"""Run with python -m unittest test_attention; CUDA/TE coverage is optional."""

import importlib.util
import os
import sys
import types
import unittest
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import torch

from attention import FlashAttention, check_flash_attention_device
from model import GPTModel, ModelConfig


def small_model(**overrides):
    config = {'num_layers': 2, 'hidden_dim': 32, 'num_heads': 4, 'num_query_groups': 2,
              'qk_dim': 8, 'v_dim': 8, 'mlp_dim': 48, 'max_seq_len': 16, 'vocab_size': 32,
              'narrow_dtype': torch.bfloat16}
    config.update(overrides)
    return GPTModel(ModelConfig(**config))


class FakeDotProductAttention(torch.nn.Module):
    """Independent SDPA reference for testing the adapter without CUDA/TE."""

    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        self.unfused_attention = torch.nn.Identity()

    def get_extra_state(self):
        # Real TE modules also have extra checkpoint state.
        return {}

    def forward(self, query, key, value):
        self.shapes = (query.shape, key.shape, value.shape)
        query, key, value = (x.permute(1, 2, 0, 3).float() for x in (query, key, value))
        repeats = query.size(1) // key.size(1)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key.repeat_interleave(repeats, dim=1),
            value.repeat_interleave(repeats, dim=1), is_causal=True,
            scale=self.options['softmax_scale'],
        )
        return output.permute(2, 0, 1, 3).flatten(2).to(torch.bfloat16)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        fake_te = types.ModuleType('transformer_engine.pytorch')
        fake_te.DotProductAttention = FakeDotProductAttention
        self.patches.enter_context(patch.dict(sys.modules, {
            'transformer_engine': types.ModuleType('transformer_engine'),
            'transformer_engine.pytorch': fake_te,
        }))
        self.patches.enter_context(patch('attention.check_flash_attention_device', return_value=(8, 0)))
        self.patches.enter_context(patch('torch.cuda.device', side_effect=lambda _: nullcontext()))
        self.patches.enter_context(patch.dict(os.environ))

    def test_outputs_gradients_and_native_gqa(self):
        inputs = torch.randint(32, (2, 7))
        for groups, value_dim in ((4, 8), (2, 8), (1, 12)):
            with self.subTest(groups=groups, value_dim=value_dim):
                reference = small_model(num_query_groups=groups, v_dim=value_dim)
                flash = small_model(num_query_groups=groups, v_dim=value_dim, use_flash_attention=True)
                flash.load_state_dict(reference.state_dict(), strict=True)
                expected, actual = reference(inputs), flash(inputs)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=3e-4)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                for name, parameter in reference.named_parameters():
                    torch.testing.assert_close(dict(flash.named_parameters())[name].grad,
                                               parameter.grad, rtol=0.04, atol=3e-3)
                adapter = flash.flash_attention._attention
                self.assertEqual(adapter.shapes[1][2], groups)
                self.assertEqual(adapter.options['kv_channels'], (8, value_dim))
                self.assertEqual(adapter.options['qkv_format'], 'sbhd')
                self.assertEqual(adapter.options['attn_mask_type'], 'causal')
                self.assertEqual(adapter.options['attention_dropout'], 0.0)

    def test_causal_and_training_mode(self):
        model = small_model(use_flash_attention=True)
        inputs = torch.randint(32, (2, 9))
        changed = inputs.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 32
        torch.testing.assert_close(model(inputs)[:, :5], model(changed)[:, :5], rtol=0, atol=0)
        self.assertTrue(model.flash_attention._attention.training)
        model.eval()
        with torch.inference_mode():
            model(inputs)
        self.assertFalse(model.flash_attention._attention.training)

    def test_checkpoint_compatibility_after_lazy_initialization(self):
        reference, flash = small_model(), small_model(use_flash_attention=True)
        self.assertIsNone(flash.flash_attention._attention)
        flash.load_state_dict(reference.state_dict(), strict=True)
        flash(torch.randint(32, (1, 5)))
        self.assertEqual(reference.state_dict().keys(), flash.state_dict().keys())
        reference.load_state_dict(flash.state_dict(), strict=True)
        self.assertEqual(len(list(reference.parameters())), len(list(flash.parameters())))

    def test_optional_dependency_is_lazy(self):
        with patch.dict(sys.modules, {'transformer_engine': None, 'transformer_engine.pytorch': None}):
            inputs = torch.randint(32, (1, 5))
            self.assertTrue(torch.isfinite(small_model()(inputs)).all())
            flash = small_model(use_flash_attention=True)
            self.assertIsNone(flash.flash_attention._attention)
            with self.assertRaisesRegex(ImportError, 'Install Transformer Engine'):
                flash(inputs)

    def test_unfused_fallback_is_rejected(self):
        model = small_model(use_flash_attention=True)
        inputs = torch.randint(32, (1, 5))
        model(inputs)
        adapter = model.flash_attention._attention
        with patch.object(adapter, 'forward', side_effect=lambda q, k, v: adapter.unfused_attention(q)), self.assertRaises(RuntimeError):
            model(inputs)

    def test_device_and_dtype_errors(self):
        for device in ('cpu', 'mps'):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, 'CUDA'):
                check_flash_attention_device(torch.device(device))
        with patch('torch.cuda.get_device_capability', return_value=(7, 5)), self.assertRaisesRegex(ValueError, 'Ampere'):
            check_flash_attention_device(torch.device('cuda', 0))
        for capability in ((8, 0), (9, 0), (10, 0)):
            with patch('torch.cuda.get_device_capability', return_value=capability):
                self.assertEqual(check_flash_attention_device(torch.device('cuda', 0)), capability)
        tensors = [torch.zeros(4, 1, 2, 8) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, 'narrow_dtype'):
            FlashAttention(4, 4, 8, 8)(*tensors, training=True)


@unittest.skipUnless(torch.cuda.is_available(), 'Requires an NVIDIA CUDA GPU')
class CudaAttentionTests(unittest.TestCase):
    def test_real_te_compiled_training_and_eval(self):
        if importlib.util.find_spec('transformer_engine') is None:
            self.skipTest('Install transformer_engine[pytorch] for CUDA coverage')
        check_flash_attention_device(torch.device('cuda'))
        torch.manual_seed(123)
        reference = small_model(qk_dim=64, v_dim=64).cuda()
        flash = small_model(qk_dim=64, v_dim=64, use_flash_attention=True).cuda()
        flash.load_state_dict(reference.state_dict(), strict=True)
        compiled = torch.compile(flash)
        inputs = torch.randint(32, (2, 16), device='cuda')
        expected, actual = reference(inputs), compiled(inputs)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=5e-4)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        actual.backward(gradient)
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(dict(flash.named_parameters())[name].grad,
                                       parameter.grad, rtol=0.05, atol=5e-3)
        compiled.eval()
        with torch.inference_mode():
            torch.testing.assert_close(compiled(inputs), reference(inputs), rtol=0.03, atol=5e-4)


if __name__ == '__main__':
    unittest.main()
