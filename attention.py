"""Optional Transformer Engine attention; Q/K/V use Bumblebee's HBSd layout."""

import os

import torch


def check_flash_attention_device(device):
    if device.type != 'cuda':
        raise ValueError('Flash attention requires an NVIDIA CUDA GPU; disable use_flash_attention on CPU/MPS.')
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise ValueError(f'Flash attention requires Ampere or newer (SM 80+); got SM {capability[0]}{capability[1]}.')
    return capability


def _reject_unfused_attention(module, inputs):
    # TE caches dispatch choices across modules. Guard even a choice cached
    # before NVTE_UNFUSED_ATTN was disabled.
    raise RuntimeError('Transformer Engine selected unfused attention; no supported flash attention kernel is available.')


class FlashAttention:
    """Cache parameter-free TE attention outside the model's checkpoint state."""

    def __init__(self, num_heads, num_query_groups, qk_dim, v_dim):
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self._attention = None
        self._device = None

    # TE's Python backend selector can graph-break on older releases. Keep this
    # one call eager while torch.compile optimizes the rest of the model.
    @torch.compiler.disable
    def __call__(self, query, key, value, training):
        check_flash_attention_device(query.device)
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('Flash attention requires narrow_dtype=torch.float16 or torch.bfloat16.')

        # TE dispatches using the current CUDA device. Each worker/input must
        # select its own device, including when multiple GPUs share a process.
        with torch.cuda.device(query.device):
            if self._attention is None or self._device != query.device:
                # TE chooses FlashAttention or cuDNN fused attention using SM,
                # dtype and shapes. Never silently use quadratic unfused attention.
                os.environ['NVTE_UNFUSED_ATTN'] = '0'
                try:
                    from transformer_engine.pytorch import DotProductAttention
                except ImportError as exc:
                    raise ImportError(
                        'Install Transformer Engine in the CUDA environment: '
                        'pip install --no-build-isolation "transformer_engine[pytorch]"'
                    ) from exc

                self._attention = DotProductAttention(
                    num_attention_heads=self.num_heads,
                    kv_channels=(self.qk_dim, self.v_dim),
                    num_gqa_groups=self.num_query_groups,
                    attention_dropout=0.0,
                    qkv_format='sbhd',
                    attn_mask_type='causal',
                    softmax_scale=self.qk_dim ** -0.5,
                )
                self._attention.unfused_attention.register_forward_pre_hook(_reject_unfused_attention)
                self._device = query.device

            self._attention.train(training)
            # Separate SBHd inputs exclude TE's older quadratic cuDNN backend.
            # TE returns SB(Hd). Preserve native GQA: no KV repeats.
            query, key, value = (x.permute(2, 1, 0, 3).contiguous() for x in (query, key, value))
            output = self._attention(query, key, value)
            return output.reshape(*query.shape[:3], self.v_dim).permute(2, 1, 0, 3)
