<p align="center">
  <img src="assets/bumblebee.png" alt="Bumblebee" width="400">
</p>

# Bumblebee

[Bumblebee](https://en.wikipedia.org/wiki/Bumblebee_(Transformers)) is a small Transformer.

## Project setup:

1. Clone repo.

2. `cp setenv.sh local_setenv.sh`

    Configure `local_setenv.sh` as detailed in that file.

## Basic training example (single worker):

1. `. local_setenv.sh`

2. `python train.py`

The first time you run `train.py`, it will download and preprocess a very large Hugging Face dataset into `$HF_HOME`. This may take hours.

## Cluster usage

See `interactive-dp8.sh` and `batch-dp32.sh`.

## Optional flash attention

The default keeps the explicit attention math. To use Transformer Engine's
optimized attention on an Ampere or newer NVIDIA GPU:

```sh
# In your CUDA environment, if Transformer Engine isn't already installed:
pip install --no-build-isolation 'transformer_engine[pytorch]>=2.5,<3'
python train.py --flash-attention
```

You can also set `ModelConfig(use_flash_attention=True)`. The training script
reports the selected GPU and SM version; TE chooses FlashAttention or cuDNN's
flash-based fused attention based on the hardware, dtype and shapes. GQA,
causality, and the existing attention scale are preserved. The option requires
FP16/BF16 QKV and raises if no supported optimized kernel is available.

TE is imported only when enabled. Its stateless attention call runs eagerly
inside `torch.compile`; the rest of the model can still be compiled. Add
`NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` to see TE's backend choice. See TE's
[installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/installation.html)
and [attention dispatcher docs](https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html).

## Development

Run `python -m unittest test_attention` for attention parity and configuration
checks. CPU tests use an SDPA stand-in to check the adapter; the real TE
forward/backward and compile smoke test requires CUDA and Transformer Engine.

I use a pre-commit Git hook that calls ruff-check. To install this hook on your clone, run the following from the top-level directory:

```
# Install pre-commit and ruff, if you don't have them already:
pip install pre-commit ruff

# Install the hook, specified in .pre-commit-config.yaml, to .git/hooks/pre-commit
pre-commit install

# Ensure that it installed correctly.
pre-commit run --all-files
```
