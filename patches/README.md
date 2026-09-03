# Runtime overlays

Files here are not part of the vLLM Python package tree. The build copies
them over an installed third-party package inside the image, after
`pip install`, instead of patching vLLM source.

## `safetensors_torch_f8_e8m0.py`

Overlay for `safetensors/torch.py` (the installed `safetensors` package,
not vLLM). Adds an `F8_E8M0` dtype entry (`torch.float8_e8m0fnu`, Torch
2.5.0+) to the dtype tables that `safetensors` uses to load and save
tensors, so a checkpoint shard that stores a `ue8m0` block-scale tensor
loads without `safetensors` raising an unrecognized-dtype error.

Applied in `docker/Dockerfile.sm80` with:

```dockerfile
COPY patches/safetensors_torch_f8_e8m0.py /opt/venv/lib/python3.12/site-packages/safetensors/torch.py
```

Source: `PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX`,
`patches/safetensors_torch.py`.
