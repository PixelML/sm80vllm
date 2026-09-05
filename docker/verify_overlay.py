"""Build-time gate: the patch overlay must have landed and vllm must still import."""
import pathlib
import sys

PKG = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")

# One representative new file per patch group, so a silently-dropped COPY fails here
# rather than 17 minutes into a boot.
REQUIRED = [
    "v1/attention/backends/mla/triton_mla_sparse.py",   # 0001/0002 SM8x fallbacks
    "v1/attention/ops/fp8_sm80.py",                     # 0002 kpool fp8 software encode
    "models/deepseek_v4/ampere/ampere_sparse.py",       # 0001 Ampere sparse MLA
    "v1/worker/gpu/spec_decode/dflash2/speculator.py",  # 0009 DFlash2 under PP
    "v1/worker/gpu/stage_timing.py",                    # 0018/0019 stage timing
    "models/glm5next/nvidia/mtp.py",                    # 0005/0017 MTP under PP
]

missing = [n for n in REQUIRED if not (PKG / n).exists()]
if missing:
    sys.exit(f"overlay incomplete, missing: {missing}")

import vllm  # noqa: E402  - after the file check, so a missing file reports first

print(f"overlay OK, vllm {vllm.__version__}")
