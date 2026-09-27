# PixelML sm80vllm: `glm53-dsv4/morrowmake-378c37b`

SM80 (Ampere: CMP 170HX, A100, RTX 3090) serving branch for GLM-5.3-Flash and DeepSeek-V4.

## Origin

- **Base:** [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) branch `ampere-glm53` @ `378c37b0098a41a5cd25b3bf8b56d158e33a6cbf`, the engine pinned by [glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) v1.4.1. Its 102 commits sit on upstream vLLM `e55d076f89` (2026-09-25) and are kept unmodified, with original authorship. Licence: Apache-2.0, same as vLLM (see `LICENSE`).
- **PixelML additions** (commits after this file): DeepSeek-V4 sparse MLA on SM8x, forward-ported from this repo's `sm80` branch (`f8ea5bb16`); its Triton backend is registered as `TRITON_MLA_SPARSE_DSV4` so it coexists with Morrowmake's DSA `TRITON_MLA_SPARSE`.

## Measured

GLM-5.3-Flash W4A16 + DFlash2 k=3, PP4, 4× CMP 170HX: see [club-170hx results](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-27-glm-5.3-flash-morrowmake-pp4-narrow-link).
