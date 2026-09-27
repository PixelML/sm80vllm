# PixelML sm80vllm: `glm53-dsv4/morrowmake-378c37b`

SM80 (Ampere: CMP 170HX, A100, RTX 3090) serving branch for GLM-5.3-Flash and DeepSeek-V4.

## Origin

- **Base:** [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) branch `ampere-glm53` @ `378c37b0098a41a5cd25b3bf8b56d158e33a6cbf`, the engine pinned by [glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) v1.4.1. Its 102 commits sit on upstream vLLM `e55d076f89` (2026-09-25) and are kept unmodified, with original authorship. Licence: Apache-2.0, same as vLLM (see `LICENSE`).
- **DeepSeek-V4 on SM8x:** merge of upstream [vllm-project/vllm#55184](https://github.com/vllm-project/vllm/pull/55184) (mikekg, head `7f82bb7486`): software fp8 e4m3, Triton sparse-MLA and paged MQA-logits fallbacks for DeepSeek-V4 Flash/Pro without DeepGEMM. It replaces this repo's older `sm80` commit `f8ea5bb16` (same fixes on an August base; 16 conflicts here versus 3). Where both sides add `TRITON_MLA_SPARSE`, Morrowmake's DSA backend is kept for GLM-5.3 / V3.2; DeepSeek-V4 reaches the PR's kernels through its own model path. Resolution notes are in the merge commit.

## Image

```bash
docker pull ghcr.io/pixelml/club-170hx@sha256:54769105a30c22996d6264b4c8f56d9538102d2324cf2f4a47575d18fed9afaa
```

Tag `vllm-glm53-dsv4-sm80-20260927`, built from `05a613f2f` of this branch with the Dockerfile of glm53-flash-cmp170hx-recipe 1.4.1 (repo/commit/labels repointed; build-time import check runs against the CUDA stub `libcuda.so.1`). Drop-in for the recipe: `IMAGE=<the digest above> ./start.sh`.

## Measured

GLM-5.3-Flash W4A16 + DFlash2 k=3, PP4, 4× CMP 170HX: identical on this image and on Morrowmake's (130.7 / 121.4 / 88.7 tok/s single user, KV pool 2,320,328); see [club-170hx results](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-27-glm-5.3-flash-morrowmake-pp4-narrow-link).
