# sm80vllm container for 4 × CMP 170HX (GLM-5.3-Flash W4A16)

Pinned build recipe for the images used in the
[PixelML/club-170hx](https://github.com/PixelML/club-170hx) GLM-5.3-Flash notebooks.

## Pull (exact bits that were measured)

| image | what it is | digest |
|---|---|---|
| `ghcr.io/pixelml/sm80vllm:tf-learnings-c93c274c8` | engine + copy drafts (`VLLM_GLM5_COPY_DRAFTS`) | `sha256:f0dbb483b000f85ac66adb6f190c11d1af395914df67a0d9b7e54dfe2258fa86` |
| `ghcr.io/pixelml/sm80vllm:tf-learnings-127c6f076` | engine only (Morrowmake `mm/ampere` @ `3a2bf16da` merged into the DSV4 branch) | `sha256:704cbb8841be7c99bd1d5d6a8136fc118ca22cfc85457ef4a669a07ef957e8ff` |

Always pull by digest: `docker pull ghcr.io/pixelml/sm80vllm@sha256:...`.

## Build it yourself

Every input is pinned: the CUDA base image by digest, the engine source by full commit,
the precompiled-extension wheel by full commit (see the header of `Dockerfile`).

```bash
# 1. Engine image at 127c6f0761b2c81d2c0ba7e2d8a5cf71e3247903
git clone https://github.com/PixelML/sm80vllm.git && cd sm80vllm
docker build -f docker/cmp170hx/Dockerfile \
  --build-arg VLLM_REPO=https://github.com/PixelML/sm80vllm.git \
  --build-arg VLLM_BRANCH=glm53-sm80-tf-learnings \
  --build-arg VLLM_COMMIT=127c6f0761b2c81d2c0ba7e2d8a5cf71e3247903 \
  -t sm80vllm-cmp170hx:127c6f076 docker/cmp170hx

# 2. Copy-drafts overlay at c93c274c86543d513c67243859480aeec46bc951 (four Python files)
git checkout c93c274c86543d513c67243859480aeec46bc951
docker build -f docker/cmp170hx/Dockerfile.overlay \
  --build-arg BASE_IMAGE=sm80vllm-cmp170hx:127c6f076 \
  -t sm80vllm-cmp170hx:c93c274c8 .
```

A self-built image is functionally the same but not byte-identical (build timestamps).
Use the published digests when you need the exact measured bits.

## Run

The image is a drop-in for the `IMAGE=` line of
[Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe)
v1.6.0 (`a242b4f`): set `IMAGE=ghcr.io/pixelml/sm80vllm@sha256:...` in the recipe's `.env`.
Copy drafts are off unless `VLLM_GLM5_COPY_DRAFTS=1`.

## Credits and licences

`Dockerfile` and `runtime_tree.py` are from the Morrowmake recipe (MIT) and build
[Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) (Apache-2.0);
this repository's changes are Apache-2.0. Model weights are not in the image.
