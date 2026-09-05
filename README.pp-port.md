# `pp-dflash2/glm53-flash-487ecf187-20260905` — GLM-5.3-Flash PP on 4x CMP 170HX

This branch is a **runtime overlay**, not a buildable vLLM source tree. It carries the
pre-patch copies of the 48 files that the upstream patch series modifies, the 24 patches
as individual commits, and the launcher/tooling needed to serve GLM-5.3-Flash with
pipeline parallelism on SM80.

## Attribution

| | |
|---|---|
| Patch series | <https://github.com/promisezackr/glm53-flash-170hx-pp8> @ `90ec72e` |
| License | Apache-2.0 |
| Taken as-is | **all 24 patches**, byte-for-byte. Every patch is one commit with an `Attribution-*` trailer naming the source repo, revision, patch file, license, and usage. |
| Adapted | only the launcher — `recipes/glm53-flash-4x170hx-pp4.sh`, derived from his `scripts/run_prod.sh`, 8 cards → 4. Deviations listed below. |
| Not reused | his benchmark **numbers**. His 123 tok/s is on 8 cards with an NVFP4 checkpoint; it is cited for comparison only and is `community-reported`, not measured by us. |
| Base image | `vllm/vllm-openai:glm53-flash`, digest `sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4`, vLLM `0.1.dev20051+g487ecf187`, Apache-2.0 (the vLLM project) |
| Drafter (not vendored, not baked into the image) | `incoai/GLM-5.3-Flash-DFlash2`, **cc-by-nc-nd-4.0 — research/measurement only, no commercial serving** |
| Upstream technique credit | DFlash2 from [z-lab/dflash](https://github.com/z-lab/dflash) / vLLM PR #52816; SM8x kernel port from [344303947/vllm170hx](https://github.com/344303947/vllm170hx), both reached us through the patch series above |

## Why an orphan branch

Upstream commit `487ecf187` is **not publicly fetchable** — it is not in `vllm-project/vllm`,
not in the GLM release PR head, and not reachable from any remote we hold. It exists only
inside the published image. A rebase onto it is therefore impossible, so the branch is
rooted at a vendored baseline extracted from the image (see `VENDOR.md`).

## Why this base and not `glm53-sm80`

Grafting the 24 patches onto our `glm53-sm80` branch (wtdcode backport) was measured, not
guessed:

* plain `git apply --check`: **21 of 24 patches conflict** (0001 fails on 33/33 files, 0009 on 13/13)
* sequential `git apply -3`: **12 of 24 conflict**, including all of 0001–0010 — the entire
  SM8x + PP + spec-decode core — and 2 files retain conflict markers after the full series

Patch 0001 is the source repo's own port of `344303947/vllm170hx` onto the glm53 tree;
`glm53-sm80` carries wtdcode's *independent* port of the same work onto a different vLLM
tree. Two divergent ports of one body of work is not a rebasable delta.

The base image loses us nothing on this box: it already ships DeepSeek-V4 (48 files, parity
with our fork), sparse MLA, mHC, dflash, and the full `glm5next/nvidia` path. Our fork's
138 extra `.py` files are AMD / CPU-SDPA / AMX / qwen4-exp / dspark paths, none of which
run on a 4x SM80 NVIDIA box.

## Verification of the port

The tree produced by these 25 commits was compared file-by-file against the tree that
actually served on this hardware (`~/WIP/verbatim-pp/glm53`, two successful PP4 boots):

```
66/66 files byte-identical, 0 differing, 0 missing
0 untracked deltas  (no file outside the tracked set differs from the pristine baseline)
```

So the image built from this branch is functionally identical to the proven verbatim run.

## What we retired rather than ported

`specdec/promisezackr-pp-port` (`7bd98cb..b63a372`) touched 6 files. Five were
re-derivations of patches **0005** (draft head loading), **0009** (DFlash2 under PP),
**0010** (aux relay), **0013** (dflash zero embeddings), **0020** (decode micro-batch cap).
On this base the originals supersede them. Only `tools/overlay-lib.sh` carried forward.

`vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py` and the mHC `ar_int8.py` kernel
were **deliberately not** added to the overlay: neither is on the serve path, and the
former needs config/registry plumbing this base lacks. The extraction tooling in
`tools/specdec/` runs against the stock image via forward hooks and needs neither.

## Launcher deviations from `scripts/run_prod.sh`

| | his | ours | forced by |
|---|---|---|---|
| cards / PP | 8, `7,5,7,5,7,5,7,2` | 4, `14,12,12,7` | card count. His KV rule (sparse-MLA layers at idx 3,7,…,43; stages holding two get fewer layers) is preserved — sparse counts 3/3/3/2, last stage cut to 7 for lm_head + drafter. |
| `--max-model-len` | 1048576 | 393216 | engine refuses 1M on 4 cards (`estimated maximum model length 483840`) |
| `--limit-mm-per-prompt` | absent | `{"image":0,"video":0}` | video-profiling hardware fault on this box; mandatory here |
| default drafter | DFlash2 k=7 | **MTP k=3** | measured: on our AWQ W4A16 checkpoint DFlash2 k=7 gives 36.0% acceptance and degenerate text, MTP k=3 gives 69.9% and clean text. `DRAFTER=dflash2` still selects his. |
| pre-launch guard | none | `scripts/check-links.sh` | this box retrains GPU1 to PCIe x1 after some cold boots; the guard refuses to benchmark below slot ceiling or on `rev ff` |

Everything else — `--gpu-memory-utilization 0.90`, `--max-num-seqs 8`,
`--max-num-batched-tokens 4096`, `--no-enable-prefix-caching`,
`VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2`, `VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, the tool-call and
reasoning parsers — is his, unchanged.

## Build

```bash
docker build -f Dockerfile.glm53-pp -t ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905 .
```

Seconds, not hours: no CUDA compilation, the patches are pure Python.

## Serve

```bash
DRAFTER=mtp     ./recipes/glm53-flash-4x170hx-pp4.sh
DRAFTER=dflash2 ./recipes/glm53-flash-4x170hx-pp4.sh
```
