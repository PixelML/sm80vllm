#!/usr/bin/env bash
# GLM-5.3-Flash on 4x CMP 170HX (SM80) - PP4 with speculative decoding.
#
# Derived from promisezackr/glm53-flash-170hx-pp8 scripts/run_prod.sh (Apache-2.0),
# adapted from 8 cards to 4. Every deviation from his script is listed in
# ../README.pp-port.md; nothing else differs.
#
#   DRAFTER=mtp     ./recipes/glm53-flash-4x170hx-pp4.sh     # k=3, clean text  (default)
#   DRAFTER=dflash2 ./recipes/glm53-flash-4x170hx-pp4.sh     # k=7, his prod drafter
#
set -uo pipefail

IMAGE=${IMAGE:-ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905}
NAME=${NAME:-glm53-pp4}
PORT=${PORT:-9004}
MODEL_DIR=${MODEL_DIR:-/models/model-cache/glm-5.3-flash-awq-w4a16}
DFLASH_DIR=${DFLASH_DIR:-/models/model-cache/dflash2-glm53-flash}
TRITON_CACHE=${TRITON_CACHE:-/models/model-cache/triton-cache-pp4}
DRAFTER=${DRAFTER:-mtp}

# --- Max model length -------------------------------------------------------
# His script uses 1048576 on 8 cards. On 4 cards the engine computes an
# estimated maximum of 483840 and refuses 1M outright, so we run 393216.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-393216}

# --- PP layer partition -----------------------------------------------------
# 45 hidden layers. His KV-balancing rule: the 11 sparse-MLA layers sit at
# idx 3,7,...,43; a stage holding two of them gets fewer total layers so its KV
# budget is larger. 14,12,12,7 gives sparse counts 3/3/3/2, and the last stage
# is cut to 7 to make room for lm_head plus the drafter.
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-14,12,12,7}

# --- His env, verbatim ------------------------------------------------------
export VLLM_PP_MAX_DECODE_REQS_PER_BATCH=${VLLM_PP_MAX_DECODE_REQS_PER_BATCH:-2}  # patch 0020
export VLLM_GLM5N_SIDECAR_BLOCK_SIZE=${VLLM_GLM5N_SIDECAR_BLOCK_SIZE:-256}        # patch 0024
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=3600

GPU_UTIL=${GPU_UTIL:-0.90}   # his value; do not raise without re-measuring KV pool

case "$DRAFTER" in
  mtp)     SPEC='{"method":"mtp","num_speculative_tokens":3}' ;;
  dflash2) SPEC='{"method":"dflash","model":"/dflash2","num_speculative_tokens":7}' ;;
  none)    SPEC="" ;;
  *) echo "DRAFTER must be mtp|dflash2|none" >&2; exit 2 ;;
esac

# --- Pre-launch guard: PCIe link width + rev ff -----------------------------
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! bash "$here/scripts/check-links.sh"; then
  echo "REFUSING TO LAUNCH: PCIe links below ceiling. Set ALLOW_DEGRADED_LINK=1 for a" >&2
  echo "correctness / serve-only boot and note the degradation in any result." >&2
  exit 3
fi

docker rm -f "$NAME" >/dev/null 2>&1
set -x
docker run -d --name "$NAME" \
  --gpus '"device=0,1,2,3"' --ipc=host --shm-size 16g \
  -p 127.0.0.1:$PORT:$PORT \
  -v "$MODEL_DIR":/model:ro \
  -v "$DFLASH_DIR":/dflash2:ro \
  -v "$TRITON_CACHE":/root/.triton \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PP_LAYER_PARTITION -e VLLM_PP_MAX_DECODE_REQS_PER_BATCH \
  -e VLLM_GLM5N_SIDECAR_BLOCK_SIZE -e CUDA_DEVICE_ORDER \
  -e VLLM_WORKER_MULTIPROC_METHOD -e VLLM_ENGINE_READY_TIMEOUT_S \
  "$IMAGE" \
  /model --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port $PORT \
  --pipeline-parallel-size 4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  ${SPEC:+--speculative-config "$SPEC"} \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
set +x
echo "started $NAME ($DRAFTER) at $(TZ=Asia/Bangkok date +%H:%M:%S) GMT+7; expect READY in ~17 min"
