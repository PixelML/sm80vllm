#!/usr/bin/env bash
# Run 3 launcher -- one shot, from the 170HX VM. Re-sync, verify, then train.
#
#   ./launch_run3.sh 1        # apollo node 1 (direct from the VM)
#   ./launch_run3.sh 2        # apollo node 2 (hops via node 1)
#   DRY=1 ./launch_run3.sh 2  # preflight only, start nothing
#
# Everything before the training launch is a check, and every check is cheap.
# This lane has repeatedly paid a 10-20 minute model load to discover a
# one-second problem, and node 2 in particular has a specdec-tools/ copy that
# predates the drafter rewrite -- running it there would silently execute the
# old convention and produce another uninterpretable number.
set -euo pipefail

NODE="${1:-1}"
DRY="${DRY:-0}"
TOOLS=/models/claude-bench/glm-sm80-port/specdec-wt/tools/specdec
DATA=/library/models/specdec-data/sliceB
SHARED=/library/models/specdec-data/target-shared.safetensors
IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
BLOCK="${BLOCK:-8}"
STEPS="${STEPS:-8000}"
LR="${LR:-1.5e-4}"
RUN="${RUN:-run3-bs${BLOCK}}"

if [ "$NODE" = "1" ]; then H="apollo"; SH() { ssh -o ConnectTimeout=20 apollo "$@"; }
elif [ "$NODE" = "2" ]; then H="apollo-2"; SH() { ssh -o ConnectTimeout=20 apollo "ssh -o ConnectTimeout=20 apollo-2 \"$*\""; }
else echo "node must be 1 or 2"; exit 2; fi
say() { echo "[run3] $*"; }

# ---- 1. node reachable, GPU free, do not disturb anyone ---------------------
say "target node $NODE ($H)"
SH "true" || { echo "node unreachable"; exit 1; }
BUSY=$(SH "docker ps --format '{{.Names}}' | grep -v autoround-setup || true")
if [ -n "$BUSY" ]; then
  say "containers running on $H: $BUSY"
  [ "${FORCE:-0}" = "1" ] || { echo "refusing: another lane holds this node (FORCE=1 to override)"; exit 1; }
fi
say "host free (autoround-setup ignored, never touched)"

# ---- 2. checksums on the VM side -------------------------------------------
say "hashing source data (once; cached in $DATA/SHA256SUMS)"
if [ ! -f "$DATA/SHA256SUMS" ]; then
  ( cd "$DATA" && sha256sum manifest.json shard-*.npz > SHA256SUMS )
fi
if [ ! -f "$(dirname "$SHARED")/SHARED.sha256" ]; then
  ( cd "$(dirname "$SHARED")" && sha256sum "$(basename "$SHARED")" > SHARED.sha256 )
fi

# ---- 3. push tools (ALWAYS -- node 2's copy is stale) ----------------------
say "syncing specdec-tools"
if [ "$NODE" = "1" ]; then
  rsync -a --delete --exclude __pycache__ --omit-dir-times "$TOOLS/" apollo:~/specdec-tools/
else
  rsync -a --delete --exclude __pycache__ --omit-dir-times "$TOOLS/" apollo:~/specdec-tools/
  ssh -o ConnectTimeout=20 apollo "rsync -a --delete --exclude __pycache__ --omit-dir-times ~/specdec-tools/ apollo-2:~/specdec-tools/"
fi
SH "ls ~/specdec-tools/ref_eval2.py ~/specdec-tools/drafter_v2.py >/dev/null" \
  || { echo "tools sync failed"; exit 1; }
say "tools synced (ref_eval2.py + drafter_v2.py present)"

# ---- 4. push data + shared weights, resumably ------------------------------
say "syncing slice B (14 GB) and target-shared (2.5 GB) -- resumable"
if [ "$NODE" = "1" ]; then
  rsync -a --append-verify --info=progress2 "$DATA/" apollo:~/specdec-data/sliceB/
  rsync -a --append-verify "$SHARED" "$(dirname "$SHARED")/SHARED.sha256" apollo:~/specdec-data/
else
  rsync -a --append-verify --info=progress2 "$DATA/" apollo:~/specdec-data/sliceB/
  rsync -a --append-verify "$SHARED" "$(dirname "$SHARED")/SHARED.sha256" apollo:~/specdec-data/
  ssh -o ConnectTimeout=20 apollo "rsync -a --append-verify ~/specdec-data/sliceB/ apollo-2:~/specdec-data/sliceB/ && rsync -a --append-verify ~/specdec-data/target-shared.safetensors ~/specdec-data/SHARED.sha256 apollo-2:~/specdec-data/"
fi

# ---- 5. verify ON THE NODE -- the copy is what trains ----------------------
say "verifying checksums on $H (this is the copy that trains, so it is the one checked)"
SH "cd ~/specdec-data/sliceB && sha256sum -c SHA256SUMS --quiet" \
  || { echo "SLICE B CHECKSUM MISMATCH on $H -- do not train"; exit 1; }
SH "cd ~/specdec-data && sha256sum -c SHARED.sha256 --quiet" \
  || { echo "target-shared CHECKSUM MISMATCH on $H -- do not train"; exit 1; }
say "checksums OK"

SH "python3 ~/specdec-tools/verify_manifest.py ~/specdec-data/sliceB 400000" \
  || { echo "manifest on $H is not the corrected-tap slice B"; exit 1; }

if [ "$DRY" = "1" ]; then say "DRY=1, preflight passed, starting nothing"; exit 0; fi

# ---- 6. train ---------------------------------------------------------------
say "starting $RUN on $H: block_size=$BLOCK steps=$STEPS lr=$LR"
SH "mkdir -p ~/drafters/$RUN && docker run -d --name glm53-$RUN --gpus all --ipc=host \
  -e PYTHONPATH=/tools \
  -v \$HOME/specdec-tools:/tools:ro \
  -v \$HOME/specdec-data:/data:ro \
  -v \$HOME/drafters/$RUN:/out \
  -w /tools --entrypoint python3 $IMAGE \
  /tools/train_drafter.py \
    --data /data/sliceB --shared /data/target-shared.safetensors \
    --out /out --block-size $BLOCK --steps $STEPS --lr $LR \
    --eval-every 500 --eval-blocks 400"
say "started. follow: ssh $H docker logs -f glm53-$RUN"
say "artifacts land on the HOST at ~/drafters/$RUN (bind-mounted, survives the container)"
