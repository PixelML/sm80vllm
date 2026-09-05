#!/bin/bash
# Pre-launch guard: refuse to benchmark when any 170HX link is below its slot ceiling.
# Slot ceilings on this box: guest GPU0 (host 3c) x8, GPU1 (host d8) x8, GPU2/3 x16.
set -e
declare -A CEIL=([0]=8 [1]=8 [2]=16 [3]=16)
bad=0
while IFS=", " read -r idx cur; do
  [ -z "$idx" ] && continue
  if [ "$cur" -lt "${CEIL[$idx]:-16}" ]; then echo "LINK BELOW CEILING: GPU$idx width x$cur (ceiling x${CEIL[$idx]})"; bad=1; fi
done < <(nvidia-smi --query-gpu=index,pcie.link.width.current --format=csv,noheader)
lspci | grep -E "0[1-4]:00" | grep -q "rev ff" && { echo "GPU rev ff (wedged)"; bad=1; }
[ "$bad" = 0 ] && echo "links OK" || { [ "${ALLOW_DEGRADED_LINK:-0}" = 1 ] && echo "continuing (ALLOW_DEGRADED_LINK=1)" || exit 3; }
