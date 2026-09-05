#!/bin/bash
# MTP acceptance-rate validation run (eager, PP8) on 9004.
set -u
docker rm -f glm53-mtp >/dev/null 2>&1

docker run -d --name glm53-mtp \
  --gpus all --ipc=host \
  \
  -p 9004:9004 \
  -v /data/vllm-deploy/glm53/vllm:/usr/local/lib/python3.12/dist-packages/vllm \
  -v /data/models/GLM-5.3-Flash:/model:ro \
  -v /data/vllm-deploy/glm53/triton-cache:/root/.triton \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_PP_LAYER_PARTITION=6,6,6,6,6,6,6,3 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm/vllm-openai:glm53-flash \
  /model --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port 9004 \
  --pipeline-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.82 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  || { echo "DOCKER-RUN-FAILED"; exit 1; }

echo "container started, waiting for readiness (max 40min)..."
for i in $(seq 1 240); do
  if curl -sf -m 3 http://localhost:9004/health >/dev/null 2>&1; then
    echo "READY after ~$((i*10))s"
    break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' glm53-mtp 2>/dev/null)" != "true" ]; then
    echo "CONTAINER-DIED"; docker logs glm53-mtp --tail 60; exit 2
  fi
  sleep 10
done
curl -sf -m 3 http://localhost:9004/health >/dev/null || { echo "TIMEOUT"; docker logs glm53-mtp --tail 40; exit 3; }

echo "=== sending test requests ==="
for p in "写一首关于秋天的七言绝句，并逐句解释含义。" \
         "Explain the difference between TCP and UDP in detail." \
         "用Python写一个快速排序，并解释复杂度。" \
         "What is 137*24? Show your work step by step." \
         "介绍一下光合作用的过程。"; do
  T0=$(date +%s.%N)
  OUT=$(curl -s -m 300 http://localhost:9004/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"GLM-5.3-Flash\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":300,\"temperature\":0}")
  T1=$(date +%s.%N)
  NTOK=$(echo "$OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo 0)
  echo "tokens=$NTOK elapsed=$(echo "$T1 $T0" | awk '{printf "%.1f", $1-$2}')s tok/s=$(echo "$NTOK $T1 $T0" | awk '{if($2>$3) printf "%.1f", $1/($2-$3)}')"
done

echo "=== spec decode metrics ==="
curl -s http://localhost:9004/metrics | grep -E "spec_decode" | grep -v "^#"
echo "=== recent log acceptance lines ==="
docker logs glm53-mtp 2>&1 | grep -iE "SpecDecoding|acceptance" | tail -10
