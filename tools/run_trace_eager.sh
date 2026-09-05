#!/bin/bash
# DFlash2 online trace capture boot (graphs on, k=7, noPC) on 9004.
set -u
docker rm -f glm53-tre glm53-mtp glm53-k7 glm53-dump >/dev/null 2>&1
rm -rf /data/vllm-deploy/glm53/triton-cache/dftrace
mkdir -p /data/vllm-deploy/glm53/triton-cache/dftrace

docker run -d --name glm53-tre \
  --gpus all --ipc=host \
  -p 9004:9004 \
  -v /data/vllm-deploy/glm53/vllm:/usr/local/lib/python3.12/dist-packages/vllm \
  -v /data/models/GLM-5.3-Flash:/model:ro -v /data/models/GLM-5.3-Flash-DFlash2:/dflash2:ro \
  -v /data/vllm-deploy/glm53/triton-cache:/root/.triton \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_PP_LAYER_PARTITION=6,6,6,6,6,6,6,3 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_DFLASH_TRACE=/root/.triton/dftrace \
  vllm/vllm-openai:glm53-flash \
  /model --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port 9004 \
  --pipeline-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.82 \
  --enforce-eager --no-enable-prefix-caching --speculative-config '{"method":"dflash","model":"/dflash2","num_speculative_tokens":7}' \
  || { echo "DOCKER-RUN-FAILED"; exit 1; }

echo "waiting for readiness..."
for i in $(seq 1 240); do
  curl -sf -m 3 http://localhost:9004/health >/dev/null 2>&1 && { echo "READY after ~$((i*10))s"; break; }
  [ "$(docker inspect -f {{.State.Running}} glm53-tre 2>/dev/null)" != "true" ] && { echo CONTAINER-DIED; docker logs glm53-tre --tail 60; exit 2; }
  sleep 10
done
curl -sf -m 3 http://localhost:9004/health >/dev/null || { echo TIMEOUT; exit 3; }

# arm tracing AFTER readiness so warmup does not burn the budget
touch /data/vllm-deploy/glm53/triton-cache/dftrace/ARM
echo ARMED

# workload 1: verbatim repetition (context-copy probe)
curl -s -m 300 http://localhost:9004/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"GLM-5.3-Flash","prompt":"Repeat exactly 30 times the line: hello world foo bar","max_tokens":120,"temperature":0}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d[\"choices\"][0][\"text\"][:200]))"
ls /data/vllm-deploy/glm53/triton-cache/dftrace/ | wc -l
