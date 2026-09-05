#!/bin/bash
# DFlash2 with embed fix: graphs on, unpinned, k=7, noPC. Port 9004.
set -u
docker rm -f glm53-dfix glm53-tre glm53-trace glm53-mtp glm53-k7 >/dev/null 2>&1

docker run -d --name glm53-dfix \
  --gpus all --ipc=host \
  -p 9004:9004 \
  -v /data/vllm-deploy/glm53/vllm:/usr/local/lib/python3.12/dist-packages/vllm \
  -v /data/models/GLM-5.3-Flash:/model:ro -v /data/models/GLM-5.3-Flash-DFlash2:/dflash2:ro \
  -v /data/vllm-deploy/glm53/triton-cache:/root/.triton \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_PP_LAYER_PARTITION=6,6,6,6,6,6,6,3 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm/vllm-openai:glm53-flash \
  /model --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port 9004 \
  --pipeline-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.82 \
  --no-enable-prefix-caching --speculative-config '{"method":"dflash","model":"/dflash2","num_speculative_tokens":7}' \
  || { echo DOCKER-RUN-FAILED; exit 1; }

echo waiting...
for i in $(seq 1 240); do
  curl -sf -m 3 http://localhost:9004/health >/dev/null 2>&1 && { echo "READY after ~$((i*10))s"; break; }
  [ "$(docker inspect -f "{{.State.Running}}" glm53-dfix 2>/dev/null)" != "true" ] && { echo CONTAINER-DIED; docker logs glm53-dfix --tail 40; exit 2; }
  sleep 10
done
curl -sf -m 3 http://localhost:9004/health >/dev/null || { echo TIMEOUT; exit 3; }
docker logs glm53-dfix 2>&1 | grep -i "loaded target embed" | head -2

snap() { curl -s http://localhost:9004/metrics | grep -E "spec_decode_num" | grep -v "^#" | grep -v created | grep -oE "[0-9.]+$" | tr "\n" " "; }
run() {
  A=$(snap); T0=$(date +%s%N)
  R=$(curl -s -m 400 http://localhost:9004/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"GLM-5.3-Flash\",\"prompt\":$2,\"max_tokens\":$3,\"temperature\":0}")
  T1=$(date +%s%N); B=$(snap)
  python3 -c "
import json
r=json.loads(''$R'')
n=r[\"usage\"][\"completion_tokens\"]
a=\"$A\".split(); b=\"$B\".split()
d=[float(y)-float(x) for x,y in zip(a,b)]
al=1+d[2]/max(d[0],1)
pp=[round(p/max(d[0],1),3) for p in d[3:8]]
print(f\"$1: tokens={n} t/s={n/(($T1-$T0)/1e9):.1f} accept_len={al:.2f} per-pos={pp}\")"
}
run warmup "\"hello\"" 30 >/dev/null
run counting "\"Count from 1 to 100, comma separated:\"" 250
run repetition "\"Repeat exactly 30 times the line: hello world foo bar\"" 200
run json "\"Output a JSON array of 20 objects with fields name, age, city. Only JSON:\"" 300
run code "\"用Python写一个快速排序，并解释复杂度。\"" 300
run math "\"Solve step by step: what is 847 times 23?\"" 250
run prose "\"写一段关于秋天的散文\"" 250
