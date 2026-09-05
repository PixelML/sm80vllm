#!/usr/bin/env python3
"""Single-stream spec-decode benchmark via /metrics counter deltas."""
import json
import time
import urllib.request

BASE = "http://localhost:9004"

WORKLOADS = [
    ("counting", "Count from 1 to 100, comma separated:", 250),
    ("repetition", "Repeat exactly 30 times the line: hello world foo bar", 200),
    ("json", "Output a JSON array of 20 objects with fields name, age, city. Only JSON:", 300),
    ("code", "用Python写一个快速排序，并解释复杂度。", 300),
    ("math", "Solve step by step: what is 847 times 23?", 250),
    ("prose", "写一段关于秋天的散文", 250),
]


def metrics():
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if "spec_decode_num" in line and "created" not in line:
            name = line.split("{")[0]
            val = float(line.rsplit(" ", 1)[1])
            out.setdefault(name, []).append(val)
    return {k: sum(v) for k, v in out.items()}


def one(name, prompt, max_tokens):
    a = metrics()
    body = json.dumps(
        {"model": "GLM-5.3-Flash", "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=400))
    dt = time.time() - t0
    b = metrics()
    n = resp["usage"]["completion_tokens"]
    drafts = b.get("vllm:spec_decode_num_drafts_total", 0) - a.get("vllm:spec_decode_num_drafts_total", 0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    per = []
    for i in range(7):
        k = f"vllm:spec_decode_num_accepted_tokens_per_pos_total"
        pass
    al = 1 + acc / max(drafts, 1)
    print(f"{name:10s} tokens={n} t/s={n/dt:.1f} accept_len={al:.2f} "
          f"(drafts={drafts:.0f} accepted={acc:.0f})")


for name, prompt, mt in WORKLOADS:
    one(name, prompt, mt)
