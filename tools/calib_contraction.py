# Offline contraction-basis calibration for the DFlash2 draft.
#
# Inputs: raw mHC stream dumps streams_L{5,14,24,33,42}_*.pt ([T, 4, 4096] fp32)
# captured during ONE short-prompt prefill, the prompt text, and the target's
# actual greedy continuation. For each candidate contraction (mean / sum /
# stream0 / stream-last), build the context feature, run the reference DFlash2
# draft forward (HF-style, from the z-lab reference implementation), draft the
# first block, and score against the true continuation.
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

DUMP = "/root/.triton/auxdump"
DRAFT = "/dflash2"
TARGET_TOK = "/model"
PROMPT = os.environ.get("CALIB_PROMPT", "Repeat exactly 30 times the line: hello world foo bar")
TRUTH = open("/root/.triton/auxdump/TRUTH.txt").read()

dev = "cuda:0"
cfg = AutoConfig.from_pretrained(DRAFT)
dc = cfg.dflash_config
LAYERS = dc["target_layer_ids"]  # [5, 14, 24, 33, 42]
MASK_ID = dc["mask_token_id"]
BLOCK = dc["block_size"]

tok = AutoTokenizer.from_pretrained(TARGET_TOK)
truth_ids = tok(TRUTH, add_special_tokens=False)["input_ids"] if TRUTH else []

# --- load dumps: chunk 000 per layer = the prompt prefill ---
streams = {}
for L in LAYERS:
    for f in sorted(glob.glob(f"{DUMP}/streams_L{L}_*.pt")):
        t = torch.load(f, map_location="cpu")
        if 10 <= t.shape[0] <= 20:
            streams[L] = t.to(dev)
            break
    assert L in streams, f"no prefill-sized chunk for L{L}"
T = streams[LAYERS[0]].shape[0]
for L in LAYERS:
    assert streams[L].shape[0] == T, (L, streams[L].shape)
for flags in (True, False):
    ids = tok(PROMPT, add_special_tokens=flags)["input_ids"]
    if len(ids) == T:
        prompt_ids = ids
        break
else:
    raise AssertionError(f"prompt tokenization mismatch: dump T={T}")
print(f"prompt tokens: {T}, truth tokens: {len(truth_ids)}")

# --- load draft weights ---
w = {}
with safe_open(f"{DRAFT}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        w[k] = f.get_tensor(k).to(dev)

# Draft shares embed_tokens/lm_head with the target: load from target shards.
tindex = json.load(open(f"{TARGET_TOK}/model.safetensors.index.json"))["weight_map"]
for src, dst in (
    ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
    ("lm_head.weight", "lm_head.weight"),
):
    shard = tindex[src]
    with safe_open(f"{TARGET_TOK}/{shard}", framework="pt") as f:
        w[dst] = f.get_tensor(src).to(dev)


def rms(x, weight, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def rope(q, k, pos, theta):
    d = q.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=dev).float() / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]

    def rot(x):
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1
        ).to(x.dtype)

    return rot(q.float()).to(q.dtype), rot(k.float()).to(k.dtype)


def grouped_conv(hidden, dynamic, base, group_size):
    # hidden [B, L, H]; dynamic [B, L, taps, groups]; base [taps, H]
    b, length, H = hidden.shape
    groups = H // group_size
    blocks = hidden.view(b, length, groups, group_size)
    dyn = dynamic.view(b, length, base.shape[0], groups, 1)
    out = torch.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else F.pad(blocks[:, :-off], (0, 0, 0, 0, off, 0))
        kern = base[off].view(1, 1, groups, group_size).to(hidden.dtype)
        out = out + kern * vals + dyn[:, :, off] * vals
    return out.view(b, length, H)


class Conv:
    def __init__(self, prefix):
        self.base = w[f"{prefix}.base_kernel"].float()  # [2, taps, H]
        self.proj = w[f"{prefix}.kernel_projection.weight"].float()
        self.gs = int(dc["conv_group_size"])
        self.taps = int(dc["conv_kernel_size"])

    def prepare(self, h):
        groups = h.shape[-1] // self.gs
        dyn = (h.float() @ self.proj.T).view(*h.shape[:-1], 2, self.taps, groups)
        return (
            grouped_conv(h.float(), dyn[..., 0, :, :], self.base[0], self.gs).to(h.dtype),
            dyn[..., 1, :, :],
        )

    def finish(self, h, dyn):
        return grouped_conv(h.float(), dyn, self.base[1], self.gs).to(h.dtype)


def draft_forward(ctx_feat, anchor_id):
    """ctx_feat [T, 5H] context features; returns BLOCK-1 draft ids (greedy chain)."""
    H = cfg.hidden_size
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    hd = cfg.head_dim
    theta = cfg.rope_parameters["rope_theta"] if isinstance(
        getattr(cfg, "rope_parameters", None), dict) else getattr(cfg, "rope_theta", 1e6)

    ctx = (ctx_feat.float() @ w["fc.weight"].float().T)  # [T, H]
    ctx = rms(ctx, w["hidden_norm.weight"])[None]  # [1, T, H]

    emb = w.get("model.embed_tokens.weight", w.get("embed_tokens.weight"))
    q_ids = torch.tensor([anchor_id] + [MASK_ID] * (BLOCK - 1), device=dev)
    hs = emb[q_ids][None].float()  # [1, B, H]
    ctx_pos = torch.arange(ctx.shape[1], device=dev)
    q_pos = torch.arange(ctx.shape[1], ctx.shape[1] + BLOCK, device=dev)

    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}"
        res = hs
        x = rms(hs, w[f"{p}.input_layernorm.weight"])
        aconv = Conv(f"{p}.attention_conv")
        x, dyn = aconv.prepare(x)
        # attention: q from x; k/v from [ctx ; x]
        q = (x.float() @ w[f"{p}.self_attn.q_proj.weight"].float().T).view(1, -1, n_q, hd)
        kv_in = torch.cat([ctx, x], dim=1)
        k = (kv_in.float() @ w[f"{p}.self_attn.k_proj.weight"].float().T).view(1, -1, n_kv, hd)
        v = (kv_in.float() @ w[f"{p}.self_attn.v_proj.weight"].float().T).view(1, -1, n_kv, hd)
        q = rms(q, w[f"{p}.self_attn.q_norm.weight"])
        k = rms(k, w[f"{p}.self_attn.k_norm.weight"])
        allpos = torch.cat([ctx_pos, q_pos])
        q, _ = rope(q, q, q_pos, theta)
        k, _ = rope(k, k, allpos, theta)
        # non-causal full attention (is_causal=False)
        q_ = q.permute(0, 2, 1, 3)
        k_ = k.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        v_ = v.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        o = F.scaled_dot_product_attention(q_.float(), k_.float(), v_.float())
        o = o.permute(0, 2, 1, 3).reshape(1, BLOCK, -1)
        o = (o @ w[f"{p}.self_attn.o_proj.weight"].float().T)
        o = aconv.finish(o, dyn)
        hs = res + o
        res = hs
        x = rms(hs, w[f"{p}.post_attention_layernorm.weight"])
        mconv = Conv(f"{p}.mlp_conv")
        x, dyn = mconv.prepare(x)
        g = x.float() @ w[f"{p}.mlp.gate_proj.weight"].float().T
        u = x.float() @ w[f"{p}.mlp.up_proj.weight"].float().T
        x = (F.silu(g) * u) @ w[f"{p}.mlp.down_proj.weight"].float().T
        x = mconv.finish(x, dyn)
        hs = res + x

    hs = rms(hs, w["norm.weight"])
    logits = hs[0, 1:].float() @ emb.float().T if "lm_head.weight" not in w else (
        hs[0, 1:].float() @ w["lm_head.weight"].float().T)
    # selector greedy chain
    topk = int(dc["selector_top_k"])
    unary, cand = torch.topk(logits, topk, dim=-1)
    hproj = hs[0, 1:].float() @ w["candidate_selector.hidden_projection.weight"].float().T
    pred_cb = w["candidate_selector.predecessor_codebook"].float()
    succ_cb = w["candidate_selector.successor_codebook"].float()
    prev = torch.tensor(anchor_id, device=dev)
    path = []
    for pos in range(BLOCK - 1):
        scores = unary[pos] + (pred_cb[prev] * hproj[pos]) @ succ_cb[cand[pos]].T
        idx = torch.argmax(scores)
        prev = cand[pos][idx]
        path.append(int(prev))
    return path


VARIANTS = {
    "mean": lambda s: s.mean(dim=1),
    "sum": lambda s: s.sum(dim=1),
    "stream0": lambda s: s[:, 0],
    "stream_last": lambda s: s[:, -1],
}

anchor = truth_ids[0] if truth_ids else prompt_ids[-1]
gt = truth_ids[1 : BLOCK] if truth_ids else []
print("ground truth block:", gt, tok.decode(gt) if gt else "")
for name, fn in VARIANTS.items():
    feat = torch.cat([fn(streams[L]) for L in LAYERS], dim=-1)  # [T, 5H]
    path = draft_forward(feat, anchor)
    hits = sum(1 for a, b in zip(path, gt) if a == b)
    run = 0
    for a, b in zip(path, gt):
        if a == b:
            run += 1
        else:
            break
    print(f"{name:12s} drafts={path} decoded={tok.decode(path)!r} "
          f"hits={hits}/{len(gt)} prefix_run={run}")
