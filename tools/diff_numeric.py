# Numeric-level diff of the engine's DFlash2 draft forward against reference math.
# Consumes fwd_*.pt / ctxkv_*.pt / cand_*.pt / trace_*.pt from an instrumented boot.
# Bisects: ctx-KV write values -> rope convention -> attention read path -> head/selector.
import glob
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig

TRACE = "/root/.triton/dftrace"
DRAFT = "/dflash2"
TARGET_TOK = "/model"
dev = "cuda:0"

cfg = AutoConfig.from_pretrained(DRAFT)
dc = cfg.dflash_config
w = {}
with safe_open(f"{DRAFT}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        w[k] = f.get_tensor(k).to(dev)
tindex = json.load(open(f"{TARGET_TOK}/model.safetensors.index.json"))["weight_map"]
for src, dst in (
    ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
    ("lm_head.weight", "lm_head.weight"),
):
    with safe_open(f"{TARGET_TOK}/{tindex[src]}", framework="pt") as f:
        w[dst] = f.get_tensor(src).to(dev)

n_kv = cfg.num_key_value_heads
n_q = cfg.num_attention_heads
hd = cfg.head_dim
H = cfg.hidden_size
theta = cfg.rope_parameters["rope_theta"] if isinstance(
    getattr(cfg, "rope_parameters", None), dict) else getattr(cfg, "rope_theta", 1e6)


def rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    s = b.float().abs().max().item()
    return f"absmax={d:.4g} (ref scale {s:.4g})"


def rms(x, weight, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def rope_rot(x, pos, theta, style="half"):
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=dev).float() / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    if style == "half":
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out.to(x.dtype)


def grouped_conv(hidden, dynamic, base, group_size):
    b, length, HH = hidden.shape
    groups = HH // group_size
    blocks = hidden.view(b, length, groups, group_size)
    dyn = dynamic.view(b, length, base.shape[0], groups, 1)
    out = torch.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else F.pad(blocks[:, :-off], (0, 0, 0, 0, off, 0))
        kern = base[off].view(1, 1, groups, group_size).to(hidden.dtype)
        out = out + kern * vals + dyn[:, :, off] * vals
    return out.view(b, length, HH)


class Conv:
    def __init__(self, prefix):
        self.base = w[f"{prefix}.base_kernel"].float()
        self.proj = w[f"{prefix}.kernel_projection.weight"].float()
        self.gs = int(dc["conv_group_size"])

    def prepare(self, h):
        groups = h.shape[-1] // self.gs
        dyn = (h.float() @ self.proj.T).view(*h.shape[:-1], 2, self.base.shape[1], groups)
        return (
            grouped_conv(h.float(), dyn[..., 0, :, :], self.base[0], self.gs).to(h.dtype),
            dyn[..., 1, :, :],
        )

    def finish(self, h, dyn):
        return grouped_conv(h.float(), dyn, self.base[1], self.gs).to(h.dtype)


ctxkv = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(f"{TRACE}/ctxkv_*.pt"))]
fwd = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(f"{TRACE}/fwd_*.pt"))]
cand = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(f"{TRACE}/cand_*.pt"))]
print(f"dumps: ctxkv={len(ctxkv)} fwd={len(fwd)} cand={len(cand)}")

# ============ Check 1-3: ctx write values (first dump, prefill) ============
c0 = ctxkv[0]
states = c0["states"].to(dev)
pos = c0["pos"].to(dev).long()
print(f"\n=== ctx write check: n={states.shape[0]} pos[:5]={pos[:5].tolist()} "
      f"is_neox(engine)={c0['is_neox']}")
normed = rms(states.float(), w["hidden_norm.weight"])
k_ref = (normed @ w["layers.0.self_attn.k_proj.weight"].float().T).view(-1, n_kv, hd)
v_ref = (normed @ w["layers.0.self_attn.v_proj.weight"].float().T).view(-1, n_kv, hd)
k_ref_n = rms(k_ref, w["layers.0.self_attn.k_norm.weight"])
print("k_prerope0 vs ref:", rel(c0["k_prerope0"].to(dev), k_ref_n))
print("v0        vs ref:", rel(c0["v0"].to(dev), v_ref))
for style in ("half", "interleaved"):
    k_roped = rope_rot(k_ref_n, pos, theta, style)
    print(f"k0 vs ref-rope[{style}]:", rel(c0["k0"].to(dev), k_roped))

# ============ Check 4: embeds ============
f0 = fwd[0]
ids = f0["input_ids"].to(dev).long()
emb_ref = w["model.embed_tokens.weight"][ids].float()
print(f"\n=== fwd0: ids={ids.tolist()} pos={f0['positions'].tolist()}")
print("embeds vs ref:", rel(f0["embeds"].to(dev), emb_ref))

# ============ Check 5: layer0 attention branch (read path) ============
# Accumulate engine-written ctx k/v by position, replaying dump order.
maxpos = max(int(c["pos"].max()) for c in ctxkv) + 1
k_store = torch.zeros(maxpos, n_kv, hd, device=dev)
v_store = torch.zeros(maxpos, n_kv, hd, device=dev)
w_store = torch.zeros(maxpos, dtype=torch.bool)

NFW = int(os.environ.get("NFWD", "6"))
ci = 0
for fi, fd in enumerate(fwd[:NFW]):
    # apply ctx writes for this step (ctxkv[fi] precedes fwd[fi] in propose)
    c = ctxkv[fi]
    p = c["pos"].to(dev).long()
    k_store[p] = c["k0"].to(dev).float()
    v_store[p] = c["v0"].to(dev).float()
    w_store[p] = True

    q_pos = fd["positions"].to(dev).long()
    B = q_pos.shape[0]
    T = int(q_pos[0].item())
    if not bool(w_store[:T].all()):
        print(f"fwd{fi}: ctx gap before T={T}, skip")
        continue
    embeds = fd["embeds"].to(dev).float()[None]  # [1, B, H]
    x = rms(embeds, w["layers.0.input_layernorm.weight"])
    aconv = Conv("layers.0.attention_conv")
    x, dyn = aconv.prepare(x)
    q = (x.float() @ w["layers.0.self_attn.q_proj.weight"].float().T).view(B, n_q, hd)
    k_q = (x.float() @ w["layers.0.self_attn.k_proj.weight"].float().T).view(B, n_kv, hd)
    v_q = (x.float() @ w["layers.0.self_attn.v_proj.weight"].float().T).view(B, n_kv, hd)
    q = rms(q, w["layers.0.self_attn.q_norm.weight"])
    k_q = rms(k_q, w["layers.0.self_attn.k_norm.weight"])
    q = rope_rot(q, q_pos, theta, "half")
    k_q = rope_rot(k_q, q_pos, theta, "half")
    k = torch.cat([k_store[:T], k_q], dim=0)  # engine-written ctx K + query K
    v = torch.cat([v_store[:T], v_q], dim=0)
    q_ = q[None].permute(0, 2, 1, 3)
    k_ = k[None].permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
    v_ = v[None].permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
    o = F.scaled_dot_product_attention(q_.float(), k_.float(), v_.float())
    o = o.permute(0, 2, 1, 3).reshape(1, B, -1)
    o = o @ w["layers.0.self_attn.o_proj.weight"].float().T
    o = aconv.finish(o, dyn)
    attn_branch_ref = o[0]
    attn_branch_eng = (fd["r0"].to(dev) - fd["embeds"].to(dev)).float()
    print(f"fwd{fi} T={T}: attn-branch(layer0) engine-vs-ref:",
          rel(attn_branch_eng, attn_branch_ref))

# ============ Check 7: head + selector given engine hidden ============
if cand:
    cd = cand[0]
    hid = cd["hidden"].to(dev).float()  # [reqs, k, H]
    logits = hid[0] @ w["lm_head.weight"].float().T
    topk = int(dc["selector_top_k"])
    unary_ref, cand_ref = torch.topk(logits, topk, dim=-1)
    eng_cand = cd["candidate_ids"][0].to(dev)
    overlap = [
        len(set(cand_ref[i].tolist()) & set(eng_cand[i].tolist())) for i in range(cand_ref.shape[0])
    ]
    print(f"\n=== cand0: top{topk} overlap per pos: {overlap}")
    print("unary engine vs ref (pos0):",
          rel(cd["unary"][0, 0].to(dev), unary_ref[0]) if eng_cand[0].tolist() == cand_ref[0].tolist() else "(id sets differ, skip)")
    # selector edge scores, pos0: anchor -> candidates
    hproj = hid[0] @ w["candidate_selector.hidden_projection.weight"].float().T
    anchor = int(cd["anchor_ids"][0])
    pred = w["candidate_selector.predecessor_codebook"].float()
    succ = w["candidate_selector.successor_codebook"].float()
    sc_ref = cd["unary"][0, 0].to(dev).float() + (pred[anchor] * hproj[0]) @ succ[eng_cand[0].long()].T
    print("selector scores pos0 engine vs ref:", rel(cd["scores"][0, 0, 0].to(dev), sc_ref))
