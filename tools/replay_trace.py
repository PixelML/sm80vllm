# Offline replay of ONLINE DFlash2 traces: reconstruct the draft's exact inputs
# from per-step engine dumps (VLLM_DFLASH_TRACE) and re-run the reference draft
# forward. If offline drafts == online drafts, the engine forward is faithful and
# the acceptance gap is in the stream content (fp8 numerics). If they differ, the
# engine has a runtime bug and the trace pinpoints where.
import glob
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

TRACE = "/root/.triton/dftrace"
DRAFT = "/dflash2"
TARGET_TOK = "/model"

dev = "cuda:0"
cfg = AutoConfig.from_pretrained(DRAFT)
dc = cfg.dflash_config
MASK_ID = dc["mask_token_id"]
BLOCK = dc["block_size"]

tok = AutoTokenizer.from_pretrained(TARGET_TOK)

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


def rms(x, weight, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def rope_rot(x, pos, theta, style="half"):
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=dev).float() / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    if style == "half":
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)
    # interleaved (GPT-J): pairs (0,1),(2,3),...
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out.to(x.dtype)


Q_STYLE = os.environ.get("REPLAY_QROPE", "half")
CTX_STYLE = os.environ.get("REPLAY_CTXROPE", "half")


def grouped_conv(hidden, dynamic, base, group_size):
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
        self.base = w[f"{prefix}.base_kernel"].float()
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


def draft_forward(ctx_fc, ctx_pos, q_ids, q_pos, return_logits=False):
    """ctx_fc [T, H] = engine's combine_hidden_states output (fc, pre-hidden_norm).
    ctx_pos [T] absolute positions. q_ids [B] anchor+masks. q_pos [B]."""
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    hd = cfg.head_dim
    theta = cfg.rope_parameters["rope_theta"] if isinstance(
        getattr(cfg, "rope_parameters", None), dict) else getattr(cfg, "rope_theta", 1e6)

    ctx = rms(ctx_fc.float(), w["hidden_norm.weight"])[None]  # [1, T, H]
    emb = w["model.embed_tokens.weight"]
    hs = emb[q_ids][None].float()
    B = q_ids.shape[0]

    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}"
        res = hs
        x = rms(hs, w[f"{p}.input_layernorm.weight"])
        aconv = Conv(f"{p}.attention_conv")
        x, dyn = aconv.prepare(x)
        q = (x.float() @ w[f"{p}.self_attn.q_proj.weight"].float().T).view(1, -1, n_q, hd)
        # context K/V from hidden_norm'd fc output; query K/V from layer stream
        k_ctx = (ctx.float() @ w[f"{p}.self_attn.k_proj.weight"].float().T).view(1, -1, n_kv, hd)
        v_ctx = (ctx.float() @ w[f"{p}.self_attn.v_proj.weight"].float().T).view(1, -1, n_kv, hd)
        k_q = (x.float() @ w[f"{p}.self_attn.k_proj.weight"].float().T).view(1, -1, n_kv, hd)
        v_q = (x.float() @ w[f"{p}.self_attn.v_proj.weight"].float().T).view(1, -1, n_kv, hd)
        q = rms(q, w[f"{p}.self_attn.q_norm.weight"])
        k_ctx = rms(k_ctx, w[f"{p}.self_attn.k_norm.weight"])
        k_q = rms(k_q, w[f"{p}.self_attn.k_norm.weight"])
        q = rope_rot(q.float(), q_pos, theta, Q_STYLE)
        k_ctx = rope_rot(k_ctx.float(), ctx_pos, theta, CTX_STYLE)
        k_q = rope_rot(k_q.float(), q_pos, theta, Q_STYLE)
        k = torch.cat([k_ctx, k_q], dim=1)
        v = torch.cat([v_ctx, v_q], dim=1)
        q_ = q.permute(0, 2, 1, 3)
        k_ = k.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        v_ = v.permute(0, 2, 1, 3).repeat_interleave(n_q // n_kv, dim=1)
        o = F.scaled_dot_product_attention(q_.float(), k_.float(), v_.float())
        o = o.permute(0, 2, 1, 3).reshape(1, B, -1)
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
    all_logits = hs[0].float() @ w["lm_head.weight"].float().T  # incl anchor row 0
    logits = all_logits[1:]
    topk = int(dc["selector_top_k"])
    unary, cand = torch.topk(logits, topk, dim=-1)
    hproj = hs[0, 1:].float() @ w["candidate_selector.hidden_projection.weight"].float().T
    pred_cb = w["candidate_selector.predecessor_codebook"].float()
    succ_cb = w["candidate_selector.successor_codebook"].float()
    prev = q_ids[0]
    path = []
    for pos in range(B - 1):
        scores = unary[pos] + (pred_cb[prev] * hproj[pos]) @ succ_cb[cand[pos]].T
        idx = torch.argmax(scores)
        prev = cand[pos][idx]
        path.append(int(prev))
    if return_logits:
        return path, all_logits, cand
    return path


# --- load traces, reconstruct context by replaying writes in step order ---
files = sorted(glob.glob(f"{TRACE}/trace_*.pt"))
print(f"{len(files)} trace steps")
H = cfg.hidden_size
maxpos = 0
traces = [torch.load(f, map_location="cpu") for f in files]
for t in traces:
    maxpos = max(maxpos, int(t["context_positions"].max()) + 1)
ctx_store = torch.zeros(maxpos, H, device=dev)
written = torch.zeros(maxpos, dtype=torch.bool)

MODE = os.environ.get("REPLAY_MODE", "after")  # after | before | drop8
DROPTAIL = int(os.environ.get("REPLAY_DROPTAIL", "0"))
QUIET = os.environ.get("REPLAY_QUIET") == "1"
pos_match_total = pos_total = 0

fc_err_max = 0.0
agree = disagree = 0
for t in traces:
    npos = t["context_positions"].long()
    comb = t["combined"].to(dev)
    # fc cross-check against cat_aux
    if t.get("cat_aux") is not None:
        ref = t["cat_aux"].to(dev).float() @ w["fc.weight"].float().T
        fc_err_max = max(fc_err_max, (ref - comb.float()).abs().max().item())
    if MODE != "before":
        ctx_store[npos] = comb.float()
        written[npos] = True

    q_ids = t["input_ids"].to(dev).long()
    q_pos = t["query_positions"].to(dev).long()
    T = int(q_pos[0].item())  # context = positions [0, anchor_pos)
    Teff = max(0, T - 8) if MODE == "drop8" else T
    Teff = max(0, Teff - DROPTAIL)
    if Teff < 1 or not bool(written[:Teff].all()):
        stat = "SKIP(ctx-gap)" if Teff >= 1 else "SKIP(T0)"
        print(f"step {t['step']:3d} {stat} T={T}")
        if MODE == "before":
            ctx_store[npos] = comb.float()
            written[npos] = True
        continue
    off_path, all_logits, cand = draft_forward(
        ctx_store[:Teff], torch.arange(Teff, device=dev), q_ids, q_pos, return_logits=True
    )
    if MODE == "before":
        ctx_store[npos] = comb.float()
        written[npos] = True
    online = t["draft_tokens"][0].tolist()
    match = sum(1 for a, b in zip(off_path, online) if a == b)
    tag = "MATCH" if match == len(online) else f"DIFF({match}/{len(online)})"
    if match == len(online):
        agree += 1
    else:
        disagree += 1
    pos_match_total += match
    pos_total += len(online)
    if not QUIET:
        row_top1 = all_logits.argmax(-1).tolist()  # 8 rows incl anchor row 0
        in_top16 = [int(o in cand[i].tolist()) for i, o in enumerate(online)]
        print(
            f"step {t['step']:3d} T={T:4d} anchor={q_ids[0].item():6d} {tag}\n"
            f"   online   {online} {tok.decode(online)!r}\n"
            f"   offline  {off_path} {tok.decode(off_path)!r}\n"
            f"   rowtop1  {row_top1} {tok.decode(row_top1)!r}\n"
            f"   online_in_off_top16 {in_top16}"
        )

print(f"\nfc max abs err: {fc_err_max:.4e}")
print(f"agree={agree} disagree={disagree} pos_match={pos_match_total}/{pos_total}")
