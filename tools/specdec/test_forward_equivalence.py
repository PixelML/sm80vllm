#!/usr/bin/env python3
"""Gate 2 -- forward equivalence of `drafter_v2` against the fork's OWN code.

Run inside the serving image of record:

  docker run --rm -v <worktree>/tools/specdec:/tools -v /models:/models \
    --entrypoint python3 ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905 \
    /tools/test_forward_equivalence.py --weights /models/model-cache/dflash2-glm53-flash/model.safetensors

Method. `DFlash2Qwen3ForCausalLM` cannot be *instantiated* on CPU (its
`Attention` layers need a backend and a KV cache), but its forward BODIES are
plain Python. So this test executes the fork's own unbound functions --

    vllm.model_executor.models.qwen3_dflash2.DFlashGroupedConv.prepare
    vllm.model_executor.models.qwen3_dflash2.DFlashGroupedConv.finish
    vllm.model_executor.models.qwen3_dflash2.DFlashGroupedConv._convolve
    vllm.model_executor.models.qwen3_dflash2._grouped_conv
    vllm.model_executor.models.qwen3_dflash2.DFlash2Qwen3DecoderLayer.forward
    vllm.model_executor.models.qwen3_dflash.DFlashQwen3Model.forward
    vllm.model_executor.models.qwen3_dflash2._score_edges

-- bound onto OUR parameter objects, and compares the result with our own
forward. Identical parameters, identical inputs: any difference is a
transcription error. The pieces that are genuinely ours (attention kernel, MLP)
are shared by both sides so they cancel; they are covered separately by the
RMSNorm and RoPE parity checks against vLLM's `forward_native`.
"""
import argparse
import sys
import types

import torch

# vLLM disables Triton with no active driver, leaving a placeholder whose
# attributes are None; one call site calls `tl.constexpr(...)` at import time.
import vllm.triton_utils as _tu
if getattr(_tu.tl, "constexpr", None) is None:
    _tu.tl.constexpr = lambda x: x

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models import qwen3_dflash as fork_dflash
from vllm.model_executor.models import qwen3_dflash2 as fork_dflash2

sys.path.insert(0, "/tools")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import drafter_v2 as ours  # noqa: E402

TOL_BF16 = 6e-2   # bf16 has ~3 decimal digits; compare in fp32 but allow slack
FAILS = []


def check(name, a, b, tol):
    d = float((a.float() - b.float()).abs().max())
    scale = float(a.float().abs().max()) or 1.0
    rel = d / scale
    ok = d <= tol or rel <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:<44} maxabs={d:.3e} rel={rel:.3e}")
    if not ok:
        FAILS.append(name)


def fork_conv_view(conv):
    """Our DFlashGroupedConv with the FORK's prepare/finish/_convolve bound on."""
    view = types.SimpleNamespace(
        base_kernel=conv.base_kernel,
        kernel_projection=conv.kernel_projection,
        block_size=conv.block_size,
        taps=conv.taps,
        group_size=conv.group_size,
        num_groups=conv.num_groups,
    )
    cls = fork_dflash2.DFlashGroupedConv
    view._convolve = types.MethodType(cls._convolve, view)
    view.prepare = types.MethodType(cls.prepare, view)
    view.finish = types.MethodType(cls.finish, view)
    return view


class ForkLayerView(torch.nn.Module):
    """Our layer's parameters, driven by the fork's DFlash2 layer forward."""

    def __init__(self, layer, ctx_k, ctx_v, attn_mask):
        super().__init__()
        self._layer = layer
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.attention_conv = fork_conv_view(layer.attention_conv)
        self.mlp_conv = fork_conv_view(layer.mlp_conv)
        self._ctx = (ctx_k, ctx_v, attn_mask)

    def self_attn(self, positions, hidden_states):
        ck, cv, mask = self._ctx
        return self._layer.self_attn(hidden_states, positions, ck, cv, mask)

    def forward(self, positions, hidden_states, residual):
        return fork_dflash2.DFlash2Qwen3DecoderLayer.forward(
            self, positions, hidden_states, residual
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/models/model-cache/dflash2-glm53-flash/model.safetensors")
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--ctx", type=int, default=37)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # ---- 1. primitive parity: _grouped_conv, _score_edges -----------------
    for bs in (8, 13, 17):
        t, h, gs, taps = bs * 3, 64, 16, 2
        g = h // gs
        hs = torch.randn(t, h)
        delta = torch.randn(t, taps, g)
        base = torch.randn(taps, h)
        check(
            f"_grouped_conv (block_size={bs})",
            fork_dflash2._grouped_conv(hs, delta, base, bs, g, gs, taps),
            ours._grouped_conv(hs, delta, base, bs, g, gs, taps),
            1e-6,
        )
    nb, nl, k, r, v = 2, 3, 4, 5, 100
    pt, st = torch.randn(v, r), torch.randn(v, r)
    cid = torch.randint(0, v, (nb, nl, k))
    ul, hid = torch.randn(nb, nl, k), torch.randn(nb, nl, r)
    anc = torch.randint(0, v, (nb,))
    check(
        "_score_edges",
        fork_dflash2._score_edges(pt, st, cid, ul, hid, anc, k),
        ours._score_edges(pt, st, cid, ul, hid, anc, k),
        1e-5,
    )

    # ---- 2. RMSNorm parity vs vLLM forward_native -------------------------
    h = 128
    vn, on = VllmRMSNorm(h, eps=1e-5), ours.RMSNorm(h, 1e-5)
    vn.weight.data.normal_()
    on.weight.data.copy_(vn.weight.data)
    x, res = torch.randn(9, h), torch.randn(9, h)
    check("RMSNorm (plain)", vn.forward_native(x.clone()), on(x.clone()), 1e-6)
    vh, vr = vn.forward_native(x.clone(), res.clone())
    oh, orr = on(x.clone(), res.clone())
    check("RMSNorm fused add (hidden)", vh, oh, 1e-6)
    check("RMSNorm fused add (residual)", vr, orr, 1e-6)

    # ---- 3. RoPE parity vs vLLM get_rope ----------------------------------
    hd = 128
    rope = get_rope(hd, max_position=4096,
                    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"})
    assert rope.is_neox_style, "drafter_v2 assumes neox-style RoPE"
    pos = torch.arange(11, 11 + 9)
    q, kk = torch.randn(9, 4 * hd), torch.randn(9, 2 * hd)
    rq, rk = rope.forward_native(pos, q.clone(), kk.clone())
    cos, sin = ours._rope_cos_sin(pos, 10000.0, hd, q.device)
    check("RoPE q", rq, ours._apply_rope(q.view(9, 4, hd), cos, sin, True).view(9, -1), 1e-5)
    check("RoPE k", rk, ours._apply_rope(kk.view(9, 2, hd), cos, sin, True).view(9, -1), 1e-5)

    # ---- 4. whole-block forward: ours vs the fork's own forward bodies ----
    from safetensors.torch import load_file
    w = load_file(args.weights)
    cfg = ours.DrafterConfig(num_speculative_tokens=args.depth)
    model = ours.DFlash2Drafter(cfg).float().eval()
    own = {k for k in model.state_dict() if not k.startswith(("embed_tokens", "lm_head"))}
    missing = own - set(w)
    extra = set(w) - own
    print(f"\nstrict-load: checkpoint {len(w)} tensors, drafter own set {len(own)}; "
          f"missing={len(missing)} extra={len(extra)}")
    if missing or extra:
        FAILS.append("strict-load")
        print("  missing:", sorted(missing)[:6])
        print("  extra:  ", sorted(extra)[:6])
    else:
        model.load_state_dict({**w, **{k: v for k, v in model.state_dict().items()
                                       if k.startswith(("embed_tokens", "lm_head"))}},
                              strict=True)
        print("  PASS  strict-load 81/81")

    nctx, d = args.ctx, args.depth
    ctx_states = torch.randn(nctx, cfg.hidden_size) * 0.5
    ctx_pos = torch.arange(nctx)
    q_ids = model.build_query_ids(1234, "cpu")
    q_pos = torch.arange(nctx, nctx + 1 + d)

    ctx_kv = model.precompute_context_kv(ctx_states, ctx_pos)
    mine = model.forward_block(q_ids, q_pos, ctx_kv, ctx_pos)

    duck = types.SimpleNamespace(
        embed_input_ids=model.embed_input_ids,
        norm=model.norm,
        layers=[
            ForkLayerView(layer, ck, cv, model.query_attn_mask(q_pos, ctx_pos))
            for layer, (ck, cv) in zip(model.layers, ctx_kv)
        ],
    )
    theirs = fork_dflash.DFlashQwen3Model.forward(duck, q_ids, q_pos)
    check("block forward (5 layers, fp32)", theirs, mine, 1e-5)

    # bf16 tolerance run
    mb = ours.DFlash2Drafter(cfg).to(torch.bfloat16).eval()
    mb.load_state_dict({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    ctx_kv_b = mb.precompute_context_kv(ctx_states.bfloat16(), ctx_pos)
    mine_b = mb.forward_block(q_ids, q_pos, ctx_kv_b, ctx_pos)
    duck_b = types.SimpleNamespace(
        embed_input_ids=mb.embed_input_ids,
        norm=mb.norm,
        layers=[
            ForkLayerView(layer, ck, cv, mb.query_attn_mask(q_pos, ctx_pos))
            for layer, (ck, cv) in zip(mb.layers, ctx_kv_b)
        ],
    )
    theirs_b = fork_dflash.DFlashQwen3Model.forward(duck_b, q_ids, q_pos)
    check("block forward (bf16)", theirs_b, mine_b, TOL_BF16)

    # ---- 4b. batched training path == per-block serving path -------------
    # The trainer flattens n_blocks query blocks of exactly conv_block_size rows.
    # That is only legitimate if it is bit-identical to running them one at a
    # time, which is what `_grouped_conv`'s per-block position reset buys.
    n_blk = 5
    anchors = torch.arange(nctx - n_blk, nctx)
    q_len = 1 + d
    bq_ids = torch.stack([model.build_query_ids(1000 + i, "cpu") for i in range(n_blk)])
    bq_pos = anchors[:, None] + torch.arange(q_len)[None, :]
    ctx_ok = ctx_pos[None, None, :] < anchors[:, None, None]
    ctx_ok = ctx_ok.expand(n_blk, q_len, nctx)
    qq = torch.ones(n_blk, q_len, q_len, dtype=torch.bool)
    bmask = torch.cat([ctx_ok, qq], dim=-1).unsqueeze(1)
    batched = model.forward_blocks(bq_ids, bq_pos, ctx_kv, bmask, n_blocks=n_blk)
    batched = batched.view(n_blk, q_len, -1)
    singles = []
    for i in range(n_blk):
        a = int(anchors[i])
        kv = [(k[:a], v[:a]) for k, v in ctx_kv]
        singles.append(model.forward_block(bq_ids[i], bq_pos[i], kv, ctx_pos[:a]))
    # fp32 GEMM/SDPA reduction order differs between a [B,h,Q,d] batch and B
    # separate [1,h,Q,d] calls, so this is a numerical-agreement check, not a
    # bitwise one. The structural claim -- no leakage across block boundaries --
    # is proved separately, and exactly, below.
    check("batched blocks ~= per-block", torch.stack(singles), batched, 2e-3)

    # Cross-block isolation: perturbing block 0's tokens must leave every other
    # block BITWISE unchanged. This is what makes flattening legitimate; if the
    # conv's per-block position reset or the attention mask were wrong, the
    # tap-1 shift or attention would carry block 0 into block 1 and this would
    # move. Exact zero is the only acceptable answer.
    poked = bq_ids.clone()
    poked[0, 0] = 4242
    batched2 = model.forward_blocks(poked, bq_pos, ctx_kv, bmask,
                                    n_blocks=n_blk).view(n_blk, q_len, -1)
    delta0 = float((batched2[0] - batched[0]).abs().max())
    delta_rest = float((batched2[1:] - batched[1:]).abs().max())
    ok = delta_rest == 0.0 and delta0 > 0.0
    print(f"{'PASS' if ok else 'FAIL'}  {'cross-block isolation (exact)':<44} "
          f"block0 moved={delta0:.3e} blocks1-4 moved={delta_rest:.3e}")
    if not ok:
        FAILS.append("cross-block isolation")

    # ---- 5. sanity: the hidden state must not explode ---------------------
    std = float(mine.float().std())
    print(f"\nfinal hidden std = {std:.4f}  (v2's bug produced 2.8e11)")
    if not (1e-3 < std < 1e3):
        FAILS.append("hidden magnitude")

    print("\n" + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL EQUIVALENCE CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    from vllm.config import DeviceConfig
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        main()
