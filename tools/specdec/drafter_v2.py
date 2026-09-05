"""DFlash2 block drafter, v2 — parameter layout matching the reference exactly.

v1 was a reimplementation from the blog description and did NOT match: 4 conv
modules per layer instead of 2, fused gate_up instead of split, no hidden_norm,
selector named `selector` instead of `candidate_selector`. Only 47 of 81/95
tensors were common, so v1's checkpoints could never load in vLLM's
`DFlash2DraftModel` path.

The reference shapes decode the design: `base_kernel [2, 2, H]` is
(pre/post, taps, hidden) and `kernel_projection [4*G, H]` is
(pre/post x taps x groups), so the two-tap conv IS applied before and after each
sublayer -- but the two kernels live in ONE module per sublayer, not two.

Acceptance test for this file is `--strict-load <reference>`: if their
checkpoint loads with strict=True, the layout is right by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DrafterConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 2048
    block_size: int = 8
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    mask_token_id: int = 154856
    target_layer_ids: tuple[int, ...] = (5, 14, 24, 33, 42)
    num_target_layers: int = 45

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dt) * self.weight


class DynamicConv(nn.Module):
    """Two-tap dynamic depthwise conv, holding BOTH the pre and post kernels.

    base_kernel: [2 (pre/post), taps, hidden]
    kernel_projection: [2 * taps * groups, hidden]
    """

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.taps = cfg.conv_kernel_size
        self.groups = cfg.num_groups
        self.group_size = cfg.conv_group_size
        self.block_size = cfg.block_size
        self.base_kernel = nn.Parameter(
            torch.zeros(2, self.taps, cfg.hidden_size)
        )
        with torch.no_grad():
            self.base_kernel[:, 0].fill_(1.0)
        self.kernel_projection = nn.Linear(
            cfg.hidden_size, 2 * self.taps * self.groups, bias=False
        )

    def forward(self, x, which: int):
        """which=0 -> pre-sublayer kernel, which=1 -> post-sublayer kernel."""
        T, H = x.shape
        delta = self.kernel_projection(x).view(T, 2, self.taps, self.groups)[:, which]
        base = self.base_kernel[which].view(self.taps, self.groups, self.group_size)
        blocks = x.view(T, self.groups, self.group_size)
        out = (base[0] + delta[:, 0].unsqueeze(-1)) * blocks
        pos = torch.arange(T, device=x.device) % self.block_size
        for tap in range(1, self.taps):
            shifted = torch.roll(blocks, shifts=tap, dims=0)
            valid = (pos >= tap) | (torch.arange(T, device=x.device) >= tap)
            shifted = shifted * valid[:, None, None].to(shifted.dtype)
            out = out + (base[tap] + delta[:, tap].unsqueeze(-1)) * shifted
        return out.view(T, H)


def rope(q, k, positions, theta: float, head_dim: int):
    half = head_dim // 2
    freqs = theta ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    ang = positions.float()[:, None] * freqs[None, :]
    cos, sin = ang.cos(), ang.sin()

    def rot(t):
        t1, t2 = t[..., :half], t[..., half:]
        c, s = cos[:, None, :].to(t.dtype), sin[:, None, :].to(t.dtype)
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)

    return rot(q), rot(k)


class Attention(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.nh, self.nkv, self.hd = (
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
        self.theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, x, positions, mask):
        T = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(T, self.nh, self.hd))
        k = self.k_norm(self.k_proj(x).view(T, self.nkv, self.hd))
        v = self.v_proj(x).view(T, self.nkv, self.hd)
        q, k = rope(q, k, positions, self.theta, self.hd)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), attn_mask=mask)
        return self.o_proj(o.squeeze(0).transpose(0, 1).reshape(T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention_conv = DynamicConv(cfg)
        self.mlp_conv = DynamicConv(cfg)

    def forward(self, x, positions, mask):
        h = self.attention_conv(self.input_layernorm(x), 0)
        x = x + self.attention_conv(self.self_attn(h, positions, mask), 1)
        h = self.mlp_conv(self.post_attention_layernorm(x), 0)
        return x + self.mlp_conv(self.mlp(h), 1)


class CandidateSelector(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        r = cfg.selector_rank
        self.predecessor_codebook = nn.Parameter(torch.randn(cfg.vocab_size, r) * 0.02)
        self.successor_codebook = nn.Parameter(torch.randn(cfg.vocab_size, r) * 0.02)
        self.hidden_projection = nn.Linear(cfg.hidden_size, r, bias=False)

    def pair_scores(self, hidden, cand_prev, cand_next):
        gate = self.hidden_projection(hidden)
        a = self.predecessor_codebook[cand_prev]
        b = self.successor_codebook[cand_next]
        return torch.einsum("tkr,tlr,tr->tkl", a, b, gate)


class DFlash2Drafter(nn.Module):
    """Trainable set only. embed_tokens and lm_head are the TARGET's, shared and
    frozen, and are never exported -- which is why the reference is 1.17B."""

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(
            len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.candidate_selector = CandidateSelector(cfg)
        # shared, frozen, not part of state_dict
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    def block_mask(self, T: int, device):
        B = self.cfg.block_size
        blk = (torch.arange(T, device=device) // B)
        return (blk[None, :] <= blk[:, None]).view(1, 1, T, T)

    def forward(self, input_ids, aux, positions):
        L, T, H = aux.shape
        x = self.embed_tokens(input_ids) + self.hidden_norm(
            self.fc(aux.permute(1, 0, 2).reshape(T, L * H)))
        mask = self.block_mask(T, x.device)
        for layer in self.layers:
            x = layer(x, positions, mask)
        return self.lm_head(self.norm(x))

    def export_state_dict(self) -> dict:
        skip = ("embed_tokens", "lm_head")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(skip)}
