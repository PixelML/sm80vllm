"""Reference DFlash2-architecture block drafter, in plain PyTorch, for training.

Trained weights are saved with the parameter names the fork's `DFlash2DraftModel`
path expects (`model.embed_tokens`, `model.layers.N.*`, `model.fc`, `model.norm`,
`lm_head`, plus the dflash2 conv and selector tensors), so a finished checkpoint
loads in vLLM with no shim -- only a `config.json` beside it.

Deliberately NOT built on vLLM's own module classes: those need a distributed
init and TP plumbing that make single-node training awkward, and the shapes are
simple enough to mirror exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    # dflash_config
    block_size: int = 8          # D = block_size - 1; widen to 13/17 later
    conv_kernel_size: int = 2    # "two-tap"
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    mask_token_id: int = 154856
    target_layer_ids: tuple[int, ...] = (5, 14, 24, 33, 42)
    num_target_layers: int = 45

    def to_hf(self) -> dict:
        return {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta},
            "sliding_window": self.sliding_window,
            "use_sliding_window": True,
            "max_window_layers": self.num_hidden_layers,
            "layer_types": ["sliding_attention"] * self.num_hidden_layers,
            "attention_bias": False,
            "hidden_act": "silu",
            "is_causal": False,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "num_target_layers": self.num_target_layers,
            "pad_token_id": 154820,
            "dflash_config": {
                "block_size": self.block_size,
                "conv_kernel_size": self.conv_kernel_size,
                "conv_group_size": self.conv_group_size,
                "selector_rank": self.selector_rank,
                "selector_top_k": self.selector_top_k,
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": list(self.target_layer_ids),
            },
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def rope(q, k, positions, theta: float, head_dim: int):
    half = head_dim // 2
    freqs = theta ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    ang = positions.float()[:, None] * freqs[None, :]
    cos, sin = ang.cos(), ang.sin()

    def rot(t):
        t1, t2 = t[..., :half], t[..., half:]
        c = cos[:, None, :].to(t.dtype)
        s = sin[:, None, :].to(t.dtype)
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)

    return rot(q), rot(k)


class GroupedConv(nn.Module):
    """Two-tap dynamic depthwise conv: each position mixes with its predecessor.

    Coefficients are a learned base kernel plus a per-group correction predicted
    from the hidden state. Position 0 of a block reads the last verified token,
    which in training is the position immediately before the block.
    """

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.taps = cfg.conv_kernel_size
        self.groups = cfg.hidden_size // cfg.conv_group_size
        self.group_size = cfg.conv_group_size
        self.block_size = cfg.block_size
        self.base_kernel = nn.Parameter(torch.zeros(self.taps, self.groups))
        with torch.no_grad():
            self.base_kernel[0].fill_(1.0)
        self.kernel_projection = nn.Linear(cfg.hidden_size, self.taps * self.groups, bias=False)

    def forward(self, x):  # x: [T, H]
        T, H = x.shape
        delta = self.kernel_projection(x).view(T, self.taps, self.groups)
        coeff = self.base_kernel.view(1, self.taps, self.groups) + delta
        blocks = x.view(T, self.groups, self.group_size)
        out = coeff[:, 0].unsqueeze(-1) * blocks
        pos = torch.arange(T, device=x.device) % self.block_size
        for tap in range(1, self.taps):
            shifted = torch.roll(blocks, shifts=tap, dims=0)
            # A block's first `tap` positions must not read across the block
            # boundary from a *later* block; they read the last verified token,
            # which teacher forcing already places immediately before.
            valid = (pos >= tap) | (torch.arange(T, device=x.device) >= tap)
            shifted = shifted * valid[:, None, None].to(shifted.dtype)
            out = out + coeff[:, tap].unsqueeze(-1) * shifted
        return out.view(T, H)


class Attention(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.window = cfg.sliding_window
        self.theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, x, positions, block_mask):
        T = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(T, self.nh, self.hd))
        k = self.k_norm(self.k_proj(x).view(T, self.nkv, self.hd))
        v = self.v_proj(x).view(T, self.nkv, self.hd)
        q, k = rope(q, k, positions, self.theta, self.hd)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=block_mask,
        )
        return self.o_proj(o.squeeze(0).transpose(0, 1).reshape(T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        g, u = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class Layer(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # two-tap conv before and after each sublayer (DFlash2)
        self.pre_attn_conv = GroupedConv(cfg)
        self.post_attn_conv = GroupedConv(cfg)
        self.pre_mlp_conv = GroupedConv(cfg)
        self.post_mlp_conv = GroupedConv(cfg)

    def forward(self, x, positions, mask):
        h = self.input_layernorm(x)
        h = self.pre_attn_conv(h)
        x = x + self.post_attn_conv(self.self_attn(h, positions, mask))
        h = self.post_attention_layernorm(x)
        h = self.pre_mlp_conv(h)
        return x + self.post_mlp_conv(self.mlp(h))


class Selector(nn.Module):
    """Low-rank bilinear scoring of adjacent candidate pairs (DFlash2 path selector)."""

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        r = cfg.selector_rank
        self.predecessor_codebook = nn.Parameter(torch.randn(cfg.vocab_size, r) * 0.02)
        self.successor_codebook = nn.Parameter(torch.randn(cfg.vocab_size, r) * 0.02)
        self.hidden_projection = nn.Linear(cfg.hidden_size, r, bias=False)

    def pair_scores(self, hidden, cand_prev, cand_next):
        """hidden [T,H]; cand_* [T,K] token ids -> [T,K,K] adjacency scores."""
        gate = self.hidden_projection(hidden)                    # [T,r]
        a = self.predecessor_codebook[cand_prev]                 # [T,K,r]
        b = self.successor_codebook[cand_next]                   # [T,K,r]
        return torch.einsum("tkr,tlr,tr->tkl", a, b, gate)


class DFlash2Drafter(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # aux fusion: concat of len(target_layer_ids) target states -> hidden
        self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.selector = Selector(cfg)

    def block_mask(self, T: int, device) -> torch.Tensor:
        """Causal across blocks, fully visible within a block.

        The block is predicted in ONE pass, so positions inside it must see each
        other -- that is what makes the drafter parallel and its cost O(1) in D.
        """
        B = self.cfg.block_size
        idx = torch.arange(T, device=device)
        blk = idx // B
        return (blk[None, :] <= blk[:, None]).view(1, 1, T, T)

    def forward(self, input_ids, aux, positions):
        """input_ids [T] (mask_token_id in drafted slots); aux [L,T,H]."""
        L, T, H = aux.shape
        x = self.embed_tokens(input_ids) + self.fc(aux.permute(1, 0, 2).reshape(T, L * H))
        mask = self.block_mask(T, x.device)
        for layer in self.layers:
            x = layer(x, positions, mask)
        return self.lm_head(self.norm(x))

    def export_state_dict(self, include_shared: bool = False) -> dict:
        """Rename to the layout the fork's DFlash2DraftModel loader expects.

        `embed_tokens` and `lm_head` are EXCLUDED by default. The reference
        checkpoint is 1.17B, which is exactly this model minus those two
        (verified: full 2.44B, minus embed+lm_head 1.17B), so the drafter shares
        the target's copies instead of shipping 1.27B of duplicates. Keep them
        only for a standalone/debug artifact.
        """
        skip = () if include_shared else ("embed_tokens", "lm_head")
        out = {}
        for k, v in self.state_dict().items():
            if skip and k.startswith(skip):
                continue
            if k.startswith(("embed_tokens", "fc", "layers", "norm")):
                out[f"model.{k}"] = v
            elif k.startswith("selector"):
                out[f"model.{k.split('.', 1)[1]}"] = v
            else:
                out[k] = v
        return out
