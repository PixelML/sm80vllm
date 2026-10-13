#!/usr/bin/env python3
"""Re-quantize a GLM-5.3-Flash W4A16 checkpoint's BF16 dense Linear weights
(KDA q/k/v/o, MLA q_b/o, shared experts, dense MLP) to the compressed-tensors
pack-quantized int4 scheme (symmetric, group 128) its routed experts already
use. Each group's scale is the one of 16 clippings (1.0 down to 0.55 of its
absmax) with the least squared error.

Credit: the MSE-searched 4-bit dense weights come from patch
0005-glm-dense-q4 (``quantize4_mse``) of MiaAI-Lab's
https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold
@ed026ef92d1650120dada1294a112acb6c8f2f48, a modification of TensorFold by
Ash Hart (@ashhart, https://github.com/ashhart/TensorFold). This port keeps
their clip grid and moves it to symmetric group-128 int4 for Marlin on sm80.

Untouched tensors are copied as raw byte ranges; only targets are decoded.
Needs only numpy. The engine side is in vllm/models/glm5next/common (KDA
in_proj split, MLA quant_config, packed_modules_mapping).

  check-pack SRC            unpack a real expert, repack, require bit-exact
  report SRC [--layers N]   per-target error: MSE clip vs plain min-max
  write SRC OUT             write the new checkpoint (shards, index, config)
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time

import numpy as np

GROUP, BITS, PACK = 128, 4, 8
CLIPS = np.linspace(1.0, 0.55, 16, dtype=np.float32)
DEFAULT_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_b_proj",
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
DTYPES = {
    "BF16": (np.uint16, 2),
    "F32": (np.float32, 4),
    "I32": (np.int32, 4),
    "I64": (np.int64, 8),
    "F16": (np.float16, 2),
    "U8": (np.uint8, 1),
    "I8": (np.int8, 1),
}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read(path, h, base, name):
    t = h[name]
    dt, _ = DTYPES[t["dtype"]]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=dt).reshape(t["shape"])


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):  # round to nearest even
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def pack(
    q,
):  # q int in [-8, 7], [n, k] -> int32 [n, k/8]; nibble i of word j = column 8j+i
    u = (q + 8).astype(np.uint32).reshape(q.shape[0], -1, PACK)
    out = np.zeros(u.shape[:2], dtype=np.uint32)
    for i in range(PACK):
        out |= u[:, :, i] << (BITS * i)
    return out.view(np.int32)


def unpack(p):
    u = p.view(np.uint32)
    q = np.stack([(u >> (BITS * i)) & 0xF for i in range(PACK)], axis=-1)
    return q.reshape(p.shape[0], -1).astype(np.int8) - 8


def quantize(w, mse=True):
    """W float32 [n, k] -> (q int8 [n, k], scale bf16-as-u16 [n, k/128], rel err)."""
    n, k = w.shape
    g = w.reshape(n, k // GROUP, GROUP)
    amax = np.abs(g).max(-1)
    best_err = best_s = None
    for c in CLIPS if mse else CLIPS[:1]:
        s = bf16_to_f32(
            f32_to_bf16(np.maximum(amax * c / 7.5, 1e-12))
        )  # the scale as stored
        q = np.clip(np.rint(g / s[..., None]), -8, 7)
        err = ((q * s[..., None] - g) ** 2).sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_s = np.where(better, s, best_s)
    q = np.clip(np.rint(g / best_s[..., None]), -8, 7).astype(np.int8).reshape(n, k)
    rel = float(best_err.sum() / max((g**2).sum(), 1e-30))
    return q, f32_to_bf16(best_s), rel


def targets_of(index, num_layers, targets):
    pat = re.compile(
        r"layers\.(\d+)\.(" + "|".join(re.escape(t) for t in targets) + r")\.weight$"
    )
    out = {}
    for name, shard in index.items():
        m = pat.search(name)
        if m and int(m.group(1)) < num_layers:
            out[name] = shard
    return out


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_meta(src):
    cfg = load_json(os.path.join(src, "config.json"))
    tc = cfg.get("text_config", cfg)
    index = load_json(os.path.join(src, "model.safetensors.index.json"))
    return cfg, tc, index


def cmd_check_pack(src):
    _, _, index = load_meta(src)
    names = [
        k
        for k in index["weight_map"]
        if k.endswith("experts.0.gate_proj.weight_packed")
    ][:3]
    for name in names:
        path = os.path.join(src, index["weight_map"][name])
        h, base = header(path)
        p = read(path, h, base, name)
        s = bf16_to_f32(
            read(path, h, base, name.replace("weight_packed", "weight_scale"))
        )
        q = unpack(p)
        w = q.astype(np.float32).reshape(q.shape[0], -1, GROUP) * s[..., None]
        q2 = np.clip(np.rint(w / s[..., None]), -8, 7).astype(np.int8).reshape(q.shape)
        same = np.array_equal(pack(q2), p)
        hist = np.bincount((q + 8).ravel(), minlength=16)
        gmax = np.abs(q.reshape(q.shape[0], -1, GROUP)).max(-1)
        print(
            f"{name}: repack bit-exact={same}  q range [{q.min()},{q.max()}]  "
            f"groups hitting |q|>=7: {(gmax >= 7).mean():.3f}  hist {hist.tolist()}"
        )
        if not same:
            sys.exit(1)


def cmd_report(src, layers):
    _, tc, index = load_meta(src)
    tg = targets_of(index["weight_map"], tc["num_hidden_layers"], DEFAULT_TARGETS)
    names = sorted(tg, key=lambda x: (int(re.search(r"layers\.(\d+)", x).group(1)), x))
    keep = set(
        range(0, tc["num_hidden_layers"], max(1, tc["num_hidden_layers"] // layers))
    )
    for name in names:
        if int(re.search(r"layers\.(\d+)", name).group(1)) not in keep:
            continue
        path = os.path.join(src, tg[name])
        h, base = header(path)
        w = bf16_to_f32(read(path, h, base, name))
        t0 = time.time()
        _, _, e0 = quantize(w, mse=False)
        _, _, e1 = quantize(w, mse=True)
        print(
            f"{name:70s} {str(list(w.shape)):14s} "
            f"rel MSE minmax {e0:.5f}  mse-clip {e1:.5f}  "
            f"({e1 / e0:.2f}x)  {time.time() - t0:.1f}s",
            flush=True,
        )


def write_shard(src_path, dst_path, replace):
    """Copy a shard, replacing each name in ``replace`` (name -> dict of new arrays)."""
    h, base = header(src_path)
    meta = h.pop("__metadata__", None)
    new_h, chunks, off = {}, [], 0
    for name in sorted(h, key=lambda x: h[x]["data_offsets"][0]):
        if name in replace:
            for nn, arr in replace[name].items():
                dt = {
                    np.dtype(np.int32): "I32",
                    np.dtype(np.uint16): "BF16",
                    np.dtype(np.int64): "I64",
                }[arr.dtype]
                b = np.ascontiguousarray(arr).tobytes()
                new_h[nn] = {
                    "dtype": dt,
                    "shape": list(arr.shape),
                    "data_offsets": [off, off + len(b)],
                }
                chunks.append(("mem", b))
                off += len(b)
        else:
            a, b = h[name]["data_offsets"]
            new_h[name] = dict(h[name], data_offsets=[off, off + b - a])
            chunks.append(("file", (base + a, b - a)))
            off += b - a
    if meta is not None:
        new_h["__metadata__"] = meta
    hb = json.dumps(new_h, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(src_path, "rb") as fi, open(dst_path + ".tmp", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for kind, c in chunks:
            if kind == "mem":
                fo.write(c)
            else:
                fi.seek(c[0])
                left = c[1]
                while left:
                    buf = fi.read(min(left, 256 << 20))
                    fo.write(buf)
                    left -= len(buf)
    os.replace(dst_path + ".tmp", dst_path)
    return {n: os.path.basename(dst_path) for n in new_h if n != "__metadata__"}


def cmd_write(src, out):
    cfg, tc, index = load_meta(src)
    wm = index["weight_map"]
    tg = targets_of(wm, tc["num_hidden_layers"], DEFAULT_TARGETS)
    os.makedirs(out, exist_ok=True)
    by_shard = {}
    for name, shard in tg.items():
        by_shard.setdefault(shard, []).append(name)
    new_wm = {k: v for k, v in wm.items() if v not in by_shard}
    errs = []
    for shard in sorted(set(wm.values())):
        dst = os.path.join(out, shard)
        if shard not in by_shard:
            if not os.path.exists(dst):
                os.symlink(os.path.join(os.path.abspath(src), shard), dst)
            continue
        h, base = header(os.path.join(src, shard))
        rep = {}
        for name in by_shard[shard]:
            w = bf16_to_f32(read(os.path.join(src, shard), h, base, name))
            q, s, e = quantize(w)
            stem = name[: -len(".weight")]
            rep[name] = {
                stem + ".weight_packed": pack(q),
                stem + ".weight_scale": s,
                stem + ".weight_shape": np.array(w.shape, dtype=np.int64),
            }
            errs.append(e)
        t0 = time.time()
        new_wm.update(write_shard(os.path.join(src, shard), dst, rep))
        print(
            f"{shard}: {len(rep)} tensors re-quantized, "
            f"written in {time.time() - t0:.0f}s",
            flush=True,
        )
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f in (
            "config.json",
            "model.safetensors.index.json",
        ):
            continue
        dst = os.path.join(out, f)
        if not os.path.exists(dst):
            (shutil.copytree if os.path.isdir(os.path.join(src, f)) else shutil.copy2)(
                os.path.join(src, f), dst
            )
    index["weight_map"] = new_wm
    dump_json(index, os.path.join(out, "model.safetensors.index.json"))
    qc = cfg.get("quantization_config") or tc["quantization_config"]
    fused = ("mlp.gate_up_proj", "mlp.shared_experts.gate_up_proj")
    mods = "|".join(re.escape(t) for t in DEFAULT_TARGETS + fused)
    drop = re.compile(r"layers\.(\d+)\.(" + mods + r")$")

    def kept(entry):
        m = drop.search(entry)
        return not (m and int(m.group(1)) < tc["num_hidden_layers"])

    before = len(qc["ignore"])
    qc["ignore"] = [e for e in qc["ignore"] if kept(e)]
    dump_json(cfg, os.path.join(out, "config.json"))
    print(
        f"ignore {before} -> {len(qc['ignore'])}; {len(errs)} tensors; "
        f"mean rel MSE {np.mean(errs):.5f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check-pack", "report", "write"])
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    {
        "check-pack": lambda: cmd_check_pack(a.src),
        "report": lambda: cmd_report(a.src, a.layers),
        "write": lambda: cmd_write(a.src, a.out),
    }[a.cmd]()
