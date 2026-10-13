# SPDX-License-Identifier: Apache-2.0
"""CPU checks for tools/glm5next_dense_q4.py (numpy only)."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parents[3] / "tools" / "glm5next_dense_q4.py"
_spec = importlib.util.spec_from_file_location("glm5next_dense_q4", _PATH)
q4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q4)


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    q = rng.integers(-8, 8, size=(64, 256)).astype(np.int8)
    p = q4.pack(q)
    assert p.dtype == np.int32 and p.shape == (64, 32)
    np.testing.assert_array_equal(q4.unpack(p), q)
    # nibble i of word j holds column 8j + i, offset by 8
    assert (p.view(np.uint32)[0, 0] & 0xF) == q[0, 0] + 8


def test_mse_clip_not_worse_than_minmax():
    rng = np.random.default_rng(1)
    w = rng.standard_t(4, size=(128, 1024)).astype(np.float32) * 0.02
    _, _, e_minmax = q4.quantize(w, mse=False)
    q, s, e_mse = q4.quantize(w, mse=True)
    assert e_mse <= e_minmax
    assert q.min() >= -8 and q.max() <= 7
    deq = q.reshape(128, -1, 128) * q4.bf16_to_f32(s)[..., None]
    rel = ((deq.reshape(w.shape) - w) ** 2).sum() / (w**2).sum()
    np.testing.assert_allclose(rel, e_mse, rtol=1e-4)
