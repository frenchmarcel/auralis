# -*- coding: utf-8 -*-
"""Coherent sum of sources: linear in pressure - two identical sources give
+6.02 dB and the same decay, opposite polarity cancels, a delay shifts the arrival."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import combine  # noqa: E402
from webapp.server.store import default_settings  # noqa: E402


def _results(fs=48000):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "irs"))
    rng = np.random.default_rng(2)
    t = np.arange(int(1.2 * fs)) / fs
    h = rng.standard_normal(len(t)) * np.exp(-6.91 * t / 0.8)
    h[:200] = 0.0
    h[200] = 5.0
    ambi = np.zeros((len(t), 9), np.float32)
    ambi[:, 0] = h
    for sid in ("A", "B"):
        np.save(os.path.join(d, "irs", "%s__R1.npy" % sid), np.stack([h, h], 1).astype(np.float32))
        np.save(os.path.join(d, "irs", "%s__R1_ambi.npy" % sid), ambi)
    pair = {"metrics": {"G_db": [10.0] * 7, "SPL_db": [84.0] * 7}}
    meta = {"fs": fs, "pairs": [dict(pair, source_id="A", receiver_id="R1"),
                                dict(pair, source_id="B", receiver_id="R1")]}
    return d, meta


def test_two_identical_sources_add_6_db_and_keep_the_decay():
    d, meta = _results()
    st = default_settings()
    _k, one = combine.combine(d, meta, [{"source_id": "A"}], "R1", st)
    _k, two = combine.combine(d, meta, [{"source_id": "A"}, {"source_id": "B"}], "R1", st)
    assert np.allclose(two["metrics"]["SPL_db"], np.array(one["metrics"]["SPL_db"]) + 6.0206, atol=1e-3)
    assert np.allclose(two["metrics"]["T30"], one["metrics"]["T30"], rtol=1e-6)
    assert one["metrics"]["SPL_db"][3] == pytest.approx(84.0, abs=1e-6)


def test_polarity_and_delay():
    d, meta = _results()
    st = default_settings()
    _k, inv = combine.combine(d, meta, [{"source_id": "A"}, {"source_id": "B", "polarity": -1}],
                              "R1", st)
    assert all(v is None or v < -200 for v in inv["metrics"]["SPL_db"])
    key, dl = combine.combine(d, meta, [{"source_id": "A", "delay_ms": 10.0}], "R1", st)
    o = np.load(os.path.join(d, "combined", key, "omni.npy"))
    assert int(np.argmax(np.abs(o))) == 200 + 480
    with pytest.raises(ValueError):
        combine.combine(d, meta, [{"source_id": "A", "delay_ms": -1}], "R1", st)
