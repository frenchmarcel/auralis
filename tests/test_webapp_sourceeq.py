# -*- coding: utf-8 -*-
"""Per-octave source EQ: each octave gets its gain, the filter is minimum phase, and a run
with an EQ is the run without it, filtered (levels follow, G and the other sources do not)."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import sourceeq  # noqa: E402
from test_webapp_resume import _store_with_sim  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Bit-for-bit comparisons: the CPU tracer (the GPU's atomics are not deterministic,
    and a validation run may hold the GPU)."""
    monkeypatch.setenv("AURALIS_GPU", "0")

EQ = [6.0, 3.0, 0.0, -3.0, -6.0, -9.0, -12.0]


def test_clean_eq():
    assert sourceeq.clean_eq(None) is None and sourceeq.clean_eq([0] * 7) is None
    assert sourceeq.clean_eq(EQ) == EQ
    for bad in ([1, 2], [0, 0, 0, 0, 0, 0, 30], [0, 0, 0, float("nan"), 0, 0, 0],
                [0, 0, 0, 0, 0, 0, None], ["a"] * 7):
        with pytest.raises(ValueError):
            sourceeq.clean_eq(bad)


@pytest.mark.parametrize("eq", [EQ, [0, 0, 10, 0, 0, 0, 0], [0, 0, 24, -24, 0, 0, 0],
                                [-12, 0, 0, 0, 0, 0, 0]])
def test_each_octave_gets_its_gain_minimum_phase(eq):
    h = sourceeq.eq_fir(eq, 44100)
    assert np.allclose(sourceeq.band_gains_db(h, 44100), eq, atol=0.06)
    e = np.cumsum(h ** 2) / np.sum(h ** 2)
    # causal and compact: >99.9 % of the energy in 20 ms (98 % for a +-24 dB step)
    assert e[int(0.02 * 44100)] > (0.98 if max(np.abs(eq)) > 20 else 0.999)


def _run(eq):
    from webapp.server import runner
    store, m = _store_with_sim(tempfile.mkdtemp())
    if eq is not None:
        m["sources"][0]["eq_db"] = eq
        store.save_simulation(m)
    runner.run(store.root, m["id"])
    out = store.results_dir(m["id"])
    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    irs = {n: np.load(os.path.join(out, "irs", n)) for n in os.listdir(os.path.join(out, "irs"))}
    return meta, irs


def test_a_run_with_an_eq_is_the_run_without_it_filtered():
    flat, ir0 = _run(None)
    eqd, ir1 = _run(EQ)
    gains = sourceeq.band_gains_db(sourceeq.eq_fir(EQ, flat["fs"]), flat["fs"])
    for p0, p1 in zip(flat["pairs"], eqd["pairs"]):
        a, b = p0["metrics"], p1["metrics"]
        if p0["source_id"] == "S1":
            assert np.allclose(np.subtract(b["SPL_db"], a["SPL_db"]), gains, atol=1e-4)
            assert b["G_db"] == a["G_db"]              # a room quantity
        else:
            assert a == b                              # the other source is untouched
    h = sourceeq.eq_fir(EQ, flat["fs"])
    for n, a in ir0.items():
        if n.startswith("S1__") and not n.endswith(("_geo.npy", "_wave.npy")):
            want = sourceeq.filter_ir(a.astype(np.float64), h)
            assert np.allclose(ir1[n], want, atol=2e-6 * np.abs(want).max()), n
        elif n.startswith("S2__"):
            assert np.array_equal(ir1[n], a), n
