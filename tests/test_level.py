# -*- coding: utf-8 -*-
"""Absolute calibration tests: SPL, A-weighting and strength factor G,
against free-field closed forms, Barron's revised theory in a diffuse
room and the absolute scale of the FDTD hybrid."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine.level import (  # noqa: E402
    A_WEIGHT_DB, band_energies, dba_total, level_stats, spl1m_from_power,
    spl_bands, strength_g,
)
from auralis.engine.metrics import schroeder_edc, _decay_time  # noqa: E402
from auralis.engine.pipeline import (  # noqa: E402
    BakeConfig, ReceiverDef, SourceDef, bake,
)

C = 343.0
_n_ok = 0
_n_fail = 0


def check(label, ok):
    global _n_ok, _n_fail
    if ok:
        _n_ok += 1
        print("  ok   %s " % label)
    else:
        _n_fail += 1
        print("  FAIL %s " % label)
        if "PYTEST_CURRENT_TEST" in os.environ:     # under pytest a failed check fails
            raise AssertionError(label)


def test_closed_forms():
    d = 2.5
    hist = np.zeros((7, 64))
    hist[:, 9] = 1.0 / d ** 2
    E = band_energies(hist)
    G, g_mid = strength_g(E)
    L = spl_bands(E, 94.0)
    check("exact free-field G (err %.1e dB)"
          % abs(g_mid - 20 * np.log10(10 / d)),
          abs(g_mid - 20 * np.log10(10 / d)) < 1e-9)
    check("exact free-field SPL (err %.1e dB)"
          % abs(L[3] - (94 - 20 * np.log10(d))),
          abs(L[3] - (94 - 20 * np.log10(d))) < 1e-9)
    check("L_W -> SPL@1m constant (%.3f, expected 95.00 +/- 0.01)"
          % spl1m_from_power(105.85),
          abs(spl1m_from_power(105.85) - 95.0) < 0.01)
    check("A-weighting: 1 kHz = 0, 125 Hz = -16.1",
          A_WEIGHT_DB[3] == 0.0 and A_WEIGHT_DB[0] == -16.1)
    flat = np.full(7, 94.0)
    la = dba_total(flat)
    la_ref = 10 * np.log10(np.sum(10 ** ((94 + A_WEIGHT_DB) / 10)))
    check("dBA energy sum (%.2f)" % la, abs(la - la_ref) < 1e-9)


def _anechoic_pair():
    mat = {"name": "anecho", "absorption": [0.99] * 7,
           "scattering": [0.1] * 7, "transmission": [0.0] * 7}
    sc = make_shoebox((12.0, 10.0, 5.0), wall_material=mat)
    src = SourceDef("S", np.array([4.0, 5.0, 2.5]), spl_1m=94.0)
    rcv = ReceiverDef("R", np.array([6.5, 5.0, 2.5]))
    return sc, src, rcv


def test_free_field_bake():
    sc, src, rcv = _anechoic_pair()
    res = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=20000,
                                        normalize=False, seed=7, ambisonic_order=1))
    d = 2.5
    g_th = 20 * np.log10(10 / d)
    check("anechoic bake: G_mid %.2f dB (theory %.2f, tol 0.15)"
          % (res.stats["g_mid"], g_th),
          abs(res.stats["g_mid"] - g_th) < 0.15)
    spl_th = 94 - 20 * np.log10(d)
    L = res.metrics["spl_bands"]
    check("SPL 1 kHz %.2f dB (theory %.2f, tol 0.15)" % (L[3], spl_th),
          abs(L[3] - spl_th) < 0.15)
    # air absorption: ~0.105 dB/m at 8 kHz -> -0.26 dB over 2.5 m
    drop = L[3] - L[6]
    check("air visible at 8 kHz (drop %.2f dB, expected ~0.26)" % drop,
          0.1 < drop < 0.45)
    # the omni channel (the binaural ear facing the source gets the head's gain), and its true
    # peak: the direct sound lands between two samples
    w = np.asarray(res.ambi[:, 0], dtype=np.float64)
    k = int(np.argmax(np.abs(w)))
    seg = w[max(k - 256, 0):k + 256]
    up = np.fft.irfft(np.fft.rfft(seg), 16 * len(seg)) * 16
    pk = float(np.abs(up).max())
    check("absolute waveform: omni true peak %.4f ~ 1/d = %.4f (tol 4%%)" % (pk, 1 / d),
          abs(pk * d - 1.0) < 0.04)


def test_normalization_invariance():
    sc, src, rcv = _anechoic_pair()
    r0 = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=8000,
                                       normalize=False, seed=11))
    r1 = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=8000,
                                       normalize=True, seed=11))
    g = 10.0 ** (r1.stats["norm_gain_db"] / 20.0)
    err = float(np.abs(r1.ir - r0.ir * g).max() / np.abs(r1.ir).max())
    check("IR_normalized = IR_absolute * gain (err %.1e)" % err, err < 1e-9)
    check("recorded gain consistent (ir_scale %.3f)" % r1.ir_scale,
          abs(r1.ir_scale - g) < 1e-12 and r1.ir_scale != 1.0)
    check("levels unaffected by normalization",
          abs(r1.stats["g_mid"] - r0.stats["g_mid"]) < 1e-12
          and abs(r1.stats["spl_dba"] - r0.stats["spl_dba"]) < 1e-12)
    check("listening peak at 0.985", abs(np.abs(r1.ir).max() - 0.985) < 1e-6)


def test_diffuse_room_barron():
    """Room 8x6x3.5, uniform alpha 0.15: G_mid against Barron's revised
    theory, with T20 measured by the engine itself (mid band)."""
    mat = {"name": "abs15", "absorption": [0.15] * 7,
           "scattering": [0.30] * 7, "transmission": [0.0] * 7}
    sc = make_shoebox((8.0, 6.0, 3.5), wall_material=mat)
    src = SourceDef("S", np.array([2.0, 2.0, 1.5]), spl_1m=94.0)
    rcv = ReceiverDef("R", np.array([5.5, 4.0, 1.7]))
    res = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=30000,
                                        normalize=False, seed=3))
    r = float(np.linalg.norm(rcv.position - src.position))
    V = 8.0 * 6.0 * 3.5
    # mid T20 from the echogram (Schroeder, -5..-25 dB)
    e_mid = res.hist_total[2] + res.hist_total[3]
    edc = schroeder_edc(e_mid)
    db = 10 * np.log10(np.maximum(edc / edc[0], 1e-12))
    t = np.arange(len(db)) * res.config.bin_dt
    T = _decay_time(db, t, -5.0, -25.0)
    g_pred = 10 * np.log10(100.0 / r ** 2
                           + (31200.0 * T / V) * np.exp(-0.04 * r / T))
    err = res.stats["g_mid"] - g_pred
    check("diffuse room: G_mid %.2f vs Barron %.2f (T20 %.2f s, gap "
          "%+.2f dB, tol 1.5)" % (res.stats["g_mid"], g_pred, T, err),
          abs(err) < 1.5)
    g_classic = 10 * np.log10(31200.0 * T / V)
    check("consistent with 10lg(T/V)+45 (%.2f, tol 2.5)" % g_classic,
          abs(res.stats["g_mid"] - g_classic) < 2.5)


def test_hybrid_absolute():
    """fdtd=True, normalize=False, nearly empty OPEN scene: the merge
    preserves the absolute scale (|H|*d ~ 1 in the low band)."""
    from auralis.engine.geometry import SceneData
    V = np.array([[15.0, -0.2, 0.8], [15.4, -0.2, 0.8],
                  [15.2, 0.2, 1.2]], float)
    sc = SceneData(tri_verts=V.reshape(1, 3, 3),
                   tri_mat=np.zeros(1, np.int32),
                   absorption=np.array([[0.03] * 7], np.float32),
                   scattering=np.full((1, 7), 0.1, np.float32),
                   transmission=np.zeros((1, 7), np.float32),
                   mat_names=["pebble"])
    src = SourceDef("S", np.array([0.0, 0.0, 1.6]), spl_1m=94.0)
    rcv = ReceiverDef("R", np.array([2.5, 0.0, 1.4]))
    kw = dict(fs=48000, n_rays=3000, seed=5, normalize=False,
              max_time=0.25)
    r_geo = bake(sc, src, rcv, BakeConfig(**kw))
    r_hyb = bake(sc, src, rcv, BakeConfig(**kw, fdtd=True,
                                          fdtd_fcross=120.0,
                                          fdtd_max_cells=6_000_000))
    d = float(np.linalg.norm(rcv.position - src.position))
    n = 1 << 15
    Hg = np.abs(np.fft.rfft(r_geo.ir[:, 0], n))
    Hh = np.abs(np.fft.rfft(r_hyb.ir[:, 0], n))
    f = np.fft.rfftfreq(n, 1.0 / 48000)
    m = (f >= 45.0) & (f <= 90.0)
    hd_low = float(np.mean(Hh[m])) * d
    check("absolute hybrid (open %s): low |H|*d 45-90 Hz = %.3f "
          "(tol [0.90, 1.08], fdtd %d cells)"
          % (bool(r_hyb.stats.get("fdtd_open", 0)), hd_low,
             r_hyb.stats.get("fdtd_cells", 0)),
          0.90 < hd_low < 1.08 and r_hyb.stats.get("fdtd_fc", 0) > 0)
    mh = (f >= 350.0) & (f <= 800.0)
    r_high = float(np.mean(Hh[mh]) / np.mean(Hg[mh]))
    check("hybrid transparent above fc: ratio 350-800 Hz %.4f (tol 1%%)"
          % r_high, abs(r_high - 1.0) < 0.01)
    check("zero normalization gain in absolute mode",
          abs(r_hyb.stats["norm_gain_db"]) < 1e-9)

if __name__ == "__main__":
    print("== closed forms ==")
    test_closed_forms()
    print("== near-anechoic bake ==")
    test_free_field_bake()
    print("== normalization invariance ==")
    test_normalization_invariance()
    print("== diffuse room (Barron) ==")
    test_diffuse_room_barron()
    print("== absolute FDTD hybrid ==")
    test_hybrid_absolute()
    print()
    print("%d ok, %d failures" % (_n_ok, _n_fail))
    sys.exit(1 if _n_fail else 0)
