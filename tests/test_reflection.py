# -*- coding: utf-8 -*-
"""Tests for the angle-dependent complex specular reflection: Paris
formula, passive admittance fit, event amplitudes, interference comb
and legacy toggle."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine.bands import CENTERS, interp_band_gains  # noqa: E402
from auralis.engine.ir_synthesis import synthesize_mono  # noqa: E402
from auralis.engine.pipeline import (  # noqa: E402
    BakeConfig, ReceiverDef, SourceDef, bake,
)
from auralis.engine.reflection import (  # noqa: E402
    ALPHA_RAND_MAX, G_ALPHA_MAX, alpha_random, band_R_mag,
    build_admittance_table, event_reflection_spectrum,
    fit_admittance_random, invert_paris, paris_alpha_real, refl_coeff,
    _gamma_of,
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


def test_paris():
    gs = np.array([0.01, 0.1, 0.3, 0.637, 1.2])
    err = np.abs(paris_alpha_real(gs)
                 - np.array([alpha_random(g + 0j) for g in gs])).max()
    check("closed-form Paris = quadrature (err %.1e)" % err, err < 1e-9)
    check("Paris maximum: g=%.3f alpha=%.4f" % (G_ALPHA_MAX,
                                                ALPHA_RAND_MAX + 0.002),
          0.55 < G_ALPHA_MAX < 0.72 and 0.948 < ALPHA_RAND_MAX + 0.002 < 0.955)
    errs = [abs(paris_alpha_real(invert_paris(a)) - min(a, ALPHA_RAND_MAX))
            for a in (0.03, 0.15, 0.5, 0.9, 0.99)]
    check("round-trip inversion (max err %.1e)" % max(errs),
          max(errs) < 1e-4)


def test_fit_materials():
    cases = {
        "rideau": [0.07, 0.30, 0.50, 0.70, 0.75, 0.70, 0.65],
        "moquette": [0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.65],
        "panneau": [0.40, 0.30, 0.20, 0.17, 0.15, 0.10, 0.10],
        "beton": [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06],
    }
    worst = 0.0
    for name, a in cases.items():
        gi, A, l = fit_admittance_random(np.array(a), CENTERS)
        af = alpha_random(_gamma_of(CENTERS, gi, A, l))
        worst = max(worst, float(np.abs(
            af - np.clip(a, 0, ALPHA_RAND_MAX)).max()))
    check("two-family fit: max err %.3f over 4 materials" % worst,
          worst < 0.08)
    tab = build_admittance_table(np.array([cases["rideau"]]))
    anc = tab["anchor"][0]
    check("gentle energy anchors (%.2f..%.2f)" % (anc.min(), anc.max()),
          anc.min() > 0.7 and anc.max() < 1.4)
    # grazing: |R| -> 1 as mu -> 0, whatever alpha (the minimum of
    # |R|(mu) sits near mu ~ |gamma|; here we test the asymptote)
    R001 = abs(refl_coeff(0.01, 0, tab, np.array([1000.0]))[0])
    Rmin = min(abs(refl_coeff(m, 0, tab, np.array([1000.0]))[0])
               for m in (0.2, 0.35, 0.5))
    check("grazing: |R|(mu=0.01)=%.3f -> 1 (dip at %.2f)" % (R001, Rmin),
          R001 > 0.90 and R001 > Rmin + 0.2)


def _floor_scene():
    dead = {"name": "anecho", "absorption": [0.99] * 7,
            "scattering": [0.1] * 7, "transmission": [0.0] * 7}
    floor = {"name": "plaster",
             "absorption": [0.10, 0.10, 0.12, 0.10, 0.08, 0.06, 0.06],
             "scattering": [0.1] * 7, "transmission": [0.0] * 7}
    sc = make_shoebox((14.0, 12.0, 6.0), wall_material=dead,
                      floor_material=floor)
    src = SourceDef("S", np.array([5.0, 6.0, 1.6]))
    rcv = ReceiverDef("R", np.array([8.4, 6.0, 1.4]))
    return sc, src, rcv


def test_event_amplitudes():
    sc, src, rcv = _floor_scene()
    res = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=3000,
                                        normalize=False, seed=2,
                                        ism_order=1))
    tab = build_admittance_table(sc.absorption)
    d2 = float(np.hypot(3.4, 3.0))
    mu = 3.0 / d2
    ev = [e for e in res.events if e.kind == "specular" and e.bounces
          and e.bounces[0][0] == 0][0]
    ev_d = [e for e in res.events if e.kind == "direct"][0]
    check("recorded cos(theta) = geometry (%.4f)" % ev.bounces[0][1],
          abs(ev.bounces[0][1] - mu) < 1e-6)
    d1 = float(np.hypot(3.4, 0.2))
    airn = (ev_d.amps * d1) ** (d2 / d1)
    pred = band_R_mag(ev.bounces, tab) * airn / d2
    err = float(np.abs(ev.amps - pred).max())
    check("amplitudes = anchored |R(theta,f_b)| (err %.1e)" % err,
          err < 1e-12)
    check("complex_refl stat is on",
          res.stats.get("complex_refl", 0.0) == 1.0)


def test_interference_comb():
    """|H(f)| of the direct + floor reflection pair against the COMPLEX
    closed form: the interference dips (phase!) must land right."""
    sc, src, rcv = _floor_scene()
    res = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=3000,
                                        normalize=False, seed=2,
                                        ism_order=1))
    tab = build_admittance_table(sc.absorption)
    ev = [e for e in res.events if e.kind == "specular" and e.bounces
          and e.bounces[0][0] == 0][0]
    ev_d = [e for e in res.events if e.kind == "direct"][0]
    h = synthesize_mono([ev_d, ev], np.zeros_like(res.hist),
                        res.config.bin_dt, 48000, adm_table=tab)
    if isinstance(h, tuple):
        h = h[0]
    n = 1 << 16
    H = np.abs(np.fft.rfft(np.ravel(h), n))
    f = np.fft.rfftfreq(n, 1.0 / 48000)
    d1 = float(np.hypot(3.4, 0.2))
    d2 = float(np.hypot(3.4, 3.0))
    air1 = ev_d.amps * d1
    airn = air1 ** (d2 / d1)
    U = event_reflection_spectrum(f, ev.bounces, tab) \
        * interp_band_gains(f, band_R_mag(ev.bounces, tab))
    Hth = np.abs(interp_band_gains(f, air1)
                 * np.exp(-2j * np.pi * f * d1 / C) / d1
                 + U * interp_band_gains(f, airn)
                 * np.exp(-2j * np.pi * f * d2 / C) / d2)
    m = (f >= 300) & (f <= 4000)
    err = (20 * np.log10(np.maximum(H[m], 1e-9))
           - 20 * np.log10(np.maximum(Hth[m], 1e-9)))
    rms = float(np.sqrt(np.mean(err ** 2)))
    check("comb 300-4000 Hz: RMS err %.2f dB (tol 0.6)" % rms, rms < 0.6)
    fm = f[m]

    def dips(x):
        k = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
        return fm[k[np.argsort(x[k])][:5]]

    # every local minimum of the theory = the whole comb; each deep
    # measured dip (1500-3500 Hz) must land on one of them
    kt = np.where((Hth[m][1:-1] < Hth[m][:-2])
                  & (Hth[m][1:-1] < Hth[m][2:]))[0] + 1
    dt = fm[kt]
    band = (fm >= 1500) & (fm <= 3500)
    xm = H[m].copy()
    xm[~band] = xm.max()
    km = np.where((xm[1:-1] < xm[:-2]) & (xm[1:-1] < xm[2:]))[0] + 1
    dm = fm[km[np.argsort(xm[km])][:3]]
    off = max(float(np.min(np.abs(dt - v))) / v for v in dm)
    check("interference dips aligned (max gap %.2f %%)" % (100 * off),
          off < 0.015)


def test_legacy_toggle():
    sc, src, rcv = _floor_scene()
    res = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=3000,
                                        normalize=False, seed=2,
                                        ism_order=1,
                                        complex_reflection=False))
    ev = [e for e in res.events if e.kind == "specular"][0]
    check("legacy: no bounces, flag at 0",
          ev.bounces is None and res.stats.get("complex_refl") == 0.0)
    a = np.asarray(sc.absorption[0], dtype=np.float64)
    d1 = float(np.hypot(3.4, 0.2))
    d2 = float(np.hypot(3.4, 3.0))
    ev_d = [e for e in res.events if e.kind == "direct"][0]
    airn = (ev_d.amps * d1) ** (d2 / d1)
    pred = np.sqrt(1.0 - a) * airn / d2
    check("legacy: sqrt(1-alpha) amplitudes (err %.1e)"
          % np.abs(ev.amps - pred).max(),
          float(np.abs(ev.amps - pred).max()) < 1e-6)  # float32 tables


def test_diffuse_consistency():
    """The diffuse-field T20 stays the one of the tables: complex
    reflection on/off within 6 %."""
    mat = {"name": "abs15", "absorption": [0.15] * 7,
           "scattering": [0.30] * 7, "transmission": [0.0] * 7}
    sc = make_shoebox((8.0, 6.0, 3.5), wall_material=mat)
    src = SourceDef("S", np.array([2.0, 2.0, 1.5]))
    rcv = ReceiverDef("R", np.array([5.5, 4.0, 1.7]))
    from auralis.engine.metrics import schroeder_edc, _decay_time

    def t20(res):
        e = res.hist_total[2] + res.hist_total[3]
        db = 10 * np.log10(np.maximum(schroeder_edc(e)
                                      / schroeder_edc(e)[0], 1e-12))
        t = np.arange(len(db)) * res.config.bin_dt
        return _decay_time(db, t, -5.0, -25.0)

    r_on = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=20000,
                                         normalize=False, seed=3))
    r_off = bake(sc, src, rcv, BakeConfig(fs=48000, n_rays=20000,
                                          normalize=False, seed=3,
                                          complex_reflection=False))
    ta, tb = t20(r_on), t20(r_off)
    check("T20 on/off: %.3f vs %.3f s (gap %.1f %%)"
          % (ta, tb, 100 * abs(ta - tb) / tb),
          abs(ta - tb) / tb < 0.06)


def test_fit_multiangle():
    """Round trip: known passive gamma -> synthetic alpha(theta) at 4
    angles -> fit_admittance_multiangle -> alphas recovered."""
    from auralis.engine.reflection import fit_admittance_multiangle
    fb = np.geomspace(125.0, 5000.0, 17)
    jw = 1j * 2 * np.pi * fb
    g_true = 0.06 + 900.0 / (jw + 400.0) + 0.35 * jw / (jw + 6000.0)
    angs = [0.0, 30.0, 45.0, 60.0]
    A = []
    for th in angs:
        c = np.cos(np.radians(th))
        R = (c - g_true) / (c + g_true)
        A.append(1.0 - np.abs(R) ** 2)
    A = np.asarray(A)
    nd, nm = 2, 2
    gi, Ab, ll = fit_admittance_multiangle(A, angs, fb, f_hi=5000.0,
                                           f_lo=125.0, n_desc=nd,
                                           n_mont=nm)
    g = gi + sum(Ab[k] / (jw + ll[k]) for k in range(nd)) \
        + sum(Ab[nd + k] * jw / (jw + ll[nd + k]) for k in range(nm))
    errs = []
    for k, th in enumerate(angs):
        c = np.cos(np.radians(th))
        R = (c - g) / (c + g)
        errs.append(float(np.max(np.abs(1 - np.abs(R) ** 2 - A[k]))))
    check("multiangle: alpha recovered (max %.3f)" % max(errs),
          max(errs) < 0.05)
    check("multiangle: l >= floor",
          bool((ll >= 2 * np.pi * 8 - 1e-6).all()))


if __name__ == "__main__":
    print("== multi-angle fit ==")
    test_fit_multiangle()
    print("== Paris formula ==")
    test_paris()
    print("== two-family fit ==")
    test_fit_materials()
    print("== event amplitudes ==")
    test_event_amplitudes()
    print("== interference comb (phase) ==")
    test_interference_comb()
    print("== legacy toggle ==")
    test_legacy_toggle()
    print("== diffuse-field consistency ==")
    test_diffuse_consistency()
    print()
    print("%d ok, %d failures" % (_n_ok, _n_fail))
    sys.exit(1 if _n_fail else 0)
