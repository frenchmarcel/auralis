# -*- coding: utf-8 -*-
"""Physical material tests: exact air layer, transfer-matrix
composition, JCA and Miki limits, passivity, gamma_normal consistency
and FDTD walls against the TMM."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import poromat as pm
from auralis.engine.bands import CENTERS

_n_ok = 0
_n_fail = 0


def check(label, cond, extra=""):
    global _n_ok, _n_fail
    if cond:
        _n_ok += 1
        print("  ok   %s %s" % (label, extra))
    else:
        _n_fail += 1
        print("  FAIL %s %s" % (label, extra))


def test_air_layer_exact():
    print("[air] 10 cm air layer on a rigid backing")
    d = 0.10
    f = np.array([60.0, 137.0, 313.0, 851.0, 1700.0, 4211.0])
    spec = {"layers": [{"model": "air", "d": d}]}
    Zs = pm.surface_impedance(spec, f, 1.0)
    k0 = 2.0 * np.pi * f / pm.C0
    ref = -1j * pm.Z0 / np.tan(k0 * d)
    err = float(np.max(np.abs(Zs - ref) / np.abs(ref)))
    check("Zs = -j Z0 cot(k0 d) (err %.1e)" % err, err < 1e-10)
    R = pm.reflection(spec, f, 1.0)
    check("|R| = 1 with no losses (err %.1e)"
          % float(np.max(np.abs(np.abs(R) - 1.0))),
          float(np.max(np.abs(np.abs(R) - 1.0))) < 1e-10)


def test_tmm_composition():
    print("[tmm] layer composition")
    f = np.geomspace(50.0, 8000.0, 40)
    a = {"layers": [{"model": "air", "d": 0.037},
                    {"model": "air", "d": 0.063}]}
    b = {"layers": [{"model": "air", "d": 0.100}]}
    for mu in (1.0, 0.62, 0.15):
        Za = pm.surface_impedance(a, f, mu)
        Zb = pm.surface_impedance(b, f, mu)
        err = float(np.max(np.abs(Za - Zb) / np.maximum(np.abs(Zb), 1.0)))
        check("air 37+63 == air 100 (mu=%.2f, err %.1e)" % (mu, err),
              err < 1e-9)


def test_lossless_oblique():
    print("[tmm] |R(theta)| = 1 for a lossless stack")
    f = np.geomspace(80.0, 6000.0, 25)
    spec = {"layers": [{"model": "air", "d": 0.05},
                       {"model": "air", "d": 0.11}]}
    worst = 0.0
    for mu in (1.0, 0.8, 0.5, 0.2, 0.05):
        R = pm.reflection(spec, f, mu)
        worst = max(worst, float(np.max(np.abs(np.abs(R) - 1.0))))
    check("|R| = 1 within 1e-9 (worst %.1e)" % worst, worst < 1e-9)


def test_jca_limits():
    print("[jca] analytic limits")
    phi, sig, ai, lv, lt = 0.96, 20000.0, 1.4, 90e-6, 180e-6
    # Darcy: j w rho_eff -> sigma as f -> 0
    f = np.array([0.5])
    Zc, k = pm.jca(f, phi, sig, ai, lv, lt)
    w = 2.0 * np.pi * f
    rho_eff = Zc * k / w                    # rho_eff by definition
    # surface velocity convention: j w rho_eff -> sigma as is
    darcy = 1j * w * rho_eff
    err = float(np.abs(darcy[0] - sig) / sig)
    check("Darcy: |j w rho phi - sigma|/sigma = %.3f" % err, err < 0.02)
    # high frequency: k -> k0 sqrt(alpha_inf) (1/sqrt(w) convergence:
    # we test the asymptote of the FORMULA, very high up)
    f = np.array([3.0e6])
    _, k = pm.jca(f, phi, sig, ai, lv, lt)
    k0 = 2.0 * np.pi * f / pm.C0
    ratio = float(np.abs(k[0]) / (k0[0] * np.sqrt(ai)))
    check("asymptote k/k0 = sqrt(alpha_inf) within 3%% (ratio %.3f)" % ratio,
          abs(ratio - 1.0) < 0.03)
    # attenuation: Im(k) < 0 everywhere
    f = np.geomspace(20.0, 20000.0, 50)
    _, k = pm.jca(f, phi, sig, ai, lv, lt)
    check("Im(k) < 0 (attenuation)", bool((k.imag < 0).all()))


def test_miki_asymptote():
    print("[miki] asymptote and absorption")
    sig = 10000.0
    f = np.array([50.0 * sig])              # X = 50 >> 1
    Zc, k = pm.miki(f, sig)
    check("Zc -> Z0 when f >> sigma (|Zc/Z0-1| = %.3f)"
          % float(np.abs(Zc[0] / pm.Z0 - 1.0)),
          float(np.abs(Zc[0] / pm.Z0 - 1.0)) < 0.02)
    spec = pm.spec_from_params("miki", sig, 50.0)
    ab = pm.alpha_bands(spec)
    check("absorption rises then saturates (125: %.2f, 2k: %.2f)"
          % (ab[0], ab[4]), ab[0] < 0.35 and 0.55 < ab[4] <= 1.0
          and ab[4] > ab[1])


def test_passivity_sweep():
    print("[passivity] random sweep of specifications")
    rng = np.random.default_rng(7)
    f = np.geomspace(30.0, 12000.0, 60)
    worst_R, worst_re = 0.0, 0.0
    for _ in range(24):
        model = "miki" if rng.random() < 0.5 else "jca"
        spec = pm.spec_from_params(
            model, float(rng.uniform(1000, 200000)),
            float(rng.uniform(5, 200)),
            air_gap_mm=float(rng.choice([0.0, rng.uniform(5, 300)])),
            phi=float(rng.uniform(0.7, 0.99)),
            alpha_inf=float(rng.uniform(1.0, 2.5)),
            lam_v_um=float(rng.uniform(20, 400)),
            lam_t_um=float(rng.uniform(40, 800)))
        for mu in (1.0, 0.6, 0.25, 0.05):
            R = pm.reflection(spec, f, mu)
            worst_R = max(worst_R, float(np.max(np.abs(R))) - 1.0)
        Zs = pm.surface_impedance(spec, f, 1.0)
        fin = np.isfinite(Zs.real)
        if fin.any():
            worst_re = min(worst_re, float(Zs.real[fin].min()) / pm.Z0)
    check("|R| <= 1 everywhere (worst excess %.1e)" % worst_R,
          worst_R < 1e-9)
    check("Re(Zs) >= 0 everywhere (worst %.1e Z0)" % worst_re,
          worst_re > -1e-9)


def test_gamma_consistency():
    print("[gamma] admittance target == normal reflection")
    f = np.geomspace(40.0, 8000.0, 30)
    spec = pm.spec_from_params("jca", 15000.0, 60.0, air_gap_mm=40.0)
    g = pm.gamma_normal(spec, f)
    R_g = (1.0 - g) / (1.0 + g)
    R = pm.reflection(spec, f, 1.0)
    err = float(np.max(np.abs(R_g - R)))
    check("R((1-g)/(1+g)) == exact R (err %.1e)" % err, err < 1e-10)


def test_fit_wall_discrete():
    """Fit alone: the 2-branch passive form, evaluated by the exact
    discrete model (wall_R_discrete), reaches the TMM target."""
    import auralis.engine.reflection as rf
    C = 343.0
    dx = C / (9.0 * 300.0)
    dt = (1.0 / np.sqrt(3.0)) * dx / C
    fgrid = np.geomspace(30.0, 360.0, 48)
    probes = np.array([60.0, 90.0, 130.0, 160.0])
    for nom, spec, tol in (
            ("solid 100mm", pm.spec_from_params("miki", 12000.0, 100.0),
             0.05),
            ("gap 50+150", pm.spec_from_params("miki", 15000.0, 50.0,
                                               air_gap_mm=150.0),
             0.075)):
        gi, A, l = rf.fit_admittance_complex(
            fgrid, pm.gamma_normal(spec, fgrid),
            l_max=0.6 / dt, dt=dt, lam=1.0 / np.sqrt(3.0))
        Rp = np.abs(rf.wall_R_discrete(probes, gi, A, l, dt,
                                       1.0 / np.sqrt(3.0)))
        Rt = np.abs(pm.reflection(spec, probes, 1.0))
        err = float(np.max(np.abs(Rp - Rt)))
        check("discrete fit %s (err %.3f <= %.3f)" % (nom, err, tol),
              err <= tol)


def test_fdtd_wall_vs_tmm():
    """240x5x5 waveguide: |R|(f) MEASURED from FDTD propagation against
    the discrete model and the TMM analytics (porous + air gap). ~90 s."""
    import auralis.engine.reflection as rf
    from auralis.engine.fdtd import (VoxelGrid, FdtdGrid,
                                     run_grid_python, source_pulse,
                                     _LAMBDA)
    C = 343.0
    dx = C / (9.0 * 300.0)
    dt = _LAMBDA * dx / C
    spec = pm.spec_from_params("miki", 15000.0, 50.0, air_gap_mm=150.0)
    fgrid = np.geomspace(30.0, 360.0, 48)
    gi, A, l = rf.fit_admittance_complex(
        fgrid, pm.gamma_normal(spec, fgrid),
        l_max=0.6 / dt, dt=dt, lam=_LAMBDA)
    nx, ny, nz = 240, 5, 5
    vox = VoxelGrid(solid=np.zeros((nx, ny, nz), bool),
                    mat=np.full((nx, ny, nz), -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    vox.solid[-1, :, :] = True
    vox.mat[-1, :, :] = 0
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.array([gi]), C,
                 filt_mat=(np.array([gi]), A[None, :], l[None, :]))
    n_steps = int((3.0 * nx * dx / C) / g.dt)
    pulse = source_pulse(g.dt, 100.0, n_steps)
    idx = np.argwhere(vox.air[6])
    si = np.array([g.cell_index[6, i, j] for i, j in idx])
    st = (si, np.full(len(si), 1.0 / len(si)))
    mid = nx // 2
    idr = np.argwhere(vox.air[mid])
    ri = np.array([g.cell_index[mid, i, j] for i, j in idr])
    rt = (ri, np.full(len(ri), 1.0 / len(ri)))
    p = run_grid_python(g, pulse, st, [rt], n_steps)[:, 0]
    t_inc = (mid - 6) * dx / C
    t_ref = ((nx - 1 - 6) + (nx - 1 - mid)) * dx / C
    w = int(0.40 * (t_ref - t_inc) / g.dt)
    FI = np.fft.rfft(p[int(t_inc / g.dt) - w:int(t_inc / g.dt) + w]
                     * np.hanning(2 * w), 1 << 14)
    FR = np.fft.rfft(p[int(t_ref / g.dt) - w:int(t_ref / g.dt) + w]
                     * np.hanning(2 * w), 1 << 14)
    f = np.fft.rfftfreq(1 << 14, g.dt)
    ep, ea = [], []
    for f0 in (60.0, 90.0, 130.0, 160.0):
        k = int(np.argmin(np.abs(f - f0)))
        kk = slice(k - 2, k + 3)
        Rm = float(np.abs(FR[kk]).mean() / np.abs(FI[kk]).mean())
        Rp = float(np.abs(rf.wall_R_discrete(
            np.array([f0]), gi, A, l, dt, _LAMBDA)[0]))
        Rt = float(np.abs(pm.reflection(spec, np.array([f0]), 1.0)[0]))
        ep.append(abs(Rm - Rp))
        ea.append(abs(Rm - Rt))
    check("machinery: measured |R| == discrete model (err %.3f <= 0.05)"
          % max(ep), max(ep) <= 0.05)
    check("chain: measured |R| == analytic TMM (err %.3f <= 0.10)"
          % max(ea), max(ea) <= 0.10)


def test_qualitative_physics():
    print("[physics] qualitative trends")
    thin = pm.alpha_bands(pm.spec_from_params("miki", 12000.0, 25.0))
    thick = pm.alpha_bands(pm.spec_from_params("miki", 12000.0, 100.0))
    check("100 mm absorbs more than 25 mm at 250 Hz (%.2f > %.2f)"
          % (thick[1], thin[1]), thick[1] > thin[1] + 0.1)
    f = np.array([300.0])
    a0 = float(pm.alpha_diffuse(
        pm.spec_from_params("miki", 9000.0, 30.0), f)[0])
    a1 = float(pm.alpha_diffuse(
        pm.spec_from_params("miki", 9000.0, 30.0, air_gap_mm=120.0),
        f)[0])
    check("the air gap boosts low-frequency absorption (300 Hz: "
          "%.2f -> %.2f)" % (a0, a1), a1 > a0 + 0.15)
    # a porous layer stops absorbing as f -> 0
    a_lo = pm.alpha_diffuse(pm.spec_from_params("miki", 12000.0, 50.0),
                            np.array([15.0]))
    check("alpha -> 0 at very low frequency (%.3f)" % a_lo[0],
          a_lo[0] < 0.12)


if __name__ == "__main__":
    print("== exact air layer ==")
    test_air_layer_exact()
    print("== transfer matrices ==")
    test_tmm_composition()
    test_lossless_oblique()
    print("== JCA ==")
    test_jca_limits()
    print("== Miki ==")
    test_miki_asymptote()
    print("== passivity ==")
    test_passivity_sweep()
    print("== engine consistency ==")
    test_gamma_consistency()
    print("== physics ==")
    test_qualitative_physics()
    print("== FDTD wall ==")
    test_fit_wall_discrete()
    test_fdtd_wall_vs_tmm()
    print()
    print("%d ok, %d failed" % (_n_ok, _n_fail))
    sys.exit(1 if _n_fail else 0)
