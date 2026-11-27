# -*- coding: utf-8 -*-
"""Fitted FV cells (conformal boundaries): conservation of true areas
and volumes, 1D end-lumping principle, T20 effect, long-run stability
and switches."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import replace                              # noqa: E402

from auralis.engine.conformal import (conformal_corrections,  # noqa: E402
                                      face_projections,
                                      fine_air_fractions)
from auralis.engine.fdtd import run_low_band, voxelize        # noqa: E402
from auralis.engine.geometry import make_shoebox              # noqa: E402
from auralis.engine.native import find_library                # noqa: E402

PASS = 0
FAIL = 0
C = 343.0
L = (3.1, 2.45, 2.15)
V_TRUE = L[0] * L[1] * L[2]
A_TRUE = 2 * (L[0] * L[1] + L[0] * L[2] + L[1] * L[2])


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %s %s" % (name, detail))
        if "PYTEST_CURRENT_TEST" in os.environ:     # under pytest a failed check fails
            raise AssertionError("%s %s" % (name, detail))


def _flat_box(alpha):
    mat = {"absorption": np.full(7, alpha),
           "scattering": np.full(7, 0.1),
           "transmission": np.zeros(7)}
    return make_shoebox(L, wall_material=mat, floor_material=mat,
                        ceiling_material=mat)


def _rotz(scene, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    tv = np.asarray(scene.tri_verts, np.float64) @ R.T
    return replace(scene, tri_verts=tv.astype(np.float32)), R


def test_conservation():
    """Mapped areas ~ true areas, discrete + donated volume ~ true
    volume, phi and weight bounds -- axis-aligned AND 30 deg box."""
    for deg in (0.0, 30.0):
        sc = _flat_box(0.12)
        s = np.asarray(L, float) / 2
        if deg:
            sc, R = _rotz(sc, deg)
            s = R @ s
        vox = voxelize(sc, 0.06, s)
        phi, fw, cst = conformal_corrections(sc, vox)
        a_map = cst["area_mapped_m2"]
        a_drop = cst["area_dropped_m2"]
        v_disc = int(vox.air.sum()) * vox.dx ** 3
        v_tot = v_disc + cst["vol_donated_m3"]
        check("areas %2.0f deg: mapped %.2f / true %.2f m2"
              % (deg, a_map, A_TRUE), abs(a_map - A_TRUE) / A_TRUE < 0.01)
        check("areas %2.0f deg: dropped %.3f m2 (< 0.5 %%)"
              % (deg, a_drop), a_drop / A_TRUE < 0.005)
        check("volume %2.0f deg: discrete+donated %.3f / true %.3f m3 "
              "(residual %+.2f %%)"
              % (deg, v_tot, V_TRUE, 100 * (1 - v_tot / V_TRUE)),
              abs(v_tot - V_TRUE) / V_TRUE < 0.015)
        okw = np.all((fw < 0) | ((fw >= 0.05 - 1e-12) & (fw <= 3.0 + 1e-12)))
        check("bounds %2.0f deg: phi in [1, 4], weights in "
              "[0.05, 3] or -1" % deg,
              phi.min() >= 1.0 - 1e-12 and phi.max() <= 4.0 + 1e-9 and okw)
    # the projection split preserves the area piece by piece
    sc = _flat_box(0.12)
    sc, R = _rotz(sc, 30.0)
    vox = voxelize(sc, 0.06, R @ (np.asarray(L, float) / 2))
    acc = face_projections(sc.tri_verts, vox.origin, vox.dx, vox.shape,
                           solid=~vox.air)
    tot = sum(acc.values())
    check("n_axis^2 projections: sum %.3f / %.3f m2" % (tot, A_TRUE),
          abs(tot - A_TRUE) / A_TRUE < 0.005)


def test_1d_end_lumping():
    """The FV principle: a Neumann chain with heavy ends (phi_e) rings
    exactly at c / (2 L_eff) with L_eff = (n + 2(phi_e - 1)) dx."""
    dx = 0.06
    n = 49
    phi_e = 2.29
    M = np.eye(n)
    M[0, 0] = M[-1, -1] = phi_e
    K = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    K[0, 0] = K[-1, -1] = 1.0
    w2 = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(M, K))))
    f1 = np.sqrt(w2[1]) * C / dx / (2 * np.pi)
    l_eff = (n + 2 * (phi_e - 1)) * dx
    f_ref = C / (2 * l_eff)
    check("1D heavy ends: f1 %.3f vs c/2L_eff %.3f Hz" % (f1, f_ref),
          abs(f1 / f_ref - 1) < 1e-3)


def _t20(ir, fs):
    e = np.asarray(ir, np.float64) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc /= edc[0]
    db = 10 * np.log10(np.maximum(edc, 1e-30))
    t = np.arange(len(db)) / fs
    m = (db <= -5.0) & (db >= -25.0)
    A = np.vstack([t[m], np.ones(int(m.sum()))]).T
    slope, _ = np.linalg.lstsq(A, db[m], rcond=None)[0]
    return -60.0 / slope


def test_t20_energy():
    """True V and A make T20 follow the true V/A. Conformal lengthens T20
    (~ +7.5 %) and stays rotation-invariant."""
    fs = 8000
    res = {}
    for deg in (0.0, 30.0):
        sc = _flat_box(0.12)
        s = np.array([0.55, 0.45, 0.40])
        r = np.array([2.45, 1.90, 1.70])
        if deg:
            sc, R = _rotz(sc, deg)
            s, r = R @ s, R @ r
        for conf in (False, True):
            # no subsonic high-pass: this broadband T20 holds the room's lowest content,
            # which the closed-room 16 Hz cut reshapes (1.159 against 1.076 with it)
            ir, info = run_low_band(sc, s, r, fs=fs, duration=1.4,
                                    f_cross=423.0, speed_of_sound=C,
                                    engine="native", conformal=conf, subsonic_hz=0.0)
            res[(deg, conf)] = _t20(ir, fs)
    t_ey = 0.161 * V_TRUE / (-A_TRUE * np.log(1 - 0.12))
    ra = res[(0.0, True)] / res[(0.0, False)]
    gap_on = abs(res[(30.0, True)] - res[(0.0, True)]) \
        / max(res[(0.0, True)], res[(30.0, True)])
    check("T20 conformal/staircase (aligned) = %.3f (expected ~1.075)" % ra,
          1.03 < ra < 1.13)
    check("T20 conformal: rotation spread %.1f %% (< 3.5 %%)"
          % (100 * gap_on), gap_on < 0.035)
    rey = res[(0.0, True)] / t_ey
    check("T20 conformal / Eyring = %.2f (band [1.08, 1.32])" % rey,
          1.08 < rey < 1.32)


def test_stability_rotated():
    """Box rotated 30 deg, near rigid (alpha 0.02), order 4, 1.5 s: the
    envelope must decay (no growing corner mode)."""
    sc = _flat_box(0.02)
    sc, R = _rotz(sc, 30.0)
    s = R @ np.array([0.55, 0.45, 0.40])
    r = R @ np.array([2.45, 1.90, 1.70])
    ir, info = run_low_band(sc, s, r, fs=8000, duration=1.5, f_cross=423.0,
                            speed_of_sound=C, engine="native", order=4,
                            conformal=True)
    q = len(ir) // 4
    e1 = float(np.abs(ir[:q]).max())
    e4 = float(np.abs(ir[-q:]).max())
    check("stability 30 deg order 4: end/start %.2e (< 1)"
          % (e4 / max(e1, 1e-30)),
          np.isfinite(ir).all() and e4 < e1)


def test_switches_and_info():
    """Flag, environment switch and FdtdInfo diagnostics."""
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="brick",
                      floor_material="brick", ceiling_material="brick")
    kw = dict(fs=8000, duration=0.5, f_cross=150.0, speed_of_sound=C)
    ir1, i1 = run_low_band(sc, (1.1, 0.8, 0.9), (2.9, 2.2, 1.5),
                           conformal=True, **kw)
    check("info: conformal=True, volume gain %.1f %% (> 3 %%)"
          % i1.vol_gain_pct, i1.conformal and i1.vol_gain_pct > 3.0)
    ir0, i0 = run_low_band(sc, (1.1, 0.8, 0.9), (2.9, 2.2, 1.5),
                           conformal=False, **kw)
    check("info: conformal=False, no corrections",
          (not i0.conformal) and i0.vol_gain_pct == 0.0)
    os.environ["AURALIS_FDTD_CONFORMAL"] = "0"
    try:
        ir2, i2 = run_low_band(sc, (1.1, 0.8, 0.9), (2.9, 2.2, 1.5),
                               conformal=True, **kw)
    finally:
        os.environ.pop("AURALIS_FDTD_CONFORMAL", None)
    check("switch AURALIS_FDTD_CONFORMAL=0 respected",
          not i2.conformal)
    d = np.abs(ir2 - ir0).max() / max(np.abs(ir0).max(), 1e-30)
    check("switch == conformal=False (diff %.1e)" % d, d < 1e-12)


def test_subgrid_fallback_factor():
    """The subgrid memory fallback must RETURN the actual factor (the
    readout slab depends on it: rad_f = 0.866 dx / F)."""
    import types
    sc = types.SimpleNamespace()
    sc.tri_verts = np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0]],
                             [[0, 0, 0], [2, 2, 0], [0, 2, 0]]], float)
    sc.tri_mat = np.zeros(2, np.int32)
    src = np.array([1.0, 1.0, 0.6])
    vox = voxelize(sc, 0.011, src, include_points=[src, [1.0, 1.0, 0.3]])
    n_fine_x3 = 27 * int(np.prod(vox.shape)) / 1e6      # in millions
    try:
        os.environ["AURALIS_SUBGRID_MAX"] = "450"
        _, f_hi = fine_air_fractions(sc, vox, factor=3)
        os.environ["AURALIS_SUBGRID_MAX"] = "50"        # < n_fine_x3
        _, f_lo = fine_air_fractions(sc, vox, factor=3)
    finally:
        os.environ.pop("AURALIS_SUBGRID_MAX", None)
    check("wide ceiling: requested factor respected (F=%d)" % f_hi,
          f_hi == 3)
    check("fallback (%.0f M > 50 M): ACTUAL factor returned (F=%d, not 3)"
          % (n_fine_x3, f_lo), f_lo == 2)
    # the readout slab follows the actual factor
    check("rad_f slab follows the actual factor (x%.2f)" % (f_hi / f_lo),
          abs((0.5 * np.sqrt(3.0) / f_lo) / (0.5 * np.sqrt(3.0) / f_hi)
              - 1.5) < 1e-12)
    # and conformal_corrections must REPORT that actual factor
    try:
        os.environ["AURALIS_SUBGRID_MAX"] = "50"
        _, _, cst = conformal_corrections(sc, vox)
    finally:
        os.environ.pop("AURALIS_SUBGRID_MAX", None)
    check("stats['factor'] = ACTUAL factor (%d), not requested"
          % cst["factor"], cst["factor"] == 2)


if __name__ == "__main__":
    if find_library() is None:
        print("native library MISSING: run native/build.sh "
              "(the T20/stability tests depend on it)")
        sys.exit(2)
    print("== conservation (areas, volumes, bounds) ==")
    test_conservation()
    print("== 1D end-lumping principle ==")
    test_1d_end_lumping()
    print("== T20: the energy goal ==")
    test_t20_energy()
    print("== stability (30 deg, order 4) ==")
    test_stability_rotated()
    print("== switches and diagnostics ==")
    test_switches_and_info()
    print("== subgrid fallback: actual factor ==")
    test_subgrid_fallback_factor()
    print("\n%d ok, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
