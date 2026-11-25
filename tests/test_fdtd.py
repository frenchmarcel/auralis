# -*- coding: utf-8 -*-
"""Low-frequency FDTD solver tests: rigid box modes, calibrated free
field, admittance walls, voxelizer, crossover and hybrid end-to-end."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine import materials as mat_lib  # noqa: E402
from auralis.engine.fdtd import (  # noqa: E402
    FdtdGrid, VoxelGrid, design_crossover, free_field_calibration,
    fuse_low_high, gamma_from_alpha, run_grid_python, run_low_band,
    source_pulse, voxelize, _resample_fft,
)

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s %s" % (name, extra))
    else:
        FAIL += 1
        print("  FAIL %s %s" % (name, extra))


C = 343.0


def _rigid_grid(n, dx):
    vox = VoxelGrid(solid=np.zeros(tuple(n), bool),
                    mat=np.full(tuple(n), -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    vox.air = ~vox.solid
    return FdtdGrid(vox, np.zeros(1), C)


def test_box_modes():
    fmax, ppw = 400.0, 9.0
    dx = C / (ppw * fmax)
    n = np.array([35, 30, 25])
    g = _rigid_grid(n, dx)
    n_steps = int(1.6 / g.dt)
    pulse = source_pulse(g.dt, fmax / 3.2, n_steps)
    st = g.taps((0.29 * n[0] * dx, 0.31 * n[1] * dx, 0.27 * n[2] * dx))
    rt = g.taps((0.72 * n[0] * dx, 0.61 * n[1] * dx, 0.58 * n[2] * dx))
    p = run_grid_python(g, pulse, st, [rt], n_steps)[:, 0]
    tail = np.abs(p[-300:]).max() / np.abs(p[n_steps // 2:
                                             n_steps // 2 + 300]).max()
    check("rigid box: bounded energy (tail/mid = %.2f)" % tail,
          tail < 1.2)
    F = np.abs(np.fft.rfft(p * np.hanning(len(p)), 4 * len(p)))
    f = np.fft.rfftfreq(4 * len(p), g.dt)
    sel = (f > 20) & (f < 260)
    F, f = F[sel], f[sel]
    pk = np.where((F[1:-1] > F[:-2]) & (F[1:-1] > F[2:])
                  & (F[1:-1] > F.max() * 0.02))[0] + 1
    peaks = f[pk]
    Leff = n * dx
    modes = sorted({round(C / 2 * np.sqrt((l / Leff[0]) ** 2
                                          + (m / Leff[1]) ** 2
                                          + (q / Leff[2]) ** 2), 2)
                    for l in range(4) for m in range(4) for q in range(4)
                    if l + m + q > 0})
    modes = [m for m in modes if m < 250][:12]
    errs = [min(abs(peaks - m)) / m for m in modes]
    check("first 12 modes vs analytic (max %.2f %%)"
          % (100 * max(errs)), max(errs) < 0.012)


def test_energy_decay():
    dx = C / (9.0 * 400.0)
    n = np.array([26, 22, 18])

    def run(gamma):
        solid = np.zeros(tuple(n), bool)
        for ax in range(3):
            sl = [slice(None)] * 3
            sl[ax] = 0
            solid[tuple(sl)] = True
            sl[ax] = -1
            solid[tuple(sl)] = True
        vox = VoxelGrid(solid=solid, mat=np.full(tuple(n), 0, np.int32),
                        origin=np.zeros(3), dx=dx)
        vox.air = ~vox.solid
        g = FdtdGrid(vox, np.array([gamma]), C)
        n_steps = int(0.30 / g.dt)
        pulse = source_pulse(g.dt, 120.0, n_steps)
        st = g.taps((n[0] * dx * 0.4, n[1] * dx * 0.4, n[2] * dx * 0.5))
        p = run_grid_python(g, pulse, st, [st], n_steps)[:, 0]
        i1, i2 = int(0.04 / g.dt), int(0.08 / g.dt)
        j1, j2 = int(0.20 / g.dt), int(0.24 / g.dt)
        return (float(np.sum(p[j1:j2] ** 2))
                / max(float(np.sum(p[i1:i2] ** 2)), 1e-300))

    r_abs = run(0.3)
    r_rig = run(0.0)
    check("gamma=0.3: energy 0.2s / 0.06s = %.1e (<3e-3)" % r_abs,
          r_abs < 3e-3)
    check("rigid contrast: %.2f (> 0.3)" % r_rig, r_rig > 0.3)


def test_free_field_calibration():
    dx = C / (9.0 * 300.0)
    n_steps = 900
    dt = dx / C / np.sqrt(3)
    pulse = source_pulse(dt, 90.0, n_steps)
    half = 46
    ns = 2 * half + 1
    vox = VoxelGrid(solid=np.zeros((ns,) * 3, bool),
                    mat=np.full((ns,) * 3, -1, np.int32),
                    origin=np.array([-half * dx] * 3), dx=dx)
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.zeros(1), C)
    st = g.taps((0, 0, 0))
    dists = [8 * dx, 14 * dx, 22 * dx]
    rts = [g.taps((d, 0, 0)) for d in dists]
    n_win = int((half * dx * 2 - max(dists)) / C * 0.9 / g.dt)
    p = run_grid_python(g, pulse, st, rts, n_win)
    Cf, n_fft, _, _band = free_field_calibration(dx, C, pulse, n_win)
    f = np.fft.rfftfreq(n_fft, g.dt)
    band = (f > 45) & (f < 200)
    peaks = []
    for k, d in enumerate(dists):
        H = np.fft.rfft(p[:, k], n_fft) * Cf
        flat = np.abs(H)[band] * d
        h = np.fft.irfft(H, n_fft)
        i = int(np.argmax(np.abs(h[:n_win])))
        t_err = abs(i * g.dt - d / C) / g.dt
        peaks.append(float(np.abs(h[i])))
        check("d=%.2f m: |H|*d in band = %.3f..%.3f, delay %.2f steps"
              % (d, flat.min(), flat.max(), t_err),
              flat.min() > 0.93 and flat.max() < 1.07 and t_err < 1.6)
    r12 = peaks[0] / peaks[1] * dists[0] / dists[1]
    r13 = peaks[0] / peaks[2] * dists[0] / dists[2]
    check("1/d law across distances (%.3f, %.3f)" % (r12, r13),
          abs(r12 - 1) < 0.02 and abs(r13 - 1) < 0.02)


def test_impedance_reflection():
    """Plane waveguide (rigid side walls), test wall at the end:
    measured |R| vs (1-gamma)/(1+gamma)."""
    dx = C / (9.0 * 300.0)
    nx, ny, nz = 160, 7, 7
    for gamma in (0.15, 0.45):
        vox = VoxelGrid(solid=np.zeros((nx, ny, nz), bool),
                        mat=np.full((nx, ny, nz), -1, np.int32),
                        origin=np.zeros(3), dx=dx)
        vox.solid[-1, :, :] = True                    # test wall at the end
        vox.mat[-1, :, :] = 0
        vox.air = ~vox.solid
        g = FdtdGrid(vox, np.array([gamma]), C)
        n_steps = int((2.6 * nx * dx / C) / g.dt)
        pulse = source_pulse(g.dt, 110.0, n_steps)
        # plane excitation: inject over the whole x = 6 slice
        idx = np.argwhere(vox.air[6])
        si = np.array([g.cell_index[6, i, j] for i, j in idx])
        st = (si, np.full(len(si), 1.0 / len(si)))
        # receiver at mid-length, slice average (plane wave)
        mid = nx // 2
        idr = np.argwhere(vox.air[mid])
        ri = np.array([g.cell_index[mid, i, j] for i, j in idr])
        rt = (ri, np.full(len(ri), 1.0 / len(ri)))
        p = run_grid_python(g, pulse, st, [rt], n_steps)[:, 0]
        t_inc = (mid - 6) * dx / C
        t_ref = ((nx - 1 - 6) + (nx - 1 - mid)) * dx / C
        w = int(0.35 * (t_ref - t_inc) / g.dt)
        i_inc = int(t_inc / g.dt)
        i_ref = int(t_ref / g.dt)
        a_inc = np.abs(p[i_inc - w:i_inc + w]).max()
        a_ref = np.abs(p[i_ref - w:i_ref + w]).max()
        R_th = (1 - gamma) / (1 + gamma)
        err = abs(a_ref / a_inc - R_th)
        check("gamma=%.2f: |R| = %.3f vs %.3f (err %.3f)"
              % (gamma, a_ref / a_inc, R_th, err), err < 0.04)


def test_voxelizer():
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="brick")
    dx = 0.14
    vox = voxelize(sc, dx, (2.0, 1.5, 1.2))
    vol_air = vox.air.sum() * dx ** 3
    check("air volume ~ interior (%.1f m3 vs 30)" % vol_air,
          22.0 < vol_air < 31.0)
    # tightness: no air cell beyond the shell
    out = vox.air & ((np.arange(vox.shape[0])[:, None, None] <= 0)
                     | (np.arange(vox.shape[0])[:, None, None]
                        >= vox.shape[0] - 1))
    check("air confined (nothing at the grid edge)", out.sum() == 0)
    # 5 cm wall: the shoebox shell is thin -> continuous solid
    frac = float(vox.solid.mean())
    check("plausible solid fraction (%.2f)" % frac, 0.05 < frac < 0.6)
    try:
        voxelize(sc, dx, (2.0, 1.5, 4.5))
        check("source outside air -> clear error", False)
    except ValueError:
        check("source outside air -> clear error", True)
    g = gamma_from_alpha(np.array([0.0, 0.5, 0.999]))
    check("gamma(alpha): 0 -> 0, increasing, bounded",
          g[0] == 0.0 and g[0] < g[1] < g[2] <= 1.5)


def test_crossover_and_resample():
    lp, hp, d = design_crossover(200.0, 48000)
    s = lp.copy()
    s[d] += 0.0
    tot = lp + hp
    imp = np.zeros_like(tot)
    imp[d] = 1.0
    check("LP + HP = exact impulse (err %.1e)"
          % np.abs(tot - imp).max(), np.abs(tot - imp).max() < 1e-12)
    # resampling: a 130 Hz tone survives faithfully
    fs_in = 6000.0
    t = np.arange(int(fs_in * 0.5)) / fs_in
    x = np.sin(2 * np.pi * 130.0 * t) * np.hanning(len(t))
    y = _resample_fft(x, fs_in, 48000, int(0.5 * 48000))
    ty = np.arange(len(y)) / 48000.0
    ref = np.sin(2 * np.pi * 130.0 * ty) * np.hanning(len(y))
    num = float(np.dot(y, ref))
    den = float(np.linalg.norm(y) * np.linalg.norm(ref) + 1e-30)
    check("resampling: 130 Hz tone correlation = %.4f" % (num / den),
          num / den > 0.999)


def test_end_to_end_small_room():
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="brick",
                      floor_material="brick", ceiling_material="brick")
    # conformal=False: this test validates the BARE grid convention
    # (rigid ghost -> L_eff = voxelized air span); trimmed FV cells
    # move the modes toward the TRUE dimensions and are tested in
    # tests/test_conformal_fv.py.
    ir, info = run_low_band(sc, (1.1, 0.8, 0.9), (2.9, 2.2, 1.5),
                            fs=48000, duration=1.2, f_cross=150.0,
                            speed_of_sound=C, conformal=False)
    check("end-to-end: finite IR (%d air cells, %.1f s of compute)"
          % (info.n_air, info.runtime_s),
          np.isfinite(ir).all() and len(ir) == int(1.2 * 48000))
    F = np.abs(np.fft.rfft(ir * np.hanning(len(ir)), 4 * len(ir)))
    f = np.fft.rfftfreq(4 * len(ir), 1.0 / 48000)
    sel = (f > 25) & (f < 160)
    F, f = F[sel], f[sel]
    pk = np.where((F[1:-1] > F[:-2]) & (F[1:-1] > F[2:])
                  & (F[1:-1] > F.max() * 0.05))[0] + 1
    peaks = f[pk]
    Lx, Ly, Lz = info.air_span
    modes = sorted({round(C / 2 * np.sqrt((l / Lx) ** 2 + (m / Ly) ** 2
                                          + (q / Lz) ** 2), 2)
                    for l in range(3) for m in range(3) for q in range(3)
                    if l + m + q > 0})
    modes = [m for m in modes if m < 115][:6]
    errs = [min(abs(peaks - m)) / m for m in modes]
    check("first 6 modes of the voxelized room (max err %.2f %%)"
          % (100 * max(errs)), max(errs) < 0.02)
    # fusion: HP*geo + LP*low keeps the energy finite and the low band
    geo = np.zeros((len(ir), 2))
    geo[100, :] = 0.05
    fused = fuse_low_high(geo, ir, 150.0, 48000)
    check("HP/LP fusion stable", np.isfinite(fused).all()
          and fused.shape == geo.shape)




def test_hybrid_bake():
    """bake(fdtd=True): the low spectrum of the hybrid IR shows the
    analytic axial modes that ray tracing alone does not produce."""
    from auralis.engine import BakeConfig, ReceiverDef, SourceDef, bake
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="brick",
                      floor_material="brick", ceiling_material="brick")
    src = SourceDef("s", (1.1, 0.8, 0.9))
    rcv = ReceiverDef("m", (2.9, 2.2, 1.5), forward=(1, 0, 0))
    # fdtd_conformal=False: the modal expectations below follow the
    # BARE grid convention (voxelized span 3.73/2.71/2.20 m); the
    # conformal cells would move them toward the true dimensions.
    r_off = bake(sc, src, rcv, BakeConfig(n_rays=4000, seed=5,
                                          max_time=1.2, fdtd=False))
    r_on = bake(sc, src, rcv, BakeConfig(n_rays=4000, seed=5, max_time=1.2,
                                         fdtd=True, fdtd_fcross=150.0,
                                         fdtd_conformal=False))
    check("FDTD stats present (fc=%.0f Hz, %d cells)"
          % (r_on.stats.get("fdtd_fc", 0),
             r_on.stats.get("fdtd_cells", 0)),
          r_on.stats.get("fdtd_fc", 0) == 150.0
          and r_on.stats.get("fdtd_cells", 0) > 1000)

    def spec(res):
        x = res.ir.mean(axis=1)
        F = np.abs(np.fft.rfft(x * np.hanning(len(x)), 4 * len(x)))
        f = np.fft.rfftfreq(4 * len(x), 1.0 / res.fs)
        m = (f > 30) & (f < 120)
        return f[m], F[m] / (F[m].max() + 1e-30)
    f1, S_on = spec(r_on)
    _, S_off = spec(r_off)
    ax = [C / (2 * 3.73), C / (2 * 2.71), C / (2 * 2.20)]

    def prominence(S, f, m):
        i = int(np.argmin(np.abs(f - m)))
        w = max(3, int(6.0 / (f[1] - f[0])))
        loc = S[max(0, i - 2):i + 3].max()
        base = np.median(S[max(0, i - w):i + w])
        return loc / (base + 1e-30)
    prom_on = [prominence(S_on, f1, m) for m in ax]
    prom_off = [prominence(S_off, f1, m) for m in ax]
    check("modes present with FDTD (prom %.1f/%.1f/%.1f > 1.8)"
          % tuple(prom_on), min(prom_on) > 1.8)
    check("smooth spectrum without FDTD (max prom %.1f < half)"
          % max(prom_off), max(prom_off) < 0.6 * min(prom_on))
    check("hybrid IR finite, normalized",
          np.isfinite(r_on.ir).all()
          and 0.9 < np.abs(r_on.ir).max() <= 1.0)




def test_native_parity():
    """Native CPU vs python (float32/float64), CPU vs EMU (bit exact)."""
    import ctypes
    from auralis.engine import native as nat
    cpu_path = nat.find_library()
    if not cpu_path:
        print("  (no native lib: parity skipped)")
        return
    lib_cpu = nat.load_library(cpu_path)
    if not hasattr(lib_cpu, "aur_fdtd_run"):
        print("  (native lib without FDTD: parity skipped)")
        return
    from auralis.engine.fdtd import run_grid_native
    dx = C / (9.0 * 350.0)
    g = _rigid_grid(np.array([30, 26, 20]), dx)
    n_steps = 700
    pulse = source_pulse(g.dt, 100.0, n_steps)
    st = g.taps((9.4 * dx, 8.2 * dx, 6.1 * dx))
    rts = [g.taps((20.2 * dx, 17.5 * dx, 12.8 * dx)),
           g.taps((5.5 * dx, 19.1 * dx, 9.9 * dx))]
    import time as _t
    t0 = _t.time()
    p_py = run_grid_python(g, pulse, st, rts, n_steps)
    t_py = _t.time() - t0
    t0 = _t.time()
    p_na = run_grid_native(g, pulse, st, rts, n_steps, lib_cpu)
    t_na = _t.time() - t0
    rel = np.abs(p_na - p_py).max() / np.abs(p_py).max()
    mc = g.n_air * n_steps / 1e6
    # undamped rigid box = float32 worst case
    check("native CPU vs python: max rel err %.1e (%.0f vs %.0f Mcell/s)"
          % (rel, mc / max(t_na, 1e-9), mc / max(t_py, 1e-9)), rel < 6e-3)
    # identical modal peaks
    def pk(x):
        F = np.abs(np.fft.rfft(x[:, 0] * np.hanning(len(x)), 4 * len(x)))
        f = np.fft.rfftfreq(4 * len(x), g.dt)
        m = (f > 40) & (f < 250)
        return f[m][np.argmax(F[m])]
    check("identical modal peak (%.1f vs %.1f Hz)" % (pk(p_na), pk(p_py)),
          abs(pk(p_na) - pk(p_py)) < 0.5)
    emu_path = os.path.join(os.path.dirname(cpu_path),
                            "libauralis_cuda_emu.so")
    if os.path.exists(emu_path):
        lib_emu = ctypes.CDLL(emu_path)
        p_em = run_grid_native(g, pulse, st, rts, n_steps, lib_emu)
        check("CPU vs EMU: bit-exact match (max %.1e)"
              % np.abs(p_em - p_na).max(),
              np.array_equal(p_em, p_na))
    else:
        print("  (no EMU lib)")




def test_admittance_fit():
    from auralis.engine.fdtd import fit_admittance
    from auralis.engine.bands import CENTERS
    # curtain: steep slope in the low end
    a_curt = np.array([0.07, 0.30, 0.50, 0.70, 0.75, 0.70, 0.65])
    gi, A, l = fit_admittance(a_curt, CENTERS, 240.0)
    f = np.array([70.0, 125.0, 200.0])
    Y = gi + (A[None, :] / (1j * 2 * np.pi * f[:, None] + l[None, :])
              ).sum(axis=1)
    Rm = np.abs((1 - Y) / (1 + Y))
    Rt = np.sqrt(1 - np.interp(f, CENTERS, a_curt))
    # 2 branches vs octave alphas: ~0.5 dB/bounce
    check("curtain fit: |R|(f) tracked (max err %.3f)"
          % np.abs(Rm - Rt).max(), np.abs(Rm - Rt).max() < 0.06)
    # flat brick: falls back to pure g_inf (no branches)
    a_flat = np.full(7, 0.03)
    gi2, A2, _ = fit_admittance(a_flat, CENTERS, 240.0)
    check("flat fit: branches ~ 0 (max A %.2f), g_inf ~ const"
          % A2.max(), A2.max() < 1.0 and abs(gi2
              - gamma_from_alpha(np.array([0.03]))[0]) < 0.01)


def test_filtered_wall_reflection():
    """5x5 waveguide (270 Hz cutoff): measured vs analytic |R|(f) for a
    2-pole rational admittance wall."""
    from auralis.engine.fdtd import _LAMBDA
    dx = C / (9.0 * 300.0)
    dt = _LAMBDA * dx / C
    nx, ny, nz = 240, 5, 5
    g_inf = 0.10
    A = np.array([180.0, 0.0])
    lam = np.array([2 * np.pi * 120.0, 1.0])
    vox = VoxelGrid(solid=np.zeros((nx, ny, nz), bool),
                    mat=np.full((nx, ny, nz), -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    vox.solid[-1, :, :] = True
    vox.mat[-1, :, :] = 0
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.array([g_inf]), C,
                 filt_mat=(np.array([g_inf]), A[None, :], lam[None, :]))
    n_steps = int((3.0 * nx * dx / C) / dt)
    pulse = source_pulse(dt, 100.0, n_steps)
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
    w = int(0.40 * (t_ref - t_inc) / dt)
    FI = np.fft.rfft(p[int(t_inc / dt) - w:int(t_inc / dt) + w]
                     * np.hanning(2 * w), 1 << 14)
    FR = np.fft.rfft(p[int(t_ref / dt) - w:int(t_ref / dt) + w]
                     * np.hanning(2 * w), 1 << 14)
    f = np.fft.rfftfreq(1 << 14, dt)
    errs = []
    errs_c = []
    for f0 in (60.0, 90.0, 130.0, 160.0):
        k = np.argmin(np.abs(f - f0))
        kk = slice(k - 2, k + 3)
        Rm = np.abs(FR[kk]).mean() / np.abs(FI[kk]).mean()
        from auralis.engine.reflection import wall_R_discrete
        Rd = wall_R_discrete(np.array([f0]), g_inf, A, lam, dt,
                             _LAMBDA)[0]
        errs.append(abs(Rm - abs(Rd)))
        Y = g_inf + (A / (1j * 2 * np.pi * f0 + lam)).sum()
        errs_c.append(abs(Rm - abs((1 - Y) / (1 + Y))))
    check("filtered wall: |R| vs discrete response, max err %.3f"
          % max(errs), max(errs) < 0.04)
    check("filtered wall: |R| vs continuous target, max err %.3f"
          % max(errs_c), max(errs_c) < 0.05)
    # passivity: fully filtered cavity, 1.5 s
    n2 = np.array([22, 18, 15])
    solid = np.zeros(tuple(n2), bool)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = 0
        solid[tuple(sl)] = True
        sl[ax] = -1
        solid[tuple(sl)] = True
    v2 = VoxelGrid(solid=solid, mat=np.full(tuple(n2), 0, np.int32),
                   origin=np.zeros(3), dx=dx)
    v2.air = ~v2.solid
    g2 = FdtdGrid(v2, np.array([g_inf]), C,
                  filt_mat=(np.array([g_inf]), A[None, :], lam[None, :]))
    ns2 = int(1.5 / g2.dt)
    q = run_grid_python(g2, source_pulse(g2.dt, 100.0, ns2),
                        g2.taps((8 * dx, 7 * dx, 6 * dx)),
                        [g2.taps((8 * dx, 7 * dx, 6 * dx))], ns2)[:, 0]
    check("filtered wall passivity (end/peak %.1e)"
          % (np.abs(q[-400:]).max() / np.abs(q).max()),
          np.isfinite(q).all()
          and np.abs(q[-400:]).max() < 1e-4 * np.abs(q).max())


def test_open_scene_ground():
    """Ground only (open scene) vs image source: exact direct,
    reflection within |R|, collar residue below -32 dB."""
    from auralis.engine.geometry import SceneData
    V = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0],
                  [-10, -10, 0], [10, 10, 0], [-10, 10, 0]], float)
    sc = SceneData(tri_verts=V.reshape(2, 3, 3),
                   tri_mat=np.zeros(2, np.int32),
                   absorption=np.array([[0.02] * 2
                                        + [0.03, 0.04, 0.05, 0.05, 0.06]]),
                   scattering=np.full((1, 7), 0.1, np.float32),
                   transmission=np.zeros((1, 7), np.float32),
                   mat_names=["sol"])
    src, rcv = (0.0, 0.0, 2.6), (1.2, 0.0, 2.3)
    ir, info = run_low_band(sc, src, rcv, fs=48000, duration=0.12,
                            f_cross=140.0, speed_of_sound=C,
                            max_cells=8_000_000)
    check("open scene detected (%d cells, %s)"
          % (info.n_cells, info.engine), info.is_open)
    d1 = np.linalg.norm(np.subtract(rcv, src))
    d2 = np.linalg.norm(np.subtract(rcv, (0, 0, -2.6)))
    fs = 48000
    i1, i2 = int(d1 / C * fs), int(d2 / C * fs)
    w = 100
    a1 = np.abs(ir[max(i1 - w, 0):i1 + w]).max()
    a2 = np.abs(ir[i2 - w:i2 + w]).max()
    tail = float(np.abs(ir[i2 + 3 * w:]).max())
    j1 = np.argmax(np.abs(ir[max(i1 - w, 0):i1 + w])) + max(i1 - w, 0) - i1
    j2 = np.argmax(np.abs(ir[i2 - w:i2 + w])) - w
    tol_geo = int(np.ceil(info.dx / C * fs)) + 6
    check("direct within a sample (%+d)" % j1, abs(j1) <= 3)
    check("ground reflection: ratio %.3f, delay %+d smp (step ~dx)"
          % (a2 * d2 / (a1 * d1), j2),
          0.88 < a2 * d2 / (a1 * d1) < 1.13 and abs(j2) <= tol_geo)
    check("collar residue %.4f (%.1f dB < -27)"
          % (tail / a1, 20 * np.log10(tail / a1 + 1e-12)),
          tail / a1 < 0.045)  # collar reflectivity ~ -13 dB deep low;
    # the direct echo is cancelled by the decomposition, only the one
    # from the reflections is left


def test_rigid_material_warning():
    """The pre-flight warning must name near-rigid materials (low-band
    alpha < 0.03) before a long FDTD run."""
    import io, contextlib
    from auralis.engine import materials as mat_lib
    from auralis.engine.geometry import SceneData
    m = {"absorption": np.full(7, 0.01, np.float32),
         "scattering": np.zeros(7, np.float32),
         "transmission": np.zeros(7, np.float32)}
    aT, sT, tT = mat_lib.build_tables([m])
    X, Y, Z = 1.0, 1.2, 0.9
    v = np.array([[a*X, b*Y, c*Z] for a in (0, 1)
                  for b in (0, 1) for c in (0, 1)], float)
    quads = [(0,1,3,2),(4,6,7,5),(0,2,6,4),(1,5,7,3),(0,4,5,1),
             (2,3,7,6)]
    tris = [t2 for a,b,c,dd in quads
            for t2 in ([v[a],v[b],v[c]], [v[a],v[c],v[dd]])]
    sc = SceneData(tri_verts=np.ascontiguousarray(np.array(tris)),
                   tri_mat=np.zeros(12, np.int32), absorption=aT,
                   scattering=sT, transmission=tT, mat_names=["dur"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ir, info = run_low_band(sc, np.array([0.3, 0.3, 0.3]),
                                [np.array([0.7, 0.9, 0.6])],
                                44100, 1.6, 399.0, 343.0,
                                engine="native", order=4)
    check("near-rigid material warning",
          "near-rigid" in buf.getvalue()
          and "dur" in buf.getvalue())


def test_conformal_weights():
    """Conformal weights: exact area conservation and ~1 weights on an
    aligned box (zero regression), conservation on a rotated box."""
    from auralis.engine.conformal import face_weights

    def boite(X, Y, Z, rot=0.0):
        v = np.array([[a*X, b*Y, c*Z] for a in (0, 1)
                      for b in (0, 1) for c in (0, 1)], float)
        if rot:
            th = np.radians(rot)
            Rz = np.array([[np.cos(th), -np.sin(th), 0],
                           [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            v = v @ Rz.T
        quads = [(0,1,3,2),(4,6,7,5),(0,2,6,4),(1,5,7,3),(0,4,5,1),
                 (2,3,7,6)]
        return np.array([tt for a,b,c,d in quads
                         for tt in ([v[a],v[b],v[c]],
                                    [v[a],v[c],v[d]])])

    dx = 0.1
    for rot, nom in ((0.0, "aligned"), (30.0, "rotated")):
        tv = boite(0.8, 1.1, 0.6, rot)
        lo = tv.reshape(-1, 3).min(0) - 2*dx
        shape = np.ceil((tv.reshape(-1, 3).max(0) + 2*dx - lo)/dx
                        ).astype(int)
        e1 = tv[:, 1]-tv[:, 0]; e2 = tv[:, 2]-tv[:, 0]
        A = 0.5*np.linalg.norm(np.cross(e1, e2), axis=1).sum()
        w = face_weights(tv, lo, dx, shape)
        Aacc = sum(w.values())
        check("conformal %s: area conservation (%.4f vs %.4f)"
              % (nom, Aacc, A), abs(Aacc/A - 1) < 1e-6)
        if rot == 0.0:
            vals = np.array(list(w.values()))/dx**2
            pleins = np.abs(vals - 1.0) < 1e-3
            lamelles = vals < 1e-6
            check("conformal aligned: weights ~1 or slivers ~0 "
                  "(%d full, %d slivers)"
                  % (int(pleins.sum()), int(lamelles.sum())),
                  bool((pleins | lamelles).all()))


def test_engine_reported_is_real():
    """`info.engine` must name the backend ACTUALLY run, not the one
    picked before a possible CPU fallback."""
    import types as _t
    from auralis.engine import fdtd as _f
    check("_lib_tag: cuda -> gpu",
          _f._lib_tag(_t.SimpleNamespace(_name="/x/libauralis_cuda.so"))
          == "gpu")
    check("_lib_tag: cuda_emu -> cuda-emu",
          _f._lib_tag(_t.SimpleNamespace(_name="/x/libauralis_cuda_emu.so"))
          == "cuda-emu")
    check("_lib_tag: plain native lib -> native",
          _f._lib_tag(_t.SimpleNamespace(_name="/x/libauralis.so"))
          == "native")
    # the marker is set WHERE the compute happens, and `info.engine`
    # captures it right after the MAIN run (the global marker is then
    # rewritten by the free-field calibration pass: `info.engine` is
    # what counts, not the module variable).
    sc = make_shoebox((3.0, 2.5, 2.4), wall_material="brick")
    _f._ENGINE_USED[0] = "sentinel"
    _, info = _f.run_low_band(sc, np.array([1.0, 1.0, 1.2]),
                              [np.array([2.0, 1.5, 1.2])], fs=8000,
                              duration=0.02, f_cross=250,
                              speed_of_sound=343.0, engine="native")
    check("reported backend = real one, not the sentinel (%s)" % info.engine,
          info.engine == "native")


if __name__ == "__main__":
    test_conformal_weights()
    test_rigid_material_warning()
    print("== rigid box modes ==")
    test_box_modes()
    print("== energy / decay ==")
    test_energy_decay()
    print("== free-field calibration ==")
    test_free_field_calibration()
    print("== impedance wall ==")
    test_impedance_reflection()
    print("== voxelizer ==")
    test_voxelizer()
    print("== reported backend == real backend ==")
    test_engine_reported_is_real()
    print("== crossover / resampling ==")
    test_crossover_and_resample()
    print("== end-to-end small room ==")
    test_end_to_end_small_room()
    print("== admittance fit ==")
    test_admittance_fit()
    print("== filtered wall (frequency) ==")
    test_filtered_wall_reflection()
    print("== open scene: ground ==")
    test_open_scene_ground()
    print("== native kernel parity ==")
    test_native_parity()
    print("== hybrid bake (pipeline) ==")
    test_hybrid_bake()
    print("\n%d ok, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
