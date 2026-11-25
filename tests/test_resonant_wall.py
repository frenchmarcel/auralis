# -*- coding: utf-8 -*-
"""Tests of the RESONANT walls (mass-spring panels) of the FDTD band:
sections (a, beta, w0^2), waveguide reflection, passivity and parity
between python / native / CUDA emulation."""

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import poromat as pm  # noqa: E402
from auralis.engine.fdtd import (  # noqa: E402
    FdtdGrid, VoxelGrid, run_grid_python, run_grid_native, source_pulse,
    _LAMBDA, _soss_prop,
)
from auralis.engine.reflection import (  # noqa: E402
    alpha_random, fit_admittance_complex, polish_resonant,
    wall_R_discrete,
)

C = 343.0
FAILED = []


def check(label, ok):
    print(("  [ok] " if ok else "  [FAIL] ") + label)
    if not ok:
        FAILED.append(label)


def _pipeline(spec, dx):
    """The run_low_band pipeline: sections + co-fitted branches."""
    dt = _LAMBDA * dx / C
    secs, _, f_res = pm.resonant_sections(spec)
    g_full = pm.gamma_normal(spec, f_res)
    gi, A, lam = fit_admittance_complex(f_res, g_full, l_max=0.6 / dt,
                                        dt=dt, lam=_LAMBDA,
                                        res_sections=secs)
    secs = polish_resonant(f_res, g_full, secs, gi, A, lam, dt, _LAMBDA)
    gi, A, lam = fit_admittance_complex(f_res, g_full, l_max=0.6 / dt,
                                        dt=dt, lam=_LAMBDA,
                                        res_sections=secs)
    return secs, gi, A, lam, f_res, g_full, dt


def test_sections_vs_transfert():
    """The section tracks the exact admittance of three typical panels."""
    for name, spec in (
            ("plaster 12.5/50 Q2", pm.panel_spec(8.8, 50.0, q=2.0)),
            ("glazing 15kg/16mm Q2.5", pm.panel_spec(15.0, 16.0, q=2.5)),
            ("panel+wool 30mm", pm.panel_spec(8.8, 60.0, q=2.0,
                                              fill_sigma=12000,
                                              fill_mm=30.0))):
        secs, _, f = pm.resonant_sections(spec)
        a_t = alpha_random(pm.gamma_normal(spec, f))
        a_s = alpha_random(pm._gamma_sections(secs, f))
        rms = float(np.sqrt(np.mean((a_s - a_t) ** 2)))
        check("section %s: alpha RMS %.4f" % (name, rms), rms < 0.01)
        f0 = np.sqrt(secs[0][2]) / (2 * np.pi)
        check("section %s: f0 %.0f Hz inside the band" % (name, f0),
              20.0 < f0 < 400.0)


def test_model_impedance_matches_target():
    """After co-fitting, the discrete reflection (rigid phase removed)
    holds the continuous target over the whole 30-700 band."""
    spec = pm.panel_spec(8.8, 50.0, q=2.0)
    dx = C / (9.0 * 300.0)
    secs, gi, A, lam, f_res, g_full, dt = _pipeline(spec, dx)
    R_t = (1.0 - g_full) / (1.0 + g_full)
    R0 = wall_R_discrete(f_res, 0.0, np.zeros(1), np.ones(1), dt, _LAMBDA)
    ph = np.conj(R0 / np.abs(R0))
    R = wall_R_discrete(f_res, gi, A, lam, dt, _LAMBDA, res=secs) * ph
    d = np.abs(R - R_t)
    check("z-model vs target: max err %.3f (<120 Hz)"
          % d[f_res < 120].max(), d[f_res < 120].max() < 0.02)
    check("z-model vs target: max err %.3f (120-250)"
          % d[(f_res >= 120) & (f_res < 250)].max(),
          d[(f_res >= 120) & (f_res < 250)].max() < 0.02)
    check("z-model vs target: max err %.3f (>250)"
          % d[f_res >= 250].max(), d[f_res >= 250].max() < 0.03)
    f_pas = np.geomspace(5.0, 0.499 / dt, 400)
    Rp = np.abs(wall_R_discrete(f_pas, gi, A, lam, dt, _LAMBDA, res=secs))
    check("model passivity: max|R| %.4f" % Rp.max(),
          Rp.max() <= 1.0 + 1e-6)


def test_waveguide_reflection():
    """Waveguide 240x5x5: |R|(f) FDTD vs discrete prediction and target."""
    spec = pm.panel_spec(8.8, 50.0, q=2.0)
    dx = C / (9.0 * 300.0)
    secs, gi, A, lam, f_res, g_full, dt = _pipeline(spec, dx)
    nx, ny, nz = 240, 5, 5
    vox = VoxelGrid(solid=np.zeros((nx, ny, nz), bool),
                    mat=np.full((nx, ny, nz), -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    vox.solid[-1, :, :] = True
    vox.mat[-1, :, :] = 0
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.array([gi]), C,
                 filt_mat=(np.array([gi]), A[None, :], lam[None, :]),
                 filt_res=[secs])
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
    errs_z, errs_c = [], []
    for f0 in (50.0, 70.0, 90.0, 110.0, 140.0, 170.0):
        k = int(np.argmin(np.abs(f - f0)))
        kk = slice(k - 2, k + 3)
        Rm = np.abs(FR[kk]).mean() / np.abs(FI[kk]).mean()
        Rz = float(np.abs(wall_R_discrete(np.array([f0]), gi, A, lam,
                                          g.dt, g.lam, res=secs))[0])
        gam = complex(pm.gamma_normal(spec, np.array([f0]))[0])
        Rc = abs((1 - gam) / (1 + gam))
        errs_z.append(abs(Rm - Rz))
        errs_c.append(abs(Rm - Rc))
    check("waveguide: |R| vs discrete prediction max err %.3f"
          % max(errs_z), max(errs_z) < 0.08)
    check("waveguide: |R| vs continuous target max err %.3f"
          % max(errs_c), max(errs_c) < 0.09)
    # the resonance dip is indeed there: |R|(90) far below
    # |R|(170)
    k90 = int(np.argmin(np.abs(f - 90.0)))
    k170 = int(np.argmin(np.abs(f - 170.0)))
    R90 = np.abs(FR[k90 - 2:k90 + 3]).mean() / np.abs(FI[k90 - 2:k90 + 3]).mean()
    R170 = (np.abs(FR[k170 - 2:k170 + 3]).mean()
            / np.abs(FI[k170 - 2:k170 + 3]).mean())
    check("waveguide: resonance dip (R(90)=%.2f < R(170)=%.2f - 0.08)"
          % (R90, R170), R90 < R170 - 0.08)


def test_cavity_is_passive():
    """All-panel cavity: energy does not grow over 2 s (the near-DC
    part, gamma(0)=0, decays slowly: that is physical)."""
    spec = pm.panel_spec(8.8, 50.0, q=2.0)
    dx = C / (9.0 * 300.0)
    secs, gi, A, lam, _, _, dt = _pipeline(spec, dx)
    n2 = (22, 18, 15)
    vox = VoxelGrid(solid=np.zeros(n2, bool),
                    mat=np.full(n2, -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = 0
        vox.solid[tuple(sl)] = True
        vox.mat[tuple(sl)] = 0
        sl[ax] = -1
        vox.solid[tuple(sl)] = True
        vox.mat[tuple(sl)] = 0
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.array([gi]), C,
                 filt_mat=(np.array([gi]), A[None, :], lam[None, :]),
                 filt_res=[secs])
    ns = int(2.0 / g.dt)
    pu = source_pulse(g.dt, 120.0, ns)
    st = (np.array([g.cell_index[10, 8, 7]]), np.array([1.0]))
    rt = (np.array([g.cell_index[5, 5, 5]]), np.array([1.0]))
    p = run_grid_python(g, pu, st, [rt], ns)[:, 0]
    seg = ns // 4
    early = np.abs(p[seg:2 * seg]).max()
    late = np.abs(p[3 * seg:]).max()
    check("panel cavity: no growth (late/early = %.3f)"
          % (late / early), late <= early * 1.02)
    check("panel cavity: no NaN", bool(np.isfinite(p).all()))


def test_parite_backends():
    """python (f64) / native (f32) / emu: emu bit-exact against native,
    native within the baseline f32 drift."""
    spec = pm.panel_spec(8.8, 50.0, q=2.0)
    dx = C / (9.0 * 300.0)
    secs, gi, A, lam, _, _, dt = _pipeline(spec, dx)
    nx, ny, nz = 240, 5, 5
    vox = VoxelGrid(solid=np.zeros((nx, ny, nz), bool),
                    mat=np.full((nx, ny, nz), -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    vox.solid[-1, :, :] = True
    vox.mat[-1, :, :] = 0
    vox.air = ~vox.solid
    from auralis.engine.native import find_library
    path = find_library()
    if not path:
        print("  [skip] no native library")
        return
    lib_n = ctypes.CDLL(path)
    emu = os.path.join(os.path.dirname(path), "libauralis_cuda_emu.so")
    lib_e = ctypes.CDLL(emu) if os.path.exists(emu) else None
    n_steps = 2400
    for order in (2, 4):
        g = FdtdGrid(vox, np.array([gi]), C,
                     filt_mat=(np.array([gi]), A[None, :], lam[None, :]),
                     filt_res=[secs], order=order)
        pulse = source_pulse(g.dt, 100.0, n_steps)
        st = (np.array([g.cell_index[6, 2, 2]]), np.array([1.0]))
        rt = (np.array([g.cell_index[nx // 2, 2, 2]]), np.array([1.0]))
        py = run_grid_python(g, pulse, st, [rt], n_steps)[:, 0]
        na = run_grid_native(g, pulse, st, [rt], n_steps, lib_n)[:, 0]
        ref = np.abs(py).max()
        dna = np.abs(py - na).max() / ref
        check("py/native parity order %d: %.2e" % (order, dna),
              dna < 3e-2)
        if lib_e is not None:
            em = run_grid_native(g, pulse, st, [rt], n_steps, lib_e)[:, 0]
            dem = np.abs(na - em).max()
            check("native/emu parity order %d: %.2e" % (order, dem),
                  dem == 0.0)


def test_soss_prop_regimes():
    """Exact propagator: reproduces expm in the three regimes."""
    dt = 2e-4
    for beta, w02, lbl in ((100.0, 3e5, "underdamped"),
                           (2000.0, 2e5, "overdamped"),
                           (2.0 * np.sqrt(3e5), 3e5, "critical")):
        E, G = _soss_prop(beta, w02, dt)
        M = np.array([[0.0, 1.0], [-w02, -beta]])
        # expm by series (M dt is small)
        X = np.eye(2)
        Eref = np.eye(2)
        for k in range(1, 25):
            X = X @ (M * dt) / k
            Eref = Eref + X
        Gref = np.linalg.solve(M, (Eref - np.eye(2)) @ np.array([0.0, 1.0]))
        ok = (np.abs(E - Eref).max() < 1e-9
              and np.abs(G - Gref).max() < 1e-12)
        check("_soss_prop %s: |dE| %.1e" % (lbl, np.abs(E - Eref).max()),
              ok)


def test_extra_sections():
    """Extra sections (f0, Q, alpha_peak): the full target is tracked
    and polishing does not wipe them out."""
    spec = pm.normalize_spec({"panel": {"m": 15.0, "cavity_mm": 16.0,
                                        "q": 2.5},
                              "extra_sections": [{"f0": 52.0, "q": 1.5,
                                                  "alpha_peak": 0.12}]})
    secs, _, f = pm.resonant_sections(spec, 30.0, 360.0)
    check("extra: two sections", len(secs) == 2)
    f0s = sorted(np.sqrt(s5[2]) / (2 * np.pi) for s5 in secs)
    check("extra: f0 %.0f and %.0f Hz" % tuple(f0s),
          abs(f0s[0] - 52) < 3 and abs(f0s[1] - 122) < 4)
    g_t = pm.gamma_full(spec, f)
    a_t = alpha_random(g_t)
    i52 = int(np.argmin(np.abs(f - 52.0)))
    check("extra: alpha_random(52) = %.2f >= 0.11" % a_t[i52],
          a_t[i52] >= 0.11)
    # full pipeline: sequential polishing preserves the low end
    dx = C / (9.0 * 300.0)
    dt = _LAMBDA * dx / C
    gi, A, lam = fit_admittance_complex(f, g_t, l_max=0.6 / dt, dt=dt,
                                        lam=_LAMBDA, res_sections=secs)
    for si in range(len(secs)):
        others = secs[:si] + secs[si + 1:]
        one = polish_resonant(f, g_t, [secs[si]], gi, A, lam, dt,
                              _LAMBDA, fixed=others)
        secs = secs[:si] + one + secs[si + 1:]
    f0s2 = sorted(np.sqrt(s5[-1]) / (2 * np.pi) for s5 in secs)
    check("extra: f0 after polishing %.0f/%.0f Hz" % tuple(f0s2),
          abs(f0s2[0] - 52) < 8 and abs(f0s2[1] - 122) < 8)


def test_run_low_band_panel():
    """End to end: run_low_band wires up a panel material (grid with
    n_rfaces > 0, finite IR that decays)."""
    import types
    from auralis.engine import make_shoebox
    from auralis.engine.fdtd import run_low_band
    scene = make_shoebox(size=(4.0, 3.2, 2.6))
    mp = [None] * len(scene.mat_names)
    mp[-1] = {"panel": {"m": 10.0, "cavity_mm": 40.0, "q": 2.0}}
    scene.mat_phys = mp
    ir, finfo = run_low_band(scene, np.array([1.2, 1.0, 1.1]),
                             [np.array([2.8, 2.2, 1.4])], fs=44100,
                             duration=0.9, f_cross=250.0,
                             speed_of_sound=C, engine="python",
                             order=2, conformal=False)
    ir = np.asarray(ir, float).ravel()
    check("run_low_band: finite IR", bool(np.isfinite(ir).all()))
    a = np.abs(ir)
    check("run_low_band: decay (tail/peak %.2e)"
          % (a[-2000:].max() / a.max()),
          a[-2000:].max() < 0.5 * a.max())


def test_gpu_runtime_fallback():
    """A GPU failure at RUN TIME falls back loudly to native/python and
    marks the GPU dead for the session (gpu_available() False)."""
    import auralis.engine.fdtd as fdtd
    import auralis.engine.native as nat
    # small valid grid for the replay
    dx = C / (9.0 * 300.0)
    n3 = (10, 8, 6)
    vox = VoxelGrid(solid=np.zeros(n3, bool),
                    mat=np.full(n3, -1, np.int32),
                    origin=np.zeros(3), dx=dx)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = 0
        vox.solid[tuple(sl)] = True
        sl[ax] = -1
        vox.solid[tuple(sl)] = True
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.zeros(1), C)
    ns = 200
    pu = source_pulse(g.dt, 200.0, ns)
    st = (np.array([g.cell_index[4, 3, 2]]), np.array([1.0]))
    rt = (np.array([g.cell_index[6, 4, 3]]), np.array([1.0]))
    ref = run_grid_python(g, pu, st, [rt], ns)

    real_rgn = fdtd.run_grid_native
    calls = {"n": 0}

    def fake_rgn(*a, **k):
        if k.get("lib") is FAKE:
            calls["n"] += 1
            raise RuntimeError("aur_fdtd_run4 failed (code 30)")
        return real_rgn(*a, **k)

    FAKE = object()
    ok_prev = nat._GPU_OK
    print("  (simulating a GPU failure, code 30 -- the two [Auralis]"
          " messages below are EXPECTED and say nothing about your"
          " card; the flag is restored at the end of the test)")
    try:
        fdtd.run_grid_native = fake_rgn
        nat._GPU_OK = True
        out = fdtd._run_gpu_or_fallback(FAKE, "aur_fdtd_run",
                                        g, pu, st, [rt], ns)
        check("GPU fallback: no exception, GPU tried once",
              calls["n"] == 1)
        check("GPU fallback: gpu_available() False after failure",
              nat.gpu_available() is False)
        d = float(np.abs(np.asarray(out, float) - ref).max()
                  / max(np.abs(ref).max(), 1e-12))
        check("GPU fallback: result == replay (%.1e)" % d, d < 3e-2)
    finally:
        fdtd.run_grid_native = real_rgn
        nat._GPU_OK = ok_prev


def main():
    print("== resonant walls (panels) ==")
    test_soss_prop_regimes()
    test_sections_vs_transfert()
    test_model_impedance_matches_target()
    test_waveguide_reflection()
    test_cavity_is_passive()
    test_parite_backends()
    test_extra_sections()
    test_run_low_band_panel()
    test_gpu_runtime_fallback()
    print()
    if FAILED:
        print("FAILURES (%d):" % len(FAILED))
        for f in FAILED:
            print("  - " + f)
        sys.exit(1)
    print("all tests pass.")


if __name__ == "__main__":
    main()
