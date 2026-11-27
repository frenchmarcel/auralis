# -*- coding: utf-8 -*-
"""Wave band suite: dispersion, CPML, open-scene stability, ITD, averaged
borders, budget crossover, fusion, backend parity."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine.fdtd import (FdtdGrid, VoxelGrid, auto_crossover,
                                 fuse_low_high, gamma_from_alpha,
                                 make_sponge, run_grid_python,
                                 run_grid_native, run_low_band,
                                 source_pulse, voxelize)
from auralis.engine.bands import CENTERS
from auralis.engine.geometry import SceneData
from auralis.engine import native as nat

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
        if "PYTEST_CURRENT_TEST" in os.environ:     # under pytest a failed check fails
            raise AssertionError("%s %s" % (label, extra))


def _lib_native():
    try:
        p = nat.find_library()
        return nat.load_library(p) if p else None
    except Exception:
        return None


def _lib_emu():
    try:
        base = os.path.join(os.path.dirname(nat.find_library() or ""),
                            "libauralis_cuda_emu.so")
        return nat.load_library(base) if os.path.exists(base) else None
    except Exception:
        return None


_LIB = _lib_native()


def _run(grid, pulse, st, rts, n_steps):
    """Fastest available backend (native, else python)."""
    if _LIB is not None:
        try:
            return run_grid_native(grid, pulse, st, rts, n_steps, _LIB)
        except Exception:
            pass
    return run_grid_python(grid, pulse, st, rts, n_steps)


def freebox(shape, dx, order=2, **kw):
    vox = VoxelGrid(solid=np.zeros(shape, bool),
                    mat=np.full(shape, -1, np.int32),
                    origin=np.array([0.0, 0.0, 0.0]), dx=dx)
    vox.air = ~vox.solid
    return FdtdGrid(vox, np.zeros(1), 343.0, order=order, **kw)


def box_scene(L=4.0, W=3.0, H=2.5, alpha=0.15, rot_z_deg=0.0):
    v = np.array([[x, y, z] for z in (0, H) for y in (0, W)
                  for x in (0, L)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4),
             (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]
    tris = []
    for a, b, c, d in quads:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    tv = np.array(tris)
    ctr = np.array([L / 2, W / 2, H / 2])
    if rot_z_deg:
        th = np.deg2rad(rot_z_deg)
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        tv = (tv.reshape(-1, 3) - ctr) @ R.T + ctr
        tv = tv.reshape(-1, 3, 3)
    sc = types.SimpleNamespace()
    sc.tri_verts = tv
    sc.tri_mat = np.zeros(len(tris), np.int64)
    sc.absorption = np.full((1, len(CENTERS)), alpha)
    sc.rot = np.eye(3) if not rot_z_deg else R
    sc.ctr = ctr
    return sc


def pebble_scene():
    V = np.array([[15.0, -0.2, 0.8], [15.4, -0.2, 0.8],
                  [15.2, 0.2, 1.2]], float)
    return SceneData(tri_verts=V.reshape(1, 3, 3),
                     tri_mat=np.zeros(1, np.int32),
                     absorption=np.array([[0.03] * 7], np.float32),
                     scattering=np.full((1, 7), 0.1, np.float32),
                     transmission=np.zeros((1, 7), np.float32),
                     mat_names=["pebble"])


# ---------------------------------------------------------------------------
def test_dispersion_order4():
    """Group delay 300-900 Hz over 2 m: order 4 must cut the order 2
    error; window STRICTLY before the first echo."""
    print("[dispersion] order 2 versus order 4 (2 m, 300-900 Hz)")
    d, c = 2.0, 343.0
    src, rcv = (0.6, 2.0, 2.0), (2.6, 2.0, 2.0)
    errs = {}
    for order in (2, 4):
        g = freebox((81, 81, 81), 0.05, order=order)
        n_steps = int((d / c + 3.2e-3) / g.dt)
        pulse = source_pulse(g.dt, 800.0, n_steps)
        p = _run(g, pulse, g.taps(src), [g.taps(rcv)], n_steps)[:, 0]
        t = np.arange(n_steps) * g.dt
        w = (t >= d / c - 1.2e-3) & (t <= d / c + 2.55e-3)
        h = np.zeros(1 << 14)
        h[:w.sum()] = p[w]
        n0 = int(np.nonzero(w)[0][0])
        f = np.fft.rfftfreq(len(h), g.dt)
        H = np.fft.rfft(h)
        pz = np.zeros(len(h))
        pz[:len(pulse)] = pulse
        # deconvolve the pulse phase (otherwise the source's own group
        # delay adds ~ +0.9 ms)
        ph = np.unwrap(np.angle(H * np.conj(np.fft.rfft(pz))))
        tau = -np.gradient(ph, 2 * np.pi * f[1]) + n0 * g.dt
        m = (f >= 300.0) & (f <= 900.0)
        errs[order] = tau[m] - d / c
    # The CONSTANT delay (~0.7 dt, of temporal origin) is absorbed by the
    # free-field calibration; what degrades the sound is the SPREAD of
    # the delay across the band (true dispersion).
    sp2 = float(errs[2].max() - errs[2].min())
    sp4 = float(errs[4].max() - errs[4].min())
    check("order 4 spread < 90 us within 300-900", sp4 < 90e-6,
          "(%.1f us)" % (sp4 * 1e6))
    check("order 2 disperses (spread > 250 us)", sp2 > 250e-6,
          "(%.1f us)" % (sp2 * 1e6))
    check("order 4 at least 3x flatter", sp4 < sp2 / 3.0,
          "(o2 %.0f us, o4 %.0f us)" % (sp2 * 1e6, sp4 * 1e6))
    check("order 4 delay bounded (< 180 us)",
          float(np.abs(errs[4]).max()) < 180e-6,
          "(max %.0f us)" % (np.abs(errs[4]).max() * 1e6))


# ---------------------------------------------------------------------------
def _echo_db(g, f_peak, n_steps):
    c = 343.0
    p = _run(g, source_pulse(g.dt, f_peak, n_steps),
             g.taps((2.0, 2.0, 2.0)), [g.taps((2.5, 2.0, 2.0))],
             n_steps)[:, 0]
    t = np.arange(n_steps) * g.dt
    tail = 7.0e-3
    w_d = (t > 0.5 / c) & (t < 0.5 / c + tail)
    w_e = (t > 3.5 / c + 0.2 * tail) & (t < t[-1] - 1e-4)
    r = 10 * np.log10(float((p[w_e] ** 2).sum())
                      / float((p[w_d] ** 2).sum()))
    return r, float(np.abs(p[-30:]).max())


def test_cpml_reflection():
    """Empty 81^3 box, 250 Hz pulse: CPML n=14 reflection <= -40 dB, at
    least 10 dB below the reference sponge, and a stable field."""
    print("[cpml] 250 Hz border reflection, empty 4 m box")
    g = freebox((81,) * 3, 0.05, order=2, pml={"n": 14})
    r_c, m_c = _echo_db(g, 250.0, 420)
    g2 = freebox((81,) * 3, 0.05, order=2,
                 sigma=make_sponge((81,) * 3, 0.05, 343.0, 18))
    r_s, m_s = _echo_db(g2, 250.0, 420)
    check("CPML n14 <= -40 dB", r_c <= -40.0, "(%.1f dB)" % r_c)
    check("gain >= 10 dB over the n=18 sponge", r_c <= r_s - 10.0,
          "(cpml %.1f / sponge %.1f dB)" % (r_c, r_s))
    check("stable and decaying field", m_c < 1e-3 and m_s < 1e-3,
          "(%.1e / %.1e)" % (m_c, m_s))


# ---------------------------------------------------------------------------
def test_open_scene_stability():
    """Solid/PML corner mode regression: the open 'pebble' scene must
    stay stable and DECAYING over 1200 steps (order 4)."""
    print("[cpml] long open-scene stability (1200 steps, order 4)")
    sc = pebble_scene()
    src, rcv = np.array([0.0, 0.0, 1.6]), np.array([2.5, 0.0, 1.4])
    dx = 343.0 / (9.0 * 180.0)
    vox = voxelize(sc, dx, src, margin_cells=16,
                   include_points=[src, rcv])
    g = FdtdGrid(vox, gamma_from_alpha(np.array([0.03])), 343.0,
                 gamma_border=1.0, order=4, pml={"n": 14})
    n_steps = 1200
    p = _run(g, source_pulse(g.dt, 56.0, n_steps),
             g.taps(src), [g.taps(rcv)], n_steps)[:, 0]
    m_mid = float(np.abs(p[500:560]).max())
    m_end = float(np.abs(p[-60:]).max())
    check("finite over 1200 steps", bool(np.isfinite(p).all()))
    check("field bounded then decaying", m_end < 1e-3
          and m_end <= m_mid * 2.0,
          "(mid %.1e -> end %.1e, %d cells)" % (m_mid, m_end, g.n_air))
    check("reduced grid margin (pml 14+2 << sponge 28+)",
          vox.shape[1] < 60, "(ny=%d)" % vox.shape[1])


# ---------------------------------------------------------------------------

def test_ears_in_grid():
    """Two probes 17.5 cm apart in a very absorbing box: the ITD (cross-correlation of the
    first 5 ms) must follow the geometry within 25 us. The 20 % onset is no estimator here:
    the fourth-order grid propagates a slight high-frequency precursor."""
    print("[ears] onset ITD, lateral and frontal (fc 400)")
    fs = 48000
    sc = box_scene(6.0, 5.0, 3.0, alpha=0.85)
    ctr = np.array([2.2, 2.5, 1.5])
    right = np.array([1.0, 0.0, 0.0])
    ears = [ctr - 0.0875 * right, ctr + 0.0875 * right]
    for src, lab, tol in ((np.array([4.4, 2.5, 1.5]), "lateral", 25e-6),
                          (np.array([2.2, 4.3, 1.5]), "frontal", 25e-6)):
        ir, fi = run_low_band(sc, src, ears, fs=fs, duration=0.06,
                              f_cross=400.0, speed_of_sound=343.0)
        d = float(np.linalg.norm(src - ctr))
        i0 = int((d / 343.0 - 2e-3) * fs)
        a, b = ir[i0:i0 + int(5e-3 * fs), 0], ir[i0:i0 + int(5e-3 * fs), 1]
        n, up = 1 << 14, 16
        xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n * up)
        k = int(np.argmax(xc))
        lag = (k - n * up if k > n * up // 2 else k) / float(fs * up)
        itd = (np.linalg.norm(src - ears[0])
               - np.linalg.norm(src - ears[1])) / 343.0
        check("ITD %s within %.0f us" % (lab, tol * 1e6),
              abs(lag - itd) < tol and ir.shape[1] == 2,
              "(geo %+.1f / measured %+.1f us)" % (itd * 1e6, lag * 1e6))


# ---------------------------------------------------------------------------
def _t20(ir, fs):
    e = np.cumsum((ir ** 2)[::-1])[::-1]
    L = 10 * np.log10(np.maximum(e / e[0], 1e-12))
    t = np.arange(len(ir)) / fs
    m = (L <= -5.0) & (L >= -25.0)
    if m.sum() < 8:
        return 0.0
    A = np.vstack([t[m], np.ones(m.sum())]).T
    k, _ = np.linalg.lstsq(A, L[m], rcond=None)[0]
    return -60.0 / k if k < 0 else 0.0


def test_face_weighting_rotation():
    """Averaged borders: a box rotated 30 degrees about z must give the
    same low-frequency T20 (within 25 %) and the same level (3 dB)."""
    print("[borders] box rotated 30 degrees versus aligned box")
    fs = 48000
    res = {}
    for rot in (0.0, 30.0):
        sc = box_scene(4.0, 3.0, 2.5, alpha=0.25, rot_z_deg=rot)
        src = sc.rot @ (np.array([1.0, 1.0, 1.1]) - sc.ctr) + sc.ctr
        rcv = sc.rot @ (np.array([2.8, 2.1, 1.4]) - sc.ctr) + sc.ctr
        ir, fi = run_low_band(sc, src, rcv, fs=fs, duration=0.5,
                              f_cross=200.0, speed_of_sound=343.0,
                              open_boundary="closed")
        res[rot] = (_t20(ir, fs), float((ir ** 2).sum()))
    t0, e0 = res[0.0]
    t30, e30 = res[30.0]
    check("T20 30deg within +/-25%% of the aligned box",
          abs(t30 - t0) < 0.25 * max(t0, 1e-6),
          "(%.2f s versus %.2f s)" % (t30, t0))
    check("energy within +/-3 dB",
          abs(10 * np.log10(max(e30, 1e-30) / max(e0, 1e-30))) < 3.0,
          "(%+.1f dB)" % (10 * np.log10(max(e30, 1e-30)
                                        / max(e0, 1e-30))))


# ---------------------------------------------------------------------------
def test_auto_crossover():
    """Budget crossover: rises with the budget, falls with the volume,
    is capped, and respects the Schroeder floor."""
    print("[budget] automatic crossover")
    f1, _ = auto_crossover(90.0, 1.5, 343.0, 80.0, budget_gcell=10.0)
    f2, _ = auto_crossover(90.0, 1.5, 343.0, 80.0, budget_gcell=40.0)
    f3, _ = auto_crossover(3000.0, 1.5, 343.0, 80.0, budget_gcell=40.0)
    check("more budget -> higher crossover", f2 > f1 * 1.2,
          "(%.0f -> %.0f Hz)" % (f1, f2))
    check("more volume -> lower crossover", f3 < f1,
          "(%.0f Hz at 3000 m3)" % f3)
    f4, _ = auto_crossover(10.0, 0.3, 343.0, 80.0, budget_gcell=500.0,
                           fcross_max=700.0)
    check("ceiling respected", f4 <= 700.0 + 1e-9, "(%.0f Hz)" % f4)
    f5, _ = auto_crossover(5000.0, 3.0, 343.0, 260.0, budget_gcell=1.0)
    check("Schroeder floor respected", f5 >= 260.0 - 1e-9,
          "(%.0f Hz)" % f5)


# ---------------------------------------------------------------------------
def test_fuse_shapes():
    print("[fusion] duplicated mono and per-channel stereo")
    fs, n = 48000, 4800
    rng = np.random.default_rng(3)
    ir_geo = rng.standard_normal((n, 2)) * np.exp(
        -np.arange(n) / (0.05 * fs))[:, None]
    low1 = rng.standard_normal(n) * 1e-2
    low2 = np.stack([low1, low1 * 0.5], axis=1)
    o1 = fuse_low_high(ir_geo, low1, 200.0, fs)
    o2 = fuse_low_high(ir_geo, low2, 200.0, fs)
    check("shapes and finiteness", o1.shape == (n, 2) and o2.shape == (n, 2)
          and np.isfinite(o1).all() and np.isfinite(o2).all())
    d_mono = float(np.linalg.norm((o1[:, 0] - ir_hi(ir_geo[:, 0], fs))
                                  - (o1[:, 1] - ir_hi(ir_geo[:, 1], fs))))
    check("duplicated mono: same low band on both channels", d_mono < 1e-9,
          "(%.1e)" % d_mono)
    dd = float(np.linalg.norm(o2[:, 0] - o2[:, 1]))
    check("stereo: distinct channels", dd > 1e-6)


def ir_hi(x, fs):
    from auralis.engine.fdtd import design_crossover
    lp, hp, delay = design_crossover(200.0, fs)
    return np.convolve(x, hp)[delay:delay + len(x)]


# ---------------------------------------------------------------------------
def test_backend_parity():
    """python / native / cuda-emulation on an order 4 filtered grid and
    a CPML grid; emulation must be bit-identical to native."""
    print("[parity] python versus native versus cuda-emulation")
    lib_n, lib_e = _LIB, _lib_emu()
    if lib_n is None:
        print("  --   native unavailable, skipped")
        return
    g = freebox((41,) * 3, 0.08, order=4, pml={"n": 10})
    n_steps = 260
    pulse = source_pulse(g.dt, 200.0, n_steps)
    st = g.taps((1.2, 1.6, 1.6))
    rts = [g.taps((2.0, 1.6, 1.6)), g.taps((1.6, 2.2, 1.2))]
    pp = run_grid_python(g, pulse, st, rts, n_steps)
    pn = run_grid_native(g, pulse, st, rts, n_steps, lib_n)
    rel = float(np.linalg.norm(pp - pn) / np.linalg.norm(pp))
    check("py/native < 5e-3 (f32)", rel < 5e-3, "(%.1e)" % rel)
    if lib_e is not None:
        pe = run_grid_native(g, pulse, st, rts, n_steps, lib_e)
        check("native == emulation (bit for bit)",
              float(np.abs(pn - pe).max()) == 0.0)
    else:
        print("  --   emulation unavailable, skipped")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("== dispersion ==")
    test_dispersion_order4()
    print("== CPML absorber ==")
    test_cpml_reflection()
    test_open_scene_stability()
    print("== ears in the grid ==")
    test_ears_in_grid()
    print("== averaged borders ==")
    test_face_weighting_rotation()
    print("== budget crossover ==")
    test_auto_crossover()
    print("== fusion ==")
    test_fuse_shapes()
    print("== backend parity ==")
    test_backend_parity()
    print()
    print("%d ok, %d failed" % (_n_ok, _n_fail))
    sys.exit(1 if _n_fail else 0)
