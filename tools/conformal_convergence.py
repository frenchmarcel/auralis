"""Conformal variants against grid convergence (box 4 x 3 x 2.5 m, aligned and rotated 30 deg):
python tools/conformal_convergence.py modes|decay stair,spread,cut,cut_ap,spread_ap 0.169,0.0847 [0,30]"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.conformal import conformal_corrections  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402

C = 343.0
L = (4.0, 3.0, 2.5)
# (smooth_sweeps, tangential apertures): spread = engine default, cut_ap = the
# fdtd_conformal_aniso option, spread_ap = apertures with the default spreading
VARIANTS = {"spread": (8, False), "cut": (0, False), "cut_ap": (0, True), "spread_ap": (8, True)}


def box(alpha, deg):
    mat = {"absorption": np.full(7, alpha), "scattering": np.full(7, 0.1),
           "transmission": np.zeros(7)}
    sc = make_shoebox(L, wall_material=mat, floor_material=mat, ceiling_material=mat)
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    if deg:
        sc = replace(sc, tri_verts=(np.asarray(sc.tri_verts, np.float64) @ R.T).astype(np.float32))
    return sc, R


def grid(name, sc, vox, gamma):
    if name == "stair":
        return F.FdtdGrid(vox, gamma, C, order=4)
    sweeps, ap = VARIANTS[name]
    phi, fw, st = conformal_corrections(sc, vox, smooth_sweeps=sweeps, per_axis=ap)
    return F.FdtdGrid(vox, gamma, C, order=4, phi=phi, face_area_w=fw,
                      excess_axis=st.get("excess_axis"))


def run(g, src, rcvs, dur, f_peak):
    n = int(dur / g.dt)
    fn, _ = F._pick_fdtd_backend("auto", g)
    y = fn(g, F.source_pulse(g.dt, f_peak, n), g.taps(src), [g.taps(r) for r in rcvs], n)
    return np.asarray(y, np.float64)


def peak(x, dt, lo, hi):
    nf = 1 << 20
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nf))
    f = np.fft.rfftfreq(nf, dt)
    sel = np.nonzero((f > lo) & (f < hi))[0]
    k = sel[np.argmax(X[sel])]
    a, b, c = np.log(X[k - 1:k + 2] + 1e-300)
    return f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])


# one axial mode per run: the source sits on the nodal planes of the other low modes
MODES = [((0.5, 1.5, 1.25), (34.0, 52.0), C / (2 * L[0])),
         ((2.0, 0.5, 1.25), (46.0, 66.0), C / (2 * L[1])),
         ((2.0, 1.5, 0.4), (56.0, 79.0), C / (2 * L[2]))]
R_MODES = [(3.3, 2.4, 1.9), (0.8, 2.3, 0.7)]


def modes(name, dx, deg):
    """Rigid box: % error of the first three axial modes."""
    sc, R = box(0.0, deg)
    pts = [R @ np.array(p) for p in [m[0] for m in MODES] + R_MODES]
    vox = F.voxelize(sc, dx, R @ np.array([2.0, 1.5, 1.25]), include_points=pts)
    g = grid(name, sc, vox, np.zeros(1))
    out = []
    for s, (lo, hi), f0 in MODES:
        y = run(g, R @ np.array(s), [R @ np.array(r) for r in R_MODES], 1.6, 60.0)
        out.append(100 * (peak(y.sum(axis=1), g.dt, lo, hi) / f0 - 1))
    return out, g


def bandpass(x, dt, f1, f2):
    """Linear-phase FIR band (never an FFT mask on a decay)."""
    fs = 1.0 / dt
    n = int(2 * round(4.0 * fs / f1) + 1)
    m = np.arange(n) - (n - 1) / 2
    h = 2 * f2 / fs * np.sinc(2 * f2 / fs * m) - 2 * f1 / fs * np.sinc(2 * f1 / fs * m)
    h *= np.kaiser(n, 8.0)
    return np.convolve(x, h)[(n - 1) // 2:(n - 1) // 2 + len(x)]


def t20(x, dt):
    edc = np.cumsum((x ** 2)[::-1])[::-1]
    db = 10 * np.log10(np.maximum(edc / edc[0], 1e-30))
    t = np.arange(len(db)) * dt
    m = (db <= -5.0) & (db >= -25.0)
    A = np.vstack([t[m], np.ones(int(m.sum()))]).T
    return -60.0 / np.linalg.lstsq(A, db[m], rcond=None)[0][0]


S_ABS = (0.6, 0.7, 0.5)
R_ABS = [(3.1, 2.2, 1.7), (2.6, 0.8, 1.3), (1.4, 2.4, 2.0), (3.4, 1.2, 0.6)]
OCT = [(63, 44.2, 88.4), (125, 88.4, 177.0), (250, 177.0, 354.0), (500, 354.0, 707.0)]


def decay(name, dx, deg, alpha=0.12):
    """alpha = 0.12 everywhere: T20 per octave below the grid crossover, mean of 4 receivers."""
    sc, R = box(alpha, deg)
    rc = [R @ np.array(r) for r in R_ABS]
    src = R @ np.array(S_ABS)
    vox = F.voxelize(sc, dx, src, include_points=[src] + rc)
    g = grid(name, sc, vox, F.gamma_from_alpha(np.full(1, alpha)))
    fc = C / (9 * 1.5 * dx)
    y = run(g, src, rc, 1.1, min(fc, 400.0))
    res = {}
    for b, f1, f2 in OCT:
        if f2 <= fc * 1.01:
            res[b] = float(np.mean([t20(bandpass(y[:, k], g.dt, f1, f2), g.dt)
                                    for k in range(y.shape[1])]))
    return res, g


def main():
    what, names = sys.argv[1], sys.argv[2].split(",")
    dxs = [float(v) for v in sys.argv[3].split(",")]
    degs = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.0, 30.0]
    for dx in dxs:
        for deg in degs:
            for name in names:
                t0 = time.time()
                r, g = (modes if what == "modes" else decay)(name, dx, deg)
                key = "err_pct" if what == "modes" else "t20"
                val = [round(v, 2) for v in r] if what == "modes" else \
                    {k: round(v, 4) for k, v in r.items()}
                print(json.dumps({"what": what, "name": name, "dx": dx, "deg": deg, key: val,
                                  "i4": len(g.i4), "n": g.n_air,
                                  "s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
