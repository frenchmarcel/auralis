# -*- coding: utf-8 -*-
"""Diffuse "rain slope" bench: stochastic tracer against the EXACT
image lattice of a box (Allen-Berkley), ratio per window.

Usage:
    python3 tools/rain_slope_bench.py                  # native if any
    python3 tools/rain_slope_bench.py --engine python  # numpy reference
    python3 tools/rain_slope_bench.py --rays 800000 --alpha 0.15 --scat 0.15
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine.bands import N_BANDS                     # noqa: E402
from auralis.engine.geometry import SceneData, make_shoebox  # noqa: E402


def flat_box(size, alpha, scat):
    """Single-material box, FLAT absorption/scattering across bands."""
    mat = {"absorption": np.full(N_BANDS, alpha),
           "scattering": np.full(N_BANDS, scat),
           "transmission": np.zeros(N_BANDS)}
    return make_shoebox(size, wall_material=mat, floor_material=mat,
                        ceiling_material=mat)


def lattice_hist(size, src, rcv, alpha, max_time, bin_dt, c):
    """Exact image-lattice energy (1-alpha)^n / d^2 per bin, direct path
    excluded (the tracer only deposits from the 1st bounce on)."""
    L = np.asarray(size, float)
    src = np.asarray(src, float)
    rcv = np.asarray(rcv, float)
    n_bins = int(round(max_time / bin_dt))
    hist = np.zeros(n_bins)
    dmax = c * max_time
    M = [int(np.ceil(dmax / (2.0 * L[a]))) + 1 for a in range(3)]
    ax_pos, ax_ref = [], []
    for a in range(3):
        m = np.arange(-M[a], M[a] + 1)
        p_plus = 2.0 * m * L[a] + src[a]          # n refl = |2m|
        p_minus = 2.0 * m * L[a] - src[a]         # n refl = |2m-1|
        ax_pos.append(np.concatenate([p_plus, p_minus]))
        ax_ref.append(np.concatenate([np.abs(2 * m), np.abs(2 * m - 1)]))
    X, Y, Z = np.meshgrid(ax_pos[0], ax_pos[1], ax_pos[2], indexing="ij")
    NX, NY, NZ = np.meshgrid(ax_ref[0], ax_ref[1], ax_ref[2], indexing="ij")
    d = np.sqrt((X - rcv[0]) ** 2 + (Y - rcv[1]) ** 2 + (Z - rcv[2]) ** 2)
    n = NX + NY + NZ
    keep = (n > 0) & (d < dmax) & (d > 1e-6)
    e = (1.0 - alpha) ** n[keep] / d[keep] ** 2
    b = (d[keep] / c / bin_dt).astype(np.int64)
    ok = b < n_bins
    np.add.at(hist, b[ok], e[ok])
    return hist


def run_tracer(scene, src, rcv, engine, n_rays, max_bounces, max_time,
               bin_dt, c, seed):
    n_bins = int(round(max_time / bin_dt))
    air0 = np.zeros(N_BANDS)
    if engine == "python":
        from auralis.engine.stochastic import trace_python
        hist, _dirs, stats = trace_python(
            scene, src, rcv, n_rays, max_bounces, max_time, bin_dt,
            n_bins, c, air0, spec_exclude_order=0, energy_cap=1e-9,
            seed=seed)
        return hist, stats, "python"
    from auralis.engine import native
    lib = native.find_gpu_library() if engine == "gpu" else None
    eng = native.NativeEngine(scene, lib_path=lib)
    hist, _dirs, stats = eng.trace(
        src, rcv, n_rays, max_bounces, max_time, bin_dt, n_bins, c, air0,
        spec_exclude_order=0, energy_cap=1e-9, seed=seed)
    return np.asarray(hist, np.float64), stats, engine


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="native",
                    choices=["native", "python", "gpu"])
    ap.add_argument("--rays", type=int, default=400_000)
    ap.add_argument("--bounces", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--scat", type=float, default=0.15)
    ap.add_argument("--time", type=float, default=0.8)
    ap.add_argument("--win", type=float, default=0.05,
                    help="ratio window (s)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    size = (10.0, 7.0, 3.0)
    src = np.array([2.3, 1.7, 1.4])
    rcv = np.array([7.1, 5.2, 1.8])
    c = 343.0
    bin_dt = 0.001

    scene = flat_box(size, args.alpha, args.scat)
    exact = lattice_hist(size, src, rcv, args.alpha, args.time, bin_dt, c)
    hist, stats, used = run_tracer(scene, src, rcv, args.engine, args.rays,
                                   args.bounces, args.time, bin_dt, c,
                                   args.seed)
    sim = hist.mean(axis=0)          # flat bands: mean = any single one

    w = int(round(args.win / bin_dt))
    n_win = len(exact) // w
    t = (np.arange(n_win) + 0.5) * args.win
    se = np.array([exact[i * w:(i + 1) * w].sum() for i in range(n_win)])
    ss = np.array([sim[i * w:(i + 1) * w].sum() for i in range(n_win)])
    good = se > 0
    ratio = np.where(good, ss / np.maximum(se, 1e-300), np.nan)
    # calibrate on the first full window (>= 1 time of flight)
    i0 = max(1, int(np.ceil((np.linalg.norm(rcv - src) / c) / args.win)))
    ratio = ratio / ratio[i0]

    esc = stats.get("escaped", 0.0)
    mb = stats.get("mean_bounces", 0.0)
    retr = stats.get("retried", float("nan"))
    print("engine=%s  rays=%d  mean bounces=%.1f  escaped=%d (%.3f%%)"
          % (used, args.rays, mb, int(esc),
             100.0 * esc / max(1.0, args.rays)))
    if esc > 0 and mb > 0:
        eps = esc / max(1.0, args.rays * mb / (1.0 - 0.5 * esc / args.rays))
        print("  -> estimated loss per bounce eps=%.4f%% ; drift "
              "predicted at %.0f ms: x%.3f"
              % (100 * eps, 1000 * t[-1],
                 (1.0 - eps) ** (t[-1] / (t[1] - t[0]) * mb
                                 * (t[1] - t[0]) / (args.time / mb))))
    if retr == retr:
        print("  recasts (anti-crack jitter): %d" % int(retr))
    print("%8s  %10s" % ("t (ms)", "hist/exact"))
    for i in range(i0, n_win):
        bar = "#" * int(round(40 * min(ratio[i], 1.3) / 1.3))
        print("%8.0f  %10.4f  %s" % (1000 * t[i], ratio[i], bar))
    tail = np.nanmean(ratio[max(i0, n_win - 4):])
    print("\nmean ratio over the last 4 windows: %.4f  (target 1.00)"
          % tail)
    return 0 if abs(tail - 1.0) < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
