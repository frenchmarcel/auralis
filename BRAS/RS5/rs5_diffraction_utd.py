#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS5 through geometric UTD (Kouyoumjian-Pathak, first order) instead
of FDTD: rigid screen in free field, no grid. Same metric and outputs
as `rs5_diffraction_edge` (per-pair windowed TF, Pearson).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))          # BRAS/
sys.path.insert(0, os.path.join(_HERE, "..", ".."))    # repo root (auralis)
import bras_simple as B                                 # noqa: E402
import rs5_diffraction_edge as R                       # noqa: E402
from auralis.engine.geometry import SceneData           # noqa: E402
from auralis.engine.pipeline import (BakeConfig, SourceDef,   # noqa: E402
                                     ReceiverDef, bake)
from auralis.engine.ir_synthesis import synthesize_mono  # noqa: E402

# Hemi-anechoic chamber, scene 05 (= scene 01): T gives c ~ 343.6 m/s.
TEMP_C = 20.3
HUMID = 41.5
FLOOR_HALF = 12.0   # floor half-extent (large -> its edges diffract
#                     well beyond the analysis window)


# Source aim axis (elevation in degrees, x-z plane, toward +x):
# LS1/2/3 = (phi,theta)=(0,0) horizontal; LS4 = (0,34.6) tilted upward
# (aims at the top edge of the screen). Diagram scene5_RIR.png.
AIM_ELEV_DEG = {1: 0.0, 2: 0.0, 3: 0.0, 4: 34.6}


def source_forward(pair):
    ls = int(pair.split("-")[0][2:])
    th = math.radians(AIM_ELEV_DEG.get(ls, 0.0))
    return np.array([math.cos(th), 0.0, math.sin(th)])


def build_scene(screen_width=R.SCREEN_W):
    """Rigid floor + finite thin screen (width `screen_width`, centered
    y=2.985, h=2.066, 25 mm thick) as a 7-band SceneData."""
    xc, yc = R.X_WALL, R.Y_CENTER
    fx0, fx1 = xc - FLOOR_HALF, xc + FLOOR_HALF
    fy0, fy1 = yc - FLOOR_HALF, yc + FLOOR_HALF
    tv = [[[fx0, fy0, 0], [fx1, fy0, 0], [fx1, fy1, 0]],
          [[fx0, fy0, 0], [fx1, fy1, 0], [fx0, fy1, 0]]]
    mats = [0, 0]
    e, h = R.E_WALL, R.H_WALL
    xw0, xw1 = xc - e / 2, xc + e / 2
    ys0, ys1 = yc - screen_width / 2, yc + screen_width / 2
    P = [(xw0, ys0, 0), (xw1, ys0, 0), (xw1, ys1, 0), (xw0, ys1, 0),
         (xw0, ys0, h), (xw1, ys0, h), (xw1, ys1, h), (xw0, ys1, h)]
    for a, b, c_, d_ in ((0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5),
                         (3, 0, 4, 7), (4, 5, 6, 7)):
        tv.append([P[a], P[b], P[c_]])
        mats.append(1)
        tv.append([P[a], P[c_], P[d_]])
        mats.append(1)
    tv = np.asarray(tv, np.float32)
    tm = np.asarray(mats, np.int32)
    absn = np.array([[0, 0, 0, 0, 0, 0, 0],                       # rigid floor
                     [0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04]],  # mdf25
                    np.float32)
    z2 = np.zeros((2, 7), np.float32)
    return SceneData(tv, tm, absn, z2, z2.copy(),
                     mat_names=["floor", "mdf25"])


def npz_path(args, pair):
    return os.path.join(args.out, "rs5utd_%s.npz" % pair)


def do_one(args, meas, pair, scene, dirv=None):
    """ONE pair -> npz. Geometric bake (UTD) then mono IR.
    dirv: source directivity (Directivity) or None (omni)."""
    src, mic, ref, fs = meas[pair]
    cfg = BakeConfig(fs=fs, temperature_c=TEMP_C, humidity_percent=HUMID,
                     diffraction=True, fdtd=False, ism_order=args.ism_order,
                     n_rays=args.n_rays, max_time=args.max_time,
                     normalize=False, engine=args.engine, seed=1234)
    res = bake(scene,
               SourceDef(name=pair + "-s", position=np.asarray(src, float),
                         directivity=dirv, forward=source_forward(pair)),
               ReceiverDef(name=pair + "-r", position=np.asarray(mic, float)),
               cfg)
    ir = np.asarray(synthesize_mono(res.events, res.hist, cfg.bin_dt,
                                    res.fs, seed=cfg.seed), float).ravel()
    # same windowing/anchoring as the FDTD: each side on its 1st arrival
    i0s = B.direct_index(ir, fs)
    i0r = B.direct_index(ref, fs)
    frs, S = B.windowed_tf(ir, fs, i0s, t_win=args.win)
    frr, Rr = B.windowed_tf(ref, fs, i0r, t_win=args.win)
    grid = frs[(frs >= args.fmin) & (frs <= args.fmax)]
    ds = np.interp(grid, frs, 20 * np.log10(np.abs(S) + 1e-15))
    dr = np.interp(grid, frr, 20 * np.log10(np.abs(Rr) + 1e-15))
    r = B.pearson(ds, dr)
    rms = float(np.sqrt(np.mean(((ds - ds.mean())
                                 - (dr - dr.mean())) ** 2)))
    n_diff = int(res.stats.get("diff_events", 0)) if hasattr(res, "stats") \
        else sum(1 for ev in res.events if ev.kind == "diffracted")
    np.savez(npz_path(args, pair), grid=grid, tf_sim=ds, tf_ref=dr,
             pearson=r, rms=rms, win=args.win, ir_sim=ir,
             i0_sim=i0s, i0_ref=i0r, fs=fs,
             screen_width=args.screen_width)
    print("[rs5-utd] %s: %d events (%d diffr.); Pearson %.3f; RMS %.2f dB "
          "(%d pts)" % (pair, len(res.events), n_diff, r, rms, len(grid)))
    return r


def write_report(args, rows):
    csv_path = os.path.join(args.out, "report.csv")
    with open(csv_path, "w") as f:
        f.write("pair,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            f.write("%s,%.0f,%.0f,%.3f,%.2f\n"
                    % (r["pair"], r["flo"], r["fhi"], r["r"], r["rms"]))
    lines = ["RS5 bench (geometric UTD): per-pair windowed TF, sim vs "
             "measurement (Pearson)",
             "scene=%s  finite screen %.2f m  window=%s s  edge x=%.3f "
             "z=%.3f" % (args.scene_dir, args.screen_width, args.win,
                         R.X_WALL, R.H_WALL), "",
             "%-10s %9s %9s" % ("pair", "Pearson", "RMS dB")]
    for r in rows:
        lines.append("%-10s %9.3f %9.2f" % (r["pair"], r["r"], r["rms"]))
    if rows:
        lines.append("%-10s %9.3f"
                     % ("mean", float(np.mean([r["r"] for r in rows]))))
    txt_path = os.path.join(args.out, "report.txt")
    open(txt_path, "w").write("\n".join(lines) + "\n")
    print("[rs5-utd] report -> %s , %s" % (txt_path, csv_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="default: the 16 pairs present")
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=2000.0,
                    help="UTD is valid over the whole band (no grid)")
    ap.add_argument("--screen-width", type=float, default=R.SCREEN_W,
                    help="finite screen width (m; actual 4.75)")
    ap.add_argument("--ism-order", type=int, default=2,
                    help="image-source order (floor reflection)")
    ap.add_argument("--n-rays", type=int, default=20000,
                    help="diffuse-tail rays (negligible in free "
                         "field)")
    ap.add_argument("--max-time", type=float, default=0.08,
                    help="IR length (s); covers arrival + window")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--directivity", default=None,
                    help="source directivity JSON (Genelec 8020c); "
                         "omni if absent. See make_genelec_directivity.py")
    ap.add_argument("--out", default="out_rs5_utd")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meas = R.load_sofa5(args.scene_dir)
    if args.pairs is None:
        args.pairs = sorted(meas.keys(),
                            key=lambda k: (int(k.split("-")[0][2:]),
                                           int(k.split("MP")[1])))
    scene = build_scene(args.screen_width)

    dirv = None
    if args.directivity:
        from auralis.engine.directivity import Directivity
        dj = json.load(open(args.directivity))
        dirv = Directivity(np.asarray(dj["table"], dtype=float),
                           name=dj.get("name", "source"))
        print("[rs5-utd] source directivity: %s (%d angles)"
              % (dj.get("name", "?"), dj.get("n_theta", 0)))

    rows = []
    for pair in args.pairs:
        if pair not in meas:
            print("[rs5-utd] %s: not in the measurements - skipped" % pair)
            continue
        pth = npz_path(args, pair)
        cached = os.path.exists(pth) and not args.force
        if not cached and not args.report_only:
            do_one(args, meas, pair, scene, dirv)
        elif not os.path.exists(pth):
            print("[rs5-utd] --report-only: %s missing - ignored" % pair)
            continue
        else:
            print("[rs5-utd] %s: cached" % pair)
        z = np.load(pth, allow_pickle=True)
        rows.append({"pair": pair, "r": float(z["pearson"]),
                     "rms": float(z["rms"]), "flo": float(z["grid"][0]),
                     "fhi": float(z["grid"][-1])})
        if not args.no_plots:
            g, ds, dr = z["grid"], z["tf_sim"], z["tf_ref"]
            off_r = -float(np.max(dr))
            off_s = float(np.median((dr + off_r) - ds))
            B._plot(os.path.join(args.out, "TF_%s.png" % pair), g,
                    [(dr + off_r, "measured", "-"),
                     (ds + off_s, "Auralis UTD", "--")],
                    "TF %s UTD (Pearson %.3f)" % (pair, float(z["pearson"])))
    if rows:
        write_report(args, rows)


if __name__ == "__main__":
    main()
