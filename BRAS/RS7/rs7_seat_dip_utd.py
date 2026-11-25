#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS7 (geometric/UTD path) -- control run on the array of 15 MDF beams
(0.34 m pitch, official BRAS OBJ) laid on a rigid floor; also carries
the beam geometry, reused by the FDTD harness."""
import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "RS5"))
import bras_simple as B                                   # noqa: E402
import rs5_diffraction_edge as R                         # noqa: E402
from auralis.engine.geometry import SceneData             # noqa: E402
from auralis.engine.pipeline import (BakeConfig, SourceDef,   # noqa: E402
                                     ReceiverDef, bake)
from auralis.engine.ir_synthesis import synthesize_mono   # noqa: E402

TEMP_C, HUMID = 19.2, 40.3     # BRAS Documentation.pdf, RS7 page -> c=342.9
FLOOR_HALF = 12.0
# array -- dimensions from the OFFICIAL BRAS OBJ (data_bras/obj/.../
# RS7_RIR.obj, SketchUp y-up frame -> z-up scene). Measured leading
# edges: 3.9584, 4.2984, ..., 8.7184 -> pitch 0.34 (the "0.22" of the
# diagram is the 0.34-0.12 free gap). Section 0.12 x 0.24 (0.108 ruled
# out).
N_BEAMS, PITCH = 15, 0.34
X_FIRST = 4.0184               # x centre of the 1st beam (edge 3.9584)
BEAM_DX, BEAM_LY, BEAM_H = 0.12, 4.1408, 0.24   # depth x, width y, height z
Y_CENTER = 2.9848              # beams y 0.9144..5.0552 (OBJ)
# source aim: LS1 (0 deg, 0 deg) horizontal; LS2 (0 deg, -10 deg) tilted down
AIM_ELEV_DEG = {1: 0.0, 2: -10.0}


def source_forward(pair):
    ls = int(pair.split("-")[0][2:])
    th = math.radians(AIM_ELEV_DEG.get(ls, 0.0))
    return np.array([math.cos(th), 0.0, math.sin(th)])


def _box(tv, mats, x0, x1, y0, y1, z0, z1, mat):
    P = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for a, b, c, d in ((0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5),
                       (3, 0, 4, 7), (4, 5, 6, 7)):
        tv.append([P[a], P[b], P[c]]); mats.append(mat)
        tv.append([P[a], P[c], P[d]]); mats.append(mat)


def build_scene():
    fx0, fx1 = X_FIRST - FLOOR_HALF, X_FIRST + FLOOR_HALF
    fy0, fy1 = Y_CENTER - FLOOR_HALF, Y_CENTER + FLOOR_HALF
    tv = [[[fx0, fy0, 0], [fx1, fy0, 0], [fx1, fy1, 0]],
          [[fx0, fy0, 0], [fx1, fy1, 0], [fx0, fy1, 0]]]
    mats = [0, 0]
    y0, y1 = Y_CENTER - BEAM_LY / 2, Y_CENTER + BEAM_LY / 2
    for k in range(N_BEAMS):
        xc = X_FIRST + k * PITCH
        _box(tv, mats, xc - BEAM_DX / 2, xc + BEAM_DX / 2, y0, y1, 0.0,
             BEAM_H, 1)
    tv = np.asarray(tv, np.float32)
    tm = np.asarray(mats, np.int32)
    # MEASURED MDF12 (ISO 10534-2 impedance tube, data_bras/3 Surface
    # descriptions/_csv/initial_estimates/mat_MDF12mm_plane_00deg.csv,
    # banded via bras_material.py)
    absn = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0.027, 0.031, 0.037, 0.043, 0.053, 0.072, 0.061]],
                    np.float32)
    z2 = np.zeros((2, 7), np.float32)
    return SceneData(tv, tm, absn, z2, z2.copy(), mat_names=["floor", "mdf12"])


def npz_path(args, pair):
    return os.path.join(args.out, "rs7utd_%s.npz" % pair)


def do_one(args, meas, pair, scene, dirv=None):
    src, mic, ref, fs = meas[pair]
    cfg = BakeConfig(fs=fs, temperature_c=TEMP_C, humidity_percent=HUMID,
                     diffraction=True, fdtd=False, ism_order=args.ism_order,
                     n_rays=args.n_rays, max_time=args.max_time,
                     normalize=False, engine=args.engine, seed=1234,
                     max_edges=256)
    res = bake(scene,
               SourceDef(name=pair + "-s", position=np.asarray(src, float),
                         directivity=dirv, forward=source_forward(pair)),
               ReceiverDef(name=pair + "-r", position=np.asarray(mic, float)),
               cfg)
    ir = np.asarray(synthesize_mono(res.events, res.hist, cfg.bin_dt,
                                    res.fs, seed=cfg.seed), float).ravel()
    i0s = B.direct_index(ir, fs)
    i0r = B.direct_index(ref, fs)
    frs, S = B.windowed_tf(ir, fs, i0s, t_win=args.win)
    frr, Rr = B.windowed_tf(ref, fs, i0r, t_win=args.win)
    grid = frs[(frs >= args.fmin) & (frs <= args.fmax)]
    ds = np.interp(grid, frs, 20 * np.log10(np.abs(S) + 1e-15))
    dr = np.interp(grid, frr, 20 * np.log10(np.abs(Rr) + 1e-15))
    r = B.pearson(ds, dr)
    rms = float(np.sqrt(np.mean(((ds - ds.mean()) - (dr - dr.mean())) ** 2)))
    n_diff = int(res.stats.get("diff_events", 0)) if hasattr(res, "stats") \
        else sum(1 for ev in res.events if ev.kind == "diffracted")
    np.savez(npz_path(args, pair), grid=grid, tf_sim=ds, tf_ref=dr,
             pearson=r, rms=rms, win=args.win, ir_sim=ir,
             i0_sim=i0s, i0_ref=i0r, fs=fs)
    print("[rs7-utd] %s: %d evts (%d diffr.); Pearson %.3f; RMS %.2f dB"
          % (pair, len(res.events), n_diff, r, rms))


def write_report(args, rows):
    with open(os.path.join(args.out, "report.csv"), "w") as f:
        f.write("pair,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            f.write("%s,%.0f,%.0f,%.3f,%.2f\n"
                    % (r["pair"], r["flo"], r["fhi"], r["r"], r["rms"]))
    lines = ["RS7 bench (geometric UTD, CONTROL): TF per pair, sim vs meas",
             "15 beams pitch 0.34 (official OBJ)  window=%s s" % args.win, "",
             "%-10s %9s %9s" % ("pair", "Pearson", "RMS dB")]
    for r in rows:
        lines.append("%-10s %9.3f %9.2f" % (r["pair"], r["r"], r["rms"]))
    if rows:
        lines.append("%-10s %9.3f"
                     % ("mean", float(np.mean([r["r"] for r in rows]))))
    open(os.path.join(args.out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("[rs7-utd] report -> %s/report.txt" % args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=1400.0)   # Treble crossover
    ap.add_argument("--ism-order", type=int, default=2)
    ap.add_argument("--n-rays", type=int, default=20000)
    ap.add_argument("--max-time", type=float, default=0.10)  # seat-dip tail
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--directivity", default=None)
    ap.add_argument("--out", default="out_rs7_utd")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meas = R.load_sofa5(args.scene_dir)
    if args.pairs is None:
        args.pairs = sorted(meas.keys(),
                            key=lambda k: (int(k.split("-")[0][2:]),
                                           int(k.split("MP")[1])))
    scene = build_scene()
    dirv = None
    if args.directivity:
        from auralis.engine.directivity import Directivity
        dj = json.load(open(args.directivity))
        dirv = Directivity(np.asarray(dj["table"], dtype=float),
                           name=dj.get("name", "source"))
        print("[rs7-utd] directivity: %s" % dj.get("name", "?"))
    rows = []
    for pair in args.pairs:
        if pair not in meas:
            print("[rs7-utd] %s: missing -- skipped" % pair)
            continue
        do_one(args, meas, pair, scene, dirv)
        z = np.load(npz_path(args, pair), allow_pickle=True)
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
