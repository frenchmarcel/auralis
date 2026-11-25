#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS6 (geometric/UTD path) - diffraction around a finite MDF body
(0.72 x 4.14 x 0.72, official BRAS OBJ) on a rigid floor; 9 pairs,
per-pair windowed TF against the measurement (Pearson)."""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))            # BRAS/
sys.path.insert(0, os.path.join(_HERE, "..", ".."))      # repo root
sys.path.insert(0, os.path.join(_HERE, "..", "RS5"))     # load_sofa5, helpers
import bras_simple as B                                   # noqa: E402
import rs5_diffraction_edge as R                         # noqa: E402 (load_sofa5)
from auralis.engine.geometry import SceneData             # noqa: E402
from auralis.engine.pipeline import (BakeConfig, SourceDef,   # noqa: E402
                                     ReceiverDef, bake)
from auralis.engine.ir_synthesis import synthesize_mono   # noqa: E402

TEMP_C, HUMID = 19.9, 40.1          # BRAS Documentation.pdf, RS6 -> c ~ 343.4
FLOOR_HALF = 12.0
# diffracting body - official BRAS OBJ (data_bras/obj/.../RS6_RIR.obj):
# x 5.1364..5.8564, y 0.9144..5.0554, z 0..0.7201 -> length 4.14 CONFIRMED
# (the "2640" dimension of the diagram is refuted), center x 5.4964 (not 5.5).
BODY_C = (5.4964, 2.9849)           # center (x, y)
BODY_DX, BODY_LY, BODY_H = 0.72, 4.141, 0.72  # x depth, y length, z height


def source_forward(pair):
    return np.array([1.0, 0.0, 0.0])   # RS6: every LS aims at (0,0) -> +x


def _box(tv, mats, x0, x1, y0, y1, z0, z1, mat):
    """Add a closed box (bottom face z0 omitted = lying on the floor)."""
    P = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for a, b, c, d in ((0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5),
                       (3, 0, 4, 7), (4, 5, 6, 7)):
        tv.append([P[a], P[b], P[c]])
        mats.append(mat)
        tv.append([P[a], P[c], P[d]])
        mats.append(mat)


def build_scene():
    """Rigid floor (large) + finite MDF body (box), as 7-band SceneData."""
    xc, yc = BODY_C
    fx0, fx1 = xc - FLOOR_HALF, xc + FLOOR_HALF
    fy0, fy1 = yc - FLOOR_HALF, yc + FLOOR_HALF
    tv = [[[fx0, fy0, 0], [fx1, fy0, 0], [fx1, fy1, 0]],
          [[fx0, fy0, 0], [fx1, fy1, 0], [fx0, fy1, 0]]]
    mats = [0, 0]
    _box(tv, mats, xc - BODY_DX / 2, xc + BODY_DX / 2,
         yc - BODY_LY / 2, yc + BODY_LY / 2, 0.0, BODY_H, 1)
    tv = np.asarray(tv, np.float32)
    tm = np.asarray(mats, np.int32)
    # MEASURED MDF12 (ISO 10534-2 impedance tube, data_bras/3 Surface
    # descriptions/_csv/initial_estimates/mat_MDF12mm_plane_00deg.csv,
    # bands via bras_material.py)
    absn = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0.027, 0.031, 0.037, 0.043, 0.053, 0.072, 0.061]],
                    np.float32)
    z2 = np.zeros((2, 7), np.float32)
    # TRANSMISSION through the body (18 HOLLOW MDF12 blocks): the BRAS
    # doc asks for it explicitly - "sound transmission can be
    # approximated based on the density of the wood". Mass law on the
    # MEASURED m''=8.908 kg/m2 (mat_MDF12mm_plane sheet), field
    # incidence (TL_normal - 5 dB): tau = 10^(-TL/10). Derived from the
    # data, never tuned. Deep in the shadow the transmitted part (-13 dB
    # at 125 Hz) can DOMINATE the diffracted one (-25/-35 dB): omitting
    # it distorts the envelope. RS6_NOTRANS=1 restores the opaque
    # body (A/B).
    trans = z2.copy()
    if os.environ.get("RS6_NOTRANS", "0") == "0":
        trans[1] = [0.0445, 0.0111, 0.0028, 0.0007, 0.00017, 4e-5, 1e-5]
    return SceneData(tv, tm, absn, z2, trans, mat_names=["floor", "mdf12"])


def npz_path(args, pair):
    return os.path.join(args.out, "rs6utd_%s.npz" % pair)


def do_one(args, meas, pair, scene, dirv=None):
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
    print("[rs6-utd] %s: %d events (%d diffr.); Pearson %.3f; RMS %.2f dB "
          "(%d pts)" % (pair, len(res.events), n_diff, r, rms, len(grid)))


def write_report(args, rows):
    csv_path = os.path.join(args.out, "report.csv")
    with open(csv_path, "w") as f:
        f.write("pair,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            f.write("%s,%.0f,%.0f,%.3f,%.2f\n"
                    % (r["pair"], r["flo"], r["fhi"], r["r"], r["rms"]))
    lines = ["RS6 bench (geometric UTD): per-pair TF, sim vs measurement "
             "(Pearson)",
             "body 0.72x4.14x0.72 center (5.4964,2.985) OBJ  window=%s s"
             % args.win, "",
             "%-10s %9s %9s" % ("pair", "Pearson", "RMS dB")]
    for r in rows:
        lines.append("%-10s %9.3f %9.2f" % (r["pair"], r["r"], r["rms"]))
    if rows:
        lines.append("%-10s %9.3f"
                     % ("mean", float(np.mean([r["r"] for r in rows]))))
    open(os.path.join(args.out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("[rs6-utd] report -> %s/report.txt" % args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=2000.0)
    ap.add_argument("--ism-order", type=int, default=2)
    ap.add_argument("--n-rays", type=int, default=20000)
    ap.add_argument("--max-time", type=float, default=0.08)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--directivity", default=None)
    ap.add_argument("--out", default="out_rs6_utd")
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
        print("[rs6-utd] directivity: %s" % dj.get("name", "?"))
    rows = []
    for pair in args.pairs:
        if pair not in meas:
            print("[rs6-utd] %s: missing - skipped" % pair)
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
