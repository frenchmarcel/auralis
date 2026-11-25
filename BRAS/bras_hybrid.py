#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRAS - generic FDTD + UTD hybrid for the object scenes (rs5, rs6,
rs7). Fuses the FDTD IR (low band) and the UTD IR (high band) with
`fuse_low_high` at the given crossover, from the npz the harnesses wrote.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import bras_simple as B                                 # noqa: E402
sys.path.insert(0, os.path.join(_HERE, "RS5"))
import rs5_diffraction_edge as R                       # noqa: E402 (load_sofa5)
from auralis.engine.fdtd import fuse_low_high           # noqa: E402


def do_one(args, meas, pair):
    """Fuse the FDTD IR (low band) and the UTD IR (high band) of a pair."""
    fdp = os.path.join(args.fdtd_dir, "%s_%s.npz" % (args.geom, pair))
    utp = os.path.join(args.utd_dir, "%sutd_%s.npz" % (args.geom, pair))
    if not (os.path.exists(fdp) and os.path.exists(utp)):
        print("[bras-hyb] %s: FDTD or UTD npz missing - skipped" % pair)
        return None
    zf = np.load(fdp, allow_pickle=True)
    zu = np.load(utp, allow_pickle=True)
    fs = int(zf["fs"])
    ref = meas[pair][2]
    irf = np.asarray(zf["ir_sim"], float)
    iru = np.asarray(zu["ir_sim"], float)
    # same time origin (both place the events at the true propagation
    # time): pad to the same length, then fuse.
    n = max(len(irf), len(iru))
    irf = np.pad(irf, (0, n - len(irf)))
    iru = np.pad(iru, (0, n - len(iru)))
    ir = fuse_low_high(iru[:, None], irf, args.fcross, fs)[:, 0]

    i0s = B.direct_index(ir, fs)
    i0r = B.direct_index(ref, fs)
    frs, S = B.windowed_tf(ir, fs, i0s, t_win=args.win)
    frr, Rr = B.windowed_tf(ref, fs, i0r, t_win=args.win)
    grid = frs[(frs >= args.fmin) & (frs <= args.fmax)]
    ds = np.interp(grid, frs, 20 * np.log10(np.abs(S) + 1e-15))
    dr = np.interp(grid, frr, 20 * np.log10(np.abs(Rr) + 1e-15))
    r = B.pearson(ds, dr)
    rms = float(np.sqrt(np.mean(((ds - ds.mean()) - (dr - dr.mean())) ** 2)))
    np.savez(os.path.join(args.out, "%shyb_%s.npz" % (args.geom, pair)),
             grid=grid, tf_sim=ds, tf_ref=dr, pearson=r, rms=rms,
             win=args.win, fcross=args.fcross, ir_sim=ir, fs=fs)
    print("[bras-hyb] %s: Pearson %.3f ; RMS %.2f dB (%d pts)"
          % (pair, r, rms, len(grid)))
    return {"pair": pair, "r": r, "rms": rms,
            "flo": float(grid[0]), "fhi": float(grid[-1])}


def write_report(args, rows):
    csv_path = os.path.join(args.out, "report.csv")
    with open(csv_path, "w") as f:
        f.write("pair,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            f.write("%s,%.0f,%.0f,%.3f,%.2f\n"
                    % (r["pair"], r["flo"], r["fhi"], r["r"], r["rms"]))
    lines = ["bench %s (FDTD+UTD hybrid): windowed TF per pair, sim vs "
             "measurement (Pearson)" % args.geom.upper(),
             "fuse_low_high blend, crossover=%.0f Hz  window=%s s  "
             "band %.0f-%.0f" % (args.fcross, args.win, args.fmin,
                                 args.fmax), "",
             "%-10s %9s %9s" % ("pair", "Pearson", "RMS dB")]
    for r in rows:
        lines.append("%-10s %9.3f %9.2f" % (r["pair"], r["r"], r["rms"]))
    if rows:
        lines.append("%-10s %9.3f"
                     % ("mean", float(np.mean([r["r"] for r in rows]))))
    txt_path = os.path.join(args.out, "report.txt")
    open(txt_path, "w").write("\n".join(lines) + "\n")
    print("[bras-hyb] report -> %s , %s" % (txt_path, csv_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", help="scene directory (measurements)")
    ap.add_argument("--geom", required=True, choices=["rs5", "rs6", "rs7"])
    ap.add_argument("--fdtd-dir", required=True,
                    help="output of the FDTD harness (npz <geom>_*.npz)")
    ap.add_argument("--utd-dir", required=True,
                    help="output of the UTD harness (npz <geom>utd_*.npz)")
    ap.add_argument("--fcross", type=float, default=700.0,
                    help="FDTD/UTD crossover frequency (Hz) - a physical "
                         "choice, NOT to be tuned on the target")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=1400.0)
    ap.add_argument("--out", default="out_hybride")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meas = R.load_sofa5(args.scene_dir)
    if args.pairs is None:
        args.pairs = sorted(meas.keys(),
                            key=lambda k: (int(k.split("-")[0][2:]),
                                           int(k.split("MP")[1])))
    rows = []
    for pair in args.pairs:
        if pair not in meas:
            print("[bras-hyb] %s: not in the measurements - skipped" % pair)
            continue
        row = do_one(args, meas, pair)
        if row is None:
            continue
        rows.append(row)
        if not args.no_plots:
            z = np.load(os.path.join(args.out, "%shyb_%s.npz"
                                     % (args.geom, pair)),
                        allow_pickle=True)
            g, ds, dr = z["grid"], z["tf_sim"], z["tf_ref"]
            off_r = -float(np.max(dr))
            off_s = float(np.median((dr + off_r) - ds))
            B._plot(os.path.join(args.out, "TF_%s.png" % pair), g,
                    [(dr + off_r, "measured", "-"),
                     (ds + off_s, "Auralis hybrid", "--")],
                    "TF %s hybrid (Pearson %.3f)"
                    % (pair, float(z["pearson"])))
    if rows:
        write_report(args, rows)


if __name__ == "__main__":
    main()
