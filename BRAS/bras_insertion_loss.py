#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRAS -- generic insertion loss of the object scenes (rs5, rs6, rs7).
IL(f) = TF_with_object - TF_free_field (dB), the free field being computed
(direct + floor, measured directivity, no object) and removed from both.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
for d in ("RS5", "RS6", "RS7"):
    sys.path.insert(0, os.path.join(_HERE, d))
import bras_simple as B                                  # noqa: E402
import rs5_diffraction_edge as R                        # noqa: E402
from auralis.engine.geometry import SceneData            # noqa: E402
from auralis.engine.pipeline import (BakeConfig, SourceDef,  # noqa: E402
                                     ReceiverDef, bake)
from auralis.engine.ir_synthesis import synthesize_mono  # noqa: E402
from auralis.engine.directivity import Directivity       # noqa: E402


def load_geom(name):
    """(UTD module, (x, y) centre of the reference floor)."""
    if name == "rs5":
        import rs5_diffraction_utd as U
        return U, (R.X_WALL, R.Y_CENTER)
    if name == "rs6":
        import rs6_diffraction_utd as U
        return U, U.BODY_C
    if name == "rs7":
        import rs7_seat_dip_utd as U
        return U, (U.X_FIRST, U.Y_CENTER)
    raise SystemExit("unknown --geom: %s" % name)


def floor_scene(xc, yc, half):
    """Rigid floor alone (large rectangle), NO object -- free field."""
    tv = np.asarray([[[xc - half, yc - half, 0], [xc + half, yc - half, 0],
                      [xc + half, yc + half, 0]],
                     [[xc - half, yc - half, 0], [xc + half, yc + half, 0],
                      [xc - half, yc + half, 0]]], np.float32)
    z = np.zeros((1, 7), np.float32)
    return SceneData(tv, np.zeros(2, np.int32), z, z.copy(), z.copy(),
                     mat_names=["floor"])


def _tf_db(ir, fs, win):
    i0 = B.direct_index(ir, fs)
    fr, S = B.windowed_tf(ir, fs, i0, t_win=win)
    return fr, 20.0 * np.log10(np.abs(S) + 1e-15)


def _smooth_oct(f, db, noct):
    """1/noct octave smoothing, applied identically to the simulated and
    the measured data (noct=3: third octave; 0 or None: raw narrowband)."""
    if not noct:
        return db
    out = np.empty_like(db)
    lo, hi = 2.0 ** (-0.5 / noct), 2.0 ** (0.5 / noct)
    for i, fc in enumerate(f):
        m = (f >= fc * lo) & (f <= fc * hi)
        out[i] = db[m].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--geom", required=True, choices=["rs5", "rs6", "rs7"])
    ap.add_argument("--sim-dir", required=True,
                    help="harness npz directory (saved simulated IR)")
    ap.add_argument("--npz-prefix", required=True,
                    help="npz prefix: rs6 | rs6utd | rs7 | rs7utd | ...")
    ap.add_argument("--directivity", required=True,
                    help="Genelec directivity JSON (for the free field)")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=1400.0)
    ap.add_argument("--out", default="out_il")
    ap.add_argument("--smooth", type=int, default=0, metavar="N",
                    help="1/N octave smoothing (3 = third octave, the "
                         "usual room-acoustics practice; 0 = raw "
                         "narrowband). Applied identically to the "
                         "simulated AND the measured data -- an analysis "
                         "resolution, not a fit.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    U, (fxc, fyc) = load_geom(args.geom)
    meas = R.load_sofa5(args.scene_dir)
    if args.pairs is None:
        args.pairs = sorted(meas.keys(),
                            key=lambda k: (int(k.split("-")[0][2:]),
                                           int(k.split("MP")[1])))
    dj = json.load(open(args.directivity))
    dirv = Directivity(np.asarray(dj["table"], dtype=float),
                       name=dj.get("name", "source"))
    floor = floor_scene(fxc, fyc, U.FLOOR_HALF)

    rows = []
    for pair in args.pairs:
        pth = os.path.join(args.sim_dir, "%s_%s.npz"
                           % (args.npz_prefix, pair))
        if pair not in meas or not os.path.exists(pth):
            print("[bras-il] %s: missing -- skipped" % pair)
            continue
        z = np.load(pth, allow_pickle=True)
        fs = int(z["fs"])
        ir_obj = np.asarray(z["ir_sim"], float)
        ref = meas[pair][2]
        # reference free field: direct + floor, directivity, NO object
        cfg = BakeConfig(fs=fs, temperature_c=U.TEMP_C,
                         humidity_percent=U.HUMID, diffraction=False,
                         fdtd=False, ism_order=1, n_rays=20000,
                         max_time=0.08, normalize=False, seed=1234)
        res = bake(floor,
                   SourceDef(name="s",
                             position=np.asarray(meas[pair][0], float),
                             directivity=dirv,
                             forward=U.source_forward(pair)),
                   ReceiverDef(name="r",
                               position=np.asarray(meas[pair][1], float)),
                   cfg)
        ir_ff = np.asarray(synthesize_mono(res.events, res.hist, cfg.bin_dt,
                                           res.fs, seed=cfg.seed), float)
        frb, db_b = _tf_db(ir_obj, fs, args.win)
        frr, db_r = _tf_db(ref, fs, args.win)
        frf, db_f = _tf_db(ir_ff, fs, args.win)
        g = frb[(frb >= args.fmin) & (frb <= args.fmax)]
        sb = np.interp(g, frb, db_b)
        sr = np.interp(g, frr, db_r)
        sff = np.interp(g, frf, db_f)
        il_sim, il_meas = sb - sff, sr - sff       # insertion loss (dB)
        if args.smooth:
            sb = _smooth_oct(g, sb, args.smooth)
            sr = _smooth_oct(g, sr, args.smooth)
            il_sim = _smooth_oct(g, il_sim, args.smooth)
            il_meas = _smooth_oct(g, il_meas, args.smooth)
        p_raw = B.pearson(sb, sr)
        p_il = B.pearson(il_sim, il_meas)
        rows.append({"pair": pair, "raw": p_raw, "il": p_il,
                     "flo": float(g[0]), "fhi": float(g[-1])})
        print("[bras-il] %s: raw TF %.3f  ->  InsLoss %.3f"
              % (pair, p_raw, p_il))
        if not args.no_plots:
            # DISPLAY alignment only (Pearson is shift invariant)
            off_r = -float(np.max(il_meas))
            off_s = float(np.median((il_meas + off_r) - il_sim))
            B._plot(os.path.join(args.out, "IL_%s.png" % pair), g,
                    [(il_meas + off_r, "measured", "-"),
                     (il_sim + off_s, "Auralis", "--")],
                    "Insertion loss %s (Pearson %.3f)" % (pair, p_il),
                    ylab="Insertion loss (dB, aligned)")
    if rows:
        csv_path = os.path.join(args.out, "report.csv")
        with open(csv_path, "w") as f:
            f.write("pair,f_lo_Hz,f_hi_Hz,pearson_tf,pearson_il\n")
            for r in rows:
                f.write("%s,%.0f,%.0f,%.3f,%.3f\n"
                        % (r["pair"], r["flo"], r["fhi"], r["raw"], r["il"]))
        lines = ["%s bench: INSERTION LOSS (object vs computed free field)"
                 % args.geom.upper(),
                 "sim=%s/%s  directivity=%s  band %.0f-%.0f"
                 % (args.sim_dir, args.npz_prefix, dj.get("name", "?"),
                    args.fmin, args.fmax), "",
                 "%-10s %10s %10s" % ("pair", "raw TF", "InsLoss")]
        for r in rows:
            lines.append("%-10s %10.3f %10.3f"
                         % (r["pair"], r["raw"], r["il"]))
        lines.append("%-10s %10.3f %10.3f"
                     % ("mean", float(np.mean([r["raw"] for r in rows])),
                        float(np.mean([r["il"] for r in rows]))))
        txt = os.path.join(args.out, "report.txt")
        open(txt, "w").write("\n".join(lines) + "\n")
        print("[bras-il] report -> %s , %s" % (txt, csv_path))


if __name__ == "__main__":
    main()
