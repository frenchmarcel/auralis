#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS7 (all-wave FDTD path) -- seat-dip effect from multiple diffraction
over 15 MDF beams (0.34 m pitch, official BRAS OBJ) laid on a rigid
floor; one subprocess per pair, npz cache, windowed TF and Pearson."""
import argparse
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "RS5"))
import bras_simple as B                                   # noqa: E402
import rs5_diffraction_edge as R                         # noqa: E402
import rs7_seat_dip_utd as U7                             # noqa: E402 (beams)
from auralis.engine.fdtd import run_low_band              # noqa: E402

# Cramer c (JASA 1993) at 19.2 degC / 40.3 % (BRAS Documentation.pdf) --
# the dry formula (331.3+0.606T=342.94) underestimates by ~0.4 m/s
C = 343.36
# MDF12 MEASURED (ISO 10534-2 impedance tube, data_bras/3 Surface
# descriptions/_csv/initial_estimates/mat_MDF12mm_plane_00deg.csv):
# full third-octave curve -> the wave band follows the true shape
MDF_FB = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                   315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                   3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
                   20000], float)
MDF_AB = np.array([0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.022,
                   0.028, 0.030, 0.030, 0.030, 0.033, 0.035, 0.036, 0.040,
                   0.044, 0.043, 0.041, 0.042, 0.053, 0.065, 0.081, 0.073,
                   0.061, 0.060, 0.061, 0.062, 0.060, 0.060, 0.059], float)


def scene_fdtd(domain):
    """Full-footprint rigid floor + 15 MDF beams."""
    x0, x1, y0, y1 = domain
    tv = [[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]],
          [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]]]
    mats = [0, 0]
    by0, by1 = U7.Y_CENTER - U7.BEAM_LY / 2, U7.Y_CENTER + U7.BEAM_LY / 2
    for k in range(U7.N_BEAMS):
        xc = U7.X_FIRST + k * U7.PITCH
        bx0, bx1 = xc - U7.BEAM_DX / 2, xc + U7.BEAM_DX / 2
        P = [(bx0, by0, 0), (bx1, by0, 0), (bx1, by1, 0), (bx0, by1, 0),
             (bx0, by0, U7.BEAM_H), (bx1, by0, U7.BEAM_H),
             (bx1, by1, U7.BEAM_H), (bx0, by1, U7.BEAM_H)]
        for a, b, c, d in ((0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5),
                           (3, 0, 4, 7), (4, 5, 6, 7)):
            tv.append([P[a], P[b], P[c]])
            tv.append([P[a], P[c], P[d]])
            mats += [1, 1]
    s = types.SimpleNamespace()
    s.tri_verts = np.asarray(tv, float)
    s.tri_mat = np.asarray(mats, np.int32)
    s.mat_names = ["floor", "mdf12"]
    s.absorption = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0.027, 0.031, 0.037, 0.043, 0.053, 0.072,
                              0.061]], np.float32)
    s.scattering = np.zeros((2, 7), np.float32)
    s.transmission = np.zeros((2, 7), np.float32)
    if os.environ.get("RS7_NOPANEL", "0") == "0":
        # DEFAULT: HOLLOW MDF12 boxes as resonant panels -- that is what
        # the beams ARE (BRAS doc). m'' MEASURED 8.908 kg/m2
        # (mat_MDF12mm_plane sheet), 96 mm cavity DEDUCED from the OBJ
        # (0.12 - 2x12 mm), q=2 default: sourced quantities only, zero
        # tuning. f0~65 Hz sits below the band -- in band it is the
        # reflection PHASE that differs from rigid, and that is what
        # matters. Against rigid beams (fc=2000, 8 pairs): raw TF
        # 0.527 -> 0.577, 1/3 oct IL 0.848 -> 0.864, better on all 8
        # pairs. RS7_NOPANEL=1 restores the rigid body (A/B).
        s.mat_phys = [None, {"panel": {"m": 8.908, "cavity_mm": 96.0}}]
    else:
        s.mat_phys = None
    s.mat_alpha_full = [None, (MDF_FB, MDF_AB)]   # rigid floor, MDF measured
    return s


def npz_path(args, pair):
    return os.path.join(args.out, "rs7_%s.npz" % pair)


def do_one(args, meas, pair):
    src, mic0, ref, fs = meas[pair]
    mic = np.asarray(mic0, float).copy()
    m = args.margin
    x_last = U7.X_FIRST + (U7.N_BEAMS - 1) * U7.PITCH
    by0, by1 = U7.Y_CENTER - U7.BEAM_LY / 2, U7.Y_CENTER + U7.BEAM_LY / 2
    dom = (min(src[0], mic[0], U7.X_FIRST - U7.BEAM_DX) - m,
           max(src[0], mic[0], x_last + U7.BEAM_DX) + m,
           min(src[1], mic[1], by0) - m, max(src[1], mic[1], by1) + m)
    sc = scene_fdtd(dom)
    # duration: the seat-dip tail (multiple diffraction) is long -> a
    # wider window is possible; we cover the latest arrival + the window.
    S, M = np.asarray(src, float), np.asarray(mic, float)
    Simg = S.copy(); Simg[2] = -Simg[2]
    t_geo = max(np.linalg.norm(M - S), np.linalg.norm(M - Simg)) / args.c
    dur = t_geo + args.win + 0.010
    dx = args.c / (9.0 * 1.5 * args.fc)
    z_true = float(mic[2])
    ir = info = None
    for _ in range(6):
        try:
            ir, info = run_low_band(sc, src, [mic], fs=fs, duration=dur,
                                    f_cross=args.fc, speed_of_sound=args.c,
                                    engine=args.engine, order=4,
                                    conformal=True, max_cells=args.max_cells)
            break
        except ValueError as e:
            if "outside the air" not in str(e):
                raise
            mic[2] += 0.6 * dx
    if ir is None:
        sys.exit("[rs7] %s: cannot place the mic in air" % pair)
    if float(mic[2]) != z_true:
        print("[rs7] WARNING %s: mic (%.1f mm) raised to %.1f mm"
              % (pair, z_true * 1000, float(mic[2]) * 1000))
    ir = np.asarray(ir, float).ravel()
    i0s = B.direct_index(ir, fs)
    i0r = B.direct_index(ref, fs)
    frs, Sf = B.windowed_tf(ir, fs, i0s, t_win=args.win)
    frr, Rf = B.windowed_tf(ref, fs, i0r, t_win=args.win)
    grid = frs[(frs >= args.fmin) & (frs <= args.fmax)]
    ds = np.interp(grid, frs, 20 * np.log10(np.abs(Sf) + 1e-15))
    dr = np.interp(grid, frr, 20 * np.log10(np.abs(Rf) + 1e-15))
    r = B.pearson(ds, dr)
    rms = float(np.sqrt(np.mean(((ds - ds.mean()) - (dr - dr.mean())) ** 2)))
    np.savez(npz_path(args, pair), grid=grid, tf_sim=ds, tf_ref=dr,
             pearson=r, rms=rms, fc=args.fc, win=args.win, ir_sim=ir,
             i0_sim=i0s, i0_ref=i0r, fs=fs)
    print("[rs7] %s: [%s] %d cells; Pearson %.3f; RMS %.2f dB (%d pts)"
          % (pair, info.engine, info.n_air, r, rms, len(grid)))


def write_report(args, rows):
    with open(os.path.join(args.out, "report.csv"), "w") as f:
        f.write("pair,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            f.write("%s,%.0f,%.0f,%.3f,%.2f\n"
                    % (r["pair"], r["flo"], r["fhi"], r["r"], r["rms"]))
    lines = ["RS7 bench (all-wave FDTD): seat-dip, TF per pair, sim vs meas.",
             "15 beams pitch 0.34 (official OBJ)  fc=%s  window=%s s"
             % (args.fc, args.win),
             "", "%-10s %9s %9s" % ("pair", "Pearson", "RMS dB")]
    for r in rows:
        lines.append("%-10s %9.3f %9.2f" % (r["pair"], r["r"], r["rms"]))
    if rows:
        lines.append("%-10s %9.3f"
                     % ("mean", float(np.mean([r["r"] for r in rows]))))
    open(os.path.join(args.out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("[rs7] report -> %s/report.txt" % args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--fc", type=float, default=1000.0)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=1400.0)
    ap.add_argument("--c", type=float, default=C)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--max-cells", type=int, default=70_000_000)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--out", default="out_rs7")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--_worker", metavar="PAIR", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    fmax_valid = 1.5 * args.fc
    if args.fmax > fmax_valid + 1e-6:
        print("[rs7] --fmax clamped to 1.5*fc=%.0f" % fmax_valid)
        args.fmax = fmax_valid
    os.makedirs(args.out, exist_ok=True)
    meas = R.load_sofa5(args.scene_dir)
    if args.pairs is None:
        args.pairs = sorted(meas.keys(),
                            key=lambda k: (int(k.split("-")[0][2:]),
                                           int(k.split("MP")[1])))
    if args._worker:
        do_one(args, meas, args._worker)
        return
    for pair in args.pairs:
        if pair not in meas:
            print("[rs7] %s: missing -- skipped" % pair)
            continue
        pth = npz_path(args, pair)
        usable = os.path.exists(pth) and not args.force
        if usable:
            z = np.load(pth)
            usable = "fc" not in z or abs(float(z["fc"]) - args.fc) < 1e-6
            if usable:
                print("[rs7] %s: cached" % pair)
        if not usable:
            if args.report_only:
                print("[rs7] --report-only: %s missing -- ignored" % pair)
                continue
            import subprocess
            cmd = [sys.executable, os.path.abspath(__file__), args.scene_dir,
                   "--_worker", pair, "--fc", str(args.fc), "--win",
                   str(args.win), "--fmin", str(args.fmin), "--fmax",
                   str(args.fmax), "--c", str(args.c), "--engine", args.engine,
                   "--max-cells", str(args.max_cells), "--margin",
                   str(args.margin), "--out", args.out]
            rc = subprocess.call(cmd)
            if rc != 0:
                sys.exit("worker %s: code %d" % (pair, rc))
    rows = []
    for pair in args.pairs:
        pth = npz_path(args, pair)
        if not os.path.exists(pth):
            continue
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
                     (ds + off_s, "Auralis", "--")],
                    "TF %s (Pearson %.3f)" % (pair, float(z["pearson"])))
    if rows:
        write_report(args, rows)


if __name__ == "__main__":
    main()
