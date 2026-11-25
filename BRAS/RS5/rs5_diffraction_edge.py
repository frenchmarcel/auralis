#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS5 edge diffraction: 25 mm MDF screen (h=2.066 m) over a rigid
floor, 16 source-mic pairs, full-wave FDTD. Single condition, so the
score is a Pearson over 160-2000 Hz on the per-pair windowed TF.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
import bras_simple as B                                  # noqa: E402
from auralis.engine.fdtd import run_low_band             # noqa: E402

# --- scene 05 constants (scene5_RIR.png diagram + SOFA) -------------
# c: same hemi-anechoic chamber as scene 01 (T ~20 C); the SOFA does
# not carry the temperature -> explicit ASSUMPTION, tunable with --c
# if Documentation.pdf gives a better value.
C_SCENE5 = 343.6
X_WALL = 5.487        # edge at x=5.487 (bottom-front center (5487,2985,0) mm)
H_WALL = 2.066        # 2066 mm
E_WALL = 0.025        # 25 mm (mat_MDF25mmB_plane)
Y_CENTER = 2.985      # screen center in y (plane of the sources/mics)
SCREEN_W = 4.75       # ACTUAL screen width (4750 mm on the drawing)


# ------------------------------------------------------------- SOFA
def load_sofa5(scene_dir):
    """Measured positions and IRs (single condition). Returns a
    dict 'LSi-MPj' -> (src(3,), mic(3,), ir(n,), fs)."""
    import h5py
    path = None
    for root, _, files in os.walk(scene_dir):
        for fn in files:
            if fn.lower().endswith("rirs.sofa") and "RIRs" in root \
                    and "BRIR" not in fn.upper():
                path = os.path.join(root, fn)
    if path is None:
        raise FileNotFoundError(
            "scene5_RIRs.sofa not found under %s (directory holding "
            "RIRs/?)" % scene_dir)
    f = h5py.File(path, "r")
    EP = f["EmitterPosition"][()]
    RP = f["ReceiverPosition"][()]
    EID = np.asarray(f["EmitterID"][()], float).ravel().astype(int)
    RID = np.asarray(f["ReceiverID"][()], float).ravel().astype(int)
    IR = f["Data.IR"][()]                 # (M, 1, N)
    fs = int(np.asarray(f["Data.SamplingRate"][()]).ravel()[0])

    def col(A, m):
        A = np.asarray(A)
        return A[0][:, m] if A.ndim == 3 else A[:, m]

    out = {}
    for m in range(IR.shape[0]):
        out["LS%d-MP%d" % (EID[m], RID[m])] = (
            col(EP, m).astype(float), col(RP, m).astype(float),
            IR[m, 0, :].astype(float), fs)
    f.close()
    return out


# ------------------------------------------------------ scene builder
def scene05(domain, x_wall=X_WALL, h=H_WALL, e=E_WALL, screen_y=None):
    """Rigid floor + thin screen at x_wall; domain = (x0, x1, y0, y1).
    screen_y=None: 'infinite' screen; (ys0, ys1): finite screen."""
    x0, x1, y0, y1 = domain
    ys0, ys1 = screen_y if screen_y is not None else (y0, y1)
    # rigid floor (2 triangles spanning the whole domain)
    tv = [[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]],
          [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]]]
    mats = [0, 0]
    # screen: closed box xw0..xw1 x ys0..ys1 x 0..h (bottom face left
    # out, it coincides with the floor; the ys0/ys1 faces are the sides)
    xw0, xw1 = x_wall - e / 2, x_wall + e / 2
    P = [(xw0, ys0, 0), (xw1, ys0, 0), (xw1, ys1, 0), (xw0, ys1, 0),
         (xw0, ys0, h), (xw1, ys0, h), (xw1, ys1, h), (xw0, ys1, h)]
    for a, b, c, d in ((0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5),
                       (3, 0, 4, 7), (4, 5, 6, 7)):
        tv.append([P[a], P[b], P[c]])
        tv.append([P[a], P[c], P[d]])
        mats += [1, 1]
    import types
    s = types.SimpleNamespace()
    s.tri_verts = np.asarray(tv, float)
    s.tri_mat = np.asarray(mats, np.int32)
    s.mat_names = ["floor", "mdf25"]
    s.absorption = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0.05, 0.05, 0.05, 0.05,
                              0.04, 0.04, 0.04]], np.float32)
    s.scattering = np.zeros((2, 7), np.float32)
    s.transmission = np.zeros((2, 7), np.float32)
    s.mat_phys = None
    return s


# ---------------------------------------------------------- processing
def npz_path(args, pair):
    return os.path.join(args.out, "rs5_%s.npz" % pair)


def do_one(args, meas, pair):
    """ONE simulation (one pair) -> npz. Runs in the current process
    (worker mode)."""
    src, mic0, ref, fs = meas[pair]
    mic = np.asarray(mic0, float).copy()
    m = args.margin
    my = args.margin_y if args.margin_y is not None else m
    # FINITE screen (--screen-width > 0): the domain must extend in y
    # past the side edges [ys0, ys1] so that they do diffract (instead
    # of being eaten by the CPML). Otherwise 'infinite' screen (y span).
    if args.screen_width > 0:
        ys0 = Y_CENTER - args.screen_width / 2
        ys1 = Y_CENTER + args.screen_width / 2
        y_lo = min(src[1], mic[1], ys0) - my
        y_hi = max(src[1], mic[1], ys1) + my
        screen_y = (ys0, ys1)
    else:
        y_lo = min(src[1], mic[1]) - my
        y_hi = max(src[1], mic[1]) + my
        screen_y = None
    dom = (min(src[0], mic[0]) - m, max(src[0], mic[0]) + m, y_lo, y_hi)
    sc = scene05(dom, screen_y=screen_y)
    # IR duration: MUST cover the latest geometric arrival time
    # (direct / floor bounce / edge diffraction) PLUS the window, or
    # the window overruns the IR and gets TRUNCATED. The ~6 m
    # source-mic separation puts the direct at ~17 ms.
    S = np.asarray(src, float)
    M = np.asarray(mic, float)
    Simg = S.copy()
    Simg[2] = -Simg[2]                     # source image (floor z=0)
    apex = np.array([X_WALL, M[1], H_WALL])  # diffracting edge
    t_late = max(np.linalg.norm(M - S),
                 np.linalg.norm(M - Simg),
                 np.linalg.norm(apex - S) + np.linalg.norm(M - apex)
                 ) / args.c
    dur = t_late + args.win + 0.008        # + t_pre + ramps + margin
    # Mic flush with the rigid floor (MP2 at z=6 mm): its cell is
    # solid -> run_low_band rejects it as outside the air. We lift it
    # to the 1st air cell (near-surface pressure, dp/dn=0 on a rigid
    # floor: the value there ~= surface pressure). LOUD fallback, not
    # silent; the true height is kept in the log.
    dx = args.c / (9.0 * 1.5 * args.fc)      # ppw=9 (run_low_band default)
    z_true = float(mic[2])
    ir = info = None
    for _ in range(6):
        try:
            ir, info = run_low_band(sc, src, [mic], fs=fs,
                                    duration=dur,
                                    f_cross=args.fc,
                                    speed_of_sound=args.c,
                                    engine=args.engine, order=4,
                                    conformal=True,
                                    max_cells=args.max_cells)
            break
        except ValueError as e:
            if "outside the air" not in str(e):
                raise
            mic[2] += 0.6 * dx
    if ir is None:
        sys.exit("[rs5] %s: cannot place the mic in the air "
                 "(6 attempts)" % pair)
    if float(mic[2]) != z_true:
        print("[rs5] WARNING %s: mic flush with the floor "
              "(%.1f mm) lifted to %.1f mm (1st air cell, "
              "near-surface pressure)"
              % (pair, z_true * 1000, float(mic[2]) * 1000))
    ir = np.asarray(ir, float).ravel()
    # anchoring: each side on its FIRST clear arrival (deep in the
    # shadow that is the diffracted wave, not the direct). The TF
    # magnitude is insensitive to a small shift (linear phase).
    i0s = B.direct_index(ir, fs)
    i0r = B.direct_index(ref, fs)
    frs, S = B.windowed_tf(ir, fs, i0s, t_win=args.win)
    frr, R = B.windowed_tf(ref, fs, i0r, t_win=args.win)
    grid = frs[(frs >= args.fmin) & (frs <= args.fmax)]
    ds = np.interp(grid, frs, 20 * np.log10(np.abs(S) + 1e-15))
    dr = np.interp(grid, frr, 20 * np.log10(np.abs(R) + 1e-15))
    r = B.pearson(ds, dr)
    # SHAPE RMS (means removed): the absolute level differs (sim in
    # 1/d vs measurement calibrated in Pascal) and is not the target,
    # only the spectral shape (diffraction + floor comb) matters, as
    # for Pearson. With no condition to subtract, do NOT compare raw dB.
    rms = float(np.sqrt(np.mean(((ds - ds.mean())
                                 - (dr - dr.mean())) ** 2)))
    # The sim IR and the anchors are saved too: window and anchoring
    # can then be replayed in post-processing WITHOUT re-simulating
    # (boundary decontamination studies then need no GPU).
    np.savez(npz_path(args, pair), grid=grid, tf_sim=ds, tf_ref=dr,
             pearson=r, rms=rms, fc=args.fc, win=args.win,
             ir_sim=ir, i0_sim=i0s, i0_ref=i0r, fs=fs,
             screen_width=args.screen_width)
    print("[rs5] %s: [%s] %d cells; Pearson %.3f; RMS %.2f dB "
          "(%d pts)" % (pair, info.engine, info.n_air, r, rms,
                        len(grid)))


def write_report(args, rows):
    """report.csv (machine) + report.txt (readable): Pearson/RMS per
    pair, same 'report then figures' flow as RS1/CR2."""
    csv_path = os.path.join(args.out, "report.csv")
    with open(csv_path, "w") as f:
        f.write("pair,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            f.write("%s,%.0f,%.0f,%.3f,%.2f\n"
                    % (r["pair"], r["flo"], r["fhi"], r["r"], r["rms"]))
    lines = ["RS5 bench: per-pair windowed TF, sim vs measurement "
             "(Pearson, Treble metric; single condition)",
             "scene=%s  fc=%s  window=%s s  edge x=%.3f z=%.3f  "
             "c=%.1f" % (args.scene_dir, args.fc, args.win,
                         X_WALL, H_WALL, args.c), "",
             "%-10s %9s %9s" % ("pair", "Pearson", "RMS dB")]
    for r in rows:
        lines.append("%-10s %9.3f %9.2f"
                     % (r["pair"], r["r"], r["rms"]))
    if rows:
        lines.append("%-10s %9.3f"
                     % ("mean", float(np.mean([r["r"] for r in rows]))))
    txt_path = os.path.join(args.out, "report.txt")
    open(txt_path, "w").write("\n".join(lines) + "\n")
    print("[rs5] report -> %s , %s" % (txt_path, csv_path))


# -------------------------------------------------------------- bench
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="default: the 16 pairs present")
    ap.add_argument("--fc", type=float, default=2000.0)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=2000.0)
    ap.add_argument("--c", type=float, default=C_SCENE5,
                    help="speed of sound (default: RS1 chamber, "
                         "assumed)")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--max-cells", type=int, default=80_000_000,
                    help="heavy RS5 grid at fc=2000 (~60-70 M "
                         "cells): GPU advised")
    ap.add_argument("--margin", type=float, default=1.6,
                    help="domain margin (m); generous because edge "
                         "artifacts do not cancel out here")
    ap.add_argument("--screen-width", type=float, default=0.0,
                    help="0 = 'infinite' screen (y span); >0 = FINITE "
                         "screen of that width (m) centered on y=2.985 "
                         "-> its side edges diffract (actual: 4.75)."
                         " Widens the domain in y (bigger grid).")
    ap.add_argument("--margin-y", type=float, default=None,
                    help="margin in y (m); default = --margin. Widens "
                         "the domain in y WITHOUT a finite screen (a "
                         "control: isolate CPML decontamination from "
                         "the physics of the finite screen).")
    ap.add_argument("--out", default="out_rs5")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the npz exists")
    ap.add_argument("--report-only", action="store_true",
                    help="simulate nothing: report/figures from the "
                         "existing npz")
    ap.add_argument("--no-plots", action="store_true",
                    help="do not generate the PNG figures")
    ap.add_argument("--_worker", metavar="PAIR", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    # HONEST scoring rule: the sim is only valid up to f_max = 1.5*fc
    # (the grid is sized for ppw=9 there). We NEVER score beyond it
    # (numerically wrong data), and we NEVER shrink below it to dodge
    # a sub-band (that would be fitting the metric). So fmax is capped
    # at 1.5*fc, no lower.
    fmax_valid = 1.5 * args.fc
    if args.fmax > fmax_valid + 1e-6:
        print("[rs5] WARNING --fmax %.0f > 1.5*fc=%.0f: the sim does "
              "not exist beyond; fmax capped at %.0f (raise --fc to "
              "score higher)." % (args.fmax, fmax_valid,
                                  fmax_valid))
        args.fmax = fmax_valid
    os.makedirs(args.out, exist_ok=True)

    meas = load_sofa5(args.scene_dir)
    if args.pairs is None:
        args.pairs = sorted(meas.keys(),
                            key=lambda k: (int(k.split("-")[0][2:]),
                                           int(k.split("MP")[1])))

    if args._worker:
        do_one(args, meas, args._worker)
        return

    # run (sub-process) or load the cache for each pair
    for pair in args.pairs:
        if pair not in meas:
            print("[rs5] %s: missing from the measurements, skipped"
                  % pair)
            continue
        pth = npz_path(args, pair)
        usable = os.path.exists(pth) and not args.force
        if usable:
            z = np.load(pth)
            sw = float(z["screen_width"]) if "screen_width" in z else 0.0
            usable = ("fc" not in z
                      or abs(float(z["fc"]) - args.fc) < 1e-6) \
                and abs(sw - args.screen_width) < 1e-6
            if usable:
                print("[rs5] %s: cached" % pair)
        if not usable:
            if args.report_only:
                print("[rs5] --report-only: no npz for %s, pair "
                      "ignored" % pair)
                continue
            import subprocess
            cmd = [sys.executable, os.path.abspath(__file__),
                   args.scene_dir, "--_worker", pair,
                   "--fc", str(args.fc), "--win", str(args.win),
                   "--fmin", str(args.fmin), "--fmax", str(args.fmax),
                   "--c", str(args.c), "--engine", args.engine,
                   "--max-cells", str(args.max_cells),
                   "--margin", str(args.margin),
                   "--screen-width", str(args.screen_width),
                   "--out", args.out]
            if args.margin_y is not None:
                cmd += ["--margin-y", str(args.margin_y)]
            rc = subprocess.call(cmd)
            if rc != 0:
                sys.exit("worker %s: code %d" % (pair, rc))

    # build report + figures from the npz
    rows = []
    for pair in args.pairs:
        pth = npz_path(args, pair)
        if not os.path.exists(pth):
            continue
        z = np.load(pth)
        rows.append({"pair": pair, "r": float(z["pearson"]),
                     "rms": float(z["rms"]), "flo": float(z["grid"][0]),
                     "fhi": float(z["grid"][-1])})
        if not args.no_plots:
            g, ds, dr = z["grid"], z["tf_sim"], z["tf_ref"]
            # display-only vertical alignment (Pearson is invariant to
            # it): measurement set to 0 at the top, sim on its median
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
