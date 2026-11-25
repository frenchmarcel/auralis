#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRAS bench - fundamental scenes (RS1, extensible to RS5/6/7).
Full-wave in an open domain (CPML); the metric is TF differences
between conditions (Pearson). Dependency: h5py (SOFA reading).
"""
import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auralis.engine.fdtd import run_low_band          # noqa: E402
from auralis.engine.reflection import (               # noqa: E402
    fit_admittance_multiangle)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bras_style                                   # noqa: E402 (shared style)

C_SCENE1 = 343.6      # 20.3 C (BRAS documentation)

# ERRATUM scene 01: the SOFA ReceiverPosition carry z+0.240 m with
# respect to the labels of the official diagram (which prevail,
# cf. Documentation.pdf) AND to the measured acoustics (Delta of the
# dips and of the arrivals checked on 4 pairs within 1-2 %). True
# heights, symmetric with the sources:
SCENE1_MP_Z = {1: 1.500, 2: 2.121, 3: 2.598, 4: 1.573}


def fix_scene1_positions(pairs_dict):
    out = {}
    for key, (src, mic, ir, fs) in pairs_dict.items():
        mp = int(key.split("MP")[1])
        mic = mic.copy()
        if mp in SCENE1_MP_Z:
            mic[2] = SCENE1_MP_Z[mp]
        out[key] = (src, mic, ir, fs)
    return out


# ---------------------------------------------------------------- SOFA
def load_sofa(scene_dir, condition):
    """Measured positions and IRs of one condition. Returns a
    dict 'LSi-MPj' -> (src(3,), mic(3,), ir(n,), fs)."""
    import h5py
    path = None
    for root, _, files in os.walk(scene_dir):
        for fn in files:
            if fn.lower().endswith("_%s.sofa" % condition.lower()) and \
                    "RIRs" in root and "BRIR" not in fn:
                path = os.path.join(root, fn)
    if path is None:
        raise FileNotFoundError("no SOFA RIR for condition %s"
                                % condition)
    f = h5py.File(path, "r")
    EP = f["EmitterPosition"][()][0]      # (3, E)
    RP = f["ReceiverPosition"][()][0]
    EID = np.asarray(f["EmitterID"][()], float).ravel().astype(int)
    RID = np.asarray(f["ReceiverID"][()], float).ravel().astype(int)
    IR = f["Data.IR"][()]                 # (M, 1, N)
    fs = int(np.asarray(f["Data.SamplingRate"][()]).ravel()[0])
    out = {}
    for m in range(IR.shape[0]):
        e, r = EID[m], RID[m]
        # unique positions per ID: the EP/RP columns follow the order
        # of the measurements in the BRAS files (E = M)
        src = EP[:, m].astype(float)
        mic = RP[:, m].astype(float)
        out["LS%d-MP%d" % (e, r)] = (src, mic,
                                     IR[m, 0, :].astype(float), fs)
    f.close()
    return out


# ------------------------------------------------------ scene builders
def scene01(condition, domain, rockfon=None, diffuse_phase="low",
            diffuse_full=False, slat_width=0.120):
    """Floor spanning the whole domain ('infinite' plane), material set
    by the condition. domain = (x0, x1, y0, y1); rockfon = (gi, A, l)."""
    x0, x1, y0, y1 = domain
    tv = np.array([[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]],
                   [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]]], float)
    s = types.SimpleNamespace()
    s.tri_verts = tv
    s.tri_mat = np.zeros(2, np.int32)
    s.mat_names = ["floor"]
    s.absorption = np.zeros((1, 7), np.float32)
    s.scattering = np.zeros((1, 7), np.float32)
    s.transmission = np.zeros((1, 7), np.float32)
    s.mat_phys = None
    if condition.lower() == "rigid":
        pass
    elif condition.lower() == "absorbing":
        # PHYSICAL model (Miki, rigid backing) fitted on the
        # multi-angle alphas: the reflection PHASE then comes out
        # right by construction (an alpha-only fit is blind to phase,
        # unusable for an interference bench).
        s.mat_phys = [{"layers": [{"model": "miki",
                                   "sigma": 127128.0, "d": 0.020}]}]
    elif condition.lower() in ("diffuse", "diffusing"):
        # BRAS 1-D diffuser (skp labels + geometric consistency):
        # 33 MDF slats 120 mm wide, 1 mm spacing, alternating heights
        # 120/240 mm (17 low / 16 high, starts and ends low),
        # length 4.14 m; anchor (3.505, 0.915, 0), center =
        # reflection point (5.5, 2.985).
        tvs = [tv[0], tv[1]]
        mats = [0, 0]
        # slats TRUNCATED to the domain extent: past the margin the
        # world is absorbed by the CPML anyway, a consistent
        # approximation, tolerated by the difference metric.
        if diffuse_full:
            yb0, yb1 = 0.915, 0.915 + 4.14
        else:
            yb0 = max(0.915, y0 + 0.02)
            yb1 = min(0.915 + 4.14, y1 - 0.02)
        for k in range(33):
            if diffuse_full:
                bx0 = 3.505 + k * 0.121
                bx1 = bx0 + slat_width
            else:
                bx0 = max(3.505 + k * 0.121, x0 + 0.02)
                bx1 = min(3.505 + k * 0.121 + slat_width, x1 - 0.02)
            if bx1 - bx0 < 0.03:
                continue
            low_first = (diffuse_phase != "high")
            bz1 = 0.120 if ((k % 2 == 0) == low_first) else 0.240
            P = [(bx0, yb0, 0), (bx1, yb0, 0), (bx1, yb1, 0),
                 (bx0, yb1, 0), (bx0, yb0, bz1), (bx1, yb0, bz1),
                 (bx1, yb1, bz1), (bx0, yb1, bz1)]
            for a, b, c_, d_ in ((0, 1, 5, 4), (2, 3, 7, 6),
                                 (1, 2, 6, 5), (3, 0, 4, 7),
                                 (4, 5, 6, 7)):
                tvs.append([P[a], P[b], P[c_]])
                tvs.append([P[a], P[c_], P[d_]])
                mats += [1, 1]
        s.tri_verts = np.asarray(tvs, float)
        s.tri_mat = np.asarray(mats, np.int32)
        s.mat_names = ["floor", "mdf12"]
        s.absorption = np.array([[0, 0, 0, 0, 0, 0, 0],
                                 [0.05, 0.05, 0.05, 0.05,
                                  0.04, 0.04, 0.04]], np.float32)
        s.scattering = np.zeros((2, 7), np.float32)
        s.transmission = np.zeros((2, 7), np.float32)
    else:
        raise NotImplementedError("unknown condition %s" % condition)
    return s


def rockfon_gamma(csv_dir):
    """Multi-angle fit of the Rockfon Sonar G from the BRAS CSVs.
    Returns (gi, A(2,), l(2,)) ready for scene.mat_gamma."""
    import csv as _csv

    def load(p):
        rows = list(_csv.reader(open(p, encoding="utf-8-sig"),
                                delimiter=';'))
        if len(rows[0]) < 3:
            rows = list(_csv.reader(open(p, encoding="utf-8-sig")))
        return (np.array([float(x) for x in rows[0] if str(x).strip()]),
                np.array([float(x) for x in rows[1] if str(x).strip()]))

    angs = [0.0, 30.0, 45.0, 60.0]
    A = []
    fb = None
    for th in angs:
        fb, a = load(os.path.join(
            csv_dir, "mat_RockFonSonarG_%02ddeg.csv" % int(th)))
        A.append(a)
    gi, Ab, l = fit_admittance_multiangle(
        np.asarray(A), angs, fb, f_hi=4000.0, f_lo=150.0,
        n_desc=1, n_mont=1)
    return (float(gi), np.asarray(Ab[:2]), np.asarray(l[:2]))


# ---------------------------------------------------------- processing
def direct_index(x, fs, hint=None):
    """Index of the direct sound: first clear onset (>1/4 of the max).
    hint: expected index, local search around it."""
    h = np.abs(x)
    thr = 0.25 * h.max()
    if hint is not None:
        a = max(int(hint) - int(0.002 * fs), 0)
        b = int(hint) + int(0.002 * fs)
        seg = h[a:b]
        k = a + int(np.argmax(seg))
        return k
    idx = np.nonzero(h > thr)[0]
    k = int(idx[0])
    # refine on the local peak that follows
    j = k + int(np.argmax(h[k:k + int(0.0008 * fs)]))
    return j


def windowed_tf(x, fs, i0, t_pre=0.001, t_win=0.030, nfft=1 << 17):
    a = max(i0 - int(t_pre * fs), 0)
    b = min(i0 + int(t_win * fs), len(x))
    w = np.zeros_like(x)
    w[a:b] = 1.0
    nr = max(int(0.001 * fs), 8)
    w[a:a + nr] *= np.linspace(0.0, 1.0, nr)
    w[b - nr:b] *= np.linspace(1.0, 0.0, nr)
    X = np.fft.rfft(x * w, nfft)
    fr = np.fft.rfftfreq(nfft, 1.0 / fs)
    return fr, X


def _plot(path, grid, curves, title, ylab="SPL rel. (dB)"):
    """curves: list of (y, label, style). Log x scale, style of the
    Treble figures (0 at the top, ~-45 at the bottom)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[bench] matplotlib missing: figures skipped "
              "(pip install matplotlib)")
        return False
    plt.rcParams.update(bras_style.rc())
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    for y, lab, sty in curves:
        ax.plot(grid, y, sty, color=bras_style.color_for(lab),
                label=lab, linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ticks = [t for t in (100, 160, 200, 315, 500, 800, 1000,
                         1250, 1600, 2000, 3150)
             if grid[0] <= t <= grid[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t if t < 1000 else "%gk" % (t/1000)
                        for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylab)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() /
                 np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def _write_report(args, rows):
    """RS1 report: report.csv + report.txt, Pearson/RMS per condition
    difference, per pair and per tier."""
    csv_path = os.path.join(args.out, "report.csv")
    with open(csv_path, "w") as fcsv:
        fcsv.write("difference,pair,band,f_lo_Hz,f_hi_Hz,pearson,rms_dB\n")
        for r in rows:
            fcsv.write("%s,%s,combined,%.0f,%.0f,%.3f,%.2f\n"
                       % (r["diff"], r["pair"], r["flo"], r["fhi"],
                          r["r"], r["rms"]))
            for name, (rp, rm, tlo, thi) in r["tiers"].items():
                fcsv.write("%s,%s,%s,%.0f,%.0f,%.3f,%.2f\n"
                           % (r["diff"], r["pair"], name, tlo, thi, rp, rm))

    groups = {}
    for r in rows:
        groups.setdefault(r["diff"], []).append(r)
    two_tier = any(r["tiers"] for r in rows)
    lines = ["RS1 bench: TF differences between conditions "
             "(Pearson, Treble metric)",
             "scene=%s  fc=%s  window=%s s"
             % (args.scene_dir, args.fc, args.win), ""]
    for diff, rs in groups.items():
        lines.append("[%s]" % diff)
        if two_tier:
            lines.append("%-10s %9s %9s %9s %9s"
                         % ("pair", "combined", "LF", "HF", "RMS dB"))
            for r in rs:
                lf = r["tiers"].get("LF", (float("nan"),))[0]
                hf = r["tiers"].get("HF", (float("nan"),))[0]
                lines.append("%-10s %9.3f %9.3f %9.3f %9.2f"
                             % (r["pair"], r["r"], lf, hf, r["rms"]))
        else:
            lines.append("%-10s %9s %9s" % ("pair", "combined", "RMS dB"))
            for r in rs:
                lines.append("%-10s %9.3f %9.2f"
                             % (r["pair"], r["r"], r["rms"]))
        lines.append("%-10s %9.3f" % ("mean",
                                      float(np.mean([r["r"] for r in rs]))))
        lines.append("")
    txt_path = os.path.join(args.out, "report.txt")
    open(txt_path, "w").write("\n".join(lines) + "\n")
    print("[bench] report -> %s , %s" % (txt_path, csv_path))


# -------------------------------------------------------------- bench
def run_pair(scene, src, mic, fs, fc, dur, engine, max_cells):
    ir, info = run_low_band(scene, src, mic if np.ndim(mic) > 1
                            else [np.asarray(mic)], fs=fs,
                            duration=dur, f_cross=fc,
                            speed_of_sound=C_SCENE1, engine=engine,
                            order=4, conformal=True,
                            max_cells=max_cells)
    return np.asarray(ir, float).ravel(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", nargs="+", default=["LS2-MP1"])
    ap.add_argument("--conditions", nargs="+",
                    default=["Rigid", "Absorbing"])
    ap.add_argument("--fc", type=float, default=2000.0)
    ap.add_argument("--win", type=float, default=0.030)
    ap.add_argument("--fmin", type=float, default=160.0)
    ap.add_argument("--fmax", type=float, default=None,
                    help="default: fc")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--max-cells", type=int, default=25_000_000)
    ap.add_argument("--rockfon-csv", default=None,
                    help="directory of the mat_RockFonSonarG_*.csv "
                         "(required for Absorbing)")
    ap.add_argument("--two-tier", action="store_true",
                    help="two assembled tiers: low band (whole "
                         "structure, --lf-fc) then high (--fc), "
                         "stitched at --lf-split; one 160-fmax curve")
    ap.add_argument("--lf-fc", type=float, default=800.0)
    ap.add_argument("--lf-split", type=float, default=700.0)
    ap.add_argument("--npz-prefix", default="",
                    help=argparse.SUPPRESS)
    ap.add_argument("--diffuse-full", action="store_true",
                    help="WHOLE diffuser structure + floor ring (low "
                         "band: the Fresnel zone overruns the margin) "
                         "- bigger grid, keep for fc <= ~900")
    ap.add_argument("--diffuse-phase", default="low",
                    choices=["low", "high"],
                    help="low (default) or high edge slat - A/B test "
                         "of the alternation phase")
    ap.add_argument("--slat-width", type=float, default=0.120,
                    help="width of the diffuser slats (m), pitch 0.121: 0.120 leaves the "
                         "1 mm gaps of the official model, 0.121 closes them (A/B test)")
    ap.add_argument("--margin", type=float, default=1.6,
                    help="lateral margin of the domain (m)")
    ap.add_argument("--out", default="out_rs1")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the npz exists")
    ap.add_argument("--report-only", action="store_true",
                    help="simulate nothing: differences from the "
                         "existing npz")
    ap.add_argument("--no-subprocess", action="store_true",
                    help="everything in this process (debug)")
    ap.add_argument("--no-plots", action="store_true",
                    help="do not generate the PNG figures")
    ap.add_argument("--_worker", nargs=2, metavar=("COND", "PAIR"),
                    default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    fmax = args.fmax or args.fc
    os.makedirs(args.out, exist_ok=True)

    rf = None

    def npz_path(cond, pair):
        return os.path.join(args.out, "%s%s_%s.npz"
                            % (args.npz_prefix, cond, pair))

    def do_one(cond, pair):
        """ONE simulation (condition, pair) -> npz. Runs in the
        current process (worker mode or --no-subprocess)."""
        meas0 = fix_scene1_positions(
            load_sofa(args.scene_dir, args.conditions[0]))
        measC = meas0 if cond == args.conditions[0] else             fix_scene1_positions(load_sofa(args.scene_dir, cond))
        s_, m_, ref, fs = measC[pair]
        src, mic = s_, m_
        dom = (min(src[0], mic[0]) - args.margin,
               max(src[0], mic[0]) + args.margin,
               min(src[1], mic[1]) - args.margin,
               max(src[1], mic[1]) + args.margin)
        if args.diffuse_full:
            # floor and domain enclose the real structure + a ring,
            # for ALL conditions: the difference only cancels the edge
            # artifacts if the domains coincide
            dom = (min(dom[0], 3.505 - 0.45),
                   max(dom[1], 7.498 + 0.45),
                   min(dom[2], 0.915 - 0.45),
                   max(dom[3], 5.055 + 0.45))
        sc = scene01(cond, dom, rockfon=rf,
                     diffuse_phase=args.diffuse_phase,
                     diffuse_full=args.diffuse_full,
                     slat_width=args.slat_width)
        ir, info = run_pair(sc, s_, m_, fs, args.fc,
                            args.win + 0.015, args.engine,
                            args.max_cells)
        i0s = direct_index(ir, fs)
        # time anchor: the direct sound of the FIRST condition
        anchor = direct_index(meas0[pair][2], fs)
        i0r = anchor if cond == args.conditions[0] else             direct_index(ref, fs, hint=anchor)
        frs, S = windowed_tf(ir, fs, i0s, t_win=args.win)
        frr, R = windowed_tf(ref, fs, i0r, t_win=args.win)
        grid = frs[(frs >= args.fmin) & (frs <= fmax)]
        ds = np.interp(grid, frs,
                       20 * np.log10(np.abs(S) + 1e-15))
        dr = np.interp(grid, frr,
                       20 * np.log10(np.abs(R) + 1e-15))
        np.savez(npz_path(cond, pair), ir=ir, fs=fs, grid=grid,
                 tf_sim=ds, tf_ref=dr, fc=args.fc, win=args.win,
                 diffuse_phase=args.diffuse_phase)
        print("[bench] %s %s: [%s] %d cells; TF %d pts"
              % (cond, pair, info.engine, info.n_air, len(grid)))

    if args._worker:
        do_one(args._worker[0], args._worker[1])
        return

    # pairs actually measured per condition (the BRAS sets are not
    # all complete: e.g. scene 01, LS2-MP4 missing from the Absorbing
    # condition)
    avail = {c: set(load_sofa(args.scene_dir, c).keys())
             for c in args.conditions}
    # --report-only: pairs skipped for lack of npz (we still report on
    # the cached ones instead of dropping the whole report)
    report_only_missing = set()

    def run_tier(prefix, fc_t, fmin_t, fmax_t, dfull):
        """Run or load every (cond, pair) of one tier; returns
        (tfs_sim, tfs_ref) on the grid of that tier."""
        t_sim, t_ref = {}, {}
        for pair in args.pairs:
            for cond in args.conditions:
                if pair not in avail[cond]:
                    print("[bench] %s %s: missing from the "
                          "measurements of this condition, skipped"
                          % (cond, pair))
                    continue
                pth = os.path.join(args.out, "%s%s_%s.npz"
                                   % (prefix, cond, pair))
                usable = False
                if os.path.exists(pth) and not args.force:
                    z = np.load(pth, allow_pickle=True)
                    if "fc" not in z or abs(float(z["fc"]) - fc_t) \
                            < 1e-6:
                        usable = True
                        print("[bench] %s%s %s: cached"
                              % (prefix, cond, pair))
                if not usable:
                    if args.report_only:
                        print("[bench] --report-only: no npz for "
                              "%s%s %s, pair ignored"
                              % (prefix, cond, pair))
                        report_only_missing.add(pair)
                        continue
                    import subprocess
                    cmd = [sys.executable,
                           os.path.abspath(__file__),
                           args.scene_dir, "--_worker", cond, pair,
                           "--pairs", pair,
                           "--conditions"] + args.conditions + [
                           "--fc", str(fc_t),
                           "--win", str(args.win),
                           "--fmin", str(fmin_t),
                           "--fmax", str(fmax_t),
                           "--engine", args.engine,
                           "--max-cells", str(args.max_cells),
                           "--margin", str(args.margin),
                           "--diffuse-phase", args.diffuse_phase,
                           "--slat-width", repr(args.slat_width),
                           "--npz-prefix", prefix]
                    if dfull:
                        cmd += ["--diffuse-full"]
                    cmd += ["--out", args.out]
                    if args.no_subprocess:
                        sys.exit("--no-subprocess is reserved for "
                                 "worker mode (debug)")
                    rc = subprocess.call(cmd)
                    if rc != 0:
                        sys.exit("worker %s%s %s: code %d"
                                 % (prefix, cond, pair, rc))
                z = np.load(pth, allow_pickle=True)
                t_sim[(cond, pair)] = (z["grid"], z["tf_sim"])
                t_ref[(cond, pair)] = (z["grid"], z["tf_ref"])
        return t_sim, t_ref

    if args.two_tier:
        sp = args.lf_split
        lf_s, lf_r = run_tier("LF_", args.lf_fc, args.fmin,
                              sp * 1.08, True)
        hf_s, hf_r = run_tier("", args.fc, sp * 0.92, fmax,
                              args.diffuse_full)
        tiers = {"LF": (lf_s, lf_r, args.fmin, sp),
                 "HF": (hf_s, hf_r, sp, fmax)}
        tfs_sim, tfs_ref = {}, {}
        for key in hf_s:
            if key not in lf_s:
                tfs_sim[key] = hf_s[key]
                tfs_ref[key] = hf_r[key]
                continue
            gl, sl = lf_s[key]
            gh, sh = hf_s[key]
            _, rl = lf_r[key]
            _, rh = hf_r[key]
            # stitching: align the low tier on the high one over the
            # overlap zone (run gains are arbitrary)
            ovl = (gl >= sp * 0.94) & (gl <= sp * 1.06)
            ovh = (gh >= sp * 0.94) & (gh <= sp * 1.06)
            if ovl.any() and ovh.any():
                sl = sl + (np.median(sh[ovh]) - np.median(sl[ovl]))
                rl = rl + (np.median(rh[ovh]) - np.median(rl[ovl]))
            ml, mh = gl < sp, gh >= sp
            g = np.concatenate([gl[ml], gh[mh]])
            tfs_sim[key] = (g, np.concatenate([sl[ml], sh[mh]]))
            tfs_ref[key] = (g, np.concatenate([rl[ml], rh[mh]]))
    else:
        tfs_sim, tfs_ref = run_tier(args.npz_prefix, args.fc,
                                    args.fmin, fmax,
                                    args.diffuse_full)
    # TF figures per (condition, pair)
    if not args.no_plots:
        for (cond, pair), (g_, ds_) in tfs_sim.items():
            _, dr_ = tfs_ref[(cond, pair)]
            off_r = -float(np.max(dr_))
            off_s = float(np.median((dr_ + off_r) - ds_))
            _plot(os.path.join(args.out,
                               "TF_%s_%s.png" % (cond, pair)),
                  g_, [(dr_ + off_r, "measured", "-"),
                       (ds_ + off_s, "Auralis", "--")],
                  "TF %s %s" % (cond, pair))
    # differences between conditions (the priority metric).
    # In two-tier mode: difference computed PER TIER (run gains cancel
    # out there with no re-levelling) then stitched, so no artificial
    # step at the junction.
    def diff_pair(c0, c1, pair):
        if not args.two_tier:
            g, s0 = tfs_sim[(c0, pair)]
            g1, s1 = tfs_sim[(c1, pair)]
            _, r0 = tfs_ref[(c0, pair)]
            _, r1 = tfs_ref[(c1, pair)]
            if len(g1) != len(g) or not np.allclose(g1, g):
                s1 = np.interp(g, g1, s1)
                r1 = np.interp(g, g1, r1)
            return g, s0 - s1, r0 - r1, {}
        segs_g, segs_s, segs_r, per = [], [], [], {}
        for name in ("LF", "HF"):
            t_s, t_r, flo, fhi = tiers[name]
            if (c0, pair) not in t_s or (c1, pair) not in t_s:
                continue
            g, s0 = t_s[(c0, pair)]
            g1, s1 = t_s[(c1, pair)]
            _, r0 = t_r[(c0, pair)]
            _, r1 = t_r[(c1, pair)]
            if len(g1) != len(g) or not np.allclose(g1, g):
                # mismatched grids (e.g. npz recycled from a slightly
                # different band): interpolate the 2nd condition
                s1 = np.interp(g, g1, s1)
                r1 = np.interp(g, g1, r1)
            m = (g >= flo) & (g < fhi)
            ds_, dr_ = (s0 - s1)[m], (r0 - r1)[m]
            per[name] = (pearson(ds_, dr_),
                         float(np.sqrt(np.mean((ds_ - dr_) ** 2))),
                         g[m][0], g[m][-1])
            segs_g.append(g[m])
            segs_s.append(ds_)
            segs_r.append(dr_)
        if not segs_g:
            return None, None, None, {}
        return (np.concatenate(segs_g), np.concatenate(segs_s),
                np.concatenate(segs_r), per)

    if len(args.conditions) >= 2:
        c0 = args.conditions[0]
        report = []                       # rows of the RS1 report
        for c1 in args.conditions[1:]:
            for pair in args.pairs:
                if not args.two_tier and (
                        (c0, pair) not in tfs_sim or
                        (c1, pair) not in tfs_sim):
                    continue
                g, dsim, dref, per = diff_pair(c0, c1, pair)
                if g is None:
                    continue
                for name, (rp, rm, fl, fh) in per.items():
                    print("[bench]   tier %s %s-%s %s: r=%.3f "
                          "RMS %.2f (%.0f-%.0f Hz)"
                          % (name, c0, c1, pair, rp, rm, fl, fh))
                r = pearson(dsim, dref)
                rms = float(np.sqrt(np.mean((dsim - dref) ** 2)))
                print("[bench] %s-%s %s: Pearson %.3f; RMS %.2f dB "
                      "(%.0f-%.0f Hz)" % (c0, c1, pair, r, rms,
                                          g[0], g[-1]))
                np.savez(os.path.join(args.out, "diff_%s-%s_%s.npz"
                                      % (c0, c1, pair)),
                         grid=g, dsim=dsim, dref=dref)
                report.append({"diff": "%s-%s" % (c0, c1), "pair": pair,
                               "r": r, "rms": rms, "flo": g[0],
                               "fhi": g[-1], "tiers": dict(per)})
                if not args.no_plots:
                    _plot(os.path.join(
                        args.out, "DIFF_%s-%s_%s.png"
                        % (c0, c1, pair)),
                        g, [(dref, "measured", "-"),
                            (dsim, "Auralis", "--")],
                        "TF_%s - TF_%s  %s (Pearson %.3f)"
                        % (c0, c1, pair, r),
                        ylab="Difference (dB)")
        if report:
            _write_report(args, report)
    if report_only_missing:
        print("[bench] --report-only: %d pair(s) ignored for lack of "
              "npz: %s - report built on the rest"
              % (len(report_only_missing),
                 ", ".join(sorted(report_only_missing))))


if __name__ == "__main__":
    main()
