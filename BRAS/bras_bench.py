#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRAS bench: Auralis simulation against measured impulse responses.
The JSON manifest describes geometry, materials and source/receiver pairs.
Outputs: <label>_sim.wav per pair, report.txt and report.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine.project import build_config, build_scene  # noqa: E402
from auralis.engine import rirmetrics as rm              # noqa: E402
from auralis.engine.pipeline import (ReceiverDef, SourceDef, bake_pair,  # noqa: E402
                                     source_low_bands)
from auralis.engine.wavio import read_wav, write_wav     # noqa: E402
from auralis.engine.air import config_speed_of_sound     # noqa: E402


# ---------------------------------------------------------------------------
# Measured references
# ---------------------------------------------------------------------------

def load_reference(path: str, fs: int):
    """Read a reference WAV, mix down to mono, resample (linear) to fs
    if needed. Returns (ir (n,), note:str)."""
    data, fs_in = read_wav(path)
    x = data.mean(axis=1)
    note = ""
    if fs_in != fs:
        n_out = int(round(len(x) * fs / fs_in))
        t_in = np.arange(len(x)) / fs_in
        t_out = np.arange(n_out) / fs
        x = np.interp(t_out, t_in, x)
        note = " (resampled %d -> %d Hz)" % (fs_in, fs)
    return x, note


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def _event_geometry(pair, cfg):
    """(t_direct, t_reflect, r_direct, r_reflect) for the 'events'
    mode: reflection off the plane z = mirror_z (image source)."""
    ev = pair["events"]
    c = config_speed_of_sound(cfg)
    S = np.asarray(pair["source"], dtype=float)
    R = np.asarray(pair["receiver"], dtype=float)
    mz = float(ev.get("mirror_z", 0.0))
    Sm = S.copy()
    Sm[2] = 2.0 * mz - Sm[2]
    r_d = float(np.linalg.norm(R - S))
    r_r = float(np.linalg.norm(R - Sm))
    if r_d < 0.05 or (r_r - r_d) < 0.05:
        raise ValueError(
            "degenerate reflection geometry for '%s': r_direct="
            "%.3f m, r_reflect=%.3f m (gap %.1f ms). Suspect positions "
            "or mirror_z -- check with sofa_pairs.py "
            "--inspect." % (pair.get("label", "?"), r_d, r_r,
                            (r_r - r_d) / c * 1e3))
    return r_d / c, r_r / c, r_d, r_r


def _fmt_R(v):
    return "   --  " if v is None else "%7.3f" % v


def _events_report(label, R_sim, R_ref):
    lines = ["[%s]  (|R| per octave)" % label,
             "band    : " + " ".join("%7s" % b for b in rm.LABELS),
             "R sim   : " + " ".join(_fmt_R(v) for v in R_sim)]
    if R_ref is not None:
        lines.append("R ref   : " + " ".join(_fmt_R(v) for v in R_ref))
        lines.append("dR      : " + " ".join(
            _fmt_R(None if (a is None or b is None) else a - b)
            for a, b in zip(R_sim, R_ref)))
    return "\n".join(lines)


def run_bench(manifest_path: str, out_dir: str = None,
              reuse: bool = False) -> int:
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    name = man.get("name", "bras")
    out_dir = out_dir or os.path.join(base_dir, name + "_report")
    os.makedirs(out_dir, exist_ok=True)

    scene = build_scene(man, base_dir)
    cfg = build_config(man)
    # "metrics_ir": "omni" judges the omnidirectional IR (ambisonic W), like the app and the
    # BRAS microphone. The mean of the two ears is not a pressure (up to 10 % on T30, 2 dB on
    # C80).
    omni = man.get("metrics_ir", "ears") == "omni"
    if man.get("metrics_ir", "ears") not in ("ears", "omni"):
        raise ValueError("metrics_ir must be 'ears' or 'omni'")
    if omni and int(getattr(cfg, "ambisonic_order", 0)) < 1:
        cfg.ambisonic_order = 1
    bb_lo = scene.tri_verts.reshape(-1, 3).min(axis=0)
    bb_hi = scene.tri_verts.reshape(-1, 3).max(axis=0)
    dims = bb_hi - bb_lo
    report_lines = ["BRAS bench: %s" % name,
                    "%d triangles, %d materials, engine=%s, fs=%d"
                    % (len(scene.tri_verts), len(scene.mat_names),
                       cfg.engine, cfg.fs),
                    "bounding box: %.2f x %.2f x %.2f m  "
                    "(z: %.2f to %.2f)"
                    % (dims[0], dims[1], dims[2],
                       bb_lo[2], bb_hi[2]), ""]
    for pr in man.get("pairs", []):
        for key in ("source", "receiver"):
            p = np.asarray(pr[key], dtype=float)
            if ((p < bb_lo - 1e-6).any()
                    or (p > bb_hi + 1e-6).any()):
                report_lines.append(
                    "WARNING [%s]: %s %s OUTSIDE the bounding "
                    "box -- export axes (Up=Z?) or units are "
                    "suspect."
                    % (pr.get("label", "?"), key, np.round(p, 2)))
    csv_rows = []

    # ---- FDTD low band per SOURCE (multi-mic batch) ----------------
    # One wave grid per source instead of one per pair: on the BRAS
    # campaigns (1 source -> N mics), the FDTD cost is divided by N.
    # Pairs already served by --reuse do not count.
    low_cache = {}
    sdef = man.get("source", {})
    dirv, fwd = None, None
    if sdef.get("directivity_json"):
        from auralis.engine.directivity import Directivity
        with open(os.path.join(base_dir, sdef["directivity_json"])) as fj:
            dj = json.load(fj)
        dirv = Directivity(np.asarray(dj["table"], dtype=float),
                           name=dj.get("name", "measured"))
        fwd = np.asarray(sdef.get("forward", [0.0, 0.0, 1.0]), dtype=float)
    fwd = fwd if fwd is not None else np.array([0.0, 0.0, -1.0])
    if getattr(cfg, "fdtd", False):
        groups = {}
        for pair in man["pairs"]:
            label = pair.get("label", "paire")
            if reuse and os.path.exists(
                    os.path.join(out_dir, "%s_sim.wav" % label)):
                continue
            key = tuple(np.round(np.asarray(pair["source"],
                                            dtype=float), 6))
            groups.setdefault(key, []).append(pair)
        for key, prs in groups.items():
            # the wave band carries the source's directivity too
            src_b = SourceDef(name="batch-src",
                              position=np.asarray(key, dtype=float),
                              directivity=dirv, forward=fwd)
            rcvs = [ReceiverDef(name=p.get("label", "paire") + "-rcv",
                                position=np.asarray(p["receiver"],
                                                    dtype=float))
                    for p in prs]
            # no fallback: an unstable wave band must stop a validation run
            low_cache[key] = source_low_bands(scene, src_b, rcvs, cfg)[0]

    for pair in man["pairs"]:
        label = pair.get("label", "paire")
        src = SourceDef(name=label + "-src",
                        position=np.asarray(pair["source"],
                                            dtype=float),
                        directivity=dirv, forward=fwd)
        rcv = ReceiverDef(name=label + "-rcv",
                          position=np.asarray(pair["receiver"],
                                              dtype=float))
        sim_path = os.path.join(out_dir, "%s_sim.wav" % label)
        if reuse and os.path.exists(sim_path):
            data, fs_sim = read_wav(sim_path)
            ir_sim = data.mean(axis=1)
            class _R:  # noqa: N801 - minimal carrier
                pass
            res = _R()
            res.fs = fs_sim
        else:
            skey = tuple(np.round(np.asarray(pair["source"],
                                             dtype=float), 6))
            res = bake_pair(scene, src, rcv, cfg, low_cache.get(skey))
            ir_sim = (np.asarray(res.ambi[:, 0], dtype=np.float64) if omni
                      else res.ir.mean(axis=1))
            write_wav(sim_path, ir_sim.astype(np.float32), res.fs)
        m_sim = rm.band_metrics(ir_sim, res.fs)

        ir_ref, note = None, ""
        if pair.get("reference_wav"):
            ref_path = os.path.join(base_dir, pair["reference_wav"])
            ir_ref, note = load_reference(ref_path, res.fs)

        if pair.get("events") or man.get("events_defaults"):
            ev = dict(man.get("events_defaults", {}))
            ev.update(pair.get("events") or {})
            pair = dict(pair)
            pair["events"] = ev
            td, tr, rd, rr = _event_geometry(pair, cfg)
            fmt = {"label": label}
            if "-" in label:
                fmt["emitter"], fmt["receiver"] = label.split("-", 1)
            rig_ref = ev.get("rigid_reference_wav") or \
                (ev["rigid_reference_pattern"].format(**fmt)
                 if ev.get("rigid_reference_pattern") else None)
            rig_sim = ev.get("rigid_sim_wav") or \
                (ev["rigid_sim_pattern"].format(**fmt)
                 if ev.get("rigid_sim_pattern") else None)
            R0 = float(ev.get("rigid_R", 0.99))
            if rig_sim:
                irr, _ = load_reference(
                    os.path.join(base_dir, rig_sim), res.fs)
                R_sim = rm.reflection_bands_relative(
                    ir_sim, irr, res.fs, td, tr, R0)
                mode = "Mommertz RELATIVE to rigid"
            else:
                R_sim = rm.reflection_bands(ir_sim, res.fs, td, tr,
                                            rd, rr)
                mode = "Mommertz absolute"
            R_ref = None
            if ir_ref is not None:
                if rig_ref:
                    ir0, _ = load_reference(
                        os.path.join(base_dir, rig_ref), res.fs)
                    R_ref = rm.reflection_bands_relative(
                        ir_ref, ir0, res.fs, td, tr, R0)
                else:
                    R_ref = rm.reflection_bands(ir_ref, res.fs, td,
                                                tr, rd, rr)
            if R_ref is not None and rig_ref:
                vals = [v for v in R_ref if v is not None]
                if vals and max(abs(v - R0) for v in vals) < 0.005:
                    report_lines.append(
                        "WARNING [%s]: R_ref == rigid_R everywhere -> "
                        "reference_wav and the rigid reference look "
                        "like the SAME file. Regenerate the pairs "
                        "from the SOFA of THIS scene." % label)
            report_lines.append(_events_report(
                "%s%s -- %s" % (label, note, mode), R_sim, R_ref))
            report_lines.append("")
            for b in range(rm.N_BANDS):
                row = {"paire": label, "bande": rm.LABELS[b],
                       "R_sim": R_sim[b]}
                if R_ref is not None:
                    row["R_ref"] = R_ref[b]
                    row["dR"] = (None if (R_sim[b] is None
                                          or R_ref[b] is None)
                                 else round(R_sim[b] - R_ref[b], 3))
                csv_rows.append(row)
            continue

        m_ref, deltas = None, None
        if ir_ref is not None:
            m_ref = rm.band_metrics(ir_ref, res.fs)
            deltas = rm.compare_metrics(m_sim, m_ref)

        report_lines.append(rm.format_report(label + note, m_sim,
                                             m_ref, deltas))
        report_lines.append("")
        for b in range(rm.N_BANDS):
            row = {"paire": label, "bande": rm.LABELS[b],
                   "T20_sim": m_sim["T20"][b], "T30_sim": m_sim["T30"][b],
                   "EDT_sim": m_sim["EDT"][b],
                   "C80_sim": m_sim["C80"][b],
                   "D50_sim": m_sim["D50"][b],
                   "L_sim": round(float(m_sim["level_db"][b]), 2)}
            if m_ref is not None:
                row.update({
                    "T20_ref": m_ref["T20"][b],
                    "T30_ref": m_ref["T30"][b],
                    "EDT_ref": m_ref["EDT"][b],
                    "C50_sim": m_sim["C50"][b],
                    "C50_ref": m_ref["C50"][b],
                    "C80_ref": m_ref["C80"][b],
                    "D50_ref": m_ref["D50"][b],
                    "Ts_sim": m_sim["Ts"][b],
                    "Ts_ref": m_ref["Ts"][b],
                    "L_ref": round(float(m_ref["level_db"][b]), 2),
                    "dT30": deltas["dT30"][b],
                    "dC50": deltas["dC50"][b],
                    "dTs": deltas["dTs"][b],
                    "dL": round(float(deltas["dLevel"][b]), 2)})
            csv_rows.append(row)

    text = "\n".join(report_lines)
    print(text)
    with open(os.path.join(out_dir, "report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    cols = ["paire", "bande", "T20_sim", "T30_sim", "EDT_sim",
            "C50_sim", "C80_sim", "D50_sim", "Ts_sim", "L_sim",
            "T20_ref", "T30_ref", "EDT_ref", "C50_ref", "C80_ref",
            "D50_ref", "Ts_ref",
            "L_ref", "dT30", "dC50", "dTs", "dL",
            "R_sim", "R_ref", "dR"]
    with open(os.path.join(out_dir, "report.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for row in csv_rows:
            wr.writerow(row)
    print("-> %s" % out_dir)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", help="scene JSON manifest")
    ap.add_argument("-o", "--out", default=None,
                    help="output folder (default: <name>_report/)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the <label>_sim.wav already"
                         " present in the output folder"
                         " (re-analyze without re-baking)")
    args = ap.parse_args(argv)
    return run_bench(args.manifest, args.out, reuse=args.reuse)


if __name__ == "__main__":
    sys.exit(main())
