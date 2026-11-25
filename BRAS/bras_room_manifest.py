#!/usr/bin/env python3
"""Bench manifest of a BRAS complex room (CR1-CR4), from the official data only.
Usage: bras_room_manifest.py CR3 --obj CR3_RIR.obj -o BRAS/CR3/cr3_dodecahedron_fitted31_dir.json
CR1 (coupled rooms) has one set per door opening: --variant DoorAngle1 or DoorAngle3."""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sofa_pairs import read_pairs  # noqa: E402

BRAS = os.path.join(REPO, "data_bras")
CSV = os.path.join(BRAS, "3 Surface descriptions", "_csv")
DODECA = os.path.join(HERE, "CR2", "ita_dodeca_directivity.json")
OCTAVES = [125, 250, 500, 1000, 2000, 4000, 8000]
# air during the measurements, Documentation.pdf (scene pages)
AIR = {"CR1": (18.2, 47.6), "CR2": (19.5, 41.7), "CR3": (22.4, 40.9), "CR4": (20.9, 37.5)}
# length of the measured RIRs (s), Documentation.pdf; 3.5 s otherwise
DURATION = {"CR1": 4.5}


def read_csv(path):
    """(third-octave freqs, absorption, scattering) of a BRAS material CSV."""
    rows = [[float(v) for v in ln.split(",") if v.strip()]
            for ln in open(path, encoding="utf-8-sig").read().splitlines() if ln.strip()]
    return np.array(rows[0]), np.array(rows[1]), np.array(rows[2])


def octave_means(f, v):
    """Octave value = mean of its three third octaves (as the CR2 manifest)."""
    return [round(float(np.mean([v[np.argmin(np.abs(f - c * 2 ** (k / 3)))] for k in (-1, 0, 1)])), 3)
            for c in OCTAVES]


def materials(scene, estimates):
    out = {}
    for path in sorted(glob.glob(os.path.join(CSV, "%s_estimates" % estimates, "mat_%s_*.csv" % scene))):
        name = os.path.basename(path)[:-4]
        f, a, s = read_csv(path)
        out[name] = {"absorption": octave_means(f, a), "scattering": octave_means(f, s),
                     "transmission": 0.0,
                     "absorption_full": {"f": f.tolist(), "alpha": a.tolist()}}
    if not out:
        raise SystemExit("no %s_estimates CSV for %s in %s" % (estimates, scene, CSV))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", choices=sorted(AIR))
    ap.add_argument("--obj", required=True, help="room OBJ, relative to the manifest folder")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--estimates", choices=("fitted", "initial"), default="fitted")
    ap.add_argument("--variant", default="",
                    help="measurement set inside the scene's file names (CR1: DoorAngle1, DoorAngle3)")
    a = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    scene_dir = glob.glob(os.path.join(BRAS, "1 Scene descriptions", a.scene + " *"))[0]
    var = (a.variant + "_") if a.variant else ""
    sofa = os.path.join(scene_dir, "RIRs", "%s_RIRs_%sDodecahedron.sofa" % (a.scene, var))
    wav = os.path.relpath(os.path.join(scene_dir, "RIRs", "wav"), out_dir)
    pairs, fs = read_pairs(sofa, os.path.join(
        wav, "%s_RIR_%s{emitter}_{receiver}_Dodecahedron.wav" % (a.scene, var)))
    t, h = AIR[a.scene]
    man = {
        "name": os.path.basename(a.out)[:-5],
        "_note_scene": "BRAS %s%s, dodecahedron source, B&K 4134 microphone (Documentation.pdf)."
                       % (a.scene, (" " + a.variant) if a.variant else ""),
        "_note_materiaux": "%s_estimates of BRAS (%s): octave = mean of its three third "
                           "octaves, wave band on the 31 third octaves. Never re-fitted here."
                           % (estimates_label(a.estimates), a.scene),
        "fs": int(fs), "duration": DURATION.get(a.scene, 3.5),
        "engine": {"n_rays": 300000, "ism_order": 2, "fdtd": True, "fdtd_fcross": 0,
                   "seed": 1234, "temperature_c": t, "humidity_percent": h},
        "geometry": {"obj": a.obj},
        "materials": materials(a.scene, a.estimates),
        "source": {"directivity_json": os.path.relpath(DODECA, out_dir), "forward": [0.0, 0.0, 1.0]},
        "pairs": pairs,
        # judged on the omni response, like the BRAS microphone
        "metrics_ir": "omni",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(man, fh, indent=1)
    print("%s: %d materials, %d pairs, %.1f C / %.1f %% -> %s"
          % (a.scene, len(man["materials"]), len(pairs), t, h, a.out))


def estimates_label(e):
    return "fitted (absorption matched by BRAS to the measured RT, Eyring)" if e == "fitted" \
        else "initial (database and in-situ estimates)"


if __name__ == "__main__":
    main()
