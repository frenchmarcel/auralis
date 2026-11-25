#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS5 - single entry point: replays the published experiment (FDTD
finite screen, UTD + Genelec directivity, 700 Hz hybrid, raw and
third-octave insertion loss). Re-running resumes from the npz cache.
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAS = os.path.join(_HERE, "..")
SCENE = os.path.join(_BRAS, "..", "data_bras", "1 Scene descriptions",
                     "RS5 diffraction (infinite wedge)")
DIRV = os.path.join(_HERE, "genelec8020_directivity.json")


def run(cmd):
    print("\n$ " + " ".join(x if " " not in x else '"%s"' % x for x in cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit("failed (code %d): %s" % (rc, cmd[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/out_rs5"))
    ap.add_argument("--fc", type=float, default=1000.0)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--quick", action="store_true",
                    help="only 2 pairs (chain check)")
    a = ap.parse_args()
    if not os.path.isdir(a.scene):
        sys.exit("RS5 scene not found: %s\n(is the data_bras dump in "
                 "place?)" % a.scene)
    pairs = ["--pairs", "LS1-MP2", "LS3-MP1"] if a.quick else []
    fdtd, utd, hyb = a.out + "_fdtd", a.out + "_utd", a.out + "_hybride"
    py = sys.executable

    # 1. FDTD, real FINITE screen (4.75 m width measured on the scene)
    run([py, os.path.join(_HERE, "rs5_diffraction_edge.py"), a.scene,
         "--fc", str(a.fc), "--margin", "1.3", "--screen-width", "4.75",
         "--max-cells", "70000000", "--out", fdtd] + pairs)
    # 2. UTD + measured directivity
    run([py, os.path.join(_HERE, "rs5_diffraction_utd.py"), a.scene,
         "--directivity", DIRV, "--out", utd] + pairs)
    # 3. Hybrid: merge of both, PHYSICAL 700 Hz crossover
    run([py, os.path.join(_BRAS, "bras_hybrid.py"), a.scene,
         "--geom", "rs5", "--fdtd-dir", fdtd, "--utd-dir", utd,
         "--fcross", "700", "--out", hyb] + pairs)
    # 4. Insertion loss of the hybrid: raw AND third-octave
    for sm, lbl in (("0", "raw"), ("3", "third_octave")):
        run([py, os.path.join(_BRAS, "bras_insertion_loss.py"), a.scene,
             "--geom", "rs5", "--sim-dir", hyb, "--npz-prefix", "rs5hyb",
             "--directivity", DIRV, "--smooth", sm,
             "--out", "%s_il_%s" % (a.out, lbl), "--no-plots"] + pairs)
    print("\n=== RS5 done ===")
    print("reports: %s, %s, %s" % (fdtd, utd, hyb))
    print("insertion loss: %s_il_*/report.txt" % a.out)


if __name__ == "__main__":
    main()
