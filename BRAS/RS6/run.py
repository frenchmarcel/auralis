#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS6 - single entry point: replays the experiment and publishes the
README figures (FDTD fc=2000 with the anisotropic conformal correction,
UTD control, raw and third-octave insertion loss). --quick: 2 pairs."""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAS = os.path.join(_HERE, "..")
SCENE = os.path.join(_BRAS, "..", "data_bras", "1 Scene descriptions",
                     "RS6 diffraction (finite body)")
DIRV = os.path.join(_BRAS, "RS5", "genelec8020_directivity.json")


def run(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    print("\n$ " + " ".join(x if " " not in x else '"%s"' % x for x in cmd))
    rc = subprocess.call(cmd, env=e)
    if rc != 0:
        sys.exit("failed (code %d): %s" % (rc, cmd[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/out_rs6"))
    ap.add_argument("--fc", type=float, default=2000.0)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--quick", action="store_true",
                    help="only 2 pairs (chain check)")
    a = ap.parse_args()
    if not os.path.isdir(a.scene):
        sys.exit("RS6 scene not found: %s\n(is the data_bras dump in "
                 "place?)" % a.scene)
    pairs = ["--pairs", "LS1-MP1", "LS3-MP3"] if a.quick else []
    fdtd, utd = a.out, a.out + "_utd"
    py = sys.executable

    # 1. FDTD - ANISOTROPIC conformal correction: neutral against conformal OFF over the
    #    9 pairs and better on RS7 (README), so all open scenes use it.
    run([py, os.path.join(_HERE, "rs6_diffraction_object.py"), a.scene,
         "--fc", str(a.fc), "--win", "0.03", "--out", fdtd] + pairs,
        env={"AURALIS_FDTD_CONFORMAL": "1", "AURALIS_FDTD_CONFORMAL_ANISO": "1"})
    # 2. UTD - geometric control (fast, no GPU)
    run([py, os.path.join(_HERE, "rs6_diffraction_utd.py"), a.scene,
         "--directivity", DIRV, "--out", utd] + pairs)
    # 3. Insertion loss: reference metric (differential), published
    #    RAW and in third-octave - never one alone.
    for tag, sim, pref in (("fdtd", fdtd, "rs6"), ("utd", utd, "rs6utd")):
        for sm, lbl in (("0", "raw"), ("3", "third_octave")):
            run([py, os.path.join(_BRAS, "bras_insertion_loss.py"), a.scene,
                 "--geom", "rs6", "--sim-dir", sim, "--npz-prefix", pref,
                 "--directivity", DIRV, "--smooth", sm,
                 "--out", "%s_il_%s_%s" % (a.out, tag, lbl),
                 "--no-plots"] + pairs)
    print("\n=== RS6 done ===")
    print("reports: %s/report.txt (FDTD), %s/report.txt (UTD)" % (fdtd, utd))
    print("insertion loss: %s_il_*/report.txt" % a.out)


if __name__ == "__main__":
    main()
