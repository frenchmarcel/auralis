"""Perforated-sheet model (poromat 'perforated', two sheets in series for the stepped holes)
against the impedance-tube data of Chen & Ren (Zenodo 23494742, CC BY 4.0, in data_materials/):
eight face-sheets with stepped apertures over a 20 mm empty cavity, 340-1600 Hz (their range):
python tools/perforated_check.py"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auralis.engine import poromat  # noqa: E402

DATA = os.path.join(ROOT, "data_materials", "chen_ren_zenodo23494742", "extracted")
FLAGGED = {"A1", "A3", "A5", "A7"}     # resonance inconsistent with their acoustic mass (README)


def sheets():
    return list(csv.DictReader(open(os.path.join(DATA, "metadata_facesheets.csv"))))


def measured(sample):
    rows = list(csv.DictReader(open(os.path.join(DATA, "data", "mean", sample + ".csv"))))
    f = np.array([float(r["frequency_Hz"]) for r in rows])
    R = np.array([float(r["R_real"]) + 1j * float(r["R_imag"]) for r in rows])
    return f, R


def spec_of(s):
    D = float(s["specimen_diameter_mm"])
    n = float(s["n_apertures"])
    layers = []
    for part in ("small", "large"):
        d = float(s[part + "_aperture_measured_mm"])
        layers.append({"model": "perforated", "d": d * 1e-3,
                       "t": float(s[part + "_section_thickness_mm"]) * 1e-3,
                       "phi": n * (d / D) ** 2})
    layers.append({"model": "air", "d": float(s["cavity_depth_mm"]) * 1e-3})
    return {"layers": layers}


def main():
    print("face-sheet  peak alpha_n measured / model (Hz, value)     |R - R_meas| 340-1600 Hz")
    errs = []
    for s in sheets():
        f, R = measured(s["facesheet_id"] + "-0")
        k = (f >= 340.0) & (f <= 1600.0)
        Zs = poromat.surface_impedance(spec_of(s), f[k], 1.0) / poromat.Z0
        Rm = (Zs - 1.0) / (Zs + 1.0)
        am, ar = 1.0 - np.abs(Rm) ** 2, 1.0 - np.abs(R[k]) ** 2
        e = float(np.mean(np.abs(Rm - R[k])))
        flag = " (flagged by the authors)" if s["facesheet_id"] in FLAGGED else ""
        if not flag:
            errs.append(e)
        print("  %-4s       %6.0f Hz %.2f / %6.0f Hz %.2f                   %.3f%s"
              % (s["facesheet_id"], f[k][np.argmax(ar)], ar.max(), f[k][np.argmax(am)], am.max(), e, flag))
    print("mean over the unflagged sheets: %.3f" % np.mean(errs))


if __name__ == "__main__":
    main()
