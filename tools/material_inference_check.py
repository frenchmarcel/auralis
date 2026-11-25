"""Blind test of matinfer: from each measured complex impedance (impedance tube, Gaborit et
al., Zenodo 3967083, CC BY 4.0, in data_materials/), keep only its octave alphas, infer a
physical model, and compare its normal-incidence R(f) at 125-1000 Hz with the measurement
and with the engine's alpha-only admittance fit: python tools/material_inference_check.py"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auralis.engine import matinfer, poromat  # noqa: E402
from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.reflection import _gamma_of, alpha_random, fit_admittance_random  # noqa: E402

GABORIT = os.path.join(ROOT, "data_materials", "gaborit_zenodo3967083", "extracted", "dataset",
                       "measurements")
FE = np.geomspace(125.0, 1000.0, 40)


def gaborit():
    """[(label, f, normalised complex Zs)] of every Gaborit measurement, 50 Hz-4.5 kHz."""
    out = []
    for p in sorted(glob.glob(GABORIT + "/*/*.rok") + glob.glob(GABORIT + "/*/*/*.rok")):
        a = np.loadtxt(p, comments="#", usecols=(0, 4, 5))
        k = (a[:, 0] >= 50.0) & (a[:, 0] <= 4500.0)
        out.append((os.path.relpath(p, GABORIT)[:-4], a[k, 0], a[k, 1] + 1j * a[k, 2]))
    return out


def octave_alpha(f, zn):
    """What a material table would hold: random-incidence alpha per octave (local reaction)."""
    g, lg, out = 1.0 / zn, np.log(f), []
    for fc in CENTERS:
        fb = np.clip(np.geomspace(fc / 2 ** 0.5, fc * 2 ** 0.5, 9), f[0], f[-1])
        gb = np.interp(np.log(fb), lg, g.real) + 1j * np.interp(np.log(fb), lg, g.imag)
        out.append(float(np.mean(alpha_random(gb))))
    return np.clip(out, 0.0, 0.99)


def r_of_gamma(g):
    return (1.0 - g) / (1.0 + g)


def main():
    if not os.path.isdir(GABORIT):
        raise SystemExit("unzip data_materials/gaborit_zenodo3967083/dataset.zip into extracted/ first")
    res = defaultdict(list)
    for lab, f, zn in gaborit():
        alpha = octave_alpha(f, zn)
        zt = np.interp(np.log(FE), np.log(f), zn.real) + 1j * np.interp(np.log(FE), np.log(f), zn.imag)
        rt = (zt - 1.0) / (zt + 1.0)
        rg = r_of_gamma(_gamma_of(FE, *fit_admittance_random(alpha, CENTERS, f_hi=500.0)))
        inf = matinfer.infer(alpha)
        ri = rg if inf["family"] is None else r_of_gamma(poromat.gamma_normal(inf["spec"], FE))
        res[lab.split("/")[0]].append((np.mean(np.abs(rg - rt)), np.mean(np.abs(ri - rt)),
                                       inf["family"]))
    print("mean |R - R_measured| at 125-1000 Hz, normal incidence")
    print("%-24s %4s %9s %9s  %s" % ("category", "n", "alpha fit", "inferred", "families (None = fit kept)"))
    tot = []
    for cat, rows in res.items():
        g = np.array([r[0] for r in rows])
        i = np.array([r[1] for r in rows])
        fams = {}
        for r in rows:
            fams[r[2]] = fams.get(r[2], 0) + 1
        tot += rows
        print("%-24s %4d %9.3f %9.3f  %s" % (cat, len(rows), g.mean(), i.mean(), fams))
    g = np.array([r[0] for r in tot])
    i = np.array([r[1] for r in tot])
    acc = np.array([r[2] is not None for r in tot])
    print("all %d: alpha fit %.3f, inferred %.3f; accepted %d: better %d, worse by > 0.01 %d"
          % (len(tot), g.mean(), i.mean(), acc.sum(), int(((i < g + 0.01) & acc).sum()),
             int(((i > g + 0.01) & acc).sum())))


if __name__ == "__main__":
    main()
