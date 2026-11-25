"""63 Hz alpha of a material in the wave band: box 10 x 8 x 4 m, one light panel everywhere.
python tools/band63_check.py [a63 ...]   (default: none, 0.10, 0.30)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import rirmetrics as rm  # noqa: E402
from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402
from auralis.engine.reflection import _gamma_of, alpha_random, fit_admittance_random  # noqa: E402

SIZE = (10.0, 8.0, 4.0)
PANEL = np.array([0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05])
FC = 250.0
F63 = np.r_[63.0, CENTERS]


def eyring(alpha):
    v = SIZE[0] * SIZE[1] * SIZE[2]
    s = 2.0 * (SIZE[0] * SIZE[1] + SIZE[0] * SIZE[2] + SIZE[1] * SIZE[2])
    return 0.161 * v / (-s * np.log(1.0 - alpha))


def run(a63):
    mat = {"absorption": PANEL.copy(), "scattering": np.full(7, 0.1), "transmission": np.zeros(7)}
    sc = make_shoebox(SIZE, wall_material=mat, floor_material=mat, ceiling_material=mat)
    full = None if a63 is None else (F63, np.r_[a63, PANEL])
    sc.mat_alpha_full = [full] * sc.n_mats
    cfg = BakeConfig(fdtd=True, fdtd_fcross=FC, ambisonic_order=1, n_rays=50_000,
                     max_time=3.0, normalize=False, seed=1234)
    res = bake(sc, SourceDef(name="s", position=np.array([2.3, 1.7, 1.4])),
               ReceiverDef(name="r", position=np.array([7.1, 5.3, 1.6])), cfg)
    omni = np.asarray(res.ambi[:, 0], np.float64)
    m = rm.band_metrics_at(omni, res.fs, [63.0, 125.0])
    fit = (fit_admittance_random(PANEL, CENTERS, f_hi=FC) if a63 is None
           else fit_admittance_random(np.r_[a63, PANEL], F63, f_hi=FC))
    a_fit = alpha_random(_gamma_of(np.array([63.0, 125.0]), *fit))
    return omni, m, a_fit, res.stats.get("fdtd_fc")


def main():
    cases = [None if a in ("none", "-") else float(a) for a in sys.argv[1:]] or [None, 0.10, 0.30]
    irs = {}
    print("a63   fc    alpha seen 63/125   T30 63 Hz (Eyring)   T30 125 Hz (Eyring)   EDT 63")
    for a63 in cases:
        omni, m, a_fit, fc = run(a63)
        irs[a63] = omni
        print("%-5s %4.0f  %.3f / %.3f       %.2f s (%.2f)          %.2f s (%.2f)          %.2f s"
              % ("none" if a63 is None else "%.2f" % a63, fc or 0, a_fit[0], a_fit[1],
                 m["T30"][0] or np.nan, eyring(a_fit[0]), m["T30"][1] or np.nan,
                 eyring(a_fit[1]), m["EDT"][0] or np.nan))
    if None in irs and PANEL[0] in irs:
        same = np.array_equal(irs[None], irs[PANEL[0]])
        print("a63 = a125 gives the same IR as no a63: %s" % ("bit for bit" if same else
              "NO, max diff %.3e" % np.max(np.abs(irs[None] - irs[PANEL[0]]))))


if __name__ == "__main__":
    main()
