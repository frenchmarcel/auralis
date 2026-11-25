# -*- coding: utf-8 -*-
"""The 63 Hz alpha of a material reaches the wave band's wall fit."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.project import build_scene  # noqa: E402
from auralis.engine.reflection import _gamma_of, alpha_random, fit_admittance_random  # noqa: E402

F63 = np.r_[63.0, CENTERS]
CASES = [np.array([0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05]),     # light panel
         np.array([0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05]),     # hard wall
         np.array([0.20, 0.45, 0.80, 0.90, 0.90, 0.85, 0.80])]     # porous absorber


def _alpha63(fit):
    gi, A, l = fit
    return float(alpha_random(_gamma_of(np.array([63.0]), gi, A, l))[0])


def test_a63_equal_to_a125_is_the_seven_band_fit_exactly():
    """Anchor: with a 63 Hz value equal to the 125 Hz one, the full-curve path is the
    seven-band fit bit for bit (that fit holds alpha_125 below 125 Hz)."""
    for ab in CASES:
        for f_hi in (141.0, 446.0, 8000.0):
            ref = fit_admittance_random(ab, CENTERS, f_hi=f_hi)
            new = fit_admittance_random(np.r_[ab[0], ab], F63, f_hi=f_hi)
            assert ref[0] == new[0]
            assert np.array_equal(ref[1], new[1]) and np.array_equal(ref[2], new[2])


def test_wall_fit_follows_the_63_hz_value():
    """A panel absorbing 0.30 at 63 Hz and 0.10 at 125 Hz: the wave band sees about 0.30
    at 63 Hz, and about 0.10 without the 63 Hz value."""
    ab = CASES[0]
    for f_hi in (141.0, 446.0):
        held = _alpha63(fit_admittance_random(ab, CENTERS, f_hi=f_hi))
        follows = _alpha63(fit_admittance_random(np.r_[0.30, ab], F63, f_hi=f_hi))
        assert abs(held - 0.10) < 0.03, held
        assert abs(follows - 0.30) < 0.04, follows


def test_manifest_full_curve_reaches_the_scene():
    man = {"materials": {"panel": {"absorption": CASES[0].tolist(),
                                   "absorption_full": {"f": F63.tolist(),
                                                       "alpha": [0.30] + CASES[0].tolist()}},
                         "wall": {"absorption": CASES[1].tolist()}},
           "geometry": {"triangles": np.zeros((2, 3, 3)), "tri_mat": [0, 1]}}
    sc = build_scene(man, "")
    fb, ab = sc.mat_alpha_full[0]
    assert np.allclose(fb, F63) and ab[0] == 0.30
    assert sc.mat_alpha_full[1] is None
    assert np.allclose(sc.absorption[0], CASES[0])      # the rays keep their seven octaves
