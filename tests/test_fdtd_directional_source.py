# -*- coding: utf-8 -*-
"""Directional source in the wave band: first-order pattern of the measured power,
from a monopole and a grid dipole run on the same grid."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.directivity import get_directivity  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402

FS = 16000


def test_first_order_weights_recover_the_analytic_patterns():
    for name, a in (("omni", 1.0), ("subcardioid", 0.75), ("cardioid", 0.5),
                    ("supercardioid", 0.366), ("hypercardioid", 0.25)):
        assert np.allclose(get_directivity(name).first_order_weights(), a, atol=0.01), name


def test_dipole_taps_are_balanced_and_oriented():
    sc = make_shoebox((4.0, 3.0, 2.5))
    vox = F.voxelize(sc, 0.1, np.array([2.0, 1.5, 1.2]))
    g = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4)
    idx, w = F._dipole_taps(g, (2.03, 1.47, 1.21), (1.0, 1.0, 0.0), 0.2)
    assert abs(w.sum()) < 1e-12 and len(np.unique(idx)) == len(idx)
    xyz = g.cells[idx] @ np.array([1.0, 1.0, 0.0])
    assert xyz[w > 0].mean() > xyz[w < 0].mean()


SRC = np.array([3.7, 2.8, 1.9])       # off the centre: no parity selection of the modes


def _run(a_bands, rcvs):
    mat = {"absorption": np.full(7, 0.5), "scattering": np.full(7, 0.1),
           "transmission": np.zeros(7)}
    sc = make_shoebox((8.0, 6.0, 4.0), wall_material=mat, floor_material=mat,
                      ceiling_material=mat)
    ir, _ = F.run_low_band(sc, SRC, rcvs, fs=FS, duration=0.5, f_cross=250.0,
                           speed_of_sound=343.0, conformal=False,
                           src_axis=None if a_bands is None else (1.0, 0.0, 0.0),
                           src_first_order=a_bands)
    return np.asarray(ir)


def _band_energy(x, lo=180.0, hi=375.0):
    """Energy of a segment between lo and hi (Parseval), where the wave band is used:
    above the room's modal region, below the grid's limit."""
    X = np.abs(np.fft.rfft(x, 4 * len(x))) ** 2
    f = np.fft.rfftfreq(4 * len(x), 1.0 / FS)
    return float(X[(f >= lo) & (f < hi)].sum())


def test_cardioid_wave_band_keeps_the_axis_and_cuts_the_back_and_the_power():
    rcvs = [SRC + (1.2, 0.0, 0.0), SRC - (1.2, 0.0, 0.0),        # front, back
            np.array([6.1, 4.6, 2.9]), np.array([1.7, 1.2, 1.1]), np.array([5.9, 0.9, 0.7]),
            np.array([1.4, 4.8, 3.0])]
    omni, card = _run(None, rcvs), _run(np.full(7, 0.5), rcvs)
    t = int(round(1.2 / 343.0 * FS))
    early = slice(max(0, t - int(0.003 * FS)), t + int(0.006 * FS))   # before any reflection
    db = lambda a, b: 10 * np.log10(a / b)                      # noqa: E731
    front = db(_band_energy(card[early, 0]), _band_energy(omni[early, 0]))
    back = db(_band_energy(card[early, 1]), _band_energy(omni[early, 1]))
    late = slice(int(0.04 * FS), None)
    power = db(sum(_band_energy(card[late, r]) for r in range(len(rcvs))),
               sum(_band_energy(omni[late, r]) for r in range(len(rcvs))))
    assert abs(front) < 1.0, front
    assert back < -12.0, back
    assert abs(power + 4.77) < 1.5, power                     # cardioid: 1/3 of the power
