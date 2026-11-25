# -*- coding: utf-8 -*-
"""Grid reuse across runs (voxelization, conformal corrections, free-field calibration):
bit-identical."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.geometry import add_box, make_shoebox  # noqa: E402


def _scene():
    mat = {"absorption": np.full(7, 0.12), "scattering": np.full(7, 0.1),
           "transmission": np.zeros(7)}
    sc = make_shoebox((3.2, 2.6, 2.2), wall_material=mat, floor_material=mat,
                      ceiling_material=mat)
    return add_box(sc, (0.6, 0.5, 0.9), (1.1, 0.9, 0.0), mat)


def _two_sources(cache):
    os.environ["AURALIS_FDTD_GRID_CACHE"] = "1" if cache else "0"
    F._VOX_CACHE.clear()
    F._CONF_CACHE.clear()
    F._CAL_CACHE.clear()
    sc = _scene()
    out = []
    # points >= 6 dx from the walls: the grid (geometry bbox) is the same for both sources
    for src in ([1.0, 0.95, 1.1], [2.2, 1.65, 1.3]):
        ir, _info = F.run_low_band(sc, np.array(src), [np.array([2.0, 1.0, 1.2]),
                                                       np.array([1.2, 1.6, 1.0])],
                                   fs=8000, duration=0.06, f_cross=180.0,
                                   speed_of_sound=343.0, engine="python")
        out.append(np.asarray(ir))
    return out


def test_reused_grid_is_bit_identical():
    try:
        a = _two_sources(cache=False)
        b = _two_sources(cache=True)
        assert len(F._VOX_CACHE) == 1 and len(F._CONF_CACHE) == 1 and len(F._CAL_CACHE) == 1
    finally:
        os.environ.pop("AURALIS_FDTD_GRID_CACHE", None)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_a_source_in_another_air_region_is_flooded_again():
    sc = _scene()
    F._VOX_CACHE.clear()
    g1 = F.voxelize(sc, 0.1, np.array([0.6, 0.7, 1.1]))
    g2 = F.voxelize(sc, 0.1, np.array([2.5, 1.9, 1.3]))
    assert np.array_equal(g1.air, g2.air) and g1.air is not g2.air
    g2.solid[0, 0, 0] = not g2.solid[0, 0, 0]          # copies: the cache is untouched
    g3 = F.voxelize(sc, 0.1, np.array([0.6, 0.7, 1.1]))
    assert np.array_equal(g3.solid, g1.solid)
