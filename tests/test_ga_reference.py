# -*- coding: utf-8 -*-
"""The geometrical solver against two exact references of a box with an absorbing floor:
image sources of all orders (specular walls) and radiosity (Lambertian walls)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ga_reference_box as g  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, speed_of_sound  # noqa: E402

ALPHA = (0.80, 0.05, 0.05)          # floor, ceiling, walls
T_MAX = 2.0


def _small_box():
    g.SIZE = (10.0, 8.0, 4.0)
    g.SRC = np.array([2.3, 3.1, 1.5])
    g.RCV = np.array([7.4, 5.2, 1.2])


def _ctx():
    c = speed_of_sound(g.T_C)
    air = float(air_attenuation_db_per_m(temperature_c=g.T_C, humidity_percent=g.HR)[g.BAND])
    return c, air


def test_diffuse_walls_match_radiosity(monkeypatch):
    """Scattering 1: the default estimator (image sources at their specular share) gives
    the radiosity's T30, EDT and C50. With rbs_scattering off the early reflections count twice."""
    from auralis.engine import geometry
    monkeypatch.setattr(geometry, "surface_dims", g.infinite_dims)
    _small_box()
    c, air = _ctx()
    ref = g.metrics(g.radiosity_hist(ALPHA, T_MAX, air, c, h=1.0))
    en = g.metrics(g.engine_hist(ALPHA, 1.0, T_MAX))
    assert abs(en["T30"] / ref["T30"] - 1.0) < 0.05, (en, ref)
    assert abs(en["EDT"] / ref["EDT"] - 1.0) < 0.05, (en, ref)
    assert abs(en["C50"] - ref["C50"]) < 0.4, (en, ref)
    old = g.metrics(g.engine_hist(ALPHA, 1.0, T_MAX, rbs=False))
    assert old["C50"] - ref["C50"] > 1.0, (old, ref)     # C50 about +2.5 dB


def test_specular_walls_match_image_sources(monkeypatch):
    from auralis.engine import geometry
    monkeypatch.setattr(geometry, "surface_dims", g.infinite_dims)
    _small_box()
    c, air = _ctx()
    ref = g.metrics(g.exact_hist(ALPHA, T_MAX, air, c))
    en = g.metrics(g.engine_hist(ALPHA, 0.0, T_MAX))
    assert abs(en["EDT"] / ref["EDT"] - 1.0) < 0.05, (en, ref)
    assert abs(en["C50"] - ref["C50"]) < 0.4, (en, ref)
    assert abs(10 * np.log10(en["E"] / ref["E"])) < 0.3, (en, ref)
