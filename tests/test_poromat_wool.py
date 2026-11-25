# -*- coding: utf-8 -*-
"""Wool flow resistivity from the bulk density: the laws reproduce the material data they
were checked on, and say when they extrapolate."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine.poromat import wool_flow_resistivity  # noqa: E402

# Kovacic et al., Forum Acusticum 2026 (CC BY 3.0), Tables 4-5: density kg/m3, sigma Pa.s/m2
KNAUF_GLASS = [(13.02, 5800), (14.50, 5400), (14.76, 6200), (16.87, 9100), (17.18, 7500),
               (18.95, 9400), (23.15, 14200), (29.83, 18600), (31.61, 26700), (32.26, 29400)]


def _rms_decade(data, kind):
    e = [np.log10(wool_flow_resistivity(r, kind)["sigma"] / s) for r, s in data]
    return float(np.sqrt(np.mean(np.square(e)))), float(np.mean(e))


def test_glass_wool_law_reproduces_the_measured_knauf_wools():
    rms, bias = _rms_decade(KNAUF_GLASS, "glass")
    assert rms < 0.065 and abs(bias) < 0.01
    w = wool_flow_resistivity(20.0, "glass")
    assert w["sigma_low"] < w["sigma"] < w["sigma_high"] and not w["extrapolated"]
    assert 14.0 < w["uncertainty_pct"] < 16.0


def test_rock_wool_law_and_extrapolation_flag():
    w = wool_flow_resistivity(40.0, "rock")
    assert 9000 < w["sigma"] < 13000 and not w["extrapolated"]
    assert wool_flow_resistivity(150.0, "rock")["extrapolated"]
    assert wool_flow_resistivity(80.0, "glass")["extrapolated"]
    # at equal density glass wool (finer fibres) is the more resistive
    assert wool_flow_resistivity(30.0, "glass")["sigma"] > wool_flow_resistivity(30.0, "rock")["sigma"]


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        wool_flow_resistivity(40.0, "hemp")
    with pytest.raises(ValueError):
        wool_flow_resistivity(1.0, "glass")
