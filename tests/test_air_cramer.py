# -*- coding: utf-8 -*-
"""Speed of sound in humid air (Cramer 1993): the BRAS harness constants, and the dry
formula kept as an option."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine.air import (config_speed_of_sound, speed_of_sound,  # noqa: E402
                                speed_of_sound_cramer)
from auralis.engine.pipeline import BakeConfig  # noqa: E402


def test_cramer_matches_the_bras_harness_value_and_literature():
    # RS6 harness: Cramer's c at 19.9 C / 40.1 % (BRAS Documentation.pdf) = 343.79
    assert speed_of_sound_cramer(19.9, 40.1) == pytest.approx(343.79, abs=0.03)
    # dry air at 0 C: 331.45 m/s (Cramer's own reference point, with CO2)
    assert speed_of_sound_cramer(0.0, 0.0) == pytest.approx(331.45, abs=0.02)
    # humidity raises c: ~+0.6 m/s from dry to 50 % at 20 C
    assert 0.4 < speed_of_sound_cramer(20.0, 50.0) - speed_of_sound_cramer(20.0, 0.0) < 0.9


def test_config_default_is_cramer_and_the_dry_formula_stays_available():
    """Cramer's formula is the default. air_cramer=False gives the dry formula."""
    cfg = BakeConfig(temperature_c=19.9, humidity_percent=40.1)
    assert config_speed_of_sound(cfg) == speed_of_sound_cramer(19.9, 40.1)
    cfg.air_cramer = False
    assert config_speed_of_sound(cfg) == speed_of_sound(19.9)
