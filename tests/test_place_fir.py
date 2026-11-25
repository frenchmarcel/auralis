# -*- coding: utf-8 -*-
"""Event placement at a fractional sample: exact delay, flat to Nyquist (two-tap linear
interpolation loses 7.7 dB at 16 kHz for half a sample), integers unchanged."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine.hrtf import place_fir  # noqa: E402

FS = 44100


def _impulse():
    h = np.zeros(129)
    h[64] = 1.0
    return h


def test_fractional_placement_is_flat_and_exactly_delayed():
    f = np.fft.rfftfreq(8192, 1 / FS)
    for frac in (0.25, 0.5, 0.9):
        buf = np.zeros(400)
        place_fir(buf, _impulse(), 100 + frac)
        H = np.fft.rfft(buf, 8192)
        db = np.abs(20 * np.log10(np.abs(H)))
        # a bare impulse is the worst case: its truncated sinc ripples near Nyquist
        assert db[(f > 50) & (f < 16000)].max() < 0.1, frac
        assert db[(f > 50) & (f < 20000)].max() < 0.25, frac
        k = int(0.2 * 8192)
        delay = -np.unwrap(np.angle(H[:k + 1]))[-1] / (2 * np.pi * k / 8192)
        assert abs(delay - (164 + frac)) < 0.01, (frac, delay)


def test_integer_placement_is_a_plain_add():
    rng = np.random.default_rng(0)
    fir = rng.standard_normal(50)
    buf = rng.standard_normal(300)
    want = buf.copy()
    want[120:170] += fir
    place_fir(buf, fir, 120.0)
    assert np.array_equal(buf, want)


def test_placement_is_clipped_at_both_ends():
    for pos in (-10.3, 40.6, -80.0, 400.5):
        buf = np.zeros(50)
        place_fir(buf, np.ones(20), pos)
        big = np.zeros(200)
        place_fir(big, np.ones(20), pos + 75)
        assert np.allclose(buf, big[75:125])
