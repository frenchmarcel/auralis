# -*- coding: utf-8 -*-
"""Subsonic high-pass of the closed-domain wave band: a loudspeaker-like roll-off."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.fdtd import subsonic_highpass  # noqa: E402


def test_subsonic_highpass_is_a_causal_fourth_order_butterworth():
    f = np.array([0.0, 5.0, 16.0, 44.0, 63.0, 1000.0])
    g = 20.0 * np.log10(np.maximum(np.abs(subsonic_highpass(f, 16.0)), 1e-30))
    assert g[0] < -300.0 and abs(g[1] + 40.4) < 0.2 and abs(g[2] + 3.01) < 0.01
    assert g[3] > -0.03 and g[4] > -0.01 and abs(g[5]) < 1e-6
    fs, n = 2000, 1 << 16
    h = np.fft.irfft(subsonic_highpass(np.fft.rfftfreq(n, 1.0 / fs), 16.0), n)
    assert np.sum(h[n // 2:] ** 2) / np.sum(h ** 2) < 1e-3        # causal (sampled analog)
    assert abs(h.sum()) < 1e-12                                   # no DC
