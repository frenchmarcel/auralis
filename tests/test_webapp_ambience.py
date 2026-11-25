# -*- coding: utf-8 -*-
"""Recorded ambiences in animated renders: loops without clicks, and a diffuse bed at its level
with the diffuse-field interaural coherence of each octave band."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine.bands import band_masks  # noqa: E402
from auralis.engine.hrtf import interaural_coherence  # noqa: E402
from auralis.engine.level import dba_total  # noqa: E402
from webapp.server import sounds  # noqa: E402

FS = 16000


def test_loop_to_repeats_without_a_jump():
    t = np.arange(int(1.3 * FS)) / FS
    x = np.sin(2 * np.pi * 220.0 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 0.7 * t))
    y = sounds.loop_to(x, 4 * FS, FS)
    assert len(y) == 4 * FS
    assert np.allclose(y[:FS // 2], x[:FS // 2])                 # starts as the recording
    assert np.abs(np.diff(y)).max() < 1.5 * np.abs(np.diff(x)).max()   # no click at the joins
    assert np.array_equal(sounds.loop_to(x, FS, FS), x[:FS])     # long enough: cut only


def test_diffuse_recording_level_and_coherence():
    rng = np.random.default_rng(3)
    rec = np.convolve(rng.standard_normal(20 * FS), np.ones(4) / 4, "same")   # a "recording"
    n = 15 * FS
    bed = sounds.diffuse_recording(rec, n, FS, 60.0)
    assert bed.shape == (n, 2)
    for ch in (0, 1):
        assert dba_total(sounds._octave_levels(bed[:, ch], FS) + 20 * np.log10(1 / sounds.P_REF)) \
            == pytest.approx(60.0, abs=0.6)
    masks = band_masks(np.fft.rfftfreq(n, 1.0 / FS), FS)
    L, R = np.fft.rfft(bed[:, 0]), np.fft.rfft(bed[:, 1])
    rho = interaural_coherence(343.0)
    for k in range(len(masks)):
        lk, rk = np.fft.irfft(L * masks[k], n), np.fft.irfft(R * masks[k], n)
        if np.sum(lk ** 2) < 1e-30:
            continue
        c = np.sum(lk * rk) / np.sqrt(np.sum(lk ** 2) * np.sum(rk ** 2))
        assert c == pytest.approx(rho[k], abs=0.08), k
