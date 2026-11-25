# -*- coding: utf-8 -*-
"""Measured HRTF sets and the diffuse tail: solid-angle weights, diffuse-field colour
and interaural coherence taken from the set, applied to the stochastic tail."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine.bands import band_masks  # noqa: E402
from auralis.engine.hrtf_sofa import HrtfSet, solid_angle_weights  # noqa: E402
from auralis.engine.ir_synthesis import synthesize_binaural  # noqa: E402

FS = 44100


def _grid(step=5.0):
    """Equal-angle grid (crowded at the poles), one point per pole."""
    az, el = np.meshgrid(np.arange(0.0, 360.0, step), np.arange(-90 + step, 90.0, step))
    az = np.concatenate([az.ravel(), [0.0, 0.0]])
    el = np.concatenate([el.ravel(), [-90.0, 90.0]])
    return az, el


def _units(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=1)


def _save_set(path, ir, delay, az, el):
    np.savez(path, ir=ir.astype(np.float32), delay=delay.astype(np.float32),
             az_deg=az.astype(np.float32), el_deg=el.astype(np.float32), fs=FS, name="test")
    return str(path)


def test_solid_angle_weights_undo_the_equal_angle_grid():
    """Nearest-neighbour areas are approximate on the worst grid (1.5 % off on
    <z^2>, whatever k); unweighted, the crowded poles put it 46 % off. The FABIAN
    grid gives 0.3339."""
    u = _units(*_grid())
    w = solid_angle_weights(u)
    assert abs((w * u[:, 2] ** 2).sum() - 1 / 3) < 0.01
    assert np.abs(w @ u).max() < 1e-3
    assert abs((u[:, 2] ** 2).mean() - 1 / 3) > 0.1


def test_uniform_set_gives_its_own_colour_and_full_coherence(tmp_path):
    az, el = _grid(10.0)
    h = np.zeros(64)
    h[:4] = [1.0, -0.6, 0.3, 0.1]
    ir = np.broadcast_to(h, (len(az), 2, 64)).copy()
    hs = HrtfSet(_save_set(tmp_path / "u.npz", ir, np.zeros((len(az), 2)), az, el), FS)
    fir = hs.diffuse_field_firs()
    f = np.fft.rfftfreq(4096, 1 / FS)
    want = np.abs(np.fft.rfft(h, 4096))
    got = np.abs(np.fft.rfft(fir[0], 4096))
    sel = (f > 100) & (f < 15000)
    assert np.abs(20 * np.log10(got[sel] / want[sel])).max() < 0.1
    assert np.allclose(fir[0], fir[1])
    assert (hs.band_coherence() > 0.99).all()


def test_coherence_of_two_points_in_a_diffuse_field(tmp_path):
    """Delays of two free-field points 18 cm apart: the band coherence is the
    band average of sinc(k d), the textbook diffuse-field result."""
    az, el = _grid(5.0)
    u = _units(az, el)
    d, c = 0.18, 343.0
    ears = np.array([[0.0, d / 2, 0.0], [0.0, -d / 2, 0.0]])
    delay = 40.0 - (u @ ears.T) / c * FS          # an ear nearer the source hears it earlier
    ir = np.zeros((len(az), 2, 32))
    ir[:, :, 0] = 1.0
    hs = HrtfSet(_save_set(tmp_path / "p.npz", ir, delay, az, el), FS)
    f = np.fft.rfftfreq(2048, 1 / FS)
    m2 = band_masks(f, FS) ** 2
    want = m2 @ np.sinc(2 * f * d / c) / m2.sum(axis=1)
    assert np.abs(hs.band_coherence() - np.clip(want, -0.999, 0.999)).max() < 0.03


def test_a_measured_set_colours_the_diffuse_tail(tmp_path):
    """Same noise, two sets: the tail spectra differ by the diffuse-field filter."""
    az, el = _grid(10.0)
    flat = np.zeros((len(az), 2, 64))
    flat[:, :, 0] = 1.0
    h = np.zeros(64)
    h[:3] = [1.0, 0.0, 0.8]                         # comb: deep dip at fs / 4
    peaked = np.broadcast_to(h, flat.shape).copy()
    zero = np.zeros((len(az), 2))
    sets = [HrtfSet(_save_set(tmp_path / n, ir, zero, az, el), FS)
            for n, ir in (("f.npz", flat), ("p.npz", peaked))]
    hist = np.zeros((7, 400))
    hist[:, 10:] = np.exp(-np.arange(390) / 80.0)
    dirs = np.tile([1.0, 0.0, 0.0], (400, 1))
    irs = [synthesize_binaural([], hist, dirs, 0.001, FS, 343.0, forward=(1, 0, 0),
                               up=(0, 0, 1), seed=3, normalize=False, hrtf_set=s)[0]
           for s in sets]
    f = np.fft.rfftfreq(len(irs[0]), 1 / FS)
    S = [np.abs(np.fft.rfft(x[:, 0])) ** 2 for x in irs]
    H2 = np.abs(np.fft.rfft(h, len(irs[0]))) ** 2
    for lo, hi in ((700, 1400), (2800, 5600), (9000, 10500)):
        b = (f >= lo) & (f < hi)
        ratio = 10 * np.log10(S[1][b].sum() / S[0][b].sum())
        want = 10 * np.log10((S[0][b] * H2[b]).sum() / S[0][b].sum())
        assert abs(ratio - want) < 0.5, (lo, ratio, want)


def test_ear_responses_do_not_depend_on_the_directions_asked_before(tmp_path):
    """Two directions less than 1/2048 apart: the second one gets its own interpolation, as
    in a fresh set. The cache key is exact, so a pair never depends on the pairs before it."""
    az, el = _grid(10.0)
    rng = np.random.default_rng(3)
    ir = rng.standard_normal((len(az), 2, 64))
    path = _save_set(tmp_path / "c.npz", ir, np.zeros((len(az), 2)), az, el)
    basis = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))
    d1 = np.array([0.6, 0.5, 0.3])
    d2 = d1 + np.array([1e-4, -1e-4, 0.0])
    warm = HrtfSet(path, FS)
    warm.ear_responses(d1, basis, 256)
    fresh = HrtfSet(path, FS)
    for a, b in zip(warm.ear_responses(d2, basis, 256), fresh.ear_responses(d2, basis, 256)):
        assert np.array_equal(a, b)
