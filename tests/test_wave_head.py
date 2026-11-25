# -*- coding: utf-8 -*-
import numpy as np

from auralis.engine.fdtd import wave_dx
from auralis.engine.hrtf import event_fir_pair
from auralis.engine.pipeline import BakeConfig, wave_head_decode, wave_head_probes

FS, C, FC = 44100, 343.0, 500.0


def _plane_wave(u, probes, n=8192, t0=0.05):
    """Probe pressures of a band-limited plane wave arriving from unit direction u."""
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spec = np.where(f < 700.0, np.cos(0.5 * np.pi * f / 700.0) ** 2, 0.0)
    out = []
    for r in probes:
        lead = float(np.dot(u, r)) / C                       # earlier where u . r > 0
        out.append(np.fft.irfft(spec * np.exp(-2j * np.pi * f * (t0 - lead)), n))
    return np.stack(out, axis=1)


def _ild(x, lo=200.0, hi=400.0):
    X = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    k = (f >= lo) & (f < hi)
    return 10 * np.log10(X[k, 0].sum() / X[k, 1].sum())


def test_probes_sit_one_cell_around_the_head_centre():
    p = wave_head_probes([1.0, 2.0, 1.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], FC, C)
    h = wave_dx(FC, C)
    assert len(p) == 7 and np.allclose(p[0], [1.0, 2.0, 1.5])
    assert np.allclose(p[1] - p[0], [0.0, h, 0.0])            # front
    assert np.allclose(p[3] - p[0], [-h, 0.0, 0.0])           # left of a head facing +y
    assert np.allclose(p[5] - p[0], [0.0, 0.0, h])            # up


import pytest  # noqa: E402


@pytest.mark.parametrize("order", [1, 2])
def test_a_plane_wave_from_the_side_is_louder_at_that_ear_like_the_head(order):
    cfg = BakeConfig(fs=FS)
    probes = np.array(wave_head_probes([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], FC, C, order=order))
    for side in (1.0, -1.0):                                  # +y = left, -y = right
        u = np.array([0.0, side, 0.0])
        cols = _plane_wave(u, probes)
        out = wave_head_decode(cols, cfg, FC, C)
        assert np.array_equal(out[:, 2], cols[:, 0])         # the centre stays W
        fl, fr, _dl, _dr = event_fir_pair(np.ones(7), u, np.array([0, -1.0, 0]), FS, C,
                                          basis=(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                                                 np.array([0, -1.0, 0])))
        want = _ild(np.stack([np.pad(fl, (0, 4096)), np.pad(fr, (0, 4096))], axis=1))
        got = _ild(out[:, :2])
        assert np.sign(got) == side and abs(got - want) < 0.5, (side, got, want)


def test_second_order_head_encodes_a_plane_wave_as_its_harmonics():
    from auralis.engine.ir_synthesis import sh_sn3d
    from auralis.engine.pipeline import wave_head_ambi
    probes = np.array(wave_head_probes([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], FC, C, order=2))
    assert len(probes) == 19
    n = 8192
    f = np.fft.rfftfreq(n, 1.0 / FS)
    k = (f > 150.0) & (f < 400.0)
    for u in ([0.0, 1.0, 0.0], [0.6, -0.48, 0.64], [-0.36, 0.48, -0.8]):
        u = np.asarray(u) / np.linalg.norm(u)
        ambi = wave_head_ambi(_plane_wave(u, probes, n=n), FS, FC, C)
        A = np.fft.rfft(ambi, axis=0)
        want = sh_sn3d(u[None, :], 2)[0]
        got = np.real(A[k] / A[k, :1]).mean(axis=0)          # each channel over W
        assert np.abs(got - want).max() < 0.06, (u, np.round(got, 3), np.round(want, 3))
