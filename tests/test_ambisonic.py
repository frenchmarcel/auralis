# -*- coding: utf-8 -*-
"""Ambisonic IR (order 2, ACN/SN3D): exact direction of the deterministic arrivals,
diffuse late field with the SN3D energy ratios, W energy-matched to the echogram,
and the wave band fused into W without changing the binaural output."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine.ir_synthesis import sh_sn3d, synthesize_ambisonic  # noqa: E402
from auralis.engine.ism import AcousticEvent  # noqa: E402
from auralis.engine.pipeline import (BakeConfig, ReceiverDef, SourceDef, bake,  # noqa: E402
                                     precompute_low_bands)


def test_harmonics_are_sn3d():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((200000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    y = sh_sn3d(d, 2)
    deg = np.floor(np.sqrt(np.arange(9)))
    assert np.allclose((y ** 2).mean(axis=0), 1.0 / (2 * deg + 1), atol=0.005)
    assert np.allclose(sh_sn3d([1, 0, 0])[0][[0, 1, 2, 3]], [1, 0, 0, 1])


def test_events_carry_their_direction_and_the_tail_is_diffuse():
    fs, n_bins = 48000, 600
    d = np.array([0.3, -0.8, 0.52])
    d /= np.linalg.norm(d)
    ev = [AcousticEvent(time=0.01, amps=np.full(7, 1.0), direction=d, order=0, kind="direct")]
    hist = np.zeros((7, n_bins))
    hist[:, 20:] = 1e-4 * np.exp(-np.arange(n_bins - 20) / 120.0)
    a = synthesize_ambisonic(ev, hist, np.zeros((n_bins, 3)), 0.001, fs, seed=4)
    k = int(0.01 * fs)
    seg = a[k - 100:k + 100]
    inten = np.array([(seg[:, 0] * seg[:, 3]).sum(), (seg[:, 0] * seg[:, 1]).sum(),
                      (seg[:, 0] * seg[:, 2]).sum()])
    assert np.allclose(inten / np.linalg.norm(inten), d, atol=1e-6)
    late = a[int(0.1 * fs):int(0.5 * fs)]
    ratio = (late ** 2).sum(axis=0) / (late[:, 0] ** 2).sum()
    deg = np.floor(np.sqrt(np.arange(9)))
    assert np.allclose(ratio, 1.0 / (2 * deg + 1), atol=0.04), ratio


def test_bake_ambisonic_w_and_wave_band():
    sc = make_shoebox((5.0, 4.0, 2.8), wall_material="plaster")
    src = SourceDef("S", np.array([1.2, 1.0, 1.4]))
    rcv = ReceiverDef("R", np.array([3.6, 2.7, 1.2]), forward=(1.0, 0.0, 0.0))
    base = dict(n_rays=20000, seed=5, normalize=False, fdtd=True, fdtd_fcross=250.0,
                max_time=0.6)
    low0 = precompute_low_bands(sc, src, [rcv], BakeConfig(**base))["R"]
    low2 = precompute_low_bands(sc, src, [rcv], BakeConfig(**base, ambisonic_order=2))["R"]
    assert low0[0].shape[1] == 2 and low2[0].shape[1] == 3
    # ears untouched by the centre probe: bit-exact on the CPU, to float32 rounding on the
    # GPU, which is not deterministic run to run (up to ~2.3e-5 relative measured)
    assert np.abs(low0[0] - low2[0][:, :2]).max() <= 1e-4 * np.abs(low0[0]).max()
    res = bake(sc, src, rcv, BakeConfig(**base, ambisonic_order=2), low_band=low2)
    ref = bake(sc, src, rcv, BakeConfig(**base), low_band=low0)
    assert res.ambi.shape == (len(res.ir), 9)
    # same rays: the binaural IR does not depend on the ambisonic option (GPU sums may
    # differ in the last float bits between two runs)
    assert np.abs(res.ir - ref.ir).max() <= 1e-6 * np.abs(ref.ir).max()
    # the wave band lands in W: below the crossover W follows the centre probe
    from auralis.engine.fdtd import design_crossover
    lp, _hp, delay = design_crossover(250.0, res.fs)
    w_low = np.convolve(res.ambi[:, 0], lp)[delay:delay + len(res.ir)]
    c_low = np.convolve(np.convolve(low2[0][:, 2], lp)[delay:delay + len(res.ir)], lp)[
        delay:delay + len(res.ir)]
    assert np.corrcoef(w_low, c_low)[0, 1] > 0.95


def test_rotation_matrices_and_decoder():
    from auralis.engine.ambidecode import decode, decoder_filters, rotation_matrix
    d = np.array([[1.0, 0.0, 0.0]])
    a = sh_sn3d(d, 2)[0]
    m = rotation_matrix(2, 90.0)                     # field turned 90 deg: front -> left
    assert np.allclose(m @ a, sh_sn3d([0, 1, 0], 2)[0], atol=1e-9)
    r = rotation_matrix(2, 37.0, -12.0, 5.0)
    assert np.allclose(r @ np.linalg.inv(r) , np.eye(9), atol=1e-9)
    f, lat = decoder_filters(2, 48000)
    assert f.shape == (9, 2, 256) and 0 < lat < 256
    # a plane wave from the left is louder in the left ear above 1 kHz
    imp = np.zeros((64, 9))
    imp[0] = sh_sn3d([0, 1, 0], 2)[0]
    y = decode(imp, f)
    spec = np.abs(np.fft.rfft(y, 1024, axis=0))
    fr = np.fft.rfftfreq(1024, 1 / 48000)
    band = (fr > 1500) & (fr < 6000)
    assert 20 * np.log10(spec[band, 0].mean() / spec[band, 1].mean()) > 5.0


@pytest.mark.parametrize("order,hi", [(1, 400.0), (2, 800.0)])
@pytest.mark.parametrize("az", [90.0, 45.0, -30.0])
def test_head_model_decoder_keeps_the_models_level_differences(order, hi, az):
    """A plane wave decoded through the analytic head keeps the model's interaural level
    difference within 0.25 dB, octave by octave, from 100 Hz to the order's limit. This needs
    the model's delays referred to the head centre."""
    from auralis.engine.ambidecode import decode, decoder_filters, hrirs
    fs = 48000
    d = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
    f, _lat = decoder_filters(order, fs)
    imp = np.zeros((64, f.shape[0]))
    imp[0] = sh_sn3d(d[None], order)[0]
    y = decode(imp, f)
    h, _ = hrirs(d[None], fs)
    fr = np.fft.rfftfreq(4096, 1 / fs)
    Y = np.abs(np.fft.rfft(y, 4096, axis=0)) ** 2
    H = np.abs(np.fft.rfft(h[0], 4096, axis=-1)) ** 2
    lo = 100.0
    while lo < hi:
        k = (fr >= lo) & (fr < 2 * lo)
        got = 10 * np.log10(Y[k, 0].sum() / Y[k, 1].sum())
        want = 10 * np.log10(H[0, k].sum() / H[1, k].sum())
        assert abs(got - want) < 0.25, (lo, got, want)
        lo *= 2


def test_solver_parts_recompose_the_hybrid_omni():
    """The geometrical and wave omni parts kept by bake fuse back into W exactly."""
    from auralis.engine.fdtd import fuse_low_high
    sc = make_shoebox((5.0, 4.0, 2.8), wall_material="plaster")
    src = SourceDef("S", np.array([1.2, 1.0, 1.4]))
    rcv = ReceiverDef("R", np.array([3.6, 2.7, 1.2]), forward=(1.0, 0.0, 0.0))
    cfg = BakeConfig(n_rays=20000, seed=5, normalize=False, fdtd=True, fdtd_fcross=250.0,
                     max_time=0.6, ambisonic_order=2)
    low = precompute_low_bands(sc, src, [rcv], cfg)["R"]
    res = bake(sc, src, rcv, cfg, low_band=low)
    p = res.omni_parts
    assert p is not None and p["fc"] == 250.0
    w = fuse_low_high(p["geo"][:, None], p["wave"], p["fc"], res.fs)[:, 0]
    assert np.allclose(w, res.ambi[:, 0], atol=1e-12 * np.abs(w).max() + 1e-15)
