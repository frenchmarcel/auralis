# -*- coding: utf-8 -*-
import numpy as np

from auralis.engine.level import level_stats
from auralis.engine.pipeline import hybrid_level_shift_db
from auralis.engine.rirmetrics import strength_db
from webapp.server.runner import band63_levels

FS = 44100


def _noise(n=FS, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * np.exp(-np.arange(n) / (0.2 * FS))


def test_strength_of_a_free_field_impulse_at_10_m_is_zero():
    ir = np.zeros(FS)
    ir[1300] = 0.1                                    # 1/r at 10 m, pressure re 1 m
    g = strength_db(ir, FS, [63.0, 125.0, 1000.0, 8000.0])
    assert np.allclose(g, 0.0, atol=1e-6)
    assert np.allclose(strength_db(np.stack([ir, ir], axis=1), FS, [125.0]), 0.0, atol=1e-6)


def test_shift_takes_the_wave_band_below_the_transition_and_the_echogram_above():
    hist = np.full((7, 10), 1e-3)                       # G of the echogram: 0 dB per band
    for amp, want in ((0.1, 0.0), (0.2, 20 * np.log10(2.0))):
        wave = np.zeros(FS)
        wave[1300] = amp                                 # G of the wave band: 0 or +6 dB
        sh = hybrid_level_shift_db(wave, hist, FS, 500.0)
        assert abs(sh[0] - want) < 0.1                   # 125 Hz: the wave band
        assert abs(sh[3]) < 0.15                         # 1 kHz: the echogram (HP share ~1)
        assert np.all(sh[4:] == 0.0)                     # above 2 fc: untouched
    silent = hybrid_level_shift_db(np.zeros(FS), hist, FS, 500.0)
    assert silent[0] < -20.0                             # no wave energy, no HP share at 125 Hz


def test_level_stats_shift():
    h = np.ones((7, 100)) * 1e-4
    base = level_stats(h)
    sh = np.array([3.0, -2.0, 0, 0, 0, 0, 0])
    lv = level_stats(h, shift_db=sh)
    assert np.allclose(np.asarray(lv["g_bands"]) - np.asarray(base["g_bands"]), sh)
    assert np.allclose(np.asarray(lv["spl_bands"]) - np.asarray(base["spl_bands"]), sh)
    assert lv["g_mid"] == base["g_mid"]


def test_band63_from_a_flat_ir_equals_the_125_hz_band():
    d = np.zeros(FS)
    d[100] = 1.0
    g, spl = band63_levels(d, FS, 5.0, 70.0)
    assert abs(g - 5.0) < 1e-6 and abs(spl - 70.0) < 1e-6
    assert band63_levels(d, FS, None, None) == (None, None)


def test_source_gain_scales_the_wave_band_too():
    from dataclasses import replace
    from auralis.engine import make_shoebox
    from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake
    sc = make_shoebox((3.0, 2.4, 2.2), wall_material="plaster", floor_material="carpet")
    cfg = BakeConfig(fs=8000, n_rays=3000, seed=2, fdtd=True, fdtd_fcross=150.0,
                     engine="python", output_layout="stereo", max_time=0.3)
    rcv = ReceiverDef("R", np.array([2.1, 1.5, 1.2]))
    out = []
    for gain in (0.0, 20 * np.log10(2.0)):
        src = SourceDef("S", np.array([0.9, 0.8, 1.3]), gain_db=gain)
        r = bake(sc, src, rcv, replace(cfg))
        out.append(np.asarray(r.ir) / r.ir_scale)
    assert np.allclose(out[1], 2.0 * out[0], rtol=1e-6, atol=1e-9 * np.abs(out[0]).max())
