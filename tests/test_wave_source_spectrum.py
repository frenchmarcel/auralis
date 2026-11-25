# -*- coding: utf-8 -*-
import numpy as np

from auralis.engine.bands import CENTERS
from auralis.engine.directivity import Directivity
from auralis.engine.octave_fir import band_gains_db
from auralis.engine.pipeline import (BakeConfig, SourceDef, _wave_spectrum,
                                     wave_source_gains_db, wave_source_pattern)

FS = 44100


def _source(table):
    return SourceDef(name="s", position=np.zeros(3), directivity=Directivity(np.asarray(table)),
                     forward=np.array([0.0, 0.0, 1.0]))


def _table(on_axis, back, power=1.0):
    """Per-band pattern going from on_axis to back as (theta / pi) ** power."""
    th = np.linspace(0.0, 1.0, 181) ** power
    return np.array([a + (b - a) * th for a, b in zip(on_axis, back)])


def test_on_axis_normalised_tables_leave_the_wave_band_untouched():
    src = _source(_table(np.ones(7), [0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2]))
    assert wave_source_gains_db(src, BakeConfig(), 400.0) is None
    ir = np.random.default_rng(1).standard_normal((2000, 2))
    assert _wave_spectrum(ir, src, BakeConfig(), 400.0) is ir
    assert wave_source_gains_db(SourceDef(name="o", position=np.zeros(3)), BakeConfig(),
                                400.0) is None


def test_a_table_carrying_the_spectrum_gives_its_power_to_the_wave_band():
    # quasi omni, louder in the bass than at 1 kHz (like the BRAS dodecahedron)
    q = np.array([1.25, 1.1, 1.05, 1.0, 0.5, 0.8, 1.0])
    src = _source(_table(q, q))
    g = wave_source_gains_db(src, BakeConfig(), 400.0)
    want = 10.0 * np.log10(q)
    assert np.allclose(g[:3], want[:3], atol=1e-4)     # mean_q: quadrature
    assert np.allclose(g[3:], g[2])               # above 2 fc: held
    delta = np.zeros(8192)
    delta[0] = 1.0
    out = _wave_spectrum(delta, src, BakeConfig(fs=FS), 400.0)
    assert np.allclose(band_gains_db(out, FS)[:3], want[:3], atol=0.06)


def test_a_directional_table_keeps_its_power_through_the_first_order_pattern():
    # on axis 1.5 (louder than at 1 kHz), narrow: the wave band runs a first-order pattern
    # of power clip(mean_q, 0.25, 1) for an on-axis level of 1; the gain restores mean_q
    src = _source(_table(np.full(7, 1.5), np.zeros(7), power=0.3))
    cfg = BakeConfig()
    axis, _ = wave_source_pattern(src, cfg, 400.0)
    assert axis is not None
    mq = src.directivity.mean_q()
    assert np.all(mq < 1.0)
    g = wave_source_gains_db(src, cfg, 400.0)
    k = CENTERS <= 800.0
    assert np.allclose(g[k], 10.0 * np.log10(mq[k] / np.clip(mq[k], 0.25, 1.0)))
