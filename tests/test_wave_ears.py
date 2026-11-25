# -*- coding: utf-8 -*-
"""Ear probes of the wave band: spaced from the head (HRTF set or rigid sphere), and used
at that spacing by the static bake, the per-source precompute and animated renders."""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import auralis.engine.fdtd as F  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402
from auralis.engine.hrtf import HEAD_RADIUS  # noqa: E402
from auralis.engine.pipeline import (BakeConfig, ReceiverDef, SourceDef, bake,  # noqa: E402
                                     source_low_bands, wave_ear_half)

FABIAN = os.path.join(REPO, "data_hrtf", "fabian", "fabian_measured_hato0.npz")


class _Seen(Exception):
    pass


def test_spacing_comes_from_the_head():
    assert wave_ear_half(BakeConfig()) == pytest.approx(1.5 * HEAD_RADIUS)
    assert wave_ear_half(BakeConfig(wave_ear_half=0.0875)) == 0.0875
    # loudspeaker output: no head, the HRTF set is ignored
    assert wave_ear_half(BakeConfig(hrtf_path=FABIAN, output_layout="stereo")) == \
        pytest.approx(1.5 * HEAD_RADIUS)


@pytest.mark.skipif(not os.path.exists(FABIAN), reason="FABIAN head not installed")
def test_fabian_spacing_reproduces_its_diffuse_coherence():
    from auralis.engine.hrtf_sofa import load_hrtf
    half = wave_ear_half(BakeConfig(hrtf_path=FABIAN, fs=44100))
    assert 0.155 < half < 0.18
    f, sll, srr, slr = load_hrtf(FABIAN, 44100)._diffuse_spectra()
    s = (f >= 100.0) & (f <= 600.0)
    ic = slr.real[s] / np.sqrt(sll[s] * srr[s])
    probes = np.sinc(2.0 * f[s] * 2.0 * half / 343.0)
    old = np.sinc(2.0 * f[s] * 0.175 / 343.0)
    assert np.sqrt(np.mean((probes - ic) ** 2)) < 0.03
    assert np.sqrt(np.mean((old - ic) ** 2)) > 0.15


def _capture(monkeypatch):
    seen = {}

    def fake(scene, src, rcv, **kw):
        seen["rcv"] = [np.asarray(r, dtype=np.float64) for r in rcv]
        raise _Seen()

    monkeypatch.setattr(F, "run_low_band", fake)
    return seen


@pytest.mark.parametrize("path", ["bake", "precompute"])
def test_probes_sit_at_the_spacing(monkeypatch, path):
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="plaster")
    R = np.array([2.0, 1.5, 1.2])
    rcv = ReceiverDef(name="r", position=R, forward=np.array([0.0, 1.0, 0.0]),
                      up=np.array([0.0, 0.0, 1.0]))
    src = SourceDef(name="s", position=np.array([1.0, 1.0, 1.4]))
    cfg = BakeConfig(fs=8000, n_rays=500, fdtd=True, fdtd_fcross=150.0, max_time=0.2,
                     wave_ear_half=0.16, diffraction=False, engine="native")
    seen = _capture(monkeypatch)
    with pytest.raises(_Seen):
        if path == "bake":
            bake(sc, src, rcv, cfg)
        else:
            source_low_bands(sc, src, [rcv], cfg)
    left, right = seen["rcv"][:2]
    assert np.allclose(right - left, [0.32, 0.0, 0.0]) or np.allclose(left - right,
                                                                        [0.32, 0.0, 0.0])
    assert np.allclose(0.5 * (left + right), R)
