# -*- coding: utf-8 -*-
import numpy as np

from auralis.engine.geometry import make_shoebox
from auralis.engine.pipeline import (BakeConfig, ReceiverDef, SourceDef, bake,
                                     effective_max_bounces)
from auralis.engine.rirmetrics import band_metrics


def _box(alpha):
    mat = {"absorption": np.full(7, alpha), "scattering": np.full(7, 0.5),
           "transmission": np.zeros(7)}
    return make_shoebox((6.0, 5.0, 4.0), wall_material=mat)


def test_the_cap_follows_the_ir_length_and_never_drops_below_the_setting():
    sc = _box(0.02)
    cfg = BakeConfig()
    short = effective_max_bounces(sc, cfg, 0.2, 343.0)
    long_ = effective_max_bounces(sc, cfg, 4.0, 343.0)
    assert short == cfg.max_bounces
    mfp = 4 * (6 * 5 * 4 * 0.85) / (2 * (30 + 24 + 20))
    assert long_ == int(np.ceil(2 * 343.0 * 4.0 / mfp))
    assert effective_max_bounces(sc, BakeConfig(max_bounces=5000), 4.0, 343.0) == 5000


def test_a_reverberant_room_keeps_its_tail():
    """alpha 0.02: Eyring T ~ 5.6 s (with air at 1 kHz). A fixed cap of 200 bounces would
    stop the rays after ~0.8 s and make T30 far too short."""
    sc = _box(0.02)
    cfg = BakeConfig(n_rays=20000, max_time=3.0, fs=16000, ism_order=1, seed=3, fdtd=False,
                     complex_reflection=False, diffraction=False, normalize=False,
                     output_layout="stereo", engine="native")
    res = bake(sc, SourceDef(name="s", position=np.array([1.5, 1.2, 1.6])),
               ReceiverDef(name="r", position=np.array([4.2, 3.6, 1.4])), cfg)
    assert res.stats["max_bounces"] > 600
    t30 = band_metrics(res.ir[:, 0], cfg.fs)["T30"][3]
    V, S = 120.0, 148.0
    a_air = 4 * 0.0012 * V                       # 1 kHz, about 20 C / 50 %
    eyring = 0.161 * V / (-S * np.log(1 - 0.02) + a_air)
    assert abs(t30 / eyring - 1.0) < 0.15, (t30, eyring)
