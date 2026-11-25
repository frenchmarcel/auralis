# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine.pipeline import (BakeConfig, ReceiverDef, SourceDef, bake,  # noqa: E402
                                     precompute_low_bands, wave_bands, wave_omni_lp)


def _case(**kw):
    sc = make_shoebox((5.0, 4.0, 2.8), wall_material="plaster")
    src = SourceDef("S", np.array([1.2, 1.0, 1.4]))
    rcv = ReceiverDef("R", np.array([3.6, 2.7, 1.2]), forward=(1.0, 0.0, 0.0))
    cfg = BakeConfig(n_rays=20000, seed=5, normalize=False, fdtd=True, fdtd_fcross=360.0,
                     max_time=0.6, ambisonic_order=1, **kw)
    low = precompute_low_bands(sc, src, [rcv], cfg)["R"]
    return bake(sc, src, rcv, cfg, low_band=low)


def test_wave_only_is_the_low_passed_wave_band_of_the_hybrid():
    """Same wave band: the wave-only W is the hybrid's wave part through the crossover's
    low-pass, its X Y Z are silent, G and SPL keep the hybrid's wave share, the parameters
    above the transition are None."""
    hyb = _case()
    wav = _case(wave_only=True)
    fc = hyb.omni_parts["fc"]
    w = wave_omni_lp(hyb.omni_parts["wave"], fc, hyb.fs)
    assert np.allclose(wav.ambi[:, 0], w, atol=1e-12 * np.abs(w).max())
    assert not np.any(wav.ambi[:, 1:])
    ok = wave_bands(fc)
    assert ok.tolist() == [True, True, False, False, False, False, False]
    m = wav.metrics
    assert m["wave_only"] and m["spl_db"] is None
    for b, lab in enumerate(["125", "250", "500", "1k", "2k", "4k", "8k"]):
        assert (m["per_band"][lab]["RT"] is not None) == ok[b]
        assert (m["g_bands"][b] is not None) == ok[b]
    # the hybrid adds the geometrical share of these octaves: the wave alone is weaker
    for b in np.nonzero(ok)[0]:
        assert m["g_bands"][b] < hyb.metrics["g_bands"][b] + 1e-9
    # no energy left above the transition (the crossover's low-pass)
    S = np.abs(np.fft.rfft(wav.ambi[:, 0])) ** 2
    f = np.fft.rfftfreq(len(wav.ambi), 1.0 / wav.fs)
    assert S[f > 2.5 * fc].sum() < 1e-4 * S[f < fc].sum()


def test_wave_only_without_the_wave_band_is_an_error():
    import pytest
    sc = make_shoebox((5.0, 4.0, 2.8), wall_material="plaster")
    src = SourceDef("S", np.array([1.2, 1.0, 1.4]))
    rcv = ReceiverDef("R", np.array([3.6, 2.7, 1.2]))
    with pytest.raises(ValueError):
        bake(sc, src, rcv, BakeConfig(n_rays=2000, fdtd=False, wave_only=True, max_time=0.3))
    from auralis.engine.pipeline import bake_pair
    with pytest.raises(RuntimeError):
        bake_pair(sc, src, rcv, BakeConfig(n_rays=2000, fdtd=True, wave_only=True,
                                           max_time=0.3), low={}, own_wave_pass=False)
