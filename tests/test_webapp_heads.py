# -*- coding: utf-8 -*-
"""Listener head setting of the web app: model by default, a measured head sets the
engine's hrtf_path, unknown or missing heads are refused, the decoder follows it."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import heads  # noqa: E402
from webapp.server import sim as simlib  # noqa: E402
from webapp.server.store import default_settings  # noqa: E402

FABIAN = heads.HEADS["fabian"]["path"]


def test_default_is_the_model_head():
    eng, _ = simlib.engine_settings(default_settings())
    assert default_settings()["head"] == "model" and "hrtf_path" not in eng


def test_unknown_or_missing_heads_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="unknown head"):
        simlib.engine_settings(dict(default_settings(), head="kemar"))
    monkeypatch.setitem(heads.HEADS, "ghost", {"name": "Ghost", "path": "/nonexistent.npz",
                                               "credit": None})
    with pytest.raises(ValueError, match="not installed"):
        simlib.engine_settings(dict(default_settings(), head="ghost"))
    assert {"id": "ghost", "name": "Ghost", "credit": None, "available": False} in heads.listing()


@pytest.mark.skipif(not os.path.exists(FABIAN), reason="FABIAN head not installed (data_hrtf)")
def test_fabian_head_reaches_the_engine_and_the_decoder():
    eng, _ = simlib.engine_settings(dict(default_settings(), head="fabian"))
    assert eng["hrtf_path"] == FABIAN
    from auralis.engine.ambidecode import decode, decoder_filters
    from auralis.engine.hrtf_sofa import load_hrtf
    from auralis.engine.ir_synthesis import sh_sn3d
    f, lat = decoder_filters(2, 48000, hrtf_set=load_hrtf(FABIAN, 48000))
    assert lat < f.shape[-1]
    # a plane wave from the left: the left ear is louder above 1.5 kHz
    imp = np.zeros((64, 9))
    imp[0] = sh_sn3d([0, 1, 0], 2)[0]
    y = decode(imp, f)
    spec = np.abs(np.fft.rfft(y, 4096, axis=0))
    fr = np.fft.rfftfreq(4096, 1 / 48000)
    band = (fr > 1500) & (fr < 6000)
    assert 20 * np.log10(spec[band, 0].mean() / spec[band, 1].mean()) > 8.0
