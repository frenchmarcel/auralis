# -*- coding: utf-8 -*-
"""STI (IEC 60268-16:2020) against the standard's own compliance cases, the
lateral-energy metrics J_LF / L_J (ISO 3382-1) and the C weighting."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import speech  # noqa: E402
from auralis.engine.ism import AcousticEvent  # noqa: E402
from auralis.engine.level import dbc_total, db_total  # noqa: E402
from auralis.engine.metrics import lateral_metrics  # noqa: E402


def test_sti_of_a_uniform_modulation_depth_c32():
    """IEC 60268-16 C.3.2: m = 0, 0.1 ... 1 in every band -> these STI values."""
    expected = [0.0, 0.18, 0.30, 0.38, 0.44, 0.5, 0.56, 0.62, 0.70, 0.82, 1.0]
    got = [round(speech.sti(np.full((7, 14), m))["sti"], 2) for m in np.linspace(0, 1, 11)]
    assert sum(abs(a - b) for a, b in zip(got, expected)) < 0.01 + 1e-9, got


def test_mtf_of_exponential_decays_c33():
    """IEC 60268-16 C.3.3: exponentially decaying sines (one carrier per octave),
    RT 0.125-8 s: m(F) = [1 + (2 pi F T / 13.8)^2]^-1/2, every error < 0.05, mean < 0.01."""
    fs, dur = 48000, 18.0
    t = np.arange(int(fs * dur)) / fs
    errs = []
    for T in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        ir = sum(np.sin(2 * np.pi * f * t) for f in (125, 250, 500, 1000, 2000, 4000, 8000))
        ir = ir * np.exp(-6.91 * t / T)
        m = speech.mtf_from_ir(ir, fs)
        ref = 1.0 / np.sqrt(1.0 + (2 * np.pi * speech.MOD_FREQS * T / 13.8) ** 2)
        errs.append(np.abs(m - ref[None, :]))
    errs = np.concatenate(errs)
    assert errs.max() < 0.05 and errs.mean() < 0.01, (errs.max(), errs.mean())


def test_levels_noise_and_masking():
    mtf = np.ones((7, 14))
    assert abs(speech.sti(mtf)["sti"] - 1.0) < 1e-12
    S = speech.speech_levels_1m("normal")
    quiet = speech.sti(mtf, S)["sti"]
    assert 0.9 < quiet < 1.0                              # threshold and masking only
    assert speech.sti(mtf, S - 30)["sti"] < quiet          # near the reception threshold
    # SNR 0 dB in every band, masking removed by a flat, low spectrum: TI = 0.5
    flat = np.full(7, 50.0)
    r = speech.sti(mtf, flat, flat)
    assert abs(r["sti"] - 0.5) < 0.03
    assert speech.sti(mtf, S, speech.noise_levels("NC-50"))["sti"] < speech.sti(
        mtf, S, speech.noise_levels("NC-20"))["sti"]
    assert np.allclose(speech.masking_level_db(np.array([50, 65, 80, 110])), [-40.0, -29.9, -19.8, -10.0])


def _ev(t, d, amp=1.0, kind="specular"):
    return AcousticEvent(time=t, amps=np.full(7, amp), direction=np.asarray(d, float),
                         order=0 if kind == "direct" else 1, kind=kind)


def test_lateral_fraction_from_event_directions():
    src, rcv = [10.0, 0.0, 1.5], [0.0, 0.0, 1.5]        # lateral axis = +-Y
    hist = np.zeros((7, 400))
    events = [_ev(0.03, [1, 0, 0], kind="direct"),       # frontal direct sound
              _ev(0.05, [0, 1, 0], 0.5),                 # side reflection, 20 ms later
              _ev(0.06, [1, 0, 0], 0.5)]                 # frontal one: no lateral energy
    jlf, lj = lateral_metrics(hist, None, events, 0.001, src, rcv)
    assert np.allclose(jlf, 0.25 / 1.5)
    assert all(x is None for x in lj)                    # nothing after 80 ms
    hist[:, 200] = 0.01                                  # isotropic late bin (no dirs)
    jlf, lj = lateral_metrics(hist, np.zeros((400, 3)), events, 0.001, src, rcv)
    assert np.allclose(lj, 10 * np.log10(0.01 / 3) + 20)


def test_c_weighting():
    flat = np.full(7, 60.0)
    assert dbc_total(flat) < db_total(flat) and db_total(flat) - dbc_total(flat) < 1.0


def test_open_plan_office_free_field():
    """ISO 3382-3 in free field: 6 dB per doubling, Lp,A,S,4m 12 dB below 1 m, and the
    comfort distance where the regression line reaches 45 dB(A)."""
    from auralis.engine import office
    r = np.array([2.0, 4.0, 8.0, 16.0])
    G = np.repeat((20.0 - 20.0 * np.log10(r))[:, None], 7, axis=1)   # 1/r: G = 20 - 20 log r
    out = office.evaluate(r, G, np.ones((4, 7, 14)))
    l1 = office._dba(office.SPEECH_1M_DB)
    assert abs(out["D2S_db"] - 6.02) < 0.01
    assert abs(out["Lp_A_S_4m_db"] - (l1 - 12.04)) < 0.02
    assert abs(out["rC_m"] - 10 ** ((l1 - 45.0) / 20.0)) < 0.02
    assert out["rD_m"] is None or out["rD_note"]                  # STI never reaches 0.5
