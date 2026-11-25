# -*- coding: utf-8 -*-
"""Ensembles and server-side render: instrument sections, stage layout
(players upstage of the conductor, no overlap) and an exact binaural mix."""

import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.wavio import read_wav, write_wav  # noqa: E402
from webapp.server import sounds  # noqa: E402


def test_sections_of_tricky_names():
    cases = {"double bassA": "bass", "CBass_1": "bass", "4Seasons_Alto": "viola",
             "contratenor": "voice", "BluesA_Voc": "voice", "Hrn_1": "horn", "Trp_2": "brass",
             "1st violinA": "violin1", "Vl_2_1_3": "violin2", "Choir_3": "choir", "cymbals": "timpani"}
    for name, sec in cases.items():
        assert sounds.section_of(name) == sec, name


def _orchestra():
    names = (["1st violin%d" % i for i in range(10)] + ["2nd violin%d" % i for i in range(8)]
             + ["viola%d" % i for i in range(6)] + ["cello%d" % i for i in range(5)]
             + ["double bass%d" % i for i in range(4)] + ["flute1", "flute2", "oboe1", "oboe2",
             "clarinet1", "clarinet2", "bassoon1", "bassoon2", "horn1", "horn2", "horn3", "horn4",
             "trumpet1", "trumpet2", "trombone1", "timpani"])
    return {"title": "test", "tracks": [{"name": n, "sound_id": "s%d" % i} for i, n in enumerate(names)]}


def test_layout_no_overlap_and_fits_the_stage():
    ens = _orchestra()
    P, scale = sounds.layout(ens["tracks"], 14.0, 10.0)
    d = np.hypot(*(P[:, None, :2] - P[None, :, :2]).transpose(2, 0, 1)) + np.eye(len(P)) * 99
    assert d.min() >= sounds.MIN_GAP * scale - 1e-9
    assert np.ptp(P[:, 1]) <= 14.0 and P[:, 0].max() <= 10.0


def test_players_sit_upstage_facing_the_audience():
    ens = _orchestra()
    src, _ = sounds.ensemble_sources(ens, [5.0, 2.0, 0.0], 90.0, 12.0, 8.0, "E1_")
    toward_audience = np.array([0.0, 1.0])
    for s in src:
        assert np.dot(np.subtract(s["position"][:2], [5.0, 2.0]), toward_audience) < 0
        assert s["azimuth_deg"] == 90.0 and s["sound_id"].startswith("s")
    v1 = [s for s in src if s["name"].startswith("1st")]
    # the audience at +Y looks towards -Y: its (and the conductor's) left is +X
    assert all(s["position"][0] > 5.0 for s in v1)


def test_render_is_the_exact_binaural_sum():
    fs, n = 44100, 4000
    rng = np.random.default_rng(3)
    d = tempfile.mkdtemp()
    paths, irs = [], []
    for k in range(2):
        x = rng.standard_normal(n) * 0.1
        p = os.path.join(d, "t%d.wav" % k)
        write_wav(p, x.astype(np.float32), fs)
        paths.append(p)
        h = np.zeros((64, 2))
        h[5 + k, 0], h[9 + 2 * k, 1] = 0.5, 0.25           # simple delays per ear
        irs.append(h)
    strips = [(paths[0], 0.0, "A", "credit A"), (paths[1], -6.0, "B", None)]
    wav, txt, info = sounds.render(strips, irs, fs, os.path.join(d, "out"))
    y, fs_out = read_wav(wav)
    x0 = read_wav(paths[0])[0].reshape(-1)
    x1 = read_wav(paths[1])[0].reshape(-1)
    g = 10 ** (-6 / 20)
    ref = np.zeros((n + 64, 2))
    ref[5:5 + n, 0] += 0.5 * x0
    ref[9:9 + n, 1] += 0.25 * x0
    ref[6:6 + n, 0] += 0.5 * g * x1
    ref[11:11 + n, 1] += 0.25 * g * x1
    k = float((y * ref).sum() / (ref * ref).sum())        # actual normalisation gain
    assert abs(20 * math.log10(k) - info["normalization_db"]) < 0.006   # reported rounded to 0.01 dB
    assert fs_out == fs
    assert np.abs(y - k * ref).max() < 1e-6                 # float32 output
    assert "credit A" in open(txt).read()


def test_render_resamples_the_ir_to_the_track_rate():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.wav")
    tone = np.sin(2 * math.pi * 1000 * np.arange(44100) / 44100) * 0.1
    write_wav(p, tone.astype(np.float32), 44100)
    h = np.zeros((480, 2))
    h[0] = 1.0                                              # identity IR at 48 kHz
    wav, _, info = sounds.render([(p, 0.0, "A", None)], [h], 48000, os.path.join(d, "o"))
    y, fs = read_wav(wav)
    assert fs == 44100 and info["fs"] == 44100
    Y = np.abs(np.fft.rfft(y[:44100, 0]))
    assert abs(np.argmax(Y) - 1000) <= 1                    # the tone stays at 1 kHz


def _bursts(fs, lengths, pause_s, lead_s=0.3, f0=220.0):
    """Tone bursts of the given lengths (s), separated by silent pauses."""
    parts = [np.zeros(int(lead_s * fs))]
    for L in lengths:
        t = np.arange(int(L * fs)) / fs
        parts += [0.5 * np.sin(2 * np.pi * f0 * t), np.zeros(int(pause_s * fs))]
    return np.concatenate(parts)


def test_utterances_are_split_at_pauses_only():
    fs = 16000
    x = _bursts(fs, [1.0, 2.0, 0.5], pause_s=0.6)
    x[int(0.8 * fs):int(0.9 * fs)] = 0.0              # 0.1 s gap inside a word: not a pause
    u = sounds.utterances(x, fs)
    assert len(u) == 3
    lengths = [(b - a) / fs - 2 * sounds.TURN_PAD_S for a, b in u]
    assert np.allclose(lengths, [1.0, 2.0, 0.5], atol=0.03)


def test_turns_alternate_without_overlap_and_no_sentence_is_cut():
    fs = 16000
    a = _bursts(fs, [1.0] * 6, pause_s=1.0, f0=200.0)
    b = _bursts(fs, [0.7] * 2, pause_s=1.0, f0=300.0)
    n = int(7.0 * fs)
    (ya, yb), turns = sounds.take_turns([a, b], fs, n, gap_s=0.4, start_s=0.5)
    assert len(ya) == len(yb) == n
    assert not ((np.abs(ya) > 0) & (np.abs(yb) > 0)).any()
    order = [i for i, *_ in turns]
    assert order[:4] == [0, 1, 0, 1] and set(order[4:]) == {0}    # b ran out: a goes on
    assert abs(turns[0][1] - 0.5) < 1e-9
    for (_i, _t0, t1), (_j, t0, _t1) in zip(turns, turns[1:]):
        assert abs(t0 - t1 - 0.4) < 2.0 / fs
    assert turns[-1][2] <= n / fs                       # the last turn ends inside the render
    used = sum(1 for i in order if i == 0)
    left = 6 - used
    assert left >= 1                                    # it stopped before cutting one short


def test_speech_calibration_and_background_noise_levels():
    fs = 16000
    x = _bursts(fs, [1.0, 1.5], pause_s=1.0)
    g = sounds.speech_gain(x, fs, 60.0)
    lv = sounds.active_octave_levels(x * g, fs) - 20 * np.log10(sounds.P_REF)
    assert abs(sounds.dba_total(lv) - 60.0) < 1e-6         # the pauses do not count
    flat = np.full(7, 30.0)
    y = sounds.diffuse_noise(flat, 8 * fs, fs, np.random.default_rng(0))
    for ch in range(2):
        L = 10 * np.log10(np.mean(y[:, ch] ** 2) / sounds.P_REF ** 2)
        # 7 bands at 30 dB, minus what the 8 kHz band loses above Nyquist at 16 kHz
        assert 36.0 < L < 38.5, L
    lo = np.fft.rfftfreq(len(y), 1 / fs) < 180
    Y = np.fft.rfft(y, axis=0)
    rho = np.real((Y[lo, 0] * np.conj(Y[lo, 1])).sum()) / np.sqrt(
        (np.abs(Y[lo, 0]) ** 2).sum() * (np.abs(Y[lo, 1]) ** 2).sum())
    assert rho > 0.85                                      # diffuse field: coherent at 125 Hz
