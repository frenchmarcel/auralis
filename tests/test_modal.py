# -*- coding: utf-8 -*-
"""Forced-response (modal) maps from the wave solver: the transfer function at chosen
frequencies equals the spectrum of the usual low-band IR, and its peaks sit on the
room modes of a box."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine.fdtd import run_low_band  # noqa: E402


def _box():
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="concrete")
    sc.absorption[:] = 0.08
    return sc


def test_transfer_matches_the_ir_spectrum_and_the_modes():
    sc = _box()
    src, rcv = [0.4, 0.4, 0.4], [3.6, 2.6, 2.1]               # opposite corners
    freqs = np.arange(30.0, 100.0, 0.25)
    kw = dict(fs=8000, duration=1.5, f_cross=150.0, speed_of_sound=343.0, conformal=False)
    tf, info = run_low_band(sc, src, [rcv], transfer_freqs=freqs, **kw)
    ir, _ = run_low_band(sc, src, [rcv], **kw)
    mag = np.abs(tf[:, 0])
    # the IR spectrum at the same bins (the IR is the same response, resampled to fs)
    spec = np.abs(np.fft.rfft(ir[:, 0], 1 << 16))
    f_ir = np.fft.rfftfreq(1 << 16, 1.0 / 8000)
    ref = np.interp(info.transfer_freqs, f_ir, spec)
    ref *= np.median(mag / ref)                               # one global factor (scaling)
    assert np.corrcoef(mag, ref)[0, 1] > 0.9                  # the IR is truncated at 1.5 s
    for lo_, hi_ in ((35, 50), (55, 63), (63, 68)):           # same resonances
        sel = (freqs > lo_) & (freqs < hi_)
        assert abs(freqs[sel][np.argmax(mag[sel])] - freqs[sel][np.argmax(ref[sel])]) < 0.5
    # the first axial mode of the 4 m length (c / 8 = 42.9 Hz for rigid walls) comes out
    # within the measured voxelisation bias (+4.6 % staircase, -8.1 % conformal)
    sel = (freqs > 35) & (freqs < 50)
    f_peak = freqs[sel][np.argmax(mag[sel])]
    assert 0.9 * 42.875 < f_peak < 1.09 * 42.875, f_peak


def test_field_animation_wavefront_travels_at_c():
    """The field frames (wave band on a plane of points, high-passed) carry a wavefront
    that travels at the speed of sound (the DC term of the closed room removed)."""
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from webapp.server.field_runner import causal_fade, high_pass
    sc = make_shoebox((8.0, 6.0, 3.0), wall_material="concrete")
    sc.absorption[:] = 0.2
    src = np.array([1.0, 3.0, 1.5])
    xs = np.arange(1.6, 6.5, 0.4)
    pts = [np.array([x, 3.0, 1.5]) for x in xs]
    fs = 1500
    ir, _ = run_low_band(sc, src, pts, fs=fs, duration=0.1, f_cross=300.0,
                         speed_of_sound=343.0, conformal=False, subsonic_hz=0.0)
    fr = np.abs(high_pass(ir, fs) * causal_fade((xs - 1.0) / 343.0, len(ir), fs, 300.0))
    t_first = np.array([np.argmax(fr[:, i] > 0.5 * fr[:, i].max()) for i in range(len(pts))]) / fs
    k = np.polyfit(xs - 1.0, t_first, 1)
    assert abs(1.0 / k[0] - 343.0) < 0.1 * 343.0, 1.0 / k[0]
