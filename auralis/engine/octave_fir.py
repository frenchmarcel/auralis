# -*- coding: utf-8 -*-
"""Minimum-phase FIR filters from octave-band gains (125 Hz - 8 kHz)."""
from __future__ import annotations

import numpy as np

BANDS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
N_TAPS = 4096


def target_db(gains_db, f):
    """Target magnitude (dB) at frequencies f: linear in dB over log2(f) between the octave
    centres, flat beyond the end bands."""
    lf = np.log2(np.maximum(np.asarray(f, dtype=np.float64), 1e-3))
    return np.interp(lf, np.log2(BANDS_HZ), np.asarray(gains_db, dtype=np.float64))


def octave_fir(gains_db, fs, n=N_TAPS, iters=12):
    """Minimum-phase FIR (n taps) whose ENERGY gain in each octave is the requested one
    (to 0.05 dB): the curve's values at the octave centres are corrected a few times, since
    a band's energy also takes in its neighbours' slopes."""
    want = np.asarray(gains_db, dtype=np.float64)
    pts = want.copy()
    for _ in range(iters):
        h = min_phase_fir(pts, fs, n)
        err = want - np.asarray(band_gains_db(h, fs))
        if np.abs(err).max() < 0.05:
            break
        pts = pts + err
    return h


def min_phase_fir(gains_db, fs, n):
    """Minimum-phase FIR (n taps) of the curve target_db, by the real cepstrum on a long
    FFT, ending in a short half-Hann fade."""
    nfft = 16 * n
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    logmag = np.log(10.0 ** (target_db(gains_db, f) / 20.0))
    cep = np.fft.irfft(logmag, nfft)
    fold = np.zeros(nfft)
    fold[0] = cep[0]
    fold[1:nfft // 2] = 2.0 * cep[1:nfft // 2]
    fold[nfft // 2] = cep[nfft // 2]
    h = np.fft.irfft(np.exp(np.fft.rfft(fold)), nfft)[:n]
    fade = n // 8
    h[-fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade)))
    return h


def band_gains_db(h, fs):
    """Energy gain (dB) of the filter in each octave band, for a flat source spectrum."""
    nfft = max(1 << int(np.ceil(np.log2(len(h)))), 1 << 16)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    p = np.abs(np.fft.rfft(h, nfft)) ** 2
    out = []
    for fc in BANDS_HZ:
        s = (f >= fc / 2 ** 0.5) & (f < min(fc * 2 ** 0.5, fs / 2.0))
        out.append(float(10.0 * np.log10(p[s].mean())) if s.any() else 0.0)
    return out


def filter_ir(ir, h):
    """IR (n,) or (n, ch) filtered by h, same length (the decayed end is cut)."""
    a = np.asarray(ir)
    n = len(a) + len(h) - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    H = np.fft.rfft(h, nfft)
    A = np.fft.rfft(a, nfft, axis=0)
    y = np.fft.irfft(A * (H if a.ndim == 1 else H[:, None]), nfft, axis=0)[:len(a)]
    return y.astype(a.dtype, copy=False)
