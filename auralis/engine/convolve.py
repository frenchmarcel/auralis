# -*- coding: utf-8 -*-
"""FFT convolution of dry signals with (binaural) impulse responses, plus a
few synthetic test signals for auditioning rooms without source material."""

from __future__ import annotations

import numpy as np


def fft_convolve(x, h):
    """Full convolution. x: (n,) or (n, cx); h: (m,) or (m, ch).
    Mono inputs broadcast against multi-channel IRs."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if h.ndim == 1:
        h = h[:, None]
    n, cx = x.shape
    m, ch = h.shape
    co = max(cx, ch)
    if cx not in (1, co) or ch not in (1, co):
        raise ValueError("incompatible channel counts: %d vs %d" % (cx, ch))
    out_len = n + m - 1
    out = np.zeros((out_len, co))

    nfft_full = 1 << int(np.ceil(np.log2(out_len)))
    if nfft_full <= (1 << 23):
        H = np.fft.rfft(h, nfft_full, axis=0)
        X = np.fft.rfft(x, nfft_full, axis=0)
        y = np.fft.irfft(X * H, nfft_full, axis=0)[:out_len]
        out[:] = y if y.shape[1] == co else np.repeat(y, co, axis=1)
        return out

    # overlap-add on x
    nfft = 1 << int(np.ceil(np.log2(4 * m)))
    step = nfft - m + 1
    H = np.fft.rfft(h, nfft, axis=0)
    for i0 in range(0, n, step):
        seg = x[i0:i0 + step]
        Y = np.fft.irfft(np.fft.rfft(seg, nfft, axis=0) * H, nfft, axis=0)
        j1 = min(out_len, i0 + nfft)
        blk = Y[:j1 - i0]
        out[i0:j1] += blk if blk.shape[1] == co else np.repeat(blk, co, axis=1)
    return out


def normalize_peak(x, peak=0.985):
    x = np.asarray(x, dtype=np.float64)
    p = float(np.abs(x).max())
    return x * (peak / p) if p > 0 else x


def make_test_signal(fs: int, kind: str = "clicks", duration: float = 3.0,
                     seed: int = 7):
    """Synthetic dry signal: 'clicks', 'speechish', 'sweep', 'noise_bursts'."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)

    if kind == "clicks":
        x = np.zeros(n)
        for tc in np.arange(0.25, duration - 0.2, 0.5):
            i = int(tc * fs)
            k = np.arange(64)
            x[i:i + 64] += np.exp(-k / 6.0) * np.cos(2 * np.pi * 1800 * k / fs)
        return normalize_peak(x, 0.9)

    if kind == "speechish":
        f0 = 130.0 * (1.0 + 0.12 * np.sin(2 * np.pi * 2.3 * t))
        phase = 2 * np.pi * np.cumsum(f0) / fs
        src = np.sign(np.sin(phase)) * 0.5 + 0.2 * rng.standard_normal(n)
        env = np.clip(np.sin(2 * np.pi * 3.1 * t) + 0.4, 0, None)
        env *= (rng.random(int(duration * 3.1) + 2) > 0.3)[
            np.minimum((t * 3.1).astype(int), int(duration * 3.1))]
        x = src * env
        # crude formant colouring
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / fs)
        shape = (np.exp(-((f - 500) / 350.0) ** 2)
                 + 0.7 * np.exp(-((f - 1500) / 500.0) ** 2)
                 + 0.4 * np.exp(-((f - 2500) / 700.0) ** 2))
        x = np.fft.irfft(X * shape, n)
        return normalize_peak(x, 0.9)

    if kind == "sweep":
        f0, f1 = 60.0, min(16000.0, 0.45 * fs)
        k = (f1 / f0) ** (1.0 / duration)
        phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
        x = np.sin(phase) * np.hanning(n) ** 0.15
        return normalize_peak(x, 0.9)

    if kind == "noise_bursts":
        x = np.zeros(n)
        for tc in np.arange(0.3, duration - 0.3, 0.6):
            i = int(tc * fs)
            m = int(0.12 * fs)
            x[i:i + m] += rng.standard_normal(m) * np.hanning(m)
        return normalize_peak(x, 0.9)

    raise ValueError("unknown kind: %s" % kind)


def resample_linear(x, fs_in, fs_out):
    """Linear resampling (mono or multi-channel, axis 0 = time). Good
    enough for source material, not for precise measurement."""
    x = np.asarray(x, dtype=np.float64)
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    if abs(fs_in - fs_out) < 1e-9 or x.shape[0] == 0:
        return x.copy()
    n_in = x.shape[0]
    n_out = max(1, int(round(n_in * fs_out / fs_in)))
    t_out = np.arange(n_out) * (fs_in / fs_out)
    i0 = np.minimum(t_out.astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    frac = t_out - i0
    if x.ndim == 1:
        return x[i0] * (1.0 - frac) + x[i1] * frac
    return x[i0] * (1.0 - frac)[:, None] + x[i1] * frac[:, None]
