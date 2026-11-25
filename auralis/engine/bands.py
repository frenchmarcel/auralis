"""Octave bands shared by the whole engine: seven bands from 125 Hz to
8 kHz (usual room-acoustics resolution, ISO 3382)."""
from __future__ import annotations

import numpy as np

#: Octave band centre frequencies (Hz)
CENTERS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
N_BANDS = len(CENTERS)

#: Human-readable labels
LABELS = ["125", "250", "500", "1k", "2k", "4k", "8k"]

_SQRT2 = np.sqrt(2.0)


def band_edges(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Low/high edges of each octave band, stretched to cover the whole
    audible range (22 Hz at the bottom, Nyquist at the top)."""
    lo = CENTERS / _SQRT2
    hi = CENTERS * _SQRT2
    lo = lo.copy()
    hi = hi.copy()
    lo[0] = 22.0
    hi[-1] = max(hi[-1], 0.499 * fs)
    hi = np.minimum(hi, 0.499 * fs)
    return lo, hi


def band_energy_fraction(n_samples: int, fs: int) -> np.ndarray:
    """Share of a UNIT delta's energy falling in each band after masking:
    ~ 2*sum(mask_b^2)/n (Parseval, edges neglected)."""
    masks = band_masks(np.fft.rfftfreq(int(n_samples), 1.0 / fs), fs)
    return 2.0 * (masks ** 2).sum(axis=1) / float(n_samples)


def band_masks(freqs: np.ndarray, fs: float, transition: float = 0.18) -> np.ndarray:
    """Complementary power masks (N_BANDS, len(freqs)) on the given rFFT
    grid; transition = relative width of the ramps."""
    lo, hi = band_edges(fs)
    masks = np.zeros((N_BANDS, len(freqs)))
    f = np.maximum(freqs, 1e-6)
    for b in range(N_BANDS):
        l, h = lo[b], hi[b]
        wl, wh = transition * l, transition * h
        m = np.ones_like(f)
        # low edge ramp (power raised-cosine -> amplitude sqrt)
        below = f < l - wl
        ramp = (f >= l - wl) & (f < l + wl)
        m[below] = 0.0
        m[ramp] = np.sqrt(0.5 * (1 + np.sin(np.pi * (f[ramp] - l) / (2 * wl))))
        # high edge ramp
        above = f > h + wh
        ramp = (f > h - wh) & (f <= h + wh)
        m[above] = 0.0
        m[ramp] *= np.sqrt(0.5 * (1 - np.sin(np.pi * (f[ramp] - h) / (2 * wh))))
        if b == 0:
            m[f <= l] = np.maximum(m[f <= l], (f[f <= l] / l) ** 2)  # gentle LF rolloff
        masks[b] = m
    return masks


def interp_band_gains(freqs: np.ndarray, band_gains: np.ndarray) -> np.ndarray:
    """Interpolate per-band amplitude gains in log frequency, with flat
    extrapolation: (..., N_BANDS) -> (..., len(freqs))."""
    f = np.clip(freqs, 1.0, None)
    lx = np.log10(f)
    lc = np.log10(CENTERS)
    band_gains = np.asarray(band_gains, dtype=np.float64)
    flat = band_gains.reshape(-1, N_BANDS)
    out = np.empty((flat.shape[0], len(freqs)))
    for i in range(flat.shape[0]):
        out[i] = np.interp(lx, lc, flat[i], left=flat[i, 0], right=flat[i, -1])
    return out.reshape(band_gains.shape[:-1] + (len(freqs),))
