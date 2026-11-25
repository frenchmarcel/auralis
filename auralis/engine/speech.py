# -*- coding: utf-8 -*-
"""Speech Transmission Index, indirect method (IEC 60268-16:2020, edition 5):
MTF from the band-filtered impulse response (or its energy envelope), then
noise, level-dependent auditory masking and the reception threshold."""
from __future__ import annotations

import numpy as np

from .bands import N_BANDS

MOD_FREQS = 10.0 ** (np.arange(-2, 12) / 10.0)          # 0.63 ... 12.5 Hz (14)
ALPHA = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])   # Table A.1, male
BETA = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
RECEPTION_THRESHOLD_DB = np.array([46.0, 27.0, 12.0, 6.5, 7.5, 8.0, 12.0])   # Table A.3
# standardised male speech, octave levels relative to the A-weighted level (A.6.1)
MALE_SPEECH_REL_DB = np.array([-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0])
VOCAL_EFFORT_DBA_1M = {"normal": 60.0, "raised": 66.0}

# NC curves (ANSI S12.2), octave levels 125 Hz .. 8 kHz (the 63 Hz value dropped)
NC_CURVES = {
    15: [36, 29, 22, 17, 14, 12, 11], 20: [40, 33, 26, 22, 19, 17, 16],
    25: [44, 37, 31, 27, 24, 22, 21], 30: [48, 41, 35, 31, 29, 28, 27],
    35: [52, 45, 40, 36, 34, 33, 32], 40: [56, 50, 45, 41, 39, 38, 37],
    45: [60, 54, 49, 46, 44, 43, 42], 50: [64, 58, 54, 51, 49, 48, 47],
    55: [67, 62, 58, 56, 54, 53, 52], 60: [71, 67, 63, 61, 59, 58, 57],
    65: [75, 71, 68, 66, 64, 63, 62], 70: [79, 75, 72, 71, 70, 69, 68],
}


def mtf_from_envelope(energy, dt):
    """(N_BANDS, 14) MTF of band energy envelopes energy (N_BANDS, n) sampled at dt."""
    e = np.asarray(energy, dtype=np.float64)
    t = np.arange(e.shape[-1]) * dt
    ph = np.exp(-2j * np.pi * np.outer(MOD_FREQS, t))           # (14, n)
    num = np.abs(e @ ph.T)
    den = e.sum(axis=-1, keepdims=True)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def mtf_from_ir(ir, fs):
    """MTF of a pressure IR (mono, or binaural averaged) per octave band."""
    from .rirmetrics import band_split
    h = band_split(ir, fs)
    return mtf_from_envelope(h * h, 1.0 / fs)


def masking_level_db(level_prev):
    """Level-dependent masking slope L_a,k from the level of band k-1 (Table A.2)."""
    L = np.asarray(level_prev, dtype=np.float64)
    return np.select([L < 63.0, L < 67.0, L < 100.0],
                     [0.5 * L - 65.0, 1.8 * L - 146.9, 0.5 * L - 59.8], -10.0)


def sti(mtf, signal_db=None, noise_db=None):
    """STI from an MTF (N_BANDS, 14). With per-band speech levels at the listener
    (signal_db) the auditory masking and reception threshold apply, and noise_db
    adds the background noise; without levels the MTF alone is rated.
    Returns {"sti", "mti" (N_BANDS,), "mtf_corrected"}."""
    m = np.asarray(mtf, dtype=np.float64).copy()
    if signal_db is not None:
        S = np.asarray(signal_db, dtype=np.float64)
        N = (np.full(N_BANDS, -np.inf) if noise_db is None
             else np.asarray(noise_db, dtype=np.float64))
        Is, In = 10.0 ** (S / 10.0), 10.0 ** (N / 10.0)
        m *= (Is / (Is + In))[:, None]                     # = 1 / (1 + 10^(-SNR/10))
        I = Is + In
        L = 10.0 * np.log10(I)
        I_am = np.zeros(N_BANDS)
        I_am[1:] = I[:-1] * 10.0 ** (masking_level_db(L[:-1]) / 10.0)
        I_rt = 10.0 ** (RECEPTION_THRESHOLD_DB / 10.0)
        m *= (I / (I + I_am + I_rt))[:, None]
    m = np.clip(m, 1e-12, 1.0 - 1e-12)
    snr = np.clip(10.0 * np.log10(m / (1.0 - m)), -15.0, 15.0)
    ti = (snr + 15.0) / 30.0
    mti = ti.mean(axis=1)
    value = float(np.dot(ALPHA, mti) - np.dot(BETA, np.sqrt(mti[:-1] * mti[1:])))
    return {"sti": float(np.clip(value, 0.0, 1.0)), "mti": mti, "mtf_corrected": m}


def speech_levels_1m(effort="normal"):
    """Octave speech levels (dB SPL at 1 m, on axis) for a vocal effort."""
    return VOCAL_EFFORT_DBA_1M[effort] + MALE_SPEECH_REL_DB


def noise_levels(spec):
    """Background noise octave levels: None / "none", "NC-35", or 7 values (dB)."""
    if spec is None or spec == "none":
        return None
    if isinstance(spec, str):
        k = int(spec.upper().replace("NC-", "").replace("NC", ""))
        if k not in NC_CURVES:
            raise ValueError("unknown noise curve '%s' (NC-15 ... NC-70)" % spec)
        return np.asarray(NC_CURVES[k], dtype=np.float64)
    v = np.asarray(spec, dtype=np.float64).reshape(-1)
    if len(v) != N_BANDS:
        raise ValueError("background noise needs %d octave levels (125 Hz - 8 kHz)" % N_BANDS)
    return v
