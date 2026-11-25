# -*- coding: utf-8 -*-
"""Octave-band impulse response metrics: band splitting, Schroeder
decay, T20/T30/EDT, C50/C80/D50/Ts, relative levels and windowed event
spectra."""
from __future__ import annotations

import numpy as np

from .bands import CENTERS, LABELS, N_BANDS, band_masks


def band_split(ir, fs: int) -> np.ndarray:
    """(N_BANDS, n): the IR filtered per band (zero-phase masks)."""
    x = np.asarray(ir, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    # one-second zero padding: spectral masking is a CIRCULAR
    # convolution; without margin the narrow filter tail (125 Hz)
    # wraps around at the end of the buffer and floors the EDC
    nfft = int(2 ** np.ceil(np.log2(n + fs)))
    X = np.fft.rfft(x, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    masks = band_masks(freqs, fs)
    out = np.empty((N_BANDS, n))
    for b in range(N_BANDS):
        out[b] = np.fft.irfft(X * masks[b], nfft)[:n]
    return out


def octave_split(ir, fs: int, centers, transition: float = 0.18) -> np.ndarray:
    """(len(centers), n): the IR filtered in octave bands of any centres (e.g. 63 Hz,
    outside the engine's 7 bands), same raised-cosine power masks as band_masks."""
    x = np.asarray(ir, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(n + fs)))
    X = np.fft.rfft(x, nfft)
    f = np.maximum(np.fft.rfftfreq(nfft, 1.0 / fs), 1e-6)
    out = np.empty((len(centers), n))
    for i, fc in enumerate(centers):
        lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
        wl, wh = transition * lo, transition * hi
        m = np.ones_like(f)
        m[f < lo - wl] = 0.0
        r = (f >= lo - wl) & (f < lo + wl)
        m[r] = np.sqrt(0.5 * (1 + np.sin(np.pi * (f[r] - lo) / (2 * wl))))
        m[f > hi + wh] = 0.0
        r = (f > hi - wh) & (f <= hi + wh)
        m[r] *= np.sqrt(0.5 * (1 - np.sin(np.pi * (f[r] - hi) / (2 * wh))))
        out[i] = np.fft.irfft(X * m, nfft)[:n]
    return out


def strength_db(ir, fs: int, centers) -> list:
    """Strength G (dB, ISO 3382-1) per true octave of an IR in the absolute 'pressure re 1 m'
    convention (free field at 1 m = a unit impulse): 10 log10(E / E_impulse) + 20, both
    through the same filters, with one second of silence in front of each (the zero-phase
    filters ring before the onset). Ears (n, 2): their mean energy. None where empty."""
    x = np.asarray(ir, dtype=np.float64)
    cols = [x] if x.ndim == 1 else [x[:, ch] for ch in range(x.shape[1])]
    pad = int(fs)
    e = sum((octave_split(np.concatenate([np.zeros(pad), c]), fs, centers) ** 2).sum(axis=1)
            for c in cols) / len(cols)
    d = np.zeros(pad + len(cols[0]))
    d[pad] = 1.0
    r = (octave_split(d, fs, centers) ** 2).sum(axis=1)
    return [float(10.0 * np.log10(a / b)) + 20.0 if a > 0 and b > 0 else None
            for a, b in zip(e, r)]


def band_metrics_at(ir, fs: int, centers) -> dict:
    """band_metrics on octave bands of the given centres (lists of len(centers))."""
    bands = octave_split(ir, fs, centers)
    k0 = _onset(ir, fs)
    out = {k: [None] * len(centers) for k in ("T20", "T30", "EDT", "C50", "C80", "D50", "Ts")}
    for b, x in enumerate(bands):
        edc = schroeder_db(x)
        out["T20"][b] = _rt_from_edc(edc, fs, -5.0, -25.0)
        out["T30"][b] = _rt_from_edc(edc, fs, -5.0, -35.0)
        out["EDT"][b] = _rt_from_edc(edc, fs, -0.5, -10.0)
        e = x[k0:] ** 2
        et = float(e.sum())
        if et > 0:
            e50 = float(e[:int(round(0.050 * fs))].sum())
            e80 = float(e[:int(round(0.080 * fs))].sum())
            out["C50"][b] = float(10.0 * np.log10(max(e50, 1e-30) / max(et - e50, 1e-30)))
            out["C80"][b] = float(10.0 * np.log10(max(e80, 1e-30) / max(et - e80, 1e-30)))
            out["D50"][b] = float(e50 / et)
            out["Ts"][b] = float(1000.0 * ((np.arange(len(e)) / fs) * e).sum() / et)
    return out


def schroeder_db(x) -> np.ndarray:
    """Schroeder decay in dB (0 dB at the start)."""
    e = np.asarray(x, dtype=np.float64) ** 2
    tail = np.cumsum(e[::-1])[::-1]
    tot = tail[0] if tail[0] > 0 else 1.0
    return 10.0 * np.log10(np.maximum(tail / tot, 1e-30))


def _rt_from_edc(edc_db: np.ndarray, fs: int, lo: float, hi: float):
    """Reverberation time extrapolated to -60 dB from the slope between
    lo and hi dB (negative). None if the dynamic range is too small."""
    idx = np.nonzero((edc_db <= lo) & (edc_db >= hi))[0]
    if len(idx) < max(8, int(0.010 * fs)):
        return None
    t = idx / float(fs)
    y = edc_db[idx]
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    if slope >= -1e-9:
        return None
    return float(-60.0 / slope)


def _onset(ir, fs: int) -> int:
    """Direct sound arrival index: first crossing above -20 dB of the
    peak, on the broadband IR."""
    x = np.abs(np.asarray(ir, dtype=np.float64))
    if x.ndim > 1:
        x = x.mean(axis=1)
    pk = float(x.max())
    idx = np.nonzero(x > 0.1 * pk)[0]
    return int(idx[0]) if len(idx) else 0


def band_metrics(ir, fs: int) -> dict:
    """ISO 3382 metrics per octave.

    T20/T30/EDT (s or None), C50/C80 (dB), D50, Ts (ms), level (dB re the sum of
    the 7 octaves). C/D/Ts are taken from the direct sound. True octaves
    (octave_split), not the engine's synthesis bands, which stretch to 22 Hz and
    to Nyquist: on a measured RIR, 22-88 Hz noise would stall the 125 Hz decay."""
    bands = octave_split(ir, fs, list(CENTERS))
    e_tot = float(np.sum(bands ** 2))
    k0 = _onset(ir, fs)
    k50 = k0 + int(round(0.050 * fs))
    k80 = k0 + int(round(0.080 * fs))
    out = {"T20": [None] * N_BANDS, "T30": [None] * N_BANDS,
           "EDT": [None] * N_BANDS, "C50": [None] * N_BANDS,
           "C80": [None] * N_BANDS, "D50": [None] * N_BANDS,
           "Ts": [None] * N_BANDS, "level_db": np.zeros(N_BANDS)}
    for b in range(N_BANDS):
        x = bands[b]
        edc = schroeder_db(x)
        out["T20"][b] = _rt_from_edc(edc, fs, -5.0, -25.0)
        out["T30"][b] = _rt_from_edc(edc, fs, -5.0, -35.0)
        out["EDT"][b] = _rt_from_edc(edc, fs, -0.5, -10.0)
        e = x[k0:] ** 2
        et = float(e.sum())
        if et > 0:
            e50 = float(e[:k50 - k0].sum())
            e80 = float(e[:k80 - k0].sum())
            tard50 = max(et - e50, 1e-30)
            tard80 = max(et - e80, 1e-30)
            out["C50"][b] = float(10.0 * np.log10(
                max(e50, 1e-30) / tard50))
            out["C80"][b] = float(10.0 * np.log10(
                max(e80, 1e-30) / tard80))
            out["D50"][b] = float(e50 / et)
            t = np.arange(len(e)) / fs
            out["Ts"][b] = float(1000.0 * (t * e).sum() / et)
        e_b = float(np.sum(x ** 2))
        out["level_db"][b] = 10.0 * np.log10(
            max(e_b, 1e-30) / max(e_tot, 1e-30))
    return out


def event_spectrum(ir, fs: int, t_center: float, half_width: float,
                   n_fft: int = 1 << 14):
    """Complex spectrum of a windowed (Hanning) event around
    t_center +/- half_width. Returns (freqs, spectrum)."""
    x = np.asarray(ir, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    k = int(round(t_center * fs))
    w = max(4, int(round(half_width * fs)))
    a, b = max(0, k - w), min(len(x), k + w)
    seg = np.zeros(2 * w)
    seg[a - (k - w):b - (k - w)] = x[a:b]
    seg *= np.hanning(2 * w)
    F = np.fft.rfft(seg, n_fft)
    return np.fft.rfftfreq(n_fft, 1.0 / fs), F


def compare_metrics(m_sim: dict, m_ref: dict) -> dict:
    """Per-band sim - ref gaps: dT20, dT30, dEDT (s, None if either
    one is missing) and dLevel (dB)."""
    out = {}
    for key in ("T20", "T30", "EDT", "C50", "C80", "D50",
                "Ts"):
        out["d" + key] = [
            (None if (a is None or b is None) else float(a - b))
            for a, b in zip(m_sim[key], m_ref[key])]
    out["dLevel"] = (np.asarray(m_sim["level_db"])
                     - np.asarray(m_ref["level_db"]))
    return out


def _fmt(v, unit="s"):
    if v is None:
        return "   --  "
    return "%7.3f" % v


def format_report(label: str, m_sim: dict, m_ref: dict = None,
                  deltas: dict = None) -> str:
    """Text table per band (columns = octaves)."""
    lines = ["[%s]" % label,
             "band    : " + " ".join("%7s" % b for b in LABELS)]
    for key in ("T20", "T30", "EDT"):
        lines.append("%s sim : " % key.ljust(4)
                     + " ".join(_fmt(v) for v in m_sim[key]))
        if m_ref is not None:
            lines.append("%s ref : " % key.ljust(4)
                         + " ".join(_fmt(v) for v in m_ref[key]))
    for key in ("C50", "C80", "D50", "Ts"):
        lines.append("%s sim : " % key.ljust(4)
                     + " ".join(_fmt(v) for v in m_sim[key]))
        if m_ref is not None:
            lines.append("%s ref : " % key.ljust(4)
                         + " ".join(_fmt(v) for v in m_ref[key]))
    def _d80(c):
        return [None if v is None
                else 1.0 / (1.0 + 10.0 ** (-v / 10.0)) for v in c]
    lines.append("D80 sim : "
                 + " ".join(_fmt(v) for v in _d80(m_sim["C80"])))
    if m_ref is not None:
        lines.append("D80 ref : "
                     + " ".join(_fmt(v) for v in _d80(m_ref["C80"])))
    lines.append("Lvl sim : " + " ".join("%7.2f" % v
                                         for v in m_sim["level_db"]))
    if m_ref is not None:
        lines.append("Lvl ref : " + " ".join("%7.2f" % v
                                             for v in m_ref["level_db"]))
    if deltas is not None:
        lines.append("dT30    : " + " ".join(_fmt(v)
                                             for v in deltas["dT30"]))
        lines.append("dC50    : " + " ".join(_fmt(v)
                                             for v in deltas["dC50"]))
        lines.append("dTs     : " + " ".join(_fmt(v)
                                             for v in deltas["dTs"]))
        lines.append("dLvl    : " + " ".join("%+7.2f" % v
                                             for v in deltas["dLevel"]))
    return "\n".join(lines)


def event_spectrum_asym(ir, fs: int, t0: float, pre: float,
                        post: float, n_fft: int = 1 << 14):
    """Spectrum of an event, ASYMMETRIC window: cosine fade-in over
    `pre` ending at t0, fade-out over `post`."""
    x = np.asarray(ir, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    npre = max(4, int(round(pre * fs)))
    npost = max(8, int(round(post * fs)))
    k = int(round(t0 * fs))
    a, b = k - npre, k + npost
    seg = np.zeros(npre + npost)
    lo, hi = max(0, a), min(len(x), b)
    seg[lo - a:hi - a] = x[lo:hi]
    w = np.ones(npre + npost)
    w[:npre] = np.sin(0.5 * np.pi * np.arange(npre) / npre) ** 2
    w[npre:] = np.cos(0.5 * np.pi * np.arange(npost) / npost) ** 2
    seg *= w
    F = np.fft.rfft(seg, n_fft)
    return np.fft.rfftfreq(n_fft, 1.0 / fs), F


def event_band_mags(ir, fs: int, t_center: float, half_width: float,
                    pre: float = None, post: float = None):
    """Mean spectral magnitude per octave of a windowed event (Hann
    +/- half_width, or asymmetric if pre/post). None if unresolved."""
    if pre is not None:
        f, F = event_spectrum_asym(ir, fs, t_center, pre, post)
        fmin_ok = 1.2 / (0.5 * (pre + post))
    else:
        f, F = event_spectrum(ir, fs, t_center, half_width)
        fmin_ok = 1.2 / half_width
    out = [None] * N_BANDS
    for b in range(N_BANDS):
        lo, hi = CENTERS[b] / np.sqrt(2.0), CENTERS[b] * np.sqrt(2.0)
        if CENTERS[b] < fmin_ok or hi > 0.5 * fs:
            continue
        m = (f >= lo) & (f < hi)
        if m.any():
            out[b] = float(np.abs(F[m]).mean())
    return out


def reflection_bands(ir, fs: int, t_direct: float, t_reflect: float,
                     r_direct: float, r_reflect: float):
    """ABSOLUTE |R| per octave of an isolated reflection, OMNI source:
    |R| = |F_refl / F_direct| * r_reflect / r_direct."""
    gap = float(t_reflect - t_direct)
    hw = max(0.35 * gap, 8.0 / fs)
    Fd = event_band_mags(ir, fs, t_direct, hw)
    Fr = event_band_mags(ir, fs, t_reflect, hw)
    spread = float(r_reflect) / max(float(r_direct), 1e-9)
    return [None if (a is None or d in (None, 0.0))
            else float(a / d * spread)
            for a, d in zip(Fr, Fd)]


def reflection_bands_relative(ir, ir_rigid, fs: int, t_direct: float,
                              t_reflect: float, rigid_R: float = 0.99):
    """|R| per octave RELATIVE to a rigid measurement of the SAME
    geometry: |R| = |F_refl / F_refl_rigid| * rigid_R."""
    gap = float(t_reflect - t_direct)
    # ASYMMETRIC window: short fade-in (bounded by the direct /
    # reflection gap) to cut the skirt of the direct sound -- far
    # louder than an absorbed reflection --, 1.8 ms fade-out for the
    # scattered tail (BRAS protocol)
    pre = min(0.6e-3, max(0.45 * gap, 4.0 / fs))
    post = 1.8e-3
    Fr = event_band_mags(ir, fs, t_reflect, None, pre=pre, post=post)
    F0 = event_band_mags(ir_rigid, fs, t_reflect, None, pre=pre,
                         post=post)
    return [None if (a is None or d in (None, 0.0))
            else float(a / d * rigid_R)
            for a, d in zip(Fr, F0)]
