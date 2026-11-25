# -*- coding: utf-8 -*-
"""Parametric binaural rendering (HRTF-lite): linear-phase FIR per ear
(band amplitudes, Brown-Duda head shadow, Woodworth ITD); stochastic
tail from band noises with diffuse-field coherence."""

from __future__ import annotations

import numpy as np

from .bands import CENTERS, band_masks, interp_band_gains, band_energy_fraction

HEAD_RADIUS = 0.0875     # m (Woodworth / Duda average head)
FIR_TAPS = 129
_FIR_NFFT = 512


def receiver_basis(forward, up):
    """Orthonormal (forward, up, right) with right = forward x up."""
    f = np.asarray(forward, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    f = f / max(np.linalg.norm(f), 1e-12)
    r = np.cross(f, u)
    rn = np.linalg.norm(r)
    if rn < 1e-9:                      # forward // up: pick any right
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        rn = np.linalg.norm(r)
        if rn < 1e-9:
            r = np.array([1.0, 0.0, 0.0])
            rn = 1.0
    r /= rn
    u2 = np.cross(r, f)
    return f, u2, r


def itd_seconds(direction, right, speed_of_sound):
    """Far-field Woodworth ITD (unit direction receiver->source). Returns
    (tau, near_is_right); tau >= 0 is the delay of the far ear."""
    x = float(np.clip(np.dot(direction, right), -1.0, 1.0))
    gamma = abs(float(np.arcsin(x)))
    tau = HEAD_RADIUS / speed_of_sound * (gamma + np.sin(gamma))
    return tau, x >= 0.0


def _shadow_magnitude(freqs, cos_theta_ear, speed_of_sound):
    """Magnitude of the spherical head shadow (Brown-Duda):
    H(w) = (alpha*jw + beta) / (jw + beta), beta = 2c/a, alpha = 1+0.9*cos."""
    beta = 2.0 * speed_of_sound / HEAD_RADIUS
    alpha = 1.0 + 0.9 * float(np.clip(cos_theta_ear, -1.0, 1.0))
    w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    return np.sqrt(beta * beta + (alpha * w) ** 2) / np.sqrt(beta * beta + w * w)


def event_fir_pair(band_amps, direction, right, fs, speed_of_sound,
                   n_taps: int = FIR_TAPS, spec_mul=None,
                   hrtf_set=None, basis=None):
    """(fir_left, fir_right, extra_delay_left_s, extra_delay_right_s); linear
    phase unless HRTFs are measured, the caller removes n_taps//2 of delay."""
    freqs = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)
    base = interp_band_gains(freqs, np.asarray(band_amps, dtype=np.float64))
    if spec_mul is not None:
        base = base * spec_mul   # complex reflection R(theta, f) (phase)
    if hrtf_set is not None and basis is not None:
        # measured HRTFs: aligned ear responses + separate delays.
        # No full window (it would crush the causal onset): short head
        # and tail ramps only, content rolled to n_taps//2 to honour
        # the placement contract of the caller.
        HL, HR, dl, dr = hrtf_set.ear_responses(direction, basis,
                                                _FIR_NFFT)
        firs = []
        for Hs in (base * HL, base * HR):
            hh = np.fft.irfft(Hs, _FIR_NFFT)
            hh = np.roll(hh, n_taps // 2)[:n_taps].copy()
            r0 = min(24, n_taps // 8)
            hh[:r0] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, r0)))
            r1 = min(48, n_taps // 6)
            hh[-r1:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, r1)))
            firs.append(hh)
        return firs[0], firs[1], dl, dr
    cosL = float(np.dot(direction, -right))
    cosR = float(np.dot(direction, right))
    magL = base * _shadow_magnitude(freqs, cosL, speed_of_sound)
    magR = base * _shadow_magnitude(freqs, cosR, speed_of_sound)

    win = np.hanning(n_taps)
    firs = []
    for mag in (magL, magR):
        h = np.fft.irfft(mag, _FIR_NFFT)
        h = np.roll(h, n_taps // 2)[:n_taps] * win
        firs.append(h)

    tau, near_right = itd_seconds(direction, right, speed_of_sound)
    dL = 0.0 if not near_right else tau
    dR = 0.0 if near_right else tau
    return firs[0], firs[1], dL, dR


_PAD = 8          # samples of fractional-shift ringing kept on each side


def place_fir(buf, fir, pos_samples: float):
    """Add `fir` into `buf` at a fractional position, clipped to the buffer.

    The fraction is an exact band-limited delay (phase ramp): a two-tap linear
    interpolation would low-pass the events, up to -7.7 dB at 16 kHz for half
    a sample at 44.1 kHz. Integer positions are unchanged."""
    i0 = int(np.floor(pos_samples))
    frac = pos_samples - i0
    if frac > 1e-9:
        n = len(fir)
        nfft = 1 << int(np.ceil(np.log2(n + 4 * _PAD)))
        F = np.fft.rfft(fir, nfft)
        F *= np.exp(-2j * np.pi * np.fft.rfftfreq(nfft) * frac)
        sh = np.fft.irfft(F, nfft)
        fir = np.concatenate([sh[-_PAD:], sh[:n + _PAD]])
        i0 -= _PAD
    a = max(0, i0)
    b = min(len(buf), i0 + len(fir))
    if b > a:
        buf[a:b] += fir[a - i0:b - i0]


def interaural_coherence(speed_of_sound):
    """Diffuse-field coherence per band: rho = sinc(2 f d / c), d = 18 cm."""
    d_ears = 2.0 * HEAD_RADIUS   # ~0.175 m effective path
    rho = np.sinc(2.0 * np.asarray(CENTERS, dtype=np.float64) * d_ears
                  / speed_of_sound)
    return np.clip(rho, -0.95, 0.95)


def pan_gains_stereo(lateral_left, base_deg):
    """Constant-power gains for a +/-base_deg stereo base. lateral_left
    lies in [-1 (right) .. +1 (left)]; centre: gL = gR = 1."""
    s = max(np.sin(np.radians(base_deg)), 1e-3)
    p = float(np.clip(lateral_left / s, -1.0, 1.0))
    th = (1.0 - p) * (np.pi / 4.0)
    return np.sqrt(2.0) * np.cos(th), np.sqrt(2.0) * np.sin(th)


def event_fir_stereo(band_amps, direction, right, fs, base_deg,
                     n_taps: int = FIR_TAPS, spec_mul=None):
    """Single FIR (smooth event spectrum, reflection phase included)
    spread over two loudspeakers by amplitude panning."""
    freqs = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)
    base = interp_band_gains(freqs, np.asarray(band_amps,
                                               dtype=np.float64))
    if spec_mul is not None:
        base = base * spec_mul
    win = np.hanning(n_taps)
    hh = np.roll(np.fft.irfft(base, _FIR_NFFT), n_taps // 2)[:n_taps] * win
    gL, gR = pan_gains_stereo(-float(np.dot(direction, right)), base_deg)
    return hh * gL, hh * gR, 0.0, 0.0


def late_tail_stereo(hist, dirs, bin_dt, fs, n_samples, right, base_deg,
                     rng):
    """Stochastic tail for loudspeakers: two INDEPENDENT noises per band,
    energy matched per bin, panned by the arrival vector."""
    nb, n_bins = hist.shape
    masks = band_masks(np.fft.rfftfreq(n_samples, 1.0 / fs), fs)

    e_bin = hist.sum(axis=0)
    dnorm = np.linalg.norm(dirs, axis=1)
    unit = np.zeros_like(dirs)
    ok = dnorm > 1e-12
    unit[ok] = dirs[ok] / dnorm[ok, None]
    lat = -(unit @ np.asarray(right, dtype=np.float64))
    s = max(np.sin(np.radians(base_deg)), 1e-3)
    p = np.clip(lat / s, -1.0, 1.0)
    th = (1.0 - p) * (np.pi / 4.0)
    wL_bin = np.cos(th) ** 2
    wR_bin = np.sin(th) ** 2

    spb = bin_dt * fs
    edges = np.minimum((np.arange(n_bins + 1) * spb).astype(np.int64),
                       n_samples)
    sample_bin = np.clip((np.arange(n_samples) / spb).astype(np.int64),
                         0, n_bins - 1)
    ampL = np.sqrt(np.maximum(2.0 * wL_bin, 0.0))[sample_bin]
    ampR = np.sqrt(np.maximum(2.0 * wR_bin, 0.0))[sample_bin]

    outL = np.zeros(n_samples)
    outR = np.zeros(n_samples)
    for b in range(nb):
        if hist[b].max() <= 0.0:
            continue
        nl = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_samples))
                          * masks[b], n_samples)
        nr = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_samples))
                          * masks[b], n_samples)
        p_ref = 0.5 * (nl * nl + nr * nr)
        fac_b = float(band_energy_fraction(n_samples, fs)[b])
        gain = np.zeros(n_bins)
        for k in range(n_bins):
            a, bb = edges[k], edges[k + 1]
            if bb <= a:
                continue
            pw = float(p_ref[a:bb].sum())
            if pw > 1e-30 and hist[b, k] > 0.0:
                gain[k] = np.sqrt(hist[b, k] * fac_b / pw)
        centers = (np.arange(n_bins) + 0.5) * spb
        g = np.interp(np.arange(n_samples), centers, gain,
                      left=gain[0] if n_bins else 0.0,
                      right=gain[-1] if n_bins else 0.0)
        outL += nl * g * ampL
        outR += nr * g * ampR
    return outL, outR


def late_tail_binaural(hist, dirs, bin_dt, fs, n_samples, right,
                       speed_of_sound, rng, rho=None):
    """Stochastic echogram (hist (NB, n_bins), dirs (n_bins, 3) toward
    arrival) rendered as a binaural tail: (left, right) of n_samples.
    rho: interaural coherence per band (default: two points in free field)."""
    nb, n_bins = hist.shape
    masks = band_masks(np.fft.rfftfreq(n_samples, 1.0 / fs), fs)
    if rho is None:
        rho = interaural_coherence(speed_of_sound)

    # per-bin lateralisation weights from the mean direction of arrival
    e_bin = hist.sum(axis=0)
    dnorm = np.linalg.norm(dirs, axis=1)
    unit = np.zeros_like(dirs)
    ok = dnorm > 1e-12
    unit[ok] = dirs[ok] / dnorm[ok, None]
    proj = unit @ np.asarray(right, dtype=np.float64)
    wR_bin = 0.5 * (1.0 + 0.85 * proj)
    wL_bin = 1.0 - wR_bin

    spb = bin_dt * fs                       # samples per bin (float)
    edges = np.minimum((np.arange(n_bins + 1) * spb).astype(np.int64),
                       n_samples)
    sample_bin = np.clip((np.arange(n_samples) / spb).astype(np.int64),
                         0, n_bins - 1)
    ampL = np.sqrt(np.maximum(2.0 * wL_bin, 0.0))[sample_bin]
    ampR = np.sqrt(np.maximum(2.0 * wR_bin, 0.0))[sample_bin]

    outL = np.zeros(n_samples)
    outR = np.zeros(n_samples)
    for b in range(nb):
        if hist[b].max() <= 0.0:
            continue
        x = rng.standard_normal(n_samples)
        y = rng.standard_normal(n_samples)
        nl = np.fft.irfft(np.fft.rfft(x) * masks[b], n_samples)
        ny = np.fft.irfft(np.fft.rfft(y) * masks[b], n_samples)
        nr = rho[b] * nl + np.sqrt(max(0.0, 1.0 - rho[b] ** 2)) * ny
        fac_b = float(band_energy_fraction(n_samples, fs)[b])

        # exact per-bin energy match on the mean of the two ears
        p_ref = 0.5 * (nl * nl + nr * nr)
        gain = np.zeros(n_bins)
        for k in range(n_bins):
            a, bb = edges[k], edges[k + 1]
            if bb <= a:
                continue
            p = float(p_ref[a:bb].sum())
            if p > 1e-30 and hist[b, k] > 0.0:
                # integrated energy -> event units (see bands)
                gain[k] = np.sqrt(hist[b, k] * fac_b / p)
        # smooth bin-to-bin gain over samples (linear interp at bin centres)
        centers = (np.arange(n_bins) + 0.5) * spb
        g = np.interp(np.arange(n_samples), centers, gain,
                      left=gain[0] if n_bins else 0.0,
                      right=gain[-1] if n_bins else 0.0)
        outL += nl * g * ampL
        outR += nr * g * ampR
    return outL, outR
