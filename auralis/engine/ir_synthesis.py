# -*- coding: utf-8 -*-
"""Impulse-response synthesis: deterministic events as fractional-delay
band FIRs (binaural or mono), stochastic echogram as a shaped noise tail."""

from __future__ import annotations

import numpy as np

from .bands import band_masks, interp_band_gains, band_energy_fraction
from .reflection import event_reflection_spectrum
from .hrtf import (FIR_TAPS, _FIR_NFFT, event_fir_pair, event_fir_stereo,
                   late_tail_binaural, late_tail_stereo,
                   place_fir, receiver_basis)


def _n_samples(n_bins, bin_dt, fs):
    return int(np.ceil(n_bins * bin_dt * fs)) + FIR_TAPS + 8


def _event_signs(events, seed):
    """Per-event +/-1 polarity: +1 for the direct path and order 1,
    random draw (seed-deterministic) beyond."""
    rng = np.random.default_rng((int(seed) ^ 0x5EED) & 0x7FFFFFFF)
    out = []
    for ev in events:
        keep = (getattr(ev, "kind", "") == "direct"
                or getattr(ev, "order", 99) <= 1)
        out.append(1.0 if keep else float(rng.choice((-1.0, 1.0))))
    return out


def synthesize_binaural(events, hist, dirs, bin_dt, fs, speed_of_sound,
                        forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0),
                        seed: int = 0, normalize: bool = True,
                        adm_table=None, hrtf_set=None):
    """Full binaural impulse response; returns (ir (n, 2), info). With
    normalize, peak at 0.985 and info['scale'] = applied factor."""
    nb, n_bins = hist.shape
    n = _n_samples(n_bins, bin_dt, fs)
    L = np.zeros(n)
    R = np.zeros(n)
    _f, _u, right = receiver_basis(forward, up)
    n_taps_ev = (FIR_TAPS if hrtf_set is None
                 else min(2 * int(hrtf_set.n_taps) + 1, 449))
    center = n_taps_ev // 2
    basis = (np.asarray(_f, dtype=np.float64),
             np.asarray(_u, dtype=np.float64),
             np.asarray(right, dtype=np.float64))

    _fir_freqs = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)
    sgn = _event_signs(events, seed)
    for ie, ev in enumerate(events):
        if float(np.max(ev.amps)) <= 0.0:
            continue
        spec = None
        if adm_table is not None and getattr(ev, 'bounces', None):
            spec = event_reflection_spectrum(_fir_freqs,
                                             ev.bounces, adm_table)
        firL, firR, dL, dR = event_fir_pair(ev.amps, ev.direction, right,
                                            fs, speed_of_sound,
                                            n_taps=n_taps_ev, spec_mul=spec,
                                            hrtf_set=hrtf_set, basis=basis)
        place_fir(L, sgn[ie] * firL, (ev.time + dL) * fs - center)
        place_fir(R, sgn[ie] * firR, (ev.time + dR) * fs - center)

    rng = np.random.default_rng(seed)
    tL, tR = late_tail_binaural(hist, dirs, bin_dt, fs, n, right,
                                speed_of_sound, rng,
                                rho=None if hrtf_set is None else hrtf_set.band_coherence())
    if hrtf_set is not None:
        # the events carry the full measured HRTFs; the diffuse tail gets the
        # set's diffuse-field colour and coherence
        dfl, dfr = hrtf_set.diffuse_field_firs()
        tL = np.convolve(tL, dfl)[:n]
        tR = np.convolve(tR, dfr)[:n]
    L += tL
    R += tR

    ir = np.stack([L, R], axis=1)
    scale = 1.0
    peak = float(np.abs(ir).max())
    if normalize and peak > 0.0:
        scale = 0.985 / peak
        ir *= scale
    return ir, {"scale": scale, "peak": peak, "n_samples": n}


def synthesize_stereo(events, hist, dirs, bin_dt, fs, speed_of_sound,
                      forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0),
                      seed: int = 0, normalize: bool = False,
                      adm_table=None, base_deg: float = 30.0):
    """LOUDSPEAKER STEREO impulse response: constant-power amplitude
    panning (base +/-base_deg), no ITD. Returns (ir, info)."""
    nb, n_bins = hist.shape
    n = _n_samples(n_bins, bin_dt, fs)
    L = np.zeros(n)
    R = np.zeros(n)
    _f, _u, right = receiver_basis(forward, up)
    center = FIR_TAPS // 2
    _fir_freqs = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)

    sgn = _event_signs(events, seed)
    for ie, ev in enumerate(events):
        if float(np.max(ev.amps)) <= 0.0:
            continue
        spec = None
        if adm_table is not None and getattr(ev, 'bounces', None):
            spec = event_reflection_spectrum(_fir_freqs,
                                             ev.bounces, adm_table)
        fl, fr, _dl, _dr = event_fir_stereo(ev.amps, ev.direction, right,
                                            fs, base_deg,
                                            spec_mul=spec)
        place_fir(L, sgn[ie] * fl, ev.time * fs - center)
        place_fir(R, sgn[ie] * fr, ev.time * fs - center)

    rng = np.random.default_rng(seed)
    tL, tR = late_tail_stereo(hist, dirs, bin_dt, fs, n, right, base_deg,
                              rng)
    L += tL
    R += tR

    ir = np.stack([L, R], axis=1)
    scale = 1.0
    peak = float(np.abs(ir).max())
    if normalize and peak > 0.0:
        scale = 0.985 / peak
        ir *= scale
    return ir, {"scale": scale, "peak": peak, "n_samples": n}


def synthesize_mono(events, hist, bin_dt, fs, seed: int = 0,
                    normalize: bool = False,
                    adm_table=None):
    """Omnidirectional IR (no head model). Handy for tests and metrics."""
    nb, n_bins = hist.shape
    n = _n_samples(n_bins, bin_dt, fs)
    out = np.zeros(n)
    center = FIR_TAPS // 2
    freqs = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)
    win = np.hanning(FIR_TAPS)

    sgn = _event_signs(events, seed)
    for ie, ev in enumerate(events):
        if float(np.max(ev.amps)) <= 0.0:
            continue
        mag = interp_band_gains(freqs, np.asarray(ev.amps, dtype=np.float64))
        if adm_table is not None and getattr(ev, "bounces", None):
            mag = mag * event_reflection_spectrum(freqs, ev.bounces,
                                                  adm_table)
        h = np.fft.irfft(mag, _FIR_NFFT)
        h = np.roll(h, center)[:FIR_TAPS] * win
        place_fir(out, sgn[ie] * h, ev.time * fs - center)

    rng = np.random.default_rng(seed)
    masks = band_masks(np.fft.rfftfreq(n, 1.0 / fs), fs)
    spb = bin_dt * fs
    edges = np.minimum((np.arange(n_bins + 1) * spb).astype(np.int64), n)
    centers = (np.arange(n_bins) + 0.5) * spb
    for b in range(nb):
        if hist[b].max() <= 0.0:
            continue
        x = rng.standard_normal(n)
        nx = np.fft.irfft(np.fft.rfft(x) * masks[b], n)
        p = nx * nx
        fac_b = float(band_energy_fraction(n, fs)[b])
        gain = np.zeros(n_bins)
        for k in range(n_bins):
            a, bb = edges[k], edges[k + 1]
            if bb <= a:
                continue
            pk = float(p[a:bb].sum())
            if pk > 1e-30 and hist[b, k] > 0.0:
                gain[k] = np.sqrt(hist[b, k] * fac_b / pk)
        g = np.interp(np.arange(n), centers, gain,
                      left=gain[0] if n_bins else 0.0,
                      right=gain[-1] if n_bins else 0.0)
        out += nx * g

    if normalize:
        peak = float(np.abs(out).max())
        if peak > 0:
            out *= 0.985 / peak
    return out


NOISE_NORM_BINS = 41


def sh_sn3d(directions, order: int = 2):
    """Real spherical harmonics, ACN order, SN3D (AmbiX) for unit directions (m, 3)
    (x front, y left, z up): (m, (order + 1)^2)."""
    d = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [y, z, x]
    if order >= 2:
        s3 = np.sqrt(3.0)
        cols += [s3 * x * y, s3 * y * z, 0.5 * (3.0 * z * z - 1.0), s3 * x * z,
                 0.5 * s3 * (x * x - y * y)]
    if order > MAX_AMBI_ORDER:
        raise ValueError("ambisonic order is limited to %d" % MAX_AMBI_ORDER)
    if order > 2:
        cols += _sh_sn3d_high(x, y, z, order)
    return np.stack(cols, axis=1)


MAX_AMBI_ORDER = 5


def _sh_sn3d_high(x, y, z, order):
    """Columns of degrees 3..order (ACN, SN3D, no Condon-Shortley phase): N_lm Q_lm(z)
    Re/Im (x + i y)^|m|, with Q the associated Legendre function divided by
    (1 - z^2)^(|m|/2) (exact at the poles: no division)."""
    from math import factorial
    xy = x + 1j * y
    pw = [np.ones_like(xy)]
    for _m in range(order):
        pw.append(pw[-1] * xy)
    q = {}
    for m in range(order + 1):
        dfact = float(np.prod(np.arange(2 * m - 1, 0, -2))) if m > 0 else 1.0
        q[(m, m)] = np.full_like(z, dfact)
        if m + 1 <= order:
            q[(m + 1, m)] = z * (2 * m + 1) * q[(m, m)]
        for l in range(m + 2, order + 1):
            q[(l, m)] = ((2 * l - 1) * z * q[(l - 1, m)] - (l + m - 1) * q[(l - 2, m)]) / (l - m)
    cols = []
    for l in range(3, order + 1):
        for m in range(-l, l + 1):
            a = abs(m)
            nrm = np.sqrt((2.0 - (a == 0)) * factorial(l - a) / factorial(l + a))
            trig = pw[a].imag if m < 0 else pw[a].real
            cols.append(nrm * q[(l, a)] * trig)
    return cols


def _sh_degree(n_ch):
    return np.floor(np.sqrt(np.arange(n_ch))).astype(int)


def synthesize_ambisonic(events, hist, dirs, bin_dt, fs, order: int = 2,
                         seed: int = 0, adm_table=None):
    """Ambisonic IR (n, (order+1)^2), ACN / SN3D, world axes.

    Events: their mono band FIR times the harmonics of their arrival direction.
    Late field, per bin: the fraction a = |mean direction| (smoothed over 21 bins,
    metrics.smoothed_directions) arrives as one plane wave from the mean direction.
    The rest is isotropic (each degree-l channel carries 1/(2l+1) of W's energy).
    W is energy-matched to the echogram exactly, as in synthesize_mono."""
    nb, n_bins = hist.shape
    n = _n_samples(n_bins, bin_dt, fs)
    n_ch = (order + 1) ** 2
    out = np.zeros((n, n_ch))
    center = FIR_TAPS // 2
    freqs = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)
    win = np.hanning(FIR_TAPS)
    sgn = _event_signs(events, seed)
    for ie, ev in enumerate(events):
        if float(np.max(ev.amps)) <= 0.0:
            continue
        mag = interp_band_gains(freqs, np.asarray(ev.amps, dtype=np.float64))
        if adm_table is not None and getattr(ev, "bounces", None):
            mag = mag * event_reflection_spectrum(freqs, ev.bounces, adm_table)
        h = np.fft.irfft(mag, _FIR_NFFT)
        h = np.roll(h, center)[:FIR_TAPS] * win
        y = sh_sn3d(ev.direction, order)[0]
        for ch in range(n_ch):
            if y[ch] != 0.0:
                place_fir(out[:, ch], sgn[ie] * y[ch] * h, ev.time * fs - center)

    from .metrics import smoothed_directions
    a_bin, unit = smoothed_directions(hist, dirs)
    spb = bin_dt * fs
    sample_bin = np.clip((np.arange(n) / spb).astype(np.int64), 0, n_bins - 1)
    sa = np.sqrt(a_bin)[sample_bin]
    sd = np.sqrt(1.0 - a_bin)[sample_bin]
    y_pw = sh_sn3d(unit, order)[sample_bin]                       # (n, n_ch)
    diff_gain = 1.0 / np.sqrt(2.0 * _sh_degree(n_ch) + 1.0)
    rng = np.random.default_rng(seed)
    # band noises drawn in the frequency domain at a power-of-two length, all the
    # channels of a band in one inverse FFT: the IR length can have large prime
    # factors, which makes FFTs at that length slow
    nfft = 1 << int(np.ceil(np.log2(n)))
    masks = band_masks(np.fft.rfftfreq(nfft, 1.0 / fs), fs)
    edges = np.minimum((np.arange(n_bins + 1) * spb).astype(np.int64), n)
    centers = (np.arange(n_bins) + 0.5) * spb
    fac = band_energy_fraction(n, fs)

    for b in range(nb):
        if hist[b].max() <= 0.0:
            continue
        spec = (rng.standard_normal((n_ch + 1, nfft // 2 + 1))
                + 1j * rng.standard_normal((n_ch + 1, nfft // 2 + 1))) * masks[b]
        noises = np.fft.irfft(spec, nfft, axis=1)[:, :n]
        n_p, n_d = noises[0], noises[1]
        w = sa * n_p + sd * n_d
        p = np.add.reduceat(w * w, np.minimum(edges[:-1], n - 1)) * (edges[1:] > edges[:-1])
        # the noise's energy per bin is smoothed before the match: matching every
        # 1 ms bin to its own (narrow-band, strongly fluctuating) realisation
        # inflates the other channels by E[1/p] > 1/E[p] (measured: order 1 at
        # 0.52 of W instead of 1/3)
        win_k = np.ones(NOISE_NORM_BINS)
        p_s = np.convolve(p, win_k, "same") / np.maximum(
            np.convolve((edges[1:] > edges[:-1]).astype(float), win_k, "same"), 1e-12)
        gain = np.where((p_s > 1e-30) & (hist[b] > 0.0),
                        np.sqrt(hist[b] * float(fac[b]) / np.maximum(p_s, 1e-30)), 0.0)
        g = np.interp(np.arange(n), centers, gain,
                      left=gain[0] if n_bins else 0.0,
                      right=gain[-1] if n_bins else 0.0)
        out[:, 0] += g * w
        for ch in range(1, n_ch):
            out[:, ch] += g * (sa * n_p * y_pw[:, ch] + sd * noises[ch + 1] * diff_gain[ch])
    return out
