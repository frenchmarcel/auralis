# -*- coding: utf-8 -*-
"""Binaural decoding of ambisonics (order <= 5, ACN/SN3D): virtual loudspeakers on a
near-uniform sphere, dual-band decoder (mode matching below the crossover, max-rE
energy-normalised above), HRIRs of the engine's head model (or a measured set);
rotation matrices for head tracking."""
from __future__ import annotations

import numpy as np

from .hrtf import FIR_TAPS, HEAD_RADIUS, event_fir_pair, place_fir
from .ir_synthesis import sh_sn3d

N_VIRTUAL = 50
CROSSOVER_HZ = 700.0
# max-rE weights per degree, 3D: g_n = P_n(cos(137.9 deg / (N + 1.51))) (Zotter & Frank, Ambisonics, 2019)
MAX_RE = {1: [1.0, 0.5774], 2: [1.0, 0.7746, 0.4]}


def max_re(order):
    """max-rE weights per degree (the tables above for orders 1-2, the formula beyond)."""
    if order in MAX_RE:
        return MAX_RE[order]
    t = np.cos(np.radians(137.9 / (order + 1.51)))
    return [float(np.polynomial.legendre.legval(t, [0] * n + [1])) for n in range(order + 1)]


def n_virtual(order):
    """Virtual loudspeakers of the decoder: 50 up to order 2, ~3 (N + 1)^2 beyond."""
    return N_VIRTUAL if order <= 2 else 3 * (order + 1) ** 2


def virtual_layout(n=N_VIRTUAL):
    """Fibonacci sphere: n near-uniform unit directions (n, 3)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (1.0 + 5 ** 0.5) * k
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _degrees(n_ch):
    return np.floor(np.sqrt(np.arange(n_ch))).astype(int)


def decoder_matrices(order, dirs=None):
    """(basic, max-rE) decoding matrices (n_spk, n_ch): speaker gains = D @ ambi."""
    dirs = virtual_layout(n_virtual(order)) if dirs is None else dirs
    Y = sh_sn3d(dirs, order)                                   # (S, n_ch)
    basic = np.linalg.pinv(Y.T)                                # mode matching
    deg = _degrees(Y.shape[1])
    g = np.asarray(max_re(order))[deg]
    return basic, basic * g[None, :]


def hrirs(dirs, fs, speed_of_sound=343.0, n_taps=256, hrtf_set=None):
    """(S, 2, n_taps) head-related IRs, head looking along +x, z up, with interaural
    delays; common latency n_taps // 4 + FIR_TAPS // 2 samples."""
    fwd, up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    lat = n_taps // 4
    out = np.zeros((len(dirs), 2, n_taps))
    n_ev = FIR_TAPS if hrtf_set is None else min(2 * int(hrtf_set.n_taps) + 1, 449)
    for s, d in enumerate(dirs):
        fl, fr, dl, dr = event_fir_pair(np.ones(7), d, right, fs, speed_of_sound,
                                        n_taps=n_ev, hrtf_set=hrtf_set,
                                        basis=(fwd, up, right))
        if hrtf_set is None:
            # the model delays the FAR ear only: each ear's delay has a kink across the median
            # plane, which no low-order projection follows (order 2: level differences of the
            # wrong sign below 700 Hz). Referred to the head centre (Woodworth: near ear
            # -a/c sin g, far ear +a/c g), the same ITD is smooth.
            lead = HEAD_RADIUS / speed_of_sound * abs(float(np.clip(np.dot(d, right), -1, 1)))
            dl, dr = dl - lead, dr - lead
        place_fir(out[s, 0], fl, lat + dl * fs)
        place_fir(out[s, 1], fr, lat + dr * fs)
    return out, lat + n_ev // 2


def magls_cutoff(order, speed_of_sound=343.0):
    """Frequency (Hz) above which order N cannot follow the interaural phase: N c / (2 pi a)."""
    return order * speed_of_sound / (2.0 * np.pi * HEAD_RADIUS)


def _magls_spectra(order, h, fs, dirs, f_cut, latency=0.0):
    """(n_ch, 2, F) decoder spectra fitted on the HRTFs h (S, 2, n) of the directions dirs.

    Complex least squares below f_cut, magnitude least squares above (the phase of the
    previous bin's solution, iterated upwards: Zaunschirm, Schoerkhuber and Hoeldrich,
    JASA 2018). The common latency (samples) is taken out first. Carried along, its linear
    phase would freeze at its cutoff value and the high band would land near t = 0."""
    Y = sh_sn3d(dirs, order)                                   # (S, C)
    P = np.linalg.pinv(Y)                                      # (C, S)
    freqs = np.fft.rfftfreq(h.shape[-1], 1.0 / fs)
    lin = np.exp(2j * np.pi * freqs * float(latency) / fs)     # e^{+j w lat}
    H = np.fft.rfft(h, axis=-1) * lin                          # (S, 2, F), latency removed
    D = np.empty((Y.shape[1], 2, len(freqs)), dtype=complex)
    for k, f in enumerate(freqs):
        if f < f_cut or k == 0:
            D[:, :, k] = P @ H[:, :, k]
        else:
            ph = np.angle(Y @ D[:, :, k - 1])
            D[:, :, k] = P @ (np.abs(H[:, :, k]) * np.exp(1j * ph))
    return D / lin[None, None, :]


def decoder_filters(order, fs, speed_of_sound=343.0, n_taps=256, hrtf_set=None,
                    crossover_hz=CROSSOVER_HZ, method="dual"):
    """(n_ch, 2, n_taps) binaural filters: ear_e = sum_ch filters[ch, e] * ambi_ch, plus
    the latency (samples) they add. method "dual": virtual loudspeakers, mode matching below
    crossover_hz and max-rE above; "magls": fitted on the HRTFs, magnitude least squares above
    magls_cutoff (the interaural phase an order cannot follow is given up for the magnitudes)."""
    if hrtf_set is not None:
        # a measured event FIR is ~2 x its taps, centred: room for it after the latency
        n_ev = min(2 * int(hrtf_set.n_taps) + 1, 449)
        n_taps = max(n_taps, 1 << int(np.ceil(np.log2(n_ev + 2 * (n_taps // 4) + 16))))
    if method not in ("dual", "magls"):
        raise ValueError("decoder method '%s': dual or magls" % method)
    dirs = virtual_layout(n_virtual(order))
    h, latency = hrirs(dirs, fs, speed_of_sound, n_taps, hrtf_set)
    freqs = np.fft.rfftfreq(n_taps, 1.0 / fs)
    if method == "magls":
        # fitted on a 4x finer frequency grid, then cut to n_taps with a fade: the iterated
        # phases above the cutoff ring longer than the HRIRs, and folded into n_taps they
        # would spoil the low end (2.4 dB off at 250 Hz at order 2, where the fit is exact)
        n_fft = 4 * n_taps
        hp_ = np.zeros(h.shape[:-1] + (n_fft,))
        hp_[..., :n_taps] = h
        d = np.fft.irfft(_magls_spectra(order, hp_, fs, dirs, magls_cutoff(order, speed_of_sound),
                                        latency), n_fft, axis=-1)[..., :n_taps]
        nf = max(n_taps // 8, 1)
        d[..., -nf:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, nf)))
        F = np.fft.rfft(d, axis=-1)
    else:
        basic, re = decoder_matrices(order, dirs)
        f_basic = np.einsum("sc,sen->cen", basic, h)
        f_re = np.einsum("sc,sen->cen", re, h)
        x = np.clip(np.log2(np.maximum(freqs, 1e-9) / crossover_hz) / 0.5 + 0.5, 0.0, 1.0)
        hp = np.sin(0.5 * np.pi * x) ** 2                       # complementary, zero phase
        F = np.fft.rfft(f_basic, axis=-1) * (1.0 - hp) + np.fft.rfft(f_re, axis=-1) * hp
    # diffuse-field equalisation, from the HRIR set only: an isotropic field (SN3D
    # channel powers 1/(2l+1)) must reach each ear with the HRIRs' mean power
    w = 1.0 / (2.0 * _degrees(F.shape[0]) + 1.0)
    p_dec = np.einsum("c,cef->ef", w, np.abs(F) ** 2)
    p_ref = (np.abs(np.fft.rfft(h, axis=-1)) ** 2).mean(axis=0)
    eq = np.sqrt(_smooth_third_octave(p_ref, freqs) / np.maximum(
        _smooth_third_octave(p_dec, freqs), 1e-30))
    out = np.fft.irfft(F * eq[None, :, :], n_taps, axis=-1)
    return out, latency


def _smooth_third_octave(p, freqs):
    """Power spectra (..., n_f) averaged over 1/3 octave around each bin."""
    out = np.empty_like(p)
    for k, f in enumerate(freqs):
        sel = (freqs >= f / 2 ** (1 / 6)) & (freqs <= f * 2 ** (1 / 6)) if f > 0 else (
            freqs <= freqs[1])
        out[..., k] = p[..., sel].mean(axis=-1)
    return out


def decode(ambi, filters):
    """Binaural (n + n_taps - 1, 2) of an ambisonic signal/IR (n, n_ch)."""
    n_ch, _two, n_taps = filters.shape
    n = len(ambi) + n_taps - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    A = np.fft.rfft(ambi[:, :n_ch], nfft, axis=0)
    F = np.fft.rfft(filters, nfft, axis=-1)
    out = np.stack([np.fft.irfft((A * F[:, e, :].T).sum(axis=1), nfft)[:n] for e in range(2)],
                   axis=1)
    return out


def rotation_matrix(order, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(n_ch, n_ch) SH-domain matrix rotating the sound FIELD by the given angles
    (yaw about +z, pitch about +y, roll about +x, applied in that order). To render a
    listener who turns the head by (yaw, pitch, roll), rotate the field by the inverse."""
    a, b, c = np.radians([yaw_deg, pitch_deg, roll_deg])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    R = rz @ ry @ rx
    dirs = virtual_layout(max(64, 3 * (order + 1) ** 2))
    Y = sh_sn3d(dirs, order)
    Yr = sh_sn3d(dirs @ R.T, order)
    # Y(R d) = Y(d) M^T  =>  ambi' = M ambi
    M = np.linalg.lstsq(Y, Yr, rcond=None)[0].T
    M[np.abs(M) < 1e-12] = 0.0
    return M
