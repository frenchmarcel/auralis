#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a SOFA HRTF file (SimpleFreeFieldHRIR) to the Auralis .npz
format. Run it ONCE per HRTF set.

Usage:
    python3 tools/sofa_to_npz.py measured.sofa hrtf_subject.npz
    python3 tools/sofa_to_npz.py measured.sofa out.npz --taps 256 --dfeq
"""
import argparse
import sys

import numpy as np


def _read_sofa(path):
    try:
        from netCDF4 import Dataset
        ds = Dataset(path, "r")
        ir = np.array(ds.variables["Data.IR"][:], dtype=np.float64)
        fs = float(np.array(ds.variables["Data.SamplingRate"][:]).ravel()[0])
        sp = np.array(ds.variables["SourcePosition"][:], dtype=np.float64)
        rp = (np.array(ds.variables["ReceiverPosition"][:], dtype=np.float64)
              if "ReceiverPosition" in ds.variables else None)
        name = getattr(ds, "GLOBAL_DatabaseName",
                       getattr(ds, "GLOBAL_Title", "hrtf"))
        ds.close()
    except ImportError:
        import h5py
        f = h5py.File(path, "r")
        ir = np.array(f["Data.IR"], dtype=np.float64)
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
        sp = np.array(f["SourcePosition"], dtype=np.float64)
        rp = (np.array(f["ReceiverPosition"], dtype=np.float64)
              if "ReceiverPosition" in f else None)
        name = "hrtf"
        for k in ("GLOBAL_DatabaseName", "GLOBAL_Title"):
            if k in f.attrs:
                name = f.attrs[k]
                break
        f.close()
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    return ir, fs, sp, rp, str(name)


def _onset_delay(h, up=8, thresh=0.05):
    """Onset delay (fractional samples) from an FFT oversampling by `up`
    and the first crossing above thresh * max|h|."""
    n = len(h)
    H = np.fft.rfft(h)
    nu = n * up
    Hu = np.zeros(nu // 2 + 1, dtype=np.complex128)
    Hu[: len(H)] = H
    hu = np.fft.irfft(Hu, nu) * up
    a = np.abs(hu)
    lim = thresh * a.max()
    k = int(np.argmax(a >= lim))
    return k / up


def _frac_shift(h, shift):
    """Shift h by -shift samples (brings the onset back to the origin)."""
    n = len(h)
    H = np.fft.rfft(h, 2 * n)                 # anti-aliasing margin
    f = np.arange(len(H))
    H *= np.exp(2j * np.pi * f * shift / (2 * n))
    return np.fft.irfft(H, 2 * n)[:n]


def _min_phase_inverse(mag, limit_db=12.0):
    """Minimum-phase inverse (real cepstrum) of a magnitude template,
    regularised and capped at +limit_db."""
    m = np.maximum(mag, 1e-6)
    inv = 1.0 / m
    inv = np.minimum(inv, 10.0 ** (limit_db / 20.0))
    n_fft = 2 * (len(m) - 1)
    logm = np.log(np.maximum(inv, 1e-9))
    cep = np.fft.irfft(logm, n_fft)
    cep[1:n_fft // 2] *= 2.0
    cep[n_fft // 2 + 1:] = 0.0
    return np.exp(np.fft.rfft(cep, n_fft))


def convert(path_in, path_out, taps=256, dfeq=False, swap=False):
    ir, fs, sp, rp, name = _read_sofa(path_in)
    if ir.ndim != 3 or ir.shape[1] != 2:
        raise SystemExit("unexpected SOFA: Data.IR must be (M, 2, N)")
    M, _, N = ir.shape
    az = np.mod(sp[:, 0], 360.0)
    el = sp[:, 1]
    # ear order: receiver y > 0 = left (SOFA convention, +y to the left)
    if rp is not None and rp.shape[0] == 2 and rp[0, 1] < rp[1, 1]:
        ir = ir[:, ::-1, :]
    if swap:
        ir = ir[:, ::-1, :]

    if dfeq:
        H = np.abs(np.fft.rfft(ir, axis=-1))
        df = np.sqrt(np.mean(H ** 2, axis=(0, 1)))
        df /= max(df[1: len(df) // 2].max(), 1e-9) * 0.5
        eq = _min_phase_inverse(df)
        n_fft = 2 * (len(df) - 1)
        IR = np.fft.rfft(ir, n_fft, axis=-1) * eq
        ir = np.fft.irfft(IR, n_fft, axis=-1)[:, :, :N]

    L = min(taps, ir.shape[-1])
    out = np.zeros((M, 2, L), dtype=np.float32)
    dly = np.zeros((M, 2), dtype=np.float32)
    fade = np.ones(L)
    nf = min(16, L // 4)
    fade[-nf:] = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, nf)))
    guard = 4.0
    for m in range(M):
        d0 = max(_onset_delay(ir[m, 0]) - guard, 0.0)
        d1 = max(_onset_delay(ir[m, 1]) - guard, 0.0)
        h0 = _frac_shift(ir[m, 0], d0)
        h1 = _frac_shift(ir[m, 1], d1)
        # correct the interaural bias of the threshold: the shadowed ear
        # is low-passed, so its edge crosses the threshold earlier.
        # Sub-sample realignment of the far ear by cross-correlation
        # (the ir + delay pair stays exactly the original signal).
        n = len(h0)
        X = np.fft.rfft(h0, 2 * n) * np.conj(np.fft.rfft(h1, 2 * n))
        up = 8
        nu = 2 * n * up
        Xu = np.zeros(nu // 2 + 1, dtype=np.complex128)
        Xu[: len(X)] = X
        xc = np.fft.irfft(Xu, nu)
        span = 8 * up
        xc = np.concatenate([xc[-span:], xc[: span + 1]])
        lag = (int(np.argmax(xc)) - span) / up
        d1 -= lag        # peak at k = -D when h1 lags by D
        h1 = _frac_shift(ir[m, 1], d1)
        dly[m, 0], dly[m, 1] = d0, d1
        out[m, 0] = (h0[:L] * fade).astype(np.float32)
        out[m, 1] = (h1[:L] * fade).astype(np.float32)
    np.savez_compressed(path_out, ir=out, delay=dly,
                        az_deg=az.astype(np.float32),
                        el_deg=el.astype(np.float32),
                        fs=int(round(fs)), name=str(name))
    print("OK: %d directions, %d taps @ %d Hz -> %s"
          % (M, L, int(round(fs)), path_out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sofa")
    ap.add_argument("npz")
    ap.add_argument("--taps", type=int, default=256)
    ap.add_argument("--dfeq", action="store_true",
                    help="diffuse-field equalisation (min-phase, +12 dB max)")
    ap.add_argument("--swap", action="store_true",
                    help="swap left/right when the set is ambiguous")
    a = ap.parse_args()
    sys.exit(convert(a.sofa, a.npz, a.taps, a.dfeq, a.swap))
