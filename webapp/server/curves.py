# -*- coding: utf-8 -*-
"""Plot data for the results view: decimated IR, EDC per band, smoothed FR."""
from __future__ import annotations

import numpy as np

from auralis.engine import rirmetrics as rm
from auralis.engine.bands import LABELS

MAX_POINTS = 4000
SMOOTHINGS = (0, 1, 3, 6, 12, 24)


def minmax_decimate(t, x, n_max=MAX_POINTS):
    """Keep min and max of each bucket so peaks survive the decimation."""
    if len(x) <= n_max:
        return t, x
    nb = n_max // 2
    edges = np.linspace(0, len(x), nb + 1).astype(int)
    tt, xx = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:b]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for k in sorted((i, j)):
            tt.append(t[k])
            xx.append(x[k])
    return np.asarray(tt), np.asarray(xx)


def _r(a, nd=5):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def curves(ir, fs, scale, fr_smoothing=0, omni=None):
    """ir: (n, 2) raw binaural IR; omni: the omnidirectional IR (ambisonic W) when the
    run produced it - EDC and frequency response then use it, as the parameters do
    (otherwise the mean of the ears); scale: common normalisation."""
    if fr_smoothing not in SMOOTHINGS:
        raise ValueError("fr_smoothing must be one of %s" % (SMOOTHINGS,))
    ir = np.asarray(ir, dtype=np.float64) * scale
    n = len(ir)
    t = np.arange(n) / fs
    tl, left = minmax_decimate(t, ir[:, 0])
    tr, right = minmax_decimate(t, ir[:, 1])
    if len(tl) != len(tr):
        right = np.interp(tl, tr, right)
    if omni is not None:
        mono = np.asarray(omni, dtype=np.float64)[:n] * scale
        to, om = minmax_decimate(t[:len(mono)], mono)
        if len(to) != len(tl):
            om = np.interp(tl, to, om)
    else:
        mono, om = ir.mean(axis=1), None

    edc = {}
    bands = rm.band_split(mono, fs)
    step = max(1, n // 1500)
    for b, lab in enumerate(LABELS):
        db = rm.schroeder_db(bands[b])
        keep = np.flatnonzero(db > -100.0)
        last = int(keep[-1]) + 1 if len(keep) else 1
        edc[lab] = {"t": _r(t[:last:step], 5), "db": _r(np.maximum(db[:last:step], -100), 2)}

    spec = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    fgrid = np.geomspace(20.0, min(20000.0, fs / 2 - 1), 600)
    if fr_smoothing:
        half = 2.0 ** (1.0 / (2 * fr_smoothing))
        cs = np.concatenate([[0.0], np.cumsum(spec)])
        lo = np.searchsorted(freqs, fgrid / half)
        hi = np.maximum(np.searchsorted(freqs, fgrid * half), lo + 1)
        p = (cs[hi] - cs[lo]) / (hi - lo)
    else:
        p = np.interp(fgrid, freqs, spec)
    fr_db = 10.0 * np.log10(np.maximum(p, 1e-30))
    return {"fs": int(fs),
            "ir": {"t": _r(tl, 6), "left": _r(left, 6), "right": _r(right, 6),
                   "omni": None if om is None else _r(om, 6)},
            "source": "omni" if omni is not None else "binaural mean",
            "edc": edc,
            "fr": {"f": _r(fgrid, 2), "db": _r(fr_db, 2)}}
