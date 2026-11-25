# -*- coding: utf-8 -*-
"""Per-octave source EQ: a minimum-phase FIR applied to the source's results. The room is
linear, so filtering its impulse responses is exactly filtering the source's signal."""
from __future__ import annotations

import numpy as np

from auralis.engine.octave_fir import (BANDS_HZ, N_TAPS, band_gains_db,  # noqa: F401
                                       filter_ir, octave_fir)

MAX_DB = 24.0


def clean_eq(eq):
    """7 gains in dB (125 Hz - 8 kHz, within +-24 dB), or None for a flat EQ."""
    if eq is None or (isinstance(eq, (list, tuple)) and len(eq) == 0):
        return None
    if not isinstance(eq, (list, tuple)) or len(eq) != len(BANDS_HZ):
        raise ValueError("eq_db: 7 octave gains (125 Hz - 8 kHz) in dB")
    try:
        g = [float(x) for x in eq]
    except (TypeError, ValueError):
        raise ValueError("eq_db: gains must be numbers (dB)")
    if not all(np.isfinite(g)) or max(abs(x) for x in g) > MAX_DB:
        raise ValueError("eq_db: gains must be finite and within +-%g dB" % MAX_DB)
    return None if all(x == 0.0 for x in g) else [round(x, 2) for x in g]


def eq_fir(eq_db, fs, n=N_TAPS, iters=12):
    """Minimum-phase FIR (n taps) whose energy gain in each octave is the requested one."""
    return octave_fir(eq_db, fs, n=n, iters=iters)


def apply_to_result(res, eq_db):
    """EQ one bake_pair result in place: its IRs (binaural, ambisonic, per-solver omni
    parts) and its sound pressure levels. G (strength) is a room quantity, normalised by the
    source's own power: unchanged. Returns the band gains applied."""
    h = eq_fir(eq_db, res.fs)
    res.ir = filter_ir(res.ir, h)
    if getattr(res, "ambi", None) is not None:
        res.ambi = filter_ir(res.ambi, h)
    parts = getattr(res, "omni_parts", None)
    if parts is not None:
        for k in ("geo", "wave"):
            if parts.get(k) is not None:
                parts[k] = filter_ir(parts[k], h)
    gains = band_gains_db(h, res.fs)
    if res.metrics.get("spl_bands") is not None:
        res.metrics["spl_bands"] = [float(x) + g for x, g in zip(res.metrics["spl_bands"], gains)]
    return gains
