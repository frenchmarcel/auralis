# -*- coding: utf-8 -*-
"""Open-plan offices, ISO 3382-3:2022: spatial decay of speech D2,S, speech level at
4 m Lp,A,S,4m, comfort distance rC and distraction distance rD along a path of
receivers, from simulated levels (G) and MTFs."""
from __future__ import annotations

import numpy as np

from . import speech
from .level import A_WEIGHT_DB

# Table 1: normal-effort speech, free-field SPL at 1 m in front of an omni source
SPEECH_1M_DB = np.array([49.9, 54.3, 58.0, 52.0, 44.8, 38.8, 33.5])
COMFORT_DBA = 45.0
DISTRACTION_STI = 0.50
R_MIN, R_MAX = 2.0, 16.0


def _dba(levels):
    return float(10.0 * np.log10(np.sum(10.0 ** ((np.asarray(levels) + A_WEIGHT_DB) / 10.0))))


def evaluate(distances, g_bands, mtfs, noise_db=None):
    """distances (N,) m from the source; g_bands (N, 7) strength G per position (dB re
    10 m free field, so the speech level is SPEECH_1M + G - 20); mtfs (N, 7, 14).
    Returns the ISO 3382-3 single numbers and the per-position values."""
    r = np.asarray(distances, dtype=np.float64)
    G = np.asarray(g_bands, dtype=np.float64)
    speech_n = SPEECH_1M_DB[None, :] + G - 20.0                 # formula 4
    la = np.array([_dba(x) for x in speech_n])                    # formula 5
    sti = np.array([speech.sti(m, s, noise_db)["sti"] for m, s in zip(mtfs, speech_n)])
    sel = (r >= R_MIN) & (r <= R_MAX)
    out = {"positions": [{"distance_m": round(float(d), 3), "Lp_A_S_db": round(float(a), 2),
                          "STI": round(float(t), 4), "Lp_S_db": np.round(s, 2).tolist()}
                         for d, a, t, s in zip(r, la, sti, speech_n)],
           "n_in_range": int(sel.sum()), "D2S_db": None, "Lp_A_S_4m_db": None,
           "rC_m": None, "rC_note": None, "rD_m": None, "rD_note": None,
           "decay_fit": None, "sti_fit": None,
           "Lp_A_B_db": None if noise_db is None else round(_dba(noise_db), 2)}
    if sel.sum() >= 2:
        x = np.log10(r[sel])
        y = la[sel]
        n = len(x)
        slope = (n * (x * y).sum() - y.sum() * x.sum()) / (n * (x * x).sum() - x.sum() ** 2)
        d2s = -np.log10(2.0) * slope                               # formula 6
        l4 = y.mean() + d2s * np.log2(r[sel] / 4.0).mean()          # formula 7
        out["D2S_db"], out["Lp_A_S_4m_db"] = round(float(d2s), 2), round(float(l4), 2)
        out["decay_fit"] = {"intercept_db": round(float(y.mean() - slope * x.mean()), 4),
                            "slope_db_per_decade": round(float(slope), 4)}
        if d2s > 0:
            rc = 2.0 ** ((l4 - COMFORT_DBA + 2.0 * d2s) / d2s)      # formula 8
            out["rC_m"] = round(float(rc), 2)
            if la[sel][np.argmin(r[sel])] < COMFORT_DBA:
                out["rC_note"] = "extrapolated: smaller than %.1f m" % r[sel].min()
            elif la[sel][np.argmax(r[sel])] > COMFORT_DBA:
                out["rC_note"] = "extrapolated: larger than %.1f m" % r[sel].max()
    use = r > 1.0 + 1e-9                                          # all but the 1 m point
    if use.sum() >= 2:
        k, c0 = np.polyfit(r[use], sti[use], 1)
        out["sti_fit"] = {"intercept": round(float(c0), 6), "slope_per_m": round(float(k), 6)}
        if abs(k) > 1e-12:
            rd = (DISTRACTION_STI - c0) / k
            out["rD_m"] = round(float(rd), 2)
            if rd < r[use].min() or rd > r[use].max():
                out["rD_note"] = "extrapolated outside %.1f-%.1f m" % (r[use].min(), r[use].max())
    return out
