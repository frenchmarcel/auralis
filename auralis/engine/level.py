"""Absolute levels: SPL at the receiver, A weighting, strength factor G.
Convention: the pressure IR is 1/d on the direct path ("pressure at 1 m"
reference); SPL_b = SPL_1m,b + 10 log10(E_b), G_b = 10 log10(E_b) + 20."""
from __future__ import annotations

import numpy as np

from .bands import CENTERS, N_BANDS

# A weighting at the octave centres 125 Hz .. 8 kHz (IEC 61672-1).
A_WEIGHT_DB = np.array([-16.1, -8.6, -3.2, 0.0, +1.2, +1.0, -1.1])

# C weighting at the same centres (IEC 61672-1).
C_WEIGHT_DB = np.array([-0.2, 0.0, 0.0, 0.0, -0.2, -0.8, -3.0])

# Reference pressure (20 uPa) and reference power (1 pW).
P_REF = 2.0e-5
W_REF = 1.0e-12


def spl1m_from_power(power_db: float, rho_c: float = 413.0) -> float:
    """Free-field SPL at 1 m from an omni source of power level L_W
    (dB re 1 pW); rho_c in rayls (413 = air at 20 C)."""
    return float(power_db
                 + 10.0 * np.log10(rho_c * W_REF / (4.0 * np.pi * P_REF ** 2)))


def band_energies(hist_total: np.ndarray) -> np.ndarray:
    """Per-band energy (7,) of the omni echogram (7, n_bins)."""
    h = np.asarray(hist_total, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != N_BANDS:
        raise ValueError("hist_total must have shape (7, n_bins)")
    return h.sum(axis=1)


def strength_g(energies: np.ndarray):
    """Strength factor G (ISO 3382-1), free-field reference at 10 m.
    Returns (G_bands (7,), G_mid = dB mean of the 500 Hz and 1 k bands)."""
    E = np.maximum(np.asarray(energies, dtype=np.float64), 1e-30)
    G = 10.0 * np.log10(E) + 20.0
    i500 = int(np.argmin(np.abs(CENTERS - 500.0)))
    i1k = int(np.argmin(np.abs(CENTERS - 1000.0)))
    return G, float(0.5 * (G[i500] + G[i1k]))


def spl_bands(energies: np.ndarray, spl_1m) -> np.ndarray:
    """Per-band SPL at the receiver for a source given in dB SPL @ 1 m
    (scalar = flat spectrum, or a (7,) per-band array)."""
    E = np.maximum(np.asarray(energies, dtype=np.float64), 1e-30)
    ref = np.asarray(spl_1m, dtype=np.float64)
    return ref + 10.0 * np.log10(E)


def dba_total(spl_per_band: np.ndarray) -> float:
    """Overall A-weighted level (energy sum of the weighted bands)."""
    L = np.asarray(spl_per_band, dtype=np.float64)
    return float(10.0 * np.log10(
        np.sum(10.0 ** ((L + A_WEIGHT_DB) / 10.0))))


def dbc_total(spl_per_band: np.ndarray) -> float:
    """Overall C-weighted level (energy sum of the weighted bands)."""
    L = np.asarray(spl_per_band, dtype=np.float64)
    return float(10.0 * np.log10(
        np.sum(10.0 ** ((L + C_WEIGHT_DB) / 10.0))))


def db_total(spl_per_band: np.ndarray) -> float:
    """Overall linear level (dB Z, energy sum of the bands)."""
    L = np.asarray(spl_per_band, dtype=np.float64)
    return float(10.0 * np.log10(np.sum(10.0 ** (L / 10.0))))


def level_stats(hist_total: np.ndarray, spl_1m=94.0, shift_db=None) -> dict:
    """Dict ready for the bake stats: per-band SPL, Z and A totals,
    per-band G and G_mid. shift_db (7,): per-band level change of the delivered IR over
    the echogram's (the wave band of a hybrid run)."""
    E = band_energies(hist_total)
    if shift_db is not None:
        E = E * 10.0 ** (np.asarray(shift_db, dtype=np.float64) / 10.0)
    G, g_mid = strength_g(E)
    L = spl_bands(E, spl_1m)
    return {
        "spl_bands": L,
        "spl_db": db_total(L),
        "spl_dba": dba_total(L),
        "g_bands": G,
        "g_mid": g_mid,
    }
