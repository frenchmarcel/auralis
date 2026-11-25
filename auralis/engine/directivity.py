# -*- coding: utf-8 -*-
"""Source directivity: per-band ENERGY gain Q_b(theta), axisymmetric
about the front axis, normalised to Q_b(0) = 1. Stored as a
(N_BANDS x n_samples) table over theta in [0, pi], linearly interpolated."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from .bands import CENTERS, N_BANDS

DEFAULT_SAMPLES = 64


class Directivity:
    """Energy table (N_BANDS, S) over uniform theta in [0, pi]."""

    def __init__(self, table: np.ndarray, name: str = "custom"):
        t = np.asarray(table, dtype=np.float64)
        if t.ndim == 1:
            t = np.broadcast_to(t, (N_BANDS, t.shape[0])).copy()
        if t.ndim != 2 or t.shape[0] != N_BANDS or t.shape[1] < 2:
            raise ValueError("invalid directivity table %r" % (t.shape,))
        self.table = np.ascontiguousarray(np.maximum(t, 0.0))
        self.name = name

    @property
    def n_samples(self) -> int:
        return self.table.shape[1]

    def is_omni(self) -> bool:
        return bool(np.allclose(self.table, 1.0))

    def energy(self, cos_theta) -> np.ndarray:
        """Q_b for one or more cos(theta). Returns (..., N_BANDS)."""
        ct = np.clip(np.asarray(cos_theta, dtype=np.float64), -1.0, 1.0)
        u = np.arccos(ct) / np.pi * (self.n_samples - 1)
        i0 = np.clip(u.astype(np.int64), 0, self.n_samples - 2)
        f = u - i0
        t = self.table                                   # (NB, S)
        out = t[:, i0] * (1.0 - f) + t[:, i0 + 1] * f    # (NB, ...)
        return np.moveaxis(out, 0, -1)

    def amp(self, cos_theta) -> np.ndarray:
        """Amplitude gain sqrt(Q_b)."""
        return np.sqrt(self.energy(cos_theta))

    def mean_q(self) -> np.ndarray:
        """Spherical mean of Q per band, (1/2) int Q(theta) sin(theta)
        dtheta: total power relative to an omni of same on-axis level."""
        th = np.linspace(0.0, np.pi, 721)
        q = self.energy(np.cos(th))              # (721, NB)
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        return 0.5 * _trapz(q * np.sin(th)[:, None], th, axis=0)

    def directivity_index_db(self) -> np.ndarray:
        """DI = 10 log10(Q(0) / mean) per band."""
        return 10.0 * np.log10(1.0 / np.maximum(self.mean_q(), 1e-9))

    def first_order_weights(self) -> np.ndarray:
        """Per band a of the pressure pattern a + (1 - a) cos(theta) of the same power.

        Same radiated power for the same on-axis level: a in [0.25, 1], from omni
        through cardioid (0.5) to hypercardioid (DI 6 dB). The wave band uses it,
        since the power sets the reverberant field."""
        p = np.clip(self.mean_q(), 0.25, 1.0)        # a^2 + (1 - a)^2 / 3 = p
        return (1.0 + np.sqrt(12.0 * p - 3.0)) / 4.0


def from_function(fn: Callable[[np.ndarray, float], np.ndarray],
                  name: str = "custom",
                  n_samples: int = DEFAULT_SAMPLES) -> Directivity:
    """fn(theta_array, band_center_hz) -> energy Q(theta) for that band."""
    th = np.linspace(0.0, np.pi, n_samples)
    rows = [np.asarray(fn(th, float(fc)), dtype=np.float64)
            for fc in CENTERS]
    return Directivity(np.stack(rows), name)


# ---------------------------------------------------------------------------
# Analytic families
# ---------------------------------------------------------------------------

def omni() -> Directivity:
    return Directivity(np.ones((N_BANDS, 2)), "omni")


def cardioid_family(a: float, name: str) -> Directivity:
    """Amplitude (1-a) + a cos(theta); energy = its square (includes the
    rear lobe of the super/hyper-cardioid)."""
    def q(th, _fc):
        v = (1.0 - a) + a * np.cos(th)
        return v * v
    return from_function(q, name)


def figure8() -> Directivity:
    return from_function(lambda th, _f: np.cos(th) ** 2, "figure8")


def cone(inner_deg: float, outer_deg: float,
         outer_gain_db: float = -18.0) -> Directivity:
    """Game-engine style cone: full level up to inner, cosine ramp down to
    outer, then constant gain outer_gain_db (energy)."""
    inner = np.radians(max(0.0, min(inner_deg, 179.9)))
    outer = np.radians(np.clip(outer_deg, inner_deg + 1e-3, 180.0))
    g = 10.0 ** (outer_gain_db / 10.0)

    def q(th, _fc):
        t = np.clip((th - inner) / max(outer - inner, 1e-9), 0.0, 1.0)
        ramp = 0.5 * (1.0 + np.cos(np.pi * t))          # 1 -> 0
        return g + (1.0 - g) * ramp
    return from_function(q, "cone")


def _beam(th, sigma_rad, floor):
    """Gaussian beam in angle with an omni floor (energy)."""
    return floor + (1.0 - floor) * np.exp(-(th / sigma_rad) ** 2)


def speaker_2way() -> Directivity:
    """Typical 2-way loudspeaker: near omni at 125 Hz, beam narrowing with
    frequency (DI ~ 2 dB -> ~9 dB)."""
    sigma_deg = {125: 170.0, 250: 140.0, 500: 105.0, 1000: 78.0,
                 2000: 58.0, 4000: 44.0, 8000: 34.0}
    floor = {125: 0.60, 250: 0.42, 500: 0.26, 1000: 0.15,
             2000: 0.09, 4000: 0.06, 8000: 0.04}

    def q(th, fc):
        k = int(round(fc))
        return _beam(th, np.radians(sigma_deg[k]), floor[k])
    return from_function(q, "speaker_2way")


def horn() -> Directivity:
    """Horn: strongly directional from the mid range up."""
    sigma_deg = {125: 120.0, 250: 90.0, 500: 60.0, 1000: 42.0,
                 2000: 32.0, 4000: 26.0, 8000: 22.0}
    floor = {125: 0.30, 250: 0.18, 500: 0.08, 1000: 0.04,
             2000: 0.025, 4000: 0.02, 8000: 0.015}

    def q(th, fc):
        k = int(round(fc))
        return _beam(th, np.radians(sigma_deg[k]), floor[k])
    return from_function(q, "horn")


def voice() -> Directivity:
    """Human voice (mouth facing front): gentle at low frequency, ~8 dB DI
    at 8 kHz, attenuated rear (from average speech measurements)."""
    sigma_deg = {125: 175.0, 250: 160.0, 500: 130.0, 1000: 100.0,
                 2000: 78.0, 4000: 62.0, 8000: 52.0}
    floor = {125: 0.70, 250: 0.55, 500: 0.38, 1000: 0.24,
             2000: 0.15, 4000: 0.10, 8000: 0.07}

    def q(th, fc):
        k = int(round(fc))
        return _beam(th, np.radians(sigma_deg[k]), floor[k])
    return from_function(q, "voice")


DIRECTIVITY_PRESETS = {
    "omni": omni,
    "subcardioid": lambda: cardioid_family(0.25, "subcardioid"),
    "cardioid": lambda: cardioid_family(0.5, "cardioid"),
    "supercardioid": lambda: cardioid_family(0.634, "supercardioid"),
    "hypercardioid": lambda: cardioid_family(0.75, "hypercardioid"),
    "figure8": figure8,
    "speaker_2way": speaker_2way,
    "horn": horn,
    "voice": voice,
}
DIRECTIVITY_NAMES = list(DIRECTIVITY_PRESETS.keys())


def get_directivity(name: str) -> Directivity:
    try:
        return DIRECTIVITY_PRESETS[name]()
    except KeyError:
        raise ValueError("unknown directivity: %r (choices: %s)"
                         % (name, ", ".join(DIRECTIVITY_NAMES)))


def as_directivity(x: Union[None, str, Directivity]) -> Optional[Directivity]:
    """None / 'omni' / omni Directivity -> None (fast path)."""
    if x is None:
        return None
    if isinstance(x, str):
        d = get_directivity(x)
    elif isinstance(x, Directivity):
        d = x
    else:
        raise TypeError("invalid directivity: %r" % (x,))
    return None if d.is_omni() else d
