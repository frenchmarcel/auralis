# -*- coding: utf-8 -*-
"""Acoustic material database: per octave band (125..8000 Hz), absorption
alpha, scattering s (share of REFLECTED energy) and transmission tau.
Reflected energy = E * (1 - alpha - tau), split by s.
"""

from __future__ import annotations

import numpy as np

from .bands import N_BANDS


def _mat(absorption, scattering=0.15, transmission=0.0):
    def vec(x):
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 0:
            a = np.full(N_BANDS, float(a), dtype=np.float32)
        if a.shape != (N_BANDS,):
            raise ValueError("material vectors must have %d bands" % N_BANDS)
        return a
    return {
        "absorption": vec(absorption),
        "scattering": vec(scattering),
        "transmission": vec(transmission),
    }


#                          125   250   500   1k    2k    4k    8k
PRESETS = {
    "default":       _mat([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], 0.15),
    "concrete":      _mat([0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04], 0.10),
    "brick":         _mat([0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07], 0.20),
    "marble":        _mat([0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02], 0.05),
    "plaster":       _mat([0.14, 0.10, 0.06, 0.05, 0.04, 0.03, 0.03], 0.10),
    "drywall":       _mat([0.29, 0.10, 0.05, 0.04, 0.07, 0.09, 0.09], 0.10,
                          [0.010, 0.006, 0.003, 0.002, 0.001, 0.001, 0.001]),
    "wood_panel":    _mat([0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.11], 0.12),
    "wood_floor":    _mat([0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07], 0.12),
    "parquet":       _mat([0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.07], 0.10),
    "glass_window":  _mat([0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.03], 0.05,
                          [0.050, 0.030, 0.018, 0.010, 0.006, 0.003, 0.002]),
    "glass_heavy":   _mat([0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02], 0.05,
                          [0.010, 0.006, 0.004, 0.002, 0.001, 0.001, 0.001]),
    "carpet_thin":   _mat([0.02, 0.04, 0.08, 0.20, 0.35, 0.40, 0.40], 0.20),
    "carpet_heavy":  _mat([0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.65], 0.25),
    "curtain_heavy": _mat([0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65], 0.35,
                          [0.080, 0.050, 0.030, 0.015, 0.010, 0.008, 0.006]),
    "acoustic_tile": _mat([0.50, 0.70, 0.85, 0.90, 0.90, 0.85, 0.80], 0.30),
    "acoustic_foam": _mat([0.15, 0.45, 0.80, 0.95, 0.95, 0.95, 0.95], 0.40),
    "audience":      _mat([0.52, 0.68, 0.85, 0.97, 0.93, 0.85, 0.85], 0.60),
    "metal_panel":   _mat([0.20, 0.20, 0.15, 0.10, 0.06, 0.05, 0.05], 0.10),
    "door_wood":     _mat([0.14, 0.10, 0.06, 0.08, 0.10, 0.10, 0.10], 0.10,
                          [0.030, 0.020, 0.012, 0.008, 0.005, 0.004, 0.003]),
    "water_surface": _mat([0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03], 0.05),
    "grass":         _mat([0.11, 0.26, 0.60, 0.69, 0.92, 0.99, 0.99], 0.50),
    # open doorway onto a large adjoining space: nothing comes back (Sabine idealisation)
    "opening":       _mat([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], 0.0),
}

PRESET_NAMES = sorted(PRESETS.keys())


def get_preset(name: str) -> dict:
    """Return a copy of a preset (falls back to 'default')."""
    m = PRESETS.get(name, PRESETS["default"])
    return {k: v.copy() for k, v in m.items()}


def build_tables(materials):
    """Stack a list of material dicts into (absorption, scattering,
    transmission) arrays of shape (n_mats, N_BANDS), float32."""
    if not materials:
        materials = [get_preset("default")]
    a = np.stack([m["absorption"] for m in materials]).astype(np.float32)
    s = np.stack([m["scattering"] for m in materials]).astype(np.float32)
    t = np.stack([m["transmission"] for m in materials]).astype(np.float32)
    return np.ascontiguousarray(a), np.ascontiguousarray(s), np.ascontiguousarray(t)
