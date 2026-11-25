#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a BRAS material CSV (3 third-octave rows: frequencies,
absorption, scattering) to the 7 bands 125 Hz - 8 kHz of bras_bench.

Usage:
    python3 BRAS/bras_material.py mat_RockFonSonarG.csv
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from auralis.engine.bands import CENTERS  # noqa: E402


def convert(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p for p in line.replace(";", ",").split(",")
                     if p.strip()]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if vals:
                rows.append(np.array(vals))
    if len(rows) < 2:
        raise ValueError("unexpected CSV: %d numeric rows" % len(rows))
    f = rows[0]
    out = {}
    for name, r in (("absorption", rows[1]),
                    ("scattering", rows[2] if len(rows) > 2 else None)):
        if r is None:
            continue
        n = min(len(f), len(r))
        vals = []
        for fc in CENTERS:
            m = (f[:n] >= fc / np.sqrt(2.0)) & (f[:n] < fc * np.sqrt(2.0))
            v = float(np.clip(r[:n][m].mean() if m.any()
                              else np.interp(fc, f[:n], r[:n]), 0.0, 0.99))
            vals.append(round(v, 3))
        out[name] = vals
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    print(json.dumps(convert(argv[0]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
