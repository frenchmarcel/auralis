"""Physical boundary models inferred from random-incidence absorption alone, for the wave band."""
from __future__ import annotations

import numpy as np

from . import poromat
from .bands import CENTERS
from .reflection import alpha_random

_PTS = 9
_CACHE: dict = {}


def _grid(centers):
    return np.concatenate([np.geomspace(fc / 2 ** 0.5, fc * 2 ** 0.5, _PTS) for fc in centers])


def band_alpha(spec: dict, centers=CENTERS) -> np.ndarray:
    """Random-incidence alpha per octave of a stack under LOCAL reaction on its normal
    impedance, which is what the FDTD boundary does with it."""
    centers = np.asarray(centers, dtype=np.float64)
    g = poromat.gamma_normal(spec, _grid(centers))
    return alpha_random(g).reshape(len(centers), _PTS).mean(axis=1)


def porous(sigma, d_mm, gap_mm):
    """Miki layer (flow resistivity Pa.s/m2, thickness mm) over an air gap, rigid backing."""
    return poromat.spec_from_params("miki", sigma, d_mm, gap_mm)


def screen(r, m, gap_mm):
    """Resistive screen (flow resistance Pa.s/m, inertia kg/m2) over an air gap."""
    return {"layers": [{"model": "screen", "r": float(r), "m": float(m)},
                       {"model": "air", "d": max(float(gap_mm), 1.0) * 1e-3}]}


def panel(m, cav_mm, q):
    """Limp panel (kg/m2) over an air cavity (mm) with resonance quality q."""
    spec = poromat.panel_spec(m, cav_mm, q=q)
    return {"layers": spec["layers"]}


# family: (builder, coarse grid, bounds of the parameters for the local search). The bounds
# are PHYSICAL ranges, not fitting freedom: without them a flexible family mimics any flat,
# low alpha (concrete as "a 7.8 kg/m2 panel over 13 mm", a floor as "a 21 Pa.s/m screen over
# 0.93 m of air"). Screens: woven / non-woven fabrics and films measure ~100-3000 Pa.s/m,
# and their inertia is that of the air in their pores (<= 0.05 kg/m2, heavier "screens"
# mimic airtight panels). Panels: 2-60 kg/m2, Q 1-15 and at least 25 mm of cavity.
FAMILIES = {
    "porous": (porous,
               [(s, d, g) for d in (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200)
                for g in (0, 25, 50, 100, 200, 400) for s in np.geomspace(2e3, 3e5, 24)],
               [(2e3, 1e6), (5.0, 300.0), (0.0, 1000.0)]),
    "screen": (screen,
               [(r, m, g) for r in np.geomspace(100, 5000, 18) for m in (0.002, 0.01, 0.05)
                for g in (25, 50, 100, 200, 400, 800)],
               [(100.0, 5000.0), (5e-4, 0.05), (10.0, 1000.0)]),
    "panel": (panel,
              [(m, c, q) for m in (2.5, 5, 7.5, 10, 15, 20, 30) for c in (25, 50, 75, 100, 150, 200, 300)
               for q in (1, 2, 4, 8)],
              [(2.0, 60.0), (25.0, 500.0), (1.0, 15.0)]),
}


def _refine(cost, make, start, bounds, iters=60):
    """Coordinate descent in log space from the best grid point; a parameter whose lower
    bound is 0 (an air gap) may also switch to exactly 0."""
    c0, x = start[0], list(start[1:])
    step = [0.3] * len(x)
    for _ in range(iters):
        moved = False
        for i in range(len(x)):
            lo, hi = bounds[i]
            for sgn in (1.0, -1.0):
                y = list(x)
                if y[i] <= 0.0:
                    if sgn < 0:
                        continue
                    y[i] = lo + 0.02 * (hi - lo)
                else:
                    y[i] = float(np.clip(y[i] * np.exp(sgn * step[i]), lo, hi))
                    if lo == 0.0 and sgn < 0 and y[i] < 1.0:
                        y[i] = 0.0
                c = cost(make(*y))
                if c < c0 - 1e-7:
                    c0, x, moved = c, y, True
        if not moved:
            step = [s * 0.5 for s in step]
            if max(step) < 0.01:
                break
    return (c0, *x)


def infer(alpha, centers=CENTERS, max_rms=0.035, max_low=0.03, min_alpha=0.2) -> dict:
    """Best physical model whose random-incidence alpha reproduces `alpha` per octave.

    Families: porous layer, resistive screen or panel over a cavity. Identification uses
    the whole spectrum (the treble betrays a wrong model). Acceptance also requires the
    bands up to 1 kHz, where the wave band lives. Returns {"family": None, "reason"} when
    no model explains the curve, or when it is near-rigid (its phase is then not
    identifiable). Checked against 172 impedance-tube samples
    (tools/material_inference_check.py)."""
    a = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 0.99)
    c = np.asarray(centers, dtype=np.float64)
    key = (np.round(a, 4).tobytes(), c.tobytes(), max_rms, max_low, min_alpha)
    if key in _CACHE:
        return _CACHE[key]
    if a.max() < min_alpha:
        out = {"family": None, "reason": "hard surface (alpha <= %.2f): too little absorption "
                                         "to identify a physical model" % a.max()}
        _CACHE[key] = out
        return out
    low = c <= 1000.0

    def cost(spec):
        return float(np.sqrt(np.mean((band_alpha(spec, c) - a) ** 2)))

    best = {}
    for fam, (make, grid, bounds) in FAMILIES.items():
        start = min((cost(make(*p)), *p) for p in grid)
        best[fam] = _refine(cost, make, start, bounds)
    fam = min(best, key=lambda k: best[k][0])
    params = tuple(float(v) for v in best[fam][1:])
    spec = FAMILIES[fam][0](*params)
    rms = best[fam][0]
    rms_low = float(np.sqrt(np.mean((band_alpha(spec, c)[low] - a[low]) ** 2)))
    if rms > max_rms or rms_low > max_low:
        out = {"family": None, "reason": "no physical model explains alpha (best: %s, rms %.3f, "
                                         "up to 1 kHz %.3f)" % (fam, rms, rms_low)}
    else:
        out = {"family": fam, "params": params, "spec": spec, "rms": rms, "rms_low": rms_low}
    _CACHE[key] = out
    return out


def describe(res: dict) -> str:
    """One line for logs and the app."""
    if res.get("family") is None:
        return "alpha-only fit kept: " + res["reason"]
    p = res["params"]
    if res["family"] == "porous":
        txt = "porous layer, flow resistivity %.0f Pa.s/m2, %.0f mm, air gap %.0f mm" % p
    elif res["family"] == "screen":
        txt = "resistive screen %.0f Pa.s/m, %.3f kg/m2, air gap %.0f mm" % p
    else:
        txt = "panel %.1f kg/m2 over %.0f mm, Q %.1f" % p
    return "inferred %s (alpha rms %.3f)" % (txt, res["rms"])


def octave_targets(absorption, alpha_full=None, a63=None):
    """(centers, alpha) to infer from: the material's octaves, the 63 Hz octave when known,
    or its full curve (e.g. BRAS third octaves) averaged per octave from 63 Hz."""
    if alpha_full is not None:
        fb, ab = (np.asarray(v, dtype=np.float64) for v in alpha_full)
        c = np.concatenate([[63.0], CENTERS])
        c = c[(c / 2 ** 0.5 >= fb.min() * 0.99) & (c * 2 ** 0.5 <= fb.max() * 1.01)]
        g = _grid(c).reshape(len(c), _PTS)
        return c, np.interp(np.log(g), np.log(fb), ab).mean(axis=1)
    a = np.asarray(absorption, dtype=np.float64)
    if a63 is not None:
        return np.concatenate([[63.0], CENTERS]), np.concatenate([[float(a63)], a])
    return np.asarray(CENTERS, dtype=np.float64), a
