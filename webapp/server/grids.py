# -*- coding: utf-8 -*-
"""Surface receivers: grids of points on a layer (offset along its normal, into the
air) or on a free rectangle, with the same validity rules as point receivers."""
from __future__ import annotations

import math

import numpy as np

from . import sim as simlib

MAX_POINTS = 2500
KINDS = ("surface", "plane")
DEFAULT_RAYS = 50000


def clean_grid(g, layer_names):
    """Validated, normalised copy of a grid definition."""
    x = dict(g)
    x["id"] = str(x.get("id") or "")
    if not x["id"] or "/" in x["id"] or x["id"].startswith("."):
        raise ValueError("a grid needs an id")
    x["name"] = str(x.get("name") or x["id"])
    kind = x.get("kind")
    if kind not in KINDS:
        raise ValueError("%s: kind must be 'surface' or 'plane'" % x["name"])
    x["spacing_m"] = float(x.get("spacing_m", 1.0))
    if not 0.1 <= x["spacing_m"] <= 20.0:
        raise ValueError("%s: spacing must be within 0.1-20 m" % x["name"])
    if kind == "surface":
        if x.get("layer") not in layer_names:
            raise ValueError("%s: unknown layer '%s'" % (x["name"], x.get("layer")))
        x["offset_m"] = float(x.get("offset_m", 1.2))
        if not 0.01 <= x["offset_m"] <= 10.0:
            raise ValueError("%s: offset must be within 0.01-10 m" % x["name"])
    else:
        c = x.get("center")
        if not (isinstance(c, (list, tuple)) and len(c) == 3):
            raise ValueError("%s: a plane grid needs a center [x, y, z]" % x["name"])
        x["center"] = [float(v) for v in c]
        x["width_m"], x["length_m"] = float(x.get("width_m", 10.0)), float(x.get("length_m", 10.0))
        if not (0.1 <= x["width_m"] <= 500 and 0.1 <= x["length_m"] <= 500):
            raise ValueError("%s: width and length must be within 0.1-500 m" % x["name"])
        x["azimuth_deg"] = float(x.get("azimuth_deg", 0.0))
        x["tilt_deg"] = float(x.get("tilt_deg", 0.0))
    return x


def _lattice_basis(n):
    """In-plane axes shared by every coplanar triangle (aligned with world X, or Y)."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - n * np.dot(ref, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _surface_points(grid, mesh, inside_fn):
    verts, faces, face_layer, _closed, names = mesh
    li = names.index(grid["layer"])
    tri = verts[faces[face_layer == li]]
    s, off = grid["spacing_m"], grid["offset_m"]
    pts = []
    for t in tri:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        a2 = float(np.linalg.norm(n))
        if a2 < 1e-12:
            continue
        n /= a2
        if n[2] < -1e-6 or (abs(n[2]) <= 1e-6 and (n[1] < -1e-6 or (abs(n[1]) <= 1e-6 and n[0] < 0))):
            n = -n                                      # canonical: shared by coplanar faces
        u, v = _lattice_basis(n)
        uv = np.stack([t @ u, t @ v], axis=1)
        lo, hi = np.floor(uv.min(0) / s - 0.5), np.ceil(uv.max(0) / s - 0.5)
        gu, gv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        cand = (np.stack([gu.ravel(), gv.ravel()], axis=1) + 0.5) * s
        a, b, c = uv
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        if abs(np.linalg.det(m)) < 1e-14:
            continue
        lam = np.linalg.solve(m, (cand - a).T).T
        ok = (lam[:, 0] >= -1e-9) & (lam[:, 1] >= -1e-9) & (lam.sum(1) <= 1 + 1e-9)
        if not ok.any():
            continue
        d = float(np.dot(n, t[0]))
        on = cand[ok, 0:1] * u + cand[ok, 1:2] * v + d * n
        pts.append((on, n, u))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    # offset towards the air: the side where the face's points land inside the volume
    probes = np.concatenate([[on.mean(axis=0) + n * off, on.mean(axis=0) - n * off]
                             for on, n, _u in pts])
    ins = inside_fn(probes).reshape(-1, 2)
    sides = [1.0 if a or not b else -1.0 for a, b in ins]
    p = np.concatenate([on + sd * off * n for (on, n, _u), sd in zip(pts, sides)])
    nrm = np.concatenate([np.tile(sd * n, (len(on), 1)) for (on, n, _u), sd in zip(pts, sides)])
    uax = np.concatenate([np.tile(u, (len(on), 1)) for on, _n, u in pts])
    _, keep = np.unique(np.round(p / 1e-4).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(keep)
    return p[keep], nrm[keep], uax[keep]


def _plane_points(grid):
    s = grid["spacing_m"]
    nu = max(1, int(math.floor(grid["width_m"] / s + 1e-9)))
    nv = max(1, int(math.floor(grid["length_m"] / s + 1e-9)))
    gu, gv = np.meshgrid((np.arange(nu) - (nu - 1) / 2.0) * s, (np.arange(nv) - (nv - 1) / 2.0) * s)
    az, tl = math.radians(grid.get("azimuth_deg", 0.0)), math.radians(grid.get("tilt_deg", 0.0))
    u = np.array([math.cos(az), math.sin(az), 0.0])                 # width axis
    v0 = np.array([-math.sin(az), math.cos(az), 0.0])
    v = v0 * math.cos(tl) + np.array([0.0, 0.0, 1.0]) * math.sin(tl)  # length axis, tilted
    p = np.asarray(grid["center"]) + gu.reshape(-1, 1) * u + gv.reshape(-1, 1) * v
    n = np.cross(u, v)
    return p, np.tile(n, (len(p), 1)), np.tile(u, (len(p), 1))


def points(grid, sim, geometry, mesh, closure, axes=False):
    """(points (P, 3), valid (P,), reasons per point index -> str, area per point m2),
    plus (normals, lattice u axes) (P, 3) each when `axes`."""
    def inside_fn(p):
        return simlib.point_inside(p, geometry, mesh, closure, sim["settings"])

    p, nrm, uax = (_surface_points(grid, mesh, inside_fn) if grid["kind"] == "surface"
                   else _plane_points(grid))
    if len(p) > MAX_POINTS:
        raise ValueError("%s: %d points (max %d): use a larger spacing"
                         % (grid["name"], len(p), MAX_POINTS))
    if len(p) == 0:
        empty = (p, np.zeros(0, bool), {}, np.zeros(0))
        return empty + (nrm, uax) if axes else empty
    ins, dist = simlib.point_checks(p, geometry, mesh, closure, sim["settings"])
    valid = ins & (dist >= simlib.MIN_SURFACE_DIST)
    why = {}
    for i in np.flatnonzero(~ins):
        why[int(i)] = "outside the air volume"
    for i in np.flatnonzero(ins & (dist < simlib.MIN_SURFACE_DIST)):
        why[int(i)] = "%.2f m from a surface" % dist[i]
    for s in sim["sources"]:
        d = np.linalg.norm(p - np.asarray(s["position"], float), axis=1)
        close = valid & (d < simlib.MIN_SRC_RCV_DIST)
        for i in np.flatnonzero(close):
            why[int(i)] = "%.2f m from %s" % (d[i], s["name"])
        valid &= ~close
    out = (p, valid, why, np.full(len(p), grid["spacing_m"] ** 2))
    return out + (nrm, uax) if axes else out


def receiver_radius(grid):
    """Receiver sphere of a grid point: half the spacing within 0.3-1 m (in a
    15 000 m3 hall the 0.3 m sphere leaves T30 noisy)."""
    return float(np.clip(0.5 * grid["spacing_m"], 0.3, 1.0))
