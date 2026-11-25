# -*- coding: utf-8 -*-
"""Geometry import and checks for the web app: units/axes, layers, open
edges, watertight shell, air volume, overlapping faces, small gaps,
point-in-volume and wall distance."""
from __future__ import annotations

import hashlib
import itertools
import math
import zlib

import numpy as np

from auralis.engine.project import read_obj

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}
DEFAULT_LAYER = "(default)"
WELD_TOL = 1e-6          # m, vertex welding for the edge topology
SHORT_EDGE = 0.02        # m, short-edge / gap threshold when no transition frequency is known
MAX_PER_KIND = 200       # findings listed per kind (the counts stay exact)
MAX_EXCLUSIONS = 5000
VOXEL_BUDGET = 3e6       # cells for the closure test (~4 s on a concert hall)
CLOSED_FRACTION = 0.25   # enclosed air must fill this much of the bbox to call it a room
C_RULE = 343.0           # m/s, speed of sound of the lambda/4 rule
CELLS_PER_WAVELENGTH = 13.5   # wave grid at the transition: dx = c / (13.5 fc), as sim.validate
# a slit through a wall, voxelized by the engine (cells within 0.87 dx of a triangle are
# solid), measured over 19 grid alignments: never open below 1.5 dx, always
# open from 2.6 dx, in between it depends on how the grid falls
SEAL_CELLS = 1.5
OPEN_CELLS = 2.6
# Overlapping faces: planes equal within PLANE_TOL (1 mm, far below any wave grid) and
# COPLANAR_DEG, sharing more than OVERLAP_FRACTION of the smaller triangle (neighbours
# that only share an edge have zero common area; float noise stays orders below).
PLANE_TOL = 1e-3
COPLANAR_DEG = 1.0
OVERLAP_FRACTION = 1e-3
# Small gaps: surfaces closer than the detail threshold but farther than GAP_MIN (below:
# they touch, as at a T-junction). Parallel faces within GAP_DEG bound a thin air layer.
GAP_MIN = 1e-3
GAP_DEG = 10.0
PAIR_BUDGET = 4e7        # candidate pairs of one broad phase before it gives up (noisily)
PAIR_CHUNK = 4e6
SEVERITIES = ("info", "warning", "error")
TAGS = {"open_edge": "oe", "nonmanifold_edge": "nm", "short_edge": "se",
        "degenerate_triangle": "dt", "overlapping_faces": "of", "small_gap": "sg"}


def import_obj(path: str, unit: str = "m", up_axis: str = "z"):
    """OBJ -> (verts (nv,3) in metres Z-up, faces (nf,3), face_layer idx, layer names)."""
    if unit not in UNIT_SCALE:
        raise ValueError("unit must be one of %s" % ", ".join(UNIT_SCALE))
    if up_axis not in ("z", "y"):
        raise ValueError("up_axis must be 'z' or 'y'")
    v, f, layer = read_obj(path)
    if len(f) == 0:
        raise ValueError("no faces found in the OBJ file")
    v = v * UNIT_SCALE[unit]
    if up_axis == "y":
        # +90 deg about X: Y-up files become Z-up, handedness preserved
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    names = []
    index = {}
    fl = np.empty(len(f), dtype=np.int32)
    for i, n in enumerate(layer):
        n = DEFAULT_LAYER if n is None else n
        if n not in index:
            index[n] = len(names)
            names.append(n)
        fl[i] = index[n]
    return v, f, fl, names


def import_model(path, unit="m", up_axis="z", ifc_space=None, ifc_exclude=None):
    """.obj, Odeon .Par, .dxf or .3dm -> (verts, faces, face_layer idx, layer names, notes)."""
    if path.lower().endswith(".par"):
        from .parfile import read_par
        v, f, fl, names, rep = read_par(path)
        notes = ["Odeon model: %d points, %d surfaces, %d layers"
                 % (rep["points"], rep["surfaces"], rep["layers"])]
        if rep["ignored_lines"]:
            notes.append("%d line(s) of free text ignored (%s)"
                         % (rep["ignored_lines"], ", ".join(rep["ignored_words"])))
        if rep["offset_fallbacks"]:
            notes.append("%d point reference(s) resolved without NumbOffSet"
                         % rep["offset_fallbacks"])
        if unit not in UNIT_SCALE or up_axis not in ("z", "y"):
            raise ValueError("unit must be m/cm/mm and up_axis z/y")
        v = v * UNIT_SCALE[unit]
        if up_axis == "y":
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
        return v, f, fl, names, notes
    if path.lower().endswith(".ifc"):
        from . import ifcfile
        v, f, fl, names, rep = ifcfile.read_ifc(
            path, exclude=ifcfile.DEFAULT_EXCLUDE if ifc_exclude is None else ifc_exclude,
            space=ifc_space or None)
        notes = ["IFC: %d triangles on %d layers%s" % (
            len(f), len(names), " (IfcSpace %s as the volume)" % ifc_space if ifc_space else "")]
        if rep["failed"]:
            notes.append("elements whose geometry failed: " + ", ".join(
                "%s x%d" % kv for kv in sorted(rep["failed"].items())))
        return v, f, fl, names, notes                   # IFC: metres, Z up by definition
    if path.lower().endswith((".dxf", ".3dm")):
        from . import cadfile
        reader = cadfile.read_dxf if path.lower().endswith(".dxf") else cadfile.read_3dm
        v, f, fl, names, rep = reader(path)
        notes = ["%d triangles on %d layers" % (len(f), len(names))]
        if rep["skipped"]:
            notes.append("entities without surface, not imported: " + ", ".join(
                "%s x%d" % kv for kv in sorted(rep["skipped"].items())))
        if rep["file_unit"] and rep["file_unit"] != unit:
            notes.append("the file declares its unit as '%s' (you chose '%s')"
                         % (rep["file_unit"], unit))
        if unit not in UNIT_SCALE or up_axis not in ("z", "y"):
            raise ValueError("unit must be m/cm/mm and up_axis z/y")
        v = v * UNIT_SCALE[unit]
        if up_axis == "y":
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
        return v, f, fl, names, notes
    v, f, fl, names = import_obj(path, unit=unit, up_axis=up_axis)
    return v, f, fl, names, []


def _weld(v):
    q = np.round(v / WELD_TOL).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.reshape(-1)


def _roots(n, a, b):
    """Connected components of the graph (range(n), edges a-b), vectorized: hook every
    edge's larger label onto its smaller one, then jump pointers. Label = smallest node."""
    lab = np.arange(n)
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    while True:
        la, lb = lab[a], lab[b]
        m = la != lb
        if not m.any():
            return lab
        np.minimum.at(lab, np.maximum(la[m], lb[m]), np.minimum(la[m], lb[m]))
        while True:
            nxt = lab[lab]
            if (nxt == lab).all():
                break
            lab = nxt


def _components(nf, a, b):
    """Faces connected through shared edges (pairs a-b) -> component label per face."""
    return _roots(nf, a, b)


def _orientation(nf, fa, sa, fb, sb):
    """Per-face sign (+1/-1) making each component consistently oriented: across a
    manifold edge, two faces must traverse it in opposite directions. Faces and their
    flipped copies form a doubled graph; a face keeps its winding when it shares a
    component with the smallest face of its own (orientable) part."""
    same = sa == sb
    lab = _roots(2 * nf, np.concatenate([fa, fa + nf]),
                 np.concatenate([np.where(same, fb + nf, fb), np.where(same, fb, fb + nf)]))
    return np.where(lab[:nf] < lab[nf:], 1.0, -1.0)


def triangle_areas(tri):
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def voxel_closure(v, f):
    """Flood the outside air on a voxel grid from beyond the model: what it cannot
    reach is enclosed. Robust to T-junctions and to gaps smaller than dx, like the
    wave solver's own grid. Returns the closure (closed, volume, grid)."""
    from auralis.engine.fdtd import voxelize
    from auralis.engine.project import build_scene
    tri = v[f]
    lo, hi = v.min(axis=0), v.max(axis=0)
    bbv = float(np.prod(np.maximum(hi - lo, 1e-6)))
    dx = max(0.1, (bbv / VOXEL_BUDGET) ** (1.0 / 3.0))
    scene = build_scene({"geometry": {"triangles": tri, "tri_mat": np.zeros(len(tri), np.int32)},
                         "materials": {"m": {"absorption": [0.1] * 7}}}, "")
    g = voxelize(scene, dx, lo - 2 * dx, margin_cells=2)
    interior = ~g.solid & ~g.air
    n = int(interior.sum())
    enclosed = n * dx ** 3
    # interior cells stop short of every surface (solid band of 0.87 dx around it):
    # add back ~0.5 dx x area, coefficient measured on a T-junction box (tests)
    volume = enclosed + 0.5 * dx * float(triangle_areas(tri).sum()) if n else 0.0
    return {"closed": bool(enclosed >= CLOSED_FRACTION * bbv), "volume": volume, "dx": float(dx),
            "origin": np.asarray(g.origin, float), "interior": interior,
            "exterior": np.asarray(g.air, dtype=bool)}


def inside_grid(points, closure):
    """Point in the enclosed air of a voxel closure (a cell or one of its neighbours:
    points near a wall fall in the solid band)."""
    P = np.asarray(points, float).reshape(-1, 3)
    grid = closure["interior"]
    idx = np.rint((P - closure["origin"]) / closure["dx"]).astype(int)
    out = np.zeros(len(P), dtype=bool)
    for k, (i, j, l) in enumerate(idx):
        sl = tuple(slice(max(c - 1, 0), min(c + 2, n)) for c, n in zip((i, j, l), grid.shape))
        out[k] = all(0 <= c < n for c, n in zip((i, j, l), grid.shape)) and bool(grid[sl].any())
    return out


def detail_rule(transition_hz=None, source=None):
    """Threshold of short edges and small gaps: lambda/4 at the wave band's transition
    (c = 343 m/s), or the fixed SHORT_EDGE when none is known."""
    if transition_hz:
        fc = float(transition_hz)
        return {"rule": "lambda/4", "threshold_m": round(C_RULE / (4.0 * fc), 5),
                "transition_hz": round(fc, 1),
                "wave_dx_m": round(C_RULE / (CELLS_PER_WAVELENGTH * fc), 5),
                "source": source or "transition given with the check"}
    return {"rule": "fixed", "threshold_m": SHORT_EDGE, "transition_hz": None, "wave_dx_m": None,
            "source": source or "no transition frequency known: fixed threshold"}


def _mm(x):
    return "%.0f mm" % (1000 * x) if x < 0.1 else "%.1f cm" % (100 * x)


def _detail_text(detail):
    if detail["rule"] == "lambda/4":
        return "below λ/4 = %s at the %.0f Hz transition" % (
            _mm(detail["threshold_m"]), detail["transition_hz"])
    return "below %s (fixed threshold: no transition frequency known)" % _mm(detail["threshold_m"])


_GOLD = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _hash_rows(tag, rows):
    """Stable 64-bit hash of integer rows (m, k): a splitmix64 chain over the columns."""
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows.reshape(len(rows), -1) if rows.size else np.zeros((len(rows), 1), np.int64)
    h = np.full(len(rows), zlib.crc32(tag.encode()), dtype=np.uint64)
    for col in rows.T:
        h = (h ^ col.astype(np.uint64)) + _GOLD
        h = (h ^ (h >> np.uint64(30))) * _MIX1
        h = (h ^ (h >> np.uint64(27))) * _MIX2
        h = h ^ (h >> np.uint64(31))
    return h


def _hash_text(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:16], 16)


def _q(x, step):
    return np.rint(np.asarray(x, dtype=float) / step).astype(np.int64)


def _canonical_pairs(qa, qb):
    """Rows [qa | qb] with the two points in lexicographic order (direction-free ids)."""
    d = qa - qb
    k = np.argmax(d != 0, axis=1)
    swap = d[np.arange(len(d)), k] > 0
    return np.where(swap[:, None], np.hstack([qb, qa]), np.hstack([qa, qb]))


class _Budget(Exception):
    pass


def _pack(c):
    return (c[:, 0] << 7) | (c[:, 1] << 25) | (c[:, 2] << 43)


def _cells(idx, lo, hi, origin, size, cls=None):
    """Grid cells covered by boxes idx (at most 2 per axis at their level): packed keys
    (with the box class in the low 7 bits), owners, and offset bits (bit k set: the cell
    is one past the box's lower cell on axis k)."""
    clo = np.floor((lo[idx] - origin) / size).astype(np.int64)
    up = np.floor((hi[idx] - origin) / size).astype(np.int64) > clo
    keys, own, bits = [], [], []
    for off in itertools.product((0, 1), repeat=3):
        m = np.ones(len(idx), dtype=bool)
        for k in range(3):
            if off[k]:
                m &= up[:, k]
        c = clo[m] + np.array(off, dtype=np.int64)
        keys.append(_pack(c) if cls is None else _pack(c) | cls[idx[m]])
        own.append(idx[m])
        bits.append(np.full(len(c), off[0] | off[1] << 1 | off[2] << 2, dtype=np.int8))
    return np.concatenate(keys), np.concatenate(own), np.concatenate(bits)


def _box_pairs(lo_a, hi_a, lo_b=None, hi_b=None, keep=None, budget=PAIR_BUDGET,
               cls_a=None, cls_b=None):
    """Index pairs (i, j) of intersecting axis-aligned boxes: A x B, or A x A with i < j
    when B is None. Hierarchical grid: a box lives at the level whose cells bound its
    extent (<= 8 cells), and a pair is reported once, at the coarser level of its two
    boxes, in the cell holding the lower corner of their overlap (the cell where neither
    box is past its own lower cell on any axis). `keep(i, j)` filters the candidates
    chunk by chunk; past `budget` raw candidates, raises _Budget. With classes (ints
    below 128, A x B only), only boxes of the same class meet."""
    self_join = lo_b is None
    if self_join:
        lo_b, hi_b = lo_a, hi_a
    out_i, out_j = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    if len(lo_a) == 0 or len(lo_b) == 0:
        return out_i[0], out_j[0]
    origin = np.minimum(lo_a.min(axis=0), lo_b.min(axis=0))
    span = float((np.maximum(hi_a.max(axis=0), hi_b.max(axis=0)) - origin).max())
    ext_a = (hi_a - lo_a).max(axis=1)
    ext_b = ext_a if self_join else (hi_b - lo_b).max(axis=1)
    s0 = max(float(np.percentile(np.concatenate([ext_a, ext_b]), 10)), span / 2.0 ** 17, 1e-9)

    def level(ext):
        return np.ceil(np.log2(np.maximum(ext * (1 + 1e-9) / s0, 1.0))).astype(np.int64)

    lev_a = level(ext_a)
    lev_b = lev_a if self_join else level(ext_b)
    cols = [[np.ascontiguousarray(x[:, k]) for k in range(3)] for x in (lo_a, hi_a, lo_b, hi_b)]
    total = 0
    for lv in range(int(max(lev_a.max(), lev_b.max())) + 1):
        size = s0 * 2.0 ** lv
        stored_b = np.flatnonzero(lev_b == lv)
        jobs = []
        if len(stored_b):
            jobs.append((np.flatnonzero(lev_a <= lv), lo_a, hi_a, stored_b, lo_b, hi_b, 0, False))
        if not self_join:
            stored_a = np.flatnonzero(lev_a == lv)
            if len(stored_a):
                jobs.append((np.flatnonzero(lev_b < lv), lo_b, hi_b, stored_a, lo_a, hi_a, 2, True))
        for q_idx, qlo, qhi, s_idx, slo, shi, ci, swapped in jobs:
            if not len(q_idx):
                continue
            qlo_c, qhi_c, slo_c, shi_c = cols[ci], cols[ci + 1], cols[2 - ci], cols[3 - ci]
            qc, sc = (cls_a, cls_b) if not swapped else (cls_b, cls_a)
            kq, oq, bq = _cells(q_idx, qlo, qhi, origin, size, qc)
            ks, os_, bs = _cells(s_idx, slo, shi, origin, size, sc)
            order = np.argsort(ks, kind="stable")
            ks, os_, bs = ks[order], os_[order], bs[order]
            left = np.searchsorted(ks, kq, "left")
            cnt = np.searchsorted(ks, kq, "right") - left
            total += int(cnt.sum())
            if total > budget:
                raise _Budget(total)
            csum = np.cumsum(cnt)
            a0 = 0
            while a0 < len(kq):
                a1 = max(int(np.searchsorted(csum, csum[a0] - cnt[a0] + PAIR_CHUNK, "right")), a0 + 1)
                c = cnt[a0:a1]
                n = int(c.sum())
                if n:
                    pos = np.repeat(left[a0:a1] - (np.cumsum(c) - c), c) + np.arange(n)
                    m = (np.repeat(bq[a0:a1], c) & bs[pos]) == 0        # the reference cell
                    qi, sj = np.repeat(oq[a0:a1], c)[m], os_[pos[m]]
                    if self_join:
                        m = (qi != sj) & ~((lev_a[qi] == lv) & (qi > sj))
                        qi, sj = qi[m], sj[m]
                    m = np.ones(len(qi), dtype=bool)
                    for k in range(3):
                        m &= (qlo_c[k][qi] <= shi_c[k][sj]) & (slo_c[k][sj] <= qhi_c[k][qi])
                    i, j = (sj[m], qi[m]) if swapped else (qi[m], sj[m])
                    if self_join:
                        i, j = np.minimum(i, j), np.maximum(i, j)
                    if keep is not None and len(i):
                        k = keep(i, j)
                        i, j = i[k], j[k]
                    out_i.append(i)
                    out_j.append(j)
                a0 = a1
    return np.concatenate(out_i), np.concatenate(out_j)


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def _plane_basis(n):
    a = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = _unit(np.cross(n, a))
    return u, np.cross(n, u)


def _clip_area(P, Q):
    """Common area of 2D triangles P[i] and Q[i], (m, 3, 2): Sutherland-Hodgman clip of
    P by the three edges of Q, vectorized over the pairs."""
    m = len(P)
    rows = np.arange(m)
    sq = ((Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
          - (Q[:, 1, 1] - Q[:, 0, 1]) * (Q[:, 2, 0] - Q[:, 0, 0]))
    Q = np.where((sq < 0)[:, None, None], Q[:, ::-1], Q)          # counter-clockwise
    poly = np.zeros((m, 9, 2))
    poly[:, :3] = P
    cnt = np.full(m, 3)
    for k in range(3):
        a, b = Q[:, k], Q[:, (k + 1) % 3]
        ab = b - a
        side = ab[:, None, 0] * (poly[..., 1] - a[:, None, 1]) - ab[:, None, 1] * (poly[..., 0] - a[:, None, 0])
        new = np.zeros_like(poly)
        ncnt = np.zeros(m, dtype=np.int64)
        for j in range(int(cnt.max())):
            valid = j < cnt
            jn = np.where(j + 1 < cnt, j + 1, 0)
            p, q = poly[:, j], poly[rows, jn]
            sp, sn = side[:, j], side[rows, jn]
            pin = sp >= 0
            e = valid & pin
            new[rows[e], ncnt[e]] = p[e]
            ncnt[e] += 1
            e = valid & (pin != (sn >= 0))
            den = np.where(sp - sn == 0, 1.0, sp - sn)
            x = p + (sp / den)[:, None] * (q - p)
            new[rows[e], ncnt[e]] = x[e]
            ncnt[e] += 1
        poly, cnt = new, ncnt
    idx = np.minimum(np.arange(9)[None, :], np.maximum(cnt - 1, 0)[:, None])
    pp = poly[rows[:, None], idx]
    x, y = pp[..., 0], pp[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))
    return np.where(cnt >= 3, area, 0.0)


def _overlap_area(t_ref, t_other, n_ref):
    """Area of t_other projected on the plane of t_ref (normal n_ref) shared with t_ref.
    Pairs separated along an edge normal of either (separating axes: neighbours that only
    share an edge or a vertex) are 0 without clipping."""
    u, w = _plane_basis(n_ref)
    o = t_ref[:, 0, :]

    def flat(t):
        return [((t[:, k] - o) * u).sum(axis=1) for k in range(3)], \
               [((t[:, k] - o) * w).sum(axis=1) for k in range(3)]
    (px, py), (qx, qy) = flat(t_other), flat(t_ref)
    depth = np.full(len(o), np.inf)
    for X, Y in ((px, py), (qx, qy)):
        for k in range(3):
            ex, ey = X[(k + 1) % 3] - X[k], Y[(k + 1) % 3] - Y[k]
            nn = np.maximum(np.hypot(ex, ey), 1e-30)
            ax, ay = -ey / nn, ex / nn
            pp = [px[i] * ax + py[i] * ay for i in range(3)]
            qq = [qx[i] * ax + qy[i] * ay for i in range(3)]
            hi = np.minimum(np.maximum(np.maximum(pp[0], pp[1]), pp[2]),
                            np.maximum(np.maximum(qq[0], qq[1]), qq[2]))
            lo = np.maximum(np.minimum(np.minimum(pp[0], pp[1]), pp[2]),
                            np.minimum(np.minimum(qq[0], qq[1]), qq[2]))
            depth = np.minimum(depth, hi - lo)
    out = np.zeros(len(o))
    k = np.flatnonzero(depth > 1e-7)
    if len(k):
        P = np.stack([np.stack([px[i][k], py[i][k]], axis=1) for i in range(3)], axis=1)
        Q = np.stack([np.stack([qx[i][k], qy[i][k]], axis=1) for i in range(3)], axis=1)
        out[k] = _clip_area(P, Q)
    return out


def _closest_on_triangles(p, a, b, c):
    """Closest points of triangles (a, b, c) to points p, all (m, 3) (Ericson 5.1.5)."""
    ab, ac = b - a, c - a
    d1, d2 = _dot(ab, p - a), _dot(ac, p - a)
    d3, d4 = _dot(ab, p - b), _dot(ac, p - b)
    d5, d6 = _dot(ab, p - c), _dot(ac, p - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def ratio(x, y):
        return x / np.where(y == 0, 1.0, y)
    den = va + vb + vc
    q = a + ab * ratio(vb, den)[:, None] + ac * ratio(vc, den)[:, None]
    for m, base, edge, t in (
            ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b, c - b, ratio(d4 - d3, d4 - d3 + d5 - d6)),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a, ac, ratio(d2, d2 - d6)),
            ((d6 >= 0) & (d5 <= d6), c, 0 * ab, 0 * d1),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a, ab, ratio(d1, d1 - d3)),
            ((d3 >= 0) & (d4 <= d3), b, 0 * ab, 0 * d1),
            ((d1 <= 0) & (d2 <= 0), a, 0 * ab, 0 * d1)):
        q[m] = base[m] + edge[m] * t[m, None]                 # later regions take priority
    return q


def _labels(n, a, b):
    """Connected components of the graph (range(n), edges a-b): one label per node."""
    return _roots(n, a, b)


def _groups(faces_a, faces_b, extra_a, extra_b):
    """Clusters of face pairs: pairs linked through a face, or through mesh edges
    (extra_a - extra_b) between involved faces. -> (faces, pair label per pair)."""
    faces = np.unique(np.concatenate([faces_a, faces_b]))
    pos = np.full(int(faces.max()) + 1 if len(faces) else 1, -1, dtype=np.int64)
    pos[faces] = np.arange(len(faces))
    ok = (extra_a < len(pos)) & (extra_b < len(pos))
    ea, eb = extra_a[ok], extra_b[ok]
    ok = (pos[ea] >= 0) & (pos[eb] >= 0)
    lab = _labels(len(faces), np.concatenate([pos[faces_a], pos[ea[ok]]]),
                  np.concatenate([pos[faces_b], pos[eb[ok]]]))
    return faces, pos, lab


def _dilate(g):
    d = g.copy()
    for ax in range(3):
        s = d.copy()
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        s[tuple(a)] |= d[tuple(b)]
        s[tuple(b)] |= d[tuple(a)]
        d = s
    return d


def _in_closure(points, closure, dilated):
    idx = np.rint((np.asarray(points, float) - closure["origin"]) / closure["dx"]).astype(np.int64)
    shape = np.array(dilated.shape)
    ok = ((idx >= 0) & (idx < shape)).all(axis=1)
    out = np.zeros(len(idx), dtype=bool)
    out[ok] = dilated[tuple(idx[ok].T)]
    return out


def _air_sides(tri, nrm, cidx, flip, c_open, shell_k, room_closed, closure, vol_c, lo_c, hi_c):
    """Side of each face's air along its own normal: +1, -1, or 0 (both sides, or unknown).
    Closed components by their signed volume (outward; the room shell inward); a face of
    another component with thickness that lies on one side of the face's plane (a seat
    with an open backrest) has its air away from it; other faces are probed on both
    sides in the voxel closure when there is one (room side inside, outside air beyond)."""
    side = np.zeros(len(tri), dtype=np.int8)
    closed = ~c_open[cidx] & (np.abs(vol_c)[cidx] > 1e-9)
    out = (np.sign(vol_c)[cidx] * flip).astype(np.int8)
    room = (cidx == shell_k) & room_closed
    side[closed] = np.where(room[closed], -out[closed], out[closed])
    rest = np.flatnonzero(~closed & (cidx != shell_k))
    if len(rest):
        k = cidx[rest]
        n = nrm[rest]
        mid, half = (lo_c[k] + hi_c[k]) / 2, (hi_c[k] - lo_c[k]) / 2
        p = _dot(n, tri[rest, 0])
        reach = (np.abs(n) * half).sum(axis=1)
        top, bot = _dot(n, mid) + reach, _dot(n, mid) - reach
        above = (bot >= p - PLANE_TOL) & (top > p + 0.01)
        below = (top <= p + PLANE_TOL) & (bot < p - 0.01)
        side[rest[above]] = -1
        side[rest[below]] = 1
    rest = np.flatnonzero(~closed & (side == 0))
    if closure is not None and len(rest):
        dil = _dilate(closure["interior"])
        c = tri[rest].mean(axis=1)
        h = 1.5 * closure["dx"]
        up = _in_closure(c + h * nrm[rest], closure, dil)
        dn = _in_closure(c - h * nrm[rest], closure, dil)
        if closure.get("exterior") is not None:
            ext = _dilate(closure["exterior"])
            up_x = _in_closure(c + h * nrm[rest], closure, ext)
            dn_x = _in_closure(c - h * nrm[rest], closure, ext)
            side[rest] = np.where(up & dn_x & ~up_x, 1, np.where(dn & up_x & ~dn_x, -1, 0))
        else:
            side[rest] = np.where(up & ~dn, 1, np.where(dn & ~up, -1, 0))
    return side


def _normal_classes(n, tol):
    """Direction classes of unit normals (bins of 0.5 per component, < 128): the exact
    class of each, and every (index, class) whose bin lies within `tol` of n or of -n."""
    def cls(b):
        return ((b[:, 0] + 2) * 25 + (b[:, 1] + 2) * 5 + (b[:, 2] + 2)).astype(np.int64)
    exact = cls(np.rint(n / 0.5).astype(np.int64))
    idx, out = [], []
    for sg in (1.0, -1.0):
        x = sg * n / 0.5
        b = np.rint(x).astype(np.int64)
        fr = x - b
        step = np.where(fr > 0.5 - tol / 0.5, 1, np.where(fr < -0.5 + tol / 0.5, -1, 0))
        for off in itertools.product((0, 1), repeat=3):
            m = np.ones(len(n), dtype=bool)
            for k in range(3):
                if off[k]:
                    m &= step[:, k] != 0
            bb = b[m] + step[m] * np.array(off, dtype=np.int64)
            ok = (np.abs(bb) <= 2).all(axis=1)
            idx.append(np.flatnonzero(m)[ok])
            out.append(cls(bb[ok]))
    idx, out = np.concatenate(idx), np.concatenate(out)
    u = np.unique(idx * 128 + out)
    return exact, u // 128, u % 128


def _hugging_boxes(tri, good):
    """Boxes that hug the triangles: one larger than the target (twice the mean triangle
    size) is bisected along its longest edge until it fits, so a big wall gets a sheet of
    small boxes instead of one box that every small triangle nearby would meet.
    -> (lo, hi, parent face)."""
    t, par = tri[good], good
    if not len(t):
        z = np.zeros((0, 3))
        return z, z, good
    ext = (t.max(axis=1) - t.min(axis=1)).max(axis=1)
    target = max(float(np.median(ext)), 2.0 * math.sqrt(float(triangle_areas(t).sum()) / len(t)))
    los, his, pars = [], [], []
    for _ in range(64):
        lo, hi = t.min(axis=1), t.max(axis=1)
        fit = (hi - lo).max(axis=1) <= target
        los.append(lo[fit])
        his.append(hi[fit])
        pars.append(par[fit])
        t, par = t[~fit], par[~fit]
        if not len(t):
            break
        ln = np.stack([np.linalg.norm(t[:, (k + 1) % 3] - t[:, k], axis=1) for k in range(3)], axis=1)
        k = np.argmax(ln, axis=1)
        r = np.arange(len(t))
        p0, p1, p2 = t[r, k], t[r, (k + 1) % 3], t[r, (k + 2) % 3]
        mid = (p0 + p1) / 2
        t = np.concatenate([np.stack([p0, mid, p2], axis=1), np.stack([mid, p1, p2], axis=1)])
        par = np.concatenate([par, par])
    if len(t):
        los.append(t.min(axis=1))
        his.append(t.max(axis=1))
        pars.append(par)
    return np.concatenate(los), np.concatenate(his), np.concatenate(pars)


def _unique_pairs(i, j):
    if not len(i):
        return i, j
    n = int(max(i.max(), j.max())) + 1
    u = np.unique(i.astype(np.int64) * n + j)
    return u // n, u % n


def _near_faces(qlo, qhi, hug, keep=None, pad=0.0):
    """(query, face) pairs whose boxes meet, deduplicated over the hugging boxes."""
    lo, hi, par = hug
    i, j = _box_pairs(qlo, qhi, lo - pad, hi + pad,
                      keep=None if keep is None else (lambda i, j: keep(i, par[j])))
    return _unique_pairs(i, par[j])


def _overlaps(tri, nrm, areas, fw, side, adj, hug, comp_area, notes):
    """Coplanar overlapping faces, clustered: [(faces, small faces, area, duplicate, contact)]."""
    cos = math.cos(math.radians(COPLANAR_DEG))
    lo, hi, par = hug
    exact, fi, fc = _normal_classes(nrm[par], 1.5 * math.radians(COPLANAR_DEG))

    def keep(i, j):
        pi, pj = par[i], par[fi[j]]
        return (pi != pj) & (np.abs(_dot(nrm[pi], nrm[pj])) >= cos)
    try:
        i, j = _box_pairs(lo - PLANE_TOL, hi + PLANE_TOL, lo[fi] - PLANE_TOL, hi[fi] + PLANE_TOL,
                          keep=keep, cls_a=exact, cls_b=fc)
    except _Budget as exc:
        notes.append("overlapping faces NOT checked: %d candidate pairs, above the budget of "
                     "%d (model too dense)" % (exc.args[0], PAIR_BUDGET))
        return []
    i, j = par[i], par[fi[j]]
    i, j = _unique_pairs(np.minimum(i, j), np.maximum(i, j))
    # the smaller side: the face of the smaller component (a seat on the floor), else the
    # smaller face (a patch inside a wall)
    ci, cj = comp_area[i], comp_area[j]
    big = (ci > cj) | ((ci == cj) & (areas[i] >= areas[j]))
    L, s = np.where(big, i, j), np.where(big, j, i)
    off = _dot(nrm[L], tri[L, 0])
    ok = np.ones(len(L), dtype=bool)
    for k in range(3):
        ok &= np.abs(_dot(nrm[L], tri[s, k]) - off) <= PLANE_TOL
    L, s = L[ok], s[ok]
    area = _overlap_area(tri[L], tri[s], nrm[L]) if len(L) else np.zeros(0)
    ok = area > np.maximum(OVERLAP_FRACTION * areas[s], 1e-10)
    L, s, area = L[ok], s[ok], area[ok]
    if not len(L):
        return []
    dup = (np.sort(fw[L], axis=1) == np.sort(fw[s], axis=1)).all(axis=1)
    # back to back: no side from which both faces are seen through air (a solid on a surface)
    rel = side[s] * np.sign(_dot(nrm[s], nrm[L])).astype(np.int8)
    contact = (side[L] != 0) & (rel != 0) & (side[L] != rel)
    # one finding per connected coplanar patch of the smaller faces (a seat on the floor,
    # a duplicated wall), with the faces it overlaps
    S = np.unique(s)
    pos = np.full(len(tri), -1, dtype=np.int64)
    pos[S] = np.arange(len(S))
    ok = (pos[adj[0]] >= 0) & (pos[adj[1]] >= 0)
    ok[ok] = np.abs(_dot(nrm[adj[0][ok]], nrm[adj[1][ok]])) >= cos
    lab = _labels(len(S), pos[adj[0][ok]], pos[adj[1][ok]])
    plab = lab[pos[s]]
    out = []
    order = np.argsort(plab, kind="stable")
    cuts = np.flatnonzero(np.r_[True, plab[order][1:] != plab[order][:-1], True])
    for a0, a1 in zip(cuts[:-1], cuts[1:]):
        k = order[a0:a1]
        out.append((np.unique(np.concatenate([s[k], L[k]])), np.unique(s[k]), float(area[k].sum()),
                    bool(dup[k].all()), bool(contact[k].all())))
    return out


def _slits(tri, nrm, vw, fw, ea, eb, F, thr, hug, notes):
    """Open edges facing another surface across a gap in [GAP_MIN, thr), sampled at 1/4,
    1/2 and 3/4 of their length; a sample touching anything (< GAP_MIN) is in contact.
    -> (edge rows, gap, near point, far point, target face, outward direction) or None."""
    a, b = vw[ea], vw[eb]
    t = b - a
    ok = (np.linalg.norm(t, axis=1) > GAP_MIN) & (np.linalg.norm(nrm[F], axis=1) > 0.5)
    rows = np.flatnonzero(ok)
    if not len(rows):
        return None
    a, t, Fr = a[rows], t[rows], F[rows]
    third = np.argmax((fw[Fr] != ea[rows, None]) & (fw[Fr] != eb[rows, None]), axis=1)
    apex = vw[fw[Fr, third]]
    o = _unit(np.cross(t, nrm[Fr]))
    o = np.where((_dot(o, apex - (a + t / 2)) > 0)[:, None], -o, o)      # away from its face
    fracs = np.array([0.25, 0.5, 0.75])
    P = (a[:, None, :] + fracs[None, :, None] * t[:, None, :]).reshape(-1, 3)
    E = np.repeat(np.arange(len(rows)), 3)

    best_c = np.full(len(P), np.inf)
    best_g = np.full(len(P), np.inf)
    tgt = np.full(len(P), -1)
    far = np.zeros((len(P), 3))
    # growing search radius: a sample is settled once it touches something, or once its
    # nearest surface in front lies within the radius searched
    live = np.arange(len(P))
    r = min(thr, max(4 * GAP_MIN, float(np.median((hug[1] - hug[0]).max(axis=1)))))
    while len(live):
        sub = live

        def keep(i, j):
            return ~(fw[j][:, :, None] == fw[Fr[E[sub[i]]]][:, None, :]).any(axis=(1, 2))
        try:
            i, j = _near_faces(P[sub] - r, P[sub] + r, hug, keep=keep)
        except _Budget as exc:
            notes.append("gaps at open edges NOT checked: %d candidate pairs, above the budget "
                         "of %d" % (exc.args[0], PAIR_BUDGET))
            return None
        i = sub[i]
        q = _closest_on_triangles(P[i], tri[j, 0], tri[j, 1], tri[j, 2])
        dv = q - P[i]
        dist = np.linalg.norm(dv, axis=1)
        np.minimum.at(best_c, i, dist)
        front = (_dot(dv, o[E[i]]) >= 0.5 * dist) & (dist >= GAP_MIN)
        i, j, q, dist = i[front], j[front], q[front], dist[front]
        order = np.lexsort((dist, i))
        first = order[np.r_[True, i[order][1:] != i[order][:-1]]] if len(order) else order
        best = dist[first] < best_g[i[first]]
        first = first[best]
        best_g[i[first]], tgt[i[first]], far[i[first]] = dist[first], j[first], q[first]
        if r >= thr:
            break
        live = live[(best_c[live] >= GAP_MIN) & (best_g[live] > r)]
        r = min(thr, 2 * r)
    gap_s = np.where((best_c >= GAP_MIN) & (best_g < thr), best_g, np.inf).reshape(-1, 3)
    gap = gap_s.min(axis=1)
    hit = np.flatnonzero(np.isfinite(gap))
    if not len(hit):
        return None
    ps = hit * 3 + np.argmin(gap_s[hit], axis=1)
    return rows[hit], gap[hit], P[ps], far[ps], tgt[ps], o[hit]


def _ray_triangles(o, d, tri):
    """Distance t along rays o + t d to triangles tri (m, 3, 3), inf when missed."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = _dot(e1, p)
    ok = np.abs(det) > 1e-14
    inv = 1.0 / np.where(ok, det, 1.0)
    s = o - tri[:, 0]
    u = _dot(s, p) * inv
    q = np.cross(s, e1)
    v = _dot(d, q) * inv
    t = _dot(e2, q) * inv
    return np.where(ok & (u >= 0) & (v >= 0) & (u + v <= 1), t, np.inf)


def _air_layers(tri, nrm, areas, fw, side, adj, thr, hug, notes):
    """Thin air layers between parallel faces: a probe from each face centroid along its
    air side(s), as long as the threshold; its first hit must be a near-parallel face that
    sees it back through air. Clustered: [(faces, min gap, max gap, area, near, far)]."""
    good = np.flatnonzero(areas > 1e-8)
    src, dirs = [], []
    for sg in (1, -1):
        k = good[(side[good] == 0) | (side[good] == sg)]
        src.append(k)
        dirs.append(sg * nrm[k])
    src, d = np.concatenate(src), np.concatenate(dirs)
    c = tri[src].mean(axis=1)
    # one probe per cell of thr/4 and direction class (the largest face): enough for a
    # layer at least as wide as thin, and bounded in densely meshed regions
    key = np.column_stack([np.floor(c / max(thr / 4, GAP_MIN)).astype(np.int64),
                           np.rint(d / 0.5).astype(np.int64)])
    order = np.lexsort((-areas[src],) + tuple(key.T[::-1]))
    ks = key[order]
    first = order[np.r_[True, (ks[1:] != ks[:-1]).any(axis=1)]] if len(order) else order
    src, d, c = src[first], d[first], c[first]
    k = max(1, int(math.ceil(thr / max(float((hug[1] - hug[0]).max(axis=1).mean()), 1e-6))))
    ends = c[:, None, :] + (thr * np.linspace(0.0, 1.0, k + 1))[None, :, None] * d[:, None, :]
    plo = np.minimum(ends[:, :-1], ends[:, 1:]).reshape(-1, 3)
    phi = np.maximum(ends[:, :-1], ends[:, 1:]).reshape(-1, 3)
    probe = np.repeat(np.arange(len(src)), k)
    try:
        i, j = _near_faces(plo, phi, hug, keep=lambda i, j: j != src[probe[i]])
    except _Budget as exc:
        notes.append("thin air layers NOT checked: %d candidate pairs, above the budget of %d "
                     "(model too dense for λ/4 = %.2f m)" % (exc.args[0], PAIR_BUDGET, thr))
        return []
    i, j = _unique_pairs(probe[i], j)
    dist = _ray_triangles(c[i], d[i], tri[j])
    ok = np.isfinite(dist) & (dist > 1e-9) & (dist < thr)
    i, j, dist = i[ok], j[ok], dist[ok]
    order = np.lexsort((dist, i))
    first = order[np.r_[True, i[order][1:] != i[order][:-1]]] if len(order) else order
    i, j, dist = i[first], j[first], dist[first]            # first hit of each probe
    s = src[i]
    cos = math.cos(math.radians(GAP_DEG))
    back = side[j] * np.sign(_dot(nrm[j], d[i])).astype(np.int8)    # its air side, along d
    ok = ((dist >= GAP_MIN) & (np.abs(_dot(nrm[s], nrm[j])) >= cos) & ((side[j] == 0) | (back == -1))
          & ~(fw[s][:, :, None] == fw[j][:, None, :]).any(axis=(1, 2)))
    i, j, s, dist = i[ok], j[ok], s[ok], dist[ok]
    if not len(i):
        return []
    par = np.abs(_dot(nrm[adj[0]], nrm[adj[1]])) >= cos
    faces, pos, lab = _groups(s, j, adj[0][par], adj[1][par])
    plab = lab[pos[s]]
    out = []
    for g in np.unique(plab):
        k = np.flatnonzero(plab == g)
        ref = d[i[k[0]]]
        below = d[i[k]] @ ref > 0                           # probe going the reference way
        lower = np.unique(np.concatenate([s[k][below], j[k][~below]]))
        upper = np.unique(np.concatenate([s[k][~below], j[k][below]]))
        area = min(float(areas[lower].sum()), float(areas[upper].sum()))
        gmin = float(dist[k].min())
        if area < gmin ** 2:                     # edge to edge proximity, not a layer
            continue
        b = k[np.argmin(dist[k])]
        near = c[i[b]]
        out.append((faces[lab == g], gmin, float(dist[k].max()), area, near, near + dist[b] * d[i[b]]))
    return out


def _report(tables, fl, names, exclusions):
    """Findings listed per kind (active first), exclusions applied, exact counts."""
    ex = exclusions or {}
    ex_ids = set(ex.get("findings") or [])
    ex_layers = [n for n in (ex.get("layers") or []) if n in names]
    face_ex = np.isin(fl, [names.index(n) for n in ex_layers]) if ex_layers else None
    findings, excluded, matched, blocked = [], {}, set(), 0
    for t in tables:
        m = len(t["hash"])
        ids = ["%s-%016x" % (t["tag"], h) for h in t["hash"].tolist()]
        by_id = np.array([x in ex_ids for x in ids], dtype=bool) if ex_ids else np.zeros(m, bool)
        by_layer = t["hit"](face_ex) if face_ex is not None and m else np.zeros(m, bool)
        matched.update(ids[k] for k in np.flatnonzero(by_id))
        excl = by_id | by_layer
        excluded[t["kind"]] = int(excl.sum())
        blocked += int((excl & (t["sev"] == 2)).sum())
        active = np.flatnonzero(~excl)
        active = active[np.argsort(-t["sev"][active], kind="stable")]
        listed = np.concatenate([active[:MAX_PER_KIND], np.flatnonzero(by_id),
                                 np.flatnonzero(by_layer & ~by_id)[:MAX_PER_KIND // 4]])
        for k in listed.tolist():
            item = {"id": ids[k], "kind": t["kind"], "severity": SEVERITIES[t["sev"][k]]}
            item.update(t["make"](k))
            item["layers"] = sorted({names[x] for x in fl[t["faces"](k)].tolist()})
            item["excluded"] = bool(excl[k])
            if excl[k]:
                item["excluded_by"] = "finding" if by_id[k] else "layer"
            findings.append(item)
    return findings, excluded, {"findings": sorted(matched), "layers": ex_layers,
                                "unmatched": sorted(ex_ids - matched)}, blocked


def _gap_ids(gaps, fl, names):
    """Ids of gap findings: kind, layers and the gap's two ends at 1 cm, order-free."""
    if not gaps:
        return np.zeros(0, dtype=np.uint64)
    ends = _canonical_pairs(_q(np.array([g[4] for g in gaps]), 1e-2), _q(np.array([g[5] for g in gaps]), 1e-2))
    lay = np.array([zlib.crc32(("%s|" % g[0] + ",".join(sorted({names[x] for x in fl[g[1]].tolist()})))
                               .encode()) for g in gaps], dtype=np.int64)
    return _hash_rows("sg", np.column_stack([lay, ends]))


def _csr_hit(owner, starts):
    def hit(rows):
        return lambda face_mask: np.logical_or.reduceat(face_mask[owner], starts)[rows]
    return hit


def analyse(v, f, fl, names, closure=None, closure_out=None, transition_hz=None,
            transition_source=None, exclusions=None):
    """Layer stats + closure check; returns (summary dict, closed-solid face mask).
    A model is closed if its edges match (topology) or, failing that, if it encloses
    air at the voxel scale (`closure`: reuse a room's voxel closure; `closure_out`
    receives the one computed here). Short edges and small gaps are judged against
    lambda/4 at `transition_hz` (the fixed SHORT_EDGE without it); `exclusions`
    ({"findings": [ids], "layers": [names]}) mark findings as known, they never change
    `watertight` nor what the solvers accept."""
    tri = v[f]
    areas = triangle_areas(tri)
    nrm = _unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    w = _weld(v)
    fw = w[f]
    e = np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]])
    sgn = np.where(e[:, 0] < e[:, 1], 1, -1)        # traversal direction in its face
    e.sort(axis=1)
    owner = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner, sgn = e[order], owner[order], sgn[order]
    brk = np.ones(len(e), dtype=bool)
    brk[1:] = (e[1:] != e[:-1]).any(axis=1)
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], len(e))
    counts = ends - starts
    grp = np.repeat(starts, counts)                   # first entry of each edge's group
    rest = ~brk
    comp = _components(len(f), owner[grp[rest]], owner[rest])
    two = starts[counts == 2]
    flip = _orientation(len(f), owner[two], sgn[two], owner[two + 1], sgn[two + 1])

    open_mask = counts == 1
    labels, cidx = np.unique(comp, return_inverse=True)
    c_open = np.zeros(len(labels), dtype=bool)
    c_open[cidx[owner[starts[open_mask]]]] = True
    lo_c = np.full((len(labels), 3), np.inf)
    hi_c = np.full((len(labels), 3), -np.inf)
    np.minimum.at(lo_c, cidx, tri.min(axis=1))
    np.maximum.at(hi_c, cidx, tri.max(axis=1))
    shell_k = int(np.argmax(np.prod(np.maximum(hi_c - lo_c, 1e-9), axis=1)))
    shell = labels[shell_k]
    topo_closed = not c_open[shell_k]
    closed = ~c_open[cidx]
    vol_c = np.bincount(cidx, weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(
        tri[:, 1], tri[:, 2])) * flip, minlength=len(labels)) / 6.0   # signed, per component
    solids = float(np.abs(vol_c[~c_open]).sum())

    lo, hi = v.min(axis=0), v.max(axis=0)
    bbv = float(np.prod(np.maximum(hi - lo, 1e-9)))
    topo_volume = None
    if topo_closed:
        topo_volume = 2 * abs(float(vol_c[shell_k])) - solids
        if topo_volume < CLOSED_FRACTION * bbv:
            topo_closed = False         # a small closed part (stage block...) is not the room
    if not topo_closed and closure is None:
        closure = voxel_closure(v, f)
    if closure_out is not None and closure is not None:
        closure_out.update(closure)
    watertight = topo_closed or bool(closure and closure["closed"])

    vw = np.zeros((w.max() + 1, 3))
    vw[w] = v
    detail = detail_rule(transition_hz, transition_source)
    thr = C_RULE / (4.0 * float(transition_hz)) if transition_hz else SHORT_EDGE
    dx_wave = C_RULE / (CELLS_PER_WAVELENGTH * float(transition_hz)) if transition_hz else None
    notes = []
    two = starts[counts == 2]
    aa, ab = [owner[two]], [owner[two + 1]]
    for s0, c in zip(starts[counts > 2].tolist(), counts[counts > 2].tolist()):
        g = owner[s0:s0 + min(c, 8)].tolist()
        pr = list(itertools.combinations(g, 2))
        aa.append(np.array([x for x, _ in pr], dtype=np.int64))
        ab.append(np.array([y for _, y in pr], dtype=np.int64))
    adj = (np.concatenate(aa), np.concatenate(ab))    # faces sharing an edge
    side = _air_sides(tri, nrm, cidx, flip, c_open, shell_k, topo_closed, closure, vol_c,
                      lo_c, hi_c)
    hit = _csr_hit(owner, starts)
    ue = e[starts]
    pa, pb = vw[ue[:, 0]], vw[ue[:, 1]]
    length = np.linalg.norm(pb - pa, axis=1)
    qedge = _canonical_pairs(_q(pa, 1e-3), _q(pb, 1e-3))
    tables = []

    def edge_table(kind, rows, sev, message):
        tables.append({
            "kind": kind, "tag": TAGS[kind], "hash": _hash_rows(TAGS[kind], qedge[rows]),
            "sev": np.asarray(sev, dtype=np.int64), "hit": hit(rows),
            "faces": lambda k: owner[starts[rows[k]]:ends[rows[k]]],
            "make": lambda k: {"message": message(rows[k]),
                               "segment": [pa[rows[k]].round(6).tolist(), pb[rows[k]].round(6).tolist()],
                               "length_m": round(float(length[rows[k]]), 6)}})

    rows = np.flatnonzero(open_mask)
    on_shell = comp[owner[starts[rows]]] == shell
    edge_table("open_edge", rows, np.where(on_shell, 0 if watertight else 2, 0), lambda r: (
        ("Open edge on the outer shell: the model is not closed here (wave solver only if it "
         "is declared an open scene)") if comp[owner[starts[r]]] == shell and not watertight else
        ("Edge not matched vertex to vertex (T-junction or gap below %.2f m): the room is still "
         "closed at that voxel scale" % closure["dx"]) if comp[owner[starts[r]]] == shell else
        "Open edge of an inner surface (allowed: thin panel, table top...)"))
    rows = np.flatnonzero(counts > 2)
    edge_table("nonmanifold_edge", rows, np.ones(len(rows)),
               lambda r: "Edge shared by %d faces" % counts[r])
    # feature edges only: an edge inside a flat, single-layer surface is triangulation
    f0, f1 = owner[starts], owner[np.minimum(starts + 1, len(owner) - 1)]
    flat = ((counts == 2) & (fl[f0] == fl[f1])
            & (_dot(nrm[f0], nrm[f1]) * flip[f0] * flip[f1] >= math.cos(math.radians(COPLANAR_DEG))))
    rows = np.flatnonzero((length > 0) & (length < thr) & ~flat)
    edge_table("short_edge", rows, np.zeros(len(rows)), lambda r: (
        "Edge of %s, %s: finer than the wave band needs (harmless for its fixed grid%s)"
        % (_mm(length[r]), _detail_text(detail),
           ", dx = %s" % _mm(dx_wave) if dx_wave else "")))

    degen = np.flatnonzero(areas < 1e-8)
    hv = np.sort(_hash_rows("v", _q(vw[fw[degen]], 1e-3).reshape(-1, 3)).reshape(-1, 3), axis=1)
    tables.append({
        "kind": "degenerate_triangle", "tag": TAGS["degenerate_triangle"],
        "hash": _hash_rows(TAGS["degenerate_triangle"], hv.view(np.int64)), "sev": np.ones(len(degen), np.int64),
        "hit": lambda m: m[degen], "faces": lambda k: degen[k:k + 1],
        "make": lambda k: {"message": "Degenerate triangle (zero area)",
                           "segment": [vw[fw[degen[k], 0]].round(6).tolist(),
                                       vw[fw[degen[k], 1]].round(6).tolist()]}})

    hug = _hugging_boxes(tri, np.flatnonzero(areas > 1e-8))
    comp_area = np.bincount(cidx, weights=areas)[cidx]
    groups = _overlaps(tri, nrm, areas, fw, side, adj, hug, comp_area, notes)

    def overlap_item(k):
        faces, small, area, dup, contact = groups[k]
        lay = sorted({names[x] for x in fl[faces].tolist()})
        pts = tri[small].reshape(-1, 3)
        wts = np.repeat(areas[small], 3)
        what = ("%d duplicated triangle(s)" % len(small) if dup else "Coplanar faces")
        if contact:
            msg = ("%s back to back over %.2f m² (%s): a solid resting on a surface, harmless "
                   "(no sound reaches either face there)" % (what, area, " / ".join(lay)))
        elif len(lay) > 1:
            msg = ("%s overlapping over %.2f m² (%s): the solvers see one surface whose material "
                   "is arbitrary (either layer)" % (what, area, " / ".join(lay)))
        else:
            msg = ("%s overlapping over %.2f m² (%s): one surface twice, its area counts twice "
                   "in the statistical estimates (Sabine, Eyring)" % (what, area, lay[0]))
        top = small[np.argsort(-areas[small], kind="stable")][:24]
        return {"message": msg, "segment": [pts.min(axis=0).round(6).tolist(),
                                            pts.max(axis=0).round(6).tolist()],
                "overlap": "duplicate" if dup else "partial", "contact": contact,
                "area_m2": round(area, 6), "n_triangles": int(len(faces)),
                "centroid": ((pts * wts[:, None]).sum(axis=0) / max(float(wts.sum()), 1e-30)).round(6).tolist(),
                "triangles": tri[top].round(6).tolist()}

    tables.append({
        "kind": "overlapping_faces", "tag": TAGS["overlapping_faces"],
        "hash": np.array([_hash_text("of|%s|%s" % (
            ",".join(sorted({names[x] for x in fl[g[0]].tolist()})),
            ",".join(str(x) for x in _q(tri[g[1]].reshape(-1, 3), 1e-2).min(axis=0).tolist()
                     + _q(tri[g[1]].reshape(-1, 3), 1e-2).max(axis=0).tolist())))
            for g in groups], dtype=np.uint64),
        "sev": np.array([0 if g[4] else 1 for g in groups], dtype=np.int64),
        "hit": lambda m: np.array([m[g[0]].any() for g in groups], dtype=bool),
        "faces": lambda k: groups[k][0], "make": overlap_item})

    gaps = []                                         # (kind, faces, gmin, gmax, near, far, extra)
    rows = np.flatnonzero(open_mask)
    sl = _slits(tri, nrm, vw, fw, ue[rows, 0], ue[rows, 1], owner[starts[rows]], thr, hug, notes)
    if sl is not None:
        er, gap, near, far, tgt, out_dir = sl
        F = owner[starts[rows[er]]]
        # one finding per slit: open edges sharing a vertex, or facing each other
        ends_w = np.concatenate([ue[rows[er], 0], ue[rows[er], 1]])
        ids = np.concatenate([np.arange(len(er))] * 2)
        o = np.argsort(ends_w, kind="stable")
        same_v = ends_w[o][1:] == ends_w[o][:-1]
        la, lb = ids[o][:-1][same_v], ids[o][1:][same_v]
        face_edge = {}
        for k, fc in enumerate(F.tolist()):
            face_edge.setdefault(fc, k)
        fa = [k for k, t_ in enumerate(tgt.tolist()) if t_ in face_edge]
        fb = [face_edge[tgt[k]] for k in fa]
        lab = _labels(len(er), np.concatenate([la, fa]).astype(np.int64),
                      np.concatenate([lb, fb]).astype(np.int64))
        dil = ext = None
        if closure is not None and closure.get("exterior") is not None:
            dil, ext = _dilate(closure["interior"]), _dilate(closure["exterior"])
        for g in np.unique(lab):
            k = np.flatnonzero(lab == g)
            b = k[np.argmin(gap[k])]
            leak = None
            if dil is not None:
                # room air on one side of the wall, outside air on the other
                h = closure["dx"]
                base = near[b] - 0.5 * h * out_dir[b]
                pr = [base + 1.5 * h * nrm[F[b]], base - 1.5 * h * nrm[F[b]]]
                ins, outs = _in_closure(pr, closure, dil), _in_closure(pr, closure, ext)
                leak = bool((ins[0] and outs[1] and not outs[0]) or (ins[1] and outs[0] and not outs[1]))
            faces = np.unique(np.concatenate([F[k], tgt[k][tgt[k] >= 0]]))
            gaps.append(("slit", faces, float(gap[k].min()), float(gap[k].max()), near[b], far[b],
                         {"length_m": round(float(length[rows[er[k]]].sum()), 4), "leak": leak}))
    for faces, gmin, gmax, area, near, far in _air_layers(tri, nrm, areas, fw, side, adj, thr, hug, notes):
        gaps.append(("layer", faces, gmin, gmax, near, far, {"area_m2": round(area, 4)}))

    def gap_item(k):
        kind, faces, gmin, gmax, near, far, extra = gaps[k]
        lay = " / ".join(sorted({names[x] for x in fl[faces].tolist()}))
        width = _mm(gmin) if gmax - gmin < 0.1 * gmin + 1e-4 else "%s to %s" % (_mm(gmin), _mm(gmax))
        if kind == "slit":
            msg = "Gap of %s at an open edge (%s), %s: close it unless it is an intended opening" % (
                width, lay, _detail_text(detail))
            if extra["leak"] and watertight and closure is not None and dx_wave:
                seal = SEAL_CELLS * dx_wave
                if gmin >= seal:
                    msg += (". It leads out of the room: the closure test (voxels of %s) seals it, but "
                            "the wave grid at %.0f Hz (dx = %s) seals gaps below ~%s only, so sound %s "
                            "leak through it" % (_mm(closure["dx"]), transition_hz, _mm(dx_wave),
                                                 _mm(seal), "will" if gmin >= OPEN_CELLS * dx_wave
                                                 else "may"))
        else:
            msg = ("Air layer of %s between parallel faces over %.2f m² (%s), %s: finer than the "
                   "wave band needs; glue the surfaces unless the layer matters"
                   % (width, extra["area_m2"], lay, _detail_text(detail)))
        item = {"message": msg, "segment": [np.round(near, 6).tolist(), np.round(far, 6).tolist()],
                "gap_type": kind, "gap_m": round(gmin, 6), "gap_max_m": round(gmax, 6)}
        item.update({x: y for x, y in extra.items() if y is not None})
        return item

    def gap_sev(g):
        kind, _f, gmin, _gx, _n, _fa, extra = g
        return int(kind == "slit" and bool(extra.get("leak")) and watertight and closure is not None
                   and dx_wave is not None and gmin >= SEAL_CELLS * dx_wave)
    tables.append({
        "kind": "small_gap", "tag": TAGS["small_gap"],
        "hash": _gap_ids(gaps, fl, names),
        "sev": np.array([gap_sev(g) for g in gaps], dtype=np.int64),
        "hit": lambda m: np.array([m[g[1]].any() for g in gaps], dtype=bool),
        "faces": lambda k: gaps[k][1], "make": gap_item})

    findings, excluded, ex_state, blocked = _report(tables, fl, names, exclusions)
    counts = {"open_edges": int(open_mask.sum()), "nonmanifold_edges": int((counts > 2).sum()),
              "short_edges": len(tables[2]["hash"]), "degenerate_triangles": int(len(degen)),
              "overlapping_faces": len(groups), "small_gaps": len(gaps)}
    blocking = []
    if not watertight:
        blocking.append("The model is not closed: the wave solver is off unless the simulation "
                        "declares an open scene (an object in free field)%s" % (
                            "; %d of the open edges that cause it are excluded, which does not "
                            "close the model" % blocked if blocked else ""))

    volume = None
    if topo_closed:
        volume = topo_volume
    elif watertight:
        # voxel interior also holds the inside of closed objects (cabinets, audience blocks)
        volume = closure["volume"] - solids

    layers = [{"name": n, "n_triangles": int((fl == i).sum()),
               "area_m2": round(float(areas[fl == i].sum()), 4)} for i, n in enumerate(names)]
    summary = {
        "n_triangles": int(len(f)),
        "bbox": {"min": v.min(axis=0).round(6).tolist(), "max": v.max(axis=0).round(6).tolist()},
        "volume_m3": None if volume is None else round(volume, 3),
        "surface_m2": round(float(areas.sum()), 4),
        "layers": layers,
        "check": {
            "watertight": bool(watertight),
            "counts": counts,
            "excluded": excluded,
            "findings": findings,
            "solvers": {"geometrical": True, "wave": bool(watertight)},
            "closure": ({"method": "topology"} if topo_closed else
                        {"method": "voxels", "dx_m": round(closure["dx"], 3),
                         "closed": bool(closure["closed"])}),
            "detail": detail,
            "exclusions": ex_state,
            "blocking": blocking,
            "notes": notes,
        },
    }
    return summary, closed


def _ray_hits(origins, direction, tri):
    """Number of crossings of rays origin+t*dir (t>0) with triangles (Moller-Trumbore)."""
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    hits = np.zeros(len(origins), dtype=np.int64)
    for k, o in enumerate(origins):
        s = o - tri[:, 0]
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        vv = (q @ direction) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hits[k] = int((ok & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 1e-9)).sum())
    return hits


def inside(points, tri_closed):
    """Point-in-volume by ray parity over closed solids, majority of 3 directions."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(tri_closed) == 0:
        return np.zeros(len(pts), dtype=bool)
    dirs = np.array([[0.5773, 0.5774, 0.5774], [-0.3, 0.8, 0.52], [0.71, -0.2, -0.67]])
    votes = sum((_ray_hits(pts, d / np.linalg.norm(d), tri_closed) % 2 == 1).astype(int)
                for d in dirs)
    return votes >= 2


def distance_to_surfaces(points, tri):
    """Minimal point-triangle distance (m) for each point."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    out = np.empty(len(pts))
    for k, p in enumerate(pts):
        ap = p - a
        d1 = np.einsum("ij,ij->i", ab, ap)
        d2 = np.einsum("ij,ij->i", ac, ap)
        n = np.cross(ab, ac)
        nn = np.einsum("ij,ij->i", n, n)
        # barycentric projection, then clamp to the triangle
        bp = p - b
        d3 = np.einsum("ij,ij->i", ab, bp)
        d4 = np.einsum("ij,ij->i", ac, bp)
        cp = p - c
        d5 = np.einsum("ij,ij->i", ab, cp)
        d6 = np.einsum("ij,ij->i", ac, cp)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = np.where(va + vb + vc == 0, 1.0, va + vb + vc)
        v_ = vb / denom
        w_ = vc / denom
        proj = a + ab * v_[:, None] + ac * w_[:, None]
        interior = (va >= 0) & (vb >= 0) & (vc >= 0) & (nn > 0)
        dist = np.where(interior, np.linalg.norm(p - proj, axis=1), np.inf)
        for s0, s1 in ((a, b), (b, c), (c, a)):
            seg = s1 - s0
            L = np.einsum("ij,ij->i", seg, seg)
            t = np.clip(np.einsum("ij,ij->i", p - s0, seg) / np.where(L > 0, L, 1.0), 0, 1)
            dist = np.minimum(dist, np.linalg.norm(p - (s0 + seg * t[:, None]), axis=1))
        out[k] = float(dist.min())
    return out
