# -*- coding: utf-8 -*-
"""Conformal voxel face weights: exact triangle clipping against the
grid planes (Sutherland-Hodgman), area deposited on the dominant-axis
face. Provides phi and face_w (areas / dx^2) to FdtdGrid."""
from __future__ import annotations

import os

import numpy as np


def _clip_poly(poly, axis, value, keep_below):
    """Sutherland-Hodgman clipping against the plane x[axis]=value."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da = a[axis] - value
        db = b[axis] - value
        # half-open convention [c0, c0+dx): the upper bound is STRICT,
        # so a piece lying exactly on a grid plane belongs to a single
        # cell (otherwise it would be counted twice).
        ina = (da <= 0) if keep_below else (da >= 0)
        inb = (db <= 0) if keep_below else (db >= 0)
        if ina:
            out.append(a)
        if ina != inb:
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    p0 = poly[0]
    s = np.zeros(3)
    for i in range(1, len(poly) - 1):
        s += np.cross(poly[i] - p0, poly[i + 1] - p0)
    return 0.5 * float(np.linalg.norm(s))


def _clip_poly_f(poly, axis, value, keep_below):
    """_clip_poly on vertices as float 3-tuples: the same IEEE operations in the same
    order (bit-identical), without the per-call cost of numpy on 3-element arrays."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da = a[axis] - value
        db = b[axis] - value
        ina = (da <= 0) if keep_below else (da >= 0)
        inb = (db <= 0) if keep_below else (db >= 0)
        if ina:
            out.append(a)
        if ina != inb:
            t = da / (da - db)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                        a[2] + t * (b[2] - a[2])))
    return out


def _poly_area_f(poly):
    """_poly_area on float 3-tuples: np.cross's component formulas, the same sequential
    accumulation, and the norm of the sum in a fixed order (as fdtd._dot3)."""
    if len(poly) < 3:
        return 0.0
    p0 = poly[0]
    s0 = s1 = s2 = 0.0
    for i in range(1, len(poly) - 1):
        u0, u1, u2 = poly[i][0] - p0[0], poly[i][1] - p0[1], poly[i][2] - p0[2]
        v0, v1, v2 = poly[i + 1][0] - p0[0], poly[i + 1][1] - p0[1], poly[i + 1][2] - p0[2]
        s0 += u1 * v2 - u2 * v1
        s1 += u2 * v0 - u0 * v2
        s2 += u0 * v1 - u1 * v0
    return 0.5 * float(np.sqrt((s0 * s0 + s1 * s1) + s2 * s2))


def face_weights(tri_verts, origin, dx, shape):
    """dict {(i,j,k,axis,side): true_area} per crossed cell; side = +1
    if the piece normal points toward +axis, -1 otherwise."""
    acc = {}
    tv = np.asarray(tri_verts, float)
    # deterministic sub-nanometer shift: removes every exact vertex /
    # grid plane coincidence (the source of all degenerate clipping
    # cases), areas unchanged to within 1e-9.
    tv = tv + dx * 1e-9 * np.array([1.0, 1.6180339887, 2.6180339887])
    for t in range(tv.shape[0]):
        tri = tv[t]
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        nrm = np.cross(e1, e2)
        a2 = np.linalg.norm(nrm)
        if a2 < 1e-14:
            continue
        nrm = nrm / a2
        ax = int(np.argmax(np.abs(nrm)))
        side = 1 if nrm[ax] > 0 else -1
        # window widened by +/-1 cell: a float floor() can miss by one
        # cell a piece lying exactly on a plane; the half-open rule
        # guarantees each piece is counted once among the visited
        # cells.
        lo = np.floor((tri.min(0) - origin) / dx).astype(int) - 1
        hi = np.floor((tri.max(0) - origin) / dx).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(shape) - 1)
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    c0 = origin + np.array([i, j, k]) * dx
                    poly = [tri[0], tri[1], tri[2]]
                    ok = True
                    for a in range(3):
                        poly = _clip_poly(poly, a, c0[a], False)
                        if len(poly) < 3:
                            ok = False
                            break
                        poly = _clip_poly(poly, a, c0[a] + dx, True)
                        if len(poly) < 3:
                            ok = False
                            break
                    if not ok:
                        continue
                    A = _poly_area(poly)
                    if A <= 0:
                        continue
                    key = (i, j, k, ax, side)
                    acc[key] = acc.get(key, 0.0) + A
    return acc


# ---------------------------------------------------------------------------
# Finite-volume fitted-cell corrections (Bilbao) for the FDTD grid.
#
# The conservative voxelization (0.87 dx, the half diagonal) keeps every
# triangle away from the air cells, so the missing air lives in the solid
# cells between the last air face and the true wall. Two corrections are
# folded into the per-cell and per-face coefficients, so the kernels do
# not change and python/native/CUDA stay in parity:
#   * volume (phi >= 1): each solid cell gives its fine air fraction to its
#     6-neighbour air cells. Added mass only relaxes the CFL condition.
#   * face areas: the exact clipped pieces replace the |n_axis| estimate
#     of the nearest triangle (nrmq) on the facing air face.
# ---------------------------------------------------------------------------



def _fine_solid_mask(tri_verts, origin, dx_f, shape_f):
    """Conservative rasterization at dx/F: True if the center lies
    <= 0.87 dx_f from a triangle. origin = center of cell (0,0,0)."""
    from .fdtd import _dot3, _point_tri_dist   # late import (avoids a cycle)
    tv = np.asarray(tri_verts, dtype=np.float64)
    n = np.asarray(shape_f, int)
    solid = np.zeros(tuple(n), dtype=bool)
    rad = 0.5 * np.sqrt(3.0) * dx_f
    for t in range(len(tv)):
        a, b, c = tv[t]
        nrm = np.cross(b - a, c - a)
        nl = float(np.sqrt(_dot3(nrm, nrm)))
        if nl < 1e-14:
            continue
        nrm = nrm / nl
        tlo = np.minimum(np.minimum(a, b), c) - rad
        thi = np.maximum(np.maximum(a, b), c) + rad
        i0 = np.maximum(np.floor((tlo - origin) / dx_f).astype(int), 0)
        i1 = np.minimum(np.ceil((thi - origin) / dx_f).astype(int) + 1, n)
        if np.any(i0 >= i1):
            continue
        gx = origin[0] + dx_f * np.arange(i0[0], i1[0])
        gy = origin[1] + dx_f * np.arange(i0[1], i1[1])
        gz = origin[2] + dx_f * np.arange(i0[2], i1[2])
        P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"),
                     axis=-1).reshape(-1, 3)
        near = np.abs(_dot3(P - a, nrm)) <= rad
        if not near.any():
            continue
        d = np.full(P.shape[0], np.inf)
        d[near] = _point_tri_dist(P[near], a, b, c)
        d = d.reshape(i1[0] - i0[0], i1[1] - i0[1], i1[2] - i0[2])
        sub = (slice(i0[0], i1[0]), slice(i0[1], i1[1]),
               slice(i0[2], i1[2]))
        solid[sub] |= d <= rad
    return solid


def subgrid_cap() -> int:
    """Largest subgrid of fine_air_fractions, in subcells: ~4 boolean masks of it are alive at
    the same time (~4 bytes/subcell). 450 M (~1.8 GB) is safe everywhere; AURALIS_SUBGRID_MAX
    (in millions) raises it on a machine with the RAM."""
    cap = 450_000_000
    env = os.environ.get("AURALIS_SUBGRID_MAX")
    if env:
        try:
            cap = max(50, int(float(env))) * 1_000_000
        except ValueError:
            print("[Auralis] AURALIS_SUBGRID_MAX='%s' invalid (expected a "
                  "count in MILLIONS of subcells) - ignored" % env)
    return cap


def fine_air_fractions(scene, vox, factor: int = 3):
    """(TRUE air fraction per coarse cell, factor ACTUALLY used), from a
    dx/factor subgrid flooded from the coarse air."""
    F = int(max(2, factor))
    shape_f = tuple(int(s) * F for s in vox.shape)
    n_fine = int(np.prod(shape_f))
    cap = subgrid_cap()
    if n_fine > cap and F > 2:
        print("[Auralis] conformal: subgrid %s too large (%.0f M > "
              "%.0f M), falling back to factor 2 - set "
              "AURALIS_SUBGRID_MAX=<millions> to raise it if RAM allows "
              "(~4 bytes/subcell)"
              % (shape_f, n_fine / 1e6, cap / 1e6))
        return fine_air_fractions(scene, vox, factor=2)
    dx_f = vox.dx / F
    # fine centers: coarse grid corner + (i+0.5) dx_f; the corner is
    # origin - dx/2 (voxelize puts the CENTERS at origin + i*dx).
    org_f = np.asarray(vox.origin, float) - 0.5 * vox.dx + 0.5 * dx_f
    from . import gridnative
    if gridnative.lib() is not None:
        # same raster, same bounded growth, same counts; 2 bytes per subcell
        return gridnative.fine_fraction(scene.tri_verts, org_f, dx_f, vox.shape, F,
                                        vox.air, 3 * F + 8), F
    fine_solid = _fine_solid_mask(scene.tri_verts, org_f, dx_f, shape_f)
    open_f = ~fine_solid
    del fine_solid
    seed = vox.air
    for ax in range(3):
        seed = np.repeat(seed, F, axis=ax)
    air_f = seed & open_f
    n_prev = int(air_f.sum())
    for _ in range(3 * F + 8):
        grown = air_f.copy()
        grown[1:, :, :] |= air_f[:-1, :, :]
        grown[:-1, :, :] |= air_f[1:, :, :]
        grown[:, 1:, :] |= air_f[:, :-1, :]
        grown[:, :-1, :] |= air_f[:, 1:, :]
        grown[:, :, 1:] |= air_f[:, :, :-1]
        grown[:, :, :-1] |= air_f[:, :, 1:]
        grown &= open_f
        n_now = int(grown.sum())
        air_f = grown
        if n_now == n_prev:
            break
        n_prev = n_now
    nx, ny, nz = vox.shape
    frac = air_f.reshape(nx, F, ny, F, nz, F).mean(axis=(1, 3, 5))
    return frac.astype(np.float64), F


def face_projections(tri_verts, origin_center, dx, shape, solid=None):
    """Exact area projections per SOLID cell: returns the dict
    {(i,j,k,axis,side): sum A*n_axis^2}. origin_center = center (0,0,0)."""
    from .fdtd import _dot3
    acc = {}
    tv = np.asarray(tri_verts, float)
    tv = tv + dx * 1e-9 * np.array([1.0, 1.6180339887, 2.6180339887])
    origin = np.asarray(origin_center, float) - 0.5 * dx   # corner
    shape = np.asarray(shape, int)
    rad = 0.5 * np.sqrt(3.0) * dx
    for t in range(tv.shape[0]):
        tri = tv[t]
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        nrm = np.cross(e1, e2)
        a2 = float(np.sqrt(_dot3(nrm, nrm)))
        if a2 < 1e-14:
            continue
        nrm = nrm / a2
        lo = np.maximum(np.floor((tri.min(0) - origin) / dx).astype(int) - 1,
                        0)
        hi = np.minimum(np.floor((tri.max(0) - origin) / dx).astype(int) + 1,
                        shape - 1)
        if np.any(lo > hi):
            continue
        ii = np.arange(lo[0], hi[0] + 1)
        jj = np.arange(lo[1], hi[1] + 1)
        kk = np.arange(lo[2], hi[2] + 1)
        ctr = np.stack(np.meshgrid(
            origin[0] + (ii + 0.5) * dx, origin[1] + (jj + 0.5) * dx,
            origin[2] + (kk + 0.5) * dx, indexing="ij"), axis=-1)
        near = np.abs(_dot3(ctr - tri[0], nrm)) <= rad
        if solid is not None:
            near &= solid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1,
                          lo[2]:hi[2] + 1]
        cand = np.argwhere(near)
        if cand.size == 0:
            continue
        w_ax = nrm * nrm                       # (n_x^2, n_y^2, n_z^2)
        sides = tuple(1 if nrm[a] > 0 else -1 for a in range(3))
        tri_f = [tuple(float(v) for v in tri[q]) for q in range(3)]
        org = [float(v) for v in origin]
        for ci, cj, ck in cand:
            i, j, k = int(lo[0] + ci), int(lo[1] + cj), int(lo[2] + ck)
            c0 = (org[0] + i * dx, org[1] + j * dx, org[2] + k * dx)
            poly = tri_f
            ok = True
            for a in range(3):
                poly = _clip_poly_f(poly, a, c0[a], False)
                if len(poly) < 3:
                    ok = False
                    break
                poly = _clip_poly_f(poly, a, c0[a] + dx, True)
                if len(poly) < 3:
                    ok = False
                    break
            if not ok:
                continue
            A = _poly_area_f(poly)
            if A <= 0:
                continue
            for a in range(3):
                if w_ax[a] < 1e-6:
                    continue
                key = (i, j, k, a, sides[a])
                acc[key] = acc.get(key, 0.0) + A * w_ax[a]
    return acc


_MAX_WALK = 3      # solid shell <= 2 cells (planes), 3 at the corners
_T_DIFF = 0.15     # anti-shielding diffusion step of conformal_corrections
_OFFS6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
          (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _nearest_air_face(air, i, j, k, ax, side, max_depth: int = 4):
    """6-connected BFS from a solid cell to the nearest AIR cell; returns
    ((i,j,k), f = 2*axis + 0 for -axis, 1 for +axis) or None."""
    shape = air.shape
    seen = {(i, j, k)}
    queue = [(i, j, k)]
    for _depth in range(max_depth):
        nxt = []
        for cell in queue:
            for (ox, oy, oz) in _OFFS6:
                p = (cell[0] + ox, cell[1] + oy, cell[2] + oz)
                if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
                        and 0 <= p[2] < shape[2]) or p in seen:
                    continue
                seen.add(p)
                if not air[p]:
                    nxt.append(p)
                    continue
                # air cell found: preferred wall face
                prefs = [2 * ax + (1 if side < 0 else 0)]
                prefs += [f for f in range(6) if f not in prefs]
                for f in prefs:
                    a2, sgn = f // 2, (1 if f % 2 else -1)
                    q = [p[0], p[1], p[2]]
                    q[a2] += sgn
                    inb = 0 <= q[a2] < shape[a2]
                    if not inb or not air[q[0], q[1], q[2]]:
                        return (p, f)
        queue = nxt
        if not queue:
            break
    return None


def wall_face_areas(scene, vox):
    """Weighted areas per (AIR cell, face f=2*axis+side), in m^2: returns
    (W (n_air,6), grid->air idx, stats, area per cell)."""
    air = vox.air
    nx, ny, nz = air.shape
    idx = np.full(air.shape, -1, dtype=np.int64)
    idx[air] = np.arange(int(air.sum()))
    from . import gridnative
    if gridnative.lib() is not None:
        W, cell_area, stats = gridnative.wall_faces(scene.tri_verts, vox.origin, vox.dx, air,
                                                    idx, _MAX_WALK)
        return W, idx, stats, cell_area
    acc = face_projections(scene.tri_verts, vox.origin, vox.dx,
                           vox.shape, solid=~air)
    W = np.zeros((int(air.sum()), 6), dtype=np.float64)
    a_tot = 0.0
    a_mapped = 0.0
    a_dropped = 0.0
    a_membrane = 0.0
    a_edge = 0.0

    def _walk_deposit(i, j, k, ax, side, A, max_walk):
        found = 0
        for s in (side, -side):
            p = [i, j, k]
            for _step in range(max_walk):
                p[ax] += s
                if not (0 <= p[ax] < air.shape[ax]):
                    break
                if air[p[0], p[1], p[2]]:
                    # air cell face oriented TOWARD the wall
                    f_air = 2 * ax + (1 if s < 0 else 0)
                    W[idx[p[0], p[1], p[2]], f_air] += A
                    found += 1
                    break
        return found

    for (i, j, k, ax, side), A in acc.items():
        a_tot += A
        found = _walk_deposit(i, j, k, ax, side, A, _MAX_WALK)
        if found == 0:
            # EDGE column: the solid bands of TWO walls cross, so every
            # axial walk stays solid; the nearest air is diagonal. Short
            # 6-connected BFS, deposit on a true wall face of the cell
            # found (the piece axis first - same split as nrmq at a
            # corner).
            hit = _nearest_air_face(air, i, j, k, ax, side)
            if hit is not None:
                ci, f_air = hit
                W[idx[ci[0], ci[1], ci[2]], f_air] += A
                a_edge += A
                found = 1
        if found == 0:
            a_dropped += A
        else:
            a_mapped += A
            if found == 2:
                a_membrane += A
    stats = {"area_true_m2": a_tot, "area_mapped_m2": a_mapped,
             "area_dropped_m2": a_dropped, "area_membrane_m2": a_membrane,
             "area_edge_m2": a_edge}
    cell_area = np.zeros(air.shape, dtype=np.float64)
    for (i, j, k, _ax, _side), A in acc.items():
        cell_area[i, j, k] += A          # sum of n_a^2 = true area
    return W, idx, stats, cell_area


def _excess_by_axis(vox, air, direct, frac, cnt_air, give, idx):
    """The donated excess split by axis, on the receiving air cells only:
    (air indices, (3, m)), exactly the dense deposit of conformal_corrections."""
    shape = vox.shape
    lim = np.asarray(shape)
    D = np.flatnonzero(direct)
    cD = np.stack(np.unravel_index(D, shape), axis=1)
    cnt = np.maximum(cnt_air.ravel()[D], 1)
    own = frac.ravel()[D] / cnt
    relay = np.zeros((3, len(D)))
    if give is not None:
        g = give.ravel()
        for s in _OFFS6:
            a = int(np.flatnonzero(s)[0])
            c = cD - np.asarray(s)
            ok = ((c >= 0) & (c < lim)).all(axis=1)
            relay[a, ok] += g[np.ravel_multi_index(c[ok].T, shape)]
        relay /= cnt
    air_f = air.ravel()
    hits = []
    for s in _OFFS6:
        c = cD + np.asarray(s)
        ok = ((c >= 0) & (c < lim)).all(axis=1)
        p = np.full(len(D), -1, dtype=np.int64)
        p[ok] = np.ravel_multi_index(c[ok].T, shape)
        ok[ok] = air_f[p[ok]]
        hits.append((s, ok, p))
    R = np.unique(np.concatenate([p[ok] for _s, ok, p in hits]))
    vals = np.zeros((3, len(R)))
    for s, ok, p in hits:
        a = int(np.flatnonzero(s)[0])
        k = np.searchsorted(R, p[ok])
        vals[a, k] += own[ok]
        for a2 in range(3):
            vals[a2, k] += relay[a2, ok]
    return idx.ravel()[R].astype(np.int64), vals


def _shift_grid(a, s, fill=False):
    """a moved by the offset s (cells), `fill` where nothing moves in."""
    out = np.full(a.shape, fill, dtype=a.dtype)
    sx, sy, sz = s
    src = a[max(0, -sx):a.shape[0] - max(0, sx) or None,
            max(0, -sy):a.shape[1] - max(0, sy) or None,
            max(0, -sz):a.shape[2] - max(0, sz) or None]
    out[max(0, sx):a.shape[0] - max(0, -sx) or None,
        max(0, sy):a.shape[1] - max(0, -sy) or None,
        max(0, sz):a.shape[2] - max(0, -sz) or None] = src
    return out


def _axis_inputs(air, frac, shifts):
    """(direct donors, air neighbour count, relay gift or None) of the donation, with integer
    passes only: the same masks and values as the numpy path of conformal_corrections, for
    _excess_by_axis when phi itself comes from gridnative.donate."""
    donors = (~air) & (frac > 1e-9)
    cnt_air = np.zeros(air.shape, dtype=np.int8)
    for s in shifts:
        cnt_air += _shift_grid(air, s).astype(np.int8)
    direct = donors & (cnt_air > 0)
    deep = donors & (cnt_air == 0)
    if not deep.any():
        return direct, cnt_air, None
    cnt_dir = np.zeros(air.shape, dtype=np.int8)
    for s in shifts:
        cnt_dir += _shift_grid(direct, s).astype(np.int8)
    relay = deep & (cnt_dir > 0)
    give = np.zeros(air.shape, dtype=np.float64)
    give[relay] = frac[relay] / np.maximum(cnt_dir[relay], 1)
    return direct, cnt_air, give


def conformal_corrections(scene, vox, factor: int = 3,
                          w_floor: float = 0.05, w_cap: float = 3.0,
                          phi_cap: float = 4.0, smooth_sweeps: int = 8,
                          per_axis: bool = False):
    """(phi, face_w, stats) for FdtdGrid.

    phi (n_air,) >= 1 is the relative cell volume, face_w (n_air, 6) the face area / dx^2
    (-1: use nrmq). per_axis also returns stats["excess_axis"] = (air cells, (3, m)), phi - 1
    split by the axis along which the volume was donated, for the tangential face apertures.
    It is sparse (only cells with an excess) and leaves phi itself unchanged."""
    dx = float(vox.dx)
    air = vox.air
    n_air = int(air.sum())
    W_m2, idx, fstats, cell_area = wall_face_areas(scene, vox)

    # F_used = REAL factor (a memory fallback can lower it): it is the
    # one that drives the slab AND the stats, never the requested one.
    frac, F = fine_air_fractions(scene, vox, factor=factor)
    # conservative subgrid slab: a rad_f layer of true air is classed
    # fine-solid along each wall, and restored analytically
    rad_f = 0.5 * np.sqrt(3.0) * dx / F
    solid_c = ~air                            # the slab, as a cell fraction, on solid cells
    frac[solid_c] = frac[solid_c] + cell_area[solid_c] * rad_f / dx ** 3
    del solid_c
    phi = np.ones(n_air, dtype=np.float64)
    v_orphan = 0.0
    shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
              (0, -1, 0), (0, 0, 1), (0, 0, -1))

    def _shift(a, s, fill=False):
        out = np.full(a.shape, fill, dtype=a.dtype)
        sx, sy, sz = s
        src = a[max(0, -sx):a.shape[0] - max(0, sx) or None,
                max(0, -sy):a.shape[1] - max(0, sy) or None,
                max(0, -sz):a.shape[2] - max(0, sz) or None]
        out[max(0, sx):a.shape[0] - max(0, -sx) or None,
            max(0, sy):a.shape[1] - max(0, -sy) or None,
            max(0, sz):a.shape[2] - max(0, -sz) or None] = src
        return out

    from . import gridnative
    fast_axis = (per_axis and int(smooth_sweeps) == 0 and gridnative.lib() is not None
                 and os.environ.get("AURALIS_CONFORMAL_AXIS_NUMPY", "0") != "1")
    if gridnative.lib() is not None and (not per_axis or fast_axis):
        # the same donation, relay and diffusion, cell by cell in the same order. With
        # conformal_aniso the split by axis is then taken sparsely, since the full-grid float
        # passes below are slow on large grids.
        phi, orphan = gridnative.donate(air, frac, int(smooth_sweeps), _T_DIFF)
        v_orphan = float(frac[orphan].sum()) * dx ** 3
        donors = None
        if fast_axis:
            direct, cnt_air, give = _axis_inputs(air, frac, shifts)
            ex_cells, ex_vals = _excess_by_axis(vox, air, direct, frac, cnt_air, give, idx)
            del direct, cnt_air, give
    else:
        donors = (~air) & (frac > 1e-9)
    if donors is not None and donors.any():
        # number of air 6-neighbors per cell
        cnt_air = np.zeros(vox.shape, dtype=np.int8)
        for s in shifts:
            cnt_air += _shift(air, s).astype(np.int8)
        direct = donors & (cnt_air > 0)
        deep = donors & (cnt_air == 0)
        # relay: deep donors first give (in equal shares) to their
        # DIRECT 6-neighbor donors
        pool = np.where(direct, frac, 0.0)
        if deep.any():
            cnt_dir = np.zeros(vox.shape, dtype=np.int8)
            for s in shifts:
                cnt_dir += _shift(direct, s).astype(np.int8)
            relay = deep & (cnt_dir > 0)
            v_orphan = float(frac[deep & (cnt_dir == 0)].sum()) * dx ** 3
            give = np.where(relay, frac / np.maximum(cnt_dir, 1), 0.0)
            for s in shifts:
                pool += np.where(direct, _shift(give, s), 0.0)
        else:
            v_orphan = 0.0
        # final deposit: equal shares between the air 6-neighbors
        share = np.where(direct, pool / np.maximum(cnt_air, 1), 0.0)
        excess = np.zeros(vox.shape, dtype=np.float64)
        for s in shifts:
            excess += np.where(air, _shift(share, s), 0.0)
        sparse_axis = per_axis and int(smooth_sweeps) == 0
        if sparse_axis:
            # no diffusion: only the direct donors and their air neighbours
            # take part -> computed on those sets (same sums, same order)
            ex_cells, ex_vals = _excess_by_axis(vox, air, direct, frac, cnt_air,
                                                give if deep.any() else None, idx)
        elif per_axis:
            # same deposit, by axis: a relayed deep donor stands for a slab
            # normal to its RELAY axis, a direct one to its deposit axis
            ex_ax = np.zeros((3,) + vox.shape, dtype=np.float64)
            own = np.where(direct, frac / np.maximum(cnt_air, 1), 0.0)
            relay_ax = np.zeros((3,) + vox.shape, dtype=np.float64)
            if deep.any():
                for s in shifts:
                    a = int(np.flatnonzero(s)[0])
                    relay_ax[a] += np.where(direct, _shift(give, s), 0.0)
            relay_ax /= np.maximum(cnt_air, 1)
            for s in shifts:
                a = int(np.flatnonzero(s)[0])
                ex_ax[a] += np.where(air, _shift(own, s), 0.0)
                for a2 in range(3):
                    ex_ax[a2] += np.where(air, _shift(relay_ax[a2], s), 0.0)
        # ---- anti-shielding: conservative diffusion in the air -------
        # Piling everything onto the distance-1 layer doubles its mass
        # (phi ~ 2). At mid band the heavy layer moves less
        # ((k dx)^2 (phi-1)), wall pressure drops and the absorber is
        # shielded (T20 +37 % instead of +14 %). The excess is spread
        # over ~2-3 layers through symmetric air-air exchanges that
        # conserve the sum exactly: the total compliance (volume, low
        # modes) does not change, and shielding falls with the local phi-1.
        t_diff = _T_DIFF
        for _ in range(int(smooth_sweeps)):
            delta = np.zeros_like(excess)
            for s in shifts:
                nb_val = _shift(excess, s)
                nb_air = _shift(air, s)
                delta += np.where(air & nb_air, nb_val - excess, 0.0)
            excess += t_diff * delta
        phi += excess[air]
        if per_axis and not sparse_axis:  # the same (linear) diffusion per axis
            for a in range(3):
                e = ex_ax[a]
                for _ in range(int(smooth_sweeps)):
                    delta = np.zeros_like(e)
                    for s in shifts:
                        delta += np.where(air & _shift(air, s), _shift(e, s) - e, 0.0)
                    e = e + t_diff * delta
                ex_ax[a] = e
    n_over = int((phi > phi_cap).sum())
    excess_axis = None
    if per_axis:
        # sparse (cells, (3, m)): the excess lives on the cells by the walls
        if fast_axis:
            ex_phi = phi[ex_cells]
        elif donors.any() and sparse_axis:
            ex_phi = phi[ex_cells]
        elif donors.any():
            ex_vals = np.stack([ex_ax[a][air] for a in range(3)])
            del ex_ax
            ex_cells = np.nonzero((ex_vals != 0.0).any(axis=0))[0]
            ex_vals = ex_vals[:, ex_cells]
            ex_phi = phi[ex_cells]
        else:
            ex_cells, ex_vals, ex_phi = np.zeros(0, np.int64), np.zeros((3, 0)), np.zeros(0)
        tot = ex_vals.sum(axis=0)
        capped = np.minimum(ex_phi, phi_cap) - 1.0
        # the cap scales the three parts alike
        ex_vals *= np.where(tot > 1e-12, capped / np.maximum(tot, 1e-12), 0.0)
        excess_axis = (ex_cells, ex_vals)
    phi = np.minimum(phi, phi_cap)

    face_w = np.full((n_air, 6), -1.0, dtype=np.float64)
    haveA = W_m2 > 0.0
    q = W_m2[haveA] / (dx * dx)               # elsewhere W <= 0: never above the cap
    n_capped = int((q > w_cap).sum())
    face_w[haveA] = np.clip(q, w_floor, w_cap)
    del q

    v_air = n_air * dx ** 3
    v_donated = float(phi.sum() - n_air) * dx ** 3
    stats = dict(fstats)
    stats.update({
        "factor": int(F),          # REAL (may be < requested: fallback)
        "vol_air_disc_m3": v_air,
        "vol_donated_m3": v_donated,
        "vol_orphan_m3": v_orphan,
        "vol_gain_pct": 100.0 * v_donated / max(v_air, 1e-12),
        "n_faces_data": int(haveA.sum()),
        "n_faces_capped": n_capped,
        "n_phi_capped": n_over,
    })
    if per_axis:
        stats["excess_axis"] = excess_axis
    return phi, face_w, stats
