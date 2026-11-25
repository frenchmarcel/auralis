# -*- coding: utf-8 -*-
"""FDTD wave solver for the low band: voxelization, locally reacting
boundary grid, free-field calibration (1/d, d/c delay) and crossover
fusion with the geometric IR."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
from types import SimpleNamespace

import contextlib
import os

import numpy as np

_LAMBDA = 1.0 / np.sqrt(3.0)          # order-2 Courant (max 3D stability)
_LAMBDA4 = float(os.environ.get("AUR_FDTD_LAM4", "0.499"))  # order-4 CFL, <1/2


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

@dataclass
class VoxelGrid:
    solid: np.ndarray          # (nx,ny,nz) bool
    mat: np.ndarray            # (nx,ny,nz) int32, -1 = no boundary
    origin: np.ndarray         # corner (x0,y0,z0) = center of cell [0,0,0]
    dx: float
    air: np.ndarray = field(default=None)   # bool, filled from the source
    nrmq: np.ndarray = field(default=None)  # (nx,ny,nz,3) uint8: |n|*255
                                            # of the nearest triangle

    @property
    def shape(self):
        return self.solid.shape

    def index_of(self, pos) -> Tuple[int, int, int]:
        i = np.rint((np.asarray(pos, float) - self.origin) / self.dx)
        return tuple(int(v) for v in i)


# Grids of the previous runs, reused when a scene runs again with the same geometry and
# cell size (one FDTD per source would otherwise rebuild an identical grid). Copies are
# returned, so the reuse is bit-identical. AURALIS_FDTD_GRID_CACHE=0 turns it off.
_VOX_CACHE: dict = {}
_CONF_CACHE: dict = {}
_CACHE_SIZE = 2


def _cache_on():
    return os.environ.get("AURALIS_FDTD_GRID_CACHE", "1") != "0"


# Grids kept between the runs of one series (keep_grids): the FdtdGrid objects of the scene
# and of the empty open-domain pass, and their native sessions. The grid stays on the GPUs
# and each run only places its source and probes. A walking listener runs the same grid
# hundreds of times.
_KEEP = {"depth": 0, "grids": {}, "sessions": {}, "vox": {}, "airkey": {}}


@contextlib.contextmanager
def keep_grids():
    """Within this block, run_low_band keeps its grids (host and GPU) from one call to the
    next when nothing but the source and the probes changes; released on exit."""
    _KEEP["depth"] += 1
    try:
        yield
    finally:
        _KEEP["depth"] -= 1
        if _KEEP["depth"] == 0:
            release_kept_grids()


def release_kept_grids():
    """Destroy the kept native sessions and drop the kept grids."""
    import ctypes as ct
    for lib, handle, _grid in _KEEP["sessions"].values():
        try:
            fd = lib.aur_fdtd_session4m_destroy
            fd.restype = None
            fd.argtypes = [ct.c_void_p]           # a pointer: never let ctypes pass a C int
            fd(ct.c_void_p(handle))
        except Exception as exc:                       # loud, never silent
            print("[Auralis] WARNING: FDTD session release failed (%s)" % exc)
    _KEEP["sessions"].clear()
    _KEEP["grids"].clear()
    _KEEP["vox"].clear()
    _KEEP["airkey"].clear()


_FIT_CACHE: dict = {}


def _fit_random_cached(alpha, centers, f_hi, l_max=None):
    """fit_admittance_random(alpha, centers, f_hi=f_hi, l_max=l_max), memoised.

    The fit is deterministic, and a series of runs would refit every material each time
    (~0.1 s each). Copies are returned."""
    from .reflection import fit_admittance_random
    key = (np.asarray(alpha, dtype=np.float64).tobytes(),
           np.asarray(centers, dtype=np.float64).tobytes(), float(f_hi),
           None if l_max is None else float(l_max))
    hit = _FIT_CACHE.get(key)
    if hit is None:
        hit = fit_admittance_random(alpha, centers, f_hi=f_hi, l_max=l_max)
        if len(_FIT_CACHE) > 512:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = hit
    gi, A, l = hit
    return gi, np.array(A, copy=True), np.array(l, copy=True)


def _kept_grid(key, build):
    """build() once per key while keep_grids() is active; plain build() otherwise."""
    if _KEEP["depth"] <= 0:
        return build()
    g = _KEEP["grids"].get(key)
    if g is None:
        g = build()
        _KEEP["grids"][key] = g
    return g


def _cache_put(cache, key, value):
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > _CACHE_SIZE:
        cache.pop(next(iter(cache)))


def _geometry_key(scene):
    import hashlib
    h = hashlib.sha1(np.ascontiguousarray(scene.tri_verts, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(scene.tri_mat, dtype=np.int64).tobytes())
    return h.hexdigest()


def _air_key(air):
    import hashlib
    hit = _KEEP["airkey"].get(id(air)) if _KEEP["depth"] > 0 else None
    if hit is not None and hit[0] is air:
        return hit[1]
    key = hashlib.sha1(np.packbits(air.ravel()).tobytes()).hexdigest() + str(air.shape)
    if _KEEP["depth"] > 0:                     # the kept voxel grids are not modified
        _KEEP["airkey"][id(air)] = (air, key)
    return key


def cached_conformal(scene, vox, **kwargs):
    """conformal_corrections, reused across the runs of an identical grid (copies)."""
    import copy
    from .conformal import conformal_corrections
    key = (_geometry_key(scene), float(vox.dx), tuple(np.round(vox.origin, 12)), vox.shape,
           _air_key(vox.air), tuple(sorted(kwargs.items())))
    if _cache_on() and key in _CONF_CACHE:
        return copy.deepcopy(_CONF_CACHE[key]) + (True,)
    out = conformal_corrections(scene, vox, **kwargs)
    if _cache_on():
        _cache_put(_CONF_CACHE, key, copy.deepcopy(out))
    return out + (False,)


def voxelize(scene, dx: float, src_pos, margin_cells: int = 2,
             include_points=None) -> VoxelGrid:
    """Voxelize the scene: a cell within 0.87 dx of a triangle is solid
    (material of the nearest one); air = component connected to source."""
    tv = np.asarray(scene.tri_verts, dtype=np.float64)
    lo = tv.reshape(-1, 3).min(axis=0)
    hi = tv.reshape(-1, 3).max(axis=0)
    if include_points is not None:
        # the domain must contain source and receivers (open scenes:
        # the bbox of the geometry alone can be flat or too small)
        pts = np.atleast_2d(np.asarray(include_points, dtype=np.float64))
        pad = 6.0 * dx
        lo = np.minimum(lo, pts.min(axis=0) - pad)
        hi = np.maximum(hi, pts.max(axis=0) + pad)
    lo = lo - margin_cells * dx
    hi = hi + margin_cells * dx
    n = np.maximum(np.ceil((hi - lo) / dx).astype(int) + 1, 4)
    origin = lo
    key = (_geometry_key(scene), float(dx), tuple(np.round(origin, 12)), tuple(int(v) for v in n))
    if _KEEP["depth"] > 0:
        # within keep_grids(), the voxel grid itself is reused (no copies: run_low_band does
        # not modify it) when the source lies in its air
        kv = _KEEP["vox"].get(key)
        if kv is not None:
            si = kv.index_of(src_pos)
            if all(0 <= si[k] < kv.shape[k] for k in range(3)) and kv.air[si]:
                return kv
    hit = _VOX_CACHE.get(key) if _cache_on() else None
    if hit is not None:
        g = VoxelGrid(solid=hit["solid"].copy(), mat=hit["mat"].copy(), origin=origin.copy(),
                      dx=dx, nrmq=hit["nrmq"].copy())
        _flood_air(g, src_pos, hit["airs"])
        if _KEEP["depth"] > 0:
            _KEEP["vox"][key] = g
        return g
    from . import gridnative
    if gridnative.lib() is not None:
        solid, matid, nrmq = gridnative.voxelize_tris(tv, scene.tri_mat, origin, dx, n)
    else:
        solid, matid, nrmq = _voxelize_tris_python(tv, scene.tri_mat, origin, dx, n)
    g = VoxelGrid(solid=solid, mat=matid, origin=origin, dx=dx, nrmq=nrmq)
    airs = []
    if _cache_on():
        _cache_put(_VOX_CACHE, key, {"solid": solid.copy(), "mat": matid.copy(),
                                     "nrmq": nrmq.copy(), "airs": airs})
    _flood_air(g, src_pos, airs)
    if _KEEP["depth"] > 0:
        _KEEP["vox"][key] = g
    return g


def _voxelize_tris_python(tv, tri_mat, origin, dx, n):
    """Python reference of the triangle loop of voxelize (native: gridnative)."""
    solid = np.zeros(tuple(n), dtype=bool)
    matid = np.full(tuple(n), -1, dtype=np.int32)
    best = np.full(tuple(n), np.inf, dtype=np.float32)
    nrmq = np.full(tuple(n) + (3,), 255, dtype=np.uint8)
    e1 = tv[:, 1] - tv[:, 0]
    e2 = tv[:, 2] - tv[:, 0]
    tn = np.cross(e1, e2)
    tn /= np.maximum(np.sqrt(_dot3(tn, tn))[:, None], 1e-30)
    tnq = np.minimum(np.abs(tn) * 255.0, 255.0).astype(np.uint8)

    rad = 0.5 * np.sqrt(3.0) * dx
    for t in range(len(tv)):
        a, b, c = tv[t]
        tlo = np.minimum(np.minimum(a, b), c) - rad
        thi = np.maximum(np.maximum(a, b), c) + rad
        i0 = np.maximum(np.floor((tlo - origin) / dx).astype(int), 0)
        i1 = np.minimum(np.ceil((thi - origin) / dx).astype(int) + 1, n)
        if np.any(i0 >= i1):
            continue
        gx = origin[0] + dx * np.arange(i0[0], i1[0])
        gy = origin[1] + dx * np.arange(i0[1], i1[1])
        gz = origin[2] + dx * np.arange(i0[2], i1[2])
        P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"),
                     axis=-1).reshape(-1, 3)
        d = _point_tri_dist(P, a, b, c).reshape(i1[0] - i0[0],
                                                i1[1] - i0[1],
                                                i1[2] - i0[2])
        sub = (slice(i0[0], i1[0]), slice(i0[1], i1[1]),
               slice(i0[2], i1[2]))
        hit = d <= rad
        upd = hit & (d < best[sub])
        solid[sub] |= hit
        matid[sub][upd] = int(tri_mat[t])
        nq = nrmq[sub]
        nq[upd] = tnq[t]
        nrmq[sub] = nq
        bb = best[sub]
        bb[upd] = d[upd]
        best[sub] = bb
    return solid, matid, nrmq


def _flood_air(g, src_pos, airs):
    """g.air = the air component connected to the source cell; an already flooded
    component that contains it is reused (copy), else flooded and remembered."""
    solid = g.solid
    n = solid.shape
    si = g.index_of(src_pos)
    if not all(0 <= si[k] < n[k] for k in range(3)) or solid[si]:
        raise ValueError("FDTD source inside a solid or outside the grid "
                         "(cell %s)" % (si,))
    for known in airs:
        if known[si]:
            g.air = known.copy()
            return
    from . import gridnative
    if gridnative.lib() is not None:
        g.air = gridnative.flood(solid, si)
        airs.append(g.air.copy())
        return
    air = np.zeros_like(solid)
    air[si] = True
    open_ = ~solid
    while True:
        grown = air.copy()
        grown[1:, :, :] |= air[:-1, :, :]
        grown[:-1, :, :] |= air[1:, :, :]
        grown[:, 1:, :] |= air[:, :-1, :]
        grown[:, :-1, :] |= air[:, 1:, :]
        grown[:, :, 1:] |= air[:, :, :-1]
        grown[:, :, :-1] |= air[:, :, 1:]
        grown &= open_
        if grown.sum() == air.sum():
            break
        air = grown
    g.air = air
    airs.append(air.copy())


def _dot3(X, v):
    """Dot products over the last axis in a fixed order, (x0 v0 + x1 v1) + x2 v2.

    Never replace with @: BLAS sums in an order that depends on the array size and the CPU.
    The native grid builder (aur_grid.cpp) uses this exact order, so both agree bit for bit."""
    return (X[..., 0] * v[..., 0] + X[..., 1] * v[..., 1]) + X[..., 2] * v[..., 2]


def _point_tri_dist(P, a, b, c):
    """Point-triangle distance, vectorized over P (N,3)."""
    ab = b - a
    ac = c - a
    ap = P - a
    d1 = _dot3(ap, ab)
    d2 = _dot3(ap, ac)
    bp = P - b
    d3 = _dot3(bp, ab)
    d4 = _dot3(bp, ac)
    cp = P - c
    d5 = _dot3(cp, ab)
    d6 = _dot3(cp, ac)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    v = vb / denom
    w = vc / denom
    proj = a + np.outer(v, ab) + np.outer(w, ac)
    inside = (d1 >= 0) | True                     # in-plane projection
    r = P - proj
    face = np.sqrt(_dot3(r, r))

    def seg(p, q):
        pq = q - p
        t = np.clip(_dot3(P - p, pq) / float(_dot3(pq, pq)), 0.0, 1.0)
        r = P - (p + np.outer(t, pq))
        return np.sqrt(_dot3(r, r))

    inface = ((va >= 0) & (vb >= 0) & (vc >= 0))
    d = np.where(inface & inside, face,
                 np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a)))
    return d


# ---------------------------------------------------------------------------
# Grid precomputations (neighbors, K, Gamma)
# ---------------------------------------------------------------------------

def gamma_from_alpha(alpha_low: np.ndarray) -> np.ndarray:
    """Normalized per-material admittance from the low-band absorption:
    R = sqrt(1-alpha), gamma = (1-R)/(1+R) (normal incidence)."""
    a = np.clip(np.asarray(alpha_low, dtype=np.float64), 0.0, 0.999)
    R = np.sqrt(1.0 - a)
    return np.clip((1.0 - R) / (1.0 + R), 0.0, 1.5)



def _alpha_at(gi, A, l, f):
    """Random-incidence alpha of a wall filter (continuous model) at f Hz."""
    from .reflection import alpha_random
    w = 2.0 * np.pi * f
    y = gi + sum(a / (1j * w + ll) for a, ll in zip(np.atleast_1d(A), np.atleast_1d(l)))
    return float(alpha_random(np.array([y]))[0])


def _phys_fit_mode():
    """Wall fit of the non-resonant physical materials.

    "auto" (fit_physical_wall) or "anchored" (the older continuous anchored fit, kept for
    comparisons). AURALIS_FDTD_PHYS_FIT overrides the default."""
    return os.environ.get("AURALIS_FDTD_PHYS_FIT", "auto")


_WALL_FIT_CACHE: dict = {}


def fit_physical_wall(gamma_fn, f_max, dt, lam, key=None):
    """Discrete wall filter of a physical model with normal admittance gamma_fn(f).

    Candidates are judged on the reflection the grid realises (wall_R_discrete, normal
    incidence, 30 Hz to the crossover), never on the continuous admittance:
    1) the complex fit, kept as is when passive,
    2) the same fit with passivity enforced, when the first one is active,
    3) when both miss the model by more than 0.05 (cavities, slotted panels), a passive
       vector fit with 1-3 resonant sections, kept if it is 20 % better.
    On 99 inferred impedance-tube samples the complex fit errs by 0.058 in |R|, against 0.149
    for an alpha-only fit. Returns (gi, A, l, sections or None, label, error)."""
    from .reflection import fit_admittance_complex, fit_wall_table, wall_R_discrete
    ck = None if key is None else (key, float(dt), float(lam), float(f_max))
    if ck is not None and ck in _WALL_FIT_CACHE:
        return _WALL_FIT_CACHE[ck]
    fg = np.geomspace(30.0, 1.2 * f_max, 48)
    gt = gamma_fn(fg)
    fe = np.geomspace(30.0, f_max / 1.5, 40)
    ge = gamma_fn(fe)
    rt = (1.0 - ge) / (1.0 + ge)
    r0 = wall_R_discrete(fe, 0.0, np.zeros(2), np.ones(2), dt, lam)
    ph = np.conj(r0 / np.abs(r0))
    fp = np.geomspace(5.0, 0.45 / dt, 300)

    def judge(gi, A, l, secs):
        err = float(np.mean(np.abs(wall_R_discrete(fe, gi, A, l, dt, lam, res=secs) * ph - rt)))
        over = float(np.abs(wall_R_discrete(fp, gi, A, l, dt, lam, res=secs)).max() - 1.0)
        if secs is None:
            over = max(over, -wall_passivity(gi, A, l, dt)[0])
        return err, over

    cands = []
    fit = fit_admittance_complex(fg, gt, l_max=0.6 / dt, dt=dt, lam=lam)
    cands.append(("complex", fit + (None,)) + judge(*fit, None))
    if cands[0][3] > 1e-4:
        fit = fit_admittance_complex(fg, gt, l_max=0.6 / dt, dt=dt, lam=lam, pass_weight=1e5)
        cands.append(("complex, passivity enforced", fit + (None,)) + judge(*fit, None))
    passive = [c for c in cands if c[3] <= 1e-4]
    best = min(passive or cands, key=lambda c: (c[3] > 1e-4, c[2]))
    if best[2] > 0.05:
        ft = np.geomspace(30.0, 1.2 * f_max, 60)
        gft = gamma_fn(ft)
        for n in (1, 2, 3):
            try:
                gi, A, l, secs, _rms = fit_wall_table(ft, gft, dt, lam, n_sections=n)
            except Exception as exc:          # noisy: the branches stay
                print("[Auralis] WARNING: resonant-section wall fit failed (%s)" % exc)
                continue
            c = ("%d resonant section(s)" % n, (gi, A, l, secs)) + judge(gi, A, l, secs)
            if c[3] <= 1e-4 and c[2] < 0.8 * best[2]:
                best = c
    gi, A, l, secs = best[1]
    out = (float(gi), np.asarray(A, float), np.asarray(l, float), secs, best[0], best[2])
    if ck is not None:
        _WALL_FIT_CACHE[ck] = out
    return out


def poromat_layers(spec):
    """Layers of a poromat spec, sugar forms ({"model": ...}, {"panel": ...}) included."""
    from . import poromat
    return poromat.normalize_spec(spec).get("layers", [])


def wall_passivity(gi, A, l, dt, n=8192):
    """(smallest real part, its frequency in Hz) of the admittance the grid realises.

    g_inf sits in the implicit centred term, the branches are explicit at time n with the
    exponential/trapezoidal states of run_grid_python. A negative value is an active wall
    that creates energy. A fit passive in continuous time is not always passive on the grid:
    a stiff high-pass branch (l dt ~ 2, negative amplitude) can make a near-rigid wall active
    and start a slowly growing boundary mode."""
    w = np.linspace(0.0, np.pi / dt, n + 2)[1:-1]
    z = np.exp(1j * w * dt)
    acc = np.zeros(len(w), dtype=np.complex128)
    for a, ll in zip(np.atleast_1d(A), np.atleast_1d(l)):
        if a == 0.0:
            continue
        e = np.exp(-ll * dt)
        acc += a * (1.0 - (1.0 - e) * (z + 1.0) / (2.0 * (z - e)))
    re = (gi + dt * acc / (1j * np.sin(w * dt))).real
    k = int(np.argmin(re))
    return float(re[k]), float(w[k] / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Frequency-dependent admittance: fitting a passive model
# ---------------------------------------------------------------------------

def fit_admittance(alpha_bands, band_centers, f_max: float):
    """Fit a passive admittance Y(w) = g_inf + sum_k A_k / (jw + l_k)
    so that |R| follows sqrt(1-alpha). Returns (g_inf, A(2,), l(2,))."""
    fb = np.asarray(band_centers, dtype=np.float64)
    ab = np.clip(np.asarray(alpha_bands, dtype=np.float64), 0.0, 0.999)
    sel = fb <= max(2.0 * f_max, fb[0] * 1.01)
    fb, ab = fb[sel], ab[sel]
    if len(fb) < 2:
        return float(gamma_from_alpha(ab[:1])[0]), np.zeros(2), np.ones(2)
    f = np.geomspace(max(20.0, 0.3 * fb[0]), min(1.2 * f_max, fb[-1]), 60)
    a_t = np.interp(f, fb, ab)
    R_t = np.sqrt(1.0 - a_t)
    w = 2.0 * np.pi * f

    def cost(g, A, l):
        Y = g + (A[0] / (1j * w + l[0]) + A[1] / (1j * w + l[1]))
        R = np.abs((1.0 - Y) / (1.0 + Y))
        return float(np.mean((R - R_t) ** 2))

    g0 = float(gamma_from_alpha(np.array([a_t.mean()]))[0])
    best = (cost(g0, np.zeros(2), np.ones(2)), g0, np.zeros(2), np.ones(2))
    lam_grid = 2.0 * np.pi * np.array([25.0, 45.0, 80.0, 140.0, 240.0,
                                       420.0, 700.0, 1100.0])
    for l1 in lam_grid:
        for l2 in lam_grid:
            if l2 < l1:
                continue
            for g in (0.25 * g0, g0, 2.0 * g0 + 0.02):
                for a1 in (0.0, 0.3 * l1, 1.0 * l1, 2.5 * l1):
                    for a2 in (0.0, 0.3 * l2, 1.0 * l2):
                        cc = cost(g, np.array([a1, a2]),
                                  np.array([l1, l2]))
                        if cc < best[0]:
                            best = (cc, g, np.array([a1, a2]),
                                    np.array([l1, l2]))
    _, g, A, l = best
    x = np.array([g, A[0], A[1], l[0], l[1]])
    step = 0.25 * np.maximum(np.abs(x), [0.02, 20.0, 20.0, 50.0, 50.0])
    for _ in range(120):
        improved = False
        for i in range(5):
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] = max(0.0, y[i] + sgn * step[i])
                if i >= 3:
                    y[i] = max(y[i], 1.0)
                c = cost(y[0], y[1:3], y[3:5])
                if c < best[0]:
                    best = (c, y[0], y[1:3].copy(), y[3:5].copy())
                    x = y
                    improved = True
        if not improved:
            step *= 0.5
            if step.max() < 1e-4:
                break
    _, g, A, l = best
    return float(g), A.astype(np.float64), l.astype(np.float64)


class _GridTimer:
    """AURALIS_FDTD_GRID_TIMING=1: time of each section of FdtdGrid, printed at the end."""
    def __init__(self):
        self.on = os.environ.get("AURALIS_FDTD_GRID_TIMING", "0") == "1"
        self.t = time.time()
        self.parts = []

    def __call__(self, name, done=False):
        if not self.on:
            return
        now = time.time()
        self.parts.append("%s %.2f s" % (name, now - self.t))
        self.t = now
        if done:
            print("[Auralis] FdtdGrid: " + ", ".join(self.parts))


class FdtdGrid:
    """Update structure: flattened air cells, neighbors with a ghost
    slot for solid faces, per-cell coefficients and Gamma."""

    def __init__(self, vox: VoxelGrid, gamma_mat: np.ndarray,
                 speed_of_sound: float, filt_mat=None, sigma=None,
                 gamma_border: float = 0.0, order: int = 2,
                 face_weighting: bool = True, pml=None,
                 phi=None, face_area_w=None, filt_res=None,
                 excess_axis=None, extra_links=None):
        """Build the grid from a flooded VoxelGrid.

        filt_mat: (gi, A, l) filtering walls, replaces gamma_mat. sigma: (nx, ny, nz) sponge.
        phi, face_area_w: conformal volumes and face areas. filt_res: resonant sections.
        excess_axis: phi - 1 by donation axis, dense (3, n_air) or sparse (cells, (3, m)),
        turned into tangential face apertures (sparse links, order 2 on those cells).
        extra_links: (flat grid cells a, b, w), symmetric conductances in face units
        (sub-cell openings)."""
        _tm = _GridTimer()
        self.vox = vox
        self.c = float(speed_of_sound)
        self.dx = float(vox.dx)
        self.order = 4 if int(order) == 4 else 2
        self.lam = _LAMBDA4 if self.order == 4 else _LAMBDA
        self.dt = self.lam * self.dx / self.c
        air = vox.air
        nx, ny, nz = air.shape
        self.n_air = int(air.sum())
        if self.n_air + 1 > np.iinfo(np.int32).max:
            raise ValueError("FDTD grid > 2^31-1 air cells (%d): "
                             "int32 indices overflow" % self.n_air)
        from . import gridnative
        nat = gridnative.has_structure()      # integer part in C++ (bit for bit)
        if nat:
            idx, ii, nb, K, nb2_n, full_n = gridnative.structure(air)
        else:
            idx = np.full(air.shape, -1, dtype=np.int32)
            idx[air] = np.arange(self.n_air)
            # int32 coordinates, same C order as argwhere (int64: 24 B per air cell)
            flat_air = np.flatnonzero(air)
            ii = np.empty((self.n_air, 3), dtype=np.int32)
            rest, ii[:, 2] = np.divmod(flat_air, nz)
            ii[:, 0], ii[:, 1] = np.divmod(rest, ny)
            del rest, flat_air
        self.cell_index = idx
        self.cells = ii

        _tm('index')
        ghost = self.n_air                       # slot kept at p = 0
        if not nat:
            nb = np.full((self.n_air, 6), ghost, dtype=np.int32)
            K = np.zeros(self.n_air, dtype=np.float64)
        gam = np.zeros(self.n_air, dtype=np.float64)
        offs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        if filt_mat is not None:
            gm = np.asarray(filt_mat[0], dtype=np.float64)
            fA = np.asarray(filt_mat[1], dtype=np.float64)
            fL = np.asarray(filt_mat[2], dtype=np.float64)
        else:
            gm = np.asarray(gamma_mat, dtype=np.float64)
            fA = fL = None
        face_cell, face_A, face_l = [], [], []
        res_cell, res_par = [], []
        # neighbours through an index padded with -2 (outside the grid, -1 =
        # solid): one int32 gather per face, and the wall work on the wall
        # faces only (less memory, same values bit for bit)
        if not nat:
            pad = np.full((nx + 2, ny + 2, nz + 2), -2, dtype=np.int32)
            pad[1:-1, 1:-1, 1:-1] = idx
            pad = pad.ravel()
            sy, sx = nz + 2, (ny + 2) * (nz + 2)
            pos = ((ii[:, 0].astype(np.int64) + 1) * sx
                   + (ii[:, 1].astype(np.int64) + 1) * sy + (ii[:, 2] + 1))
        shape_i = np.asarray(air.shape, dtype=np.int32)
        for f, (ox, oy, oz) in enumerate(offs):
            off_f = np.array([ox, oy, oz], dtype=np.int32)
            if nat:
                wl = gridnative.wall_list(ii, nb, air.shape, f)   # C++ scan (ABI 2.6)
                if wl is not None:
                    wi, inb = wl
                else:
                    wi = np.flatnonzero(nb[:, f] == ghost)  # wall faces
                    q = ii[wi] + off_f
                    inb = ((q >= 0) & (q < shape_i)).all(axis=1)  # solid inside the grid
                    del q
            else:
                flat = pad[pos + (ox * sx + oy * sy + oz)]
                isair = flat >= 0
                nb[isair, f] = flat[isair]
                K += isair
                wi = np.nonzero(~isair)[0]                 # wall faces
                del isair
                inb = flat[wi] == -1                   # solid inside the grid
                del flat
            if len(wi):
                jw = ii[wi[inb]] + off_f
                m = np.full(len(wi), -1, dtype=np.int32)
                m[inb] = vox.mat[jw[:, 0], jw[:, 1], jw[:, 2]]
                g = np.zeros(len(wi))
                has = m >= 0
                g[has] = gm[np.clip(m[has], 0, len(gm) - 1)]
                # edge averaging: the staircase exposes |nx|+|ny|+|nz|
                # times too much area; weighting each face by the
                # |n_axis| of the nearest triangle restores the true
                # area (sum of |n_axis|^2 = 1).
                w = np.ones(len(wi))
                if face_weighting and vox.nrmq is not None:
                    ax3 = f // 2
                    nrm = vox.nrmq[jw[:, 0], jw[:, 1], jw[:, 2], ax3]
                    ww = np.maximum(nrm.astype(np.float64) / 255.0, 0.05)
                    w[inb] = np.where(m[inb] >= 0, ww, 1.0)
                if face_area_w is not None:
                    # true areas (conformal): override nrmq wherever
                    # a patch has been mapped onto this face
                    fa = np.asarray(face_area_w[wi, f], dtype=np.float64)
                    ov = inb & (fa >= 0.0)
                    w[ov] = fa[ov]
                # grid-border face: rigid (0) or matched (1)
                g[~inb] = gamma_border
                gam[wi] += g * w
                if fA is not None:
                    # the dynamic faces of this direction, in order: one array per face
                    # direction gives the same values and order as one entry per face
                    mm = np.clip(m, 0, len(gm) - 1)
                    dyn = (m >= 0) & (np.abs(fA[mm]).sum(axis=1) > 0)
                    kd = np.nonzero(dyn)[0]
                    face_cell.append(wi[kd])
                    face_A.append(fA[mm[kd]] * w[kd][:, None])
                    face_l.append(fL[mm[kd]])
                if filt_res is not None:
                    mm = np.clip(m, 0, len(filt_res) - 1)
                    for k in np.nonzero(m >= 0)[0]:
                        secs = filt_res[mm[k]]
                        if not secs:
                            continue
                        for sec in secs:
                            # amplitude x face area (nrmq or
                            # conformal), like first-order branches
                            res_cell.append(wi[k])
                            res_par.append((float(w[k]), tuple(sec)))
        if not nat:
            del pad, pos
        _tm('neighbours + wall faces')
        self.nb = nb
        self.K = K
        gam_sum = gam                          # (lambda/2) weighted g_inf_f sum, below
        # volume damping (absorbing layer): s = sigma*dt/2, folded into
        # the SAME coefficients -> the kernels do not change
        if sigma is not None:
            sg = np.asarray(sigma, dtype=np.float64)[tuple(ii.T)]
            self.s_half = 0.5 * sg * self.dt
        else:
            self.s_half = np.zeros(self.n_air)
        if phi is not None:
            ph = np.asarray(phi, dtype=np.float64)
            if ph.shape != (self.n_air,):
                raise ValueError("phi: expected (n_air,) = (%d,), got %s"
                                 % (self.n_air, ph.shape))
            # phi < 1 would tighten the CFL condition (small cells):
            # the DONATION construction never yields that case, we
            # clamp it as a safeguard.
            ph = np.maximum(ph, 1.0)
        else:
            ph = np.ones(self.n_air, dtype=np.float64)
        self.phi = ph
        need_drv = any(len(x) for x in face_cell) or bool(res_cell)
        co = (gridnative.coefficients(0.5 * self.lam, self.lam ** 2, self.lam * self.dt,
                                      gam_sum, K, self.s_half, ph, need_drv) if nat else None)
        if co is not None:                     # the same operations, in C++ (bit for bit)
            self.Gamma, self.c_center, self.c_old, self.c_now, self.c_nb, c_drv_n = co
            den = None
        else:
            self.Gamma = 0.5 * self.lam * gam_sum
            den = ph + self.Gamma + self.s_half
            self.c_center = 2.0 * ph - K * (self.lam ** 2)
            self.c_old = (self.Gamma + self.s_half - ph) / den
            self.c_now = self.c_center / den
            self.c_nb = (self.lam ** 2) / den
        # filtering walls: per-face states, per-cell drive
        self.f_cell = (np.concatenate(face_cell).astype(np.int32)
                       if face_cell else np.zeros(0, dtype=np.int32))
        self.f_A = (np.concatenate(face_A).astype(np.float64)
                    if face_A else np.zeros((0, 2)))
        self.f_l = (np.concatenate(face_l).astype(np.float64)
                    if face_l else np.zeros((0, 2)))
        self.n_faces = len(self.f_cell)
        if self.n_faces:
            dt = self.dt
            self.f_e = np.exp(-self.f_l * dt)          # e^{-l dt}
            self.f_g = np.where(self.f_l > 0,
                                self.f_A / np.maximum(self.f_l, 1e-30)
                                * (1.0 - self.f_e), 0.0)
            self.f_Asum = self.f_A.sum(axis=1)
        # ---- resonant sections (panels): exact propagator ------------
        # Drive convention (see wall_R_discrete): D carries gamma*s*p.
        # A section = physical poles (b, w0^2) + THREE free output
        # coefficients (rA, rc1, rc2) -- zeros tuned in z by
        # polish_resonant. Physical 3-tuple form (a, b, w2) accepted:
        # (rA, rc1, rc2) = (a, a w2, a b). The face weight w scales all
        # three (flux), like the branch amplitudes.
        self.r_cell = np.asarray(res_cell, dtype=np.int32)
        self.n_rfaces = len(self.r_cell)
        if self.n_rfaces:
            from .reflection import _res_five
            fv = [(_res_five(sec), wf) for (wf, sec) in res_par]
            self.r_A = np.array([wf * s5[0] for s5, wf in fv])
            self.r_c1 = np.array([wf * s5[1] for s5, wf in fv])
            self.r_c2 = np.array([wf * s5[2] for s5, wf in fv])
            self.r_E = np.empty((self.n_rfaces, 4), dtype=np.float64)
            self.r_G = np.empty((self.n_rfaces, 2), dtype=np.float64)
            prop_cache = {}
            for i, (s5, _wf) in enumerate(fv):
                rb, rw2 = float(s5[3]), float(s5[4])
                key = (round(rb, 9), round(rw2, 6))
                if key not in prop_cache:
                    prop_cache[key] = _soss_prop(rb, rw2, self.dt)
                E, G = prop_cache[key]
                self.r_E[i] = E.ravel()
                self.r_G[i] = G
        else:
            self.r_A = np.zeros(0)
            self.r_c1 = np.zeros(0)
            self.r_c2 = np.zeros(0)
            self.r_E = np.zeros((0, 4))
            self.r_G = np.zeros((0, 2))
        if self.n_faces or self.n_rfaces:
            self.c_drv = c_drv_n if den is None else (self.lam * self.dt) / den  # dyn term

        # ---- second-step neighbors + cell classes (order 4) ----------
        # column by column: a 2-D fancy index would turn nb into intp (48 B/cell)
        _tm('coefficients + face tables')
        if nat:
            self.nb2 = nb2_n
        else:
            self.nb2 = np.empty_like(nb)
            for f in range(6):
                self.nb2[:, f] = np.append(nb[:, f], ghost)[nb[:, f]]
        # ---- tangential apertures of the conformal cells (sparse) ----------
        # A face of axis b between air cells i and j gets the aperture
        # w = 1 + mean over (i, j) of the excess donated along the OTHER
        # axes: the donated slab conducts the flow parallel to the wall.
        # Stored as links (cell, neighbour, k = lam^2 (w - 1) / den) in CSR
        # by cell: p2[i] += sum k (p[j] - p[i]). Default (None): no links.
        self.l_cell = np.zeros(0, dtype=np.int32)
        self.l_nb = np.zeros(0, dtype=np.int32)
        self.l_k = np.zeros(0, dtype=np.float64)
        self.l_ptr = np.zeros(1, dtype=np.int64)
        self.l_rows = np.zeros(0, dtype=np.int32)
        has_link = np.zeros(self.n_air, dtype=bool)
        cells, nbs, ks = [], [], []
        if excess_axis is not None:
            # only a cell with an excess, or its neighbour, can get w > 0: the
            # links are built on those pairs alone (same values and order as
            # a pass over every cell). excess_axis: dense (3, n_air) or sparse
            # (cells, (3, m)) as conformal_corrections returns it.
            if isinstance(excess_axis, tuple):
                ecell = np.asarray(excess_axis[0], dtype=np.int64)
                ex = np.asarray(excess_axis[1], dtype=np.float64)
                srt = np.argsort(ecell, kind="stable")
                ecell, ex = ecell[srt], ex[:, srt]
            else:
                exd = np.asarray(excess_axis, dtype=np.float64)
                ecell = np.nonzero((exd != 0.0).any(axis=0))[0]
                ex = exd[:, ecell]
            perp = np.stack([ex[1] + ex[2], ex[0] + ex[2], ex[0] + ex[1]], axis=1)
            n_e = len(ecell)

            def _rank(cand):
                """Position of cand in ecell, -1 if absent."""
                k = np.minimum(np.searchsorted(ecell, cand), max(n_e - 1, 0))
                return np.where((n_e > 0) & (ecell[k] == cand), k, -1)
            if den is None:                    # coefficients came from C++: the same den
                den = ph + self.Gamma + self.s_half
            for f in range(6):
                b = f // 2
                # pairs (e, j): the excess cell reads its neighbour j
                j = nb[ecell, f].astype(np.int64)
                ok = j < ghost
                kj = _rank(j[ok])
                pj = np.where(kj >= 0, perp[np.maximum(kj, 0), b], 0.0)
                c1, n1 = ecell[ok], j[ok]
                w1 = 0.5 * (perp[ok, b] + pj)
                # pairs (i, e), i without excess: i's face f reaches e
                i = nb[ecell, f ^ 1].astype(np.int64)
                ok = i < ghost
                ok[ok] = _rank(i[ok]) < 0
                c2, n2 = i[ok], ecell[ok]
                w2 = 0.5 * (0.0 + perp[ok, b])
                c, n_, w = (np.concatenate([c1, c2]), np.concatenate([n1, n2]),
                            np.concatenate([w1, w2]))
                keep = w > 1e-9
                o = np.argsort(c[keep], kind="stable")
                sel = c[keep][o]
                w = w[keep][o]
                cells.append(sel)
                nbs.append(n_[keep][o])
                ks.append((self.lam ** 2) * w / den[sel])
        if extra_links is not None and len(extra_links[2]):
            # sub-cell openings (subcell.open_pockets): a symmetric conductance w between two
            # air cells, in face units, after the tangential apertures (same CSR)
            if den is None:
                den = ph + self.Gamma + self.s_half
            fl = idx.ravel()
            ea = fl[np.asarray(extra_links[0], dtype=np.int64)].astype(np.int64)
            eb = fl[np.asarray(extra_links[1], dtype=np.int64)].astype(np.int64)
            if (ea < 0).any() or (eb < 0).any():
                raise ValueError("sub-cell link on a cell outside the air")
            ew = np.asarray(extra_links[2], dtype=np.float64)
            c2 = np.concatenate([ea, eb])
            o = np.argsort(c2, kind="stable")
            cells.append(c2[o])
            nbs.append(np.concatenate([eb, ea])[o])
            ks.append(((self.lam ** 2) * np.concatenate([ew, ew]) / den[c2])[o])
        if cells:
            c_all = np.concatenate(cells)
            order_ = np.argsort(c_all, kind="stable")
            self.l_cell = c_all[order_].astype(np.int32)
            self.l_nb = np.concatenate(nbs)[order_].astype(np.int32)
            self.l_k = np.concatenate(ks)[order_]
            rows, start = np.unique(self.l_cell, return_index=True)
            self.l_rows = rows.astype(np.int32)
            self.l_ptr = np.append(start, len(self.l_cell)).astype(np.int64)
            has_link[rows] = True
        _tm('nb2 + links')
        self.n_links = len(self.l_cell)
        if self.order == 4:
            if nat:
                all1, all2 = full_n, True
            else:
                all1 = (nb < ghost).all(axis=1)
                all2 = (self.nb2 < ghost).all(axis=1)
            # volume-corrected cells (phi != 1) all touch a solid, so
            # they already sit at order 2 through all1; explicit guard
            # in case a caller passes an arbitrary phi. Cells with links
            # (tangential apertures) also stay at order 2.
            is4 = all1 & all2 & (self.s_half <= 0.0) & (ph == 1.0) & ~has_link
        else:
            is4 = np.zeros(self.n_air, dtype=bool)

        # ---- CPML (pressure formulation, kappa = 1) -------------------
        # The sigma profile is renormalized over (n-1) ACTIVE cells and
        # any cell with ANY of its 6 neighbors ghost is EXCLUDED from
        # ALL the tables: the phi recursion against a rigid termination
        # -- axial OR tangential (corners, solids inside the slab) --
        # is a GKS-type boundary instability, invisible to von Neumann.
        _tm('order-4 classes')
        self.pml_axes = []
        self.pml_n = int(pml.get("n", 0)) if pml is not None else 0
        if pml is not None and int(pml.get("n", 0)) > 0:
            npml = int(pml["n"])
            m_gr = float(pml.get("m", 3.0))
            R0 = float(pml.get("R0", 1e-6))
            smax_dt = (m_gr + 1.0) * (-np.log(R0)) * self.lam \
                / (2.0 * npml)
            amax_dt = float(pml.get("alpha_dt", 0.06))
            # a slab's sub grid (SlabGrid): the CPML of the WHOLE box, in its coordinates
            shp = tuple(int(v) for v in pml.get("shape", vox.shape))
            off_p = pml.get("offset")
            ii_p = ii if off_p is None else ii + np.asarray(off_p, dtype=np.int32)
            eff = max(npml - 1, 1)
            any_ghost = None if nat else (nb >= self.n_air).any(axis=1)
            for a in range(3):
                if nat:
                    # cells and minus-neighbour positions in C++; u on those cells only
                    # (the same values as on every cell, then indexed)
                    sel, locm_a = gridnative.pml_axis(ii_p, nb, shp, npml, a)
                    if len(sel) == 0:
                        continue
                    ia = ii_p[sel, a]
                    us = np.clip(np.maximum((npml - 1 - ia) / eff,
                                            (ia - (shp[a] - npml)) / eff), 0.0, 1.0)
                    nm = nb[sel, 2 * a]
                    np_ = nb[sel, 2 * a + 1]
                    kp = np.minimum(np.searchsorted(sel, np_), len(sel) - 1)
                    locp_a = np.where(sel[kp] == np_, kp, -1).astype(np.int32)
                else:
                    ia = ii_p[:, a]
                    u = np.maximum((npml - 1 - ia) / eff,
                                   (ia - (shp[a] - npml)) / eff)
                    u = np.clip(u, 0.0, 1.0)
                    mask = (u > 0.0) & ~any_ghost
                    sel = np.nonzero(mask)[0]
                    if len(sel) == 0:
                        continue
                    us = u[sel]
                    loc = np.full(self.n_air + 1, -1, dtype=np.int32)
                    loc[sel] = np.arange(len(sel), dtype=np.int32)
                    nm = nb[:, 2 * a][sel]
                    np_ = nb[:, 2 * a + 1][sel]
                    locm_a, locp_a = loc[nm], loc[np_]
                    del u, mask, loc
                sdt = smax_dt * us ** m_gr
                adt = amax_dt * (1.0 - us)
                tot = sdt + adt
                b = np.exp(-tot)
                aa = np.where(tot > 0,
                              sdt / np.maximum(tot, 1e-30) * (b - 1.0),
                              0.0)
                a_psi = np.where(np_ >= self.n_air, 0.0, aa)
                self.pml_axes.append({
                    "cells": sel, "b": b, "a": aa, "a_psi": a_psi,
                    "nbm": nm, "nbp": np_,
                    "locm": locm_a, "locp": locp_a})
        _tm('CPML tables')
        if self.pml_axes:
            # collar: the order-4 interior must not read the slab
            # through its stencil (0.4 % CFL margin -> no coupling)
            pmlm = np.zeros(self.n_air + 1, dtype=bool)
            for ax in self.pml_axes:
                pmlm[ax["cells"]] = True
            if nat:
                is4 &= ~gridnative.collar(nb, self.nb2, pmlm)
            else:
                bad = np.zeros(self.n_air, dtype=bool)
                for f in range(6):
                    bad |= pmlm[nb[:, f]]
                    bad |= pmlm[self.nb2[:, f]]
                is4 &= ~(pmlm[:-1][np.arange(self.n_air)] | bad)
        _tm('CPML collar')
        self.is4 = is4
        self.i4 = np.nonzero(is4)[0]
        self.i2 = np.nonzero(~is4)[0]
        l2 = self.lam ** 2
        self.c4_c = 2.0 - 7.5 * l2
        self.c4_nb = (4.0 / 3.0) * l2
        self.c4_nb2 = -l2 / 12.0
        self.lam2 = l2
        _tm('class lists', done=True)

    # -- trilinear sampling ---------------------------------------------
    def taps(self, pos):
        """(8 indices, 8 weights) to read/inject at an exact position."""
        f = (np.asarray(pos, float) - self.vox.origin) / self.dx
        i0 = np.floor(f).astype(int)
        fr = f - i0
        out_i, out_w = [], []
        for dxi in (0, 1):
            for dyi in (0, 1):
                for dzi in (0, 1):
                    j = i0 + (dxi, dyi, dzi)
                    w = ((fr[0] if dxi else 1 - fr[0])
                         * (fr[1] if dyi else 1 - fr[1])
                         * (fr[2] if dzi else 1 - fr[2]))
                    ci = -1
                    if all(0 <= j[k] < self.vox.shape[k] for k in range(3)):
                        ci = int(self.cell_index[tuple(j)])
                    if ci >= 0 and w > 0:
                        out_i.append(ci)
                        out_w.append(w)
        if not out_i:
            raise ValueError("position outside the air: %s" % (pos,))
        w = np.asarray(out_w)
        return np.asarray(out_i, dtype=np.int32), w / w.sum()

    def source_taps(self, pos):
        """taps() of a source: the same volume velocity whatever the cells' volumes.

        A soft source adds pressure, and a cell of volume phi dx^3 (the conformal cut layer,
        phi up to ~2.3) turns it into phi times the volume, so the weights are divided by phi.
        Without this, a source within ~3 cells of a wall radiates up to 3.4 dB too much.
        Interior cells (phi = 1) are unchanged, bit for bit."""
        i, w = self.taps(pos)
        return i, w / self.volumes(i)

    def volumes(self, i):
        """phi of cells i (a kept grid holds only its cut cells', _slim_kept_grid)."""
        if self.phi is not None:
            return self.phi[i]
        cut, val = self.phi_cut
        out = np.ones(len(i))
        if len(cut):
            k = np.clip(np.searchsorted(cut, i), 0, len(cut) - 1)
            hit = cut[k] == i
            out[hit] = val[k[hit]]
        return out


def surface_taps(grid, scene, mat_k: int):
    """(cells, weights) of a vibrating surface of material mat_k.

    The faces between an air cell and a solid cell whose nearest triangle (from the face
    centre, within two cells) is of material mat_k, each weighted by abs(n . axis) of that
    triangle (its true area). Weights are summed per air cell, normalised to 1 (the volume
    velocity of the unit point source) and divided by the cells' volumes phi.
    The material comes from the geometry, not the voxels: an edge cell holds one material
    only, so part of the rim would be missed and a uniform wall would excite transverse modes."""
    vox = grid.vox
    ci = grid.cell_index
    dx = float(vox.dx)
    tv = np.asarray(scene.tri_verts, dtype=np.float64).reshape(-1, 3, 3)
    tm = np.asarray(scene.tri_mat)
    panel = tm == int(mat_k)
    if not panel.any():
        raise ValueError("no surface of material %d in the scene" % int(mat_k))
    lo = tv[panel].min(axis=(0, 1)) - 2.5 * dx
    hi = tv[panel].max(axis=(0, 1)) + 2.5 * dx
    tlo, thi = tv.min(axis=1), tv.max(axis=1)
    near = np.nonzero(((thi >= lo) & (tlo <= hi)).all(axis=1))[0]
    nrm = np.cross(tv[near, 1] - tv[near, 0], tv[near, 2] - tv[near, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    org = np.asarray(vox.origin, dtype=np.float64)
    i0 = np.maximum(np.floor((lo - org) / dx).astype(int) - 1, 0)
    i1 = np.minimum(np.ceil((hi - org) / dx).astype(int) + 2, np.asarray(vox.shape))
    box = tuple(slice(a, b) for a, b in zip(i0, i1))
    cib = ci[box]
    solb = vox.solid[box]
    acc = np.zeros(grid.n_air)
    for ax in range(3):
        for sgn in (1, -1):
            a_sl = [slice(None)] * 3
            s_sl = [slice(None)] * 3
            m = cib.shape[ax]
            a_sl[ax], s_sl[ax] = ((slice(0, m - 1), slice(1, m)) if sgn > 0
                                  else (slice(1, m), slice(0, m - 1)))
            a_idx = cib[tuple(a_sl)]
            hit = (a_idx >= 0) & solb[tuple(s_sl)]
            if not hit.any():
                continue
            ijk = np.argwhere(hit) + i0
            if sgn < 0:
                ijk[:, ax] += 1
            fc = org + ijk * dx
            fc[:, ax] += 0.5 * sgn * dx                # the face centre
            best = np.full(len(fc), np.inf)
            arg = np.full(len(fc), -1)
            for t, k in enumerate(near):
                sel = np.nonzero(((fc >= tlo[k] - 2 * dx) & (fc <= thi[k] + 2 * dx)).all(axis=1))[0]
                if not len(sel):
                    continue
                d = _point_tri_dist(fc[sel], tv[k, 0], tv[k, 1], tv[k, 2])
                upd = d < best[sel]
                best[sel[upd]] = d[upd]
                arg[sel[upd]] = t
            ok = (arg >= 0) & (best <= 2.0 * dx)          # the staircase lies up to ~1.4 dx off
            ok[ok] &= panel[near[arg[ok]]]
            if not ok.any():
                continue
            w = np.abs(nrm[arg[ok], ax])
            np.add.at(acc, a_idx[hit][ok], w)
    cells = np.flatnonzero(acc > 0)
    if not len(cells) or acc.sum() <= 0:
        raise ValueError("the vibrating surface (material %d) touches no air cell of the wave "
                         "grid" % int(mat_k))
    w = acc[cells]
    # volume velocity per cell ~ its face area; a soft source in a cell of volume phi dx^3
    # gives phi times the volume (FdtdGrid.source_taps)
    return cells.astype(np.int32), w / w.sum() / grid.volumes(cells)


def _soss_prop(beta: float, w02: float, dt: float):
    """Exact propagator of the face resonant system x' = M x + B p,
    M = [[0,1],[-w0^2,-beta]]: returns (E = expm(dt M), M^-1 (E-I) B)."""
    beta = float(max(beta, 0.0))
    w02 = float(max(w02, 1e-12))
    d = 0.5 * beta
    disc = w02 - d * d
    ed = np.exp(-d * dt)
    if disc > 1e-12:                      # underdamped
        om = np.sqrt(disc)
        c = ed * np.cos(om * dt)
        s = ed * np.sin(om * dt) / om
    elif disc < -1e-12:                   # overdamped
        om = np.sqrt(-disc)
        c = ed * np.cosh(om * dt)
        s = ed * np.sinh(om * dt) / om
    else:                                 # critical
        c = ed
        s = ed * dt
    E = np.array([[c + d * s, s],
                  [-w02 * s, c - d * s]], dtype=np.float64)
    # G = M^-1 (E - I) B; M^-1 = [[-beta/w0^2, -1/w0^2], [1, 0]]
    e12 = E[0, 1]
    e22m = E[1, 1] - 1.0
    G = np.array([(-beta * e12 - e22m) / w02, e12], dtype=np.float64)
    return E, G


def source_pulse(dt: float, f_peak: float, n_steps: int) -> np.ndarray:
    """Differentiated Gaussian (zero at f = 0), spectral peak at f_peak."""
    sig = 1.0 / (2.0 * np.pi * f_peak)
    t0 = 5.0 * sig
    t = np.arange(n_steps) * dt
    s = -(t - t0) / sig * np.exp(-0.5 * ((t - t0) / sig) ** 2)
    m = np.abs(s).max()
    return (s / m if m > 0 else s).astype(np.float64)


def run_grid_python(grid: FdtdGrid, pulse: np.ndarray, src_taps, rec_taps,
                    n_steps: int) -> np.ndarray:
    """Reference time loop (numpy). Returns (n_steps, n_rec): order-4
    interior, boundaries / sponge / PML at order 2."""
    _ENGINE_USED[0] = "python"        # tags WHERE the compute happens
    n = grid.n_air
    p0 = np.zeros(n + 1)
    p1 = np.zeros(n + 1)
    nb = grid.nb
    nb2 = grid.nb2
    i4, i2 = grid.i4, grid.i2
    c_now, c_old, c_nb = grid.c_now, grid.c_old, grid.c_nb
    cn2, co2, cb2 = c_now[i2], c_old[i2], c_nb[i2]
    nb_2 = nb[i2]
    nb_4 = nb[i4]
    nb2_4 = nb2[i4]
    nl = getattr(grid, "n_links", 0)
    if nl:
        l_cell, l_nb, l_k = grid.l_cell, grid.l_nb, grid.l_k
    nf = getattr(grid, "n_faces", 0)
    if nf:
        fc, fAs = grid.f_cell, grid.f_Asum
        fe, fg, fl = grid.f_e, grid.f_g, grid.f_l
        phi = np.zeros((nf, 2))
    nr = getattr(grid, "n_rfaces", 0)
    if nr:
        rc, rA = grid.r_cell, grid.r_A
        rc1, rc2 = grid.r_c1, grid.r_c2
        rE, rG = grid.r_E, grid.r_G
        psiR = np.zeros((nr, 2))
    if nf or nr:
        cdrv = grid.c_drv
    pml = grid.pml_axes
    if pml:
        psi = [np.zeros(len(ax["cells"])) for ax in pml]
        pphi = [np.zeros(len(ax["cells"])) for ax in pml]
    si, sw = src_taps
    recs = [rec_taps] if isinstance(rec_taps, tuple) else list(rec_taps)
    out = np.zeros((n_steps, len(recs)))
    for it in range(n_steps):
        p2 = np.empty(n + 1)
        if len(i4):
            s4 = (p1[nb_4[:, 0]] + p1[nb_4[:, 1]] + p1[nb_4[:, 2]]
                  + p1[nb_4[:, 3]] + p1[nb_4[:, 4]] + p1[nb_4[:, 5]])
            s4b = (p1[nb2_4[:, 0]] + p1[nb2_4[:, 1]] + p1[nb2_4[:, 2]]
                   + p1[nb2_4[:, 3]] + p1[nb2_4[:, 4]] + p1[nb2_4[:, 5]])
            p2[i4] = (grid.c4_c * p1[i4] + grid.c4_nb * s4
                      + grid.c4_nb2 * s4b - p0[i4])
        s2 = (p1[nb_2[:, 0]] + p1[nb_2[:, 1]] + p1[nb_2[:, 2]]
              + p1[nb_2[:, 3]] + p1[nb_2[:, 4]] + p1[nb_2[:, 5]])
        p2[i2] = cn2 * p1[i2] + cb2 * s2 + co2 * p0[i2]
        if nl:
            np.add.at(p2, l_cell, l_k * (p1[l_nb] - p1[l_cell]))
        if nf or nr:
            D = np.zeros(n)
            if nf:
                # filtering walls: d = Asum*p - sum_k l_k phi_k (time n)
                drive = fAs * p1[fc] - (fl * phi).sum(axis=1)
                np.add.at(D, fc, drive)
            if nr:
                # resonant sections: gamma*s*p drive (same convention
                # as the branches): y = a p - a w0^2 psi1 - a b psi2
                np.add.at(D, rc, rA * p1[rc]
                          - (rc1 * psiR[:, 0] + rc2 * psiR[:, 1]))
            p2[:n] -= cdrv * D
        if pml:
            # stable staggered form: psi lives at the (+) half-node of
            # each cell, compact differences (the checkerboard is not in
            # the operator null space, unlike 2h-centered ones)
            for k, ax in enumerate(pml):
                cc = ax["cells"]
                dh = p1[ax["nbp"]] - p1[cc]          # h * dp/dx at i+1/2
                psi[k] = (ax.get("b_psi", ax["b"]) * psi[k]
                          + ax["a_psi"] * dh)
            for k, ax in enumerate(pml):
                cc = ax["cells"]
                pm = np.where(ax["locm"] >= 0, psi[k][ax["locm"]], 0.0)
                dpsi = psi[k] - pm                   # h^2 * d(psi)/dx at i
                d2 = p1[ax["nbp"]] - 2.0 * p1[cc] + p1[ax["nbm"]]
                pphi[k] = ax["b"] * pphi[k] + ax["a"] * (d2 + dpsi)
                p2[cc] += grid.lam2 * (dpsi + pphi[k])
        p2[n] = 0.0
        if it < len(pulse):
            p2[si] += pulse[it] * sw
        for r, (ri, rw) in enumerate(recs):
            out[it, r] = float(p2[ri] @ rw)
        if nf:
            # trapezoidal branch: input (p^{n+1} + p^n)/2
            phi = fe * phi + fg * (0.5 * (p2[fc] + p1[fc]))[:, None]
        if nr:
            pb = 0.5 * (p2[rc] + p1[rc])
            s1 = rE[:, 0] * psiR[:, 0] + rE[:, 1] * psiR[:, 1] \
                + rG[:, 0] * pb
            s2 = rE[:, 2] * psiR[:, 0] + rE[:, 3] * psiR[:, 1] \
                + rG[:, 1] * pb
            psiR[:, 0] = s1
            psiR[:, 1] = s2
        p0, p1 = p1, p2
    return out



# ---------------------------------------------------------------------------
# Native bridge (threaded CPU / CUDA): same aur_fdtd_run ABI for both
# ---------------------------------------------------------------------------

def _bind_fdtd(lib):
    import ctypes as ct
    fn = lib.aur_fdtd_run
    fn.restype = ct.c_int32
    fn.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_float),
                   ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                   ct.c_int64,
                   ct.POINTER(ct.c_float), ct.c_int32,
                   ct.POINTER(ct.c_int32), ct.POINTER(ct.c_float),
                   ct.c_int32,
                   ct.POINTER(ct.c_int32), ct.POINTER(ct.c_float),
                   ct.c_int32, ct.c_int32, ct.c_int32,
                   ct.POINTER(ct.c_float)]
    return fn


_PROGRESS_HOOK = [None]


def set_progress_hook(fn):
    """fn(step, n_steps), called 4 times a second during native FDTD runs (ABI 2.3,
    the kernels write their step into a counter); None turns it off."""
    _PROGRESS_HOOK[0] = fn


def _with_progress(lib, n_steps, call):
    """Run `call` (a native FDTD run) while a thread reports its real progress."""
    hook = _PROGRESS_HOOK[0]
    if hook is None or not hasattr(lib, "aur_fdtd_progress"):
        return call()
    import ctypes as ct
    import threading
    counter = ct.c_int64(0)
    lib.aur_fdtd_progress.argtypes = [ct.POINTER(ct.c_int64)]
    lib.aur_fdtd_progress.restype = None
    done = threading.Event()

    def poll():
        while not done.wait(0.25):
            hook(int(counter.value), int(n_steps))

    th = threading.Thread(target=poll, daemon=True)
    lib.aur_fdtd_progress(ct.byref(counter))
    th.start()
    try:
        return call()
    finally:
        done.set()
        th.join()
        lib.aur_fdtd_progress(None)


def _fdtd_devices():
    """GPUs of a split run, AURALIS_FDTD_GPUS="0,1,2,3" (CUDA ordinals; "0,0" splits on one
    card, for tests); None: one device."""
    v = os.environ.get("AURALIS_FDTD_GPUS", "").strip()
    if not v:
        return None
    try:
        devs = [int(x) for x in v.replace(" ", "").split(",") if x != ""]
    except ValueError:
        print("[Auralis] WARNING: AURALIS_FDTD_GPUS='%s' is not a list of GPU numbers: one "
              "device." % v)
        return None
    return devs or None


def slab_cuts(grid, n_dev):
    """(cuts (n_dev+1,), halo (2 n_dev,)) of n_dev slabs of whole x planes holding about as
    many air cells each, with their two-plane copies; cuts kept 3 planes clear of the x PML
    and slabs at least 3 planes thick. None if the grid cannot be split that way."""
    x = np.asarray(grid.cells[:, 0], dtype=np.int64)
    nx = int(grid.vox.shape[0])
    start = np.searchsorted(x, np.arange(nx + 1))           # first air cell of each x plane
    npml = int(getattr(grid, "pml_n", 0) or 0)
    lo, hi = npml + 3, nx - npml - 3
    # a link reaching 3 or more planes away (sub-cell openings) would leave a slab's two-plane
    # copy: no cut through it
    banned = np.zeros(nx + 1, dtype=bool)
    if getattr(grid, "n_links", 0):
        xa = x[np.asarray(grid.l_cell, dtype=np.int64)]
        xb = x[np.asarray(grid.l_nb, dtype=np.int64)]
        far = np.abs(xa - xb) >= 3
        for a_, b_ in set(zip(np.minimum(xa, xb)[far].tolist(), np.maximum(xa, xb)[far].tolist())):
            banned[a_ + 1:b_ + 1] = True
    planes = [0]
    for k in range(1, n_dev):
        first = max(lo, planes[-1] + 3)
        last = hi - 3 * (n_dev - 1 - k)
        if first > last:
            return None
        cand = np.arange(first, last + 1)
        cand = cand[~banned[cand]]
        if len(cand) == 0:
            return None
        target = grid.n_air * k / n_dev
        planes.append(int(cand[np.argmin(np.abs(start[cand] - target))]))
    planes.append(nx)
    if nx - planes[-2] < 3:
        return None
    cuts = start[planes].astype(np.int64)
    if np.any(np.diff(cuts) <= 0):
        return None
    halo = np.empty(2 * n_dev, dtype=np.int64)
    for k in range(n_dev):
        halo[2 * k] = start[max(planes[k] - 2, 0)]
        halo[2 * k + 1] = start[min(planes[k + 1] + 2, nx)]
    return cuts, halo


def _run_kept(lib, handle, grid, pulse, src_taps, rec_taps, n_steps):
    """One run on a kept grid: only the pulse, the source taps and the probes go over."""
    import ctypes as ct
    P = ct.POINTER
    ghost = grid.n_air
    pu = np.ascontiguousarray(pulse, dtype=np.float32)
    si = np.ascontiguousarray(src_taps[0], dtype=np.int32)
    sw = np.ascontiguousarray(src_taps[1], dtype=np.float32)
    recs = [rec_taps] if isinstance(rec_taps, tuple) else list(rec_taps)
    ri = np.full((len(recs), 8), ghost, dtype=np.int32)
    rw = np.zeros((len(recs), 8), dtype=np.float32)
    for r, (i, w) in enumerate(recs):
        ri[r, :len(i)] = i
        rw[r, :len(w)] = w
    out = np.zeros((n_steps, len(recs)), dtype=np.float32)
    frun = lib.aur_fdtd_session4m_run
    frun.restype = ct.c_int32
    frun.argtypes = [ct.c_void_p, P(ct.c_float), ct.c_int32, P(ct.c_int32), P(ct.c_float),
                     ct.c_int32, P(ct.c_int32), P(ct.c_float), ct.c_int32, ct.c_int32,
                     P(ct.c_float)]
    rc = _with_progress(lib, n_steps, lambda: frun(
        handle, pu.ctypes.data_as(P(ct.c_float)), len(pu), si.ctypes.data_as(P(ct.c_int32)),
        sw.ctypes.data_as(P(ct.c_float)), len(si), ri.ctypes.data_as(P(ct.c_int32)),
        rw.ctypes.data_as(P(ct.c_float)), len(recs), n_steps, out.ctypes.data_as(P(ct.c_float))))
    if rc != 0:
        raise RuntimeError("aur_fdtd_session4m_run failed (code %d%s)" % (
            rc, ": instability caught by the watchdog" if rc == 41 else ""))
    return out.astype(np.float64)


def _slim_kept_grid(grid):
    """Once its session holds the grid on the device(s), the host arrays are only needed to
    place sources and probes: the rest is dropped (host memory of a large scene)."""
    from types import SimpleNamespace
    ph = np.asarray(grid.phi)
    cut = np.flatnonzero(ph != 1.0)
    grid.phi_cut = (cut, ph[cut])          # the cut cells' volumes, for source_taps
    for k in ("nb", "nb2", "K", "Gamma", "s_half", "phi", "c_center", "c_old", "c_now", "c_nb",
              "c_drv", "f_cell", "f_A", "f_l", "f_e", "f_g", "f_Asum", "r_cell", "r_A", "r_c1",
              "r_c2", "r_E", "r_G", "l_cell", "l_nb", "l_k", "l_ptr", "l_rows", "is4", "i4",
              "i2", "pml_axes", "cells"):
        if hasattr(grid, k):
            setattr(grid, k, None)
    v = grid.vox
    grid.vox = SimpleNamespace(origin=v.origin, dx=v.dx, shape=v.shape)


def _kept_session(lib, grid, devs, grid_args, link_args=()):
    """The native session of a kept grid (created at its first run, then reused), or None
    (loud) if the library refuses it. link_args: the 5 tangential-link arguments of run5
    (conformal_aniso), through aur_fdtd_session5m_create (ABI 2.7)."""
    import ctypes as ct
    hit = _KEEP["sessions"].get(id(grid))
    if hit is not None and hit[2] is grid and hit[0] is lib:
        return hit[1]
    n_dev = len(devs) if devs else 1
    if n_dev > 1:
        cut = slab_cuts(grid, n_dev)
        if cut is None:
            print("[Auralis] WARNING: kept FDTD grid: %d slabs impossible (too thin along x): "
                  "one device." % n_dev)
            n_dev, devs = 1, None
    if n_dev == 1:
        cuts = np.array([0, grid.n_air], dtype=np.int64)
        halo = np.array([0, grid.n_air], dtype=np.int64)
        dv = np.array([devs[0] if devs else 0], dtype=np.int32)
    else:
        cuts, halo = cut
        dv = np.ascontiguousarray(devs, dtype=np.int32)
    P = ct.POINTER
    fc = lib.aur_fdtd_session5m_create if link_args else lib.aur_fdtd_session4m_create
    fc.restype = ct.c_void_p
    fc.argtypes = ([ct.c_int32, P(ct.c_int32), P(ct.c_int64), P(ct.c_int64)]
                   + [P(ct.c_int32), P(ct.c_int32), P(ct.c_uint8),
                      P(ct.c_float), P(ct.c_float), P(ct.c_float), P(ct.c_float),
                      ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_int64,
                      ct.c_int64, P(ct.c_int32), P(ct.c_float), P(ct.c_float), P(ct.c_float),
                      P(ct.c_float),
                      ct.c_int64, P(ct.c_int32), P(ct.c_float), P(ct.c_float), P(ct.c_float),
                      P(ct.c_float), P(ct.c_float),
                      P(ct.c_int64), P(ct.c_int32), P(ct.c_float), P(ct.c_float), P(ct.c_float),
                      P(ct.c_int32), P(ct.c_int32), P(ct.c_int32)]
                   + ([ct.c_int64, P(ct.c_int32), P(ct.c_int64), P(ct.c_int32), P(ct.c_float)]
                      if link_args else [ct.c_int64])
                   + [P(ct.c_int32)])
    status = ct.c_int32(0)
    handle = fc(ct.c_int32(n_dev), dv.ctypes.data_as(P(ct.c_int32)),
                cuts.ctypes.data_as(P(ct.c_int64)), halo.ctypes.data_as(P(ct.c_int64)),
                *grid_args, *(list(link_args) if link_args else [ct.c_int64(0)]),
                ct.byref(status))
    if not handle:
        print("[Auralis] WARNING: kept FDTD grid refused by the library (code %d): a fresh "
              "upload per run." % status.value)
        return None
    _KEEP["sessions"][id(grid)] = (lib, handle, grid)
    _slim_kept_grid(grid)
    print("[Auralis] FDTD grid kept on %s for the next runs (%.1f M air cells)"
          % ("GPUs " + ",".join(str(int(d)) for d in dv) if n_dev > 1
             else "device %d" % int(dv[0]), grid.n_air / 1e6))
    return handle


def _multi_device_args(lib, grid, devs, n_links):
    """(aur_fdtd_run4m, or aur_fdtd_run5m with tangential links, and its leading ctypes
    arguments), or None with a LOUD warning."""
    import ctypes as ct
    why = None
    if not hasattr(lib, "aur_fdtd_run4m"):
        why = "this library has no aur_fdtd_run4m (ABI < 2.5): rebuild native/build_cuda.sh"
    elif n_links and not hasattr(lib, "aur_fdtd_run5m"):
        why = ("tangential links (conformal_aniso) need aur_fdtd_run5m (ABI 2.7): rebuild "
               "native/build_cuda.sh")
    cut = None if why else slab_cuts(grid, len(devs))
    if why is None and cut is None:
        why = "the grid is too thin along x for %d slabs" % len(devs)
    if why:
        print("[Auralis] WARNING: split FDTD on %s refused (%s): ONE device." % (devs, why))
        return None
    cuts, halo = cut
    dv = np.ascontiguousarray(devs, dtype=np.int32)
    P = ct.POINTER
    fnm = lib.aur_fdtd_run5m if n_links else lib.aur_fdtd_run4m
    fnm.restype = ct.c_int32
    head_t = [ct.c_int32, P(ct.c_int32), P(ct.c_int64), P(ct.c_int64)]
    run4_t = [P(ct.c_int32), P(ct.c_int32), P(ct.c_uint8),
              P(ct.c_float), P(ct.c_float), P(ct.c_float), P(ct.c_float),
              ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_int64,
              ct.c_int64, P(ct.c_int32), P(ct.c_float), P(ct.c_float), P(ct.c_float),
              P(ct.c_float),
              ct.c_int64, P(ct.c_int32), P(ct.c_float), P(ct.c_float), P(ct.c_float),
              P(ct.c_float), P(ct.c_float),
              P(ct.c_int64), P(ct.c_int32), P(ct.c_float), P(ct.c_float), P(ct.c_float),
              P(ct.c_int32), P(ct.c_int32), P(ct.c_int32)]
    tail_t = [P(ct.c_float), ct.c_int32, P(ct.c_int32), P(ct.c_float), ct.c_int32,
              P(ct.c_int32), P(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32, P(ct.c_float)]
    link_t = ([ct.c_int64, P(ct.c_int32), P(ct.c_int64), P(ct.c_int32), P(ct.c_float)]
              if n_links else [ct.c_int64])
    fnm.argtypes = head_t + run4_t + link_t + tail_t
    sizes = np.diff(cuts)
    print("[Auralis] FDTD split over GPUs %s: %s M air cells per slab" % (
        ",".join(str(d) for d in devs), "/".join("%.1f" % (v / 1e6) for v in sizes)))
    keep = (dv, cuts, halo)                     # alive while the call runs
    head = [ct.c_int32(len(devs)), dv.ctypes.data_as(P(ct.c_int32)),
            cuts.ctypes.data_as(P(ct.c_int64)), halo.ctypes.data_as(P(ct.c_int64))]
    _multi_device_args.keep = keep
    return fnm, head


def run_grid_native(grid: FdtdGrid, pulse: np.ndarray, src_taps, rec_taps,
                    n_steps: int, lib, n_threads: int = 0) -> np.ndarray:
    """Same computation as run_grid_python through the native library
    (float32). rec_taps: list of trilinear (idx, w)."""
    import ctypes as ct
    # version guard: since ABI 2.0 the index arrays are int32 (a third less GPU memory). An
    # older lib expects int64 and would silently corrupt the run, so fall back loudly.
    nr = int(getattr(grid, "n_rfaces", 0))
    ver_ok = False
    vs = "?"
    try:
        lib.aur_version.restype = ct.c_char_p
        vs = lib.aur_version().decode()
        num = vs.rsplit(" ", 1)[-1].split(".")
        ver_ok = (int(num[0]), int(num[1])) >= (2, 0)
    except Exception:
        ver_ok = False
    if not ver_ok:
        print("[Auralis] WARNING: native lib '%s' < 2.0 (int32 index "
              "ABI); python fallback (slow). Rebuild with "
              "native/build.sh and native/build_cuda.sh." % vs)
        return run_grid_python(grid, pulse, src_taps, rec_taps,
                               n_steps).astype(np.float32)
    _ENGINE_USED[0] = _lib_tag(lib)   # after the ABI guard: this lib
                                      # is the one that will compute
    hit = _KEEP["sessions"].get(id(grid)) if _KEEP["depth"] > 0 else None
    if hit is not None and hit[2] is grid and hit[0] is lib:
        return _run_kept(lib, hit[1], grid, pulse, src_taps, rec_taps, n_steps)
    if getattr(grid, "nb", 0) is None:
        raise RuntimeError("this grid was handed over to the GPU (keep_grids) and has no host "
                           "copy left: no replay on another engine inside the series")
    fn = _bind_fdtd(lib)
    ghost = grid.n_air
    nb = np.ascontiguousarray(grid.nb, dtype=np.int32)
    cn = np.ascontiguousarray(grid.c_now, dtype=np.float32)
    co = np.ascontiguousarray(grid.c_old, dtype=np.float32)
    cb = np.ascontiguousarray(grid.c_nb, dtype=np.float32)
    pu = np.ascontiguousarray(pulse, dtype=np.float32)
    si = np.ascontiguousarray(src_taps[0], dtype=np.int32)
    sw = np.ascontiguousarray(src_taps[1], dtype=np.float32)
    recs = [rec_taps] if isinstance(rec_taps, tuple) else list(rec_taps)
    n_rec = len(recs)
    ri = np.full((n_rec, 8), ghost, dtype=np.int32)
    rw = np.zeros((n_rec, 8), dtype=np.float32)
    for r, (i, w) in enumerate(recs):
        ri[r, :len(i)] = i
        rw[r, :len(w)] = w
    out = np.zeros((n_steps, n_rec), dtype=np.float32)
    P = ct.POINTER
    nf = int(getattr(grid, "n_faces", 0))
    if nr:
        rce = np.ascontiguousarray(grid.r_cell, dtype=np.int32)
        rA = np.ascontiguousarray(grid.r_A, dtype=np.float32)
        rc1 = np.ascontiguousarray(grid.r_c1, dtype=np.float32)
        rc2 = np.ascontiguousarray(grid.r_c2, dtype=np.float32)
        rE = np.ascontiguousarray(grid.r_E.reshape(-1), dtype=np.float32)
        rG = np.ascontiguousarray(grid.r_G.reshape(-1), dtype=np.float32)
    else:
        rce = np.zeros(1, np.int32)
        rA = rc1 = rc2 = np.zeros(1, np.float32)
        rE = np.zeros(4, np.float32)
        rG = np.zeros(2, np.float32)
    n_links = int(getattr(grid, "n_links", 0))
    if (getattr(grid, "order", 2) == 4 or getattr(grid, "pml_axes", None)
            or n_links):
        fn4 = lib.aur_fdtd_run5 if n_links else lib.aur_fdtd_run4
        fn4.restype = ct.c_int32
        argt = [P(ct.c_int32), P(ct.c_int32), P(ct.c_uint8),
                        P(ct.c_float), P(ct.c_float), P(ct.c_float),
                        P(ct.c_float),
                        ct.c_float, ct.c_float, ct.c_float, ct.c_float,
                        ct.c_int64,
                        ct.c_int64, P(ct.c_int32), P(ct.c_float),
                        P(ct.c_float), P(ct.c_float), P(ct.c_float),
                        ct.c_int64, P(ct.c_int32), P(ct.c_float),
                        P(ct.c_float), P(ct.c_float), P(ct.c_float),
                        P(ct.c_float),
                        P(ct.c_int64), P(ct.c_int32), P(ct.c_float),
                        P(ct.c_float), P(ct.c_float), P(ct.c_int32),
                        P(ct.c_int32), P(ct.c_int32),
                        P(ct.c_float), ct.c_int32,
                        P(ct.c_int32), P(ct.c_float), ct.c_int32,
                        P(ct.c_int32), P(ct.c_float), ct.c_int32,
                        ct.c_int32, ct.c_int32, P(ct.c_float)]
        link_t = [ct.c_int64, P(ct.c_int32), P(ct.c_int64), P(ct.c_int32), P(ct.c_float)]
        i_pulse = 33                  # links go after pml_locm, before pulse
        fn4.argtypes = argt[:i_pulse] + link_t + argt[i_pulse:] if n_links else argt
        if n_links:
            lr = np.ascontiguousarray(grid.l_rows, dtype=np.int32)
            lp = np.ascontiguousarray(grid.l_ptr, dtype=np.int64)
            ln = np.ascontiguousarray(grid.l_nb, dtype=np.int32)
            lk = np.ascontiguousarray(grid.l_k, dtype=np.float32)
            link_args = [ct.c_int64(len(lr)), lr.ctypes.data_as(P(ct.c_int32)),
                         lp.ctypes.data_as(P(ct.c_int64)), ln.ctypes.data_as(P(ct.c_int32)),
                         lk.ctypes.data_as(P(ct.c_float))]
        else:
            link_args = []
        nb2 = np.ascontiguousarray(grid.nb2, dtype=np.int32)
        m4 = np.ascontiguousarray(grid.is4.astype(np.uint8))
        cdrv = getattr(grid, "c_drv", None)
        if cdrv is None:
            cdrv = np.zeros(grid.n_air, dtype=np.float32)
        cdrv = np.ascontiguousarray(cdrv, dtype=np.float32)
        if nf:
            fce = np.ascontiguousarray(grid.f_cell, dtype=np.int32)
            fe = np.ascontiguousarray(grid.f_e, dtype=np.float32)
            fg = np.ascontiguousarray(grid.f_g, dtype=np.float32)
            fl = np.ascontiguousarray(grid.f_l, dtype=np.float32)
            fA = np.ascontiguousarray(grid.f_Asum, dtype=np.float32)
        else:
            fce = np.zeros(1, np.int32)
            fe = fg = fl = np.zeros(2, np.float32)
            fA = np.zeros(1, np.float32)
        axes = list(getattr(grid, "pml_axes", []) or [])
        cnt = np.zeros(3, dtype=np.int64)
        for k, ax in enumerate(axes):
            cnt[k] = len(ax["cells"])
        def _cat(key, dt):
            if not axes:
                return np.zeros(1, dt)
            return np.ascontiguousarray(
                np.concatenate([np.asarray(ax[key]) for ax in axes]),
                dtype=dt)
        pc = _cat("cells", np.int32)
        pb = _cat("b", np.float32)
        pa = _cat("a", np.float32)
        pap = _cat("a_psi", np.float32)
        pnm = _cat("nbm", np.int32)
        pnp = _cat("nbp", np.int32)
        plm = _cat("locm", np.int32)
        args = [nb.ctypes.data_as(P(ct.c_int32)),
                 nb2.ctypes.data_as(P(ct.c_int32)),
                 m4.ctypes.data_as(P(ct.c_uint8)),
                 cn.ctypes.data_as(P(ct.c_float)),
                 co.ctypes.data_as(P(ct.c_float)),
                 cb.ctypes.data_as(P(ct.c_float)),
                 cdrv.ctypes.data_as(P(ct.c_float)),
                 ct.c_float(grid.c4_c), ct.c_float(grid.c4_nb),
                 ct.c_float(grid.c4_nb2), ct.c_float(grid.lam2),
                 ct.c_int64(grid.n_air),
                 ct.c_int64(nf),
                 fce.ctypes.data_as(P(ct.c_int32)),
                 fe.ctypes.data_as(P(ct.c_float)),
                 fg.ctypes.data_as(P(ct.c_float)),
                 fl.ctypes.data_as(P(ct.c_float)),
                 fA.ctypes.data_as(P(ct.c_float)),
                 ct.c_int64(nr),
                 rce.ctypes.data_as(P(ct.c_int32)),
                 rA.ctypes.data_as(P(ct.c_float)),
                 rc1.ctypes.data_as(P(ct.c_float)),
                 rc2.ctypes.data_as(P(ct.c_float)),
                 rE.ctypes.data_as(P(ct.c_float)),
                 rG.ctypes.data_as(P(ct.c_float)),
                 cnt.ctypes.data_as(P(ct.c_int64)),
                 pc.ctypes.data_as(P(ct.c_int32)),
                 pb.ctypes.data_as(P(ct.c_float)),
                 pa.ctypes.data_as(P(ct.c_float)),
                 pap.ctypes.data_as(P(ct.c_float)),
                 pnm.ctypes.data_as(P(ct.c_int32)),
                 pnp.ctypes.data_as(P(ct.c_int32)),
                 plm.ctypes.data_as(P(ct.c_int32))] + link_args + [
                 pu.ctypes.data_as(P(ct.c_float)), ct.c_int32(len(pu)),
                 si.ctypes.data_as(P(ct.c_int32)),
                 sw.ctypes.data_as(P(ct.c_float)), ct.c_int32(len(si)),
                 ri.ctypes.data_as(P(ct.c_int32)),
                 rw.ctypes.data_as(P(ct.c_float)), ct.c_int32(n_rec),
                 ct.c_int32(n_steps), ct.c_int32(n_threads),
                 out.ctypes.data_as(P(ct.c_float))]
        devs = _fdtd_devices()
        i_tail = i_pulse + len(link_args)            # pulse, n, si, sw, n, ri, rw, n, ...
        sess = None
        if _KEEP["depth"] > 0 and hasattr(lib, "aur_fdtd_session4m_create") and (
                not n_links or hasattr(lib, "aur_fdtd_session5m_create")):
            sess = _kept_session(lib, grid, devs, args[:i_pulse], link_args)
        multi = None
        if sess is None and devs and len(devs) > 1:
            multi = _multi_device_args(lib, grid, devs, n_links)
        if sess is not None:
            frun = lib.aur_fdtd_session4m_run
            frun.restype = ct.c_int32
            frun.argtypes = [ct.c_void_p, P(ct.c_float), ct.c_int32, P(ct.c_int32),
                             P(ct.c_float), ct.c_int32, P(ct.c_int32), P(ct.c_float),
                             ct.c_int32, ct.c_int32, P(ct.c_float)]
            tail = args[i_tail:]                     # pulse, n, si, sw, n, ri, rw, n, steps, thr, out
            rc = _with_progress(lib, n_steps, lambda: frun(sess, *(tail[:9] + tail[10:])))
        elif multi is not None:
            fnm, head = multi
            mid = link_args if n_links else [ct.c_int64(0)]
            rc = _with_progress(lib, n_steps, lambda: fnm(*(head + args[:i_pulse] + mid
                                                             + args[i_tail:])))
        else:
            rc = _with_progress(lib, n_steps, lambda: fn4(*args))
        if rc in (42, 43):
            raise RuntimeError("aur_fdtd_run4m refused the slabs (code %d): %s" % (
                rc, "a cut through the x PML, a slab too thin, or a link leaving the slab"
                if rc == 42 else "tangential links without their arrays"))
        if rc != 0:
            if rc == 41:
                raise RuntimeError(
                    "FDTD instability caught by the watchdog: "
                    "slow-growing boundary mode. The wall filters are "
                    "checked for passivity in the grid's discrete time "
                    "(the known cause in closed rooms); "
                    "open domains keep a known GKS edge mode. "
                    "Workarounds: shorten the duration, raise the "
                    "material absorption, or turn the FDTD off (rays "
                    "only, validated). Please report the scene.")
            raise RuntimeError("aur_fdtd_run%d failed (code %d)" % (5 if n_links else 4, rc))
        return out.astype(np.float64)
    if nf or nr:
        fn2 = lib.aur_fdtd_run2
        fn2.restype = ct.c_int32
        if nf:
            fce = np.ascontiguousarray(grid.f_cell, dtype=np.int32)
            fe = np.ascontiguousarray(grid.f_e, dtype=np.float32)
            fg = np.ascontiguousarray(grid.f_g, dtype=np.float32)
            fl = np.ascontiguousarray(grid.f_l, dtype=np.float32)
            fA = np.ascontiguousarray(grid.f_Asum, dtype=np.float32)
        else:
            fce = np.zeros(1, np.int32)
            fe = fg = fl = np.zeros(2, np.float32)
            fA = np.zeros(1, np.float32)
        cdrv = np.ascontiguousarray(grid.c_drv, dtype=np.float32)
        rc = _with_progress(lib, n_steps, lambda: fn2(
                 nb.ctypes.data_as(P(ct.c_int32)),
                 cn.ctypes.data_as(P(ct.c_float)),
                 co.ctypes.data_as(P(ct.c_float)),
                 cb.ctypes.data_as(P(ct.c_float)),
                 cdrv.ctypes.data_as(P(ct.c_float)),
                 ct.c_int64(grid.n_air),
                 ct.c_int64(nf),
                 fce.ctypes.data_as(P(ct.c_int32)),
                 fe.ctypes.data_as(P(ct.c_float)),
                 fg.ctypes.data_as(P(ct.c_float)),
                 fl.ctypes.data_as(P(ct.c_float)),
                 fA.ctypes.data_as(P(ct.c_float)),
                 ct.c_int64(nr),
                 rce.ctypes.data_as(P(ct.c_int32)),
                 rA.ctypes.data_as(P(ct.c_float)),
                 rc1.ctypes.data_as(P(ct.c_float)),
                 rc2.ctypes.data_as(P(ct.c_float)),
                 rE.ctypes.data_as(P(ct.c_float)),
                 rG.ctypes.data_as(P(ct.c_float)),
                 pu.ctypes.data_as(P(ct.c_float)), ct.c_int32(len(pu)),
                 si.ctypes.data_as(P(ct.c_int32)),
                 sw.ctypes.data_as(P(ct.c_float)), ct.c_int32(len(si)),
                 ri.ctypes.data_as(P(ct.c_int32)),
                 rw.ctypes.data_as(P(ct.c_float)), ct.c_int32(n_rec),
                 ct.c_int32(n_steps), ct.c_int32(n_threads),
                 out.ctypes.data_as(P(ct.c_float))))
        if rc != 0:
            raise RuntimeError("aur_fdtd_run2 failed (code %d)" % rc)
        return out.astype(np.float64)
    rc = _with_progress(lib, n_steps, lambda: fn(
            nb.ctypes.data_as(P(ct.c_int32)),
            cn.ctypes.data_as(P(ct.c_float)),
            co.ctypes.data_as(P(ct.c_float)),
            cb.ctypes.data_as(P(ct.c_float)),
            ct.c_int64(grid.n_air),
            pu.ctypes.data_as(P(ct.c_float)), ct.c_int32(len(pu)),
            si.ctypes.data_as(P(ct.c_int32)),
            sw.ctypes.data_as(P(ct.c_float)), ct.c_int32(len(si)),
            ri.ctypes.data_as(P(ct.c_int32)),
            rw.ctypes.data_as(P(ct.c_float)), ct.c_int32(n_rec),
            ct.c_int32(n_steps), ct.c_int32(n_threads),
            out.ctypes.data_as(P(ct.c_float))))
    if rc != 0:
        raise RuntimeError("aur_fdtd_run failed (code %d)" % rc)
    return out.astype(np.float64)


# Backend actually executed on the last call, GPU->CPU, ABI and python fallbacks included.
# It is written where the compute happens, never where the choice is made, so a run that
# fell back to the CPU cannot report "gpu".
_ENGINE_USED = [None]


def _lib_tag(lib):
    """Honest tag for a loaded library."""
    name = str(getattr(lib, "_name", "") or "").lower()
    if "cuda_emu" in name or "cuda-emu" in name:
        return "cuda-emu"
    return "gpu" if "cuda" in name else "native"


def _run_gpu_or_fallback(gpu_lib, need, *a, **k):
    """GPU run with a LOUD fallback: a runtime failure marks the GPU
    dead for the session and replays the same grid on CPU."""
    from . import native as _nat
    try:
        return run_grid_native(*a, lib=gpu_lib, **k)
    except RuntimeError as exc:
        _nat.mark_gpu_dead(str(exc))
    run, name = _pick_backend_impl("native", need)
    print("[Auralis] FDTD: replaying on '%s'%s"
          % (name, " (slow)" if name == "python" else ""))
    return run(*a, **k)


def _slab_grid(scene, vox, make, engine, pml_cfg, order, conf_on, aniso, src_pos, rcvs,
               src_axis, dx):
    """slabs.SlabGrid of the scene: make(sub_vox, pml, conformal kwargs) is the scene's
    FdtdGrid on one slab's sub box, with the conformal corrections of that sub box. The CUDA
    library on AURALIS_FDTD_GPUS (all GPUs: one slab each), or AURALIS_FDTD_SLABS_LIB (the
    emulation, for tests) with AURALIS_FDTD_GPUS as slab count."""
    from . import native as _nat
    from . import slabs
    path = os.environ.get("AURALIS_FDTD_SLABS_LIB") or (
        _nat.find_gpu_library() if engine in ("auto", "gpu") and _nat.gpu_available() else None)
    lib = _nat.load_library(path) if path else None
    if not slabs.available(lib):
        raise RuntimeError("a scene beyond 2^31 air cells needs the CUDA library 2.8 "
                           "(aur_fdtd_slabs_create) on its GPUs")
    devs = _fdtd_devices() or [0]
    from .conformal import subgrid_cap
    # ONE subgrid factor for every slab (the largest one decides): a slab at factor 3 next to
    # one fallen back to 2 would hold two corrections of the same scene.
    npml = int(pml_cfg.get("n", 0)) if pml_cfg else 0
    n_max = max(sb - sa for sa, sb in slabs.slab_boxes(vox.air, len(devs), npml)) * int(
        vox.shape[1]) * int(vox.shape[2])
    factor = 3 if 27 * n_max <= subgrid_cap() else 2
    if conf_on and factor < 3:
        print("[Auralis] conformal: the largest slab's subgrid is beyond AURALIS_SUBGRID_MAX "
              "(%.0f M > %.0f M subcells): every slab at factor 2 (one correction for the "
              "whole scene) - raise AURALIS_SUBGRID_MAX if RAM allows (~4 bytes/subcell)"
              % (27 * n_max / 1e6, subgrid_cap() / 1e6))

    def build(sub, psub):
        from .conformal import conformal_corrections
        cf = {}
        if conf_on:                     # not cached: a slab's arrays are tens of GB
            phi, face_w, cst = conformal_corrections(scene, sub, factor=factor, per_axis=aniso,
                                                     smooth_sweeps=0 if aniso else 8)
            cf = {"phi": phi, "face_area_w": face_w, "excess_axis": cst.pop("excess_axis", None)}
        return make(sub, psub, cf)
    pts = [np.asarray(src_pos, dtype=np.float64)] + [np.asarray(r, dtype=np.float64) for r in rcvs]
    if src_axis is not None:
        u = np.asarray(src_axis, float) / np.linalg.norm(src_axis)
        pts += [pts[0] + dx * u, pts[0] - dx * u]          # the dipole's ends (h = 2 dx)
    return slabs.SlabGrid(vox, build, lib, devs, pml_cfg, order, pts)


def _dense_empty_lib(engine: str):
    """The CUDA library for the dense empty-box pass (dense.py), or None: the GPU backend
    only (the CPU and python engines keep the indexed empty grid), AURALIS_FDTD_EMPTY_DENSE=0
    turns it off."""
    if os.environ.get("AURALIS_FDTD_EMPTY_DENSE", "1") == "0":
        return None
    from . import native as _nat
    if os.environ.get("AURALIS_FDTD_DENSE_LIB"):       # the emulation, for tests
        lib = _nat.load_library(os.environ["AURALIS_FDTD_DENSE_LIB"])
        return lib if hasattr(lib, "aur_fdtd_dense_empty") else None
    if engine not in ("auto", "gpu"):
        return None
    try:
        if not _nat.gpu_available():
            return None
        path = _nat.find_gpu_library()
        lib = _nat.load_library(path) if path else None
        if (lib is None or not hasattr(lib, "aur_fdtd_dense_empty")
                or not int(lib.aur_cuda_available())):
            return None
        return lib
    except Exception:
        return None


def _pick_fdtd_backend(engine: str, grid=None):
    if getattr(grid, "wide_index", False):           # slabs.SlabGrid: its own session
        return grid.runner, "gpu-slabs"
    need = "aur_fdtd_run"
    if grid is not None:
        if getattr(grid, "n_links", 0):
            need = "aur_fdtd_run5"     # tangential apertures (conformal_aniso)
        elif getattr(grid, "order", 2) == 4 or getattr(grid, "pml_axes",
                                                       None):
            need = "aur_fdtd_run4"     # order 4 / CPML / multi-probe
        elif getattr(grid, "n_faces", 0):
            need = "aur_fdtd_run2"
    return _pick_backend_impl(engine, need)


def _pick_backend_impl(engine: str, need: str):
    """(run_function, name) depending on library availability."""
    from . import native as _nat
    err = ""
    err_n = ""
    if engine in ("auto", "gpu"):
        try:
            if not _nat.gpu_available():
                # honors AURALIS_GPU=0 and the session marking
                # (mark_gpu_dead after a runtime failure)
                err = "gpu_available() False (env, driver or session)"
            else:
                path = _nat.find_gpu_library()
                if path:
                    lib = _nat.load_library(path)
                    if not hasattr(lib, need):
                        err = "symbol %s missing from %s" % (need, path)
                    elif not (hasattr(lib, "aur_cuda_available")
                              and int(lib.aur_cuda_available()) == 1):
                        err = ("aur_cuda_available() != 1 "
                               "(driver/context)")
                    else:
                        return (lambda *a, **k: _run_gpu_or_fallback(
                            lib, need, *a, **k), "gpu")
                else:
                    err = "libauralis_cuda not found"
        except Exception as exc:
            err = repr(exc)
        if engine == "gpu":
            raise RuntimeError("CUDA library unavailable: "
                               + err)
    if engine in ("auto", "native"):
        try:
            path = _nat.find_library()
            if not path:
                err_n = "libauralis not found"
            else:
                lib = _nat.load_library(path)
                if hasattr(lib, need):
                    return (lambda *a, **k: run_grid_native(*a, lib=lib,
                                                            **k),
                            "native")
                err_n = "symbol %s missing from %s" % (need, path)
        except Exception as exc:
            err_n = repr(exc)
        if engine == "native":
            raise RuntimeError("native library unavailable: "
                               + err_n)
    if engine == "auto" and (err or err_n):
        print("[Auralis] FDTD: falling back to python (SLOW). gpu: %s; "
              "native: %s. Rebuild native/build.sh (and "
              "build_cuda.sh) on THIS machine if the shipped "
              "libraries fail to load (glibc)."
              % (err or "not tried", err_n or "not tried"))
    return run_grid_python, "python"



def make_sponge(shape, dx: float, c: float, n_cells: int = 18,
                s_max_dt: float = 0.35) -> np.ndarray:
    """Volume damping sigma (nx,ny,nz): cubic ramp over n_cells from
    the grid border. s_max_dt = sigma_max * dt."""
    dt = _LAMBDA * dx / c
    nx, ny, nz = shape
    ax = [np.minimum(np.arange(n), np.arange(n)[::-1])
          for n in (nx, ny, nz)]
    d = np.minimum.reduce(np.meshgrid(ax[0], ax[1], ax[2], indexing="ij"))
    r = np.clip((n_cells - d) / float(n_cells), 0.0, 1.0)
    return (s_max_dt / dt) * r ** 3

# ---------------------------------------------------------------------------
# Free-field calibration (1/d convention, d/c delay)
# ---------------------------------------------------------------------------

_CAL_CACHE: dict = {}


def free_field_calibration(dx: float, c: float, pulse: np.ndarray,
                           n_steps: int, d_ref_cells: int = 12,
                           reg: float = 1e-3, order: int = 2):
    """Wiener filter C(f) such that C * p_free_field = delta(t-d/c)/d, on an empty grid.

    Returns (C, n_fft, d_ref, band). It is the same for every source of a grid, so it is
    cached with the grids (AURALIS_FDTD_GRID_CACHE). Copies are returned."""
    import hashlib
    key = (float(dx), float(c), int(n_steps), int(d_ref_cells), float(reg), int(order),
           hashlib.sha1(np.ascontiguousarray(pulse, dtype=np.float64).tobytes()).hexdigest())
    if _cache_on() and key in _CAL_CACHE:
        C, n_fft, d_ref, band = _CAL_CACHE[key]
        return C.copy(), n_fft, d_ref, band.copy()
    out = _free_field_calibration(dx, c, pulse, n_steps, d_ref_cells, reg, order)
    if _cache_on():
        _cache_put(_CAL_CACHE, key, (out[0].copy(), out[1], out[2], out[3].copy()))
    return out


def _free_field_calibration(dx, c, pulse, n_steps, d_ref_cells, reg, order):
    """free_field_calibration, computed."""
    d_ref = d_ref_cells * dx
    dt = _LAMBDA * dx / c
    a = np.abs(np.asarray(pulse, float))
    nz = np.nonzero(a > 1e-6 * (a.max() or 1.0))[0]
    support = int(nz[-1]) + 1 if len(nz) else 8
    t_need = (support + 6) * dt + d_ref / c
    # first border echo >= window end: half >= (c*t + d)/(2 dx)
    half = int(np.ceil((c * t_need + d_ref) / (2.0 * dx))) + 3
    n_side = 2 * half + 1

    class _Vox:
        pass
    vox = VoxelGrid(solid=np.zeros((n_side,) * 3, bool),
                    mat=np.full((n_side,) * 3, -1, np.int32),
                    origin=np.array([-half * dx] * 3), dx=dx)
    vox.air = ~vox.solid
    g = FdtdGrid(vox, np.zeros(1), c, order=order)
    st = g.taps((0.0, 0.0, 0.0))
    rt = g.taps((d_ref, 0.0, 0.0))
    n_win = min(n_steps, int(np.floor(
        ((2 * half * dx - d_ref) / c * 0.98) / g.dt)),
        int(np.ceil(t_need / g.dt)) + 4)
    p = run_grid_python(g, pulse, st, [rt], n_win)[:, 0]

    n_fft = int(2 ** np.ceil(np.log2(max(n_steps, n_win) * 2)))
    S = np.fft.rfft(p, n_fft)
    f = np.fft.rfftfreq(n_fft, g.dt)
    target = np.exp(-2j * np.pi * f * (d_ref / c)) / d_ref
    den = np.abs(S) ** 2 + reg * float(np.abs(S).max()) ** 2
    C = target * np.conj(S) / den
    band = np.abs(S) ** 2 / den          # effective Wiener band
    return C, n_fft, d_ref, band


def subsonic_highpass(f: np.ndarray, f_c: float) -> np.ndarray:
    """Frequency response of a causal 4th-order Butterworth high-pass at f_c."""
    s = 1j * np.asarray(f, float) / f_c
    s2 = s * s
    return s2 * s2 / ((s2 + 2.0 * np.sin(np.pi / 8) * s + 1.0)
                      * (s2 + 2.0 * np.sin(3 * np.pi / 8) * s + 1.0))


# ---------------------------------------------------------------------------
# Complementary linear-phase crossover
# ---------------------------------------------------------------------------

def design_crossover(f_cross: float, fs: int, n_taps: int = 0):
    """Windowed low-pass + high-pass = delta - low-pass: the two sum
    EXACTLY to an impulse delayed by (N-1)/2 samples."""
    if n_taps <= 0:
        n_taps = int(2 * round(3.0 * fs / f_cross) + 1)
    n_taps |= 1
    m = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = 2.0 * f_cross / fs * np.sinc(2.0 * f_cross / fs * m)
    h *= np.kaiser(n_taps, 8.0)
    h /= h.sum()
    hp = -h
    hp[(n_taps - 1) // 2] += 1.0
    return h, hp, (n_taps - 1) // 2


def _resample_fft(x: np.ndarray, fs_in: float, fs_out: int,
                  n_out: int) -> np.ndarray:
    """Resampling by spectral zero-padding (aperiodized signal)."""
    n_in = len(x)
    n_pad = int(2 ** np.ceil(np.log2(n_in * 2)))
    X = np.fft.rfft(x, n_pad)
    n2 = int(round(n_pad * fs_out / fs_in))
    Y = np.zeros(n2 // 2 + 1, dtype=complex)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    y = np.fft.irfft(Y, n2) * (n2 / n_pad)
    return y[:n_out] if len(y) >= n_out else np.pad(y, (0, n_out - len(y)))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class FdtdInfo:
    f_cross: float
    f_max_grid: float
    dx: float
    dt: float
    n_cells: int
    n_air: int
    n_steps: int
    runtime_s: float
    engine: str
    air_span: tuple = (0.0, 0.0, 0.0)   # voxelized air extent (m)
    is_open: bool = False
    order: int = 4
    absorber: str = "closed"   # closed | cpml | sponge
    conformal: bool = False    # adjusted FV cells active
    vol_gain_pct: float = 0.0  # donated volume / discrete air volume
    area_stair_pct: float = 0.0  # staircase area inflation vs exact
    missing: tuple = ()        # receivers outside the air (missing_rcv_ok): zero IRs


def wave_dx(f_cross: float, speed_of_sound: float, ppw: float = 9.0) -> float:
    """Grid spacing of run_low_band for a crossover f_cross (grid band 1.5 f_cross)."""
    f_max = 1.5 * float(f_cross)
    return speed_of_sound / (ppw * f_max)


def _dipole_taps(grid, pos, axis, h):
    """Injection taps of a dipole: unit sources +1 / -1 at pos +/- h/2 along axis,
    shared cells merged."""
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    p = np.asarray(pos, float)
    taps = getattr(grid, "source_taps", grid.taps)
    i1, w1 = taps(p + 0.5 * h * u)
    i2, w2 = taps(p - 0.5 * h * u)
    idx, inv = np.unique(np.concatenate([i1, i2]), return_inverse=True)
    w = np.bincount(inv, weights=np.concatenate([w1, -w2]))
    keep = np.abs(w) > 1e-12
    return idx[keep].astype(np.int64 if getattr(grid, "wide_index", False) else np.int32), w[keep]


def _first_order_filters(f, a_bands, h, c):
    """(a(f), 1 - a(f), 1/D(f)) for the pattern a + (1 - a) cos(theta): a from the
    per-band weights in log frequency, omni (1) at 31.25 Hz and below; D = 2j
    sin(pi f h / c) is the on-axis gain of the grid dipole, inverted with a floor
    at 20 Hz (the dipole share vanishes there anyway)."""
    from .bands import CENTERS
    lc = np.log(np.concatenate([[31.25], CENTERS]))
    a = np.interp(np.log(np.maximum(f, 1.0)), lc,
                  np.concatenate([[1.0], np.asarray(a_bands, float)]))
    d = 2j * np.sin(np.pi * f * h / c)
    eps = 2.0 * np.sin(np.pi * 20.0 * h / c)
    return a, 1.0 - a, np.conj(d) / (np.abs(d) ** 2 + eps ** 2)


def grid_alignment_angle(scene, step_deg: float = 0.25, min_gain: float = 0.05):
    """(angle in rad, relative gain) of the rotation about z that minimises the area of the
    horizontal bounding box of the triangles; (0, 0) below a min_gain reduction."""
    xy = np.asarray(scene.tri_verts, dtype=np.float64).reshape(-1, 3)[:, :2]
    xy = np.unique(np.round(xy, 6), axis=0)
    ths = np.radians(np.arange(0.0, 90.0, step_deg))
    c, s_ = np.cos(ths)[:, None], np.sin(ths)[:, None]
    xs = c * xy[None, :, 0] - s_ * xy[None, :, 1]
    ys = s_ * xy[None, :, 0] + c * xy[None, :, 1]
    area = np.ptp(xs, axis=1) * np.ptp(ys, axis=1)
    k = int(np.argmin(area))
    gain = 1.0 - area[k] / max(area[0], 1e-30)
    if gain < min_gain:
        return 0.0, 0.0
    return float(ths[k]), float(gain)


def run_low_band(scene, src_pos, rcv_pos, fs: int, duration: float,
                 f_cross: float, speed_of_sound: float,
                 ppw: float = 9.0, engine: str = "auto",
                 max_cells: int = 3_500_000,
                 boundary_filters: bool = True,
                 open_boundary: str = "auto",
                 sponge_cells: int = 0, order: int = 4,
                 boundary_absorber: str = "cpml",
                 pml_cells: int = 14,
                 conformal: bool = True,
                 transfer_freqs=None,
                 conformal_aniso: bool = False,
                 src_axis=None, src_first_order=None,
                 subsonic_hz=None, align_grid=None,
                 missing_rcv_ok: bool = False,
                 domain_points=None,
                 subcell_openings=None,
                 src_surface=None) -> Tuple[np.ndarray, FdtdInfo]:
    """Low-band IR (1/d convention) at fs, length round(duration * fs), and an FdtdInfo.

    rcv_pos: one position gives (n,), a list gives (n, n_rcv).
    transfer_freqs (Hz): return the calibrated complex transfer function (n_freqs, n_rcv) at
    the nearest FFT bins instead, with info.transfer_freqs the bins used.
    conformal_aniso: also widen the faces parallel to the walls by the donated volume, which
    makes the modes exact. AURALIS_FDTD_CONFORMAL_ANISO=1 turns it on too.
    src_axis, src_first_order: pattern a(f) + (1 - a(f)) cos(theta) about src_axis, from a
    second run with a grid dipole. Ignored with transfer_freqs.
    subsonic_hz: corner of the causal high-pass of closed-domain IRs (None: env, 16 Hz).
    align_grid: turn the scene about z to shrink its horizontal box (None: AURALIS_FDTD_ALIGN).
    missing_rcv_ok: a receiver outside the air gets a zero IR, listed in info.missing.
    domain_points: points besides the source that the domain box must hold (None: the
    receivers), so that extra probes inside its margin do not move the grid.
    subcell_openings: join enclosed pockets through gaps thinner than the cell, closed
    domains only (None: AURALIS_FDTD_SUBCELL, on).
    src_surface: material (name or index) of a vibrating surface replacing the point source.
    src_pos then only anchors the air. Closed domains only, omni."""
    t0 = time.time()
    if align_grid is None:
        align_grid = os.environ.get("AURALIS_FDTD_ALIGN", "0") == "1"
    if align_grid:
        th, gain = grid_alignment_angle(scene)
        if th != 0.0:
            import copy as _copy
            Rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0],
                           [0.0, 0.0, 1.0]])
            rotated = np.ascontiguousarray(np.asarray(scene.tri_verts, np.float64) @ Rz.T,
                                           dtype=np.float32)
            scene = _copy.copy(scene)           # every attribute kept (mat_gamma, ...)
            scene.tri_verts = rotated
            src_pos = Rz @ np.asarray(src_pos, dtype=np.float64)
            rp = np.asarray(rcv_pos, dtype=np.float64)
            rcv_pos = rp @ Rz.T if rp.ndim > 1 else Rz @ rp
            if isinstance(rcv_pos, np.ndarray) and rcv_pos.ndim > 1:
                rcv_pos = [r for r in rcv_pos]
            if src_axis is not None:
                src_axis = Rz @ np.asarray(src_axis, dtype=np.float64)
            if domain_points is not None:
                domain_points = [Rz @ np.asarray(q, dtype=np.float64) for q in domain_points]
            print("[Auralis] FDTD grid turned by %.2f deg about the vertical, along the "
                  "scene: horizontal box %.0f %% smaller" % (np.degrees(th), 100.0 * gain))
    f_max = 1.5 * float(f_cross)
    dx = wave_dx(f_cross, speed_of_sound, ppw)
    rcvs = np.atleast_2d(np.asarray(rcv_pos, dtype=np.float64))
    # points the domain box is built around (6 dx of margin): source and receivers, or the
    # caller's own (the dipole probes of a moving source must not move the grid)
    dom_pts = ([src_pos] + [r for r in rcvs] if domain_points is None
               else [src_pos] + [np.asarray(q, dtype=np.float64) for q in domain_points])
    single = np.asarray(rcv_pos, dtype=np.float64).ndim == 1
    vox = voxelize(scene, dx, src_pos,
                   include_points=dom_pts)
    a = vox.air                          # air on a face of the box: open domain
    touches = bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()
                   or a[:, :, 0].any() or a[:, :, -1].any())
    is_open = touches and open_boundary != "closed"
    sigma = None
    pml_cfg = None
    g_border = 0.0
    absorber = "closed"
    if sponge_cells <= 0:
        lam_p = speed_of_sound / (f_max / 3.2)
        sponge_cells = int(np.clip(round(1.4 * lam_p / dx), 28, 56))
    if is_open:
        absorber = boundary_absorber if boundary_absorber in (
            "cpml", "sponge") else "cpml"
        marge = (pml_cells + 2) if absorber == "cpml" \
            else (sponge_cells + 2)
        vox = voxelize(scene, dx, src_pos, margin_cells=marge,
                       include_points=dom_pts)
        if absorber == "cpml":
            pml_cfg = {"n": int(pml_cells)}
        else:
            sigma = make_sponge(vox.shape, dx, speed_of_sound,
                                n_cells=sponge_cells)
        g_border = 1.0
    n_cells = int(np.prod(vox.shape))
    if n_cells > max_cells:
        raise ValueError("FDTD grid too large (%d cells > %d): "
                         "lower the crossover frequency"
                         % (n_cells, max_cells))
    # ---- sub-cell openings (gaps narrower than the voxel rule) ---------
    sub_links = None
    if subcell_openings is None:
        subcell_openings = os.environ.get("AURALIS_FDTD_SUBCELL", "1") != "0"
    # closed rooms only: in an open domain the enclosed pockets are the insides of buildings
    # and objects, and their whole shells would be searched for nothing
    if subcell_openings and not is_open:
        from . import subcell
        t_sc = time.time()
        vox, sub_links, sst = subcell.open_pockets(scene, vox)
        for o in sst["opened"]:
            print("[Auralis] sub-cell opening: pocket of %.1f m3 joined to the air through "
                  "gaps narrower than the cell, conductance %.2f faces (h = dx/%d: %.2f, "
                  "dx/%d: %.2f), %d links between %d and %d mouth cells, CG %d/%d iterations, "
                  "flux balance %.1e, %.1f s"
                  % (o["m3"], o["w"], o["F"], o["w_h"], o["F"] // 2, o["w_2h"], o["n_links"],
                     o["n_mouth"][0], o["n_mouth"][1], o["cg_iter"][0], o["cg_iter"][1],
                     o["flux_balance"], o["t_s"]))
        closed = sst["n_pockets"] - len(sst["opened"])
        if closed:
            print("[Auralis] sub-cell openings: %d enclosed pocket(s) next to the air stay "
                  "closed (no gap in the geometry), %.1f s" % (closed, time.time() - t_sc))
    # ---- adjusted FV cells (conformal) --------------------------------
    # True volumes (donation from the solids) + exact face areas.
    # Everything is folded into the per-cell/per-face coefficients: no
    # kernel changes, python/native/CUDA parity comes for free.
    phi = None
    face_w = None
    excess_axis = None
    conf_on = bool(conformal) and os.environ.get(
        "AURALIS_FDTD_CONFORMAL", "1") != "0"
    vol_gain_pct = 0.0
    area_stair_pct = 0.0
    aniso = bool(conformal_aniso) or os.environ.get(
        "AURALIS_FDTD_CONFORMAL_ANISO", "0") == "1"
    # a scene beyond the int32 indices of one grid is built slab by slab (slabs.py); the
    # conformal corrections are then computed on each slab's sub box
    slab_mode = (os.environ.get("AURALIS_FDTD_SLABS", "0") == "1"
                 or int(vox.air.sum()) + 1 > np.iinfo(np.int32).max)
    if slab_mode and (sub_links is not None or sigma is not None):
        raise ValueError("slab-built wave grid: no sub-cell openings and no sponge layer "
                         "(CPML only)")
    if conf_on and not slab_mode:
        try:
            t_cf = time.time()
            phi, face_w, cst, reused = cached_conformal(
                scene, vox, per_axis=aniso, smooth_sweeps=0 if aniso else 8)
            excess_axis = cst.pop("excess_axis", None)
            n_wf = cst["n_faces_data"]
            a_map = cst["area_mapped_m2"]
            if n_wf > 0 and a_map > 0:
                area_stair_pct = 100.0 * (n_wf * dx * dx / a_map - 1.0)
            vol_gain_pct = cst["vol_gain_pct"]
            print("[Auralis] conformal FDTD: volume +%.2f %% "
                  "(%.2f m3 recovered, %.3f m3 orphaned), staircase "
                  "area -> exact %+.1f %% over %d faces (%.1f m2 true, "
                  "%.2f m2 dropped), subgrid x%d, %.1f s"
                  % (vol_gain_pct, cst["vol_donated_m3"],
                     cst["vol_orphan_m3"], area_stair_pct, n_wf,
                     cst["area_true_m2"], cst["area_dropped_m2"],
                     cst["factor"], time.time() - t_cf)
                  + (" (reused: same grid as the previous run)" if reused else ""))
            if cst["n_faces_capped"] or cst["n_phi_capped"]:
                print("[Auralis] conformal: %d faces and %d volumes "
                      "capped (w<=3, phi<=4) -- very oblique corners."
                      % (cst["n_faces_capped"], cst["n_phi_capped"]))
        except Exception as exc:            # LOUD fallback, never mute
            phi = face_w = excess_axis = None
            print("[Auralis] conformal UNAVAILABLE (%s): plain "
                  "staircase grid (nrmq). Please report with the scene."
                  % exc)
            conf_on = False
    if boundary_filters:
        # DIFFUSE FIELD fit (inverse Paris): the same target gamma as
        # the specular reflection R(theta, f) of the geometric engine.
        from .bands import CENTERS as _BC
        from .reflection import (fit_admittance_complex,
                                  fit_admittance_random)
        nm = len(scene.absorption)
        gi = np.zeros(nm)
        fA = np.zeros((nm, 2))
        fL = np.ones((nm, 2))
        filt_res = [None] * nm
        refit = [None] * nm                    # (alpha, bands) of random-incidence fits
        res_on = os.environ.get("AURALIS_FDTD_RESONANT", "1") != "0"
        for m in range(nm):
            mg_all = getattr(scene, "mat_gamma", None)
            if mg_all is not None and m < len(mg_all) and \
                    mg_all[m] is not None:
                # precomputed branches (gi, A(2,), l(2,)) -- e.g. a
                # multi-angle fit: used as is, with no re-fit
                gi[m], fA[m], fL[m] = mg_all[m]
                continue
            spec = None
            mp = getattr(scene, "mat_phys", None)
            if mp is not None and m < len(mp):
                spec = mp[m]
            if spec and _phys_fit_mode() == "anchored" and not any(
                    str(la.get("model", "")) == "panel" for la in poromat_layers(spec)):
                # comparison only: the older continuous anchored fit
                from . import poromat
                from .reflection import alpha_random
                lam_o = _LAMBDA4 if int(order) == 4 else _LAMBDA
                dt_est = lam_o * dx / speed_of_sound
                sp_n = poromat.normalize_spec(spec)
                if spec.get("inferred"):
                    afull = getattr(scene, "mat_alpha_full", None)
                    afull = afull[m] if afull and m < len(afull) else None
                    fb, ab = afull if afull is not None else (_BC, scene.absorption[m])
                else:
                    fb = np.geomspace(20.0, 1.2 * f_max, 64)
                    ab = alpha_random(poromat.gamma_full(sp_n, fb))
                gi[m], fA[m], fL[m] = fit_admittance_random(
                    ab, fb, f_hi=f_max, l_max=0.6 / dt_est,
                    gamma_target=lambda ff, sp=sp_n: poromat.gamma_full(sp, ff),
                    dt=dt_est)
                continue
            if spec:
                from . import poromat
                spec = poromat.normalize_spec(spec)
                fg = np.geomspace(30.0, 1.2 * f_max, 48)
                lam_o = _LAMBDA4 if int(order) == 4 else _LAMBDA
                dt_est = lam_o * dx / speed_of_sound
                secs = []
                if res_on:
                    secs, g_res, f_res = poromat.resonant_sections(
                        spec, f_lo=30.0, f_hi=1.2 * f_max)
                if secs:
                    # panel: resonant section PRE-COMPENSATED for the
                    # grid -- branches fitted against the FULL target
                    # with the section in the discrete model, section
                    # polished with frozen branches, branches refitted.
                    from .reflection import polish_resonant
                    g_full = poromat.gamma_full(spec, f_res)
                    gi[m], fA[m], fL[m] = fit_admittance_complex(
                        f_res, g_full, l_max=0.6 / dt_est, dt=dt_est,
                        lam=lam_o, res_sections=secs)
                    for si_ in range(len(secs)):
                        others = secs[:si_] + secs[si_ + 1:]
                        one = polish_resonant(
                            f_res, g_full, [secs[si_]], gi[m], fA[m],
                            fL[m], dt_est, lam_o, fixed=others)
                        secs = secs[:si_] + one + secs[si_ + 1:]
                    gi[m], fA[m], fL[m] = fit_admittance_complex(
                        f_res, g_full, l_max=0.6 / dt_est, dt=dt_est,
                        lam=lam_o, res_sections=secs)
                    s5 = secs[0]
                    b_, w2_ = float(s5[-2]), float(s5[-1])
                    a_ = float(s5[0])
                    nm_lbl = scene.mat_names[m] if m < len(
                        scene.mat_names) else str(m)
                    print("[Auralis] resonant wall '%s': f0=%.0f Hz "
                          "Q=%.2f a=%.1f (panel, %d section(s), "
                          "grid pre-compensated)"
                          % (nm_lbl, np.sqrt(w2_) / (2 * np.pi),
                             np.sqrt(w2_) / max(b_, 1e-9), a_,
                             len(secs)))
                    filt_res[m] = secs
                else:
                    # any other physical model (porous stacks, screens, perforated
                    # sheets, measured complex tables, inferred models): the passive
                    # discrete wall that best follows it (fit_physical_wall). gamma_full
                    # reads measured complex tables too, unlike gamma_normal.
                    nm_lbl = scene.mat_names[m] if m < len(scene.mat_names) else str(m)
                    gi[m], fA[m], fL[m], wsecs, how, werr = fit_physical_wall(
                        lambda ff, sp=spec: poromat.gamma_full(sp, ff), f_max, dt_est,
                        lam_o, key=repr(spec))
                    if wsecs:
                        filt_res[m] = list(wsecs)
                    if how != "complex" or werr > 0.05:
                        print("[Auralis] wall of '%s': %s (discrete |R - R_model| %.3f in "
                              "the wave band)" % (nm_lbl, how, werr))
            else:
                afull = getattr(scene, "mat_alpha_full", None)
                afull = afull[m] if afull and m < len(afull) else None
                if afull is not None:
                    # full tabulated curve (third octaves from 20 Hz):
                    # the wave band follows the TRUE shape below the
                    # first octave instead of the flat clamp
                    fb, ab = afull
                    refit[m] = (ab, fb)
                    gi[m], fA[m], fL[m] = _fit_random_cached(ab, fb, f_max)
                else:
                    refit[m] = (scene.absorption[m], _BC)
                    gi[m], fA[m], fL[m] = _fit_random_cached(scene.absorption[m], _BC, f_max)
        # passivity in the GRID's discrete time (a passive continuous fit
        # can still create energy there); passive fits are left untouched
        dt_g = (_LAMBDA4 if int(order) == 4 else _LAMBDA) * float(vox.dx) / speed_of_sound
        for m in range(nm):
            if filt_res[m]:
                continue             # resonant sections: z-exact, l <= 0.6/dt
            low, f_low = wall_passivity(gi[m], fA[m], fL[m], dt_g)
            if low >= 0.0:
                continue
            # the explicit high-pass branch over-subtracts at low frequency
            # by (1/(1-e^-l dt) - 1/2) l dt instead of 1 (31 % at l dt = 2):
            # stiff poles are capped first, then g_inf takes the remainder
            low0, a125_0 = low, _alpha_at(gi[m], fA[m], fL[m], 125.0)
            if refit[m] is not None:
                r = _fit_random_cached(refit[m][0], refit[m][1], f_max, l_max=0.6 / dt_g)
                low_r, _f = wall_passivity(*r, dt_g)
                if low_r > low:
                    (gi[m], fA[m], fL[m]), low = r, low_r
            how = "poles capped at 0.6/dt" if low >= 0.0 else ""
            if low < 0.0:
                gi[m] += 1.05 * (-low)
                how += ("%sg_inf raised by %.1e (alpha at 125 Hz %.4f -> %.4f)"
                        % ("; " if how else "", 1.05 * (-low), a125_0,
                           _alpha_at(gi[m], fA[m], fL[m], 125.0)))
            nm_lbl = (scene.mat_names[m] if m < len(scene.mat_names)
                      else str(m))
            print("[Auralis] WARNING: wall filter of '%s' was ACTIVE on the "
                  "grid (discrete admittance %+.1e at %.0f Hz: it creates "
                  "energy, the slow boundary mode of near-rigid rooms): %s."
                  % (nm_lbl, low0, f_low, how))
        # Quasi-DC damping of the physical walls in a closed room. A rigid-backed physical
        # model is rigid at DC (gamma -> 0). The room's quasi-DC mode is then undamped, and
        # the smallest numerical error makes it drift and grow (+20 dB below 10 Hz within
        # seconds, then leaking into the octaves). Alpha-only fits keep a DC conductance, and
        # in an open domain the mode leaves through the absorbing layer. The second-order
        # section a jw / ((jw)^2 + b jw + w0^2), w0 = 2 pi 0.2 Hz, b = 2 pi 4 Hz, a = g b, is a
        # conductance g = 0.01 from ~0.5 to ~3 Hz and ~1e-5 at 125 Hz. It damps the drift
        # without touching the octaves. AURALIS_FDTD_DC_DAMP=0 turns it off.
        dc_g = float(os.environ.get("AURALIS_FDTD_DC_DAMP", "0.01"))
        if not is_open and dc_g > 0.0:
            b_dc, w2_dc = 2.0 * np.pi * 4.0, (2.0 * np.pi * 0.2) ** 2
            for m in range(nm):
                if refit[m] is not None:
                    continue                 # alpha-only fit: its own DC conductance
                filt_res[m] = list(filt_res[m] or []) + [(dc_g * b_dc, b_dc, w2_dc)]
        if not any(filt_res):
            filt_res = None
        grid_key = None
        if _KEEP["depth"] > 0:
            # everything the grid depends on (conformal arrays: through the grid itself)
            grid_key = ("scene", _geometry_key(scene), float(dx), tuple(np.round(vox.origin, 12)),
                        vox.shape, _air_key(vox.air), bool(conf_on), bool(conformal_aniso),
                        int(order), repr(pml_cfg), None if sigma is None else int(sponge_cells),
                        float(g_border), float(speed_of_sound), gi.tobytes(), fA.tobytes(),
                        fL.tobytes(), repr(filt_res))
        if slab_mode:
            grid = _slab_grid(scene, vox, lambda sub, psub, cf: FdtdGrid(
                sub, gi, speed_of_sound, filt_mat=(gi, fA, fL), sigma=None,
                gamma_border=g_border, order=order, pml=psub, filt_res=filt_res, **cf),
                engine, pml_cfg, order, conf_on, aniso, src_pos, rcvs, src_axis, dx)
        else:
            grid = _kept_grid(grid_key, lambda: FdtdGrid(
                vox, gi, speed_of_sound, filt_mat=(gi, fA, fL), sigma=sigma,
                gamma_border=g_border, order=order, pml=pml_cfg, phi=phi, face_area_w=face_w,
                filt_res=filt_res, excess_axis=excess_axis, extra_links=sub_links))
    else:
        alpha_low = scene.absorption[:, :2].mean(axis=1)
        grid_key = None
        if _KEEP["depth"] > 0:
            grid_key = ("plain", _geometry_key(scene), float(dx), tuple(np.round(vox.origin, 12)),
                        vox.shape, _air_key(vox.air), bool(conf_on), int(order), repr(pml_cfg),
                        None if sigma is None else int(sponge_cells), float(g_border),
                        float(speed_of_sound), np.asarray(alpha_low).tobytes())
        if slab_mode:
            grid = _slab_grid(scene, vox, lambda sub, psub, cf: FdtdGrid(
                sub, gamma_from_alpha(alpha_low), speed_of_sound, sigma=None,
                gamma_border=g_border, order=order, pml=psub, **cf),
                engine, pml_cfg, order, conf_on, aniso, src_pos, rcvs, src_axis, dx)
        else:
            grid = _kept_grid(grid_key, lambda: FdtdGrid(
                vox, gamma_from_alpha(alpha_low), speed_of_sound, sigma=sigma,
                gamma_border=g_border, order=order, pml=pml_cfg, phi=phi, face_area_w=face_w,
                excess_axis=excess_axis, extra_links=sub_links))
    del phi, face_w, excess_axis, sub_links   # folded into the grid: free them before the run
    n_steps = int(np.ceil(duration / grid.dt)) + 8
    pulse = source_pulse(grid.dt, f_max / 3.2, n_steps)
    if src_surface is not None:
        if is_open:
            raise ValueError("a vibrating surface needs a closed room (open wave domain)")
        if slab_mode:
            raise ValueError("a vibrating surface needs a grid below 2^31 air cells")
        k = (src_surface if isinstance(src_surface, (int, np.integer))
             else list(scene.mat_names).index(str(src_surface)))
        st = surface_taps(grid, scene, int(k))
        print("[Auralis] wave band: vibrating surface '%s' (%d air cells beside it)"
              % (scene.mat_names[int(k)], len(st[0])))
        src_axis = None
    else:
        st = grid.source_taps(src_pos)
        if hasattr(grid, "volumes"):
            i_p, w_p = grid.taps(src_pos)
            vol = float(np.sum(w_p * grid.volumes(i_p)))   # the volume factor it had
        else:
            vol = 1.0
        if abs(vol - 1.0) > 1e-3:
            print("[Auralis] wave band: the source sits in the walls' cut layer (volume factor "
                  "%.3f): its injection is divided by the cells' volumes (without this it "
                  "would radiate %+.2f dB too much)" % (vol, 20.0 * np.log10(vol)))
    missing = []
    if missing_rcv_ok:
        rts = []
        for k, r in enumerate(rcvs):
            try:
                rts.append(grid.taps(r))
            except ValueError:
                missing.append(k)
                rts.append(st)            # recorded anywhere, zeroed at the end
    else:
        rts = [grid.taps(r) for r in rcvs]
    st_d, h_dip = None, 2.0 * dx
    if src_axis is not None and src_first_order is not None and transfer_freqs is None:
        try:
            st_d = _dipole_taps(grid, src_pos, src_axis, h_dip)
        except ValueError as exc:
            print("[Auralis] WARNING: directional source impossible in the wave band "
                  "(%s, too close to a surface): OMNI below the crossover." % exc)
    _amean = scene.absorption[:, :3].mean(axis=1)
    _rigid = [scene.mat_names[i] for i in np.unique(scene.tri_mat)
              if _amean[i] < 0.03]
    if _rigid and duration > 1.5:
        print("[Auralis] note: near-rigid materials in the wave band "
              "(low-frequency alpha < 0.03): %s. Their wall filters "
              "are checked for passivity on the grid (repairs are "
              "reported above); the watchdog still stops a run "
              "cleanly if a boundary mode grows. Realistic values "
              "(furnished room: 0.05-0.15) also sound truer."
              % ", ".join(_rigid))
    _vair = float(grid.n_air) * grid.dx ** 3
    _tvr = np.asarray(scene.tri_verts, float).reshape(-1, 3)
    _vbox = float(np.prod(_tvr.max(0) - _tvr.min(0)))
    if _vbox > 1.0 and _vair < 0.25 * _vbox:
        print("[Auralis] WARNING: the air connected to the SOURCE is "
              "only %.1f m3 (%.0f %% of the scene) -- source probably "
              "boxed in (corner, furniture, niche): voxelization "
              "inflates every surface by ~%.0f cm. Move the source at "
              "least %.0f cm away from any surface, otherwise the "
              "whole rest of the room counts as 'outside the air'."
              % (_vair, 100 * _vair / _vbox, 87 * grid.dx,
                 2 * 100 * grid.dx))
    if is_open and duration > 1.5:
        print("[Auralis] WARNING: open FDTD domain (leaky room or "
              "source/mic outside) over %.1f s -- envelope not "
              "validated on long durations; the watchdog will stop "
              "cleanly if an instability shows up."
              % duration)
    runner, eng_asked = _pick_fdtd_backend(engine, grid)
    _ENGINE_USED[0] = None
    p = runner(grid, pulse, st, rts, n_steps)          # (n_steps, n_rcv)
    p_d = None if st_d is None else runner(grid, pulse, st_d, rts, n_steps)
    if slab_mode:
        grid.close()                     # free the devices for the empty-box pass
    # only its scalars are used from here on: drop the arrays before the
    # empty-grid pass of open domains (two grids at once = the RAM peak)
    grid = SimpleNamespace(dt=grid.dt, n_air=grid.n_air, order=grid.order)
    # REAL backend (GPU->CPU / ABI / python fallbacks), not the asked one
    eng_used = _ENGINE_USED[0] or eng_asked
    if eng_used != eng_asked:
        print("[Auralis] FDTD: REAL backend '%s' (asked: '%s') -- a "
              "fallback took place." % (eng_used, eng_asked))

    C, n_fft, _, band = free_field_calibration(dx, speed_of_sound,
                                               pulse, n_steps, order=order)
    # The calibrated source is flat down to ~5 Hz. Below its first mode a closed room is
    # pressurised: a flat volume acceleration gives a pressure rising as 1/f^2. Walls rigid at
    # DC keep that content for seconds, and its truncation at the end of the IR leaks into the
    # 63-125 Hz analysis bands. No real loudspeaker radiates there, so a causal 4th-order
    # Butterworth high-pass at AURALIS_FDTD_SUBSONIC_HZ (16 Hz, 0 = off) does what its roll-off
    # would do (-0.02 dB at 44 Hz, -40 dB at 5 Hz). Open domains are not pressurised and
    # transfer functions (transfer_freqs) are not filtered either.
    # subsonic_hz=0 is for callers that remove the DC term with a zero-phase filter (field
    # animation: a causal roll-off would leave a 10-20 Hz transient on the frames).
    f_ss = float(os.environ.get("AURALIS_FDTD_SUBSONIC_HZ", "16")
                 if subsonic_hz is None else subsonic_hz)
    subsonic = 1.0
    if not is_open and f_ss > 0.0:
        subsonic = subsonic_highpass(np.fft.rfftfreq(n_fft, grid.dt), f_ss)
    if is_open:
        # scattered-field decomposition: a second pass on the SAME
        # EMPTY grid exactly cancels the collar echo of the direct
        # path; the ideal direct is reinjected in closed form.
        def _empty_grid():
            vempty = VoxelGrid(solid=np.zeros(vox.shape, bool),
                               mat=np.full(vox.shape, -1, np.int32),
                               origin=vox.origin, dx=dx)
            vempty.air = ~vempty.solid
            return FdtdGrid(vempty, np.zeros(1), speed_of_sound,
                            sigma=sigma, gamma_border=g_border,
                            order=order, pml=pml_cfg)
        # dense pass (12 bytes a cell, split over the same GPUs, about twice as fast) unless the
        # grids are kept: the animated wave band reuses its indexed empty grid instead of
        # rebuilding tables every run. Always dense for a slab-built scene.
        devs = _fdtd_devices()
        use_dense = sigma is None and (slab_mode or _KEEP["depth"] <= 0)
        dlib = _dense_empty_lib(engine) if use_dense else None
        p0 = None
        if dlib is not None:
            # the same floats without neighbour tables: 12 bytes a cell instead of ~75 (dense.py)
            from . import dense
            try:
                box = dense.DenseBox(vox.shape, vox.origin, dx)
                rt0s = [box.taps(r) for r in rcvs]
                kw0 = dict(lib=dlib, shape=vox.shape, origin=vox.origin, dx=dx,
                           c=speed_of_sound, order=order, pml=pml_cfg, g_border=g_border,
                           pulse=pulse, rec_taps=rt0s, n_steps=n_steps,
                           devs=devs if devs and len(devs) > 1 else None)
                p0 = dense.run_empty(src_taps=box.taps(src_pos), **kw0)
                if p_d is not None:
                    p0_d = dense.run_empty(src_taps=dense.dipole_taps(box, src_pos, src_axis,
                                                                      h_dip), **kw0)
            except (ValueError, RuntimeError) as exc:
                print("[Auralis] dense empty-box pass UNAVAILABLE (%s): indexed empty grid"
                      % exc)
                p0 = None
        if p0 is None and slab_mode:
            raise RuntimeError("slab-built wave grid: the empty-box pass needs the dense "
                               "kernel of the CUDA library (aur_fdtd_dense_empty)")
        if p0 is None:
            gempty = _kept_grid(None if _KEEP["depth"] <= 0 else (
                "empty", float(dx), tuple(np.round(vox.origin, 12)), vox.shape, int(order),
                repr(pml_cfg), None if sigma is None else int(sponge_cells), float(g_border),
                float(speed_of_sound)), _empty_grid)
            st0 = gempty.taps(src_pos)
            rt0s = [gempty.taps(r) for r in rcvs]
            run0, _ = _pick_fdtd_backend(engine, gempty)
            p0 = run0(gempty, pulse, st0, rt0s, n_steps)
            if p_d is not None:
                p0_d = run0(gempty, pulse, _dipole_taps(gempty, src_pos, src_axis, h_dip),
                            rt0s, n_steps)
        f = np.fft.rfftfreq(n_fft, grid.dt)
        h = np.empty((n_fft, p.shape[1]))
        for r in range(p.shape[1]):
            d1 = float(np.linalg.norm(rcvs[r]
                                      - np.asarray(src_pos, float)))
            direct = (np.exp(-2j * np.pi * f
                             * (d1 / speed_of_sound))
                      / d1) * band     # same band as C
            P = np.fft.rfft(p[:, r] - p0[:, r], n_fft) * C
            if p_d is not None:
                wa, wb, dinv = _first_order_filters(f, src_first_order, h_dip, speed_of_sound)
                u = np.asarray(src_axis, float) / np.linalg.norm(src_axis)
                cos_r = float(np.dot(rcvs[r] - np.asarray(src_pos, float), u)) / max(d1, 1e-9)
                P = (wa * P + wb * dinv * np.fft.rfft(p_d[:, r] - p0_d[:, r], n_fft) * C
                     + direct * (wa + wb * cos_r))
            else:
                P = P + direct
            h[:, r] = np.fft.irfft(P * subsonic, n_fft)
    elif transfer_freqs is not None:
        h = None
    else:
        h = np.empty((n_fft, p.shape[1]))
        if p_d is not None:
            wa, wb, dinv = _first_order_filters(np.fft.rfftfreq(n_fft, grid.dt),
                                                src_first_order, h_dip, speed_of_sound)
        for r in range(p.shape[1]):
            P = np.fft.rfft(p[:, r], n_fft)
            if p_d is not None:
                P = wa * P + wb * dinv * np.fft.rfft(p_d[:, r], n_fft)
            h[:, r] = np.fft.irfft(P * C * subsonic, n_fft)
    if transfer_freqs is not None:
        fb = np.fft.rfftfreq(n_fft, grid.dt)
        idx = np.clip(np.round(np.asarray(transfer_freqs, float) * n_fft * grid.dt)
                      .astype(np.int64), 0, len(fb) - 1)
        if h is None:                                  # closed domain: P = FFT(p) C
            tf = (np.fft.rfft(p, n_fft, axis=0)[idx] * C[idx, None])
        else:                                          # open domain: from h
            tf = np.fft.rfft(h, n_fft, axis=0)[idx]
        info = FdtdInfo(f_cross=float(f_cross), f_max_grid=f_max, dx=dx,
                        dt=grid.dt, n_cells=n_cells, n_air=grid.n_air,
                        n_steps=n_steps, runtime_s=time.time() - t0,
                        engine=eng_used, air_span=(0.0, 0.0, 0.0),
                        is_open=is_open, order=grid.order, absorber=absorber,
                        conformal=conf_on, vol_gain_pct=vol_gain_pct,
                        area_stair_pct=area_stair_pct)
        info.transfer_freqs = fb[idx]
        return tf, info
    n_out = int(round(duration * fs))
    # Resampling preserves the time-domain VALUES; for an IR used in
    # convolution at fs, the in-band DFT magnitude must stay 1/d: we
    # correct by fs_grid/fs (without it the fused low band comes out
    # fs/fs_grid too loud, ~ +24 dB).
    ir = np.stack([_resample_fft(h[:, r], 1.0 / grid.dt, fs, n_out)
                   * ((1.0 / grid.dt) / float(fs))
                   for r in range(h.shape[1])], axis=1)
    if missing:
        ir[:, missing] = 0.0
    if single:
        ir = ir[:, 0]
    # extent of the air per axis, from three reductions (np.argwhere would allocate 24 bytes
    # per air cell)
    span = []
    for k in range(3):
        on = np.flatnonzero(vox.air.any(axis=tuple(j for j in range(3) if j != k)))
        span.append(float((on[-1] - on[0] + 1) * dx) if len(on) else 0.0)
    span = tuple(span)
    info = FdtdInfo(f_cross=float(f_cross), f_max_grid=f_max, dx=dx,
                    dt=grid.dt, n_cells=n_cells, n_air=grid.n_air,
                    n_steps=n_steps, runtime_s=time.time() - t0,
                    engine=eng_used, air_span=span, is_open=is_open,
                    order=grid.order, absorber=absorber,
                    conformal=conf_on, vol_gain_pct=vol_gain_pct,
                    area_stair_pct=area_stair_pct, missing=tuple(missing))
    return ir, info


def auto_crossover(volume_m3: float, duration_s: float,
                   speed_of_sound: float, f_schroeder: float,
                   order: int = 4, budget_gcell: float = 40.0,
                   fcross_max: float = 700.0) -> tuple:
    """Automatic crossover frequency from a compute budget
    (cost ~ V*duration*c/(lam h^4)). Returns (fc, fc_budget) in Hz."""
    lam = _LAMBDA4 if int(order) == 4 else _LAMBDA
    B = max(float(budget_gcell), 0.5) * 1e9
    h_star = (1.3 * max(volume_m3, 1.0) * max(duration_s, 0.05)
              * speed_of_sound / (lam * B)) ** 0.25
    fc_budget = speed_of_sound / (13.5 * h_star)
    fc = max(float(f_schroeder), min(fc_budget, float(fcross_max)))
    fc = float(np.clip(fc, 60.0, max(float(fcross_max), 60.0)))
    return fc, float(fc_budget)


def fuse_low_high(ir_geo: np.ndarray, ir_low: np.ndarray,
                  f_cross: float, fs: int):
    """ir = HP*ir_geo + LP*ir_low, delays aligned. ir_low: (n,) mono
    duplicated across channels, or (n, n_ch) per channel."""
    lp, hp, delay = design_crossover(f_cross, fs)
    n = ir_geo.shape[0]
    out = np.zeros_like(ir_geo)
    low2 = np.atleast_2d(ir_low.T).T
    for ch in range(ir_geo.shape[1]):
        lo = np.convolve(low2[:n, min(ch, low2.shape[1] - 1)],
                         lp)[delay:delay + n]
        hi = np.convolve(ir_geo[:, ch], hp)[delay:delay + n]
        out[:, ch] = hi + lo
    return out
