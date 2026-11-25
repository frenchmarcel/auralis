# -*- coding: utf-8 -*-
"""ctypes bridge to the native wave-band grid builder (libauralis, ABI 2.4)."""
from __future__ import annotations

import ctypes as ct
import os

import numpy as np

_STATE = {"lib": None, "tried": False}
_D = ct.POINTER(ct.c_double)
_I32 = ct.POINTER(ct.c_int32)
_I64 = ct.POINTER(ct.c_int64)
_U8 = ct.POINTER(ct.c_uint8)


def lib():
    """libauralis with the aur_grid_* functions bound, else None with one loud warning.

    AURALIS_GRID_NATIVE=0 selects the Python reference without a warning. Every function
    gets its argtypes: without them ctypes passes pointers as 32-bit ints on some platforms."""
    if os.environ.get("AURALIS_GRID_NATIVE", "1") == "0":
        return None
    if _STATE["tried"]:
        return _STATE["lib"]
    _STATE["tried"] = True
    try:
        from .native import load_library
        L = load_library()
    except OSError as exc:
        print("[Auralis] WARNING: native grid builder unavailable (%s): Python grid "
              "construction (slow). Build it with native/build.sh." % exc)
        return None
    if not hasattr(L, "aur_grid_wall_faces"):
        print("[Auralis] WARNING: libauralis has no grid builder (ABI < 2.4): Python grid "
              "construction (slow). Rebuild it with native/build.sh.")
        return None
    L.aur_grid_voxelize.restype = ct.c_int32
    L.aur_grid_voxelize.argtypes = [_D, ct.c_int32, _I32, _D, ct.c_double, _I32, _U8, _I32,
                                    _U8, ct.c_int32]
    L.aur_grid_flood.restype = ct.c_int64
    L.aur_grid_flood.argtypes = [_U8, _I32, ct.c_int64, _U8]
    L.aur_grid_fine_fraction.restype = ct.c_int32
    L.aur_grid_fine_fraction.argtypes = [_D, ct.c_int32, _D, ct.c_double, _I32, ct.c_int32,
                                         _U8, ct.c_int32, _D, ct.c_int32]
    L.aur_grid_donate.restype = ct.c_int32
    L.aur_grid_donate.argtypes = [_I32, _U8, _D, ct.c_int32, ct.c_double, _D, _U8, ct.c_int32]
    if hasattr(L, "aur_grid_structure"):
        L.aur_grid_structure.restype = ct.c_int64
        L.aur_grid_structure.argtypes = [_U8, _I32, _I32, _I32, _I32, _D, _I32, _U8, ct.c_int32]
        L.aur_grid_pml_axis.restype = ct.c_int64
        L.aur_grid_pml_axis.argtypes = [_I32, _I32, ct.c_int64, _I32, ct.c_int32, ct.c_int32,
                                        _I64, _I32, ct.c_int32]
        if hasattr(L, "aur_grid_wall_list"):
            L.aur_grid_wall_list.restype = ct.c_int64
            L.aur_grid_wall_list.argtypes = [_I32, _I32, ct.c_int64, _I32, ct.c_int32, _I64,
                                             _U8, ct.c_int32]
        if hasattr(L, "aur_grid_coefficients"):
            L.aur_grid_coefficients.restype = None
            L.aur_grid_coefficients.argtypes = [ct.c_int64, ct.c_double, ct.c_double,
                                                ct.c_double, _D, _D, _D, _D, _D, _D, _D, _D,
                                                _D, _D, ct.c_int32]
        L.aur_grid_collar.restype = None
        L.aur_grid_collar.argtypes = [_I32, _I32, ct.c_int64, _U8, _U8, ct.c_int32]
    L.aur_grid_wall_faces.restype = ct.c_int64
    L.aur_grid_wall_faces.argtypes = [_D, ct.c_int32, _D, ct.c_double, _I32, _U8, _I64,
                                      ct.c_int32, _D, _D, _D, ct.c_int32]
    _STATE["lib"] = L
    return L


def _p(a, t):
    return a.ctypes.data_as(t)


def _threads():
    try:
        return int(os.environ.get("AURALIS_GRID_THREADS", "0"))
    except ValueError:
        return 0


def _tris(tv):
    return np.ascontiguousarray(np.asarray(tv, dtype=np.float64).reshape(-1, 9))


def voxelize_tris(tv, tri_mat, origin, dx, n):
    """(solid, matid, nrmq) of fdtd.voxelize's triangle loop."""
    L = lib()
    t = _tris(tv)
    tm = np.ascontiguousarray(tri_mat, dtype=np.int32)
    o = np.ascontiguousarray(origin, dtype=np.float64)
    n32 = np.ascontiguousarray(n, dtype=np.int32)
    shape = tuple(int(v) for v in n)
    solid = np.empty(shape, dtype=bool)
    matid = np.empty(shape, dtype=np.int32)
    nrmq = np.empty(shape + (3,), dtype=np.uint8)
    L.aur_grid_voxelize(_p(t, _D), len(t), _p(tm, _I32), _p(o, _D), float(dx), _p(n32, _I32),
                        _p(solid, _U8), _p(matid, _I32), _p(nrmq, _U8), _threads())
    return solid, matid, nrmq


def flood(solid, seed):
    """Air component (bool) of fdtd._flood_air, from the seed cell (i, j, k)."""
    L = lib()
    s = np.ascontiguousarray(solid, dtype=bool)
    n32 = np.asarray(s.shape, dtype=np.int32)
    air = np.empty(s.shape, dtype=bool)
    L.aur_grid_flood(_p(s, _U8), _p(n32, _I32), int(np.ravel_multi_index(seed, s.shape)),
                     _p(air, _U8))
    return air


def fine_fraction(tv, org_f, dx_f, shape, factor, air, n_iter):
    """conformal.fine_air_fractions after its RAM check: TRUE air fraction per coarse cell."""
    L = lib()
    t = _tris(tv)
    o = np.ascontiguousarray(org_f, dtype=np.float64)
    n32 = np.ascontiguousarray(shape, dtype=np.int32)
    a = np.ascontiguousarray(air, dtype=bool)
    frac = np.empty(tuple(int(v) for v in shape), dtype=np.float64)
    L.aur_grid_fine_fraction(_p(t, _D), len(t), _p(o, _D), float(dx_f), _p(n32, _I32),
                             int(factor), _p(a, _U8), int(n_iter), _p(frac, _D), _threads())
    return frac


def wall_faces(tv, origin_center, dx, air, idx, max_walk):
    """(W, cell_area, stats) of conformal.wall_face_areas."""
    L = lib()
    t = _tris(tv)
    o = np.ascontiguousarray(origin_center, dtype=np.float64)
    a = np.ascontiguousarray(air, dtype=bool)
    n32 = np.asarray(a.shape, dtype=np.int32)
    ix = np.ascontiguousarray(idx, dtype=np.int64)
    W = np.zeros((int(a.sum()), 6), dtype=np.float64)
    cell_area = np.zeros(a.shape, dtype=np.float64)
    st = np.zeros(5, dtype=np.float64)
    L.aur_grid_wall_faces(_p(t, _D), len(t), _p(o, _D), float(dx), _p(n32, _I32), _p(a, _U8),
                          _p(ix, _I64), int(max_walk), _p(W, _D), _p(cell_area, _D), _p(st, _D),
                          _threads())
    stats = {"area_true_m2": float(st[0]), "area_mapped_m2": float(st[1]),
             "area_dropped_m2": float(st[2]), "area_membrane_m2": float(st[3]),
             "area_edge_m2": float(st[4])}
    return W, cell_area, stats


def donate(air, frac, smooth_sweeps, t_diff):
    """(phi before the cap, orphan mask) of the default path of conformal_corrections."""
    L = lib()
    a = np.ascontiguousarray(air, dtype=bool)
    f = np.ascontiguousarray(frac, dtype=np.float64)
    n32 = np.asarray(a.shape, dtype=np.int32)
    phi = np.empty(int(a.sum()), dtype=np.float64)
    orphan = np.empty(a.shape, dtype=bool)
    L.aur_grid_donate(_p(n32, _I32), _p(a, _U8), _p(f, _D), int(smooth_sweeps), float(t_diff),
                      _p(phi, _D), _p(orphan, _U8), _threads())
    return phi, orphan


def has_structure():
    """True if the library also builds the FdtdGrid structure (ABI 2.5)."""
    L = lib()
    return L is not None and hasattr(L, "aur_grid_structure")


def structure(air):
    """(idx, cells, nb, K, nb2, full) of FdtdGrid: grid -> air index (-1), coordinates,
    neighbours (ghost = n_air), air-neighbour count, second neighbours, all 12 in air."""
    L = lib()
    a = np.ascontiguousarray(air, dtype=bool)
    n32 = np.asarray(a.shape, dtype=np.int32)
    n_air = int(a.sum())
    idx = np.empty(a.shape, dtype=np.int32)
    cells = np.empty((n_air, 3), dtype=np.int32)
    nb = np.empty((n_air, 6), dtype=np.int32)
    K = np.empty(n_air, dtype=np.float64)
    nb2 = np.empty((n_air, 6), dtype=np.int32)
    full = np.empty(n_air, dtype=bool)
    L.aur_grid_structure(_p(a, _U8), _p(n32, _I32), _p(idx, _I32), _p(cells, _I32),
                         _p(nb, _I32), _p(K, _D), _p(nb2, _I32), _p(full, _U8), _threads())
    return idx, cells, nb, K, nb2, full


def pml_axis(cells, nb, shape, npml, axis):
    """(sel int64, locm int32) of the CPML table of one axis."""
    L = lib()
    n_air = len(cells)
    n32 = np.asarray(shape, dtype=np.int32)
    sel = np.empty(n_air, dtype=np.int64)
    locm = np.empty(n_air, dtype=np.int32)
    m = int(L.aur_grid_pml_axis(_p(cells, _I32), _p(nb, _I32), n_air, _p(n32, _I32),
                                int(npml), int(axis), _p(sel, _I64), _p(locm, _I32),
                                _threads()))
    return sel[:m].copy(), locm[:m].copy()


def collar(nb, nb2, pmlm):
    """Air cells that are CPML cells or reach one through nb / nb2 (bool)."""
    L = lib()
    pm = np.ascontiguousarray(pmlm, dtype=bool)
    out = np.empty(len(nb), dtype=bool)
    L.aur_grid_collar(_p(nb, _I32), _p(nb2, _I32), len(nb), _p(pm, _U8), _p(out, _U8),
                      _threads())
    return out


def wall_list(cells, nb, shape, f):
    """(wi int64, inb bool) of the wall faces of direction f; None if the library is older
    than ABI 2.6 (the numpy scan is used then)."""
    L = lib()
    if L is None or not hasattr(L, "aur_grid_wall_list"):
        return None
    n_air = len(cells)
    n32 = np.asarray(shape, dtype=np.int32)
    wi = np.empty(n_air, dtype=np.int64)
    inb = np.empty(n_air, dtype=bool)
    m = int(L.aur_grid_wall_list(_p(cells, _I32), _p(nb, _I32), n_air, _p(n32, _I32), int(f),
                                 _p(wi, _I64), _p(inb, _U8), _threads()))
    return wi[:m].copy(), inb[:m].copy()


def coefficients(half_lam, lam2, lam_dt, gam, K, s_half, ph, with_drv):
    """(Gamma, c_center, c_old, c_now, c_nb, c_drv or None) of FdtdGrid, or None if the library
    is older than ABI 2.6."""
    L = lib()
    if L is None or not hasattr(L, "aur_grid_coefficients"):
        return None
    arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in (gam, K, s_half, ph)]
    n = len(arrs[0])
    out = [np.empty(n) for _ in range(5)]
    drv = np.empty(n) if with_drv else None
    L.aur_grid_coefficients(n, float(half_lam), float(lam2), float(lam_dt),
                            *[_p(a, _D) for a in arrs], *[_p(o, _D) for o in out],
                            _p(drv, _D) if drv is not None else None, _threads())
    return tuple(out) + (drv,)
