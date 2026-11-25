import os
import time

import numpy as np


def _flood(blocked, seed):
    """Cells 6-connected to seed through ~blocked (bool mask of the grid)."""
    from . import gridnative
    if gridnative.lib() is not None:
        return gridnative.flood(blocked, tuple(int(v) for v in seed))
    out = np.zeros_like(blocked)
    out[tuple(seed)] = True
    open_ = ~blocked
    while True:
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[:, :, 1:] |= out[:, :, :-1]
        grown[:, :, :-1] |= out[:, :, 1:]
        grown &= open_
        if grown.sum() == out.sum():
            return out
        out = grown


def _dilate(mask, r):
    """Chebyshev dilation of a bool grid by r cells (separable, one axis at a time)."""
    out = mask.copy()
    for ax in range(3):
        acc = out.copy()
        for s in range(1, int(r) + 1):
            if s >= out.shape[ax]:
                break
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            dst[ax], src[ax] = slice(s, None), slice(None, -s)
            acc[tuple(dst)] |= out[tuple(src)]
            dst[ax], src[ax] = slice(None, -s), slice(s, None)
            acc[tuple(dst)] |= out[tuple(src)]
        out = acc
    return out


def _exterior(free):
    """Free cells connected to a face of the box (what lies outside a closed room)."""
    ext = np.zeros_like(free)
    faces = [(0, 0), (0, -1), (1, 0), (1, -1), (2, 0), (2, -1)]
    for _ in range(32):
        seed = None
        for ax, side in faces:
            sl = [slice(None)] * 3
            sl[ax] = side
            m = free[tuple(sl)] & ~ext[tuple(sl)]
            if m.any():
                loc = list(np.unravel_index(int(np.argmax(m)), m.shape))
                loc.insert(ax, side % free.shape[ax])
                seed = tuple(loc)
                break
        if seed is None:
            return ext
        ext |= _flood(~free, seed)
    print("[Auralis] sub-cell openings: more than 32 separate free regions on the faces "
          "of the box; the rest is taken as interior")
    return ext


def find_pockets(vox, reach, min_m3, max_pockets=64):
    """Enclosed free regions (not the main air, not outside the box) that come within
    `reach` cells of the main air through solid cells, of at least min_m3: a list of bool
    masks, and the number of smaller ones skipped."""
    free = ~vox.solid
    rest = free & ~vox.air
    if not rest.any():
        return [], 0
    near = rest & _dilate(vox.air, reach + 1)
    if not near.any():
        return [], 0
    interior = rest & ~_exterior(free)
    near &= interior
    pockets, n_small = [], 0
    cell_m3 = float(vox.dx) ** 3
    while near.any():
        if len(pockets) + n_small >= max_pockets:
            print("[Auralis] sub-cell openings: more than %d pockets next to the air, the "
                  "others stay closed" % max_pockets)
            break
        seed = np.unravel_index(int(np.argmax(near)), near.shape)
        comp = _flood(~interior, seed)
        near &= ~comp
        if comp.sum() * cell_m3 >= min_m3:
            pockets.append(comp)
        else:
            n_small += 1
    return pockets, n_small


def _fine_solid(scene, origin_f, h, nf):
    """Solid mask of a fine box (cell centres origin_f + i h): a cell within h/2 of a triangle.
    Thinner than the 0.87 dx of fdtd.voxelize and still without leaks for 6-neighbour steps:
    a step of length h that crosses a triangle has an end within h/2 of it."""
    from .fdtd import _point_tri_dist
    tv = np.asarray(scene.tri_verts, dtype=np.float64)
    org = np.asarray(origin_f, dtype=np.float64)
    nf = np.asarray(nf, dtype=np.int64)
    rad = 0.5 * h * (1.0 + 1e-9)
    solid = np.zeros(tuple(int(v) for v in nf), dtype=bool)
    blo, bhi = org - rad, org + (nf - 1) * h + rad
    tlo, thi = tv.min(axis=1), tv.max(axis=1)
    inside = np.all((thi >= blo) & (tlo <= bhi), axis=1)
    for t in np.flatnonzero(inside):
        a, b, c = tv[t]
        i0 = np.maximum(np.floor((tlo[t] - rad - org) / h).astype(np.int64), 0)
        i1 = np.minimum(np.ceil((thi[t] + rad - org) / h).astype(np.int64) + 1, nf)
        if np.any(i0 >= i1):
            continue
        nrm = np.cross(b - a, c - a)
        ln = float(np.sqrt(nrm @ nrm))
        if ln <= 0.0:
            continue
        nrm = nrm / ln
        gx = org[0] + h * np.arange(i0[0], i1[0])
        gy = org[1] + h * np.arange(i0[1], i1[1])
        gz = org[2] + h * np.arange(i0[2], i1[2])
        pd = np.abs((gx[:, None, None] - a[0]) * nrm[0] + (gy[None, :, None] - a[1]) * nrm[1]
                    + (gz[None, None, :] - a[2]) * nrm[2])
        cand = np.argwhere(pd <= rad)
        if len(cand) == 0:
            continue
        P = np.stack([gx[cand[:, 0]], gy[cand[:, 1]], gz[cand[:, 2]]], axis=1)
        hit = _point_tri_dist(P, a, b, c) <= rad
        k = cand[hit] + i0
        solid[k[:, 0], k[:, 1], k[:, 2]] = True
    return solid


def _reach(open_, seeds, max_floods=256):
    """Cells of open_ 6-connected to a seed cell (floods until every seed is reached); None
    past max_floods separate pieces."""
    out = np.zeros_like(open_)
    left = seeds & open_
    blocked = ~open_
    for _ in range(max_floods):
        if not left.any():
            return out
        seed = np.unravel_index(int(np.argmax(left)), left.shape)
        comp = _flood(blocked, seed)
        out |= comp
        left &= ~comp
    print("[Auralis] sub-cell openings: more than %d separate pieces of a mouth in a fine box"
          % max_floods)
    return None


def _pcg(deg, nbs, b, tol, max_iter):
    """Jacobi-preconditioned CG on A u = b, A = diag(deg) - adjacency (nbs: (6, n) indices,
    n = none). Returns (u, iterations, relative residual)."""
    n = len(deg)
    minv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)

    def mat(x):
        xe = np.append(x, 0.0)
        y = deg * x
        for d in range(nbs.shape[0]):
            y -= xe[nbs[d]]
        return y
    u = np.zeros(n)
    r = b.copy()
    bn = float(np.sqrt(b @ b)) or 1.0
    z = minv * r
    p = z.copy()
    rz = float(r @ z)
    it = 0
    res = 1.0
    for it in range(1, max_iter + 1):
        ap = mat(p)
        pap = float(p @ ap)
        if pap <= 0:
            break
        a = rz / pap
        u += a * p
        r -= a * ap
        res = float(np.sqrt(r @ r)) / bn
        if res < tol:
            break
        z = minv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return u, it, res


def channel_flux(scene, vox, pocket, lo, hi, F, tol=2e-5, max_iter=40000):
    """Potential flow through the sub-cell channels between the main air (u = 1) and the
    pocket (u = 0) on a fine box (cell h = dx / F, coarse cells lo..hi): coarse air cells
    are equipotential terminals, fine air inside coarse solid cells is unknown.
    Returns ({coarse flat index: flux} main side, same pocket side, total flux reaching the
    pocket in fine-face units, CG iterations, residual)."""
    dx = float(vox.dx)
    h = dx / F
    nc = hi - lo
    nf = nc * F
    origin_f = np.asarray(vox.origin, dtype=np.float64) + (lo - 0.5 + 0.5 / F) * dx
    fsolid = _fine_solid(scene, origin_f, h, nf)
    sub = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
    cls_c = np.zeros(tuple(nc), dtype=np.int8)
    cls_c[vox.solid[sub]] = 3
    cls_c[vox.air[sub]] = 1
    cls_c[pocket[sub]] = 2
    cls = np.repeat(np.repeat(np.repeat(cls_c, F, axis=0), F, axis=1), F, axis=2)
    cls[fsolid] = 0
    del fsolid
    # only fine air reached from both sides carries flux: dead ends and closed pockets leave
    # before the solve
    from_main = _reach(cls != 0, cls == 1)
    if from_main is None or not (from_main & (cls == 2)).any():
        return {}, {}, 0.0, 0, 0.0
    from_pocket = _reach(cls != 0, cls == 2)
    if from_pocket is None:
        return {}, {}, 0.0, 0, 0.0
    cls[(cls == 3) & ~(from_main & from_pocket)] = 0
    del from_main, from_pocket
    # swapping the sides gives the same problem in 1 - u. CG from 0 converges far faster with
    # the larger mouth at u = 1 (about 13 times fewer iterations on a door), so solve that way
    unk_mask = cls == 3
    def _touch(side):
        m = cls == side
        t = np.zeros_like(m)
        t[1:] |= m[:-1]; t[:-1] |= m[1:]
        t[:, 1:] |= m[:, :-1]; t[:, :-1] |= m[:, 1:]
        t[:, :, 1:] |= m[:, :, :-1]; t[:, :, :-1] |= m[:, :, 1:]
        return int(np.count_nonzero(t & unk_mask))
    swap = _touch(2) > _touch(1)
    del unk_mask
    if swap:
        cls[cls == 1] = 4
        cls[cls == 2] = 1
        cls[cls == 4] = 2
    flat = cls.ravel()
    unk = np.flatnonzero(flat == 3)
    n_u = len(unk)
    if n_u == 0:
        return {}, {}, 0.0, 0, 0.0
    idx = np.full(flat.size, n_u, dtype=np.int32)
    idx[unk] = np.arange(n_u, dtype=np.int32)
    coord = np.stack(np.unravel_index(unk, tuple(nf)), axis=1)
    strides = (nf[1] * nf[2], nf[2], 1)
    deg = np.zeros(n_u)
    b = np.zeros(n_u)
    nbs = np.full((6, n_u), n_u, dtype=np.int32)
    term = []                        # (unknown, fine flat of the terminal, side)
    for d in range(6):
        ax, sgn = d // 2, (1 if d % 2 else -1)
        ok = (coord[:, ax] + sgn >= 0) & (coord[:, ax] + sgn < nf[ax])
        nb = unk + sgn * strides[ax]
        c = np.zeros(n_u, dtype=np.int8)
        c[ok] = flat[nb[ok]]
        deg += (c != 0)
        is_u = c == 3
        nbs[d, is_u] = idx[nb[is_u]]
        b += (c == 1)
        for side in (1, 2):
            m = np.flatnonzero(c == side)
            if len(m):
                term.append((m, nb[m], side))
    u, it, res = _pcg(deg, nbs, b, tol, max_iter)
    sides = {1: {}, 2: {}}
    for m, tf, side in term:
        q = (1.0 - u[m]) if side == 1 else u[m]
        fc = np.stack(np.unravel_index(tf, tuple(nf)), axis=1) // F + lo
        cflat = np.ravel_multi_index(fc.T, vox.shape)
        keys, inv = np.unique(cflat, return_inverse=True)
        sums = np.bincount(inv, weights=q)
        for k_, s_ in zip(keys, sums):
            sides[side][int(k_)] = sides[side].get(int(k_), 0.0) + float(s_)
    # what reaches the pocket: the main side also counts the residual of dead ends
    if swap:
        sides = {1: sides[2], 2: sides[1]}
    total = sum(sides[2].values())
    return sides[1], sides[2], total, it, res


def _sinkhorn_plan(X, a, Y, b, eps, n_iter=2000, tol=1e-9):
    """Entropic optimal transport plan between masses a (at X) and b (at Y), cost |x - y|^2,
    regularisation eps (m^2), in the log domain: P (n_a, n_b) with P 1 = a, P^T 1 = b."""
    C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    la, lb = np.log(a), np.log(b)
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    for _ in range(n_iter):
        M = (g[None, :] - C) / eps
        f = eps * (la - (M.max(axis=1) + np.log(np.exp(M - M.max(axis=1)[:, None]).sum(axis=1))))
        M = (f[:, None] - C) / eps
        g = eps * (lb - (M.max(axis=0) + np.log(np.exp(M - M.max(axis=0)[None, :]).sum(axis=0))))
        P = np.exp((f[:, None] + g[None, :] - C) / eps)
        if np.abs(P.sum(axis=1) - a).max() < tol:
            break
    return P


def pocket_links(scene, vox, pocket, reach, h_target, max_fine):
    """((cells a, cells b, w), stats) linking the main air to one pocket, or None if no channel.

    w is in FDTD face units (a whole face = 1). It is the channel's potential-flow conductance,
    extrapolated to h -> 0 from h and 2 h because the fine rule narrows each channel by about
    h. It is shared between the cells of both mouths by an entropic optimal transport of their
    fluxes, so each part of a mouth feeds the nearest part of the other."""
    dx = float(vox.dx)
    C = vox.solid & _dilate(vox.air, reach) & _dilate(pocket, reach)
    if not C.any():
        return None
    ijk = np.argwhere(C)
    lo = np.maximum(ijk.min(axis=0) - 1, 0)
    hi = np.minimum(ijk.max(axis=0) + 2, np.asarray(vox.shape))
    nc = hi - lo
    F = max(8, int(np.ceil(dx / h_target)))     # a gap of half a cell spans 4 fine cells
    F = F + (F % 2)
    while F > 4 and np.prod(nc * F, dtype=np.float64) > max_fine:
        F -= 2
    if np.prod(nc * F, dtype=np.float64) > max_fine:
        print("[Auralis] sub-cell openings: channel box %s cells too large for a fine grid of "
              "%.0f M cells, pocket of %.2f m3 left CLOSED"
              % (tuple(int(v) for v in nc), max_fine / 1e6, pocket.sum() * dx ** 3))
        return None
    t0 = time.time()
    qa, qb, G, it, res = channel_flux(scene, vox, pocket, lo, hi, F)
    if G / F < 1e-3 or not qa or not qb:
        return None                     # no gap (or one of less than a thousandth of a face)
    _, _, G2, it2, _ = channel_flux(scene, vox, pocket, lo, hi, F // 2)
    w_h, w_2h = G / F, G2 / (F // 2)
    w0 = float(np.clip(2.0 * w_h - w_2h, w_h, 1.5 * w_h))
    # diagnostic only (sensitivity of a result to the conductance): AURALIS_FDTD_SUBCELL_SCALE
    w0 *= float(os.environ.get("AURALIS_FDTD_SUBCELL_SCALE", "1"))
    ka = np.array(sorted(qa))
    kb = np.array(sorted(qb))
    a = np.array([qa[k] for k in ka])
    b = np.array([qb[k] for k in kb])
    keep_a, keep_b = a > 1e-9 * a.sum(), b > 1e-9 * b.sum()
    ka, a, kb, b = ka[keep_a], a[keep_a], kb[keep_b], b[keep_b]
    a, b = a / a.sum(), b / b.sum()
    if len(a) * len(b) > 4e6:
        print("[Auralis] sub-cell openings: %d x %d mouth cells, too many to pair; pocket of "
              "%.2f m3 left CLOSED" % (len(a), len(b), pocket.sum() * dx ** 3))
        return None
    org = np.asarray(vox.origin, dtype=np.float64)
    X = org + dx * np.stack(np.unravel_index(ka, vox.shape), axis=1)
    Y = org + dx * np.stack(np.unravel_index(kb, vox.shape), axis=1)
    P = _sinkhorn_plan(X, a, Y, b, eps=2.0 * dx * dx)
    P[P < 1e-6 * P.max()] = 0.0
    P *= 1.0 / P.sum()
    ia, ib = np.nonzero(P)
    w = w0 * P[ia, ib]
    stats = {"w": w0, "w_h": w_h, "w_2h": w_2h, "F": F, "cg_iter": (it, it2), "cg_res": res,
             "n_mouth": (len(ka), len(kb)), "n_links": len(w), "t_s": time.time() - t0,
             "flux_balance": abs(G - sum(qa.values())) / G}
    return (ka[ia].astype(np.int64), kb[ib].astype(np.int64), w), stats


_CACHE: dict = {}
_CACHE_JOINED: dict = {}    # joined airs (last two): a source on the other side reuses the links


def _cache_put(cache, key, value):
    if len(cache) >= 2:
        cache.pop(next(iter(cache)))
    cache[key] = value


def open_pockets(scene, vox, reach_m=0.6, h_target=0.006, max_fine=1.2e8, min_m3=None):
    """Open the enclosed pockets that the geometry joins to the air through sub-cell gaps.

    A gap narrower than ~1.7 dx (a door ajar, a slot under a door) through walls up to
    reach_m thick is closed by the voxelisation. The pocket is added to the air and joined by
    links carrying the channel's conductance. Returns (vox, links, stats). vox is a new
    VoxelGrid sharing the solid arrays, or the same object when nothing opens. links is
    (cells a, cells b, w) on flat coarse indices, or None. The links are symmetric, so a
    source in the pocket gets the same ones."""
    from .fdtd import VoxelGrid, _air_key, _geometry_key
    dx = float(vox.dx)
    reach = max(4, int(np.ceil(float(reach_m) / dx)))     # cells: the walls a gap may cross
    if min_m3 is None:
        # a pocket below 1 % of the air cannot change the room's response, and each one
        # costs minutes of fine grid (a hall can hold dozens under its seating)
        min_m3 = 0.01 * float(vox.air.sum()) * dx ** 3
    # no pocket threshold in the base key: it is 1 % of this side's air, so the other side of
    # the same gap would miss the joined-air cache. The joined air itself is compared below.
    base = (_geometry_key(scene), dx, tuple(np.round(vox.origin, 12)), vox.shape, int(reach),
            float(h_target), float(max_fine))

    def _new_vox(air):
        nv = VoxelGrid(solid=vox.solid, mat=vox.mat, origin=vox.origin, dx=vox.dx,
                       nrmq=vox.nrmq)
        nv.air = air.copy()
        return nv
    key = base + (float(min_m3), _air_key(vox.air))
    hit = _CACHE.get(key)
    if hit is not None:
        air, links, stats = hit
        if links is None:
            return vox, None, stats
        return _new_vox(air), tuple(a.copy() for a in links), stats
    pockets, n_small = find_pockets(vox, reach, min_m3)
    for fa, links, stats in _CACHE_JOINED.get(base, []):
        # the same joined air seen from a pocket: this air and the pockets it opened
        if (vox.air & ~fa).any():
            continue
        cand = vox.air.copy()
        for pk in pockets:
            if (pk & fa).any():
                cand |= pk
        if np.array_equal(cand, fa):
            print("[Auralis] sub-cell openings: the links of the same joined air (a source on "
                  "the other side) are reused")
            return _new_vox(fa), tuple(a.copy() for a in links), stats
    out = _open_pockets(scene, vox, reach, h_target, max_fine, pockets, n_small)
    _cache_put(_CACHE, key, (out[0].air.copy(),
                             None if out[1] is None else tuple(a.copy() for a in out[1]), out[2]))
    if out[1] is not None:
        kept = _CACHE_JOINED.get(base, [])[-1:]
        _CACHE_JOINED.clear()
        _CACHE_JOINED[base] = kept + [(out[0].air.copy(), tuple(a.copy() for a in out[1]),
                                       out[2])]
    return out


def _open_pockets(scene, vox, reach, h_target, max_fine, pockets, n_small):
    """open_pockets without its caches, on the pockets find_pockets gave."""
    from .fdtd import VoxelGrid
    dx = float(vox.dx)
    stats = {"n_pockets": len(pockets), "n_small": n_small, "opened": []}
    if not pockets:
        return vox, None, stats
    air = vox.air.copy()
    la, lb, lw = [], [], []
    for pk in pockets:
        out = pocket_links(scene, vox, pk, reach, h_target, max_fine)
        if out is None:
            continue
        (ca, cb, w), st = out
        st["m3"] = float(pk.sum()) * dx ** 3
        stats["opened"].append(st)
        air |= pk
        la.append(ca)
        lb.append(cb)
        lw.append(w)
    if not la:
        return vox, None, stats
    nv = VoxelGrid(solid=vox.solid, mat=vox.mat, origin=vox.origin, dx=vox.dx, nrmq=vox.nrmq)
    nv.air = air
    return nv, (np.concatenate(la), np.concatenate(lb), np.concatenate(lw)), stats
