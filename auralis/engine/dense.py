import ctypes as ct

import numpy as np

_REF: dict = {}


def _ref_grid(order, c, pml, g_border):
    """A small empty FdtdGrid with the same order, CPML and border: its coefficients and CPML
    values are the ones every cell of a large empty box takes (they depend on the count of
    box faces a cell touches and on the depth in the CPML only)."""
    from .fdtd import FdtdGrid, VoxelGrid
    npml = int(pml.get("n", 0)) if pml else 0
    S = max(2 * npml + 8, 12)
    key = (int(order), float(c), repr(pml), float(g_border), S)
    hit = _REF.get(key)
    if hit is not None:
        return hit
    shp = (S, S, S)
    v = VoxelGrid(solid=np.zeros(shp, bool), mat=np.full(shp, -1, np.int32),
                  origin=np.zeros(3), dx=1.0)
    v.air = ~v.solid
    g = FdtdGrid(v, np.zeros(1), c, sigma=None, gamma_border=g_border, order=order, pml=pml)
    _REF.clear()
    _REF[key] = (g, S, npml)
    return _REF[key]


def _core_npml(order, npml, shape):
    """The kernel's npml for its order-4 test: the CPML's, or one that leaves no core (order 2)."""
    return int(npml) if int(order) == 4 else int(max(shape))


def _order4_rule(ii, shape, npml):
    """The dense kernel's order-4 cells: two cells or more from the faces, outside the CPML
    and its collar (npml: _core_npml)."""
    n = np.asarray(shape)
    d = np.minimum(ii, n - 1 - ii).min(axis=1)
    ok = d >= 2
    if npml > 0:
        ok &= ((ii >= npml + 1) & (ii <= n - npml - 2)).all(axis=1)
    return ok


def box_tables(order, c, pml, g_border):
    """(c4c, c4nb, c4nb2, lam2, coef (12,) float32, {axis: {depth key: (b, a, a_psi)}}) of an
    empty box, read on the reference grid; ValueError if that grid does not follow the dense
    kernel's rules (then the indexed empty grid is used)."""
    g, S, npml = _ref_grid(order, c, pml, g_border)
    ii = np.asarray(g.cells, dtype=np.int64)
    if not np.array_equal(_order4_rule(ii, (S, S, S), _core_npml(order, npml, (S, S, S))),
                          np.asarray(g.is4, bool)):
        raise ValueError("order-4 cells of the reference box differ from the dense rule")
    m = ((ii == 0) | (ii == S - 1)).sum(axis=1)
    cn = np.asarray(g.c_now, dtype=np.float32)
    cb = np.asarray(g.c_nb, dtype=np.float32)
    co = np.asarray(g.c_old, dtype=np.float32)
    coef = np.zeros(12, dtype=np.float32)
    non4 = ~np.asarray(g.is4, bool)
    for mm in range(4):
        sel = non4 & (m == mm)
        trip = np.unique(np.stack([cn[sel], cb[sel], co[sel]], axis=1), axis=0)
        if len(trip) != 1:
            raise ValueError("order-2 coefficients of the reference box are not one per face count")
        coef[3 * mm:3 * mm + 3] = trip[0]
    axes = {}
    if npml > 0:
        if len(g.pml_axes) != 3:
            raise ValueError("the reference box has CPML on %d axes" % len(g.pml_axes))
        for a, ax in enumerate(g.pml_axes):
            ia = ii[np.asarray(ax["cells"]), a]
            key = np.where(ia < S // 2, ia, ia - S)
            vals = {}
            for k_, b_, a_, p_ in zip(key, np.float32(ax["b"]), np.float32(ax["a"]),
                                      np.float32(ax["a_psi"])):
                v_ = (b_, a_, p_)
                if vals.setdefault(int(k_), v_) != v_:
                    raise ValueError("CPML values of the reference box vary at one depth")
            axes[a] = vals
    return (np.float32(g.c4_c), np.float32(g.c4_nb), np.float32(g.c4_nb2), np.float32(g.lam2),
            coef, axes)


class DenseBox:
    """An empty box of `shape` cells at origin/dx, the dense kernel's layout (two zero cells of
    padding on every side); taps() as FdtdGrid.taps, on padded flat indices."""

    def __init__(self, shape, origin, dx):
        self.shape = tuple(int(v) for v in shape)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.dx = float(dx)
        nx, ny, nz = self.shape
        self.PY, self.PZ = ny + 4, nz + 4

    def flat(self, ijk):
        ijk = np.asarray(ijk, dtype=np.int64)
        return ((ijk[..., 0] + 2) * self.PY + (ijk[..., 1] + 2)) * self.PZ + (ijk[..., 2] + 2)

    def taps(self, pos):
        f = (np.asarray(pos, float) - self.origin) / self.dx
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
                    if all(0 <= j[k] < self.shape[k] for k in range(3)) and w > 0:
                        out_i.append(int(self.flat(j)))
                        out_w.append(w)
        if not out_i:
            raise ValueError("position outside the box: %s" % (pos,))
        w = np.asarray(out_w)
        return np.asarray(out_i, dtype=np.int64), w / w.sum()

    def pml_tables(self, npml, axes):
        """The CPML entries of FdtdGrid on this box, axis by axis in ascending cell order:
        (counts (3,), cells, nbm, nbp, locm (int64), b, a, a_psi (float32))."""
        n = np.asarray(self.shape)
        strides = (self.PY * self.PZ, self.PZ, 1)
        out = {k: [] for k in ("cells", "nbm", "nbp", "locm", "b", "a", "a_psi")}
        counts = np.zeros(3, dtype=np.int64)
        for a in range(3):
            if npml <= 0:
                continue
            rng = []
            for b in range(3):
                inner = np.arange(1, n[b] - 1, dtype=np.int64)
                if b == a:
                    inner = inner[(inner <= npml - 2) | (inner >= n[b] - npml + 1)]
                rng.append(inner)
            I, J, K = np.meshgrid(*rng, indexing="ij")
            ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)
            del I, J, K
            ia = ijk[:, a]
            key = np.where(ia < n[a] // 2, ia, ia - n[a])
            cells = self.flat(ijk)
            del ijk
            table = axes[a]
            keys = np.array(sorted(table))
            vb = np.array([table[k][0] for k in keys], dtype=np.float32)
            va = np.array([table[k][1] for k in keys], dtype=np.float32)
            vp = np.array([table[k][2] for k in keys], dtype=np.float32)
            pos = np.searchsorted(keys, key)
            if np.any(pos >= len(keys)) or np.any(keys[np.minimum(pos, len(keys) - 1)] != key):
                raise ValueError("a CPML depth of the box is missing from the reference")
            nm = cells - strides[a]
            k_m = np.minimum(np.searchsorted(cells, nm), len(cells) - 1)
            locm = np.where(cells[k_m] == nm, k_m, -1).astype(np.int64)
            out["cells"].append(cells)
            out["nbm"].append(nm)
            out["nbp"].append(cells + strides[a])
            out["locm"].append(locm)
            out["b"].append(vb[pos])
            out["a"].append(va[pos])
            out["a_psi"].append(vp[pos])
            counts[a] = len(cells)
        cat = {k: (np.ascontiguousarray(np.concatenate(v)) if v else
                   np.zeros(1, np.int64 if k in ("cells", "nbm", "nbp", "locm") else np.float32))
               for k, v in out.items()}
        return counts, cat


def available(lib):
    return lib is not None and hasattr(lib, "aur_fdtd_dense_empty")


def dense_cuts(nx, n_dev, npml):
    """x planes [0, ..., nx] of n_dev slabs of nearly equal width, kept 3 planes clear of the
    x CPML (an x entry's minus neighbour stays in its slab), at least 2 planes each."""
    lo, hi = npml + 3, nx - npml - 3
    cuts = [0]
    for k in range(1, n_dev):
        x = int(round(nx * k / n_dev))
        x = min(max(x, lo, cuts[-1] + 2), hi)
        cuts.append(x)
    cuts.append(nx)
    if any(b - a < 2 for a, b in zip(cuts, cuts[1:])):
        raise ValueError("%d dense slabs do not fit in %d x planes" % (n_dev, nx))
    return np.asarray(cuts, dtype=np.int64)


def run_empty(lib, shape, origin, dx, c, order, pml, g_border, pulse, src_taps, rec_taps,
              n_steps, devs=None):
    """Receiver pressures (n_steps, n_rec) float64 of the empty box, as run_grid_native on the
    empty FdtdGrid of the same box would give them (the same floats). src_taps, rec_taps:
    from DenseBox.taps (padded flat indices). devs: CUDA devices of a split run (whole x planes
    each, aur_fdtd_dense_empty_m), None or one: a single device."""
    c4c, c4nb, c4nb2, lam2, coef, axes = box_tables(order, c, pml, g_border)
    box = DenseBox(shape, origin, dx)
    npml = int(pml.get("n", 0)) if pml else 0
    counts, T = box.pml_tables(npml, axes)
    si = np.ascontiguousarray(src_taps[0], dtype=np.int64)
    sw = np.ascontiguousarray(src_taps[1], dtype=np.float32)
    recs = [rec_taps] if isinstance(rec_taps, tuple) else list(rec_taps)
    ri = np.zeros((len(recs), 8), dtype=np.int64)          # padded cell 0: always 0
    rw = np.zeros((len(recs), 8), dtype=np.float32)
    for r, (i, w) in enumerate(recs):
        ri[r, :len(i)] = i
        rw[r, :len(w)] = w
    out = np.zeros((int(n_steps), len(recs)), dtype=np.float32)
    pulse = np.ascontiguousarray(pulse, dtype=np.float32)
    P = ct.POINTER
    I64, F32 = P(ct.c_int64), P(ct.c_float)
    multi = devs is not None and len(devs) > 1 and hasattr(lib, "aur_fdtd_dense_empty_m")
    if multi:
        fn = lib.aur_fdtd_dense_empty_m
        dv = np.asarray(devs, dtype=np.int32)
        cuts = dense_cuts(int(shape[0]), len(devs), npml)
        head_t = [ct.c_int32, P(ct.c_int32), I64]
        head = [ct.c_int32(len(dv)), dv.ctypes.data_as(P(ct.c_int32)), cuts.ctypes.data_as(I64)]
    else:
        fn = lib.aur_fdtd_dense_empty
        head_t, head = [], []
    fn.restype = ct.c_int32
    fn.argtypes = head_t + ([ct.c_int64] * 4 + [ct.c_float] * 3 + [F32, ct.c_float, I64]
                            + [I64] * 4 + [F32] * 3 + [F32, ct.c_int32, I64, F32, ct.c_int32,
                                                       I64, F32, ct.c_int32, ct.c_int32, F32])
    rc = fn(*head, *[ct.c_int64(v) for v in shape], ct.c_int64(_core_npml(order, npml, shape)),
            ct.c_float(c4c),
            ct.c_float(c4nb), ct.c_float(c4nb2), coef.ctypes.data_as(F32), ct.c_float(lam2),
            counts.ctypes.data_as(I64), T["cells"].ctypes.data_as(I64),
            T["nbm"].ctypes.data_as(I64), T["nbp"].ctypes.data_as(I64),
            T["locm"].ctypes.data_as(I64), T["b"].ctypes.data_as(F32),
            T["a"].ctypes.data_as(F32), T["a_psi"].ctypes.data_as(F32),
            pulse.ctypes.data_as(F32), ct.c_int32(min(len(pulse), int(n_steps))),
            si.ctypes.data_as(I64), sw.ctypes.data_as(F32), ct.c_int32(len(si)),
            ri.ctypes.data_as(I64), rw.ctypes.data_as(F32), ct.c_int32(len(recs)),
            ct.c_int32(int(n_steps)), out.ctypes.data_as(F32))
    if rc != 0:
        raise RuntimeError("aur_fdtd_dense_empty%s failed (code %d)" % ("_m" if multi else "", rc))
    return out.astype(np.float64)


def dipole_taps(box, pos, axis, h):
    """fdtd._dipole_taps on a DenseBox (int64 indices)."""
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    p = np.asarray(pos, float)
    i1, w1 = box.taps(p + 0.5 * h * u)
    i2, w2 = box.taps(p - 0.5 * h * u)
    idx, inv = np.unique(np.concatenate([i1, i2]), return_inverse=True)
    w = np.bincount(inv, weights=np.concatenate([w1, -w2]))
    keep = np.abs(w) > 1e-12
    return idx[keep].astype(np.int64), w[keep]
