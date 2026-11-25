import ctypes as ct
import gc
import time

import numpy as np

BUILD_HALO = 12      # planes around a slab for its sub grid: conformal donation, diffusion
COPY_HALO = 2        # planes exchanged with the neighbours each step (order 4)


def _plane_cuts(counts, n_dev, npml):
    """x planes [0, p1, ..., nx) of n_dev slabs with about as many air cells, kept 3 planes
    clear of the x CPML and at least 3 planes thick (fdtd.slab_cuts on plane counts)."""
    nx = len(counts)
    start = np.concatenate([[0], np.cumsum(counts)])
    total = start[-1]
    lo, hi = npml + 3, nx - npml - 3
    planes = [0]
    for k in range(1, n_dev):
        first = max(lo, planes[-1] + 3)
        last = hi - 3 * (n_dev - 1 - k)
        if first > last:
            raise ValueError("%d slabs do not fit in %d x planes" % (n_dev, nx))
        cand = np.arange(first, last + 1)
        planes.append(int(cand[np.argmin(np.abs(start[cand] - total * k / n_dev))]))
    planes.append(nx)
    if nx - planes[-2] < 3:
        raise ValueError("last slab thinner than 3 planes")
    return planes, start


def slab_boxes(air, n_dev, npml):
    """[(sa, sb)]: the x planes of every slab's sub box (its owned planes and BUILD_HALO
    around), as SlabGrid builds them."""
    planes, _ = _plane_cuts(air.sum(axis=(1, 2)).astype(np.int64), n_dev, npml)
    nx = air.shape[0]
    return [(max(0, planes[k] - BUILD_HALO), min(nx, planes[k + 1] + BUILD_HALO))
            for k in range(n_dev)]


def native_arrays(g):
    """The arrays of FdtdGrid g as run_grid_native hands them to the library (same dtypes)."""
    f32 = np.float32
    a = {"nb": np.ascontiguousarray(g.nb, dtype=np.int32),
         "nb2": np.ascontiguousarray(g.nb2, dtype=np.int32),
         "m4": np.ascontiguousarray(g.is4.astype(np.uint8)),
         "cn": np.ascontiguousarray(g.c_now, dtype=f32),
         "co": np.ascontiguousarray(g.c_old, dtype=f32),
         "cb": np.ascontiguousarray(g.c_nb, dtype=f32)}
    cd = getattr(g, "c_drv", None)
    a["cd"] = np.ascontiguousarray(np.zeros(g.n_air, f32) if cd is None else cd, dtype=f32)
    nf = int(getattr(g, "n_faces", 0))
    if nf:
        a.update(fc=np.ascontiguousarray(g.f_cell, dtype=np.int32),
                 fe=np.ascontiguousarray(g.f_e, dtype=f32),
                 fg=np.ascontiguousarray(g.f_g, dtype=f32),
                 fl=np.ascontiguousarray(g.f_l, dtype=f32),
                 fA=np.ascontiguousarray(g.f_Asum, dtype=f32))
    else:
        a.update(fc=np.zeros(1, np.int32), fe=np.zeros(2, f32), fg=np.zeros(2, f32),
                 fl=np.zeros(2, f32), fA=np.zeros(1, f32))
    nr = int(getattr(g, "n_rfaces", 0))
    if nr:
        a.update(rc=np.ascontiguousarray(g.r_cell, dtype=np.int32),
                 rA=np.ascontiguousarray(g.r_A, dtype=f32),
                 rc1=np.ascontiguousarray(g.r_c1, dtype=f32),
                 rc2=np.ascontiguousarray(g.r_c2, dtype=f32),
                 rE=np.ascontiguousarray(g.r_E.reshape(-1), dtype=f32),
                 rG=np.ascontiguousarray(g.r_G.reshape(-1), dtype=f32))
    else:
        a.update(rc=np.zeros(1, np.int32), rA=np.zeros(1, f32), rc1=np.zeros(1, f32),
                 rc2=np.zeros(1, f32), rE=np.zeros(4, f32), rG=np.zeros(2, f32))
    axes = list(getattr(g, "pml_axes", []) or [])
    cnt = np.zeros(3, dtype=np.int64)
    for k, ax in enumerate(axes):
        cnt[k] = len(ax["cells"])

    def cat(key, dt):
        if not axes:
            return np.zeros(1, dt)
        return np.ascontiguousarray(np.concatenate([np.asarray(ax[key]) for ax in axes]),
                                    dtype=dt)
    a.update(pcnt=cnt, pc=cat("cells", np.int32), pb=cat("b", f32), pa=cat("a", f32),
             pap=cat("a_psi", f32), pnm=cat("nbm", np.int32), pnp=cat("nbp", np.int32),
             plm=cat("locm", np.int32))
    nl = int(getattr(g, "n_links", 0))
    if nl:
        a.update(lr=np.ascontiguousarray(g.l_rows, dtype=np.int32),
                 lp=np.ascontiguousarray(g.l_ptr, dtype=np.int64),
                 ln=np.ascontiguousarray(g.l_nb, dtype=np.int32),
                 lk=np.ascontiguousarray(g.l_k, dtype=f32))
    else:
        a.update(lr=np.zeros(1, np.int32), lp=np.zeros(1, np.int64),
                 ln=np.zeros(1, np.int32), lk=np.zeros(1, f32))
    a["n_faces"], a["n_rfaces"], a["n_links"] = nf, nr, nl
    a["n_lrows"] = len(a["lr"]) if nl else 0           # rows of the CSR, as run_grid_native
    return a


def _check_copy(a, o0, o1, hl, hr, n_sub, k):
    """Check that every reference of the owned cells [o0, o1) stays in the copy [hl, hr).

    The ghost n_sub is allowed. The library would refuse such a slab (code 42) without
    saying where, so the offending tables are named here."""
    def out(x):
        # int32 as stored (a slab holds < 2^31 cells): no 8-byte copy of tens of GB
        x = np.asarray(x)
        return int(np.count_nonzero((x != n_sub) & ((x < hl) | (x >= hr))))
    bad = {"nb": out(a["nb"].reshape(-1, 6)[o0:o1]), "nb2": out(a["nb2"].reshape(-1, 6)[o0:o1])}
    if a["n_links"]:
        rows = a["lr"]
        m = (rows >= o0) & (rows < o1)
        ptr = a["lp"]
        owned_rows = np.nonzero(m)[0]
        if len(owned_rows):
            q0, q1 = ptr[owned_rows[0]], ptr[owned_rows[-1] + 1]
            bad["links"] = out(a["ln"][q0:q1])
    cnt = a["pcnt"]
    off = np.concatenate([[0], np.cumsum(cnt)])
    for ax in range(3):
        cells = a["pc"][off[ax]:off[ax + 1]]
        m = (cells >= o0) & (cells < o1)
        if m.any():
            bad["pml%d" % ax] = out(a["pnm"][off[ax]:off[ax + 1]][m]) + out(
                a["pnp"][off[ax]:off[ax + 1]][m])
    bad = {key: int(v) for key, v in bad.items() if v}
    if bad:
        raise RuntimeError("slab %d: references outside its copy [%d, %d) of %d cells: %s"
                           % (k, hl, hr, n_sub, bad))


def _bind(lib):
    P = ct.POINTER
    I32, I64, F32, U8 = P(ct.c_int32), P(ct.c_int64), P(ct.c_float), P(ct.c_uint8)
    lib.aur_fdtd_slabs_create.restype = ct.c_void_p
    lib.aur_fdtd_slabs_create.argtypes = [ct.c_int32, I32, I64, I64, ct.c_int64, ct.c_int32,
                                          ct.c_int32, ct.c_float, ct.c_float, ct.c_float,
                                          ct.c_float, I32]
    lib.aur_fdtd_slabs_add.restype = ct.c_int32
    lib.aur_fdtd_slabs_add.argtypes = ([ct.c_void_p, ct.c_int32, ct.c_int64, ct.c_int64, I32,
                                        I32, U8, F32, F32, F32, F32, ct.c_int64, I32, F32,
                                        F32, F32, F32, ct.c_int64, I32, F32, F32, F32, F32,
                                        F32, I64, I32, F32, F32, F32, I32, I32, I32,
                                        ct.c_int64, I32, I64, I32, F32])
    lib.aur_fdtd_session6m_run.restype = ct.c_int32
    lib.aur_fdtd_session6m_run.argtypes = [ct.c_void_p, F32, ct.c_int32, I64, F32, ct.c_int32,
                                           I64, F32, ct.c_int32, ct.c_int32, F32]
    lib.aur_fdtd_session4m_destroy.restype = None
    lib.aur_fdtd_session4m_destroy.argtypes = [ct.c_void_p]


def available(lib):
    return lib is not None and hasattr(lib, "aur_fdtd_slabs_create")


class SlabGrid:
    """The wave grid of a scene built slab by slab, for scenes beyond 2^31 air cells.

    Each slab is its own FdtdGrid on a box of whole x planes (its owned planes and BUILD_HALO
    around), uploaded to its device and freed, so the whole-scene grid never exists on the
    host. Indices are virtual int64: the slabs' owned cells end to end. The owned cells hold
    the same arrays as in a grid of the whole scene. build(sub_vox, pml) returns the FdtdGrid
    of a sub box. points are the only positions taps() will accept (source, receivers,
    dipole ends)."""
    wide_index = True

    def __init__(self, vox, build, lib, devs, pml, order, points):
        from .fdtd import VoxelGrid
        _bind(lib)
        self.lib, self.devs = lib, list(devs)
        self.dx = float(vox.dx)
        self.order = 4 if int(order) == 4 else 2
        self.handle = None
        npml = int(pml.get("n", 0)) if pml else 0
        counts = vox.air.sum(axis=(1, 2)).astype(np.int64)
        K = len(self.devs)
        planes, start = _plane_cuts(counts, K, npml)
        nx = len(counts)
        self.n_air = int(start[-1])
        cuts = np.array([start[p] for p in planes], dtype=np.int64)
        halo = np.empty(2 * K, dtype=np.int64)
        for k in range(K):
            halo[2 * k] = start[max(planes[k] - COPY_HALO, 0)] if k > 0 else cuts[k]
            halo[2 * k + 1] = (start[min(planes[k + 1] + COPY_HALO, nx)] if k < K - 1
                               else cuts[k + 1])
        pts = [np.asarray(p, dtype=np.float64) for p in points]
        base = [int(np.floor((p[0] - vox.origin[0]) / self.dx)) for p in pts]
        self._taps, self._staps = {}, {}
        t0 = time.time()
        for k in range(K):
            xa, xb = planes[k], planes[k + 1]
            sa, sb = max(0, xa - BUILD_HALO), min(nx, xb + BUILD_HALO)
            sub = VoxelGrid(solid=vox.solid[sa:sb], mat=vox.mat[sa:sb],
                            origin=vox.origin + np.array([sa * self.dx, 0.0, 0.0]), dx=self.dx,
                            nrmq=None if vox.nrmq is None else vox.nrmq[sa:sb])
            sub.air = vox.air[sa:sb]
            psub = None if not pml else dict(pml, offset=(sa, 0, 0), shape=tuple(vox.shape))
            t_b = time.time()
            g = build(sub, psub)
            t_b = time.time() - t_b
            sx = np.asarray(g.cells[:, 0], dtype=np.int64)
            o0 = int(np.searchsorted(sx, xa - sa))
            o1 = int(np.searchsorted(sx, xb - sa))
            if o1 - o0 != cuts[k + 1] - cuts[k]:
                raise RuntimeError("slab %d: %d owned cells, %d expected"
                                   % (k, o1 - o0, cuts[k + 1] - cuts[k]))
            shift = int(cuts[k]) - o0
            if k == 0:
                self.dt = float(g.dt)
                st = ct.c_int32(0)
                dv = np.asarray(self.devs, dtype=np.int32)
                self.handle = lib.aur_fdtd_slabs_create(
                    K, dv.ctypes.data_as(ct.POINTER(ct.c_int32)),
                    cuts.ctypes.data_as(ct.POINTER(ct.c_int64)),
                    halo.ctypes.data_as(ct.POINTER(ct.c_int64)), ct.c_int64(self.n_air), 1, 1,
                    ct.c_float(g.c4_c), ct.c_float(g.c4_nb), ct.c_float(g.c4_nb2),
                    ct.c_float(g.lam2), ct.byref(st))
                if not self.handle:
                    raise RuntimeError("aur_fdtd_slabs_create failed (code %d)" % st.value)
            for q, p in enumerate(pts):
                if xa <= base[q] < xb or (k == 0 and base[q] < 0) or (k == K - 1 and base[q] >= nx):
                    try:
                        i, w = g.taps(p)
                        self._taps[q] = (np.asarray(i, dtype=np.int64) + shift, w)
                        self._staps[q] = (self._taps[q][0], w / g.phi[i])
                    except ValueError as exc:
                        self._taps[q] = self._staps[q] = exc
            t_a = time.time()
            a = native_arrays(g)
            n_sub = int(g.n_air)
            hl_s = int(halo[2 * k]) - shift
            hr_s = int(halo[2 * k + 1]) - shift
            _check_copy(a, o0, o1, hl_s, hr_s, n_sub, k)
            del g
            t_a = time.time() - t_a
            t_u = time.time()
            P = ct.POINTER

            def ptr(x, t):
                return x.ctypes.data_as(P(t))
            I32, F32 = ct.c_int32, ct.c_float
            rc = lib.aur_fdtd_slabs_add(
                self.handle, k, ct.c_int64(shift), ct.c_int64(len(a["cn"])),
                ptr(a["nb"], I32), ptr(a["nb2"], I32), ptr(a["m4"], ct.c_uint8),
                ptr(a["cn"], F32), ptr(a["co"], F32), ptr(a["cb"], F32), ptr(a["cd"], F32),
                ct.c_int64(a["n_faces"]), ptr(a["fc"], I32), ptr(a["fe"], F32),
                ptr(a["fg"], F32), ptr(a["fl"], F32), ptr(a["fA"], F32),
                ct.c_int64(a["n_rfaces"]), ptr(a["rc"], I32), ptr(a["rA"], F32),
                ptr(a["rc1"], F32), ptr(a["rc2"], F32), ptr(a["rE"], F32), ptr(a["rG"], F32),
                ptr(a["pcnt"], ct.c_int64), ptr(a["pc"], I32), ptr(a["pb"], F32),
                ptr(a["pa"], F32), ptr(a["pap"], F32), ptr(a["pnm"], I32), ptr(a["pnp"], I32),
                ptr(a["plm"], I32), ct.c_int64(a["n_lrows"]), ptr(a["lr"], I32),
                ptr(a["lp"], ct.c_int64), ptr(a["ln"], I32), ptr(a["lk"], F32))
            del a
            gc.collect()
            if rc != 0:
                self.close()
                raise RuntimeError("aur_fdtd_slabs_add(%d) failed (code %d%s)" % (
                    k, rc, ": a reference leaves the slab's copy" if rc == 42 else ""))
            print("[Auralis] FDTD slab %d/%d: x planes %d-%d, %d air cells, device %d (%.0f s: "
                  "grid %.0f s, arrays %.0f s, fill and upload %.0f s)"
                  % (k + 1, K, xa, xb, cuts[k + 1] - cuts[k], self.devs[k], time.time() - t0,
                     t_b, t_a, time.time() - t_u), flush=True)
        self._pts = pts

    def taps(self, pos):
        return self._prepared(self._taps, pos)

    def source_taps(self, pos):
        """FdtdGrid.source_taps of the slab that holds the point."""
        return self._prepared(self._staps, pos)

    def _prepared(self, table, pos):
        p = np.asarray(pos, dtype=np.float64)
        for q, r in enumerate(self._pts):
            if np.array_equal(r, p):
                t = table.get(q)
                if t is None:
                    break
                if isinstance(t, Exception):
                    raise t
                return t
        raise ValueError("position not prepared for the slab grid: %s" % (p,))

    def runner(self, _grid, pulse, src_taps, rec_taps, n_steps):
        """run_grid_native's contract: (n_steps, n_rec) float64."""
        si = np.ascontiguousarray(src_taps[0], dtype=np.int64)
        sw = np.ascontiguousarray(src_taps[1], dtype=np.float32)
        recs = [rec_taps] if isinstance(rec_taps, tuple) else list(rec_taps)
        ri = np.full((len(recs), 8), self.n_air, dtype=np.int64)
        rw = np.zeros((len(recs), 8), dtype=np.float32)
        for r, (i, w) in enumerate(recs):
            ri[r, :len(i)] = i
            rw[r, :len(w)] = w
        out = np.zeros((int(n_steps), len(recs)), dtype=np.float32)
        pl = np.ascontiguousarray(pulse, dtype=np.float32)
        P = ct.POINTER
        rc = self.lib.aur_fdtd_session6m_run(
            self.handle, pl.ctypes.data_as(P(ct.c_float)), ct.c_int32(min(len(pl), int(n_steps))),
            si.ctypes.data_as(P(ct.c_int64)), sw.ctypes.data_as(P(ct.c_float)), len(si),
            ri.ctypes.data_as(P(ct.c_int64)), rw.ctypes.data_as(P(ct.c_float)), len(recs),
            int(n_steps), out.ctypes.data_as(P(ct.c_float)))
        if rc != 0:
            raise RuntimeError("aur_fdtd_session6m_run failed (code %d)" % rc)
        return out.astype(np.float64)

    def close(self):
        if self.handle:
            self.lib.aur_fdtd_session4m_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
