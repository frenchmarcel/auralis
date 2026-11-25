"""Conformal donation tracked per axis with tangential face apertures (prototype, order 2,
rigid box): python tools/conformal_aniso_prototype.py [dx]. Not engine code."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import make_shoebox
from auralis.engine import conformal as cf
from auralis.engine.fdtd import voxelize, FdtdGrid

L = (4.0, 3.0, 2.5)
sc = make_shoebox(L, wall_material="concrete")
c = 343.0
dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0847
src, rcv = np.array([0.4, 0.4, 0.4]), np.array([3.6, 2.6, 2.1])
vox = voxelize(sc, dx, src, include_points=[src, rcv])
air = vox.air

def per_axis_excess():
    """conformal_corrections' donation, but the excess is kept per donation axis."""
    W_m2, idx, fstats, cell_area = cf.wall_face_areas(sc, vox)
    frac, F = cf.fine_air_fractions(sc, vox, factor=3)
    rad_f = 0.5 * np.sqrt(3.0) * dx / F
    frac = frac + np.where(air, 0.0, cell_area * rad_f / dx ** 3)
    shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    def sh(a, s, fill=False):
        out = np.full(a.shape, fill, dtype=a.dtype); sx, sy, sz = s
        src_ = a[max(0, -sx):a.shape[0] - max(0, sx) or None, max(0, -sy):a.shape[1] - max(0, sy) or None, max(0, -sz):a.shape[2] - max(0, sz) or None]
        out[max(0, sx):a.shape[0] - max(0, -sx) or None, max(0, sy):a.shape[1] - max(0, -sy) or None, max(0, sz):a.shape[2] - max(0, -sz) or None] = src_
        return out
    donors = (~air) & (frac > 1e-9)
    cnt_air = np.zeros(vox.shape, np.int8)
    for s in shifts: cnt_air += sh(air, s).astype(np.int8)
    direct = donors & (cnt_air > 0)
    deep = donors & (cnt_air == 0)
    # deep donors relay to their direct-donor neighbours (as the engine does); the slab
    # they represent is normal to the RELAY axis
    pool_ax = np.zeros((3,) + vox.shape)
    cnt_dir = np.zeros(vox.shape, np.int8)
    for s in shifts: cnt_dir += sh(direct, s).astype(np.int8)
    relay = deep & (cnt_dir > 0)
    give = np.where(relay, frac / np.maximum(cnt_dir, 1), 0.0)
    for s in shifts:
        ax = int(np.flatnonzero(s)[0])
        pool_ax[ax] += np.where(direct, sh(give, s), 0.0)
    own = np.where(direct, frac, 0.0)
    ex = np.zeros((3,) + vox.shape)
    for s in shifts:
        ax = int(np.flatnonzero(s)[0])
        ex[ax] += np.where(air, sh(own / np.maximum(cnt_air, 1), s), 0.0)
        for a2 in range(3):
            ex[a2] += np.where(air, sh(pool_ax[a2] / np.maximum(cnt_air, 1), s), 0.0)
    for a in range(3):                          # same conservative diffusion, per axis
        e = ex[a]
        for _ in range(8):
            d = np.zeros_like(e)
            for s in shifts:
                d += np.where(air & sh(air, s), sh(e, s) - e, 0.0)
            e = e + 0.15 * d
        ex[a] = e
    return ex

def run(mode, n_steps):
    phi = None
    ex = None
    if mode != "stair":
        ex = per_axis_excess()
        phi = 1.0 + ex.sum(axis=0)[air]
    g = FdtdGrid(vox, np.zeros(1), c, order=2, phi=phi)
    n = g.n_air
    nb = g.nb
    ghost = n
    lam2 = g.lam ** 2
    ph = g.phi
    # face weights (n, 6): 1 (isotropic) or the anisotropic aperture
    W = np.ones((n, 6))
    if mode == "aniso":
        ii = np.argwhere(air)
        perp = np.stack([ex[1] + ex[2], ex[0] + ex[2], ex[0] + ex[1]])   # E_perp per axis b
        Ecell = np.stack([perp[b][air] for b in range(3)], axis=1)        # (n, 3)
        for f in range(6):
            b = f // 2
            j = nb[:, f]
            ok = j < ghost
            W[ok, f] = 1.0 + 0.5 * (Ecell[ok, b] + Ecell[j[ok], b])
    isair = nb < ghost
    Wa = np.where(isair, W, 0.0)
    cc = 2.0 - lam2 * Wa.sum(axis=1) / ph
    cnb = lam2 * Wa / ph[:, None]
    si = int(np.argmin(np.linalg.norm(np.argwhere(air) * dx + vox.origin - src, axis=1)))
    ri = int(np.argmin(np.linalg.norm(np.argwhere(air) * dx + vox.origin - rcv, axis=1)))
    p0 = np.zeros(n + 1); p1 = np.zeros(n + 1)
    out = np.zeros(n_steps)
    nbc = np.where(isair, nb, ghost)
    for it in range(n_steps):
        p2 = np.zeros(n + 1)
        p2[:n] = cc * p1[:n] + (cnb * p1[nbc]).sum(axis=1) - p0[:n]
        if it == 0:
            p2[si] += 1.0
        out[it] = p2[ri]
        p0, p1 = p1, p2
    return out, g.dt, float(ph.sum() * dx ** 3)

f_true = np.array([c / (2 * L[0]), c / (2 * L[1]), c / (2 * L[2])])
for mode in ("stair", "conformal", "aniso"):
    t0 = time.time()
    g0 = FdtdGrid(vox, np.zeros(1), c, order=2)
    n_steps = int(1.2 / g0.dt)
    x, dt, vol = run(mode, n_steps)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 18)); f = np.fft.rfftfreq(1 << 18, dt)
    pk = []
    for ft in f_true:
        sel = (f > 0.85 * ft) & (f < 1.15 * ft); pk.append(f[sel][np.argmax(X[sel])])
    err = 100 * (np.array(pk) / f_true - 1)
    print("%-9s dx %.4f vol %.2f m3 | axial modes %s Hz, error %% %s  (%.0f s)" % (mode, dx, vol, np.round(pk, 2), np.round(err, 2), time.time() - t0), flush=True)
