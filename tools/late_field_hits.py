"""Diagnostic ray tracer (the engine's reflection rule, one band, no air): every wall hit
with its time, material, incidence and energy, to see how the late field meets each
surface: python tools/late_field_hits.py <manifest.json> [out.npz]"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine.project import build_scene
from auralis.engine.geometry import NumpyRaycaster

B = 3   # 1 kHz
def run(man_path, n_rays=6000, t_max=1.5, seed=1):
    man = json.load(open(man_path)); base = os.path.dirname(man_path)
    sc = build_scene(man, base)
    rc = NumpyRaycaster(sc)
    rng = np.random.default_rng(seed)
    alpha = sc.absorption[:, B].astype(float); scat = sc.scattering[:, B].astype(float)
    tri_area = sc.triangle_areas(); mat_area = np.bincount(sc.tri_mat, weights=tri_area, minlength=sc.n_mats)
    c = 343.0
    src = np.asarray(man["pairs"][0]["source"], float)
    n = n_rays
    z = 1 - 2 * rng.random(n); r = np.sqrt(1 - z * z); ph = 2 * np.pi * rng.random(n)
    d = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1); o = np.tile(src, (n, 1))
    E = np.ones(n); t = np.zeros(n); alive = np.ones(n, bool)
    rec = []    # (time, mat, cos, energy before)
    while alive.any():
        idx = np.where(alive)[0]
        tri, th = rc.cast_batch(o[idx], d[idx])
        miss = tri < 0
        alive[idx[miss]] = False
        idx, tri, th = idx[~miss], tri[~miss], th[~miss]
        t[idx] += th / c
        over = t[idx] > t_max
        alive[idx[over]] = False
        idx, tri, th = idx[~over], tri[~over], th[~over]
        if idx.size == 0: break
        hp = o[idx] + d[idx] * th[:, None]
        nrm = rc.normals[tri].copy(); flip = (nrm * d[idx]).sum(1) > 0; nrm[flip] *= -1
        m = rc.tri_mat[tri]
        cos = np.abs((nrm * d[idx]).sum(1))
        rec.append(np.stack([t[idx], m, cos, E[idx]], 1))
        E[idx] *= 1 - alpha[m]
        go = rng.random(idx.size) < scat[m]
        spec = d[idx] - 2 * (d[idx] * nrm).sum(1)[:, None] * nrm
        u1, u2 = rng.random(idx.size), rng.random(idx.size)
        rr = np.sqrt(u1); a = np.where(np.abs(nrm[:, :1]) > 0.9, [[0, 1, 0]], [[1, 0, 0]])
        tt = np.cross(a, nrm); tt /= np.linalg.norm(tt, axis=1, keepdims=True); bb = np.cross(nrm, tt)
        diff = tt * (rr * np.cos(2 * np.pi * u2))[:, None] + bb * (rr * np.sin(2 * np.pi * u2))[:, None] + nrm * np.sqrt(1 - u1)[:, None]
        nd = np.where(go[:, None], diff, spec); d[idx] = nd / np.linalg.norm(nd, axis=1, keepdims=True)
        o[idx] = hp + nrm * 1e-4
    rec = np.concatenate(rec)
    return sc, rec, mat_area, alpha

def report(rec, area, alpha, names, n_rays):
    """Late-field share of the hits per material against its area share, and the decay."""
    t, m, E = rec[:, 0], rec[:, 1].astype(int), rec[:, 3]
    S = area.sum()
    abar = (area * alpha).sum() / S
    k = t >= 0.3
    for i, n in enumerate(names):
        print("  %-26s late hits %5.1f %%  area %5.1f %%  alpha %.2f"
              % (n, 100 * E[k & (m == i)].sum() / E[k].sum(), 100 * area[i] / S, alpha[i]))
    bins = np.arange(0.3, 1.4, 0.01)
    h, _ = np.histogram(t, bins=bins, weights=E)
    slope = np.polyfit(bins[:-1], 10 * np.log10(h), 1)[0]
    mfp = 343.0 / (((t > 0.3) & (t < 1.4)).sum() / n_rays / 1.1)
    print("  energy-weighted alpha per late hit %.3f (area-weighted %.3f); late mean free path "
          "%.2f m; decay T %.2f s" % ((E[k] * alpha[m[k]]).sum() / E[k].sum(), abar, mfp, -60 / slope))


if __name__ == "__main__":
    t0 = time.time()
    sc, rec, area, alpha = run(os.path.abspath(sys.argv[1]))
    if len(sys.argv) > 2:
        np.savez(sys.argv[2], rec=rec, area=area, alpha=alpha, names=np.array(sc.mat_names))
    print("%d hits, %.0f s" % (len(rec), time.time() - t0))
    report(rec, area, alpha, sc.mat_names, 6000)
