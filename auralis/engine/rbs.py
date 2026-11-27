# -*- coding: utf-8 -*-
"""Reflection-based scattering: Christensen & Rindel, Forum Acusticum 2005."""
from __future__ import annotations

import numpy as np


def s_diffraction(f, l, w, cos_theta, a_star, c):
    """Scattering due to the finite size of a surface l x w (w <= l), eqs. (2)-(5):
    s_d = 1 - K_w K_l, K = min(1, f / f_lim), f_w = c a* / (2 (w cos theta)^2),
    f_l = c a* / (2 l^2). Broadcasts over all arguments."""
    ws = np.maximum(np.asarray(w, dtype=np.float64) * np.abs(cos_theta), 1e-3)
    ls = np.maximum(np.asarray(l, dtype=np.float64), 1e-3)
    a = np.maximum(np.asarray(a_star, dtype=np.float64), 1e-6)
    k_w = np.minimum(1.0, f / (c * a / (2.0 * ws * ws)))
    k_l = np.minimum(1.0, f / (c * a / (2.0 * ls * ls)))
    return 1.0 - k_w * k_l


def a_star(d_inc, d_refl):
    """Characteristic distance a* = d_inc d_refl / (2 (d_inc + d_refl)), eq. (4)."""
    d_inc = np.asarray(d_inc, dtype=np.float64)
    d_refl = np.asarray(d_refl, dtype=np.float64)
    return d_inc * d_refl / np.maximum(2.0 * (d_inc + d_refl), 1e-12)


def combine(s_d, s_s):
    """Reflection-based scattering coefficient, eq. (1): 1 - (1 - s_d)(1 - s_s)."""
    return 1.0 - (1.0 - s_d) * (1.0 - s_s)


def weight_image_sources(events, scene, dims, band_hz, c, src, rcv):
    """Scale each image source by sqrt(prod_k (1 - s_r,k)) per band: its SPECULAR share
    after every reflection. s_r comes from that reflection's surface, its incidence
    (incoming leg vs the triangle normal) and its image distances (path length before
    it, and after it to the receiver). The rays carry the scattered share, so no
    energy is counted twice."""
    tri_l, tri_w = dims
    f = np.asarray(band_hz, dtype=np.float64)
    tv = np.asarray(scene.tri_verts, dtype=np.float64)
    ng = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ng /= np.maximum(np.linalg.norm(ng, axis=1), 1e-12)[:, None]
    for ev in events:
        pts = getattr(ev, "points", None)
        if ev.kind != "specular" or not pts:
            continue
        chain = [np.asarray(src, float)] + [np.asarray(p, float) for p in pts] + \
            [np.asarray(rcv, float)]
        legs = np.linalg.norm(np.diff(np.array(chain), axis=0), axis=1)
        before = np.cumsum(legs)[:-1]            # source -> each reflection point
        total = float(legs.sum())
        spec = np.ones(len(f))
        for k, p in enumerate(chain[1:-1]):
            t = _triangle_at(tv, p)
            if t < 0:
                continue
            d_in = (p - chain[k]) / max(legs[k], 1e-12)
            cos_t = abs(float(np.dot(ng[t], d_in)))
            s_r = combine(s_diffraction(f, tri_l[t], tri_w[t], cos_t,
                                        a_star(before[k], total - before[k]), c),
                          scene.scattering[scene.tri_mat[t]])
            spec *= 1.0 - s_r
        ev.amps = ev.amps * np.sqrt(np.clip(spec, 0.0, 1.0))
        if getattr(ev, "incoherent", None) is not None:
            ev.incoherent = ev.incoherent * np.clip(spec, 0.0, 1.0)
    return events


def _triangle_at(tv, p, tol=1e-3):
    """Index of a triangle containing point p (on its plane, inside its edges), or -1."""
    a, b, c = tv[:, 0], tv[:, 1], tv[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    dist = np.abs(np.einsum('ij,ij->i', n, p - a)) / np.maximum(nn, 1e-12)
    cand = np.flatnonzero(ok & (dist < tol))
    for t in cand:
        nt = n[t]
        if (np.dot(np.cross(b[t] - a[t], p - a[t]), nt) >= -1e-9 * nn[t] and
                np.dot(np.cross(c[t] - b[t], p - b[t]), nt) >= -1e-9 * nn[t] and
                np.dot(np.cross(a[t] - c[t], p - c[t]), nt) >= -1e-9 * nn[t]):
            return int(t)
    return -1
