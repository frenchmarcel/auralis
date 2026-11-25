# -*- coding: utf-8 -*-
"""Pure Python stochastic tracer (vectorised NumPy), mirror of the
native kernel: diffuse rain for the scattered share, sphere detection
(weight 4/rcv_radius^2) for the specular share past spec_exclude_order."""

from __future__ import annotations

import numpy as np

from .bands import CENTERS, N_BANDS
from . import rbs as _rbs
from .geometry import NumpyRaycaster, SceneData

_EPS = 1e-4
_JIT = 0.008           # ~0.5 deg: anti-crack recast
_MAX_RETRY = 2


def trace_python(scene: SceneData, src, rcv, n_rays: int, max_bounces: int,
                 max_time: float, bin_dt: float, n_bins: int,
                 speed_of_sound: float, air_db_per_m,
                 spec_exclude_order: int = 2, energy_cap: float = 1e-6,
                 seed: int = 1234, batch: int = 2048, progress=None,
                 src_forward=None, dir_table=None, rcv_radius: float = 0.3,
                 rbs=None):
    """Returns (hist (NB, n_bins) float64, dirs (n_bins, 3), stats dict).
    rbs = (tri_l, tri_w) per triangle: reflection-based scattering (rbs.py) instead of
    the materials' scattering alone; the incident distance is the path since the
    last diffuse event (the image distance of the specular chain)."""
    rc = NumpyRaycaster(scene)
    rng = np.random.default_rng(seed)
    if rbs is not None:
        rbs = (np.asarray(rbs[0], dtype=np.float64), np.asarray(rbs[1], dtype=np.float64))
        rbs_f = np.asarray(CENTERS, dtype=np.float64)[None, :]

    refl = scene.reflectance().astype(np.float64)          # (n_mats, NB)
    scat = scene.scattering.astype(np.float64)
    kair = (np.log(10.0) / 10.0) * np.asarray(air_db_per_m, dtype=np.float64)

    hist = np.zeros((N_BANDS, n_bins), dtype=np.float64)
    dirs = np.zeros((n_bins, 3), dtype=np.float64)
    total_bounces = 0.0
    total_escaped = 0.0
    total_deposits = 0.0
    total_detect = 0.0
    total_retried = 0.0
    inv_n = 1.0 / max(1, n_rays)
    K = int(max(0, spec_exclude_order))
    cap = max(1e-9, float(energy_cap))
    r_det = max(1e-3, float(rcv_radius))
    det_w = 4.0 / (r_det * r_det)      # expectation = lattice (1/d^2)
    src = np.asarray(src, dtype=np.float64)
    rcv = np.asarray(rcv, dtype=np.float64)
    c = float(speed_of_sound)

    fwd = None
    tab = None
    if dir_table is not None and src_forward is not None:
        tab = np.asarray(dir_table, dtype=np.float64)      # (NB, S)
        fwd = np.asarray(src_forward, dtype=np.float64)
        nf = np.linalg.norm(fwd)
        if nf > 1e-9 and tab.shape[1] >= 2:
            fwd = fwd / nf
        else:
            fwd = tab = None

    done = 0
    while done < n_rays:
        n = min(batch, n_rays - done)
        done += n

        # ray state
        z = 1.0 - 2.0 * rng.random(n)
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        phi = 2.0 * np.pi * rng.random(n)
        d = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
        o = np.broadcast_to(src, (n, 3)).copy()
        if tab is not None:
            S = tab.shape[1]
            ct = np.clip(d @ fwd, -1.0, 1.0)
            u = np.arccos(ct) / np.pi * (S - 1)
            i0 = np.clip(u.astype(np.int64), 0, S - 2)
            fr = (u - i0)[:, None]
            E = tab[:, i0].T * (1.0 - fr) + tab[:, i0 + 1].T * fr
        else:
            E = np.ones((n, N_BANDS))
        t_acc = np.zeros(n)
        order = np.zeros(n, dtype=np.int32)
        pure_spec = np.ones(n, dtype=bool)
        spec_seg = np.ones(n, dtype=bool)    # segment born from a specular
        retries = np.zeros(n, dtype=np.int32)  # branch (launch included)
        alive = np.ones(n, dtype=bool)
        dspec = np.zeros(n)                  # path since the last diffuse event
        wsp = np.ones((n, N_BANDS))          # rbs: (1 - s_b) / (1 - s_mean) of a
                                             # specular segment (unbiased per band)

        for _bounce in range(max_bounces):
            idx = np.where(alive)[0]
            if idx.size == 0:
                break
            tri, t_hit = rc.cast_batch(o[idx], d[idx])
            miss = tri < 0

            # ---- numerical cracks: re-aim before giving up --------------
            jit = miss & (retries[idx] < _MAX_RETRY)
            if jit.any():
                ji = idx[jit]
                retries[ji] += 1
                total_retried += float(ji.size)
                dj = d[ji] + _JIT * rng.standard_normal((ji.size, 3))
                d[ji] = dj / np.linalg.norm(dj, axis=1, keepdims=True)
            final = miss & ~jit
            total_escaped += float(final.sum())

            # ---- sphere detection: energy carried SPECULARLY ------------
            # final segments only (hit, or confirmed escape); purely
            # specular chains <= K stay with the image sources; the
            # launch segment (order 0) is the direct path.
            segl = np.full(idx.size, -1.0)
            segl[~miss] = t_hit[~miss]
            segl[final] = np.maximum(0.0, c * (max_time - t_acc[idx[final]]))
            cd = segl > 0.0
            cd &= spec_seg[idx]
            cd &= ~(pure_spec[idx] & (order[idx] <= K))
            if cd.any():
                ci = idx[cd]
                w = rcv[None, :] - o[ci]
                tc = np.einsum('ij,ij->i', w, d[ci])
                inseg = (tc > 0.0) & (tc < segl[cd])
                if inseg.any():
                    ci = ci[inseg]
                    tc = tc[inseg]
                    miss2 = (np.einsum('ij,ij->i', w[inseg], w[inseg])
                             - tc * tc)
                    close = miss2 < r_det * r_det
                    if close.any():
                        ci = ci[close]
                        tc = tc[close]
                        b = ((t_acc[ci] + tc / c) / bin_dt).astype(np.int64)
                        okb = (b >= 0) & (b < n_bins)
                        ci, tc, b = ci[okb], tc[okb], b[okb]
                        if ci.size:
                            dep = (E[ci] * np.exp(-np.outer(tc, kair))
                                   * det_w * inv_n)
                            if rbs is not None:
                                dep = dep * wsp[ci]
                            np.add.at(hist, (slice(None), b), dep.T)
                            wsum = dep.sum(axis=1)
                            np.add.at(dirs, b, -d[ci] * wsum[:, None])
                            total_detect += float(ci.size)

            alive[idx[final]] = False
            hit = idx[~miss]
            if hit.size == 0:
                continue
            tri = tri[~miss]
            th = t_hit[~miss]

            # propagation: air absorption + time
            E[hit] *= np.exp(-np.outer(th, kair))
            t_acc[hit] += th / speed_of_sound
            over = t_acc[hit] > max_time
            alive[hit[over]] = False
            hit = hit[~over]
            if hit.size == 0:
                continue
            tri = tri[~over]
            th = th[~over]

            hp = o[hit] + d[hit] * th[:, None]
            nrm = rc.normals[tri].copy()
            flip = np.einsum('ij,ij->i', nrm, d[hit]) > 0.0
            nrm[flip] *= -1.0

            m = np.clip(rc.tri_mat[tri], 0, scene.n_mats - 1)
            Eref = E[hit] * refl[m]                 # (H, NB)
            s_here = scat[m]                        # (H, NB)
            s_cont = s_here
            if rbs is not None:
                d_in = dspec[hit] + th
                cos_i = np.abs(np.einsum('ij,ij->i', nrm, d[hit]))[:, None]
                tl, tw = rbs[0][tri][:, None], rbs[1][tri][:, None]
                s_cont = _rbs.combine(_rbs.s_diffraction(
                    rbs_f, tl, tw, cos_i, (d_in / 4.0)[:, None], c), s_here)
            s_mean = s_cont.mean(axis=1)

            # ---- diffuse rain toward the receiver -----------------------
            # only the SCATTERED fraction radiates in a Lambert lobe; the
            # specular share is measured by the sphere detection.
            to_r = rcv[None, :] - hp
            dR = np.linalg.norm(to_r, axis=1)
            okR = dR > 1e-4
            ur = np.zeros_like(to_r)
            ur[okR] = to_r[okR] / dR[okR, None]
            costh = np.einsum('ij,ij->i', nrm, ur)
            bins = ((t_acc[hit] + dR / speed_of_sound) / bin_dt).astype(np.int64)
            cand = okR & (costh > 0.0) & (bins >= 0) & (bins < n_bins)
            if cand.any():
                ci = np.where(cand)[0]
                off = hp[ci] + nrm[ci] * (_EPS * 4.0)
                blocked = rc.occluded_batch(off, ur[ci], dR[ci] - _EPS * 8.0)
                ci = ci[~blocked]
                if ci.size:
                    s_rain = s_here[ci]
                    if rbs is not None:          # the last leg's true length
                        s_rain = _rbs.combine(_rbs.s_diffraction(
                            rbs_f, tl[ci], tw[ci], cos_i[ci],
                            _rbs.a_star(d_in[ci], dR[ci])[:, None], c), s_here[ci])
                    geo = 4.0 * costh[ci] / (dR[ci] ** 2)
                    dep = (Eref[ci] * s_rain * geo[:, None]
                           * np.exp(-np.outer(dR[ci], kair)) * inv_n)  # (C,NB)
                    b = bins[ci]
                    np.add.at(hist, (slice(None), b), dep.T)
                    w = dep.sum(axis=1)
                    doa = -ur[ci]                     # receiver -> wall
                    np.add.at(dirs, b, doa * w[:, None])
                    total_deposits += float(ci.size)

            # ---- continue rays ------------------------------------------
            total_bounces += float(hit.size)
            order[hit] += 1
            E[hit] = Eref

            e_max = Eref.max(axis=1)
            low = e_max < cap
            if low.any():
                surv = rng.random(hit.size) <= 0.25
                kill = low & (~surv | (e_max < cap * 1e-4))
                boost = low & surv & (e_max >= cap * 1e-4)
                E[hit[boost]] *= 4.0
                alive[hit[kill]] = False

            live = alive[hit]
            hl = hit[live]
            if hl.size:
                go_diff = rng.random(hl.size) < s_mean[live]
                nl = nrm[live]
                dl = d[hl]
                # specular
                spec = dl - 2.0 * np.einsum('ij,ij->i', dl, nl)[:, None] * nl
                # cosine-weighted diffuse
                u1 = rng.random(hl.size)
                u2 = rng.random(hl.size)
                rr = np.sqrt(u1)
                ph = 2.0 * np.pi * u2
                x = rr * np.cos(ph)
                y = rr * np.sin(ph)
                zz = np.sqrt(np.maximum(0.0, 1.0 - u1))
                a = np.where(np.abs(nl[:, 0:1]) > 0.9,
                             np.array([[0.0, 1.0, 0.0]]),
                             np.array([[1.0, 0.0, 0.0]]))
                tt = np.cross(a, nl)
                tt /= np.linalg.norm(tt, axis=1, keepdims=True)
                bb = np.cross(nl, tt)
                diff = tt * x[:, None] + bb * y[:, None] + nl * zz[:, None]
                newd = np.where(go_diff[:, None], diff, spec)
                newd /= np.linalg.norm(newd, axis=1, keepdims=True)
                d[hl] = newd
                pure_spec[hl] &= ~go_diff
                spec_seg[hl] = ~go_diff
                if rbs is not None:
                    dspec[hl] = np.where(go_diff, 0.0, d_in[live])
                    sl = s_cont[live]
                    q = 1.0 / np.maximum(1.0 - s_mean[live], 1e-6)
                    sp = ~go_diff
                    wsp[hl[sp]] = (1.0 - sl[sp]) * q[sp, None]
                o[hl] = hp[live] + nl * (_EPS * 4.0)

        if progress is not None:
            if progress(done, n_rays) is False:
                break

    stats = {"rays": float(done), "mean_bounces": total_bounces / max(1.0, done),
             "escaped": total_escaped, "deposits": total_deposits,
             "detections": total_detect, "retried": total_retried}
    return hist, dirs, stats
