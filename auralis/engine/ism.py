# -*- coding: utf-8 -*-
"""Deterministic early part: direct sound (with transmission) and image
sources (ISM) for low-order specular reflections. The stochastic tracer
excludes exactly these paths (spec_exclude_order)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .air import air_energy_factor
from .bands import N_BANDS
from .geometry import Plane, SceneData

_EPS = 1e-4


@dataclass
class AcousticEvent:
    """One deterministic arrival: time (s), per-band amps (energy =
    amps**2), unit direction from the receiver to the incoming wave."""
    time: float
    amps: np.ndarray
    direction: np.ndarray
    order: int
    kind: str
    emit_dir: np.ndarray = None   # unit vector: emission direction from
                                  # the real source (for directivity)
    bounces: object = None        # [(mat, cos_theta), ...] -> R(theta, f)
    points: object = None         # reflection / diffraction points, source side first
    planes: object = None         # patch index of each reflection, source side first


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


def compute_direct(scene: SceneData, tracer, src, rcv, air_db_per_m,
                   speed_of_sound: float) -> AcousticEvent:
    """Direct arrival, damped by transmission through crossed surfaces:
    amp_b = sqrt(tau_b * air_b(d)) / d, zero on full occlusion."""
    src = np.asarray(src, dtype=np.float64)
    rcv = np.asarray(rcv, dtype=np.float64)
    d = float(np.linalg.norm(rcv - src))
    d = max(d, 1e-3)
    trans, _n = tracer.segment_transmission(src, rcv, scene.transmission)
    energy = np.asarray(trans, dtype=np.float64) * air_energy_factor(d, air_db_per_m)
    amps = np.sqrt(np.maximum(energy, 0.0)) / d
    emit = _unit(rcv - src)
    return AcousticEvent(time=d / speed_of_sound,
                         amps=amps.astype(np.float64),
                         direction=-emit,
                         order=0, kind="direct", emit_dir=emit)


def _mirror(point, plane: Plane):
    dist = float(np.dot(plane.normal, point) - plane.d)
    return point - 2.0 * dist * plane.normal


_PATCH_TOL = 1e-2   # m: > 5e-3 (cast tolerance in _validate_path) + float32 error


class _PatchTable:
    """In-plane inward edge normals of every patch triangle, to bound from
    below the distance between a point and a patch (cheap, vectorised)."""

    def __init__(self, scene, planes):
        tv = scene.tri_verts.astype(np.float64)
        self.per_plane = []
        for pl in planes:
            v = tv[pl.tris]                                    # (t, 3, 3)
            n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
            en = []
            for i in range(3):
                e = v[:, (i + 1) % 3] - v[:, i]
                m = np.cross(n, e)                             # in-plane, inward
                m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-300)
                en.append(m)
            self.per_plane.append((v, np.stack(en, axis=1)))   # (t,3,3), (t,3,3)

    def far_from(self, pid, pts, tol):
        """True where every triangle of patch `pid` is farther than tol."""
        v, en = self.per_plane[pid]
        out = np.empty(len(pts), dtype=bool)
        step = max(1, int(2_000_000 // (9 * len(v))))
        for a in range(0, len(pts), step):
            rel = pts[a:a + step, None, None, :] - v[None, :, :, :]   # (m, t, 3, 3)
            outside = -np.einsum("mtij,tij->mti", rel, en).min(axis=2)
            out[a:a + step] = outside.min(axis=1) > tol
        return out


def compute_image_sources(scene: SceneData, planes: List[Plane],
                          tri_plane: np.ndarray, tracer, src, rcv,
                          max_order: int, air_db_per_m,
                          speed_of_sound: float,
                          adm_table=None,
                          min_amp: float = 1e-5) -> List[AcousticEvent]:
    """Image sources validated up to `max_order`: every path is retraced
    from the receiver back to the source (finite patches, occlusions).
    Same events, in the same order, as compute_image_sources_reference:
    candidates are enumerated in numpy and a CONSERVATIVE pre-filter drops
    those whose reflection points miss their patch by more than the cast
    tolerance; survivors go through the reference validation unchanged."""
    src = np.asarray(src, dtype=np.float64)
    rcv = np.asarray(rcv, dtype=np.float64)
    if max_order <= 0 or not planes:
        return []
    refl = scene.reflectance()
    normals = np.array([pl.normal for pl in planes], dtype=np.float64)
    ds = np.array([pl.d for pl in planes], dtype=np.float64)
    mats = np.array([pl.material for pl in planes], dtype=np.int64)
    patches = _PatchTable(scene, planes)
    n_pl = len(planes)
    k_air = (np.log(10.0) / 10.0) * np.asarray(air_db_per_m, dtype=np.float64)
    events: List[AcousticEvent] = []

    # frontier, BFS order: sequences (m, o), image chains (m, o, 3), reflectance
    seqs = np.zeros((1, 0), dtype=np.int64)
    chains = np.zeros((1, 0, 3))
    rprods = np.ones((1, N_BANDS))
    for order in range(1, max_order + 1):
        m = len(seqs)
        parent = np.repeat(np.arange(m), n_pl)
        pid = np.tile(np.arange(n_pl), m)
        last = chains[parent, -1] if order > 1 else np.broadcast_to(src, (len(pid), 3))
        keep = np.ones(len(pid), dtype=bool)
        if order > 1:
            keep &= seqs[parent, -1] != pid
        dist = np.einsum("ij,ij->i", normals[pid], last) - ds[pid]
        # exact zero test of the reference: decide near-zero values in scalar
        near = keep & (np.abs(dist) < 1e-9)
        for i in np.flatnonzero(near):
            if float(np.dot(planes[pid[i]].normal, last[i]) - planes[pid[i]].d) == 0.0:
                keep[i] = False
        r2 = rprods[parent] * refl[mats[pid]]                   # same products as BFS
        keep &= ~(r2.max(axis=1) <= min_amp * min_amp)
        parent, pid, dist, r2, last = parent[keep], pid[keep], dist[keep], r2[keep], last[keep]
        img = last - 2.0 * dist[:, None] * normals[pid]
        seqs = np.concatenate([seqs[parent], pid[:, None]], axis=1)
        chains = np.concatenate([chains[parent], img[:, None, :]], axis=1)
        rprods = r2
        if len(seqs) == 0:
            break
        cand = _prefilter(seqs, chains, rprods, rcv, normals, ds, patches, k_air, min_amp)
        for i in np.flatnonzero(cand):
            seq = tuple(int(x) for x in seqs[i])
            final = src.copy()                                  # exact scalar chain
            for q in seq:
                final = _mirror(final, planes[q])
            ev = _validate_path(scene, planes, tri_plane, tracer,
                                src, rcv, seq, final, rprods[i],
                                air_db_per_m, speed_of_sound, min_amp,
                                adm_table)
            if ev is not None:
                events.append(ev)
    return events


def _prefilter(seqs, chains, rprods, rcv, normals, ds, patches, k_air, min_amp):
    """Candidates the reference validation MIGHT accept (never drops a valid one)."""
    m, order = seqs.shape
    ok = np.ones(m, dtype=bool)
    L = np.linalg.norm(chains[:, -1] - rcv, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        amps = np.sqrt(np.maximum(rprods * np.exp(-np.multiply.outer(L, k_air)), 0.0)) \
            / L[:, None]
    ok &= ~(amps.max(axis=1) < min_amp * (1.0 - 1e-9))         # NaN (L = 0) kept
    point = np.broadcast_to(rcv, (m, 3)).copy()
    for k in range(order - 1, -1, -1):
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            break
        q = seqs[idx, k]
        d = chains[idx, k] - point[idx]
        dist = np.linalg.norm(d, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = d / dist[:, None]
            denom = np.einsum("ij,ij->i", normals[q], d)
            t = (ds[q] - np.einsum("ij,ij->i", normals[q], point[idx])) / denom
        finite = np.isfinite(t)
        bad = finite & ((t < _EPS - 1e-9) | (t > dist - _EPS + 1e-9))
        p = point[idx] + d * np.where(finite, t, 0.0)[:, None]
        far = np.zeros(len(idx), dtype=bool)
        for pid in np.unique(q[finite & ~bad]):
            sel = np.flatnonzero((q == pid) & finite & ~bad)
            far[sel] = patches.far_from(int(pid), p[sel], _PATCH_TOL)
        ok[idx[bad | far]] = False
        point[idx] = p
    return ok


def compute_image_sources_reference(scene: SceneData, planes: List[Plane],
                                    tri_plane: np.ndarray, tracer, src, rcv,
                                    max_order: int, air_db_per_m,
                                    speed_of_sound: float,
                                    adm_table=None,
                                    min_amp: float = 1e-5) -> List[AcousticEvent]:
    """Reference (one Python validation per candidate): the anchor of the
    vectorised compute_image_sources, which must return the same events."""
    src = np.asarray(src, dtype=np.float64)
    rcv = np.asarray(rcv, dtype=np.float64)
    if max_order <= 0 or not planes:
        return []

    refl = scene.reflectance()          # (n_mats, NB)
    events: List[AcousticEvent] = []

    # ---- enumerate plausible sequences (BFS, no immediate repeats) -------
    # Each node: (sequence tuple, image position, reflectance product)
    frontier = [((), src, np.ones(N_BANDS))]
    for _order in range(1, max_order + 1):
        nxt = []
        for seq, img, rprod in frontier:
            for pid, pl in enumerate(planes):
                if seq and seq[-1] == pid:
                    continue
                # the source (image) must be on the front side of the patch
                if float(np.dot(pl.normal, img) - pl.d) == 0.0:
                    continue
                img2 = _mirror(img, pl)
                r2 = rprod * refl[pl.material]
                if float(r2.max()) <= min_amp * min_amp:
                    continue
                nxt.append((seq + (pid,), img2, r2))
        for seq, img, rprod in nxt:
            ev = _validate_path(scene, planes, tri_plane, tracer,
                                src, rcv, seq, img, rprod,
                                air_db_per_m, speed_of_sound, min_amp,
                                adm_table)
            if ev is not None:
                events.append(ev)
        frontier = nxt
    return events


def _validate_path(scene, planes, tri_plane, tracer, src, rcv, seq,
                   final_image, rprod, air_db_per_m, c, min_amp,
                   adm_table=None):
    """Backtrack rcv -> ... -> src, casting real rays at every leg."""
    L = float(np.linalg.norm(final_image - rcv))
    if L < 1e-6:
        return None
    air_e = air_energy_factor(L, air_db_per_m)
    energy = rprod * air_e
    amps = np.sqrt(np.maximum(energy, 0.0)) / L
    if float(amps.max()) < min_amp:
        return None

    # image chain: images[k] = src mirrored across seq[0..k]
    images = []
    img = src.copy()
    for pid in seq:
        img = _mirror(img, planes[pid])
        images.append(img.copy())

    point = rcv.copy()
    first_leg_dir = None
    mu_list = []
    hits = []
    for k in range(len(seq) - 1, -1, -1):
        target = images[k]
        pl = planes[seq[k]]
        d = target - point
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return None
        d /= dist
        denom = float(np.dot(pl.normal, d))
        if abs(denom) < 1e-9:
            return None
        mu_list.append((pl.material, abs(denom)))
        t_plane = (pl.d - float(np.dot(pl.normal, point))) / denom
        if t_plane <= _EPS or t_plane >= dist - _EPS:
            return None
        tri, t_hit, _n = tracer.cast(point, d, dist)
        # the nearest surface along this leg must be the expected patch,
        # at the expected distance (no occluder in front of it)
        if tri < 0 or tri_plane[tri] != seq[k] or abs(t_hit - t_plane) > 5e-3:
            return None
        if first_leg_dir is None:
            first_leg_dir = d.copy()
        point = point + d * t_plane
        hits.append(point)
        # nudge off the surface toward the next target
    # final leg: last reflection point -> real source, unoccluded
    d = src - point
    dist = float(np.linalg.norm(d))
    emit = np.array([0.0, 0.0, 1.0])
    if dist > _EPS * 10:
        d /= dist
        emit = -d                     # source -> first reflection point
        tri, t_hit, _n = tracer.cast(point + d * (_EPS * 4), d, dist - _EPS * 8)
        if tri >= 0:
            return None
    if adm_table is not None:
        from .reflection import band_R_mag
        amps = band_R_mag(mu_list, adm_table) \
            * np.sqrt(np.maximum(air_e, 0.0)) / L
    return AcousticEvent(time=L / c, amps=amps.astype(np.float64),
                         direction=np.asarray(first_leg_dir, dtype=np.float64),
                         order=len(seq), kind="specular",
                         emit_dir=emit,
                         bounces=(list(mu_list) if adm_table is not None
                                  else None),
                         points=[p.copy() for p in hits[::-1]], planes=tuple(seq))
