# -*- coding: utf-8 -*-
"""Edge diffraction (UTD): convex edge extraction, per-band D
coefficient, first-order diffracted events.
Amplitude: |D| / sqrt(s' * s * (s' + s)), free field = 1/d."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .air import air_energy_factor
from .bands import CENTERS, N_BANDS
from .geometry import SceneData
from .ism import AcousticEvent

_EPS = 1e-4
_SQRT2PI = np.sqrt(2.0 * np.pi)
_EJP4 = np.exp(1j * np.pi / 4.0)          # e^{+j pi/4}
_EMJP4 = np.exp(-1j * np.pi / 4.0)


# ---------------------------------------------------------------------------
# Fresnel transition function F(X) = 2j sqrt(X) e^{jX} I(sqrt(X)),
# I(u) = int_u^inf e^{-j tau^2} d tau -- complex numerical table.
# ---------------------------------------------------------------------------

_F_X = None
_F_TAB = None


def _build_fresnel_table():
    global _F_X, _F_TAB
    xs = np.logspace(-6, np.log10(30.0), 1024)
    umax = np.sqrt(xs[-1])
    tau = np.linspace(0.0, umax, 20000)
    integ = np.exp(-1j * tau * tau)
    # cumulative primitive from 0 to u (trapezoid rule)
    cum = np.concatenate([[0.0 + 0.0j],
                          np.cumsum(0.5 * (integ[1:] + integ[:-1])
                                    * np.diff(tau))])
    I_inf = 0.5 * np.sqrt(np.pi) * _EMJP4        # int_0^inf e^{-j t^2} dt
    us = np.sqrt(xs)
    cum_u = np.interp(us, tau, cum.real) + 1j * np.interp(us, tau, cum.imag)
    I_u = I_inf - cum_u
    _F_X = xs
    _F_TAB = 2j * us * np.exp(1j * xs) * I_u


def fresnel_F(X):
    """Complex F(X), X >= 0 (scalar or array)."""
    if _F_TAB is None:
        _build_fresnel_table()
    X = np.asarray(X, dtype=np.float64)
    out = np.ones(X.shape, dtype=np.complex128)
    small = X < _F_X[-1]
    if np.any(small):
        xs = np.clip(X[small], _F_X[0], _F_X[-1])
        re = np.interp(np.log(xs), np.log(_F_X), _F_TAB.real)
        im = np.interp(np.log(xs), np.log(_F_X), _F_TAB.imag)
        out[small] = re + 1j * im
    return out


# ---------------------------------------------------------------------------
# UTD coefficient of a wedge
# ---------------------------------------------------------------------------

def _term(beta, sigma, n, kL):
    """One of the four cot * F terms, limit form near the boundaries.
    beta: phi -/+ phi0; sigma: +1/-1; kL: k * L (per band, array)."""
    A = np.pi + sigma * beta
    N = np.round(A / (2.0 * np.pi * n))
    eps = A - 2.0 * np.pi * n * N            # small near a boundary
    kL = np.asarray(kL, dtype=np.float64)
    if abs(eps) < 1e-3:
        sg = 1.0 if eps >= 0.0 else -1.0
        return n * _EJP4 * (np.sqrt(2.0 * np.pi * kL) * sg
                            - 2.0 * kL * eps * _EJP4)
    a = 2.0 * (np.sin(0.5 * eps) ** 2)
    return (1.0 / np.tan(eps / (2.0 * n))) * fresnel_F(kL * a)


def utd_coefficient(phi, phi0, n, k, L, R0=None, Rn=None):
    """Per-band diffraction coefficient D (complex). n = air angle / pi;
    R0, Rn: amplitude reflectances of the faces (None = rigid wedge)."""
    k = np.asarray(k, dtype=np.float64)
    kL = k * max(L, 1e-9)
    one = np.ones(k.shape)
    R0 = one if R0 is None else np.asarray(R0, dtype=np.float64)
    Rn = one if Rn is None else np.asarray(Rn, dtype=np.float64)

    d = (_term(phi - phi0, +1.0, n, kL)
         + _term(phi - phi0, -1.0, n, kL)
         + Rn * _term(phi + phi0, +1.0, n, kL)
         + R0 * _term(phi + phi0, -1.0, n, kL))
    pref = -_EMJP4 / (2.0 * n * _SQRT2PI * np.sqrt(k))
    return pref * d


# ---------------------------------------------------------------------------
# Diffracting edge extraction
# ---------------------------------------------------------------------------

@dataclass
class DiffEdge:
    a: np.ndarray          # end points
    b: np.ndarray
    e: np.ndarray          # unit vector a -> b
    length: float
    t0: np.ndarray         # direction in face 0, perpendicular to e,
    w0: np.ndarray         # away from the edge; w0 = e x t0 (air side)
    theta: float           # air angle (rad), > pi (convex); 2*pi = screen
    n: float               # theta / pi
    mat0: int              # materials of the two faces
    mat1: int
    thick: float = 0.0     # merged thickness (thin-slab edges)


def _quant(v, q=1e-4):
    return (round(float(v[0]) / q), round(float(v[1]) / q),
            round(float(v[2]) / q))


def extract_edges(scene: SceneData, min_deviation_deg: float = 15.0,
                  min_length: float = 0.02,
                  thin_merge: float = 0.06,
                  max_edges: Optional[int] = None) -> List[DiffEdge]:
    """Convex edges (and free borders) of the triangle soup: air angle
    > pi + min_deviation; a single-face border becomes a half-plane."""
    tv = np.asarray(scene.tri_verts, dtype=np.float64)
    tm = np.asarray(scene.tri_mat)
    e1 = tv[:, 1] - tv[:, 0]
    e2 = tv[:, 2] - tv[:, 0]
    ng = np.cross(e1, e2)
    nl = np.linalg.norm(ng, axis=1, keepdims=True)
    nl[nl == 0] = 1.0
    ng = ng / nl
    cen = tv.mean(axis=1)

    emap = {}
    for ti in range(len(tv)):
        for i in range(3):
            va = tv[ti, i]
            vb = tv[ti, (i + 1) % 3]
            ka, kb = _quant(va), _quant(vb)
            if ka == kb:
                continue
            key = (ka, kb) if ka < kb else (kb, ka)
            emap.setdefault(key, []).append(ti)

    min_dev = np.radians(min_deviation_deg)
    edges: List[DiffEdge] = []
    for key, tris in emap.items():
        if len(tris) > 2:
            continue                        # non-manifold edge: skipped
        a = np.array(key[0], dtype=np.float64) * 1e-4
        b = np.array(key[1], dtype=np.float64) * 1e-4
        length = float(np.linalg.norm(b - a))
        if length < min_length:
            continue
        e = (b - a) / length

        def face_frame(ti):
            n0 = ng[ti]
            t = cen[ti] - 0.5 * (a + b)
            t = t - e * np.dot(t, e)
            tn = np.linalg.norm(t)
            if tn < 1e-9:
                return None
            return t / tn, n0

        f0 = face_frame(tris[0])
        if f0 is None:
            continue
        t0, n0 = f0
        # orient e so that e x t0 = +n0 (w0 points to the air side of face 0)
        w0 = np.cross(e, t0)
        if np.dot(w0, n0) < 0.0:
            e = -e
            a, b = b, a
            w0 = -w0

        if len(tris) == 1:                  # free border -> half-plane
            theta = 2.0 * np.pi
            m0 = m1 = int(tm[tris[0]])
        else:
            f1 = face_frame(tris[1])
            if f1 is None:
                continue
            t1, n1 = f1
            # Starting from t0 and turning toward +w0 (= n0, air side of
            # face 0), we stay in the air until we meet face 1:
            # Theta_air = angle of t1 in that direction, directly.
            phi1 = np.arctan2(np.dot(t1, w0), np.dot(t1, t0))
            theta = phi1 if phi1 > 0.0 else phi1 + 2.0 * np.pi
            # Orientation check: n1 must face the air arc, that is
            # phi(n1) = theta - pi/2 (mod 2 pi). Otherwise the geometry
            # is badly oriented for this edge: skip it.
            phin1 = np.arctan2(np.dot(n1, w0), np.dot(n1, t0))
            dchk = (phin1 - (theta - 0.5 * np.pi)) % (2.0 * np.pi)
            if min(dchk, 2.0 * np.pi - dchk) > 0.15:
                continue
            m0, m1 = int(tm[tris[0]]), int(tm[tris[1]])

        if theta <= np.pi + min_dev or theta > 2.0 * np.pi + 1e-6:
            continue
        edges.append(DiffEdge(a=a, b=b, e=e, length=length, t0=t0, w0=w0,
                              theta=float(theta), n=float(theta / np.pi),
                              mat0=m0, mat1=m1))
    edges = _merge_thin_pairs(edges, thin_merge)
    edges.sort(key=lambda d: -d.length)
    if max_edges is not None:
        edges = edges[:max_edges]
    return edges


def _merge_thin_pairs(edges: List[DiffEdge], tol: float) -> List[DiffEdge]:
    """Merge the parallel edge pairs of a THIN slab (wall, screen, door)
    into a single half-plane edge (n = 2) on the mid-line."""
    if tol <= 0.0 or len(edges) < 2:
        return edges
    used = [False] * len(edges)
    out: List[DiffEdge] = []
    for i, ei in enumerate(edges):
        if used[i]:
            continue
        best = -1
        for j in range(i + 1, len(edges)):
            if used[j]:
                continue
            ej = edges[j]
            if abs(abs(float(np.dot(ei.e, ej.e))) - 1.0) > 2e-3:
                continue
            if abs(ei.length - ej.length) > 0.25 * ei.length + 1e-6:
                continue
            mi = 0.5 * (ei.a + ei.b)
            mj = 0.5 * (ej.a + ej.b)
            gap = mj - mi
            if float(np.linalg.norm(gap - ei.e * np.dot(gap, ei.e))) > tol                or float(np.linalg.norm(gap)) > tol + 0.5 * abs(
                       float(np.dot(gap, ei.e))) + tol:
                continue
            if not (1.1 < ei.n < 1.9 and 1.1 < ej.n < 1.9):
                continue
            best = j
            break
        if best < 0:
            out.append(ei)
            continue
        ej = edges[best]
        used[i] = used[best] = True
        a2, b2 = ej.a, ej.b
        if float(np.dot(ej.e, ei.e)) < 0.0:
            a2, b2 = b2, a2
        a = 0.5 * (ei.a + a2)
        b = 0.5 * (ei.b + b2)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        e = (b - a) / length
        slab = (0.5 * (ej.a + ej.b)) - (0.5 * (ei.a + ei.b))
        slab = slab - e * np.dot(slab, e)
        sn = float(np.linalg.norm(slab))
        slab = slab / sn if sn > 1e-9 else ei.w0
        # screen-plane direction, away from the edge: among the t0 of the
        # two original wedges, the one running along the LARGE face
        # (largest projection off the thickness axis) points into the
        # slab material -- that is the t0 of the equivalent half-plane.
        # candidates: the t0 (in-face directions, already turned toward
        # the material when the face is the large one) and the FLIPPED
        # w0 (the air normal of a small face -- jamb, lintel -- points to
        # the opening; its opposite points into the slab).
        cands = []
        for tc in (ei.t0, ej.t0, -ei.w0, -ej.w0):
            p = tc - slab * np.dot(tc, slab)
            p = p - e * np.dot(p, e)
            cands.append((float(np.linalg.norm(p)), p))
        ln, t0 = max(cands, key=lambda c: c[0])
        if ln < 1e-6:
            continue
        t0 = t0 / ln
        w0 = np.cross(e, t0)
        out.append(DiffEdge(a=a, b=b, e=e, length=length, t0=t0, w0=w0,
                            theta=2.0 * np.pi, n=2.0,
                            mat0=ei.mat0, mat1=ej.mat0, thick=sn))
    return out


# ---------------------------------------------------------------------------
# First-order diffraction events
# ---------------------------------------------------------------------------

def _angle_in_wedge(v, edge: DiffEdge):
    """(angle from face 0 through the air, perpendicular radius)."""
    vp = v - edge.e * np.dot(v, edge.e)
    r = float(np.linalg.norm(vp))
    if r < 1e-9:
        return None, 0.0
    u = vp / r
    phi = float(np.arctan2(np.dot(u, edge.w0), np.dot(u, edge.t0)))
    if phi < 0.0:
        phi += 2.0 * np.pi
    return phi, r


def compute_diffraction(scene: SceneData, tracer, edges: List[DiffEdge],
                        src, rcv, air_db_per_m, speed_of_sound: float,
                        min_amp: float = 1e-5,
                        refl_amp: Optional[np.ndarray] = None
                        ) -> List[AcousticEvent]:
    """Source -> edge -> receiver paths (first order), ray-validated.
    refl_amp: (n_mats, N_BANDS) in AMPLITUDE; None = rigid wedge."""
    src = np.asarray(src, dtype=np.float64)
    rcv = np.asarray(rcv, dtype=np.float64)
    k = 2.0 * np.pi * np.asarray(CENTERS, dtype=np.float64) / speed_of_sound
    events: List[AcousticEvent] = []

    for ed in edges:
        # Fermat point on the edge line (equal angles with e)
        zs = float(np.dot(src - ed.a, ed.e))
        zr = float(np.dot(rcv - ed.a, ed.e))
        rs = float(np.linalg.norm((src - ed.a) - ed.e * zs))
        rr = float(np.linalg.norm((rcv - ed.a) - ed.e * zr))
        if rs < 1e-6 or rr < 1e-6:
            continue
        z = zs + rs * (zr - zs) / (rs + rr)
        margin = min(0.01, 0.05 * ed.length)
        if z < margin or z > ed.length - margin:
            continue                         # point off the finite edge
        D = ed.a + ed.e * z

        vs = src - D
        vr = rcv - D
        sp = float(np.linalg.norm(vs))       # s'
        s = float(np.linalg.norm(vr))
        if sp < 1e-4 or s < 1e-4:
            continue

        phi0, _ = _angle_in_wedge(vs, ed)
        phi, _ = _angle_in_wedge(vr, ed)
        if phi0 is None or phi is None:
            continue
        eps_a = 1e-4
        if not (eps_a < phi0 < ed.theta - eps_a):
            continue                         # source outside the wedge air
        if not (eps_a < phi < ed.theta - eps_a):
            continue

        sinb = float(np.clip(np.sqrt(max(0.0, 1.0 -
                     (np.dot(vs, ed.e) / sp) ** 2)), 1e-4, 1.0))
        L = (s * sp / (s + sp)) * sinb * sinb

        R0 = Rn = None
        if refl_amp is not None:
            R0 = refl_amp[min(ed.mat0, len(refl_amp) - 1)]
            Rn = refl_amp[min(ed.mat1, len(refl_amp) - 1)]
        Dc = utd_coefficient(phi, phi0, ed.n, k, L, R0=R0, Rn=Rn)
        geo = 1.0 / np.sqrt(sp * s * (sp + s))
        amps = np.abs(Dc) * geo
        amps *= np.sqrt(air_energy_factor(sp + s, air_db_per_m))
        if float(amps.max()) < min_amp:
            continue

        # ray-casting validation: both segments must be clear (target
        # nudged into the air along the bisector)
        half = 0.5 * ed.theta
        bis = ed.t0 * np.cos(half) + ed.w0 * np.sin(half)
        Doff = D + bis * (_EPS * 8.0)
        # margin on the D side: at least half the merged thickness -- the
        # slab an equivalent edge stands for must not occlude itself
        pull = 0.75 * ed.thick + _EPS * 12.0
        ds = Doff - src
        ls = float(np.linalg.norm(ds))
        if ls <= pull + _EPS * 8.0:
            continue
        tri, _t, _n = tracer.cast(src + (ds / ls) * (_EPS * 4.0), ds / ls,
                                  ls - pull)
        if tri >= 0:
            continue
        dr = rcv - Doff
        lr = float(np.linalg.norm(dr))
        if lr <= pull + _EPS * 8.0:
            continue
        tri, _t, _n = tracer.cast(Doff + (dr / lr) * pull, dr / lr,
                                  lr - pull - _EPS * 4.0)
        if tri >= 0:
            continue

        events.append(AcousticEvent(
            time=(sp + s) / speed_of_sound,
            amps=amps.astype(np.float64),
            direction=(D - rcv) / max(s, 1e-9),
            order=1, kind="diffracted",
            emit_dir=(D - src) / max(sp, 1e-9), points=[np.asarray(D, float).copy()]))
    return events
