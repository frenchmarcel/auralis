# -*- coding: utf-8 -*-
"""Scene geometry: triangle soup + materials, a vectorised NumPy raycaster
(pure-Python fallback for the native kernel) and planar-patch extraction
for the image-source method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .bands import N_BANDS
from .materials import build_tables, get_preset

_EPS = 1e-4


@dataclass
class SceneData:
    """Triangle soup (n_tris, 3, 3) with per-triangle material indices
    and (n_mats, N_BANDS) absorption / scattering / transmission tables."""
    tri_verts: np.ndarray
    tri_mat: np.ndarray
    absorption: np.ndarray
    scattering: np.ndarray
    transmission: np.ndarray
    mat_phys: list = None      # poromat specs per material (or None)
    mat_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.tri_verts = np.ascontiguousarray(self.tri_verts, dtype=np.float32)
        self.tri_mat = np.ascontiguousarray(self.tri_mat, dtype=np.int32)
        n = len(self.tri_verts)
        if self.tri_mat.shape != (n,):
            raise ValueError("tri_mat must be (n_tris,)")
        for name in ("absorption", "scattering", "transmission"):
            a = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != N_BANDS:
                raise ValueError("%s must be (n_mats, %d)" % (name, N_BANDS))
            setattr(self, name, a)
        if not self.mat_names:
            self.mat_names = ["mat_%d" % i for i in range(self.n_mats)]

    # ---- convenience ----------------------------------------------------
    @property
    def n_tris(self) -> int:
        return int(len(self.tri_verts))

    @property
    def n_mats(self) -> int:
        return int(self.absorption.shape[0])

    def triangle_areas(self) -> np.ndarray:
        e1 = self.tri_verts[:, 1] - self.tri_verts[:, 0]
        e2 = self.tri_verts[:, 2] - self.tri_verts[:, 0]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    def bbox(self):
        v = self.tri_verts.reshape(-1, 3)
        return v.min(axis=0), v.max(axis=0)

    def reflectance(self) -> np.ndarray:
        """(n_mats, N_BANDS) reflected-energy fraction, clamped >= 0."""
        return np.clip(1.0 - self.absorption - self.transmission, 0.0, 1.0)

    @staticmethod
    def from_materials(tri_verts, tri_mat, materials, mat_names=None) -> "SceneData":
        a, s, t = build_tables(materials)
        return SceneData(np.asarray(tri_verts), np.asarray(tri_mat), a, s, t,
                         mat_names=list(mat_names or []))


# --------------------------------------------------------------------------
# Procedural helpers
# --------------------------------------------------------------------------

def _quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def box_triangles(size, origin=(0, 0, 0), inward=True):
    """12 triangles of an axis-aligned box; inward=True -> normals point
    inside. Returns (12, 3, 3) float32 and one face name per triangle."""
    lx, ly, lz = size
    ox, oy, oz = origin
    v = np.array([
        [ox,      oy,      oz],
        [ox + lx, oy,      oz],
        [ox + lx, oy + ly, oz],
        [ox,      oy + ly, oz],
        [ox,      oy,      oz + lz],
        [ox + lx, oy,      oz + lz],
        [ox + lx, oy + ly, oz + lz],
        [ox,      oy + ly, oz + lz],
    ], dtype=np.float32)
    if inward:
        faces = [
            ((0, 1, 2, 3), "floor"),      # z = oz,   normal +z
            ((7, 6, 5, 4), "ceiling"),    # z = oz+lz normal -z
            ((4, 5, 1, 0), "wall_y0"),    # y = oy    normal +y
            ((2, 6, 7, 3), "wall_y1"),
            ((1, 5, 6, 2), "wall_x1"),
            ((3, 7, 4, 0), "wall_x0"),
        ]
    else:
        faces = [
            ((3, 2, 1, 0), "bottom"),
            ((4, 5, 6, 7), "top"),
            ((0, 1, 5, 4), "side_y0"),
            ((3, 7, 6, 2), "side_y1"),
            ((2, 6, 5, 1), "side_x1"),
            ((0, 4, 7, 3), "side_x0"),
        ]
    tris, names = [], []
    for (a, b, c, d), name in faces:
        tris += _quad(v[a], v[b], v[c], v[d])
        names += [name, name]
    return np.asarray(tris, dtype=np.float32), names


def make_shoebox(size=(10.0, 7.0, 3.0),
                 wall_material="plaster",
                 floor_material=None,
                 ceiling_material=None) -> SceneData:
    """Simple rectangular room. Materials given as preset names or dicts."""
    def resolve(m):
        return get_preset(m) if isinstance(m, str) else m

    wall = resolve(wall_material)
    floor = resolve(floor_material) if floor_material else wall
    ceil = resolve(ceiling_material) if ceiling_material else wall
    mats = [floor, ceil, wall]
    names = ["floor", "ceiling", "walls"]

    tris, face_names = box_triangles(size, inward=True)
    mat_of_face = {"floor": 0, "ceiling": 1}
    tri_mat = np.array([mat_of_face.get(n, 2) for n in face_names], dtype=np.int32)
    return SceneData.from_materials(tris, tri_mat, mats, names)


def add_box(scene: SceneData, size, origin, material, name="obstacle") -> SceneData:
    """Return a new SceneData with an outward-facing box appended."""
    mat = get_preset(material) if isinstance(material, str) else material
    tris, _ = box_triangles(size, origin, inward=False)
    mat_idx = scene.n_mats
    tri_verts = np.concatenate([scene.tri_verts, tris], axis=0)
    tri_mat = np.concatenate([scene.tri_mat,
                              np.full(len(tris), mat_idx, dtype=np.int32)])
    absorption = np.concatenate([scene.absorption, mat["absorption"][None]], axis=0)
    scattering = np.concatenate([scene.scattering, mat["scattering"][None]], axis=0)
    transmission = np.concatenate([scene.transmission, mat["transmission"][None]], axis=0)
    return SceneData(tri_verts, tri_mat, absorption, scattering, transmission,
                     mat_names=scene.mat_names + [name])


# --------------------------------------------------------------------------
# Vectorised NumPy raycaster (fallback path, mirrors the native kernel)
# --------------------------------------------------------------------------

class NumpyRaycaster:
    """Batched Moller-Trumbore over the whole triangle list, chunked to
    bound the working set. Slower than the BVH kernel, dependency-free."""

    MAX_PAIRS = 6_000_000

    def __init__(self, scene: SceneData):
        tv = scene.tri_verts.astype(np.float64)
        self.v0 = np.ascontiguousarray(tv[:, 0])
        self.e1 = np.ascontiguousarray(tv[:, 1] - tv[:, 0])
        self.e2 = np.ascontiguousarray(tv[:, 2] - tv[:, 0])
        ng = np.cross(self.e1, self.e2)
        nl = np.linalg.norm(ng, axis=1, keepdims=True)
        nl[nl == 0] = 1.0
        self.normals = ng / nl
        self.n_tris = len(self.v0)
        self.tri_mat = scene.tri_mat

    # ---- core batched intersection --------------------------------------
    def _intersect_chunk(self, o, d, j0, j1, t_best, i_best, any_hit, t_max):
        """Rays (N,3) against triangles [j0:j1)."""
        v0 = self.v0[j0:j1]           # (M,3)
        e1 = self.e1[j0:j1]
        e2 = self.e2[j0:j1]
        # p = d x e2 -> (N,M,3)
        p = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum('mk,nmk->nm', e1, p)
        inv = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1.0, det), 0.0)
        tv = o[:, None, :] - v0[None, :, :]
        u = np.einsum('nmk,nmk->nm', tv, p) * inv
        q = np.cross(tv, e1[None, :, :])
        v = np.einsum('nk,nmk->nm', d, q) * inv
        t = np.einsum('mk,nmk->nm', e2, q) * inv
        ok = ((np.abs(det) > 1e-12)
              & (u >= -1e-6) & (u <= 1.0 + 1e-6)
              & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
              & (t > _EPS))
        if t_max is not None:
            ok &= t < t_max[:, None]
        t = np.where(ok, t, np.inf)
        if any_hit:
            return ok.any(axis=1)
        jrel = np.argmin(t, axis=1)
        tj = t[np.arange(len(o)), jrel]
        better = tj < t_best
        t_best[better] = tj[better]
        i_best[better] = jrel[better] + j0
        return None

    def cast_batch(self, origins, dirs, t_max=None):
        """Nearest hit for N rays. Returns (tri_idx int32 (-1 = miss), t)."""
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        n = len(o)
        t_best = np.full(n, np.inf)
        i_best = np.full(n, -1, dtype=np.int64)
        tm = None if t_max is None else np.asarray(t_max, dtype=np.float64).reshape(-1)
        chunk = max(1, int(self.MAX_PAIRS // max(n, 1)))
        for j0 in range(0, self.n_tris, chunk):
            self._intersect_chunk(o, d, j0, min(self.n_tris, j0 + chunk),
                                  t_best, i_best, False, tm)
        return i_best.astype(np.int32), t_best

    def occluded_batch(self, origins, dirs, t_max):
        """True where any triangle blocks the segment within t_max."""
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        tm = np.asarray(t_max, dtype=np.float64).reshape(-1)
        n = len(o)
        blocked = np.zeros(n, dtype=bool)
        chunk = max(1, int(self.MAX_PAIRS // max(n, 1)))
        for j0 in range(0, self.n_tris, chunk):
            live = ~blocked
            if not live.any():
                break
            hit = self._intersect_chunk(o[live], d[live], j0,
                                        min(self.n_tris, j0 + chunk),
                                        None, None, True, tm[live])
            idx = np.where(live)[0]
            blocked[idx[hit]] = True
        return blocked

    def cast(self, o, d, t_max=np.inf):
        """Single-ray convenience. Returns (tri, t, normal) or
        (-1, inf, None)."""
        i, t = self.cast_batch(np.asarray(o)[None], np.asarray(d)[None],
                               np.asarray([t_max]))
        if i[0] < 0:
            return -1, np.inf, None
        return int(i[0]), float(t[0]), self.normals[int(i[0])].copy()

    def segment_transmission(self, a, b, transmission_table):
        """Product of per-band transmission across every surface crossed by
        the open segment a->b (Vercidium permeation / muffling)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = b - a
        dist = float(np.linalg.norm(d))
        out = np.ones(N_BANDS, dtype=np.float64)
        if dist < 1e-6:
            return out, 0
        d /= dist
        o = a.copy()
        remaining = dist - _EPS
        crossings = 0
        for _ in range(256):
            if remaining <= _EPS:
                break
            tri, t, _n = self.cast(o, d, remaining)
            if tri < 0:
                break
            m = int(np.clip(self.tri_mat[tri], 0, len(transmission_table) - 1))
            out *= transmission_table[m]
            crossings += 1
            o = o + d * (t + _EPS)
            remaining -= (t + _EPS)
            if out.max() < 1e-7:
                break
        return out, crossings


# --------------------------------------------------------------------------
# Planar-patch extraction (image-source method)
# --------------------------------------------------------------------------

@dataclass
class Plane:
    normal: np.ndarray      # unit
    d: float                # plane: dot(n, x) = d
    material: int
    area: float
    tris: np.ndarray        # triangle indices belonging to this patch


def extract_planes(scene: SceneData, max_planes: int = 48, quant: float = 1e-3):
    """Group coplanar, same-material triangles into planar patches, the
    max_planes largest. Returns (planes, tri_plane), -1 = no patch."""
    tv = scene.tri_verts.astype(np.float64)
    e1 = tv[:, 1] - tv[:, 0]
    e2 = tv[:, 2] - tv[:, 0]
    ng = np.cross(e1, e2)
    area2 = np.linalg.norm(ng, axis=1)
    ok = area2 > 1e-12
    n = np.zeros_like(ng)
    n[ok] = ng[ok] / area2[ok, None]
    # canonical orientation so opposite windings merge
    flip = (n[:, 0] < -quant) | ((np.abs(n[:, 0]) <= quant) & (n[:, 1] < -quant)) \
        | ((np.abs(n[:, 0]) <= quant) & (np.abs(n[:, 1]) <= quant) & (n[:, 2] < 0))
    n[flip] *= -1.0
    d = np.einsum('ij,ij->i', n, tv[:, 0])

    keys = {}
    groups = []
    for i in range(scene.n_tris):
        if not ok[i]:
            continue
        key = (int(round(n[i, 0] / quant)), int(round(n[i, 1] / quant)),
               int(round(n[i, 2] / quant)), int(round(d[i] / (quant * 10))),
               int(scene.tri_mat[i]))
        g = keys.get(key)
        if g is None:
            keys[key] = len(groups)
            groups.append([i])
        else:
            groups[g].append(i)

    areas = 0.5 * area2
    planes = []
    for g in groups:
        idx = np.asarray(g, dtype=np.int64)
        a = float(areas[idx].sum())
        i0 = idx[0]
        planes.append(Plane(n[i0].copy(), float(d[i0]),
                            int(scene.tri_mat[i0]), a, idx))
    planes.sort(key=lambda p: -p.area)
    planes = planes[:max_planes]

    tri_plane = np.full(scene.n_tris, -1, dtype=np.int32)
    for pid, p in enumerate(planes):
        tri_plane[p.tris] = pid
    return planes, tri_plane


def surface_dims(scene: SceneData, quant: float = 1e-3, weld: float = 1e-4):
    """Per triangle (l, w) in metres of the SURFACE it belongs to: a connected patch of
    coplanar, same-material triangles (sharing a vertex). l = its largest extent along
    its principal in-plane axis, w = area / l (the rectangle of the same area and
    length: exact for a rectangle, narrower for irregular shapes). For the
    reflection-based scattering of Christensen & Rindel (Forum Acusticum 2005)."""
    tv = scene.tri_verts.astype(np.float64)
    nt = len(tv)
    ng = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area2 = np.linalg.norm(ng, axis=1)
    ok = area2 > 1e-12
    n = np.zeros_like(ng)
    n[ok] = ng[ok] / area2[ok, None]
    flip = (n[:, 0] < -quant) | ((np.abs(n[:, 0]) <= quant) & (n[:, 1] < -quant)) \
        | ((np.abs(n[:, 0]) <= quant) & (np.abs(n[:, 1]) <= quant) & (n[:, 2] < 0))
    n[flip] *= -1.0
    d = np.einsum('ij,ij->i', n, tv[:, 0])
    key = np.concatenate([np.round(n / quant), np.round(d / (quant * 10))[:, None],
                          np.asarray(scene.tri_mat, dtype=np.float64)[:, None]],
                         axis=1).astype(np.int64)
    _, plane_id = np.unique(key, axis=0, return_inverse=True)
    plane_id = plane_id.reshape(-1)
    _, vid = np.unique(np.round(tv.reshape(-1, 3) / weld).astype(np.int64), axis=0,
                       return_inverse=True)
    vid = vid.reshape(nt, 3)
    parent = np.arange(nt)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for t in np.flatnonzero(ok):
        for v in vid[t]:
            k = (int(plane_id[t]), int(v))
            if k in first:
                a, b = find(t), find(first[k])
                if a != b:
                    parent[a] = b
            else:
                first[k] = t
    root = np.array([find(i) for i in range(nt)])
    tri_l = np.zeros(nt, dtype=np.float32)
    tri_w = np.zeros(nt, dtype=np.float32)
    for r in np.unique(root[ok]):
        idx = np.flatnonzero((root == r) & ok)
        p = tv[idx].reshape(-1, 3)
        nn = n[idx[0]]
        ref = np.array([1.0, 0.0, 0.0]) if abs(nn[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - nn * np.dot(ref, nn)
        u /= np.linalg.norm(u)
        v = np.cross(nn, u)
        q = np.unique(np.round(np.stack([p @ u, p @ v], axis=1) / weld), axis=0) * weld
        q -= q.mean(axis=0)              # shared corners counted once (else the axis tilts)
        _, vec = np.linalg.eigh(q.T @ q)
        ext = np.ptp(q @ vec, axis=0)
        l = float(max(ext.max(), 1e-6))
        a = float(0.5 * area2[idx].sum())
        tri_l[idx] = l
        tri_w[idx] = min(a / l, l)
    return tri_l, tri_w
