# -*- coding: utf-8 -*-
"""Sub-surfaces of the layers, for the material list (display only: materials stay per layer)."""
from __future__ import annotations

import os
import threading

import numpy as np

SMOOTH_DEG = 20.0       # same angle as the viewer's outline edges (EdgesGeometry threshold)
WELD_TOL = 1e-6         # m, as meshcheck: vertices closer than this are one vertex
_CACHE_MAX = 16
_cache: dict = {}
_cache_lock = threading.Lock()


def obj_groups(path):
    """(faces (n,3), label per triangle) of an OBJ, in the triangle order of the engine's
    read_obj (fan triangulation). The label joins the current `o` and `g` names."""
    nv, tris, labels = 0, [], []
    o = g = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                nv += 1
            elif t[0] == "o":
                o = " ".join(t[1:]) or None
            elif t[0] == "g":
                g = " ".join(t[1:]) or None
            elif t[0] == "f":
                idx = [int(w.split("/")[0]) for w in t[1:]]
                idx = [i - 1 if i > 0 else nv + i for i in idx]
                lab = " / ".join(x for x in (o, g) if x) or None
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
                    labels.append(lab)
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3), labels


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def smooth_patches(tri, weld, groups=None):
    """Patch label per triangle: triangles sharing an edge are joined when their planes are
    within SMOOTH_DEG (orientation ignored) and, given `groups`, when they are in the same
    group. Degenerate triangles join a neighbour without bridging two patches."""
    n = len(tri)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    p = tri.astype(float)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    unit = np.zeros_like(nrm)
    unit[ok] = nrm[ok] / ln[ok, None]
    w = weld
    nw = int(w.max()) + 1
    e = np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]])
    owner = np.tile(np.arange(n), 3)
    e.sort(axis=1)
    keep = e[:, 0] != e[:, 1]
    key = e[keep, 0] * nw + e[keep, 1]
    owner = owner[keep]
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    count = np.diff(np.r_[starts, len(key)])
    # candidate pairs: every pair of triangles sharing an edge (non-manifold edges: all pairs)
    pa, pb = [owner[starts[count == 2]]], [owner[starts[count == 2] + 1]]
    for s, c in zip(starts[count > 2], count[count > 2]):
        fs = owner[s:s + c]
        ii, jj = np.triu_indices(c, 1)
        pa.append(fs[ii])
        pb.append(fs[jj])
    a, b = np.concatenate(pa), np.concatenate(pb)
    m = a != b
    if groups is not None:
        code = {}
        gc = np.array([code.setdefault(x, len(code)) for x in groups])
        m &= gc[a] == gc[b]
    a, b = a[m], b[m]
    both = ok[a] & ok[b]
    flat = both & (np.abs(np.einsum("ij,ij->i", unit[a], unit[b]))
                   >= np.cos(np.radians(SMOOTH_DEG)))
    parent = list(range(n))
    for x, y in zip(a[flat].tolist(), b[flat].tolist()):
        rx, ry = _find(parent, x), _find(parent, y)
        if rx != ry:
            parent[ry] = rx
    late = list(zip(a[~both].tolist(), b[~both].tolist()))
    attached = set()
    for a, b in late:
        d, other = (a, b) if not ok[a] else (b, a)
        if d in attached or not ok[other]:
            continue
        attached.add(d)
        ra, rb = _find(parent, other), _find(parent, d)
        if ra != rb:
            parent[rb] = ra
    return np.array([_find(parent, i) for i in range(n)], dtype=np.int64)


def _orientation(tri):
    """'horizontal' | 'vertical' | 'sloped' | 'curved', whatever the triangle winding."""
    p = tri.astype(float)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])      # length = 2 x area
    ref = nrm[np.argmax(np.linalg.norm(nrm, axis=1))]
    s = np.where(nrm @ ref < 0, -1.0, 1.0)
    tot = np.linalg.norm(nrm, axis=1).sum()
    if tot <= 0:
        return None
    mean = (nrm * s[:, None]).sum(0) / tot
    m = np.linalg.norm(mean)
    if m < 0.98:
        return "curved"
    nz = abs(mean[2]) / m
    return "horizontal" if nz > 0.98 else "vertical" if nz < 0.17 else "sloped"


def _room_groups(source_path, room_faces, source_filename):
    """Group label per room triangle from the stored OBJ, or (None, why not)."""
    if not (source_filename or "").lower().endswith(".obj"):
        ext = os.path.splitext(source_filename or "")[1] or "unknown"
        return None, ("the source format (%s) carries no group the app reads: surfaces are "
                      "derived from the triangles" % ext)
    if not os.path.exists(source_path):
        return None, "no source model stored: surfaces are derived from the triangles"
    try:
        faces, labels = obj_groups(source_path)
    except (OSError, ValueError, IndexError) as exc:
        return None, "the source OBJ could not be re-read (%s): surfaces derived from the " \
                     "triangles" % exc
    if faces.shape != room_faces.shape or not np.array_equal(faces, room_faces):
        return None, ("the groups of the source OBJ do not match the stored triangles: "
                      "surfaces derived from the triangles only")
    if not any(labels):
        return None, "the source OBJ has no g/o groups: surfaces derived from the triangles"
    return labels, None


def compute(verts, faces, face_layer, names, owners=None, object_names=None,
            room_groups=None):
    """Sub-surfaces per layer: [{"name", "by", "subsurfaces": [...]}]. `triangles` index
    the layer's triangles in the order of GET .../mesh (np.flatnonzero(face_layer == i))."""
    q = np.round(np.asarray(verts, float) / WELD_TOL).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    v = np.asarray(verts, float)
    out = []
    for li, name in enumerate(names):
        sel = np.flatnonzero(face_layer == li)
        tri_idx = faces[sel]
        tri = v[tri_idx]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        own = None
        if owners is not None:
            own = [owners[k] for k in sel]
        if own is not None and own and all(o is not None for o in own):
            by = "instances"
            keys = own
        else:
            by = "patches"
            grp = None
            if room_groups is not None and all(k < len(room_groups) for k in sel):
                grp = [room_groups[k] for k in sel]
            keys = smooth_patches(tri, weld[tri_idx], grp).tolist()
        parts = {}
        for i, k in enumerate(keys):
            parts.setdefault(k, []).append(i)
        subs = []
        for k, idx in parts.items():
            idx = np.asarray(idx)
            sub = {"area_m2": round(float(area[idx].sum()), 3), "n_triangles": int(len(idx)),
                   "triangles": idx.tolist()}
            if by == "instances":
                sub.update(label=(object_names or {}).get(k, k), object_id=k, group=None,
                           orientation=None)
            else:
                g = room_groups[sel[idx[0]]] if room_groups is not None and grp else None
                sub.update(group=g, orientation=_orientation(tri[idx]))
            subs.append(sub)
        if by == "patches":
            # collinear triangles (exactly zero area) with no neighbour: one entry at the end
            zero = [s for s in subs if s["orientation"] is None]
            subs = [s for s in subs if s["orientation"] is not None]
            subs.sort(key=lambda s: (-s["area_m2"], s["triangles"][0]))
            for i, s in enumerate(subs):
                s["label"] = "Surface %d" % (i + 1)
            if zero:
                idx = sorted(i for s in zero for i in s["triangles"])
                subs.append({"label": "Zero-area triangles", "group": None, "orientation": None,
                             "area_m2": 0.0, "n_triangles": len(idx), "triangles": idx,
                             "degenerate": True})
        else:
            subs.sort(key=lambda s: s["label"])
        out.append({"name": name, "by": by,
                    "subsurfaces": [{k: s[k] for k in ("label", "group", "orientation", "object_id",
                                                       "degenerate", "area_m2", "n_triangles",
                                                       "triangles")
                                     if k in s} for s in subs]})
    return out


def for_geometry(store, gid):
    """The GET /api/geometries/{gid}/subsurfaces payload (cached per mesh file version)."""
    g = store.get_geometry(gid)
    d = store.geometry_dir(gid)
    files = [os.path.join(d, x) for x in ("mesh.npz", "mesh_assembled.npz", "source.obj")]
    stamp = tuple(os.path.getmtime(f) if os.path.exists(f) else None for f in files)
    names_by_id = {o["id"]: o.get("name") or o["id"] for o in g.get("objects") or []}
    key = (store.root, gid)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp and hit[1] == names_by_id:
            return hit[2]
    v, f, fl, _closed, names = store.load_mesh(gid)
    owners = store.load_owners(gid)
    _rv, rf, _rfl, _rn = store.load_room(gid)
    groups, why = _room_groups(files[2], rf, g.get("source_filename"))
    layers = compute(v, f, fl, names, owners, names_by_id, groups)
    note = ("Surfaces = triangles joined across edges flatter than %g deg (the outlines of the "
            "3D view)" % SMOOTH_DEG)
    if groups is not None:
        note += ", split where the g/o groups of the source OBJ change"
    note += ". Object layers list their placed instances. Materials stay assigned per layer."
    out = {"method": "smooth_patches" + ("+obj_groups" if groups is not None else ""),
           "angle_deg": SMOOTH_DEG, "groups_from_source": groups is not None,
           "note": note if why is None else note + " Groups: " + why + ".",
           "layers": layers}
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (stamp, names_by_id, out)
    return out
