# -*- coding: utf-8 -*-
"""Object library: realistic CC0 visual models for display, simplified
acoustic proxies for the solvers. Local object frame: x = width, y = depth
(front towards -y), z = height, origin at the floor, centre of the footprint."""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(HERE, "data", "objects_catalog.json")
VISUAL_DIR = os.path.join(os.path.dirname(HERE), "assets", "visual")
LAYER_PREFIX = "obj:"


def _box(x0, x1, y0, y1, z0, z1):
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[c[a], c[b], c[d]] for a, b, _, d in quads] +
                    [[c[b], c[cc], c[d]] for _, b, cc, d in quads])


def _quad(p0, p1, p2, p3):
    return np.array([[p0, p1, p2], [p0, p2, p3]], dtype=float)


def _extrude_l(w, d, h, hs, tb):
    """Closed L-shaped block (seat + backrest) extruded along x, no T-junctions."""
    yb = d / 2 - tb
    prof = [(-d / 2, 0), (d / 2, 0), (d / 2, hs), (d / 2, h), (yb, h), (yb, hs), (-d / 2, hs)]
    tris = []
    for i in range(len(prof)):
        (ya, za), (yc, zc) = prof[i], prof[(i + 1) % len(prof)]
        tris.extend(_quad((-w / 2, ya, za), (w / 2, ya, za), (w / 2, yc, zc), (-w / 2, yc, zc)))
    for x in (-w / 2, w / 2):
        P = {k: (x, y, z) for k, (y, z) in enumerate(prof)}
        tris.extend([[P[0], P[1], P[2]], [P[0], P[2], P[5]], [P[0], P[5], P[6]],
                     [P[5], P[2], P[3]], [P[5], P[3], P[4]]])
    return np.array(tris, dtype=float)


def proxy_parts(kind, dims):
    """Acoustic proxy of one object in its local frame: {part: triangles (n,3,3)}."""
    w, d, h = (max(float(v), 1e-3) for v in dims)
    if kind == "none":
        return {}
    if kind == "box":
        return {"body": _box(-w / 2, w / 2, -d / 2, d / 2, 0, h)}
    if kind == "bed":
        return {"mattress": _box(-w / 2, w / 2, -d / 2, d / 2, 0, min(h, 0.6))}
    if kind == "plane_top":
        return {"top": _quad((-w / 2, -d / 2, h), (w / 2, -d / 2, h), (w / 2, d / 2, h), (-w / 2, d / 2, h))}
    if kind == "vertical_plane":
        return {"surface": _quad((-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, 0, h), (-w / 2, 0, h))}
    if kind == "chair":
        hs, yb = min(0.47, 0.55 * h), 0.4 * d
        return {"seat": _quad((-w / 2, -d / 2, hs), (w / 2, -d / 2, hs), (w / 2, yb, hs), (-w / 2, yb, hs)),
                "back": _quad((-w / 2, yb, hs), (w / 2, yb, hs), (w / 2, yb, h), (-w / 2, yb, h))}
    if kind == "upholstered":
        return {"upholstery": _extrude_l(w, d, h, min(0.45, 0.5 * h), max(0.15, 0.25 * d))}
    raise ValueError("unknown proxy kind '%s'" % kind)


def load_catalog():
    """Catalog entries with their measured dims and visual model URLs when downloaded."""
    with open(CATALOG_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    index_path = os.path.join(VISUAL_DIR, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    out = []
    for e in data["objects"]:
        e = dict(e)
        vis = index.get(e["id"])
        has = bool(vis) and os.path.exists(os.path.join(VISUAL_DIR, e["id"], vis["gltf"]))
        e["visual"] = has
        e["license"] = "CC0 (Poly Haven)" if e["source"] == "polyhaven" else "Auralis"
        e["visual_scale"] = float(e.get("visual_scale", 1.0))
        if has:
            k = e["visual_scale"]           # corrects source models authored at a wrong scale
            e["dims"] = [round(k * x, 4) for x in vis["dims"]]
            e["visual_offset"] = [round(k * x, 5) for x in vis["offset"]]
            e["visual_url"] = "/api/objects/visual/%s/%s" % (e["id"], vis["gltf"])
            e["thumbnail_url"] = "/api/objects/visual/%s/thumbnail.png" % e["id"]
        else:
            e["visual_offset"] = e["visual_url"] = e["thumbnail_url"] = None
        e["acoustic"] = e["proxy"] != "none"
        out.append(e)
    return {"license_note": data.get("license_note", ""), "objects": out}


def layer_name(catalog_id, part):
    return "%s%s:%s" % (LAYER_PREFIX, catalog_id, part)


def instance_triangles(obj, entry):
    """World triangles per layer for one placed object (scale, Z rotation, translation)."""
    s = np.asarray(obj.get("scale") or [1, 1, 1], dtype=float)
    if s.shape != (3,) or (s <= 0).any() or not np.isfinite(s).all():
        raise ValueError("object %s: scale must be 3 positive numbers" % obj.get("id"))
    a = math.radians(float(obj.get("rotation_deg", 0.0)))
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    t = np.asarray(obj["position"], dtype=float)
    out = {}
    for part, tri in proxy_parts(entry["proxy"], entry["dims"]).items():
        out[layer_name(entry["id"], part)] = (tri * s) @ R.T + t
    return out


def assemble(room, objects, catalog):
    """Room mesh + every object's proxy -> (verts, faces, face_layer, names, owners)."""
    v, f, fl, names = room
    by_id = {e["id"]: e for e in catalog["objects"]}
    names = list(names)
    verts, faces, flay, owners = [v], [f], [fl], [None] * len(f)
    n = len(v)
    for obj in objects:
        entry = by_id.get(obj.get("catalog_id"))
        if entry is None:
            raise ValueError("unknown object type '%s'" % obj.get("catalog_id"))
        for lname, tri in instance_triangles(obj, entry).items():
            if lname not in names:
                names.append(lname)
            k = names.index(lname)
            verts.append(tri.reshape(-1, 3))
            faces.append(n + np.arange(len(tri) * 3).reshape(-1, 3))
            flay.append(np.full(len(tri), k, dtype=np.int32))
            owners.extend([obj["id"]] * len(tri))
            n += len(tri) * 3
    return (np.concatenate(verts), np.concatenate(faces), np.concatenate(flay).astype(np.int32),
            names, owners)


def default_layer_settings(catalog, layer):
    """Catalog default material/scattering of an object layer ('obj:<id>:<part>')."""
    _, cid, part = layer.split(":", 2)
    for e in catalog["objects"]:
        if e["id"] == cid:
            p = e["parts"].get(part, {})
            return {"material_id": p.get("material"), "scattering": p.get("scattering")}
    return {"material_id": None, "scattering": None}
