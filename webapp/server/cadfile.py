# -*- coding: utf-8 -*-
"""CAD imports: DXF (ezdxf) and Rhino 3DM (rhino3dm), both MIT. Faces become triangles,
CAD layers become assignment layers; unsupported entities are counted, never dropped
silently."""
from __future__ import annotations

import collections

import numpy as np

DXF_UNITS = {1: "inch", 2: "feet", 4: "mm", 5: "cm", 6: "m"}


def _fan(poly):
    """Triangles of a planar polygon given as a vertex list (fan; quads split)."""
    pts = [tuple(map(float, p)) for p in poly]
    dedup = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup = dedup[:-1]
    return [(dedup[0], dedup[k], dedup[k + 1]) for k in range(1, len(dedup) - 1)]


def _collect(tris_by_layer):
    names = sorted(tris_by_layer)
    tris, fl = [], []
    for i, n in enumerate(names):
        tris += tris_by_layer[n]
        fl += [i] * len(tris_by_layer[n])
    if not tris:
        raise ValueError("no surface found in the file")
    v = np.asarray(tris, dtype=np.float64).reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    return v, f, np.asarray(fl, np.int32), names


def read_dxf(path):
    """(verts, faces, face_layer, names, report): 3DFACE, MESH, POLYFACE / POLYMESH
    polylines, inside INSERTed blocks too (exploded with their transforms)."""
    import ezdxf
    doc = ezdxf.readfile(path)
    by_layer = collections.defaultdict(list)
    skipped = collections.Counter()

    def visit(entities, depth=0):
        for e in entities:
            t = e.dxftype()
            layer = e.dxf.get("layer", "0") or "0"
            if t == "INSERT" and depth < 16:
                visit(e.virtual_entities(), depth + 1)
            elif t == "3DFACE":
                by_layer[layer] += _fan([e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2, e.dxf.vtx3])
            elif t == "MESH":
                vs = [tuple(map(float, p)) for p in e.vertices]
                for face in e.faces:
                    by_layer[layer] += _fan([vs[i] for i in face])
            elif t == "POLYLINE" and (e.is_poly_face_mesh or e.is_polygon_mesh):
                from ezdxf.render import MeshBuilder
                mb = MeshBuilder.from_polyface(e)
                vs = [tuple(map(float, p)) for p in mb.vertices]
                for face in mb.faces:
                    by_layer[layer] += _fan([vs[i] for i in face])
            elif t in ("LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "TEXT", "MTEXT",
                       "DIMENSION", "HATCH", "POINT", "SPLINE", "3DSOLID", "BODY", "REGION",
                       "SURFACE", "SOLID", "ELLIPSE", "LEADER", "VIEWPORT"):
                skipped[t] += 1
            else:
                skipped[t] += 1

    visit(doc.modelspace())
    v, f, fl, names = _collect(by_layer)
    units = DXF_UNITS.get(int(doc.header.get("$INSUNITS", 0)))
    return v, f, fl, names, {"skipped": dict(skipped), "file_unit": units}


def read_3dm(path):
    """(verts, faces, face_layer, names, report): meshes, and the render meshes stored
    with Breps / extrusions (Rhino saves them unless 'save small' is used)."""
    import rhino3dm as r3
    model = r3.File3dm.Read(path)
    if model is None:
        raise ValueError("not a readable .3dm file")
    layers = {i: (lay.FullPath or lay.Name or "Layer %d" % i)
              for i, lay in enumerate(model.Layers)}
    by_layer = collections.defaultdict(list)
    skipped = collections.Counter()
    no_mesh = 0

    def add_mesh(mesh, layer):
        vs = [(p.X, p.Y, p.Z) for p in mesh.Vertices]
        for k in range(len(mesh.Faces)):
            a, b, c, d = mesh.Faces[k]
            by_layer[layer] += _fan([vs[a], vs[b], vs[c]] + ([vs[d]] if d != c else []))

    for obj in model.Objects:
        g = obj.Geometry
        layer = layers.get(obj.Attributes.LayerIndex, "Default")
        kind = g.ObjectType
        if kind == r3.ObjectType.Mesh:
            add_mesh(g, layer)
        elif kind in (r3.ObjectType.Brep, r3.ObjectType.Extrusion):
            brep = g if kind == r3.ObjectType.Brep else g.ToBrep(True)
            got = False
            for face in (brep.Faces if brep is not None else []):
                m = face.GetMesh(r3.MeshType.Any)
                if m is not None and len(m.Faces):
                    add_mesh(m, layer)
                    got = True
            if not got:
                no_mesh += 1
        else:
            skipped[str(kind).split(".")[-1]] += 1
    if no_mesh:
        skipped["Brep/Extrusion without render mesh"] = no_mesh
    v, f, fl, names = _collect(by_layer)
    unit = str(model.Settings.ModelUnitSystem).split(".")[-1].lower()
    return v, f, fl, names, {"skipped": dict(skipped),
                             "file_unit": {"millimeters": "mm", "centimeters": "cm",
                                           "meters": "m", "inches": "inch",
                                           "feet": "feet"}.get(unit, unit)}
