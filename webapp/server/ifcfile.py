# -*- coding: utf-8 -*-
"""IFC import (IfcOpenShell, LGPL): building elements triangulated in world coordinates,
one layer per class and material; or one IfcSpace as the acoustic volume, its faces
labelled by the nearest building element (a closed model by construction)."""
from __future__ import annotations

import collections

import numpy as np

# classes that are not room surfaces by default (the user can include them)
DEFAULT_EXCLUDE = ("IfcSpace", "IfcOpeningElement", "IfcAnnotation", "IfcGrid",
                   "IfcVirtualElement", "IfcSite", "IfcBuildingElementProxy")


def _settings():
    import ifcopenshell.geom
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    return st


def _layer(el):
    import ifcopenshell.util.element as ue
    mat = ue.get_material(el)
    name = getattr(mat, "Name", None) if mat is not None else None
    if mat is not None and name is None and mat.is_a("IfcMaterialLayerSetUsage"):
        name = getattr(mat.ForLayerSet, "LayerSetName", None)
    return el.is_a() + (" / %s" % name if name else "")


def _shape_tris(shape):
    v = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
    return v[f]


def _nearest(a, b, chunk=256):
    """Index of the nearest point of b for every point of a (numpy only)."""
    out = np.empty(len(a), dtype=np.int64)
    bb = (b * b).sum(axis=1)
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk]
        d = bb[None, :] - 2.0 * x @ b.T
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def inventory(path):
    """Classes with geometry and their counts, and the spaces (GlobalId, name)."""
    import ifcopenshell
    model = ifcopenshell.open(path)
    counts = collections.Counter(p.is_a() for p in model.by_type("IfcProduct")
                                 if getattr(p, "Representation", None) is not None)
    spaces = [{"id": s.GlobalId, "name": s.LongName or s.Name or s.GlobalId}
              for s in model.by_type("IfcSpace") if s.Representation is not None]
    return {"classes": dict(counts), "spaces": spaces, "schema": model.schema}


def read_ifc(path, exclude=DEFAULT_EXCLUDE, space=None):
    """(verts, faces, face_layer, names, report). space = an IfcSpace GlobalId: that
    space's closed shell, each face labelled by the nearest included element's layer."""
    import ifcopenshell
    import ifcopenshell.geom
    model = ifcopenshell.open(path)
    st = _settings()
    by_layer = collections.defaultdict(list)
    failed = collections.Counter()
    space_tris = None
    excl = set(exclude or ())
    for el in model.by_type("IfcProduct"):
        if getattr(el, "Representation", None) is None:
            continue
        cls = el.is_a()
        if space is not None and cls == "IfcSpace" and el.GlobalId == space:
            space_tris = _shape_tris(ifcopenshell.geom.create_shape(st, el))
            continue
        if cls in excl:
            continue
        try:
            tris = _shape_tris(ifcopenshell.geom.create_shape(st, el))
        except Exception:
            failed[cls] += 1
            continue
        by_layer[_layer(el)].append(tris)
    report = {"failed": dict(failed), "space": None}
    if space is not None:
        if space_tris is None:
            raise ValueError("IfcSpace %s not found (or without geometry)" % space)
        names = sorted(by_layer) or ["IfcSpace"]
        if by_layer:
            el_tris = np.concatenate([np.concatenate(by_layer[n]) for n in names])
            el_lab = np.concatenate([np.full(sum(len(t) for t in by_layer[n]), i)
                                     for i, n in enumerate(names)])
            fl = el_lab[_nearest(space_tris.mean(axis=1), el_tris.mean(axis=1))].astype(np.int32)
        else:
            fl = np.zeros(len(space_tris), np.int32)
        used = np.unique(fl)
        remap = {int(u): i for i, u in enumerate(used)}
        fl = np.array([remap[int(x)] for x in fl], np.int32)
        names = [names[int(u)] for u in used]
        v = space_tris.reshape(-1, 3)
        report["space"] = space
        return v, np.arange(len(v)).reshape(-1, 3), fl, names, report
    names = sorted(by_layer)
    if not names:
        raise ValueError("no building element with geometry (check the excluded classes)")
    tris = np.concatenate([np.concatenate(by_layer[n]) for n in names])
    fl = np.concatenate([np.full(sum(len(t) for t in by_layer[n]), i)
                         for i, n in enumerate(names)]).astype(np.int32)
    v = tris.reshape(-1, 3)
    return v, np.arange(len(v)).reshape(-1, 3), fl, names, report
