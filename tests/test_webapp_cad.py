# -*- coding: utf-8 -*-
"""DXF and Rhino 3DM import: a closed box written with each library is read back as a
closed room with its layers and its exact volume (skipped if the libraries are absent)."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import meshcheck  # noqa: E402


def _box():
    tri, names = box_triangles((5.0, 4.0, 3.0))
    return np.asarray(tri, float), names


def test_dxf_box_with_3dfaces_mesh_and_blocks():
    ezdxf = pytest.importorskip("ezdxf")
    tri, names = _box()
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    blk = doc.blocks.new("CEIL")
    for t, n in zip(tri, names):
        if n == "ceiling":                               # through a block, offset by INSERT
            blk.add_3dface([tuple(p - [0, 0, 1.0]) for p in t] + [tuple(t[2] - [0, 0, 1.0])],
                           dxfattribs={"layer": "ceiling"})
        elif n == "floor":
            m = msp.add_mesh(dxfattribs={"layer": "floor"})
            with m.edit_data() as md:
                md.vertices = [tuple(p) for p in t]
                md.faces = [(0, 1, 2)]
        else:
            msp.add_3dface([tuple(p) for p in t] + [tuple(t[2])], dxfattribs={"layer": "walls"})
    msp.add_blockref("CEIL", (0, 0, 1.0))
    msp.add_line((0, 0, 0), (1, 1, 1))
    path = os.path.join(tempfile.mkdtemp(), "box.dxf")
    doc.saveas(path)
    v, f, fl, lay, notes = meshcheck.import_model(path, "m", "z")
    assert sorted(lay) == ["ceiling", "floor", "walls"]
    s, _closed = meshcheck.analyse(v, f, fl, lay)
    assert s["check"]["watertight"] and abs(s["volume_m3"] - 60.0) < 1e-6
    assert any("LINE x1" in n for n in notes)


def test_3dm_box_mesh_by_layer():
    r3 = pytest.importorskip("rhino3dm")
    tri, names = _box()
    model = r3.File3dm()
    model.Settings.ModelUnitSystem = r3.UnitSystem.Meters
    idx = {}
    for n in ("floor", "rest"):
        lay = r3.Layer()
        lay.Name = n
        idx[n] = model.Layers.Add(lay)
    for key, sel in (("floor", [n == "floor" for n in names]), ("rest", [n != "floor" for n in names])):
        mesh = r3.Mesh()
        for t in tri[np.asarray(sel)]:
            a = mesh.Vertices.Add(*t[0])
            b = mesh.Vertices.Add(*t[1])
            c = mesh.Vertices.Add(*t[2])
            mesh.Faces.AddFace(a, b, c)
        attr = r3.ObjectAttributes()
        attr.LayerIndex = idx[key]
        model.Objects.AddMesh(mesh, attr)
    path = os.path.join(tempfile.mkdtemp(), "box.3dm")
    assert model.Write(path, 7)
    v, f, fl, lay, notes = meshcheck.import_model(path, "m", "z")
    assert sorted(lay) == ["floor", "rest"]
    s, _closed = meshcheck.analyse(v, f, fl, lay)
    assert s["check"]["watertight"] and abs(s["volume_m3"] - 60.0) < 1e-6
