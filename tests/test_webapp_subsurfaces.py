# -*- coding: utf-8 -*-
"""Sub-surfaces of the layer list: smooth patches cover every triangle once, split at
sharp edges and at the source OBJ's groups, and object layers list their instances."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import meshcheck, subsurfaces  # noqa: E402
from webapp.server.store import Store  # noqa: E402


def _box_mesh(size=(6.0, 5.0, 3.0)):
    v = np.asarray(box_triangles(size)[0], float).reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    return v, f


def test_box_faces_are_six_surfaces():
    v, f = _box_mesh()
    out = subsurfaces.compute(v, f, np.zeros(len(f), np.int32), ["walls"])
    subs = out[0]["subsurfaces"]
    assert out[0]["by"] == "patches" and len(subs) == 6
    assert sorted(i for s in subs for i in s["triangles"]) == list(range(len(f)))
    assert abs(sum(s["area_m2"] for s in subs) - 2 * (30 + 18 + 15)) < 1e-6
    assert [s["label"] for s in subs] == ["Surface %d" % k for k in range(1, 7)]
    assert subs[0]["area_m2"] >= subs[-1]["area_m2"]
    orient = sorted(s["orientation"] for s in subs)
    assert orient == ["horizontal"] * 2 + ["vertical"] * 4


def test_triangles_index_the_layer_in_mesh_order():
    v, f = _box_mesh()
    fl = np.zeros(len(f), np.int32)
    fl[::3] = 1                                    # interleaved layers
    out = subsurfaces.compute(v, f, fl, ["a", "b"])
    for li, layer in enumerate(out):
        n = int((fl == li).sum())
        assert sorted(i for s in layer["subsurfaces"] for i in s["triangles"]) == list(range(n))


def test_smooth_curve_stays_one_surface_and_zero_area_is_grouped():
    # a half cylinder in 12 facets (15 deg apart) plus a collinear sliver
    a = np.linspace(0, np.pi, 13)
    ring = np.stack([np.cos(a), np.sin(a)], 1)
    v = np.concatenate([np.c_[ring, np.zeros(13)], np.c_[ring, np.ones(13)],
                        [[5.0, 0, 0], [6.0, 0, 0], [7.0, 0, 0]]])
    f = [(i, i + 1, i + 14) for i in range(12)] + [(i, i + 14, i + 13) for i in range(12)]
    f = np.asarray(f + [(26, 27, 28)])
    out = subsurfaces.compute(v, f, np.zeros(len(f), np.int32), ["vault"])[0]["subsurfaces"]
    assert len(out) == 2 and out[0]["orientation"] == "curved" and out[0]["n_triangles"] == 24
    assert out[1]["degenerate"] and out[1]["triangles"] == [24]


def test_obj_groups_split_and_align(tmp_path):
    v, f = _box_mesh((2.0, 2.0, 2.0))
    lines = ["v %.3f %.3f %.3f" % tuple(p) for p in v] + ["usemtl walls"]
    for k, t in enumerate(f):
        if k % 2 == 0:
            lines.append("g %s" % ("left" if k < len(f) // 2 else "right"))
        lines.append("f %d %d %d" % tuple(t + 1))
    src = tmp_path / "box.obj"
    src.write_text("\n".join(lines) + "\n")
    v2, f2, fl2, names = meshcheck.import_obj(str(src))
    faces, labels = subsurfaces.obj_groups(str(src))
    assert np.array_equal(faces, f2) and len(labels) == len(f2)
    out = subsurfaces.compute(v2, f2, fl2, names, room_groups=labels)[0]["subsurfaces"]
    assert {s["group"] for s in out} == {"left", "right"}
    assert sum(s["n_triangles"] for s in out) == len(f2)


def test_for_geometry_with_store_and_objects(tmp_path):
    store = Store(str(tmp_path))
    p = store.create_project("P")
    s = store.create_space(p["id"], "S")
    v, f = _box_mesh()
    fl = np.zeros(len(f), np.int32)
    obj = "".join("v %.3f %.3f %.3f\n" % tuple(x) for x in v) + "usemtl walls\n" + \
        "".join("f %d %d %d\n" % tuple(t + 1) for t in f)
    g = store.create_geometry(s["id"], {"name": "G", "layers": [], "objects": []}, v, f, fl,
                              ["walls"], np.ones(len(f), bool), obj.encode(), "box.obj")
    out = subsurfaces.for_geometry(store, g["id"])
    assert not out["groups_from_source"] and "no g/o groups" in out["note"]
    assert len(out["layers"][0]["subsurfaces"]) == 6
    # an object layer appended after the room: one sub-surface per placed instance
    tri = np.asarray(box_triangles((0.5, 0.5, 0.5))[0], float)
    v2 = np.concatenate([v, tri.reshape(-1, 3) + 1.0, tri.reshape(-1, 3) + 2.0])
    n0, nt = len(v), len(tri)
    f2 = np.concatenate([f, n0 + np.arange(2 * nt * 3).reshape(-1, 3)])
    fl2 = np.r_[fl, np.ones(2 * nt, np.int32)]
    owners = [None] * len(f) + ["O1"] * nt + ["O2"] * nt
    store.save_assembled(g["id"], v2, f2, fl2, np.ones(len(f2), bool),
                         ["walls", "obj:box:body"], owners)
    g["objects"] = [{"id": "O1", "name": "Box A"}, {"id": "O2", "name": "Box B"}]
    store.save_geometry(g)
    out = subsurfaces.for_geometry(store, g["id"])
    objl = out["layers"][1]
    assert objl["by"] == "instances"
    assert [x["label"] for x in objl["subsurfaces"]] == ["Box A", "Box B"]
    assert objl["subsurfaces"][1]["triangles"] == list(range(nt, 2 * nt))
    assert len(out["layers"][0]["subsurfaces"]) == 6
