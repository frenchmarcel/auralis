# -*- coding: utf-8 -*-
"""Object library: acoustic proxies are closed or open as intended, keep their
volume, follow the placement transform, and leave the room watertight."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import meshcheck, objects  # noqa: E402


def _analyse(tri, names=("x",)):
    tri = np.asarray(tri, dtype=float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    return meshcheck.analyse(v, f, np.zeros(len(f), np.int32), list(names))[0]


def test_closed_proxies_have_exact_volume():
    w, d, h = 2.0, 0.9, 0.85
    hs, tb = min(0.45, 0.5 * h), max(0.15, 0.25 * d)
    cases = {"box": w * d * h, "bed": w * d * min(h, 0.6),
             "upholstered": w * (d * hs + tb * (h - hs))}
    for kind, vol in cases.items():
        s = _analyse(np.concatenate(list(objects.proxy_parts(kind, (w, d, h)).values())))
        assert s["check"]["watertight"], kind
        assert abs(s["volume_m3"] - vol) < 6e-4, kind      # summary rounds to 1e-3 m3


def test_open_proxies_and_visual_only():
    for kind in ("chair", "plane_top", "vertical_plane"):
        s = _analyse(np.concatenate(list(objects.proxy_parts(kind, (1.0, 0.5, 0.9)).values())))
        assert not s["check"]["watertight"], kind
    assert objects.proxy_parts("none", (1, 1, 1)) == {}


def test_placement_transform():
    entry = {"id": "t", "proxy": "box", "dims": [2.0, 1.0, 0.5], "parts": {}}
    obj = {"id": "O1", "position": [5.0, 3.0, 0.2], "rotation_deg": 90, "scale": [1, 1, 2]}
    tri = objects.instance_triangles(obj, entry)["obj:t:body"].reshape(-1, 3)
    lo, hi = tri.min(0), tri.max(0)
    assert np.allclose(lo, [4.5, 2.0, 0.2]) and np.allclose(hi, [5.5, 4.0, 1.2])


def test_furnished_room_stays_watertight_and_loses_object_volume():
    room = box_triangles((6.0, 5.0, 3.0))[0]
    v = np.asarray(room, float).reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.zeros(len(f), np.int32)
    catalog = {"objects": [
        {"id": "cab", "proxy": "box", "dims": [1.0, 0.5, 2.0], "parts": {}},
        {"id": "tab", "proxy": "plane_top", "dims": [1.6, 0.9, 0.75], "parts": {}},
    ]}
    objs = [{"id": "O1", "catalog_id": "cab", "position": [1.0, 1.0, 0.0]},
            {"id": "O2", "catalog_id": "tab", "position": [3.0, 2.5, 0.0]}]
    va, fa, fla, names, owners = objects.assemble((v, f, fl, ["walls"]), objs, catalog)
    s, closed = meshcheck.analyse(va, fa, fla, names)
    assert s["check"]["watertight"]
    assert abs(s["volume_m3"] - (90.0 - 1.0)) < 1e-9
    assert names == ["walls", "obj:cab:body", "obj:tab:top"]
    assert owners[:len(f)] == [None] * len(f) and set(owners[len(f):]) == {"O1", "O2"}
    ins = meshcheck.inside([[1.0, 1.0, 1.0], [3.0, 2.5, 1.5]], va[fa][closed])
    assert ins.tolist() == [False, True]          # inside the cabinet = not in the air
