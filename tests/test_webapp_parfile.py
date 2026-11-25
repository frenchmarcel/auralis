# -*- coding: utf-8 -*-
"""Odeon .Par reader and voxel closure: mirror convention, constants and degree
trigonometry, LSurf layer names, NumbOffSet Auto PtAbsRef, UTF-16 files, loud
refusal of parametric models, and T-junction rooms recognised as closed."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import meshcheck, parfile  # noqa: E402

# A 10 x 6 x 3 m box written the Odeon way: half of it (y >= 0) plus mirrored
# surfaces (negative numbers), coordinates with constants and TanD().
BOX = """### test box
Const L 10
Const H 3*TanD(45)
Pt 1 0 0 0
Pt 2 L 0 0
Pt 3 L 3 0
Pt 4 0 3 0
Pt 5 0 0 H
Pt 6 L 0 H
Pt 7 L 3 H
Pt 8 0 3 H
Some free text line from the author = 42
LSurf -1 1 Floor
1 2 3 4
LSurf -2 1 Ceiling
5 6 7 8
LSurf -3 1 Walls
4 3 7 8
LSurf -4 1 Walls
1 4 8 5
LSurf -5 1 Walls
2 3 7 6
"""


def _write(text, encoding="utf-8"):
    fd, path = tempfile.mkstemp(suffix=".Par")
    with os.fdopen(fd, "wb") as fh:
        fh.write(text.encode(encoding))
    return path


def test_mirrored_box_is_closed_with_exact_volume():
    v, f, fl, names, rep = parfile.read_par(_write(BOX))
    assert names == ["Floor", "Ceiling", "Walls"]
    assert rep["ignored_lines"] == 1 and rep["ignored_words"] == ["Some"]
    lo, hi = v.min(0), v.max(0)
    assert np.allclose(lo, [0, -3, 0]) and np.allclose(hi, [10, 3, 3])
    s, _ = meshcheck.analyse(v, f, fl, names)
    assert s["check"]["watertight"] and s["check"]["closure"]["method"] == "topology"
    assert abs(s["volume_m3"] - 180.0) < 1e-6


def test_utf16_files_are_read():
    v, f, _fl, names, _ = parfile.read_par(_write("﻿" + BOX, "utf-16-le"))
    assert len(f) > 0 and "Walls" in names


def test_numboffset_auto_with_absolute_refs():
    text = BOX + """NumbOffSet Auto PtAbsRef
LSurf 9 1 Panel
1 2 6 5
"""
    _v, f, _fl, names, rep = parfile.read_par(_write(text))
    assert "Panel" in names and rep["offset_fallbacks"] == 0


def test_parametric_models_are_refused_loudly():
    with pytest.raises(ValueError, match="parametric commands not supported"):
        parfile.read_par(_write("For i 1 3\nPt i 0 0 0\nEnd\n"))


def test_t_junction_room_is_closed_at_the_voxel_scale():
    # floor split in two quads whose shared edge ends mid-way on the long walls:
    # edges no longer match vertex to vertex, but the room is closed
    L, W, H = 8.0, 5.0, 3.0
    quads = [
        [(0, 0, 0), (4, 0, 0), (4, W, 0), (0, W, 0)], [(4, 0, 0), (L, 0, 0), (L, W, 0), (4, W, 0)],
        [(0, 0, H), (L, 0, H), (L, W, H), (0, W, H)],
        [(0, 0, 0), (L, 0, 0), (L, 0, H), (0, 0, H)], [(0, W, 0), (L, W, 0), (L, W, H), (0, W, H)],
        [(0, 0, 0), (0, W, 0), (0, W, H), (0, 0, H)], [(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)],
    ]
    tri = []
    for q in quads:
        tri += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    tri = np.array(tri, dtype=float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.zeros(len(f), np.int32)
    s, _ = meshcheck.analyse(v, f, fl, ["room"])
    assert s["check"]["counts"]["open_edges"] > 0                 # topology says "open"...
    assert s["check"]["watertight"] and s["check"]["closure"]["method"] == "voxels"
    assert abs(s["volume_m3"] - L * W * H) / (L * W * H) < 0.05  # ...voxels say closed, volume ok
    assert {x["severity"] for x in s["check"]["findings"] if x["kind"] == "open_edge"} == {"info"}
