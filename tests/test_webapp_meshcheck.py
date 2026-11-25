# -*- coding: utf-8 -*-
"""Web app geometry checks: watertight shell, air volume, inner open
surfaces, overlapping faces, small gaps and short edges against lambda/4,
exclusions, point-in-volume, wall distance, unit/axis import."""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import meshcheck  # noqa: E402
from webapp.server.store import Store  # noqa: E402


def _mesh(tri, layer=0):
    tri = np.asarray(tri, dtype=float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    return v, f, np.full(len(f), layer, dtype=np.int32)


def test_closed_box_volume_and_points():
    v, f, fl = _mesh(box_triangles((10.0, 7.0, 3.0))[0])
    s, closed = meshcheck.analyse(v, f, fl, ["walls"])
    assert s["check"]["watertight"] and s["check"]["solvers"]["wave"]
    assert abs(s["volume_m3"] - 210.0) < 1e-9
    tri = v[f]
    ins = meshcheck.inside([[5, 3.5, 1.5], [12, 3.5, 1.5]], tri[closed])
    assert ins.tolist() == [True, False]
    d = meshcheck.distance_to_surfaces([[5, 3.5, 1.5], [5, 3.5, 0.2]], tri)
    assert np.allclose(d, [1.5, 0.2])


def test_open_shell_disables_wave_solver():
    tri = box_triangles((4.0, 3.0, 2.5))[0][:-2]          # remove one face (2 triangles)
    v, f, fl = _mesh(tri)
    s, _ = meshcheck.analyse(v, f, fl, ["walls"])
    assert not s["check"]["watertight"] and s["volume_m3"] is None
    assert s["check"]["counts"]["open_edges"] > 0
    assert any(x["severity"] == "error" for x in s["check"]["findings"])


def test_inner_open_panel_is_allowed():
    shell = box_triangles((4.0, 3.0, 2.5))[0]
    panel = [[[1, 1, 0.75], [2, 1, 0.75], [2, 2, 0.75]], [[1, 1, 0.75], [2, 2, 0.75], [1, 2, 0.75]]]
    v, f, fl = _mesh(np.concatenate([shell, panel]))
    s, _ = meshcheck.analyse(v, f, fl, ["walls"])
    assert s["check"]["watertight"]
    assert abs(s["volume_m3"] - 30.0) < 1e-9
    assert {x["severity"] for x in s["check"]["findings"] if x["kind"] == "open_edge"} == {"info"}


def test_import_units_and_y_up():
    fd, path = tempfile.mkstemp(suffix=".obj")
    with os.fdopen(fd, "w") as fh:
        fh.write("v 0 0 0\nv 100 0 0\nv 0 200 0\nusemtl a\nf 1 2 3\n")
    v, f, fl, names = meshcheck.import_obj(path, unit="cm", up_axis="y")
    assert names == ["a"]
    assert np.allclose(v, [[0, 0, 0], [1, 0, 0], [0, 0, 2]])   # Y-up height -> Z


def test_volume_ignores_face_orientation():
    tri = np.array(box_triangles((3.0, 2.0, 2.5))[0], dtype=float)
    tri[::2] = tri[::2, ::-1]                       # flip every other triangle
    v, f, fl = _mesh(tri)
    s, _ = meshcheck.analyse(v, f, fl, ["walls"])
    assert abs(s["volume_m3"] - 15.0) < 1e-9


def _parts(*parts):
    """Triangle soups (one per layer) -> (verts, faces, face layers)."""
    tri = np.concatenate([np.asarray(p, dtype=float) for p in parts])
    fl = np.concatenate([np.full(len(p), k, dtype=np.int32) for k, p in enumerate(parts)])
    v = tri.reshape(-1, 3)
    return v, np.arange(len(v)).reshape(-1, 3), fl


def _quad(p0, p1, p2, p3):
    return [[p0, p1, p2], [p0, p2, p3]]


BOX = np.asarray(box_triangles((4.0, 3.0, 2.5))[0], dtype=float)
EAST = np.all(np.abs(BOX[:, :, 0] - 4.0) < 1e-9, axis=1)          # the wall x = 4


def _slit_box(width=0.05):
    y0, y1 = 1.5 - width / 2, 1.5 + width / 2
    wall = (_quad([4, 0, 0], [4, y0, 0], [4, y0, 2.5], [4, 0, 2.5])
            + _quad([4, y1, 0], [4, 3, 0], [4, 3, 2.5], [4, y1, 2.5]))
    return _parts(BOX[~EAST], wall)


def _kind(s, kind):
    return [x for x in s["check"]["findings"] if x["kind"] == kind]


def test_duplicated_wall_is_one_overlap_finding():
    v, f, fl = _parts(BOX, BOX[EAST])
    s, _ = meshcheck.analyse(v, f, fl, ["walls", "copy"])
    assert s["check"]["watertight"] and s["check"]["counts"]["overlapping_faces"] == 1
    (x,) = _kind(s, "overlapping_faces")
    assert x["overlap"] == "duplicate" and not x["contact"] and x["severity"] == "warning"
    assert abs(x["area_m2"] - 7.5) < 1e-9 and x["layers"] == ["copy", "walls"]
    assert np.allclose(x["centroid"], [4.0, 1.5, 1.25]) and len(x["triangles"]) == 2


def test_partial_overlap_and_solid_resting_on_the_floor():
    patch = _quad([1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0])         # a rug drawn on the floor
    v, f, fl = _parts(BOX, patch)
    s, _ = meshcheck.analyse(v, f, fl, ["walls", "rug"])
    (x,) = _kind(s, "overlapping_faces")
    assert x["overlap"] == "partial" and x["severity"] == "warning" and abs(x["area_m2"] - 1.0) < 1e-9
    block = np.asarray(box_triangles((1.0, 1.0, 1.0))[0], dtype=float) + [1.0, 1.0, 0.0]
    v, f, fl = _parts(BOX, block)
    s, _ = meshcheck.analyse(v, f, fl, ["walls", "cabinet"])
    (x,) = _kind(s, "overlapping_faces")
    assert x["contact"] and x["severity"] == "info"                   # back to back: harmless


def test_slit_is_judged_against_lambda_over_4():
    """A 5 cm slit: below lambda/4 at 250 Hz (34 cm) and 1000 Hz (8.6 cm), above it at
    2000 Hz (4.3 cm); the fixed 2 cm threshold without a transition does not flag it."""
    v, f, fl = _slit_box(0.05)
    closure = {}
    found = {}
    for fc in (None, 250, 1000, 2000):
        s, _ = meshcheck.analyse(v, f, fl, ["walls", "east"], closure=closure or None,
                                 closure_out=closure, transition_hz=fc)
        assert s["check"]["watertight"] and s["check"]["closure"]["method"] == "voxels"
        found[fc] = _kind(s, "small_gap")
        d = s["check"]["detail"]
        assert d["rule"] == ("fixed" if fc is None else "lambda/4")
        assert abs(d["threshold_m"] - (0.02 if fc is None else 343.0 / (4 * fc))) < 1e-4
    assert not found[None] and not found[2000]
    (lo,), (hi,) = found[250], found[1000]
    assert abs(lo["gap_m"] - 0.05) < 1e-6 and lo["gap_type"] == "slit" and lo["leak"]
    # the closure test (10 cm voxels) seals it; the 1000 Hz wave grid (dx 2.5 cm) does not
    assert lo["severity"] == "info" and hi["severity"] == "warning" and "leak" in hi["message"]
    assert lo["id"] == hi["id"]                                        # same finding


def test_air_layer_between_parallel_faces():
    panel = np.asarray(box_triangles((0.5, 1.0, 1.0))[0], dtype=float) + [4.0 - 0.53, 1.0, 0.5]
    v, f, fl = _parts(BOX, panel)
    s, _ = meshcheck.analyse(v, f, fl, ["walls", "panel"], transition_hz=1000)
    (x,) = _kind(s, "small_gap")
    assert x["gap_type"] == "layer" and abs(x["gap_m"] - 0.03) < 1e-6 and x["layers"] == ["panel", "walls"]
    s, _ = meshcheck.analyse(v, f, fl, ["walls", "panel"], transition_hz=3000)   # lambda/4 = 2.9 cm
    assert not _kind(s, "small_gap")


def test_short_edges_follow_lambda_over_4_and_skip_triangulation():
    stud = np.asarray(box_triangles((0.05, 0.05, 0.05))[0], dtype=float) + [2.0, 1.5, 1.0]
    v, f, fl = _parts(BOX, stud)
    for fc, n in ((None, 0), (1000, 12), (8000, 0)):     # 12 feature edges of 5 cm
        s, _ = meshcheck.analyse(v, f, fl, ["walls", "stud"], transition_hz=fc)
        assert s["check"]["counts"]["short_edges"] == n, (fc, s["check"]["counts"])
    # a finely triangulated flat wall has no "short edges": its diagonals are not detail
    grid = []
    for i in range(20):
        for j in range(20):
            grid += _quad([0.1 * i, 0.1 * j, 0], [0.1 * i + 0.1, 0.1 * j, 0],
                          [0.1 * i + 0.1, 0.1 * j + 0.1, 0], [0.1 * i, 0.1 * j + 0.1, 0])
    v, f, fl = _parts(grid)
    s, _ = meshcheck.analyse(v, f, fl, ["floor"], transition_hz=250)
    assert s["check"]["counts"]["short_edges"] == 80                   # its 80 border edges only


def test_exclusions_persist_and_never_close_the_model():
    tri = BOX[:-2]                                         # a face missing: not closed
    v, f, fl = _parts(tri)
    s, closed = meshcheck.analyse(v, f, fl, ["walls"])
    errors = [x for x in s["check"]["findings"] if x["severity"] == "error"]
    assert errors and not s["check"]["watertight"] and s["check"]["blocking"]
    store = Store(tempfile.mkdtemp())
    p = store.create_project("p")
    sp = store.create_space(p["id"], "s")
    g = store.create_geometry(sp["id"], dict(s, name="open box"), v, f, fl, ["walls"], closed,
                              b"", "box.obj")
    g["check_exclusions"] = {"findings": [errors[0]["id"]], "layers": []}
    store.save_geometry(g)
    ex = store.get_geometry(g["id"])["check_exclusions"]
    s2, _ = meshcheck.analyse(v, f, fl, ["walls"], exclusions=ex)
    c = s2["check"]
    first = next(x for x in c["findings"] if x["id"] == errors[0]["id"])
    assert first["excluded"] and first["excluded_by"] == "finding"
    assert c["excluded"]["open_edge"] == 1 and c["exclusions"]["findings"] == [errors[0]["id"]]
    # excluding never changes what the solvers accept
    assert not c["watertight"] and not c["solvers"]["wave"] and "excluded" in c["blocking"][0]
    s3, _ = meshcheck.analyse(v, f, fl, ["walls"], exclusions={"findings": [], "layers": ["walls"]})
    assert all(x["excluded"] for x in s3["check"]["findings"])
    assert s3["check"]["excluded"]["open_edge"] == s3["check"]["counts"]["open_edges"]
    assert not s3["check"]["watertight"]


def test_finding_ids_are_stable_across_rechecks_and_thresholds():
    v, f, fl = _slit_box(0.05)
    a, _ = meshcheck.analyse(v, f, fl, ["walls", "east"], transition_hz=250)
    b, _ = meshcheck.analyse(v, f, fl, ["walls", "east"], transition_hz=1000)
    ids = lambda s, k: sorted(x["id"] for x in _kind(s, k))         # noqa: E731
    assert ids(a, "open_edge") == ids(b, "open_edge") and ids(a, "small_gap") == ids(b, "small_gap")
    assert len(set(x["id"] for x in a["check"]["findings"])) == len(a["check"]["findings"])


def test_voxel_closure_keeps_the_outside_air():
    v, f, fl = _slit_box(0.05)
    closure = {}
    meshcheck.analyse(v, f, fl, ["walls", "east"], closure_out=closure)
    store = Store(tempfile.mkdtemp())
    p = store.create_project("p")
    sp = store.create_space(p["id"], "s")
    s, closed = meshcheck.analyse(v, f, fl, ["walls", "east"], closure=closure)
    g = store.create_geometry(sp["id"], dict(s, name="slit"), v, f, fl, ["walls", "east"], closed,
                              b"", "box.obj", closure=closure)
    back = store.load_closure(g["id"])
    assert np.array_equal(back["exterior"], closure["exterior"])
    assert not (back["exterior"] & back["interior"]).any()
