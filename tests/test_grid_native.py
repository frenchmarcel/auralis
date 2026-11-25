# -*- coding: utf-8 -*-
"""Native grid builder (aur_grid.cpp) against the Python reference: bit for bit."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import conformal as CF  # noqa: E402
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine import gridnative  # noqa: E402
from auralis.engine.geometry import SceneData, add_box, make_shoebox  # noqa: E402

MAT = {"absorption": np.full(7, 0.12), "scattering": np.full(7, 0.1),
       "transmission": np.zeros(7)}


def _rot(sc, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc.tri_verts = np.ascontiguousarray(sc.tri_verts @ R.T, dtype=np.float32)
    return R


def _room():
    sc = make_shoebox((4.3, 3.1, 2.6), wall_material=MAT, floor_material=MAT,
                      ceiling_material=MAT)
    sc = add_box(sc, (0.9, 0.6, 0.8), (1.4, 0.9, 0.0), MAT)
    sc = add_box(sc, (0.4, 0.4, 1.5), (3.2, 2.2, 0.0), MAT)
    R = _rot(sc, 23.0)
    return sc, R @ np.array([0.8, 0.7, 1.4]), [R @ np.array([3.4, 2.4, 1.1])]


def _street():
    g = np.array([[[-8, -5, 0], [8, -5, 0], [8, 5, 0]], [[-8, -5, 0], [8, 5, 0], [-8, 5, 0]]])
    sc = SceneData(g, np.zeros(2, np.int32), MAT["absorption"][None], MAT["scattering"][None],
                   MAT["transmission"][None])
    for x in (-7.0, -1.5, 3.5):
        sc = add_box(sc, (4.0, 3.0, 5.0 + x % 3), (x, 2.0, 0.0), MAT)
    sc = add_box(sc, (2.2, 1.0, 1.2), (-2.0, -1.5, 0.0), MAT)
    R = _rot(sc, 7.0)
    return sc, R @ np.array([-5.0, -0.5, 1.2]), [R @ np.array([5.0, -1.0, 1.6])]


def _grids(sc, src, rcvs, dx, native, margin=2):
    os.environ["AURALIS_GRID_NATIVE"] = "1" if native else "0"
    os.environ["AURALIS_FDTD_GRID_CACHE"] = "0"
    try:
        vox = F.voxelize(sc, dx, src, margin_cells=margin, include_points=[src] + rcvs)
        W, _idx, st, ca = CF.wall_face_areas(sc, vox)
        frac, fac = CF.fine_air_fractions(sc, vox)
        phi, fw, cst = CF.conformal_corrections(sc, vox)
    finally:
        os.environ.pop("AURALIS_GRID_NATIVE", None)
        os.environ.pop("AURALIS_FDTD_GRID_CACHE", None)
    return vox, W, st, ca, frac, fac, phi, fw, cst


@pytest.mark.parametrize("scene, dx, margin", [(_room, 0.09, 2), (_street, 0.25, 16)])
def test_native_grid_equals_the_python_reference(scene, dx, margin):
    if gridnative.lib() is None:
        pytest.skip("libauralis without the grid builder (ABI < 2.4)")
    sc, src, rcvs = scene()
    a = _grids(sc, src, rcvs, dx, native=False, margin=margin)
    b = _grids(sc, src, rcvs, dx, native=True, margin=margin)
    for name in ("solid", "mat", "nrmq", "air"):
        assert np.array_equal(getattr(a[0], name), getattr(b[0], name)), name
    assert a[0].solid.any() and a[0].air.sum() > 1000
    for k, name in ((1, "W"), (3, "cell_area"), (4, "frac"), (6, "phi"), (7, "face_w")):
        assert np.array_equal(a[k], b[k]), name
    assert a[2] == b[2] and a[5] == b[5] and a[8] == b[8]
    assert a[2]["area_true_m2"] > 1.0 and (a[6] > 1.0).any()


def test_native_flood_keeps_the_source_component():
    """Two rooms that do not communicate: the air is the one holding the source."""
    if gridnative.lib() is None:
        pytest.skip("libauralis without the grid builder (ABI < 2.4)")
    sc = make_shoebox((3.0, 2.0, 2.0), wall_material=MAT, floor_material=MAT,
                      ceiling_material=MAT)
    tv = np.concatenate([sc.tri_verts, sc.tri_verts + np.float32([4.0, 0.0, 0.0])])
    two = SceneData(tv, np.concatenate([sc.tri_mat, sc.tri_mat]), sc.absorption,
                    sc.scattering, sc.transmission)
    solid, _m, _q = gridnative.voxelize_tris(two.tri_verts, two.tri_mat,
                                             np.array([-0.3, -0.3, -0.3]), 0.1, (77, 27, 27))
    air_a = gridnative.flood(solid, (12, 10, 10))
    air_b = gridnative.flood(solid, (52, 10, 10))
    assert air_a[12, 10, 10] and not air_a[52, 10, 10]
    assert air_b[52, 10, 10] and not air_b[12, 10, 10]
    assert not (air_a & solid).any()


def _grids_of_run(case, native, monkeypatch):
    """The FdtdGrid objects a run_low_band call hands to the solver (solver replaced)."""
    caught = []

    def runner(grid, pulse, st, rts, n):
        caught.append(grid)
        return np.zeros((n, len(rts)))
    monkeypatch.setattr(F, "_pick_fdtd_backend", lambda engine, grid=None: (runner, "fake"))
    monkeypatch.setenv("AURALIS_GRID_NATIVE", "1" if native else "0")
    monkeypatch.setenv("AURALIS_FDTD_GRID_CACHE", "0")
    sc, src, rcvs, fc = case()
    F.run_low_band(sc, src, rcvs, fs=8000, duration=0.05, f_cross=fc, speed_of_sound=343.0)
    return caught


def _same(a, b, where):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype, where
        assert a.shape == b.shape and np.array_equal(a, b), where
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for k in a:
            _same(a[k], b[k], "%s.%s" % (where, k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, "%s[%d]" % (where, i))
    elif isinstance(a, (int, float, bool, np.generic, type(None))):
        assert a == b, where


def _panel_room():
    sc, src, _r = _room()
    R = np.array([[np.cos(np.radians(23.0)), -np.sin(np.radians(23.0)), 0],
                  [np.sin(np.radians(23.0)), np.cos(np.radians(23.0)), 0], [0, 0, 1]])
    rcvs = [R @ np.array([3.0, 1.5, 1.2])]               # in the air, clear of both boxes
    sc.mat_phys = [None] * sc.n_mats
    sc.mat_phys[1] = {"layers": [{"model": "panel", "m": 8.0, "q": 4.0},
                                 {"model": "air", "d": 0.08}]}
    return sc, src, rcvs, 260.0


def _open_street():
    sc, src, rcvs = _street()
    return sc, src, rcvs, 150.0


@pytest.mark.parametrize("case", [_panel_room, _open_street], ids=["room", "street"])
def test_native_fdtd_grid_structure_equals_the_python_one(case, monkeypatch):
    """FdtdGrid with its integer part in C++ (aur_grid_structure, CPML tables, collar):
    every array identical, dtype included, to the numpy construction."""
    if not gridnative.has_structure():
        pytest.skip("libauralis without the FdtdGrid structure (ABI < 2.5)")
    ref = _grids_of_run(case, False, monkeypatch)
    nat = _grids_of_run(case, True, monkeypatch)
    assert len(ref) == len(nat) >= 1
    for g_ref, g_nat in zip(ref, nat):
        assert g_ref.n_air > 1000
        keys = [k for k in vars(g_ref) if k != "vox"]
        assert keys == [k for k in vars(g_nat) if k != "vox"]
        for k in keys:
            _same(getattr(g_ref, k), getattr(g_nat, k), k)
    if case is _open_street:
        assert ref[0].pml_axes and len(ref) == 2          # scene and empty-grid passes
    else:
        assert ref[0].n_rfaces > 0 and ref[0].n_faces > 0
