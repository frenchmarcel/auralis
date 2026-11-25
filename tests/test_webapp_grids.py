# -*- coding: utf-8 -*-
"""Receiver grids: lattice on a layer offset into the air, free rectangles, validity
rules, and energy-based parameters consistent with the point receivers."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import grids, meshcheck  # noqa: E402
from webapp.server.store import default_settings  # noqa: E402


def _room(size=(10.0, 8.0, 3.0)):
    tri, names = box_triangles(size)
    tri = np.asarray(tri, float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.array([0 if n == "floor" else 1 for n in names], np.int32)
    summary, closed = meshcheck.analyse(v, f, fl, ["floor", "walls"])
    sim = {"sources": [{"id": "S1", "name": "S1", "position": [2.0, 2.0, 1.5]}],
           "receivers": [], "settings": default_settings()}
    return sim, dict(summary), (v, f, fl, closed, ["floor", "walls"])


def test_floor_grid_is_a_lattice_offset_into_the_room():
    sim, g, mesh = _room()
    gr = grids.clean_grid({"id": "G1", "kind": "surface", "layer": "floor", "offset_m": 1.2,
                           "spacing_m": 1.0}, mesh[4])
    p, valid, why, area = grids.points(gr, sim, g, mesh, None)
    assert len(p) == 80 and np.allclose(p[:, 2], 1.2)           # 10 x 8 cells, centred
    assert np.allclose(np.sort(np.unique(p[:, 0])), np.arange(0.5, 10, 1.0))
    assert area.sum() == pytest.approx(80.0)
    near = np.linalg.norm(p - [2.0, 2.0, 1.5], axis=1) < 0.5   # too close to the source
    assert not valid[near].any() and valid[~near].all() and len(why) == int(near.sum())


def test_ceiling_side_grid_offsets_downwards_and_plane_grid():
    sim, g, mesh = _room()
    gr = grids.clean_grid({"id": "G1", "kind": "surface", "layer": "walls", "offset_m": 0.5,
                           "spacing_m": 2.0}, mesh[4])
    p, valid, why, _a = grids.points(gr, sim, g, mesh, None)
    inside = (p > [0, 0, 0]).all(axis=1) & (p <= [10, 8, 3]).all(axis=1)
    assert inside.all() and valid.sum() > 0.6 * len(p)       # offset into the room
    # lattice points on an edge of the layer end up against the next surface: disabled
    assert all("surface" in why[i] or "outside" in why[i] for i in np.flatnonzero(~valid))
    pl = grids.clean_grid({"id": "G2", "kind": "plane", "center": [5, 4, 1.2], "width_m": 4,
                           "length_m": 2, "spacing_m": 1.0, "azimuth_deg": 90}, mesh[4])
    p, valid, _w, _a = grids.points(pl, sim, g, mesh, None)
    assert len(p) == 8 and np.allclose(np.ptp(p[:, 1]), 3.0) and np.allclose(np.ptp(p[:, 0]), 1.0)


def test_grid_rules():
    sim, g, mesh = _room()
    with pytest.raises(ValueError):
        grids.clean_grid({"id": "G", "kind": "surface", "layer": "nope"}, mesh[4])
    with pytest.raises(ValueError):
        grids.clean_grid({"id": "G", "kind": "plane", "center": [0, 0]}, mesh[4])
    big = grids.clean_grid({"id": "G", "kind": "plane", "center": [5, 4, 1], "width_m": 60,
                            "length_m": 60, "spacing_m": 1.0}, mesh[4])
    with pytest.raises(ValueError):
        grids.points(big, sim, g, mesh, None)
    assert grids.receiver_radius({"spacing_m": 1.0}) == 0.5
    assert grids.receiver_radius({"spacing_m": 0.2}) == 0.3


def test_grid_parameters_match_the_energy_method_of_a_point():
    from auralis.engine import make_shoebox
    from auralis.engine.grid import grid_metrics
    from auralis.engine.pipeline import BakeConfig, SourceDef, bake
    from auralis.engine.grid import energy_metrics
    sc = make_shoebox((9.0, 6.0, 3.0), wall_material="plaster", floor_material="carpet")
    cfg = BakeConfig(n_rays=20000, seed=3, engine="native" if _native() else "python")
    src = SourceDef("S", np.array([2.0, 2.0, 1.5]))
    rcv = np.array([6.0, 4.0, 1.2])
    g = grid_metrics(sc, src, [rcv], cfg)
    from auralis.engine.pipeline import ReceiverDef
    res = bake(sc, src, ReceiverDef("R", rcv), cfg)
    em = energy_metrics(res.hist_total, cfg.bin_dt, res.events[0].time)
    for k in ("T30", "EDT", "C80", "D50"):
        assert np.allclose(g[k][0], em[k], equal_nan=True), k   # same rays, same events
    assert np.allclose(g["G_db"][0], res.metrics["g_bands"])
    assert 0.0 < g["STI"][0] < 1.0 and np.all(g["JLF"][0] >= 0)


def _native():
    try:
        from auralis.engine import native_available
        return native_available()
    except Exception:
        return False


def test_shrunk_receiver_sphere_removes_the_near_wall_bias():
    """A 0.3 m sphere 0.10 m from a wall loses ~0.8 dB of late energy; shrunk to the
    wall distance the level matches the one in the middle of the room."""
    from auralis.engine import make_shoebox
    from auralis.engine.grid import grid_metrics
    from auralis.engine.pipeline import BakeConfig, SourceDef
    sc = make_shoebox((10.0, 8.0, 3.2), wall_material="plaster", floor_material="carpet")
    src = SourceDef("S", np.array([3.0, 4.0, 1.6]))
    pts = [[7.0, 0.1, 1.6], [7.0, 1.5, 1.6]]
    cfg = BakeConfig(n_rays=200000, seed=1, rcv_radius=0.3,
                     engine="native" if _native() else "python")
    fixed = grid_metrics(sc, src, pts, cfg)["G_db"]
    adapted = grid_metrics(sc, src, pts, cfg, rcv_radii=[0.1, 0.3])["G_db"]
    assert (fixed[0] - fixed[1])[2:5].mean() < -0.5                # the bias is real
    assert abs((adapted[0] - adapted[1])[2:5].mean()) < 0.25        # and gone


def test_receiver_radii_follow_the_wall_distance():
    from webapp.server import runner
    from auralis.engine.pipeline import ReceiverDef
    sim, g, mesh = _room()
    r = runner.receiver_radii([ReceiverDef("a", np.array([5.0, 0.15, 1.2])),
                               ReceiverDef("b", np.array([5.0, 4.0, 1.2])),
                               ReceiverDef("c", np.array([5.0, 0.05, 1.2]))], mesh, 0.3)
    assert r == {"a": pytest.approx(0.15), "b": 0.3, "c": 0.1}
