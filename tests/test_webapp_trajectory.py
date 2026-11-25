# -*- coding: utf-8 -*-
"""Trajectories in the web app: path rules, sampling handed to the engine, checks
along the whole path, and an animated render whose Doppler shift is the physical one."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import meshcheck  # noqa: E402
from webapp.server import sim as simlib  # noqa: E402
from webapp.server import trajectory as traj  # noqa: E402
from webapp.server.library import Library  # noqa: E402
from webapp.server.store import Store, default_settings  # noqa: E402


def _path(points, times, orientation="fixed", target=None):
    return traj.clean_path({"waypoints": [{"position": p, "t": t} for p, t in zip(points, times)],
                            "orientation": orientation, "target_id": target}, "P")


def _box_case(size=(10.0, 8.0, 3.0), alpha=0.05):
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    material = lib.add_custom("walls", "test", "", [alpha] * 7, 0.1)["id"]
    tri = np.asarray(box_triangles(size)[0], float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.zeros(len(f), np.int32)
    summary, closed = meshcheck.analyse(v, f, fl, ["walls"])
    sim = {"id": "mtest", "geometry_id": "g", "layers": {"walls": {"material_id": material,
                                                                   "scattering": None}},
           "sources": [{"id": "S1", "name": "S1", "position": [2.0, 4.0, 1.5],
                        "azimuth_deg": 0, "elevation_deg": 0}],
           "receivers": [{"id": "R1", "name": "R1", "position": [8.0, 4.0, 1.5],
                          "azimuth_deg": 180, "elevation_deg": 0}],
           "settings": default_settings()}
    return sim, dict(summary), (v, f, fl, closed, ["walls"]), lib


def test_path_rules():
    for bad in ([[0, 0, 0]], [0.0]), ([[0, 0, 0], [1, 0, 0]], [1.0, 1.0]), \
               ([[0, 0, 0], [1, 0, 0]], [-1.0, 1.0]):
        with pytest.raises(ValueError):
            _path(*bad)
    with pytest.raises(ValueError):
        _path([[0, 0, 0], [1, 0, 0]], [0, 1], orientation="target")
    with pytest.raises(ValueError):
        _path([[0, 0, 0], [1, 0, 0]], [0, 1], orientation="spin")
    assert traj.clean_path(None, "P") is None


def test_positions_are_held_outside_the_waypoint_times():
    it = {"position": [0, 0, 0], "path": _path([[0, 0, 1], [4, 0, 1]], [1.0, 3.0])}
    p = traj.position_at(it, [0.0, 1.0, 2.0, 3.0, 9.0])
    assert np.allclose(p[:, 0], [0, 0, 2, 4, 4]) and np.allclose(p[:, 2], 1)


def test_follow_heading_turns_the_corner_and_keeps_elevation():
    it = {"position": [0, 0, 0], "elevation_deg": 30.0,
          "path": _path([[0, 0, 0], [4, 0, 0], [4, 4, 0]], [1.0, 3.0, 5.0], "follow")}
    f = traj.forward_at(it, [0.0, 2.0, 4.0, 9.0], {})
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert np.allclose(f[0], [c, 0, s]) and np.allclose(f[1], [c, 0, s])   # held before start
    assert np.allclose(f[2], [0, c, s]) and np.allclose(f[3], [0, c, s])   # held after the end
    assert np.allclose(np.linalg.norm(traj.forward_at(it, np.linspace(0, 6, 61), {}), axis=1), 1)


def test_target_orientation_tracks_a_moving_target():
    tgt = {"id": "T", "position": [0, 0, 0], "path": _path([[0, 5, 0], [10, 5, 0]], [0, 10])}
    it = {"id": "A", "position": [0, 0, 0],
          "path": _path([[0, 0, 0], [0, 0.001, 0]], [0, 10], "target", "T")}
    f = traj.forward_at(it, [0.0, 5.0], {"T": tgt})
    assert np.allclose(f[0], [0, 1, 0], atol=1e-3)
    assert np.allclose(f[1], np.array([5, 5, 0]) / np.sqrt(50), atol=1e-3)


def test_engine_trajectory_matches_the_polyline():
    it = {"id": "A", "position": [0, 0, 0], "azimuth_deg": 90,
          "path": _path([[0, 0, 0], [3, 0, 0], [3, 3, 0]], [0.0, 1.5, 4.5])}
    tr = traj.engine_trajectory(it, {"A": it}, 6.0)
    t = np.linspace(0, 6, 37)
    assert np.allclose(tr.position(t), traj.position_at(it, t), atol=1e-12)
    assert np.allclose(tr.forward(t), [0, 1, 0])          # fixed orientation: azimuth 90
    assert 1.5 in tr.times and 4.5 in tr.times            # kinks sampled exactly


def test_paths_are_checked_along_their_whole_length():
    sim, g, mesh, lib = _box_case()
    sim["sources"][0]["path"] = _path([[2, 4, 1.5], [12, 4, 1.5]], [0, 5])      # leaves the box
    sim["receivers"][0]["path"] = _path([[8, 4, 1.5], [8, 7.95, 1.5]], [0, 5])  # grazes a wall
    by = {p["id"]: p for p in simlib.validate_paths(sim, g, mesh)}
    assert not by["S1"]["valid"] and "leaves the air volume" in by["S1"]["reasons"][0]
    assert any("from a surface" in r for r in by["R1"]["reasons"])
    assert by["S1"]["length_m"] == 10.0 and by["S1"]["max_speed_mps"] == 2.0
    sim["sources"][0]["path"] = _path([[2, 4, 1.5], [6, 2, 1.5]], [0, 5])
    sim["receivers"][0]["path"] = None
    assert all(p["valid"] for p in simlib.validate_paths(sim, g, mesh))
    # a path never blocks a static run
    assert simlib.validate(sim, g, mesh, lib)["ok_to_run"]


def test_motion_check_rules():
    sim, g, mesh, lib = _box_case()
    v = simlib.validate_motion(sim, g, mesh, lib, None, "R1", ["S1"])
    assert not v["ok"] and any("nothing moves" in b for b in v["blocking"])
    # the source walks through the listener
    sim["sources"][0]["path"] = _path([[2, 4, 1.5], [9, 4, 1.5]], [0, 7])
    v = simlib.validate_motion(sim, g, mesh, lib, None, "R1", ["S1"])
    assert not v["ok"] and any("comes 0.0" in b and "t = 6.00" in b for b in v["blocking"])
    sim["sources"][0]["path"] = _path([[2, 4, 1.5], [7, 2, 1.5]], [0, 7])
    v = simlib.validate_motion(sim, g, mesh, lib, None, "R1", ["S1"])
    assert v["ok"], v["blocking"]
    assert v["duration_s"] == 7.0 and v["paths"][0]["id"] == "S1"
    assert not simlib.validate_motion(sim, g, mesh, lib, None, "R1", ["S1"], 0.2)["ok"]


def test_motion_render_storage():
    store = Store(tempfile.mkdtemp())
    m = {"id": "mtest"}
    os.makedirs(store.sim_dir("mtest"))
    info = store.create_motion(m["id"], {"receiver_id": "R1", "strips": [{}]})
    assert store.get_motion(m["id"], info["id"])["status"] == "queued"
    store.set_motion(m["id"], info["id"], status="done")
    assert [x["id"] for x in store.motions(m["id"])] == [info["id"]]
    with pytest.raises(KeyError):
        store.motion_dir(m["id"], "../x")
    store.delete_motion(m["id"], info["id"])
    assert store.motions(m["id"]) == []


def _dominant_hz(x, fs, f_lo, f_hi):
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w, 16 * len(x)))
    f = np.fft.rfftfreq(16 * len(x), 1 / fs)
    band = (f > f_lo) & (f < f_hi)
    return float(f[band][np.argmax(X[band])])


def test_animated_render_doppler_shift_is_physical():
    """A 1 kHz tone on a source passing the listener at 17 m/s, in a dead 80 x 20 m hall:
    approaching, the listener hears f0 c / (c - v cos θ), receding f0 c / (c + v cos θ)."""
    from auralis.engine import native_available
    from auralis.engine.air import speed_of_sound
    from auralis.engine.wavio import read_wav, write_wav
    from webapp.server import motion_runner
    if not native_available():
        pytest.skip("native engine absent")
    d = tempfile.mkdtemp()
    sim, g, mesh, lib = _box_case((80.0, 20.0, 6.0), alpha=0.9)
    fs, f0, v = 44100, 1000.0, 17.0
    t = np.arange(int(3.5 * fs)) / fs
    wav = os.path.join(d, "tone.wav")
    write_wav(wav, (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), fs)
    sim["settings"].update(n_rays=4000)
    sim["sources"][0]["path"] = _path([[2, 10, 3], [2 + v * 3.5, 10, 3]], [0, 3.5])
    sim["receivers"][0]["position"] = [31.75, 13.0, 3.0]   # passes 3 m abeam at t = 1.75 s
    sim["receivers"][0]["azimuth_deg"] = 270
    req = {"simulation": sim, "receiver_id": "R1", "duration_s": 3.5, "doppler": True,
           "strips": [{"source_id": "S1", "path": wav, "credit": "test tone", "gain_db": 0.0}]}
    motion_runner.render(req, g, mesh, lib, None, d)
    mdir = d
    y, fs_out = read_wav(os.path.join(mdir, "render.wav"))
    assert fs_out == fs and y.shape[1] == 2
    c = speed_of_sound(20.0)
    # approach: t in [0.2, 0.9] s, the source 25-13 m away, cos θ ~ 0.99-0.97
    y = y.mean(axis=1)
    pr = np.array([31.75, 13.0])
    for (a, b), sign in (((0.2, 0.9), +1), ((2.6, 3.3), -1)):
        te = np.linspace(a, b, 50)                          # emission times (approximately)
        rel = pr - np.column_stack([2 + v * te, np.full_like(te, 10.0)])
        cos = np.mean(np.abs(rel[:, 0]) / np.linalg.norm(rel, axis=1))
        expect = f0 * c / (c - sign * v * cos)
        got = _dominant_hz(y[int((a + 0.07) * fs):int((b + 0.07) * fs)], fs, 900, 1100)
        assert abs(got - expect) < 2.0, (sign, got, expect)
    assert "test tone" in open(os.path.join(mdir, "credits.txt")).read()


def test_jobs_cancelled_in_the_cooldown_never_run_and_interrupted_ones_are_released():
    import time
    from webapp.server import jobs as jobslib
    store = Store(tempfile.mkdtemp())
    os.makedirs(store.sim_dir("mtest"))
    m = {"id": "mtest", "status": "running", "sources": []}
    store.save_simulation(m)
    info = store.create_motion("mtest", {"receiver_id": "R1", "strips": [{}]})
    jm = jobslib.JobManager(store)                     # recovers the interrupted ones
    assert store.get_simulation("mtest")["status"] == "failed"
    assert store.get_motion("mtest", info["id"])["status"] == "failed"
    ran = []
    jm._run = lambda job: ran.append(job["id"])
    jm.last_end = time.time()                          # next job waits for the cooldown
    job = jm.submit_motion(m, store.create_motion("mtest", {"receiver_id": "R1", "strips": [{}]}))
    time.sleep(0.3)
    jm.cancel(job["id"])
    time.sleep(jobslib.COOLDOWN_S + 0.5)
    assert ran == [] and job["status"] == "cancelled"


def test_smooth_paths_round_corners_and_keep_holds():
    wps = [{"position": [0, 0, 1.6], "t": 0}, {"position": [2, 0, 1.6], "t": 2},
           {"position": [2, 0, 1.6], "t": 4}, {"position": [2, 3, 1.6], "t": 7},
           {"position": [5, 3, 1.6], "t": 10}]
    lin = {"position": [0, 0, 1.6], "path": traj.clean_path({"waypoints": wps}, "a")}
    sm = {"position": [0, 0, 1.6],
          "path": traj.clean_path({"waypoints": wps, "interpolation": "smooth"}, "b")}
    assert lin["path"]["interpolation"] == "linear"
    ts = np.array([w["t"] for w in wps], float)
    assert np.allclose(traj.position_at(sm, ts), [w["position"] for w in wps])
    assert np.allclose(traj.position_at(sm, np.linspace(2, 4, 21)), [2, 0, 1.6])  # the hold
    tt = np.linspace(0, 10, 20001)
    v = np.diff(traj.position_at(sm, tt), axis=0) / np.diff(tt)[:, None]
    k = np.searchsorted(tt, 7.0)
    assert np.linalg.norm(v[k - 2] - v[k + 1]) < 0.01 and np.linalg.norm(v[k]) > 0.5   # C1, no stop
    lin_ref = np.stack([np.interp(tt, ts, [w["position"][i] for w in wps]) for i in range(3)], 1)
    assert np.array_equal(traj.position_at(lin, tt), lin_ref)        # linear unchanged
    pts = traj.path_points(sm)
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() <= traj.CHECK_STEP_M + 1e-9
    info = traj.path_info(sm)
    assert info["length_m"] > traj.path_info(lin)["length_m"] and len(info["curve"]) == len(pts)
    with pytest.raises(ValueError):
        traj.clean_path({"waypoints": wps, "interpolation": "bezier"}, "c")
