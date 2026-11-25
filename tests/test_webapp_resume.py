# -*- coding: utf-8 -*-
"""Jobs left in flight by a server stop: queued again by the real server (marked failed by
a mere import), and a simulation resumes after the sources it had finished."""

import json
import os
import sys
import tempfile
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.geometry import box_triangles  # noqa: E402
from webapp.server import jobs as jobslib  # noqa: E402
from webapp.server import meshcheck  # noqa: E402
from webapp.server.store import Store, default_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Bit-for-bit comparisons: the CPU tracer (the GPU's atomics are not deterministic,
    and a validation run may hold the GPU)."""
    monkeypatch.setenv("AURALIS_GPU", "0")


def _store_with_sim(root):
    store = Store(root)
    p = store.create_project("p")
    sp = store.create_space(p["id"], "s")
    tri, names = box_triangles((6.0, 5.0, 3.0))
    v = np.asarray(tri, float).reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.array([0 if n == "floor" else 1 for n in names], np.int32)
    summary, closed = meshcheck.analyse(v, f, fl, ["floor", "walls"])
    g = store.create_geometry(sp["id"], dict(summary), v, f, fl, ["floor", "walls"], closed,
                              b"", "room.obj")
    m = store.create_simulation(g["id"], "t")
    m["layers"] = {"floor": {"material_id": "auralis:carpet_heavy", "scattering": None},
                   "walls": {"material_id": "auralis:plaster", "scattering": None}}
    m["sources"] = [{"id": "S1", "name": "S1", "position": [1.5, 1.5, 1.5]},
                    {"id": "S2", "name": "S2", "position": [4.5, 3.0, 1.5]}]
    m["receivers"] = [{"id": "R1", "name": "R1", "position": [3.0, 2.5, 1.2]},
                      {"id": "R2", "name": "R2", "position": [4.0, 1.5, 1.2]}]
    m["settings"] = dict(default_settings(), mode="survey", n_rays=3000, ism_order=1,
                         diffraction=False)
    store.save_simulation(m)
    return store, m


def test_real_server_queues_interrupted_jobs_again_simulations_first(monkeypatch):
    monkeypatch.setattr(jobslib, "COOLDOWN_S", 0.0)
    store = Store(tempfile.mkdtemp())
    for mid in ("ma", "mb"):
        os.makedirs(store.sim_dir(mid))
    store.save_simulation({"id": "ma", "status": "running", "sources": []})
    store.save_simulation({"id": "mb", "status": "done", "sources": []})
    task = store.create_motion("mb", {"receiver_id": "R1", "strips": [{}]})
    store.set_task("mb", "motion", task["id"], status="running")
    ran = []
    orig = jobslib.JobManager._run
    jobslib.JobManager._run = lambda self, job: ran.append((job["kind"], job["simulation_id"]))
    try:
        jm = jobslib.JobManager(store, resume=True)
        t0 = time.time()
        while len(ran) < 2 and time.time() - t0 < 5:
            time.sleep(0.05)
    finally:
        jobslib.JobManager._run = orig
    assert ran == [("simulation", "ma"), ("motion", "mb")]
    assert store.get_simulation("ma")["status"] == "queued"     # set by submit
    assert any("resumed" in line for j in jm.list() for line in j["log"])


def test_import_marks_interrupted_jobs_failed():
    store = Store(tempfile.mkdtemp())
    os.makedirs(store.sim_dir("ma"))
    store.save_simulation({"id": "ma", "status": "queued", "sources": []})
    jobslib.JobManager(store)                          # resume=False: the import path
    assert store.get_simulation("ma")["status"] == "failed"


def test_interrupted_run_resumes_after_its_finished_sources(monkeypatch):
    from webapp.server import runner
    store, m = _store_with_sim(tempfile.mkdtemp())
    runner.run(store.root, m["id"])                       # reference: one uninterrupted run
    ref = store.results_dir(m["id"])
    with open(os.path.join(ref, "meta.json")) as fh:
        meta_ref = json.load(fh)
    ir_ref = {n: np.load(os.path.join(ref, "irs", n)) for n in os.listdir(os.path.join(ref, "irs"))}

    store2, m2 = _store_with_sim(tempfile.mkdtemp())
    calls = []
    real = runner.bake_pair

    def stop_at_second_source(scene, s, r, cfg, low, **kw):
        if s.name == "S2":
            raise KeyboardInterrupt("server stopped")
        calls.append((s.name, r.name))
        return real(scene, s, r, cfg, low, **kw)

    monkeypatch.setattr(runner, "bake_pair", stop_at_second_source)
    with pytest.raises(KeyboardInterrupt):
        runner.run(store2.root, m2["id"])
    assert calls == [("S1", "R1"), ("S1", "R2")]
    assert not os.path.exists(store2.results_dir(m2["id"]))
    assert os.path.exists(os.path.join(store2.results_dir(m2["id"]) + ".tmp", "checkpoint.json"))

    calls.clear()

    def count(scene, s, r, cfg, low, **kw):
        calls.append((s.name, r.name))
        return real(scene, s, r, cfg, low, **kw)

    monkeypatch.setattr(runner, "bake_pair", count)
    runner.run(store2.root, m2["id"])
    assert calls == [("S2", "R1"), ("S2", "R2")]          # S1 kept from the first attempt
    out = store2.results_dir(m2["id"])
    assert not os.path.exists(os.path.join(out, "checkpoint.json"))
    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    assert [p["metrics"] for p in meta["pairs"]] == [p["metrics"] for p in meta_ref["pairs"]]
    assert meta["ir_scale"] == meta_ref["ir_scale"]
    for n, a in ir_ref.items():
        assert np.array_equal(np.load(os.path.join(out, "irs", n)), a), n


def test_changed_simulation_does_not_reuse_a_checkpoint(monkeypatch):
    from webapp.server import runner
    store, m = _store_with_sim(tempfile.mkdtemp())
    real = runner.bake_pair

    def stop(scene, s, r, cfg, low, **kw):
        if s.name == "S2":
            raise KeyboardInterrupt("server stopped")
        return real(scene, s, r, cfg, low, **kw)

    monkeypatch.setattr(runner, "bake_pair", stop)
    with pytest.raises(KeyboardInterrupt):
        runner.run(store.root, m["id"])
    m = store.get_simulation(m["id"])
    m["sources"][0]["position"] = [2.0, 1.5, 1.5]         # edited before the restart
    store.save_simulation(m)
    calls = []

    def count(scene, s, r, cfg, low, **kw):
        calls.append(s.name)
        return real(scene, s, r, cfg, low, **kw)

    monkeypatch.setattr(runner, "bake_pair", count)
    runner.run(store.root, m["id"])
    assert calls == ["S1", "S1", "S2", "S2"]


def _results(store, mid):
    out = store.results_dir(mid)
    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    irs = {n: np.load(os.path.join(out, "irs", n)) for n in os.listdir(os.path.join(out, "irs"))}
    return meta, irs


def test_pairs_in_worker_processes_match_the_runner_bit_for_bit(monkeypatch):
    """PairPool: the pairs baked in worker processes give the responses and parameters
    of the runner's own loop, bit for bit with the CPU tracer, in the same order."""
    from webapp.server import runner
    store, m = _store_with_sim(tempfile.mkdtemp())
    monkeypatch.setenv("AURALIS_PAIR_WORKERS", "1")
    runner.run(store.root, m["id"])
    meta_ref, ir_ref = _results(store, m["id"])
    store2, m2 = _store_with_sim(tempfile.mkdtemp())
    monkeypatch.setenv("AURALIS_PAIR_WORKERS", "3")
    assert runner.pair_workers(4, None) == 3
    runner.run(store2.root, m2["id"])
    meta, irs = _results(store2, m2["id"])
    strip = lambda ps: [{k: v for k, v in p.items() if k != "timings"} for p in ps]  # noqa: E731
    assert strip(meta["pairs"]) == strip(meta_ref["pairs"])
    assert [(p["source_id"], p["receiver_id"]) for p in meta["pairs"]] == \
        [("S1", "R1"), ("S1", "R2"), ("S2", "R1"), ("S2", "R2")]
    assert meta["ir_scale"] == meta_ref["ir_scale"]
    assert meta["engine_used"] == meta_ref["engine_used"]
    assert sorted(irs) == sorted(ir_ref)
    for n, a in ir_ref.items():
        assert np.array_equal(irs[n], a), n
