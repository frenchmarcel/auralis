# -*- coding: utf-8 -*-
"""Example project: concerts in Odeon's hall models (local data, skipped without it), the
BRAS rooms built from the bench's manifests in BRAS/, and the Paris street with real
recordings, delivered with its finished renders."""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import examples  # noqa: E402
from webapp.server.library import Library  # noqa: E402
from webapp.server.store import Store  # noqa: E402

needs_odeon = pytest.mark.skipif(not os.path.isdir(examples.ROOMS),
                                 reason="local Odeon data absent")


class NoJobs:
    def submit(self, m):
        raise AssertionError("must not be queued")

    def submit_motion(self, m, info):
        raise AssertionError("must not be queued")


def _store_at(version):
    store = Store(tempfile.mkdtemp())
    lib = Library(os.path.join(store.root, "materials_custom.json"))
    p = store.create_project("Examples")
    with open(os.path.join(store.root, examples.MARK), "w") as fh:
        json.dump({"version": version, "project_id": p["id"], "examples": [{"example": "old"}]}, fh)
    return store, lib, p


def _spaces(store, p):
    return [s["name"] for s in store._list("spaces") if s["project_id"] == p["id"]]


def test_fresh_store_gets_halls_bras_rooms_and_paris_only():
    store = Store(tempfile.mkdtemp())
    lib = Library(os.path.join(store.root, "materials_custom.json"))
    info = examples.install(store, lib, jobs=None, run=False)
    assert info["version"] == examples.VERSION == 7
    names = [e["example"] for e in info["examples"]]
    rooms = [x[1] for x in examples.BRAS_ROOMS]
    halls = [x[1] for x in examples.HALLS] if os.path.isdir(examples.ROOMS) else []
    assert [n for n in names if n in halls] == halls
    assert [n for n in names if n in rooms] == rooms
    assert names[-1] == "Paris street"
    assert len(names) == len(halls) + len(rooms) + 1
    p = store.get_project(info["project_id"])
    assert sorted(_spaces(store, p)) == sorted(
        (["Concert halls"] if halls else []) + ["BRAS rooms (TU Berlin)",
                                                "Paris street"])
    assert all(e.get("ok") for e in info["examples"]), [e for e in info["examples"]
                                                        if not e.get("ok")]
    assert examples.install(store, lib, jobs=None, run=False)["project_id"] == info["project_id"]


def test_bras_rooms_are_the_bench_scenes():
    """Each BRAS room carries the manifest's materials (absorption and per-octave scattering,
    63 Hz from its three third octaves), the measured positions exactly, the session's air and
    the dodecahedron, imported once for all rooms."""
    store = Store(tempfile.mkdtemp())
    lib = Library(os.path.join(store.root, "materials_custom.json"))
    sid = store.create_space(store.create_project("t")["id"], "BRAS")["id"]
    out = [examples._bras_room(store, lib, sid, *spec) for spec in examples.BRAS_ROOMS]
    dodeca = {m["sources"][0]["directivity"] for m in
              (store.get_simulation(o["simulation"]) for o in out)}
    assert len(dodeca) == 1 and next(iter(dodeca)).startswith("custom-")
    for spec, o in zip(examples.BRAS_ROOMS, out):
        assert o["ok"], (spec[0], o["blocking"])
        with open(os.path.join(examples.BRAS_DIR, spec[2])) as fh:
            man = json.load(fh)
        m = store.get_simulation(o["simulation"])
        pos = {}
        for pr in man["pairs"]:
            ls, mp = pr["label"].split("-")
            pos[ls], pos[mp] = pr["source"], pr["receiver"]
        assert {s["id"]: s["position"] for s in m["sources"]} == {
            k: v for k, v in pos.items() if k.startswith("LS")}
        assert {r["id"]: r["position"] for r in m["receivers"]} == {
            k: v for k, v in pos.items() if k.startswith("MP")}
        assert all(s["elevation_deg"] == 90.0 for s in m["sources"])     # forward +z, as the bench
        r0, s0 = m["receivers"][0], m["sources"][0]
        az = math.degrees(math.atan2(s0["position"][1] - r0["position"][1],
                                     s0["position"][0] - r0["position"][0])) % 360
        assert abs(r0["azimuth_deg"] - az) < 0.06                        # facing LS1
        assert m["settings"]["temperature_c"] == man["engine"]["temperature_c"]
        assert m["settings"]["humidity_percent"] == man["engine"]["humidity_percent"]
        assert set(m["layers"]) == set(man["materials"])
        for layer, spec_m in man["materials"].items():
            mat = lib.get(m["layers"][layer]["material_id"])
            assert mat["library"] == "bras"
            assert np.allclose(mat["absorption"], spec_m["absorption"])
            assert m["layers"][layer]["scattering"] == [round(x, 4) for x in spec_m["scattering"]]
            fa = dict(zip(spec_m["absorption_full"]["f"], spec_m["absorption_full"]["alpha"]))
            assert mat["absorption_63"] == round(float(np.mean([fa[50.0], fa[63.0], fa[80.0]])), 4)


@needs_odeon
def test_halls_use_their_own_odeon_materials():
    store = Store(tempfile.mkdtemp())
    lib = Library(os.path.join(store.root, "materials_custom.json"))
    info = examples.install(store, lib, jobs=None, run=False)
    built = {e["example"]: e for e in info["examples"]}
    for _key, hall, _model, _ens in examples.HALLS:
        m = store.get_simulation(built[hall]["simulation"])
        assert {lib.get(v["material_id"])["library"] for v in m["layers"].values()} == \
            {"odeon_rooms"}, hall
    m = store.get_simulation(built["Musikverein, Vienna"]["simulation"])
    assert len(m["sources"]) == 15 and all(s.get("sound_id") for s in m["sources"])
    r1 = m["receivers"][0]["position"]
    # Odeon's own listening point in this model is (25.3, -1.25, 1.2): ours is derived
    # from the surface names only and must land close to it
    assert abs(r1[0] - 25.3) < 3.0 and abs(r1[2] - 1.2) < 0.5


def test_older_store_gets_only_the_new_examples():
    """A v6 store (it has CR2 and the Paris street) gets the other BRAS rooms only; a store
    from before v6 also gets the Paris street. The removed examples are never added."""
    for version, extra in ((6, []), (5, ["Paris street"])):
        store, lib, p = _store_at(version)
        info = examples.install(store, lib, jobs=None, run=False)
        assert info["version"] == examples.VERSION and info["project_id"] == p["id"]
        assert [e["example"] for e in info["examples"]] == ["old"] + [
            x[1] for x in examples.BRAS_ROOMS if x[0] != "cr2"] + extra
        assert sorted(_spaces(store, p)) == sorted(
            ["BRAS rooms (TU Berlin)"] + (["Paris street"] if extra else []))
        assert examples.install(store, lib, jobs=None, run=False) == info


def test_bras_rooms_are_queued_when_running():
    store, lib, p = _store_at(6)
    queued = []

    class Jobs:
        def submit(self, m):
            queued.append(m["id"])

        def submit_motion(self, m, info):
            raise AssertionError("no animated render for the BRAS rooms")
    info = examples.install(store, lib, jobs=Jobs(), run=True)
    assert queued == [e["simulation"] for e in info["examples"][1:]]


def test_paris_with_real_recordings_comes_rendered_with_its_credits():
    """The Paris street with real recordings: 15 sources on CC0 Freesound recordings
    (credited), a recorded hum as the diffuse bed, the wave band to 250 Hz in its settings;
    nothing queued, both shipped renders are finished and carry every credit."""
    store, lib, p = _store_at(6)
    with open(os.path.join(store.root, examples.MARK)) as fh:
        mark = json.load(fh)
    mark["version"] = 5
    with open(os.path.join(store.root, examples.MARK), "w") as fh:
        json.dump(mark, fh)
    queued = []

    class Jobs(NoJobs):
        def submit(self, m):
            queued.append(m["name"])
    info = examples.install(store, lib, jobs=Jobs(), run=True)
    assert queued == ["BRAS measurement positions"] * 4            # the Paris street: nothing
    (e,) = [x for x in info["examples"] if x["example"] == "Paris street"]
    assert e["ok"] and e["queue"] is False
    m = store.get_simulation(e["simulation"])
    st = m["settings"]
    assert st["solvers"] == "both" and st["transition_hz"] == 250.0 and st["open_scene"]
    assert os.path.exists(st["background_recording"]["path"])
    assert "CC0" in st["background_recording"]["credit"]
    assert len(m["sources"]) == 15 and all(s.get("sound_id") for s in m["sources"])
    with open(os.path.join(examples.PARIS_SOUNDS_DIR, "sounds.json")) as fh:
        index = json.load(fh)
    assert all(v["licence"] == "CC0 1.0" for v in index.values()) and len(index) == 9
    renders = store.motions(m["id"])
    assert len(renders) == len(examples.PARIS_REC_RENDERS) == 2
    for r in renders:
        assert r["status"] == "done"
        with open(os.path.join(store.motion_dir(m["id"], r["id"]), "credits.txt"),
                  encoding="utf-8") as fh:
            text = fh.read()
        assert examples.FABIAN_CREDIT in text and text.count("freesound.org/s/") == 9
