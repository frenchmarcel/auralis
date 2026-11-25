# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_webapp_resume import _store_with_sim  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("AURALIS_GPU", "0")


def test_wave_solver_only_runs_and_reports_the_octaves_below_the_transition():
    """Advanced mode, wave solver only: the run is allowed (with its warning), the IR has no
    energy above the transition, the parameters exist in the octaves the wave band covers and
    are None elsewhere, with no STI or overall level."""
    from webapp.server import runner
    from webapp.server import sim as simlib
    store, m = _store_with_sim(tempfile.mkdtemp())
    m["sources"], m["receivers"] = m["sources"][:1], m["receivers"][:1]
    m["settings"].update(mode="advanced", solvers="wave", transition_hz=400.0, ir_length_s=0.8)
    store.save_simulation(m)
    eng, _dur = simlib.engine_settings(m["settings"])
    assert eng["wave_only"] and eng["fdtd"]
    runner.run(store.root, m["id"])
    out = store.results_dir(m["id"])
    with open(os.path.join(out, "meta.json")) as fh:
        text = fh.read()
    assert "NaN" not in text and "Infinity" not in text
    meta = json.loads(text)
    met = meta["pairs"][0]["metrics"]
    assert met["wave_only"] and met["STI"] is None and met["SPL_dba"] is None
    assert all(v is None for v in met["JLF"] + met["LJ_db"] + met["MTI"])
    for key in ("T30", "C50", "G_db", "SPL_db"):
        assert all(v is not None for v in met[key][:2]), key          # 125, 250 Hz
        assert all(v is None for v in met[key][2:]), key              # 500 Hz and above
    assert "by_solver" not in met
    from webapp.server.combine import combine
    _key, info = combine(out, meta, [{"source_id": "S1", "gain_db": -3.0}], "R1", m["settings"])
    json.loads(json.dumps(info, allow_nan=False))                    # no NaN for the browser
    assert info["metrics"]["STI"] is None and info["metrics"]["G_db"][2] is None
    assert abs(info["metrics"]["G_db"][0] - (met["G_db"][0] - 3.0)) < 0.01
    ir = np.load(os.path.join(out, "irs", "S1__R1.npy"))
    S = (np.abs(np.fft.rfft(ir, axis=0)) ** 2).sum(axis=1)
    f = np.fft.rfftfreq(len(ir), 1.0 / meta["fs"])
    assert S[f > 1000.0].sum() < 1e-4 * S[f < 400.0].sum()


def test_wave_solver_only_refuses_receiver_grids():
    from webapp.server import sim as simlib
    from webapp.server.library import Library
    store, m = _store_with_sim(tempfile.mkdtemp())
    m["settings"].update(mode="advanced", solvers="wave")
    geometry = store.get_geometry(m["geometry_id"])
    mesh = store.load_mesh(m["geometry_id"])
    from webapp.server import grids
    m["receiver_grids"] = [dict(grids.clean_grid({"id": "G1", "kind": "surface", "layer": "floor",
                                                  "offset_m": 1.2, "spacing_m": 1.0}, mesh[4]),
                                name="g")]
    lib = Library(os.path.join(store.root, "materials_custom.json"))
    chk = simlib.validate(m, geometry, mesh, lib)
    assert any("receiver grids" in b for b in chk["blocking"])
    assert any("wave solver only" in w for w in chk["warnings"])


def test_a_vibrating_layer_source_runs_with_the_wave_solver_only():
    """A source with surface_layer: refused in hybrid mode, run in wave-only mode (the
    layer's surfaces vibrate; the IR has energy)."""
    from webapp.server import runner
    from webapp.server import sim as simlib
    from webapp.server.library import Library
    store, m = _store_with_sim(tempfile.mkdtemp())
    m["sources"], m["receivers"] = m["sources"][:1], m["receivers"][:1]
    m["sources"][0]["surface_layer"] = "floor"
    m["settings"].update(mode="advanced", solvers="both", transition_hz=300.0, ir_length_s=0.6)
    geometry = store.get_geometry(m["geometry_id"])
    mesh = store.load_mesh(m["geometry_id"])
    lib = Library(os.path.join(store.root, "materials_custom.json"))
    assert any("vibrating layer" in b for b in simlib.validate(m, geometry, mesh, lib)["blocking"])
    m["settings"]["solvers"] = "wave"
    assert not simlib.validate(m, geometry, mesh, lib)["blocking"]
    store.save_simulation(m)
    runner.run(store.root, m["id"])
    out = store.results_dir(m["id"])
    ir = np.load(os.path.join(out, "irs", "S1__R1.npy"))
    assert np.abs(ir).max() > 0
