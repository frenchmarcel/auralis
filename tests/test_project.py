# -*- coding: utf-8 -*-
"""Manifest -> scene layer shared by the BRAS harnesses and the web app:
OBJ layers, noisy material 0 fallback, and a single code path."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "BRAS"))

from auralis.engine import project  # noqa: E402

OBJ = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
f 1 2 5
usemtl floor
f 1 2 3 4
usemtl wall
f 1 4 5
"""


def _write_obj(text):
    fd, path = tempfile.mkstemp(suffix=".obj")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return path


def test_read_obj_layers_and_fan():
    path = _write_obj(OBJ)
    v, f, layer = project.read_obj(path)
    assert v.shape == (5, 3)
    assert f.tolist() == [[0, 1, 4], [0, 1, 2], [0, 2, 3], [0, 3, 4]]
    assert layer == [None, "floor", "floor", "wall"]


def test_load_obj_material_zero_is_loud(capsys):
    path = _write_obj(OBJ)
    tv, tm = project.load_obj(path, {"floor": 0, "wall": 1})
    assert tv.shape == (4, 3, 3)
    assert tm.tolist() == [0, 0, 0, 1]
    assert "before any usemtl" in capsys.readouterr().out


def test_load_obj_unknown_layer_raises():
    path = _write_obj(OBJ)
    with pytest.raises(ValueError, match="not in the manifest"):
        project.load_obj(path, {"floor": 0})


def test_build_scene_and_config_from_manifest():
    path = os.path.join(ROOT, "BRAS", "manifests", "rs1_rigid.json")
    with open(path) as fh:
        man = json.load(fh)
    sc = project.build_scene(man, os.path.dirname(path))
    assert sc.tri_verts.shape == (2, 3, 3)
    assert sc.absorption.shape[1] == 7
    cfg = project.build_config(man)
    assert cfg.normalize is False
    assert cfg.n_rays == man["engine"]["n_rays"]


def test_harness_uses_the_engine_code_path():
    import bras_bench
    assert bras_bench.build_scene is project.build_scene
    assert bras_bench.build_config is project.build_config


def test_one_per_source_loop_for_the_app_and_the_bench(monkeypatch):
    """The runner and bras_bench share source_low_bands / bake_pair. Their policies differ
    only by explicit arguments."""
    import inspect
    from auralis.engine import pipeline
    sys.path.insert(0, os.path.join(ROOT, "BRAS"))
    import bras_bench
    from webapp.server import runner
    for mod in (bras_bench, runner):
        src = inspect.getsource(mod)
        assert "source_low_bands(" in src and "bake_pair(" in src
        assert "precompute_low_bands(" not in src and " bake(" not in src
    s = pipeline.SourceDef(name="s", position=np.zeros(3))
    r = pipeline.ReceiverDef(name="r", position=np.ones(3))

    def unstable(*_a, **_k):
        raise RuntimeError("watchdog")
    monkeypatch.setattr(pipeline, "precompute_low_bands", unstable)
    assert pipeline.source_low_bands(None, s, [r], pipeline.BakeConfig(fdtd=True),
                                     fallback=True) == ({}, "watchdog")
    try:
        pipeline.source_low_bands(None, s, [r], pipeline.BakeConfig(fdtd=True))
        raise AssertionError("a bench must stop on an unstable wave band")
    except RuntimeError:
        pass
    seen = []
    monkeypatch.setattr(pipeline, "bake", lambda sc, so, re_, cfg, progress=None, low_band=None:
                        seen.append((cfg.fdtd, cfg.rcv_radius, low_band)))
    cfg = pipeline.BakeConfig(fdtd=True)
    pipeline.bake_pair(None, s, r, cfg, {}, own_wave_pass=True)        # bench: own FDTD pass
    pipeline.bake_pair(None, s, r, cfg, {}, own_wave_pass=False)       # app: geometrical
    pipeline.bake_pair(None, s, r, cfg, {"r": "LOW"}, rcv_radius=0.12, own_wave_pass=False)
    assert seen == [(True, cfg.rcv_radius, None), (False, cfg.rcv_radius, None),
                    (True, 0.12, "LOW")]
