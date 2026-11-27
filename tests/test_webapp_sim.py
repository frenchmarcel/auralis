# -*- coding: utf-8 -*-
"""Web app simulation rules for non-closed models: the wave solver only for a
DECLARED open scene, conformal correction forced off there, free-field positions."""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import meshcheck  # noqa: E402
from webapp.server import sim as simlib  # noqa: E402
from webapp.server.library import Library  # noqa: E402
from webapp.server.store import default_settings  # noqa: E402


def _panel_case(open_scene, mode="default"):
    tri = np.array([[[-1, 0, 0], [1, 0, 0], [1, 0, 1.2]], [[-1, 0, 0], [1, 0, 1.2], [-1, 0, 1.2]]], float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.zeros(len(f), np.int32)
    summary, closed = meshcheck.analyse(v, f, fl, ["mdf"])
    geometry = dict(summary)
    settings = dict(default_settings(), mode=mode, open_scene=open_scene)
    sim = {"id": "mtest", "layers": {"mdf": {"material_id": "auralis:wood_panel", "scattering": None}},
           "sources": [{"id": "S1", "name": "S1", "position": [0.3, -1.5, 0.6]}],
           "receivers": [{"id": "R1", "name": "R1", "position": [-0.4, -1.2, 0.6]}],
           "settings": settings}
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    return sim, geometry, (v, f, fl, closed, ["mdf"]), lib


def test_wave_solver_needs_a_declared_open_scene():
    sim, g, mesh, lib = _panel_case(open_scene=False)
    v = simlib.validate(sim, g, mesh, lib)
    assert not v["ok_to_run"] and not v["wave_available"]
    assert any("open scene" in b for b in v["blocking"])
    sim, g, mesh, lib = _panel_case(open_scene=True)
    v = simlib.validate(sim, g, mesh, lib)
    assert v["ok_to_run"] and v["wave_available"] and v["open_scene"]
    assert v["suggested_transition_hz"] is not None


def test_free_field_positions_are_valid_around_an_open_model():
    sim, g, mesh, lib = _panel_case(open_scene=False, mode="survey")
    v = simlib.validate(sim, g, mesh, lib)
    assert v["ok_to_run"], v["blocking"]          # source/receiver outside the panel's bbox


def test_open_scenes_get_the_anisotropic_conformal_correction():
    st = dict(default_settings(), mode="default", conformal=True)
    op = simlib.engine_settings(st, open_scene=True)[0]
    cl = simlib.engine_settings(st, open_scene=False)[0]
    assert op["fdtd_conformal"] and op["fdtd_conformal_aniso"]
    assert cl["fdtd_conformal"] and not cl["fdtd_conformal_aniso"]
    off = simlib.engine_settings(dict(st, conformal=False), open_scene=True)[0]
    assert not off["fdtd_conformal"] and not off["fdtd_conformal_aniso"]


def test_open_flag_ignored_on_a_closed_model():
    g = {"check": {"watertight": True}}
    assert not simlib.is_open_scene({"open_scene": True}, g)


def test_decay_criterion_ir_length():
    from auralis.engine.geometry import box_triangles
    tri = np.asarray(box_triangles((10.0, 8.0, 3.0))[0], float)
    v = tri.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    fl = np.zeros(len(f), np.int32)
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    mid = lib.add_custom("w", "t", "", [0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2], 0.1)["id"]
    sim = {"layers": {"walls": {"material_id": mid}}}
    mesh = (v, f, fl, np.ones(len(f), bool), ["walls"])
    t35 = simlib.ir_length_for_decay(sim, mesh, lib, 240.0, 35.0)
    t60 = simlib.ir_length_for_decay(sim, mesh, lib, 240.0, 60.0)
    S, V = 2 * (80 + 30 + 24), 240.0
    t125 = 0.161 * V / (-S * np.log(1 - 0.05))                  # the longest band decides
    assert abs(t35 - (t125 * 45 / 60 + np.sqrt(173) / 343)) < 1e-6 and t60 > t35


def test_complex_table_material_and_details():
    """A material given as Z / rho c reaches the manifest as the engine's gamma table, with
    the random-incidence alphas of that table for the rays."""
    from webapp.server.library import gamma_from_input, table_alpha
    f = np.geomspace(50, 10000, 24)
    z = 1.5 - 1j * 300.0 / f                        # stiff, then nearly matched: absorbs high
    fq, g = gamma_from_input(f, z.real, z.imag, "z")
    a = table_alpha(fq, g)
    assert np.all(np.diff(a) > -0.02) and a[-1] > a[0]
    R = (z - 1) / (z + 1)
    fq2, g2 = gamma_from_input(f, R.real, R.imag, "r")
    assert np.allclose(g, g2)
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    m = lib.add_custom("t", "t", "", a, 0.1, physical={
        "type": "table", "gamma_table": {"f": fq.tolist(), "re": g.real.tolist(),
                                         "im": g.imag.tolist()}})
    sim = {"id": "m", "layers": {"w": {"material_id": m["id"], "scattering": [0.1] * 7}},
           "settings": default_settings()}
    mesh = (np.zeros((3, 3)), np.zeros((1, 3), int), np.zeros(1, np.int32), np.ones(1, bool), ["w"])
    man = simlib.build_manifest(sim, mesh, lib)
    assert "gamma_table" in man["materials"]["w"]["phys"] and man["materials"]["w"]["scattering"] == [0.1] * 7
    from auralis.engine.project import build_scene
    from auralis.engine.reflection import build_admittance_table
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    man["geometry"] = {"triangles": tri, "tri_mat": np.zeros(1, np.int32)}
    sc = build_scene(man, "")
    tab = build_admittance_table(sc.absorption, getattr(sc, "mat_phys", None))
    assert np.isfinite(tab["gi"]).all()


def test_63_hz_value_reaches_the_wave_band_only():
    """A material's 63 Hz alpha becomes the manifest's full curve (wave band). The rays
    keep the seven octaves. No full curve without the value."""
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    a = [0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05]
    with63 = lib.add_custom("panel", "t", "", a, 0.1, a63=0.30)
    plain = lib.add_custom("plain", "t", "", a, 0.1)
    assert with63["absorption_63"] == 0.30 and plain["absorption_63"] is None
    sim = {"id": "m", "layers": {"p": {"material_id": with63["id"], "scattering": None},
                                 "q": {"material_id": plain["id"], "scattering": None}},
           "settings": default_settings()}
    mesh = (np.zeros((3, 3)), np.zeros((1, 3), int), np.zeros(1, np.int32), np.ones(1, bool),
            ["p", "q"])
    man = simlib.build_manifest(sim, mesh, lib)
    full = man["materials"]["p"]["absorption_full"]
    assert full["f"][0] == 63.0 and full["f"][1:] == [125.0, 250.0, 500.0, 1000.0, 2000.0,
                                                      4000.0, 8000.0]
    assert full["alpha"] == [0.30] + a and man["materials"]["p"]["absorption"] == a
    assert "absorption_full" not in man["materials"]["q"]
    import pytest
    with pytest.raises(ValueError):
        lib.add_custom("bad", "t", "", a, 0.1, a63=-0.1)
    with pytest.raises(ValueError):
        lib.add_custom("phys", "t", "", a, 0.1, a63=0.2, physical={"type": "porous"})


def test_perforated_panel_reaches_the_wave_band_anchored():
    """A perforated panel built in the app: its stack goes to the manifest with anchor_alpha
    (the wave band keeps its alphas and takes its phase), through normalize_spec."""
    from webapp.server.library import perforated_alpha, perforated_spec
    import pytest
    params = dict(kind="slotted", size_mm=3.0, t_mm=16.0, open_pct=8.0, plenum_mm=100.0,
                  wool_mm=40.0, wool_sigma=12000.0, fleece_rayl=150.0)
    a = perforated_alpha(**params)
    assert max(a) > 0.5 and a[0] < max(a)                 # a resonant absorber, not flat
    spec = perforated_spec(**params)
    assert [la["model"] for la in spec["layers"]] == ["slotted", "screen", "miki", "air"]
    with pytest.raises(ValueError):
        perforated_spec(**dict(params, open_pct=80.0))
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    m = lib.add_custom("slots", "t", "", a, 0.1, physical=dict(type="perforated", **params))
    sim = {"id": "m", "layers": {"w": {"material_id": m["id"], "scattering": None}},
           "settings": default_settings()}
    mesh = (np.zeros((3, 3)), np.zeros((1, 3), int), np.zeros(1, np.int32), np.ones(1, bool), ["w"])
    man = simlib.build_manifest(sim, mesh, lib)
    assert man["materials"]["w"]["phys"]["anchor_alpha"]
    from auralis.engine.project import build_scene
    man["geometry"] = {"triangles": np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float),
                       "tri_mat": np.zeros(1, np.int32)}
    sc = build_scene(man, "")
    assert sc.mat_phys[0]["anchor_alpha"] and sc.mat_phys[0]["layers"][0]["model"] == "slotted"


def test_open_scene_has_no_statistical_reverberation():
    """An open model has no reverberant field: no Sabine / Eyring (the signed volume of an
    open mesh is meaningless and can even be negative)."""
    sim, g, mesh, lib = _panel_case(open_scene=True)
    v = simlib.validate(sim, g, mesh, lib)
    assert v["rt_estimate"] is None and "open scene" in v["rt_note"]


def test_open_scenes_get_spherical_wave_reflections():
    st = dict(default_settings(), mode="default")
    assert simlib.engine_settings(st, open_scene=True)[0]["spherical_reflection"]
    assert not simlib.engine_settings(st, open_scene=False)[0]["spherical_reflection"]
    assert not simlib.engine_settings(dict(st, spherical_reflection=False),
                                      open_scene=True)[0]["spherical_reflection"]


def test_ground_classes_reach_the_engine_with_their_model():
    """The Nord2000 grounds of the library: alphas from the model, and the model itself in
    the manifest (wave band, complex and spherical-wave reflections)."""
    from auralis.engine import ground, poromat
    from auralis.engine.project import build_scene
    lib = Library(os.path.join(tempfile.mkdtemp(), "custom.json"))
    ids = [m["id"] for m in lib.all()["materials"] if m["category"] == "Outdoor ground"]
    assert ids == ["auralis:ground_%s" % k for k in "ABCDEFG"]
    m = lib.get("auralis:ground_C")
    assert m["physical"]["type"] == "ground" and m["physical"]["sigma"] == 80e3
    assert np.allclose(m["absorption"], poromat.alpha_bands(ground.ground_spec("C")), atol=1e-4)
    sim = {"id": "g", "layers": {"g": {"material_id": m["id"], "scattering": None}},
           "settings": default_settings()}
    mesh = (np.zeros((3, 3)), np.zeros((1, 3), int), np.zeros(1, np.int32), np.ones(1, bool), ["g"])
    man = simlib.build_manifest(sim, mesh, lib)
    assert man["materials"]["g"]["phys"] == ground.ground_spec("C")
    man["geometry"] = {"triangles": np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float),
                       "tri_mat": np.zeros(1, np.int32)}
    sc = build_scene(man, "")
    assert sc.mat_phys[0]["layers"][0] == {"model": "delany_bazley", "sigma": 80e3,
                                           "semi_infinite": True}
