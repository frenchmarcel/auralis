# -*- coding: utf-8 -*-
"""Physical models inferred from alpha alone (matinfer) and their anchored wall fit."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import matinfer, poromat  # noqa: E402
from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.fdtd import _LAMBDA4, wall_passivity  # noqa: E402
from auralis.engine.project import build_scene  # noqa: E402
from auralis.engine.reflection import _gamma_of, alpha_random, fit_admittance_random  # noqa: E402

FE = np.geomspace(125.0, 1000.0, 30)


def _r(g):
    return (1.0 - g) / (1.0 + g)


def test_porous_material_is_recognised_and_its_phase_recovered():
    truth = matinfer.porous(15000, 50, 0)
    res = matinfer.infer(matinfer.band_alpha(truth))
    assert res["family"] == "porous", res
    r_true = _r(poromat.gamma_normal(truth, FE))
    r_inf = _r(poromat.gamma_normal(res["spec"], FE))
    assert np.mean(np.abs(r_inf - r_true)) < 0.05


def test_hard_surfaces_and_flat_low_alpha_keep_the_alpha_fit():
    """Concrete-like and window-like curves: too little absorption to identify a model.
    The physical bounds keep concrete from becoming a 7.8 kg/m2 panel over 13 mm."""
    for alpha in ([0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04],
                  [0.18, 0.12, 0.10, 0.07, 0.05, 0.04, 0.04]):
        res = matinfer.infer(alpha)
        assert res["family"] is None and "hard surface" in res["reason"]
    assert "alpha-only fit kept" in matinfer.describe(matinfer.infer([0.01] * 7))


def test_screens_stay_light():
    """An airtight panel must not be mimicked by a heavy 'screen' over a deep gap."""
    for fam, (_make, _grid, bounds) in matinfer.FAMILIES.items():
        if fam == "screen":
            assert bounds[1][1] <= 0.05 and bounds[0][0] >= 100.0


def test_anchored_fit_keeps_alpha_follows_the_phase_and_is_passive():
    model = matinfer.porous(25000, 20, 200)          # ceiling tile over a plenum
    alpha = matinfer.band_alpha(model)
    f_hi = 500.0
    dt = _LAMBDA4 * (343.0 / (9.0 * f_hi)) / 343.0
    plain = fit_admittance_random(alpha, CENTERS, f_hi=f_hi, l_max=0.6 / dt)
    anch = fit_admittance_random(alpha, CENTERS, f_hi=f_hi, l_max=0.6 / dt,
                                 gamma_target=lambda f: poromat.gamma_normal(model, f), dt=dt)
    fe = np.geomspace(63.0, 500.0, 25)
    r_t = _r(poromat.gamma_normal(model, fe))
    assert np.mean(np.abs(_r(_gamma_of(fe, *anch)) - r_t)) < np.mean(np.abs(_r(_gamma_of(fe, *plain)) - r_t))
    a_t = np.interp(fe, CENTERS, alpha)
    assert np.mean(np.abs(alpha_random(_gamma_of(fe, *anch)) - a_t)) < 0.04
    assert wall_passivity(*anch, dt)[0] > -1e-3


def test_build_scene_infers_alpha_only_absorbers_only():
    tri = np.zeros((4, 3, 3))
    absorber = matinfer.band_alpha(matinfer.porous(15000, 50, 0)).tolist()
    mats = {"wool": {"absorption": absorber},
            "concrete": {"absorption": [0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04]},
            "declared": {"absorption": absorber, "phys": {"model": "miki", "sigma": 9000,
                                                          "thickness_mm": 40}}}
    man = {"materials": mats, "geometry": {"triangles": tri, "tri_mat": [0, 1, 2, 0]}}
    sc = build_scene(man, "")
    assert sc.mat_phys[0]["inferred"] and sc.mat_phys[1] is None
    assert "inferred" not in sc.mat_phys[2]
    man["engine"] = {"infer_materials": False}
    sc = build_scene(man, "")
    assert sc.mat_phys[0] is None


def test_perforated_sheets_follow_the_textbook_resonance():
    """Helmholtz check of the perforated sheet over an empty cavity: the alpha peak sits at
    f0 = c/(2 pi) sqrt(phi / (t' D)), t' = t + 0.85 (1 - 1.13 sqrt(phi) ...) d."""
    d, t, phi, D = 5e-3, 10e-3, 0.05, 0.1
    spec = {"layers": [{"model": "perforated", "d": d, "t": t, "phi": phi}, {"model": "air", "d": D}]}
    f = np.linspace(100.0, 1000.0, 901)
    a = 1.0 - np.abs(poromat.reflection(spec, f, 1.0)) ** 2
    delta = 0.85 * (1 - 1.13 * phi ** 0.5 - 0.09 * phi + 0.27 * phi ** 1.5)
    f0 = 343.0 / (2 * np.pi) * np.sqrt(phi / ((t + delta * d) * D))
    assert abs(f[np.argmax(a)] / f0 - 1.0) < 0.05
    for kind, extra in (("slotted", {"w_slot": 2e-3}), ("mpp", {"d": 0.5e-3})):
        sp = {"layers": [dict({"model": kind, "t": 1e-3 if kind == "mpp" else 16e-3, "phi": 0.02}, **extra),
                         {"model": "air", "d": 0.1}]}
        z = poromat.surface_impedance(sp, f, 1.0)
        assert np.all(np.isfinite(z)) and np.all(z.real > 0)


def test_physical_wall_fit_keeps_the_validated_filter_and_reads_tables():
    """fit_physical_wall returns the complex fit unchanged, bit for bit, for a well-behaved
    porous layer (the RS1 Rockfon). A measured complex table gives an absorbing wall."""
    from auralis.engine.fdtd import fit_physical_wall
    from auralis.engine.reflection import fit_admittance_complex, wall_R_discrete
    f_max = 1.5 * 800.0
    dt = _LAMBDA4 * (343.0 / (9.0 * f_max)) / 343.0
    rock = poromat.normalize_spec({"layers": [{"model": "miki", "sigma": 127128.0, "d": 0.020}]})
    fg = np.geomspace(30.0, 1.2 * f_max, 48)
    old = fit_admittance_complex(fg, poromat.gamma_normal(rock, fg), l_max=0.6 / dt, dt=dt, lam=_LAMBDA4)
    new = fit_physical_wall(lambda f: poromat.gamma_full(rock, f), f_max, dt, _LAMBDA4)
    assert new[4] == "complex" and new[3] is None and new[0] == old[0]
    assert np.array_equal(new[1], old[1]) and np.array_equal(new[2], old[2])
    table = {"gamma_table": {"f": [50.0, 500.0, 5000.0], "re": [0.3, 0.5, 0.8], "im": [-0.4, -0.2, 0.0]}}
    gi, A, l, secs, _how, err = fit_physical_wall(lambda f: poromat.gamma_full(table, f), f_max, dt, _LAMBDA4)
    fe = np.geomspace(60.0, 600.0, 10)
    alpha_n = 1.0 - np.abs(wall_R_discrete(fe, gi, A, l, dt, _LAMBDA4, res=secs)) ** 2
    assert alpha_n.min() > 0.5 and err < 0.25             # absorbing, not rigid
