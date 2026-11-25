# -*- coding: utf-8 -*-
"""Wall filters must be passive in the grid's DISCRETE time, not only in the
continuous model. Otherwise near-rigid rooms grow a slow boundary mode."""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402
from auralis.engine.reflection import fit_admittance_random  # noqa: E402

BRICK = np.array([0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07])   # near-rigid, rising


def _dt(fc):
    f_max = 1.5 * fc
    return F._LAMBDA4 * (343.0 / (9.0 * f_max)) / 343.0, f_max


def test_continuous_passivity_is_not_enough():
    dt, f_max = _dt(398.0)
    gi, A, l = fit_admittance_random(BRICK, CENTERS, f_hi=f_max)
    w = np.linspace(1.0, np.pi / dt, 4000)
    cont = (gi + sum(a / (1j * w + ll) for a, ll in zip(A, l))).real
    assert cont.min() > 0                                   # passive on paper
    low, f_low = F.wall_passivity(gi, A, l, dt)
    assert low < -1e-2 and f_low < 300.0                    # active on the grid
    capped = fit_admittance_random(BRICK, CENTERS, f_hi=f_max, l_max=0.6 / dt)
    assert F.wall_passivity(*capped, dt)[0] > 0
    assert abs(F._alpha_at(*capped, 250.0) - 0.03) < 0.002  # same absorption
    flat = fit_admittance_random(np.full(7, 0.03), CENTERS, f_hi=f_max)
    assert F.wall_passivity(*flat, dt)[0] > 0


def test_only_active_walls_are_repaired(capsys):
    sc = make_shoebox((3.0, 2.5, 2.2), wall_material="concrete")
    sc.mat_names = ["brick", "flat"]
    sc.absorption = np.stack([BRICK, np.full(7, 0.03)]).astype(np.float32)
    sc.tri_mat = (np.arange(len(sc.tri_verts)) % 2).astype(np.int32)
    F.run_low_band(sc, [1.0, 1.0, 1.0], [[2.0, 1.4, 1.2]], fs=4000, duration=0.05,
                   f_cross=398.0, speed_of_sound=343.0, engine="python")
    out = capsys.readouterr().out
    assert "wall filter of 'brick' was ACTIVE" in out and "poles capped" in out
    assert "wall filter of 'flat'" not in out


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "BRAS", "CR2", "CR2_RIR.obj")),
                    reason="CR2 geometry absent")
def test_validated_cr2_walls_are_untouched():
    """At the published CR2 configuration no wall is active: the repair never runs,
    so the validated results are unchanged bit for bit."""
    from auralis.engine.air import speed_of_sound
    from auralis.engine.ir_synthesis import _n_samples
    from auralis.engine.pipeline import _scene_volume_hint, estimate_max_time
    from auralis.engine.project import build_config, build_scene
    path = os.path.join(ROOT, "BRAS", "CR2", "cr2_dodecahedron_fitted31.json")
    with open(path) as fh:
        man = json.load(fh)
    sc, cfg = build_scene(man, os.path.dirname(path)), build_config(man)
    c = speed_of_sound(cfg.temperature_c)
    max_time = float(cfg.max_time or estimate_max_time(sc, c))
    dur = _n_samples(max(8, int(np.ceil(max_time / cfg.bin_dt))), cfg.bin_dt, cfg.fs) / cfg.fs
    vol, rt = _scene_volume_hint(sc), estimate_max_time(sc, c) / 1.5
    fc, _ = F.auto_crossover(vol, dur, c, 2000.0 * np.sqrt(max(rt, 0.05) / vol),
                             order=int(cfg.fdtd_order), budget_gcell=float(cfg.fdtd_budget_gcell),
                             fcross_max=float(cfg.fdtd_fcross_max))
    f_max = 1.5 * fc
    dt = F._LAMBDA4 * (c / (9.0 * f_max)) / c
    for m, (fb, ab) in enumerate(sc.mat_alpha_full):
        assert F.wall_passivity(*fit_admittance_random(ab, fb, f_hi=f_max), dt)[0] > 0, \
            sc.mat_names[m]


def test_native_fdtd_reports_its_steps():
    """ABI 2.3: the native kernels write their current step for a host poller."""
    from auralis.engine import native as nat
    if not nat.find_library():
        pytest.skip("no native library")
    lib = nat.load_library(nat.find_library())
    if not hasattr(lib, "aur_fdtd_progress"):
        pytest.skip("native library older than ABI 2.3")
    sc = make_shoebox((6.0, 5.0, 3.0), wall_material="concrete")
    sc.absorption[:] = 0.05
    src, rcv = [1.0, 1.0, 1.2], [4.0, 3.0, 1.4]
    vox = F.voxelize(sc, 343.0 / (9 * 600.0), src, include_points=[src, rcv])
    g = F.FdtdGrid(vox, F.gamma_from_alpha(np.full(1, 0.05)), 343.0, order=4)
    n = 6000
    seen = []
    F.set_progress_hook(lambda k, total: seen.append((k, total)))
    try:
        F.run_grid_native(g, F.source_pulse(g.dt, 450.0, n), g.taps(src), [g.taps(rcv)], n, lib)
    finally:
        F.set_progress_hook(None)
    if not seen:
        pytest.skip("run shorter than one poll (0.25 s)")
    ks = [k for k, _ in seen]
    assert all(t == n for _, t in seen) and ks == sorted(ks) and 0 <= ks[-1] < n
