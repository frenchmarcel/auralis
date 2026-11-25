# -*- coding: utf-8 -*-
"""Reflection-based scattering (Christensen & Rindel 2005): formulas, surface
dimensions, image-source weighting, and the native path with nothing to add."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import rbs  # noqa: E402
from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.geometry import SceneData, make_shoebox, surface_dims  # noqa: E402

F = np.asarray(CENTERS, dtype=float)


def test_formulas():
    c, a = 343.0, 2.0
    # a 0.5 x 3 m panel at normal incidence: f_w = c a / (2 w^2) = 2744 Hz
    sd = rbs.s_diffraction(F, 3.0, 0.5, 1.0, a, c)
    fw, fl = c * a / (2 * 0.25), c * a / (2 * 9.0)
    want = 1 - np.minimum(1, F / fw) * np.minimum(1, F / fl)
    assert np.allclose(sd, want)
    assert sd[-1] == 0.0 and (np.diff(sd) <= 1e-12).all()        # falls with frequency
    assert rbs.s_diffraction(1000.0, 3.0, 0.5, 1e-6, a, c) > 0.99  # grazing: all scattered
    assert rbs.s_diffraction(1000.0, 50.0, 30.0, 0.7, a, c) == 0.0
    assert np.isclose(rbs.a_star(4.0, 4.0), 1.0)                   # d_inc = d_refl: d / 4
    assert np.isclose(rbs.combine(0.5, 0.2), 0.6)


def test_surface_dimensions():
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="concrete")
    l, w = surface_dims(sc)
    assert sorted({(round(float(a), 6), round(float(b), 6)) for a, b in zip(l, w)}) == \
        [(3.0, 2.5), (4.0, 2.5), (4.0, 3.0)]
    # two coplanar panels apart are two surfaces
    q = lambda x0: [[[x0, 0, 0], [x0 + 1, 0, 0], [x0 + 1, 2, 0]], [[x0, 0, 0], [x0 + 1, 2, 0], [x0, 2, 0]]]
    tv = np.array(q(0.0) + q(5.0), dtype=np.float32)
    z = np.zeros((1, 7))
    l, w = surface_dims(SceneData(tv, np.zeros(4, np.int32), z, z, z))
    assert np.allclose(l, 2.0) and np.allclose(w, 1.0)


def test_image_sources_keep_only_their_specular_share():
    from auralis.engine.geometry import extract_planes, NumpyRaycaster
    from auralis.engine.ism import compute_image_sources

    def ratios(size, src, rcv):
        sc = make_shoebox(size, wall_material="concrete")
        sc.scattering[:] = 0.2
        planes, tri_plane = extract_planes(sc, 48)
        ev = compute_image_sources(sc, planes, tri_plane, NumpyRaycaster(sc), src, rcv, 1,
                                   np.zeros(7), 343.0)
        before = [e.amps.copy() for e in ev]
        rbs.weight_image_sources(ev, sc, surface_dims(sc), F, 343.0, src, rcv)
        return np.array([e.amps / b for e, b in zip(ev, before)]) ** 2

    big = ratios((20.0, 15.0, 6.0), np.array([5.0, 5.0, 1.5]), np.array([12.0, 9.0, 1.2]))
    assert len(big) == 6 and np.allclose(big, 0.8, atol=1e-6)   # large walls: 1 - s_s only
    # a small panel (1 x 0.5 m) hung 6 m below the ceiling of a large hall, seen from
    # about 5 m: its reflection keeps the specular share the formula gives
    from auralis.engine.geometry import add_box
    sc = add_box(make_shoebox((30.0, 20.0, 12.0), wall_material="concrete"), (1.0, 0.5, 0.02),
                 (14.5, 9.75, 6.0), "concrete", name="panel")
    sc.scattering[:] = 0.05
    planes, tri_plane = extract_planes(sc, 48)
    src, rcv = np.array([12.0, 10.0, 1.5]), np.array([17.0, 10.0, 1.5])
    ev = compute_image_sources(sc, planes, tri_plane, NumpyRaycaster(sc), src, rcv, 1,
                               np.zeros(7), 343.0)
    pan = [e for e in ev if abs(e.points[0][2] - 6.0) < 1e-6]
    assert len(pan) == 1
    before = pan[0].amps.copy()
    rbs.weight_image_sources(pan, sc, surface_dims(sc), F, 343.0, src, rcv)
    p = pan[0].points[0]
    d1, d2 = np.linalg.norm(p - src), np.linalg.norm(rcv - p)
    want = (1 - rbs.s_diffraction(F, 1.0, 0.5, (6.0 - 1.5) / d1, d1 * d2 / (2 * (d1 + d2)),
                                  343.0)) * 0.95
    got = (pan[0].amps / before) ** 2
    assert np.allclose(got, want, atol=1e-6) and got[0] < 0.2   # mostly scattered at 125 Hz


def test_native_rbs_path_with_nothing_to_add():
    from auralis.engine import native as nat
    if not nat.find_library():
        pytest.skip("no native library")
    from auralis.engine.native import NativeEngine
    eng = NativeEngine(make_shoebox((8.0, 6.0, 3.0), wall_material="concrete"))
    if not hasattr(eng.lib, "aur_trace2"):
        pytest.skip("native library older than ABI 2.2")
    a = dict(n_rays=4000, max_bounces=40, max_time=0.5, bin_dt=0.001, n_bins=500,
             speed_of_sound=343.0, air_db_per_m=np.zeros(7), seed=5, n_threads=1)
    huge = (np.full(12, 1e6, np.float32), np.full(12, 1e6, np.float32))   # s_d = 0
    h0 = eng.trace([2, 2, 1.5], [6, 4, 1.2], **a)[0]
    h1 = eng.trace([2, 2, 1.5], [6, 4, 1.2], rbs=huge, **a)[0]
    assert np.allclose(h1, h0, rtol=1e-4, atol=1e-6 * h0.max())
