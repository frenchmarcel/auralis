# -*- coding: utf-8 -*-
"""Conformal correction with tangential face apertures (conformal_aniso): exact box modes,
volume kept in the cut layer, every backend equal to the Python reference, default path
unchanged."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.conformal import conformal_corrections  # noqa: E402

SRC, RCV = np.array([0.4, 0.4, 0.4]), np.array([3.6, 2.6, 2.1])


def _box(alpha=0.01):
    sc = make_shoebox((4.0, 3.0, 2.5), wall_material="concrete")
    sc.absorption[:] = alpha
    return sc


def _modes(ir, fs):
    x = ir * np.hanning(len(ir))
    X = np.abs(np.fft.rfft(x, 1 << 19))
    f = np.fft.rfftfreq(1 << 19, 1.0 / fs)
    out = []
    for lo, hi in ((40, 46), (54, 60), (70, 74)):
        sel = (f > lo) & (f < hi)
        out.append(f[sel][np.argmax(X[sel])])
    return np.array(out)


def test_box_modes_are_exact_with_tangential_apertures():
    from auralis.engine import native_available
    if not native_available():
        pytest.skip("native engine absent")
    true = np.array([343.0 / 8.0, 343.0 / 6.0, 343.0 / 2 * np.hypot(1 / 4.0, 1 / 3.0)])
    kw = dict(fs=2000, duration=2.0, f_cross=300.0, speed_of_sound=343.0, engine="native")
    err = {}
    for name, conf, aniso in (("stair", False, False), ("conformal", True, False),
                              ("aniso", True, True)):
        ir, _ = F.run_low_band(_box(), SRC, [RCV], conformal=conf, conformal_aniso=aniso, **kw)
        err[name] = np.abs(_modes(ir[:, 0], 2000) / true - 1)
    assert err["aniso"].max() < 0.01, err             # measured +0.2 to +0.4 %
    assert err["conformal"][:2].min() > 0.04          # measured -6.7 / -5.5 %
    assert err["stair"][:2].min() > 0.03              # measured +5.0 / +5.0 %


def test_backends_equal_the_python_reference_and_default_is_unchanged():
    from auralis.engine import native as nat
    sc = _box(0.05)
    vox = F.voxelize(sc, 343.0 / (9 * 450.0), SRC, include_points=[SRC, RCV])
    phi, fw, st = conformal_corrections(sc, vox, per_axis=True)
    phi0, fw0, _st0 = conformal_corrections(sc, vox)
    assert np.array_equal(phi, phi0) and np.array_equal(fw, fw0)   # per_axis changes nothing
    cells, ex = st["excess_axis"]                   # sparse: only the cells by the walls
    full = np.zeros(len(phi))
    full[cells] = ex.sum(axis=0)
    assert np.allclose(full, phi - 1.0, atol=1e-9)
    g0 = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4, phi=phi, face_area_w=fw)
    assert g0.n_links == 0
    g = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4, phi=phi, face_area_w=fw,
                   excess_axis=st["excess_axis"])
    assert g.n_links > 0 and np.array_equal(g.c_nb, g0.c_nb)
    n = 300
    pulse = F.source_pulse(g.dt, 450.0, n)
    s_, r_ = g.taps(SRC), g.taps(RCV)
    ref = F.run_grid_python(g, pulse, s_, [r_], n)
    libs = []
    if nat.find_library():
        libs.append(nat.load_library(nat.find_library()))
    emu = os.path.join(os.path.dirname(nat.find_library() or ""), "libauralis_cuda_emu.so")
    if os.path.exists(emu):
        libs.append(nat.load_library(emu))
    if nat.gpu_available() and nat.find_gpu_library():
        libs.append(nat.load_library(nat.find_gpu_library()))
    if not libs:
        pytest.skip("no native library")
    for lib in libs:
        if not hasattr(lib, "aur_fdtd_run5"):
            continue
        out = F.run_grid_native(g, pulse, s_, [r_], n, lib)
        assert np.abs(out - ref).max() < 1e-4 * np.abs(ref).max()


def test_the_volume_stays_in_the_cut_layer():
    """With the apertures there is no slow wall layer, so the volume is not spread. It stays
    on the cells that touch a solid, same total, and the order-4 interior stays as whole as
    the staircase's. Spreading drops 95 % of it to order 2 and breaks reciprocity."""
    sc = _box(0.05)
    vox = F.voxelize(sc, 343.0 / (9 * 450.0), SRC, include_points=[SRC, RCV])
    phi8, fw8, st8 = conformal_corrections(sc, vox, per_axis=True)
    phi0, fw0, st0 = conformal_corrections(sc, vox, per_axis=True, smooth_sweeps=0)
    solid = np.pad(~vox.air, 1, constant_values=True)
    near = np.zeros(vox.shape, dtype=bool)
    for ax in range(3):
        for s in (0, 2):
            sl = [slice(1, -1)] * 3
            sl[ax] = slice(s, s + vox.shape[ax])
            near |= solid[tuple(sl)]
    touch = near[vox.air]
    assert (phi0[~touch] == 1.0).all() and (phi0[touch] > 1.0).mean() > 0.9
    # same volume but for the phi <= 4 cap, which bites on the unspread edges and corners
    # (3 walls: phi 7.9 uncapped); the true volume holds to 1 % either way
    assert phi0.sum() <= phi8.sum() and abs(phi0.sum() * vox.dx ** 3 / 30.0 - 1) < 0.01
    stair = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4)
    cut = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4, phi=phi0, face_area_w=fw0,
                     excess_axis=st0["excess_axis"])
    spread = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4, phi=phi8, face_area_w=fw8,
                        excess_axis=st8["excess_axis"])
    assert len(cut.i4) > 0.99 * len(stair.i4) and len(spread.i4) < 0.5 * len(stair.i4)


def test_run_low_band_spreads_only_without_apertures(monkeypatch):
    from auralis.engine import conformal as cf
    monkeypatch.delenv("AURALIS_FDTD_CONFORMAL", raising=False)
    monkeypatch.delenv("AURALIS_FDTD_CONFORMAL_ANISO", raising=False)
    seen = []
    orig = cf.conformal_corrections

    def spy(*a, **k):
        seen.append(k.get("smooth_sweeps"))
        return orig(*a, **k)
    monkeypatch.setattr(cf, "conformal_corrections", spy)
    for aniso in (False, True):
        F.run_low_band(_box(0.05), SRC, [RCV], fs=1000, duration=0.02, f_cross=120.0,
                       speed_of_sound=343.0, engine="python", conformal_aniso=aniso)
    assert seen == [8, 0]


@pytest.mark.parametrize("order", [4, 2])
def test_cut_layer_apertures_are_stable_on_oblique_walls(order):
    """Unspread excess and apertures on oblique walls stay stable: box rotated 30 deg, near
    rigid (alpha 0.02), 1.5 s, the raw grid envelope decays. The raw output is used because
    the Wiener deconvolution of a truncated response lifts its last samples."""
    from dataclasses import replace
    from auralis.engine import native_available
    if not native_available():
        pytest.skip("native engine absent")
    th = np.radians(30.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc = make_shoebox((3.1, 2.45, 2.15), wall_material="concrete")
    sc = replace(sc, tri_verts=(np.asarray(sc.tri_verts, np.float64) @ R.T).astype(np.float32))
    s_, r_ = R @ np.array([0.55, 0.45, 0.40]), R @ np.array([2.45, 1.90, 1.70])
    vox = F.voxelize(sc, 343.0 / (9 * 1.5 * 423.0), s_, include_points=[s_, r_])
    phi, fw, st = conformal_corrections(sc, vox, per_axis=True, smooth_sweeps=0)
    g = F.FdtdGrid(vox, F.gamma_from_alpha(np.full(1, 0.02)), 343.0, order=order, phi=phi,
                   face_area_w=fw, excess_axis=st["excess_axis"])
    n = int(1.5 / g.dt)
    run, _ = F._pick_fdtd_backend("native", g)
    y = np.asarray(run(g, F.source_pulse(g.dt, 423.0, n), g.taps(s_), [g.taps(r_)], n))[:, 0]
    q = n // 8
    env = [np.abs(y[k * q:(k + 1) * q]).max() for k in range(8)]
    assert np.isfinite(y).all() and env[-1] < env[0] / 10 and env[-1] <= max(env[-3:-1])


def test_dense_and_sparse_excess_give_the_same_links():
    """FdtdGrid takes the excess dense (3, n_air) or sparse (cells, (3, m)); the links are
    built on the cells with an excess and their neighbours only, identically."""
    from dataclasses import replace
    th = np.radians(23.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc = _box(0.05)
    sc = replace(sc, tri_verts=(np.asarray(sc.tri_verts, np.float64) @ R.T).astype(np.float32))
    vox = F.voxelize(sc, 343.0 / (9 * 450.0), R @ SRC, include_points=[R @ SRC, R @ RCV])
    phi, fw, st = conformal_corrections(sc, vox, per_axis=True, smooth_sweeps=0)
    cells, ex = st["excess_axis"]
    dense = np.zeros((3, len(phi)))
    dense[:, cells] = ex
    a = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4, phi=phi, face_area_w=fw, excess_axis=dense)
    b = F.FdtdGrid(vox, np.zeros(1), 343.0, order=4, phi=phi, face_area_w=fw,
                   excess_axis=(cells, ex))
    assert a.n_links > 0 and len(cells) < 0.5 * len(phi)
    for k in ("l_cell", "l_nb", "l_k", "l_ptr", "l_rows", "is4"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
