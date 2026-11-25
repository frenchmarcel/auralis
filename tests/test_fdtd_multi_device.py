# -*- coding: utf-8 -*-
"""FDTD split over several devices (aur_fdtd_run4m, run5m with the tangential links of
conformal_aniso): bit for bit against one domain."""

import ctypes
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine import native as N  # noqa: E402
from auralis.engine.geometry import SceneData, add_box, make_shoebox  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAT = {"absorption": np.full(7, 0.12), "scattering": np.full(7, 0.1),
       "transmission": np.zeros(7)}


def _lib(kind):
    """The emulation or CUDA library with aur_fdtd_run4m (env AURALIS_CUDA_EMU_LIB /
    AURALIS_CUDA_LIB first), or None."""
    if kind == "emu":
        path = os.environ.get("AURALIS_CUDA_EMU_LIB") or os.path.join(
            ROOT, "auralis", "lib", N.platform_tag(), "libauralis_cuda_emu.so")
    else:
        path = N.find_gpu_library()
    if not path or not os.path.isfile(path):
        return None
    lib = ctypes.CDLL(path)
    if not hasattr(lib, "aur_fdtd_run4m"):
        return None
    if kind == "gpu" and not int(lib.aur_cuda_available()):
        return None
    return lib


def _rot(sc, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc.tri_verts = np.ascontiguousarray(sc.tri_verts @ R.T, dtype=np.float32)
    return R


def _closed():
    """Rotated room, a panel (resonant sections) wall, a box: faces, sections, order 2/4."""
    sc = make_shoebox((5.2, 3.1, 2.6), wall_material=MAT, floor_material=MAT,
                      ceiling_material=MAT)
    sc = add_box(sc, (0.9, 0.6, 0.8), (1.4, 0.9, 0.0), MAT)
    sc.mat_phys = [None] * sc.n_mats
    sc.mat_phys[1] = {"layers": [{"model": "panel", "m": 8.0, "q": 4.0},
                                 {"model": "air", "d": 0.08}]}
    R = _rot(sc, 17.0)
    return (sc, R @ np.array([0.8, 0.7, 1.4]),
            [R @ np.array([4.4, 2.4, 1.1]), R @ np.array([2.6, 1.5, 1.6])], 220.0)


def _open():
    """Ground and a box in free field: CPML, and the empty-grid second pass."""
    g = np.array([[[-4, -3, 0], [4, -3, 0], [4, 3, 0]], [[-4, -3, 0], [4, 3, 0], [-4, 3, 0]]])
    sc = SceneData(g, np.zeros(2, np.int32), MAT["absorption"][None], MAT["scattering"][None],
                   MAT["transmission"][None])
    sc = add_box(sc, (1.5, 1.2, 2.0), (-0.5, 0.3, 0.0), MAT)
    R = _rot(sc, 7.0)
    return (sc, R @ np.array([-2.5, -0.5, 1.2]),
            [R @ np.array([2.5, -1.0, 1.6]), R @ np.array([0.3, -1.8, 0.9])], 150.0)


def _run(case, lib, devs, monkeypatch, aniso=False):
    monkeypatch.setattr(F, "_pick_fdtd_backend",
                        lambda engine, grid=None: (
                            lambda *a, **k: F.run_grid_native(*a, lib=lib, **k), "native"))
    if devs:
        monkeypatch.setenv("AURALIS_FDTD_GPUS", devs)
    else:
        monkeypatch.delenv("AURALIS_FDTD_GPUS", raising=False)
    sc, src, rcvs, fc = case()
    ir, _info = F.run_low_band(sc, src, rcvs, fs=8000, duration=0.06, f_cross=fc,
                               speed_of_sound=343.0, conformal_aniso=aniso)
    return np.asarray(ir)


@pytest.mark.parametrize("kind", ["emu", "gpu"])
@pytest.mark.parametrize("case", [_closed, _open], ids=["closed", "open"])
def test_split_run_equals_one_domain(kind, case, monkeypatch):
    lib = _lib(kind)
    if lib is None:
        pytest.skip("no %s library with aur_fdtd_run4m (ABI 2.5)" % kind)
    ref = _run(case, lib, None, monkeypatch)
    assert np.abs(ref).max() > 0
    for devs in ("0,0", "0,0,0"):
        out = _run(case, lib, devs, monkeypatch)
        assert np.array_equal(out, ref), devs


def test_slab_cuts_balance_air_and_keep_clear_of_the_pml():
    sc, src, rcvs, fc = _open()
    dx = 343.0 / (9.0 * 1.5 * fc)
    vox = F.voxelize(sc, dx, src, margin_cells=16, include_points=[src] + rcvs)
    grid = F.FdtdGrid(vox, np.zeros(sc.n_mats), 343.0, order=4, pml={"n": 14},
                      gamma_border=1.0)
    cuts, halo = F.slab_cuts(grid, 3)
    x = grid.cells[:, 0]
    assert cuts[0] == 0 and cuts[-1] == grid.n_air
    sizes = np.diff(cuts)
    assert sizes.max() / sizes.min() < 1.3
    for k in range(1, 3):                       # whole planes, clear of the x PML
        p = x[cuts[k]]
        assert x[cuts[k] - 1] == p - 1 and 14 + 3 <= p <= vox.shape[0] - 14 - 3
    for k in range(3):                          # copies = two planes on each side
        assert halo[2 * k] <= cuts[k] and halo[2 * k + 1] >= cuts[k + 1]
    assert F.slab_cuts(grid, 40) is None        # too thin: refused


def test_grid_alignment_finds_the_street_and_keeps_the_points(monkeypatch):
    """A street at 7 deg to the axes: the grid is turned by 7 deg, the air cells drop, and the
    points are carried along (an aligned street gives the same grid)."""
    from tools_street import street
    sc, src, rcvs = street(7.0)
    th, gain = F.grid_alignment_angle(sc)
    assert abs(np.degrees(th) - (90.0 - 7.0)) < 0.3 or abs(np.degrees(th) - 7.0) < 0.3
    assert gain > 0.2
    caught = []

    def runner(grid, pulse, st, rts, n):
        caught.append(grid.n_air)
        return np.zeros((n, len(rts)))
    monkeypatch.setattr(F, "_pick_fdtd_backend", lambda engine, grid=None: (runner, "fake"))
    monkeypatch.setenv("AURALIS_FDTD_CONFORMAL", "0")
    for align in (False, True):
        F.run_low_band(sc, src, rcvs, fs=8000, duration=0.02, f_cross=120.0,
                       speed_of_sound=343.0, align_grid=align)
    assert caught[2] < 0.8 * caught[0]           # scene pass, aligned vs not
    sc0, src0, rcvs0 = street(0.0)
    F.run_low_band(sc0, src0, rcvs0, fs=8000, duration=0.02, f_cross=120.0,
                   speed_of_sound=343.0, align_grid=False)
    assert abs(caught[4] - caught[2]) < 0.03 * caught[4]


def _series(case, lib, devs, keep, monkeypatch, aniso=False):
    """Two runs on the same scene with another source and other probes each time."""
    monkeypatch.setattr(F, "_pick_fdtd_backend",
                        lambda engine, grid=None: (
                            lambda *a, **k: F.run_grid_native(*a, lib=lib, **k), "native"))
    if devs:
        monkeypatch.setenv("AURALIS_FDTD_GPUS", devs)
    else:
        monkeypatch.delenv("AURALIS_FDTD_GPUS", raising=False)
    sc, src, rcvs, fc = case()
    shift = np.array([0.35, 0.2, 0.0])
    outs = []
    import contextlib
    with (F.keep_grids() if keep else contextlib.nullcontext()):
        for k in range(2):
            ir, _info = F.run_low_band(sc, src + k * shift, [r - k * shift for r in rcvs],
                                       fs=8000, duration=0.06, f_cross=fc, speed_of_sound=343.0,
                                       conformal_aniso=aniso)
            outs.append(np.asarray(ir))
        if keep:
            assert len(F._KEEP["sessions"]) >= 1      # the grid stayed on the device(s)
    assert not F._KEEP["sessions"] and not F._KEEP["grids"]   # released on exit
    return outs


@pytest.mark.parametrize("kind", ["emu", "gpu"])
@pytest.mark.parametrize("case", [_closed, _open], ids=["closed", "open"])
@pytest.mark.parametrize("devs", [None, "0,0"], ids=["one", "split"])
def test_kept_grid_series_equals_fresh_runs(kind, case, devs, monkeypatch):
    """keep_grids(): the grid uploaded once, every run of the series only places its source
    and probes; results identical bit for bit to fresh runs."""
    lib = _lib(kind)
    if lib is None or not hasattr(lib, "aur_fdtd_session4m_create"):
        pytest.skip("no %s library with FDTD sessions (ABI 2.6)" % kind)
    fresh = _series(case, lib, None, False, monkeypatch)
    kept = _series(case, lib, devs, True, monkeypatch)
    for a, b in zip(fresh, kept):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("kind", ["emu", "gpu"])
@pytest.mark.parametrize("case", [_closed, _open], ids=["closed", "open"])
def test_split_run_with_tangential_links_equals_one_domain(kind, case, monkeypatch):
    """conformal_aniso split across devices (aur_fdtd_run5m, ABI 2.7): each slab applies the
    links of the cells it owns right after the stencil, as run5."""
    lib = _lib(kind)
    if lib is None or not hasattr(lib, "aur_fdtd_run5m"):
        pytest.skip("no %s library with aur_fdtd_run5m (ABI 2.7)" % kind)
    seen = []
    real = F._multi_device_args
    monkeypatch.setattr(F, "_multi_device_args",
                        lambda lib_, grid, devs, n_links: seen.append(n_links)
                        or real(lib_, grid, devs, n_links))
    ref = _run(case, lib, None, monkeypatch, aniso=True)
    assert np.abs(ref).max() > 0
    for devs in ("0,0", "0,0,0"):
        out = _run(case, lib, devs, monkeypatch, aniso=True)
        assert np.array_equal(out, ref), devs
    assert any(n > 0 for n in seen)       # the links were split (an empty-grid pass has none)


@pytest.mark.parametrize("kind", ["emu", "gpu"])
@pytest.mark.parametrize("devs", [None, "0,0"], ids=["one", "split"])
def test_kept_grid_series_with_tangential_links(kind, devs, monkeypatch):
    lib = _lib(kind)
    if lib is None or not hasattr(lib, "aur_fdtd_session5m_create"):
        pytest.skip("no %s library with linked FDTD sessions (ABI 2.7)" % kind)
    fresh = _series(_closed, lib, None, False, monkeypatch, aniso=True)
    kept = _series(_closed, lib, devs, True, monkeypatch, aniso=True)
    for a, b in zip(fresh, kept):
        assert np.array_equal(a, b)
