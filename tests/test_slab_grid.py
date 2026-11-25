# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine import native as N  # noqa: E402
from test_fdtd_multi_device import _closed, _open  # noqa: E402

EMU = os.environ.get("AURALIS_CUDA_EMU_LIB") or os.path.join(
    ROOT, "auralis", "lib", N.platform_tag(), "libauralis_cuda_emu.so")


def _run(case, monkeypatch, slabs, aniso):
    lib = N.load_library(EMU)
    monkeypatch.setattr(F, "_pick_fdtd_backend", lambda engine, grid=None: (
        (grid.runner, "slabs") if getattr(grid, "wide_index", False) else
        (lambda *a, **k: F.run_grid_native(*a, lib=lib, **k), "native")))
    monkeypatch.setenv("AURALIS_FDTD_DENSE_LIB", EMU)
    if slabs:
        monkeypatch.setenv("AURALIS_FDTD_SLABS", "1")
        monkeypatch.setenv("AURALIS_FDTD_SLABS_LIB", EMU)
        monkeypatch.setenv("AURALIS_FDTD_GPUS", slabs)
    else:
        monkeypatch.delenv("AURALIS_FDTD_SLABS", raising=False)
        monkeypatch.delenv("AURALIS_FDTD_GPUS", raising=False)
    F._VOX_CACHE.clear()
    F._CONF_CACHE.clear()
    sc, src, rcvs, fc = case()
    ir, _info = F.run_low_band(sc, src, rcvs, fs=8000, duration=0.06, f_cross=fc,
                               speed_of_sound=343.0, conformal_aniso=aniso,
                               src_axis=np.array([0.6, 0.3, 0.2]),
                               src_first_order=np.full(7, 0.5))
    return np.asarray(ir)


@pytest.mark.parametrize("aniso", [False, True], ids=["default", "aniso"])
@pytest.mark.parametrize("case", [_closed, _open], ids=["closed", "open"])
def test_a_scene_built_slab_by_slab_equals_the_whole_grid(case, aniso, monkeypatch):
    """Every slab its own FdtdGrid on a box of whole x planes (no grid of the scene): the same
    floats as the grid of the whole scene, monopole and dipole runs, 3 slabs."""
    if not os.path.isfile(EMU) or not hasattr(N.load_library(EMU), "aur_fdtd_slabs_create"):
        pytest.skip("no emulation library with aur_fdtd_slabs_create")
    ref = _run(case, monkeypatch, None, aniso)
    out = _run(case, monkeypatch, "0,1,2", aniso)
    assert np.abs(ref).max() > 0
    assert np.array_equal(out, ref), np.abs(out - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("case", [_closed, _open], ids=["closed", "open"])
def test_slabs_share_one_subgrid_factor(case, monkeypatch):
    """A subgrid cap that only the largest slab exceeds: every slab falls back to factor 2 with
    it, as the grid of the whole scene does, and the floats stay the same."""
    if not os.path.isfile(EMU) or not hasattr(N.load_library(EMU), "aur_fdtd_slabs_create"):
        pytest.skip("no emulation library with aur_fdtd_slabs_create")
    from auralis.engine import conformal as C
    from auralis.engine import slabs as S
    cap = [10 ** 12]
    real = S.slab_boxes

    def spy(air, n_dev, npml):
        boxes = real(air, n_dev, npml)
        sizes = sorted((sb - sa) * air.shape[1] * air.shape[2] for sa, sb in boxes)
        assert sizes[0] < sizes[-1]
        cap[0] = 27 * sizes[-1] - 1
        return boxes
    monkeypatch.setattr(S, "slab_boxes", spy)
    monkeypatch.setattr(C, "subgrid_cap", lambda: cap[0])
    out = _run(case, monkeypatch, "0,1,2", True)
    assert cap[0] < 10 ** 12
    ref = _run(case, monkeypatch, None, True)
    assert np.abs(ref).max() > 0
    assert np.array_equal(out, ref), np.abs(out - ref).max() / np.abs(ref).max()
