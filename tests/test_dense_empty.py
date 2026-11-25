# -*- coding: utf-8 -*-
import ctypes
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import dense  # noqa: E402
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine import native as N  # noqa: E402


def _lib(kind):
    if kind == "emu":
        path = os.environ.get("AURALIS_CUDA_EMU_LIB") or os.path.join(
            ROOT, "auralis", "lib", N.platform_tag(), "libauralis_cuda_emu.so")
    else:
        path = N.find_gpu_library()
    if not path or not os.path.isfile(path):
        return None
    lib = ctypes.CDLL(path)
    if not dense.available(lib):
        return None
    if kind == "gpu" and not int(lib.aur_cuda_available()):
        return None
    return lib


@pytest.mark.parametrize("kind", ["emu", "gpu"])
@pytest.mark.parametrize("order,npml", [(4, 8), (2, 6), (4, 0)])
def test_dense_empty_box_equals_the_indexed_empty_grid(kind, order, npml):
    """The second pass of open domains without neighbour tables gives the same floats."""
    lib = _lib(kind)
    if lib is None:
        pytest.skip("no %s library with aur_fdtd_dense_empty" % kind)
    shape, dx, c = (36, 31, 27), 0.05, 343.0
    origin = np.array([-0.4, 0.3, 0.1])
    pml = {"n": npml} if npml else None
    vox = F.VoxelGrid(solid=np.zeros(shape, bool), mat=np.full(shape, -1, np.int32),
                      origin=origin, dx=dx)
    vox.air = ~vox.solid
    g = F.FdtdGrid(vox, np.zeros(1), c, sigma=None, gamma_border=1.0, order=order, pml=pml)
    n_steps = 500
    pulse = F.source_pulse(g.dt, 250.0, n_steps)
    src = origin + dx * np.array([12.3, 14.6, 9.2])
    rcvs = [origin + dx * np.array([25.5, 9.1, 18.8]), origin + dx * np.array([3.2, 27.7, 2.4])]
    axis = np.array([0.3, -0.5, 0.8])
    ref = F.run_grid_native(g, pulse, g.taps(src), [g.taps(r) for r in rcvs], n_steps, lib=lib)
    ref_d = F.run_grid_native(g, pulse, F._dipole_taps(g, src, axis, 2 * dx),
                              [g.taps(r) for r in rcvs], n_steps, lib=lib)
    box = dense.DenseBox(shape, origin, dx)
    rts = [box.taps(r) for r in rcvs]
    out = dense.run_empty(lib, shape, origin, dx, c, order, pml, 1.0, pulse, box.taps(src),
                          rts, n_steps)
    out_d = dense.run_empty(lib, shape, origin, dx, c, order, pml, 1.0, pulse,
                            dense.dipole_taps(box, src, axis, 2 * dx), rts, n_steps)
    assert np.abs(ref).max() > 0
    if kind == "emu":
        assert np.array_equal(out, ref) and np.array_equal(out_d, ref_d)
    else:                       # nvcc contracts the two kernels' sums into FMAs differently
        for a, b in ((out, ref), (out_d, ref_d)):
            assert np.abs(a - b).max() <= 1e-5 * np.abs(b).max(), np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("kind", ["emu", "gpu"])
def test_dense_empty_box_split_over_devices_gives_the_same_floats(kind):
    """3 slabs of whole x planes (their copies exchanged every step) against one device."""
    lib = _lib(kind)
    if lib is None or not hasattr(lib, "aur_fdtd_dense_empty_m"):
        pytest.skip("no %s library with aur_fdtd_dense_empty_m" % kind)
    shape, dx, c, pml = (48, 31, 27), 0.05, 343.0, {"n": 8}
    origin = np.array([-0.4, 0.3, 0.1])
    box = dense.DenseBox(shape, origin, dx)
    dt = F._LAMBDA4 * dx / c
    n_steps = 400
    pulse = F.source_pulse(dt, 250.0, n_steps)
    src = origin + dx * np.array([23.3, 14.6, 9.2])            # in the middle slab
    rcvs = [origin + dx * np.array([30.5, 9.1, 18.8]), origin + dx * np.array([15.9, 27.7, 2.4]),
            origin + dx * np.array([24.0, 12.2, 11.1])]          # one on a cut's copy
    rts = [box.taps(r) for r in rcvs]
    one = dense.run_empty(lib, shape, origin, dx, c, 4, pml, 1.0, pulse, box.taps(src), rts,
                          n_steps)
    devs = [0, 0, 0] if kind == "gpu" else [0, 1, 2]
    split = dense.run_empty(lib, shape, origin, dx, c, 4, pml, 1.0, pulse, box.taps(src), rts,
                            n_steps, devs=devs)
    assert np.abs(one).max() > 0 and np.array_equal(split, one)
