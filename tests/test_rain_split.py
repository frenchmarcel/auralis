# -*- coding: utf-8 -*-
"""Split estimator (sphere detection + s*E rain): the simulated-energy
to EXACT image-lattice ratio must stay FLAT over time at zero
scattering, in python, C kernel and emulated CUDA kernel.
"""
import os
import sys

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from rain_slope_bench import flat_box, lattice_hist, run_tracer  # noqa: E402
from auralis.engine.native import find_library                   # noqa: E402

PASS = 0
FAIL = 0
SIZE = (10.0, 7.0, 3.0)
SRC = np.array([2.3, 1.7, 1.4])
RCV = np.array([7.1, 5.2, 1.8])
C = 343.0
BIN = 0.001
T = 0.6
WIN = 0.075
EMU = os.path.join(_ROOT, "auralis", "lib", "linux-x86_64",
                   "libauralis_cuda_emu.so")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %s %s" % (name, detail))


def _ratio(engine, n_rays, seed, lib_path=None):
    """(mean tail, max deviation, stats) of the windowed simulation to
    exact-lattice ratio (zero scattering), normalised to window 1."""
    sc = flat_box(SIZE, 0.15, 0.0)
    exact = lattice_hist(SIZE, SRC, RCV, 0.15, T, BIN, C)
    if lib_path is None:
        hist, stats, _ = run_tracer(sc, SRC, RCV, engine, n_rays, 400,
                                    T, BIN, C, seed=11)
    else:
        from auralis.engine import native
        eng = native.NativeEngine(sc, lib_path=lib_path)
        hist, _d, stats = eng.trace(SRC, RCV, n_rays, 400, T, BIN,
                                    int(round(T / BIN)), C, np.zeros(7),
                                    spec_exclude_order=0,
                                    energy_cap=1e-9, seed=11)
        hist = np.asarray(hist, np.float64)
    sim = np.asarray(hist).mean(axis=0)
    w = int(round(WIN / BIN))
    n_win = len(exact) // w
    se = np.array([exact[i * w:(i + 1) * w].sum() for i in range(n_win)])
    ss = np.array([sim[i * w:(i + 1) * w].sum() for i in range(n_win)])
    i0 = max(1, int(np.ceil((np.linalg.norm(RCV - SRC) / C) / WIN)))
    r = ss / np.maximum(se, 1e-300)
    r = r / r[i0]
    tail = float(np.nanmean(r[max(i0, n_win - 3):]))
    dev = float(np.nanmax(np.abs(r[i0:] - 1)))
    return tail, dev, stats


def test_python_reference():
    tail, dev, stats = _ratio("python", 60_000, seed=11)
    check("python: ratio tail %.3f (|.-1| < 0.12)" % tail,
          abs(tail - 1.0) < 0.12)
    check("python: max deviation %.3f (< 0.20)" % dev, dev < 0.20)
    check("python: jitter re-aiming active (retried=%d > 0)"
          % int(stats.get("retried", 0)), stats.get("retried", 0) > 0)


def test_native_kernel():
    if find_library() is None:
        print("native library MISSING: run native/build.sh")
        check("C kernel present", False)
        return
    tail, dev, _ = _ratio("native", 250_000, seed=11)
    check("C kernel: ratio tail %.3f (|.-1| < 0.05)" % tail,
          abs(tail - 1.0) < 0.05)
    check("C kernel: max deviation %.3f (< 0.10)" % dev, dev < 0.10)


def test_cuda_emulation():
    if not os.path.exists(EMU):
        print("CUDA emulation missing (%s): run "
              "native/build_cuda.sh --emulate -- DELIBERATE test skip"
              % EMU)
        return
    tail, dev, _ = _ratio("native", 250_000, seed=11, lib_path=EMU)
    check("CUDA kernel (emu): ratio tail %.3f (|.-1| < 0.05)" % tail,
          abs(tail - 1.0) < 0.05)
    check("CUDA kernel (emu): max deviation %.3f (< 0.10)" % dev,
          dev < 0.10)


if __name__ == "__main__":
    print("== split rain: python (reference) ==")
    test_python_reference()
    print("== split rain: C kernel ==")
    test_native_kernel()
    print("== split rain: emulated CUDA kernel ==")
    test_cuda_emulation()
    print("\n%d ok, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
