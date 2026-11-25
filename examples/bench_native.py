# -*- coding: utf-8 -*-
"""Auralis kernel benchmark: python / native CPU / CUDA. Traces the same
scene with every available backend and prints throughput in
Msegments/s.  Usage: python3 examples/bench_native.py [n_rays]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import native  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, \
    speed_of_sound  # noqa: E402
from auralis.engine.stochastic import trace_python  # noqa: E402
from examples.standalone_demo import build_club  # noqa: E402


def bench(name, fn, n_rays):
    t0 = time.time()
    hist, dirs, st = fn()
    dt = time.time() - t0
    segs = st["mean_bounces"] * n_rays
    print("%-22s %7.2f s   %8.2f Msegments/s   E=%.4e  bounces=%.1f"
          % (name, dt, segs / dt / 1e6, hist.sum(), st["mean_bounces"]))
    return hist


def main():
    n_rays = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    sc = build_club()
    src = np.array([10.0, 4.0, 2.2])
    rcv = np.array([3.5, 4.0, 1.7])
    air = air_attenuation_db_per_m()
    c = speed_of_sound(20.0)
    max_time, bin_dt = 1.2, 0.001
    n_bins = int(max_time / bin_dt)
    kw = dict(spec_exclude_order=2, energy_cap=1e-6, seed=42)
    print("Scene: club 48 tris | %d rays | IR %.1f s\n" % (n_rays, max_time))

    ref = None
    gp = native.find_gpu_library()
    if gp and native.gpu_available():
        eng = native.NativeEngine(sc, lib_path=gp)
        print("GPU:", eng.version)
        ref = bench("CUDA (GPU)", lambda: eng.trace(
            src, rcv, n_rays, 200, max_time, bin_dt, n_bins, c, air, **kw),
            n_rays)
        eng.close()

    if native.is_available():
        eng = native.NativeEngine(sc)
        h = bench("native CPU (all cores)", lambda: eng.trace(
            src, rcv, n_rays, 200, max_time, bin_dt, n_bins, c, air,
            n_threads=0, **kw), n_rays)
        eng.close()
        ref = ref if ref is not None else h

    emu = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "auralis", "lib",
        native.platform_tag(), "libauralis_cuda_emu" +
        (".dylib" if sys.platform == "darwin" else
         ".dll" if sys.platform.startswith("win") else ".so"))
    if os.path.isfile(emu):
        eng = native.NativeEngine(sc, lib_path=emu)
        n_small = min(n_rays, 60_000)
        bench("CUDA-emu (1 core)*", lambda: eng.trace(
            src, rcv, n_small, 200, max_time, bin_dt, n_bins, c, air, **kw),
            n_small)
        eng.close()
        print("  * validation tool, not optimized")

    n_py = min(n_rays, 30_000)
    bench("python (numpy)", lambda: trace_python(
        sc, src, rcv, n_py, 200, max_time, bin_dt, n_bins, c, air, **kw),
        n_py)

    if ref is not None:
        print("\nPer-band energies (reference):",
              np.round(ref.sum(axis=1), 4))


if __name__ == "__main__":
    main()
