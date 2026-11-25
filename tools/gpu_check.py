#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the real CUDA engine: loading, parity against the CPU native
engine on a full bake, and timing. Run it from the project root. The
expected parity is STATISTICAL (a few % per band), not bit exact.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import gpu_available, native            # noqa: E402
from auralis.engine import materials as mat_lib             # noqa: E402
from auralis.engine.geometry import SceneData               # noqa: E402
from auralis.engine.pipeline import (BakeConfig, ReceiverDef,  # noqa: E402
                                     SourceDef, bake)
import auralis.engine.rirmetrics as rm                      # noqa: E402


def box_scene():
    X, Y, Z = 6.2, 8.6, 3.0
    v = np.array([[bx * X, by * Y, bz * Z] for bx in (0, 1)
                  for by in (0, 1) for bz in (0, 1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4),
             (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]
    tris = []
    for a, b, c, d in quads:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    m = {"absorption": np.full(7, 0.10, np.float32),
         "scattering": np.full(7, 0.15, np.float32),
         "transmission": np.zeros(7, np.float32)}
    a, s, t = mat_lib.build_tables([m])
    return SceneData(tri_verts=np.ascontiguousarray(np.array(tris)),
                     tri_mat=np.zeros(12, np.int32), absorption=a,
                     scattering=s, transmission=t, mat_names=["m"])


def main():
    print("gpu_available() :", gpu_available())
    gp = native.find_gpu_library()
    print("library         :", gp or "(not found)")
    if not gpu_available():
        print("-> no usable GPU here: build with "
              "native/build_cuda.sh on an NVIDIA machine, then "
              "run this script again from the project root.")
        return 1

    scene = box_scene()
    S = SourceDef(name="s", position=np.array([4.0, 1.8, 1.5]))
    R = ReceiverDef(name="r", position=np.array([2.1, 5.7, 1.3]))

    # ---- RAW tracer (direct call, no pipeline, no synthesis) ----
    from auralis.engine import air as airmod
    air = airmod.air_attenuation_db_per_m(
        np.array(rm.CENTERS, float), 20.0, 50.0)
    hists = {}
    for nom, lp in (("natif", None), ("cuda", gp)):
        eng = native.NativeEngine(scene, lib_path=lp)
        t0 = time.time()
        h = eng.trace(np.array([4.0, 1.8, 1.5]),
                      np.array([2.1, 5.7, 1.3]), 400000, 200, 1.5,
                      1e-3, 1500, 343.0, air, spec_exclude_order=2,
                      seed=7)[0]
        dt = time.time() - t0
        hists[nom] = (h, dt)
        print("raw trace %-5s : %6.2f s  (version %s)"
              % (nom, dt, eng.version))
    eb = [(hists["cuda"][0][b].sum(), hists["natif"][0][b].sum())
          for b in range(7)]
    dmax = max(abs(a / max(b, 1e-30) - 1) for a, b in eb) * 100
    print("raw trace: max per-band energy gap %.2f %% | "
          "speed-up x%.1f"
          % (dmax, hists["natif"][1] / max(hists["cuda"][1], 1e-9)))

    out = {}
    for eng in ("native", "gpu"):
        cfg = BakeConfig(normalize=False, max_time=1.5, n_rays=400000,
                         ism_order=2, seed=7, fs=44100, engine=eng,
                         diffraction=False)
        t0 = time.time()
        res = bake(scene, S, R, cfg)
        dt = time.time() - t0
        ir = res.ir.mean(axis=1)
        e = np.array([float((rm.band_split(ir, res.fs)[b] ** 2).sum())
                      for b in range(rm.N_BANDS)])
        m = rm.band_metrics(ir, res.fs)
        out[eng] = (e, m, dt)
        print("%-6s : bake %5.2f s | T30(1k) %.2f s | C50(1k) %+.2f dB"
              % (eng, dt, m["T30"][3] or -1, m["C50"][3] or 0))
    # ---- FDTD: same small box, 400 Hz crossover, both engines ----
    for eng in ("native", "gpu"):
        cfg = BakeConfig(normalize=False, max_time=0.8, n_rays=20000,
                         ism_order=1, seed=7, fs=44100, engine=eng,
                         fdtd=True, fdtd_fcross=400.0,
                         diffraction=False)
        t0 = time.time()
        res = bake(scene, S, R, cfg)
        dt = time.time() - t0
        ir = res.ir.mean(axis=1)
        el = np.array([float((rm.band_split(ir, res.fs)[b] ** 2).sum())
                       for b in (0, 1, 2)])
        out["fdtd_" + eng] = (el, dt)
        print("FDTD %-6s : bake %5.2f s | energy 125/250/500 %s"
              % (eng, dt, np.round(el / el.max(), 3)))
    dF = 100.0 * np.abs(out["fdtd_gpu"][0] / out["fdtd_native"][0]
                        - 1.0)
    print("FDTD: max low-band energy gap GPU vs native %.2f %% | "
          "speed-up x%.1f"
          % (dF.max(), out["fdtd_native"][1]
             / max(out["fdtd_gpu"][1], 1e-9)))

    dE = 100.0 * np.abs(out["gpu"][0] / out["native"][0] - 1.0)
    dT = abs((out["gpu"][1]["T30"][3] or 0) -
             (out["native"][1]["T30"][3] or 0))
    print("per-band energy gap GPU vs native: max %.2f %% "
          "(expected < ~3 %%)" % dE.max())
    print("T30(1k) gap: %.3f s | tracer speed-up: x%.1f"
          % (dT, out["native"][2] / max(out["gpu"][2], 1e-9)))
    ok = dE.max() < 3.0 and dT < 0.05
    print("VERDICT:", "OK - the GPU reproduces the native engine" if ok
          else "SUSPECT GAP - report this output")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
