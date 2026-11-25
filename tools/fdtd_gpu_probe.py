#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a scene from nan_dump.npz on the GPU FDTD with the divergence
scanner armed (AUR_FDTD_DEBUG): names the offending array and step.

Usage:
    python3 tools/fdtd_gpu_probe.py nan_dump.npz [f_cross] [dur_s] [engine]

engine: gpu (default) | native -- native has no step-by-step scanner.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AUR_FDTD_DEBUG", "100")

from auralis.engine.geometry import SceneData        # noqa: E402
from auralis.engine.fdtd import run_low_band          # noqa: E402
from auralis.engine import gpu_available              # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = np.load(sys.argv[1], allow_pickle=True)
    fc = float(sys.argv[2]) if len(sys.argv) > 2 else 398.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    eng = sys.argv[4] if len(sys.argv) > 4 else "gpu"
    absorber = sys.argv[5] if len(sys.argv) > 5 else "cpml"
    order = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    print("absorber:", absorber, "| order:", order,
          "| lam4:", os.environ.get("AUR_FDTD_LAM4", "0.499"))
    print("requested engine:", eng)
    print("gpu_available:", gpu_available(),
          "| AUR_FDTD_DEBUG =", os.environ["AUR_FDTD_DEBUG"])
    scene = SceneData(
        tri_verts=np.ascontiguousarray(
            d["tri_verts"].astype(np.float64)),
        tri_mat=d["tri_mat"].astype(np.int32),
        absorption=d["absorption"], scattering=d["scattering"],
        transmission=d["transmission"],
        mat_names=["m%d" % i for i in range(d["absorption"].shape[0])])
    src = d["src"].astype(float)
    rcv = d["rcv"].astype(float)
    try:
        ir, info = run_low_band(scene, src, [rcv], 44100, dur, fc,
                                343.0, engine=eng, order=order,
                                max_cells=12_000_000,
                                boundary_absorber=absorber)
    except Exception as exc:
        print("EXCEPTION:", exc)
        print("-> the [aur_fdtd dbg] message on stderr (just above) "
              "names the offending array and step.")
        return 2
    x = ir[:, 0] if ir.ndim > 1 else ir
    print("full run: %d cells, %d steps, engine=%s"
          % (info.n_air, info.n_steps, info.engine))
    print("finite = %s | NaN = %d | max|p| = %.3e"
          % (bool(np.isfinite(x).all()),
             int((~np.isfinite(x)).sum()),
             float(np.nanmax(np.abs(x)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
