#!/usr/bin/env python3
"""Room bench with the tracer's two estimator corrections ALONE: per-band specular weight
(1 - s_b)/(1 - s_mean) and image sources weighted by (1 - s), without the finite-size
diffraction. Runs the rbs_scattering mode with every surface infinitely large, so that
s_d = 0. Usage: bras_estimator_only.py manifest_rbs.json -o out_dir"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from auralis.engine import geometry  # noqa: E402

HUGE = 1.0e6  # metres: f_lim ~ 1e-10 Hz, so K = 1 and s_d = 0 at every band


def infinite_dims(scene, *args, **kwargs):
    """surface_dims replacement: every triangle on an infinite surface."""
    n = len(scene.tri_verts)
    print("[estimator-only] surface sizes forced to %.0e m (no finite-size scattering)" % HUGE)
    return np.full(n, HUGE, np.float32), np.full(n, HUGE, np.float32)


def main():
    import json
    man = json.load(open(sys.argv[1]))
    if not man.get("engine", {}).get("rbs_scattering"):
        raise SystemExit("the manifest must set engine.rbs_scattering = true")
    geometry.surface_dims = infinite_dims
    import bras_bench
    return bras_bench.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
