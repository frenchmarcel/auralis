"""Bit-exact checks of the GPU splits without pytest (HPC nodes): a closed room and an open
scene, plain and with the anisotropic conformal correction (tangential links, ABI 2.7), run on
one device, split over the devices given, and as a kept-grid series; every output must equal
the single-device run. python tools/split_check.py 0,1 0,1,2,3"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.geometry import SceneData, add_box, make_shoebox  # noqa: E402

MAT = {"absorption": np.full(7, 0.12), "scattering": np.full(7, 0.1),
       "transmission": np.zeros(7)}


def _rot(sc, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc.tri_verts = np.ascontiguousarray(sc.tri_verts @ R.T, dtype=np.float32)
    return R


def closed():
    sc = make_shoebox((5.2, 3.1, 2.6), wall_material=MAT, floor_material=MAT,
                      ceiling_material=MAT)
    sc = add_box(sc, (0.9, 0.6, 0.8), (1.4, 0.9, 0.0), MAT)
    sc.mat_phys = [None] * sc.n_mats
    sc.mat_phys[1] = {"layers": [{"model": "panel", "m": 8.0, "q": 4.0},
                                 {"model": "air", "d": 0.08}]}
    R = _rot(sc, 17.0)
    return (sc, R @ np.array([0.8, 0.7, 1.4]),
            [R @ np.array([4.4, 2.4, 1.1]), R @ np.array([2.6, 1.5, 1.6])], 220.0)


def opened():
    g = np.array([[[-4, -3, 0], [4, -3, 0], [4, 3, 0]], [[-4, -3, 0], [4, 3, 0], [-4, 3, 0]]])
    sc = SceneData(g, np.zeros(2, np.int32), MAT["absorption"][None], MAT["scattering"][None],
                   MAT["transmission"][None])
    sc = add_box(sc, (1.5, 1.2, 2.0), (-0.5, 0.3, 0.0), MAT)
    R = _rot(sc, 7.0)
    return (sc, R @ np.array([-2.5, -0.5, 1.2]),
            [R @ np.array([2.5, -1.0, 1.6]), R @ np.array([0.3, -1.8, 0.9])], 150.0)


def series(case, devs, keep, aniso=False):
    if devs:
        os.environ["AURALIS_FDTD_GPUS"] = devs
    else:
        os.environ.pop("AURALIS_FDTD_GPUS", None)
    sc, src, rcvs, fc = case()
    shift = np.array([0.35, 0.2, 0.0])
    outs = []
    with (F.keep_grids() if keep else contextlib.nullcontext()):
        for k in range(2):
            ir, info = F.run_low_band(sc, src + k * shift, [r - k * shift for r in rcvs],
                                      fs=8000, duration=0.3, f_cross=fc, speed_of_sound=343.0,
                                      engine="gpu", conformal_aniso=aniso)
            outs.append(np.asarray(ir))
    return outs


def main():
    splits = sys.argv[1:] or ["0,1", "0,1,2,3"]
    ok = True
    for name, case, aniso in (("closed", closed, False), ("open", opened, False),
                              ("closed+aniso", closed, True), ("open+aniso", opened, True)):
        ref = series(case, None, False, aniso)
        # Kept grids use the indexed empty-box pass of open domains, fresh runs the dense one.
        # The two differ by ~1e-6 on a GPU (FMA), so kept series are compared to the indexed one.
        os.environ["AURALIS_FDTD_EMPTY_DENSE"] = "0"
        ref_kept = series(case, None, False, aniso)
        os.environ.pop("AURALIS_FDTD_EMPTY_DENSE", None)
        for devs in [None] + splits:
            for keep in (False, True):
                if devs is None and not keep:
                    continue
                out = series(case, devs, keep, aniso)
                r0 = ref_kept if keep else ref
                same = all(np.array_equal(a, b) for a, b in zip(r0, out))
                d = max(float(np.max(np.abs(a - b)) / np.max(np.abs(a))) for a, b in zip(r0, out))
                ok &= same
                print("%-12s devices %-8s kept %-5s: %s (max rel diff %.1e)"
                      % (name, devs or "one", keep, "IDENTICAL" if same else "DIFFERENT", d),
                      flush=True)
    print("SPLIT CHECK %s" % ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    main()
