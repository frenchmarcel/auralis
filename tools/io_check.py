import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.geometry import SceneData, box_triangles, get_preset  # noqa: E402

L = (3.0, 2.2, 1.9)


def _scene():
    tris, names = box_triangles(L, inward=True)
    wall = get_preset("plaster")
    tm = np.array([1 if n == "wall_x0" else 0 for n in names], dtype=np.int32)
    return SceneData.from_materials(tris, tm, [wall, wall], ["walls", "panel"])


def _run(sc, rcv, engine, **kw):
    t0 = time.time()
    ir, info = F.run_low_band(sc, np.array([0.5, 1.1, 0.95]), rcv, fs=8000, duration=0.25,
                              f_cross=700.0, speed_of_sound=343.0, engine=engine, **kw)
    return np.asarray(ir), info, time.time() - t0


def main():
    """GPU against the native CPU solver with more taps and receivers than one block holds
    (1500 receivers; a vibrating wall of > 1024 taps): the CUDA libs must agree to ~1e-6 and
    leave no receiver silent. AURALIS_CUDA_LIB picks the library under test."""
    sc = _scene()
    rng = np.random.default_rng(3)
    rcv = [np.array([0.4, 0.4, 0.4]) + rng.random(3) * (np.array(L) - 0.8) for _ in range(1500)]
    for label, kw in (("point source, 1500 receivers", {}),
                      ("vibrating wall, 1500 receivers", {"src_surface": "panel"})):
        cpu, info, tc = _run(sc, rcv, "native", **kw)
        try:
            gpu, ginfo, tg = _run(sc, rcv, "gpu", **kw)
            err = np.abs(gpu - cpu).max() / np.abs(cpu).max()
            zero = int((np.abs(gpu).max(axis=0) == 0).sum())
            print("%s: engine %s, %d air cells, max rel diff GPU-CPU %.2e, silent receivers %d "
                  "(CPU %.0f s, GPU %.0f s)" % (label, ginfo.engine, info.n_air, err, zero, tc, tg))
        except Exception as exc:
            print("%s: GPU FAILED: %s" % (label, exc))
    print("lib", os.environ.get("AURALIS_CUDA_LIB") or "default")


if __name__ == "__main__":
    main()
