"""Wave band of a synthetic street (buildings, side streets, cars): grid size, preparation and
GPU time, GPU and host memory. python tools/street_bench.py --fc 250 [--length 200 ...]"""
import argparse
import os
import resource
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine.geometry import SceneData, add_box  # noqa: E402

FACADE = (0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05)
GROUND = (0.01, 0.01, 0.015, 0.02, 0.02, 0.02, 0.02)
CAR = (0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05)


def street_scene(length=200.0, width=15.0, depth=12.0, h_range=(15.0, 21.0), cars=6,
                 rot_deg=7.0, seed=1):
    """Ground, two rows of building blocks (side streets between some), cars on the road.
    x along the street, y across (road between -width/2 and +width/2), z up."""
    rng = np.random.default_rng(seed)
    ab = np.array([FACADE, GROUND, CAR])
    y0, y1 = -width / 2 - depth, width / 2 + depth
    g = np.array([[[0, y0, 0], [length, y0, 0], [length, y1, 0]],
                  [[0, y0, 0], [length, y1, 0], [0, y1, 0]]], float)
    sc = SceneData(g, np.full(2, 1, np.int32), ab, np.full((3, 7), 0.1), np.zeros((3, 7)),
                   mat_names=["facade", "ground", "car"])
    fac = {"absorption": ab[0], "scattering": np.full(7, 0.1), "transmission": np.zeros(7)}
    car = {"absorption": ab[2], "scattering": np.full(7, 0.1), "transmission": np.zeros(7)}
    for side in (-1, 1):
        x = 0.0
        while x < length - 6.0:
            w = min(rng.uniform(12.0, 30.0), length - x)
            h = rng.uniform(*h_range)
            ys = width / 2 if side > 0 else -width / 2 - depth
            sc = add_box(sc, (w, depth, h), (x, ys, 0.0), fac)
            x += w + (8.0 if rng.random() < 0.25 else 0.0)        # side street
    for k in range(cars):
        x = (k + 0.5) * length / cars + rng.uniform(-5.0, 5.0)
        lane = -2.8 if k % 2 else 2.0
        sc = add_box(sc, (4.3, 1.8, 1.5), (x - 2.15, lane - 0.9, 0.0), car)
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc.tri_verts = np.ascontiguousarray(sc.tri_verts @ R.T, dtype=np.float32)
    return sc, R


class GpuMem(threading.Thread):
    """Peak memory.used (MiB) of the physical GPUs listed, polled twice a second."""
    def __init__(self, gpus):
        super().__init__(daemon=True)
        self.gpus, self.peak, self.stop = gpus, 0, False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits", "-i", self.gpus],
                                     capture_output=True, text=True, timeout=10).stdout
                self.peak = max(self.peak, sum(int(v) for v in out.split()))
            except Exception:
                pass
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc", type=float, default=250.0, help="crossover (Hz)")
    ap.add_argument("--duration", type=float, default=1.5, help="IR length (s)")
    ap.add_argument("--length", type=float, default=200.0)
    ap.add_argument("--width", type=float, default=15.0)
    ap.add_argument("--fs", type=int, default=8000)
    ap.add_argument("--rot", type=float, default=7.0,
                    help="street direction against the grid axes (degrees)")
    ap.add_argument("--conformal", action="store_true",
                    help="conformal correction ON (default OFF: open scene with sharp edges)")
    ap.add_argument("--aniso", action="store_true",
                    help="the ANISOTROPIC conformal correction (used by the app for open scenes, "
                         "split over several GPUs since ABI 2.7)")
    ap.add_argument("--save", default="", help="write the IR to this .npy")
    ap.add_argument("--max-cells", type=float, default=2 ** 31 - 2,
                    help="grid size limit (cells); above 2^31 air cells the grid is built slab "
                         "by slab on AURALIS_FDTD_GPUS")
    args = ap.parse_args()
    if not (args.conformal or args.aniso):
        os.environ["AURALIS_FDTD_CONFORMAL"] = "0"
    sc, R = street_scene(args.length, args.width, rot_deg=args.rot)
    src = R @ np.array([0.3 * args.length, -1.0, 0.5])
    rcv = [R @ np.array([0.6 * args.length, args.width / 2 - 1.0, 1.6])]
    gpu_time = [0.0]
    orig = F.run_grid_native

    def timed(*a, **k):
        t = time.time()
        try:
            return orig(*a, **k)
        finally:
            gpu_time[0] += time.time() - t
    F.run_grid_native = timed
    from auralis.engine import dense, slabs
    for mod, name in ((dense, "run_empty"), (slabs.SlabGrid, "runner")):
        def wrap(f):
            def w(*a, **k):
                t = time.time()
                try:
                    return f(*a, **k)
                finally:
                    gpu_time[0] += time.time() - t
            return w
        setattr(mod, name, wrap(getattr(mod, name)))
    gpus = os.environ.get("AURALIS_FDTD_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpus = ",".join(sorted(set(g for g in gpus.replace(" ", "").split(",")[:8] if g)))
    mon = GpuMem(gpus)
    mon.start()
    t0 = time.time()
    ir, info = F.run_low_band(sc, src, rcv, fs=args.fs, duration=args.duration,
                              f_cross=args.fc, speed_of_sound=343.0,
                              max_cells=int(args.max_cells),
                              conformal_aniso=args.aniso)
    if args.save:
        np.save(args.save, np.asarray(ir))
    total = time.time() - t0
    mon.stop = True
    mon.join(2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    upd = 2.0 * info.n_air * info.n_steps if info.is_open else info.n_air * info.n_steps
    print("street %.0f x %.0f m at %.0f deg, fc %.0f Hz, dx %.4f m: %.1f M cells, %.1f M air, "
          "%d steps, %s, open %s" % (args.length, args.width, args.rot, args.fc, info.dx,
                                     info.n_cells / 1e6,
                           info.n_air / 1e6, info.n_steps, info.engine, info.is_open))
    print("total %.1f s: solver %.1f s (%.2f Gcell-updates/s), rest %.1f s; GPU peak %.1f GiB "
          "over GPU(s) %s (%.0f B per air cell); host peak %.1f GB (%.0f B per cell)" % (
              total, gpu_time[0], upd / max(gpu_time[0], 1e-9) / 1e9, total - gpu_time[0],
              mon.peak / 1024.0, gpus, mon.peak * 2 ** 20 / max(info.n_air, 1), rss,
              rss * 1e9 / max(info.n_cells, 1)))
    if not np.all(np.isfinite(ir)):
        print("WARNING: non-finite IR")


if __name__ == "__main__":
    main()
