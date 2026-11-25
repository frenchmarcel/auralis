"""A walk along a synthetic street while cars pass, rendered binaurally with the wave band
animated (sources and listener both moving, the grid kept on the GPUs between runs): run
count, timing per stage, and the WAV.
python tools/street_walk.py [--fc 250] [--walk 20] [--length 200] [--out walk.wav]
Multi-GPU: AURALIS_FDTD_GPUS=0,1,2,3."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auralis.engine.motion import (MotionConfig, MovingReceiver, MovingSource,  # noqa: E402
                                   Trajectory, render_scene_moving)
from auralis.engine.pipeline import BakeConfig, wave_ear_half  # noqa: E402
from auralis.engine.wavio import write_wav  # noqa: E402
from street_bench import street_scene  # noqa: E402


def engine_noise(dur, fs, f0, seed):
    """A car going by: firing harmonics of f0 (engine), low rumble and tyre noise."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k in range(1, 12):
        x += np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) / k ** 0.7
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    tyre = np.fft.irfft(spec * np.exp(-((f - 900.0) / 600.0) ** 2), n)
    rumble = np.fft.irfft(spec * (1.0 / np.maximum(f, 20.0)) * (f < 300), n)
    y = x / np.abs(x).max() + 0.6 * tyre / np.abs(tyre).max() + 0.8 * rumble / np.abs(rumble).max()
    return 0.3 * y / np.abs(y).max()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc", type=float, default=250.0, help="crossover (Hz)")
    ap.add_argument("--walk", type=float, default=20.0, help="walk duration (s)")
    ap.add_argument("--length", type=float, default=200.0, help="street length (m)")
    ap.add_argument("--fs", type=int, default=32000)
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--max-runs", type=int, default=100000)
    ap.add_argument("--out", default="street_walk.wav")
    ap.add_argument("--hrtf", default="", help="measured head (.npz); '' = spherical model")
    ap.add_argument("--ear-half", type=float, default=0.0,
                    help="wave-band ear probes, half spacing (m); 0 = from the head")
    args = ap.parse_args()
    os.environ.setdefault("AURALIS_FDTD_CONFORMAL", "0")    # open scene, sharp edges
    L, W, T = args.length, 15.0, args.walk
    sc, R = street_scene(L, W, rot_deg=0.0, cars=4)
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.array([1.0, 0.0, 0.0])
    x0 = 0.42 * L
    walker = MovingReceiver(name="walker", trajectory=Trajectory(
        times=[0.0, T], positions=[[x0, W / 2 - 1.2, 1.6], [x0 + 1.4 * T, W / 2 - 1.2, 1.6]],
        forwards=[fwd, fwd], ups=[up, up]))
    v1, v2 = 12.0, 10.0
    c1 = MovingSource(name="car east", trajectory=Trajectory(
        times=[0.0, T], positions=[[x0 - 0.5 * v1 * T + 15.0, -0.5, 0.5],
                                   [x0 + 0.5 * v1 * T + 15.0, -0.5, 0.5]]))
    c2 = MovingSource(name="car west", trajectory=Trajectory(
        times=[0.0, T], positions=[[x0 + 0.5 * v2 * T + 10.0, 0.5, 0.5],
                                   [x0 - 0.5 * v2 * T + 10.0, 0.5, 0.5]]))
    for c in (c1, c2):                       # keep the cars on the street
        c.trajectory.positions[:, 0] = np.clip(c.trajectory.positions[:, 0], 3.0, L - 3.0)
    cfg = BakeConfig(fs=args.fs, n_rays=args.rays, ism_order=2, fdtd=False,
                     fdtd_fcross=args.fc, fdtd_conformal=False, max_time=1.0,
                     fdtd_max_cells=2 ** 31 - 2, normalize=True, seed=11,
                     hrtf_path=args.hrtf, wave_ear_half=args.ear_half)
    mc = MotionConfig(base=cfg, wave=True, wave_max_runs=args.max_runs, reverb_dt=0.5)
    print("[walk] head: %s, wave-band ears at +-%.1f cm"
          % (os.path.basename(args.hrtf) or "spherical model", 100.0 * wave_ear_half(cfg)),
          flush=True)
    dry = [engine_noise(T, args.fs, 31.0, 1), engine_noise(T, args.fs, 37.0, 2)]
    stage = {"name": "", "t": time.time()}

    def progress(frac, st):
        name = st.split(":")[0] if "wave band" not in st else "wave band"
        if name != stage["name"]:
            print("[walk] %.0f s  %s" % (time.time() - stage["t"], st), flush=True)
            stage["name"] = name
        return True
    t0 = time.time()
    mix, info = render_scene_moving(sc, [(c1, dry[0]), (c2, dry[1])], walker, mc,
                                    progress=progress)
    wall = time.time() - t0
    write_wav(args.out, mix, args.fs)
    print("[walk] street %.0f m, walk %.0f s at 1.4 m/s, crossover %.0f Hz: %.0f s in all"
          % (L, T, args.fc, wall))
    print("[walk] wave band: %s" % (info.get("wave") or {}).get("note"))
    for s in info["sources"]:
        print("[walk] %s timings %s" % (s.get("source", "?"),
                                        {k: round(v, 1) for k, v in s["timings"].items()}))
    print("[walk] wrote %s (%.1f s, %d Hz, binaural)" % (args.out, len(mix) / args.fs, args.fs))


if __name__ == "__main__":
    main()
