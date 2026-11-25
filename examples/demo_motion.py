# -*- coding: utf-8 -*-
"""Standalone demo of moving sources: a flyby through the club.
Writes to motion_out/ the dry signal, the animated binaural render and
a duo (flyby + static pad on stage).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import (  # noqa: E402
    BakeConfig, MotionConfig, MovingSource, ReceiverDef, SourceDef,
    Trajectory, make_test_signal, render_moving, render_scene_moving,
    write_wav,
)
from examples.standalone_demo import build_club  # noqa: E402

FS = 48_000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "motion_out")


def engine_like(duration):
    """Engine-like sound: a clear fundamental (obvious Doppler) plus
    2-7 kHz high content, which makes the panning audible."""
    t = np.arange(int(duration * FS)) / FS
    f0 = 165.0
    x = np.zeros_like(t)
    for k in range(1, 37):                       # harmonics up to ~6 kHz
        x += (1.0 / k ** 0.8) * np.sin(2 * np.pi * f0 * k * t + 0.7 * k)
    # slightly warbling turbine whistle
    x += 0.35 * np.sin(2 * np.pi * 2400.0 * t
                       + 4.0 * np.sin(2 * np.pi * 2.6 * t))
    # 2-7 kHz band-pass hiss
    rng = np.random.default_rng(3)
    w = rng.standard_normal(len(t))
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(len(t), 1.0 / FS)
    W *= np.exp(-0.5 * ((np.log(np.maximum(f, 1.0) / 3800.0)) / 0.45) ** 2)
    x += 0.5 * np.fft.irfft(W, len(t))
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (duration - t) / 0.1)
    x *= env
    return 0.6 * x / np.abs(x).max()


def main():
    os.makedirs(OUT, exist_ok=True)
    scene = build_club()

    dur = 6.0
    dry = engine_like(dur)
    write_wav(os.path.join(OUT, "flyby_sec.wav"), dry, FS)

    # Listener at the centre, facing the stage (+X, as in the static demo)
    listener = ReceiverDef("listener", np.array([3.5, 4.0, 1.7]),
                           forward=(1.0, 0.0, 0.0))

    # The source crosses the room (12 x 8) along Y, 0.8 m in front of the
    # listener (x = 4.3): round trip y = 0.6 -> 7.4 -> 0.6 in 6 s
    # (~4.5 m/s). The speaker KEEPS FACING THE LISTENER throughout: the
    # trajectory also carries the front-axis animation (33 nodes).
    times = np.linspace(0.0, dur, 33)
    yy = np.interp(times, [0.0, dur * 0.5, dur], [0.6, 7.4, 0.6])
    way = np.stack([np.full_like(yy, 4.3), yy, np.full_like(yy, 1.5)],
                   axis=1)
    fwds = listener.position - way
    flyby = MovingSource("flyby", Trajectory(times, way, forwards=fwds),
                         directivity="speaker_2way")

    cfg = BakeConfig(n_rays=60_000, seed=2026, fs=FS, normalize=False)
    mc = MotionConfig(base=cfg, reverb_dt=0.5, doppler=True)

    print("Rendering the flyby (Doppler + panning)...")
    t0 = time.time()
    last = [-1]

    def prog(fr, st):
        p = int(fr * 100)
        if p // 10 != last[0]:
            last[0] = p // 10
            print("  %3d %%  %s" % (p, st))
        return True

    out, info = render_moving(scene, flyby, listener, dry, mc, progress=prog)
    peak = np.abs(out).max()
    out *= 10 ** (-1.0 / 20.0) / peak
    write_wav(os.path.join(OUT, "flyby_binaural.wav"), out, FS)
    print("  engine %s | %d late keys | %d hops | %.1f s of compute"
          % (info["engine_used"], info["n_keys"], info["n_hops"],
             time.time() - t0))

    # Two simultaneous sources: the flyby + a static pad on stage
    pad = make_test_signal(FS, "speechish", duration=dur, seed=11)
    pad = 0.5 * pad / np.abs(pad).max()
    stage = SourceDef("scene_pa", (10.0, 4.0, 2.2), gain_db=-4.0,
                      directivity="voice", forward=(-1.0, 0.0, 0.0))
    print("Rendering two sources (flyby + voice on stage)...")
    t0 = time.time()
    mix, _ = render_scene_moving(scene, [(flyby, dry), (stage, pad)],
                                 listener, mc, progress=prog)
    write_wav(os.path.join(OUT, "duo_binaural.wav"), mix, FS)
    print("  %.1f s of compute" % (time.time() - t0))
    print("Done -> %s" % os.path.abspath(OUT))


if __name__ == "__main__":
    main()
