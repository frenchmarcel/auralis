# -*- coding: utf-8 -*-
"""Standalone demo: small furnished club, binaural impulse response
bake, then auralization of test signals.
Usage: python3 examples/standalone_demo.py [out_dir]
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from auralis.engine import (BakeConfig, ReceiverDef, SourceDef, add_box, bake,
                            fft_convolve, format_report, make_shoebox,
                            make_test_signal, native_available,
                            normalize_peak, write_wav)


def build_club():
    scene = make_shoebox((12.0, 8.0, 4.0),
                         wall_material="brick",
                         floor_material="parquet",
                         ceiling_material="acoustic_tile")
    scene = add_box(scene, (3.0, 1.0, 1.1), (7.5, 0.5, 0.0),
                    "concrete", "bar")
    scene = add_box(scene, (0.1, 5.0, 3.2), (0.6, 1.5, 0.0),
                    "curtain_heavy", "curtain")
    scene = add_box(scene, (1.2, 1.2, 0.9), (3.0, 5.8, 0.0),
                    "wood_panel", "table")
    return scene


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo_out"
    os.makedirs(out_dir, exist_ok=True)

    scene = build_club()
    print("Scene: %d triangles, %d materials (%s)"
          % (scene.n_tris, scene.n_mats, ", ".join(scene.mat_names)))
    print("Native engine:", "yes" if native_available() else
          "no (NumPy fallback, slower)")

    src = SourceDef("scene_PA", np.array([10.0, 4.0, 2.2]), gain_db=0.0)
    rcv = ReceiverDef("listener", np.array([3.5, 4.0, 1.7]),
                      forward=np.array([1.0, 0.0, 0.0]),   # faces the stage
                      up=np.array([0.0, 0.0, 1.0]))

    cfg = BakeConfig(
        n_rays=150_000 if native_available() else 8_000,
        max_bounces=200,
        ism_order=2,
        bin_dt=0.001,
        fs=48_000,
        temperature_c=21.0,
        humidity_percent=45.0,
        seed=2026,
    )

    last = [-1]

    def progress(frac, stage):
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print("\r  [%3d%%] %s" % (pct, stage), end="", flush=True)
        return True

    t0 = time.time()
    res = bake(scene, src, rcv, cfg, progress=progress)
    print("\nBake in %.2f s (engine %s)  -  rays: %d, mean bounces: %.1f,"
          " deposits: %d"
          % (time.time() - t0, res.engine_used, res.stats["rays"],
             res.stats["mean_bounces"], res.stats["deposits"]))

    ir_path = os.path.join(out_dir, "ir_binaurale.wav")
    write_wav(ir_path, res.ir, res.fs)
    print("Binaural IR ->", ir_path,
          "(%.2f s, %d deterministic events)"
          % (res.ir.shape[0] / res.fs, len(res.events)))

    print("\nAcoustic report (ISO 3382):")
    print(format_report(res.metrics))
    with open(os.path.join(out_dir, "rapport.json"), "w") as f:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o
        json.dump(clean(res.metrics), f, indent=2)

    for kind in ("clicks", "speechish", "noise_bursts"):
        dry = make_test_signal(res.fs, kind, duration=4.0)
        wet = fft_convolve(dry, res.ir)
        mix = normalize_peak(wet, 0.95)
        p = os.path.join(out_dir, "auralise_%s.wav" % kind)
        write_wav(p, mix, res.fs)
        write_wav(os.path.join(out_dir, "sec_%s.wav" % kind), dry, res.fs)
        print("Auralization '%s' -> %s" % (kind, p))

    print("\nDone. Listen to sec_*.wav then auralise_*.wav on headphones.")


if __name__ == "__main__":
    main()
