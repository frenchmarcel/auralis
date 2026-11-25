"""Energy per time window, relative to the direct sound, simulated against measured, per
octave and averaged over the pairs: python BRAS/bras_energy_windows.py <manifest> <run_dir>"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from auralis.engine import rirmetrics as rm  # noqa: E402
from auralis.engine.wavio import read_wav  # noqa: E402
from bras_bench import load_reference  # noqa: E402

CENTERS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
EDGES_MS = [0.0, 5.0, 20.0, 50.0, 80.0, 150.0, 300.0, 1e9]
LABELS = ["0-5", "5-20", "20-50", "50-80", "80-150", "150-300", "300+"]


def windows_db(ir, fs):
    """Energy per window and octave (dB rel. the 0-5 ms window of the same octave)."""
    k0 = rm._onset(ir, fs)
    k0 = max(0, k0 - int(0.001 * fs))          # 1 ms before the onset: the band filters ring
    bands = rm.octave_split(ir[k0:], fs, CENTERS)
    out = np.zeros((len(CENTERS), len(LABELS)))
    for b, x in enumerate(bands):
        e = x ** 2
        for w in range(len(LABELS)):
            i0 = int(EDGES_MS[w] * 1e-3 * fs)
            i1 = min(len(e), int(min(EDGES_MS[w + 1], 1e6) * 1e-3 * fs))
            out[b, w] = e[i0:i1].sum()
    return 10.0 * np.log10(np.maximum(out, 1e-30) / np.maximum(out[:, :1], 1e-30))


def main():
    man_path, run_dir = sys.argv[1], sys.argv[2]
    base = os.path.dirname(os.path.abspath(man_path))
    man = json.load(open(man_path))
    diffs = []
    for p in man["pairs"]:
        sim_path = os.path.join(run_dir, "%s_sim.wav" % p["label"])
        if not os.path.exists(sim_path) or not p.get("reference_wav"):
            continue
        data, fs = read_wav(sim_path)
        sim = data.mean(axis=1) if data.ndim == 2 else data
        ref, _ = load_reference(os.path.join(base, p["reference_wav"]), fs)
        diffs.append(windows_db(sim, fs) - windows_db(ref, fs))
    d = np.mean(diffs, axis=0)
    print("%s: %d pairs; simulated minus measured, dB, energy per window relative to the "
          "0-5 ms window (direct sound)" % (os.path.basename(man_path), len(diffs)))
    print("octave " + "".join("%9s" % lab for lab in LABELS[1:]))
    for b, c in enumerate(CENTERS):
        print("%6s " % ("%g" % c if c < 1000 else "%gk" % (c / 1000)) + "".join("%+9.1f" % v for v in d[b, 1:]))


if __name__ == "__main__":
    main()
