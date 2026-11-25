"""ITA dodecahedron: radiated power from the BRAS unit spectra against the bench's table, and
the source spectrum seen in the rooms (direct sound and whole response, simulated minus
measured, re 1 kHz): python BRAS/bras_dodeca_check.py [run_dir_name]"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from auralis.engine import rirmetrics as rm  # noqa: E402
from auralis.engine.wavio import read_wav  # noqa: E402
from bras_bench import load_reference  # noqa: E402

DODE = os.path.join(os.path.dirname(HERE), "data_bras", "2 Source and receiver descriptions",
                    "ITA dodecahedron")
TABLE = os.path.join(HERE, "CR2", "ita_dodeca_directivity.json")
F3 = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
               1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
               20000])
CENTERS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
OCT = [[i for i in range(31) if c / 2 ** 0.5 <= F3[i] < c * 2 ** 0.5] for c in CENTERS]
ROOMS = [("CR2", "CR2/cr2_dodecahedron_fitted31_dir.json"),
         ("CR3", "CR3/cr3_dodecahedron_fitted31_dir.json"),
         ("CR4", "CR4/cr4_dodecahedron_fitted31_dir.json")]
CX = re.compile(r"([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*([+-])\s*([\d.]+(?:e[+-]?\d+)?)i")


def _complex(text):
    return [complex(float(a.replace(" ", "")), float(s + b)) for a, s, b in CX.findall(text)]


def unit_spectra():
    """(theta (N,), Mid (N, 31), High (N, 31), Low (31,)) complex third-octave spectra."""
    out = {}
    for name in ("Mid", "High"):
        th, z = [], []
        with open(os.path.join(DODE, "Dode_%s_1x1_64442_mps_front_pole.csv" % name)) as fh:
            next(fh)
            for line in fh:
                m = re.match(r"\s*P(\d+)T(\d+)", line)
                v = _complex(line[m.end():]) if m else []
                if len(v) == 31:
                    th.append(float(m.group(2)))
                    z.append(v)
        out[name] = (np.array(th), np.array(z))
    low = np.array(_complex(open(os.path.join(DODE, "Dode_Low_mps_omnidirectional.csv"))
                            .readlines()[1]))
    assert np.array_equal(out["Mid"][0], out["High"][0]) and len(low) == 31
    return out["Mid"][0], out["Mid"][1], out["High"][1], low


def rel_db(p):
    x = 10.0 * np.log10(p)
    return x - x[3]


def source_power():
    th, mid, high, low = unit_spectra()
    w = np.maximum(np.sin(np.radians(th)), 1e-6)[:, None]
    sph = lambda p: (w * p).sum(0) / w.sum()   # noqa: E731
    inc = np.abs(low) ** 2 + sph(np.abs(mid) ** 2) + sph(np.abs(high) ** 2)
    coh = sph(np.abs(low[None, :] + mid + high) ** 2)
    tab = np.asarray(json.load(open(TABLE))["table"])
    t = np.linspace(0.0, np.pi, tab.shape[1])
    wt = np.sin(t) / np.sin(t).sum()
    octs = lambda p: np.array([p[i].sum() for i in OCT])   # noqa: E731
    return {"table": rel_db((tab * wt).sum(1)), "incoherent": rel_db(octs(inc)),
            "coherent": rel_db(octs(coh))}


def room_spectra(run):
    """Per room: (direct 0-2.5 ms, whole response) simulated minus measured, dB re 1 kHz,
    mean and spread over the pairs."""
    out = {}
    for room, man in ROOMS:
        man = os.path.join(HERE, man)
        m, base = json.load(open(man)), os.path.dirname(man)
        dd, tt = [], []
        for p in m["pairs"]:
            x, fs = read_wav(os.path.join(base, run, p["label"] + "_sim.wav"))
            x = x.mean(axis=1) if x.ndim == 2 else x
            r, _ = load_reference(os.path.join(base, p["reference_wav"]), fs)
            e = []
            for ir in (x, r):
                k0 = max(0, rm._onset(ir, fs) - int(0.001 * fs))
                b = rm.octave_split(ir[k0:], fs, CENTERS) ** 2
                e.append((b[:, :int(0.0025 * fs)].sum(1), b.sum(1)))
            dd.append(rel_db(e[0][0] / e[1][0]))
            tt.append(rel_db(e[0][1] / e[1][1]))
        out[room] = (np.mean(dd, 0), np.std(dd, 0), np.mean(tt, 0), np.std(tt, 0))
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "reference_run"
    print("octave power of the dodecahedron, dB re 1 kHz:  125   250   500    1k    2k    4k    8k")
    for k, v in source_power().items():
        print("  %-44s" % ("BRAS unit spectra, " + k if k != "table" else "bench table (ita_dodeca_directivity.json)")
              + "".join("%6.1f" % x for x in v))
    print("\nsource as seen in the rooms (simulated - measured, dB re 1 kHz, %s):" % run)
    for room, (d, ds, t, ts) in room_spectra(run).items():
        print("  %s direct 0-2.5 ms  " % room + "".join("%+6.1f" % x for x in d) + "   (2k spread %.1f)" % ds[4])
        print("  %s whole response   " % room + "".join("%+6.1f" % x for x in t) + "   (2k spread %.1f)" % ts[4])


if __name__ == "__main__":
    main()
