#!/usr/bin/env python3
"""JND summary of a room bench report (report.csv of bras_bench): mean |d|/JND over T30
(5 %), C50 (1 dB), Ts (10 ms) and level (1 dB), and the share within 1 JND, per metric
and per octave. Usage: bras_jnd.py report.csv [report2.csv ...]"""
import csv
import sys

import numpy as np

JND = (("dT30", 0.05, "T30_ref"), ("dC50", 1.0, None), ("dTs", 10.0, None), ("dL", 1.0, None))


def summary(path):
    rows = list(csv.DictReader(open(path)))
    cells = {}
    for r in rows:
        for key, jnd, rel in JND:
            if r[key] in ("", None):
                continue
            x = abs(float(r[key])) / (jnd * float(r[rel])) if rel else abs(float(r[key])) / jnd
            cells.setdefault((key, r["bande"]), []).append(x)
    bands = list(dict.fromkeys(r["bande"] for r in rows))
    allv = np.concatenate([np.asarray(v) for v in cells.values()])
    print("%s: %.2f JND mean, %.0f %% <= 1 JND (n=%d)" % (path, allv.mean(), 100 * (allv <= 1).mean(), len(allv)))
    print("          " + "".join("%7s" % b for b in bands) + "    all")
    for key, _j, _r in JND:
        row = [np.mean(cells.get((key, b), [np.nan])) for b in bands]
        v = np.concatenate([np.asarray(cells[(key, b)]) for b in bands if (key, b) in cells])
        print("  %-6s  " % key + "".join("%7.2f" % x for x in row) + "   %.2f" % v.mean())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summary(p)
