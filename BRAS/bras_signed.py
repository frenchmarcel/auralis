"""Signed mean deviation (simulated - measured) per octave over the pairs, for several runs:
python BRAS/bras_signed.py report.csv [report.csv ...]"""
import csv
import sys

import numpy as np

BANDS = ["125", "250", "500", "1k", "2k", "4k", "8k"]


def signed(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for key, rel in (("T30", True), ("EDT", True), ("C50", False), ("Ts", False)):
        vals = []
        for b in BANDS:
            d = []
            for r in rows:
                if r["bande"] != b or not r[key + "_sim"] or not r[key + "_ref"]:
                    continue
                s, m = float(r[key + "_sim"]), float(r[key + "_ref"])
                d.append(100.0 * (s / m - 1.0) if rel else s - m)
            vals.append(np.mean(d) if d else np.nan)
        out[key] = vals
    return out


def main():
    print("signed mean deviation, simulated - measured: T30 / EDT in %, C50 in dB, Ts in ms")
    print("%-34s %-4s" % ("run", "") + "".join("%7s" % b for b in ("125", "250", "500", "1k", "2k", "4k", "8k")))
    for p in sys.argv[1:]:
        s = signed(p)
        for k in ("T30", "EDT", "C50", "Ts"):
            print("%-34s %-4s" % (p.replace("/report.csv", "")[-34:], k) + "".join("%+7.1f" % v for v in s[k]))


if __name__ == "__main__":
    main()
