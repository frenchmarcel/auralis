#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CR2 -- run the room bench (../bras_bench.py) on a manifest, then
plot the validation figure (out/validation.png) from the report.
Usage: cr2_run.py <manifest.json> [-o out_dir]"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "bras_bench.py")
PLOT = os.path.join(HERE, "plot.py")


def _out_dir(argv):
    """Mirror the output-directory resolution of bras_bench: -o/--out
    if given, else <manifest_dir>/<name>_report."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("-o", "--out", default=None)
    try:
        known, _ = ap.parse_known_args(argv)
    except SystemExit:
        return None
    if known.out:
        return known.out
    if not known.manifest:
        return None
    try:
        with open(known.manifest, encoding="utf-8") as fh:
            name = json.load(fh).get("name", "bras")
    except Exception:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(known.manifest)),
                        name + "_report")


def main():
    argv = sys.argv[1:]
    rc = subprocess.call([sys.executable, BENCH] + argv)
    if rc != 0:
        sys.exit(rc)
    # plot the figure from the report written by bras_bench
    out = _out_dir(argv)
    report = os.path.join(out, "report.txt") if out else None
    if not report or not os.path.exists(report):
        print("cr2_run: report.txt not found (%s) -- figure skipped"
              % report)
        sys.exit(0)
    png = os.path.join(out, "validation.png")
    prc = subprocess.call([sys.executable, PLOT, report, "-o", png])
    if prc != 0:
        print("cr2_run: figure not generated (plot.py code %d) -- the "
              "report stays in %s" % (prc, out))
    sys.exit(0)


if __name__ == "__main__":
    main()
