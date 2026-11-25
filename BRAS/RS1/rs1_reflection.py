#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RS1 -- Reflection off an (infinite) plane: bare floor / Rockfon
20 mm / diffuser; delegates to the bras_simple harness (Pearson 160-2000 Hz).
Usage: rs1_reflection.py <scene 01 dir> [bras_simple options]
Defaults: --pairs LS1-MP1 LS2-MP1 LS3-MP3
          --conditions Rigid Absorbing --fc 2000"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
import bras_simple  # noqa: E402


def main():
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    if "--pairs" not in argv:
        argv += ["--pairs", "LS1-MP1", "LS2-MP1", "LS3-MP3"]
    if "--conditions" not in argv:
        argv += ["--conditions", "Rigid", "Absorbing"]
    if "--fc" not in argv:
        argv += ["--fc", "2000"]
    if "--out" not in argv:
        argv += ["--out", "out_rs1"]
    sys.argv = ["bras_simple.py"] + argv
    bras_simple.main()


if __name__ == "__main__":
    main()
