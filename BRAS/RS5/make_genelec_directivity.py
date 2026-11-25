#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the measured Genelec 8020c directivity (BRAS mps CSV) into a
`Directivity` JSON table: Q_b(theta) per octave (azimuth mean), Q_b(0)=1.

Usage: make_genelec_directivity.py <Genelec...mps...csv> [output.json]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bras_directivity import OCT, octave_table, read_mps  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    csv = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "genelec8020_directivity.json"
    freqs, E = read_mps(csv)
    tbl = octave_table(freqs, E)
    nT = tbl.shape[1]

    d = {"name": "Genelec_8020c",
         "note": "BRAS measured directivity (mps, azimuth mean, energy "
                 "per octave, on-axis normalised Q_b(0)=1); on-axis "
                 "response flat within +/-1 dB, hence not reapplied",
         "n_theta": nT, "table": tbl.tolist()}
    json.dump(d, open(out, "w"))
    # diagnostic: off-axis roll-off per octave
    print("wrote %s  (table %dx%d)" % (out, len(OCT), nT))
    print("Off-axis roll-off (dB rel. on-axis), theta in degrees:")
    print("  theta:   " + "".join("%8d" % t for t in (0, 15, 30, 45, 90)))
    for b, fc in enumerate(OCT):
        print("  %5.0f Hz:" % fc + "".join(
            "%8.1f" % (10 * np.log10(max(tbl[b, t], 1e-9)))
            for t in (0, 15, 30, 45, 90)))


if __name__ == "__main__":
    main()
