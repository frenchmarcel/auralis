#!/usr/bin/env python3
"""CR2 binaural low end against the measured BRIRs (FABIAN, QSC-K8, LS3-LS7 -> MP6): IACC in
the first 80 ms and in the reverberation, and the level difference between the ears, per
octave 63-500 Hz, for the wave band's heads ("ears": two probes; "hrtf" / "hrtf2": first- /
second-order ambisonics decoded through the HRTF set).
Usage: binaural_lowband.py out_dir [--heads ears hrtf hrtf2]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import listening_test as LT  # noqa: E402
from auralis.engine import rirmetrics as rm  # noqa: E402
from auralis.engine.directivity import Directivity  # noqa: E402
from auralis.engine.pipeline import (ReceiverDef, SourceDef, bake_pair,  # noqa: E402
                                     source_low_bands)
from auralis.engine.project import build_config, build_scene  # noqa: E402

OCT = [63.0, 125.0, 250.0, 500.0]
EARLY_S, LATE_S = 0.08, 0.6


def simulate(out_dir, emitters, head, mode):
    """{LS: (n, 2)} with the published CR2 configuration, FABIAN, the QSC-K8 aimed as
    measured (listening_test.simulate), the wave band's head `mode`; one npz per source."""
    man = json.load(open(LT.MANIFEST))
    scene = build_scene(man, HERE)
    cfg = build_config(man)
    cfg.hrtf_path = LT.HRTF
    cfg.wave_head = mode
    dirv = Directivity(np.asarray(LT.qsc()["table"]), name="QSC_K8")
    rcv = ReceiverDef("MP6", position=head, forward=np.array([0.0, -1.0, 0.0]))
    out = {}
    for ls, pos in emitters.items():
        path = os.path.join(out_dir, "sim_%s_%s.npz" % (ls, mode))
        if not os.path.exists(path):
            t0 = time.time()
            fwd = (np.array([0.0, 1.0, 0.0]) if ls == "LS7"
                   else (head - pos) / np.linalg.norm(head - pos))
            src = SourceDef(ls, pos, directivity=dirv, forward=fwd)
            low, err = source_low_bands(scene, src, [rcv], cfg)
            if err:
                raise SystemExit("wave band of %s failed: %s" % (ls, err))
            res = bake_pair(scene, src, rcv, cfg, low)
            np.savez(path, ir=res.ir)
            print("  %s %s simulated in %.0f s" % (ls, mode, time.time() - t0), flush=True)
        out[ls] = np.asarray(np.load(path)["ir"], dtype=np.float64)
    return out


def iacc(x, fs):
    """Max over |tau| <= 1 ms of the normalised cross-correlation of the two ears."""
    l, r = x[:, 0], x[:, 1]
    n = len(l)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(l, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
    m = int(round(0.001 * fs))
    lags = np.concatenate([cc[:m + 1], cc[-m:]])
    den = np.sqrt(np.sum(l * l) * np.sum(r * r))
    return float(np.max(np.abs(lags)) / den) if den > 0 else float("nan")


def measures(ir, fs):
    """{octave: (IACC early, IACC late, level left - right in dB over 0-80 ms)}."""
    k = LT.onset(ir, fs)
    out = {}
    for fc in OCT:
        b = np.stack([rm.octave_split(ir[:, e], fs, [fc])[0] for e in range(2)], axis=1)
        a, c = k, k + int(EARLY_S * fs)
        early, late = b[a:c], b[c:k + int(LATE_S * fs)]
        ild = 10 * np.log10(np.sum(early[:, 0] ** 2) / np.sum(early[:, 1] ** 2))
        out[fc] = (iacc(early, fs), iacc(late, fs), ild)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--heads", nargs="+", default=["ears", "hrtf", "hrtf2"])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    meas, emitters, head, fs = LT.measured()
    rows = {"measured": {ls: measures(meas[ls], fs) for ls in sorted(meas)}}
    for mode in a.heads:
        sims = simulate(a.out_dir, emitters, head, mode)
        rows[mode] = {ls: measures(sims[ls], fs) for ls in sorted(sims)}
    lines = []
    for q, name in ((0, "IACC 0-80 ms"), (1, "IACC 80-600 ms"), (2, "level left - right, 0-80 ms (dB)")):
        lines.append("%s, per octave %s Hz:" % (name, "/".join("%d" % f for f in OCT)))
        for ls in sorted(meas):
            lines.append("  %s  " % ls + "   ".join(
                "%s %s" % (k, " ".join("%5.2f" % rows[k][ls][fc][q] for fc in OCT))
                for k in rows))
        mean = {k: [np.mean([rows[k][ls][fc][q] for ls in meas]) for fc in OCT] for k in rows}
        lines.append("  mean " + "   ".join("%s %s" % (k, " ".join("%5.2f" % v for v in mean[k]))
                                            for k in rows))
        if q == 2:
            for k in a.heads:
                err = np.mean([[abs(rows[k][ls][fc][2] - rows["measured"][ls][fc][2]) for fc in OCT]
                               for ls in meas], axis=0)
                lines.append("  |error| %s %s" % (k, " ".join("%5.2f" % v for v in err)))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out_dir, "report.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
