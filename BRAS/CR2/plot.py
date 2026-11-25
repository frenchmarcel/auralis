#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation figures (Treble style) from the console output of a BRAS
bench: one panel per metric, octaves on the x axis, measurement +/-1 JND
against simulation. Usage: plot.py report.txt -o out/validation.png"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
import bras_style                 # noqa: E402  (shared visual identity)

BLABS = ["125", "250", "500", "1k", "2k", "4k", "8k"]
TEAL, INK = bras_style.TEAL, bras_style.INK
METRICS = [
    ("T30", "T30 (s)", ("rel", 0.05)),
    ("EDT", "EDT (s)", ("rel", 0.05)),
    ("C50", "C50 (dB)", ("abs", 1.0)),
    ("C80", "C80 (dB)", ("abs", 1.0)),
    ("D50", "D50", ("abs", 0.05)),
    ("D80", "D80", ("abs", 0.05)),
    ("Ts", "Ts (ms)", ("abs", 10.0)),
    ("Lvl", "SPL/band (dB, rel.)", ("abs", 1.0)),
]


def _num(tok):
    try:
        return float(tok)
    except ValueError:
        return np.nan       # "--" (band not resolved)


def parse(text):
    pairs, cur = {}, None
    for line in text.splitlines():
        s = line.strip()
        # pair header: [LS1-MP1] or [LS1-MP1 — fitted] ...
        m = re.match(r"\[([^\]]+?)\]\s*$", s)
        if m:
            cur = m.group(1).split("—")[0].split("  ")[0].strip()
            pairs.setdefault(cur, {})
            continue
        m = re.match(r"(T20|T30|EDT|C50|C80|D50|D80|Ts|Lvl)\s+"
                     r"(sim|ref)\s*:\s*(.+)", s)
        if m and cur is not None:
            vals = [_num(t) for t in m.group(3).split()]
            if len(vals) == 7:
                pairs[cur][(m.group(1), m.group(2))] = np.array(vals)
    return {k: v for k, v in pairs.items() if v}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("-o", "--out", default="validation.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--metrics", default=None, metavar="LIST",
                    help="subset of metrics, comma-separated and IN THE "
                         "WANTED ORDER (e.g. EDT,Ts,C50,D80). Default: "
                         "every metric present in the report.")
    a = ap.parse_args(argv)
    metrics = METRICS
    if a.metrics:
        want = [m.strip() for m in a.metrics.split(",") if m.strip()]
        table = {k: (k, lab, jnd) for k, lab, jnd in METRICS}
        inconnues = [w for w in want if w not in table]
        if inconnues:
            sys.exit("unknown metric(s): %s - available: %s"
                     % (", ".join(inconnues),
                        ", ".join(k for k, _, _ in METRICS)))
        metrics = [table[w] for w in want]
    pairs = parse(open(a.report, encoding="utf-8",
                       errors="replace").read())
    if not pairs:
        sys.exit("no pair found in %s (does the report contain the "
                 "[pair] / T30 sim : ... blocks?)"
                 % a.report)
    x = np.arange(7)
    plt.rcParams.update(bras_style.rc())
    present = []
    for key, label, jnd in metrics:
        sims = np.array([p[(key, "sim")] for p in pairs.values()
                         if (key, "sim") in p])
        refs = np.array([p[(key, "ref")] for p in pairs.values()
                         if (key, "ref") in p])
        if len(sims) and len(refs):
            present.append((key, label, jnd, sims, refs))
    if not present:
        sys.exit("pairs found but no 'ref' column: the report has no "
                 "measurements (pairs without reference_wav?)")
    # Grid: one panel per metric. The legend takes an EXTRA slot only
    # when the natural grid leaves one free; otherwise it sits inside
    # the first panel (the case of a 2x2 selection).
    nm = len(present)
    if nm == 4:
        ncols, nrows = 2, 2
    else:
        n = nm + 1
        ncols = 4 if n > 6 else 3
        nrows = -(-n // ncols)
    n_slots = ncols * nrows
    legend_in_axes = (n_slots == nm)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax in axes[nm if legend_in_axes else nm + 1:]:
        ax.set_visible(False)
    for ax, (key, label, (kind, w), sims, refs) in zip(axes, present):
        rm = np.nanmean(refs, 0)
        sm = np.nanmean(sims, 0)
        lo, hi = ((rm * (1 - w), rm * (1 + w)) if kind == "rel"
                  else (rm - w, rm + w))
        ax.fill_between(x, lo, hi, color=bras_style.BAND, alpha=0.45, lw=0,
                        label="measured +/- 1 JND")
        for r in refs:
            ax.plot(x, r, color=bras_style.FAINT, lw=0.7, alpha=0.5)
        for sglob in sims:
            ax.plot(x, sglob, color=TEAL, lw=0.7, alpha=0.4)
        ax.plot(x, rm, color=INK, lw=2.3, marker="o", ms=5,
                label="measured (mean)")
        ax.plot(x, sm, color=TEAL, lw=2.3, marker="s", ms=5,
                label="Auralis (mean)")
        ax.set_title("%s   mean |delta| = %.2f"
                     % (label, np.nanmean(np.abs(sm - rm))),
                     fontsize=10, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(BLABS)
        ax.set_xlabel("octave (Hz)")
        ax.margins(x=0.03)
    h, l = axes[0].get_legend_handles_labels()
    if legend_in_axes:
        axes[0].legend(h, l, loc="best", frameon=True, framealpha=0.85,
                       fontsize=8)
        fig.text(0.995, 0.005, "%d source-receiver pairs" % len(pairs),
                 ha="right", va="bottom", fontsize=9, color="#666")
    else:
        axes[nm].axis("off")
        axes[nm].legend(h, l, loc="center", frameon=False, fontsize=10)
        axes[nm].text(0.5, 0.15, "%d source-receiver pairs" % len(pairs),
                      ha="center", fontsize=9, color="#666")
    if a.title:
        fig.suptitle(a.title, fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if a.title else 1))
    fig.savefig(a.out, dpi=140)
    print("figure ->", a.out)


if __name__ == "__main__":
    main()