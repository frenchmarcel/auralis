#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Engine test (slow, ~2 min): open domain (CPML) vs analytic image
solution. No external data required."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auralis.engine.fdtd import run_low_band  # noqa: E402

C = 343.0


def scene(with_plate):
    tv = [[[-1.5, -1.5, 0], [1.5, -1.5, 0], [1.5, 1.5, 0]],
          [[-1.5, -1.5, 0], [1.5, 1.5, 0], [-1.5, 1.5, 0]]]
    if not with_plate:
        tv = [[[-1.5, -1.5, -0.4], [-1.49, -1.5, -0.4],
               [-1.5, -1.49, -0.4]]]
    s = types.SimpleNamespace()
    s.tri_verts = np.asarray(tv, float)
    s.tri_mat = np.zeros(len(tv), np.int32)
    s.mat_names = ["plate"]
    s.absorption = np.zeros((1, 7), np.float32)
    s.scattering = np.zeros((1, 7), np.float32)
    s.transmission = np.zeros((1, 7), np.float32)
    s.mat_phys = None
    return s


def peak_t(x, i, fs):
    a, b, c = x[i - 1], x[i], x[i + 1]
    return (i + 0.5 * (a - c) / (a - 2 * b + c + 1e-30)) / fs


def test_open_domain():
    fs = 44100
    src = np.array([0.0, 0.0, 0.60])
    mic = np.array([1.20, 0.0, 0.40])
    irs = {}
    for lbl, wp in (("libre", False), ("plaque", True)):
        ir, fi = run_low_band(scene(wp), src, [mic], fs=fs,
                              duration=0.028, f_cross=1000.0,
                              speed_of_sound=C, engine="auto",
                              order=4, conformal=True,
                              max_cells=4_000_000)
        irs[lbl] = np.asarray(ir, float).ravel()
        print("[%s] %s: %d cells" % (fi.engine, lbl, fi.n_air))
    r1g = np.linalg.norm(mic - src)
    r2g = np.linalg.norm(mic - src * np.array([1, 1, -1]))
    h = np.abs(irs["libre"])
    t1 = peak_t(h, int(np.argmax(h)), fs)
    e_libre = abs(t1 * C - r1g)
    print("free field: r1_eff %.4f m (geo %.4f, err %.1f mm)"
          % (t1 * C, r1g, 1e3 * e_libre))
    d = irs["plaque"] - irs["libre"]
    h2 = np.abs(d)
    t2 = peak_t(h2, int(np.argmax(h2)), fs)
    off = (t2 * C - r2g)
    print("reflection: r2_eff %.4f m (geo %.4f, plane %.1f mm below "
          "the surface)" % (t2 * C, r2g, 1e3 * off / 2))
    # |R_eff| over a time window on the isolated image
    g = np.zeros_like(d)
    a, b = int(0.0040 * fs), int(0.0059 * fs)
    g[a:b] = np.hanning(b - a) ** 0.3
    n = len(d)
    Hd = np.fft.rfft(irs["libre"], 8 * n)
    Hr = np.fft.rfft(d * g, 8 * n)
    fr = np.fft.rfftfreq(8 * n, 1.0 / fs)
    sel = (fr > 200) & (fr < 1000)
    Reff = (np.abs(Hr[sel]) / np.abs(Hd[sel])) * ((t2 * C) / (t1 * C))
    ph = np.degrees(np.angle(Hr[sel] * np.conj(Hd[sel]) * np.exp(
        1j * 2 * np.pi * fr[sel] * (t2 - t1))))
    print("|R_eff| 200-1000 Hz: mean %.3f (min %.3f, max %.3f); "
          "mean phase %+.1f deg" % (Reff.mean(), Reff.min(),
                                    Reff.max(), ph.mean()))
    # effective plane expected ~0.1-0.35 dx; dx at fc=1000 = 2.54 cm
    ok = (e_libre < 1e-3 and 0.85 < Reff.mean() < 1.05
          and abs(ph.mean()) < 6.0 and 0.001 < off / 2 < 0.009)
    assert ok, "open domain out of thresholds"
    print("test_open_domain: ok")


if __name__ == "__main__":
    test_open_domain()
