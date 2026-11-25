#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRAS supporting evidence turned into unit tests. Each test skips
when its environment variable (BRAS_ROCKFON_CSV, BRAS_GENELEC_MPS,
BRAS_SCENE01_DIR, BRAS_DODE_DIR, BRAS_CR2_WAV) is not set."""
import csv
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auralis.engine import poromat            # noqa: E402
from auralis.engine import rirmetrics as rm   # noqa: E402

C = 343.6
FS = 44100


def _skip(name, var):
    print("%s: skipped (set %s)" % (name, var))
    return True


def _read_wav(p):
    raw = open(p, "rb").read()
    i, data, fs = 12, None, 44100
    while i < len(raw) - 8:
        cid = raw[i:i + 4]
        sz = struct.unpack("<I", raw[i + 4:i + 8])[0]
        if cid == b"fmt ":
            _, _, fs = struct.unpack("<HHI", raw[i + 8:i + 16])
        elif cid == b"data":
            data = raw[i + 8:i + 8 + sz]
        i += 8 + sz + (sz & 1)
    return np.frombuffer(data, dtype="<f4").astype(float), fs


# ------------------------------------------------------------- RS1
def test_rockfon_fit():
    """Miki fit with rigid backing d=20 mm on the RockFon alphas at
    0/30/45/60 deg: sigma ~127 k, R phase small and negative."""
    d = os.environ.get("BRAS_ROCKFON_CSV")
    if not d:
        return _skip("test_rockfon_fit", "BRAS_ROCKFON_CSV")

    def load(p):
        rows = list(csv.reader(open(p, encoding="utf-8-sig"),
                               delimiter=';'))
        if len(rows[0]) < 3:
            rows = list(csv.reader(open(p, encoding="utf-8-sig")))
        return (np.array([float(x) for x in rows[0]
                          if str(x).strip()]),
                np.array([float(x) for x in rows[1]
                          if str(x).strip()]))

    angs = [0.0, 30.0, 45.0, 60.0]
    A, fb = [], None
    for th in angs:
        fb, a = load(os.path.join(
            d, "mat_RockFonSonarG_%02ddeg.csv" % int(th)))
        A.append(a)
    A = np.asarray(A)
    m = (fb >= 200) & (fb <= 4000)
    f, Am = fb[m], A[:, m]
    best = (1e9, None)
    for sig in np.geomspace(3e3, 3e5, 120):
        spec = poromat.normalize_spec(
            {"layers": [{"model": "miki", "sigma": float(sig),
                         "d": 0.020}]})
        g = poromat.gamma_normal(spec, f)
        err = 0.0
        for k, th in enumerate(angs):
            c = np.cos(np.radians(th))
            R = (c - g) / (c + g)
            err += float(np.mean((1 - np.abs(R) ** 2 - Am[k]) ** 2))
        if err < best[0]:
            best = (err, sig)
    err, sig = best
    spec = poromat.normalize_spec(
        {"layers": [{"model": "miki", "sigma": float(sig),
                     "d": 0.020}]})
    g = poromat.gamma_normal(spec, f)
    ph = np.degrees(np.angle((1 - g) / (1 + g)))
    print("test_rockfon_fit: sigma %.0f, rms %.3f, phase "
          "[%.0f..%.0f] deg" % (sig, np.sqrt(err / 4),
                                ph.min(), ph.max()))
    assert 80e3 < sig < 200e3 and np.sqrt(err / 4) < 0.13
    assert (ph < 0).all() and (ph > -60).all()


def test_genelec_phase():
    """The inter-angle phase shift of the Genelec 8020c (direct vs
    reflection) is tiny: the source does not explain the comb shift."""
    path = os.environ.get("BRAS_GENELEC_MPS")
    if not path:
        return _skip("test_genelec_phase", "BRAS_GENELEC_MPS")
    aim = np.array([np.cos(np.radians(-45)), 0,
                    np.sin(np.radians(-45))])
    LS2 = np.array([3.379, 2.985, 2.121])
    MP1 = np.array([8.098, 2.985, 1.500])
    sp = np.array([5.5, 2.985, 0.0])

    def ang(u):
        u = u / np.linalg.norm(u)
        th = np.degrees(np.arccos(np.clip(float(np.dot(u, aim)),
                                          -1, 1)))
        up = np.cross(np.cross(aim, np.array([0, 1, 0])), aim)
        up /= np.linalg.norm(up)
        return (0 if float(np.dot(u, up)) >= 0 else 180), round(th)

    pd, td = ang(MP1 - LS2)
    pr, tr = ang(sp - LS2)
    want = {"P%03dT%03d" % (pd, td): "d",
            "P%03dT%03d" % (pr, tr): "r"}
    rows = {}
    with open(path) as fh:
        F = np.array([float(x) for x in next(fh).split(",")[1:]
                      if x.strip()])
        for line in fh:
            lab = line.split(",", 1)[0].strip()
            if lab in want:
                toks = [t.strip() for t in
                        line.strip().rstrip(",").split(",")[1:]]
                rows[want[lab]] = np.array(
                    [complex(t.replace(" ", "").replace("i", "j"))
                     for t in toks])
                if len(rows) == 2:
                    break
    Hd, Hr = rows["d"], rows["r"]
    sel = ((F >= 100) & (F <= 1000) & np.isfinite(Hd)
           & np.isfinite(Hr))
    dphi = np.degrees(np.unwrap(np.angle(Hd[sel]
                                         * np.conj(Hr[sel]))))
    tau = dphi / 360.0 / F[sel] * 1e3
    print("test_genelec_phase: |dphi| max %.1f deg, |tau| max "
          "%.3f ms (~65 deg / 0.5 ms would have been needed)"
          % (np.abs(dphi).max(), np.abs(tau).max()))
    assert np.abs(dphi).max() <= 35.0
    assert np.abs(tau).max() <= 0.12


def test_scene01_erratum():
    """SOFA scene 01 erratum (ReceiverPosition z+0.240 m): arrival
    deltas match the diagram (<=3 %), not the SOFA heights."""
    scene = os.environ.get("BRAS_SCENE01_DIR")
    if not scene:
        return _skip("test_scene01_erratum", "BRAS_SCENE01_DIR")
    LS = {1: np.array([2.902, 2.985, 1.500]),
          2: np.array([3.379, 2.985, 2.121]),
          3: np.array([4.000, 2.985, 2.598])}
    MP_X = {1: 8.098, 3: 7.000}
    Z_SOFA = {1: 1.740, 3: 2.838}
    Z_VRAI = {1: 1.500, 3: 2.598}
    PAIRS = [(1, 1), (2, 1), (3, 1), (1, 3)]
    wavdir = None
    for root, _, files in os.walk(scene):
        if any(f.startswith("scene1_RIR_Rigid") and
               f.endswith(".wav") for f in files):
            wavdir = root
    assert wavdir, "Rigid wav files not found under %s" % scene

    def delta_measured(x):
        h = np.abs(x)
        ip = int(np.argmax(h))
        thr = 0.20 * h[ip]
        k = int(np.nonzero(h > thr)[0][0])
        k += int(np.argmax(h[k:k + 40]))
        if abs(k - ip) < 20:
            j0 = ip + 60
            j = j0 + int(np.argmax(h[j0:j0 + int(0.006 * FS)]))
            return (j - ip) / FS * C
        return (ip - k) / FS * C

    def delta_geo(ls, mp, z_mp):
        s = LS[ls]
        m = np.array([MP_X[mp], 2.985, z_mp])
        r1 = np.linalg.norm(m - s)
        return np.linalg.norm(m - s * np.array([1, 1, -1])) - r1

    ev, es = [], []
    for ls, mp in PAIRS:
        x, _ = _read_wav(os.path.join(
            wavdir, "scene1_RIR_Rigid_LS%d_MP%d.wav" % (ls, mp)))
        dm = delta_measured(x)
        ev.append(abs(dm - delta_geo(ls, mp, Z_VRAI[mp]))
                  / delta_geo(ls, mp, Z_VRAI[mp]))
        es.append(abs(dm - delta_geo(ls, mp, Z_SOFA[mp]))
                  / delta_geo(ls, mp, Z_SOFA[mp]))
    mv, ms = float(np.mean(ev)), float(np.mean(es))
    print("test_scene01_erratum: diagram %.1f %% / SOFA %.1f %%"
          % (100 * mv, 100 * ms))
    assert mv < 0.03 and ms > 2.5 * mv


# ------------------------------------------------------------- CR2
def test_dodeca_source_lvl():
    """The dodecahedron crossovers explain the CR2 dLvl: 3-gain
    model, RMS <=1.2 dB."""
    d = os.environ.get("BRAS_DODE_DIR")
    if not d:
        return _skip("test_dodeca_source_lvl", "BRAS_DODE_DIR")
    F = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
                  250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                  2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
                  12500, 16000, 20000], float)
    EDGES = [(88, 177), (177, 354), (354, 707), (707, 1414),
             (1414, 2828), (2828, 5657), (5657, 11314)]
    DLVL = np.array([3.67, 0.82, -0.81, -1.19, 4.76, -0.52, -0.21])
    CX = re.compile(r"^([+-]?[\d.]+)\s*([+-])\s*([\d.]+)i$")

    def parse_line(line):
        toks = [t.strip() for t in line.strip().rstrip(",").split(",")]
        m = re.match(r"P(\d+)T(\d+)", toks[0])
        th = float(m.group(2)) if m else None
        vals = []
        for tok in toks[1 if m else 0:]:
            if not tok:
                continue
            mm = CX.match(tok)
            if mm:
                a = float(mm.group(1))
                b = float(mm.group(2) + mm.group(3))
                vals.append(a * a + b * b)
            else:
                v = float(tok.rstrip("i"))
                vals.append(v * v)
        return th, np.array(vals)

    def sphere_avg(path):
        acc, ws = np.zeros(31), 0.0
        with open(path) as fh:
            next(fh)
            for line in fh:
                th, p = parse_line(line)
                if len(p) != 31:
                    continue
                w = max(np.sin(np.radians(th)), 1e-6) \
                    if th is not None else 1.0
                acc += w * p
                ws += w
        return acc / ws

    def octp(p):
        return np.array([p[(F >= lo) & (F < hi)].mean()
                         for lo, hi in EDGES])

    _, low = parse_line(open(os.path.join(
        d, "Dode_Low_mps_omnidirectional.csv")).readlines()[1])
    mid = sphere_avg(os.path.join(
        d, "Dode_Mid_1x1_64442_mps_front_pole.csv"))
    high = sphere_avg(os.path.join(
        d, "Dode_High_1x1_64442_mps_front_pole.csv"))
    L, M, H = octp(low), octp(mid), octp(high)
    m_re = 10 * np.log10(M / M[2:4].mean())

    def cost(g):
        s = g[0] * L + g[1] * M + g[2] * H
        if (s <= 0).any():
            return 1e9, None
        pred = -10 * np.log10(s)
        pred = pred + float(np.mean(DLVL - pred))
        return float(np.sum((DLVL - pred) ** 2)), pred

    g = np.array([1.0 / L[0], 1.0 / M[2], 1.0 / H[6]])
    best, pred = cost(g)
    step = np.array([0.5, 0.5, 0.5])
    for _ in range(300):
        moved = False
        for i in range(3):
            for s_ in (1, -1):
                gg = g.copy()
                gg[i] *= (1 + s_ * step[i])
                c_, aux = cost(gg)
                if c_ < best:
                    g, best, pred, moved = gg, c_, aux, True
        if not moved:
            step *= 0.5
            if step.max() < 1e-4:
                break
    res = DLVL - pred
    rms = float(np.sqrt(np.mean(res ** 2)))
    print("test_dodeca_source_lvl: Mid re 500-1k %.1f/%.1f dB "
          "(125/2k); residual(125) %.2f; RMS %.2f"
          % (m_re[0], m_re[4], res[0], rms))
    assert m_re[0] < -10 and m_re[4] < -13
    assert abs(res[0]) < 0.3 and rms < 1.2


def test_cr2_8k_comparability():
    """11.3 kHz low-pass on the CR2 measurement: C50(8k) drops >1.5 dB
    and Ts(8k) rises >10 ms, the raw 8k octave is not comparable."""
    p = os.environ.get("BRAS_CR2_WAV")
    if not p:
        return _skip("test_cr2_8k_comparability", "BRAS_CR2_WAV")
    x, fs = _read_wav(p)
    m0 = rm.band_metrics(x, fs)
    nt = 1201
    n = np.arange(nt) - nt // 2
    fc = 11314.0 / (fs / 2)
    h = fc * np.sinc(fc * n) * np.kaiser(nt, 8.0)
    xl = np.convolve(x, h, mode="same")
    m1 = rm.band_metrics(xl, fs)
    print("test_cr2_8k_comparability: C50 %.2f->%.2f dB; "
          "Ts %.1f->%.1f ms"
          % (m0["C50"][6], m1["C50"][6], m0["Ts"][6], m1["Ts"][6]))
    assert m0["C50"][6] - m1["C50"][6] > 1.5
    assert m1["Ts"][6] - m0["Ts"][6] > 10.0


if __name__ == "__main__":
    fails = 0
    for t in (test_rockfon_fit, test_genelec_phase,
              test_scene01_erratum, test_dodeca_source_lvl,
              test_cr2_8k_comparability):
        try:
            t()
        except AssertionError as e:
            fails += 1
            print("%s: FAILED (%s)" % (t.__name__, e))
    print("%d failure(s)" % fails)
    sys.exit(1 if fails else 0)
