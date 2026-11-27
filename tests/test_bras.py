# -*- coding: utf-8 -*-
"""BRAS bench tests: IR metrics and full CLI round trip.
Without the BRAS database the validation is SYNTHETIC (simulated box)."""
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import rirmetrics as rm
from auralis.engine.geometry import SceneData
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake
from auralis.engine.wavio import write_wav
from auralis.engine import materials as mat_lib

_n_ok = 0
_n_fail = 0


def check(label, cond):
    global _n_ok, _n_fail
    if cond:
        _n_ok += 1
        print("  ok   %s" % label)
    else:
        _n_fail += 1
        print("  FAIL %s" % label)
        if "PYTEST_CURRENT_TEST" in os.environ:     # under pytest a failed check fails
            raise AssertionError(label)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_rt_synthetic():
    """Exponential decay T=0.6 s: T20/T30/EDT must recover it.
    Averaged over 3 draws, tolerances widened at low frequency."""
    fs = 24000
    T = 0.6
    t = np.arange(int(1.4 * fs)) / fs
    acc = {k: np.zeros(rm.N_BANDS)
           for k in ("T20", "T30", "EDT", "C50", "Ts")}
    seeds = (3, 5, 11)
    for sd in seeds:
        rng = np.random.default_rng(sd)
        ir = rng.standard_normal(len(t)) * np.exp(-6.9077 * t / T)
        m = rm.band_metrics(ir, fs)
        for key in acc:
            for b in range(rm.N_BANDS):
                v = m[key][b]
                if v is None:
                    check("%s band %s: None (seed %d)"
                          % (key, rm.LABELS[b], sd), False)
                    return
                acc[key][b] += v / len(seeds)
    def tol(key, b):
        # windows: T30 30 dB, T20 20 dB, EDT 9.5 dB -> the estimation
        # variance grows as sqrt(60/range) and as 1/sqrt(bandwidth)
        if key == "EDT":
            # positive bias at low frequency: the zero-phase filter
            # spread (~1/bandwidth, two-sided) flattens the first 10
            # dB -- exactly the EDT window
            return 0.45 if b == 0 else (0.30 if b == 1 else 0.12)
        if key == "T20":
            return 0.15 if b == 0 else 0.08
        return 0.12 if b == 0 else 0.08

    for b in range(rm.N_BANDS):
        for key in ("T20", "T30", "EDT"):
            v = acc[key][b]
            if abs(v - T) > tol(key, b) * T:
                check("%s band %s = %.3f (expected %.2f +/- %d%%)"
                      % (key, rm.LABELS[b], v, T,
                         tol(key, b) * 100), False)
                return
    check("T20/T30/EDT recover T=0.6 s over 7 bands (3 draws)",
          True)
    # analytic C50/Ts of an exponential decay (averaged over the 3
    # draws): C50 = 10log10((1-q)/q), q = exp(-13.82*0.05/T);
    # Ts = 1000*T/13.82
    q = np.exp(-13.8155 * 0.05 / 0.6)
    c50_th = 10 * np.log10((1 - q) / q)
    ts_th = 1000.0 * 0.6 / 13.8155
    okc = all(abs(acc["C50"][b] - c50_th) < 0.6 for b in (2, 3, 4))
    okt = all(abs(acc["Ts"][b] - ts_th) / ts_th < 0.12
              for b in (2, 3, 4))
    check("analytic C50 recovered (%.1f dB +/- 0.6, 3 draws)"
          % c50_th, okc)
    check("analytic Ts recovered (%.0f ms +/- 12%%, 3 draws)"
          % ts_th, okt)


def test_level_delta():
    """A 1 kHz band boost is seen by dLevel, the other bands barely
    move (broadband renormalization effect only)."""
    fs = 24000
    rng = np.random.default_rng(5)
    t = np.arange(int(0.8 * fs)) / fs
    ir = rng.standard_normal(len(t)) * np.exp(-6.9077 * t / 0.5)
    bands = rm.band_split(ir, fs)
    ir2 = bands.sum(axis=0) + bands[3]        # 1k band doubled (+6 dB)
    d = rm.compare_metrics(rm.band_metrics(ir2, fs),
                           rm.band_metrics(ir, fs))
    check("1 kHz boost detected (dLevel=%.1f dB > 3)" % d["dLevel"][3],
          d["dLevel"][3] > 3.0)
    check("250 Hz band barely affected (|%.2f| < 1.5 dB)" % d["dLevel"][1],
          abs(d["dLevel"][1]) < 1.5)


def test_reflection_bands():
    """|R| of an isolated reflection from two synthetic impulses:
    |R|=0.6 recovered in resolved bands, None in the low ones."""
    fs = 44100
    S = np.array([0.0, 0.0, 1.5])
    R = np.array([2.0, 0.0, 1.5])
    r_d = float(np.linalg.norm(R - S))
    Sm = S * np.array([1, 1, -1])
    r_r = float(np.linalg.norm(R - Sm))
    c = 343.0
    td, tr = r_d / c, r_r / c
    ir = np.zeros(int(0.1 * fs))
    ir[int(round(td * fs))] = 1.0 / r_d
    ir[int(round(tr * fs))] = 0.6 / r_r
    Rb = rm.reflection_bands(ir, fs, td, tr, r_d, r_r)
    ok = all(v is not None and abs(v - 0.6) < 0.05
             for v in Rb[3:])
    check("isolated reflection: |R|=0.6 recovered (1k-8k) %s"
          % ["%.2f" % v if v else "--" for v in Rb], ok)
    check("unresolved low bands -> None", Rb[0] is None)


def test_reflection_relative():
    """Relative mode: a stray directivity biases the absolute
    estimator but cancels in the material/rigid ratio."""
    fs = 44100
    rd, rr = 2.0, 3.606
    td, tr = rd / 343.0, rr / 343.0
    g = 2.0          # fake source boost toward the reflection
    R0, Rtrue = 0.99, 0.6
    n = int(0.1 * fs)
    ir_rig = np.zeros(n)
    ir_rig[int(round(td * fs))] = 1.0 / rd
    ir_rig[int(round(tr * fs))] = R0 * g / rr
    ir_mat = np.zeros(n)
    ir_mat[int(round(td * fs))] = 1.0 / rd
    ir_mat[int(round(tr * fs))] = Rtrue * g / rr
    Rabs = rm.reflection_bands(ir_mat, fs, td, tr, rd, rr)
    Rrel = rm.reflection_bands_relative(ir_mat, ir_rig, fs, td, tr,
                                        R0)
    check("absolute polluted by directivity (%.2f ~ %.2f)"
          % (Rabs[5], Rtrue * g), abs(Rabs[5] - Rtrue * g) < 0.06)
    ok = all(v is not None and abs(v - Rtrue) < 0.03
             for v in Rrel[3:])
    check("relative recovers |R|=0.6 despite directivity", ok)


def test_sofa_pairs():
    """The SOFA extractor reads positions and labels from a compliant
    synthetic file (cartesian SourcePosition, EmitterID)."""
    h5py = pytest.importorskip("h5py")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "tools"))
    import sofa_pairs
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.sofa")
        with h5py.File(p, "w") as f:
            f["Data.IR"] = np.zeros((2, 1, 64))
            f["Data.SamplingRate"] = np.array([44100.0])
            sp = f.create_dataset("SourcePosition",
                                  data=np.array([[1.0, 2.0, 1.5],
                                                 [1.0, 2.0, 1.5]]))
            sp.attrs["Type"] = np.bytes_("cartesian")
            lp = f.create_dataset("ListenerPosition",
                                  data=np.array([[4.0, 3.0, 1.2],
                                                 [5.0, 3.0, 1.2]]))
            lp.attrs["Type"] = np.bytes_("cartesian")
            f["EmitterID"] = np.array([[1.0], [1.0]])
            f["ReceiverID"] = np.array([[1.0], [2.0]])
        pairs, fs = sofa_pairs.read_pairs(
            p, pattern="wav/X_{emitter}_{receiver}.wav")
        ok = (fs == 44100.0 and len(pairs) == 2
              and pairs[0]["label"] == "LS1-MP1"
              and pairs[1]["receiver"] == [5.0, 3.0, 1.2]
              and pairs[0]["reference_wav"] == "wav/X_LS1_MP1.wav")
        check("SOFA extractor: per-measurement fields", ok)

        # real BRAS layout: dummy Source/ListenerPosition at zero,
        # Emitter/ReceiverPosition as (1, 3, M) -- axis 1 = coordinate,
        # axis 2 = measurement (checked on RS1_RIRs_Rigid.sofa)
        p2 = os.path.join(tmp, "t2.sofa")
        xyz_e = np.array([[1.0, 3.0, 3.0],       # x of the measurements
                          [2.0, 2.0, 2.0],       # y
                          [1.8, 1.8, 1.8]])      # z
        xyz_r = np.array([[4.0, 4.0, 4.0],
                          [1.0, 1.0, 2.0],
                          [1.2, 1.2, 1.2]])
        with h5py.File(p2, "w") as f:
            f["Data.IR"] = np.zeros((3, 1, 64))
            f["Data.SamplingRate"] = np.array([44100.0])
            f["SourcePosition"] = np.zeros((1, 3))
            f["ListenerPosition"] = np.zeros((1, 3))
            ep = f.create_dataset("EmitterPosition",
                                  data=xyz_e[None, :, :])
            ep.attrs["Type"] = np.bytes_("cartesian")
            rp = f.create_dataset("ReceiverPosition",
                                  data=xyz_r[None, :, :])
            rp.attrs["Type"] = np.bytes_("cartesian")
            f["EmitterID"] = np.array([[1.0], [2.0], [2.0]])
            f["ReceiverID"] = np.array([[1.0], [1.0], [2.0]])
        pairs, _ = sofa_pairs.read_pairs(p2)
        ok = (pairs[0]["source"] == [1.0, 2.0, 1.8]
              and pairs[1]["source"] == [3.0, 2.0, 1.8]
              and pairs[2]["receiver"] == [4.0, 2.0, 1.2]
              and pairs[2]["label"] == "LS2-MP2")
        check("SOFA extractor: BRAS layout (1,3,M)", ok)

        # standard SOFA table (E,3) indexed by the IDs
        p3 = os.path.join(tmp, "t3.sofa")
        with h5py.File(p3, "w") as f:
            f["Data.IR"] = np.zeros((3, 1, 64))
            f["Data.SamplingRate"] = np.array([44100.0])
            f["SourcePosition"] = np.zeros((1, 3))
            f["ListenerPosition"] = np.zeros((1, 3))
            f["EmitterPosition"] = np.array([[1.0, 2.0, 1.8],
                                             [3.0, 2.0, 1.8]])
            f["ReceiverPosition"] = np.array([[4.0, 1.0, 1.2],
                                              [4.0, 2.0, 1.2]])
            f["EmitterID"] = np.array([[1.0], [2.0], [2.0]])
            f["ReceiverID"] = np.array([[1.0], [1.0], [2.0]])
        pairs, _ = sofa_pairs.read_pairs(p3)
        ok3 = (pairs[0]["source"] == [1.0, 2.0, 1.8]
               and pairs[2]["receiver"] == [4.0, 2.0, 1.2])
        check("SOFA extractor: (E,3) table indexed by ID", ok3)

        # complex room: Data.IR as M x R x E x N, positions as
        # tables (mic grid x loudspeakers)
        p4 = os.path.join(tmp, "t4.sofa")
        with h5py.File(p4, "w") as f:
            f["Data.IR"] = np.zeros((1, 2, 2, 64))
            f["Data.SamplingRate"] = np.array([44100.0])
            f["SourcePosition"] = np.zeros((1, 3))
            f["ListenerPosition"] = np.zeros((1, 3))
            f["EmitterPosition"] = np.array([[1.0, -2.5, 0.7],
                                             [-1.4, -2.5, 0.7]])
            f["ReceiverPosition"] = np.array([[-1.0, 1.4, 1.2],
                                              [0.8, 2.0, 1.2]])
            f["EmitterID"] = np.array([[1.0], [2.0]])
            f["ReceiverID"] = np.array([[1.0], [2.0]])
        pairs, _ = sofa_pairs.read_pairs(
            p4, pattern="w/{emitter}_{receiver}.wav")
        ok4 = (len(pairs) == 4
               and pairs[0]["label"] == "LS1-MP1"
               and pairs[1]["label"] == "LS1-MP2"
               and pairs[3]["source"] == [-1.4, -2.5, 0.7]
               and pairs[3]["receiver"] == [0.8, 2.0, 1.2])
        check("SOFA extractor: 4-D grid for complex rooms", ok4)

        # --into: direct injection into a manifest
        mp = os.path.join(tmp, "man.json")
        with open(mp, "w") as fh:
            json.dump({"name": "x", "pairs": [{"vieux": True}]}, fh)
        rc = sofa_pairs.main([p2, "--into", mp])
        with open(mp) as fh:
            man = json.load(fh)
        check("--into replaces the manifest pairs",
              rc == 0 and len(man["pairs"]) == 3
              and man["pairs"][2]["label"] == "LS2-MP2")


# ---------------------------------------------------------------------------
# Bench round trip
# ---------------------------------------------------------------------------

def _box_scene(alpha):
    X, Y, Z = 4.0, 3.0, 2.5
    v = np.array([[bx * X, by * Y, bz * Z]
                  for bx in (0, 1) for by in (0, 1) for bz in (0, 1)],
                 dtype=np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4),
             (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]
    tris = []
    for a, b, c, d in quads:
        tris.append([v[a], v[b], v[c]])
        tris.append([v[a], v[c], v[d]])
    tv = np.ascontiguousarray(np.array(tris))
    mats = [{"absorption": np.full(7, alpha, np.float32),
             "scattering": np.full(7, 0.2, np.float32),
             "transmission": np.zeros(7, np.float32)}]
    a, s, t = mat_lib.build_tables(mats)
    return SceneData(tri_verts=tv,
                     tri_mat=np.zeros(len(tv), np.int32),
                     absorption=a, scattering=s, transmission=t,
                     mat_names=["boite"]), tv


_ENGINE = {"n_rays": 20000, "ism_order": 1, "seed": 7, "n_threads": 1,
           "engine": "auto"}


def _csv_by_band(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["bande"]] = row
    return out


def test_hybrid_box_c50():
    """Hybrid seam: in a specular box with CR2 dimensions, C50 must
    follow the exact image lattice. ~25 s."""
    from auralis.engine import air as airmod
    from auralis.engine.pipeline import BakeConfig as BC
    L = np.array([6.2, 8.6, 3.0]); alpha = 0.10
    S = np.array([4.03, 1.75, 0.72]); R = np.array([2.11, 5.73, 1.23])
    c = airmod.speed_of_sound(20.0)
    air = airmod.air_attenuation_db_per_m(
        np.array(rm.CENTERS, float), 20.0, 50.0)
    scene, _tv = _box_scene(alpha)
    scene.tri_verts[:] = scene.tri_verts * (L / np.array([4.0, 3.0, 2.5]))
    scene.scattering[:] = 0.0
    n = 44; k = np.arange(-n, n + 1)
    ax = []
    for d in range(3):
        pos = np.where(k % 2 == 0, k * L[d] + S[d],
                       (k + 1) * L[d] - S[d])
        ax.append(pos - R[d])
    DX, DY, DZ = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    dist = np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2)
    NBc = (np.abs(k)[:, None, None] + np.abs(k)[None, :, None]
           + np.abs(k)[None, None, :])
    kair = np.log(10) / 10 * air[3]
    E = ((1.0 - alpha) ** NBc / np.maximum(dist, 1e-9) ** 2
         * np.exp(-kair * dist))
    r = dist / c - float((dist / c).min())
    e50 = float(E[r < 0.05].sum()); et = float(E.sum())
    C50_th = 10 * np.log10(e50 / (et - e50))
    for order, tol in ((1, 0.7), (2, 0.7)):
        cfg = BC(normalize=False, max_time=1.2, n_rays=120000,
                 ism_order=order, seed=7, n_threads=1, fs=44100,
                 temperature_c=20.0, humidity_percent=50.0,
                 diffraction=False)
        res = bake(scene, SourceDef(name="s", position=S),
                   ReceiverDef(name="r", position=R), cfg)
        m = rm.band_metrics(res.ir.mean(axis=1), res.fs)
        d = abs(m["C50"][3] - C50_th)
        check("hybrid seam ism%d: C50(1k) %.2f vs lattice %.2f "
              "(|d| %.2f <= %.1f)" % (order, m["C50"][3], C50_th, d,
                                      tol), d <= tol)


def test_bench_roundtrip():
    """Reference = direct bake; with the same seed the bench must find
    itself back (dT30 ~ 0), then detect a perturbed absorption."""
    scene, tv = _box_scene(0.20)
    cfg = BakeConfig(normalize=False, max_time=0.4, **_ENGINE)
    src = SourceDef(name="s", position=np.array([1.0, 1.0, 1.2]))
    rcv = ReceiverDef(name="r", position=np.array([2.8, 2.0, 1.4]))
    res = bake(scene, src, rcv, cfg)
    tool = os.path.join(os.path.dirname(__file__), "..", "BRAS",
                        "bras_bench.py")
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "ref.wav")
        write_wav(ref, res.ir.mean(axis=1).astype(np.float32), res.fs)

        def manifest(alpha, out_name):
            man = {"name": out_name, "fs": res.fs, "duration": 0.4,
                   "engine": dict(_ENGINE),
                   "geometry": {"triangles": tv.tolist()},
                   "materials": {"boite": {
                       "absorption": [alpha] * 7,
                       "scattering": 0.2, "transmission": 0.0}},
                   "pairs": [{"label": "P1",
                              "source": [1.0, 1.0, 1.2],
                              "receiver": [2.8, 2.0, 1.4],
                              "reference_wav": "ref.wav"}]}
            p = os.path.join(tmp, out_name + ".json")
            with open(p, "w") as fh:
                json.dump(man, fh)
            return p

        # 1) self comparison: same parameters, same seed
        r = subprocess.run([sys.executable, tool,
                            manifest(0.20, "self"),
                            "-o", os.path.join(tmp, "self_out")],
                           capture_output=True, text=True)
        check("bench: exit 0 (self)", r.returncode == 0)
        if r.returncode:
            print(r.stdout[-800:], r.stderr[-800:])
            return
        rows = _csv_by_band(os.path.join(tmp, "self_out", "report.csv"))
        ok = True
        for b in ("500", "1k", "2k"):
            dt = rows[b]["dT30"]
            dl = rows[b]["dL"]
            if dt in ("", "None") or abs(float(dt)) > 0.02:
                ok = False
            if dl in ("", "None") or abs(float(dl)) > 0.3:
                ok = False
        check("self comparison: dT30<=0.02 s and dL<=0.3 dB (mids)",
              ok)

        # 2) perturbation: alpha 0.20 -> 0.55, reverb must drop
        r = subprocess.run([sys.executable, tool,
                            manifest(0.55, "pert"),
                            "-o", os.path.join(tmp, "pert_out")],
                           capture_output=True, text=True)
        check("bench: exit 0 (perturbed)", r.returncode == 0)
        if r.returncode:
            print(r.stdout[-800:], r.stderr[-800:])
            return
        rows = _csv_by_band(os.path.join(tmp, "pert_out", "report.csv"))
        dts = [float(rows[b]["dT30"]) for b in ("500", "1k", "2k")
               if rows[b]["dT30"] not in ("", "None")]
        check("perturbation detected (median dT30 %.3f s < -0.05)"
              % (np.median(dts) if dts else 0.0),
              len(dts) >= 2 and np.median(dts) < -0.05)


if __name__ == "__main__":
    print("== metrics ==")
    test_rt_synthetic()
    test_level_delta()
    print("== isolated reflection / SOFA ==")
    test_reflection_bands()
    test_reflection_relative()
    test_sofa_pairs()
    print("== hybrid seam ==")
    test_hybrid_box_c50()
    print("== BRAS bench ==")
    test_bench_roundtrip()
    print()
    print("%d ok, %d failed" % (_n_ok, _n_fail))
    sys.exit(1 if _n_fail else 0)
