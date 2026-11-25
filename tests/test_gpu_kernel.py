# -*- coding: utf-8 -*-
"""Validation of the CUDA kernel through its EMULATION build
(g++ -DAUR_EMU). Bit-exact parity of raycast and permeation with the
CPU kernel, statistical tracer equivalence, determinism, directivity.
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import BakeConfig, ReceiverDef, SourceDef, bake, \
    make_shoebox  # noqa: E402
from auralis.engine import materials as mat_lib  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, \
    speed_of_sound  # noqa: E402
from auralis.engine.directivity import get_directivity  # noqa: E402
from auralis.engine.geometry import SceneData, box_triangles  # noqa: E402
from auralis.engine.native import NativeEngine, find_library  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMU = os.path.join(ROOT, "auralis", "lib", "linux-x86_64",
                   "libauralis_cuda_emu.so")

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s %s" % (name, extra))
    else:
        FAIL += 1
        print("  FAIL %s %s" % (name, extra))


def ensure_emu():
    if os.path.isfile(EMU):
        return True
    src = os.path.join(ROOT, "native", "src", "aur_cuda.cu")
    cmd = ["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-DAUR_EMU",
           "-x", "c++", src, "-o", EMU]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:  # pragma: no cover
        print("  SKIP emulation build failed:", e)
        return False


def build_brick_scene():
    L, W, H, T = 8.0, 6.0, 3.0, 0.05
    Y0, Y1, Z0, Z1 = 2.2, 3.8, 1.0, 2.2
    names = ["brick", "glass_window", "concrete", "plaster"]
    mats = [mat_lib.get_preset(n) for n in names]
    BR, GL, BE, PL = range(4)
    boxes = [((L + 4.0, W, T), (0, 0, -T), BE), ((L, W, T), (0, 0, H), PL),
             ((L, T, H), (0, -T, 0), BR), ((L, T, H), (0, W, 0), BR),
             ((T, W, H), (-T, 0, 0), BR),
             ((T, W, Z0), (L, 0, 0), BR), ((T, W, H - Z1), (L, 0, Z1), BR),
             ((T, Y0, Z1 - Z0), (L, 0, Z0), BR),
             ((T, W - Y1, Z1 - Z0), (L, Y1, Z0), BR),
             ((0.02, Y1 - Y0, Z1 - Z0), (L, Y0, Z0), GL)]
    tv, tm = [], []
    for size, orig, mi in boxes:
        t3, _ = box_triangles(size, orig, inward=False)
        tv.append(t3)
        tm.append(np.full(len(t3), mi, np.int32))
    return SceneData.from_materials(np.concatenate(tv), np.concatenate(tm),
                                    mats, names)


def test_host_parity(sc):
    """raycast + permeation: same host algorithms -> bit-exact identity."""
    cpu = NativeEngine(sc)
    emu = NativeEngine(sc, lib_path=EMU)
    check("versions", "emu" in emu.version and "emu" not in cpu.version,
          "(%s vs %s)" % (cpu.version, emu.version))

    rng = np.random.default_rng(3)
    same_tri = same_t = 0
    N = 400
    for _ in range(N):
        o = rng.uniform([0.5, 0.5, 0.2], [7.5, 5.5, 2.8])
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        t1 = cpu.cast(o, d)
        t2 = emu.cast(o, d)
        same_tri += (t1[0] == t2[0])
        if t1[0] >= 0 and t1[0] == t2[0]:
            same_t += (abs(t1[1] - t2[1]) < 1e-6)
    check("raycast: %d/%d same triangles" % (same_tri, N), same_tri == N)
    check("raycast: identical distances", same_t == same_tri)

    tr1, n1 = cpu.segment_transmission((10.5, 3.0, 1.6), (4.0, 3.0, 1.5))
    tr2, n2 = emu.segment_transmission((10.5, 3.0, 1.6), (4.0, 3.0, 1.5))
    check("permeation through glass: identical (crossings %d)" % n1,
          n1 == n2 and np.allclose(tr1, tr2, rtol=0, atol=0))
    cpu.close()
    emu.close()


def _trace(engine, sc, src, rcv, n_rays, seed, dirv=None, fwd=None,
           max_time=1.0):
    air = air_attenuation_db_per_m()
    c = speed_of_sound(20.0)
    n_bins = int(max_time / 0.002)
    kw = {}
    if dirv is not None:
        kw = dict(src_forward=fwd, dir_table=dirv.table)
    return engine.trace(src, rcv, n_rays, 150, max_time, 0.002, n_bins,
                        c, air, spec_exclude_order=2, energy_cap=1e-6,
                        seed=seed, **kw)


def test_tracer_equivalence():
    sc = make_shoebox((8.0, 6.0, 3.0), wall_material="brick",
                      floor_material="parquet",
                      ceiling_material="acoustic_tile")
    src = np.array([2.0, 2.0, 1.5])
    rcv = np.array([6.0, 4.0, 1.6])
    cpu = NativeEngine(sc)
    emu = NativeEngine(sc, lib_path=EMU)

    # The split estimator (sphere detection + s*E rain) has low bias but
    # detection is a RARE-EVENT estimator: the per-band variance decays
    # as 1/sqrt(N_detections). Measured (2 seeds, identical conditions):
    # 60k rays give max|rel-1| 3.5-8.2 % and r 0.976-0.979. 480k give
    # <= 1.2 % and r >= 0.997. Hence 480k rays with a 2x margin.
    N_EQ = 480_000
    h1, d1, s1 = _trace(cpu, sc, src, rcv, N_EQ, seed=42)
    h2, d2, s2 = _trace(emu, sc, src, rcv, N_EQ, seed=42)
    rel = h2.sum(axis=1) / np.maximum(h1.sum(axis=1), 1e-30)
    check("per-band energy CPU~EMU (< 2.5 %%): max %.2f %%"
          % (100 * np.abs(rel - 1).max()), np.abs(rel - 1).max() < 0.025)
    check("mean bounces ~ (%.1f vs %.1f)"
          % (s1["mean_bounces"], s2["mean_bounces"]),
          abs(s2["mean_bounces"] / s1["mean_bounces"] - 1) < 0.02)
    leak1, leak2 = s1["escaped"] / N_EQ, s2["escaped"] / N_EQ
    check("closed box: edge leaks negligible and comparable "
          "(%.3f %% vs %.3f %%)" % (100 * leak1, 100 * leak2),
          leak1 < 2e-3 and leak2 < 2e-3)
    prof1 = h1.sum(axis=0)
    prof2 = h2.sum(axis=0)
    n = min(len(prof1), len(prof2))
    corr = float(np.corrcoef(prof1[:n], prof2[:n])[0, 1])
    check("time profiles correlated (r=%.4f)" % corr, corr > 0.995)

    # emulation determinism: bit-exact
    h3, d3, _ = _trace(emu, sc, src, rcv, 20000, seed=7)
    h4, d4, _ = _trace(emu, sc, src, rcv, 20000, seed=7)
    check("emulation is deterministic (bit-exact)",
          np.array_equal(h3, h4) and np.array_equal(d3, d4))

    # directivity
    sp = get_directivity("speaker_2way")
    fwd = (rcv - src) / np.linalg.norm(rcv - src)
    # 480 k rays like the other parity checks: detection is a RARE-event
    # estimator, its per-band variance decays as 1/sqrt(N) (measured:
    # 60 k -> 5.5 %, 480 k -> 1.5 %).
    hd1, _, _ = _trace(cpu, sc, src, rcv, 480000, seed=42, dirv=sp, fwd=fwd)
    hd2, _, _ = _trace(emu, sc, src, rcv, 480000, seed=42, dirv=sp, fwd=fwd)
    reld = hd2.sum(axis=1) / np.maximum(hd1.sum(axis=1), 1e-30)
    check("directional: per-band energy (< 2.5 %%): max %.2f %%"
          % (100 * np.abs(reld - 1).max()), np.abs(reld - 1).max() < 0.025)
    cpu.close()
    emu.close()

    # free field: a single distant triangle -> 100 % escape
    tri = np.array([[[-500.0, -500.0, -200.0], [500.0, -500.0, -200.0],
                     [0.0, 500.0, -200.0]]])
    free = SceneData(tri_verts=tri, tri_mat=np.zeros(1, np.int32),
                     absorption=np.full((1, 7), 0.9, np.float32),
                     scattering=np.full((1, 7), 0.2, np.float32),
                     transmission=np.zeros((1, 7), np.float32),
                     mat_names=["far"])
    # rays hitting the plate at ~200 m break on max_time BEFORE they
    # escape, so CPU and EMU are compared against each other.
    ec = NativeEngine(free)
    ee = NativeEngine(free, lib_path=EMU)
    _, _, sc_ = _trace(ec, free, np.array([0, 0, 1.0]),
                       np.array([3, 0, 1.0]), 20000, seed=1, max_time=0.3)
    _, _, se_ = _trace(ee, free, np.array([0, 0, 1.0]),
                       np.array([3, 0, 1.0]), 20000, seed=1, max_time=0.3)
    ec.close()
    ee.close()
    fc, fe = sc_["escaped"] / 20000, se_["escaped"] / 20000
    check("free field: escape fraction CPU~EMU (%.1f %% vs %.1f %%)"
          % (100 * fc, 100 * fe), fc > 0.5 and abs(fe - fc) < 0.02)


def test_full_bake_via_emu():
    """A full pipeline bake, forcing the emulation library."""
    from auralis.engine import pipeline
    sc = build_brick_scene()
    orig = pipeline._select_engine

    def sel(scene, mode):
        eng = NativeEngine(scene, lib_path=EMU)
        return "cuda-emu", eng, eng
    pipeline._select_engine = sel
    try:
        r = bake(sc, SourceDef("s", (2.0, 1.5, 1.4)),
                 ReceiverDef("m", (4.0, 3.0, 1.5), forward=(1, 0, 0)),
                 BakeConfig(n_rays=20000, seed=7, max_time=1.5))
    finally:
        pipeline._select_engine = orig
    rt = r.metrics["per_band"]["1k"]["RT"]
    check("full bake through the CUDA-emu kernel: RT(1k)=%.2f s [%s]"
          % (rt if rt else -1, r.engine_used),
          r.engine_used == "cuda-emu" and rt is not None and 2.0 < rt < 4.5
          and np.isfinite(r.ir).all())


if __name__ == "__main__":
    if not ensure_emu():
        print("0 ok, 0 failures (emulation unavailable)")
        sys.exit(0)
    sc = build_brick_scene()
    print("== host parity (raycast / permeation) ==")
    test_host_parity(sc)
    print("== tracer equivalence ==")
    test_tracer_equivalence()
    print("== full pipeline ==")
    test_full_bake_via_emu()
    print("\n%d ok, %d failures" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
