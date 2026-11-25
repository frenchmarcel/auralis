# -*- coding: utf-8 -*-
"""Tests for source directivity."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import (  # noqa: E402
    BakeConfig, ReceiverDef, SourceDef, bake, get_directivity, make_shoebox,
    native_available,
)
from auralis.engine import stochastic  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, speed_of_sound  # noqa: E402
from auralis.engine.bands import N_BANDS  # noqa: E402
from auralis.engine.directivity import as_directivity, cone, omni  # noqa: E402
from auralis.engine.geometry import NumpyRaycaster, extract_planes  # noqa: E402
from auralis.engine.ism import compute_direct, compute_image_sources  # noqa: E402
from auralis.engine.native import NativeEngine  # noqa: E402
from auralis.engine.pipeline import apply_source_directivity  # noqa: E402

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


def test_tables():
    card = get_directivity("cardioid")
    q = card.energy(np.array([1.0, -1.0]))       # (2, NB)
    check("cardioid Q(0)=1", np.allclose(q[0], 1.0))
    check("cardioid Q(pi)=0", np.allclose(q[1], 0.0, atol=1e-12))
    f8 = get_directivity("figure8")
    check("figure8 Q(pi)=1", np.allclose(f8.energy(-1.0), 1.0))
    check("figure8 Q(pi/2)~0", float(f8.energy(0.0).max()) < 3e-3)
    check("omni mean = 1", np.allclose(omni().mean_q(), 1.0))
    check("cardioid mean ~ 1/3",
          np.allclose(card.mean_q(), 1.0 / 3.0, rtol=0.02),
          "(%.4f)" % card.mean_q()[0])
    sp = get_directivity("speaker_2way")
    di = sp.directivity_index_db()
    check("loudspeaker DI increasing", bool(np.all(np.diff(di) > 0)),
          "(%.1f -> %.1f dB)" % (di[0], di[-1]))
    cn = cone(30.0, 90.0, -20.0)
    check("cone: 1 on axis, outer gain behind",
          np.allclose(cn.energy(1.0), 1.0)
          and np.allclose(cn.energy(-1.0), 10 ** (-2.0), rtol=1e-6))
    check("as_directivity('omni') -> None",
          as_directivity("omni") is None and as_directivity(None) is None)


def test_direct_and_ism():
    sc = make_shoebox((10.0, 7.0, 3.0))
    rc = NumpyRaycaster(sc)
    air = air_attenuation_db_per_m()
    c = speed_of_sound(20.0)
    src = np.array([3.0, 3.5, 1.0])
    rcv = np.array([7.0, 3.5, 1.0])

    # figure8 perpendicular to the direct path -> near silence
    ev = compute_direct(sc, rc, src, rcv, air, c)
    ref = ev.amps.copy()
    f8 = get_directivity("figure8")
    apply_source_directivity([ev], f8, np.array([0.0, 0.0, 1.0]))
    check("direct figure8 perpendicular ~ 0",
          float((ev.amps / ref).max()) < 0.05)
    ev2 = compute_direct(sc, rc, src, rcv, air, c)
    apply_source_directivity([ev2], f8, np.array([1.0, 0.0, 0.0]))
    check("direct figure8 on axis = omni",
          np.allclose(ev2.amps, ref, rtol=1e-9))

    # floor bounce: exact analytic amplitude ratio
    planes, tp = extract_planes(sc, 48)
    evs_o = compute_image_sources(sc, planes, tp, rc, src, rcv, 1, air, c)
    evs_d = compute_image_sources(sc, planes, tp, rc, src, rcv, 1, air, c)
    card = get_directivity("cardioid")
    fwd = np.array([0.0, 0.0, -1.0])            # source aimed at the floor
    apply_source_directivity(evs_d, card, fwd)

    def floor_ev(evs):
        for e in evs:
            if e.order == 1 and e.emit_dir is not None and e.emit_dir[2] < -0.3:
                return e
        return None
    fo, fd = floor_ev(evs_o), floor_ev(evs_d)
    check("floor bounce found", fo is not None and fd is not None)
    if fo is not None:
        ct = 1.0 / np.sqrt(5.0)                 # emission (2,0,-1)/sqrt(5)
        expected = card.amp(ct)                 # value read in the table
        ratio = fd.amps / fo.amps
        check("floor bounce: exact cardioid ratio (table)",
              np.allclose(ratio, expected, rtol=1e-9),
              "(%.6f)" % float(ratio.mean()))
        check("floor bounce: ratio ~ analytic (1+ct)/2",
              abs(float(ratio.mean()) - (1.0 + ct) / 2.0) < 5e-4)
        check("emit_dir is unit length",
              abs(np.linalg.norm(fo.emit_dir) - 1.0) < 1e-9)


def test_tracers():
    sc = make_shoebox((8.0, 6.0, 3.0))
    air = air_attenuation_db_per_m()
    c = speed_of_sound(20.0)
    src = np.array([2.0, 2.0, 1.5])
    rcv = np.array([6.0, 4.0, 1.6])
    kw = dict(n_rays=30000, max_bounces=120, max_time=1.2, bin_dt=0.002,
              n_bins=600, speed_of_sound=c, air_db_per_m=air,
              spec_exclude_order=2, energy_cap=1e-6, seed=42)

    ones = np.ones((N_BANDS, 64))
    h0, _, _ = stochastic.trace_python(sc, src, rcv, **kw)
    h1, _, _ = stochastic.trace_python(sc, src, rcv, **kw,
                                       src_forward=[0, 0, -1], dir_table=ones)
    check("python: omni table == no table", np.allclose(h0, h1))

    sp = get_directivity("speaker_2way")
    fwd = (rcv - src) / np.linalg.norm(rcv - src)
    hp, _, _ = stochastic.trace_python(sc, src, rcv, **kw,
                                       src_forward=fwd, dir_table=sp.table)
    mq = sp.mean_q()
    rel = hp.sum(axis=1) / np.maximum(h0.sum(axis=1), 1e-30)
    check("python: total energy ~ mean(Q) per band",
          np.allclose(rel, mq, rtol=0.12),
          "(max gap %.1f %%)" % (100 * np.abs(rel / mq - 1).max()))

    if native_available():
        eng = NativeEngine(sc)
        n0, _, _ = eng.trace(src, rcv, kw["n_rays"], kw["max_bounces"],
                             kw["max_time"], kw["bin_dt"], kw["n_bins"], c,
                             air, seed=42)
        n1, _, _ = eng.trace(src, rcv, kw["n_rays"], kw["max_bounces"],
                             kw["max_time"], kw["bin_dt"], kw["n_bins"], c,
                             air, seed=42, src_forward=[0, 0, -1],
                             dir_table=ones)
        check("native: omni table == no table", np.allclose(n0, n1))
        nd, _, _ = eng.trace(src, rcv, kw["n_rays"], kw["max_bounces"],
                             kw["max_time"], kw["bin_dt"], kw["n_bins"], c,
                             air, seed=42, src_forward=fwd,
                             dir_table=sp.table)
        eng.close()
        eN = nd.sum()
        eP = hp.sum()
        check("native vs python (directive): energy < 2.5 %",
              abs(eN / eP - 1.0) < 0.025,
              "(%.2f %%)" % (100 * abs(eN / eP - 1)))


def test_bake_c50():
    sc = make_shoebox((12.0, 8.0, 3.5), wall_material="brick",
                      floor_material="parquet")
    src = SourceDef("pa", (10.0, 4.0, 2.0), forward=(-1.0, 0.0, 0.0))
    rcv = ReceiverDef("aud", (3.0, 4.0, 1.7), forward=(1.0, 0.0, 0.0))
    cfg = BakeConfig(n_rays=40000, seed=7, normalize=False)
    r_omni = bake(sc, src, rcv, cfg)
    src_h = SourceDef("pa", (10.0, 4.0, 2.0), forward=(-1.0, 0.0, 0.0),
                      directivity="horn")
    r_horn = bake(sc, src_h, rcv, cfg)
    c_o = r_omni.metrics["per_band"]["2k"]["C50"]
    c_h = r_horn.metrics["per_band"]["2k"]["C50"]
    check("horn aimed at listener: C50(2k) rises by at least 2 dB",
          c_h > c_o + 2.0, "(%.1f -> %.1f dB)" % (c_o, c_h))
    d_o = r_omni.metrics["eax"]["DecayTime_s"]
    d_h = r_horn.metrics["eax"]["DecayTime_s"]
    check("decay time barely changed",
          abs(d_h - d_o) / d_o < 0.35, "(%.2f vs %.2f s)" % (d_o, d_h))


if __name__ == "__main__":
    print("== tables & presets ==")
    test_tables()
    print("== direct & ISM ==")
    test_direct_and_ism()
    print("== tracers ==")
    test_tracers()
    print("== full bake ==")
    test_bake_c50()
    print("\n%d ok, %d failures" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
