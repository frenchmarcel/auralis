# -*- coding: utf-8 -*-
"""Auralis engine validation suite: physics (air, ISM, RT),
native/python consistency, I/O round trips.
Run: python3 tests/test_engine.py"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import (BakeConfig, N_BANDS, NumpyRaycaster, ReceiverDef,
                            SceneData, SourceDef, air_attenuation_db_per_m,
                            add_box, bake, compute_direct,
                            compute_image_sources, extract_planes,
                            fft_convolve, get_preset, make_shoebox,
                            make_test_signal, native_available, read_wav,
                            synthesize_mono, trace_python, write_wav)
from auralis.engine.bands import band_masks
from auralis.engine.metrics import events_to_hist, schroeder_edc

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %s %s" % (name, detail))


def rt_from_hist(hist, bin_dt, lo=-5.0, hi=-25.0):
    e = hist.sum(axis=0)
    edc = schroeder_edc(e)
    db = 10 * np.log10(np.maximum(edc, 1e-12))
    t = np.arange(len(e)) * bin_dt
    i = np.where(db <= lo)[0][0]
    j = np.where(db <= hi)[0][0]
    A = np.vstack([t[i:j], np.ones(j - i)]).T
    slope, _ = np.linalg.lstsq(A, db[i:j], rcond=None)[0]
    return -60.0 / slope


def test_air():
    print("[air] ISO 9613-1")
    a = air_attenuation_db_per_m()
    check("increases with frequency", bool(np.all(np.diff(a) > 0)),
          "8k=%.4f dB/m" % a[-1])
    a_dry = air_attenuation_db_per_m(humidity_percent=10)
    check("dry air absorbs more at 8k", a_dry[-1] > a[-1])
    check("1 kHz order of magnitude", 0.001 < a[3] < 0.02, "%.4f" % a[3])


def test_bands():
    print("[bands] rFFT masks")
    fs = 48000
    f = np.fft.rfftfreq(8192, 1.0 / fs)
    m = band_masks(f, fs)
    s = (m ** 2).sum(axis=0)
    mid = (f > 100) & (f < 20000)
    check("partition ~1 (energy)", float(np.abs(s[mid] - 1).max()) < 0.35,
          "max dev %.3f" % float(np.abs(s[mid] - 1).max()))


def test_direct_calibration():
    print("[ism] 1/d calibration and transmission")
    scene = make_shoebox((20.0, 20.0, 10.0), "concrete")
    rc = NumpyRaycaster(scene)
    air0 = np.zeros(N_BANDS)
    for d in (1.0, 3.0, 7.5):
        ev = compute_direct(scene, rc, (2, 2, 2), (2 + d, 2, 2), air0, 343.0)
        check("amp=1/d (d=%.1f)" % d,
              abs(ev.amps[3] - 1.0 / d) < 1e-6, "%.5f" % ev.amps[3])
        check("time=d/c (d=%.1f)" % d, abs(ev.time - d / 343.0) < 1e-9)

    # opaque wall in between -> silence; leaky door -> sqrt(tau)/d
    scene2 = add_box(scene, (0.2, 6.0, 6.0), (9.9, 7.0, 0.0), "concrete")
    rc2 = NumpyRaycaster(scene2)
    ev = compute_direct(scene2, rc2, (5, 10, 2), (15, 10, 2), air0, 343.0)
    check("opaque wall -> zero direct", float(ev.amps.max()) < 1e-9)

    door = get_preset("door_wood")
    scene3 = add_box(scene, (0.2, 6.0, 6.0), (9.9, 7.0, 0.0), door)
    rc3 = NumpyRaycaster(scene3)
    ev = compute_direct(scene3, rc3, (5, 10, 2), (15, 10, 2), air0, 343.0)
    tau2 = float(door["transmission"][3]) ** 2      # two crossings (box)
    expect = math.sqrt(tau2) / 10.0
    check("door -> sqrt(tau^2)/d", abs(ev.amps[3] - expect) < 1e-6,
          "%.6f vs %.6f" % (ev.amps[3], expect))


def test_image_sources():
    print("[ism] image sources (box 8x6x3)")
    scene = make_shoebox((8.0, 6.0, 3.0), "plaster")
    rc = NumpyRaycaster(scene)
    planes, tri_plane = extract_planes(scene)
    check("6 planes extracted", len(planes) == 6, str(len(planes)))
    src = np.array([2.0, 2.0, 1.2])
    rcv = np.array([6.0, 4.0, 1.6])
    air0 = np.zeros(N_BANDS)
    evs = compute_image_sources(scene, planes, tri_plane, rc, src, rcv,
                                1, air0, 343.0)
    check("6 first-order reflections", len(evs) == 6, str(len(evs)))
    # floor bounce analytic check
    d_floor = math.hypot(math.hypot(4.0, 2.0), src[2] + rcv[2])
    times = sorted(ev.time for ev in evs)
    check("floor arrival", any(abs(t - d_floor / 343.0) < 1e-4 for t in times))
    r = 1.0 - get_preset("plaster")["absorption"][3]
    ev_floor = min(evs, key=lambda e: abs(e.time - d_floor / 343.0))
    check("floor bounce amp = sqrt(r)/L",
          abs(ev_floor.amps[3] - math.sqrt(r) / d_floor) < 1e-6)
    evs2 = compute_image_sources(scene, planes, tri_plane, rc, src, rcv,
                                 2, air0, 343.0)
    check("order 2 present", len(evs2) > 6, str(len(evs2)))
    # occluded first-order path must be rejected
    scene_o = add_box(scene, (0.4, 0.4, 3.0), (3.8, 2.8, 0.0), "concrete")
    rc_o = NumpyRaycaster(scene_o)
    planes_o, tp_o = extract_planes(scene_o)
    evs_o = compute_image_sources(scene_o, planes_o, tp_o, rc_o, src, rcv,
                                  1, air0, 343.0)
    check("occlusion filters out paths", len(evs_o) < 6,
          "%d < 6" % len(evs_o))


def _shoebox_rt_reference(alpha=0.15):
    V = 10 * 7 * 3.0
    S = 2 * (10 * 7 + 7 * 3 + 10 * 3.0)
    sab = 0.161 * V / (S * alpha)
    eyr = 0.161 * V / (-S * math.log(1 - alpha))
    return sab, eyr


def test_python_tracer_rt():
    print("[stochastic] RT box 10x7x3, alpha=0.15 (Python)")
    scene = make_shoebox((10.0, 7.0, 3.0),
                         {"absorption": np.full(N_BANDS, 0.15, np.float32),
                          "scattering": np.full(N_BANDS, 0.3, np.float32),
                          "transmission": np.zeros(N_BANDS, np.float32)})
    hist, dirs, st = trace_python(scene, (2, 2, 1.5), (7.5, 5, 1.6),
                                  n_rays=6000, max_bounces=160, max_time=2.2,
                                  bin_dt=0.002, n_bins=1100,
                                  speed_of_sound=343.0,
                                  air_db_per_m=np.zeros(N_BANDS),
                                  spec_exclude_order=0, energy_cap=1e-6,
                                  seed=11)
    t20 = rt_from_hist(hist, 0.002)
    sab, eyr = _shoebox_rt_reference()
    check("T20 in [Eyring-25%%, Sabine+25%%]",
          eyr * 0.75 < t20 < sab * 1.25,
          "T20=%.2f (Sabine %.2f / Eyring %.2f)" % (t20, sab, eyr))
    check("directional deposits present", float(np.abs(dirs).sum()) > 0)
    return hist


def test_native_tracer_rt():
    print("[stochastic] RT box 10x7x3 (native)")
    if not native_available():
        print("  --   native unavailable, skipped")
        return None
    from auralis.engine.native import NativeEngine
    scene = make_shoebox((10.0, 7.0, 3.0),
                         {"absorption": np.full(N_BANDS, 0.15, np.float32),
                          "scattering": np.full(N_BANDS, 0.3, np.float32),
                          "transmission": np.zeros(N_BANDS, np.float32)})
    eng = NativeEngine(scene)
    hist, dirs, st = eng.trace((2, 2, 1.5), (7.5, 5, 1.6), 60000, 200, 2.2,
                               0.002, 1100, 343.0, np.zeros(N_BANDS),
                               spec_exclude_order=0, energy_cap=1e-6, seed=11)
    t20 = rt_from_hist(hist, 0.002)
    sab, eyr = _shoebox_rt_reference()
    check("native T20 in [Eyring-25%%, Sabine+25%%]",
          eyr * 0.75 < t20 < sab * 1.25,
          "T20=%.2f (Sabine %.2f / Eyring %.2f)" % (t20, sab, eyr))
    # permeation via native
    tr, n = eng.segment_transmission((1, 1, 1), (9, 6, 2))
    check("interior segment is clear", n == 0 and abs(tr[0] - 1) < 1e-6)
    eng.close()
    return hist


def test_cross_consistency(hp, hn):
    if hp is None or hn is None:
        return
    print("[cross] Python vs native")
    ep = hp.sum()
    en = hn.sum()
    check("total energy within 20%", abs(ep - en) / max(en, 1e-12) < 0.20,
          "py=%.3f nat=%.3f" % (ep, en))
    tp = rt_from_hist(hp, 0.002)
    tn = rt_from_hist(hn, 0.002)
    check("T20 within 15%", abs(tp - tn) / tn < 0.15,
          "py=%.2f nat=%.2f" % (tp, tn))


def test_wav_roundtrip():
    print("[wavio] round trip")
    fs = 48000
    x = make_test_signal(fs, "clicks", 0.5)
    st = np.stack([x, -0.5 * x], axis=1)
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "f32.wav")
        write_wav(p1, st, fs, float32=True)
        y, fs2 = read_wav(p1)
        check("float32 exact", fs2 == fs and y.shape == st.shape
              and float(np.abs(y - st).max()) < 1e-6)
        p2 = os.path.join(td, "i16.wav")
        write_wav(p2, st, fs, float32=False)
        y, _ = read_wav(p2)
        check("pcm16 ~1e-4", float(np.abs(y - st).max()) < 1e-3)


def test_convolution_identity():
    print("[convolve] Dirac identity")
    fs = 48000
    x = make_test_signal(fs, "speechish", 0.6)
    h = np.zeros(256)
    h[0] = 1.0
    y = fft_convolve(x, h)
    check("delta * x = x", float(np.abs(y[:len(x), 0] - x).max()) < 1e-9)
    h2 = np.zeros((256, 2))
    h2[10, 0] = 1.0
    h2[20, 1] = 0.5
    y2 = fft_convolve(x, h2)
    check("stereo delays", abs(y2[10 + 100, 0] - x[100]) < 1e-9
          and abs(y2[20 + 100, 1] - 0.5 * x[100]) < 1e-9)


def test_full_bake():
    print("[pipeline] full bake (%s engine)" %
          ("native" if native_available() else "python"))
    scene = make_shoebox((9.0, 6.0, 3.2), "plaster", "parquet", "drywall")
    cfg = BakeConfig(n_rays=40000 if native_available() else 4000,
                     max_bounces=150, ism_order=2, bin_dt=0.001,
                     fs=48000, seed=3)
    res = bake(scene,
               SourceDef("src", np.array([2.0, 1.5, 1.4])),
               ReceiverDef("rcv", np.array([6.5, 4.5, 1.6]),
                           forward=np.array([-1.0, 0.0, 0.0]),
                           up=np.array([0.0, 0.0, 1.0])),
               cfg)
    check("finite stereo IR", res.ir.shape[1] == 2
          and np.isfinite(res.ir).all() and float(np.abs(res.ir).max()) > 0)
    rt = res.metrics["per_band"]["1k"]["RT"]
    check("RT 1 kHz plausible", rt is not None and 0.2 < rt < 3.0,
          "RT=%.2f" % (rt or -1))
    check("direct + reflections", len(res.events) >= 7,
          "%d events" % len(res.events))
    d = float(np.linalg.norm(np.array([6.5, 4.5, 1.6]) - np.array([2.0, 1.5, 1.4])))
    check("first peak ~ direct",
          abs(res.events[0].time - d / res.metrics["speed_of_sound"]) < 1e-6)
    # mono synthesis energy sanity: direct band energy ~ 1/d^2 within noise
    mono = synthesize_mono(res.events[:1], np.zeros_like(res.hist),
                           cfg.bin_dt, cfg.fs)
    check("finite mono synthesis", np.isfinite(mono).all())
    return res


def test_events_hist():
    print("[metrics] events -> hist")
    from auralis.engine.ism import AcousticEvent
    ev = AcousticEvent(0.0105, np.full(N_BANDS, 0.5), np.array([0, 1, 0.0]),
                       0, "direct")
    h = events_to_hist([ev], 100, 0.001, N_BANDS)
    check("bin correct", h[3, 10] == 0.25 and h.sum() == 0.25 * N_BANDS)


def test_wav_float32_headroom():
    """A float32 WAV keeps |x| > 1 (no hidden clipping); int16 clips
    by nature."""
    import tempfile, os
    from auralis.engine.wavio import read_wav, write_wav
    x = np.array([0.5, -2.5, 3.0, -0.1])
    with tempfile.TemporaryDirectory() as d:
        pf = os.path.join(d, "f.wav")
        write_wav(pf, x, 44100, float32=True)
        y, _ = read_wav(pf)
        check("float32 wav keeps |x|>1",
              abs(float(np.abs(y).max()) - 3.0) < 1e-5)
        pi = os.path.join(d, "i.wav")
        write_wav(pi, x, 44100, float32=False)
        z, _ = read_wav(pi)
        check("int16 wav clips at 1.0",
              abs(float(np.abs(z).max()) - 1.0) < 1e-3)


if __name__ == "__main__":
    test_wav_float32_headroom()
    print("Auralis - engine test suite")
    print("native available:", native_available())
    test_air()
    test_bands()
    test_direct_calibration()
    test_image_sources()
    hp = test_python_tracer_rt()
    hn = test_native_tracer_rt()
    test_cross_consistency(hp, hn)
    test_wav_roundtrip()
    test_convolution_identity()
    test_events_hist()
    test_full_bake()
    print("\n%d ok, %d failures" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
