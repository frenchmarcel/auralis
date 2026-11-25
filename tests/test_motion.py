# -*- coding: utf-8 -*-
"""Tests for moving sources (time-varying convolution + Doppler)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import (  # noqa: E402
    BakeConfig, MotionConfig, MovingSource, ReceiverDef, SceneData, SourceDef,
    Trajectory, bake, fft_convolve, make_shoebox, render_moving,
    render_scene_moving,
)
from auralis.engine.air import speed_of_sound  # noqa: E402

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


def test_trajectory():
    tr = Trajectory(times=[0.0, 1.0, 3.0],
                    positions=[[0, 0, 0], [2, 0, 0], [2, 4, 0]],
                    forwards=[[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    p = tr.position(np.array([-1.0, 0.5, 2.0, 9.0]))
    check("interp positions + clamp",
          np.allclose(p, [[0, 0, 0], [1, 0, 0], [2, 2, 0], [2, 4, 0]]))
    f = tr.forward(2.0)
    check("axes renormalized", abs(np.linalg.norm(f) - 1.0) < 1e-12)
    cst = Trajectory.constant((5, 6, 7))
    check("constant", np.allclose(cst.position(123.0), [5, 6, 7]))


def test_static_equivalence():
    """Constant trajectories, doppler OFF, python engine: time-varying
    convolution must reproduce bake(...) + convolution EXACTLY."""
    sc = make_shoebox((8.0, 6.0, 3.0))
    cfg = BakeConfig(n_rays=4000, engine="python", seed=11, normalize=False,
                     max_time=0.7, fs=48000)
    src = SourceDef("s", (2.0, 2.0, 1.5))
    rcv = ReceiverDef("r", (6.0, 4.0, 1.6), forward=(-1.0, 0.0, 0.0))

    res = bake(sc, src, rcv, cfg)
    rng = np.random.default_rng(0)
    dry = rng.standard_normal(48000)             # 1 s of noise
    ref = fft_convolve(dry, res.ir)              # (L, 2)

    mc = MotionConfig(base=cfg, doppler=False, reverb_dt=0.4)
    out, info = render_moving(sc, src, rcv, dry, mc)
    n = min(len(ref), len(out))
    err = float(np.abs(out[:n] - ref[:n]).max())
    peak = float(np.abs(ref).max())
    check("static TVC == bake+convolution (exact)",
          err < 1e-8 * peak, "(rel err %.1e)" % (err / peak))
    check("residual outside support ~ 0",
          float(np.abs(out[n:]).max()) < 1e-10 * peak)


def test_doppler_pitch():
    """Near free field: a source approaching at 30 m/s shifts a 1 kHz
    tone to f * c / (c - v)."""
    tri = np.array([[[-500.0, -500.0, -200.0],
                     [500.0, -500.0, -200.0],
                     [0.0, 500.0, -200.0]]])
    sc = SceneData(tri_verts=tri, tri_mat=np.zeros(1, np.int32),
                   absorption=np.full((1, 7), 0.9, np.float32),
                   scattering=np.full((1, 7), 0.2, np.float32),
                   transmission=np.zeros((1, 7), np.float32),
                   mat_names=["far"])
    fs = 48000
    cfg = BakeConfig(n_rays=64, engine="python", seed=5, normalize=True,
                     max_time=0.5, fs=fs, ism_order=0)
    v = 30.0
    traj = Trajectory(times=[0.0, 3.0],
                      positions=[[110.0, 0.0, 1.7], [110.0 - 3 * v, 0.0, 1.7]])
    msrc = MovingSource("mobile", traj)
    rcv = ReceiverDef("fixed", (0.0, 0.0, 1.7), forward=(1.0, 0.0, 0.0))
    t = np.arange(int(3.0 * fs)) / fs
    dry = np.sin(2 * np.pi * 1000.0 * t)

    out, _ = render_moving(sc, msrc, rcv, dry, MotionConfig(base=cfg))
    mono = out[:, 0] + out[:, 1]
    seg = mono[int(1.0 * fs):int(2.0 * fs)] * np.hanning(fs)
    sp = np.abs(np.fft.rfft(seg))
    k = int(np.argmax(sp))
    # parabolic refinement
    if 0 < k < len(sp) - 1:
        d = 0.5 * (sp[k - 1] - sp[k + 1]) / (sp[k - 1] - 2 * sp[k] + sp[k + 1])
    else:
        d = 0.0
    f_meas = (k + d) * 1.0                       # 1 s -> 1 Hz / bin
    c = speed_of_sound(20.0)
    f_exp = 1000.0 * c / (c - v)
    check("Doppler approaching: %.1f Hz measured vs %.1f expected"
          % (f_meas, f_exp), abs(f_meas - f_exp) < 4.0)

    # receding source
    traj2 = Trajectory(times=[0.0, 3.0],
                       positions=[[20.0, 0.0, 1.7], [20.0 + 3 * v, 0.0, 1.7]])
    out2, _ = render_moving(sc, MovingSource("m2", traj2), rcv, dry,
                            MotionConfig(base=cfg))
    mono2 = out2[:, 0] + out2[:, 1]
    seg2 = mono2[int(1.0 * fs):int(2.0 * fs)] * np.hanning(fs)
    sp2 = np.abs(np.fft.rfft(seg2))
    k2 = int(np.argmax(sp2))
    f_exp2 = 1000.0 * c / (c + v)
    check("Doppler receding: %.1f Hz vs %.1f" % (k2, f_exp2),
          abs(k2 - f_exp2) < 4.0)


def test_pan_and_mix():
    sc = make_shoebox((12.0, 8.0, 3.5), wall_material="brick",
                      floor_material="carpet_heavy",
                      ceiling_material="acoustic_tile")
    fs = 48000
    cfg = BakeConfig(n_rays=2500, engine="auto", seed=3, normalize=True,
                     fs=fs, max_time=0.6)
    dur = 2.5
    traj = Trajectory(times=[0.0, dur],
                      positions=[[1.0, 6.0, 1.8], [11.0, 6.0, 1.8]])
    msrc = MovingSource("crossing", traj)
    rcv = ReceiverDef("aud", (6.0, 4.0, 1.7), forward=(0.0, 1.0, 0.0))
    rng = np.random.default_rng(1)
    dry = rng.standard_normal(int(dur * fs)) * 0.5

    out, info = render_moving(sc, msrc, rcv, dry,
                              MotionConfig(base=cfg, reverb_dt=0.6))
    check("finite output", bool(np.isfinite(out).all()) and
          float(np.abs(out).max()) > 0.01)

    def rms(x):
        return float(np.sqrt(np.mean(x * x)))
    a0, a1 = int(0.2 * fs), int(0.8 * fs)
    b0, b1 = int((dur - 0.8) * fs), int((dur - 0.2) * fs)
    l0, r0 = rms(out[a0:a1, 0]), rms(out[a0:a1, 1])
    l1, r1 = rms(out[b0:b1, 0]), rms(out[b0:b1, 1])
    check("source on the left at start (L>R)", l0 > 1.10 * r0,
          "(%.3f vs %.3f)" % (l0, r0))
    check("source on the right at end (R>L)", r1 > 1.10 * l1,
          "(%.3f vs %.3f)" % (l1, r1))

    # sum of two sources == mixdown
    s_a = SourceDef("a", (2.0, 5.0, 1.5))
    s_b = SourceDef("b", (10.0, 5.0, 1.5))
    d_a = rng.standard_normal(fs)
    d_b = rng.standard_normal(fs)
    cfgN = BakeConfig(**{**cfg.__dict__, "normalize": False})
    mcN = MotionConfig(base=cfgN, reverb_dt=0.5)
    oa, _ = render_moving(sc, s_a, rcv, d_a, mcN)
    ob, _ = render_moving(sc, s_b, rcv, d_b, mcN)
    mix, _ = render_scene_moving(sc, [(s_a, d_a), (s_b, d_b)], rcv, mcN)
    n = min(len(mix), max(len(oa), len(ob)))
    ref = np.zeros((n, 2))
    ref[:len(oa)] += oa[:n]
    ref[:len(ob)] += ob[:n]
    check("multi-source mix == sum",
          np.allclose(mix[:n], ref, atol=1e-10))




def test_itd_sign():
    """The near ear must get the direct sound FIRST: L/R lag from cross
    correlation, compared with the Woodworth ITD."""
    from auralis.engine.hrtf import itd_seconds, receiver_basis
    tri = np.array([[[-500.0, -500.0, -200.0],
                     [500.0, -500.0, -200.0],
                     [0.0, 500.0, -200.0]]])
    sc = SceneData(tri_verts=tri, tri_mat=np.zeros(1, np.int32),
                   absorption=np.full((1, 7), 0.9, np.float32),
                   scattering=np.full((1, 7), 0.2, np.float32),
                   transmission=np.zeros((1, 7), np.float32),
                   mat_names=["far"])
    fs = 48000
    cfg = BakeConfig(n_rays=16, engine="python", seed=2, normalize=False,
                     max_time=0.25, fs=fs, ism_order=0)
    rcv = ReceiverDef("fixed", (0.0, 0.0, 1.7), forward=(1.0, 0.0, 0.0))
    rng = np.random.default_rng(4)
    dry = rng.standard_normal(fs // 2)
    c = speed_of_sound(20.0)
    _f, _u, right = receiver_basis(rcv.forward, rcv.up)

    for y, side in ((-3.0, "right"), (3.0, "left")):
        src = SourceDef("s", (2.0, y, 1.7))
        out, _ = render_moving(sc, src, rcv, dry, MotionConfig(base=cfg))
        L, R = out[:, 0], out[:, 1]
        n = len(L)
        lag_max = 64
        xc = np.correlate(L, R, "full")[n - 1 - lag_max:n + lag_max]
        k = int(np.argmax(xc)) - lag_max      # >0: L lags behind R
        d = np.asarray(src.position, float) - np.asarray(rcv.position, float)
        d /= np.linalg.norm(d)
        tau, near_right = itd_seconds(d, right, c)
        expected = tau * fs * (1.0 if near_right else -1.0)
        check("ITD %s: lag=%+d samples (expected %+.1f)" % (side, k, expected),
              abs(k - expected) < 2.0)


if __name__ == "__main__":
    print("== trajectories ==")
    test_trajectory()
    print("== static equivalence ==")
    test_static_equivalence()
    print("== Doppler ==")
    test_doppler_pitch()
    print("== panning & mixdown ==")
    test_pan_and_mix()
    print("== direct-path ITD ==")
    test_itd_sign()
    print("\n%d ok, %d failures" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
