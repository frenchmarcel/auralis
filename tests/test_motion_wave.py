# -*- coding: utf-8 -*-
"""Wave band of animated renders: with nothing moving it reproduces the static
hybrid IR below the crossover; source and receiver both moving go through the receiver's
key frames."""

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine.fdtd import design_crossover  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402
from auralis.engine.motion import (MotionConfig, MovingReceiver, MovingSource,  # noqa: E402
                                   Trajectory, render_scene_moving, wave_band_keys)
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402

FS, FC = 8000, 200.0


def _cfg():
    return BakeConfig(fs=FS, n_rays=4000, ism_order=1, fdtd=True, fdtd_fcross=FC,
                      normalize=False, fdtd_conformal=False, seed=3, max_time=0.4,
                      diffraction=False)


def test_static_scene_matches_the_static_hybrid_below_the_crossover():
    sc = make_shoebox((3.2, 2.6, 2.2), wall_material="plaster")
    S, R = np.array([0.8, 0.9, 1.3]), np.array([2.4, 1.8, 1.1])
    fwd, up = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    cfg = _cfg()
    st = bake(sc, SourceDef(name="s", position=S),
              ReceiverDef(name="r", position=R, forward=fwd, up=up), cfg)
    k0 = int(0.05 * FS)
    dry = np.zeros(int(0.3 * FS))
    dry[k0] = 1.0
    ms = MovingSource(name="s", trajectory=Trajectory.constant(S))
    mr = MovingReceiver(name="r", trajectory=Trajectory.constant(R, forward=fwd, up=up))
    y, info = render_scene_moving(sc, [(ms, dry)], mr,
                                  MotionConfig(base=dataclasses.replace(cfg, fdtd=False),
                                               wave=True))
    assert info["wave"]["animated"]
    lp, _hp, dl = design_crossover(FC / 2, FS)

    def low(x):
        return np.stack([np.convolve(x[:, c], lp)[dl:dl + len(x)] for c in range(2)], 1)
    ref = np.zeros_like(y)
    n = min(len(st.ir), len(ref) - k0)
    ref[k0:k0 + n] = st.ir[:n]
    a, b = low(y), low(ref)
    assert np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)) < 0.03


def test_both_moving_runs_at_the_receiver_keys(monkeypatch):
    """Both moving: at a receiver key frame, the source key at the same time is exactly the
    run made at that key (weight 1), and the keys follow both paths."""
    from auralis.engine.motion import _ears_at
    from auralis.engine.fdtd import run_low_band
    from auralis.engine.pipeline import low_band_setup, wave_ear_half
    sc = make_shoebox((3.2, 2.6, 2.2), wall_material="plaster")
    ms = MovingSource(name="s", trajectory=Trajectory(times=[0, 2], positions=[[0.8, 0.9, 1.3],
                                                                               [0.8, 1.8, 1.3]]))
    mr = MovingReceiver(name="r", trajectory=Trajectory(times=[0, 2], positions=[[2.4, 1.0, 1.1],
                                                                                 [2.4, 1.8, 1.1]]))
    cfg = dataclasses.replace(_cfg(), engine="native")
    mc = MotionConfig(base=cfg, wave=True, wave_dt=1.0)
    lows, note = wave_band_keys(sc, [ms], mr, mc, [2.0])
    assert lows is not None and "both move" in note
    key_t, irs, fc, fs_low = lows[0]
    rk = np.linspace(0.0, 2.0, 3)
    assert all(np.any(np.isclose(key_t, t)) for t in rk)
    c, duration, fc0 = low_band_setup(sc, cfg)
    t = rk[1]
    ear = _ears_at(mr.trajectory, t, wave_ear_half(cfg))[0, 0]
    ref, _i = run_low_band(sc, ear, [ms.trajectory.position(t)], fs=fs_low, duration=duration,
                           f_cross=fc0, speed_of_sound=c, engine="native",
                           max_cells=cfg.fdtd_max_cells, order=int(cfg.fdtd_order),
                           boundary_absorber=cfg.fdtd_absorber, conformal=False)
    j = int(np.argmin(np.abs(key_t - t)))
    got = irs[j, :, 0]
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    assert np.max(np.abs(got - ref)) <= 1e-6 * np.max(np.abs(ref))
    # a capped run count spreads the receiver keys (loudly); the solver is not needed for it
    import auralis.engine.fdtd as F
    monkeypatch.setattr(F, "run_low_band",
                        lambda scene, s, r, **kw: (np.zeros((64, len(r))), None))
    lows2, note2 = wave_band_keys(sc, [ms], mr, dataclasses.replace(mc, wave_dt=0.1,
                                                                     wave_max_runs=6), [2.0])
    assert "3 receiver key frames (6 runs)" in note2


def _walk_render(wave_dt, align):
    """A listener walking 2.8 m straight away from a talking source over a ground patch in
    free field (the worst case for the cross-fade: the path grows 1:1); the low band."""
    from auralis.engine.geometry import SceneData
    g = np.array([[[-2, -6, 0], [14, -6, 0], [14, 6, 0]], [[-2, -6, 0], [14, 6, 0], [-2, 6, 0]]],
                 float)
    mat = np.full((1, 7), 0.05)
    sc = SceneData(g, np.zeros(2, np.int32), mat, np.full((1, 7), 0.1), np.zeros((1, 7)))
    S = np.array([0.0, 1.5, 1.3])
    fwd, up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    ms = MovingSource(name="s", trajectory=Trajectory.constant(S))
    mr = MovingReceiver(name="r", trajectory=Trajectory(
        times=[0.0, 2.8], positions=[[0.9, 1.5, 1.2], [3.7, 1.5, 1.2]], forwards=[fwd, fwd],
        ups=[up, up]))
    dry = np.random.default_rng(7).standard_normal(int(2.8 * FS))
    cfg = dataclasses.replace(_cfg(), fdtd=False, n_rays=2000)       # GPU when there is one
    y, info = render_scene_moving(sc, [(ms, dry)], mr,
                                  MotionConfig(base=cfg, wave=True, wave_dt=wave_dt,
                                               wave_align=align))
    assert info["wave"]["animated"]
    lp, _hp, dl = design_crossover(FC / 2, FS)
    return np.stack([np.convolve(y[:, c], lp)[dl:dl + len(y)] for c in range(2)], 1)


def test_aligned_cross_fade_in_free_field():
    """Keys one wavelength apart, walking straight away from the source in free field: the
    variable delay line aligned on the direct sound cuts the error of the plain cross-fade
    (6.8 % -> 1.6 % measured); in reflective scenes it does not (wave_align is off)."""
    lam = 343.0 / FC
    fine = _walk_render(lam / 16, False)
    e = {a: np.sum((_walk_render(lam, a) - fine) ** 2) / np.sum(fine ** 2) for a in (False, True)}
    assert e[True] < 0.5 * e[False]
