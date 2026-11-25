# -*- coding: utf-8 -*-
"""Directional sources in the animated wave band: probes at +-h/2 along the source's axis
stand for the static grid dipole (reciprocity). A source that barely moves gives the static
directional IR below the crossover in every motion branch. An omni wave band does not."""

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine.fdtd import design_crossover  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402
from auralis.engine.motion import (MotionConfig, MovingReceiver, MovingSource,  # noqa: E402
                                   Trajectory, render_scene_moving)
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402

FS, FC = 8000, 200.0
S, R = np.array([0.8, 0.9, 1.3]), np.array([2.4, 1.8, 1.1])
FWD_S = np.array([-0.6, -0.8, 0.0])          # the cardioid faces away from the listener
FWD_R, UP = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])


def _cfg(**kw):
    return BakeConfig(fs=FS, n_rays=4000, ism_order=1, fdtd=True, fdtd_fcross=FC,
                      normalize=False, fdtd_conformal=False, seed=3, max_time=0.4,
                      diffraction=False, engine="native", **kw)


def _crawl(p, fwd=None):
    """A path that moves 0.1 mm in 0.3 s: not static for the renderer, static for the field."""
    t = np.array([0.0, 0.3])
    pos = np.stack([p, p + [1e-4, 0.0, 0.0]])
    f = None if fwd is None else np.stack([fwd, fwd])
    return Trajectory(times=t, positions=pos, forwards=f,
                      ups=None if fwd is None else np.stack([UP, UP]))


def _low(x):
    lp, _hp, dl = design_crossover(FC / 2, FS)
    return np.stack([np.convolve(x[:, c], lp)[dl:dl + len(x)] for c in range(2)], 1)


@pytest.fixture(scope="module")
def static_ref():
    sc = make_shoebox((3.2, 2.6, 2.2), wall_material="plaster")
    st = bake(sc, SourceDef(name="s", position=S, directivity="cardioid", forward=FWD_S),
              ReceiverDef(name="r", position=R, forward=FWD_R, up=UP), _cfg())
    return sc, st


def _render(sc, src_traj, rcv_traj, **cfg_kw):
    k0 = int(0.05 * FS)
    dry = np.zeros(int(0.3 * FS))
    dry[k0] = 1.0
    ms = MovingSource(name="s", trajectory=src_traj, directivity="cardioid")
    mr = MovingReceiver(name="r", trajectory=rcv_traj)
    y, info = render_scene_moving(sc, [(ms, dry)], mr, MotionConfig(
        base=dataclasses.replace(_cfg(**cfg_kw), fdtd=False), wave=True))
    assert info["wave"]["animated"]
    return y, k0


def _err(y, k0, st):
    ref = np.zeros_like(y)
    n = min(len(st.ir), len(ref) - k0)
    ref[k0:k0 + n] = st.ir[:n]
    a, b = _low(y), _low(ref)
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))


@pytest.mark.parametrize("branch", ["receiver static", "sources static", "both move"])
def test_directional_moving_source_matches_the_static_one(static_ref, branch):
    sc, st = static_ref
    src = Trajectory.constant(S, forward=FWD_S) if branch == "sources static" else _crawl(S, FWD_S)
    rcv = (Trajectory.constant(R, forward=FWD_R, up=UP) if branch == "receiver static"
           else _crawl(R, FWD_R))
    y, k0 = _render(sc, src, rcv)
    assert _err(y, k0, st) < 0.03


def test_the_omni_wave_band_was_wrong_for_a_directional_source(static_ref):
    sc, st = static_ref
    y, k0 = _render(sc, _crawl(S, FWD_S), Trajectory.constant(R, forward=FWD_R, up=UP),
                    fdtd_source_directivity=False)
    assert _err(y, k0, st) > 0.10
