# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.fdtd import run_low_band  # noqa: E402
from auralis.engine.geometry import SceneData, box_triangles, get_preset  # noqa: E402

L = (3.0, 2.2, 1.9)
FS, C = 8000, 343.0


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("AURALIS_GPU", "0")


def _box(panel=True):
    tris, names = box_triangles(L, inward=True)
    wall = get_preset("plaster")
    tm = np.array([1 if (panel and n == "wall_x0") else 0 for n in names], dtype=np.int32)
    return SceneData.from_materials(tris, tm, [wall, wall], ["walls", "panel"])


def _spec(ir):
    n = 1 << 16
    return np.fft.rfftfreq(n, 1.0 / FS), np.fft.rfft(np.asarray(ir), n, axis=0)


def test_a_point_source_near_a_wall_keeps_its_volume_velocity():
    """Below the first mode a monopole fills the box whatever its position: 6 cm and 10 cm
    from a wall (in the conformal cut layer, phi up to 1.45) give the level of a source in
    the interior within 0.3 dB. Taps not divided by phi would give +1.1 and +3.4 dB."""
    sc = _box(panel=False)
    lv = []
    for x in (0.06, 0.10, 0.38):
        ir, _ = run_low_band(sc, np.array([x, 1.1, 0.95]), [np.array([2.1, 0.9, 1.1])], fs=FS,
                             duration=2.0, f_cross=400.0, speed_of_sound=C, engine="native")
        f, P = _spec(ir)
        k = (f > 25) & (f < 40)
        lv.append(10 * np.log10((np.abs(P[k, 0]) ** 2).mean()))
    assert max(lv) - min(lv) < 0.3, lv


def test_a_vibrating_wall_drives_the_axial_field_of_a_piston():
    """The whole x = 0 wall of a box vibrating uniformly: a 1-D field (the same complex
    pressure across y and z) at the level of the piston solution
    p = 4 pi / (k A) cos(k (L - x)) / sin(k L) for the volume velocity of the unit point source
    (0.5 dB below the first mode); a point source in a corner gives no such field."""
    sc = _box()
    rc = [np.array([1.0, 0.7, 0.6]), np.array([1.0, 1.5, 1.3]), np.array([2.1, 0.9, 1.1])]
    out = {}
    for surf, sp in (("panel", (0.3, 1.1, 0.95)), (None, (0.3, 0.4, 0.5))):
        ir, _ = run_low_band(sc, np.array(sp), rc, fs=FS, duration=2.0, f_cross=400.0,
                             speed_of_sound=C, engine="native", src_surface=surf,
                             conformal_aniso=True)
        out[surf] = _spec(ir)
    f, P = out["panel"]
    for lo, hi, tol in ((30, 120, 0.02), (120, 250, 0.06)):
        b = (f >= lo) & (f < hi)
        e = (np.abs(P[b, 0] - P[b, 1]) ** 2).sum() / (np.abs(P[b, 0]) ** 2).sum()
        assert e < tol, (lo, hi, e)
    _f, Q = out[None]
    b = (f >= 30) & (f < 120)
    assert (np.abs(Q[b, 0] - Q[b, 1]) ** 2).sum() / (np.abs(Q[b, 0]) ** 2).sum() > 0.5
    A = L[1] * L[2]
    for f0 in (30.0, 40.0, 45.0):
        kk = int(np.argmin(np.abs(f - f0)))
        k = 2 * np.pi * f[kk] / C
        ana = 4 * np.pi / (k * A) * abs(np.cos(k * (L[0] - 2.1)) / np.sin(k * L[0]))
        assert abs(20 * np.log10(abs(P[kk, 2]) / ana)) < 0.5, (f0, abs(P[kk, 2]), ana)


def test_a_vibrating_surface_is_wave_only():
    from auralis.engine.pipeline import (BakeConfig, ReceiverDef, SourceDef, bake,
                                         precompute_low_bands)
    sc = _box()
    src = SourceDef("wall", np.array([0.5, 1.1, 0.95]), surface="panel")
    rcv = ReceiverDef("R", np.array([2.1, 0.9, 1.1]))
    with pytest.raises(ValueError):
        bake(sc, src, rcv, BakeConfig(n_rays=2000, fdtd=True, max_time=0.3))
    cfg = BakeConfig(n_rays=2000, seed=3, normalize=False, fdtd=True, fdtd_fcross=300.0,
                     max_time=0.5, wave_only=True, ambisonic_order=1)
    low = precompute_low_bands(sc, src, [rcv], cfg)["R"]
    res = bake(sc, src, rcv, cfg, low_band=low)
    assert res.metrics["wave_only"] and np.abs(res.ambi[:, 0]).max() > 0
