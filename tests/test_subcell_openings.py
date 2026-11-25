# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import fdtd as F  # noqa: E402
from auralis.engine import subcell  # noqa: E402
from auralis.engine.geometry import add_box, make_shoebox  # noqa: E402

DX = 0.1
FC = 343.0 / (9.0 * 1.5 * DX)          # wave_dx(FC) = DX
SRC, RCV = np.array([0.5, 0.4, 0.4]), np.array([1.7, 0.4, 0.4])


def _rooms(gap, thick=0.2, alpha=0.05, x0=1.0):
    """Two rooms 0.8 m wide and high behind a partition (x x0 to x0 + thick; x0 = 1: two of
    1 m) with a full-height slot of width `gap` (0: none) at y = 0.4."""
    mat = {"absorption": np.full(7, alpha), "scattering": np.full(7, 0.1),
           "transmission": np.zeros(7)}
    sc = make_shoebox((2.0 + thick, 0.8, 0.8), wall_material=mat, floor_material=mat,
                      ceiling_material=mat)
    y0 = 0.4 - 0.5 * gap
    sc = add_box(sc, (thick, y0, 0.8), (x0, 0.0, 0.0), mat)
    return add_box(sc, (thick, 0.8 - y0 - gap, 0.8), (x0, y0 + gap, 0.0), mat)


def test_a_slot_narrower_than_the_cell_joins_the_rooms_with_its_conductance():
    sc = _rooms(0.04)
    vox = F.voxelize(sc, DX, SRC)
    assert not vox.air[vox.index_of(RCV)]                     # the voxel rule closes the slot
    nv, links, st = subcell.open_pockets(sc, vox)
    assert nv is not vox and not vox.air[vox.index_of(RCV)]   # the cached grid is untouched
    assert nv.air[nv.index_of(RCV)] and len(st["opened"]) == 1
    w = st["opened"][0]["w"]
    # g H / (L dx) = 0.04 * 0.8 / (0.2 * 0.1) = 1.6 faces, less the spreading at both mouths
    assert 1.0 < w < 1.8, w
    assert abs(links[2].sum() - w) < 1e-9 * w
    assert st["opened"][0]["flux_balance"] < 1e-2
    # each part of the slot feeds the facing part of the other mouth: along the slot's height
    # (z) the links join cells about one cell apart at most, never the top to the bottom
    za = np.unravel_index(links[0], nv.shape)[2]
    zb = np.unravel_index(links[1], nv.shape)[2]
    assert np.sum(links[2] * np.abs(za - zb)) / w < 1.0


def test_a_wall_without_gap_stays_closed():
    sc = _rooms(0.0)
    vox = F.voxelize(sc, DX, SRC)
    nv, links, st = subcell.open_pockets(sc, vox)
    assert links is None and nv is vox and st["n_pockets"] >= 1 and not st["opened"]


def test_the_rooms_resonate_through_the_slot_at_the_helmholtz_frequency():
    """Two closed rooms joined by a neck: f = c / (2 pi) sqrt(G (1/V1 + 1/V2)) with
    G = g H / (L + 2 d), d the end correction of a slit of width g in a wall W wide,
    (g / pi)(1 - ln(pi g / (2 W))). A 3 cm slit through 0.5 m of wall, 4 cm cells: the
    links carry the slit's inertance, the grid its radiation at both mouths."""
    g, L, W, H, V = 0.03, 0.5, 0.8, 0.8, 0.64
    sc = _rooms(g, thick=L, alpha=0.02)
    rcv = np.array([1.0 + L + 0.5, 0.4, 0.4])
    kw = dict(fs=2000, duration=4.0, f_cross=FC, speed_of_sound=343.0, engine="auto")
    os.environ["AURALIS_FDTD_SUBCELL"] = "0"
    try:
        with pytest.raises(ValueError):
            F.run_low_band(sc, SRC, rcv, **kw)
    finally:
        os.environ.pop("AURALIS_FDTD_SUBCELL", None)
    os.environ["AURALIS_FDTD_SUBSONIC_HZ"] = "0"
    try:
        ir, _info = F.run_low_band(sc, SRC, rcv, **kw)
    finally:
        os.environ.pop("AURALIS_FDTD_SUBSONIC_HZ", None)
    assert np.all(np.isfinite(ir))
    n = len(ir)
    spec = np.abs(np.fft.rfft(ir, 16 * n))
    f = np.fft.rfftfreq(16 * n, 1.0 / kw["fs"])
    k = (f > 5.0) & (f < 120.0)
    f_peak = f[k][np.argmax(spec[k])]
    d = (g / np.pi) * (1.0 - np.log(np.pi * g / (2.0 * W)))
    f_h = 343.0 / (2.0 * np.pi) * np.sqrt(g * H / (L + 2.0 * d) * 2.0 / V)
    assert abs(f_peak / f_h - 1.0) < 0.05, (f_peak, f_h)


@pytest.mark.parametrize("x0", [1.0, 0.7], ids=["equal", "unequal"])
def test_a_source_on_the_other_side_gets_the_same_links(x0):
    """The joined air seen from either room: the same links, computed once. Also for rooms of
    unequal volume, where the pocket threshold (1 % of the source's air) differs by side."""
    sc = _rooms(0.04, x0=x0)
    subcell._CACHE.clear()
    subcell._CACHE_JOINED.clear()
    va = F.voxelize(sc, DX, SRC)
    na, la, _sa = subcell.open_pockets(sc, va)
    vb = F.voxelize(sc, DX, RCV)
    nb, lb, _sb = subcell.open_pockets(sc, vb)
    assert np.array_equal(na.air, nb.air)
    assert all(np.array_equal(x, y) for x, y in zip(la, lb))
