# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine import ambidecode as A  # noqa: E402
from auralis.engine.ir_synthesis import sh_sn3d  # noqa: E402


def test_harmonics_up_to_order_5_are_sn3d():
    """SN3D: mean of Y_i Y_j over the sphere = delta_ij / (2l + 1); orders 0-2 keep their
    explicit formulas, order 3 matches the AmbiX table, a rotation is exact at order 5."""
    d = A.virtual_layout(40000)
    Y = sh_sn3d(d, 5)
    deg = np.floor(np.sqrt(np.arange(36))).astype(int)
    assert np.abs(Y.T @ Y / len(d) - np.diag(1.0 / (2 * deg + 1))).max() < 1e-5
    assert np.array_equal(Y[:, :9], sh_sn3d(d, 2))
    x, y, z = d.T
    ref = [np.sqrt(5 / 8) * y * (3 * x * x - y * y), np.sqrt(15) * x * y * z,
           np.sqrt(3 / 8) * y * (5 * z * z - 1), 0.5 * z * (5 * z * z - 3),
           np.sqrt(3 / 8) * x * (5 * z * z - 1), np.sqrt(15) / 2 * z * (x * x - y * y),
           np.sqrt(5 / 8) * x * (x * x - 3 * y * y)]
    assert max(np.abs(r - Y[:, 9 + k]).max() for k, r in enumerate(ref)) < 1e-12
    M = A.rotation_matrix(5, 37.0, -12.0, 5.0)
    a, b, c = np.radians([37.0, -12.0, 5.0])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    u = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
    assert np.allclose(M @ sh_sn3d(u[None], 5)[0], sh_sn3d((rz @ ry @ rx @ u)[None], 5)[0],
                       atol=1e-12)
    with pytest.raises(ValueError):
        sh_sn3d(d[:3], 6)


def _ild(order, az, bands):
    fs = 48000
    d = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
    F = A.decoder_filters(order, fs)[0]
    imp = np.zeros((64, F.shape[0]))
    imp[0] = sh_sn3d(d[None], order)[0]
    Y = np.abs(np.fft.rfft(A.decode(imp, F), 4096, axis=0)) ** 2
    H = np.abs(np.fft.rfft(A.hrirs(d[None], fs)[0][0], 4096, axis=-1)) ** 2
    fr = np.fft.rfftfreq(4096, 1 / fs)
    out = []
    for lo in bands:
        k = (fr >= lo) & (fr < 2 * lo)
        out.append((10 * np.log10(Y[k, 0].sum() / Y[k, 1].sum()),
                    10 * np.log10(H[0, k].sum() / H[1, k].sum())))
    return out


@pytest.mark.parametrize("order", [3, 4, 5])
def test_higher_orders_keep_the_heads_level_differences_further_up(order):
    """The head model's interaural level difference within 0.25 dB from 100 to 800 Hz at
    orders 3-5, and closer than at order 2 in the 800-1600 Hz octave (45 deg)."""
    for got, want in _ild(order, 45.0, [100, 200, 400]):
        assert abs(got - want) < 0.25, (got, want)
    (g_n, w), = _ild(order, 45.0, [800])
    (g_2, _w), = _ild(2, 45.0, [800])
    assert abs(g_n - w) < abs(g_2 - w)


def test_bake_at_order_3_keeps_ws_energy_and_adds_the_third_degree():
    from auralis.engine import make_shoebox
    from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake
    sc = make_shoebox((5.0, 4.0, 2.8), wall_material="plaster")
    src = SourceDef("S", np.array([1.2, 1.0, 1.4]))
    rcv = ReceiverDef("R", np.array([3.6, 2.7, 1.2]))
    out = {o: bake(sc, src, rcv, BakeConfig(n_rays=4000, seed=5, max_time=0.4,
                                            ambisonic_order=o)) for o in (2, 3)}
    a2, a3 = out[2].ambi, out[3].ambi
    assert a3.shape[1] == 16 and np.abs(a3[:, 9:]).max() > 0
    # the late noise is drawn for every channel at once (another realisation per order); W
    # keeps the echogram's energy up to that noise (smoothed bins: ~0.8 dB at 125 Hz over 0.4 s)
    from auralis.engine import rirmetrics as rm
    e2 = (rm.band_split(a2[:, 0], 48000) ** 2).sum(axis=1)
    e3 = (rm.band_split(a3[:, 0], 48000) ** 2).sum(axis=1)
    d = 10 * np.log10(e3 / e2)
    assert np.abs(d).max() < 1.2 and np.abs(d[2:]).max() < 0.4, d
    # a degree-3 channel of an isotropic late field carries 1/7 of W's energy
    late = slice(int(0.15 * 48000), int(0.35 * 48000))
    r = (a3[late, 9:16] ** 2).sum() / 7 / (a3[late, 0] ** 2).sum()
    assert 0.08 < r < 0.2, r


def _errors(F, test, fs=48000):
    n = 8192
    fr = np.fft.rfftfreq(n, 1 / fs)
    href, _ = A.hrirs(test, fs, n_taps=F.shape[-1])
    Y = sh_sn3d(test, int(np.sqrt(F.shape[0])) - 1)
    D = np.abs(np.fft.rfft(np.einsum("dc,cen->den", Y, F), n, axis=-1)) ** 2
    R = np.abs(np.fft.rfft(href, n, axis=-1)) ** 2
    lev, ild = [], []
    for lo in (250, 2000, 4000, 8000):
        k = (fr >= lo / 2 ** 0.5) & (fr < lo * 2 ** 0.5)
        ed, er = D[..., k].sum(-1), R[..., k].sum(-1)
        lev.append(np.mean(np.abs(10 * np.log10(ed / er))))
        ild.append(np.mean(np.abs(10 * np.log10(ed[:, 0] / ed[:, 1] * er[:, 1] / er[:, 0]))))
    return np.array(lev), np.array(ild)


def test_magls_decoder_keeps_the_heads_magnitudes_above_the_cutoff():
    """MagLS (magnitude least squares above N c / 2 pi a, the latency taken out): the same as the
    dual-band decoder at 250 Hz, and within 0.6 dB of the head's levels and level differences
    at 2-8 kHz over 300 test directions, where the dual-band decoder is 2-6 dB off."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal((300, 3))
    test = v / np.linalg.norm(v, axis=1, keepdims=True)
    lev_d, ild_d = _errors(A.decoder_filters(2, 48000, method="dual")[0], test)
    lev_m, ild_m = _errors(A.decoder_filters(2, 48000, method="magls")[0], test)
    assert abs(lev_m[0] - lev_d[0]) < 0.05 and abs(ild_m[0] - ild_d[0]) < 0.05
    assert (lev_m[1:] < 0.6).all() and (ild_m[1:] < 0.6).all(), (lev_m, ild_m)
    assert (lev_d[1:] > 2.0).all(), lev_d
    with pytest.raises(ValueError):
        A.decoder_filters(2, 48000, method="nope")
