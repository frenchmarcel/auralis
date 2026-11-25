# -*- coding: utf-8 -*-
import numpy as np

from auralis.engine import make_shoebox
from auralis.engine.grid import grid_metrics
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake


def test_map_levels_follow_the_hybrid_ir_like_a_point_receiver():
    sc = make_shoebox((3.4, 2.6, 2.3), wall_material="plaster", floor_material="carpet")
    cfg = BakeConfig(fs=8000, n_rays=4000, seed=5, fdtd=True, fdtd_fcross=200.0,
                     engine="python", output_layout="stereo", max_time=0.4)
    src = SourceDef("S", np.array([0.9, 0.8, 1.3]), gain_db=3.0)
    pts = np.array([[2.4, 1.7, 1.1], [1.9, 0.9, 1.6]])
    g = grid_metrics(sc, src, pts, cfg)
    assert g["wave_fc"] == 200.0
    geo = grid_metrics(sc, src, pts, BakeConfig(**{**cfg.__dict__, "fdtd": False}))
    for i, p in enumerate(pts):
        res = bake(sc, src, ReceiverDef("R", p), cfg)
        assert np.allclose(g["G_db"][i], res.metrics["g_bands"], atol=1e-6)
        assert np.allclose(g["SPL_db"][i], res.metrics["spl_bands"], atol=1e-6)
        shift = np.asarray(res.metrics["wave_level_shift_db"])
        assert np.allclose(g["G_db"][i] - geo["G_db"][i], shift, atol=1e-6)
        assert np.abs(shift[:2]).max() > 0.05 and np.all(shift[3:] == 0.0)
        assert np.allclose(g["T30"][i], geo["T30"][i], equal_nan=True)   # decays unchanged
