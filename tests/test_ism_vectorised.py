# -*- coding: utf-8 -*-
"""Anchor of the vectorised image-source method: it must return EXACTLY the
events of the per-candidate reference (same set, same order, same bits), with
occluders, split patches and the complex-reflection table, orders 1-3."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import make_shoebox  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, speed_of_sound  # noqa: E402
from auralis.engine.geometry import NumpyRaycaster, add_box, extract_planes  # noqa: E402
from auralis.engine.ism import (compute_image_sources,  # noqa: E402
                                compute_image_sources_reference)
from auralis.engine.reflection import build_admittance_table  # noqa: E402


def _same(a, b):
    return len(a) == len(b) and all(
        x.time == y.time and np.array_equal(x.amps, y.amps)
        and np.array_equal(x.direction, y.direction) and x.order == y.order
        and np.array_equal(x.emit_dir, y.emit_dir) and x.bounces == y.bounces
        for x, y in zip(a, b))


def _tracers(scene):
    out = [NumpyRaycaster(scene)]
    try:
        from auralis.engine.native import NativeEngine, is_available
        if is_available():
            out.append(NativeEngine(scene))
    except Exception:
        pass
    return out


def test_vectorised_ism_equals_the_reference():
    sc = make_shoebox((9.0, 6.0, 3.2), wall_material="plaster", floor_material="carpet")
    sc = add_box(sc, (1.2, 2.5, 1.8), (4.0, 1.5, 0.0), "concrete")   # occluder
    sc = add_box(sc, (0.6, 0.6, 3.2), (2.0, 4.2, 0.0), "wood_panel")  # column
    planes, tp = extract_planes(sc, 48)
    c = speed_of_sound(20.0)
    air = air_attenuation_db_per_m()
    adm = build_admittance_table(sc.absorption, getattr(sc, "mat_phys", None))
    rng = np.random.default_rng(11)
    lo, hi = np.array([0.3, 0.3, 0.3]), np.array([8.7, 5.7, 2.9])
    pts = rng.uniform(lo, hi, size=(40, 3))
    inside_box = ((pts[:, 0] > 3.9) & (pts[:, 0] < 5.3) & (pts[:, 1] > 1.4) & (pts[:, 1] < 4.1)
                  & (pts[:, 2] < 1.9)) | ((pts[:, 0] > 1.9) & (pts[:, 0] < 2.7)
                                          & (pts[:, 1] > 4.1) & (pts[:, 1] < 4.9))
    pts = pts[~inside_box]
    n_events = 0
    for tracer in _tracers(sc):
        for order, step in ((1, 1), (2, 2), (3, 7)):
            for i in range(0, len(pts) - 1, step):
                for table in (None, adm):
                    ref = compute_image_sources_reference(sc, planes, tp, tracer, pts[i],
                                                          pts[i + 1], order, air, c,
                                                          adm_table=table)
                    new = compute_image_sources(sc, planes, tp, tracer, pts[i], pts[i + 1],
                                                order, air, c, adm_table=table)
                    assert _same(ref, new), (type(tracer).__name__, order, i)
                    n_events += len(ref)
        if hasattr(tracer, "close"):
            tracer.close()
    assert n_events > 1000           # the comparison is not vacuous
