# -*- coding: utf-8 -*-
"""A small street of boxes on a ground plane, turned by a given angle (tests helper)."""
import numpy as np

from auralis.engine.geometry import SceneData, add_box

MAT = {"absorption": np.full(7, 0.05), "scattering": np.full(7, 0.1),
       "transmission": np.zeros(7)}


def street(deg):
    g = np.array([[[0, -9, 0], [40, -9, 0], [40, 9, 0]], [[0, -9, 0], [40, 9, 0], [0, 9, 0]]], float)
    sc = SceneData(g, np.zeros(2, np.int32), MAT["absorption"][None], MAT["scattering"][None],
                   MAT["transmission"][None])
    for x in (0.0, 13.0, 27.0):
        sc = add_box(sc, (12.0, 5.0, 8.0), (x, 4.0, 0.0), MAT)
        sc = add_box(sc, (12.0, 5.0, 6.0), (x, -9.0, 0.0), MAT)
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sc.tri_verts = np.ascontiguousarray(sc.tri_verts @ R.T, dtype=np.float32)
    return sc, R @ np.array([8.0, -1.0, 1.2]), [R @ np.array([30.0, 1.0, 1.6])]
