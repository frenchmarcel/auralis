#!/usr/bin/env python3
"""CR1 room OBJ from the full SketchUp export (skp_to_obj.py without --keep): the official
DoorAngle1 model leaves one concrete wall without material, and paints its two dodecahedron
loudspeakers (at LS1 and LS2) with a copy of paintedConcrete; they are source markers, dropped,
as any small loose object that holds a source position (both files: a 16 cm body around LS1).
Usage: relabel_cr1_obj.py full.obj out.obj"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from skp_to_obj import areas, write_obj  # noqa: E402
from auralis.engine.project import read_obj  # noqa: E402

# wall y = -3.648 m, x in [-6.251, 0]: concrete in the DoorAngle3 model, no material in DoorAngle1
WALL_Y, WALL_X_MAX, TOL = -3.648, 0.01, 2e-3
# the dodecahedron models of the DoorAngle1 file (0.24 m2, 10788 triangles around LS1 and LS2):
# kept, they would enclose the sources (no wave band, rays trapped in the ball)
MARKERS = {"mat_CR1_paintedConcrete1"}
# BRAS source positions (bras_room_manifest.py, CR1): a loose body smaller than MARKER_SIZE whose
# box holds one is the loudspeaker itself (its directivity is the source's), never the room
SOURCES = np.array([[1.5, -2.225, 0.773], [-1.77, -2.28, 0.723]])
MARKER_SIZE = 0.5


def components(faces):
    """Label of the vertex-connected component of each triangle."""
    lab = np.arange(int(faces.max()) + 1)
    while True:
        m = lab[faces].min(axis=1)
        new = lab.copy()
        np.minimum.at(new, faces.ravel(), np.repeat(m, 3))
        new = new[new]
        if np.array_equal(new, lab):
            return lab[faces[:, 0]]
        lab = new


def relabel(verts, faces, layers):
    """{material: (n, 3, 3) triangles} after the CR1 fixes, and a log of what changed."""
    tris = verts[faces]
    lay = np.array(["marker:" + m if m in MARKERS else m for m in layers], dtype=object)
    log = []
    comp = components(faces)
    for c in np.unique(comp):
        sel = comp == c
        lo, hi = tris[sel].reshape(-1, 3).min(axis=0), tris[sel].reshape(-1, 3).max(axis=0)
        holds = ((SOURCES >= lo) & (SOURCES <= hi)).all(axis=1)
        if np.linalg.norm(hi - lo) < MARKER_SIZE and holds.any():
            log.append("loose body of %d triangles (%.3f m2, %s m) around LS%d -> source marker"
                       % (sel.sum(), areas(tris[sel]).sum(), np.round(hi - lo, 3),
                          int(np.argmax(holds)) + 1))
            lay[sel] = ["marker:" + str(m) for m in lay[sel]]
    on_wall = ((np.abs(tris[:, :, 1] - WALL_Y).max(axis=1) < TOL)
               & (tris.mean(axis=1)[:, 0] < WALL_X_MAX) & (lay == "default"))
    if on_wall.any():
        lay[on_wall] = "mat_CR1_concrete"
        log.append("%.2f m2 of the wall y = %.3f m (no material) -> mat_CR1_concrete, as in the "
                   "DoorAngle3 model" % (areas(tris[on_wall]).sum(), WALL_Y))
    keep = np.array([str(m).startswith("mat_CR1_") for m in lay])
    if (~keep).any():
        log.append("dropped %.2f m2 outside mat_CR1_* (source and microphone markers): %s"
                   % (areas(tris[~keep]).sum(), ", ".join(sorted(set(map(str, lay[~keep]))))))
    groups = {m: tris[keep & (lay == m)] for m in sorted(set(map(str, lay[keep])))}
    return groups, log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("full_obj")
    ap.add_argument("out_obj")
    a = ap.parse_args()
    verts, faces, layers = read_obj(a.full_obj)
    groups, log = relabel(np.asarray(verts, dtype=np.float64), np.asarray(faces), list(layers))
    for line in log:
        print(line)
    for m, t in groups.items():
        print("  %-26s %8.2f m2  %6d triangles" % (m, areas(t).sum(), len(t)))
    n = write_obj(a.out_obj, groups)
    print("%d vertices -> %s" % (n, a.out_obj))


if __name__ == "__main__":
    main()
