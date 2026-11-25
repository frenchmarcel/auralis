#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X-ray of the material groups of an OBJ: per `usemtl`, area, dominant
normal orientation, height range and extent.

Usage:
    python3 tools/obj_groups.py CR2_RIR.obj
"""
from __future__ import annotations

import sys

import numpy as np


def analyse(path):
    verts, groups, objets = [], {}, {}
    cur, obj = "(none)", "(root)"
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] in ("o", "g"):
                obj = " ".join(t[1:]) or "(empty)"
            elif t[0] == "usemtl":
                cur = t[1] if len(t) > 1 else "(empty)"
                groups.setdefault(cur, [])
            elif t[0] == "f":
                idx = [int(w.split("/")[0]) for w in t[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    tri = (idx[0], idx[k], idx[k + 1])
                    groups.setdefault(cur, []).append(tri)
                    objets.setdefault(obj, []).append(tri)
    v = np.asarray(verts, dtype=np.float64)
    print("%d vertices, %d groups" % (len(v), len(groups)))
    print("%-10s %6s %9s %28s %18s" % ("usemtl", "tris", "area m2",
                                       "mean normal |nx ny nz|",
                                       "z min..max"))
    infos = []
    for name, tris in sorted(groups.items()):
        f = np.asarray(tris, dtype=np.int64)
        p = v[f]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        a2 = np.linalg.norm(n, axis=1)
        area = float(a2.sum() / 2.0)
        nm = (np.abs(n) / np.maximum(a2, 1e-12)[:, None])
        nmean = (nm * a2[:, None]).sum(axis=0) / max(a2.sum(), 1e-12)
        zlo, zhi = float(p[..., 2].min()), float(p[..., 2].max())
        infos.append((name, len(f), area, nmean, zlo, zhi))
        print("%-10s %6d %9.2f      %.2f %.2f %.2f %18s"
              % (name, len(f), area, nmean[0], nmean[1], nmean[2],
                 "%.2f..%.2f" % (zlo, zhi)))
    # suggestion: floor / ceiling from the z normal and the height, then
    # the verticals sorted by area (plaster >> concrete/windows usually)
    horiz = [i for i in infos if i[3][2] > 0.7]
    vert = [i for i in infos if i[3][2] <= 0.7]
    print()
    if horiz:
        sol = min(horiz, key=lambda i: 0.5 * (i[4] + i[5]))
        print("suggestion : floor    -> %s (lowest horizontal)"
              % sol[0])
        if len(horiz) > 1:
            plaf = max(horiz, key=lambda i: 0.5 * (i[4] + i[5]))
            print("suggestion : ceiling  -> %s (highest horizontal)"
                  % plaf[0])
    for i in sorted(vert, key=lambda i: -i[2]):
        print("vertical   : %-8s area %.1f m2, z %.2f..%.2f"
              % (i[0], i[2], i[4], i[5]))
    print("(verticals: the largest one is usually the plaster; "
          "windows span a narrow z range ~0.8-2.5 m; "
          "cross-check with the Pictures/Details photos)")


    print()
    print("--- per OBJECT (o/g lines) ---")
    print("%-28s %6s %9s  %s" % ("object", "tris", "area m2",
                                 "extent x / y / z"))
    for name, tris in sorted(objets.items(),
                             key=lambda kv: -len(kv[1])):
        f = np.asarray(tris, dtype=np.int64)
        p = v[f]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = float(np.linalg.norm(n, axis=1).sum() / 2.0)
        lo = p.reshape(-1, 3).min(axis=0)
        hi = p.reshape(-1, 3).max(axis=0)
        print("%-28s %6d %9.1f  [%.1f,%.1f] [%.1f,%.1f] [%.1f,%.1f]"
              % (name[:28], len(f), area, lo[0], hi[0], lo[1], hi[1],
                 lo[2], hi[2]))
    print()
    print("hint: room CR2 is about 8.4 x 6.7 x 3.0 m -- any object"
          " whose extent goes well beyond that is scenery to delete"
          " in Blender before re-export (or export with 'Selection"
          " Only' and the room alone selected).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyse(sys.argv[1])
