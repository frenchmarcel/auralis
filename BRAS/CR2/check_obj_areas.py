#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the per-material areas of an OBJ against reference areas.
Without --ref: list areas per usemtl; with --ref cr2: compare with the
official BRAS scene 09 areas and flag the deviations."""
import argparse
import sys

import numpy as np

REF = {
    "cr2": {
        # official BRAS scene 09 areas (m2)
        "mat_CR2_plaster": 34.93,
        "mat_CR2_concrete": 56.94,
        "mat_CR2_windows": 9.75,
        "mat_CR2_floor": 49.29,
        "mat_CR2_ceiling": 51.62,
    },
}


def obj_areas(path):
    verts, faces = [], []
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("usemtl"):
            cur = line.split()[1] if len(line.split()) > 1 else "(none)"
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((cur, idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts, dtype=np.float64)
    areas, counts = {}, {}
    for m, a, b, c in faces:
        ar = 0.5 * np.linalg.norm(np.cross(v[b] - v[a], v[c] - v[a]))
        areas[m] = areas.get(m, 0.0) + ar
        counts[m] = counts.get(m, 0) + 1
    return areas, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--ref", choices=sorted(REF), default=None)
    ap.add_argument("--tol", type=float, default=5.0,
                    help="tolerance in %% (default 5)")
    args = ap.parse_args()
    areas, counts = obj_areas(args.obj)
    tot = sum(areas.values())
    print("%-24s %9s %7s" % ("material (usemtl)", "area m2", "tris"))
    for m, a in sorted(areas.items(), key=lambda kv: -kv[1]):
        print("%-24s %9.2f %7d" % (m, a, counts[m]))
    print("%-24s %9.2f" % ("TOTAL", tot))
    if not args.ref:
        return 0
    ref = REF[args.ref]
    print("\ncomparison with the official areas (%s):" % args.ref)
    bad = 0
    for m, a_ref in sorted(ref.items()):
        a = areas.get(m, 0.0)
        dev = 100.0 * (a - a_ref) / a_ref
        flag = "" if abs(dev) <= args.tol else "   <-- DEVIATION"
        if abs(dev) > args.tol:
            bad += 1
        print("  %-24s %8.2f  official %7.2f  (%+6.1f %%)%s"
              % (m, a, a_ref, dev, flag))
    extra = set(areas) - set(ref)
    if extra:
        print("  usemtl outside reference:", ", ".join(sorted(extra)))
    if bad:
        print("\n%d material(s) out of tolerance (%.0f %%)." % (bad, args.tol))
        return 1
    print("\nall areas within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
