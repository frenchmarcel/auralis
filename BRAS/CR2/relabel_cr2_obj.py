#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize the CR2_RIR.obj materials to the five BRAS categories
(mat_CR2_floor/ceiling/windows/plaster/concrete) and put the mesh back
to Z-up. Usage: relabel_cr2_obj.py CR2_RIR.obj [CR2_RIR_fixed.obj]"""
import sys

import numpy as np

ORDER = ["mat_CR2_floor", "mat_CR2_ceiling", "mat_CR2_windows",
         "mat_CR2_plaster", "mat_CR2_concrete"]
OFFICIAL = {  # _descr/mat_scene09.txt (BRAS, scene 09)
    "mat_CR2_floor": 49.29, "mat_CR2_ceiling": 51.62,
    "mat_CR2_windows": 9.75, "mat_CR2_plaster": 34.93,
    "mat_CR2_concrete": 56.94,
}
ALERTE_M2 = 1.5  # gap above which we refuse to conclude here


def canon(mat):
    """Canonical name if the material is one of the five BRAS, else None."""
    if mat is None:
        return None
    m = mat.replace("mat_scene09_", "mat_CR2_")
    return m if m in OFFICIAL else None


def main(src, dst):
    verts = []
    for line in open(src):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    V = np.asarray(verts)

    # --- Orientation: the engine (and classify() below) assume Z-up, floor at Z=0.
    # A Y-up mesh (e.g. from convert3d.org) fails SILENTLY: the Z-up manifest positions
    # fall outside the room and the simulation comes out empty. The vertical axis is the
    # one resting on the floor (min ~ 0, the two others bracketing 0). The mesh is turned
    # to Z-up by a PROPER rotation (det +1 keeps winding and normals). A plain Y<->Z swap
    # would be a reflection.
    vmin, vmax = V.min(axis=0), V.max(axis=0)
    span = vmax - vmin
    floor_like = np.abs(vmin) < 0.05 * span          # min sits on the floor
    axes = [i for i in range(3) if floor_like[i]]
    if axes == [2]:                                  # already Z-up
        R = np.eye(3)
        print("vertical axis: Z (already Z-up)")
    elif axes == [1]:                                # Y-up -> (x,y,z)->(x,-z,y)
        R = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        print("vertical axis: Y detected -> Z-up normalization "
              "(rotation (x,y,z)->(x,-z,y))")
    else:
        sys.exit("ORIENTATION ERROR: no single vertical axis resting on "
                 "the floor (min~0). candidate axes=%s, min=%s. The OBJ "
                 "must be Z-up or Y-up, floor at 0; redo the conversion "
                 "(README step 1)."
                 % (axes, np.round(vmin, 3).tolist()))
    V = V @ R.T                                      # vertices in Z-up
    zmax, ymax = V[:, 2].max(), V[:, 1].max()

    def indices(line):
        idx = [int(t.split("/")[0]) for t in line.split()[1:]]
        return [i - 1 if i > 0 else len(verts) + i for i in idx]

    def normal_cen(idx):
        n = np.cross(V[idx[1]] - V[idx[0]], V[idx[2]] - V[idx[0]])
        n = n / (np.linalg.norm(n) + 1e-12)
        return n, V[idx].mean(axis=0)

    def area(idx):
        a, s = V[idx[0]], 0.0
        for b, c in zip(idx[1:-1], idx[2:]):
            s += 0.5 * np.linalg.norm(np.cross(V[b] - a, V[c] - a))
        return float(s)

    def classify(idx):
        """Geometric fallback for faces with no reliable material."""
        n, cen = normal_cen(idx)
        if abs(n[2]) >= 0.7:                       # ~horizontal
            return ("mat_CR2_floor" if cen[2] < 0.5 * zmax
                    else "mat_CR2_ceiling")
        if cen[1] >= ymax - 0.5:                   # facade strip:
            return "mat_CR2_concrete"              # frames, spandrels
        return "mat_CR2_plaster"

    header, cur = [], None
    groups = {m: [] for m in ORDER}
    aires = {m: 0.0 for m in ORDER}
    recases = {}
    vi = 0
    for line in open(src):
        if line.startswith("v "):                    # rotated vertex (Z-up)
            header.append("v %.6f %.6f %.6f\n" % tuple(V[vi]))
            vi += 1
        elif line.startswith("vn "):                 # normal: same rotation
            n = R @ np.asarray([float(x) for x in line.split()[1:4]])
            header.append("vn %.6f %.6f %.6f\n" % tuple(n))
        elif line.startswith(("vt ", "mtllib", "o ", "#")):
            header.append(line)
        elif line.startswith("usemtl"):
            cur = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            idx = indices(line)
            m = canon(cur)
            if m is None:
                m = classify(idx)
                recases[cur] = recases.get(cur, 0.0) + area(idx)
            groups[m].append(line)
            aires[m] += area(idx)

    with open(dst, "w") as out:
        out.writelines(header)
        for m in ORDER:
            if groups[m]:
                out.write("usemtl %s\n" % m)
                out.writelines(groups[m])

    print("wrote %s" % dst)
    if recases:
        print("faces with no reliable material, classified by geometry:")
        for mat, a in sorted(recases.items(), key=lambda x: -x[1]):
            print("  %6.2f m2  %s" % (a, mat))
    print("areas per category (got / official / gap):")
    alerte = False
    for m in ORDER:
        off = OFFICIAL[m]
        d = aires[m] - off
        flag = ""
        if abs(d) > ALERTE_M2:
            flag, alerte = "  <-- ALERT", True
        print("  %-18s %7.2f / %6.2f / %+5.2f m2%s"
              % (m.replace("mat_CR2_", ""), aires[m], off, d, flag))
    if alerte:
        print("ALERT: partition too far from the official areas -- "
              "the conversion has probably lost the material names; "
              "redo step 2 of the README.")
        sys.exit(1)
    print("OK -- confirm with: python3 check_obj_areas.py %s --ref cr2"
          % dst)


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else "CR2_RIR_fixed.obj")
