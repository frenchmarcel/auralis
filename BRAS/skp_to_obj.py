#!/usr/bin/env python3
"""SketchUp (.skp, 2013-2020) -> OBJ for the bench: metres, Z up (SketchUp's own frame, the
one of the BRAS SOFA positions), one `usemtl` per material. Needs OpenSKP (MIT):
python BRAS/skp_to_obj.py in.skp out.obj [--keep mat_CR3_]"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

INCH = 0.0254


def resolve(mat_id, names):
    return names.get(mat_id) if mat_id is not None else None


def instance_material(inst, names):
    """Material painted on a group / component instance (legacy key or VFF D107 tag)."""
    from openskp import _core
    mid = inst.get("material_id")
    if mid is None:
        d007 = next((c for c in inst.get("children", []) if c.get("tag") == "D007"), None)
        d107 = d007 and next((c for c in d007["children"] if c["tag"] == "D107"), None)
        if d107:
            mid = _core.parse_var_int(d107["payload"], 0, len(d107["payload"]))
    return resolve(mid, names)


def triangles_by_material(path):
    """{material name: (n, 3, 3) triangles in metres}, every placed instance flattened. A
    face takes its front material, else its back one, else its group's (SketchUp's rule)."""
    from openskp import _core
    from openskp._face_groups import reconstruct_loop_vertices
    parsed = _core.full_parse(path)
    defs, names = parsed["defs_dict"], parsed.get("material_id_to_name", {})
    out = defaultdict(list)

    def walk(def_id, matrix, inherited, active):
        d = defs.get(def_id)
        if d is None:
            return
        b = d["builder"]
        for f in b.faces.values():
            mat = (resolve(f.get("material_id"), names) or resolve(f.get("back_material_id"), names)
                   or inherited or "default")
            loops = [lv for lv in (reconstruct_loop_vertices(lp, b.edges) for lp in f["loops"]) if lv]
            if not loops:
                continue
            for tri in _core.triangulate_face_3d(b.vertices, loops, f["normal"]):
                if all(v in b.vertices for v in tri):
                    out[mat].append([_core.transform_point(b.vertices[v], matrix) for v in tri])
        for inst in b.instances:
            ref = inst["ref_idx"]
            if ref in active:
                raise SystemExit("recursive component %r" % ref)
            walk(ref, _core.multiply_matrices(matrix, inst["matrix"]),
                 instance_material(inst, names) or inherited, active | {ref})

    walk("ROOT", [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1.0], None, frozenset())
    return {k: np.asarray(v, dtype=np.float64) * INCH for k, v in out.items()}


def areas(tris):
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)


def write_obj(path, groups):
    verts, faces, idx = [], [], {}
    lines = ["# from SketchUp by BRAS/skp_to_obj.py (OpenSKP): metres, Z up"]
    for mat in sorted(groups):
        lines.append("g %s\nusemtl %s" % (mat, mat))
        for tri in groups[mat]:
            ids = []
            for p in tri:
                key = tuple(np.round(p, 6))
                if key not in idx:
                    idx[key] = len(verts) + 1
                    verts.append(key)
                ids.append(idx[key])
            if len(set(ids)) == 3:
                lines.append("f %d %d %d" % tuple(ids))
    with open(path, "w") as fh:
        fh.write("\n".join("v %.6f %.6f %.6f" % v for v in verts) + "\n" + "\n".join(lines) + "\n")
    return len(verts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("skp")
    ap.add_argument("obj")
    ap.add_argument("--keep", default=None,
                    help="keep only the materials starting with this prefix (e.g. mat_CR3_): "
                         "source / receiver markers and their colours are dropped, loudly")
    a = ap.parse_args()
    groups = triangles_by_material(os.path.abspath(a.skp))
    print("materials (area m2, triangles):")
    for m in sorted(groups):
        t = groups[m]
        print("  %-28s %10.2f  %6d%s" % (m, areas(t).sum(), len(t),
                                        "" if not a.keep or m.startswith(a.keep) else "   DROPPED"))
    if a.keep:
        dropped = {m: areas(t).sum() for m, t in groups.items() if not m.startswith(a.keep)}
        groups = {m: t for m, t in groups.items() if m.startswith(a.keep)}
        if dropped:
            print("WARNING: %.2f m2 dropped outside %s*: %s"
                  % (sum(dropped.values()), a.keep, ", ".join(sorted(dropped))))
    allt = np.concatenate(list(groups.values()))
    lo, hi = allt.reshape(-1, 3).min(0), allt.reshape(-1, 3).max(0)
    n_v = write_obj(a.obj, groups)
    print("bounding box %s .. %s m; %d triangles, %d vertices -> %s"
          % (np.round(lo, 3), np.round(hi, 3), len(allt), n_v, a.obj))


if __name__ == "__main__":
    sys.exit(main())
