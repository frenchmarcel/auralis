# -*- coding: utf-8 -*-
"""Scene manifest -> engine objects, shared by the BRAS harnesses and the web app."""
from __future__ import annotations

import os

import numpy as np

from . import materials as mat_lib
from . import matinfer
from . import poromat
from .geometry import SceneData
from .pipeline import BakeConfig


def _bands7(v, default):
    if v is None:
        return np.full(mat_lib.N_BANDS, float(default), dtype=np.float32)
    a = np.asarray(v, dtype=np.float32)
    if a.ndim == 0:
        return np.full(mat_lib.N_BANDS, float(a), dtype=np.float32)
    if len(a) != mat_lib.N_BANDS:
        raise ValueError("expected %d bands, got %d"
                         % (mat_lib.N_BANDS, len(a)))
    return a


def build_materials(mspec: dict):
    """manifest dict -> (mats:list[engine dict], names, index by name)."""
    mats, names, index = [], [], {}
    for name, m in mspec.items():
        d = {"scattering": _bands7(m.get("scattering"), 0.15),
             "transmission": _bands7(m.get("transmission"), 0.0)}
        if "absorption_full" in m:
            af = m["absorption_full"]
            d["alpha_full"] = (np.asarray(af["f"], np.float64),
                               np.asarray(af["alpha"], np.float64))
        if "phys" in m:
            spec = poromat.normalize_spec(dict(m["phys"]))
            for flag in ("anchor_alpha", "inferred"):
                if m["phys"].get(flag):
                    spec = dict(spec, **{flag: True})
            if "absorption" in m:
                # explicit alphas (e.g. BRAS fitted) for the geometric
                # band; the physical spec serves the wave band
                # (complex admittance / resonant wall)
                d["absorption"] = _bands7(m["absorption"], 0.1)
            else:
                d["absorption"] = poromat.alpha_bands(spec).astype(
                    np.float32)
            d["phys"] = spec
        elif "absorption" in m:
            d["absorption"] = _bands7(m["absorption"], 0.1)
        else:
            raise ValueError("material '%s': neither 'absorption' nor "
                             "'phys'" % name)
        index[name] = len(mats)
        mats.append(d)
        names.append(name)
    return mats, names, index


def read_obj(path: str):
    """Minimal OBJ reader: v / f (fan) / usemtl.
    Returns (verts (nv,3), faces (nf,3) int64, face_layer list[str|None])."""
    verts, tris, layer = [], [], []
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "usemtl":
                # the WHOLE name: "Wall A" and "Wall B" are two layers
                cur = line.split(None, 1)[1].strip() if len(t) > 1 else ""
            elif t[0] == "f":
                idx = [int(w.split("/")[0]) for w in t[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
                    layer.append(cur)
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(tris, dtype=np.int64).reshape(-1, 3), layer)


def load_obj(path: str, mat_index: dict):
    """OBJ -> (tri_verts (n,3,3), tri_mat (n,)); every usemtl must be in
    mat_index, faces before the first usemtl get material 0."""
    v, f, layer = read_obj(path)
    unknown = sorted({n for n in layer if n is not None and n not in mat_index})
    if unknown:
        raise ValueError("usemtl '%s' not in the manifest (known: %s)"
                         % (unknown[0], ", ".join(sorted(mat_index))))
    n_none = sum(1 for n in layer if n is None)
    if n_none and len(mat_index) > 1:
        print("[Auralis] WARNING: %d faces of %s come before any usemtl: "
              "material 0 assigned." % (n_none, os.path.basename(path)))
    tmat = [0 if n is None else mat_index[n] for n in layer]
    return v[f], np.asarray(tmat, dtype=np.int32)


def build_scene(man: dict, base_dir: str):
    """Manifest -> SceneData (geometry from 'obj' or inline 'triangles')."""
    mats, names, index = build_materials(man["materials"])
    geo = man["geometry"]
    if "obj" in geo:
        tv, tm = load_obj(os.path.join(base_dir, geo["obj"]), index)
    else:
        tv = np.ascontiguousarray(np.asarray(geo["triangles"],
                                             dtype=np.float64))
        tm = np.asarray(geo.get("tri_mat", [0] * len(tv)),
                        dtype=np.int32)
    if tv.ndim != 3 or tv.shape[1:] != (3, 3):
        raise ValueError("triangles: expected shape (n, 3, 3)")
    a, s, t = mat_lib.build_tables(mats)
    sc = SceneData(tri_verts=tv, tri_mat=tm, absorption=a,
                    scattering=s, transmission=t,
                    mat_phys=[m.get("phys") for m in mats],
                    mat_names=names)
    # FULL absorption curves (BRAS third-octave, 20 Hz-20 kHz) for the
    # wave band: avoids the flat clamp below the first octave
    sc.mat_alpha_full = [m.get("alpha_full") for m in mats]
    if man.get("engine", {}).get("infer_materials", BakeConfig.infer_materials):
        infer_physics(sc)
    return sc


def infer_physics(sc: SceneData) -> list:
    """Give every alpha-only material the physical model that reproduces its alpha
    (matinfer), for the wave band and the complex reflections; materials with their own
    physical description are left alone. Every decision is printed."""
    phys = list(sc.mat_phys or [None] * sc.n_mats)
    full = getattr(sc, "mat_alpha_full", None) or [None] * sc.n_mats
    out = []
    for i in range(sc.n_mats):
        if phys[i]:
            out.append(None)
            continue
        c, a = matinfer.octave_targets(sc.absorption[i], full[i])
        res = matinfer.infer(a, c)
        print("[Auralis] material '%s': %s" % (sc.mat_names[i], matinfer.describe(res)))
        if res["family"] is not None:
            phys[i] = dict(res["spec"], inferred=True)
        out.append(res)
    sc.mat_phys = phys
    return out


def build_config(man: dict) -> BakeConfig:
    """Manifest -> BakeConfig (fs, duration, engine.* fields; never normalized)."""
    cfg = BakeConfig(normalize=False)
    cfg.fs = int(man.get("fs", cfg.fs))
    if man.get("duration"):
        cfg.max_time = float(man["duration"])
    eng = man.get("engine", {})
    for key, val in eng.items():
        if not hasattr(cfg, key):
            raise ValueError("engine.%s: unknown BakeConfig field" % key)
        setattr(cfg, key, type(getattr(cfg, key))(val)
                if getattr(cfg, key) is not None else val)
    cfg.normalize = False          # levels comparable across pairs
    return cfg
