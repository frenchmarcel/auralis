# -*- coding: utf-8 -*-
"""Simulation document -> engine manifest (the BRAS manifest format, so the
app and the harnesses share auralis.engine.project), plus pre-run validation."""
from __future__ import annotations

import math
import os

import numpy as np

from auralis.engine.bands import CENTERS

BAND63_HZ = 63.0        # nominal centre of the 63 Hz octave (same as the band63 metrics)
PERFORATED_KEYS = ("kind", "size_mm", "t_mm", "open_pct", "plenum_mm", "wool_mm", "wool_sigma",
                   "fleece_rayl")
from . import heads
from . import meshcheck
from . import trajectory as trajlib

MIN_SURFACE_DIST = 0.10    # m, provisional rule (to be derived from the FDTD grid)
MIN_SRC_RCV_DIST = 0.50    # m
MODES = ("survey", "default", "advanced")


def direction(az_deg, el_deg):
    az, el = math.radians(float(az_deg or 0.0)), math.radians(float(el_deg or 0.0))
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def is_open_scene(settings, geometry):
    """Declared open scene: honoured unless the model closes topologically (a model
    closed only at the voxel scale may be a hill under open-air seats, not a room)."""
    method = geometry["check"].get("closure", {}).get("method", "topology")
    return bool(settings.get("open_scene")) and (not geometry["check"]["watertight"]
                                                 or method == "voxels")


def uses_wave(settings):
    mode = settings.get("mode", "survey")
    if mode == "default":
        return True
    if mode == "advanced":
        return settings.get("solvers", "both") in ("both", "wave")
    return False


def wave_only(settings):
    """Advanced mode, wave solver only: the wave band alone, up to the transition."""
    return settings.get("mode") == "advanced" and settings.get("solvers") == "wave"


def mask_wave_only(met, fc):
    """A wave-only pair: its parameters in the octaves the wave band covers whole, None in
    the others; no STI, J_LF, L_J or overall level (they need every octave)."""
    from auralis.engine.pipeline import wave_bands
    ok = [bool(x) for x in wave_bands(fc)]
    out = {}
    for k, v in met.items():
        if k in ("STI", "SPL_dbz", "SPL_dba", "SPL_dbc"):
            out[k] = None
        elif k in ("MTI", "STI_signal_db", "JLF", "LJ_db"):
            out[k] = [None] * len(v)
        elif isinstance(v, list) and len(v) == len(ok):
            out[k] = [x if o else None for x, o in zip(v, ok)]
        else:
            out[k] = v
    out["wave_only"] = True
    return out


def engine_settings(settings, open_scene=False):
    """App settings -> (engine dict of BakeConfig fields, duration or None).

    An open scene gets the ANISOTROPIC conformal correction (cut layer + tangential
    apertures). The default one degrades open scenes with sharp edges. This one is better
    on RS7 and neutral on RS6. Closed rooms keep the default (CR2 1.07 against 1.24 JND)."""
    mode = settings.get("mode", "survey")
    if mode not in MODES:
        raise ValueError("unknown mode '%s'" % mode)
    eng = {"ism_order": int(settings.get("ism_order", 2)),
           "n_rays": int(settings.get("n_rays", 200000)),
           "temperature_c": float(settings.get("temperature_c", 20.0)),
           "humidity_percent": float(settings.get("humidity_percent", 50.0)),
           "diffraction": bool(settings.get("diffraction", True)),
           "fdtd_conformal": bool(settings.get("conformal", True)),
           "fdtd_conformal_aniso": bool(settings.get("conformal", True)) and bool(open_scene),
           "fdtd": uses_wave(settings),
           "fdtd_fcross": 0.0,
           "ambisonic_order": 2}
    path = heads.hrtf_path(settings.get("head", "model"))
    if path:
        eng["hrtf_path"] = path
    if mode == "advanced" and settings.get("transition_hz"):
        eng["fdtd_fcross"] = float(settings["transition_hz"])
    if mode == "advanced":
        # surface-size scattering and specular-share image sources (Christensen & Rindel
        # 2005): the engine default, off only on request
        eng["rbs_scattering"] = bool(settings.get("rbs_scattering", True))
    if mode != "advanced":
        eng["ism_order"], eng["n_rays"] = 2, 200000
    cap = os.environ.get("AURALIS_FDTD_MAX_CELLS")
    if cap:
        # a deployment knob, not a setting: the default cap (12 M cells) suits a laptop GPU; a
        # node of large GPUs raises it (the Paris street at 250 Hz: 1.08 G cells on 4 B200)
        eng["fdtd_max_cells"] = int(float(cap))
    if wave_only(settings):
        # the geometrical part is computed and dropped: a light pass is enough
        eng.update(wave_only=True, n_rays=min(eng["n_rays"], 5000), ism_order=1)
    duration = settings.get("ir_length_s") if mode == "advanced" else None
    return eng, (float(duration) if duration else None)


def ir_length_for_decay(sim, mesh, library, volume_m3, decay_db):
    """IR length (s) for a decay of `decay_db` + 10 dB of margin in EVERY band: Eyring
    per band (the automatic rule only looks at 500 Hz - 1 kHz), plus the room's diagonal
    in flight time. Clamped to 0.3-8 s."""
    verts, faces, face_layer, _closed, names = mesh
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    A = np.zeros(len(CENTERS))
    for i, n in enumerate(names):
        a = np.asarray(library.get(sim["layers"][n]["material_id"])["absorption"], float)
        A += area[face_layer == i].sum() * a
    S = float(area.sum())
    ext = np.ptp(verts.reshape(-1, 3), axis=0)
    V = float(volume_m3 or np.prod(np.maximum(ext, 0.1)))
    abar = np.clip(A / max(S, 1e-9), 1e-3, 0.99)
    t60 = 0.161 * V / (-S * np.log(1.0 - abar))
    t = float(t60.max()) * (float(decay_db) + 10.0) / 60.0 + float(np.linalg.norm(ext)) / 343.0
    return float(np.clip(t, 0.3, 8.0))


def build_manifest(sim, mesh, library, open_scene=False):
    """One manifest material per layer (layer-level scattering overrides)."""
    verts, faces, face_layer, _closed, names = mesh
    mats = {}
    for name in names:
        entry = sim["layers"].get(name) or {}
        mid = entry.get("material_id")
        if not mid:
            raise ValueError("layer '%s' has no material" % name)
        m = library.get(mid)
        if m is None:
            raise ValueError("layer '%s': unknown material '%s'" % (name, mid))
        sc = entry.get("scattering")
        if sc is None:
            sc = m["scattering"]
        d = {"scattering": [float(x) for x in sc] if isinstance(sc, (list, tuple)) else float(sc)}
        phys = m.get("physical")
        if phys and phys.get("type") == "porous":
            # anchor_alpha: the wave band keeps the model's alphas and takes its phase
            d["phys"] = {"model": phys["model"], "sigma": phys["sigma"],
                         "thickness_mm": phys["thickness_mm"], "air_gap_mm": phys["air_gap_mm"],
                         "anchor_alpha": True}
        elif phys and phys.get("type") == "perforated":
            from .library import perforated_spec
            d["phys"] = dict(perforated_spec(**{k: phys[k] for k in PERFORATED_KEYS if k in phys}),
                             anchor_alpha=True)
            d["absorption"] = list(m["absorption"])
        elif phys and phys.get("type") == "table":
            # measured complex admittance: the wave band and the complex reflections
            # use it, the rays its random-incidence alphas
            d["phys"] = {"gamma_table": phys["gamma_table"]}
            d["absorption"] = list(m["absorption"])
        else:
            d["absorption"] = list(m["absorption"])
            if m.get("absorption_63") is not None:
                # the wave band follows the material down to 63 Hz instead of holding
                # its 125 Hz value below 125 Hz. The rays keep the 7 octaves.
                d["absorption_full"] = {"f": [BAND63_HZ] + [float(c) for c in CENTERS],
                                        "alpha": [float(m["absorption_63"])]
                                        + [float(x) for x in m["absorption"]]}
        mats[name] = d
    eng, duration = engine_settings(sim["settings"], open_scene)
    st = sim["settings"]
    if duration is None and st.get("mode") == "advanced" and st.get("stop_decay_db"):
        duration = ir_length_for_decay(sim, mesh, library, None, st["stop_decay_db"])
    man = {"name": sim["id"], "fs": 48000, "engine": eng,
           "geometry": {"triangles": verts[faces], "tri_mat": face_layer.astype(np.int32)},
           "materials": mats}
    if duration:
        man["duration"] = duration
    return man


def point_inside(pts, geometry, mesh, closure, settings):
    """True where a point is in the air volume (always, around an open model)."""
    verts, faces, _fl, closed, _names = mesh
    tri = verts[faces]
    watertight = bool(geometry["check"]["watertight"])
    if watertight and is_open_scene(settings, geometry):
        return np.ones(len(pts), dtype=bool)
    if watertight and closure is not None:
        # enclosed air at the voxel scale, minus the inside of closed objects
        return meshcheck.inside_grid(pts, closure) & ~meshcheck.inside(pts, tri[closed])
    if watertight:
        return meshcheck.inside(pts, tri[closed])
    # open scene (single panel, wedge, object in free field): positions
    # around the model are valid, only the distance to surfaces is checked
    return np.ones(len(pts), dtype=bool)


def point_checks(pts, geometry, mesh, closure, settings):
    """(inside the air volume, distance to the nearest surface) for each point."""
    tri = mesh[0][mesh[1]]
    return (point_inside(pts, geometry, mesh, closure, settings),
            meshcheck.distance_to_surfaces(pts, tri))


def validate_paths(sim, geometry, mesh, closure=None):
    """Per moving point: the whole path must stay in the air volume, MIN_SURFACE_DIST
    from the surfaces (same rule as a static point, sampled every 0.1 m)."""
    out = []
    for item in sim["sources"] + sim["receivers"]:
        if not item.get("path"):
            continue
        info = trajlib.path_info(item)
        pts = trajlib.path_points(item)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        ins, dist = point_checks(pts, geometry, mesh, closure, sim["settings"])
        reasons = []
        if not ins.all():
            k = int(np.argmin(ins))
            reasons.append("leaves the air volume %.1f m along the path, at (%.2f, %.2f, %.2f)"
                           % ((cum[k],) + tuple(pts[k])))
        if dist.min() < MIN_SURFACE_DIST:
            k = int(np.argmin(dist))
            reasons.append("passes %.2f m from a surface %.1f m along the path (min %.2f m)"
                           % (dist[k], cum[k], MIN_SURFACE_DIST))
        tid = item["path"].get("target_id")
        ids = {x["id"] for x in sim["sources"] + sim["receivers"]}
        if tid and (tid not in ids or tid == item["id"]):
            reasons.append("its target '%s' is not another source or receiver" % tid)
        out.append(dict(info, id=item["id"], valid=not reasons, reasons=reasons))
    return out


def validate_motion(sim, geometry, mesh, library, closure, receiver_id, source_ids,
                    duration_s=None):
    """Checks before an animated render of `source_ids` at `receiver_id`."""
    blocking, warnings = [], []
    names = mesh[4]
    unassigned = [n for n in names
                  if not (sim["layers"].get(n) or {}).get("material_id")
                  or library.get(sim["layers"][n]["material_id"]) is None]
    if unassigned:
        blocking.append("%d layer(s) without material" % len(unassigned))
    rcv = next((r for r in sim["receivers"] if r["id"] == receiver_id), None)
    by_id = {x["id"]: x for x in sim["sources"] + sim["receivers"]}
    srcs = [by_id[s] for s in source_ids if s in by_id and s not in
            {r["id"] for r in sim["receivers"]}]
    if rcv is None:
        blocking.append("unknown receiver '%s'" % receiver_id)
    if not srcs:
        blocking.append("no source with a sound to render")
    vib = [x["name"] for x in srcs if x.get("surface_layer")]
    if vib:
        blocking.append("vibrating layer source(s) %s: animated renders need point sources "
                        "(the geometrical solver)" % ", ".join(vib))
    involved = srcs + ([rcv] if rcv else [])
    involved_ids = {x["id"] for x in involved}
    moving = [x for x in involved if x.get("path")]
    if involved and not moving:
        blocking.append("nothing moves: give a trajectory to the receiver or to a playing "
                        "source (or use Render to WAV for a static scene)")
    paths = {p["id"]: p for p in validate_paths(sim, geometry, mesh, closure)}
    fixed = [x for x in involved if not x.get("path")]
    if fixed:
        ins, dist = point_checks([x["position"] for x in fixed], geometry, mesh, closure,
                                 sim["settings"])
        for x, i, d in zip(fixed, ins, dist):
            if not i:
                blocking.append("%s: not in the air volume" % x["name"])
            if d < MIN_SURFACE_DIST:
                blocking.append("%s: %.2f m from a surface (min %.2f m)"
                                % (x["name"], d, MIN_SURFACE_DIST))
    for x in moving:
        p = paths[x["id"]]
        if not p["valid"]:
            blocking.append("%s: %s" % (x["name"], "; ".join(p["reasons"])))
        tid = x["path"].get("target_id")
        if tid and tid in by_id and tid not in involved_ids:
            warnings.append("%s aims at %s, which is not part of this render"
                            % (x["name"], by_id[tid]["name"]))
    t_path = max([trajlib.end_time(x) for x in involved] + [0.0])
    duration = float(duration_s) if duration_s else t_path
    if not 0.5 <= duration <= trajlib.MAX_DURATION_S:
        blocking.append("render length must be within 0.5-%.0f s" % trajlib.MAX_DURATION_S)
    if rcv is not None:
        for s in srcs:
            d, t = trajlib.closest_approach(s, rcv, duration)
            if d < MIN_SRC_RCV_DIST:
                blocking.append("%s comes %.2f m from %s at t = %.2f s (min %.2f m)"
                                % (s["name"], d, rcv["name"], t, MIN_SRC_RCV_DIST))
    fast = [(x["name"], paths[x["id"]]["max_speed_mps"]) for x in moving
            if paths[x["id"]]["max_speed_mps"] > 60.0]
    for name, v in fast:
        warnings.append("%s moves at up to %.0f m/s: the Doppler shift is large and the late "
                        "field is only updated every key frame" % (name, v))
    if uses_wave(sim["settings"]) and rcv is not None and rcv.get("path") and \
            any(x.get("path") for x in srcs):
        warnings.append("a source and the receiver both move: the wave band (FDTD) takes two "
                        "runs per key frame of the receiver (a static receiver or static "
                        "sources take 1-2 wave runs in all)")
    return {"ok": not blocking, "blocking": blocking, "warnings": warnings,
            "duration_s": round(duration, 3), "path_end_s": round(t_path, 3),
            "paths": [paths[x["id"]] for x in moving]}


def validate(sim, geometry, mesh, library, closure=None):
    """Checks shown before Run; ok_to_run gates the run endpoint. `closure`: the
    room's voxel closure when its edges do not close topologically."""
    verts, faces, face_layer, closed, names = mesh
    tri = verts[faces]
    blocking = []
    unassigned = [n for n in names
                  if not (sim["layers"].get(n) or {}).get("material_id")
                  or library.get(sim["layers"][n]["material_id"]) is None]
    if unassigned:
        blocking.append("%d layer(s) without material" % len(unassigned))
    watertight = bool(geometry["check"]["watertight"])
    settings = sim["settings"]
    open_scene = is_open_scene(settings, geometry)
    if uses_wave(settings) and not (watertight or open_scene):
        blocking.append("the model is not closed: fix its open edges, or declare it an "
                        "open scene (object in free field) in Settings")
    vib = [s["name"] for s in sim["sources"] if s.get("surface_layer")]
    if vib and not wave_only(settings):
        blocking.append("vibrating layer source(s) %s: they radiate in the wave band only - "
                        "Settings: Advanced, Wave solver only" % ", ".join(vib))
    if vib and open_scene:
        blocking.append("vibrating layer source(s) %s need a closed model" % ", ".join(vib))
    if wave_only(settings) and sim.get("receiver_grids"):
        blocking.append("receiver grids need the geometrical solver: not available with the "
                        "wave solver only")
    srcs, rcvs = sim["sources"], sim["receivers"]
    if not srcs:
        blocking.append("no source")
    if not rcvs and not sim.get("receiver_grids"):
        blocking.append("no receiver")

    pts = [s["position"] for s in srcs] + [r["position"] for r in rcvs]
    min_dist, why_min = MIN_SURFACE_DIST, ""
    if uses_wave(settings) and (watertight or open_scene) and not unassigned:
        # the wave solver samples the field on its grid: a point must lie at least one
        # cell inside the air (the engine refuses points within ~half a cell)
        fc = (float(settings["transition_hz"]) if settings.get("mode") == "advanced"
              and settings.get("transition_hz") else suggested_transition(sim, mesh, library,
                                                                           open_scene))
        dx = 343.0 / (13.5 * fc)
        if dx > min_dist:
            min_dist, why_min = dx, " = one wave-solver cell at %.0f Hz" % fc
    if pts:
        ins, dist = point_checks(pts, geometry, mesh, closure, settings)
    out_items = []
    for k, item in enumerate(srcs + rcvs):
        reasons = []
        if not ins[k]:
            reasons.append("not in the air volume (outside the room or inside an object)"
                           + ("; if this is an open-air model whose ground forms a closed "
                              "solid, declare it an open scene in Settings"
                              if closure is not None else ""))
        if dist[k] < min_dist:
            reasons.append("%.2f m from a surface (min %.2f m%s)" % (dist[k], min_dist, why_min))
        out_items.append(reasons)
    for i, s in enumerate(srcs):
        for j, r in enumerate(rcvs):
            d = float(np.linalg.norm(np.subtract(s["position"], r["position"])))
            if d < MIN_SRC_RCV_DIST:
                msg = "%.2f m from %s (min %.2f m)"
                out_items[i].append(msg % (d, r["name"], MIN_SRC_RCV_DIST))
                out_items[len(srcs) + j].append(msg % (d, s["name"], MIN_SRC_RCV_DIST))
    src_v = [{"id": s["id"], "valid": not out_items[i], "reasons": out_items[i]}
             for i, s in enumerate(srcs)]
    rcv_v = [{"id": r["id"], "valid": not out_items[len(srcs) + j],
              "reasons": out_items[len(srcs) + j]} for j, r in enumerate(rcvs)]
    if any(not x["valid"] for x in src_v + rcv_v):
        blocking.append("invalid source or receiver position")
    from . import grids as gridlib
    grid_v = []
    for gr in sim.get("receiver_grids", []):
        try:
            p, valid, _why, _a = gridlib.points(gr, sim, geometry, mesh, closure)
            grid_v.append({"id": gr["id"], "n_points": int(len(p)), "n_valid": int(valid.sum()),
                           "error": None})
            if not valid.any():
                blocking.append("receiver grid %s has no valid point" % gr["name"])
        except ValueError as exc:
            grid_v.append({"id": gr["id"], "n_points": None, "n_valid": 0, "error": str(exc)})
            blocking.append(str(exc))

    warnings = []
    if wave_only(settings):
        warnings.append("wave solver only: results in the octaves below the transition only "
                        "(the others are -), and the IR stops at the transition: it sounds "
                        "muffled. No STI, J_LF or overall level (they need every octave)")
    if uses_wave(settings) and not unassigned:
        rigid = [n for n in names
                 if library.get(sim["layers"][n]["material_id"])["absorption"][0] < 0.03]
        if rigid:
            warnings.append("near-rigid materials in the wave band (alpha < 0.03 at 125 Hz): %s. "
                            "Their wall filters are checked and made passive when needed (log); "
                            "if the wave solver still becomes unstable, the affected sources fall "
                            "back to the geometrical solver (reported)."
                            % ", ".join(rigid[:5]) + (" ..." if len(rigid) > 5 else ""))
    volume = geometry.get("volume_m3")
    surface = geometry["surface_m2"]
    rt = None
    rt_note = None
    if open_scene or (volume is not None and volume <= 0):
        # sound leaves an open model: no reverberant field, no statistical decay (the signed
        # volume of an open mesh can even come out negative: -2367 m3 for the street example)
        rt_note = ("no statistical estimate for an open scene: the sound leaves the model "
                   "(run the simulation for its decay)")
    elif not unassigned:
        areas = {l["name"]: l["area_m2"] for l in geometry["layers"]}
        A = np.zeros(len(CENTERS))
        for n in names:
            A += areas[n] * np.asarray(library.get(sim["layers"][n]["material_id"])["absorption"])
        V = volume if volume else float(np.prod(tri.reshape(-1, 3).ptp(axis=0)))
        abar = np.clip(A / surface, 1e-4, 0.999)
        rt = {"sabine": (0.161 * V / np.maximum(A, 1e-9)).round(3).tolist(),
              "eyring": (0.161 * V / (-surface * np.log(1 - abar))).round(3).tolist()}
    return {"ok_to_run": not blocking, "blocking": blocking, "warnings": warnings,
            "unassigned_layers": unassigned,
            "sources": src_v, "receivers": rcv_v, "rt_estimate": rt, "rt_note": rt_note,
            "volume_m3": volume, "surface_m2": surface,
            "suggested_transition_hz": suggested_transition(sim, mesh, library, open_scene) if (
                (watertight or open_scene) and not unassigned) else None,
            "wave_available": watertight or open_scene,
            "watertight": watertight, "open_scene": open_scene,
            "closure_method": geometry["check"].get("closure", {}).get("method", "topology"),
            "paths": validate_paths(sim, geometry, mesh, closure), "grids": grid_v}


def suggested_transition(sim, mesh, library, open_scene=False):
    """Same rule as pipeline.precompute_low_bands (budget + Schroeder) - display only."""
    from auralis.engine.air import speed_of_sound, config_speed_of_sound
    from auralis.engine.fdtd import auto_crossover
    from auralis.engine.ir_synthesis import _n_samples
    from auralis.engine.pipeline import _scene_volume_hint, estimate_max_time
    from auralis.engine.project import build_config, build_scene
    s = dict(sim, settings=dict(sim["settings"], mode="default"))
    man = build_manifest(s, mesh, library, open_scene)
    scene = build_scene(man, "")
    cfg = build_config(man)
    c = config_speed_of_sound(cfg)
    max_time = float(cfg.max_time or estimate_max_time(scene, c))
    n_bins = max(8, int(np.ceil(max_time / cfg.bin_dt)))
    duration = _n_samples(n_bins, cfg.bin_dt, cfg.fs) / float(cfg.fs)
    vol = _scene_volume_hint(scene)
    f_sch = 2000.0 * np.sqrt(max(estimate_max_time(scene, c) / 1.5, 0.05) / vol)
    fc, _ = auto_crossover(vol, duration, c, f_sch, order=int(cfg.fdtd_order),
                           budget_gcell=float(cfg.fdtd_budget_gcell),
                           fcross_max=float(cfg.fdtd_fcross_max))
    return round(float(fc), 1)


# unit costs measured on a laptop GPU (RTX 2000 Ada)
COST = {"pair_fixed_s": 1.0, "trace_200k_gpu_s": 0.4, "trace_200k_cpu_s": 3.5,
        "ism_order2_s": 0.4, "fdtd_gcell_per_s_gpu": 2.0, "fdtd_gcell_per_s_cpu": 0.04,
        "fdtd_setup_s": 15.0, "grid_point_s": 0.15}


def wave_runs(s, root, fc):
    """FDTD runs of an app source: 2 when it is directional below ~2 fc (monopole and
    dipole, as pipeline.wave_source_pattern), else 1."""
    from auralis.engine.directivity import as_directivity
    from . import directivities as dirlib
    name = s.get("directivity") or "omni"
    try:
        d = as_directivity(dirlib.engine_directivity(root, name) if root
                           else (None if name == "omni" else name))
    except (ValueError, OSError, KeyError):
        return 1
    if d is None:
        return 1
    return 2 if np.any(d.first_order_weights()[CENTERS <= 2.0 * fc] <= 0.98) else 1


def estimate_runtime(sim, geometry, mesh, library, gpu=True, closure=None, root=None):
    """Rough run time (s) from the measured unit costs: pairs, wave grids, grids."""
    from . import grids as gridlib
    from auralis.engine.air import speed_of_sound_cramer
    from auralis.engine.fdtd import _LAMBDA4
    st = sim["settings"]
    eng, duration = engine_settings(st, is_open_scene(st, geometry))
    n_s, n_r = len(sim["sources"]), len(sim["receivers"])
    rays = eng["n_rays"] / 200000.0
    t_pair = (COST["pair_fixed_s"] + rays * COST["trace_200k_gpu_s" if gpu else "trace_200k_cpu_s"]
              + COST["ism_order2_s"] * (eng["ism_order"] / 2.0) ** 2)
    out = {"pairs_s": n_s * n_r * t_pair, "wave_s": 0.0, "grids_s": 0.0}
    fc, dur = None, None
    wave_one = 0.0                    # one wave grid per source (point receivers, or a grid)
    if eng["fdtd"] and (n_r or sim.get("receiver_grids")):
        fc = suggested_transition(sim, mesh, library, is_open_scene(st, geometry)) \
            if not eng["fdtd_fcross"] else eng["fdtd_fcross"]
        c = speed_of_sound_cramer(eng["temperature_c"], eng["humidity_percent"])
        V = geometry.get("volume_m3") or float(np.prod(np.ptp(mesh[0], axis=0)))
        # the IR length the run will use: fixed, from the decay criterion, or automatic
        man = build_manifest(sim, mesh, library, is_open_scene(st, geometry))
        if man.get("duration"):
            dur = float(man["duration"])
        else:
            from auralis.engine.pipeline import estimate_max_time
            from auralis.engine.project import build_scene
            dur = estimate_max_time(build_scene(man, ""), c)
        h = c / (13.5 * float(fc))
        gcells = 1.3 * V * dur * c / (_LAMBDA4 * h ** 4) / 1e9
        rate = COST["fdtd_gcell_per_s_gpu" if gpu else "fdtd_gcell_per_s_cpu"]
        wave_one = sum(COST["fdtd_setup_s"] + wave_runs(s, root, fc) * gcells / rate
                       for s in sim["sources"])
        out["wave_s"] = wave_one if n_r else 0.0
    for g in sim.get("receiver_grids", []):
        try:
            _p, valid, _w, _a = gridlib.points(g, sim, geometry, mesh, closure)
        except ValueError:
            continue
        out["grids_s"] += n_s * int(valid.sum()) * COST["grid_point_s"] * (
            st.get("grid_rays", 50000) / 50000.0) ** 0.5
        if valid.any():
            out["grids_s"] += wave_one        # G and SPL of the grid take the wave band
    total = sum(out.values())
    return dict({k: round(v, 1) for k, v in out.items()}, total_s=round(total, 1),
                transition_hz=fc, gpu=bool(gpu), ir_length_s=None if dur is None else round(dur, 2),
                note="rough: unit costs measured on the GPU of this machine")
