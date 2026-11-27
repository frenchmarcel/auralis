# -*- coding: utf-8 -*-
"""Reflection tracking: the deterministic arrivals of one source-receiver pair (direct
sound, image sources, diffraction) with their paths, levels and directions, computed
the way bake() computes them. CPU only (the server never opens a CUDA context)."""
from __future__ import annotations

import math

import numpy as np

from auralis.engine.air import air_attenuation_db_per_m, speed_of_sound, config_speed_of_sound
from auralis.engine.diffraction import compute_diffraction, extract_edges
from auralis.engine.directivity import as_directivity
from auralis.engine.geometry import extract_planes
from auralis.engine.ism import compute_direct, compute_image_sources
from auralis.engine.pipeline import (_select_engine, apply_source_directivity, estimate_max_time,
                                     resolve_spherical)
from auralis.engine.project import build_config, build_scene
from auralis.engine.reflection import build_admittance_table

from . import sim as simlib

MAX_ORDER = 4
SPL_1M = 94.0           # SourceDef.spl_1m, the level the results are given for


def _angles(direction, forward):
    """(world azimuth, azimuth relative to the receiver's look direction, elevation), deg."""
    d = np.asarray(direction, float)
    az_w = math.degrees(math.atan2(d[1], d[0])) % 360.0
    el = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
    f = np.asarray(forward, float)
    az_f = math.degrees(math.atan2(f[1], f[0])) if np.hypot(f[0], f[1]) > 1e-9 else 0.0
    rel = (az_w - az_f + 180.0) % 360.0 - 180.0          # + = to the listener's left
    return round(az_w, 2), round(rel, 2), round(el, 2)


def track(sim, mesh, library, geometry, source_id, receiver_id, max_order=None,
          look_azimuth_deg=None, look_elevation_deg=None, store_root=None):
    """List of arrivals sorted by time (see API.md, Reflection tracking)."""
    src = next((s for s in sim["sources"] if s["id"] == source_id), None)
    rcv = next((r for r in sim["receivers"] if r["id"] == receiver_id), None)
    if src is None or rcv is None:
        raise KeyError(source_id if src is None else receiver_id)
    open_scene = simlib.is_open_scene(sim["settings"], geometry)
    man = simlib.build_manifest(sim, mesh, library, open_scene)
    scene = build_scene(man, "")
    cfg = build_config(man)
    order = int(cfg.ism_order if max_order is None else max_order)
    if not 0 <= order <= MAX_ORDER:
        raise ValueError("max_order must be within 0-%d" % MAX_ORDER)
    c = config_speed_of_sound(cfg)
    air = air_attenuation_db_per_m(temperature_c=cfg.temperature_c,
                                   humidity_percent=cfg.humidity_percent,
                                   pressure_kpa=cfg.pressure_kpa)
    max_time = float(cfg.max_time or estimate_max_time(scene, c))
    ps, pr = np.asarray(src["position"], float), np.asarray(rcv["position"], float)
    _name, eng, tracer = _select_engine(scene, "native")
    try:
        adm = (build_admittance_table(scene.absorption, getattr(scene, "mat_phys", None))
               if cfg.complex_reflection else None)
        planes, tri_plane = extract_planes(scene, cfg.max_planes)
        sph = resolve_spherical(scene, cfg, tracer, ps)
        events = [compute_direct(scene, tracer, ps, pr, air, c)]
        if order > 0:
            events += compute_image_sources(scene, planes, tri_plane, tracer, ps, pr, order,
                                            air, c, adm_table=adm,
                                            spherical=sph)
        if cfg.diffraction:
            edges = extract_edges(scene, max_edges=cfg.max_edges)
            if edges:
                events += compute_diffraction(scene, tracer, edges, ps, pr, air, c,
                                              refl_amp=np.sqrt(scene.reflectance()),
                                              planes=planes if sph else None,
                                              tri_plane=tri_plane, adm_table=adm,
                                              spherical=sph)
    finally:
        if eng is not None:
            eng.close()
    events = [ev for ev in events if ev.time < max_time]
    from . import directivities as dirlib
    dname = src.get("directivity") or "omni"
    dirv = as_directivity(dirlib.engine_directivity(store_root, dname))
    if dirv is not None:
        apply_source_directivity(events, dirv, simlib.direction(src.get("azimuth_deg", 0),
                                                                 src.get("elevation_deg", 0)))
    gain = 10.0 ** (float(src.get("gain_db", 0.0)) / 20.0)
    fwd = simlib.direction(rcv.get("azimuth_deg", 0) if look_azimuth_deg is None
                           else look_azimuth_deg,
                           rcv.get("elevation_deg", 0) if look_elevation_deg is None
                           else look_elevation_deg)
    names = list(scene.mat_names)
    direct = next((ev for ev in events if ev.kind == "direct"), None)
    t_direct = direct.time if direct is not None else 0.0
    e_direct = float(((direct.amps * gain) ** 2).sum()) if direct is not None else 0.0
    out = []
    for ev in sorted(events, key=lambda e: e.time):
        amps = np.asarray(ev.amps, float) * gain
        e = float((amps ** 2).sum())
        if e <= 0.0:
            continue
        az_w, az_rel, el = _angles(ev.direction, fwd)
        pts = [p.tolist() for p in (ev.points or [])]
        surfaces = ([names[planes[q].material] for q in ev.planes] if ev.planes else [])
        out.append({
            "kind": ev.kind, "order": int(ev.order),
            "time_ms": round(ev.time * 1000.0, 3),
            "delay_ms": round((ev.time - t_direct) * 1000.0, 3),
            "distance_m": round(ev.time * c, 3),
            "spl_db": [round(SPL_1M + 20.0 * math.log10(a), 2) if a > 0 else None for a in amps],
            "level_db": round(SPL_1M + 10.0 * math.log10(e), 2),
            "level_rel_db": round(10.0 * math.log10(e / e_direct), 2) if e_direct > 0 else None,
            "azimuth_deg": az_w, "azimuth_rel_deg": az_rel, "elevation_deg": el,
            "surfaces": surfaces,
            "path": [[round(float(x), 4) for x in p] for p in [ps.tolist()] + pts + [pr.tolist()]]})
    return {"source_id": source_id, "receiver_id": receiver_id, "max_order": order,
            "speed_of_sound": round(c, 3), "spl_1m_db": SPL_1M, "events": out,
            "n_direct_occluded": int(direct is not None and e_direct == 0.0)}
