# -*- coding: utf-8 -*-
"""Many receivers per source (surface / plane grids): room-acoustic parameters from
the energy-time curve of each point (deterministic events + stochastic echogram),
without synthesising an IR; G and SPL follow the hybrid IR in the wave band's octaves."""
from __future__ import annotations

import time as _time
from typing import Callable, Optional

import numpy as np

from .air import air_attenuation_db_per_m
from .air import config_speed_of_sound
from .bands import N_BANDS
from .diffraction import compute_diffraction, extract_edges
from .directivity import as_directivity
from .geometry import extract_planes
from .ism import compute_direct, compute_image_sources
from .level import dba_total, strength_g
from .metrics import _decay_time, events_to_hist, lateral_metrics, schroeder_edc
from .pipeline import (BakeConfig, SourceDef, _select_engine, apply_source_directivity,
                       effective_max_bounces, estimate_max_time, hybrid_level_shift_db,
                       scene_cached)
from .reflection import build_admittance_table
from . import speech

PARAMS = ("T20", "T30", "EDT", "C50", "C80", "D50", "Ts_ms", "G_db", "SPL_db", "JLF",
          "LJ_db", "MTI", "decay_range_db")
SINGLE = ("STI", "SPL_dba")


def energy_metrics(hist_total, bin_dt, t_direct):
    """Per-band T20, T30, EDT, C50, C80, D50, Ts (ms) from band echograms (NB, n)."""
    nb, n = hist_total.shape
    t = np.arange(n) * bin_dt
    out = {k: np.full(nb, np.nan) for k in ("T20", "T30", "EDT", "C50", "C80", "D50",
                                             "Ts_ms")}
    k0 = int(t_direct / bin_dt)
    k50, k80 = k0 + int(round(0.050 / bin_dt)), k0 + int(round(0.080 / bin_dt))
    for b in range(nb):
        e = hist_total[b]
        tot = float(e.sum())
        if tot <= 0:
            continue
        db = 10.0 * np.log10(np.maximum(schroeder_edc(e), 1e-12))
        for key, lo, hi in (("T20", -5.0, -25.0), ("T30", -5.0, -35.0), ("EDT", 0.0, -10.0)):
            v = _decay_time(db, t, lo, hi)
            out[key][b] = np.nan if v is None else v
        e50, e80 = float(e[:k50].sum()), float(e[:k80].sum())
        if tot > e50 > 0:
            out["C50"][b] = 10.0 * np.log10(e50 / (tot - e50))
        if tot > e80 > 0:
            out["C80"][b] = 10.0 * np.log10(e80 / (tot - e80))
        out["D50"][b] = e50 / tot
        out["Ts_ms"][b] = 1000.0 * float(((t - t[k0]) * e).sum() / tot)
    return out


def decay_range(hist_total, bin_dt, t_direct):
    """Per band: level reached at the end of the echogram (dB re its early peak, 50 ms
    means); above -45 dB the T30 of that band is unreliable."""
    k = np.ones(50) / 50.0
    k0 = int(t_direct / bin_dt)
    out = np.full(hist_total.shape[0], np.nan)
    for b, e in enumerate(hist_total):
        sm = np.convolve(e, k, "same")
        peak = float(sm[k0:k0 + 100].max()) if len(sm) > k0 else 0.0
        tail = float(sm[-80:-30].mean()) if len(sm) > 80 else float(sm[-1])
        if peak > 0:
            out[b] = max(10.0 * np.log10(max(tail, 1e-30) / peak), -200.0)
    return out


WAVE_FS_MAX = 16000


def wave_band_points(scene, source: SourceDef, points, cfg: BakeConfig):
    """The wave band at every point (omni probes, one FDTD grid, the source's pattern and
    spectrum as for the point receivers): dict(ir (n, P) float32 at fs, fs, fc, missing (P,)
    bool: points outside the air), or None (loud) when the wave band cannot run."""
    from dataclasses import replace
    from .fdtd import run_low_band
    from .pipeline import _wave_spectrum, low_band_setup, wave_source_pattern
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c, duration, fc = low_band_setup(scene, cfg)
    fs = int(min(int(cfg.fs), WAVE_FS_MAX))         # levels only: up to 2 fc, not the full band
    axis, a_bands = wave_source_pattern(source, cfg, fc)
    p_lo = source.position_low if source.position_low is not None else source.position
    t0 = _time.time()
    try:
        ir, info = run_low_band(scene, p_lo, [p for p in pts], fs=fs, duration=duration,
                                f_cross=fc, speed_of_sound=c, engine=cfg.engine,
                                max_cells=cfg.fdtd_max_cells, order=int(cfg.fdtd_order),
                                boundary_absorber=cfg.fdtd_absorber,
                                conformal=bool(cfg.fdtd_conformal),
                                conformal_aniso=bool(cfg.fdtd_conformal_aniso),
                                align_grid=bool(cfg.fdtd_align), src_axis=axis,
                                src_first_order=a_bands, missing_rcv_ok=True)
    except (ValueError, RuntimeError) as exc:
        print("[Auralis] receiver grid: wave band IMPOSSIBLE (%s); G and SPL stay those of "
              "the geometrical solver" % exc)
        return None
    ir = np.asarray(ir, dtype=np.float64).reshape(len(ir), -1)
    ir = _wave_spectrum(ir, source, replace(cfg, fs=fs), fc)
    missing = np.zeros(len(pts), dtype=bool)
    missing[list(info.missing)] = True
    if missing.any():
        print("[Auralis] receiver grid: %d of %d points outside the wave band's air keep the "
              "geometrical G and SPL" % (int(missing.sum()), len(pts)))
    print("[Auralis] receiver grid: wave band of %d points in one grid [%s, %d cells, "
          "crossover %.0f Hz, %.1f s]" % (len(pts), info.engine, info.n_air, fc,
                                          _time.time() - t0))
    return {"ir": ir.astype(np.float32), "fs": fs, "fc": float(fc), "missing": missing}


def grid_metrics(scene, source: SourceDef, points, cfg: BakeConfig = None,
                 n_rays: Optional[int] = None, speech_effort="normal", noise_db=None,
                 progress: Optional[Callable[[float, str], Optional[bool]]] = None,
                 rcv_radii=None, wave=None):
    """Parameters at every point of `points` (P, 3) for one source; rcv_radii (P,)
    overrides cfg.rcv_radius per point (a sphere cut by a wall under-counts).
    With cfg.fdtd, G, SPL, SPL(A) and the speech levels of the STI follow the hybrid IR in
    the octaves the wave band reaches, as for a point receiver (pipeline.hybrid_level_shift_db);
    `wave` = wave_band_points(...) of these points if already computed, else it is run here.
    Decay times and clarity stay those of the energy-time curve.
    Returns {param: (P, NB) array, "STI": (P,), "SPL_dba": (P,)}; NaN = undefined."""
    cfg = cfg or BakeConfig()
    if getattr(source, "surface", ""):
        raise ValueError("receiver grids need the geometrical solver: the vibrating surface "
                         "'%s' radiates in the wave band only" % source.name)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if wave is None and cfg.fdtd and len(pts):
        wave = wave_band_points(scene, source, pts, cfg)
    n_rays = int(n_rays or cfg.n_rays)
    c = config_speed_of_sound(cfg)
    air = air_attenuation_db_per_m(temperature_c=cfg.temperature_c,
                                   humidity_percent=cfg.humidity_percent,
                                   pressure_kpa=cfg.pressure_kpa)
    max_time = float(cfg.max_time or estimate_max_time(scene, c))
    _n_bounce = effective_max_bounces(scene, cfg, max_time, c)
    n_bins = max(8, int(np.ceil(max_time / cfg.bin_dt)))
    src = np.asarray(source.position, dtype=np.float64)
    dirv = as_directivity(source.directivity)
    sfwd = np.asarray(source.forward, dtype=np.float64)
    sfwd = sfwd / max(float(np.linalg.norm(sfwd)), 1e-12)
    g_lin2 = 10.0 ** (float(source.gain_db) / 10.0)
    _name, engine, tracer = _select_engine(scene, cfg.engine)
    planes, tri_plane = extract_planes(scene, cfg.max_planes)
    edges = (list(scene_cached(scene, extract_edges, max_edges=cfg.max_edges))
             if cfg.diffraction else [])
    refl_a = np.sqrt(scene.reflectance())
    adm = (build_admittance_table(scene.absorption, getattr(scene, "mat_phys", None))
           if cfg.complex_reflection else None)
    S_1m = speech.speech_levels_1m(speech_effort)
    rbs_dims = None
    if cfg.rbs_scattering:
        from . import rbs as _rbs
        from .bands import CENTERS as _BC
        from .geometry import surface_dims
        rbs_dims = scene_cached(scene, surface_dims)
    out = {k: np.full((len(pts), N_BANDS), np.nan) for k in PARAMS}
    out["MTI"] = np.full((len(pts), N_BANDS), np.nan)
    out["STI"] = np.full(len(pts), np.nan)
    out["SPL_dba"] = np.full(len(pts), np.nan)
    t0 = _time.time()
    try:
        for i, rcv in enumerate(pts):
            events = [compute_direct(scene, tracer, src, rcv, air, c)]
            if cfg.ism_order > 0:
                events += compute_image_sources(scene, planes, tri_plane, tracer, src, rcv,
                                                cfg.ism_order, air, c, adm_table=adm,
                                                spherical=cfg.spherical_reflection)
            if edges:
                events += compute_diffraction(scene, tracer, edges, src, rcv, air, c,
                                              refl_amp=refl_a,
                                              planes=(planes if cfg.spherical_reflection
                                                      else None), tri_plane=tri_plane,
                                              adm_table=adm, spherical=cfg.spherical_reflection)
            if rbs_dims is not None:
                _rbs.weight_image_sources(events, scene, rbs_dims, _BC, c, src, rcv)
            events = [ev for ev in events if ev.time < max_time]
            if dirv is not None:
                apply_source_directivity(events, dirv, sfwd)
            if engine is not None:
                hist, dirs, _st = engine.trace(
                    src, rcv, n_rays, _n_bounce, max_time, cfg.bin_dt, n_bins, c, air,
                    spec_exclude_order=cfg.ism_order, energy_cap=cfg.energy_cap,
                    seed=cfg.seed, n_threads=cfg.n_threads,
                    rcv_radius=float(cfg.rcv_radius if rcv_radii is None else rcv_radii[i]),
                    src_forward=(sfwd if dirv is not None else None),
                    dir_table=(dirv.table if dirv is not None else None),
                    rbs=rbs_dims)
            else:
                from .stochastic import trace_python
                hist, dirs, _st = trace_python(
                    scene, src, rcv, n_rays, _n_bounce, max_time, cfg.bin_dt, n_bins,
                    c, air, spec_exclude_order=cfg.ism_order, energy_cap=cfg.energy_cap,
                    seed=cfg.seed, src_forward=(sfwd if dirv is not None else None),
                    dir_table=(dirv.table if dirv is not None else None),
                    rcv_radius=float(cfg.rcv_radius if rcv_radii is None else rcv_radii[i]),
                    rbs=rbs_dims)
            hist = hist * g_lin2
            dirs = dirs * g_lin2
            for ev in events:
                ev.amps = ev.amps * np.sqrt(g_lin2)
            total = hist + events_to_hist(events, n_bins, cfg.bin_dt, N_BANDS)
            t_direct = events[0].time if events and events[0].kind == "direct" else 0.0
            for k, v in energy_metrics(total, cfg.bin_dt, t_direct).items():
                out[k][i] = v
            out["decay_range_db"][i] = decay_range(total, cfg.bin_dt, t_direct)
            G, _gm = strength_g(total.sum(axis=1))
            if wave is not None and not wave["missing"][i]:
                G = np.asarray(G) + hybrid_level_shift_db(
                    wave["ir"][:, i].astype(np.float64) * np.sqrt(g_lin2), total, wave["fs"],
                    wave["fc"])
            out["G_db"][i] = G
            out["SPL_db"][i] = float(source.spl_1m) + G - 20.0
            out["SPL_dba"][i] = dba_total(out["SPL_db"][i])
            jlf, lj = lateral_metrics(hist, dirs, events, cfg.bin_dt, src, rcv)
            out["JLF"][i] = [np.nan if x is None else x for x in jlf]
            out["LJ_db"][i] = [np.nan if x is None else x for x in lj]
            st = speech.sti(speech.mtf_from_envelope(total, cfg.bin_dt), S_1m + G - 20.0,
                            noise_db)
            out["STI"][i], out["MTI"][i] = st["sti"], st["mti"]
            if progress is not None and progress((i + 1) / len(pts), "grid point %d / %d"
                                                 % (i + 1, len(pts))) is False:
                raise KeyboardInterrupt("grid cancelled")
    finally:
        if engine is not None:
            engine.close()
    out["timings"] = {"total_s": _time.time() - t0, "per_point_s": (_time.time() - t0)
                      / max(len(pts), 1)}
    out["wave_fc"] = None if wave is None else wave["fc"]
    return out
