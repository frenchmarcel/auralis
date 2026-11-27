# -*- coding: utf-8 -*-
"""Bake pipeline: geometry in, binaural impulse response + report out.
Steps: atmosphere, direct + ISM, stochastic trace, synthesis, metrics.
"""

from __future__ import annotations

import hashlib
import os
import time as _time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional

import numpy as np

from .air import air_attenuation_db_per_m
from .air import config_speed_of_sound
from .diffraction import compute_diffraction, extract_edges
from .directivity import Directivity, as_directivity
from .bands import CENTERS, LABELS, N_BANDS
from .geometry import NumpyRaycaster, SceneData, extract_planes
from .ir_synthesis import synthesize_binaural, synthesize_stereo
from .level import level_stats
from .ism import AcousticEvent, compute_direct, compute_image_sources
from .metrics import compute_metrics, events_to_hist
from .stochastic import trace_python

try:
    from . import native as _native
except Exception:                                    # pragma: no cover
    _native = None


@dataclass
class SourceDef:
    name: str
    position: np.ndarray
    gain_db: float = 0.0
    #: Position of the low-frequency driver of a multi-way source (e.g. the
    #: 3-way BRAS dodecahedron), used by the wave band. None = position.
    position_low: "Optional[np.ndarray]" = None
    spl_1m: float = 94.0        # source level, dB SPL at 1 m (omni)
    dry_path: Optional[str] = None
    # Directivity: None/'omni', a DIRECTIVITY_PRESETS name, or a
    # Directivity object. `forward` = source front axis (default -Z,
    # the Blender speaker/camera convention).
    directivity: Optional[object] = None
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    # A vibrating surface instead of a point: the name of the scene material whose surfaces
    # move with one normal velocity (a panel, a flush-mounted loudspeaker), with the volume
    # velocity of the unit point source. Wave band only (BakeConfig.wave_only). `position`
    # is then a point of the room that anchors the air.
    surface: str = ""


@dataclass
class ReceiverDef:
    name: str
    position: np.ndarray
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class BakeConfig:
    n_rays: int = 200_000
    max_bounces: int = 200
    ism_order: int = 2
    complex_reflection: bool = True   # complex R(theta, f) on speculars
    # Point-source (spherical-wave) reflection on the image sources and ground paths around
    # diffracting edges: the ground effect and screens outdoors. Physical materials then use
    # their exact admittance. None: on in an open scene (test rays escape), off in a closed
    # room, where it changes nothing it was checked on (CR2 1.10 against 1.09 JND).
    spherical_reflection: Optional[bool] = None
    output_layout: str = "binaural"   # "binaural" (headphones) | "stereo"
    hrtf_path: str = ""               # measured HRTF .npz ('' = model)
    stereo_base_deg: float = 30.0     # half-angle of the speaker base
    bin_dt: float = 0.001            # EchogramGranularity
    max_time: Optional[float] = None  # MaxEchogramTime; None = auto (Eyring)
    fs: int = 48_000
    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    # Speed of sound with humidity (Cramer 1993). False = the dry formula 331.3 + 0.606 T.
    air_cramer: bool = True
    pressure_kpa: float = 101.325
    energy_cap: float = 1e-6         # ReverbEnergyCap
    seed: int = 1234
    n_threads: int = 0
    engine: str = "auto"             # auto | gpu | native | python
    rcv_radius: float = 0.3
    normalize: bool = True
    max_planes: int = 48
    diffraction: bool = True         # UTD edge diffraction (1st order)
    max_edges: int = 64
    fdtd: bool = False               # low-frequency wave solver
    fdtd_fcross: float = 0.0         # Hz; 0 = auto (budget + Schroeder)
    fdtd_max_cells: int = 12_000_000
    fdtd_order: int = 4              # 2 | 4 (interior stencil)
    fdtd_absorber: str = "cpml"      # cpml | sponge (open scenes)
    fdtd_budget_gcell: float = 40.0  # auto-mode budget (Gcells)
    fdtd_fcross_max: float = 1000.0  # auto crossover ceiling (Hz)
    fdtd_conformal: bool = True      # cut FV cells (true volumes/areas)
    ambisonic_order: int = 0         # 0 = off; 1-5: BakeResult.ambi (ACN/SN3D)
    # Conformal cells plus tangential face apertures, volume kept in the cut layer:
    # exact box modes and converged T20. Off by default: worse in the closed BRAS rooms.
    fdtd_conformal_aniso: bool = False
    # Reflection-based scattering (Christensen & Rindel 2005, rbs.py): the materials'
    # scattering plus that of the surfaces' finite size, per reflection. Image sources
    # keep only their specular share, so nothing is counted twice (exact against image
    # sources and radiosity, tools/ga_reference_box.py).
    rbs_scattering: bool = True
    # Directional source in the wave band too: a first-order pattern of the same power,
    # two FDTD runs. An omni wave band sends about 3 dB too much of a directional
    # loudspeaker's energy into the room (QSC K8, 250-315 Hz).
    fdtd_source_directivity: bool = True
    # Rotate the wave-band grid along the scene (smallest horizontal box). A street at an
    # angle to the axes then takes half the cells. Off by default to keep the validated grids.
    fdtd_align: bool = False
    # Corners ((x, y, z), (x, y, z)) of the wave domain, or None (the geometry's box). The
    # geometry is cut there and goes on through the absorbing layer: a terrain larger than the
    # box acts as an infinite ground, with no edge to diffract.
    fdtd_box: Optional[tuple] = None
    # Wave-band grid points per wavelength at 1.5 x the transition. Grazing reflections over a
    # long ground need more: the cells next to a wall run at second order, and the reflection
    # they give at 0.1 rad grazing is 15-36 % too strong at 13-18 points per wavelength (it
    # falls as dx^2).
    fdtd_ppw: float = 9.0
    # Materials given by alpha only get the physical model (porous layer, resistive screen,
    # panel) that reproduces their alpha, for the wave band and the complex reflections
    # (matinfer, applied by project.build_scene). The rays keep the tabulated alpha.
    infer_materials: bool = True
    # Half spacing (m) of the wave band's two ear probes, which have no head between them.
    # 0 = from the head: the spacing whose diffuse coherence matches the HRTF set (FABIAN
    # 0.167 m), else Kuhn's low-frequency sphere limit 1.5 a (0.131 m).
    wave_ear_half: float = 0.0
    # The head in the wave band (binaural). "ears" = the two probes above (no head, no level
    # difference between the ears). "hrtf" = first-order ambisonics captured around the
    # listener (centre and +- one cell along the head's front, left and up), decoded
    # through the HRTF set, so the head's shadow and delay come from the HRTFs.
    # "hrtf2" = second order (19 probes, adding the 12 diagonals in the head's three planes).
    wave_head: str = "ears"
    # The wave band alone: the IR is the FDTD band through the crossover's low-pass, G and
    # SPL come from its energy, and the ISO 3382 parameters are its own. Octaves whose upper
    # edge lies above the transition hold no result. Needs fdtd, else an error is raised.
    wave_only: bool = False


@dataclass
class BakeResult:
    ir: np.ndarray                   # (n, 2) binaural IR
    fs: int
    hist: np.ndarray                 # stochastic echogram (NB, n_bins)
    hist_total: np.ndarray           # stochastic + deterministic
    dirs: np.ndarray                 # (n_bins, 3)
    events: List[AcousticEvent]
    metrics: dict
    stats: dict
    engine_used: str
    timings: dict
    config: BakeConfig
    ir_scale: float = 1.0
    ambi: Optional[np.ndarray] = None   # (n, (order+1)^2) ACN/SN3D, same scale as ir
    # omni components before the crossover (ambisonics + wave band only):
    # {"geo": W of the geometrical solver, "wave": centre-probe low band, "fc": Hz}
    omni_parts: Optional[dict] = None



def effective_max_bounces(scene: SceneData, cfg, max_time: float, c: float) -> int:
    """The tracer's bounce cap for an IR of max_time.

    Never below cfg.max_bounces, and at least twice the reflections a ray meets in max_time
    (mean free path 4V/S, bounded to 20000). energy_cap's roulette already ends rays without
    bias. A cap that binds before max_time cuts the tail of a reverberant room."""
    tv = np.asarray(scene.tri_verts, dtype=np.float64).reshape(-1, 3, 3)
    area = float(0.5 * np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]),
                                      axis=1).sum())
    base = int(cfg.max_bounces)
    if area <= 0.0 or max_time <= 0.0:
        return base
    mfp = max(4.0 * _scene_volume_hint(scene) / area, 0.05)
    need = int(np.ceil(2.0 * c * float(max_time) / mfp))
    return max(base, min(need, 20000))


def warn_bounce_cap(stats: dict, cap: int, label: str = "") -> None:
    """LOUD warning when the rays still met the bounce cap on average: late energy is missing."""
    if float(stats.get("mean_bounces", 0.0)) >= 0.95 * cap:
        print("[Auralis] WARNING: %sthe rays met the bounce cap (%d, mean %.0f) before the end "
              "of the IR: the late tail is truncated; raise BakeConfig.max_bounces."
              % (label + ": " if label else "", cap, float(stats["mean_bounces"])), flush=True)


def _scene_volume_hint(scene: SceneData) -> float:
    """Approximate volume (m^3): bounding box x 0.85."""
    tv = np.asarray(scene.tri_verts, dtype=np.float64).reshape(-1, 3)
    ext = tv.max(axis=0) - tv.min(axis=0)
    return float(max(np.prod(np.maximum(ext, 0.1)) * 0.85, 1.0))


def estimate_max_time(scene: SceneData, speed_c: float) -> float:
    """1.5 x Eyring RT from bbox volume + surface, clamped to [0.5, 8] s."""
    lo, hi = scene.bbox()
    V = float(np.prod(np.maximum(hi - lo, 1e-3)))
    S = float(scene.triangle_areas().sum())
    if S <= 0:
        return 1.0
    areas = scene.triangle_areas()
    a_mid = scene.absorption[:, 2:4].mean(axis=1)     # 500 Hz - 1 kHz
    abar = float((areas * a_mid[scene.tri_mat]).sum() / areas.sum())
    abar = float(np.clip(abar, 0.01, 0.99))
    t_ey = 0.161 * V / (-S * np.log(1.0 - abar)) * (343.0 / speed_c)
    diag = float(np.linalg.norm(hi - lo))
    return float(np.clip(1.5 * t_ey + diag / speed_c, 0.5, 8.0))


def _select_engine(scene: SceneData, mode: str):
    """Returns (name, engine_or_None, tracer) for mode auto | gpu |
    native | python; auto falls back GPU -> native CPU -> Python."""
    if mode in ("auto", "gpu") and _native is not None:
        gp = _native.find_gpu_library() if _native.gpu_available() else None
        if gp is not None:
            try:
                eng = _native.NativeEngine(scene, lib_path=gp)
                return "cuda", eng, eng
            except Exception as exc:
                if mode == "gpu":
                    raise
                print("[Auralis] GPU unavailable (%s): falling back to "
                      "the native CPU engine" % exc)
        elif mode == "gpu":
            raise OSError("engine 'gpu' requested but libauralis_cuda "
                          "is missing or no GPU is present (see "
                          "native/build_cuda.sh)")
    if mode in ("auto", "native") and _native is not None:
        try:
            eng = _native.NativeEngine(scene)
            return "native", eng, eng
        except Exception:
            if mode == "native":
                raise
    return "python", None, NumpyRaycaster(scene)


_SCENE_CACHE: "OrderedDict" = OrderedDict()
_SCENE_CACHE_SIZE = 8


_OPEN_TOLD = set()


def resolve_spherical(scene: SceneData, cfg, tracer, src) -> bool:
    """BakeConfig.spherical_reflection resolved. None turns it on when more than 5 % of 256
    test rays from the source escape the model (an open scene), and says so once."""
    v = getattr(cfg, "spherical_reflection", None)
    if v is not None:
        return bool(v)
    src = np.asarray(src, dtype=np.float64)
    tv = np.asarray(scene.tri_verts, dtype=np.float64).reshape(-1, 3)
    reach = 4.0 * float(np.linalg.norm(tv.max(axis=0) - tv.min(axis=0))) + 1.0
    n = 256
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    ph = np.pi * (1.0 + 5.0 ** 0.5) * i
    r = np.sqrt(1.0 - z * z)
    dirs = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)
    out = sum(1 for d in dirs if tracer.cast(src, d, reach)[0] < 0)
    on = out > 0.05 * n
    key = (id(scene), on)
    if on and key not in _OPEN_TOLD:
        _OPEN_TOLD.add(key)
        print("[Auralis] open scene (%d of %d test rays escape): spherical-wave reflections and "
              "ground paths around edges are on" % (out, n), flush=True)
    return on


def scene_cached(scene: SceneData, fn: Callable, **kw):
    """fn(scene, **kw) for a function of the triangles and their materials only.

    Used for extract_edges and surface_dims, memoized on the content of tri_verts and
    tri_mat. The cached value is shared: callers must not modify it."""
    h = hashlib.blake2b(digest_size=16)
    for a in (scene.tri_verts, scene.tri_mat):
        a = np.ascontiguousarray(a)
        h.update(("%s%s" % (a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    key = (fn.__module__, fn.__qualname__, h.hexdigest(), tuple(sorted(kw.items())))
    if key in _SCENE_CACHE:
        _SCENE_CACHE.move_to_end(key)
        return _SCENE_CACHE[key]
    value = fn(scene, **kw)
    _SCENE_CACHE[key] = value
    while len(_SCENE_CACHE) > _SCENE_CACHE_SIZE:
        _SCENE_CACHE.popitem(last=False)
    return value


def bake(scene: SceneData, source: SourceDef, receiver: ReceiverDef,
         config: BakeConfig = None,
         progress: Optional[Callable[[float, str], Optional[bool]]] = None,
         low_band=None) -> BakeResult:
    """Bake one source -> one receiver; progress(frac, stage) -> False raises
    KeyboardInterrupt; low_band from precompute_low_bands() skips the FDTD."""
    cfg = config or BakeConfig()
    timings = {}

    def tick(frac, stage):
        if progress is not None and progress(frac, stage) is False:
            raise KeyboardInterrupt("bake cancelled")

    t0 = _time.time()
    air = air_attenuation_db_per_m(temperature_c=cfg.temperature_c,
                                   humidity_percent=cfg.humidity_percent,
                                   pressure_kpa=cfg.pressure_kpa)
    c = config_speed_of_sound(cfg)
    max_time = cfg.max_time or estimate_max_time(scene, c)
    n_bins = max(8, int(np.ceil(max_time / cfg.bin_dt)))

    engine_name, engine, tracer = _select_engine(scene, cfg.engine)
    timings["setup_s"] = _time.time() - t0
    tick(0.02, "geometry")

    # ---- deterministic early part ----------------------------------------
    t0 = _time.time()
    src = np.asarray(source.position, dtype=np.float64)
    rcv = np.asarray(receiver.position, dtype=np.float64)
    dirv = as_directivity(source.directivity)
    sfwd = np.asarray(source.forward, dtype=np.float64)
    nf = float(np.linalg.norm(sfwd))
    sfwd = sfwd / nf if nf > 1e-9 else np.array([0.0, 0.0, -1.0])
    adm_table = None
    if cfg.complex_reflection:
        from .reflection import build_admittance_table
        adm_table = build_admittance_table(
            scene.absorption,
            getattr(scene, 'mat_phys', None))
    sph = resolve_spherical(scene, cfg, tracer, src)
    events: List[AcousticEvent] = [
        compute_direct(scene, tracer, src, rcv, air, c)]
    # the direct sound interferes with the spherical-wave reflections: same precision
    events[0].exact = bool(sph and adm_table is not None)
    planes, tri_plane = extract_planes(scene, cfg.max_planes)
    if cfg.ism_order > 0:
        events += compute_image_sources(scene, planes, tri_plane, tracer,
                                        src, rcv, cfg.ism_order, air, c,
                                        adm_table=adm_table,
                                        spherical=sph)
    n_edges = 0
    if cfg.diffraction:
        edges = list(scene_cached(scene, extract_edges, max_edges=cfg.max_edges))
        n_edges = len(edges)
        if edges:
            events += compute_diffraction(
                scene, tracer, edges, src, rcv, air, c,
                refl_amp=np.sqrt(scene.reflectance()),
                planes=planes if sph else None, tri_plane=tri_plane,
                adm_table=adm_table, spherical=sph)
    rbs_dims = None
    if cfg.rbs_scattering:
        from . import rbs as _rbs
        from .bands import CENTERS as _BC
        from .geometry import surface_dims
        rbs_dims = scene_cached(scene, surface_dims)
        _rbs.weight_image_sources(events, scene, rbs_dims, _BC, c, src, rcv)
    events = [ev for ev in events if ev.time < max_time]
    if dirv is not None:
        apply_source_directivity(events, dirv, sfwd)
    timings["ism_s"] = _time.time() - t0
    tick(0.10, "early: %d events (%d edges)" % (len(events),
                                                n_edges))

    # ---- stochastic tail ---------------------------------------------------
    t0 = _time.time()
    hist = np.zeros((N_BANDS, n_bins), dtype=np.float64)
    dirs = np.zeros((n_bins, 3), dtype=np.float64)
    stats = {"rays": 0.0, "mean_bounces": 0.0, "escaped": 0.0,
             "deposits": 0.0, "diff_edges": float(n_edges),
             "diff_events": float(sum(1 for ev in events
                                      if ev.kind == "diffracted"))}

    n_bounce = effective_max_bounces(scene, cfg, max_time, c)
    if engine_name in ("native", "cuda"):
        chunks = (1 if engine_name == "cuda"
                  else int(np.clip(cfg.n_rays // 25_000, 1, 16)))
        per = int(np.ceil(cfg.n_rays / chunks))
        done = 0
        for i in range(chunks):
            nr = min(per, cfg.n_rays - done)
            if nr <= 0:
                break
            h, dd, st = engine.trace(
                src, rcv, nr, n_bounce, max_time, cfg.bin_dt, n_bins,
                c, air, spec_exclude_order=cfg.ism_order,
                energy_cap=cfg.energy_cap, seed=cfg.seed + 7919 * i,
                n_threads=cfg.n_threads, rcv_radius=cfg.rcv_radius,
                src_forward=(sfwd if dirv is not None else None),
                dir_table=(dirv.table if dirv is not None else None),
                rbs=rbs_dims)
            w = nr / cfg.n_rays
            hist += h * w
            dirs += dd * w
            stats["mean_bounces"] += st["mean_bounces"] * w
            stats["escaped"] += st["escaped"]
            stats["deposits"] += st["deposits"]
            done += nr
            stats["rays"] = float(done)
            tick(0.10 + 0.60 * done / cfg.n_rays,
                 "trace %s %d / %d rays"
                 % (engine_name, done, cfg.n_rays))
    else:
        def cb(done, total):
            tick(0.10 + 0.60 * done / total,
                 "trace Python %d / %d rays" % (done, total))
            return True
        hist, dirs, stats = trace_python(
            scene, src, rcv, cfg.n_rays, n_bounce, max_time,
            cfg.bin_dt, n_bins, c, air,
            spec_exclude_order=cfg.ism_order, energy_cap=cfg.energy_cap,
            seed=cfg.seed, progress=cb, rcv_radius=cfg.rcv_radius,
            src_forward=(sfwd if dirv is not None else None),
            dir_table=(dirv.table if dirv is not None else None),
            rbs=rbs_dims)
    timings["trace_s"] = _time.time() - t0
    stats["max_bounces"] = float(n_bounce)
    warn_bounce_cap(stats, n_bounce)

    # NaN tripwire: name the guilty stage instead of writing poisoned
    # files (one NaN in the IR -> the whole mix goes NaN through FFT)
    if not np.isfinite(dirs).all():
        raise RuntimeError("NaN/Inf in the tracer directions "
                           "(engine %s)" % engine_name)
    if not np.isfinite(hist).all():
        raise RuntimeError("NaN/Inf in the tracer histogram (engine "
                           "%s) - degenerate geometry or instability; "
                           "please report this error."
                           % engine_name)
    for _ev in events:
        if not np.isfinite(np.asarray(_ev.amps)).all():
            raise RuntimeError("NaN/Inf in a %s event (order %s)"
                               % (getattr(_ev, "kind", "?"),
                                  getattr(_ev, "order", "?")))

    # ---- gain, synthesis, metrics -----------------------------------------
    g = 10.0 ** (source.gain_db / 20.0)
    if g != 1.0:
        hist *= g * g
        dirs *= g * g
        for ev in events:
            ev.amps = ev.amps * g

    t0 = _time.time()
    if cfg.output_layout == "stereo":
        tick(0.72, "stereo loudspeaker synthesis")
        ir, info = synthesize_stereo(
            events, hist, dirs, cfg.bin_dt, cfg.fs, c,
            forward=receiver.forward, up=receiver.up,
            seed=cfg.seed ^ 0x5DEECE66D,
            normalize=False, adm_table=adm_table,
            base_deg=cfg.stereo_base_deg)
    else:
        hset = None
        if cfg.hrtf_path:
            try:
                from .hrtf_sofa import load_hrtf
                hset = load_hrtf(cfg.hrtf_path, cfg.fs)
            except Exception as exc:
                print("[Auralis] HRTF ignored (%s), parametric model"
                      % exc)
        tick(0.72, "binaural synthesis"
             + (" (HRTF %s)" % hset.name if hset is not None else ""))
        ir, info = synthesize_binaural(
            events, hist, dirs, cfg.bin_dt, cfg.fs, c,
            forward=receiver.forward, up=receiver.up,
            seed=cfg.seed ^ 0x5DEECE66D,
            # normalization is centralized below (gain logged)
            normalize=False, adm_table=adm_table, hrtf_set=hset)
    ambi = None
    omni_parts = None
    if int(cfg.ambisonic_order) > 0:
        from .ir_synthesis import synthesize_ambisonic
        ambi = synthesize_ambisonic(events, hist, dirs, cfg.bin_dt, cfg.fs,
                                    order=int(cfg.ambisonic_order),
                                    seed=cfg.seed ^ 0x5DEECE66D,
                                    adm_table=adm_table)
    timings["synthesis_s"] = _time.time() - t0
    if not np.isfinite(ir).all():
        raise RuntimeError("NaN/Inf in the SYNTHESIS (events and "
                           "histogram healthy on input) - please "
                           "report with the scene.")

    wave_omni, wave_fc = None, 0.0
    if getattr(source, "surface", "") and not cfg.wave_only:
        raise ValueError("source '%s' is a vibrating surface: it radiates in the wave band only "
                         "(BakeConfig.wave_only)" % source.name)
    if cfg.wave_only:
        if not cfg.fdtd:
            raise ValueError("wave_only needs the wave solver (fdtd=True)")
        # fused with silence, the IR keeps the low-passed wave band alone
        ir = np.zeros_like(ir)
        ambi = None if ambi is None else np.zeros_like(ambi)
    if cfg.fdtd:
        from .fdtd import auto_crossover, fuse_low_high, run_low_band
        dur_ir = len(ir) / cfg.fs
        # the source gain scales the wave band as it scales the echogram and the events
        if low_band is not None:
            # PRECOMPUTED low band (one grid per source): fused as is,
            # padded/trimmed to the IR duration.
            ir_low, finfo, fc = low_band
            ir_low = np.asarray(ir_low, dtype=np.float64)
            n_ir = len(ir)
            if ir_low.shape[0] < n_ir:
                pad = [(0, n_ir - ir_low.shape[0])] + \
                    [(0, 0)] * (ir_low.ndim - 1)
                ir_low = np.pad(ir_low, pad)
            if g != 1.0:
                ir_low = ir_low * g
            tick(0.80, "low-frequency FDTD (crossover %.0f Hz, "
                 "precomputed)" % fc)
            t0 = _time.time()
            if ambi is not None:
                omni_parts = {"geo": ambi[:, 0].copy(),
                              "wave": _omni_low(ir_low, ambi.shape[0]), "fc": float(fc)}
                ambi = _fuse_ambi_w(ambi, ir_low, fc, cfg.fs)
            wave_omni = _omni_low(ir_low, len(ir))     # centre probe, else the ears
            wave_fc = fc
            if ir_low.ndim == 2 and ir_low.shape[1] == 3:
                ir_low = ir_low[:, :2]           # centre probe: W only
            ir = fuse_low_high(ir, ir_low, fc, cfg.fs)
            stats.update({"fdtd_fc": fc, "fdtd_cells": float(finfo.n_air),
                          "fdtd_dx": finfo.dx,
                          "fdtd_open": 1.0 if finfo.is_open else 0.0,
                          "fdtd_order": float(finfo.order),
                          "fdtd_fc_budget": 0.0,
                          "fdtd_precomputed": 1.0,
                          "fdtd_conformal": 1.0 if getattr(
                              finfo, "conformal", False) else 0.0,
                          "fdtd_ears": 1.0 if ir_low.ndim == 2 else 0.0})
            stats["fdtd_engine"] = {"python": 0.0, "native": 1.0,
                                    "gpu": 2.0}.get(finfo.engine, 0.0)
            timings["fdtd_s"] = _time.time() - t0
        else:
            vol = _scene_volume_hint(scene)
            rt_e = estimate_max_time(scene, c) / 1.5
            f_sch = 2000.0 * np.sqrt(max(rt_e, 0.05) / vol)
            fc = float(cfg.fdtd_fcross)
            fc_budget = 0.0
            if fc <= 0.0:
                fc, fc_budget = auto_crossover(
                    vol, dur_ir, c, f_sch, order=int(cfg.fdtd_order),
                    budget_gcell=float(cfg.fdtd_budget_gcell),
                    fcross_max=float(cfg.fdtd_fcross_max))
            else:
                fc = float(np.clip(fc, 60.0,
                                   max(float(cfg.fdtd_fcross_max), 60.0)))
            tick(0.80, "low-frequency FDTD (crossover %.0f Hz)" % fc)
            t0 = _time.time()
            wave_axis, wave_a = wave_source_pattern(source, cfg, fc)
            layout = str(getattr(cfg, "output_layout", "binaural")).lower()
            rcv_arg = receiver.position
            if layout != "stereo":
                fwd = np.asarray(receiver.forward, dtype=np.float64)
                upv = np.asarray(receiver.up, dtype=np.float64)
                r = np.cross(fwd, upv)
                nr = np.linalg.norm(r)
                if nr > 1e-9:
                    r = r / nr
                    pos = np.asarray(receiver.position, dtype=np.float64)
                    h = wave_ear_half(cfg)
                    rcv_arg = [pos - h * r, pos + h * r]
                    if ambi is not None:           # centre probe for W
                        rcv_arg.append(pos)
                    if _wave_head_on(cfg):
                        rcv_arg = wave_head_probes(pos, fwd, upv, fc, c, _head_order(cfg))
            try:
                try:
                    ir_low, finfo = run_low_band(
                        scene,
                        (source.position_low
                         if source.position_low is not None
                         else source.position), rcv_arg, fs=cfg.fs,
                        duration=dur_ir, f_cross=fc, speed_of_sound=c,
                        engine=cfg.engine, max_cells=cfg.fdtd_max_cells,
                        order=int(cfg.fdtd_order),
                        boundary_absorber=cfg.fdtd_absorber,
                        conformal=bool(cfg.fdtd_conformal),
                            conformal_aniso=bool(cfg.fdtd_conformal_aniso),
                            align_grid=bool(cfg.fdtd_align), domain_box=cfg.fdtd_box,
                        ppw=float(cfg.fdtd_ppw),
                        src_axis=wave_axis, src_first_order=wave_a,
                        src_surface=source.surface or None)
                except ValueError as exc:
                    if isinstance(rcv_arg, list):
                        print("[Auralis] in-grid ears impossible (%s): "
                              "mono fallback" % exc)
                        ir_low, finfo = run_low_band(
                            scene, source.position, receiver.position,
                            fs=cfg.fs, duration=dur_ir, f_cross=fc,
                            speed_of_sound=c, engine=cfg.engine,
                            max_cells=cfg.fdtd_max_cells,
                            order=int(cfg.fdtd_order),
                            boundary_absorber=cfg.fdtd_absorber,
                            conformal=bool(cfg.fdtd_conformal),
                            conformal_aniso=bool(cfg.fdtd_conformal_aniso),
                            align_grid=bool(cfg.fdtd_align), domain_box=cfg.fdtd_box,
                        ppw=float(cfg.fdtd_ppw),
                            src_axis=wave_axis, src_first_order=wave_a,
                            src_surface=source.surface or None)
                    else:
                        raise
                ir_low = _wave_spectrum(ir_low, source, cfg, fc)
                if np.ndim(ir_low) == 2 and np.shape(ir_low)[1] in HEAD_PROBES:
                    ir_low = wave_head_decode(ir_low, cfg, fc, c)
                if g != 1.0:
                    ir_low = ir_low * g
                if not np.isfinite(ir_low).all():
                    raise RuntimeError(
                        "NaN/Inf in the FDTD band (engine %s, crossover "
                        "%.0f Hz, %s cells) - wave solver instability: "
                        "leaky room, source/mic outside the air volume, "
                        "or pathological geometry."
                        % (getattr(finfo, "engine", "?"), fc,
                           getattr(finfo, "n_air", "?")))
                if ambi is not None:
                    omni_parts = {"geo": ambi[:, 0].copy(),
                                  "wave": _omni_low(ir_low, ambi.shape[0]), "fc": float(fc)}
                    ambi = _fuse_ambi_w(ambi, ir_low, fc, cfg.fs)
                ir = fuse_low_high(ir, ir_low, fc, cfg.fs)
                wave_omni = _omni_low(ir_low, len(ir))
                wave_fc = fc
                stats.update({"fdtd_fc": fc,
                              "fdtd_cells": float(finfo.n_air),
                              "fdtd_dx": finfo.dx,
                              "fdtd_open": 1.0 if finfo.is_open else 0.0,
                              "fdtd_order": float(finfo.order),
                              "fdtd_fc_budget": float(fc_budget),
                              "fdtd_conformal": 1.0 if getattr(
                                  finfo, "conformal", False) else 0.0,
                              "fdtd_ears": 1.0 if (isinstance(rcv_arg, list)
                                                   and np.asarray(ir_low).ndim
                                                   == 2) else 0.0})
                stats["fdtd_engine"] = {"python": 0.0, "native": 1.0,
                                        "gpu": 2.0}.get(finfo.engine, 0.0)
            except ValueError as exc:
                print("[Auralis] FDTD skipped: %s" % exc)
                stats["fdtd_fc"] = 0.0
            timings["fdtd_s"] = _time.time() - t0

    # ---- listening normalization (optional, gain logged) ----------------
    # The un-normalized IR is in the absolute "pressure re 1 m" convention:
    # IR_absolute = IR_rendered / scale. Levels below are computed on the
    # echogram, which is never rescaled: they do not depend on this choice.
    scale = 1.0
    if cfg.normalize:
        pk = float(np.abs(ir).max())
        if pk > 0.0:
            scale = 0.985 / pk
            ir *= scale
    stats["norm_gain_db"] = float(20.0 * np.log10(scale))

    t0 = _time.time()
    hist_total = hist + events_to_hist(events, n_bins, cfg.bin_dt, N_BANDS)
    if cfg.wave_only and wave_omni is None:
        raise RuntimeError("wave-only output: the wave band did not run (%s)"
                           % ("crossover below 125 Hz" if cfg.fdtd else "fdtd off"))
    level_shift = (None if wave_omni is None else
                   hybrid_level_shift_db(wave_omni, hist_total, cfg.fs, wave_fc,
                                         geometrical=not cfg.wave_only))
    lv = level_stats(hist_total, spl_1m=float(source.spl_1m), shift_db=level_shift)
    stats.update({"spl_db": lv["spl_db"], "spl_dba": lv["spl_dba"],
                  "g_mid": lv["g_mid"],
                  "complex_refl": 1.0 if adm_table is not None else 0.0,
                  "stereo": 1.0 if cfg.output_layout == "stereo" else 0.0})
    metrics = compute_metrics(hist_total, dirs, events, cfg.bin_dt)
    metrics["spl_bands"] = lv["spl_bands"]
    metrics["g_bands"] = lv["g_bands"]
    metrics["spl_db"] = lv["spl_db"]
    metrics["spl_dba"] = lv["spl_dba"]
    metrics["g_mid"] = lv["g_mid"]
    metrics["wave_level_shift_db"] = (None if level_shift is None
                                      else [float(x) for x in level_shift])
    metrics["max_time_s"] = max_time
    metrics["speed_of_sound"] = c
    if cfg.wave_only:
        metrics.update(wave_only_metrics(_omni_low(wave_omni_lp(wave_omni, wave_fc, cfg.fs),
                                                   len(ir)), cfg.fs, wave_fc, metrics))
    timings["metrics_s"] = _time.time() - t0

    if engine is not None:
        engine.close()
    tick(1.0, "done")
    if not np.isfinite(ir).all():
        raise RuntimeError("NaN/Inf after FUSION (synthesis and FDTD "
                           "both healthy on their own) - fusion bug; "
                           "please report.")
    if ambi is not None:
        ambi = ambi * scale
    if omni_parts is not None:
        omni_parts = dict(omni_parts, geo=omni_parts["geo"] * scale,
                          wave=omni_parts["wave"] * scale)
    return BakeResult(ir=ir, fs=cfg.fs, hist=hist, hist_total=hist_total,
                      dirs=dirs, events=events, metrics=metrics, stats=stats,
                      engine_used=engine_name, timings=timings, config=cfg,
                      ir_scale=scale, ambi=ambi, omni_parts=omni_parts)


def wave_bands(fc: float) -> np.ndarray:
    """Octaves (CENTERS) the wave band covers whole: upper edge (centre x sqrt 2) <= fc."""
    return CENTERS * 2.0 ** 0.5 <= float(fc)


def wave_omni_lp(wave_omni, fc: float, fs: int) -> np.ndarray:
    """The wave band's omni IR through the crossover's low-pass (zero phase, as fused)."""
    from .fdtd import design_crossover
    w = np.asarray(wave_omni, dtype=np.float64)
    lp, _hp, delay = design_crossover(fc, fs)
    return np.convolve(w, lp)[delay:delay + len(w)]


def wave_only_metrics(omni, fs: int, fc: float, metrics: dict) -> dict:
    """What a wave-only result reports: per_band from its IR (ISO 3382, true octaves), G and SPL
    of the octaves it covers, None elsewhere (the echogram's values are the geometrical
    solver's, not this result's)."""
    from .rirmetrics import band_metrics as ir_band_metrics
    ok = wave_bands(fc)
    m = ir_band_metrics(np.asarray(omni, dtype=np.float64), fs)
    per_band = {}
    for b, lab in enumerate(LABELS):
        if not ok[b]:
            per_band[lab] = {k: None for k in ("RT", "RT_method", "EDT", "C50", "C80", "D50",
                                               "Ts")}
            continue
        rt, meth = (m["T30"][b], "T30") if m["T30"][b] is not None else (m["T20"][b], "T20")
        per_band[lab] = {"RT": rt, "RT_method": meth if rt is not None else None,
                         "EDT": m["EDT"][b], "C50": m["C50"][b], "C80": m["C80"][b],
                         "D50": m["D50"][b],
                         "Ts": None if m["Ts"][b] is None else m["Ts"][b] / 1000.0}
    g = [float(v) if o else None for v, o in zip(metrics["g_bands"], ok)]
    spl = [float(v) if o else None for v, o in zip(metrics["spl_bands"], ok)]
    # overall levels need every octave: none here
    return {"per_band": per_band, "g_bands": g, "spl_bands": spl,
            "spl_db": None, "spl_dba": None, "g_mid": None, "wave_only": True,
            "wave_bands": [bool(x) for x in ok], "eax": {k: None for k in metrics["eax"]}}


def hybrid_level_shift_db(wave_omni, hist_total, fs: int, fc: float,
                          geometrical: bool = True) -> np.ndarray:
    """Per-band shift (dB) from the echogram's levels (G, SPL) to those of the hybrid IR.

    Applies to the bands the wave band reaches (centre <= 2 fc), 0 above. Per true octave:
    the energy of the low-passed wave band (omni IR, absolute 'pressure re 1 m' convention)
    plus the echogram's energy times the share of the octave the crossover's high-pass lets
    through. The geometrical part is not taken from its synthesized noise (one realisation
    varies by +-1.6 dB in the 125 Hz octave).
    geometrical=False: the wave band's energy alone (wave-only output)."""
    from .fdtd import design_crossover
    from .level import band_energies, strength_g
    from .rirmetrics import strength_db
    k = CENTERS <= 2.0 * float(fc)
    out = np.zeros(N_BANDS)
    if not k.any():
        return out
    w = np.asarray(wave_omni, dtype=np.float64)
    lp, hp, delay = design_crossover(fc, fs)
    w_lp = np.convolve(w, lp)[delay:delay + len(w)]
    hp_ir = np.zeros(len(hp) + 2 * delay)
    hp_ir[:len(hp)] = hp                               # an impulse through the high-pass
    cen = CENTERS[k]

    def lin(g):
        return np.array([0.0 if v is None else 10.0 ** ((v - 20.0) / 10.0) for v in g])
    e_wave = lin(strength_db(w_lp, fs, cen))
    share_hp = lin(strength_db(hp_ir, fs, cen))
    e_echo = band_energies(hist_total)[k]
    e = e_wave + e_echo * share_hp if geometrical else e_wave
    g_echo = strength_g(band_energies(hist_total))[0][k]
    ok = e > 0
    out[np.nonzero(k)[0][ok]] = 10.0 * np.log10(e[ok]) + 20.0 - g_echo[ok]
    return out


def wave_ear_half(cfg) -> float:
    """Half spacing (m) of the wave band's ear probes: cfg.wave_ear_half if set, else the
    HRTF set's diffuse-coherence equivalent, else Kuhn's low-frequency limit of a rigid
    sphere (ITD 3a/c sin, i.e. probes 1.5 a from the centre)."""
    half = float(getattr(cfg, "wave_ear_half", 0.0) or 0.0)
    if half > 0.0:
        return half
    path = getattr(cfg, "hrtf_path", "")
    if path and str(getattr(cfg, "output_layout", "binaural")).lower() != "stereo":
        from .hrtf_sofa import load_hrtf
        return load_hrtf(path, int(cfg.fs)).wave_ear_half()
    from .hrtf import HEAD_RADIUS
    return 1.5 * HEAD_RADIUS


HEAD_PROBES = {7: 1, 19: 2}      # probes of the 'hrtf' wave heads -> ambisonic order


def _wave_head_on(cfg) -> bool:
    """An 'hrtf' / 'hrtf2' wave head is used (binaural output only)."""
    return (str(getattr(cfg, "wave_head", "ears")).lower() in ("hrtf", "hrtf2")
            and str(getattr(cfg, "output_layout", "binaural")).lower() != "stereo")


def wave_head_probes(pos, forward, up, fc: float, c: float, order: int = 1):
    """The probes of the 'hrtf' wave head: the centre, then +- h along the head's front,
    left and up (7, first order); second order adds the 12 diagonals (+-h, +-h) in the
    front-left, front-up and left-up planes (19). h = one cell of the grid (fdtd.wave_dx)."""
    from .fdtd import wave_dx
    pos = np.asarray(pos, dtype=np.float64)
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float64)
    u = u - np.dot(u, f) * f
    u = u / np.linalg.norm(u)
    left = np.cross(u, f)
    h = head_probe_spacing(fc, c, order)
    out = [pos] + [pos + s * h * a for a in (f, left, u) for s in (1.0, -1.0)]
    if int(order) >= 2:
        for a, b in ((f, left), (f, u), (left, u)):
            out += [pos + h * (sa * a + sb * b) for sa in (1.0, -1.0) for sb in (1.0, -1.0)]
    return out


def head_probe_spacing(fc: float, c: float, order: int = 1) -> float:
    """h of the wave head's probes: one cell of the grid (fdtd.wave_dx).

    Second order uses AURALIS_HEAD2_CELLS cells instead. 2 and 3 cells give the same
    free-field level differences as 1: the trilinear taps are not the limit."""
    from .fdtd import wave_dx
    k = 1.0 if int(order) < 2 else float(os.environ.get("AURALIS_HEAD2_CELLS", "1"))
    return k * wave_dx(fc, c)


def _head_order(cfg) -> int:
    return 2 if str(getattr(cfg, "wave_head", "ears")).lower() == "hrtf2" else 1


_HEAD_DECODERS: dict = {}


def wave_head_ambi(cols, fs: int, fc: float, c: float):
    """(n, 4) or (n, 9) ACN/SN3D ambisonics in the head frame from the 7 or 19 probes of
    wave_head_probes: W = centre; order 1 = pressure differences across 2h over 2j sin(kh);
    order 2 = second differences along the axes and across the diagonals, over -4 sin^2(kh/2)
    and -4 sin^2(kh) (a plane wave's values); regularised below 20 Hz, 0 above kh = pi/2."""
    from .fdtd import wave_dx
    p = np.asarray(cols, dtype=np.float64)
    n = p.shape[0]
    order = HEAD_PROBES[p.shape[1]]
    h = head_probe_spacing(fc, c, order)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1.0 / int(fs))
    kh = 2.0 * np.pi * f / c * h
    kh0 = 2.0 * np.pi * 20.0 / c * h
    band = kh < 0.5 * np.pi

    def inv(d, d0):
        """Regularised 1/d below 20 Hz, 0 above kh = pi/2."""
        return np.where(band, d / (d * d + d0 * d0), 0.0)
    g = inv(np.sin(kh), np.sin(kh0)) / 2j

    def spec(x, gain):
        return np.fft.irfft(np.fft.rfft(x, nfft) * gain, nfft)[:n]

    def grad(a, b):
        return spec(p[:, a] - p[:, b], g)
    ambi = [p[:, 0], grad(3, 4), grad(5, 6), grad(1, 2)]              # W Y Z X
    if order >= 2:
        # plane wave e^{jk u.r}: (P+ - 2 P0 + P-) = -4 sin^2(k h u_a / 2) P0 ~ u_a^2 times
        # -4 sin^2(kh/2); the four diagonals of a plane: -4 sin(k h u_a) sin(k h u_b) P0
        g2 = inv(-4.0 * np.sin(0.5 * kh) ** 2, -4.0 * np.sin(0.5 * kh0) ** 2)
        gx = inv(-4.0 * np.sin(kh) ** 2, -4.0 * np.sin(kh0) ** 2)
        xx = spec(p[:, 1] - 2.0 * p[:, 0] + p[:, 2], g2)              # front
        yy = spec(p[:, 3] - 2.0 * p[:, 0] + p[:, 4], g2)              # left
        zz = spec(p[:, 5] - 2.0 * p[:, 0] + p[:, 6], g2)              # up

        def cross(k0):
            return spec(p[:, k0] - p[:, k0 + 1] - p[:, k0 + 2] + p[:, k0 + 3], gx)
        xy, xz, yz = cross(7), cross(11), cross(15)
        r3 = np.sqrt(3.0)
        ambi += [r3 * xy, r3 * yz, 0.5 * (3.0 * zz - p[:, 0]), r3 * xz,
                 0.5 * r3 * (xx - yy)]                                 # ACN 4-8, SN3D
    return np.stack(ambi, axis=1)


def wave_head_decode(cols, cfg, fc: float, c: float):
    """(n, 3) left, right, centre of the wave band from its 7 or 19 probes: first- or
    second-order ambisonics (wave_head_ambi) decoded through the HRTF set of cfg.hrtf_path
    (the head model otherwise) by ambidecode's decoder, its latency removed."""
    from .ambidecode import decode, decoder_filters
    p = np.asarray(cols, dtype=np.float64)
    n = p.shape[0]
    order = HEAD_PROBES[p.shape[1]]
    fs = int(cfg.fs)
    ambi = wave_head_ambi(p, fs, fc, c)
    path = str(getattr(cfg, "hrtf_path", "") or "")
    key = (fs, round(float(c), 6), path, order)
    if key not in _HEAD_DECODERS:
        hset = None
        if path:
            from .hrtf_sofa import load_hrtf
            hset = load_hrtf(path, fs)
        _HEAD_DECODERS[key] = decoder_filters(order, fs, c, hrtf_set=hset)
    filt, lat = _HEAD_DECODERS[key]
    ears = decode(ambi, filt)[lat:lat + n]
    if len(ears) < n:
        ears = np.pad(ears, ((0, n - len(ears)), (0, 0)))
    return np.column_stack([ears[:, 0], ears[:, 1], p[:, 0]])


def _omni_low(ir_low, n):
    """Centre pressure of a low band (3 columns), else the mean of the ears / mono."""
    lo = np.asarray(ir_low, dtype=np.float64)
    if lo.ndim == 2:
        lo = lo[:, 2] if lo.shape[1] == 3 else lo.mean(axis=1)
    return np.pad(lo, (0, max(0, n - len(lo))))[:n]


def _fuse_ambi_w(ambi, ir_low, fc, fs):
    """Wave band into W only: the centre probe (3-column low band), else the mean
    of the ears (approximation), else the mono probe. Higher orders keep the
    geometrical solver's directions over the whole band."""
    from .fdtd import fuse_low_high
    lo = np.asarray(ir_low, dtype=np.float64)
    if lo.ndim == 2:
        lo = lo[:, 2] if lo.shape[1] == 3 else lo.mean(axis=1)
    n = ambi.shape[0]
    if len(lo) < n:
        lo = np.pad(lo, (0, n - len(lo)))
    out = ambi.copy()
    out[:, :1] = fuse_low_high(ambi[:, :1], lo[:n], fc, fs)
    return out


def low_band_setup(scene: SceneData, cfg: BakeConfig):
    """(speed of sound, duration, crossover) of the wave band, as bake() uses them."""
    from .fdtd import auto_crossover
    from .ir_synthesis import _n_samples
    c = config_speed_of_sound(cfg)
    # duration = EXACT length of the IR bake() will synthesize (same
    # _n_samples rule): at a different duration the FFT deconvolution
    # (different n_fft) makes the modal tails differ point by point -
    # measured 15 % spread for 0.24 % of duration.
    max_time = float(cfg.max_time or estimate_max_time(scene, c))
    n_bins = max(8, int(np.ceil(max_time / cfg.bin_dt)))
    duration = _n_samples(n_bins, cfg.bin_dt, cfg.fs) / float(cfg.fs)
    vol = _scene_volume_hint(scene)
    rt_e = estimate_max_time(scene, c) / 1.5
    f_sch = 2000.0 * np.sqrt(max(rt_e, 0.05) / vol)
    fc = float(cfg.fdtd_fcross)
    if fc <= 0.0:
        fc, _ = auto_crossover(vol, duration, c, f_sch,
                               order=int(cfg.fdtd_order),
                               budget_gcell=float(cfg.fdtd_budget_gcell),
                               fcross_max=float(cfg.fdtd_fcross_max))
    else:
        fc = float(np.clip(fc, 60.0, max(float(cfg.fdtd_fcross_max), 60.0)))
    return c, duration, fc


def wave_source_pattern(source: SourceDef, cfg: BakeConfig, fc: float):
    """(axis, per-band a) of a directional source for the wave band, or (None, None)
    when it is omni below ~2 fc or the option is off: then a single FDTD run."""
    if not getattr(cfg, "fdtd_source_directivity", True) or getattr(source, "surface", ""):
        return None, None
    d = as_directivity(source.directivity)
    if d is None:
        return None, None
    a = d.first_order_weights()
    if np.all(a[CENTERS <= 2.0 * fc] > 0.98):
        return None, None
    return np.asarray(source.forward, dtype=np.float64), a


def wave_source_gains_db(source: SourceDef, cfg: BakeConfig, fc: float):
    """Per-octave gains (dB, 125 Hz - 8 kHz) giving the wave band the source's power per octave.

    For a directivity table that carries the radiated spectrum (on-axis level not 1 in every
    band, e.g. the BRAS dodecahedron normalised to its spherical mean at 1 kHz). The
    geometrical solver radiates that table as is, but the wave band's monopole or first-order
    pattern has an on-axis level of 1. None for tables normalised on axis (app directivities,
    the QSC K8): the wave band is then unchanged, bit for bit. Bands above 2 fc hold the last
    value."""
    d = as_directivity(source.directivity)
    if d is None or np.allclose(d.table[:, 0], 1.0) or getattr(source, "surface", ""):
        return None
    axis, _ = wave_source_pattern(source, cfg, fc)
    mq = np.maximum(d.mean_q(), 1e-9)
    p = np.ones_like(mq) if axis is None else np.clip(mq, 0.25, 1.0)
    g = 10.0 * np.log10(mq / p)
    k = np.nonzero(CENTERS <= 2.0 * fc)[0]
    last = int(k[-1]) if len(k) else 0
    g[last + 1:] = g[last]
    return g


def _wave_spectrum(ir_low, source: SourceDef, cfg: BakeConfig, fc: float):
    """The wave band (n,) or (n, probes) filtered by wave_source_gains_db; as is when None."""
    g = wave_source_gains_db(source, cfg, fc)
    if g is None:
        return ir_low
    from .octave_fir import filter_ir, octave_fir
    print("[Auralis] wave band: spectrum of the source '%s' from its directivity table, "
          "%s dB (125 Hz-1 kHz)" % (source.name, np.round(g[:4], 2).tolist()))
    return filter_ir(np.asarray(ir_low, dtype=np.float64), octave_fir(g, cfg.fs))


def precompute_low_bands(scene: SceneData, source: SourceDef,
                         receivers: List[ReceiverDef],
                         config: BakeConfig = None) -> dict:
    """FDTD low band per source: one grid for all receivers.
    {name: (ir_low, finfo, fc)} for bake(low_band=); key absent on failure."""
    from .fdtd import run_low_band
    cfg = config or BakeConfig()
    if not cfg.fdtd or not receivers:
        return {}
    c, duration, fc = low_band_setup(scene, cfg)
    layout = str(getattr(cfg, "output_layout", "binaural")).lower()

    def probes_of(rcv):
        pos = np.asarray(rcv.position, dtype=np.float64)
        if layout == "stereo":
            return [pos]
        fwd = np.asarray(rcv.forward, dtype=np.float64)
        upv = np.asarray(rcv.up, dtype=np.float64)
        r = np.cross(fwd, upv)
        nr = np.linalg.norm(r)
        if nr <= 1e-9:
            return [pos]
        r = r / nr
        if _wave_head_on(cfg):
            return wave_head_probes(pos, fwd, upv, fc, c, _head_order(cfg))
        h = wave_ear_half(cfg)
        ears = [pos - h * r, pos + h * r]
        # ambisonic W needs the pressure at the centre: one more probe, the
        # ears' recordings are unchanged (columns: left, right, centre)
        return ears + [pos] if int(getattr(cfg, "ambisonic_order", 0)) > 0 else ears

    axis, a_bands = wave_source_pattern(source, cfg, fc)
    if axis is not None:
        print("[Auralis] wave band: directional source '%s', first-order pattern "
              "a = %s (125 Hz-1 kHz), 2 FDTD runs"
              % (source.name, np.round(a_bands[:4], 2).tolist()))

    def run_probes(plist):
        p_lo = (source.position_low if source.position_low is not None
                else source.position)
        ir_p, info_p = run_low_band(scene, p_lo, plist, fs=cfg.fs,
                                    duration=duration, f_cross=fc,
                                    speed_of_sound=c, engine=cfg.engine,
                                    max_cells=cfg.fdtd_max_cells,
                                    order=int(cfg.fdtd_order),
                                    boundary_absorber=cfg.fdtd_absorber,
                                    conformal=bool(cfg.fdtd_conformal),
                                    conformal_aniso=bool(cfg.fdtd_conformal_aniso),
                                    align_grid=bool(cfg.fdtd_align), domain_box=cfg.fdtd_box,
                        ppw=float(cfg.fdtd_ppw),
                                    src_axis=axis, src_first_order=a_bands,
                                    src_surface=source.surface or None)
        return _wave_spectrum(ir_p, source, cfg, fc), info_p

    t0 = _time.time()
    plan = [(r, probes_of(r)) for r in receivers]
    flat = [p for _, ps in plan for p in ps]
    out = {}
    try:
        ir_all, finfo = run_probes(flat)
        ir_all = np.atleast_2d(np.asarray(ir_all, dtype=np.float64))
        if ir_all.shape[0] == len(flat) and ir_all.shape[1] != len(flat):
            ir_all = ir_all.T
        off = 0
        for rcv, ps in plan:
            cols = ir_all[:, off:off + len(ps)]
            off += len(ps)
            if len(ps) in HEAD_PROBES:
                cols = wave_head_decode(cols, cfg, fc, c)
            out[rcv.name] = (cols if len(ps) > 1 else cols[:, 0],
                             finfo, fc)
        print("[Auralis] low band precomputed: source '%s', "
              "%d receivers (%d probes) in ONE grid [%s, %d cells, "
              "crossover %.0f Hz, %.1f s]"
              % (source.name, len(receivers), len(flat), finfo.engine,
                 finfo.n_air, fc, _time.time() - t0))
        return out
    except ValueError as exc:
        print("[Auralis] grouped precompute impossible (%s): "
              "per-receiver fallback" % exc)
    for rcv, ps in plan:
        try:
            ir_low, finfo = run_probes(ps)
            if len(ps) in HEAD_PROBES:
                ir_low = wave_head_decode(np.atleast_2d(np.asarray(ir_low).T).T, cfg, fc, c)
            out[rcv.name] = (ir_low, finfo, fc)
        except ValueError as exc:
            try:
                ir_low, finfo = run_probes([np.asarray(rcv.position,
                                                       dtype=np.float64)])
                out[rcv.name] = (np.asarray(ir_low)[:, 0], finfo, fc)
                print("[Auralis] receiver '%s': in-grid ears "
                      "impossible (%s), MONO low band" % (rcv.name, exc))
            except ValueError as exc2:
                print("[Auralis] receiver '%s': precomputed low band "
                      "IMPOSSIBLE (%s) - bake() will run its own pass"
                      % (rcv.name, exc2))
    if out:
        print("[Auralis] low band precomputed (fallback): source '%s', "
              "%d/%d receivers, %.1f s"
              % (source.name, len(out), len(receivers), _time.time() - t0))
    return out


# ---------------------------------------------------------------------------
# The per-source loop shared by the app's runner and the BRAS harnesses, so both
# take the same two decisions on one execution path.
# ---------------------------------------------------------------------------

def source_low_bands(scene: SceneData, source: SourceDef,
                     receivers: List[ReceiverDef], config: BakeConfig = None,
                     fallback: bool = False):
    """Wave band of one source for all its receivers (one FDTD grid, see
    precompute_low_bands). Returns (low, error). With `fallback`, an unstable
    wave band (the watchdog's RuntimeError) gives ({}, message) and that
    source's pairs stay geometrical; without it the error propagates (a
    validation bench must stop)."""
    try:
        return precompute_low_bands(scene, source, receivers, config), None
    except RuntimeError as exc:
        if not fallback:
            raise
        return {}, str(exc)


def bake_pair(scene: SceneData, source: SourceDef, receiver: ReceiverDef,
              config: BakeConfig = None, low: Optional[dict] = None,
              rcv_radius: Optional[float] = None, own_wave_pass: bool = True,
              progress: Optional[Callable[[float, str], Optional[bool]]] = None
              ) -> BakeResult:
    """One pair with its source's precomputed wave band `low` (the dict of
    source_low_bands). A pair whose wave band is missing runs bake()'s own FDTD
    pass, or with own_wave_pass=False the geometrical solver alone (the app:
    that pass is the one that just failed). rcv_radius overrides
    config.rcv_radius (smaller spheres near walls)."""
    cfg = config or BakeConfig()
    lb = (low or {}).get(receiver.name)
    if cfg.fdtd and lb is None and not own_wave_pass:
        if cfg.wave_only:
            raise RuntimeError("wave-only output: no wave band for %s -> %s (it failed)"
                               % (source.name, receiver.name))
        cfg = replace(cfg, fdtd=False)
    if rcv_radius is not None and rcv_radius != cfg.rcv_radius:
        cfg = replace(cfg, rcv_radius=rcv_radius)
    return bake(scene, source, receiver, cfg, progress=progress, low_band=lb)


def apply_source_directivity(events: List[AcousticEvent],
                             dirv: Directivity, src_forward) -> None:
    """Scale event amplitudes by sqrt(Q_b(theta)) according to their
    emission direction from the source."""
    fwd = np.asarray(src_forward, dtype=np.float64)
    for ev in events:
        e = getattr(ev, "emit_dir", None)
        if e is None:
            continue
        ct = float(np.clip(np.dot(fwd, np.asarray(e, dtype=np.float64)),
                           -1.0, 1.0))
        ev.amps = ev.amps * dirv.amp(ct)
        if getattr(ev, "incoherent", None) is not None:
            ev.incoherent = ev.incoherent * dirv.amp(ct) ** 2


def add_incoherent(events, hist, dirs, bin_dt) -> None:
    """Add the energy that rough grounds take from the coherent reflections to the echogram,
    at their arrival time and from their direction. For energy metrics (receiver grids): the
    impulse responses render it with its own spectrum (ir_synthesis.scattered_part)."""
    n_bins = hist.shape[1]
    for ev in events:
        e = getattr(ev, "incoherent", None)
        if e is None:
            continue
        k = int(ev.time / bin_dt)
        if 0 <= k < n_bins:
            hist[:, k] += e
            if dirs is not None:
                dirs[k] += float(np.sum(e)) * np.asarray(ev.direction, dtype=np.float64)


def native_available() -> bool:
    return _native is not None and _native.is_available()


def gpu_available() -> bool:
    return _native is not None and _native.gpu_available()
