# -*- coding: utf-8 -*-
"""Moving sources and receivers: time-varying-filter binaural render.
Late field baked at key frames then cross-faded, early part recomputed
at every hop, optional direct path as a delay line (Doppler)."""

from __future__ import annotations

import dataclasses
import time as _time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .air import air_attenuation_db_per_m
from .air import config_speed_of_sound
from .bands import N_BANDS
from .directivity import as_directivity
from .diffraction import compute_diffraction, extract_edges
from .geometry import NumpyRaycaster, SceneData, extract_planes
from .reflection import (build_admittance_table,
                         event_reflection_spectrum)
from .ir_synthesis import _event_signs
from .hrtf import (FIR_TAPS, _FIR_NFFT, event_fir_pair, event_fir_stereo,
                   itd_seconds, late_tail_binaural, late_tail_stereo,
                   place_fir, receiver_basis)
from .ism import compute_direct, compute_image_sources
from .pipeline import (BakeConfig, ReceiverDef, SourceDef,
                       apply_source_directivity, effective_max_bounces, estimate_max_time,
                       wave_ear_half, _select_engine)

__all__ = ["Trajectory", "MovingSource", "MovingReceiver", "MotionConfig",
           "render_moving", "render_scene_moving"]


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def _unit_rows(a, fallback):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3).copy()
    n = np.linalg.norm(a, axis=1, keepdims=True)
    bad = n[:, 0] < 1e-9
    a[bad] = fallback
    n[bad] = 1.0
    return a / n


@dataclass
class Trajectory:
    """Positions (and optional axes) sampled at increasing times.
    Sampling at an arbitrary time is linear."""
    times: np.ndarray                       # (K,)
    positions: np.ndarray                   # (K, 3)
    forwards: Optional[np.ndarray] = None   # (K, 3)
    ups: Optional[np.ndarray] = None        # (K, 3)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.positions = np.asarray(self.positions,
                                    dtype=np.float64).reshape(-1, 3)
        if len(self.times) != len(self.positions) or len(self.times) == 0:
            raise ValueError("empty or inconsistent trajectory")
        if len(self.times) > 1 and np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")
        if self.forwards is not None:
            self.forwards = _unit_rows(self.forwards, (0.0, 0.0, -1.0))
        if self.ups is not None:
            self.ups = _unit_rows(self.ups, (0.0, 0.0, 1.0))

    @classmethod
    def constant(cls, position, forward=(0.0, 0.0, -1.0),
                 up=(0.0, 0.0, 1.0)) -> "Trajectory":
        p = np.asarray(position, dtype=np.float64)
        return cls(times=np.array([0.0]), positions=p[None, :],
                   forwards=np.asarray(forward, float)[None, :],
                   ups=np.asarray(up, float)[None, :])

    def _interp(self, table, t):
        t = np.asarray(t, dtype=np.float64)
        if len(self.times) == 1:
            out = np.broadcast_to(table[0], t.shape + (3,)).copy()
            return out
        return np.stack([np.interp(t, self.times, table[:, i])
                         for i in range(3)], axis=-1)

    def position(self, t) -> np.ndarray:
        return self._interp(self.positions, t)

    def forward(self, t) -> np.ndarray:
        if self.forwards is None:
            return np.broadcast_to(np.array([0.0, 0.0, -1.0]),
                                   np.shape(t) + (3,)).copy()
        f = self._interp(self.forwards, t)
        return _unit_rows(f, (0.0, 0.0, -1.0)).reshape(f.shape)

    def up(self, t) -> np.ndarray:
        if self.ups is None:
            return np.broadcast_to(np.array([0.0, 0.0, 1.0]),
                                   np.shape(t) + (3,)).copy()
        u = self._interp(self.ups, t)
        return _unit_rows(u, (0.0, 0.0, 1.0)).reshape(u.shape)

    @property
    def t_end(self) -> float:
        return float(self.times[-1])


@dataclass
class MovingSource:
    name: str
    trajectory: Trajectory
    gain_db: float = 0.0
    directivity: Optional[object] = None
    dry_path: Optional[str] = None


@dataclass
class MovingReceiver:
    name: str
    trajectory: Trajectory


def _as_moving_receiver(r) -> MovingReceiver:
    if isinstance(r, MovingReceiver):
        return r
    if isinstance(r, ReceiverDef):
        return MovingReceiver(r.name, Trajectory.constant(
            r.position, r.forward, r.up))
    raise TypeError("invalid receiver: %r" % (r,))


def _as_moving_source(s) -> MovingSource:
    if isinstance(s, MovingSource):
        return s
    if isinstance(s, SourceDef):
        return MovingSource(s.name, Trajectory.constant(
            s.position, s.forward), gain_db=s.gain_db,
            directivity=s.directivity, dry_path=s.dry_path)
    raise TypeError("invalid source: %r" % (s,))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MotionConfig:
    base: BakeConfig = field(default_factory=BakeConfig)
    hop: int = 2048            # samples per half-window (~43 ms @48k)
    reverb_dt: float = 0.75    # s between late-field key frames
    doppler: bool = True       # direct path as a variable delay line
    tail_rays: Optional[int] = None   # rays per key frame (def: base.n_rays)
    # Late field convolved once per key frame with the input weighted by its
    # cross-fade: an exact rewrite of the per-hop cross-fade, far cheaper.
    # False = the per-hop reference (every hop convolves the full-length IR).
    tail_by_keys: bool = True
    # wave band (FDTD) below the crossover, from static IRs at key frames
    # cross-faded like the late field (render_scene_moving); 0 = key spacing
    # from the speed: the path moves at most lambda/8 at the crossover
    wave: bool = False
    wave_dt: float = 0.0
    # source AND receiver moving: two runs (one per ear) per receiver key frame; above this
    # many runs the receiver keys are spread out, with a loud warning
    wave_max_runs: int = 64
    # Align every key IR on the direct-sound delay of the current geometry before the
    # cross-fade (a variable delay line per key, like the Doppler of the direct path). It is
    # exact for the direct sound only. It divides the error by 4 in free field at a
    # 1-wavelength key spacing, but doubles it in a street and multiplies it by 6-9 in a
    # room at 1/2-1 wavelength, because the reflections are shifted too. Off by default.
    wave_align: bool = False
    # Key spacing of the receiver when it and a source both move (two runs per key), as a
    # fraction of the wavelength at the crossover. Source keys (probes, free) stay at 1/8.
    # 1/4 differs from 1/16 by 0.03 % in a street and 0.02 % in a room.
    wave_rkey_lambda: float = 0.25


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ir_len(n_bins, bin_dt, fs):
    return int(np.ceil(n_bins * bin_dt * fs)) + FIR_TAPS + 8


def _seg(x, a0, L):
    """x[a0:a0+L] with zero padding out of bounds (a0 may be negative)."""
    out = np.zeros(L)
    b0 = max(0, a0)
    b1 = min(len(x), a0 + L)
    if b1 > b0:
        out[b0 - a0:b1 - a0] = x[b0:b1]
    return out


def _add(out, y, base, ch):
    """out[base:base+len(y), ch] += y, clipped on both sides."""
    s0 = max(0, -base)
    b0 = max(0, base)
    ln = min(len(y) - s0, len(out) - b0)
    if ln > 0:
        out[b0:b0 + ln, ch] += y[s0:s0 + ln]


def render_moving(scene: SceneData, source, receiver, dry: np.ndarray,
                  mconfig: MotionConfig = None,
                  progress: Optional[Callable[[float, str],
                                              Optional[bool]]] = None,
                  low_keys=None) -> Tuple[np.ndarray, dict]:
    """Render `dry` (mono, at base.fs) emitted by a source (static or
    moving) to a receiver (static or moving). Returns (out (n, 2), info).
    low_keys = (key times, IRs (K, n, 2), crossover, their rate) from
    wave_band_keys: the wave band replaces the geometry below the crossover."""
    mc = mconfig or MotionConfig()
    cfg = mc.base
    if getattr(cfg, "fdtd", False):
        print("[Auralis] FDTD skipped in motion (v1: static sources)")
        import dataclasses as _dc
        cfg = _dc.replace(cfg, fdtd=False)
    fs = int(cfg.fs)
    msrc = _as_moving_source(source)
    mrcv = _as_moving_receiver(receiver)
    dry = np.asarray(dry, dtype=np.float64).reshape(-1)
    if dry.ndim != 1 or len(dry) == 0:
        raise ValueError("a non-empty mono dry signal is required")
    dur = len(dry) / fs
    timings = {}

    def tick(fr, st):
        if progress is not None and progress(fr, st) is False:
            raise KeyboardInterrupt("render cancelled")

    # ---- shared context ---------------------------------------------------
    t0 = _time.time()
    air = air_attenuation_db_per_m(temperature_c=cfg.temperature_c,
                                   humidity_percent=cfg.humidity_percent,
                                   pressure_kpa=cfg.pressure_kpa)
    c = config_speed_of_sound(cfg)
    max_time = cfg.max_time or estimate_max_time(scene, c)
    n_bins = max(8, int(np.ceil(max_time / cfg.bin_dt)))
    _n_bounce = effective_max_bounces(scene, cfg, max_time, c)
    n_ir = _ir_len(n_bins, cfg.bin_dt, fs)
    engine_name, engine, tracer = _select_engine(scene, cfg.engine)
    planes, tri_plane = extract_planes(scene, cfg.max_planes)
    edges = (extract_edges(scene, max_edges=cfg.max_edges)
             if getattr(cfg, "diffraction", True) else [])
    refl_a = np.sqrt(scene.reflectance())
    dirv = as_directivity(msrc.directivity)
    g_lin = 10.0 ** (msrc.gain_db / 20.0)
    tail_seed = cfg.seed ^ 0x5DEECE66D
    n_rays = int(mc.tail_rays or cfg.n_rays)
    rbs_dims = None
    if getattr(cfg, "rbs_scattering", False):
        from . import rbs as _rbs
        from .bands import CENTERS as _BC
        from .geometry import surface_dims
        rbs_dims = surface_dims(scene)
    timings["setup_s"] = _time.time() - t0

    # ---- late-field key frames --------------------------------------------
    t0 = _time.time()
    n_keys = max(2, int(np.ceil(dur / max(mc.reverb_dt, 1e-3))) + 1)
    key_t = np.linspace(0.0, max(dur, 1e-6), n_keys)
    stereo_out = (getattr(cfg, "output_layout", "binaural") == "stereo")
    hset = None
    if not stereo_out and getattr(cfg, "hrtf_path", ""):
        try:
            from .hrtf_sofa import load_hrtf
            hset = load_hrtf(cfg.hrtf_path, fs)
        except Exception as exc:
            print("[Auralis] HRTFs ignored (%s), parametric model" % exc)
    n_taps_ev = (FIR_TAPS if hset is None
                 else min(2 * int(hset.n_taps) + 1, 449))
    tails = []
    for j, tj in enumerate(key_t):
        ps = msrc.trajectory.position(tj)
        pr = mrcv.trajectory.position(tj)
        fw = mrcv.trajectory.forward(tj)
        upv = mrcv.trajectory.up(tj)
        sf = msrc.trajectory.forward(tj)
        if engine is not None:           # native CPU or CUDA: same trace API
            hist, dirs, _st = engine.trace(
                ps, pr, n_rays, _n_bounce, max_time, cfg.bin_dt,
                n_bins, c, air, spec_exclude_order=cfg.ism_order,
                energy_cap=cfg.energy_cap, seed=cfg.seed,
                n_threads=cfg.n_threads, rcv_radius=cfg.rcv_radius,
                src_forward=(sf if dirv is not None else None),
                dir_table=(dirv.table if dirv is not None else None),
                rbs=rbs_dims)
        else:
            from .stochastic import trace_python
            hist, dirs, _st = trace_python(
                scene, ps, pr, n_rays, _n_bounce, max_time,
                cfg.bin_dt, n_bins, c, air,
                spec_exclude_order=cfg.ism_order,
                energy_cap=cfg.energy_cap, seed=cfg.seed,
                src_forward=(sf if dirv is not None else None),
                dir_table=(dirv.table if dirv is not None else None),
                rcv_radius=cfg.rcv_radius, rbs=rbs_dims)
        hist *= g_lin * g_lin
        _f, _u, right = receiver_basis(fw, upv)
        rng = np.random.default_rng(tail_seed)      # same noise at each key
        if stereo_out:
            tl, tr = late_tail_stereo(hist, dirs, cfg.bin_dt, fs, n_ir,
                                      right, cfg.stereo_base_deg, rng)
        else:
            tl, tr = late_tail_binaural(hist, dirs, cfg.bin_dt, fs, n_ir,
                                        right, c, rng,
                                        rho=None if hset is None else hset.band_coherence())
            if hset is not None:                     # as in synthesize_binaural
                dfl, dfr = hset.diffuse_field_firs()
                tl = np.convolve(tl, dfl)[:n_ir]
                tr = np.convolve(tr, dfr)[:n_ir]
        tails.append(np.stack([tl, tr], axis=1))
        tick(0.05 + 0.35 * (j + 1) / n_keys,
             "late field %d / %d" % (j + 1, n_keys))
    timings["tail_keys_s"] = _time.time() - t0

    # ---- hops: early IR + tail cross-fade, time-varying convolution ------
    t0 = _time.time()
    H = int(max(256, mc.hop))
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(2 * H) / (2 * H))
    n_hops = int(np.ceil(len(dry) / H))
    n_out = len(dry) + n_ir + 2 * H
    out = np.zeros((n_out, 2))
    center = n_taps_ev // 2

    hop_tc = (np.arange(n_hops) * H + H) / fs
    by_keys = bool(mc.tail_by_keys)

    def key_pos(tc):
        """(key index j0, weight a of key j0 + 1) of the late field at time tc."""
        u = tc / max(key_t[-1], 1e-9) * (n_keys - 1)
        j0 = int(np.clip(np.floor(u), 0, n_keys - 2))
        return j0, float(np.clip(u - j0, 0.0, 1.0))

    if by_keys:
        # cross-fade weight of each key over its own support only (memory ~ 2 x dry)
        lo = np.full(n_keys, len(dry))
        hi = np.zeros(n_keys, dtype=np.int64)
        for k in range(-1, n_hops):
            j0, _a = key_pos(float(hop_tc[max(k, 0)]))
            b0, b1 = max(0, k * H), min(len(dry), k * H + 2 * H)
            for j in (j0, j0 + 1):
                lo[j], hi[j] = min(lo[j], b0), max(hi[j], b1)
        key_w = [np.zeros(max(0, int(hi[j] - lo[j]))) for j in range(n_keys)]

    dir_amp_d = np.zeros((n_hops, N_BANDS))       # direct amps * distance
    dir_vec = np.zeros((n_hops, 3))               # arrival (rcv -> source)
    itdLR = np.zeros((n_hops, 2))
    rights = np.zeros((n_hops, 3))    
    fwds = np.zeros_like(rights)
    ups = np.zeros_like(rights)

    for k in range(-1, n_hops):
        kk = max(k, 0)
        tc = float(hop_tc[kk])
        ps = msrc.trajectory.position(tc)
        pr = mrcv.trajectory.position(tc)
        fw = mrcv.trajectory.forward(tc)
        upv = mrcv.trajectory.up(tc)
        sf = msrc.trajectory.forward(tc)
        _f, _u, right = receiver_basis(fw, upv)
        rights[kk] = right
        fwds[kk] = np.asarray(fw, dtype=np.float64)
        ups[kk] = np.asarray(upv, dtype=np.float64)

        adm = (build_admittance_table(
            scene.absorption,
            getattr(scene, 'mat_phys', None))   # cached
               if getattr(cfg, "complex_reflection", True) else None)
        events = [compute_direct(scene, tracer, ps, pr, air, c)]
        if cfg.ism_order > 0:
            events += compute_image_sources(scene, planes, tri_plane, tracer,
                                            ps, pr, cfg.ism_order, air, c,
                                            adm_table=adm)
        if edges:
            events += compute_diffraction(scene, tracer, edges, ps, pr,
                                          air, c, refl_amp=refl_a)
        if rbs_dims is not None:
            _rbs.weight_image_sources(events, scene, rbs_dims, _BC, c, ps, pr)
        events = [ev for ev in events if ev.time < max_time]
        if dirv is not None:
            apply_source_directivity(events, dirv, sf)
        if g_lin != 1.0:
            for ev in events:
                ev.amps = ev.amps * g_lin

        d_ev = events[0]
        d_dist = max(float(np.linalg.norm(ps - pr)), 1e-3)
        dir_amp_d[kk] = d_ev.amps * d_dist
        dir_vec[kk] = d_ev.direction
        if stereo_out:
            itdLR[kk] = (0.0, 0.0)
        else:
            tau, near_right = itd_seconds(d_ev.direction, right, c)
            itdLR[kk] = (tau, 0.0) if near_right else (0.0, tau)

        ir_events = events[1:] if mc.doppler else events
        off = 1 if mc.doppler else 0
        sgn = _event_signs(events, cfg.seed ^ 0x5DEECE66D)
        placed = []
        for iev, ev in enumerate(ir_events):
            if float(np.max(ev.amps)) <= 0.0:
                continue
            spec = None
            if adm is not None and getattr(ev, "bounces", None):
                spec = event_reflection_spectrum(
                    np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs), ev.bounces, adm)
            if stereo_out:
                fl, fr2, dL, dR = event_fir_stereo(
                    ev.amps, ev.direction, right, fs,
                    cfg.stereo_base_deg, spec_mul=spec)
            else:
                fl, fr2, dL, dR = event_fir_pair(
                    ev.amps, ev.direction, right, fs, c,
                    n_taps=n_taps_ev, spec_mul=spec,
                    hrtf_set=hset, basis=(fw, upv, right))
            sg = sgn[iev + off]
            placed.append((sg * fl, (ev.time + dL) * fs - center,
                           sg * fr2, (ev.time + dR) * fs - center))
        n_e = n_ir
        if by_keys:
            # shortest buffer that changes no placement (place_fir clips at the end)
            last = max([max(pl, pr) + max(len(f1), len(f2))
                        for f1, pl, f2, pr in placed] + [0.0])
            n_e = int(min(n_ir, np.ceil(last) + 8))
        early = np.zeros((n_e, 2))
        for f1, pl, f2, pr in placed:
            place_fir(early[:, 0], f1, pl)
            place_fir(early[:, 1], f2, pr)

        j0, a = key_pos(tc)
        base = k * H
        if by_keys:
            ir_k = early
            b0, b1 = max(0, base), min(len(dry), base + 2 * H)
            if b1 > b0:
                for j, w in ((j0, 1.0 - a), (j0 + 1, a)):
                    key_w[j][b0 - lo[j]:b1 - lo[j]] += w * win[b0 - base:b1 - base]
        else:
            ir_k = early + (1.0 - a) * tails[j0] + a * tails[j0 + 1]

        xw = _seg(dry, k * H, 2 * H) * win
        n_c = 2 * H + len(ir_k) - 1
        X = np.fft.rfft(xw, n_c)
        for ch in range(2):
            y = np.fft.irfft(X * np.fft.rfft(ir_k[:, ch], n_c), n_c)
            _add(out, y, base, ch)
        if (k & 7) == 0:
            tick(0.40 + 0.45 * (k + 2) / (n_hops + 1),
                 "hops %d / %d" % (k + 2, n_hops + 1))
    if by_keys:
        for j in range(n_keys):
            if len(key_w[j]) == 0 or not key_w[j].any():
                continue
            b0, b1 = int(lo[j]), int(hi[j])
            z = dry[b0:b1] * key_w[j]
            n_c = 1 << int(np.ceil(np.log2(len(z) + n_ir - 1)))
            Z = np.fft.rfft(z, n_c)
            for ch in range(2):
                y = np.fft.irfft(Z * np.fft.rfft(tails[j][:, ch], n_c), n_c)
                _add(out, y[:len(z) + n_ir - 1], b0, ch)
    timings["hops_s"] = _time.time() - t0

    # ---- direct path: variable delay line (Doppler) -----------------------
    t0 = _time.time()
    if mc.doppler:
        n_dir = len(dry) + int(np.ceil((max_time + 0.05) * fs)) + 2 * H
        n_dir = min(n_dir, n_out)
        tt = np.arange(n_dir) / fs
        pr_t = mrcv.trajectory.position(tt)
        ps_t = msrc.trajectory.position(tt)
        d0 = np.linalg.norm(ps_t - pr_t, axis=1)
        tau = d0 / c
        for _ in range(2):                       # retarded emission time
            ps_e = msrc.trajectory.position(tt - tau)
            tau = np.linalg.norm(ps_e - pr_t, axis=1) / c
        d_t = np.maximum(tau * c, 1e-3)
        inv_d = 1.0 / d_t

        itdL = np.interp(tt, hop_tc, itdLR[:, 0])
        itdR = np.interp(tt, hop_tc, itdLR[:, 1])

        for ch, itd in ((0, itdL), (1, itdR)):
            idx = (tt - tau - itd) * fs
            i0 = np.floor(idx).astype(np.int64)
            frv = idx - i0
            ok = (i0 >= 0) & (i0 < len(dry) - 1)
            sig = np.zeros(n_dir)
            sig[ok] = (dry[i0[ok]] * (1.0 - frv[ok])
                       + dry[i0[ok] + 1] * frv[ok])
            sig *= inv_d

            extra = int(np.ceil((n_dir - len(dry)) / H)) + 1
            for k in range(-1, n_hops + extra):
                kk = int(np.clip(k, 0, n_hops - 1))
                amps = dir_amp_d[kk]
                if float(amps.max()) <= 0.0:
                    continue
                if stereo_out:
                    fl, fr2, _dL, _dR = event_fir_stereo(
                        amps, dir_vec[kk], rights[kk], fs,
                        cfg.stereo_base_deg)
                else:
                    fl, fr2, _dL, _dR = event_fir_pair(
                        amps, dir_vec[kk], rights[kk], fs, c,
                        n_taps=n_taps_ev, hrtf_set=hset,
                        basis=(fwds[kk], ups[kk], rights[kk])
                        if hset is not None else None)
                fir = fl if ch == 0 else fr2
                a0 = k * H
                if a0 >= n_dir:
                    break
                xw = _seg(sig, a0, 2 * H) * win
                y = np.fft.irfft(np.fft.rfft(xw, 2 * H + FIR_TAPS - 1)
                                 * np.fft.rfft(fir, 2 * H + FIR_TAPS - 1),
                                 2 * H + FIR_TAPS - 1)
                _add(out, y, a0 - center, ch)
        tick(0.92, "direct path (Doppler)")
    timings["direct_s"] = _time.time() - t0

    if engine is not None:
        engine.close()
    if low_keys is not None:
        t0 = _time.time()
        tick(0.94, "wave band: %d key frames" % len(low_keys[0]))
        out = _mix_wave_band(out, dry, low_keys, fs, g_lin,
                             delays=(lambda tt: _direct_delays(msrc, mrcv, tt, c,
                                                               wave_ear_half(cfg)))
                             if mc.wave_align else None)
        timings["wave_s"] = _time.time() - t0
    scale = 1.0
    if cfg.normalize:
        peak = float(np.abs(out).max())
        if peak > 0:
            scale = 10 ** (-1.0 / 20.0) / peak
            out *= scale
    tick(1.0, "done")
    info = {"fs": fs, "n_keys": n_keys, "n_hops": n_hops, "n_ir": n_ir,
            "engine_used": engine_name, "timings": timings, "scale": scale,
            "max_time_s": max_time, "doppler": bool(mc.doppler),
            "wave_keys": None if low_keys is None else len(low_keys[0])}
    return out, info


# ---------------------------------------------------------------------------
# Wave band (FDTD) of animated renders
# ---------------------------------------------------------------------------



def _is_static(traj) -> bool:
    if float(np.ptp(traj.positions, axis=0).max()) > 1e-6:
        return False
    return traj.forwards is None or float(np.ptp(traj.forwards, axis=0).max()) < 1e-6


def _ears_at(traj, t, half) -> np.ndarray:
    """(n, 2, 3) left and right ear probes of a trajectory at times t, `half` metres from
    the centre (pipeline.wave_ear_half)."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    p, fw, up = traj.position(t), traj.forward(t), traj.up(t)
    r = np.cross(fw, up)
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
    return np.stack([p - half * r, p + half * r], axis=1)


def _wave_key_times(traj, dur, fc, c, dt_max, dt_fixed=0.0, frac=0.125) -> np.ndarray:
    """Key times of the wave band: between two keys the point moves at most `frac`
    wavelength at the crossover (1/8 by default), so that cross-faded static IRs stay
    coherent."""
    if _is_static(traj):
        return np.array([0.0])
    if dt_fixed > 0.0:
        dt = float(dt_fixed)
    else:
        v = (np.linalg.norm(np.diff(traj.positions, axis=0), axis=1)
             / np.maximum(np.diff(traj.times), 1e-9))
        dt = float(np.clip(frac * c / (fc * max(float(v.max()), 1e-6)), 0.02, dt_max))
    return np.linspace(0.0, dur, max(2, int(np.ceil(dur / dt)) + 1))


def _moving_pattern(m, cfg, fc):
    """Per-band weights a of a moving source's first-order wave-band pattern
    a + (1 - a) cos(theta) (pipeline.wave_source_pattern), or None: omni."""
    from .pipeline import wave_source_pattern
    _axis, a = wave_source_pattern(SourceDef(name=m.name, position=np.zeros(3),
                                             directivity=m.directivity), cfg, fc)
    return a


def _key_probes(m, tk, a, h):
    """Probes of a source's key frames tk: its positions; for a directional source (a)
    also +-h/2 along its forward axis at each key, the reciprocal of the grid dipole
    of the static source (fdtd._dipole_taps, h = 2 dx): (3K, 3) then."""
    P = np.atleast_2d(m.trajectory.position(tk))
    if a is None:
        return P
    U = np.atleast_2d(m.trajectory.forward(tk))
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    return np.concatenate([P, P + 0.5 * h * U, P - 0.5 * h * U])


def _mono_points(probe_sets, keys):
    """The key positions alone (first len(k) rows of each _key_probes set): the domain box
    of the run, so that dipole probes read the grid an omni source would."""
    return [q for ps, k in zip(probe_sets, keys) for q in ps[:len(k)]]


def _key_irs(cols, a, h, c, fs, missing=()):
    """(K, n) key IRs from the (3K or K, n) probe IRs of _key_probes: the pressure, plus
    for a directional source the first-order pattern a P + (1 - a) D^-1 (P+ - P-) per
    frequency, as fdtd.run_low_band combines its monopole and dipole runs. A key whose
    dipole probes left the air (in `missing`, indices into cols) keeps its pressure:
    omni there, counted in the second value."""
    if a is None:
        return cols, 0
    from .fdtd import _first_order_filters
    K = cols.shape[0] // 3
    mono, dif = cols[:K], cols[K:2 * K] - cols[2 * K:]
    n = cols.shape[1]
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    wa, wb, dinv = _first_order_filters(np.fft.rfftfreq(nf, 1.0 / fs), a, h, c)
    out = np.fft.irfft(wa * np.fft.rfft(mono, nf, axis=1)
                       + wb * dinv * np.fft.rfft(dif, nf, axis=1), nf, axis=1)[:, :n]
    bad = sorted({int(i) % K for i in missing if i >= K})
    if bad:
        out[bad] = mono[bad]
    return out, len(bad)


def wave_band_keys(scene, sources, receiver, mc: MotionConfig, durations,
                   progress=None):
    """_wave_band_keys with the grid kept between its runs (fdtd.keep_grids): uploaded once,
    each run only places its source and probes; bit for bit the same IRs."""
    from .fdtd import keep_grids
    with keep_grids():
        return _wave_band_keys(scene, sources, receiver, mc, durations, progress)


def _wave_band_keys(scene, sources, receiver, mc: MotionConfig, durations,
                    progress=None):
    """Wave-band IRs at key frames, one entry per source.

    Each entry is (key times, IRs (K, n, 2), crossover, their rate), or (None, reason).
    Static receiver: by reciprocity, one run per ear covers every key of every source
    (omni wave band, exact to 0.4 % without the conformal correction, 8-11 % with it).
    Static sources: one run per source, probes at the ears along the receiver's keys.
    Both moving: reciprocity at the receiver's key frames, two runs (one per ear) per key,
    probes at the source positions of that key's time window. Each source key cross-fades
    the two receiver keys around it, and the mix cross-fades the source keys: a bilinear
    interpolation over the two paths."""
    from .fdtd import run_low_band, wave_dx
    from .pipeline import low_band_setup
    cfg = mc.base
    c, duration, fc = low_band_setup(scene, cfg)
    half = wave_ear_half(cfg)
    h_dip = 2.0 * wave_dx(fc, c)
    fs_low = int(min(cfg.fs, max(1000, round(8.0 * fc))))
    kw = dict(fs=fs_low, duration=duration, f_cross=fc, speed_of_sound=c,
              engine=cfg.engine, max_cells=cfg.fdtd_max_cells,
              order=int(cfg.fdtd_order), boundary_absorber=cfg.fdtd_absorber,
              conformal=bool(cfg.fdtd_conformal),
              conformal_aniso=bool(cfg.fdtd_conformal_aniso),
              align_grid=bool(getattr(cfg, "fdtd_align", False)))
    mr = _as_moving_receiver(receiver)
    ms = [_as_moving_source(s) for s in sources]
    pats = [_moving_pattern(m, cfg, fc) for m in ms]
    for m, a in zip(ms, pats):
        if a is not None:
            print("[Auralis] animated wave band: directional source '%s', first-order pattern "
                  "a = %s (125 Hz-1 kHz) along its path's forward axis: 3 probes per key "
                  "frame" % (m.name, np.round(a[:4], 2).tolist()))
    n_omni = [0] * len(ms)
    dt_max = max(0.02, float(mc.reverb_dt))

    def tick(frac, stage):
        if progress is not None and progress(frac, stage) is False:
            raise KeyboardInterrupt("render cancelled")

    out = []
    if _is_static(mr.trajectory):
        ears = _ears_at(mr.trajectory, 0.0, half)[0]
        keys = [_wave_key_times(m.trajectory, d, fc, c, dt_max, mc.wave_dt)
                for m, d in zip(ms, durations)]
        pl = [_key_probes(m, k, a, h_dip) for m, k, a in zip(ms, keys, pats)]
        probes = np.concatenate(pl)
        offs = np.cumsum([0] + [len(x) for x in pl])
        per = [[] for _ in ms]
        for e in (0, 1):
            tick(0.5 * e, "wave band: %s ear, %d probes (reciprocity)"
                 % (("left", "right")[e], len(probes)))
            ir, info = run_low_band(scene, ears[e], [p for p in probes],
                                    missing_rcv_ok=any(a is not None for a in pats),
                                    domain_points=_mono_points(pl, keys), **kw)
            ir = np.asarray(ir, dtype=np.float64).reshape(len(ir), -1)
            miss = np.asarray(getattr(info, "missing", ()), dtype=np.int64)
            for j, a in enumerate(pats):
                lo, hi = offs[j], offs[j + 1]
                mj = miss[(miss >= lo) & (miss < hi)] - lo
                kir, nb = _key_irs(ir[:, lo:hi].T, a, h_dip, c, fs_low, mj)
                n_omni[j] = max(n_omni[j], nb)
                per[j].append(kir)
        _warn_omni_keys(ms, n_omni, keys)
        for j, k in enumerate(keys):
            out.append((k, np.ascontiguousarray(np.stack(per[j], axis=2)), fc, fs_low))
        return out, ("reciprocity: 2 runs (one per ear), %d key frames, crossover %.0f Hz"
                     % (sum(len(k) for k in keys), fc))
    if all(_is_static(m.trajectory) for m in ms):
        keys = _wave_key_times(mr.trajectory, max(durations), fc, c, dt_max, mc.wave_dt)
        ears = _ears_at(mr.trajectory, keys, half).reshape(-1, 3)
        for i, m in enumerate(ms):
            tick(i / len(ms), "wave band: source %d / %d, %d key frames"
                 % (i + 1, len(ms), len(keys)))
            axis = None if pats[i] is None else np.atleast_2d(m.trajectory.forward(0.0))[0]
            ir, _info = run_low_band(scene, np.atleast_2d(m.trajectory.position(0.0))[0],
                                     [p for p in ears], src_axis=axis,
                                     src_first_order=pats[i], **kw)
            ir = np.asarray(ir, dtype=np.float64).reshape(len(ir), len(keys), 2)
            out.append((keys, np.ascontiguousarray(np.transpose(ir, (1, 0, 2))), fc, fs_low))
        return out, ("%d runs (one per source), %d key frames, crossover %.0f Hz"
                     % (len(ms), len(keys), fc))
    rkeys = _wave_key_times(mr.trajectory, max(durations), fc, c, dt_max, mc.wave_dt,
                            frac=float(mc.wave_rkey_lambda))
    cap = max(2, int(mc.wave_max_runs) // 2)
    if len(rkeys) > cap:
        # spread the receiver keys: coherence of the cross-fade is kept up to ~1/8
        # wavelength of motion between keys, beyond it the crossover region colours
        step = float(np.max(np.linalg.norm(np.diff(
            mr.trajectory.position(np.linspace(0.0, max(durations), cap)), axis=0), axis=1)))
        print("[Auralis] WARNING: animated wave band: the receiver and a source both move; "
              "%d receiver key frames would take %d runs, spread to %d (%d runs, up to "
              "%.2f m between keys = %.2f wavelength at the %.0f Hz crossover; above ~0.12 the "
              "cross-fade colours the top of the wave band). Raise wave_max_runs or set wave_dt."
              % (len(rkeys), 2 * len(rkeys), cap, 2 * cap, step, step * fc / c, fc), flush=True)
        rkeys = np.linspace(0.0, max(durations), cap)
    K = len(rkeys)
    skeys = []
    for m, d in zip(ms, durations):
        sk = _wave_key_times(m.trajectory, d, fc, c, dt_max, mc.wave_dt)
        skeys.append(np.unique(np.concatenate([sk, rkeys[rkeys <= d]])))
    ears = _ears_at(mr.trajectory, rkeys, half)                       # (K, 2, 3)
    acc = [None] * len(ms)
    eye = np.eye(K)
    for k in range(K):
        wk = [np.interp(sk, rkeys, eye[k]) for sk in skeys]       # hat weights of key k
        sel = [np.flatnonzero(w > 0.0) for w in wk]
        pts = [_key_probes(m, sk[i], a, h_dip) for m, sk, i, a in zip(ms, skeys, sel, pats)
               if len(i)]
        if not pts:
            continue
        probes = np.concatenate(pts)
        for e in (0, 1):
            tick((2 * k + e) / (2.0 * K), "wave band: receiver key %d / %d, %s ear, %d probes"
                 % (k + 1, K, ("left", "right")[e], len(probes)))
            ir, info = run_low_band(scene, ears[k, e], [p for p in probes],
                                    missing_rcv_ok=any(a is not None for a in pats),
                                    domain_points=_mono_points(
                                        pts, [i for i in sel if len(i)]), **kw)
            ir = np.asarray(ir, dtype=np.float64).reshape(len(ir), -1)
            miss = np.asarray(getattr(info, "missing", ()), dtype=np.int64)
            off = 0
            for j, i in enumerate(sel):
                if not len(i):
                    continue
                if acc[j] is None:
                    acc[j] = np.zeros((len(skeys[j]), ir.shape[0], 2))
                width = len(i) * (1 if pats[j] is None else 3)
                mj = miss[(miss >= off) & (miss < off + width)] - off
                cols, nb = _key_irs(ir[:, off:off + width].T, pats[j], h_dip, c, fs_low, mj)
                if e == 0:
                    n_omni[j] += nb
                if mc.wave_align:
                    # each source key gets the geometry of its own time: the receiver key's
                    # IR moved by the change of the direct-path delay (ear at the key time)
                    sp = np.atleast_2d(ms[j].trajectory.position(skeys[j][i]))
                    ear_now = _ears_at(mr.trajectory, skeys[j][i], half)[:, e]
                    d1 = (np.linalg.norm(sp - ear_now, axis=1)
                          - np.linalg.norm(sp - ears[k, e], axis=1)) / c
                    cols = _shift_rows(cols, d1, fs_low)
                acc[j][i, :, e] += wk[j][i, None] * cols
                off += width
    _warn_omni_keys(ms, n_omni, skeys)
    for j, sk in enumerate(skeys):
        out.append((sk, acc[j], fc, fs_low))
    return out, ("both move: reciprocity at %d receiver key frames (%d runs), %s source key "
                 "frames, crossover %.0f Hz" % (K, 2 * K, "/".join(str(len(k)) for k in skeys),
                                               fc))


def _warn_omni_keys(ms, n_omni, keys):
    for m, nb, k in zip(ms, n_omni, keys):
        if nb:
            print("[Auralis] WARNING: animated wave band: directional source '%s' too close to "
                  "a surface at %d of its %d key frames (dipole probes outside the air): "
                  "OMNI there below the crossover." % (m.name, nb, len(k)), flush=True)


def _shift_rows(h, d, fs):
    """Each row of h delayed by d[row] seconds (fractional, through the spectrum, zero-padded
    so nothing wraps)."""
    h = np.atleast_2d(h)
    n = h.shape[1]
    nf = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    f = np.fft.rfftfreq(nf, 1.0 / fs)
    ph = np.exp(-2j * np.pi * f[None, :] * np.asarray(d, dtype=np.float64)[:, None])
    return np.fft.irfft(np.fft.rfft(h, nf, axis=1) * ph, nf, axis=1)[:, :n]


def _vdelay(y, d_samples):
    """y read at n - d[n] (variable fractional delay, 4-point Lagrange; zero outside)."""
    n = len(d_samples)
    p = np.arange(n, dtype=np.float64) - d_samples
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    w = ((-f * (f - 1) * (f - 2) / 6.0), ((f + 1) * (f - 1) * (f - 2) / 2.0),
         (-(f + 1) * f * (f - 2) / 2.0), ((f + 1) * f * (f - 1) / 6.0))
    out = np.zeros(n)
    for k, wk in zip((-1, 0, 1, 2), w):
        idx = i0 + k
        ok = (idx >= 0) & (idx < len(y))
        out[ok] += wk[ok] * y[idx[ok]]
    return out


def _direct_delays(msrc, mrcv, t, c, half):
    """(len(t), 2) direct-path delays from the source to each ear at times t."""
    s = np.atleast_2d(msrc.trajectory.position(t))
    ears = _ears_at(mrcv.trajectory, t, half)
    return np.linalg.norm(ears - s[:, None, :], axis=2) / c


def _fftconv(a, b):
    n = len(a) + len(b) - 1
    nf = 1 << int(np.ceil(np.log2(max(n, 2))))
    return np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:n]


def _mix_wave_band(out, dry, low_keys, fs, gain, delays=None):
    """out (geometric render) high-passed at the crossover, plus the dry signal
    through the key IRs (low-passed, linearly cross-faded in time), computed at the
    wave band's own rate and resampled: the static fusion's crossover (fdtd).
    delays(t) -> (len(t), 2): direct-path delay per ear; given, every key's output goes
    through a variable delay line delays(t) - delays(t_key) before the cross-fade."""
    from .fdtd import _resample_fft, design_crossover
    key_t, irs, fc, fs_low = low_keys
    _lp, hp, dl = design_crossover(fc, fs)
    n_out = len(out)
    res = np.empty_like(out)
    for ch in range(out.shape[1]):
        res[:, ch] = _fftconv(out[:, ch], hp)[dl:dl + n_out]
    x = _resample_fft(np.asarray(dry, dtype=np.float64), fs, fs_low,
                      int(np.ceil(len(dry) * fs_low / fs)))
    lpl, _hpl, dll = design_crossover(fc, fs_low)
    n_low = int(np.ceil(n_out * fs_low / fs)) + 1
    low = np.zeros((n_low, 2))
    t = np.arange(len(x)) / fs_low
    K = len(key_t)
    tau = tau_k = None
    if delays is not None and K > 1:
        tau = delays(np.arange(n_low) / fs_low)                  # (n_low, 2)
        tau_k = delays(np.asarray(key_t, dtype=np.float64))      # (K, 2)
    for j in range(K):
        w = np.ones(len(x)) if K == 1 else np.interp(t, key_t, np.eye(K)[j])
        nz = np.flatnonzero(w > 0.0)
        if len(nz) == 0:
            continue
        a, b = nz[0], nz[-1] + 1
        z = x[a:b] * w[a:b]
        for ch in range(2):
            h = np.convolve(irs[j][:, ch], lpl)[dll:dll + irs.shape[1]]
            y = _fftconv(z, h)[:n_low - a]
            if tau is not None:
                d = (tau[a:, ch] - tau_k[j, ch]) * fs_low       # samples, output time base
                m = min(int(np.ceil(np.max(np.abs(d)))) + 2, len(d) - len(y))
                if m > 0:                                      # room for a delayed tail
                    y = np.concatenate([y, np.zeros(m)])
                y = _vdelay(y, d[:len(y)])
            low[a:a + len(y), ch] += y
    for ch in range(2):
        res[:, ch] += gain * _resample_fft(low[:, ch], fs_low, fs, n_out)
    return res


def render_scene_moving(scene: SceneData,
                        sources_dry: List[Tuple[Union[MovingSource, SourceDef],
                                                np.ndarray]],
                        receiver, mconfig: MotionConfig = None,
                        progress=None) -> Tuple[np.ndarray, dict]:
    """Sum several moving sources into the same receiver.
    `sources_dry`: list of (source, mono_dry_signal)."""
    mc = mconfig or MotionConfig()
    outs = []
    infos = []
    n = len(sources_dry)
    lows, wave_note = None, None
    if mc.wave:
        try:
            lows, wave_note = wave_band_keys(
                scene, [x[0] for x in sources_dry], receiver, mc,
                [len(d) / float(mc.base.fs) for _s, d in sources_dry],
                progress=(lambda fr, st: progress(0.15 * fr, st)) if progress else None)
        except RuntimeError as exc:                 # watchdog: LOUD fallback
            lows, wave_note = None, "wave band FAILED (%s): geometrical solver only" % exc
        if lows is None:
            print("[Auralis] WARNING: animated render: %s." % wave_note, flush=True)
    for i, (src, dry) in enumerate(sources_dry):
        def sub(fr, st, _i=i):
            if progress is not None:
                return progress((_i + fr) / n,
                                "%s : %s" % (_as_moving_source(src).name, st))
            return True
        base = mc.base
        cfg_i = BakeConfig(**{**base.__dict__, "normalize": False})
        mo, info = render_moving(scene, src, receiver, dry,
                                 dataclasses.replace(mc, base=cfg_i),
                                 progress=sub,
                                 low_keys=None if lows is None else lows[i])
        outs.append(mo)
        infos.append(info)
    n_out = max(len(o) for o in outs)
    mix = np.zeros((n_out, 2))
    for o in outs:
        mix[:len(o)] += o
    scale = 1.0
    if mc.base.normalize:
        peak = float(np.abs(mix).max())
        if peak > 0:
            scale = 10 ** (-1.0 / 20.0) / peak
            mix *= scale
    return mix, {"sources": infos, "scale": scale,
                 "wave": None if not mc.wave else {"animated": lows is not None,
                                                   "note": wave_note}}
