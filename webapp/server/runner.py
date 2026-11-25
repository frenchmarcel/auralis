# -*- coding: utf-8 -*-
"""Run one simulation in its own process (one GPU job at a time, OOM-safe).
Usage: python -m webapp.server.runner <store_root> <simulation_id>
Progress goes to stdout as lines '@@{json}'; everything else is the engine log."""
from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import pickle
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from auralis.engine import rirmetrics as rm                      # noqa: E402
from auralis.engine import speech                                # noqa: E402
from auralis.engine.level import db_total, dba_total, dbc_total  # noqa: E402
from auralis.engine.metrics import lateral_metrics               # noqa: E402
from auralis.engine.pipeline import (ReceiverDef, SourceDef, bake_pair,  # noqa: E402
                                     gpu_available, source_low_bands)
from auralis.engine.project import build_config, build_scene     # noqa: E402
from auralis.engine.grid import grid_metrics                     # noqa: E402
from webapp.server import directivities as dirlib                # noqa: E402
from webapp.server import grids as gridlib                       # noqa: E402
from webapp.server import meshcheck                              # noqa: E402
from webapp.server import sim as simlib                          # noqa: E402
from webapp.server import sourceeq                                # noqa: E402
from webapp.server.library import Library                         # noqa: E402
from webapp.server.store import Store                             # noqa: E402

GPU_WAIT_S = 300
GPU_POLL_S = 15


def emit(**kw):
    print("@@" + json.dumps(kw), flush=True)


def choose_engine(cfg):
    """Handle a GPU that refuses new CUDA contexts (seen on a laptop after a session).

    The wave solver waits for it (the CPU is ~50x slower there). The geometrical solver
    goes to the CPU at once. Either way the fallback is explicit, never silent."""
    if cfg.engine not in ("auto", "gpu") or gpu_available():
        return
    if cfg.fdtd and os.environ.get("AURALIS_GPU", "1") != "0":    # GPU not disabled on purpose
        t0 = time.time()
        while time.time() - t0 < GPU_WAIT_S:
            emit(stage="waiting for the GPU (CUDA context refused)")
            time.sleep(GPU_POLL_S)
            if gpu_available():
                return
    print("[Auralis] WARNING: the GPU refuses CUDA contexts: running on the CPU%s." % (
        " (wave solver: much slower)" if cfg.fdtd else ""), flush=True)
    cfg.engine = "native"
    os.environ["AURALIS_GPU"] = "0"


def receiver_def(r):
    fwd = simlib.direction(r.get("azimuth_deg", 0), r.get("elevation_deg", 0))
    up = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    return ReceiverDef(name=r["id"], position=np.asarray(r["position"], float),
                       forward=fwd, up=up)


def source_def(s, root=None):
    """Engine source of an app source; imported directivities are read from the store
    `root` (with their on-axis level at 1 m when they carry one)."""
    d = s.get("directivity") or "omni"
    if root is not None:
        dirv, spl = dirlib.engine_directivity(root, d), dirlib.on_axis_spl(root, d)
    else:
        dirv, spl = (None if d == "omni" else d), 94.0
    return SourceDef(name=s["id"], position=np.asarray(s["position"], float),
                     gain_db=float(s.get("gain_db", 0.0)), directivity=dirv, spl_1m=spl,
                     forward=simlib.direction(s.get("azimuth_deg", 0), s.get("elevation_deg", 0)),
                     surface=str(s.get("surface_layer") or ""))


class WaveTicker:
    """Progress of the wave solver every 2 s, never going back.

    While the grid is built (no step yet): an estimate kept within the first 30 %.
    Then the step the native kernels report (ABI 2.3, longest run only) fills 30-100 %,
    shared between the `runs` of a directional source (a monopole and a dipole run).
    Without ABI 2.3: the estimate, capped at 95 %."""

    def __init__(self, estimate_s, report, runs=1):
        import threading
        from auralis.engine import fdtd as _fdtd
        self.t0, self.est, self.report = time.time(), float(estimate_s), report
        self.step, self.last = None, 0.0
        self.runs, self.run_i = max(int(runs), 1), 0
        self._fdtd = _fdtd
        _fdtd.set_progress_hook(self._hook)
        self.done = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _hook(self, k, n):
        if n > 0 and (self.step is None or n >= self.step[1]):
            if self.step is not None and n == self.step[1] and k + n // 10 < self.step[0]:
                self.run_i = min(self.run_i + 1, self.runs - 1)     # the next run began
            self.step = (k, n)

    def _loop(self):
        while not self.done.wait(2.0):
            est = (time.time() - self.t0) / self.est
            if self.step is not None:
                f = 0.3 + 0.7 * (self.run_i + self.step[0] / self.step[1]) / self.runs
                kind = "steps"
            else:
                f, kind = min(0.3, 0.3 * est), "building"
            self.last = max(self.last, min(0.99, f))
            self.report(self.last, kind)

    def stop(self):
        self.done.set()
        self._fdtd.set_progress_hook(None)


def receiver_radii(rcvs, mesh, nominal):
    """Ray receiver sphere per receiver: the nominal radius, shrunk to the distance of
    the nearest surface (0.1 m min). A sphere cut by a wall under-counts: measured
    -0.8 dB at 0.10 m with 0.3 m, ~0 once the radius <= distance."""
    from webapp.server import meshcheck
    if not rcvs:
        return {}
    d = meshcheck.distance_to_surfaces([r.position for r in rcvs], mesh[0][mesh[1]])
    out = {}
    for r, di in zip(rcvs, d):
        out[r.name] = float(nominal) if di >= nominal else float(max(0.1, di))
        if di < nominal:
            print("[Auralis] receiver %s is %.2f m from a surface: ray receiver radius %.2f m "
                  "instead of %.2f m (unbiased level, a little more noise)."
                  % (r.name, di, out[r.name], nominal), flush=True)
    return out


def part_metrics(ir, fs, f_max=None, ref_ir=None, ref_g=None, ref_spl=None):
    """ISO 3382 parameters of one solver's omni IR; bands whose upper edge is above
    f_max (the wave band's transition) are None. G and SPL follow the hybrid's by the
    band-energy ratio of the two omni IRs (same scale, same source)."""
    ir = np.asarray(ir, dtype=np.float64)
    m = rm.band_metrics(ir, fs)
    out = {k: [clean(x) for x in m[k]] for k in ("T20", "T30", "EDT", "C50", "C80", "D50")}
    out["Ts_ms"] = [clean(x) for x in m["Ts"]]
    if ref_ir is not None and ref_g is not None:
        bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]    # true octaves, as G
        e = (rm.octave_split(ir, fs, bands) ** 2).sum(axis=1)
        e_ref = (rm.octave_split(np.asarray(ref_ir, dtype=np.float64), fs, bands) ** 2).sum(axis=1)
        delta = 10.0 * np.log10(np.maximum(e, 1e-300) / np.maximum(e_ref, 1e-300))
        out["G_db"] = [clean(g + d) for g, d in zip(ref_g, delta)]
        if ref_spl is not None:
            out["SPL_db"] = [clean(x + d) for x, d in zip(ref_spl, delta)]
    if f_max is not None:
        ok = [c * 2 ** 0.5 <= f_max for c in (125, 250, 500, 1000, 2000, 4000, 8000)]
        out = {k: [v if o else None for v, o in zip(vals, ok)] for k, vals in out.items()}
    return out


def band63_levels(omni, fs, g125, spl125):
    """(G, SPL) of the 63 Hz octave: the 125 Hz values shifted by the energy ratio of the two
    octaves in the omni IR, over that of a flat impulse through the same filters. The
    source's level at 63 Hz is taken as its 125 Hz one."""
    if g125 is None:
        return None, None
    # one second of silence in front: the zero-phase octave filters ring before the onset
    pad = int(fs)
    x = np.concatenate([np.zeros(pad), np.asarray(omni, dtype=np.float64)])
    d = np.zeros(len(x))
    d[pad] = 1.0
    e63 = float((rm.octave_split(x, fs, [63.0])[0] ** 2).sum())
    e125 = float((rm.octave_split(x, fs, [125.0])[0] ** 2).sum())
    r63 = float((rm.octave_split(d, fs, [63.0])[0] ** 2).sum())
    r125 = float((rm.octave_split(d, fs, [125.0])[0] ** 2).sum())
    if min(e63, e125, r63, r125) <= 0.0:
        return None, None
    shift = 10.0 * np.log10((e63 / r63) / (e125 / r125))
    return clean(g125 + shift), (None if spl125 is None else clean(spl125 + shift))


def decay_reached(res):
    """Per band: energy level reached at the end of the echogram (dB re its early peak,
    50 ms averages)."""
    h = res.hist_total
    k = np.ones(50) / 50.0
    k0 = int(res.events[0].time / res.config.bin_dt) if res.events else 0
    out = []
    for b in range(h.shape[0]):
        sm = np.convolve(h[b], k, "same")
        peak = float(sm[k0:k0 + 100].max()) if len(sm) > k0 else 0.0
        tail = float(sm[-80:-30].mean()) if len(sm) > 80 else float(sm[-1])
        out.append(10.0 * np.log10(max(tail, 1e-30) / peak) if peak > 0 else None)
    return [None if v is None else max(v, -200.0) for v in out]


def ir_decay_reached(omni, fs):
    """decay_reached from the IR itself (true octaves, 50 ms averages): a wave-only result has
    no echogram of its own."""
    x = np.asarray(omni, dtype=np.float64)
    bands = rm.octave_split(x, fs, [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]) ** 2
    w = max(1, int(0.05 * fs))
    k = np.ones(w) / w
    out = []
    for e in bands:
        sm = np.convolve(e, k, "same")
        peak = float(sm.max())
        tail = float(sm[-int(0.08 * fs):-int(0.03 * fs)].mean()) if len(sm) > 0.08 * fs else 0.0
        out.append(max(10.0 * np.log10(max(tail, 1e-30) / peak), -200.0) if peak > 0 else None)
    return out


def extra_metrics(res, ir, src, rcv, settings):
    """STI (IEC 60268-16, MTF of the full-band IR, speech level at the listener from
    G, background noise from the settings), J_LF / L_J from the geometrical solver's
    directional energy (ISO 3382-1), overall SPL in dB Z / A / C."""
    g = np.asarray(res.metrics["g_bands"], dtype=np.float64)
    signal = speech.speech_levels_1m(settings.get("speech_effort", "normal")) + g - 20.0
    noise = speech.noise_levels(settings.get("background_noise", "none"))
    st = speech.sti(speech.mtf_from_ir(ir, res.fs), signal, noise)
    jlf, lj = lateral_metrics(res.hist, res.dirs, res.events, res.config.bin_dt,
                              src.position, rcv.position)
    spl = np.asarray(res.metrics["spl_bands"], dtype=np.float64)
    return {"STI": clean(st["sti"]), "MTI": [clean(x) for x in st["mti"]],
            "STI_signal_db": [clean(x) for x in signal],
            "JLF": [clean(x) for x in jlf], "LJ_db": [clean(x) for x in lj],
            "SPL_dbz": clean(db_total(spl)), "SPL_dba": clean(dba_total(spl)),
            "SPL_dbc": clean(dbc_total(spl))}


def run_grids(sim, geometry, mesh, closure, scene, cfg, srcs, out_dir, emit_fn, ck=None):
    """Receiver grids: energy-based parameters of the geometrical solver at every valid
    point, with G and SPL taking the wave band in its octaves when cfg.fdtd (one FDTD grid per
    grid and source), one npz per grid and source (those of an interrupted run, in `ck`, are
    kept)."""
    grids = sim.get("receiver_grids", [])
    if not grids:
        return []
    os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)
    noise = speech.noise_levels(sim["settings"].get("background_noise", "none"))
    effort = sim["settings"].get("speech_effort", "normal")
    n_rays = int(sim["settings"].get("grid_rays", gridlib.DEFAULT_RAYS))
    out = []
    work = [(g, s) for g in grids for s in srcs]
    for k, (g, s) in enumerate(work):
        kept = ck.data["grids"].get("%s__%s" % (g["id"], s.name)) if ck is not None else None
        if kept is not None:
            out.append(kept)
            continue
        p, valid, why, area, nrm, uax = gridlib.points(g, sim, geometry, mesh, closure, axes=True)
        gcfg = dataclasses.replace(cfg, rcv_radius=gridlib.receiver_radius(g))
        t0 = time.time()

        def cb(frac, stage, _k=k):
            emit_fn(stage="grid %s, %s: %s" % (g["name"], s.name, stage),
                    progress=(_k + float(frac)) / len(work))

        dist = meshcheck.distance_to_surfaces(p[valid], mesh[0][mesh[1]])
        res = grid_metrics(scene, s, p[valid], gcfg, n_rays=n_rays, speech_effort=effort,
                           noise_db=noise, progress=cb,
                           rcv_radii=np.clip(dist, 0.1, gcfg.rcv_radius))
        arrays = {}
        for key in grid_params():
            a = np.full((len(p),) + res[key].shape[1:], np.nan)
            a[valid] = res[key]
            arrays[key] = a
        np.savez(os.path.join(out_dir, "grids", "%s__%s.npz" % (g["id"], s.name)), points=p,
                 valid=valid, area_m2=area, normals=nrm, u_axes=uax, **arrays)
        with open(os.path.join(out_dir, "grids", "%s__%s_reasons.json" % (g["id"], s.name)),
                  "w") as fh:
            json.dump({str(k): v for k, v in why.items()}, fh)
        out.append({"grid_id": g["id"], "source_id": s.name, "n_points": int(len(p)),
                    "n_valid": int(valid.sum()), "n_rays": n_rays,
                    "rcv_radius_m": gcfg.rcv_radius, "duration_s": round(time.time() - t0, 1)})
        if ck is not None:
            ck.grid_done("%s__%s" % (g["id"], s.name), out[-1])
    by = {}
    for x in out:
        by.setdefault(x["grid_id"], []).append(x)
    return [{"grid_id": gid, "sources": v} for gid, v in by.items()]


def fingerprint(sim, man):
    """Hash of everything a result depends on: a checkpoint is reused only for the same
    model, materials, settings, sources, receivers and grids."""
    import hashlib
    key = {"geometry_id": sim.get("geometry_id"), "manifest": man, "sources": sim["sources"],
           "receivers": sim["receivers"], "grids": sim.get("receiver_grids", []),
           "settings": sim.get("settings", {})}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


class Checkpoint:
    """Per-source progress of a run in its .tmp results folder, so a run interrupted by
    a server stop resumes after its finished sources (and grids). A folder with another
    fingerprint, or none, is wiped."""

    def __init__(self, tmp, fp):
        self.path = os.path.join(tmp, "checkpoint.json")
        self.data = None
        try:
            with open(self.path) as fh:
                d = json.load(fh)
            if d.get("fingerprint") == fp:
                self.data = d
        except (OSError, ValueError):
            pass
        if self.data is None:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(os.path.join(tmp, "irs"))
            self.data = {"fingerprint": fp, "sources": {}, "grids": {}, "fc_used": None,
                         "engine_used": None, "wave_failed": []}
        else:
            print("[Auralis] resuming an interrupted run: %d source(s) and %d grid run(s) "
                  "kept from it." % (len(self.data["sources"]), len(self.data["grids"])),
                  flush=True)

    def _save(self):
        part = self.path + ".part"
        with open(part, "w") as fh:
            json.dump(self.data, fh)
        os.replace(part, self.path)

    def source_done(self, sid, pairs, fc_used, engine_used, wave_failed):
        self.data["sources"][sid] = pairs
        self.data["fc_used"], self.data["engine_used"] = fc_used, engine_used
        self.data["wave_failed"].extend(wave_failed)
        self._save()

    def grid_done(self, key, entry):
        self.data["grids"][key] = entry
        self._save()


def grid_params():
    from auralis.engine.grid import PARAMS, SINGLE
    return PARAMS + SINGLE


def clean(v):
    if v is None:
        return None
    v = float(v)
    return round(v, 6) if np.isfinite(v) else None


def measure_pair(scene, cfg, s, r, low, radius, eq, label, settings, tmp, progress=None):
    """Bakes one pair, saves its responses in tmp/irs and measures them. Returns
    (pair entry, peak of the binaural IR, engine used). Runs in the runner, or in a
    worker of PairPool."""
    t0 = time.time()
    # no precomputed wave band: geometrical solver only (bake would otherwise
    # run its own FDTD pass, the one that just failed)
    res = bake_pair(scene, s, r, cfg, low, rcv_radius=radius,
                    own_wave_pass=False, progress=progress)
    if eq:                                  # per-octave source EQ (sourceeq)
        sourceeq.apply_to_result(res, eq)
    ir = np.asarray(res.ir, dtype=np.float32)
    peak = float(np.abs(ir).max())
    np.save(os.path.join(tmp, "irs", "%s__%s.npy" % (s.name, r.name)), ir)
    # ISO 3382 parameters on the OMNIDIRECTIONAL IR (ambisonic W): the mean of
    # the two ears is not a pressure (up to 10 % on T30, 2 dB on C80)
    omni = ir.mean(axis=1)
    if res.ambi is not None:
        np.save(os.path.join(tmp, "irs", "%s__%s_ambi.npy" % (s.name, r.name)),
                np.asarray(res.ambi, dtype=np.float32))
        omni = np.asarray(res.ambi[:, 0], dtype=np.float64)
    parts = {}
    only_wave = bool(res.metrics.get("wave_only"))
    if res.omni_parts is not None and not only_wave:
        for part in ("geo", "wave"):
            np.save(os.path.join(tmp, "irs", "%s__%s_omni_%s.npy" % (s.name, r.name, part)),
                    np.asarray(res.omni_parts[part], dtype=np.float32))
            parts[part] = part_metrics(res.omni_parts[part], res.fs,
                                       res.omni_parts["fc"] if part == "wave" else None,
                                       ref_ir=res.ambi[:, 0],
                                       ref_g=res.metrics.get("g_bands"),
                                       ref_spl=res.metrics.get("spl_bands"))
    m = rm.band_metrics(omni, res.fs)
    met = {k2: [clean(x) for x in m[k2]] for k2 in ("T20", "T30", "EDT", "C50", "C80", "D50")}
    met["Ts_ms"] = [clean(x) for x in m["Ts"]]
    met["SPL_db"] = [clean(x) for x in res.metrics.get("spl_bands", [None] * 7)]
    met["G_db"] = [clean(x) for x in res.metrics.get("g_bands", [None] * 7)]
    met.update(extra_metrics(res, omni, s, r, settings))
    met["decay_range_db"] = [clean(x) for x in (
        ir_decay_reached(omni, res.fs) if only_wave else decay_reached(res))]
    if only_wave:
        met = simlib.mask_wave_only(met, res.stats.get("fdtd_fc") or 0.0)
    if parts:
        met["by_solver"] = parts
    fc_pair = res.stats.get("fdtd_fc") or 0.0
    if res.ambi is not None and fc_pair >= 125.0:
        # the 63 Hz octave (44-88 Hz) from the wave band alone,
        # when the whole octave lies below the transition
        m63 = rm.band_metrics_at(omni, res.fs, [63.0])
        met["band63"] = {k2: clean(m63[k2][0]) for k2 in ("T20", "T30", "EDT", "C50",
                                                            "C80", "D50")}
        met["band63"]["Ts_ms"] = clean(m63["Ts"][0])
        met["band63"]["G_db"], met["band63"]["SPL_db"] = band63_levels(
            omni, res.fs, met["G_db"][0], met["SPL_db"][0])
    short = [lab for lab, x in zip(("125", "250", "500", "1k", "2k", "4k", "8k"),
                                   met["decay_range_db"]) if x is not None and x > -45.0]
    if short:
        print("[Auralis] WARNING: %s -> %s: the IR ends before a 45 dB decay in the %s "
              "band(s): T30 is unreliable there; lengthen the IR (Advanced: IR length "
              "or decay criterion)." % (label, r.name, ", ".join(short)), flush=True)
    entry = {"source_id": s.name, "receiver_id": r.name, "metrics": met,
             "timings": {"total_s": round(time.time() - t0, 2)}}
    return entry, peak, res.engine_used


_WORKER = {}


def _pair_worker_init(payload):
    try:                                    # Linux: a worker dies with its runner (cancel)
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGKILL))
    except Exception:
        pass
    _WORKER.update(zip(("scene", "cfg", "settings", "tmp"), pickle.loads(payload)))


def _pair_worker_run(s, r, low, radius, eq, label):
    w = _WORKER
    return measure_pair(w["scene"], w["cfg"], s, r, low, radius, eq, label, w["settings"],
                        w["tmp"])


def pair_workers(n_pairs, cfg):
    """Number of worker processes for the pairs.

    The geometrical part of a pair is mostly CPU work (the synthesis of its responses:
    ~85 % of a pair in a concert hall, the GPU tracer ~10 %), so pairs run side by side.
    Each worker opens its own CUDA context for the tracer (6 of them and an FDTD run
    share a laptop GPU). AURALIS_PAIR_WORKERS sets the count (1 = all pairs in the
    runner). Without it: workers only with the GPU tracer (the CPU tracer already uses
    every core), at most 6 and a third of the cores."""
    env = os.environ.get("AURALIS_PAIR_WORKERS", "").strip()
    if env:
        return max(1, min(int(env), n_pairs))
    if cfg.engine not in ("auto", "gpu") or os.environ.get("AURALIS_GPU", "1") == "0" \
            or not gpu_available():
        return 1
    return max(1, min(6, (os.cpu_count() or 1) // 3, n_pairs))


class PairPool:
    """Pairs baked in worker processes. Each worker receives the runner's scene once
    (pickled: rebuilding it there would redo the material inference, 10 s for a hall).
    Each pair is a fixed function of its inputs (seeded noise): the same responses as
    in the runner, bit for bit with the CPU tracer. The GPU tracer differs by ~1e-10
    from one run to the next anyway (float atomics)."""

    def __init__(self, n, scene, cfg, settings, tmp):
        payload = pickle.dumps((scene, cfg, settings, tmp), protocol=pickle.HIGHEST_PROTOCOL)
        self.ex = ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"),
                                      initializer=_pair_worker_init, initargs=(payload,))
        self.futs = {}

    def submit(self, tag, *args):
        self.futs[self.ex.submit(_pair_worker_run, *args)] = tag

    def finished(self, block):
        """(tag, result) of the pairs done so far; block = wait for at least one."""
        if not self.futs:
            return []
        done, _ = wait(list(self.futs), timeout=None if block else 0,
                       return_when=FIRST_COMPLETED)
        return [(self.futs.pop(f), f.result()) for f in done]

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)


def run(store_root, mid):
    store = Store(store_root)
    sim = store.get_simulation(mid)
    geometry = store.get_geometry(sim["geometry_id"])
    mesh = store.load_mesh(sim["geometry_id"])
    library = Library(os.path.join(store.root, "materials_custom.json"))
    check = simlib.validate(sim, geometry, mesh, library, store.load_closure(sim["geometry_id"]))
    if not check["ok_to_run"]:
        raise ValueError("not runnable: " + "; ".join(check["blocking"]))
    open_scene = simlib.is_open_scene(sim["settings"], geometry)
    man = simlib.build_manifest(sim, mesh, library, open_scene)
    scene = build_scene(man, "")
    cfg = build_config(man)
    if open_scene and cfg.fdtd and sim["settings"].get("conformal", True):
        print("[Auralis] open scene: ANISOTROPIC conformal correction (cut layer + tangential "
              "apertures). The default correction degrades open scenes with sharp edges.",
              flush=True)
    choose_engine(cfg)

    tmp = store.results_dir(mid) + ".tmp"
    ck = Checkpoint(tmp, fingerprint(sim, man))
    srcs = [source_def(s, store.root) for s in sim["sources"]]
    eqs = {s["id"]: sourceeq.clean_eq(s.get("eq_db")) for s in sim["sources"]}
    rcvs = [receiver_def(r) for r in sim["receivers"]]
    radii = receiver_radii(rcvs, mesh, cfg.rcv_radius)
    n_s, n_r = len(srcs), len(rcvs)
    prog = {s.name: 0.0 for s in srcs}
    pairs, fc_used, engine_used, peak = [], ck.data["fc_used"], ck.data["engine_used"], 0.0
    wave_failed = list(ck.data["wave_failed"])
    t_start = time.time()

    pair_share = 0.5 if sim.get("receiver_grids") and n_r else (1.0 if n_r else 0.0)
    wave_estimate_s, fc_guess = 60.0, None
    if cfg.fdtd:
        try:
            est = simlib.estimate_runtime(sim, geometry, mesh, library, gpu=cfg.engine != "native",
                                          root=store.root)
            wave_estimate_s = max(5.0, est["wave_s"] / max(n_s, 1))
            fc_guess = est.get("transition_hz")
        except Exception:
            pass

    def report(stage):
        emit(stage=stage, progress=pair_share * sum(prog.values()) / n_s, sources=prog)

    results, shares, left, engines = {}, {}, {}, []

    def pair_finished(sid, label, entry, pk, eng):
        nonlocal peak, engine_used
        results[(sid, entry["receiver_id"])] = entry
        peak, engine_used = max(peak, pk), eng
        engines.append(eng)
        left[sid] -= 1
        prog[sid] = shares[sid] + (1 - shares[sid]) * (n_r - left[sid]) / n_r
        report("%s - done %d/%d" % (label, n_r - left[sid], n_r))
        if not left[sid]:
            ck.source_done(sid, [results[(sid, r.name)] for r in rcvs], fc_used, engine_used,
                           [w for w in wave_failed if w["source_id"] == sid])

    def run_sources(sources, pool):
        """Wave band of each source in the runner (GPU), then its pairs: in the
        runner, or handed to the pool while the next source's wave band runs."""
        nonlocal fc_used, peak
        for s in sources:
            label = next(x["name"] for x in sim["sources"] if x["id"] == s.name)
            if s.name in ck.data["sources"]:           # finished before the server stopped
                for e in ck.data["sources"][s.name]:
                    results[(s.name, e["receiver_id"])] = e
                for r in rcvs:
                    peak = max(peak, float(np.abs(np.load(os.path.join(
                        tmp, "irs", "%s__%s.npy" % (s.name, r.name)))).max()))
                prog[s.name] = 1.0
                report("%s - kept from the interrupted run" % label)
                continue
            low, share = {}, 0.0
            if cfg.fdtd:
                share = 0.5
                report("wave band (FDTD) - %s" % label)
                runs = simlib.wave_runs(next(x for x in sim["sources"] if x["id"] == s.name),
                                        store.root, fc_guess) if fc_guess else 1
                ticker = WaveTicker(wave_estimate_s, lambda f, kind, _l=label, _n=s.name: (
                    prog.__setitem__(_n, share * f),
                    report("wave band (FDTD) - %s (%s)"
                           % (_l, "building the grid" if kind == "building"
                              else "%d%%" % round(100 * f)))), runs=runs)
                try:
                    low, err = source_low_bands(scene, s, rcvs, cfg, fallback=True)
                finally:
                    ticker.stop()
                if err:                                # watchdog: unstable boundary mode
                    print("[Auralis] WARNING: wave band FAILED for %s (%s): this source uses the "
                          "geometrical solver only." % (label, err), flush=True)
                    wave_failed.append({"source_id": s.name, "error": err})
                for _nm, (_ir, _finfo, fc) in low.items():
                    fc_used = float(fc)
                missing = [r.name for r in rcvs if r.name not in low]
                if missing and low:
                    print("[Auralis] WARNING: wave band failed for %s -> %s: those pairs use "
                          "the geometrical solver only." % (label, ", ".join(missing)), flush=True)
                prog[s.name] = share
            shares[s.name], left[s.name] = share, n_r
            for k, r in enumerate(rcvs):
                lr = {r.name: low[r.name]} if r.name in low else {}
                if pool is not None:
                    pool.submit((s.name, label), s, r, lr, radii[r.name], eqs.get(s.name), label)
                    for tag, out in pool.finished(block=False):
                        pair_finished(*tag, *out)
                    continue

                def cb(frac, stage, _k=k):
                    prog[s.name] = share + (1 - share) * (_k + float(frac)) / n_r
                    report("%s - %s" % (label, stage))
                pair_finished(s.name, label, *measure_pair(scene, cfg, s, r, lr, radii[r.name],
                                                           eqs.get(s.name), label,
                                                           sim["settings"], tmp, cb))

    todo = sum(1 for s in srcs if s.name not in ck.data["sources"]) * n_r
    n_workers = pair_workers(todo, cfg) if todo > 1 else 1
    pool = PairPool(n_workers, scene, cfg, sim["settings"], tmp) if n_workers > 1 else None
    if pool is not None:
        print("[Auralis] %d pairs on %d worker processes (AURALIS_PAIR_WORKERS=1: one at a "
              "time in the runner)." % (todo, n_workers), flush=True)
    try:
        run_sources(srcs if rcvs else [], pool)
        while pool is not None and pool.futs:
            for tag, out in pool.finished(block=True):
                pair_finished(*tag, *out)
    finally:
        if pool is not None:
            pool.close()
    pairs = [results[(s.name, r.name)] for s in srcs for r in rcvs]
    if len(set(engines)) > 1:
        print("[Auralis] WARNING: the pairs did not all use the same engine (%s): a worker "
              "fell back from the GPU." % ", ".join("%s x%d" % (e, engines.count(e))
                                                     for e in sorted(set(engines))),
              flush=True)

    def grid_emit(stage, progress):
        emit(stage=stage, progress=pair_share + (1.0 - pair_share) * progress)

    grid_meta = run_grids(sim, geometry, mesh, store.load_closure(sim["geometry_id"]), scene, cfg,
                          srcs, tmp, grid_emit, ck)

    meta = {"bands_hz": [125, 250, 500, 1000, 2000, 4000, 8000], "fs": int(cfg.fs),
            "metrics_ir": "omni" if cfg.ambisonic_order > 0 else "binaural mean",
            "transition_hz": fc_used, "engine_used": engine_used,
            "ir_scale": (0.9 / peak) if peak > 0 else 1.0,
            "settings": sim["settings"], "pairs": pairs, "grids": grid_meta,
            "wave_failed": wave_failed,
            "duration_s": round(time.time() - t_start, 1)}
    with open(os.path.join(tmp, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if os.path.exists(ck.path):
        os.remove(ck.path)
    final = store.results_dir(mid)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    emit(stage="done", progress=1.0, sources={s.name: 1.0 for s in srcs},
         engine_used=engine_used, duration_s=meta["duration_s"])


if __name__ == "__main__":
    try:
        run(sys.argv[1], sys.argv[2])
    except Exception as exc:
        traceback.print_exc()
        emit(error="%s: %s" % (type(exc).__name__, exc))
        sys.exit(1)
