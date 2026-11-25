# -*- coding: utf-8 -*-
"""Wave-field animation in its own process: the wave solver's impulse response on every
point of a receiver grid, resampled to the animation frame rate.
Usage: python -m webapp.server.field_runner <store_root> <simulation_id> <task_id>"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from auralis.engine.air import config_speed_of_sound, speed_of_sound  # noqa: E402
from auralis.engine.fdtd import run_low_band                     # noqa: E402
from auralis.engine.project import build_config, build_scene     # noqa: E402
from webapp.server import grids as gridlib                       # noqa: E402
from webapp.server import meshcheck                              # noqa: E402
from webapp.server import sim as simlib                          # noqa: E402
from webapp.server.library import Library                         # noqa: E402
from webapp.server.runner import choose_engine, emit             # noqa: E402
from webapp.server.store import Store                             # noqa: E402

MAX_DURATION_S = 1.0
MAX_FRAMES = 2000


HP_HZ = 15.0


def high_pass(x, fs, f_hp=HP_HZ):
    """Zero-phase high-pass (raised-cosine ramp up to f_hp) of traces (n, P): a source in a
    closed room injects a net volume, and the deconvolution spreads that DC term over the
    whole trace, before the wavefront even arrives (measured: -0.11 everywhere at t = 0)."""
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    g = np.clip(f / f_hp, 0.0, 1.0)
    g = 0.5 - 0.5 * np.cos(np.pi * g)
    return np.fft.irfft(np.fft.rfft(x, nfft, axis=0) * g[:, None], nfft, axis=0)[:n]


def causal_fade(t_direct, n, fs, f_max):
    """(n, P) display window: 0 before t_direct - 2 / f_max, raised-cosine ramp over
    1 / f_max. The band-limited (zero-phase) wave band rings BEFORE each arrival
    (measured: 40 % of the peak ahead of the wavefront); nothing physical arrives there."""
    t = np.arange(n)[:, None] / fs
    a = (t - (np.asarray(t_direct)[None, :] - 2.0 / f_max)) * f_max
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(a, 0.0, 1.0))


def run(store_root, mid, rid):
    store = Store(store_root)
    d = store.task_dir(mid, "field", rid)
    with open(os.path.join(d, "request.json"), encoding="utf-8") as fh:
        req = json.load(fh)
    sim = req["simulation"]
    gid = sim["geometry_id"]
    geometry, mesh = store.get_geometry(gid), store.load_mesh(gid)
    closure = store.load_closure(gid)
    library = Library(os.path.join(store.root, "materials_custom.json"))
    grid = next(g for g in sim["receiver_grids"] if g["id"] == req["grid_id"])
    src = next(s for s in sim["sources"] if s["id"] == req["source_id"])
    open_scene = simlib.is_open_scene(sim["settings"], geometry)
    man = simlib.build_manifest(sim, mesh, library, open_scene)
    scene = build_scene(man, "")
    cfg = build_config(man)
    cfg.fdtd = True
    choose_engine(cfg)
    c = config_speed_of_sound(cfg)
    f_max = float(req["f_max_hz"])
    fps = int(req["fps"])
    duration = float(req["duration_s"])
    p, valid, _why, area, nrm, uax = gridlib.points(grid, sim, geometry, mesh, closure, axes=True)
    dx = c / (13.5 * f_max)
    near = np.zeros(len(p), bool)
    near[valid] = meshcheck.distance_to_surfaces(p[valid], mesh[0][mesh[1]]) < dx
    if near.any():
        print("[Auralis] %d grid points closer than one wave-solver cell (%.2f m) to a surface "
              "are left out." % (int(near.sum()), dx), flush=True)
        valid &= ~near
    emit(stage="wave solver: %d points, up to %.0f Hz, %.2f s at %d frames/s"
         % (int(valid.sum()), f_max, duration, fps), progress=0.05)
    t0 = time.time()
    ir, info = run_low_band(scene, src["position"], [q for q in p[valid]], fs=fps,
                            duration=duration, f_cross=f_max, speed_of_sound=c,
                            engine=cfg.engine, max_cells=cfg.fdtd_max_cells,
                            order=int(cfg.fdtd_order), boundary_absorber=cfg.fdtd_absorber,
                            conformal=bool(cfg.fdtd_conformal), subsonic_hz=0.0)
    frames = np.zeros((ir.shape[0], len(p)), dtype=np.float32)
    tr = high_pass(np.asarray(ir, dtype=np.float64), fps)
    tr *= causal_fade(np.linalg.norm(p[valid] - np.asarray(src["position"], float), axis=1) / c,
                      tr.shape[0], fps, f_max)
    frames[:, valid] = tr.astype(np.float32)
    np.savez(os.path.join(d, "field.npz"), points=p, valid=valid, normals=nrm, u_axes=uax,
             frames=frames)
    meta = {"source_id": src["id"], "grid_id": grid["id"], "n_points": int(len(p)),
            "n_valid": int(valid.sum()), "fps": fps, "n_frames": int(ir.shape[0]),
            "duration_s": duration, "f_max_hz": f_max, "dx_m": round(float(info.dx), 4),
            "engine": info.engine, "conformal": bool(info.conformal),
            "render_s": round(time.time() - t0, 1),
            "note": "impulse response of the wave band (pressure, re 1 m), band-limited to "
                    "about %.0f Hz, high-passed at %.0f Hz (the net volume of a source in a "
                    "closed room is a DC term), faded in just before the direct arrival "
                    "(band-limited pre-ringing); display only, slowed down" % (f_max, HP_HZ)}
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    emit(stage="done", progress=1.0)


if __name__ == "__main__":
    try:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as exc:
        traceback.print_exc()
        emit(error="%s: %s" % (type(exc).__name__, exc))
        sys.exit(1)
