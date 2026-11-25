# -*- coding: utf-8 -*-
"""Forced-response (modal) maps in their own process: the wave solver's transfer
function at chosen frequencies on every point of a receiver grid, for one source.
Usage: python -m webapp.server.modal_runner <store_root> <simulation_id> <task_id>"""
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
from auralis.engine.pipeline import estimate_max_time            # noqa: E402
from auralis.engine.project import build_config, build_scene     # noqa: E402
from webapp.server import grids as gridlib                       # noqa: E402
from webapp.server import meshcheck                              # noqa: E402
from webapp.server import sim as simlib                          # noqa: E402
from webapp.server.library import Library                         # noqa: E402
from webapp.server.runner import choose_engine, emit             # noqa: E402
from webapp.server.store import Store                             # noqa: E402

MAX_FREQS = 400
MAX_HZ = 1000.0


def run(store_root, mid, rid):
    store = Store(store_root)
    d = store.task_dir(mid, "modal", rid)
    with open(os.path.join(d, "request.json"), encoding="utf-8") as fh:
        req = json.load(fh)
    sim = req["simulation"]
    gid = sim["geometry_id"]
    geometry, mesh = store.get_geometry(gid), store.load_mesh(gid)
    closure = store.load_closure(gid)
    library = Library(os.path.join(store.root, "materials_custom.json"))
    grid = next(g for g in sim["receiver_grids"] if g["id"] == req["grid_id"])
    src = next(s for s in sim["sources"] if s["id"] == req["source_id"])
    freqs = np.asarray(req["freqs"], dtype=np.float64)

    open_scene = simlib.is_open_scene(sim["settings"], geometry)
    man = simlib.build_manifest(sim, mesh, library, open_scene)
    scene = build_scene(man, "")
    cfg = build_config(man)
    cfg.fdtd = True
    choose_engine(cfg)
    if cfg.engine == "native":
        print("[Auralis] WARNING: no GPU: the modal map runs the wave solver on the CPU "
              "(much slower).", flush=True)
    p, valid, _why, area = gridlib.points(grid, sim, geometry, mesh, closure)
    c = config_speed_of_sound(cfg)
    duration = float(cfg.max_time or estimate_max_time(scene, c))
    f_grid = max(60.0, 1.1 * float(freqs.max()))
    # the wave solver refuses probes within about half a cell of a surface
    dx = c / (13.5 * f_grid)
    near = np.zeros(len(p), bool)
    near[valid] = meshcheck.distance_to_surfaces(p[valid], mesh[0][mesh[1]]) < dx
    if near.any():
        print("[Auralis] %d grid points closer than one wave-solver cell (%.2f m) to a surface "
              "are left out." % (int(near.sum()), dx), flush=True)
        valid &= ~near
    emit(stage="wave solver: %d points, up to %.0f Hz, %.1f s" % (int(valid.sum()), f_grid,
                                                                   duration), progress=0.05)
    t0 = time.time()
    tf, info = run_low_band(scene, src["position"], [q for q in p[valid]], fs=int(cfg.fs),
                            duration=duration, f_cross=f_grid, speed_of_sound=c,
                            engine=cfg.engine, max_cells=cfg.fdtd_max_cells,
                            order=int(cfg.fdtd_order), boundary_absorber=cfg.fdtd_absorber,
                            conformal=bool(cfg.fdtd_conformal), transfer_freqs=freqs)
    level = np.full((len(freqs), len(p)), np.nan)
    # level re the same source in free field at 10 m (the G convention), per frequency
    level[:, valid] = 20.0 * np.log10(np.maximum(np.abs(tf), 1e-12) / 0.1)
    np.savez(os.path.join(d, "maps.npz"), points=p, valid=valid, area_m2=area,
             freqs=np.asarray(info.transfer_freqs, float), level_db=level)
    meta = {"source_id": src["id"], "grid_id": grid["id"], "n_points": int(len(p)),
            "n_valid": int(valid.sum()), "freqs_hz": np.round(info.transfer_freqs, 3).tolist(),
            "dx_m": round(float(info.dx), 4), "duration_s": round(duration, 2),
            "engine": info.engine, "conformal": bool(info.conformal),
            "render_s": round(time.time() - t0, 1),
            "note": "mode frequencies carry the voxelisation bias of the wave solver: "
                    "measured on a 4 m box, first mode +4.6 % without the conformal "
                    "correction, -8.1 % with it"}
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
