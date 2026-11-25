# -*- coding: utf-8 -*-
"""Animated binaural render (moving sources and/or receiver) in its own process.
Usage: python -m webapp.server.motion_runner <store_root> <simulation_id> <render_id>
Same protocol as runner.py: progress lines '@@{json}', the rest is the engine log."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from auralis.engine.level import dba_total                       # noqa: E402
from auralis.engine.motion import (MotionConfig, MovingReceiver,  # noqa: E402
                                   MovingSource, render_scene_moving)
from auralis.engine.air import config_speed_of_sound, speed_of_sound  # noqa: E402
from auralis.engine.project import build_config, build_scene     # noqa: E402
from auralis.engine import speech, write_wav                     # noqa: E402
from webapp.server import directivities as dirlib                # noqa: E402
from webapp.server import sourceeq  # noqa: E402
from webapp.server import heads  # noqa: E402
from webapp.server import sim as simlib                          # noqa: E402
from webapp.server import sounds as soundlib                     # noqa: E402
from webapp.server import trajectory as trajlib                  # noqa: E402
from webapp.server.library import Library                         # noqa: E402
from webapp.server.runner import choose_engine, emit             # noqa: E402
from webapp.server.store import Store                             # noqa: E402

def run(store_root, mid, rid):
    store = Store(store_root)
    rdir = store.motion_dir(mid, rid)
    with open(os.path.join(rdir, "request.json"), encoding="utf-8") as fh:
        req = json.load(fh)
    req.setdefault("store_root", store.root)
    gid = req["simulation"]["geometry_id"]     # snapshot taken when the render was queued
    render(req, store.get_geometry(gid), store.load_mesh(gid),
           Library(os.path.join(store.root, "materials_custom.json")), store.load_closure(gid),
           rdir)


def _background(req, sim):
    """Background noise of the simulation's settings, when the request mixes it in: a
    recorded ambience (settings.background_recording {"path", "dba", "name"}) when set, else
    the octave levels of settings.background_noise."""
    if not req.get("background_noise"):
        return None
    st = sim["settings"]
    rec = st.get("background_recording")
    if rec and rec.get("path"):
        effort = st.get("speech_effort", "normal")
        return {"curve": "recording: %s" % rec.get("name", os.path.basename(rec["path"])),
                "recording": rec["path"], "levels_db": None, "dba": float(rec["dba"]),
                "speech_dba_1m": speech.VOCAL_EFFORT_DBA_1M[effort], "effort": effort,
                "credit": rec.get("credit")}
    levels = speech.noise_levels(st.get("background_noise", "none"))
    if levels is None:
        print("[Auralis] WARNING: background noise asked, but the simulation has none "
              "(Settings > background noise): rendered without.", flush=True)
        return None
    effort = st.get("speech_effort", "normal")
    return {"curve": st["background_noise"] if isinstance(st["background_noise"], str)
            else "custom", "levels_db": levels, "dba": round(dba_total(levels), 1),
            "speech_dba_1m": speech.VOCAL_EFFORT_DBA_1M[effort], "effort": effort}


def render(req, geometry, mesh, library, closure, rdir):
    """Render request `req` into rdir/render.wav (+ credits.txt, meta.json)."""
    sim = req["simulation"]
    strips = req["strips"]
    check = simlib.validate_motion(sim, geometry, mesh, library, closure, req["receiver_id"],
                                   [s["source_id"] for s in strips], req.get("duration_s"))
    if not check["ok"]:
        raise ValueError("not renderable: " + "; ".join(check["blocking"]))
    for w in check["warnings"]:
        print("[Auralis] WARNING: %s." % w, flush=True)
    duration = check["duration_s"]

    open_scene = simlib.is_open_scene(sim["settings"], geometry)
    man = simlib.build_manifest(sim, mesh, library, open_scene)
    scene = build_scene(man, "")
    cfg = build_config(man)
    cfg.fdtd = False
    choose_engine(cfg)

    tracks = [soundlib.load_sound(s["path"]) for s in strips]
    fs = tracks[0][1]
    cfg.fs = fs
    n = int(round(duration * fs))
    by_id = {x["id"]: x for x in sim["sources"] + sim["receivers"]}
    t_end = duration           # positions are held after the end, while the room decays
    xs = [soundlib._resample(x, f, fs) for x, f in tracks]
    # a strip may start later in its recording (offset_s: several sources sharing one
    # recording then play different passages); continuous sounds (traffic, crowds) shorter
    # than the render are looped, cross-faded
    xs = [np.roll(x, -int(round(float(st.get("offset_s", 0.0)) * fs)) % max(len(x), 1))
          if st.get("offset_s") else x for st, x in zip(strips, xs)]
    xs = [soundlib.loop_to(x, n, fs) if st.get("loop") and len(x) < n else x
          for st, x in zip(strips, xs)]
    noise = _background(req, sim)
    if noise is not None:          # in pascals: the tracks become speech at the vocal effort
        xs = [x * soundlib.speech_gain(x, fs, noise["speech_dba_1m"]) for x in xs]
    turns = None
    if req.get("turn_taking"):
        xs, turns = soundlib.take_turns(xs, fs, n, gap_s=float(req["turn_taking"]["gap_s"]))
    else:
        xs = [np.pad(x[:n], (0, n - len(x[:n]))) for x in xs]
    pairs = []
    for st, x in zip(strips, xs):
        src = by_id[st["source_id"]]
        eq = sourceeq.clean_eq(src.get("eq_db"))
        if eq is not None:              # the source's per-octave EQ, on its signal
            x = sourceeq.filter_ir(x, sourceeq.eq_fir(eq, fs))
        ms = MovingSource(name=src["id"], trajectory=trajlib.engine_trajectory(src, by_id, t_end),
                          gain_db=float(src.get("gain_db", 0.0)) + float(st.get("gain_db", 0.0)),
                          directivity=dirlib.engine_directivity(req.get("store_root", ""),
                                                                src.get("directivity")))
        pairs.append((ms, x))
    rcv = by_id[req["receiver_id"]]
    mr = MovingReceiver(name=rcv["id"], trajectory=trajlib.engine_trajectory(rcv, by_id, t_end))
    mc = MotionConfig(base=cfg, hop=int(req.get("hop", 2048)),
                      reverb_dt=float(req.get("reverb_dt", 0.75)),
                      doppler=bool(req.get("doppler", True)),
                      tail_rays=req.get("tail_rays") or None,
                      wave=simlib.uses_wave(sim["settings"]),
                      # the wave band's run cap when source and listener both move (64 by
                      # default: a laptop); a big node can afford every key frame
                      wave_max_runs=max(2, int(req.get("wave_max_runs", 64))))
    names = {s["source_id"]: by_id[s["source_id"]]["name"] for s in strips}

    def progress(frac, stage):
        emit(stage=stage, progress=float(frac))
        return True

    t0 = time.time()
    mix, info = render_scene_moving(scene, pairs, mr, mc, progress=progress)
    if noise is not None and noise.get("recording"):
        # a recorded ambience as the diffuse bed (settings.background_recording)
        rec, rfs = soundlib.load_sound(noise["recording"])
        mix += soundlib.diffuse_recording(soundlib._resample(rec, rfs, fs), len(mix), fs,
                                          noise["dba"], config_speed_of_sound(cfg))
    elif noise is not None:
        mix += soundlib.diffuse_noise(noise["levels_db"], len(mix), fs,
                                      np.random.default_rng(cfg.seed ^ 0xB0B),
                                      config_speed_of_sound(cfg))
    mix *= 10 ** (float(req.get("master_db", 0.0)) / 20.0)
    peak = float(np.abs(mix).max()) or 1.0
    norm = 0.891 / peak                                   # -1 dBFS, as the static renders
    mix *= norm
    write_wav(os.path.join(rdir, "render.wav"), mix.astype(np.float32), fs)
    credits = sorted({s["credit"] for s in strips if s.get("credit")})
    head = heads.HEADS.get(sim["settings"].get("head", "model"), {})
    if head.get("credit"):                       # a measured head: its licence asks for credit
        credits.append(head["credit"])
    if noise is not None and noise.get("credit"):          # a recorded ambience
        credits.append(noise["credit"])
    with open(os.path.join(rdir, "credits.txt"), "w", encoding="utf-8") as fh:
        fh.write("Auralis animated binaural render - %d sources, receiver %s\n\n"
                 % (len(strips), rcv["name"]))
        if credits:
            fh.write("Credits (required by the recordings' licences):\n")
            for c in credits:
                fh.write("  - %s\n" % c)
            fh.write("\n")
        fh.write("Sources and recordings:\n")
        for s in strips:
            fh.write("  - %s: %s (%+.1f dB)%s\n" % (
                names[s["source_id"]], os.path.basename(s["path"]), float(s.get("gain_db", 0.0)),
                " - moving" if by_id[s["source_id"]].get("path") else ""))
        if noise is not None:
            fh.write("\nBackground noise: %s (%.1f dB(A) at the ears, diffuse); the recordings "
                     "are taken as speech at %.0f dB(A) at 1 m on axis (%s effort).\n"
                     % (noise["curve"], noise["dba"], noise["speech_dba_1m"], noise["effort"]))
        if turns is not None:
            fh.write("\nConversation: the recordings take turns, cut at their own pauses, "
                     "%.2f s apart (%d turns).\n" % (float(req["turn_taking"]["gap_s"]),
                                                     len(turns)))
    meta = {"fs": fs, "duration_s": round(len(mix) / fs, 2), "render_s": round(time.time() - t0, 1),
            "normalization_db": round(20 * math.log10(norm), 2), "credits": credits,
            "engine_used": info["sources"][0]["engine_used"] if info["sources"] else None,
            "doppler": mc.doppler, "reverb_dt": mc.reverb_dt, "hop": mc.hop,
            "n_keys": [i["n_keys"] for i in info["sources"]],
            "timings": [i["timings"] for i in info["sources"]], "warnings": check["warnings"],
            "wave": info.get("wave"),
            "background_noise": None if noise is None else {
                k: noise[k] for k in ("curve", "dba", "speech_dba_1m", "effort")},
            "turns": None if turns is None else [
                {"source_id": strips[i]["source_id"], "start_s": round(a, 3), "end_s": round(b, 3)}
                for i, a, b in turns]}
    with open(os.path.join(rdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    emit(stage="done", progress=1.0)


if __name__ == "__main__":
    try:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as exc:
        traceback.print_exc()
        emit(error="%s: %s" % (type(exc).__name__, exc))
        sys.exit(1)
