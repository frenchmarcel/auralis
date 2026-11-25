# -*- coding: utf-8 -*-
"""Example project built on first launch: concerts in Odeon hall models (local data), the BRAS
rooms CR1-CR4 of TU Berlin, and a Paris street with real recordings.
Everything is derived from the source files, nothing is tuned."""
from __future__ import annotations

import json
import math
import os
import re
import threading

import numpy as np

from . import directivities as dirlib
from . import heads as headlib
from . import meshcheck, parfile
from . import sim as simlib
from . import sounds as soundlib
from . import trajectory as trajlib
from .store import new_id, now

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOMS = os.path.join(REPO, "data_odeon", "rooms")
MARK = "examples.json"
VERSION = 7               # 1: halls + CR2. 6: the Paris street with real recordings. 7: the
                          # BRAS rooms CR1, CR3 and CR4. The examples of versions 2-5 are no
                          # longer built: a store that has them keeps them.
BRAS_DIR = os.path.join(REPO, "BRAS")
# key, geometry name, bench manifest (BRAS/: geometry converted from BRAS, materials fitted by
# BRAS, measurement positions and air of the session), CC BY-SA 4.0
BRAS_ROOMS = [
    ("cr1_door30", "Coupled rooms, door open 30.4° (BRAS CR1)",
     "CR1/cr1_door3_dodecahedron_fitted31_dir.json"),
    ("cr1_door4", "Coupled rooms, door open 4.1° (BRAS CR1)",
     "CR1/cr1_door1_dodecahedron_fitted31_dir.json"),
    ("cr2", "Seminar room (BRAS CR2)", "CR2/cr2_dodecahedron_fitted31_dir.json"),
    ("cr3", "Chamber music hall (BRAS CR3)", "CR3/cr3_dodecahedron_fitted31_dir.json"),
    ("cr4", "Auditorium (BRAS CR4)", "CR4/cr4_dodecahedron_fitted31_dir.json"),
]
BRAS_DODECA = os.path.join(BRAS_DIR, "CR2", "ita_dodeca_directivity.json")
BRAS_DODECA_NAME = "ITA dodecahedron (BRAS)"
BRAS_CREDIT = ("BRAS database, TU Berlin (Brinkmann et al., DepositOnce), CC BY-SA 4.0")
HALLS = [  # key, hall, Odeon model, ensemble
    ("musikverein", "Musikverein, Vienna", "Musikverein", "mozart40"),
    ("goteborg", "Konserthus, Gothenburg", "Goteborg_Koncerthus", "mahler1"),
    ("rfh", "Royal Festival Hall, London", "Royal_Festival_Hall", "bruckner8"),
]
STAGE = re.compile(r"stage|podium|platform", re.I)
AUDIENCE = re.compile(r"audience|seat|stall|parterre|chair|parkett", re.I)
NOT_MAIN = re.compile(r"back ?stage|choir|box|balcon|gallery|wall|riser", re.I)
_lock = threading.Lock()


def _horizontal(tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.abs(n[:, 2]) > 0.9 * np.linalg.norm(n, axis=1)


def _stage_and_audience(v, f, fl, names):
    """Horizontal faces of the stage and of the main audience area, from surface names."""
    tri = v[f]
    horiz = _horizontal(tri)
    stage = np.array([bool(STAGE.search(names[k])) and not NOT_MAIN.search(names[k]) for k in fl])
    aud = np.array([bool(AUDIENCE.search(names[k])) and not NOT_MAIN.search(names[k]) for k in fl])
    return tri[stage & horiz], tri[aud & horiz]


def _floor_under(tri, xy):
    """Height of the lowest audience floor right under `xy` (stalls under a balcony)."""
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]

    def side(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[0] - p[:, 0])

    d1, d2, d3 = side(a, b, xy), side(b, c, xy), side(c, a, xy)
    hit = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    if hit.any():
        return float(tri[hit][:, :, 2].mean(axis=1).min())
    near = tri.reshape(-1, 3)
    near = near[np.linalg.norm(near[:, :2] - xy, axis=1) < 3.0]
    return float(near[:, 2].min()) if len(near) else float(np.median(tri[:, :, 2]))


def _placement(stage, aud):
    """Conductor on the stage front edge facing the audience, stage size, and two
    listening points on the hall axis (a third and three quarters into the stalls)."""
    w_s = meshcheck.triangle_areas(stage)
    w_a = meshcheck.triangle_areas(aud)
    cs = (stage.mean(axis=1) * w_s[:, None]).sum(0) / w_s.sum()
    ca = (aud.mean(axis=1) * w_a[:, None]).sum(0) / w_a.sum()
    a = (ca - cs)[:2]
    a /= np.linalg.norm(a)
    lat = np.array([-a[1], a[0]])
    ps, pa = stage.reshape(-1, 3), aud.reshape(-1, 3)
    u_s = (ps[:, :2] - cs[:2]) @ a
    front, back = u_s.max(), u_s.min()
    conductor = np.r_[cs[:2] + (front - 0.6) * a, np.median(ps[:, 2])]
    width = 0.9 * float(np.ptp((ps[:, :2] - cs[:2]) @ lat))
    depth = max(2.0, 0.9 * float(front - back) - 0.6)
    u_a = (pa[:, :2] - conductor[:2]) @ a
    far = float(np.percentile(u_a, 95))
    listeners = []
    for frac in (0.35, 0.75):
        xy = conductor[:2] + frac * far * a
        listeners.append(np.r_[xy, _floor_under(aud, xy) + 1.2])
    facing = math.degrees(math.atan2(a[1], a[0]))
    return conductor, facing, width, depth, listeners


def _nudge(sim, geometry, mesh, library, closure):
    """Move invalid receivers sideways/up until the validation accepts them."""
    for _ in range(12):
        v = simlib.validate(sim, geometry, mesh, library, closure)
        bad = [r for r in v["receivers"] if not r["valid"]]
        if not bad:
            return v
        for r in sim["receivers"]:
            if any(b["id"] == r["id"] for b in bad):
                r["position"][0] += 0.7
                r["position"][2] += 0.2
    return simlib.validate(sim, geometry, mesh, library, closure)


def _hall(store, library, sid, key, hall, model, ensemble_id):
    par = os.path.join(ROOMS, model + ".Par")
    matfile = os.path.join(ROOMS, model + ".Mat")
    ens = next((e for e in soundlib.ensembles() if e["id"] == ensemble_id), None)
    if not (os.path.exists(par) and os.path.exists(matfile) and ens):
        return None
    v, f, fl, surf_names, rep = parfile.read_par(par)
    mat = parfile.read_mat(matfile)
    if mat is None:
        return None
    stage, aud = _stage_and_audience(v, f, fl, surf_names)
    if not len(stage) or not len(aud):
        return None
    # one layer per material of the Odeon model, with the model's own coefficients
    table = mat["materials"]
    face_mat, face_scat = [], []
    for s in rep["face_surface"]:
        midx, scat = mat["surfaces"].get(int(s), mat["surfaces"].get(-int(s), (1, 0.05)))
        face_mat.append(min(max(midx, 1), len(table)) - 1)
        face_scat.append(scat)
    face_mat, face_scat = np.array(face_mat), np.array(face_scat)
    used = sorted(set(face_mat.tolist()))
    names = []
    for k in used:
        nm = table[k][2] or "Material %d" % (k + 1)
        names.append(nm if nm not in names else "%s (%d)" % (nm, k + 1))
    fl2 = np.array([used.index(k) for k in face_mat], dtype=np.int32)
    closure = {}
    summary, closed = meshcheck.analyse(v, f, fl2, names, closure_out=closure)
    for layer in summary["layers"]:
        layer["kind"] = "room"
    with open(par, "rb") as fh:
        src = fh.read()
    meta = dict(summary, name=hall, unit_in="m", up_axis_in="z", objects=[],
                import_notes=["Odeon model with its own material table (example)"])
    g = store.create_geometry(sid, meta, v, f, fl2, names, closed, src, model + ".Par",
                              closure=closure or None)
    m = store.create_simulation(g["id"], "Concert – " + ens["title"].split(" (")[0])
    for k, name in zip(used, names):
        lib_id, alpha, _ = table[k]
        scat = float(np.mean(face_scat[face_mat == k]))
        mm = library.add_local("odeon_rooms", "%s:%d" % (key, k + 1), table[k][2], "Odeon · " + hall,
                               alpha[1:8], scat,
                               "Odeon model of the %s, material #%d (library id %d)" % (hall, k + 1, lib_id),
                               a63=alpha[0])
        m["layers"][name] = {"material_id": mm["id"], "scattering": round(scat, 3)}
    conductor, facing, width, depth, listeners = _placement(stage, aud)
    mesh, closure_g = store.load_mesh(g["id"]), store.load_closure(g["id"])
    for _ in range(8):
        # the stage outline from surface names can reach past its back wall: shrink
        # the layout until every player is in the air
        m["sources"], _scale = soundlib.ensemble_sources(ens, conductor.tolist(), facing, width,
                                                         depth, "E1_")
        m["receivers"] = []
        chk = simlib.validate(m, g, mesh, library, closure_g)
        if all(x["valid"] for x in chk["sources"]):
            break
        depth *= 0.85
        width *= 0.95
    m["receivers"] = [{"id": "R%d" % (i + 1), "name": nm, "position": p.round(3).tolist(),
                       "azimuth_deg": round((facing + 180) % 360, 1), "elevation_deg": 0.0}
                      for i, (nm, p) in enumerate(zip(("Stalls, centre", "Stalls, rear"), listeners))]
    m["description"] = ("%s on the stage of the %s, with the materials of the Odeon model."
                        % (ens["title"].split(" (")[0], hall))
    v_ = _nudge(m, g, mesh, library, closure_g)
    store.save_simulation(m)
    return {"simulation": m["id"], "ok": v_["ok_to_run"], "blocking": v_["blocking"]}


def _bras_dodecahedron(store):
    """Id of the ITA dodecahedron of BRAS among the store's imported directivities, imported
    once from the bench's table (its spectrum included)."""
    for d in dirlib.listing(store.root):
        if d.get("kind") == "imported" and d.get("name") == BRAS_DODECA_NAME:
            return d["id"]
    return dirlib.import_file(store.root, BRAS_DODECA, BRAS_DODECA_NAME,
                              os.path.basename(BRAS_DODECA))["id"]


def _bras_room(store, library, sid, key, name, manifest_path):
    """A BRAS room as in the validation bench: its geometry, its materials (absorption and
    scattering per octave, 63 Hz from the mean of its three third octaves), the dodecahedron at
    the measured LS positions, the microphones at the MP positions (facing LS1) and the air of
    the session. The positions are kept exactly: nothing is nudged."""
    path = os.path.join(BRAS_DIR, manifest_path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        man = json.load(fh)
    obj = os.path.join(os.path.dirname(path), man["geometry"]["obj"])
    v, f, fl, names, notes = meshcheck.import_model(obj, "m", "z")
    closure = {}
    summary, closed = meshcheck.analyse(v, f, fl, names, closure_out=closure)
    for layer in summary["layers"]:
        layer["kind"] = "room"
    with open(obj, "rb") as fh:
        src = fh.read()
    meta = dict(summary, name=name, unit_in="m", up_axis_in="z", objects=[],
                import_notes=["BRAS geometry converted for the validation bench (BRAS/%s), "
                              "CC BY-SA 4.0" % os.path.dirname(manifest_path)] + list(notes))
    g = store.create_geometry(sid, meta, v, f, fl, names, closed, src, os.path.basename(obj),
                              closure=closure or None)
    room = key.split("_")[0]
    m = store.create_simulation(g["id"], "BRAS measurement positions")
    for layer in names:
        spec = man["materials"].get(layer)
        if spec is None:
            continue
        full = spec.get("absorption_full") or {}
        a63 = None
        if full.get("f"):
            fa = dict(zip(full["f"], full["alpha"]))
            if all(x in fa for x in (50.0, 63.0, 80.0)):
                a63 = round(float(np.mean([fa[50.0], fa[63.0], fa[80.0]])), 4)
        scat = spec.get("scattering", 0.1)
        scat_mean = float(np.mean(scat))
        label = layer.replace("mat_%s_" % room.upper(), "")
        mm = library.add_local("bras", "%s:%s" % (room, layer), "BRAS %s – %s" % (room.upper(), label),
                               "BRAS " + room.upper(), spec["absorption"], scat_mean,
                               "BRAS %s bench manifest (absorption fitted by BRAS to the measured "
                               "reverberation time), CC BY-SA 4.0" % room.upper(), a63=a63)
        m["layers"][layer] = {"material_id": mm["id"],
                              "scattering": ([round(float(x), 4) for x in scat]
                                             if isinstance(scat, list) else round(scat_mean, 4))}
    eng = man.get("engine", {})
    m["settings"].update(temperature_c=float(eng.get("temperature_c", 20.0)),
                         humidity_percent=float(eng.get("humidity_percent", 50.0)))
    if room == "cr1":
        # the reverberation chamber decays in 5-7.5 s (T30 at 125 Hz-2 kHz): the automatic length,
        # from the whole model's Eyring estimate, ends before a 45 dB decay there
        m["settings"].update(mode="advanced", solvers="geometrical", ir_length_s=8.0)
    dodeca = _bras_dodecahedron(store)
    pos = {}
    for pr in man["pairs"]:
        ls, mp = pr["label"].split("-")
        pos[ls], pos[mp] = pr["source"], pr["receiver"]
    sources = sorted(k for k in pos if k.startswith("LS"))
    mics = sorted((k for k in pos if k.startswith("MP")), key=lambda k: int(k[2:]))
    speech = next((s for s in soundlib.odeon_sounds().values()
                   if s["track"] == "SI Harvard Word Lists Male"), None)
    m["sources"] = [{"id": ls, "name": "Dodecahedron %s" % ls,
                     "position": [float(x) for x in pos[ls]], "directivity": dodeca,
                     "azimuth_deg": 0.0, "elevation_deg": 90.0, "gain_db": 0.0,
                     **({"sound_id": speech["id"]} if speech and ls == sources[0] else {})}
                    for ls in sources]
    s1 = np.asarray(pos[sources[0]], float)
    m["receivers"] = []
    for mp in mics:
        d = s1 - np.asarray(pos[mp], float)
        m["receivers"].append({"id": mp, "name": "Microphone %s" % mp,
                               "position": [float(x) for x in pos[mp]],
                               "azimuth_deg": round(math.degrees(math.atan2(d[1], d[0])) % 360, 1),
                               "elevation_deg": 0.0})
    m["description"] = ("%s from the BRAS database: measured geometry, materials, positions and "
                        "air. Data: %s." % (name, BRAS_CREDIT))
    store.save_simulation(m)
    v_ = simlib.validate(m, g, store.load_mesh(g["id"]), library, store.load_closure(g["id"]))
    return {"simulation": m["id"], "ok": v_["ok_to_run"], "blocking": v_["blocking"],
            "example": name}


FABIAN_CREDIT = ("FABIAN HRTF database, TU Berlin (Brinkmann et al., JAES 2017), CC BY 4.0")


def _street_sound(store, name, x, fs, credit=None):
    """A sound of the store's library (sounds/<id>.wav + .json), like an uploaded one."""
    from auralis.engine.wavio import write_wav
    d = os.path.join(store.root, "sounds")
    sid = new_id("u")
    write_wav(os.path.join(d, sid + ".wav"), np.asarray(x, np.float32), fs)
    meta = {"id": sid, "name": name, "kind": "uploaded", "duration_s": round(len(x) / fs, 2),
            "created": now()}
    if credit:
        meta["credit"] = credit
    with open(os.path.join(d, sid + ".json"), "w") as fh:
        json.dump(meta, fh)
    return sid, os.path.join(d, sid + ".wav")


PARIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "examples", "paris")
PARIS_T, PARIS_FS = 40.0, 32000
# City hum: ISO 717-1 spectrum No. 2 (urban road traffic, A-weighted, normalised), summed into
# octaves (-14.5 / -10.2 / -7.2 / -3.9 / -6.4 dB, the standard's octave spectrum), 4 and 8 kHz
# continued at its -2 dB per third-octave slope, set to 60 dB(A): the background of a busy
# city centre. Octave levels, dB re 20 uPa, 125 Hz - 8 kHz.
CITY_HUM_60DBA = [61.5, 58.3, 55.9, 55.9, 52.2, 46.9, 43.0]
# Emission levels (orders of magnitude of the road-noise literature, CNOSSOS-EU / Harmonoise; not
# fitted): a car at 30 km/h passes at ~66 dB(A) at 7.5 m, i.e. 83.5 dB(A) at 1 m; with a
# background noise the renders set every track to 60 dB(A) at 1 m first (speech), hence the gains.
CAR_GAIN_DB = 83.5 - 60.0
FLOW_GAIN_DB = {"boulevard": 24.0, "street": 18.0}     # a lane of traffic seen from its end
PARIS_LIGHTING = {"preset": "daylight", "env_intensity": 1.0, "sun_intensity": 3.0,
                  "sun_azimuth_deg": 210.0, "sun_elevation_deg": 38.0, "sun_color": "#fff3e0",
                  "view_lights": True, "exposure_ev": 0.0}


def _paris_simulation(store, library, sid, sim_name):
    """The Paris street's geometry (OpenStreetMap; its textured display model and lighting), its
    three materials and a simulation of it: (geometry, simulation, layer names)."""
    import shutil
    obj = os.path.join(PARIS_DIR, "street.obj")
    v, f, fl, names, notes = meshcheck.import_model(obj, unit="m", up_axis="z")
    summary, closed = meshcheck.analyse(v, f, fl, names)
    for layer in summary["layers"]:
        layer["kind"] = "room"
    meta = dict(summary, name="Réaumur Street, Paris (OpenStreetMap)", unit_in="m", up_axis_in="z",
                objects=[], import_notes=["(c) OpenStreetMap contributors, ODbL; heights from "
                                          "storey counts (tools/osm_street.py)"] + list(notes))
    with open(obj, "rb") as fh:
        g = store.create_geometry(sid, meta, v, f, fl, names, closed, fh.read(), "rue_reaumur.obj")
    glb = os.path.join(PARIS_DIR, "reaumur.glb")
    if os.path.exists(glb):
        shutil.copyfile(glb, os.path.join(store.geometry_dir(g["id"]), "visual.glb"))
        g["visual"] = {"filename": "reaumur.glb", "bytes": os.path.getsize(glb), "uploaded": now(),
                       "transform": {"position": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0],
                                     "scale": 1.0, "up_axis": "y"},
                       "lighting": dict(PARIS_LIGHTING)}
        store.save_geometry(g)
    stone = [0.6 * a + 0.4 * b for a, b in zip((0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05),
                                               (0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02))]
    mats = {}
    for n, label, ab, sc in (("facade", "facade (limestone, 40 % windows)", stone, 0.2),
                             ("roof", "roof (zinc)", (0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04), 0.1),
                             ("ground", "road and sidewalks (asphalt)",
                              (0.01, 0.01, 0.015, 0.02, 0.02, 0.02, 0.02), 0.1)):
        # names are unique: a store that has the other Paris example reuses its materials
        found = [e for e in library.custom if e.get("name") == "Paris street example – " + label]
        mats[n] = (found[0]["id"] if found else library.add_custom(
            "Paris street example – " + label, "Paris street example",
            "Paris street example.", list(ab), sc)["id"])
    m = store.create_simulation(g["id"], sim_name)
    for n in names:
        m["layers"][n] = {"material_id": mats.get(n, mats["facade"]), "scattering": None}
    return g, m, names


def _paris_drive(a, b, speed, t0, T=PARIS_T):
    """Hold at a until t0, drive to b at speed, hold there; waypoints within [0, T]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    t1 = t0 + L / speed
    wps = [(a, 0.0)] + ([(a, t0)] if t0 > 0 else [])
    if t1 <= T:
        wps += [(b, t1)] + ([(b, T)] if t1 < T else [])
    else:
        wps += [(a + (b - a) * (T - t0) / (t1 - t0), T)]
    return trajlib.clean_path({"waypoints": [{"position": [round(float(x), 3) for x in p],
                                              "t": round(t, 3)} for p, t in wps],
                               "orientation": "follow"}, "path")


PARIS_LANES = (-2.2, 1.3, 4.8)                   # lanes centred between the facades
PARIS_V30 = 30.0 / 3.6


def _paris_pedestrian(m, T=PARIS_T):
    a, b = (70.0, -6.8, 1.6), (70.0 + 1.4 * T, -6.8, 1.6)
    m["receivers"] = [{"id": "R1", "name": "Pedestrian (south sidewalk, walking east)",
                       "position": list(a), "azimuth_deg": 0.0, "elevation_deg": 0.0,
                       "path": trajlib.clean_path({"waypoints": [{"position": list(a), "t": 0.0},
                                                                 {"position": list(b), "t": T}],
                                                   "orientation": "follow"}, "path")}]


def _paris_ship_render(store, m, req, wav, meta_json, credits):
    """A finished animated render of m from the example's files: True when they exist."""
    import shutil
    if not os.path.exists(wav):
        return False
    info = store.create_motion(m["id"], req)
    d = store.motion_dir(m["id"], info["id"])
    shutil.copyfile(wav, os.path.join(d, "render.wav"))
    meta = json.load(open(meta_json)) if os.path.exists(meta_json) else {
        "fs": PARIS_FS, "duration_s": PARIS_T, "render_s": None, "credits": [FABIAN_CREDIT],
        "doppler": True, "warnings": [], "background_noise": None, "turns": None}
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(d, "credits.txt"), "w", encoding="utf-8") as fh:
        fh.write("Paris street walk. Buildings: (c) OpenStreetMap contributors, ODbL.\n"
                 "Credits (required by the licences):\n" + "".join(
                     "  - %s\n" % c for c in credits))
    store.set_motion(m["id"], info["id"], status="done", finished=now())
    return True


PARIS_SOUNDS_DIR = os.path.join(PARIS_DIR, "sounds")
# a heavy vehicle at 30 km/h: ~6 dB over a car (order of magnitude of CNOSSOS-EU, not fitted)
BUS_GAIN_DB = CAR_GAIN_DB + 6.0
CROWD_GAIN_DB = 6.0                  # a group of a few people: +6 dB of a normal voice
PARIS_HUM_DBA = 60.0                 # the recorded city hum at the pedestrian, as CITY_HUM_60DBA
# pedestrian groups: (id, name, position, recording, offset in it (s)), 1.1-3 m off the facades
# as OpenStreetMap draws them (the south side steps back to y -13...-17 west of Saint-Denis Street;
# the walk is at y -6.8); Saint-Denis Street (pedestrian) between x 43.5 and 59.5
PARIS_CROWDS = [("G1", "People, south side (Saint-Denis corner)", (38.0, -11.8, 1.6), 566891, 0.0),
                ("G2", "People, south sidewalk (cafe)", (88.0, -7.9, 1.6), 634880, 0.0),
                ("G3", "People, south sidewalk (boulevard corner)", (141.0, -9.0, 1.6), 869048, 0.0),
                ("G4", "People, north sidewalk (west)", (8.0, 9.4, 1.6), 634880, 25.0),
                ("G5", "People, north sidewalk (Saint-Denis corner)", (62.0, 9.8, 1.6), 869048, 30.0),
                ("G6", "People, north sidewalk (east)", (110.0, 9.9, 1.6), 566891, 30.0),
                ("G7", "People, Saint-Denis Street", (51.0, 20.0, 1.6), 566891, 15.0),
                ("G8", "People, south side (west)", (-30.0, -14.0, 1.6), 869048, 15.0)]
PARIS_REC_RENDERS = (
    ("walk_recordings_full_physics", "Full physics (wave band to 250 Hz)", "both"),
    ("walk_recordings_geometrical", "Geometrical solver only", "geometrical"))
# short names and credits of the Freesound recordings (titles and links are in sounds.json)
PARIS_SOUND_NAMES = {522221: "Car (Peugeot 308)", 268622: "Car (VW Golf)", 451668: "Bus engine",
                     635757: "Boulevard traffic", 451516: "Side street traffic",
                     512142: "Paris traffic hum", 566891: "People talking 1",
                     634880: "People talking 2", 869048: "People talking 3"}


def _short_credit(e):
    """'author (freesound.org/s/id), CC0' from a sounds.json entry."""
    m = re.search(r"^.* by ([^()]+?) \((freesound\.org/s/\d+)/?\), (CC0)", e["credit"])
    return "%s (%s), %s" % m.groups() if m else e["credit"]


def _measured_head():
    """The FABIAN head when it is installed (data_hrtf/, not in the repository), else the model."""
    return "fabian" if any(h["id"] == "fabian" and h["available"] for h in headlib.listing()) \
        else "model"


def _paris_recording(store, rec_id, index, cache):
    """(sound id, path, credit) of a recording of data/examples/paris/sounds in the store's
    library, added once per recording."""
    if rec_id not in cache:
        e = index[str(rec_id)]
        x, fs = soundlib.load_sound(os.path.join(PARIS_SOUNDS_DIR, e["file"]))
        credit = _short_credit(e)
        sid, wav = _street_sound(store, PARIS_SOUND_NAMES[rec_id], x, fs, credit=credit)
        cache[rec_id] = (sid, wav, credit)
    return cache[rec_id]


def _paris_recordings(store, library, sid):
    """The Paris street walk with real CC0 recordings from Freesound. Its two renders come
    finished: nothing runs."""
    with open(os.path.join(PARIS_SOUNDS_DIR, "sounds.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    g, m, names = _paris_simulation(store, library, sid,
                                    "Walk to Sébastopol Boulevard")
    cache = {}
    hum_sid, hum_wav, hum_credit = _paris_recording(store, 512142, index, cache)
    m["settings"].update(mode="advanced", solvers="both", transition_hz=250.0, open_scene=True,
                         head=_measured_head(), ir_length_s=1.5, background_noise=list(CITY_HUM_60DBA),
                         background_recording={"sound_id": hum_sid, "dba": PARIS_HUM_DBA,
                                               "path": hum_wav, "name": PARIS_SOUND_NAMES[512142],
                                               "credit": hum_credit})
    lanes, v30 = PARIS_LANES, PARIS_V30
    movers = [("C1", "Car 1", (-100, lanes[1]), (200, lanes[1]), 0.0, 522221, 0.0, CAR_GAIN_DB),
              ("C2", "Car 2", (-100, lanes[0]), (200, lanes[0]), 11.0, 268622, 0.0, CAR_GAIN_DB),
              ("C3", "Car 3", (-100, lanes[2]), (200, lanes[2]), 22.0, 522221, 20.0, CAR_GAIN_DB),
              ("B1", "Bus (boulevard)", (157.2, -60), (162.0, 60), 5.0, 451668, 0.0, BUS_GAIN_DB)]
    flows = [("F1", "Traffic, west end", (-98.0, 1.3, 0.6), 451516, 0.0, "street"),
             ("F2", "Traffic, boulevard south", (156.5, -60.0, 0.6), 635757, 0.0, "boulevard"),
             ("F3", "Traffic, boulevard north", (162.0, 60.0, 0.6), 635757, 30.0, "boulevard")]
    strips = []

    def add(src, rec_id, offset):
        snd, wav, credit = _paris_recording(store, rec_id, index, cache)
        m["sources"].append(dict(src, directivity="omni", azimuth_deg=0.0, elevation_deg=0.0,
                                 sound_id=snd))
        strips.append({"source_id": src["id"], "sound_id": snd, "gain_db": 0.0, "path": wav,
                       "credit": credit, "loop": True, "offset_s": float(offset)})
    for cid, cname, a, b, t0, rec, off, gain in movers:
        path = _paris_drive((a[0], a[1], 0.5), (b[0], b[1], 0.5), v30, t0)
        add({"id": cid, "name": cname, "position": path["waypoints"][0]["position"],
             "gain_db": gain, "path": path}, rec, off)
    for fid, fname, pos, rec, off, kind in flows:
        add({"id": fid, "name": fname, "position": list(pos), "gain_db": FLOW_GAIN_DB[kind]},
            rec, off)
    for gid, gname, pos, rec, off in PARIS_CROWDS:
        add({"id": gid, "name": gname, "position": list(pos), "gain_db": CROWD_GAIN_DB},
            rec, off)
    _paris_pedestrian(m)
    m["description"] = ("Réaumur Street in Paris, from OpenStreetMap, with real recordings. A "
                        "pedestrian walks to Sébastopol Boulevard while cars and a bus pass. "
                        "Two finished renders: full physics, and the geometrical solver only.")
    store.save_simulation(m)
    v_ = simlib.validate(m, g, store.load_mesh(g["id"]), library, store.load_closure(g["id"]))
    out = {"simulation": m["id"], "ok": v_["ok_to_run"], "blocking": v_["blocking"],
           "queue": False, "example": "Paris street"}
    credits = sorted({st["credit"] for st in strips} | {hum_credit}) + [FABIAN_CREDIT]
    requests = {}
    for stem, label, solvers in PARIS_REC_RENDERS:
        sim = json.loads(json.dumps(m))
        sim["settings"]["solvers"] = solvers
        req = {"simulation": sim, "receiver_id": "R1", "strips": strips, "duration_s": PARIS_T,
               "doppler": True, "reverb_dt": 0.75, "master_db": 0.0, "label": label,
               "turn_taking": None, "background_noise": True}
        if solvers == "both":
            req["wave_max_runs"] = 4000          # a run per key of the walker's ears, uncapped
        if not _paris_ship_render(store, m, req, os.path.join(PARIS_DIR, stem + ".wav"),
                                  os.path.join(PARIS_DIR, stem + ".json"), credits):
            requests[stem] = req
    if requests:
        out["motion_requests"] = requests
    return out


def install(store, library, jobs=None, run=True, force=False):
    """Create the 'Examples' project (once, unless forced) and queue its runs; a store
    made by an older version only gets the examples added since."""
    with _lock:
        mark = os.path.join(store.root, MARK)
        old = None
        if os.path.exists(mark) and not force:
            with open(mark) as fh:
                old = json.load(fh)
            if old.get("version", 1) >= VERSION:
                return old
        since = old.get("version", 1) if old else 0
        p = None
        if old:
            try:
                p = store.get_project(old["project_id"])
            except KeyError:                      # deleted: the new examples get a new one
                p = None
        if p is None:
            p = store.create_project("Examples",
                                     "Ready-made scenes: concerts in Odeon's hall models (their own "
                                     "materials, local licence), the BRAS rooms of TU Berlin "
                                     "(measured geometry, materials and positions) and a Paris "
                                     "street with real recordings.")

        def guarded(fn, *a):
            try:
                return fn(store, library, *a)
            except Exception as exc:              # an example must never break the app
                return {"error": "%s: %s" % (type(exc).__name__, exc)}

        built = []
        if since < 1:
            s = store.create_space(p["id"], "Concert halls", "")
            halls = [dict(r, example=spec[1]) for spec in HALLS
                     for r in [guarded(_hall, s["id"], *spec)] if r]
            built.extend(halls)
            if not halls:                         # no local Odeon data: no empty space
                store.delete_space(s["id"])
        if since < 7:
            # a store from before version 7 already has CR2 (its own space): the others join
            rooms = BRAS_ROOMS if since < 1 else [x for x in BRAS_ROOMS if x[0] != "cr2"]
            s = store.create_space(p["id"], "BRAS rooms (TU Berlin)", "")
            for spec in rooms:
                r = guarded(_bras_room, s["id"], *spec)
                if r:
                    built.append(dict(r, example=spec[1]))
        if since < 6:
            r = guarded(_paris_recordings,
                        store.create_space(p["id"], "Paris street", "")["id"])
            if r:
                r.pop("motion_requests", None)
                built.append(dict(r, example=r.get("example",
                                                   "Paris street")))
        motions = {b["simulation"]: b.pop("motion", None) for b in built if b.get("simulation")}
        if run and jobs is not None:
            for b in built:
                if b.get("ok") and b.get("queue", True):      # the street comes pre-rendered
                    m = store.get_simulation(b["simulation"])
                    jobs.submit(m)
                    req = motions.get(b["simulation"])
                    if req:
                        rinfo = store.create_motion(m["id"], req)
                        jobs.submit_motion(m, rinfo)
                        b["motion_render"] = rinfo["id"]
        info = {"version": VERSION, "project_id": p["id"],
                "examples": (old["examples"] if old and old.get("project_id") == p["id"] else [])
                + built}
        with open(mark, "w") as fh:
            json.dump(info, fh, indent=1)
        return info
