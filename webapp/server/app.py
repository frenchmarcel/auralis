# -*- coding: utf-8 -*-
"""FastAPI application: the HTTP contract of webapp/API.md."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from auralis.engine.convolve import make_test_signal
from auralis.engine import speech as speechlib
from auralis.engine.wavio import read_wav, write_wav

from . import curves as curveslib
from . import combine as combinelib
from . import directivities as dirlib
from . import examples
from . import grids as gridlib
from . import heads as headlib
from . import exports as exportlib
from . import meshcheck
from . import objects as objlib
from . import reflections as refllib
from . import sim as simlib
from . import sounds as soundlib
from . import subsurfaces as sublib
from . import trajectory as trajlib
from .jobs import JobManager
from . import sourceeq
from .library import Library, porous_alpha, porous_spec
from .store import Store, default_settings, new_id, now

WEB_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "dist")
SOUND_FS = 48000
GENERATED = {"clicks": "Clicks", "speechish": "Speech-like (synthetic)",
             "sweep": "Sine sweep", "noise_bursts": "Noise bursts"}

store = Store()
library = Library(os.path.join(store.root, "materials_custom.json"))
# the real server (python -m webapp.server) resumes interrupted jobs; an import does not
jobs = JobManager(store, resume=os.environ.get("AURALIS_RESUME_JOBS") == "1")
app = FastAPI(title="Auralis", version="0.1")


def _install_examples():
    """First launch: build the Examples project from local data and queue its runs."""
    try:
        examples.install(store, library, jobs, run=True)
    except Exception as exc:
        print("[Auralis] WARNING: example project not installed: %s" % exc, flush=True)


threading.Thread(target=_install_examples, daemon=True).start()


@app.exception_handler(KeyError)
async def _not_found(_request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": "not found: %s" % exc.args[0]})


@app.exception_handler(ValueError)
async def _bad_request(_request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# -- health / hierarchy ---------------------------------------------------------
@app.get("/api/health")
def health():
    return jobs.health()


@app.get("/api/projects")
def projects():
    return store.tree()


@app.get("/api/recent-simulations")
def recent():
    return store.recent()


@app.post("/api/projects")
def create_project(body: dict):
    return store.create_project(body.get("name"), body.get("description", ""))


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, body: dict):
    return store.patch("projects", pid, body)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    store.delete_project(pid)
    return {"ok": True}


@app.post("/api/projects/{pid}/spaces")
def create_space(pid: str, body: dict):
    return store.create_space(pid, body.get("name"), body.get("description", ""))


@app.patch("/api/spaces/{sid}")
def patch_space(sid: str, body: dict):
    return store.patch("spaces", sid, body)


@app.delete("/api/spaces/{sid}")
def delete_space(sid: str):
    store.delete_space(sid)
    return {"ok": True}


# -- geometries -----------------------------------------------------------------
IMPORT_EXTS = (".obj", ".par", ".dxf", ".3dm", ".ifc", ".skp")


@app.post("/api/import/ifc-inventory")
async def ifc_inventory(file: UploadFile = File(...)):
    """Classes with geometry and spaces of an IFC file, to choose before the import."""
    from . import ifcfile
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        return dict(ifcfile.inventory(tmp), default_exclude=list(ifcfile.DEFAULT_EXCLUDE))
    finally:
        os.remove(tmp)


@app.post("/api/spaces/{sid}/geometries")
async def import_geometry(sid: str, file: UploadFile = File(...), name: str = Form("Geometry"),
                          up_axis: str = Form("z"), unit: str = Form("m"),
                          ifc_space: str = Form(""), ifc_exclude: str = Form(None)):
    store.get_space(sid)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in IMPORT_EXTS:
        raise ValueError("supported formats: .obj, Odeon .Par, .dxf, Rhino .3dm, .ifc and "
                         "SketchUp .skp")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    skp_notes, path = [], tmp
    try:
        if ext == ".skp":            # SketchUp: metres and Z up, whatever the form says
            from . import skpfile
            path = tmp[:-4] + ".obj"
            report = skpfile.to_obj(tmp, path)
            unit, up_axis = "m", "z"
            skp_notes = ["SketchUp model converted by OpenSKP (metres, Z up, one layer per "
                         "material)"] + [ln.split(" -> ")[0].strip() for ln in report
                                         if ln.startswith(("bounding", "WARNING"))]
        v, f, fl, names, notes = meshcheck.import_model(
            path, unit=unit, up_axis=up_axis, ifc_space=ifc_space or None,
            ifc_exclude=None if ifc_exclude is None else [c for c in ifc_exclude.split(",") if c])
        notes = skp_notes + list(notes)
    finally:
        os.remove(tmp)
        if path != tmp and os.path.exists(path):
            os.remove(path)
    closure = {}
    summary, closed = meshcheck.analyse(v, f, fl, names, closure_out=closure)
    for layer in summary["layers"]:
        layer["kind"] = "room"
    meta = dict(summary, name=name or "Geometry", unit_in=unit, up_axis_in=up_axis, objects=[],
                import_notes=notes)
    return store.create_geometry(sid, meta, v, f, fl, names, closed, data, file.filename,
                                 closure=closure or None)


@app.get("/api/geometries/{gid}")
def get_geometry(gid: str):
    g = store.get_geometry(gid)
    if g.get("visual") and "lighting" not in g["visual"]:
        g["visual"]["lighting"] = None          # older geometries lack the key
    return g


@app.patch("/api/geometries/{gid}")
def patch_geometry(gid: str, body: dict):
    return store.patch("geometries", gid, body)


@app.delete("/api/geometries/{gid}")
def delete_geometry(gid: str):
    store.delete_geometry(gid)
    return {"ok": True}


@app.get("/api/geometries/{gid}/mesh")
def get_mesh(gid: str):
    v, f, fl, _closed, names = store.load_mesh(gid)
    owners = store.load_owners(gid)
    layers = []
    for i, n in enumerate(names):
        sel = np.flatnonzero(fl == i)
        layer = {"name": n, "kind": "object" if n.startswith(objlib.LAYER_PREFIX) else "room",
                 "indices": f[sel].reshape(-1).tolist()}
        if layer["kind"] == "object":
            layer["owners"] = [owners[k] for k in sel]
        layers.append(layer)
    return {"positions": np.round(v, 6).reshape(-1).tolist(), "layers": layers}


@app.get("/api/geometries/{gid}/subsurfaces")
def get_subsurfaces(gid: str):
    """Sub-surfaces of each layer, for display: materials stay assigned per layer."""
    return sublib.for_geometry(store, gid)


SOURCE_TYPES = {".skp": "application/vnd.sketchup.skp",
                ".obj": "model/obj", ".par": "text/plain", ".dxf": "image/vnd.dxf",
                ".3dm": "application/octet-stream", ".ifc": "application/x-step"}


@app.get("/api/geometries/{gid}/source")
def get_source_model(gid: str):
    """The model file as imported (kept as source.obj whatever its format), under its name."""
    g = store.get_geometry(gid)
    p = os.path.join(store.geometry_dir(gid), "source.obj")
    if not os.path.exists(p):
        raise HTTPException(404, "no source model is stored for geometry '%s'" % g["name"])
    name = os.path.basename(str(g.get("source_filename") or "").replace("\\", "/"))
    if not name:
        name = exportlib._safe(g["name"]) + "_source"
    return FileResponse(p, filename=name, media_type=SOURCE_TYPES.get(
        os.path.splitext(name)[1].lower(), "application/octet-stream"))


VISUAL_MAX_BYTES = 200 * 1024 * 1024


@app.post("/api/geometries/{gid}/visual")
async def upload_visual(gid: str, file: UploadFile = File(...)):
    """A display-only .glb model over the geometry (never seen by the solvers)."""
    g = store.get_geometry(gid)
    if not (file.filename or "").lower().endswith(".glb"):
        raise ValueError("a binary glTF (.glb) is expected")
    data = await file.read()
    if len(data) > VISUAL_MAX_BYTES or data[:4] != b"glTF":
        raise ValueError("not a .glb file (or larger than 200 MB)")
    with store.lock:
        g = store.get_geometry(gid)
        with open(os.path.join(store.geometry_dir(gid), "visual.glb"), "wb") as fh:
            fh.write(data)
        # a new file needs a new alignment; its lighting is a choice of the user: kept
        g["visual"] = {"filename": file.filename, "bytes": len(data), "uploaded": now(),
                       "transform": {"position": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0],
                                     "scale": 1.0, "up_axis": "y"},
                       "lighting": (g.get("visual") or {}).get("lighting")}
        return store.save_geometry(g)


@app.get("/api/geometries/{gid}/visual.glb")
def get_visual(gid: str):
    p = os.path.join(store.geometry_dir(gid), "visual.glb")
    if not os.path.exists(p):
        raise KeyError("no visual model")
    return FileResponse(p, media_type="model/gltf-binary")


LIGHTING_PRESETS = ("default", "daylight", "evening", "overcast", "studio", "custom")
LIGHTING_RANGES = {"env_intensity": (0.0, 3.0), "sun_intensity": (0.0, 8.0),
                   "sun_azimuth_deg": (0.0, 360.0), "sun_elevation_deg": (-10.0, 90.0),
                   "exposure_ev": (-3.0, 3.0)}


def _clean_lighting(x):
    """Display lighting of the visual model (never seen by the solvers), or None."""
    if x is None:
        return None
    if not isinstance(x, dict):
        raise ValueError("lighting must be an object or null")
    out = {"preset": str(x.get("preset", "custom"))}
    if out["preset"] not in LIGHTING_PRESETS:
        raise ValueError("lighting.preset must be one of %s" % ", ".join(LIGHTING_PRESETS))
    for k, (lo, hi) in LIGHTING_RANGES.items():
        if k not in x:
            raise ValueError("lighting.%s is missing" % k)
        v = float(x[k])
        if not (math.isfinite(v) and lo <= v <= hi):
            raise ValueError("lighting.%s must be within %g - %g" % (k, lo, hi))
        out[k] = v
    col = str(x.get("sun_color", ""))
    if len(col) != 7 or col[0] != "#" or any(ch not in "0123456789abcdefABCDEF" for ch in col[1:]):
        raise ValueError("lighting.sun_color must be #rrggbb")
    out["sun_color"] = col.lower()
    out["view_lights"] = bool(x.get("view_lights", True))
    return out


@app.put("/api/geometries/{gid}/visual")
def set_visual_transform(gid: str, body: dict):
    """Alignment of the visual model: position (m), rotation (deg, XYZ), uniform scale,
    up axis of the file (glTF is Y-up); and its display `lighting` (object, or null)."""
    with store.lock:                     # read-modify-write of geometry.json
        return _set_visual(gid, body)


def _set_visual(gid, body):
    g = store.get_geometry(gid)
    if not g.get("visual"):
        raise KeyError("no visual model")
    if "lighting" in body:
        g["visual"]["lighting"] = _clean_lighting(body["lighting"])
    t = dict(g["visual"]["transform"])
    if "position" in body:
        t["position"] = _point(body["position"], "position")
    if "rotation_deg" in body:
        t["rotation_deg"] = _point(body["rotation_deg"], "rotation")
    if "scale" in body:
        sc = float(body["scale"])
        if not 1e-4 <= sc <= 1e4:
            raise ValueError("scale must be within 1e-4 - 1e4")
        t["scale"] = sc
    if "up_axis" in body:
        if body["up_axis"] not in ("y", "z"):
            raise ValueError("up_axis must be y or z")
        t["up_axis"] = body["up_axis"]
    g["visual"]["transform"] = t
    return store.save_geometry(g)


@app.delete("/api/geometries/{gid}/visual")
def delete_visual(gid: str):
    with store.lock:
        g = store.get_geometry(gid)
        p = os.path.join(store.geometry_dir(gid), "visual.glb")
        if os.path.exists(p):
            os.remove(p)
        g.pop("visual", None)
        return store.save_geometry(g)


@app.post("/api/geometries/{gid}/duplicate")
def duplicate_geometry(gid: str, body: dict):
    return store.duplicate_geometry(gid, body.get("name"))


FINDING_ID = re.compile(r"^[a-z]{2}-[0-9a-f]{16}$")


def _check_args(g):
    """Transition and exclusions of a geometry's last check, kept when it is re-checked."""
    d = (g.get("check") or {}).get("detail") or {}
    return {"transition_hz": d.get("transition_hz"), "transition_source": d.get("source"),
            "exclusions": g.get("check_exclusions")}


def _check_closure(gid):
    """The room's voxel closure. Older geometries lack its outside air (needed to tell a
    slit that leads out of the room): it is rebuilt from the room then."""
    closure = store.load_closure(gid)
    if closure is not None and closure.get("exterior") is None:
        rv, rf, _fl, _names = store.load_room(gid)
        closure = meshcheck.voxel_closure(rv, rf)
    return closure


def _recheck(gid, transition_hz=None, source=None):
    """Re-run the geometry check on the solver mesh (room + objects) and save it."""
    with store.lock:
        g = store.get_geometry(gid)
        v, f, fl, _closed, names = store.load_mesh(gid)
        summary, _ = meshcheck.analyse(v, f, fl, names, closure=_check_closure(gid),
                                       transition_hz=transition_hz, transition_source=source,
                                       exclusions=g.get("check_exclusions"))
        g["check"] = summary["check"]
        return store.save_geometry(g)


def _sim_transition(m, g):
    """(transition in Hz or None, why) of a simulation's wave band, as its run computes it."""
    st, name = m["settings"], m.get("name") or m["id"]
    if not simlib.uses_wave(st):
        return None, ("simulation '%s' uses the geometrical solver only (no wave band): fixed "
                      "threshold" % name)
    if st.get("mode") == "advanced" and st.get("transition_hz"):
        return float(st["transition_hz"]), "transition fixed in the settings of '%s'" % name
    open_scene = simlib.is_open_scene(st, g)
    if not (g["check"]["watertight"] or open_scene):
        return None, ("simulation '%s' has no wave band on this model (not closed, not declared "
                      "an open scene): fixed threshold" % name)
    try:
        fc = simlib.suggested_transition(m, store.load_mesh(g["id"]), library, open_scene)
    except ValueError as exc:
        return None, "automatic transition of '%s' unknown (%s): fixed threshold" % (name, exc)
    return fc, "automatic transition of '%s' (same rule as its run)" % name


@app.post("/api/geometries/{gid}/check")
def check_geometry(gid: str, body: dict):
    """Re-check the geometry: short edges and small gaps against lambda/4 at `transition_hz`,
    else at the transition of `simulation_id`, else the fixed threshold."""
    g = store.get_geometry(gid)
    fc, why = body.get("transition_hz"), None
    if fc is not None:
        fc = float(fc)
        if not (math.isfinite(fc) and 20.0 <= fc <= 20000.0):
            raise ValueError("transition_hz must be within 20-20000 Hz")
        why = "transition given with the check"
    elif body.get("simulation_id"):
        m = store.get_simulation(str(body["simulation_id"]))
        if m["geometry_id"] != gid:
            raise ValueError("simulation %s does not belong to this geometry" % m["id"])
        fc, why = _sim_transition(m, g)
    return _recheck(gid, fc, why)


@app.put("/api/geometries/{gid}/exclusions")
def put_exclusions(gid: str, body: dict):
    """Findings (ids) and whole layers marked as known: they stop counting, and never change
    what the solvers accept. Replaces the set, then re-checks at the same transition."""
    with store.lock:
        g = store.get_geometry(gid)
        fids, layers = body.get("findings") or [], body.get("layers") or []
        if not (isinstance(fids, list) and all(isinstance(x, str) and FINDING_ID.match(x)
                                               for x in fids)):
            raise ValueError("findings: a list of finding ids (as listed in check.findings)")
        if len(set(fids)) > meshcheck.MAX_EXCLUSIONS:
            raise ValueError("at most %d excluded findings" % meshcheck.MAX_EXCLUSIONS)
        known = {x["name"] for x in g["layers"]} | set(
            (g.get("check_exclusions") or {}).get("layers") or [])
        if not (isinstance(layers, list) and all(isinstance(x, str) and x in known for x in layers)):
            raise ValueError("layers: names of layers of this geometry")
        g["check_exclusions"] = {"findings": sorted(set(fids)), "layers": sorted(set(layers))}
        store.save_geometry(g)
        a = _check_args(g)
        return _recheck(gid, a["transition_hz"], a["transition_source"])


def _object_item(o, catalog_ids):
    x = dict(o)
    x["id"] = str(x.get("id") or new_id("O"))
    if x.get("catalog_id") not in catalog_ids:
        raise ValueError("unknown object type '%s'" % x.get("catalog_id"))
    x["name"] = str(x.get("name") or x["catalog_id"])
    x["position"] = _point(x.get("position"), x["name"])
    x["rotation_deg"] = float(x.get("rotation_deg", 0.0))
    s = x.get("scale") or [1.0, 1.0, 1.0]
    if not (isinstance(s, list) and len(s) == 3 and all(
            isinstance(k, (int, float)) and math.isfinite(k) and k > 0 for k in s)):
        raise ValueError("%s: scale must be 3 positive numbers" % x["name"])
    x["scale"] = [float(k) for k in s]
    return x


@app.put("/api/geometries/{gid}/objects")
def put_objects(gid: str, body: dict):
    """Place objects in a geometry: re-assembles the solver mesh, re-checks it and
    syncs every simulation's layers (new object layers get the catalog defaults)."""
    with store.lock:
        g = store.get_geometry(gid)
        sims = store.simulations_of(gid)
        if any(m["status"] in ("queued", "running") for m in sims):
            raise HTTPException(409, "a simulation of this geometry is running")
        catalog = objlib.load_catalog()
        ids = {e["id"] for e in catalog["objects"]}
        items = [_object_item(o, ids) for o in (body.get("objects") or [])]
        if len({o["id"] for o in items}) != len(items):
            raise ValueError("objects: duplicate ids")
        room = store.load_room(gid)
        if items:
            v, f, fl, names, owners = objlib.assemble(room, items, catalog)
        else:
            (v, f, fl, names), owners = room, None
        summary, closed = meshcheck.analyse(v, f, fl, names, closure=_check_closure(gid),
                                            **_check_args(g))
        store.save_assembled(gid, v, f, fl, closed, names, owners)
        for layer in summary["layers"]:
            layer["kind"] = "object" if layer["name"].startswith(objlib.LAYER_PREFIX) else "room"
        g.update(summary)
        g["objects"] = items
        store.save_geometry(g)
        for m in sims:
            layers = {}
            for n in names:
                if n in m["layers"]:
                    layers[n] = m["layers"][n]
                elif n.startswith(objlib.LAYER_PREFIX):
                    layers[n] = objlib.default_layer_settings(catalog, n)
                else:
                    layers[n] = {"material_id": None, "scattering": None}
            m["layers"] = layers
            if os.path.isdir(store.results_dir(m["id"])):
                m["results_stale"] = True
                if m["status"] == "done":
                    m["status"] = "draft"
            store.save_simulation(m)
        return g


@app.post("/api/objects/proxy")
def object_proxy(body: dict):
    """Acoustic proxies of objects at a placement, nothing saved: what a gizmo drag
    shows before it is released (the saved mesh comes from PUT .../objects)."""
    catalog = objlib.load_catalog()
    by_id = {e["id"]: e for e in catalog["objects"]}
    out = []
    for o in body.get("objects") or []:
        x = _object_item(o, set(by_id))
        tri = objlib.instance_triangles(x, by_id[x["catalog_id"]])
        out.append({"id": x["id"], "layers": {k: np.round(v, 4).tolist() for k, v in tri.items()}})
    return {"objects": out}


@app.get("/api/objects/catalog")
def objects_catalog():
    return objlib.load_catalog()


@app.get("/api/objects/visual/{cid}/{path:path}")
def object_file(cid: str, path: str):
    base = os.path.realpath(os.path.join(objlib.VISUAL_DIR, cid))
    f = os.path.realpath(os.path.join(base, path))
    if not f.startswith(base + os.sep) or not os.path.isfile(f):
        raise KeyError("%s/%s" % (cid, path))
    return FileResponse(f)


# -- surface textures of the realistic view (display only) --------------------------
TEXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures")


@app.get("/api/textures")
def textures_index():
    """Downloaded texture sets and HDRIs (fetch_textures.py); entries whose files are missing
    are left out, and `available` is false when nothing was downloaded."""
    p = os.path.join(TEXTURE_DIR, "index.json")
    idx = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    has = lambda rel: os.path.isfile(os.path.join(TEXTURE_DIR, rel))
    tex = {k: dict(v, maps={m: "/api/textures/" + rel for m, rel in v.get("maps", {}).items() if has(rel)})
           for k, v in (idx.get("textures") or {}).items() if has(v.get("maps", {}).get("diff", ""))}
    hdri = {k: dict(v, url_file="/api/textures/" + v["file"])
            for k, v in (idx.get("hdri") or {}).items() if has(v.get("file", ""))}
    return {"available": bool(tex), "textures": tex, "hdri": hdri}


@app.get("/api/textures/{path:path}")
def texture_file(path: str):
    base = os.path.realpath(TEXTURE_DIR)
    f = os.path.realpath(os.path.join(base, path))
    if not f.startswith(base + os.sep) or not os.path.isfile(f) or f.endswith(".json"):
        raise KeyError("texture %s" % path)
    return FileResponse(f, headers={"Cache-Control": "public, max-age=86400"})


# -- materials --------------------------------------------------------------------
@app.get("/api/materials")
def materials():
    return library.all()


@app.post("/api/materials")
def create_material(body: dict):
    return library.add_custom(body.get("name"), body.get("category"), body.get("description"),
                              body.get("absorption"), body.get("scattering", 0.15),
                              a63=body.get("absorption_63"))


def _porous_sigma(body: dict):
    """Flow resistivity of the porous builder: `sigma`, or a glass / rock wool `wool` of bulk
    density `density_kg_m3` (poromat.wool_flow_resistivity; its band and law come back)."""
    if body.get("sigma") is not None:
        return float(body["sigma"]), None
    if body.get("density_kg_m3") is None or not body.get("wool"):
        raise HTTPException(400, "give the flow resistivity sigma, or a wool type and its density")
    from auralis.engine.poromat import wool_flow_resistivity
    try:
        w = wool_flow_resistivity(body["density_kg_m3"], body["wool"])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return w["sigma"], dict(w, wool=str(body["wool"]).lower(),
                            density_kg_m3=float(body["density_kg_m3"]))


@app.post("/api/materials/wool-sigma")
def wool_sigma(body: dict):
    """Flow resistivity of a glass or rock wool from its bulk density, with its band."""
    return _porous_sigma({"wool": body.get("wool"), "density_kg_m3": body.get("density_kg_m3")})[1]


@app.post("/api/materials/porous-preview")
def porous_preview(body: dict):
    sigma, wool = _porous_sigma(body)
    out = {"absorption": [round(x, 4) for x in porous_alpha(
        body.get("model", "miki"), sigma, body["thickness_mm"], body.get("air_gap_mm", 0))],
        "sigma": round(sigma, 1)}
    if wool is not None:
        out["wool"] = wool
        out["absorption_band"] = [[round(x, 4) for x in porous_alpha(
            body.get("model", "miki"), sg, body["thickness_mm"], body.get("air_gap_mm", 0))]
            for sg in (wool["sigma_low"], wool["sigma_high"])]
    return out


@app.post("/api/materials/porous")
def create_porous(body: dict):
    model = body.get("model", "miki")
    sigma, wool = _porous_sigma(body)
    porous_spec(model, sigma, body["thickness_mm"], body.get("air_gap_mm", 0))
    alpha = porous_alpha(model, sigma, body["thickness_mm"], body.get("air_gap_mm", 0))
    phys = {"type": "porous", "model": model, "sigma": float(sigma),
            "thickness_mm": float(body["thickness_mm"]),
            "air_gap_mm": float(body.get("air_gap_mm", 0))}
    if wool is not None:
        phys["sigma_from"] = {k: wool[k] for k in ("wool", "density_kg_m3", "sigma_low",
                                                     "sigma_high", "extrapolated", "law", "source")}
    return library.add_custom(body.get("name"), body.get("category", "Porous absorbers"),
                              body.get("description"), alpha, body.get("scattering", 0.15),
                              physical=phys)


def _perforated_params(body: dict) -> dict:
    return {"kind": body.get("kind", "perforated"), "size_mm": body["size_mm"], "t_mm": body["t_mm"],
            "open_pct": body["open_pct"], "plenum_mm": body["plenum_mm"],
            "wool_mm": body.get("wool_mm", 0.0), "wool_sigma": body.get("wool_sigma", 10000.0),
            "fleece_rayl": body.get("fleece_rayl", 0.0)}


@app.post("/api/materials/perforated-preview")
def perforated_preview(body: dict):
    """Random-incidence alpha of a perforated / slotted / micro-perforated panel stack."""
    from .library import perforated_alpha
    return {"absorption": [round(x, 4) for x in perforated_alpha(**_perforated_params(body))]}


@app.post("/api/materials/perforated")
def create_perforated(body: dict):
    from .library import perforated_alpha
    params = _perforated_params(body)
    alpha = perforated_alpha(**params)
    phys = dict(type="perforated", **{k: (v if k == "kind" else float(v)) for k, v in params.items()})
    return library.add_custom(body.get("name"), body.get("category", "Perforated panels"),
                              body.get("description"), alpha, body.get("scattering", 0.15),
                              physical=phys)


@app.post("/api/materials/table-preview")
def preview_table_material(body: dict):
    """Random-incidence alpha per octave of a Z / R table, without creating anything."""
    from .library import gamma_from_input, table_alpha
    f, g = gamma_from_input(body.get("freqs"), body.get("re"), body.get("im"), body.get("kind"))
    return {"absorption": [round(float(a), 4) for a in table_alpha(f, g)]}


@app.post("/api/materials/table")
def create_table_material(body: dict):
    """Material from a complex normal-incidence quantity per frequency (e.g. 1/3 octaves
    50 Hz-10 kHz): Z / rho c (kind "z") or R (kind "r"). Rays get the random-incidence
    alphas; the wave band and the complex reflections use the table itself."""
    from .library import gamma_from_input, table_alpha
    f, g = gamma_from_input(body.get("freqs"), body.get("re"), body.get("im"), body.get("kind"))
    alpha = table_alpha(f, g)
    phys = {"type": "table", "input": body.get("kind"),
            "gamma_table": {"f": f.round(3).tolist(), "re": g.real.round(6).tolist(),
                            "im": g.imag.round(6).tolist()}}
    return library.add_custom(body.get("name"), body.get("category", "Measured (complex)"),
                              body.get("description"), alpha, body.get("scattering", 0.15),
                              physical=phys)


@app.get("/api/materials/{mid:path}/details")
def material_details(mid: str):
    """Random-incidence alpha (table vs the engine's fit) and the normal-incidence complex
    reflection factor the engine uses, 50 Hz-10 kHz."""
    from auralis.engine import poromat
    from auralis.engine.reflection import _gamma_of, alpha_random, fit_admittance_random, \
        fit_admittance_complex
    from auralis.engine.bands import CENTERS
    m = library.get(mid)
    if m is None:
        raise KeyError(mid)
    f = np.geomspace(50.0, 10000.0, 64)
    phys = m.get("physical") or {}
    fg = np.geomspace(40.0, 1.2 * CENTERS[-1], 60)
    wave_model = None
    if phys.get("type") == "porous":
        # as in the wave band (sim.build_manifest: anchor_alpha): the model's alphas, its phase
        spec = porous_spec(phys["model"], phys["sigma"], phys["thickness_mm"], phys["air_gap_mm"])
        gi, A, l = fit_admittance_random(np.asarray(m["absorption"], float), CENTERS,
                                         gamma_target=lambda ff: poromat.gamma_normal(spec, ff))
        source = "porous model (%s): phase known" % phys["model"]
        sf = phys.get("sigma_from")
        if sf:
            source += ("; flow resistivity %.0f Pa.s/m2 from the density of a %s wool (%.0f kg/m3, "
                       "1-sigma %.0f-%.0f%s): %s" % (
                           phys["sigma"], sf["wool"], sf["density_kg_m3"], sf["sigma_low"],
                           sf["sigma_high"], ", EXTRAPOLATED beyond the data the law was checked on"
                           if sf["extrapolated"] else "", sf["law"]))
    elif phys.get("type") == "perforated":
        from .library import perforated_spec
        spec = perforated_spec(**{k: phys[k] for k in simlib.PERFORATED_KEYS if k in phys})
        ab = np.asarray(m["absorption"], float)
        gi, A, l = fit_admittance_random(ab, CENTERS,
                                         gamma_target=lambda ff: poromat.gamma_normal(spec, ff))
        source = ("%s panel model: phase known from the geometry (not measured), absorption "
                  "from the model" % phys.get("kind", "perforated"))
    elif phys.get("type") == "table":
        spec = {"gamma_table": phys["gamma_table"]}
        gi, A, l = fit_admittance_complex(fg, poromat._gamma_table(spec, fg))
        source = "measured complex %s: phase known" % ("impedance" if phys.get("input") == "z"
                                                       else "reflection factor")
    else:
        # alpha-only material: what the engine does with it (BakeConfig.infer_materials)
        from auralis.engine import matinfer
        a63 = m.get("absorption_63")
        fb = np.r_[simlib.BAND63_HZ, CENTERS] if a63 is not None else CENTERS
        ab = np.r_[a63, m["absorption"]] if a63 is not None else np.asarray(m["absorption"], float)
        inf = matinfer.infer(ab, fb)
        wave_model = matinfer.describe(inf)
        panel = inf["family"] == "panel"
        if inf["family"] is None:
            gi, A, l = fit_admittance_random(ab, fb)
            source = ("fitted on the random-incidence alphas only: the PHASE is a model "
                      "assumption (an alpha-only fit is blind to it). " + wave_model)
        elif panel:
            gi, A, l = fit_admittance_complex(fg, poromat.gamma_normal(inf["spec"], fg))
            source = ("inferred physical model, phase from the model (not measured): "
                      + wave_model)
        else:
            gi, A, l = fit_admittance_random(
                ab, fb, gamma_target=lambda ff: poromat.gamma_normal(inf["spec"], ff))
            source = ("inferred physical model, phase from the model (not measured), "
                      "absorption from the table: " + wave_model)
    g = _gamma_of(f, gi, A, l)
    R = (1.0 - g) / (1.0 + g)
    return {"id": mid, "bands_hz": [int(c) for c in CENTERS],
            "alpha_table": m["absorption"],
            "alpha_table_63": m.get("absorption_63"),
            # before the per-band anchoring the engine applies to match alpha_table
            "alpha_fit_random": np.round(np.clip(alpha_random(_gamma_of(CENTERS, gi, A, l)),
                                                 0, 1), 4).tolist(),
            # the 63 Hz octave: wave band only, so no anchoring (the fit IS what it sees)
            "alpha_fit_63": round(float(np.clip(alpha_random(
                _gamma_of(np.array([simlib.BAND63_HZ]), gi, A, l))[0], 0, 1)), 4),
            "freqs_hz": np.round(f, 2).tolist(),
            "R_normal": {"re": np.round(R.real, 5).tolist(), "im": np.round(R.imag, 5).tolist(),
                         "mag": np.round(np.abs(R), 5).tolist(),
                         "phase_deg": np.round(np.degrees(np.angle(R)), 2).tolist()},
            "alpha_normal": np.round(1.0 - np.abs(R) ** 2, 5).tolist(),
            "phase_source": source,
            "wave_model": wave_model}


@app.delete("/api/materials/{mid:path}")
def delete_material(mid: str):
    library.delete_custom(mid)
    return {"ok": True}


@app.patch("/api/materials/{mid:path}")
def rename_material(mid: str, body: dict):
    """Rename one of your materials. Simulations refer to its id, which never changes."""
    return library.rename_custom(mid, body.get("name"))


@app.post("/api/materials/{mid:path}/duplicate")
def duplicate_material(mid: str, body: dict | None = None):
    """Copy any material into your own ones (new id), keeping its licence."""
    return library.duplicate(mid, (body or {}).get("name"))


@app.get("/api/directivities")
def directivities():
    """Presets and imported directivities, with DI per band and a polar subsample."""
    return dirlib.listing(store.root)


@app.post("/api/directivities")
async def import_directivity(file: UploadFile = File(...), name: str = Form("")):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".json", ".csv", ".sofa"):
        raise ValueError("supported: .json (engine table), .csv, .sofa (AES69); the CLF .CF2 "
                         "binary has no public specification")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        rec = dirlib.import_file(store.root, tmp, name, file.filename)
    finally:
        os.remove(tmp)
    return dict({k: v for k, v in rec.items() if k != "table"}, kind="imported",
                **dirlib.summary(rec["table"]))


@app.delete("/api/directivities/{did}")
def delete_directivity(did: str):
    used = [m["name"] for m in store._list("simulations")
            if any(s.get("directivity") == did for s in m.get("sources", []))]
    if used:
        raise HTTPException(409, "used by: %s" % ", ".join(used[:5]))
    dirlib.delete(store.root, did)
    return {"ok": True}


# -- simulations ------------------------------------------------------------------
def _point(p, what):
    if not (isinstance(p, (list, tuple)) and len(p) == 3
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in p)):
        raise ValueError("%s: position must be 3 finite numbers" % what)
    return [float(x) for x in p]


@app.post("/api/geometries/{gid}/simulations")
def create_simulation(gid: str, body: dict):
    m = store.create_simulation(gid, body.get("name"), body.get("copy_from"))
    catalog = None
    for n, v in m["layers"].items():
        if n.startswith(objlib.LAYER_PREFIX) and not v.get("material_id"):
            catalog = catalog or objlib.load_catalog()
            m["layers"][n] = objlib.default_layer_settings(catalog, n)
    return store.save_simulation(m) if catalog else m


@app.get("/api/simulations/{mid}")
def get_simulation(mid: str):
    return store.get_simulation(mid)


@app.put("/api/simulations/{mid}")
def put_simulation(mid: str, body: dict):
    with store.lock:
        m = store.get_simulation(mid)
        if m["status"] in ("queued", "running"):
            raise HTTPException(409, "the simulation is running: wait or cancel it first")
        g = store.get_geometry(m["geometry_id"])
        changed = False
        if "layers" in body:
            layers = {}
            catalog = None
            for l in g["layers"]:
                v = body["layers"].get(l["name"])
                if v is None:           # stale client: keep what is stored, never wipe it
                    v = m["layers"].get(l["name"])
                    if v is None and l["name"].startswith(objlib.LAYER_PREFIX):
                        catalog = catalog or objlib.load_catalog()
                        v = objlib.default_layer_settings(catalog, l["name"])
                    v = v or {}
                sc = v.get("scattering")
                if isinstance(sc, (list, tuple)):       # one value per octave
                    if len(sc) != 7:
                        raise ValueError("layer '%s': scattering per octave needs 7 values"
                                         % l["name"])
                    sc = [float(x) for x in sc]
                    if not all(0.0 <= x <= 1.0 for x in sc):
                        raise ValueError("layer '%s': scattering must be within [0, 1]"
                                         % l["name"])
                elif sc is not None:
                    sc = float(sc)
                    if not 0.0 <= sc <= 1.0:
                        raise ValueError("layer '%s': scattering must be within [0, 1]"
                                         % l["name"])
                layers[l["name"]] = {"material_id": v.get("material_id"), "scattering": sc}
            changed |= layers != m["layers"]
            m["layers"] = layers
        for key in ("sources", "receivers"):
            if key in body:
                items = []
                for it in body[key]:
                    x = dict(it)
                    x["id"] = str(x.get("id") or new_id(key[0].upper()))
                    x["name"] = str(x.get("name") or x["id"])
                    x["path"] = trajlib.clean_path(x.get("path"), x["name"])
                    if x["path"]:         # a moving point's static position is its start
                        x["position"] = list(x["path"]["waypoints"][0]["position"])
                    x["position"] = _point(x.get("position"), x["name"])
                    x["azimuth_deg"] = float(x.get("azimuth_deg", 0))
                    x["elevation_deg"] = float(x.get("elevation_deg", 0))
                    if key == "sources":
                        x["directivity"] = x.get("directivity") or "omni"
                        if not dirlib.known(store.root, x["directivity"]):
                            raise ValueError("unknown directivity '%s'" % x["directivity"])
                        x["gain_db"] = float(x.get("gain_db", 0))
                        sl = x.get("surface_layer")
                        if sl in (None, ""):
                            x.pop("surface_layer", None)
                        elif sl not in [l["name"] for l in g["layers"]]:
                            raise ValueError("source '%s': unknown layer '%s'" % (x["name"], sl))
                        eq = sourceeq.clean_eq(x.get("eq_db"))
                        if eq is None:
                            x.pop("eq_db", None)
                        else:
                            x["eq_db"] = eq
                    items.append(x)
                if len({x["id"] for x in items}) != len(items):
                    raise ValueError("%s: duplicate ids" % key)
                # a trajectory only feeds animated renders: editing it keeps the static
                # results (its start point is the static position, compared below)
                # reordering alone (same items, another order) keeps the results too
                changed |= (sorted(_without_path(items), key=lambda x: x["id"])
                            != sorted(_without_path(m[key]), key=lambda x: x["id"]))
                m[key] = items
        if "receiver_grids" in body:
            names = [l["name"] for l in g["layers"]]
            grids = [gridlib.clean_grid(x, names) for x in body["receiver_grids"] or []]
            if len({x["id"] for x in grids}) != len(grids):
                raise ValueError("receiver grids: duplicate ids")
            changed |= grids != m.get("receiver_grids", [])
            m["receiver_grids"] = grids
        if "settings" in body:
            st = default_settings()
            st.update({k: v for k, v in (body["settings"] or {}).items() if k in st})
            simlib.engine_settings(st)          # raises on a bad mode
            if st["speech_effort"] not in speechlib.VOCAL_EFFORT_DBA_1M:
                raise ValueError("speech_effort must be one of %s"
                                 % ", ".join(speechlib.VOCAL_EFFORT_DBA_1M))
            speechlib.noise_levels(st["background_noise"])   # raises on a bad curve
            st["background_recording"] = _clean_background_recording(st["background_recording"])
            if st["stop_decay_db"] is not None:
                st["stop_decay_db"] = float(st["stop_decay_db"])
                if not 20.0 <= st["stop_decay_db"] <= 90.0:
                    raise ValueError("stop_decay_db must be within 20-90 dB")
            st["grid_rays"] = int(st["grid_rays"])
            if not 1000 <= st["grid_rays"] <= 2_000_000:
                raise ValueError("grid_rays must be within 1000-2000000")
            st["rbs_scattering"] = bool(st["rbs_scattering"])
            changed |= st != dict(default_settings(), **m["settings"])   # new keys: not a change
            m["settings"] = st
        for k in ("name", "description"):
            if body.get(k) is not None:
                m[k] = str(body[k])
        if changed and m["status"] == "done":
            m["status"], m["results_stale"] = "draft", True
        elif changed and os.path.isdir(store.results_dir(mid)):
            m["results_stale"] = True
        return store.save_simulation(m)


def _without_path(items):
    return [{k: v for k, v in x.items() if k != "path"} for x in items]


@app.delete("/api/simulations/{mid}")
def delete_simulation(mid: str):
    store.delete_simulation(mid)
    return {"ok": True}


@app.get("/api/simulations/{mid}/validation")
def validation(mid: str):
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    return simlib.validate(m, g, store.load_mesh(m["geometry_id"]), library,
                           store.load_closure(m["geometry_id"]))


@app.get("/api/simulations/{mid}/estimate")
def estimate(mid: str):
    """Rough run time before Run, from the unit costs measured on this machine."""
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    check = simlib.validate(m, g, store.load_mesh(m["geometry_id"]), library,
                            store.load_closure(m["geometry_id"]))
    if check["unassigned_layers"]:
        raise HTTPException(409, "assign a material to every layer first")
    return simlib.estimate_runtime(m, g, store.load_mesh(m["geometry_id"]), library,
                                   gpu=bool(jobs.gpu), closure=store.load_closure(m["geometry_id"]),
                                   root=store.root)


@app.post("/api/simulations/{mid}/run")
def run(mid: str):
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    check = simlib.validate(m, g, store.load_mesh(m["geometry_id"]), library,
                            store.load_closure(m["geometry_id"]))
    if not check["ok_to_run"]:
        raise HTTPException(409, "not ready to run: " + "; ".join(check["blocking"]))
    return jobs.submit(m)


@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return jobs.get(jid)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    return jobs.cancel(jid)


# -- results ----------------------------------------------------------------------
def _meta(mid):
    p = os.path.join(store.results_dir(mid), "meta.json")
    if not os.path.exists(p):
        raise KeyError("results of %s" % mid)
    with open(p) as fh:
        return json.load(fh)


def _ir(mid, src, rcv):
    if "/" in src + rcv or src.startswith(".") or rcv.startswith("."):
        raise KeyError(src)
    p = os.path.join(store.results_dir(mid), "irs", "%s__%s.npy" % (src, rcv))
    if not os.path.exists(p):
        raise KeyError("%s -> %s" % (src, rcv))
    return np.load(p), p


@app.get("/api/simulations/{mid}/results")
def results(mid: str):
    m = store.get_simulation(mid)
    meta = _meta(mid)
    meta["stale"] = bool(m.get("results_stale"))
    return meta


@app.get("/api/simulations/{mid}/results/{src}/{rcv}/curves")
def result_curves(mid: str, src: str, rcv: str, fr_smoothing: int = 0, part: str = "hybrid"):
    """part: hybrid (default), geo (geometrical solver alone) or wave (wave band alone)."""
    meta = _meta(mid)
    ir, path = _ir(mid, src, rcv)
    if part not in ("hybrid", "geo", "wave"):
        raise ValueError("part must be hybrid, geo or wave")
    if part != "hybrid":
        pp = path[:-4] + "_omni_%s.npy" % part
        if not os.path.exists(pp):
            raise KeyError("no %s part for %s -> %s (a wave-solver run from v0.6 has it)"
                           % (part, src, rcv))
        omni = np.load(pp)
        return dict(curveslib.curves(np.stack([omni, omni], axis=1), meta["fs"],
                                     meta["ir_scale"], fr_smoothing, omni), part=part)
    ap = path[:-4] + "_ambi.npy"
    omni = np.load(ap, mmap_mode="r")[:, 0] if os.path.exists(ap) else None
    return dict(curveslib.curves(ir, meta["fs"], meta["ir_scale"], fr_smoothing, omni), part=part)


@app.get("/api/simulations/{mid}/results/{src}/{rcv}/ambi.wav")
def result_ambi_wav(mid: str, src: str, rcv: str):
    """Ambisonic IR, order 2, ACN / SN3D (AmbiX), world axes, 9 channels float32."""
    meta = _meta(mid)
    _ir_, path = _ir(mid, src, rcv)
    ap = path[:-4] + "_ambi.npy"
    if not os.path.exists(ap):
        raise KeyError("no ambisonic IR for %s -> %s (run again)" % (src, rcv))
    wav = ap[:-4] + ".wav"
    if not os.path.exists(wav) or os.path.getmtime(wav) < os.path.getmtime(ap):
        write_wav(wav, (np.load(ap) * meta["ir_scale"]).astype(np.float32), meta["fs"],
                  float32=True)
    return FileResponse(wav, media_type="audio/wav",
                        filename="%s_%s_%s_ambix.wav" % (mid, src, rcv))


@app.get("/api/simulations/{mid}/results/{src}/{rcv}/ir.wav")
def result_wav(mid: str, src: str, rcv: str):
    meta = _meta(mid)
    ir, path = _ir(mid, src, rcv)
    wav = path[:-4] + ".wav"
    if not os.path.exists(wav) or os.path.getmtime(wav) < os.path.getmtime(path):
        write_wav(wav, (ir * meta["ir_scale"]).astype(np.float32), meta["fs"], float32=True)
    return FileResponse(wav, media_type="audio/wav",
                        filename="%s_%s_%s.wav" % (mid, src, rcv))


# -- sounds -----------------------------------------------------------------------
def _sounds_dir():
    return os.path.join(store.root, "sounds")


@app.get("/api/sounds")
def sounds():
    out = [{"id": "gen:" + k, "name": v, "kind": "generated", "duration_s": 6.0}
           for k, v in GENERATED.items()]
    d = _sounds_dir()
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                out.append(json.load(fh))
    return out + [{k: e[k] for k in ("id", "name", "kind", "duration_s", "credit", "license", "group")}
                  for e in soundlib.odeon_sounds().values()]


def _sound_path(sid):
    """(path, credit) of a sound id: generated, uploaded or Odeon."""
    d = _sounds_dir()
    if sid.startswith("gen:"):
        kind = sid[4:]
        if kind not in GENERATED:
            raise KeyError(sid)
        p = os.path.join(d, "_gen_%s.wav" % kind)
        if not os.path.exists(p):
            write_wav(p, make_test_signal(SOUND_FS, kind, 6.0).astype(np.float32), SOUND_FS)
        return p, None
    if sid.startswith("odeon-"):
        e = soundlib.odeon_sounds().get(sid)
        if e is None:
            raise KeyError(sid)
        return os.path.join(os.path.dirname(soundlib.ODEON_SOUNDS_INDEX), e["file"]), e["credit"]
    if "/" in sid or sid.startswith("."):
        raise KeyError(sid)
    p = os.path.join(d, sid + ".wav")
    if not os.path.exists(p):
        raise KeyError(sid)
    try:                                    # an uploaded sound may carry its licence credit
        with open(os.path.join(d, sid + ".json")) as fh:
            return p, json.load(fh).get("credit")
    except (OSError, ValueError):
        return p, None


@app.get("/api/sounds/{sid}.wav")
def sound_wav(sid: str):
    return FileResponse(_sound_path(sid)[0], media_type="audio/wav")


@app.post("/api/examples/install")
def install_examples(body: dict):
    """(Re)build the Examples project; `force` makes a new copy, `run` queues the runs."""
    return examples.install(store, library, jobs, run=bool(body.get("run", True)),
                            force=bool(body.get("force", False)))


@app.get("/api/ensembles")
def ensembles():
    return soundlib.ensembles()


@app.post("/api/simulations/{mid}/ensemble")
def add_ensemble(mid: str, body: dict):
    """One source per player, seated on a stage: conductor at `center` (floor),
    audience towards `facing_deg`; the layout shrinks to fit width x depth."""
    with store.lock:
        m = store.get_simulation(mid)
        if m["status"] in ("queued", "running"):
            raise HTTPException(409, "the simulation is running: wait or cancel it first")
        ens = next((e for e in soundlib.ensembles() if e["id"] == body.get("ensemble_id")), None)
        if ens is None:
            raise KeyError(body.get("ensemble_id"))
        center = _point(body.get("center"), "center")
        width, depth = float(body.get("width_m", 12.0)), float(body.get("depth_m", 8.0))
        if not (0.5 <= width <= 100 and 0.5 <= depth <= 100):
            raise ValueError("stage width and depth must be within 0.5-100 m")
        keep = [] if body.get("replace_sources") else m["sources"]
        prefix = "E%d_" % (1 + len({s.get("ensemble") for s in keep if s.get("ensemble")}))
        new, scale = soundlib.ensemble_sources(ens, center, float(body.get("facing_deg", 0.0)),
                                               width, depth, prefix)
        if body.get("preview"):
            # the layout as it would be added, with each player's validity; nothing saved
            g = store.get_geometry(m["geometry_id"])
            mesh = store.load_mesh(m["geometry_id"])
            ins, dist = simlib.point_checks([x["position"] for x in new], g, mesh,
                                            store.load_closure(m["geometry_id"]), m["settings"])
            for x, i, d in zip(new, ins, dist):
                x["valid"] = bool(i and d >= simlib.MIN_SURFACE_DIST)
                x["reason"] = (None if x["valid"] else "outside the air volume" if not i
                               else "%.2f m from a surface" % d)
            return {"preview": True, "sources": new, "scale": round(scale, 3), "n_sources": len(new),
                    "n_invalid": sum(not x["valid"] for x in new)}
        if len(keep) + len(new) > 100:
            raise ValueError("at most 100 sources per simulation (%d requested)" % (len(keep) + len(new)))
        m["sources"] = keep + new
        if os.path.isdir(store.results_dir(mid)):
            m["results_stale"] = True
            if m["status"] == "done":
                m["status"] = "draft"
        store.save_simulation(m)
        return {"simulation": m, "scale": round(scale, 3), "n_sources": len(new)}


def _mix_request(mid, body):
    """Checked inputs of a server-side mix: (strips, irs, ir_fs, format, sample format)."""
    meta = _meta(mid)
    rid = body.get("receiver_id")
    fmt = body.get("format", "binaural")
    if fmt not in ("binaural", "ambisonic"):
        raise ValueError("format must be 'binaural' or 'ambisonic'")
    strips, irs = [], []
    for st in body.get("strips") or []:
        if st.get("muted") or not st.get("sound_id"):
            continue
        path, credit = _sound_path(st["sound_id"])
        ir, ipath = _ir(mid, st["source_id"], rid)
        if fmt == "ambisonic":
            ap = ipath[:-4] + "_ambi.npy"
            if not os.path.exists(ap):
                raise HTTPException(409, "no ambisonic IR for %s: run the simulation again"
                                    % st["source_id"])
            ir = np.load(ap)
        strips.append((path, float(st.get("gain_db", 0.0)), st["source_id"], credit))
        irs.append(ir * meta["ir_scale"])
    if not strips:
        raise ValueError("no source with a sound to render")
    sample = body.get("sample_format") or ("pcm16" if fmt == "ambisonic" else "float32")
    if sample not in ("pcm16", "float32"):
        raise ValueError("sample_format must be 'pcm16' or 'float32'")
    return strips, irs, meta["fs"], fmt, sample


def _mix(mid, body, prep, progress=None):
    strips, irs, ir_fs, fmt, sample = prep
    wav, txt, info = soundlib.render(strips, irs, ir_fs, os.path.join(store.results_dir(mid), "renders"),
                                     float(body.get("master_db", 0.0)),
                                     tag="ambix" if fmt == "ambisonic" else "",
                                     pcm16=sample == "pcm16", progress=progress)
    base = "/api/simulations/%s/renders/" % mid
    return dict(info, receiver_id=body.get("receiver_id"), n_sources=len(strips), format=fmt,
                sample_format=sample, url=base + os.path.basename(wav),
                credits_url=base + os.path.basename(txt))


@app.post("/api/simulations/{mid}/render")
def render(mid: str, body: dict):
    """Server-side binaural mix of every assigned source at one receiver (synchronous)."""
    return _mix(mid, body, _mix_request(mid, body))


_mix_pool = ThreadPoolExecutor(max_workers=1)      # CPU only: never waits behind a GPU run
_mix_live = {}                                      # task id -> cancel requested


def _mix_task(mid, rid, body):
    def progress(frac, stage):
        store.set_task(mid, "mix", rid, progress=round(float(frac), 3), stage=stage)
        return not _mix_live.get(rid)

    try:
        if _mix_live.get(rid):
            raise KeyboardInterrupt("render cancelled")
        store.set_task(mid, "mix", rid, status="running", started=now())
        result = _mix(mid, body, _mix_request(mid, body), progress)
        store.set_task(mid, "mix", rid, status="done", progress=1.0, stage="done", result=result,
                       finished=now())
    except KeyboardInterrupt:
        store.set_task(mid, "mix", rid, status="cancelled", stage="cancelled", finished=now())
    except Exception as exc:                        # reported on the task, never silent
        store.set_task(mid, "mix", rid, status="failed", error="%s: %s" % (type(exc).__name__, exc),
                       finished=now())
    finally:
        _mix_live.pop(rid, None)


@app.post("/api/simulations/{mid}/mixes")
def mix_start(mid: str, body: dict):
    """The same mix as POST .../render, as a background task with progress and cancel."""
    _mix_request(mid, body)                         # bad requests fail here, synchronously
    info = store.create_task(mid, "mix", dict(body), progress=0.0, stage="queued", result=None)
    _mix_live[info["id"]] = False
    _mix_pool.submit(_mix_task, mid, info["id"], dict(body))
    return info


@app.get("/api/simulations/{mid}/mixes")
def mix_list(mid: str):
    """The simulation's mix tasks, newest first (a page reload picks up a running one)."""
    store.get_simulation(mid)
    return [mix_get(mid, x["id"]) for x in store.tasks(mid, "mix")]


@app.get("/api/simulations/{mid}/mixes/{rid}")
def mix_get(mid: str, rid: str):
    info = store.get_task(mid, "mix", rid)
    if info["status"] in ("queued", "running") and rid not in _mix_live:
        info = store.set_task(mid, "mix", rid, status="failed",
                              error="interrupted (the server restarted)")
    return info


@app.post("/api/simulations/{mid}/mixes/{rid}/cancel")
def mix_cancel(mid: str, rid: str):
    info = store.get_task(mid, "mix", rid)
    if rid in _mix_live:
        _mix_live[rid] = True
    return info


@app.get("/api/simulations/{mid}/renders/{name}")
def render_file(mid: str, name: str):
    if "/" in name or name.startswith(".") or not name.startswith("render_"):
        raise KeyError(name)
    p = os.path.join(store.results_dir(mid), "renders", name)
    if not os.path.exists(p):
        raise KeyError(name)
    return FileResponse(p, filename="%s_%s" % (mid, name))


def _clean_background_recording(x):
    """None, or a recorded ambience of the sound library as the diffuse bed of animated renders:
    {"sound_id", "dba"} -> with its file path, name and credit."""
    if not x:
        return None
    if not isinstance(x, dict) or not x.get("sound_id"):
        raise ValueError("background_recording needs a sound_id (and dba)")
    path, credit = _sound_path(str(x["sound_id"]))
    dba = float(x.get("dba", 55.0))
    if not 20.0 <= dba <= 100.0:
        raise ValueError("background_recording: dba must be within 20-100 dB(A)")
    name = next((e.get("name") for e in sounds() if e.get("id") == x["sound_id"]), x["sound_id"])
    return {"sound_id": str(x["sound_id"]), "dba": dba, "path": path, "name": name,
            "credit": credit}


def _motion_strips(body):
    out = []
    for st in body.get("strips") or []:
        if st.get("muted") or not st.get("sound_id"):
            continue
        path, credit = _sound_path(st["sound_id"])
        out.append({"source_id": st["source_id"], "sound_id": st["sound_id"], "path": path,
                    "credit": credit, "gain_db": float(st.get("gain_db", 0.0)),
                    "loop": bool(st.get("loop", False)),
                    "offset_s": max(0.0, float(st.get("offset_s", 0.0) or 0.0))})
    return out


def _turn_taking(spec, strips):
    """None, or {"gap_s"}: the strips' recordings speak in turn (sounds.take_turns)."""
    if not spec:
        return None
    gap = float(spec.get("gap_s", 0.5)) if isinstance(spec, dict) else 0.5
    if not 0.0 <= gap <= 5.0:
        raise ValueError("turn_taking: gap_s must be within 0-5 s")
    if len(strips) < 2:
        raise ValueError("turn_taking needs at least two playing sources")
    return {"gap_s": gap}


def _motion_check(mid, body):
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    strips = _motion_strips(body)
    check = simlib.validate_motion(m, g, store.load_mesh(m["geometry_id"]), library,
                                   store.load_closure(m["geometry_id"]), body.get("receiver_id"),
                                   [s["source_id"] for s in strips], body.get("duration_s"))
    return m, strips, check


@app.get("/api/simulations/{mid}/grids/{grid_id}/points")
def grid_points(mid: str, grid_id: str):
    """Points of a receiver grid with their validity (preview before a run)."""
    m = store.get_simulation(mid)
    grid = next((x for x in m.get("receiver_grids", []) if x["id"] == grid_id), None)
    if grid is None:
        raise KeyError(grid_id)
    gid = m["geometry_id"]
    p, valid, why, area, nrm, uax = gridlib.points(grid, m, store.get_geometry(gid),
                                                   store.load_mesh(gid), store.load_closure(gid),
                                                   axes=True)
    return {"grid_id": grid_id, "points": np.round(p, 4).tolist(), "valid": valid.tolist(),
            "normals": np.round(nrm, 5).tolist(), "u_axes": np.round(uax, 5).tolist(),
            "reasons": {str(k): v for k, v in why.items()}, "area_m2": area.tolist(),
            "n_points": int(len(p)), "n_valid": int(valid.sum())}


@app.get("/api/simulations/{mid}/grids/{grid_id}/results")
def grid_results(mid: str, grid_id: str, source_id: str, param: str = None):
    """Grid parameters of one source (all, or one `param`)."""
    meta = _meta(mid)
    g = next((x for x in meta.get("grids", []) if x["grid_id"] == grid_id), None)
    if g is None:
        raise KeyError(grid_id)
    if "/" in source_id or source_id.startswith("."):
        raise KeyError(source_id)
    path = os.path.join(store.results_dir(mid), "grids", "%s__%s.npz" % (grid_id, source_id))
    if not os.path.exists(path):
        raise KeyError("%s / %s" % (grid_id, source_id))
    z = np.load(path)
    keys = [k for k in z.files if k not in ("points", "valid", "area_m2", "normals", "u_axes")]
    if param is not None:
        if param not in keys:
            raise KeyError(param)
        keys = [param]

    def num(a):
        return [None if not np.isfinite(v) else round(float(v), 5) for v in a.reshape(-1)] \
            if a.ndim == 1 else [num(r) for r in a]

    rp = path[:-4] + "_reasons.json"
    reasons = json.load(open(rp)) if os.path.exists(rp) else {}
    axes = {k: np.round(z[k], 5).tolist() for k in ("normals", "u_axes") if k in z.files}
    return dict({"grid_id": grid_id, "source_id": source_id,
                 "points": np.round(z["points"], 4).tolist(), "valid": z["valid"].tolist(),
                 "area_m2": z["area_m2"].tolist(), "reasons": reasons,
                 "bands_hz": meta["bands_hz"], "params": {k: num(z[k]) for k in keys}}, **axes)


@app.post("/api/simulations/{mid}/combine")
def combine_sources(mid: str, body: dict):
    """Coherent sum of sources at one receiver (gain, delay, polarity per source)."""
    m = store.get_simulation(mid)
    meta = _meta(mid)
    key, info = combinelib.combine(store.results_dir(mid), meta, body.get("sources") or [],
                                   body.get("receiver_id"), m["settings"])
    base = "/api/simulations/%s/combined/%s/" % (mid, key)
    return dict(info, curves_url=base + "curves", wav_url=base + "ir.wav")


def _combined(mid, key):
    if "/" in key or key.startswith("."):
        raise KeyError(key)
    d = os.path.join(store.results_dir(mid), "combined", key)
    if not os.path.isdir(d):
        raise KeyError(key)
    return d


@app.get("/api/simulations/{mid}/combined/{key}/curves")
def combined_curves(mid: str, key: str, fr_smoothing: int = 0):
    d = _combined(mid, key)
    meta = _meta(mid)
    return curveslib.curves(np.load(os.path.join(d, "ir.npy")), meta["fs"], meta["ir_scale"],
                            fr_smoothing, np.load(os.path.join(d, "omni.npy")))


@app.get("/api/simulations/{mid}/combined/{key}/ir.wav")
def combined_wav(mid: str, key: str):
    d = _combined(mid, key)
    meta = _meta(mid)
    wav = os.path.join(d, "ir.wav")
    if not os.path.exists(wav):
        write_wav(wav, (np.load(os.path.join(d, "ir.npy")) * meta["ir_scale"]).astype(np.float32),
                  meta["fs"], float32=True)
    return FileResponse(wav, media_type="audio/wav", filename="%s_combined_%s.wav" % (mid, key))


@app.get("/api/simulations/{mid}/iso3382-3")
def open_plan_office(mid: str, source_id: str, receiver_ids: str):
    """ISO 3382-3 along a path of receivers (comma-separated ids) for one source."""
    from auralis.engine import office
    from auralis.engine import speech as sp
    m = store.get_simulation(mid)
    meta = _meta(mid)
    src = next((s for s in m["sources"] if s["id"] == source_id), None)
    if src is None:
        raise KeyError(source_id)
    rcv = {r["id"]: r for r in m["receivers"]}
    ids = [x for x in receiver_ids.split(",") if x]
    if len(ids) < 2:
        raise ValueError("ISO 3382-3 needs at least 2 receivers along the path")
    dist, g, mtfs = [], [], []
    for rid in ids:
        if rid not in rcv:
            raise KeyError(rid)
        pair = next((p for p in meta["pairs"] if p["source_id"] == source_id
                     and p["receiver_id"] == rid), None)
        if pair is None:
            raise KeyError("%s -> %s" % (source_id, rid))
        ir, path = _ir(mid, source_id, rid)
        ap = path[:-4] + "_ambi.npy"
        omni = np.load(ap)[:, 0] if os.path.exists(ap) else ir.mean(axis=1)
        dist.append(float(np.linalg.norm(np.subtract(rcv[rid]["position"], src["position"]))))
        g.append([np.nan if v is None else v for v in pair["metrics"]["G_db"]])
        mtfs.append(sp.mtf_from_ir(omni, meta["fs"]))
    order = np.argsort(dist)
    out = office.evaluate(np.asarray(dist)[order], np.asarray(g)[order],
                          np.asarray(mtfs)[order],
                          sp.noise_levels(m["settings"].get("background_noise", "none")))
    for k, i in enumerate(order):
        out["positions"][k]["receiver_id"] = ids[i]
    out["directivity_note"] = (None if (src.get("directivity") or "omni") == "omni" else
                               "ISO 3382-3 assumes an omnidirectional source; this one is not")
    return out


@app.post("/api/simulations/{mid}/modal")
def modal_create(mid: str, body: dict):
    """Queue a forced-response (modal) map: one source, one receiver grid, frequencies."""
    from . import modal_runner
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    check = simlib.validate(m, g, store.load_mesh(m["geometry_id"]), library,
                            store.load_closure(m["geometry_id"]))
    if check["unassigned_layers"]:
        raise HTTPException(409, "assign a material to every layer first")
    if not check["wave_available"]:
        raise HTTPException(409, "the wave solver needs a closed model or a declared open scene")
    if not any(s["id"] == body.get("source_id") for s in m["sources"]):
        raise KeyError(body.get("source_id"))
    if not any(x["id"] == body.get("grid_id") for x in m.get("receiver_grids", [])):
        raise KeyError(body.get("grid_id"))
    if body.get("freqs"):
        freqs = sorted({round(float(f), 3) for f in body["freqs"]})
    else:
        f0, f1 = float(body.get("f_min", 20.0)), float(body.get("f_max", 200.0))
        step = float(body.get("step_hz", 1.0))
        if step <= 0 or f1 < f0:
            raise ValueError("need f_min <= f_max and step_hz > 0")
        freqs = np.round(np.arange(f0, f1 + 1e-9, step), 3).tolist()
    if not freqs or len(freqs) > modal_runner.MAX_FREQS:
        raise ValueError("1 to %d frequencies" % modal_runner.MAX_FREQS)
    if min(freqs) <= 0 or max(freqs) > modal_runner.MAX_HZ:
        raise ValueError("frequencies must be within 0-%.0f Hz" % modal_runner.MAX_HZ)
    req = {"simulation": m, "source_id": body["source_id"], "grid_id": body["grid_id"],
           "freqs": freqs, "label": str(body.get("label") or "")}
    info = store.create_task(mid, "modal", req, source_id=body["source_id"],
                             grid_id=body["grid_id"], n_freqs=len(freqs))
    job = jobs.submit_task(m, "modal", info)
    return {"modal": store.get_task(mid, "modal", info["id"]), "job": job}


@app.get("/api/simulations/{mid}/modal")
def modal_list(mid: str):
    store.get_simulation(mid)
    return store.tasks(mid, "modal")


@app.get("/api/simulations/{mid}/modal/{rid}")
def modal_get(mid: str, rid: str):
    return store.get_task(mid, "modal", rid)


@app.get("/api/simulations/{mid}/modal/{rid}/map")
def modal_map(mid: str, rid: str, freq: float = None):
    """Level (dB re free field at 10 m) per grid point, at the nearest computed frequency
    (all frequencies when `freq` is omitted)."""
    p = os.path.join(store.task_dir(mid, "modal", rid), "maps.npz")
    if not os.path.exists(p):
        raise KeyError(rid)
    z = np.load(p)
    f = z["freqs"]
    lv = z["level_db"]
    if freq is not None:
        k = int(np.argmin(np.abs(f - freq)))
        f, lv = f[k:k + 1], lv[k:k + 1]

    def num(a):
        return [None if not np.isfinite(v) else round(float(v), 2) for v in a]

    return {"points": np.round(z["points"], 4).tolist(), "valid": z["valid"].tolist(),
            "area_m2": z["area_m2"].tolist(), "freqs_hz": np.round(f, 3).tolist(),
            "level_db": [num(row) for row in lv]}


@app.delete("/api/simulations/{mid}/modal/{rid}")
def modal_delete(mid: str, rid: str):
    info = store.get_task(mid, "modal", rid)
    if info["status"] in ("queued", "running"):
        raise HTTPException(409, "queued or running: cancel its job first")
    store.delete_task(mid, "modal", rid)
    return {"ok": True}


@app.post("/api/simulations/{mid}/field")
def field_create(mid: str, body: dict):
    """Queue a wave-field animation: impulse response of the wave band on a grid."""
    from . import field_runner
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    check = simlib.validate(m, g, store.load_mesh(m["geometry_id"]), library,
                            store.load_closure(m["geometry_id"]))
    if check["unassigned_layers"]:
        raise HTTPException(409, "assign a material to every layer first")
    if not check["wave_available"]:
        raise HTTPException(409, "the wave solver needs a closed model or a declared open scene")
    if not any(s["id"] == body.get("source_id") for s in m["sources"]):
        raise KeyError(body.get("source_id"))
    if not any(x["id"] == body.get("grid_id") for x in m.get("receiver_grids", [])):
        raise KeyError(body.get("grid_id"))
    f_max = float(body.get("f_max_hz", 300.0))
    duration = float(body.get("duration_s", 0.2))
    fps = int(body.get("fps", max(4 * f_max, 400)))
    if not 50.0 <= f_max <= 1000.0:
        raise ValueError("f_max_hz must be within 50-1000 Hz")
    if not 0.02 <= duration <= field_runner.MAX_DURATION_S:
        raise ValueError("duration_s must be within 0.02-%.1f s" % field_runner.MAX_DURATION_S)
    if fps < 2.5 * f_max or duration * fps > field_runner.MAX_FRAMES:
        raise ValueError("fps must be >= 2.5 x f_max and duration x fps <= %d frames"
                         % field_runner.MAX_FRAMES)
    req = {"simulation": m, "source_id": body["source_id"], "grid_id": body["grid_id"],
           "f_max_hz": f_max, "duration_s": duration, "fps": fps,
           "label": str(body.get("label") or "")}
    info = store.create_task(mid, "field", req, source_id=body["source_id"],
                             grid_id=body["grid_id"])
    job = jobs.submit_task(m, "field", info)
    return {"field": store.get_task(mid, "field", info["id"]), "job": job}


@app.get("/api/simulations/{mid}/field")
def field_list(mid: str):
    store.get_simulation(mid)
    return store.tasks(mid, "field")


@app.get("/api/simulations/{mid}/field/{rid}")
def field_get(mid: str, rid: str):
    return store.get_task(mid, "field", rid)


@app.get("/api/simulations/{mid}/field/{rid}/frames")
def field_frames(mid: str, rid: str):
    """Frames quantised to int8 with one scale: value = byte / 127 * scale (base64,
    frames x points, row-major)."""
    import base64
    p = os.path.join(store.task_dir(mid, "field", rid), "field.npz")
    if not os.path.exists(p):
        raise KeyError(rid)
    z = np.load(p)
    fr = z["frames"]
    scale = float(np.abs(fr).max()) or 1.0
    q = np.clip(np.round(fr / scale * 127.0), -127, 127).astype(np.int8)
    return {"points": np.round(z["points"], 4).tolist(), "valid": z["valid"].tolist(),
            "normals": np.round(z["normals"], 4).tolist(), "u_axes": np.round(z["u_axes"], 4).tolist(),
            "n_frames": int(fr.shape[0]), "n_points": int(fr.shape[1]), "scale": scale,
            "data_b64": base64.b64encode(q.tobytes()).decode("ascii")}


@app.delete("/api/simulations/{mid}/field/{rid}")
def field_delete(mid: str, rid: str):
    info = store.get_task(mid, "field", rid)
    if info["status"] in ("queued", "running"):
        raise HTTPException(409, "queued or running: cancel its job first")
    store.delete_task(mid, "field", rid)
    return {"ok": True}


UI_STATE_KEYS = ("results", "auralizer", "editor")


@app.get("/api/simulations/{mid}/ui-state/{key}")
def get_ui_state(mid: str, key: str):
    """Saved views (result configurations, auralizer set-ups): opaque JSON for the UI."""
    if key not in UI_STATE_KEYS:
        raise KeyError(key)
    p = os.path.join(store.sim_dir(mid), "ui_%s.json" % key)
    store.get_simulation(mid)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@app.put("/api/simulations/{mid}/ui-state/{key}")
def put_ui_state(mid: str, key: str, body: dict):
    if key not in UI_STATE_KEYS:
        raise KeyError(key)
    store.get_simulation(mid)
    data = json.dumps(body)
    if len(data) > 512 * 1024:
        raise ValueError("saved view too large (max 512 kB)")
    with open(os.path.join(store.sim_dir(mid), "ui_%s.json" % key), "w", encoding="utf-8") as fh:
        fh.write(data)
    return body


_DECODERS = {}


@app.get("/api/heads")
def list_heads():
    """Listener heads of the binaural rendering (settings.head)."""
    return headlib.listing()


@app.get("/api/ambisonic/decoder")
def ambisonic_decoder(fs: int = 48000, head: str = "model"):
    """Binaural decoding filters for 2nd-order ambisonics (ACN/SN3D), head along +x."""
    if fs not in (44100, 48000):
        raise ValueError("fs must be 44100 or 48000")
    path = headlib.hrtf_path(head)
    if (fs, head) not in _DECODERS:
        from auralis.engine.ambidecode import decoder_filters
        from auralis.engine.hrtf_sofa import load_hrtf
        f, lat = decoder_filters(2, fs, hrtf_set=load_hrtf(path, fs))
        _DECODERS[(fs, head)] = {"order": 2, "fs": fs, "head": head, "n_taps": int(f.shape[-1]),
                                 "latency_samples": int(lat),
                                 "convention": "ACN/SN3D, x front, y left, z up",
                                 "filters": np.round(f, 7).tolist()}
    return _DECODERS[(fs, head)]


@app.get("/api/simulations/{mid}/export.xlsx")
def export_xlsx(mid: str):
    m = store.get_simulation(mid)
    meta = _meta(mid)
    data = exportlib.xlsx(m, meta, store.get_geometry(m["geometry_id"]).get("name", ""))
    return Response(data, media_type="application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet",
                    headers={"Content-Disposition": 'attachment; filename="%s.xlsx"'
                             % exportlib._safe(m["name"])})


@app.get("/api/simulations/{mid}/irs.zip")
def export_irs(mid: str):
    m = store.get_simulation(mid)
    data = exportlib.ir_zip(m, _meta(mid), store.results_dir(mid))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="%s_irs.zip"'
                             % exportlib._safe(m["name"])})


@app.get("/api/simulations/{mid}/reflections")
def reflections(mid: str, source_id: str, receiver_id: str, max_order: int = None,
                azimuth_deg: float = None, elevation_deg: float = None):
    """Deterministic arrivals of one pair, with their paths (reflection tracking)."""
    m = store.get_simulation(mid)
    g = store.get_geometry(m["geometry_id"])
    check = simlib.validate(m, g, store.load_mesh(m["geometry_id"]), library,
                            store.load_closure(m["geometry_id"]))
    if check["unassigned_layers"]:
        raise HTTPException(409, "assign a material to every layer first")
    return refllib.track(m, store.load_mesh(m["geometry_id"]), library, g, source_id,
                         receiver_id, max_order, azimuth_deg, elevation_deg, store.root)


@app.post("/api/simulations/{mid}/motion/check")
def motion_check(mid: str, body: dict):
    """Checks shown before an animated render (paths, distances, duration)."""
    return _motion_check(mid, body)[2]


@app.post("/api/simulations/{mid}/motion")
def motion_render(mid: str, body: dict):
    """Queue an animated binaural render: moving sources and/or receiver."""
    m, strips, check = _motion_check(mid, body)
    if not check["ok"]:
        raise HTTPException(409, "not ready to render: " + "; ".join(check["blocking"]))
    req = {"simulation": m, "receiver_id": body["receiver_id"], "strips": strips,
           "duration_s": check["duration_s"], "doppler": bool(body.get("doppler", True)),
           "reverb_dt": float(body.get("reverb_dt", 0.75)),
           "master_db": float(body.get("master_db", 0.0)), "label": str(body.get("label") or ""),
           "turn_taking": _turn_taking(body.get("turn_taking"), strips),
           "background_noise": bool(body.get("background_noise", False)),
           "wave_max_runs": max(2, min(100000, int(body.get("wave_max_runs", 64))))}
    if req["background_noise"] and not m["settings"].get("background_recording") and \
            speechlib.noise_levels(m["settings"].get("background_noise", "none")) is None:
        raise ValueError("background_noise: the simulation has no background noise "
                         "(Settings > background noise)")
    if not 0.1 <= req["reverb_dt"] <= 5.0:
        raise ValueError("reverb_dt must be within 0.1-5 s")
    info = store.create_motion(mid, req)
    job = jobs.submit_motion(m, info)
    return {"render": store.get_motion(mid, info["id"]), "job": job}


@app.get("/api/simulations/{mid}/motion")
def motion_list(mid: str):
    store.get_simulation(mid)
    return [_motion_public(mid, x) for x in store.motions(mid)]


@app.get("/api/simulations/{mid}/motion/{rid}")
def motion_get(mid: str, rid: str):
    return _motion_public(mid, store.get_motion(mid, rid))


@app.delete("/api/simulations/{mid}/motion/{rid}")
def motion_delete(mid: str, rid: str):
    info = store.get_motion(mid, rid)
    if info["status"] in ("queued", "running"):
        raise HTTPException(409, "the render is queued or running: cancel its job first")
    store.delete_motion(mid, rid)
    return {"ok": True}


@app.get("/api/simulations/{mid}/motion/{rid}/{name}")
def motion_file(mid: str, rid: str, name: str):
    if name not in ("render.wav", "credits.txt"):
        raise KeyError(name)
    p = os.path.join(store.motion_dir(mid, rid), name)
    if not os.path.exists(p):
        raise KeyError(name)
    return FileResponse(p, filename="%s_%s_%s" % (mid, rid, name))


def _motion_public(mid, info):
    base = "/api/simulations/%s/motion/%s/" % (mid, info["id"])
    if info["status"] == "done":
        info = dict(info, url=base + "render.wav", credits_url=base + "credits.txt")
    return info


@app.post("/api/sounds")
async def upload_sound(file: UploadFile = File(...), name: str = Form(""), credit: str = Form("")):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        x, fs = read_wav(tmp)
    except Exception as exc:
        raise ValueError("not a readable WAV file (%s)" % exc)
    finally:
        os.remove(tmp)
    x = np.asarray(x, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    sid = new_id("u")
    write_wav(os.path.join(_sounds_dir(), sid + ".wav"), mono, fs)
    meta = {"id": sid, "name": name or os.path.splitext(file.filename or sid)[0],
            "kind": "uploaded", "duration_s": round(len(mono) / fs, 2), "created": now(),
            "credit": credit.strip() or None}
    with open(os.path.join(_sounds_dir(), sid + ".json"), "w") as fh:
        json.dump(meta, fh)
    return meta


# -- built UI (served last so /api wins) -----------------------------------------------
@app.get("/{path:path}")
def ui(path: str):
    if path.startswith("api/"):
        raise KeyError(path)
    if not os.path.isdir(WEB_DIST):
        return JSONResponse({"detail": "UI not built: run `npm run build` in webapp/web"},
                            status_code=503)
    f = os.path.normpath(os.path.join(WEB_DIST, path))
    if path and f.startswith(WEB_DIST) and os.path.isfile(f) and not f.endswith("index.html"):
        return FileResponse(f)          # hashed assets: cacheable
    # the page itself must never be cached, or a rebuilt UI stays invisible
    return FileResponse(os.path.join(WEB_DIST, "index.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
