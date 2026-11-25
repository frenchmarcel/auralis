# -*- coding: utf-8 -*-
"""On-disk storage of projects, spaces, geometries, simulations and sounds
(JSON + npz, one folder per object, outside the repository)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import shutil
import threading

import numpy as np

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "AuralisProjects")


def now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def new_id(prefix):
    return prefix + secrets.token_hex(4)


def default_settings():
    return {"mode": "survey", "transition_hz": None, "ir_length_s": None,
            "ism_order": 2, "n_rays": 200000, "solvers": "both",
            "temperature_c": 20.0, "humidity_percent": 50.0,
            "diffraction": True, "conformal": True, "open_scene": False,
            "speech_effort": "normal", "background_noise": "none", "grid_rays": 50000,
            "stop_decay_db": None, "rbs_scattering": True, "head": "model",
            "background_recording": None}


class Store:
    """Flat collections: projects/<id>.json, spaces/<id>.json,
    geometries/<id>/, simulations/<id>/, sounds/<id>.*"""

    def __init__(self, root=None):
        self.root = os.path.abspath(root or os.environ.get("AURALIS_PROJECTS", DEFAULT_ROOT))
        for d in ("projects", "spaces", "geometries", "simulations", "sounds"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.lock = threading.RLock()

    # -- low level -------------------------------------------------------
    def _read(self, path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp, path)

    def _list(self, kind):
        d = os.path.join(self.root, kind)
        out = []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if kind in ("projects", "spaces") and name.endswith(".json"):
                out.append(self._read(p))
            elif os.path.isdir(p):
                f = os.path.join(p, {"geometries": "geometry.json",
                                     "simulations": "simulation.json"}[kind])
                if os.path.exists(f):
                    out.append(self._read(f))
        return out

    def _path(self, kind, oid):
        if not oid or "/" in oid or oid.startswith("."):
            raise KeyError(oid)
        if kind in ("projects", "spaces"):
            return os.path.join(self.root, kind, oid + ".json")
        return os.path.join(self.root, kind, oid)

    # -- projects / spaces ------------------------------------------------
    def create_project(self, name, description=""):
        p = {"id": new_id("p"), "name": name or "Project", "description": description or "",
             "created": now()}
        self._write(self._path("projects", p["id"]), p)
        return p

    def create_space(self, pid, name, description=""):
        self.get_project(pid)
        s = {"id": new_id("s"), "project_id": pid, "name": name or "Space",
             "description": description or "", "created": now()}
        self._write(self._path("spaces", s["id"]), s)
        return s

    def get_project(self, pid):
        p = self._path("projects", pid)
        if not os.path.exists(p):
            raise KeyError(pid)
        return self._read(p)

    def get_space(self, sid):
        p = self._path("spaces", sid)
        if not os.path.exists(p):
            raise KeyError(sid)
        return self._read(p)

    def patch(self, kind, oid, fields):
        with self.lock:
            if kind == "projects":
                obj, path = self.get_project(oid), self._path(kind, oid)
            elif kind == "spaces":
                obj, path = self.get_space(oid), self._path(kind, oid)
            else:
                obj = self.get_geometry(oid)
                path = os.path.join(self._path(kind, oid), "geometry.json")
            for k in ("name", "description"):
                if fields.get(k) is not None:
                    obj[k] = str(fields[k])
            self._write(path, obj)
            return obj

    def delete_project(self, pid):
        self.get_project(pid)
        for s in self._list("spaces"):
            if s["project_id"] == pid:
                self.delete_space(s["id"])
        os.remove(self._path("projects", pid))

    def delete_space(self, sid):
        self.get_space(sid)
        for g in self._list("geometries"):
            if g["space_id"] == sid:
                self.delete_geometry(g["id"])
        os.remove(self._path("spaces", sid))

    # -- geometries --------------------------------------------------------
    def geometry_dir(self, gid):
        return self._path("geometries", gid)

    def create_geometry(self, sid, meta, verts, faces, face_layer, layer_names, closed_mask,
                        source_bytes, source_filename, closure=None):
        self.get_space(sid)
        gid = new_id("g")
        d = self.geometry_dir(gid)
        os.makedirs(d)
        extra = {}
        if closure:
            extra = {"vox_interior": np.packbits(closure["interior"]),
                     "vox_shape": np.array(closure["interior"].shape),
                     "vox_origin": closure["origin"], "vox_dx": closure["dx"],
                     "vox_closed": closure["closed"], "vox_volume": closure["volume"]}
            if closure.get("exterior") is not None:
                extra["vox_exterior"] = np.packbits(closure["exterior"])
        np.savez_compressed(os.path.join(d, "mesh.npz"), verts=verts, faces=faces,
                            face_layer=face_layer, closed=closed_mask,
                            layer_names=np.array(layer_names, dtype=object), **extra)
        with open(os.path.join(d, "source.obj"), "wb") as fh:
            fh.write(source_bytes)
        g = dict(meta)
        g.update({"id": gid, "space_id": sid, "created": now(),
                  "source_filename": source_filename})
        self._write(os.path.join(d, "geometry.json"), g)
        return g

    def get_geometry(self, gid):
        p = os.path.join(self.geometry_dir(gid), "geometry.json")
        if not os.path.exists(p):
            raise KeyError(gid)
        return self._read(p)

    def load_room(self, gid):
        """The imported model only (no objects): (verts, faces, face_layer, names)."""
        z = np.load(os.path.join(self.geometry_dir(gid), "mesh.npz"), allow_pickle=True)
        return z["verts"], z["faces"], z["face_layer"], [str(n) for n in z["layer_names"]]

    def load_closure(self, gid):
        """Voxel closure of the imported room (None when its edges close topologically)."""
        z = np.load(os.path.join(self.geometry_dir(gid), "mesh.npz"), allow_pickle=True)
        if "vox_interior" not in z:
            return None
        shape = tuple(int(x) for x in z["vox_shape"])
        interior = np.unpackbits(z["vox_interior"])[:int(np.prod(shape))].reshape(shape).astype(bool)
        out = {"closed": bool(z["vox_closed"]), "volume": float(z["vox_volume"]),
               "dx": float(z["vox_dx"]), "origin": z["vox_origin"], "interior": interior}
        if "vox_exterior" in z:                 # outside air (missing from older geometries)
            out["exterior"] = np.unpackbits(z["vox_exterior"])[:int(np.prod(shape))].reshape(
                shape).astype(bool)
        return out

    def load_mesh(self, gid):
        """What the solvers see: room + object proxies when objects are placed."""
        d = self.geometry_dir(gid)
        p = os.path.join(d, "mesh_assembled.npz")
        z = np.load(p if os.path.exists(p) else os.path.join(d, "mesh.npz"), allow_pickle=True)
        return (z["verts"], z["faces"], z["face_layer"], z["closed"],
                [str(n) for n in z["layer_names"]])

    def load_owners(self, gid):
        """Object id per triangle of the assembled mesh (None for the room)."""
        p = os.path.join(self.geometry_dir(gid), "mesh_assembled.npz")
        if not os.path.exists(p):
            return None
        return [None if o == "" else str(o) for o in np.load(p, allow_pickle=True)["owners"]]

    def save_assembled(self, gid, verts, faces, face_layer, closed, names, owners):
        p = os.path.join(self.geometry_dir(gid), "mesh_assembled.npz")
        if owners is None:
            if os.path.exists(p):
                os.remove(p)
            return
        np.savez_compressed(p, verts=verts, faces=faces, face_layer=face_layer, closed=closed,
                            layer_names=np.array(names, dtype=object),
                            owners=np.array([o or "" for o in owners], dtype=object))

    def save_geometry(self, g):
        self._write(os.path.join(self.geometry_dir(g["id"]), "geometry.json"), g)
        return g

    def duplicate_geometry(self, gid, name):
        g = self.get_geometry(gid)
        nid = new_id("g")
        shutil.copytree(self.geometry_dir(gid), self.geometry_dir(nid))
        g = dict(g, id=nid, name=name or g["name"] + " (copy)", created=now())
        return self.save_geometry(g)

    def simulations_of(self, gid):
        return [m for m in self._list("simulations") if m["geometry_id"] == gid]

    def delete_geometry(self, gid):
        self.get_geometry(gid)
        for m in self._list("simulations"):
            if m["geometry_id"] == gid:
                self.delete_simulation(m["id"])
        shutil.rmtree(self.geometry_dir(gid))

    # -- simulations -------------------------------------------------------
    def sim_dir(self, mid):
        return self._path("simulations", mid)

    def create_simulation(self, gid, name, copy_from=None):
        g = self.get_geometry(gid)
        layers = {l["name"]: {"material_id": None, "scattering": None} for l in g["layers"]}
        m = {"id": new_id("m"), "geometry_id": gid, "name": name or "Simulation",
             "description": "", "created": now(), "status": "draft", "last_run": None,
             "results_stale": False, "layers": layers, "sources": [], "receivers": [],
             "receiver_grids": [],
             "settings": default_settings()}
        if copy_from:
            src = self.get_simulation(copy_from)
            for k in ("sources", "receivers", "settings", "receiver_grids"):
                m[k] = json.loads(json.dumps(src.get(k, [])))
            for ln, v in src["layers"].items():
                if ln in layers:            # same geometry, or a layer of the same name
                    layers[ln] = dict(v)
            m["receiver_grids"] = [x for x in m["receiver_grids"]
                                   if x.get("kind") != "surface" or x.get("layer") in layers]
        os.makedirs(self.sim_dir(m["id"]))
        self._write(os.path.join(self.sim_dir(m["id"]), "simulation.json"), m)
        return m

    def get_simulation(self, mid):
        p = os.path.join(self.sim_dir(mid), "simulation.json")
        if not os.path.exists(p):
            raise KeyError(mid)
        return self._read(p)

    def save_simulation(self, m):
        with self.lock:
            self._write(os.path.join(self.sim_dir(m["id"]), "simulation.json"), m)
        return m

    def delete_simulation(self, mid):
        self.get_simulation(mid)
        shutil.rmtree(self.sim_dir(mid))

    def results_dir(self, mid):
        return os.path.join(self.sim_dir(mid), "results")

    # -- tasks run beside the static results: animated renders, modal maps ----
    TASK_PREFIX = {"motion": "a", "modal": "f", "field": "w", "mix": "x"}

    def task_dir(self, mid, kind, rid):
        if (kind not in self.TASK_PREFIX or not rid.startswith(self.TASK_PREFIX[kind])
                or not rid.isalnum()):
            raise KeyError(rid)
        return os.path.join(self.sim_dir(mid), kind, rid)

    def create_task(self, mid, kind, request, **fields):
        rid = new_id(self.TASK_PREFIX[kind])
        d = self.task_dir(mid, kind, rid)
        os.makedirs(d)
        info = dict({"id": rid, "simulation_id": mid, "kind": kind, "created": now(),
                     "status": "queued", "label": request.get("label") or "", "error": None},
                    **fields)
        self._write(os.path.join(d, "request.json"), request)
        self._write(os.path.join(d, "info.json"), info)
        return info

    def get_task(self, mid, kind, rid):
        p = os.path.join(self.task_dir(mid, kind, rid), "info.json")
        if not os.path.exists(p):
            raise KeyError(rid)
        info = self._read(p)
        mp = os.path.join(os.path.dirname(p), "meta.json")
        if info["status"] == "done" and os.path.exists(mp):
            info["result"] = self._read(mp)
        return info

    def set_task(self, mid, kind, rid, **fields):
        with self.lock:
            p = os.path.join(self.task_dir(mid, kind, rid), "info.json")
            info = self._read(p)
            info.update(fields)
            self._write(p, info)
        return info

    def tasks(self, mid, kind):
        d = os.path.join(self.sim_dir(mid), kind)
        if not os.path.isdir(d):
            return []
        out = []
        for rid in os.listdir(d):
            try:
                out.append(self.get_task(mid, kind, rid))
            except (KeyError, ValueError):
                continue
        return sorted(out, key=lambda x: x["created"], reverse=True)

    def delete_task(self, mid, kind, rid):
        self.get_task(mid, kind, rid)
        shutil.rmtree(self.task_dir(mid, kind, rid))

    # animated renders (kept names)
    def motion_dir(self, mid, rid):
        return self.task_dir(mid, "motion", rid)

    def create_motion(self, mid, request):
        return self.create_task(mid, "motion", request, receiver_id=request["receiver_id"],
                                n_sources=len(request["strips"]))

    def get_motion(self, mid, rid):
        return self.get_task(mid, "motion", rid)

    def set_motion(self, mid, rid, **fields):
        return self.set_task(mid, "motion", rid, **fields)

    def motions(self, mid):
        return self.tasks(mid, "motion")

    def delete_motion(self, mid, rid):
        self.delete_task(mid, "motion", rid)


    # -- tree ----------------------------------------------------------------
    def tree(self):
        sims = self._list("simulations")
        geos = self._list("geometries")
        spaces = self._list("spaces")
        out = []
        for p in sorted(self._list("projects"), key=lambda x: x["created"]):
            pp = dict(p, spaces=[])
            for s in sorted((s for s in spaces if s["project_id"] == p["id"]),
                            key=lambda x: x["created"]):
                ss = dict(s, geometries=[])
                for g in sorted((g for g in geos if g["space_id"] == s["id"]),
                                key=lambda x: x["created"]):
                    ss["geometries"].append({
                        "id": g["id"], "space_id": s["id"], "name": g["name"],
                        "created": g["created"], "n_triangles": g["n_triangles"],
                        "n_layers": len(g["layers"]), "watertight": g["check"]["watertight"],
                        "simulations": [self.sim_summary(m) for m in sorted(
                            (m for m in sims if m["geometry_id"] == g["id"]),
                            key=lambda x: x["created"])]})
                pp["spaces"].append(ss)
            out.append(pp)
        return out

    def sim_summary(self, m):
        out = {k: m.get(k) for k in ("id", "geometry_id", "name", "created", "status",
                                     "last_run", "results_stale")}
        out["has_results"] = os.path.exists(os.path.join(self.results_dir(m["id"]), "meta.json"))
        return out

    def recent(self, limit=6):
        out = []
        for m in self._list("simulations"):
            if not m.get("last_run"):
                continue
            try:
                g = self.get_geometry(m["geometry_id"])
                s = self.get_space(g["space_id"])
                p = self.get_project(s["project_id"])
            except KeyError:
                continue
            r = self.sim_summary(m)
            r.update(project_name=p["name"], space_name=s["name"], geometry_name=g["name"],
                     duration_s=m.get("last_duration_s"))
            out.append(r)
        out.sort(key=lambda r: r["last_run"], reverse=True)
        return out[:limit]
