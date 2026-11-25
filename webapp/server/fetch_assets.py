# -*- coding: utf-8 -*-
"""Download the CC0 visual models of the object catalog from Poly Haven
(glTF, 1k textures) into webapp/assets/visual/ and measure their bounding box.
Usage: python -m webapp.server.fetch_assets [--force]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "data", "objects_catalog.json")
OUT = os.path.join(os.path.dirname(HERE), "assets", "visual")
API = "https://api.polyhaven.com/files/%s"
THUMB = "https://cdn.polyhaven.com/asset_img/thumbs/%s.png?width=256&height=256"
UA = {"User-Agent": "Auralis-asset-fetch/0.1 (local research tool)"}
COMP = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def _matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    t = np.eye(4)
    t[:3, 3] = node.get("translation", [0, 0, 0])
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    r = np.eye(4)
    r[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    s = np.diag(list(node.get("scale", [1, 1, 1])) + [1.0])
    return t @ r @ s


def gltf_bbox(path):
    """Axis-aligned bbox of the scene in glTF space (Y up), node transforms applied."""
    g = json.load(open(path, encoding="utf-8"))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def visit(ni, parent):
        node = g["nodes"][ni]
        m = parent @ _matrix(node)
        if "mesh" in node:
            for prim in g["meshes"][node["mesh"]]["primitives"]:
                acc = g["accessors"][prim["attributes"]["POSITION"]]
                a, b = np.array(acc["min"], float), np.array(acc["max"], float)
                corners = np.array([[x, y, z, 1.0] for x in (a[0], b[0])
                                    for y in (a[1], b[1]) for z in (a[2], b[2])])
                p = (m @ corners.T).T[:, :3]
                lo[:] = np.minimum(lo, p.min(0))
                hi[:] = np.maximum(hi, p.max(0))
        for c in node.get("children", []):
            visit(c, m)

    scene = g["scenes"][g.get("scene", 0)]
    for ni in scene["nodes"]:
        visit(ni, np.eye(4))
    return lo, hi


def to_z_up(lo, hi):
    """glTF Y-up box -> Z-up (rotation +90 deg about X: (x, y, z) -> (x, -z, y))."""
    c = np.array([[x, -z, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return c.min(0), c.max(0)


def fetch(entry, force=False):
    aid = entry["id"]
    d = os.path.join(OUT, aid)
    main = os.path.join(d, aid + ".gltf")
    if not os.path.exists(main) or force:
        files = json.loads(get(API % aid))
        g = files["gltf"]["1k"]["gltf"]
        os.makedirs(d, exist_ok=True)
        for rel, info in g.get("include", {}).items():
            p = os.path.normpath(os.path.join(d, rel))
            if not p.startswith(d):
                raise ValueError("unsafe path in %s: %s" % (aid, rel))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(get(info["url"]))
        with open(main, "wb") as fh:
            fh.write(get(g["url"]))
        with open(os.path.join(d, "thumbnail.png"), "wb") as fh:
            fh.write(get(THUMB % aid))
    lo, hi = to_z_up(*gltf_bbox(main))
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
    return {"dims": (hi - lo).round(4).tolist(),
            "offset": (-np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])).round(5).tolist(),
            "gltf": aid + ".gltf", "bytes": size, "license": "CC0 (Poly Haven)"}


def main():
    ap = argparse.ArgumentParser(description="Fetch the CC0 visual models (Poly Haven)")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    args = ap.parse_args()
    cat = json.load(open(CATALOG, encoding="utf-8"))["objects"]
    os.makedirs(OUT, exist_ok=True)
    index_path = os.path.join(OUT, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    failed = []
    for e in cat:
        if e["source"] != "polyhaven":
            continue
        try:
            index[e["id"]] = fetch(e, args.force)
            print("ok   %-28s %6.1f MB  dims %s" % (e["id"], index[e["id"]]["bytes"] / 1e6,
                                                   index[e["id"]]["dims"]), flush=True)
        except Exception as exc:
            failed.append(e["id"])
            print("FAIL %-28s %s" % (e["id"], exc), flush=True)
        time.sleep(0.2)
        with open(index_path, "w") as fh:
            json.dump(index, fh, indent=1)
    total = sum(v["bytes"] for v in index.values())
    print("%d models, %.0f MB in %s; failed: %s" % (len(index), total / 1e6, OUT, failed or "none"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
