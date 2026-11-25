# -*- coding: utf-8 -*-
"""Download the CC0 surface textures and environment maps of the realistic 3D view
from Poly Haven into webapp/assets/textures/. Usage: python -m webapp.server.fetch_textures [--force]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "assets", "textures")
FILES = "https://api.polyhaven.com/files/%s"
INFO = "https://api.polyhaven.com/info/%s"
UA = {"User-Agent": "Auralis-asset-fetch/0.1 (local research tool)"}
LICENSE = "CC0 (Poly Haven)"

# Surface key used by the UI -> Poly Haven texture id. The repeat size is the scanned
# area of the asset (its `dimensions`), so the texture keeps its physical scale.
TEXTURES = {
    "parquet": "herringbone_parquet",
    "wood_floor": "wood_floor",
    "wood_panel": "oak_veneer_01",
    "carpet": "poly_wool_herringbone",
    "plaster": "painted_plaster_wall",
    "concrete": "concrete_wall_008",
    "brick": "red_brick_03",
    "stone": "sandstone_blocks_08",
    "marble": "marble_01",
    "tiles": "floor_tiles_08",
    "metal_sheet": "box_profile_metal_sheet",
    "velvet": "velour_velvet",
    "fabric": "hessian_230",
    "ceiling_tiles": "ceiling_interior",
    "linoleum": "linoleum_brown",
    "asphalt": "asphalt_02",
    "paving": "rectangular_paving",
    "grass": "leafy_grass",
    "gravel": "gravel_floor_02",
    "render": "beige_wall_001",
}
# Poly Haven map name -> file stem in our folder (1k JPG; normal map in the OpenGL convention).
MAPS = {"Diffuse": "diff", "nor_gl": "nor", "Rough": "rough"}
# Image-based lighting: a soft interior and an overcast sky (1k Radiance .hdr).
HDRIS = {"interior": "photo_studio_london_hall", "exterior": "kloofendal_overcast_puresky"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def download(url, path, md5=None, force=False):
    """Write url to path (atomically), checking the md5 Poly Haven publishes."""
    if os.path.exists(path) and not force:
        return os.path.getsize(path)
    data = get(url)
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise ValueError("md5 mismatch for %s" % url)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return len(data)


def credit(info):
    return ", ".join("%s (%s)" % (k, v) for k, v in (info.get("authors") or {}).items())


def fetch_texture(key, aid, force=False):
    d = os.path.join(OUT, key)
    os.makedirs(d, exist_ok=True)
    files = json.loads(get(FILES % aid))
    info = json.loads(get(INFO % aid))
    maps, size = {}, 0
    for name, stem in MAPS.items():
        f = (files.get(name) or {}).get("1k", {}).get("jpg")
        if not f:
            continue
        rel = "%s.jpg" % stem
        size += download(f["url"], os.path.join(d, rel), f.get("md5"), force)
        maps[stem] = "%s/%s" % (key, rel)
    if "diff" not in maps:
        raise ValueError("%s has no 1k diffuse JPG" % aid)
    dims = info.get("dimensions") or [1000, 1000]
    return {"id": aid, "name": info.get("name", aid), "credit": credit(info),
            "size_m": [round(dims[0] / 1000.0, 4), round(dims[1] / 1000.0, 4)],
            "maps": maps, "bytes": size, "license": LICENSE,
            "url": "https://polyhaven.com/a/%s" % aid}


def fetch_hdri(key, aid, force=False):
    os.makedirs(os.path.join(OUT, "hdri"), exist_ok=True)
    f = json.loads(get(FILES % aid))["hdri"]["1k"]["hdr"]
    info = json.loads(get(INFO % aid))
    rel = "hdri/%s.hdr" % key
    size = download(f["url"], os.path.join(OUT, rel), f.get("md5"), force)
    return {"id": aid, "name": info.get("name", aid), "credit": credit(info), "file": rel,
            "bytes": size, "license": LICENSE, "url": "https://polyhaven.com/a/%s" % aid}


def main():
    ap = argparse.ArgumentParser(description="Fetch the CC0 surface textures and HDRIs (Poly Haven)")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    index_path = os.path.join(OUT, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    index.setdefault("textures", {})
    index.setdefault("hdri", {})
    failed = []
    jobs = [("textures", k, v, fetch_texture) for k, v in TEXTURES.items()]
    jobs += [("hdri", k, v, fetch_hdri) for k, v in HDRIS.items()]
    for group, key, aid, fn in jobs:
        try:
            index[group][key] = fn(key, aid, args.force)
            print("ok   %-14s %-30s %5.1f MB" % (key, aid, index[group][key]["bytes"] / 1e6), flush=True)
        except Exception as exc:
            failed.append(key)
            print("FAIL %-14s %-30s %s" % (key, aid, exc), flush=True)
        time.sleep(0.2)
        with open(index_path, "w") as fh:
            json.dump(index, fh, indent=1)
    total = sum(v["bytes"] for g in ("textures", "hdri") for v in index[g].values())
    print("%d textures, %d HDRIs, %.0f MB in %s; failed: %s"
          % (len(index["textures"]), len(index["hdri"]), total / 1e6, OUT, failed or "none"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
