"""OpenStreetMap buildings -> a street scene: footprints extruded to heights derived from their
storey counts, a ground plane, in a local frame whose x axis follows a named street.
python tools/osm_street.py overpass.json --street "Rue Réaumur" --x -70 250 --y 50 --out dir
Data: (c) OpenStreetMap contributors, ODbL."""
import argparse
import json
import math
import os

import numpy as np

# Storey heights of a Parisian apartment building (ASSUMED, typical Haussmann values; OSM gives
# storey counts, not metres here): a tall ground floor, then regular floors; a mansard storey
# adds less than a full floor because it is set back under the roof.
GROUND_FLOOR_M = 4.0
FLOOR_M = 3.1
ROOF_LEVEL_M = 1.5
DEFAULT_LEVELS = 6           # most common value of the area when a building has no tag


def load(path):
    """(nodes, ways): building multipolygon relations become ways whose node ring is their
    OUTER outline (their courtyards, the inner rings, are filled: a closed courtyard reaches
    the street only over the roofs)."""
    d = json.load(open(path, encoding="utf-8"))
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in d["elements"] if e["type"] == "node"}
    ways = [e for e in d["elements"] if e["type"] == "way"]
    by_id = {w["id"]: w for w in ways}
    member_ways = set()
    for r in d["elements"]:
        if r["type"] != "relation" or "building" not in r.get("tags", {}):
            continue
        parts = [list(by_id[m["ref"]]["nodes"]) for m in r["members"]
                 if m["type"] == "way" and m.get("role") == "outer" and m["ref"] in by_id]
        member_ways.update(m["ref"] for m in r["members"] if m["type"] == "way")
        for ring in _rings(parts):
            ways.append({"type": "way", "id": r["id"], "nodes": ring,
                         "tags": dict(r["tags"], **{"_inner_filled": str(sum(
                             m.get("role") == "inner" for m in r["members"]))})})
    # an untagged member way only draws a relation's outline: not a building of its own
    ways = [w for w in ways if not (w["id"] in member_ways and "building" not in w.get("tags", {}))]
    return nodes, ways


def _rings(parts):
    """Closed rings joined from way node lists (end to end, reversing when needed)."""
    parts = [p for p in parts if len(p) >= 2]
    out = []
    while parts:
        ring = parts.pop(0)
        while ring[0] != ring[-1]:
            for k, p in enumerate(parts):
                if p[0] == ring[-1]:
                    ring += p[1:]
                elif p[-1] == ring[-1]:
                    ring += p[::-1][1:]
                else:
                    continue
                parts.pop(k)
                break
            else:
                break                             # open ring: dropped below
        if ring[0] == ring[-1] and len(ring) >= 4:
            out.append(ring)
    return out


class Frame:
    """Local metric frame: equirectangular projection at lat0, then turned so that the street
    axis is +x, origin at `origin` (projected metres before the turn)."""

    def __init__(self, lat0, lon0, angle, origin):
        self.lat0, self.lon0, self.angle = lat0, lon0, angle
        self.origin = np.asarray(origin, dtype=np.float64)

    def project(self, lat, lon):
        return np.array([(lon - self.lon0) * math.cos(math.radians(self.lat0)) * 111320.0,
                         (lat - self.lat0) * 110540.0])

    def __call__(self, lat, lon):
        p = self.project(lat, lon) - self.origin
        c, s = math.cos(-self.angle), math.sin(-self.angle)
        return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def street_frame(nodes, ways, street):
    """Frame whose x axis is the principal direction of the named street, origin at its
    point nearest to the centroid of its nodes."""
    lat0 = np.mean([nodes[n][0] for w in ways for n in w["nodes"][:1]])
    lon0 = np.mean([nodes[n][1] for w in ways for n in w["nodes"][:1]])
    f0 = Frame(lat0, lon0, 0.0, (0.0, 0.0))
    pts = np.array([f0.project(*nodes[n]) for w in ways
                    if w.get("tags", {}).get("name") == street for n in w["nodes"]])
    if len(pts) < 2:
        raise SystemExit("street '%s' not found in the data" % street)
    c = pts.mean(axis=0)
    u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0] if vt[0][0] >= 0 else -vt[0]
    origin = pts[np.argmin(np.linalg.norm(pts - c, axis=1))]
    return Frame(lat0, lon0, math.atan2(axis[1], axis[0]), origin)


def building_height(tags):
    """Cornice height (m) from building:levels and roof:levels, with the storey heights
    above; (height, levels, roof levels, levels were tagged)."""
    try:
        lv = int(float(tags.get("building:levels")))
        tagged = True
    except (TypeError, ValueError):
        lv, tagged = DEFAULT_LEVELS, False
    try:
        rl = int(float(tags.get("roof:levels", 0)))
    except (TypeError, ValueError):
        rl = 0
    lv = max(lv, 1)
    return GROUND_FLOOR_M + FLOOR_M * (lv - 1) + ROOF_LEVEL_M * rl, lv, rl, tagged


def _area2(p):
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def ear_clip(poly):
    """Triangles (index triples) of a simple polygon given counter-clockwise; None if the
    clipping gets stuck (self-intersecting outline)."""
    idx = list(range(len(poly)))
    out = []

    def inside(p, a, b, c):
        d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
        d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
        d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)
    guard = 0
    while len(idx) > 3 and guard < 10 * len(poly) ** 2:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = poly[i0], poly[i1], poly[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-9:
                continue                          # reflex or flat corner
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            return None
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def footprints(nodes, ways, frame, xr, ymax):
    """Buildings whose footprint centroid lies in x in xr, |y| < ymax (local frame):
    [{"id", "poly" (n,2) CCW, "height", "levels", "roof_levels", "tagged"}]."""
    out = []
    for w in ways:
        t = w.get("tags", {})
        if "building" not in t or len(w["nodes"]) < 4 or w["nodes"][0] != w["nodes"][-1]:
            continue
        p = np.array([frame(*nodes[n]) for n in w["nodes"][:-1]])
        c = p.mean(axis=0)
        if not (xr[0] <= c[0] <= xr[1] and abs(c[1]) <= ymax):
            continue
        keep = [0]                                 # drop repeated points
        for k in range(1, len(p)):
            if np.linalg.norm(p[k] - p[keep[-1]]) > 0.05:
                keep.append(k)
        p = p[keep]
        if len(p) >= 3 and np.linalg.norm(p[0] - p[-1]) <= 0.05:
            p = p[:-1]
        if len(p) < 3 or abs(_area2(p)) < 2.0:
            continue
        if _area2(p) < 0:
            p = p[::-1]
        h, lv, rl, tagged = building_height(t)
        out.append({"id": w["id"], "poly": p, "height": h, "levels": lv, "roof_levels": rl,
                    "tagged": tagged})
    return out


def mesh(buildings, ground_box):
    """(verts, faces, face layer, names, report): walls 'facade', flat roofs 'roof', and the
    ground rectangle 'ground' (two triangles; under the buildings too: their bottoms are
    open, the ground closes them)."""
    V, F, L = [], [], []
    names = ["facade", "roof", "ground"]
    stuck = 0

    def vid(p):
        V.append(p)
        return len(V) - 1
    for b in buildings:
        p, h = b["poly"], b["height"]
        lo = [vid((x, y, 0.0)) for x, y in p]
        hi = [vid((x, y, h)) for x, y in p]
        n = len(p)
        for k in range(n):
            a, c = k, (k + 1) % n
            F += [(lo[a], lo[c], hi[c]), (lo[a], hi[c], hi[a])]   # outward for a CCW outline
            L += [0, 0]
        tri = ear_clip(p)
        if tri is None:
            stuck += 1
            cc = vid((float(p[:, 0].mean()), float(p[:, 1].mean()), h))
            tri = [(None, k, (k + 1) % n) for k in range(n)]
            F += [(cc, hi[b_], hi[c_]) for _a, b_, c_ in tri]
        else:
            F += [(hi[a], hi[b_], hi[c_]) for a, b_, c_ in tri]
        L += [1] * len(tri)
    (x0, x1), (y0, y1) = ground_box
    g = [vid((x0, y0, 0.0)), vid((x1, y0, 0.0)), vid((x1, y1, 0.0)), vid((x0, y1, 0.0))]
    F += [(g[0], g[1], g[2]), (g[0], g[2], g[3])]
    L += [2, 2]
    rep = {"buildings": len(buildings), "triangulation_fallbacks": stuck,
           "levels_tagged": sum(b["tagged"] for b in buildings)}
    return (np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64),
            np.asarray(L, dtype=np.int32), names, rep)


def write_obj(path, v, f, fl, names, header):
    with open(path, "w", encoding="utf-8") as fh:
        for line in header:
            fh.write("# %s\n" % line)
        for p in v:
            fh.write("v %.3f %.3f %.3f\n" % tuple(p))
        for k, n in enumerate(names):
            fh.write("g %s\nusemtl %s\n" % (n, n))
            for t in f[fl == k]:
                fh.write("f %d %d %d\n" % tuple(t + 1))


def street_lines(nodes, ways, frame, xr, ymax, kinds=("primary", "secondary", "tertiary",
                                                       "residential", "living_street",
                                                       "pedestrian", "unclassified")):
    """Street centre lines in the local frame, clipped loosely to the box: [{"name",
    "highway", "lanes", "oneway", "points" (n,2)}] (for the visual ground and the sources)."""
    out = []
    for w in ways:
        t = w.get("tags", {})
        if t.get("highway") not in kinds:
            continue
        p = np.array([frame(*nodes[n]) for n in w["nodes"]])
        if not ((p[:, 0] >= xr[0] - 30) & (p[:, 0] <= xr[1] + 30) & (np.abs(p[:, 1]) <= ymax + 30)).any():
            continue
        out.append({"name": t.get("name", ""), "highway": t["highway"],
                    "lanes": int(t.get("lanes", 1) or 1), "oneway": t.get("oneway") == "yes",
                    "points": p.round(2).tolist()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("osm")
    ap.add_argument("--street", default="Rue Réaumur")
    ap.add_argument("--x", type=float, nargs=2, default=(-70.0, 250.0))
    ap.add_argument("--y", type=float, default=50.0)
    ap.add_argument("--context", type=float, default=0.0,
                    help="also list the buildings up to this distance (m) around the box, for "
                         "the visual model only (the acoustic model keeps the box)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nodes, ways = load(a.osm)
    fr = street_frame(nodes, ways, a.street)
    b = footprints(nodes, ways, fr, a.x, a.y)
    allp = np.concatenate([x["poly"] for x in b])
    box = ((float(allp[:, 0].min()) - 5.0, float(allp[:, 0].max()) + 5.0),
           (float(allp[:, 1].min()) - 5.0, float(allp[:, 1].max()) + 5.0))
    v, f, fl, names, rep = mesh(b, box)
    os.makedirs(a.out, exist_ok=True)
    write_obj(os.path.join(a.out, "street.obj"), v, f, fl, names,
              ["Auralis: %s from OpenStreetMap (c) OpenStreetMap contributors, ODbL" % a.street,
               "local frame: x along the street, z up, metres; heights from storey counts"])
    hs = np.array([x["height"] for x in b])
    info = {"street": a.street, "frame": {"lat0": fr.lat0, "lon0": fr.lon0,
                                          "angle_rad": fr.angle, "origin": fr.origin.tolist()},
            "box": box, "report": rep,
            "heights_m": {"min": float(hs.min()), "median": float(np.median(hs)),
                          "max": float(hs.max())},
            "storeys": {"ground_floor_m": GROUND_FLOOR_M, "floor_m": FLOOR_M,
                        "roof_level_m": ROOF_LEVEL_M, "default_levels": DEFAULT_LEVELS},
            "buildings": [{"id": x["id"], "poly": x["poly"].round(3).tolist(),
                           "height": round(x["height"], 2), "levels": x["levels"],
                           "roof_levels": x["roof_levels"], "tagged": x["tagged"]} for x in b],
            "streets": street_lines(nodes, ways, fr, a.x, a.y),
            "context": [], "visual_box": box,
            "license": "(c) OpenStreetMap contributors, ODbL 1.0 (openstreetmap.org/copyright)"}
    if a.context > 0:
        cx = (a.x[0] - a.context, a.x[1] + a.context)
        ids = {x["id"] for x in b}
        ctx = [x for x in footprints(nodes, ways, fr, cx, a.y + a.context) if x["id"] not in ids]
        info["context"] = [{"id": x["id"], "poly": x["poly"].round(3).tolist(),
                            "height": round(x["height"], 2), "levels": x["levels"],
                            "roof_levels": x["roof_levels"], "tagged": x["tagged"]} for x in ctx]
        allc = np.concatenate([x["poly"] for x in b + ctx])
        info["visual_box"] = ((float(allc[:, 0].min()) - 5.0, float(allc[:, 0].max()) + 5.0),
                              (float(allc[:, 1].min()) - 5.0, float(allc[:, 1].max()) + 5.0))
        info["streets"] = street_lines(nodes, ways, fr, cx, a.y + a.context)
        print("context: %d more buildings for the visual model" % len(ctx))
    with open(os.path.join(a.out, "street.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False)
    print("%d buildings (%d with a storey count), heights %.1f / %.1f / %.1f m (min / median / "
          "max), %d triangles, %d triangulation fallbacks, box x %.0f..%.0f y %.0f..%.0f"
          % (rep["buildings"], rep["levels_tagged"], hs.min(), np.median(hs), hs.max(), len(f),
             rep["triangulation_fallbacks"], box[0][0], box[0][1], box[1][0], box[1][1]))


if __name__ == "__main__":
    main()
