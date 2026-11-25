"""Display-only .glb of an OpenStreetMap street (tools/osm_street.py output): Parisian
apartment-building facades drawn procedurally (stone courses, tall windows, balconies on the
2nd and 5th floors, shop fronts, zinc mansards), zinc roofs, asphalt roads and paved sidewalks.
The solvers never see it; the acoustic model is street.obj.
python tools/street_visual.py street.json --road asphalt.jpg --sidewalk paving.jpg --out street.glb
Needs Pillow."""
import argparse
import io
import json
import math
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BAY_M = 3.0                  # width of one window bay
STONE_TINTS = [(226, 214, 190), (214, 203, 182), (232, 222, 203), (205, 193, 171)]
SIGNS = [(40, 60, 48), (92, 28, 34), (24, 30, 44), (70, 52, 30)]


def _noise(w, h, amp, seed, blur=1.2):
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, amp, (h, w)).astype(np.float32)
    im = Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(blur))
    return np.asarray(im, dtype=np.float32) - 128.0


def _stone(w, h, seed, course_px):
    """Light limestone with courses and staggered joints (white: tinted per building)."""
    base = np.full((h, w, 3), 236.0, dtype=np.float32)
    base += _noise(w, h, 9.0, seed)[..., None] * np.array([1.0, 0.97, 0.9])
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(seed + 1)
    for k, y in enumerate(range(0, h, course_px)):
        d.line([(0, y), (w, y)], fill=(196, 190, 178), width=2)
        off = (k % 2) * course_px
        for x in range(off, w, 2 * course_px + int(rng.integers(-8, 8))):
            d.line([(x, y), (x, y + course_px)], fill=(205, 199, 187), width=1)
    return im


def _window(d, x0, y0, x1, y1, shutters=False):
    """A tall French window: stone surround, cream frame, dark glass with a reflection."""
    d.rectangle([x0 - 14, y0 - 16, x1 + 14, y1 + 6], fill=(244, 238, 226))      # surround
    d.rectangle([x0 - 14, y0 - 22, x1 + 14, y0 - 16], fill=(214, 206, 190))     # lintel shadow
    d.rectangle([x0, y0, x1, y1], fill=(236, 232, 222))                          # frame
    g0, g1 = x0 + 7, x1 - 7
    for yy in range(y0 + 7, y1 - 6):
        t = (yy - y0) / max(y1 - y0, 1)
        c = (int(52 + 40 * (1 - t)), int(62 + 40 * (1 - t)), int(74 + 44 * (1 - t)))
        d.line([(g0, yy), (g1, yy)], fill=c)
    xm = (x0 + x1) // 2
    d.line([(xm, y0 + 7), (xm, y1 - 6)], fill=(236, 232, 222), width=5)         # mullion
    yt = y0 + int(0.28 * (y1 - y0))
    d.line([(g0, yt), (g1, yt)], fill=(236, 232, 222), width=5)                 # transom
    if shutters:
        sw = (x1 - x0) // 2 + 10
        for sx in (x0 - 18 - sw, x1 + 18):
            d.rectangle([sx, y0, sx + sw, y1], fill=(120, 132, 128))
            for yy in range(y0 + 6, y1, 9):
                d.line([(sx + 3, yy), (sx + sw - 3, yy)], fill=(98, 110, 106))


def _railing(d, x0, x1, y0, y1):
    """Wrought-iron balcony railing with a stone slab under it."""
    d.rectangle([x0, y1, x1, y1 + 14], fill=(200, 192, 176))
    d.rectangle([x0, y1 + 14, x1, y1 + 20], fill=(150, 144, 132))
    d.line([(x0, y0), (x1, y0)], fill=(28, 28, 30), width=5)
    for x in range(x0, x1, 12):
        d.line([(x, y0), (x, y1)], fill=(28, 28, 30), width=2)
        d.arc([x - 6, y0 + 10, x + 6, y0 + 30], 0, 180, fill=(28, 28, 30), width=2)


def facade_storey(balcony, seed, w=512, h=530):
    """One bay of a regular floor (3.0 m x 3.1 m)."""
    im = _stone(w, h, seed, 76)
    d = ImageDraw.Draw(im)
    x0, x1 = int(0.29 * w), int(0.71 * w)
    y0, y1 = int(0.13 * h), int(0.82 * h)
    _window(d, x0, y0, x1, y1)
    if balcony:
        _railing(d, 0, w, int(0.62 * h), int(0.82 * h))
    else:
        _railing(d, x0 - 10, x1 + 10, int(0.66 * h), int(0.82 * h))             # guard rail
    d.rectangle([0, h - 10, w, h], fill=(210, 202, 186))                          # floor band
    return im


def facade_ground(seed, w=512, h=683):
    """One bay of the ground floor (3.0 m x 4.0 m): rusticated pier and a shop front."""
    rng = np.random.default_rng(seed)
    im = _stone(w, h, seed, 46)
    d = ImageDraw.Draw(im)
    for y in range(0, h, 46):                                                      # rustication
        d.line([(0, y), (w, y)], fill=(170, 164, 150), width=4)
    sign = SIGNS[int(rng.integers(len(SIGNS)))]
    x0, x1 = int(0.12 * w), int(0.88 * w)
    d.rectangle([x0, int(0.10 * h), x1, int(0.22 * h)], fill=sign)                # fascia
    for k in range(0, x1 - x0 - 40, 28):
        d.rectangle([x0 + 20 + k, int(0.14 * h), x0 + 34 + k, int(0.18 * h)],
                    fill=(222, 206, 160))                                          # lettering
    gy0, gy1 = int(0.24 * h), int(0.96 * h)
    d.rectangle([x0, gy0, x1, gy1], fill=sign)
    for yy in range(gy0 + 8, gy1 - 6):
        t = (yy - gy0) / (gy1 - gy0)
        d.line([(x0 + 8, yy), (x1 - 8, yy)], fill=(int(40 + 50 * (1 - t)), int(46 + 50 * (1 - t)),
                                                   int(52 + 56 * (1 - t))))
    d.line([((x0 + x1) // 2, gy0), ((x0 + x1) // 2, gy1)], fill=sign, width=8)
    d.rectangle([0, h - 8, w, h], fill=(150, 146, 138))                           # plinth
    return im


def mansard(seed, w=512, h=256):
    """One bay of a zinc mansard (3.0 m x 1.5 m) with a dormer window."""
    a = np.full((h, w, 3), (128, 138, 148), dtype=np.float32)
    a += _noise(w, h, 6.0, seed)[..., None]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 32):
        d.line([(x, 0), (x, h)], fill=(102, 112, 122), width=3)                   # standing seams
    x0, x1, y0, y1 = int(0.36 * w), int(0.64 * w), int(0.12 * h), int(0.92 * h)
    d.polygon([(x0 - 16, y0 + 10), ((x0 + x1) // 2, y0 - 22), (x1 + 16, y0 + 10)],
              fill=(236, 230, 218))                                              # pediment
    d.rectangle([x0 - 12, y0 + 10, x1 + 12, y1], fill=(236, 230, 218))
    d.rectangle([x0, y0 + 20, x1, y1 - 8], fill=(58, 68, 80))
    d.line([((x0 + x1) // 2, y0 + 20), ((x0 + x1) // 2, y1 - 8)], fill=(236, 230, 218), width=4)
    return im


def zinc_roof(seed, w=512, h=512):
    a = np.full((h, w, 3), (122, 132, 142), dtype=np.float32)
    a += _noise(w, h, 7.0, seed)[..., None]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 40):
        d.line([(x, 0), (x, h)], fill=(96, 106, 116), width=3)
    return im


def _grey(im, lift=1.25):
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    y = a.mean(axis=2, keepdims=True)
    a = 0.15 * a + 0.85 * y                                                       # desaturate
    a = 255.0 - (255.0 - a) / lift                                                # lighten
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _clip_polyline(p, x0, x1, y0, y1):
    """The polyline cut to the box (segments clipped with Liang-Barsky), None if outside."""
    out = []
    for a, b in zip(p[:-1], p[1:]):
        d = b - a
        t0, t1 = 0.0, 1.0
        ok = True
        for q, r in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
            if abs(q) < 1e-12:
                if r < 0:
                    ok = False
                    break
                continue
            t = r / q
            if q < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if ok and t0 < t1:
            seg = [a + t0 * d, a + t1 * d]
            if out and np.linalg.norm(out[-1] - seg[0]) < 1e-6:
                out.append(seg[1])
            else:
                if out:
                    break                                  # keep the first inside piece
                out.extend(seg)
    return np.asarray(out) if len(out) >= 2 else None


def _jpeg(im, q=86):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    return b.getvalue()


class Glb:
    """Minimal glTF 2.0 binary writer: one mesh per material, embedded JPEG textures."""

    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors, self.images, self.textures = [], [], [], []
        self.materials, self.meshes, self.nodes = [], [], []

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin += b"\0"
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def _acc(self, arr, kind, comp, target):
        a = np.ascontiguousarray(arr)
        v = self._view(a.tobytes(), target)
        acc = {"bufferView": v, "componentType": comp, "count": int(len(a)), "type": kind}
        if kind == "VEC3" and comp == 5126:
            acc["min"], acc["max"] = a.min(axis=0).tolist(), a.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def material(self, name, jpeg, tint=(1.0, 1.0, 1.0), rough=0.85, metal=0.0):
        self.images.append({"bufferView": self._view(jpeg), "mimeType": "image/jpeg"})
        self.textures.append({"source": len(self.images) - 1, "sampler": 0})
        self.materials.append({"name": name, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": len(self.textures) - 1},
            "baseColorFactor": [tint[0], tint[1], tint[2], 1.0],
            "metallicFactor": metal, "roughnessFactor": rough}})
        return len(self.materials) - 1

    def mesh(self, name, mat, pos, nrm, uv, idx):
        """pos/nrm in the scene frame (z up): written Y-up as glTF wants (x, z, -y)."""
        P = np.asarray(pos, np.float32)[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
        N = np.asarray(nrm, np.float32)[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
        prim = {"attributes": {"POSITION": self._acc(P, "VEC3", 5126, 34962),
                               "NORMAL": self._acc(N, "VEC3", 5126, 34962),
                               "TEXCOORD_0": self._acc(np.asarray(uv, np.float32), "VEC2", 5126,
                                                       34962)},
                "indices": self._acc(np.asarray(idx, np.uint32).reshape(-1), "SCALAR", 5125,
                                     34963), "material": mat}
        self.meshes.append({"name": name, "primitives": [prim]})
        self.nodes.append({"mesh": len(self.meshes) - 1, "name": name})

    def write(self, path, credits):
        js = {"asset": {"version": "2.0", "generator": "Auralis tools/street_visual.py",
                        "copyright": credits},
              "scene": 0, "scenes": [{"nodes": list(range(len(self.nodes)))}],
              "nodes": self.nodes, "meshes": self.meshes, "materials": self.materials,
              "textures": self.textures, "images": self.images,
              "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
              "accessors": self.accessors, "bufferViews": self.views,
              "buffers": [{"byteLength": len(self.bin)}]}
        j = json.dumps(js, separators=(",", ":")).encode()
        j += b" " * ((4 - len(j) % 4) % 4)
        b = bytes(self.bin) + b"\0" * ((4 - len(self.bin) % 4) % 4)
        with open(path, "wb") as fh:
            fh.write(b"glTF" + struct.pack("<II", 2, 12 + 8 + len(j) + 8 + len(b)))
            fh.write(struct.pack("<I", len(j)) + b"JSON" + j)
            fh.write(struct.pack("<I", len(b)) + b"BIN\0" + b)


class Batch:
    def __init__(self):
        self.p, self.n, self.uv, self.i = [], [], [], []

    def quad(self, a, b, c, d, n, uva, uvb, uvc, uvd):
        k = len(self.p)
        self.p += [a, b, c, d]
        self.n += [n] * 4
        self.uv += [uva, uvb, uvc, uvd]
        self.i += [k, k + 1, k + 2, k, k + 2, k + 3]


def build(street, road_jpg, sidewalk_jpg, out, ground_floor=4.0, floor=3.1, roof_level=1.5):
    g = Glb()
    groups = {}

    def mat(key, maker, tint=(1, 1, 1), rough=0.85, metal=0.0):
        if key not in groups:
            groups[key] = (g.material(key, _jpeg(maker()), tint, rough, metal), Batch())
        return groups[key][1]
    for bi, b in enumerate(street["buildings"] + street.get("context", [])):
        poly = np.asarray(b["poly"], float)
        # ~0.8 of white reads as limestone under the realistic lighting. Brighter values
        # (0.87-0.98) wash the facades out.
        tint = 0.84 * np.array(STONE_TINTS[bi % len(STONE_TINTS)]) / 236.0
        seed = int(b["id"]) % 1000
        lv, rl, H = b["levels"], b["roof_levels"], b["height"]
        bands = [(0.0, min(ground_floor, H), "ground")]
        z = ground_floor
        for k in range(1, lv):                       # floors 1 .. lv-1 above the ground floor
            if z >= H - 0.05:
                break
            bands.append((z, min(z + floor, H), "balcony" if k in (2, 5) else "storey"))
            z += floor
        if rl > 0 and z < H - 0.05:
            bands.append((z, H, "mansard"))
        u = 0.0
        n = len(poly)
        for e in range(n):
            a2, c2 = poly[e], poly[(e + 1) % n]
            L = float(np.linalg.norm(c2 - a2))
            if L < 0.05:
                continue
            nrm = (float(c2[1] - a2[1]) / L, float(a2[0] - c2[0]) / L, 0.0)
            u0, u1 = u / BAY_M, (u + L) / BAY_M
            for z0, z1, kind in bands:
                if kind == "ground":
                    bt = mat("ground_%d" % (seed % 3), lambda s=seed: facade_ground(s % 3), tuple(tint))
                elif kind == "mansard":
                    bt = mat("mansard", lambda: mansard(7), rough=0.55, metal=0.35)
                else:
                    bt = mat("%s_%d" % (kind, bi % len(STONE_TINTS)),
                             lambda k=kind, s=seed: facade_storey(k == "balcony", s % 5),
                             tuple(tint))
                v1 = (z1 - z0) / (floor if kind in ("storey", "balcony") else
                                  ground_floor if kind == "ground" else roof_level)
                bt.quad((a2[0], a2[1], z0), (c2[0], c2[1], z0), (c2[0], c2[1], z1),
                        (a2[0], a2[1], z1), nrm, (u0, 1.0), (u1, 1.0), (u1, 1.0 - v1),
                        (u0, 1.0 - v1))
            u += L
        # flat roof: fan of the CCW outline from its centroid (display only)
        bt = mat("roof", lambda: zinc_roof(3), rough=0.5, metal=0.4)
        cx, cy = poly.mean(axis=0)
        for e in range(n):
            a2, c2 = poly[e], poly[(e + 1) % n]
            k = len(bt.p)
            bt.p += [(cx, cy, H), (a2[0], a2[1], H), (c2[0], c2[1], H)]
            bt.n += [(0, 0, 1)] * 3
            bt.uv += [(cx / 6, cy / 6), (a2[0] / 6, a2[1] / 6), (c2[0] / 6, c2[1] / 6)]
            bt.i += [k, k + 1, k + 2]
    (x0, x1), (y0, y1) = street.get("visual_box") or street["box"]
    # Paris sidewalks are light grey asphalt: the paving photo, desaturated and lightened
    side = mat("sidewalk", lambda: _grey(Image.open(sidewalk_jpg)), rough=0.9)
    side.quad((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0), (0, 0, 1),
              (x0 / 3, y0 / 3), (x1 / 3, y0 / 3), (x1 / 3, y1 / 3), (x0 / 3, y1 / 3))
    road = mat("road", lambda: Image.open(road_jpg), rough=0.95)
    for si, s in enumerate(street["streets"]):
        if s["highway"] in ("pedestrian", "footway", "living_street"):
            continue
        w = 3.2 * max(1, s["lanes"]) + (1.0 if s["highway"] in ("primary", "secondary") else 0.0)
        p = _clip_polyline(np.asarray(s["points"], float), x0, x1, y0, y1)
        if p is None:
            continue
        zr = 0.03 + 0.002 * si
        along = 0.0
        for k in range(len(p) - 1):
            a2, c2 = p[k], p[k + 1]
            L = float(np.linalg.norm(c2 - a2))
            if L < 0.05:
                continue
            t = (c2 - a2) / L
            q = np.array([-t[1], t[0]]) * w / 2
            road.quad((*(a2 - q), zr), (*(c2 - q), zr), (*(c2 + q), zr), (*(a2 + q), zr),
                      (0, 0, 1), (along / 6, 0), ((along + L) / 6, 0),
                      ((along + L) / 6, w / 6), (along / 6, w / 6))
            along += L
    for key, (m, bt) in groups.items():
        if bt.p:
            g.mesh(key, m, bt.p, bt.n, bt.uv, bt.i)
    g.write(out, "Buildings: (c) OpenStreetMap contributors, ODbL. Road and sidewalk textures: "
                 "Poly Haven, CC0. Facades: procedural (Auralis).")
    return len(groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("street_json")
    ap.add_argument("--road", required=True)
    ap.add_argument("--sidewalk", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = json.load(open(a.street_json, encoding="utf-8"))
    n = build(st, a.road, a.sidewalk, a.out, st["storeys"]["ground_floor_m"],
              st["storeys"]["floor_m"], st["storeys"]["roof_level_m"])
    import os
    print("%s: %d materials, %.1f MB" % (a.out, n, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
