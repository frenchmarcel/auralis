# -*- coding: utf-8 -*-
"""Reader for Odeon room models (.Par): points, surfaces (Surf / LSurf), layers,
constants and expressions, NumbOffSet, MTranslate/MReset, and Odeon's mirror
convention (negative numbers = mirrored about the XZ plane). The parametric
part of the language (For loops, Cylinder, RevSurf, UCS...) is refused loudly."""
from __future__ import annotations

import ast
import math
import operator
import re

import numpy as np

UNSUPPORTED = {"for", "end", "cylinder", "revsurf", "ucs", "countpt", "countsurf", "msurf",
               "box", "dome", "elevsurf", "rotatesurf", "mrotatex", "mrotatey", "mrotatez",
               "mscale", "copysurf", "mirrorsurf", "hemisphere", "sphere"}
IGNORED = {"blockbegin", "blockend", "sketchupsafe", "numbering", "layer"}
FUNCS = {"sind": lambda x: math.sin(math.radians(x)), "cosd": lambda x: math.cos(math.radians(x)),
         "tand": lambda x: math.tan(math.radians(x)),
         "arcsind": lambda x: math.degrees(math.asin(x)), "arccosd": lambda x: math.degrees(math.acos(x)),
         "arctand": lambda x: math.degrees(math.atan(x)),
         "sin": math.sin, "cos": math.cos, "tan": math.tan, "arcsin": math.asin, "arccos": math.acos,
         "arctan": math.atan, "sqrt": math.sqrt, "sqr": lambda x: x * x, "abs": abs,
         "round": round, "trunc": math.trunc, "int": math.trunc, "ln": math.log, "exp": math.exp}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


def _text(raw):
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", "replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


class _Eval:
    """Arithmetic with Odeon constants and functions only (no Python eval)."""

    def __init__(self, consts):
        self.consts = consts

    def __call__(self, expr):
        expr = expr.strip().replace("^", "**")
        node = ast.parse(expr, mode="eval").body
        return float(self._ev(node))

    def _ev(self, n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](self._ev(n.left), self._ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = self._ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Name):
            key = n.id.lower()
            if key == "pi":
                return math.pi
            if key in self.consts:
                return self.consts[key]
            raise ValueError("unknown constant '%s'" % n.id)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.lower() in FUNCS:
            return FUNCS[n.func.id.lower()](*[self._ev(a) for a in n.args])
        raise ValueError("unsupported expression: %s" % ast.dump(n)[:60])


def _split_exprs(tokens, n, ev):
    """Split whitespace tokens into n expressions (expressions may contain spaces)."""
    if len(tokens) == n:
        return [ev(t) for t in tokens]
    for cut in _cuts(len(tokens), n):
        parts = [" ".join(tokens[a:b]) for a, b in zip((0,) + cut, cut + (len(tokens),))]
        try:
            return [ev(p) for p in parts]
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            continue
    raise ValueError("cannot read %d values from: %s" % (n, " ".join(tokens)))


def _cuts(m, n):
    if n == 1:
        yield ()
        return
    for i in range(1, m - n + 2):
        for rest in _cuts(m - i, n - 1):
            yield (i,) + tuple(i + r for r in rest)


def _triangulate(poly):
    """Ear clipping of a planar polygon (n, 3) -> index triplets; handles concave
    polygons and Odeon's zero-width 'keyhole' bridges."""
    n = len(poly)
    if n < 3:
        return []
    normal = np.zeros(3)
    for i in range(n):                                   # Newell normal
        a, b = poly[i], poly[(i + 1) % n]
        normal += np.cross(a, b)
    if np.linalg.norm(normal) < 1e-12:
        return []
    drop = int(np.argmax(np.abs(normal)))
    p2 = np.delete(poly, drop, axis=1)
    sign = 1.0 if normal[drop] > 0 else -1.0
    if drop == 1:
        sign = -sign                                     # (x, z) keeps handedness reversed
    idx = list(range(n))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = p2[i0], p2[i1], p2[i2]
            area = sign * cross(a, b, c)
            if area <= 1e-12:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = p2[j]
                if np.allclose(p, a) or np.allclose(p, b) or np.allclose(p, c):
                    continue
                d1, d2, d3 = sign * cross(a, b, p), sign * cross(b, c, p), sign * cross(c, a, p)
                if d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                clipped = True
                break
        if not clipped:
            # degenerate remainder (collinear points): drop the flattest vertex
            areas = [abs(cross(p2[idx[k - 1]], p2[idx[k]], p2[idx[(k + 1) % len(idx)]]))
                     for k in range(len(idx))]
            idx.pop(int(np.argmin(areas)))
    if len(idx) == 3 and abs(cross(*(p2[i] for i in idx))) > 1e-12:
        tris.append(tuple(idx))
    return tris


def read_par(path):
    """Odeon .Par -> (verts (nv,3), faces (nf,3), face_layer idx, layer names, report)."""
    with open(path, "rb") as fh:
        text = _text(fh.read())
    used = {l.split(";", 1)[0].split()[0].lower() for l in text.splitlines()
            if l.split(";", 1)[0].split()}
    if used & UNSUPPORTED:
        raise ValueError("this Odeon model is built with parametric commands not supported yet "
                         "(%s): export it from Odeon in another format, or use a simpler model"
                         % ", ".join(sorted(used & UNSUPPORTED)))
    consts, pts, surfs = {}, {}, []
    offset, trans = 0, np.zeros(3)
    ev = _Eval(consts)
    unsupported, ignored = set(), []
    abs_refs = False                    # NumbOffSet ... PtAbsRef: surfaces use absolute point numbers
    current = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        try:
            line = raw.split(";", 1)[0].strip()
            if not line or line.startswith(("#", ":", "'")):
                continue
            tok = line.split()
            key = tok[0].lower()
            if re.match(r"^[-+]?\d+$", tok[0]) and current is not None:
                current["refs"].extend(int(t) for t in tok if re.match(r"^[-+]?\d+$", t))
                continue
            current = None
            if key == "const":
                consts[tok[1].lower()] = ev(" ".join(tok[2:]))
            elif key == "pt":
                num = int(ev(tok[1]))
                p = np.array(_split_exprs(tok[2:], 3, ev)) + trans
                pts[num + offset] = p
            elif key in ("surf", "lsurf"):
                num = int(tok[1])
                name = " ".join(tok[3:] if key == "lsurf" else tok[2:]).strip()
                current = {"num": num, "name": name or "Surface %d" % abs(num),
                           "refs": [], "offset": 0 if abs_refs else offset}
                surfs.append(current)
            elif key == "numboffset":
                abs_refs = any(t.lower() == "ptabsref" for t in tok[1:])
                arg = " ".join(t for t in tok[1:] if t.lower() != "ptabsref")
                if arg.lower() == "auto":
                    top = max([abs(k) for k in pts] or [0])
                    offset = int(math.ceil((top + 1) / 1000.0) * 1000)
                else:
                    consts["numboffset"] = offset
                    offset = int(ev(arg))
            elif key == "mtranslate":
                trans = trans + np.array(_split_exprs(tok[1:], 3, ev))
            elif key == "mreset":
                trans = np.zeros(3)
            elif key in IGNORED:
                continue
            elif key in UNSUPPORTED:
                unsupported.add(tok[0])
            else:
                ignored.append(tok[0])
        except (ValueError, SyntaxError, ZeroDivisionError, IndexError) as exc:
            raise ValueError("line %d: %s  [%s]" % (lineno, exc, raw.strip()[:70]))
    if unsupported:
        raise ValueError("this Odeon model uses parametric commands not supported yet (%s): "
                         "export it from Odeon in another format, or simplify it"
                         % ", ".join(sorted(unsupported)))

    fallbacks = [0]

    def point(ref, off, mirror):
        k = abs(ref) + off
        if k not in pts and off and abs(ref) in pts:
            k = abs(ref)                        # a point defined before the offset: counted, reported
            fallbacks[0] += 1
        if k not in pts:
            raise ValueError("surface refers to undefined point %d" % ref)
        p = pts[k].copy()
        if (ref < 0) != mirror:
            p[1] = -p[1]
        return p

    verts, faces, flay, names, fsurf = [], [], [], [], []
    missing = []
    for s in surfs:
        lname = s["name"]
        if lname not in names:
            names.append(lname)
        lk = names.index(lname)
        for mirror in ((False, True) if s["num"] < 0 else (False,)):
            try:
                poly = np.array([point(r, s["offset"], mirror) for r in s["refs"]])
            except ValueError as exc:
                missing.append(str(exc))
                continue
            for a, b, c in _triangulate(poly):
                base = len(verts)
                verts.extend([poly[a], poly[b], poly[c]])
                faces.append([base, base + 1, base + 2])
                flay.append(lk)
                fsurf.append(-s["num"] if mirror else s["num"])     # Odeon number of this copy
    if missing:
        raise ValueError("%d surface(s) refer to undefined points (first: %s)" % (len(missing), missing[0]))
    if not faces:
        raise ValueError("no surface found in the Odeon model")
    report = {"points": len(pts), "surfaces": len(surfs), "layers": len(names),
              "constants": len(consts), "offset_fallbacks": fallbacks[0],
              "ignored_lines": len(ignored), "ignored_words": sorted(set(ignored))[:12],
              "face_surface": np.array(fsurf, dtype=np.int64)}
    return (np.array(verts), np.array(faces, dtype=np.int64), np.array(flay, dtype=np.int32),
            names, report)


def read_mat(path):
    """Odeon .Mat (classic binary layout): per-surface (material index, scattering)
    and the room's material table [(library id, 8 alpha 63 Hz-8 kHz, name)].
    Returns None for layouts this reader does not know (newer Odeon versions)."""
    import struct
    with open(path, "rb") as fh:
        m = fh.read()
    n = struct.unpack("<i", m[:4])[0]
    if not 0 < n < 100000 or 4 + 16 * n > len(m):
        return None
    surf = {}
    for i in range(n):
        sid, midx, scat, _flag = struct.unpack("<iifi", m[4 + 16 * i: 20 + 16 * i])
        surf[sid] = (midx, scat)
    tail = m[4 + 16 * n:]
    k = struct.unpack("<i", tail[:4])[0]
    for width in (90, 80):                      # name field width differs between versions
        size = 4 + 32 + 1 + width
        if not 0 < k < 1000 or 4 + k * size > len(tail):
            continue
        table = []
        for i in range(k):
            r = tail[4 + i * size: 4 + (i + 1) * size]
            alpha = struct.unpack("<8f", r[4:36])
            length = r[36]
            name = r[37:37 + length]
            if length > width or not all(0 <= a <= 1.5 for a in alpha) or any(c < 9 for c in name):
                break
            table.append((struct.unpack("<i", r[:4])[0], list(alpha),
                          name.decode("latin-1").split("\t")[0].strip()))
        if len(table) == k:
            return {"surfaces": surf, "materials": table}
    return None
