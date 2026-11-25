#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the source/receiver pairs of a BRAS SOFA for bras_bench.
Requires h5py; prints the pairs as JSON or injects them via --into.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _to_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return "".join(_to_str(x) for x in v.ravel().tolist())
    return str(v)


def _sph_to_cart(pos):
    az = np.radians(pos[:, 0])
    el = np.radians(pos[:, 1])
    r = pos[:, 2]
    return np.stack([r * np.cos(el) * np.cos(az),
                     r * np.cos(el) * np.sin(az),
                     r * np.sin(el)], axis=1)


def _normalize(arr, n_ir):
    """Reduce a SOFA position field to (n, 3) whatever its layout.
    Returns (pos, per_measurement, description)."""
    a = np.array(arr, dtype=np.float64)
    if a.ndim == 3:
        if a.shape[1] == 3 and a.shape[2] == n_ir:
            return a[0].T, True, "(*,3,M) per measurement"
        if a.shape[1] == 3 and a.shape[2] == 1:
            return a[:, :, 0], a.shape[0] == n_ir, "(E,3,1)"
        a = a.reshape(a.shape[0], -1)
    if a.ndim == 2:
        if a.shape[0] == 3 and a.shape[1] == n_ir and n_ir != 3:
            return a.T, True, "(3,M) per measurement"
        if a.shape[1] >= 3:
            return a[:, :3], a.shape[0] == n_ir, \
                "(%d,3)" % a.shape[0]
    if a.ndim == 1 and a.size >= 3:
        return a[None, :3], n_ir == 1, "(3,)"
    raise ValueError("unexpected position shape: %s"
                     % (np.array(arr).shape,))


def _ids_int(f, key, n):
    """Numeric indices (1-based) from EmitterID/ReceiverID, or None if
    the field is absent or not numeric."""
    if key not in f:
        return None
    try:
        v = np.array(f[key], dtype=np.float64).ravel()[:n]
        return np.clip(np.round(v).astype(int) - 1, 0, None)
    except (ValueError, TypeError):
        return None


def _resolve(f, per_m_key, table_key, ids, n_ir):
    """Per-IR positions: the per-measurement field if informative, else
    Emitter/ReceiverPosition (per measurement, or table indexed by ID)."""
    for key in (per_m_key, table_key):
        if key not in f:
            continue
        ds = f[key]
        pos, per_ir, desc = _normalize(ds, n_ir)
        typ = _to_str(ds.attrs.get("Type", b"cartesian")).lower()
        if typ.startswith("spher"):
            pos = _sph_to_cart(pos)
        informative = (len(np.unique(np.round(pos, 6), axis=0)) > 1
                       or float(np.abs(pos).max()) > 1e-6)
        if not informative:
            continue
        if per_ir:
            return pos[:n_ir], "%s %s" % (key, desc)
        if len(pos) == 1:
            return np.repeat(pos, n_ir, axis=0), "%s %s" % (key, desc)
        if ids is not None:
            idx = np.clip(ids, 0, len(pos) - 1)
            return pos[idx], "%s %s indexed by ID" % (key, desc)
    raise ValueError("positions not found or all degenerate "
                     "(%s / %s) -- run --inspect"
                     % (per_m_key, table_key))


def _labels(f, key, n, prefix):
    """LSx / MPx labels from EmitterID/ReceiverID, stored either as
    strings ('LS1') or as numeric indices (then prefixed)."""
    if key not in f:
        return ["%s%d" % (prefix, i + 1) for i in range(n)]
    raw = f[key]
    out = []
    for i in range(min(n, raw.shape[0])):
        lab = _to_str(np.array(raw[i])).strip().strip("\x00")
        try:
            lab = "%s%d" % (prefix, int(round(float(lab))))
        except ValueError:
            pass
        out.append(lab)
    while len(out) < n:
        out.append("%s%d" % (prefix, len(out) + 1))
    return out


def inspect(path):
    import h5py
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if hasattr(obj, "shape"):
                attrs = {k: _to_str(np.array(v)) for k, v in
                         obj.attrs.items() if k in ("Type", "Units")}
                try:
                    head = np.array(obj).ravel()[:6]
                    head = np.round(head.astype(np.float64), 3).tolist()
                except (ValueError, TypeError):
                    head = "(text)"
                print("%-24s %-14s %s %s"
                      % (name, obj.shape, attrs or "", head))
        f.visititems(walk)


def _read_pairs_grid(f, pattern=None):
    """Rooms CR1-4: Data.IR is M x R x E x N; the pairs are the R x E
    grid, positions from Receiver/EmitterPosition."""
    M, R, E, N = f["Data.IR"].shape
    fs = float(np.array(f["Data.SamplingRate"]).ravel()[0]) \
        if "Data.SamplingRate" in f else None
    rpos, r_from = _resolve(f, "ListenerPosition",
                            "ReceiverPosition", None, R)
    epos, e_from = _resolve(f, "SourcePosition",
                            "EmitterPosition", None, E)
    print("grid %dx%d | source: %s | receiver: %s"
          % (E, R, e_from, r_from), file=sys.stderr)
    e_lab = _labels(f, "EmitterID", E, "LS")
    r_lab = _labels(f, "ReceiverID", R, "MP")
    pairs = []
    for e in range(E):
        for r in range(R):
            p = {"label": "%s-%s" % (e_lab[e], r_lab[r]),
                 "source": [round(float(x), 4) for x in epos[e]],
                 "receiver": [round(float(x), 4)
                              for x in rpos[r]]}
            if pattern:
                p["reference_wav"] = pattern.format(
                    emitter=e_lab[e], receiver=r_lab[r])
            pairs.append(p)
    return pairs, fs


def read_pairs(path, pattern=None):
    import h5py
    with h5py.File(path, "r") as f:
        shp = f["Data.IR"].shape
        if len(shp) == 4:
            return _read_pairs_grid(f, pattern)
        n_ir = shp[0]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0]) \
            if "Data.SamplingRate" in f else None
        e_ids = _ids_int(f, "EmitterID", n_ir)
        r_ids = _ids_int(f, "ReceiverID", n_ir)
        src, src_from = _resolve(f, "SourcePosition",
                                 "EmitterPosition", e_ids, n_ir)
        lis, lis_from = _resolve(f, "ListenerPosition",
                                 "ReceiverPosition", r_ids, n_ir)
        print("source positions: %s | receiver: %s"
              % (src_from, lis_from), file=sys.stderr)
        emit = _labels(f, "EmitterID", n_ir, "LS")
        recv = _labels(f, "ReceiverID", n_ir, "MP")
    pairs = []
    for i in range(n_ir):
        p = {"label": "%s-%s" % (emit[i], recv[i]),
             "source": [round(float(x), 4) for x in src[i]],
             "receiver": [round(float(x), 4) for x in lis[i]]}
        if pattern:
            p["reference_wav"] = pattern.format(emitter=emit[i],
                                                receiver=recv[i])
        pairs.append(p)
    return pairs, fs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sofa", help="BRAS .sofa file")
    ap.add_argument("--into", default=None, metavar="MANIFEST",
                    help="inject the pairs directly into this JSON "
                         "manifest (replaces 'pairs') instead of "
                         "printing them")
    ap.add_argument("--inspect", action="store_true",
                    help="list every SOFA field (names, shapes, first "
                         "values) and exit")
    ap.add_argument("--events-mirror-z", type=float, default=None,
                    help="add the reflection mode to each pair "
                         "('events': {'mirror_z': Z}) for the RS1 "
                         "scenes (reflecting plane at z=Z)")
    ap.add_argument("--pattern", default=None,
                    help="reference_wav template, with {emitter} and "
                         "{receiver} (e.g. RIRs/wav/RS1_RIR_Rigid_"
                         "{emitter}_{receiver}.wav)")
    args = ap.parse_args(argv)
    if args.inspect:
        inspect(args.sofa)
        return 0
    pairs, fs = read_pairs(args.sofa, args.pattern)
    if args.events_mirror_z is not None:
        for p in pairs:
            p["events"] = {"mirror_z": args.events_mirror_z}
    if fs:
        print('"fs": %d,' % int(fs), file=sys.stderr)
    if args.into:
        import os
        with open(args.into, "r", encoding="utf-8") as fh:
            man = json.load(fh)
        rrp = man.get("events_defaults", {}).get(
            "rigid_reference_pattern")
        if rrp and args.pattern == rrp:
            print("ERROR: the reference template (--pattern) is "
                  "IDENTICAL to the manifest rigid reference (%s). "
                  "You are pointing at the SOFA/WAVs of the RIGID "
                  "scene: use those of the manifest scene (e.g. "
                  "Absorbing)." % rrp, file=sys.stderr)
            return 1
        man["pairs"] = pairs
        with open(args.into, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, ensure_ascii=False)
        base = os.path.dirname(os.path.abspath(args.into))
        manquants = [p["reference_wav"] for p in pairs
                     if p.get("reference_wav") and not os.path.exists(
                         os.path.join(base, p["reference_wav"]))]
        print("%d pairs injected into %s: %s"
              % (len(pairs), args.into,
                 ", ".join(p["label"] for p in pairs)),
              file=sys.stderr)
        if manquants:
            print("WARNING: %d reference(s) not found -- wrong SOFA "
                  "or wrong template? %s"
                  % (len(manquants), manquants[:3]), file=sys.stderr)
            return 1
        return 0
    print(json.dumps(pairs, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
