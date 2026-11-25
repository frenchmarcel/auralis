# -*- coding: utf-8 -*-
"""Source directivities: engine presets, imports (engine JSON, CSV, AES69 SOFA, Odeon .SO8)
and the local Odeon patterns (read in place, never copied). The engine's model is
axisymmetric: imports are averaged in rings around the axis and the lost asymmetry is reported.
CLF .CF2 has no public specification, so it is not read."""
from __future__ import annotations

import json
import os
import re

import numpy as np

from auralis.engine.bands import CENTERS, N_BANDS
from auralis.engine.directivity import (DIRECTIVITY_NAMES, Directivity,
                                        get_directivity)

N_THETA = 181
PRESET_LABELS = {"omni": "Omnidirectional", "speaker_2way": "Loudspeaker (2-way)",
                 "figure8": "Figure-of-eight"}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ODEON_ARCHIVES = os.path.join(REPO, "data_odeon", "anechoic", "raw")
ODEON_PREFIX = "odeon-"
ODEON_LICENSE = "Odeon A/S - research licence, local only, never redistributed"
# instrument of an ensemble track (lower-case name) -> Odeon pattern; strict: an
# instrument without its own pattern (double bass, tuba, harp, percussion) stays omni
ODEON_INSTRUMENTS = [
    (r"viola|^vla_|_alto$|^alto", "odeon-viola"),
    (r"violin|^vl_", "odeon-violin"),
    (r"cello|^vc_", "odeon-cello"),
    (r"flute|^fl_", "odeon-flute"),
    (r"oboe|^ob_", "odeon-oboe"),
    (r"clarinet|^cl_", "odeon-clarinet"),
    (r"bassoon|^bn_", "odeon-bassoon"),
    (r"horn|^hrn_", "odeon-french-horn"),
    (r"trumpet|^tp$|^trp_", "odeon-trumpet"),
    (r"trombone", "odeon-trombone"),
    (r"guitar|_git[lr]$", "odeon-guitar"),
    (r"sax", "odeon-alto-saxophone"),
    (r"_voc$|singer|tenor|soprano|motetus|triplum|choir", "odeon-singer"),
]


def _dir(root):
    d = os.path.join(root, "directivities")
    os.makedirs(d, exist_ok=True)
    return d


def _path(root, did):
    if not re.fullmatch(r"custom-[0-9a-f]{6,12}", did or ""):
        raise KeyError(did)
    return os.path.join(_dir(root), did + ".json")


def known(root, name):
    return name in DIRECTIVITY_NAMES or name in odeon_patterns() or (
        str(name).startswith("custom-") and os.path.exists(_path(root, name)))


def load(root, did):
    if str(did).startswith(ODEON_PREFIX):
        rec = odeon_patterns().get(did)
        if rec is None:
            raise KeyError(did)
        return rec
    p = _path(root, did)
    if not os.path.exists(p):
        raise KeyError(did)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_so8(data):
    """Odeon .SO8 pattern -> (table (NB, N_THETA) energy re on-axis, asymmetry dB,
    on-axis SPL at 1 m per band or None). Layout derived from the files
    (power-normalised patterns integrate to 1 within 0.4 %, talkers radiate forward,
    the French horn backward): a 16-byte header, then 705 records of 8 octave levels
    (63 Hz - 8 kHz, dB): the total power level, a second record (not used), and a
    37 x 19 grid - roll angle about the source axis 0..360 deg by 10 (360 = 0), polar
    angle 180 - 10 j from the axis (j = 18 on axis). 63 Hz is dropped (engine bands)."""
    v = np.frombuffer(data[16:], dtype="<f4")
    if len(v) != 705 * 8:
        raise ValueError(".SO8: %d values, 5640 expected" % len(v))
    r = v.reshape(705, 8).astype(np.float64)
    grid = r[2:].reshape(37, 19, 8)[:36, :, 1:]               # (roll, j, 125 Hz-8 kHz)
    theta = np.broadcast_to(180.0 - 10.0 * np.arange(19), (36, 19)).reshape(-1)
    e = 10.0 ** ((grid - r[0, 1:]) / 10.0).reshape(-1, N_BANDS)
    table, asym = _ring_table(theta, e)
    # absolute files (talkers) carry levels: the total power level is far from 0 dB
    spl = grid[:, 18].mean(axis=0) if np.abs(r[0]).max() > 1.0 else None
    return table, asym, spl


_odeon_cache = {}


def odeon_patterns():
    """{id: record} of the distinct .SO8 patterns of the local Odeon archives, read in
    place (research licence: never copied into the store or a delivery). Empty when
    data_odeon is absent."""
    if "all" in _odeon_cache:
        return _odeon_cache["all"]
    out = {}
    if os.path.isdir(ODEON_ARCHIVES):
        import glob
        import zipfile
        for f in sorted(glob.glob(os.path.join(ODEON_ARCHIVES, "**", "*.OdeonZipArchive"),
                                  recursive=True)):
            try:
                z = zipfile.ZipFile(f)
            except zipfile.BadZipFile:
                print("[Auralis] WARNING: unreadable Odeon archive %s" % f, flush=True)
                continue
            for n in z.namelist():
                base = n.split("/")[-1]
                if not base.lower().endswith(".so8"):
                    continue
                stem = re.sub(r"_NATURAL$", "", base[:-4], flags=re.I)
                did = ODEON_PREFIX + stem.lower().replace("_", "-")
                if did in out:
                    continue
                try:
                    table, asym, spl = read_so8(z.read(n))
                except ValueError as exc:
                    print("[Auralis] WARNING: %s/%s skipped: %s" % (os.path.basename(f), base, exc),
                          flush=True)
                    continue
                notes = ["Odeon pattern, averaged in rings about the source axis; 63 Hz "
                         "not used (the engine starts at 125 Hz)"]
                if asym > 1.0:
                    notes.append("not axisymmetric: up to %.1f dB between directions at the "
                                 "same angle from the axis; the engine uses their mean" % asym)
                out[did] = {"id": did, "name": stem.replace("_", " ").capitalize() + " (Odeon)",
                            "source": "%s / %s" % (os.path.basename(f), base),
                            "method": "Odeon .SO8 (10 deg grid), averaged in rings about the axis",
                            "notes": notes, "asymmetry_db": round(asym, 2), "license": ODEON_LICENSE,
                            "on_axis_spl_1m_db": None if spl is None else [round(float(x), 2) for x in spl],
                            "table": np.round(np.maximum(table, 0.0), 6).tolist()}
    _odeon_cache["all"] = out
    return out


def instrument_pattern(track_name):
    """Odeon pattern id for an ensemble track, or None (no pattern of its own, or no data)."""
    t = str(track_name).lower()
    pats = odeon_patterns()
    for rx, did in ODEON_INSTRUMENTS:
        if re.search(rx, t):
            return did if did in pats else None
    return None


def engine_directivity(root, name):
    """None (omni), a preset name, or a Directivity for an imported one."""
    if not name or name == "omni":
        return None
    if name in DIRECTIVITY_NAMES:
        return name
    d = load(root, name)
    return Directivity(np.asarray(d["table"], float), name=d["name"])


def on_axis_spl(root, name, default=94.0):
    """Source level per band at 1 m on axis, when the imported data carries it."""
    if name and str(name).startswith(("custom-", ODEON_PREFIX)):
        v = load(root, name).get("on_axis_spl_1m_db")
        if v:
            return np.asarray(v, float)
    return default


def summary(table):
    """Directivity index per band and a polar subsample (every 5 degrees, dB)."""
    d = Directivity(np.asarray(table, float))
    th = np.arange(0, 181, 5)
    q = d.energy(np.cos(np.radians(th)))                   # (37, NB)
    return {"di_db": np.round(d.directivity_index_db(), 2).tolist(),
            "polar": {"theta_deg": th.tolist(),
                      "db": np.round(10 * np.log10(np.maximum(q.T, 1e-6)), 2).tolist()}}


def listing(root):
    out = []
    for n in DIRECTIVITY_NAMES:
        out.append(dict({"id": n, "name": PRESET_LABELS.get(n, n.replace("_", " ").capitalize()),
                         "kind": "preset", "on_axis_spl_1m_db": None},
                        **summary(get_directivity(n).table)))
    for f in sorted(os.listdir(_dir(root))):
        if f.endswith(".json"):
            d = load(root, f[:-5])
            out.append(dict({k: d.get(k) for k in ("id", "name", "source", "method", "notes",
                                                    "on_axis_spl_1m_db", "asymmetry_db")},
                            kind="imported", **summary(d["table"])))
    for d in odeon_patterns().values():
        out.append(dict({k: d.get(k) for k in ("id", "name", "source", "method", "notes",
                                                "on_axis_spl_1m_db", "asymmetry_db", "license")},
                        kind="odeon_local", **summary(d["table"])))
    return out


def _ring_table(theta_deg, energy):
    """(NB, N_THETA) energy table from samples at angles theta (any order): mean per 1
    degree ring, linear interpolation across empty rings, on-axis normalised to 1."""
    t = np.clip(np.asarray(theta_deg, float), 0.0, 180.0)
    e = np.asarray(energy, float)                           # (M, NB)
    idx = np.round(t).astype(int)
    table = np.full((N_BANDS, N_THETA), np.nan)
    spread = []
    for k in np.unique(idx):
        sel = idx == k
        table[:, k] = e[sel].mean(axis=0)
        if sel.sum() > 1:
            db = 10 * np.log10(np.maximum(e[sel], 1e-12))
            spread.append(float(np.max(db.max(axis=0) - db.min(axis=0))))
    grid = np.arange(N_THETA)
    for b in range(N_BANDS):
        ok = np.isfinite(table[b])
        table[b] = np.interp(grid, grid[ok], table[b, ok])
    table /= np.maximum(table[:, :1], 1e-30)
    return table, (max(spread) if spread else 0.0)


def import_file(root, path, name, original_name):
    ext = os.path.splitext(original_name)[1].lower()
    notes = []
    spl = None
    asym = 0.0
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        t = np.asarray(d["table"], float)
        if t.ndim == 1:
            t = np.broadcast_to(t, (N_BANDS, len(t)))
        if t.shape[0] != N_BANDS:
            raise ValueError("table: %d bands expected (125 Hz - 8 kHz)" % N_BANDS)
        theta = np.linspace(0.0, 180.0, t.shape[1])
        table = np.stack([np.interp(np.arange(N_THETA), theta, t[b]) for b in range(N_BANDS)])
        spl = d.get("on_axis_spl_1m_db")
        method = "engine table (energy vs angle)"
    elif ext == ".csv":
        rows = [r for r in (line.strip() for line in open(path, encoding="utf-8-sig"))
                if r and not r.startswith("#")]
        head = [h.strip().lower() for h in rows[0].split(",")]
        if len(head) != N_BANDS + 1 or not head[0].startswith("theta"):
            raise ValueError("CSV header: theta_deg,125,250,500,1000,2000,4000,8000")
        data = np.array([[float(x) for x in r.split(",")] for r in rows[1:]])
        e = 10.0 ** (data[:, 1:] / 10.0)
        table, asym = _ring_table(data[:, 0], e)
        method = "CSV of octave levels (dB) vs angle from the axis"
    elif ext == ".sofa":
        table, asym, notes = _read_sofa(path)
        method = "AES69 SOFA FreeFieldDirectivityTF, averaged in rings about the axis"
    elif ext == ".so8":
        with open(path, "rb") as fh:
            table, asym, spl = read_so8(fh.read())
        notes.append("63 Hz not used (the engine starts at 125 Hz)")
        method = "Odeon .SO8 (10 deg grid), averaged in rings about the axis"
    else:
        raise ValueError("supported: .json (engine table), .csv, .sofa (AES69), .so8 (Odeon)")
    if asym > 1.0:
        notes.append("not axisymmetric: up to %.1f dB between directions at the same angle "
                     "from the axis; the engine uses their mean" % asym)
    did = "custom-" + os.urandom(4).hex()
    rec = {"id": did, "name": name or os.path.splitext(original_name)[0],
           "source": original_name, "method": method, "notes": notes,
           "asymmetry_db": round(asym, 2),
           "on_axis_spl_1m_db": [round(float(x), 2) for x in spl] if spl is not None else None,
           "table": np.round(np.maximum(table, 0.0), 6).tolist()}
    with open(_path(root, did), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return rec


def delete(root, did):
    os.remove(_path(root, did))


def _read_sofa(path):
    import h5py
    with h5py.File(path, "r") as f:
        conv = f.attrs.get("SOFAConventions", b"")
        conv = conv.decode() if isinstance(conv, bytes) else str(conv)
        if "Directivity" not in conv:
            raise ValueError("SOFA convention '%s': FreeFieldDirectivityTF expected" % conv)
        re_ = np.asarray(f["Data.Real"])
        im = np.asarray(f["Data.Imag"])
        freqs = np.asarray(f["N"]).reshape(-1)
        pos = np.asarray(f["ReceiverPosition"])
        ptype = f["ReceiverPosition"].attrs.get("Type", b"spherical")
        ptype = ptype.decode() if isinstance(ptype, bytes) else str(ptype)
        view = np.asarray(f["SourceView"]).reshape(-1, 3)[0] if "SourceView" in f else \
            np.array([0.0, 0.0, 1.0])
        vtype = f["SourceView"].attrs.get("Type", b"spherical") if "SourceView" in f else b"spherical"
        vtype = vtype.decode() if isinstance(vtype, bytes) else str(vtype)
    mag2 = re_ ** 2 + im ** 2                                # (M, R, N)
    if mag2.ndim == 3:
        mag2 = mag2.mean(axis=1)
    pos = pos.reshape(-1, pos.shape[-1])[:, :3]
    if pos.shape[0] != mag2.shape[0]:
        pos = np.broadcast_to(pos[0], (mag2.shape[0], 3))

    def cart(p, kind):
        if "spher" in kind:
            az, el = np.radians(p[..., 0]), np.radians(p[..., 1])
            return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
        v = np.asarray(p, float)
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    d = cart(pos, ptype)
    ax = cart(view, vtype)
    theta = np.degrees(np.arccos(np.clip(d @ ax, -1.0, 1.0)))
    e = np.zeros((mag2.shape[0], N_BANDS))
    for b, fc in enumerate(CENTERS):
        sel = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
        if not sel.any():
            sel = np.array([np.argmin(np.abs(freqs - fc))])
        e[:, b] = mag2[:, sel].mean(axis=1)
    table, asym = _ring_table(theta, e)
    notes = ["%d directions, %d frequencies (%.0f-%.0f Hz)" % (len(theta), len(freqs),
                                                                freqs.min(), freqs.max())]
    return table, asym, notes
