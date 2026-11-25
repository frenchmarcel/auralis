# -*- coding: utf-8 -*-
"""Result exports: an xlsx workbook (settings + parameters) and a zip of the IRs."""
from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np

from auralis.engine.wavio import write_wav

BAND_LABELS = ["125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz", "8 kHz"]
PER_BAND = [("T20", "T20 (s)"), ("T30", "T30 (s)"), ("EDT", "EDT (s)"), ("C50", "C50 (dB)"),
            ("C80", "C80 (dB)"), ("D50", "D50"), ("Ts_ms", "Ts (ms)"), ("SPL_db", "SPL (dB)"),
            ("G_db", "G (dB)"), ("MTI", "MTI"), ("JLF", "J_LF"), ("LJ_db", "L_J (dB)")]
SINGLE = [("STI", "STI"), ("SPL_dbz", "SPL (dB Z)"), ("SPL_dba", "SPL (dB A)"),
          ("SPL_dbc", "SPL (dB C)")]


def _names(sim):
    return ({s["id"]: s["name"] for s in sim["sources"]},
            {r["id"]: r["name"] for r in sim["receivers"]})


def xlsx(sim, meta, geometry_name=""):
    """Workbook bytes: Settings, Parameters (one row per pair and parameter), Single
    numbers, Points."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    src_n, rcv_n = _names(sim)
    wb = Workbook()
    ws = wb.active
    ws.title = "Settings"
    rows = [("Simulation", sim["name"]), ("Geometry", geometry_name),
            ("Engine used", meta.get("engine_used")),
            ("Transition frequency (Hz)", meta.get("transition_hz")),
            ("Computation time (s)", meta.get("duration_s")),
            ("Results stale", bool(meta.get("stale")))]
    rows += [("settings." + k, json.dumps(v) if isinstance(v, (list, dict)) else v)
             for k, v in sorted(meta.get("settings", {}).items())]
    for r in rows:
        ws.append(list(r))
    ws.column_dimensions["A"].width = 32
    bold = Font(bold=True)

    ws = wb.create_sheet("Parameters")
    ws.append(["Source", "Receiver", "Parameter"] + BAND_LABELS)
    for p in meta["pairs"]:
        for key, label in PER_BAND:
            vals = p["metrics"].get(key)
            if vals is not None:
                ws.append([src_n.get(p["source_id"], p["source_id"]),
                           rcv_n.get(p["receiver_id"], p["receiver_id"]), label] + list(vals))
    ws = wb.create_sheet("Single numbers")
    ws.append(["Source", "Receiver"] + [lab for _k, lab in SINGLE])
    for p in meta["pairs"]:
        ws.append([src_n.get(p["source_id"], p["source_id"]),
                   rcv_n.get(p["receiver_id"], p["receiver_id"])]
                  + [p["metrics"].get(k) for k, _lab in SINGLE])
    ws = wb.create_sheet("Points")
    ws.append(["Kind", "Id", "Name", "x (m)", "y (m)", "z (m)", "Azimuth (deg)",
               "Elevation (deg)", "Directivity", "Gain (dB)"])
    for kind, items in (("source", sim["sources"]), ("receiver", sim["receivers"])):
        for it in items:
            ws.append([kind, it["id"], it["name"]] + list(it["position"])
                      + [it.get("azimuth_deg"), it.get("elevation_deg"),
                         it.get("directivity", ""), it.get("gain_db", "")])
    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = bold
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def ir_zip(sim, meta, results_dir):
    """Zip bytes: one binaural float32 WAV per pair (shared ir_scale, so relative levels
    hold), the parameters as JSON and a README."""
    src_n, rcv_n = _names(sim)
    buf = io.BytesIO()
    scale = float(meta.get("ir_scale", 1.0))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in meta["pairs"]:
            path = os.path.join(results_dir, "irs", "%s__%s.npy" % (p["source_id"],
                                                                   p["receiver_id"]))
            if not os.path.exists(path):
                continue
            stem = "%s__%s" % (_safe(src_n.get(p["source_id"], p["source_id"])),
                               _safe(rcv_n.get(p["receiver_id"], p["receiver_id"])))
            files = [("binaural", np.load(path) * scale)]
            ap = path[:-4] + "_ambi.npy"
            if os.path.exists(ap):
                ambi = np.load(ap) * scale
                files += [("omni", ambi[:, 0]), ("ambix", ambi)]
            for kind, data in files:
                tmp = io.BytesIO()
                _wav_bytes(tmp, data, int(meta["fs"]))
                z.writestr("%s_%s.wav" % (stem, kind), tmp.getvalue())
        z.writestr("parameters.json", json.dumps(meta, indent=1))
        z.writestr("README.txt",
                   "Auralis impulse responses - %s\n\n"
                   "Per source/receiver pair, float32 WAV at %d Hz:\n"
                   "  *_binaural.wav  2 channels (left, right) for headphones\n"
                   "  *_omni.wav      1 channel, omnidirectional pressure (the parameters are\n"
                   "                  computed on it)\n"
                   "  *_ambix.wav     9 channels, 2nd-order ambisonics, ACN order, SN3D\n"
                   "                  (AmbiX), world axes: x, y, z of the model, z up\n"
                   "All files share one scale factor (%.6g), so their relative levels are those\n"
                   "of the simulation. parameters.json holds the room-acoustic parameters.\n"
                   % (sim["name"], int(meta["fs"]), scale))
    return buf.getvalue()


def _wav_bytes(buf, data, fs):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        write_wav(path, np.asarray(data, dtype=np.float32), fs)
        with open(path, "rb") as fh:
            buf.write(fh.read())
    finally:
        os.remove(path)


def _safe(name):
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return keep.strip("_") or "x"
