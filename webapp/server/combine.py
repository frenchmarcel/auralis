# -*- coding: utf-8 -*-
"""Coherent sum of sources: pressure IRs added with a gain, a delay and a
polarity per source - a loudspeaker system, an ensemble played as one source."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from auralis.engine import rirmetrics as rm
from auralis.engine import speech
from auralis.engine.level import db_total, dba_total, dbc_total

MAX_DELAY_MS = 2000.0


def _shift(x, n):
    out = np.zeros((len(x) + n,) + x.shape[1:], dtype=np.float64)
    out[n:] = x
    return out


def combine(results_dir, meta, sources, receiver_id, settings):
    """sources = [{"source_id", "gain_db", "delay_ms", "polarity" (+1/-1)}]. Returns
    (key, {"metrics", "fs", ...}); the summed IRs are cached in results/combined/<key>/."""
    if not sources:
        raise ValueError("no source to combine")
    clean = []
    for s in sources:
        d = float(s.get("delay_ms", 0.0))
        if not 0.0 <= d <= MAX_DELAY_MS:
            raise ValueError("delay must be within 0-%.0f ms" % MAX_DELAY_MS)
        pol = int(s.get("polarity", 1))
        if pol not in (1, -1):
            raise ValueError("polarity must be 1 or -1")
        sid = str(s["source_id"])
        if "/" in sid or sid.startswith("."):
            raise KeyError(sid)
        clean.append({"source_id": sid, "gain_db": float(s.get("gain_db", 0.0)),
                      "delay_ms": d, "polarity": pol})
    fs = int(meta["fs"])
    key = hashlib.sha1(json.dumps([receiver_id, clean, settings.get("speech_effort"),
                                   settings.get("background_noise")]).encode()).hexdigest()[:12]
    out_dir = os.path.join(results_dir, "combined", key)
    info_p = os.path.join(out_dir, "info.json")
    if os.path.exists(info_p):
        with open(info_p) as fh:
            return key, json.load(fh)
    bin_sum, omni_sum = None, None
    for s in clean:
        base = os.path.join(results_dir, "irs", "%s__%s" % (s["source_id"], receiver_id))
        if not os.path.exists(base + ".npy"):
            raise KeyError("%s -> %s" % (s["source_id"], receiver_id))
        g = s["polarity"] * 10.0 ** (s["gain_db"] / 20.0)
        n = int(round(s["delay_ms"] * 1e-3 * fs))
        b = _shift(np.load(base + ".npy").astype(np.float64) * g, n)
        o = (_shift(np.load(base + "_ambi.npy")[:, 0].astype(np.float64) * g, n)
             if os.path.exists(base + "_ambi.npy") else b.mean(axis=1))
        bin_sum = b if bin_sum is None else _add(bin_sum, b)
        omni_sum = o if omni_sum is None else _add(omni_sum, o)
    m = rm.band_metrics(omni_sum, fs)
    met = {k: _clean_list(m[k]) for k in ("T20", "T30", "EDT", "C50", "C80", "D50")}
    met["Ts_ms"] = _clean_list(m["Ts"])
    # level: energy of the summed omni IR relative to each source's own (G, SPL known
    # per pair): the sum's band energy over the reference pair's band energy, in dB
    ref = next(p for p in meta["pairs"] if p["source_id"] == clean[0]["source_id"]
               and p["receiver_id"] == receiver_id)
    ref_o = _omni(results_dir, clean[0]["source_id"], receiver_id)
    e_sum = (rm.band_split(omni_sum, fs) ** 2).sum(axis=1)
    e_ref = (rm.band_split(ref_o, fs) ** 2).sum(axis=1)
    delta = 10.0 * np.log10(np.maximum(e_sum, 1e-300) / np.maximum(e_ref, 1e-300))
    g_ref = np.asarray([np.nan if v is None else v for v in ref["metrics"]["G_db"]], float)
    spl_ref = np.asarray([np.nan if v is None else v for v in ref["metrics"]["SPL_db"]], float)
    G, spl = g_ref + delta, spl_ref + delta
    met["G_db"], met["SPL_db"] = _clean_list(G), _clean_list(spl)
    signal = speech.speech_levels_1m(settings.get("speech_effort", "normal")) + G - 20.0
    st = speech.sti(speech.mtf_from_ir(omni_sum, fs), signal,
                    speech.noise_levels(settings.get("background_noise", "none")))
    met.update({"STI": round(st["sti"], 6), "MTI": _clean_list(st["mti"]),
                "SPL_dbz": round(db_total(spl), 3), "SPL_dba": round(dba_total(spl), 3),
                "SPL_dbc": round(dbc_total(spl), 3)})
    if ref["metrics"].get("wave_only"):
        from .sim import mask_wave_only
        met = mask_wave_only(met, float(meta.get("transition_hz") or 0.0))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ir.npy"), bin_sum.astype(np.float32))
    np.save(os.path.join(out_dir, "omni.npy"), omni_sum.astype(np.float32))
    info = {"key": key, "receiver_id": receiver_id, "sources": clean, "fs": fs, "metrics": met}
    with open(info_p, "w") as fh:
        json.dump(info, fh)
    return key, info


def _omni(results_dir, sid, rid):
    base = os.path.join(results_dir, "irs", "%s__%s" % (sid, rid))
    if os.path.exists(base + "_ambi.npy"):
        return np.load(base + "_ambi.npy")[:, 0].astype(np.float64)
    return np.load(base + ".npy").astype(np.float64).mean(axis=1)


def _add(a, b):
    n = max(len(a), len(b))
    out = np.zeros((n,) + a.shape[1:])
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def _clean_list(v):
    return [None if (x is None or not np.isfinite(x)) else round(float(x), 6) for x in v]
