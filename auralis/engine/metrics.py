# -*- coding: utf-8 -*-
"""Room-acoustics metrics from band echograms (ISO 3382 style) and an
EAX-like summary (DecayTime, DecayHFRatio, reflection / reverb delays)."""

from __future__ import annotations

import numpy as np

from .bands import CENTERS, LABELS


def events_to_hist(events, n_bins: int, bin_dt: float, n_bands: int):
    """Bin deterministic event energies (amps**2) into an echogram."""
    h = np.zeros((n_bands, n_bins), dtype=np.float64)
    for ev in events:
        k = int(ev.time / bin_dt)
        if 0 <= k < n_bins:
            h[:, k] += np.asarray(ev.amps, dtype=np.float64) ** 2
    return h


def schroeder_edc(e):
    """Backward-integrated energy decay curve, normalised to 1 at t=0."""
    edc = np.cumsum(e[::-1])[::-1]
    m = edc[0] if len(edc) and edc[0] > 0 else 1.0
    return edc / m


def _decay_time(edc_db, t, lo_db, hi_db):
    """Least-squares slope between lo_db and hi_db, extrapolated to -60."""
    below_lo = np.where(edc_db <= lo_db)[0]
    below_hi = np.where(edc_db <= hi_db)[0]
    if below_lo.size == 0 or below_hi.size == 0:
        return None
    i, j = below_lo[0], below_hi[0]
    if j - i < 3:
        return None
    A = np.vstack([t[i:j], np.ones(j - i)]).T
    slope, _ = np.linalg.lstsq(A, edc_db[i:j], rcond=None)[0]
    if slope >= -1e-9:
        return None
    return float(-60.0 / slope)


def band_metrics(e, bin_dt, t0_bin=0):
    """Per-band ISO 3382 metrics from one band echogram e (n_bins,)."""
    n = len(e)
    t = np.arange(n) * bin_dt
    total = float(e.sum())
    out = {"RT": None, "RT_method": None, "EDT": None,
           "C50": None, "C80": None, "D50": None, "Ts": None}
    if total <= 0:
        return out

    edc = schroeder_edc(e)
    db = 10.0 * np.log10(np.maximum(edc, 1e-12))
    rt = _decay_time(db, t, -5.0, -25.0)
    method = "T20"
    if rt is None:
        rt = _decay_time(db, t, -5.0, -35.0)
        method = "T30"
    out["RT"] = rt
    out["RT_method"] = method if rt is not None else None
    out["EDT"] = _decay_time(db, t, 0.0, -10.0)

    k50 = t0_bin + int(round(0.050 / bin_dt))
    k80 = t0_bin + int(round(0.080 / bin_dt))
    e50, l50 = float(e[:k50].sum()), float(e[k50:].sum())
    e80, l80 = float(e[:k80].sum()), float(e[k80:].sum())
    if l50 > 0 and e50 > 0:
        out["C50"] = 10.0 * np.log10(e50 / l50)
    if l80 > 0 and e80 > 0:
        out["C80"] = 10.0 * np.log10(e80 / l80)
    out["D50"] = e50 / total if total > 0 else None
    out["Ts"] = float((t * e).sum() / total)
    return out


def compute_metrics(hist_total, dirs, events, bin_dt):
    """Full report from hist_total (NB, n_bins), the directional sums
    dirs (n_bins, 3, may be None) and the deterministic events."""
    nb, n_bins = hist_total.shape
    direct = [ev for ev in events if ev.kind == "direct"]
    spec = sorted([ev for ev in events if ev.kind == "specular"],
                  key=lambda e: e.time)
    t_direct = direct[0].time if direct else 0.0
    t0_bin = int(t_direct / bin_dt)

    per_band = {}
    for b in range(nb):
        per_band[LABELS[b]] = band_metrics(hist_total[b], bin_dt, t0_bin)

    def rt_of(idx):
        vals = [per_band[LABELS[i]]["RT"] for i in idx]
        vals = [v for v in vals if v]
        return float(np.mean(vals)) if vals else None

    rt_mid = rt_of([2, 3])           # 500 Hz, 1 kHz
    rt_hf = rt_of([4, 5])            # 2, 4 kHz
    rt_lf = rt_of([0, 1])            # 125, 250 Hz

    # diffusion proxy: 1 - mean anisotropy of the directional echogram
    diffusion = None
    if dirs is not None:
        e_bin = hist_total.sum(axis=0)
        ok = e_bin > (e_bin.max() * 1e-4 if e_bin.max() > 0 else 0)
        if ok.any():
            aniso = np.linalg.norm(dirs[ok], axis=1) / np.maximum(e_bin[ok], 1e-30)
            diffusion = float(np.clip(1.0 - np.mean(np.clip(aniso, 0, 1)), 0, 1))

    eax = {
        "DecayTime_s": rt_mid,
        "DecayHFRatio": (rt_hf / rt_mid) if (rt_hf and rt_mid) else None,
        "DecayLFRatio": (rt_lf / rt_mid) if (rt_lf and rt_mid) else None,
        "ReflectionsDelay_s": (spec[0].time - t_direct) if spec else None,
        "LateReverbDelay_s": (spec[-1].time - t_direct) if spec else None,
        "Diffusion": diffusion,
        "DirectTime_s": t_direct,
    }
    return {"bands_hz": list(CENTERS), "per_band": per_band, "eax": eax}


DIR_WINDOW_BINS = 21


def smoothed_directions(hist, dirs, win=DIR_WINDOW_BINS):
    """(fraction (n_bins,), unit direction (n_bins, 3)) of the echogram's mean arrival.

    Smoothed over a sliding window of `win` bins. Per 1 ms bin, the mean of few
    deposits is biased towards 1 (0.20 per bin against 0.047 over 21 bins in a
    9 x 6 m room, decaying as 1/sqrt(window)), which makes the late field look far
    more directional than it is."""
    e = hist.sum(axis=0)
    n_bins = len(e)
    if dirs is None:
        return np.zeros(n_bins), np.tile([1.0, 0.0, 0.0], (n_bins, 1))
    k = np.ones(int(win))
    es = np.convolve(e, k, "same")
    ds = np.stack([np.convolve(dirs[:, i], k, "same") for i in range(3)], axis=1)
    dn = np.linalg.norm(ds, axis=1)
    frac = np.clip(dn / np.maximum(es, 1e-300), 0.0, 1.0)
    unit = np.tile([1.0, 0.0, 0.0], (n_bins, 1))
    ok = dn > 1e-12
    unit[ok] = ds[ok] / dn[ok, None]
    return frac, unit


def lateral_metrics(hist, dirs, events, bin_dt, src, rcv):
    """J_LF and L_J per band (ISO 3382-1) from the DIRECTIONAL energy of the
    geometrical solver: the figure-of-eight points sideways, its null towards the
    source (horizontal plane). Deterministic events weigh cos^2 of their exact
    direction; each stochastic bin is modelled as its mean direction for the
    fraction |mean| of its energy (smoothed_directions), isotropic (1/3) for the rest.
    hist: stochastic echogram (NB, n_bins), dirs: its directional sums (n_bins, 3).
    Returns (J_LF (NB,), L_J dB (NB,)); None per band where undefined."""
    to_src = np.asarray(src, float) - np.asarray(rcv, float)
    lat = np.array([-to_src[1], to_src[0], 0.0])
    if np.linalg.norm(lat) < 1e-9:            # source straight above/below
        lat = np.array([1.0, 0.0, 0.0])
    lat /= np.linalg.norm(lat)
    nb, n_bins = hist.shape
    t = (np.arange(n_bins) + 0.5) * bin_dt
    frac, unit = smoothed_directions(hist, dirs)
    kappa = frac * (unit @ lat) ** 2 + (1.0 - frac) / 3.0
    direct = [ev for ev in events if ev.kind == "direct"]
    t0 = direct[0].time if direct else 0.0
    w_omni_e = np.zeros(nb)       # 0-80 ms, omni
    w_lat_e = np.zeros(nb)        # 5-80 ms, figure-of-eight
    w_lat_l = np.zeros(nb)        # > 80 ms, figure-of-eight
    for ev in events:
        e = np.asarray(ev.amps, float) ** 2
        c2 = float(np.dot(ev.direction, lat)) ** 2
        dt = ev.time - t0
        if dt < 0.080:
            w_omni_e += e
            if dt >= 0.005:
                w_lat_e += e * c2
        else:
            w_lat_l += e * c2
    rel = t - t0
    early, lat_e, late = rel < 0.080, (rel >= 0.005) & (rel < 0.080), rel >= 0.080
    w_omni_e += hist[:, early].sum(axis=1)
    w_lat_e += (hist[:, lat_e] * kappa[lat_e]).sum(axis=1)
    w_lat_l += (hist[:, late] * kappa[late]).sum(axis=1)
    jlf = [float(a / b) if b > 0 else None for a, b in zip(w_lat_e, w_omni_e)]
    lj = [float(10.0 * np.log10(x) + 20.0) if x > 0 else None for x in w_lat_l]
    return jlf, lj


def format_report(metrics) -> str:
    """Human-readable text table."""
    lines = ["Band       RT       EDT     C50     C80     D50     Ts"]
    for lab in LABELS:
        m = metrics["per_band"][lab]
        def f(v, fmt="%6.2f"):
            return (fmt % v) if v is not None else "   --  "
        lines.append("%-8s %s%s %s %s %s %s %s" % (
            lab,
            f(m["RT"]), ("(%s)" % (m["RT_method"] or "  ")).ljust(6),
            f(m["EDT"]), f(m["C50"]), f(m["C80"]),
            f(m["D50"], "%6.3f"), f(m["Ts"], "%6.3f")))
    e = metrics["eax"]
    lines.append("")
    for k, v in e.items():
        lines.append("%-22s %s" % (k, ("%.4f" % v) if v is not None else "--"))
    if "spl_bands" in metrics:
        lines.append("")
        lines.append("SPL per band (dB)  " + "  ".join(
            "%5.1f" % v if v is not None else "  -- " for v in metrics["spl_bands"]))
        if metrics.get("spl_db") is not None:
            lines.append("Overall SPL %.1f dB / %.1f dB(A)    G_mid %.1f dB"
                         % (metrics["spl_db"], metrics["spl_dba"],
                            metrics["g_mid"]))
    return "\n".join(lines)
