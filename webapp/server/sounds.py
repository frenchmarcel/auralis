# -*- coding: utf-8 -*-
"""Odeon anechoic library (local licence), ensembles (one source per player,
seated like on a stage) and the server-side binaural render of a mix."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import numpy as np

from auralis.engine.bands import band_masks
from auralis.engine.hrtf import interaural_coherence
from auralis.engine.level import dba_total
from auralis.engine.wavio import read_wav, write_wav

from . import directivities as dirlib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ODEON_SOUNDS_INDEX = os.path.join(REPO, "data_odeon", "anechoic", "index.json")

# Which tracks play together: explicit, because some Odeon packs mix several
# songs or add bar noises to a trio.
ENSEMBLES = [
    ("mozart40", "Mozart – Symphony 40 (15 players)", "Mozart_40thSymphony", None),
    ("beethoven7", "Beethoven – Symphony 7, 1st mvt (21 players)", "Beethoven_Symphony7_1stmov", None),
    ("beethoven8", "Beethoven – Symphony 8 (59 players)", "Beethoven_Symphony8_mov1_2_4", None),
    ("mahler1", "Mahler – Symphony 1, 4th mvt (39 players)", "Mahler_Symphony1_4thmov", None),
    ("bruckner8", "Bruckner – Symphony 8, 2nd mvt (47 players)", "Bruckner_Symphony8_2ndmov", None),
    ("puccini", "Puccini – O mio babbino caro (63 tracks)", "GPuccini_OMioBabbinoCaro", None),
    ("verdi", "Verdi – Di tale amor (52 tracks)", "GVerdi_DiTaleAmorCheDirsi", None),
    ("donizetti", "Donizetti – Come Paride vezzoso (53 tracks)", "GDonizetti_ComeParideVezzoso", None),
    ("machaut", "Machaut – Kyrie (4 voices)", "GuillaumedeMachaut_Kyrie", None),
    ("vivaldi", "Vivaldi – Summer, string quartet", "AVivaldi_Summer", "4Seasons_"),
    ("reinhardt", "Reinhardt – Minor Swing, jazz trio", "DReinhardt_MinorSwing", "Reinhardt_"),
    ("blues_a", "Blues A (voice + 2 guitars)", "BluesFlamenco", "BluesA_"),
    ("hellofaguy", "Hell of a Guy (voice + 2 guitars)", "BluesFlamenco", "HellOfAGuy_"),
]

# section: (keywords, ring radius m, angle range deg from audience axis
# (negative = audience's left), height above floor m)
SECTIONS = [
    ("voice", r"voc|voice|tenor|soprano|contratenor|motetus|triplum|singer|baritone|mezzo",
     1.2, (-60, 60), 1.6),
    ("violin1", r"(1st|first)[ _]?violin|vl_1_|violin1|^vl1", 2.0, (-85, -45), 1.1),
    ("violin2", r"(2nd|second)[ _]?violin|vl_2_|violin2|^vl2", 2.0, (-45, -8), 1.1),
    ("viola", r"viola|vla|\balto\b|_alto|^alto", 2.0, (8, 45), 1.1),
    ("cello", r"cello|vc", 2.0, (45, 85), 1.0),
    ("bass", r"double ?bass|c ?bass|contrabass|^db|doublebass", 4.5, (60, 88), 1.2),
    ("flute", r"flute|^fl|piccolo|oboe|^ob", 4.5, (-18, 0), 1.1),
    ("reed", r"clarinet|^cl|bassoon|^bn|fagott", 5.5, (-5, 18), 1.1),
    ("horn", r"horn|^hn|^hrn|^cor", 5.5, (-55, -20), 1.1),
    ("brass", r"trumpet|^tp|^trp|^tr\b|trombone|^tb|tuba", 6.5, (15, 45), 1.1),
    ("timpani", r"timpani|^timp|perc|drum|cymbal", 6.5, (-12, 12), 1.0),
    ("harp", r"harp", 4.0, (-88, -65), 1.1),
    ("choir", r"choir|chorus", 7.5, (-60, 60), 1.8),
    ("guitar", r"guitar|git", 1.5, (-45, 45), 1.0),
]
SPACING = 1.1           # m between players on an arc
ROW_STEP = 1.0          # m between successive arcs of a section
MIN_GAP = 0.8           # m, closest two players may sit


def odeon_sounds():
    """{id: entry} of the Odeon recordings; a piece shipped with several rooms
    is listed once. Empty when data_odeon/anechoic is absent."""
    if not os.path.exists(ODEON_SOUNDS_INDEX):
        return {}
    with open(ODEON_SOUNDS_INDEX) as fh:
        index = json.load(fh)
    out, seen = {}, set()
    for e in index:
        work = e["piece"].split("-", 1)[1] if e["group"] == "concerts" and "-" in e["piece"] else e["piece"]
        if (work, e["track"]) in seen:
            continue
        seen.add((work, e["track"]))
        title = work.replace("_", " – ")
        out[e["id"]] = dict(e, work=work, kind="odeon_local", name="%s · %s" % (title, e["track"]))
    return out


def ensembles():
    sounds = odeon_sounds()
    out = []
    for eid, title, work, prefix in ENSEMBLES:
        tracks = [s for s in sounds.values()
                  if s["work"] == work and (prefix is None or s["track"].startswith(prefix))]
        if not tracks:
            continue
        out.append({"id": eid, "title": title, "n_tracks": len(tracks),
                    "duration_s": max(t["duration_s"] or 0 for t in tracks),
                    "credit": tracks[0]["credit"], "license": tracks[0]["license"],
                    "tracks": [{"sound_id": t["id"], "name": t["track"]} for t in tracks]})
    return out


def section_of(track):
    t = track.lower().replace("4seasons_", "").replace("reinhardt_", "")
    for name, pattern, *_ in SECTIONS:
        if re.search(pattern, t):
            return name
    return None


def layout(tracks, width_m, depth_m):
    """Stage-local positions (u = distance upstage of the conductor, v = towards the
    audience's left, h = height) for each track, scaled to fit width x depth."""
    groups = {}
    for i, t in enumerate(tracks):
        groups.setdefault(section_of(t["name"]), []).append(i)
    solo_band = len(tracks) <= 6
    pos = [None] * len(tracks)
    spec = {s[0]: s[2:] for s in SECTIONS}
    for sec, idx in groups.items():
        r0, (a0, a1), h = spec.get(sec, (3.0, (-25, 25), 1.1))
        if solo_band:
            r0, (a0, a1) = 1.5, (-50, 50)
        left = list(idx)
        r = r0
        while left:
            arc = math.radians(a1 - a0) * r
            k = max(1, min(len(left), int(arc // SPACING)))
            row, left = left[:k], left[k:]
            for j, i in enumerate(row):
                # centre of the j-th of k equal bins: never on a boundary shared
                # with the neighbouring section
                a = math.radians(a0 + (a1 - a0) * (j + 0.5) / k)
                pos[i] = (r * math.cos(a), -r * math.sin(a), h)
            r += ROW_STEP
    P = np.array(pos)
    if not solo_band:
        # overflow rows of a section can land on another section's arc: push any
        # player closer than MIN_GAP to an already placed one further upstage
        for i in range(1, len(P)):
            for _ in range(40):
                d = np.hypot(*(P[:i, :2] - P[i, :2]).T)
                if d.min() >= MIN_GAP:
                    break
                r = math.hypot(P[i, 0], P[i, 1])
                P[i, :2] *= (r + 0.5) / r
    if solo_band:
        # every player in one arc whatever its section: spread them evenly
        n = len(P)
        for j in range(n):
            a = math.radians(-50 + 100 * j / max(1, n - 1))
            P[j, :2] = 1.5 * math.cos(a), -1.5 * math.sin(a)
    ext_u = max(P[:, 0].max(), 1e-6)
    ext_v = max(np.ptp(P[:, 1]), 1e-6)
    s = min(1.0, depth_m / (ext_u + 0.5), width_m / (ext_v + 0.5))
    P[:, :2] *= s
    return P, s


def ensemble_sources(ens, center, facing_deg, width_m, depth_m, id_prefix):
    """Sources for an ensemble: conductor at `center` (on the floor), audience
    towards `facing_deg` (azimuth, CCW from +X); each player gets the Odeon pattern of
    its instrument when there is one (facing the audience), else omni."""
    P, scale = layout(ens["tracks"], width_m, depth_m)
    phi = math.radians(facing_deg)
    a = np.array([math.cos(phi), math.sin(phi)])           # towards the audience
    left = np.array([math.sin(phi), -math.cos(phi)])        # audience's left
    c = np.asarray(center, dtype=float)
    out = []
    for k, (t, (u, v, h)) in enumerate(zip(ens["tracks"], P)):
        xy = c[:2] - u * a + v * left
        out.append({"id": "%s%d" % (id_prefix, k + 1), "name": t["name"],
                    "position": [round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(c[2] + h), 3)],
                    "directivity": dirlib.instrument_pattern(t["name"]) or "omni",
                    "azimuth_deg": round(facing_deg % 360, 1),
                    "elevation_deg": 0.0, "gain_db": 0.0, "sound_id": t["sound_id"],
                    "ensemble": ens["title"]})
    return out, scale


def _resample(x, fs_in, fs_out):
    """Band-limited FFT resampling (x: (n,) or (n, c))."""
    if fs_in == fs_out:
        return x
    n_out = int(round(len(x) * fs_out / fs_in))
    X = np.fft.rfft(x, axis=0)
    m = n_out // 2 + 1
    Y = np.zeros((m,) + X.shape[1:], dtype=complex)
    k = min(m, len(X))
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n_out, axis=0) * (n_out / len(x))


def load_sound(path):
    x, fs = read_wav(path)
    x = np.asarray(x, dtype=np.float64)
    return (x.mean(axis=1) if x.ndim == 2 else x), int(fs)


TURN_FRAME_S = 0.02     # s, energy frames of the pause detector
TURN_DB = -35.0         # dB re the 95th percentile frame: below = pause
TURN_PAUSE_S = 0.25     # s, shortest pause that ends an utterance
TURN_PAD_S = 0.05       # s kept on each side of an utterance (inside the pause)
TURN_FADE_S = 0.01      # s, raised-cosine fade at each cut


def utterances(x, fs):
    """[(start, end)] samples of the utterances of a speech track, split at its pauses
    (checked on Odeon's Harvard lists: one sentence each, 1.4-3.4 s)."""
    h = max(1, int(round(TURN_FRAME_S * fs)))
    n = len(x) // h
    if n == 0:
        return [(0, len(x))] if len(x) else []
    e = (np.asarray(x[:n * h], dtype=np.float64).reshape(n, h) ** 2).mean(axis=1)
    ref = float(np.percentile(e, 95))
    if ref <= 0:
        return []
    act = 10.0 * np.log10(np.maximum(e / ref, 1e-12)) > TURN_DB
    min_pause, pad = int(math.ceil(TURN_PAUSE_S / TURN_FRAME_S)), int(round(TURN_PAD_S * fs))
    out, k = [], 0
    while k < n:
        if not act[k]:
            k += 1
            continue
        j, quiet = k, 0
        while j < n and quiet < min_pause:
            quiet = quiet + 1 if not act[j] else 0
            j += 1
        end = (j - quiet) * h
        out.append((max(0, k * h - pad), min(len(x), end + pad)))
        k = j
    return out


P_REF = 20e-6           # Pa


def active_octave_levels(x, fs):
    """Octave levels (dB re 1, 125 Hz-8 kHz, engine bands) of a track over its
    utterances only: the pauses would lower a speech level. None if it is silent."""
    segs = utterances(x, fs)
    if not segs:
        return None
    y = np.concatenate([np.asarray(x[a:b], dtype=np.float64) for a, b in segs])
    Y = np.fft.rfft(y)
    masks = band_masks(np.fft.rfftfreq(len(y), 1.0 / fs), fs)
    ms = 2.0 * (np.abs(Y) ** 2 * masks ** 2).sum(axis=1) / float(len(y)) ** 2   # Parseval
    return 10.0 * np.log10(np.maximum(ms, 1e-30))


def speech_gain(x, fs, dba_1m):
    """Gain that makes track x the on-axis free-field pressure (Pa) at 1 m of speech at
    dba_1m dB(A), measured over its utterances (the renders' IRs are re 1 m on axis)."""
    lv = active_octave_levels(x, fs)
    if lv is None:
        return 0.0
    return P_REF * 10.0 ** ((dba_1m - dba_total(lv)) / 20.0)


def diffuse_noise(levels_db, n, fs, rng, speed_of_sound=343.0):
    """(n, 2) diffuse-field noise in Pa at the two ears: octave levels levels_db (dB SPL,
    125 Hz-8 kHz) at each ear, the diffuse-field interaural coherence in each band."""
    masks = band_masks(np.fft.rfftfreq(n, 1.0 / fs), fs)
    rho = interaural_coherence(speed_of_sound)
    out = np.zeros((n, 2))
    for b, lv in enumerate(np.asarray(levels_db, dtype=np.float64)):
        nl = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * masks[b], n)
        ny = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * masks[b], n)
        nr = rho[b] * nl + math.sqrt(max(0.0, 1.0 - rho[b] ** 2)) * ny
        p = P_REF * 10.0 ** (lv / 20.0)
        out[:, 0] += nl * (p / max(float(np.sqrt(np.mean(nl ** 2))), 1e-30))
        out[:, 1] += nr * (p / max(float(np.sqrt(np.mean(nr ** 2))), 1e-30))
    return out


def loop_to(x, n, fs, xfade_s=0.5):
    """Track x repeated up to n samples, each join an equal-power cross-fade of xfade_s
    (continuous sounds: traffic, crowds, ambiences shorter than the render)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) >= n:
        return x[:n].copy()
    f = int(min(xfade_s * fs, len(x) // 4))
    if f < 2:
        return np.resize(x, n)
    t = np.linspace(0.0, np.pi / 2, f)
    fin, fout = np.sin(t), np.cos(t)
    out = x.copy()
    while len(out) < n:
        head = out[-f:] * fout + x[:f] * fin
        out = np.concatenate([out[:-f], head, x[f:]])
    return out[:n]


def _octave_levels(x, fs):
    X = np.fft.rfft(np.asarray(x, dtype=np.float64))
    masks = band_masks(np.fft.rfftfreq(len(x), 1.0 / fs), fs)
    ms = 2.0 * (np.abs(X) ** 2 * masks ** 2).sum(axis=1) / float(len(x)) ** 2
    return 10.0 * np.log10(np.maximum(ms, 1e-30))


def diffuse_recording(x, n, fs, dba, speed_of_sound=343.0):
    """(n, 2) diffuse bed in Pa at the two ears from a mono recording x: per octave band, the
    left ear hears the recording, the right one the same passage mixed with another passage
    (half its length later) so that the two ears have the diffuse-field coherence of
    diffuse_noise; scaled to dba dB(A) at each ear. The recording keeps its own grain."""
    a = loop_to(x, n, fs)
    shift = max(1, len(x) // 2)
    b = loop_to(np.roll(np.asarray(x, dtype=np.float64), -shift), n, fs)
    masks = band_masks(np.fft.rfftfreq(n, 1.0 / fs), fs)
    rho = interaural_coherence(speed_of_sound)
    A, B = np.fft.rfft(a), np.fft.rfft(b)
    out = np.zeros((n, 2))
    for k in range(len(masks)):
        la = np.fft.irfft(A * masks[k], n)
        lb = np.fft.irfft(B * masks[k], n)
        out[:, 0] += la
        out[:, 1] += rho[k] * la + math.sqrt(max(0.0, 1.0 - rho[k] ** 2)) * lb
    lv = _octave_levels(out[:, 0], fs)
    g = P_REF * 10.0 ** ((float(dba) - dba_total(lv)) / 20.0)
    return out * g


def take_turns(tracks, fs, n, gap_s=0.5, start_s=0.5):
    """Conversation from speech tracks (all at fs): they speak in turn, one utterance
    each, `gap_s` apart, every track silent while another speaks. A track that runs
    out of utterances drops out; the first turn that would not end before n ends the
    conversation (no sentence cut short). Returns the (n,) tracks and the turns
    [(track, t0, t1)]."""
    cuts = [utterances(x, fs) for x in tracks]
    out = [np.zeros(n) for _ in tracks]
    nf = max(1, int(round(TURN_FADE_S * fs)))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    turns, nxt, pos = [], [0] * len(tracks), int(round(start_s * fs))
    while any(nxt[i] < len(c) for i, c in enumerate(cuts)):
        for i, x in enumerate(tracks):
            if nxt[i] >= len(cuts[i]):
                continue
            a, b = cuts[i][nxt[i]]
            nxt[i] += 1
            if pos + (b - a) > n:
                return out, turns
            seg = np.array(x[a:b], dtype=np.float64)
            if len(seg) > 2 * nf:
                seg[:nf] *= ramp
                seg[-nf:] *= ramp[::-1]
            out[i][pos:pos + len(seg)] = seg
            turns.append((i, pos / fs, (pos + len(seg)) / fs))
            pos += len(seg) + int(round(gap_s * fs))
    return out, turns


def render(strips, irs, ir_fs, out_dir, master_db=0.0, tag="", pcm16=False, progress=None):
    """Mix: strips = [(sound_path, gain_db, label, credit)], irs = matching (n, ch) IRs at
    ir_fs (2 ch binaural, or 9 ch ambisonic). Spectra are accumulated so each player
    costs one track FFT. progress(frac, stage) returning False cancels (KeyboardInterrupt).
    Returns (wav path, credits path, info)."""
    def tick(frac, stage):
        if progress is not None and progress(frac, stage) is False:
            raise KeyboardInterrupt("render cancelled")

    tick(0.0, "loading %d recordings" % len(strips))
    tracks = [load_sound(p) for p, *_ in strips]
    fs = tracks[0][1]
    tracks = [_resample(x, f, fs) for x, f in tracks]
    hs = [_resample(np.asarray(h, dtype=np.float64), ir_fs, fs) for h in irs]
    n = max(len(x) for x in tracks) + max(len(h) for h in hs)
    nfft = 1 << int(math.ceil(math.log2(n)))
    acc = np.zeros((nfft // 2 + 1, hs[0].shape[1]), dtype=complex)
    for k, (x, h, (_p, gain_db, *_rest)) in enumerate(zip(tracks, hs, strips)):
        tick(0.1 + 0.8 * k / len(strips), "track %d / %d" % (k + 1, len(strips)))
        acc += np.fft.rfft(x * 10 ** (gain_db / 20.0), nfft)[:, None] * np.fft.rfft(h, nfft, axis=0)
    tick(0.9, "writing the file")
    y = np.fft.irfft(acc, nfft, axis=0)[:n] * 10 ** (master_db / 20.0)
    peak = float(np.abs(y).max()) or 1.0
    norm = 0.891 / peak                                   # -1 dBFS
    y *= norm
    key = hashlib.sha1(json.dumps([(s[0], s[1]) for s in strips] + [master_db, tag, pcm16]).encode()
                       ).hexdigest()[:10] + ("_" + tag if tag else "")
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "render_%s.wav" % key)
    write_wav(wav, y.astype(np.float32), fs, float32=not pcm16)    # normalised: 16 bit is safe
    credits = sorted({s[3] for s in strips if s[3]})
    txt = os.path.join(out_dir, "render_%s_credits.txt" % key)
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("Auralis %s render - %d sources\n\n" % (
            "2nd-order ambisonic (ACN/SN3D, 9 channels)" if tag == "ambix" else "binaural",
            len(strips)))
        if credits:
            fh.write("Credits (required by the recordings' licences):\n")
            for c in credits:
                fh.write("  - %s\n" % c)
            fh.write("\n")
        fh.write("Sources and recordings:\n")
        for path, gain_db, label, _c in strips:
            fh.write("  - %s: %s (%+.1f dB)\n" % (label, os.path.basename(path), gain_db))
    return wav, txt, {"fs": fs, "duration_s": round(n / fs, 2),
                      "normalization_db": round(20 * math.log10(norm), 2), "credits": credits}
