#!/usr/bin/env python3
"""CR2 blind listening test, measured against simulated BRIRs (FABIAN at MP6, QSC-K8 at
LS3-LS7). Usage: listening_test.py [-o out_dir] [--quick]"""
import argparse
import base64
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, ".."))
from auralis.engine import rirmetrics as rm  # noqa: E402
from auralis.engine.directivity import Directivity  # noqa: E402
from auralis.engine.hrtf_sofa import min_phase_fir  # noqa: E402
from auralis.engine.pipeline import (ReceiverDef, SourceDef, bake_pair, wave_ear_half,  # noqa: E402
                                     source_low_bands)
from auralis.engine.project import build_config, build_scene  # noqa: E402
from auralis.engine.wavio import read_wav, write_wav  # noqa: E402
from bras_directivity import octave_table, read_mps  # noqa: E402

BRAS = os.path.join(REPO, "data_bras")
SOFA = os.path.join(BRAS, "1 Scene descriptions", "CR2 small room (seminar room)", "BRIRs",
                    "CR2_BRIRs.sofa")
AUDIO = os.path.join(BRAS, "4 Additional data", "Audio")
QSC_CSV = os.path.join(BRAS, "2 Source and receiver descriptions", "QSC K8",
                       "QSCK8_1x1_64442_MPS_front_pole.csv")
QSC_JSON = os.path.join(HERE, "qsc_k8_directivity.json")
HRTF = os.path.join(REPO, "data_hrtf", "fabian", "fabian_measured_hato0.npz")
MANIFEST = os.path.join(HERE, "cr2_dodecahedron_fitted31_dir.json")
PAGE = os.path.join(HERE, "listening_test_page.html")
ODEON = os.path.join(REPO, "data_odeon", "anechoic")
# program: label, file, [(channel, speaker)], start (s), source length (s, None = all),
# total length with the tail (s). "bras" renders as renderAuralizations.m (quartet
# channels 1-4 on LS3-LS6, noise pulse on LS7); "voice" uses Odeon recordings, under a
# LOCAL research licence: that page must never be redistributed
PROGRAM_SETS = {
    "bras": {"trials": (5, 6), "credit": "Anechoic recordings: BRAS (CC BY-SA 4.0).",
             "programs": {
                 "quartet": ("String quartet",
                             os.path.join(AUDIO, "AnechoicAudio", "StringQuartet_4channel.wav"),
                             [(0, "LS3"), (1, "LS4"), (2, "LS5"), (3, "LS6")], 0.0, None, 13.0),
                 "noise": ("Noise pulse",
                           os.path.join(AUDIO, "AnechoicAudio", "NoisePulsesCR2_1channel.wav"),
                           [(0, "LS7")], 0.0, None, 5.0)}},
    "voice": {"trials": (2, 3),
              "credit": "Anechoic recordings: Odeon A/S, local research licence. "
                        "Do not redistribute this page.",
              "programs": {
                  "speech_fr": ("Speech (French)",
                                os.path.join(ODEON, "concerts", "Canteen-MiscSpeech",
                                             "Le_Corbeau_et_le_Renard_french_.wav"),
                                [(0, "LS7")], 0.0, 8.8, 11.3),
                  "words_en": ("Short sentences (male voice)",
                               os.path.join(ODEON, "misc",
                                            "Harvard_Word_List_Speech_Inteligibility_Test_Recordings",
                                            "SI_Harvard_Word_Lists_Male.wav"),
                               [(0, "LS4")], 2.0, 7.0, 9.5),
                  "claps": ("Single claps",
                            os.path.join(ODEON, "concerts", "Argos", "Clapping_slow_3_.wav"),
                            [(0, "LS3")], 0.0, None, 8.8),
                  "hammer": ("Hammer on steel",
                             os.path.join(ODEON, "misc", "NoiseSignals", "Hammer_on_steel.WAV"),
                             [(0, "LS6")], 0.0, 4.0, 6.5)}},
}
OCTAVES = [125, 250, 500, 1000, 2000, 4000, 8000]


def measured():
    """Measured BRIRs at the neutral head orientation: ({LS: (n, 2) left first},
    {LS: emitter position}, head position, fs)."""
    import h5py
    with h5py.File(SOFA, "r") as f:
        m = int(np.argmin(np.abs(np.array(f["ListenerView"])[:, 0])))
        ir = np.array(f["Data.IR"][m], dtype=np.float64)                # (2, E, N)
        ids = np.array(f["EmitterID"]).ravel().astype(int)
        em = np.array(f["EmitterPosition"], dtype=np.float64)[:, :, 0]
        head = np.array(f["ListenerPosition"], dtype=np.float64)[0]
        ears_y = np.array(f["ReceiverPosition"], dtype=np.float64)[:, 1, 0]
        fs = int(np.array(f["Data.SamplingRate"]).ravel()[0])
    if ears_y[0] < ears_y[1]:                                            # SOFA: +y = left
        ir = ir[::-1]
    names = ["LS%d" % i for i in ids]
    return ({n: ir[:, e].T.copy() for e, n in enumerate(names)},
            {n: em[e] for e, n in enumerate(names)}, head, fs)


def qsc():
    """QSC-K8 (BRAS, equalised on axis): octave directivity and on-axis magnitude per
    third octave, parsed once into a JSON next to this script."""
    if not os.path.exists(QSC_JSON):
        freqs, E = read_mps(QSC_CSV)
        with open(QSC_JSON, "w") as fh:
            json.dump({"name": "QSC_K8",
                       "note": "BRAS measured directivity (mps, azimuth mean, energy per "
                               "octave, Q_b(0)=1) and on-axis magnitude per third octave",
                       "n_theta": E.shape[0], "table": octave_table(freqs, E).tolist(),
                       "on_axis_freqs": freqs.tolist(),
                       "on_axis_mag": np.sqrt(E[0]).tolist()}, fh)
    with open(QSC_JSON) as fh:
        return json.load(fh)


def on_axis_fir(q, fs, n_taps=4096):
    """Minimum-phase FIR of the speaker's on-axis magnitude (third octaves, interpolated
    in log frequency), 0 dB at 1 kHz."""
    fq, mq = np.asarray(q["on_axis_freqs"]), np.asarray(q["on_axis_mag"])
    ok = mq > 0
    f = np.fft.rfftfreq(16384, 1.0 / fs)
    db = np.interp(np.log(np.maximum(f, 1.0)), np.log(fq[ok]), 20 * np.log10(mq[ok]))
    db -= np.interp(np.log(1000.0), np.log(fq[ok]), 20 * np.log10(mq[ok]))
    return min_phase_fir(10 ** (db / 20), n_taps)


def butter_response(f, fs, fc, high, order=4):
    """Digital Butterworth (bilinear, prewarped) frequency response on the frequencies f."""
    wc = 2 * fs * np.tan(np.pi * fc / fs)
    p = wc * np.exp(1j * np.pi * (2 * np.arange(1, order + 1) + order - 1) / (2 * order))
    if high:
        p = wc ** 2 / p
    zp = (2 * fs + p) / (2 * fs - p)
    zero = 1.0 if high else -1.0
    z = np.exp(2j * np.pi * np.asarray(f, dtype=np.float64) / fs)
    ref = -1.0 if high else 1.0                     # unit gain at Nyquist / DC
    return (np.prod((z[:, None] - zero) / (z[:, None] - zp), axis=1)
            / np.prod((ref - zero) / (ref - zp)))


def chain(ir, fs, eq):
    """Speaker on-axis response, then the band-pass BRAS applied to its BRIRs (4th-order
    Butterworth, -3 dB at 50 Hz and 20 kHz), on a simulated BRIR (n, 2)."""
    n = len(ir)
    nfft = 1 << int(np.ceil(np.log2(n + len(eq) + fs)))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    H = (np.fft.rfft(eq, nfft) * butter_response(f, fs, 50.0, True)
         * butter_response(f, fs, 20000.0, False))
    return np.fft.irfft(np.fft.rfft(ir, nfft, axis=0) * H[:, None], nfft, axis=0)[:n]


def with_extra_absorption(man, dalpha):
    """Manifest with dalpha (per octave, OCTAVES) added to every material, on the
    geometric bands and on the third-octave curves of the wave band."""
    man = json.loads(json.dumps(man))
    lo = np.log(OCTAVES)
    for m in man["materials"].values():
        m["absorption"] = np.clip(np.asarray(m["absorption"]) + dalpha, 0.0, 0.99).tolist()
        if "absorption_full" in m:
            f = np.asarray(m["absorption_full"]["f"], dtype=np.float64)
            add = np.interp(np.log(f), lo, dalpha)
            m["absorption_full"]["alpha"] = np.clip(
                np.asarray(m["absorption_full"]["alpha"]) + add, 0.0, 0.99).tolist()
    return man


def sabine_increment(scene, t_now, t_goal):
    """Uniform absorption coefficient to add per octave so that the Sabine decay goes
    from t_now to t_goal: 0.161 V (1/t_goal - 1/t_now) / S, V and S from the model."""
    tv = np.asarray(scene.tri_verts, dtype=np.float64)
    cross = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()
    vol = abs(np.einsum("ij,ij->i", tv[:, 0], np.cross(tv[:, 1], tv[:, 2])).sum()) / 6.0
    return 0.161 * vol * (1.0 / np.asarray(t_goal) - 1.0 / np.asarray(t_now)) / area, vol, area


def cfg_wave_dir():
    """True when the engine gives the wave band the source directivity (cache tag)."""
    from auralis.engine.pipeline import BakeConfig
    return bool(getattr(BakeConfig(), "fdtd_source_directivity", False))


def simulate(out_dir, emitters, head, quick, dalpha=None):
    """Simulated BRIRs {LS: (n, 2)}: published CR2 configuration (plus dalpha per
    octave if given), FABIAN head, QSC-K8 aimed as measured; one npz per source,
    so a rerun resumes."""
    man = json.load(open(MANIFEST))
    tag = ("_quick" if quick else "") + ("_wavedir" if cfg_wave_dir() else "")
    if dalpha is not None:
        man = with_extra_absorption(man, dalpha)
        tag += "_fit" + "_".join("%03d" % round(1000 * a) for a in dalpha)
    scene = build_scene(man, HERE)
    cfg = build_config(man)
    if not os.path.exists(HRTF):
        raise SystemExit("FABIAN head missing: %s (see data_hrtf/README.md)" % HRTF)
    cfg.hrtf_path = HRTF
    half = wave_ear_half(cfg)
    if abs(half - 0.0875) > 1e-9:               # 8.75 cm ear probes keep the untagged cache
        tag += "_ear%03d" % round(1000 * half)
    if quick:
        cfg.n_rays, cfg.fdtd = 20000, False
    dirv = Directivity(np.asarray(qsc()["table"]), name="QSC_K8")
    # FABIAN faces LS7 (ListenerView 0); LS3-LS6 were turned and tilted towards its
    # head, LS7 (on a stand) was not (BRAS documentation, 3.4.2)
    rcv = ReceiverDef("MP6", position=head, forward=np.array([0.0, -1.0, 0.0]))
    sims = {}
    for ls, pos in emitters.items():
        path = os.path.join(out_dir, "sim_%s%s.npz" % (ls, tag))
        if not os.path.exists(path):
            t0 = time.time()
            fwd = (np.array([0.0, 1.0, 0.0]) if ls == "LS7"
                   else (head - pos) / np.linalg.norm(head - pos))
            src = SourceDef(ls, pos, directivity=dirv, forward=fwd)
            low = None
            if cfg.fdtd:
                low, err = source_low_bands(scene, src, [rcv], cfg)
                if err:
                    raise SystemExit("wave band of %s failed: %s" % (ls, err))
            res = bake_pair(scene, src, rcv, cfg, low)
            if res.fs != 44100:
                raise SystemExit("simulated at %d Hz, the BRIRs are at 44100" % res.fs)
            np.savez(path, ir=res.ir)
            print("  %s simulated in %.0f s" % (ls, time.time() - t0), flush=True)
        sims[ls] = np.asarray(np.load(path)["ir"], dtype=np.float64)
    return sims


def onset(ir, fs):
    """First sample within 20 dB of the strongest arrival of the first 100 ms."""
    e = np.abs(ir[:int(0.1 * fs)]).max(axis=1)
    return int(np.argmax(e >= 0.1 * e.max()))


def shift(ir, d, n):
    """ir delayed by d samples (advanced if d < 0), cut or zero-padded to n."""
    out = np.zeros((n, ir.shape[1]))
    src = ir[max(-d, 0):]
    m = min(len(src), n - max(d, 0))
    out[max(d, 0):max(d, 0) + m] = src[:m]
    return out


def bras_gain(ir, fs):
    """AKnormalize(brir, 'abs', 'mean', 'mean', 1, [300 1000]) of BRAS: the mean linear
    magnitude over 300 Hz - 1 kHz and both ears, which BRAS set to 1 on the nearest
    speaker (LS7)."""
    H = np.abs(np.fft.rfft(ir, axis=0))
    f = np.fft.rfftfreq(len(ir), 1.0 / fs)
    return float(H[(f >= 300) & (f <= 1000)].mean())


def band_energy(x, fs, fc):
    """Energy of x (n, 2) in the octave around fc, both ears."""
    X = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(X[(f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))].sum())


def report(meas, sim, fs, t0_meas, t0_sim):
    """Objective comparison, text lines: arrival times, early/late levels per octave,
    decay times, and the transducer chain on LS7's direct sound."""
    out = ["Arrival times relative to LS7 (geometry check), sim - measured:"]
    for ls in sorted(meas):
        dm = (onset(meas[ls], fs) - t0_meas) / fs * 1e3
        ds = (onset(sim[ls], fs) - t0_sim) / fs * 1e3
        out.append("  %s  %+.2f ms  (measured %+.2f ms)" % (ls, ds - dm, dm))
    out.append("")
    out.append("Level sim - measured (dB), early 0-50 ms / late 50 ms-end after the direct:")
    out.append("        " + "".join("%11d" % c for c in OCTAVES))
    for ls in sorted(meas):
        k = onset(meas[ls], fs)
        cut = k + int(0.05 * fs)
        row = []
        for c in OCTAVES:
            e = 10 * np.log10(band_energy(sim[ls][:cut], fs, c) / band_energy(meas[ls][:cut], fs, c))
            la = 10 * np.log10(band_energy(sim[ls][cut:], fs, c) / band_energy(meas[ls][cut:], fs, c))
            row.append("%5.1f/%5.1f" % (e, la))
        out.append("  %s " % ls + "".join(row))
    out.append("")
    for key in ("T20", "EDT", "C50"):
        mm = [rm.band_metrics_at(meas[ls].mean(axis=1), fs, OCTAVES)[key] for ls in meas]
        ss = [rm.band_metrics_at(sim[ls].mean(axis=1), fs, OCTAVES)[key] for ls in sim]
        mean = lambda v: [np.mean([x[b] for x in v if x[b] is not None]) for b in range(len(OCTAVES))]  # noqa: E731
        unit = "dB" if key == "C50" else "s"
        out.append("%s (%s, mean of 5 speakers, ear average)  measured / simulated:" % (key, unit))
        out.append("  " + "  ".join("%d: %.2f / %.2f" % (c, a, b)
                                    for c, a, b in zip(OCTAVES, mean(mm), mean(ss))))
    k = onset(meas["LS7"], fs)
    w = slice(k - int(0.0003 * fs), k + int(0.0012 * fs))
    win = np.hanning(w.stop - w.start)[:, None]
    hm = np.abs(np.fft.rfft(meas["LS7"][w] * win, 4096, axis=0)) ** 2
    hs = np.abs(np.fft.rfft(sim["LS7"][w] * win, 4096, axis=0)) ** 2
    f = np.fft.rfftfreq(4096, 1.0 / fs)
    out.append("")
    out.append("Transducer chain, LS7 direct sound (1.5 ms window), sim - measured (dB):")
    out.append("  " + "  ".join("%dk: %+.1f" % (c // 1000, 10 * np.log10(
        hs[(f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))].sum()
        / hm[(f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))].sum())) for c in (2000, 4000, 8000, 16000)))
    return out


def render(irs, spec, fs):
    """A program through the BRIRs, cut to its total length: (n, 2)."""
    _label, path, routes, start, length, total = spec
    x, fs_x = read_wav(path)
    if fs_x != fs:
        raise SystemExit("%s is at %d Hz, the BRIRs at %d" % (path, fs_x, fs))
    i0 = int(round(start * fs))
    x = x[i0:i0 + (int(round(length * fs)) if length else len(x))]
    n = int(round(total * fs))
    y = np.zeros((n, 2))
    nfft = 1 << int(np.ceil(np.log2(n + len(irs[routes[0][1]]))))
    for ch, ls in routes:
        X = np.fft.rfft(x[:n, ch], nfft)
        y += np.fft.irfft(X[:, None] * np.fft.rfft(irs[ls], nfft, axis=0), nfft, axis=0)[:n]
    return y


def wav16(y, fs):
    """16-bit WAV bytes, mono (n,) or stereo (n, 2)."""
    ch = 1 if y.ndim == 1 else y.shape[1]
    pcm = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2").tobytes()
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, ch, fs, fs * 2 * ch, 2 * ch, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def dry(spec, fs):
    """The program without the room: its source channels summed, mono, at full length."""
    _label, path, routes, start, length, total = spec
    x, _fs = read_wav(path)
    i0 = int(round(start * fs))
    x = x[i0:i0 + (int(round(length * fs)) if length else len(x))]
    n = int(round(total * fs))
    y = np.zeros(n)
    for ch, _ls in routes:
        y[:min(n, len(x))] += x[:n, ch]
    return y


def prepare(raw, meas, fs, eq):
    """Speaker response and band-pass, then one delay and one gain for the whole set:
    BRAS removed the leading silence of LS7 and normalised it with bras_gain, the
    relative times and levels of the five speakers stay as simulated."""
    n = len(meas["LS7"])
    sim = {ls: chain(ir, fs, eq) for ls, ir in raw.items()}
    d = onset(meas["LS7"], fs) - onset(sim["LS7"], fs)
    sim = {ls: shift(ir, d, n) for ls, ir in sim.items()}
    g = bras_gain(meas["LS7"], fs) / bras_gain(sim["LS7"], fs)
    return {ls: ir * g for ls, ir in sim.items()}, d, g


def t20(irs, fs):
    """T20 per octave of the ear average, mean over the speakers."""
    v = [rm.band_metrics_at(ir.mean(axis=1), fs, OCTAVES)["T20"] for ir in irs.values()]
    return np.array([np.mean([x[b] for x in v if x[b] is not None])
                     for b in range(len(OCTAVES))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "listening_out"))
    ap.add_argument("--quick", action="store_true",
                    help="no wave band, 20k rays: tests the chain in a minute")
    ap.add_argument("--no-session-fit", action="store_true",
                    help="keep the published materials (the BRIR session decays ~12 %% faster)")
    ap.add_argument("--set", choices=sorted(PROGRAM_SETS), default="bras",
                    help="stimuli: BRAS quartet and noise, or Odeon speech and isolated "
                         "events (local licence)")
    a = ap.parse_args()
    pset = PROGRAM_SETS[a.set]
    os.makedirs(a.out, exist_ok=True)
    meas, emitters, head, fs = measured()
    print("measured BRIRs: %s, head at %s, %d Hz" % (sorted(meas), head, fs), flush=True)
    eq = on_axis_fir(qsc(), fs)
    t0 = onset(meas["LS7"], fs)
    sim, d, g = prepare(simulate(a.out, emitters, head, a.quick), meas, fs, eq)
    lines = ["CR2 listening test %s (%s)" % (time.strftime("%Y-%m-%d %H:%M"),
                                              "QUICK" if a.quick else "full"),
             "measured LS7 under the BRAS normalisation: %.3f (1 = the rule is reproduced)"
             % bras_gain(meas["LS7"], fs), "",
             "=== PUBLISHED materials (cr2_dodecahedron_fitted31_dir) ===",
             "simulated set: delay %+d samples, gain %+.1f dB" % (d, 20 * np.log10(g)), ""]
    lines += report(meas, sim, fs, t0, t0)

    t_meas, fit = t20(meas, fs), None
    if not a.no_session_fit:
        # LABELLED EXCEPTION to "never refit against the targets":
        # the BRIR session decays ~12 % faster than the RIR session the materials
        # come from; a uniform absorption from the measured T20 of THIS session puts
        # the room right, so the test judges the rendering, not the room prediction
        scene = build_scene(json.load(open(MANIFEST)), HERE)
        fit, t_sim = np.zeros(len(OCTAVES)), t20(sim, fs)
        lines += ["", "=== SESSION FIT (labelled exception, BRAS 'fitted' protocol) ==="]
        for it in range(2):
            inc, vol, area = sabine_increment(scene, t_sim, t_meas)
            fit = np.round(fit + inc, 3)
            sim, d, g = prepare(simulate(a.out, emitters, head, a.quick, fit), meas, fs, eq)
            t_sim = t20(sim, fs)
            lines.append("pass %d (V %.0f m3, S %.0f m2): added alpha %s -> T20 sim / measured %s"
                         % (it + 1, vol, area, fit.tolist(),
                            np.round(t_sim / t_meas, 3).tolist()))
            if np.abs(t_sim / t_meas - 1).max() < 0.03:
                break
        lines += ["simulated set: delay %+d samples, gain %+.1f dB" % (d, 20 * np.log10(g)), ""]
        lines += report(meas, sim, fs, t0, t0)

    stimuli, levels = {}, {}
    lines.append("")
    for prog, spec in pset["programs"].items():
        real, fake = render(meas, spec, fs), render(sim, spec, fs)
        if prog == "quartet":
            ref, _ = read_wav(os.path.join(AUDIO, "Auralization", "string_CR2.wav"))
            m = min(len(ref), len(real))
            cc = min(np.corrcoef(ref[:m, e], real[:m, e])[0, 1] for e in (0, 1))
            lines.append("measured quartet against BRAS string_CR2.wav: correlation %.4f, "
                         "level %+.2f dB" % (cc, 10 * np.log10((real[:m] ** 2).sum()
                                                               / (ref[:m] ** 2).sum())))
        lv = 10 * np.log10((fake ** 2).sum() / (real ** 2).sum())
        levels[prog] = round(float(lv), 2)
        lines.append("%s: simulated - measured level %+.1f dB (broadband, both ears)" % (prog, lv))
        k = 0.9 / max(np.abs(real).max(), np.abs(fake).max())   # one gain: levels kept
        for tag, y in (("real", real * k), ("sim", fake * k)):
            write_wav(os.path.join(a.out, "%s_%s.wav" % (prog, tag)), y.astype(np.float32), fs)
            stimuli["%s_%s" % (prog, tag)] = base64.b64encode(wav16(y, fs)).decode("ascii")
        # anechoic reference, diotic, at the RMS of the real version (peak capped: an
        # impulsive dry sound is much peakier than its reverberant version)
        d = dry(spec, fs)
        d *= np.sqrt(((real * k) ** 2).mean() / max((d ** 2).mean(), 1e-30))
        d *= min(1.0, 0.95 / max(np.abs(d).max(), 1e-30))
        stimuli["%s_dry" % prog] = base64.b64encode(wav16(d, fs)).decode("ascii")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(txt + "\n")
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "quick": a.quick, "set": a.set,
            "programs": [{"id": k, "label": v[0]} for k, v in pset["programs"].items()],
            "trials": pset["trials"], "credit": pset["credit"],
            "levels_sim_minus_real_db": levels,
            "session_fit_added_alpha": None if fit is None else dict(zip(OCTAVES, fit.tolist())),
            "t20_measured": dict(zip(OCTAVES, np.round(t_meas, 3).tolist())),
            "t20_simulated": dict(zip(OCTAVES, np.round(t20(sim, fs), 3).tolist())),
            "config": "CR2 published (cr2_dodecahedron_fitted31_dir), FABIAN measured HATO 0, "
                      "QSC-K8 BRAS directivity + on-axis response, BRAS band-pass and gain rule"}
    with open(PAGE) as fh:
        page = fh.read()
    page = page.replace("__AUDIO__", json.dumps(stimuli)).replace("__META__", json.dumps(meta))
    html = os.path.join(a.out, "listening_test%s%s.html" % (
        "" if a.set == "bras" else "_" + a.set, "_quick" if a.quick else ""))
    with open(html, "w") as fh:
        fh.write(page)
    print("\npage: %s (%.1f MB)" % (html, os.path.getsize(html) / 1e6))


if __name__ == "__main__":
    main()
