"""Wave band of a walking listener: error of the cross-fade between key positions against the
response computed at every position. One FDTD run (reciprocity: the source fixed, probes every
2 cm along the walk), then key spacings of lambda/8 to 2 lambda at the crossover, cross-faded
as the renderer does (plain) or after aligning each key on the direct-sound delay of the
current position (aligned, what a variable delay line does).
python tools/key_spacing_check.py [--fc 250] [--walk 6] [--length 30]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auralis.engine.fdtd import run_low_band  # noqa: E402
from street_bench import street_scene  # noqa: E402

C = 343.0


def third_octave_db(h, fs, f_lo, f_hi):
    """Energy per third octave between f_lo and f_hi (dB)."""
    n = 1 << int(np.ceil(np.log2(len(h) * 2)))
    H = np.abs(np.fft.rfft(h, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    centres = 1000.0 * 2.0 ** (np.arange(-30, 1) / 3.0)
    centres = centres[(centres >= f_lo) & (centres <= f_hi)]
    out = []
    for fc in centres:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10.0 * np.log10(max(H[m].sum(), 1e-30)))
    return np.array(out)


def shift(h, d, fs):
    """h delayed by d seconds (fractional, through the spectrum; zero-padded, no wrap)."""
    n = 1 << int(np.ceil(np.log2(len(h) * 2)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(np.fft.rfft(h, n) * np.exp(-2j * np.pi * f * d), n)[:len(h)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc", type=float, default=250.0)
    ap.add_argument("--walk", type=float, default=6.0, help="walked length (m)")
    ap.add_argument("--length", type=float, default=30.0, help="street length (m)")
    ap.add_argument("--duration", type=float, default=0.6)
    ap.add_argument("--radial", action="store_true",
                    help="the source on the same sidewalk, 4 m before the walk: the listener "
                         "walks straight away from it (the worst case: the path grows 1:1)")
    args = ap.parse_args()
    os.environ.setdefault("AURALIS_FDTD_CONFORMAL", "0")
    sc, R = street_scene(args.length, 15.0, rot_deg=0.0, cars=2)
    src = np.array([0.5 * args.length, -2.0, 0.5])                 # a car on the road
    if args.radial:
        src = np.array([0.5 * args.length - 0.5 * args.walk - 4.0, 15.0 / 2 - 1.2, 1.6])
    step = 0.02
    xs = np.arange(0.5 * args.length - 0.5 * args.walk, 0.5 * args.length + 0.5 * args.walk, step)
    probes = [np.array([x, 15.0 / 2 - 1.2, 1.6]) for x in xs]       # the sidewalk, ear height
    fs = int(round(8.0 * args.fc))
    ir, info = run_low_band(sc, src, probes, fs=fs, duration=args.duration, f_cross=args.fc,
                            speed_of_sound=C, max_cells=2 ** 31 - 2)
    ir = np.asarray(ir, dtype=np.float64)                         # (n, P)
    lam = C / args.fc
    tau = np.array([np.linalg.norm(p - src) / C for p in probes])
    print(("radial walk, " if args.radial else "walk past the source, ") + "street %.0f m, fc %.0f Hz (lambda %.2f m), %d probes every %.0f cm over %.1f m, "
          "%.1f M air cells" % (args.length, args.fc, lam, len(xs), 100 * step, args.walk,
                                info.n_air / 1e6))
    ref_db = [third_octave_db(ir[:, j], fs, 0.25 * args.fc, args.fc) for j in range(len(xs))]
    for frac in (0.125, 0.25, 0.5, 1.0, 2.0):
        d = frac * lam
        k_step = max(1, int(round(d / step)))
        keys = np.arange(0, len(xs), k_step)
        if keys[-1] != len(xs) - 1:
            keys = np.append(keys, len(xs) - 1)
        err = {"plain": [], "aligned": []}
        coh = {"plain": [], "aligned": []}
        for a, b in zip(keys[:-1], keys[1:]):
            for j in range(a + 1, b):
                w = (j - a) / float(b - a)
                hp = (1 - w) * ir[:, a] + w * ir[:, b]
                ha = ((1 - w) * shift(ir[:, a], tau[j] - tau[a], fs)
                      + w * shift(ir[:, b], tau[j] - tau[b], fs))
                for name, h in (("plain", hp), ("aligned", ha)):
                    err[name].append(np.abs(third_octave_db(h, fs, 0.25 * args.fc, args.fc)
                                            - ref_db[j]))
                    coh[name].append(np.sum((h - ir[:, j]) ** 2) / np.sum(ir[:, j] ** 2))
        line = "keys every %.3f lambda (%.2f m): " % (frac, d)
        for name in ("plain", "aligned"):
            e = np.array(err[name])
            line += "%s: third-octave error mean %.2f dB, worst band %.2f dB, waveform error %.1f %%;  " % (
                name, e.mean(), e.mean(axis=0).max(), 100 * np.mean(coh[name]))
        print(line, flush=True)


if __name__ == "__main__":
    main()
