"""A thin screen on a ground: rays (one edge path, or the four paths) against the wave solver.

Level re free field in third octaves behind a rigid screen. The wave solver is the
reference: a finite plate cut by a time gate acts as an infinite ground (see
ground_effect_check.py). The ray responses get the same gate.
Usage: python tools/screen_ground_check.py [--ground D] [--height 2.0] [--fc 600]
       [--gate-ms 40] [--ppw 9]
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AURALIS_FDTD_SUBSONIC_HZ", "0")

from auralis.engine import ground, poromat                       # noqa: E402
from auralis.engine.air import config_speed_of_sound             # noqa: E402
from auralis.engine.fdtd import run_low_band                     # noqa: E402
from auralis.engine.geometry import SceneData                    # noqa: E402
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ground_effect_check import _plate_for_gate, plane         # noqa: E402

FS = 48000
THIRDS = 1000.0 * 2.0 ** (np.arange(-10, 8) / 3.0)            # 100 Hz - 5 kHz


def screen_scene(spec, x0, x1, ly, xs, height):
    """Ground plate (material 0, the ground model) and a rigid thin screen (material 1)."""
    tv = np.concatenate([plane(x0, x1, -ly, ly, 0.0),
                         np.array([[[xs, -ly, 0.0], [xs, ly, 0.0], [xs, ly, height]],
                                   [[xs, -ly, 0.0], [xs, ly, height], [xs, -ly, height]]])])
    al = np.vstack([poromat.alpha_bands(spec), np.zeros(7)])
    return SceneData(tv, np.array([0, 0, 1, 1], np.int32), al, np.zeros_like(al),
                     np.zeros_like(al), mat_phys=[spec, None], mat_names=["ground", "screen"])


def probe_scene(x, z):
    tv = plane(x, x + 0.01, 0.0, 0.01, z)
    return SceneData(tv, np.zeros(2, np.int32), np.zeros((1, 7)), np.zeros((1, 7)),
                     np.zeros((1, 7)), mat_names=["probe"])


def gated_thirds(h, h0, t_gate, taper=0.005):
    n = 1 << 17
    t = np.arange(max(len(h), len(h0))) / FS
    w = np.where(t < t_gate, 1.0, np.where(t < t_gate + taper,
                                           0.5 + 0.5 * np.cos(np.pi * (t - t_gate) / taper), 0.0))
    G = np.abs(np.fft.rfft(np.pad(h, (0, len(w) - len(h))) * w, n)) ** 2
    G0 = np.abs(np.fft.rfft(np.pad(h0, (0, len(w) - len(h0))) * w, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / FS)
    out = []
    for x in THIRDS:
        m = (f >= x * 2 ** (-1 / 6)) & (f < x * 2 ** (1 / 6))
        out.append(10 * np.log10(G[m].sum() / G0[m].sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ground", default="D", help="Nord2000 ground class A-G")
    ap.add_argument("--height", type=float, default=2.0, help="screen height, m")
    ap.add_argument("--fc", type=float, default=600.0, help="wave solver limit, Hz")
    ap.add_argument("--gate-ms", type=float, default=40.0)
    ap.add_argument("--ppw", type=float, default=9.0, help="wave solver points per wavelength")
    a = ap.parse_args()
    spec = ground.ground_spec(a.ground)
    src, rcv, xs = np.array([0.0, 0.0, 0.5]), np.array([10.0, 0.0, 1.5]), 5.0
    top = np.array([xs, 0.0, a.height])
    imgs = [np.array([0.0, 0.0, -0.5]), src]
    longest = max(np.linalg.norm(top - s) for s in imgs) + np.linalg.norm(
        top - np.array([10.0, 0.0, -1.5]))
    cfg0 = BakeConfig(fs=FS, n_rays=2000, ism_order=2, max_time=0.2, normalize=False,
                      ambisonic_order=1, rbs_scattering=False, diffraction=True)
    c = config_speed_of_sound(cfg0)
    t_gate = longest / c + a.gate_ms / 1e3
    lx, ly, _ = _plate_for_gate(src, rcv, c * (t_gate + 0.007))
    sc = screen_scene(spec, -lx, 10.0 + lx, ly, xs, a.height)
    sc0 = probe_scene(5.0, a.height + 1.0)
    S, R = SourceDef("S", src), ReceiverDef("R", rcv)
    rows = {}
    free = np.asarray(bake(sc0, S, R, cfg0).ambi[:, 0])
    for name, sph in (("rays, one edge path", False), ("rays, four paths", True)):
        cfg = replace(cfg0, spherical_reflection=sph)
        rows[name] = gated_thirds(np.asarray(bake(sc, S, R, cfg).ambi[:, 0]), free, t_gate)
    kw = dict(fs=FS, duration=t_gate + 0.03, f_cross=a.fc, speed_of_sound=c,
              max_cells=90_000_000, conformal_aniso=True, ppw=a.ppw)
    h, info = run_low_band(sc, src, rcv, **kw)
    h0, _ = run_low_band(sc0, src, rcv, **kw)
    wv = gated_thirds(h, h0, t_gate)
    wv[THIRDS > 1.2 * a.fc] = np.nan
    rows["waves (reference)"] = wv
    print("screen %.1f m at 5 m, source 0.5 m, receiver 1.5 m at 10 m, ground class %s; plate "
          "%.0f x %.0f m, gate %.0f ms; wave solver dx %.1f cm, %.1f M cells, %.0f s, %s"
          % (a.height, a.ground, 10 + 2 * lx, 2 * ly, a.gate_ms, 100 * info.dx,
             info.n_cells / 1e6, info.runtime_s, info.engine))
    print("level re free field (dB)")
    print("%-22s" % "third octave (Hz)" + "".join("%7.0f" % f for f in THIRDS))
    for k, v in rows.items():
        print("%-22s" % k + "".join("%7.1f" % x for x in v))
    ok = ~np.isnan(wv)
    for k in ("rays, one edge path", "rays, four paths"):
        print("%-22s max deviation from the waves %.1f dB, mean %.1f dB"
              % (k, np.abs(rows[k] - wv)[ok].max(), np.abs(rows[k] - wv)[ok].mean()))


if __name__ == "__main__":
    main()
