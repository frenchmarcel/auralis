"""Ground effect of a point source above a porous ground: engine against references.

Compares the excess attenuation (level re free field, third octaves) of the geometrical
solver (plane-wave and spherical-wave reflection) with the analytic plane-wave and
Weyl-Van der Pol solutions, and the wave solver alone with the exact solution behind the
same time gate (the GPU helps: 30-40 M cells).
Usage: python tools/ground_effect_check.py [--sigma 200e3 [--depth 1.0] | --ground D]
       [--hs 1.0] [--hr 1.5] [--dist 10 25] [--fc 600] [--no-fdtd]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AURALIS_FDTD_SUBSONIC_HZ", "0")

from auralis.engine import ground, poromat                      # noqa: E402
from auralis.engine.air import config_speed_of_sound             # noqa: E402
from auralis.engine.geometry import SceneData                    # noqa: E402
from dataclasses import replace                                   # noqa: E402
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402

FS = 48000
THIRDS = 1000.0 * 2.0 ** (np.arange(-13, 8) / 3.0)            # 50 Hz - 5 kHz


def plane(x0, x1, y0, y1, z):
    a, b, c, d = [x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]
    return np.array([[a, b, c], [a, c, d]], dtype=np.float64)


def ground_scene(spec, d, z):
    tv = plane(-3.0, d + 3.0, -6.0, 6.0, z)
    alpha = poromat.alpha_bands(spec)[None]
    return SceneData(tv, np.zeros(len(tv), np.int32), alpha, np.zeros_like(alpha),
                     np.zeros_like(alpha), mat_phys=[spec], mat_names=["ground"])


def omni(res):
    return np.asarray(res.ambi[:, 0], dtype=np.float64)


def third_levels(h, ref):
    n = 1 << int(np.ceil(np.log2(max(len(h), len(ref)) * 2)))
    H, R = np.fft.rfft(h, n), np.fft.rfft(ref, n)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    out = []
    for fc in THIRDS:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10 * np.log10(np.sum(np.abs(H[m]) ** 2) / np.sum(np.abs(R[m]) ** 2)))
    return np.array(out)


def _edge_path(src, rcv, a, b, n=4001):
    """Shortest source -> edge -> receiver path over the segment a-b."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    p = a[None] + t * (b - a)[None]
    return float(np.min(np.linalg.norm(p - src, axis=1) + np.linalg.norm(p - rcv, axis=1)))


def _plate_for_gate(src, rcv, path_min):
    """Half sizes of a plate whose edges are all heard after a path of path_min."""
    d = rcv[0] - src[0]
    lx, ly = 1.0, 1.0
    for _ in range(400):
        x0, x1 = src[0] - lx, rcv[0] + lx
        corners = [np.array(v) for v in ((x0, -ly, 0.0), (x1, -ly, 0.0), (x1, ly, 0.0),
                                         (x0, ly, 0.0))]
        paths = [_edge_path(src, rcv, corners[i], corners[(i + 1) % 4]) for i in range(4)]
        if min(paths[1], paths[3]) < path_min:
            lx += 0.25
        if min(paths[0], paths[2]) < path_min:
            ly += 0.25
        if min(paths) >= path_min:
            return lx, ly, d
    raise ValueError("no plate size found")


def wave_rows(spec, hs, hr, d, c, fc, gate_ms=40.0):
    """Wave solver alone over a finite plate whose edges are heard only after the gate. Both
    responses and the exact solution are cut at the same time (ground reflection + gate_ms,
    5 ms taper), so the plate acts as an infinite ground in the window. The free-field
    reference is the same run over a 1 cm triangle (same grid step, pulse and length, so the
    same calibration band). Thirds up to 1.2 fc; the gate smooths the lowest ones alike."""
    from auralis.engine.fdtd import run_low_band
    src, rcv = np.array([0.0, 0.0, hs]), np.array([d, 0.0, hr])
    r2 = np.hypot(d, hr + hs)
    t_gate, taper = r2 / c + gate_ms / 1e3, 0.005
    lx, ly, _ = _plate_for_gate(src, rcv, c * (t_gate + taper + 0.002))
    plate = plane(-lx, d + lx, -ly, ly, 0.0)
    sc = SceneData(plate, np.zeros(2, np.int32), poromat.alpha_bands(spec)[None],
                   np.zeros((1, 7)), np.zeros((1, 7)), mat_phys=[spec], mat_names=["ground"])
    tiny = plane(d / 2, d / 2 + 0.01, 0.0, 0.01, max(hs, hr) + 1.0)
    sc0 = SceneData(tiny, np.zeros(2, np.int32), np.zeros((1, 7)), np.zeros((1, 7)),
                    np.zeros((1, 7)), mat_names=["probe"])
    # the anisotropic conformal correction, as the app uses for open scenes: the default one
    # slows the waves that graze a wall (here +50 mm of path over 10 m, the dip 4 dB off)
    kw = dict(fs=FS, duration=t_gate + taper + 0.02, f_cross=fc, speed_of_sound=c,
              max_cells=80_000_000, conformal_aniso=True)
    h, info = run_low_band(sc, src, rcv, **kw)
    h0, _ = run_low_band(sc0, src, rcv, **kw)
    n = 1 << 16
    t = np.arange(len(h)) / FS
    w = np.where(t < t_gate, 1.0, np.where(t < t_gate + taper,
                                           0.5 + 0.5 * np.cos(np.pi * (t - t_gate) / taper), 0.0))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    H, H0 = np.fft.rfft(h, n), np.fft.rfft(h0, n)
    a = np.fft.irfft(H0 * _exact(spec, hs, hr, d, c, f), n)[:len(h)]
    # the direct sound with the grid's own dispersion (offline estimate): the reinjected
    # direct is exact while the reflection travels through the grid
    lag = _grid_lag(f, info.dx, info.dt, c, rcv - src) * (np.linalg.norm(rcv - src) - 12 * info.dx)
    h_m = np.fft.irfft(H + H0 * (np.exp(-2j * np.pi * f * lag) - 1.0), n)[:len(h)]
    h0_m = np.fft.irfft(H0 * np.exp(-2j * np.pi * f * lag), n)[:len(h)]
    G, G0, A, Gm, G0m = (np.abs(np.fft.rfft(x * w, n)) ** 2 for x in (h, h0, a, h_m, h0_m))
    wm = np.full(len(THIRDS), np.nan)
    wav, ana = np.full(len(THIRDS), np.nan), np.full(len(THIRDS), np.nan)
    for i, x in enumerate(THIRDS):
        m = (f >= x * 2 ** (-1 / 6)) & (f < x * 2 ** (1 / 6))
        if x <= 1.2 * fc:
            wav[i] = 10 * np.log10(G[m].sum() / G0[m].sum())
            ana[i] = 10 * np.log10(A[m].sum() / G0[m].sum())
            wm[i] = 10 * np.log10(Gm[m].sum() / G0m[m].sum())
    print("wave solver: plate %.1f x %.1f m, gate %.0f ms, dx %.1f cm, %.1f M cells, %.0f s, %s"
          % (d + 2 * lx, 2 * ly, gate_ms, 100 * info.dx, info.n_cells / 1e6, info.runtime_s,
             info.engine), flush=True)
    return {"waves, gated": wav, "waves, matched direct": wm, "exact, same gate": ana}


def _grid_lag(f, dx, dt, c, direction):
    """Delay per metre (s/m) of the 4th-order scheme against c, along `direction`."""
    nrm = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    out = np.zeros(len(f))
    for i, x in enumerate(f):
        if x <= 0.0 or x * dt >= 0.25:
            continue
        lhs = (2.0 * np.cos(2 * np.pi * x * dt) - 2.0) / (c * dt) ** 2
        lo, hi = 0.0, np.pi / dx
        for _ in range(60):
            k = 0.5 * (lo + hi)
            q = k * nrm * dx
            S = np.sum(-2.0 * np.cos(2.0 * q) + 32.0 * np.cos(q) - 30.0) / (12.0 * dx * dx)
            lo, hi = (k, hi) if S > lhs else (lo, k)
        out[i] = 0.5 * (lo + hi) / (2 * np.pi * x) - 1.0 / c
    return out


def _exact(spec, hs, hr, d, c, f):
    """p / p_direct of the spherical-wave solution at the frequencies f."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 1.0)
    r1, r2 = np.hypot(d, hr - hs), np.hypot(d, hr + hs)
    q = ground.spherical_reflection(f, poromat.Z0 / poromat.surface_impedance(spec, f, 1.0),
                                    (hr + hs) / r2, r2, c)
    return 1.0 + q * r1 / r2 * np.exp(-2j * np.pi * f * (r2 - r1) / c)


def analytic(spec, hs, hr, d, c, spherical):
    f = np.linspace(1.0, FS / 2, 200001)
    r1 = np.hypot(d, hr - hs)
    r2 = np.hypot(d, hr + hs)
    mu = (hr + hs) / r2
    zs = poromat.surface_impedance(spec, f, 1.0)
    beta = poromat.Z0 / zs
    q = (ground.spherical_reflection(f, beta, mu, r2, c) if spherical
         else (mu - beta) / (mu + beta))
    k = 2 * np.pi * f / c
    p = np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2
    p0 = np.exp(-1j * k * r1) / r1
    out = []
    for fc in THIRDS:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10 * np.log10(np.mean(np.abs(p[m]) ** 2) / np.mean(np.abs(p0[m]) ** 2)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sigma", type=float, default=200e3, help="flow resistivity, Pa.s/m2")
    ap.add_argument("--depth", type=float, default=1.0, help="porous layer depth, m")
    ap.add_argument("--ground", default="", help="Nord2000 ground class A-G (Delany-Bazley, "
                    "semi-infinite) instead of --sigma and --depth")
    ap.add_argument("--hs", type=float, default=1.0)
    ap.add_argument("--hr", type=float, default=1.5)
    ap.add_argument("--dist", type=float, nargs="+", default=[10.0, 25.0])
    ap.add_argument("--fc", type=float, default=600.0, help="wave band limit, Hz")
    ap.add_argument("--no-fdtd", action="store_true")
    a = ap.parse_args()
    spec = (ground.ground_spec(a.ground) if a.ground
            else {"layers": [{"model": "miki", "sigma": a.sigma, "d": a.depth}]})
    if a.ground:
        a.sigma = spec["layers"][0]["sigma"]
    for d in a.dist:
        src = SourceDef("S", np.array([0.0, 0.0, a.hs]))
        rcv = ReceiverDef("R", np.array([d, 0.0, a.hr]))
        t_max = round(1.2 * np.hypot(d, a.hs + a.hr) / 340.0 + 0.1, 2)
        cfg = BakeConfig(fs=FS, n_rays=2000, ism_order=1, diffraction=False, max_time=t_max,
                         normalize=False, ambisonic_order=1, rbs_scattering=False)
        c = config_speed_of_sound(cfg)
        free = omni(bake(ground_scene(spec, d, -500.0), src, rcv, cfg))
        ga = third_levels(omni(bake(ground_scene(spec, d, 0.0), src, rcv, cfg)), free)
        cfg_s = replace(cfg, spherical_reflection=True)
        ga_s = third_levels(omni(bake(ground_scene(spec, d, 0.0), src, rcv, cfg_s)), free)
        an_p = analytic(spec, a.hs, a.hr, d, c, spherical=False)
        an_s = analytic(spec, a.hs, a.hr, d, c, spherical=True)
        rows = {"rays, plane (engine)": ga, "rays, spherical (eng.)": ga_s,
                "analytic, plane wave": an_p, "analytic, spherical": an_s}
        if not a.no_fdtd:
            rows.update(wave_rows(spec, a.hs, a.hr, d, c, a.fc))
        print("\nd = %.0f m, hs = %.2f m, hr = %.2f m, sigma = %.0f kPa.s/m2: level re free "
              "field (dB)" % (d, a.hs, a.hr, a.sigma / 1e3))
        print("%-24s" % "third octave (Hz)" + "".join("%7.0f" % f for f in THIRDS))
        for name, v in rows.items():
            print("%-24s" % name + "".join("%7.1f" % x for x in v))


if __name__ == "__main__":
    main()
