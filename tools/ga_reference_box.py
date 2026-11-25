"""Geometrical solver against two exact references of a box, uniform absorption or an absorbing
floor (an audience): image sources of all orders (specular walls) and radiosity (Lambertian
walls): python tools/ga_reference_box.py [scattering ...] [--old] [--no-refs]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import rirmetrics as rm  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, speed_of_sound  # noqa: E402
from auralis.engine.geometry import make_shoebox  # noqa: E402
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402

SIZE = (20.0, 15.0, 8.0)
SRC = np.array([4.3, 6.1, 1.5])
RCV = np.array([13.7, 9.2, 1.2])
T_C, HR = 20.0, 50.0
BAND = 3                      # 1 kHz
BIN = 0.001
CASES = {"uniform": (0.15, 0.15, 0.15), "absorbing floor": (0.80, 0.05, 0.05)}   # floor, ceiling, walls


def exact_hist(alpha, t_max, air_db_m, c):
    """Energy echogram (1 ms bins) of the specular box: sum of prod(R) exp(-m d) / d^2
    over the image lattice; image x = (1 - 2q) s + 2 m L reflects |m - q| times on the
    wall at 0 and |m| times on the wall at L."""
    a_floor, a_ceil, a_wall = alpha
    walls = [(a_wall, a_wall), (a_wall, a_wall), (a_floor, a_ceil)]
    d_max = c * t_max
    axes = []
    for k in range(3):
        L, s, r = SIZE[k], SRC[k], RCV[k]
        m = np.arange(-int(d_max / (2 * L)) - 2, int(d_max / (2 * L)) + 3)
        off, gain = [], []
        for q in (0, 1):
            off.append((1 - 2 * q) * s + 2 * m * L - r)
            gain.append((1 - walls[k][0]) ** np.abs(m - q) * (1 - walls[k][1]) ** np.abs(m))
        axes.append((np.concatenate(off), np.concatenate(gain)))
    (ox, gx), (oy, gy), (oz, gz) = axes
    m_air = np.log(10.0) / 10.0 * air_db_m
    n_bins = int(np.ceil(t_max / BIN))
    hist = np.zeros(n_bins)
    for i in range(len(ox)):
        d = np.sqrt(ox[i] ** 2 + oy[:, None] ** 2 + oz[None, :] ** 2)
        keep = d < d_max
        e = (gx[i] * gy[:, None] * gz[None, :] * np.exp(-m_air * d) / d ** 2)[keep]
        hist += np.bincount((d[keep] / c / BIN).astype(np.int64), weights=e, minlength=n_bins)[:n_bins]
    return hist


def _patches(h):
    """Square patches of side ~h on the 6 faces: centres (P, 3), inward normals, areas, face
    index (0 floor, 1 ceiling, 2 walls) and 4 x 4 sub-points per patch (P, 16, 3)."""
    lx, ly, lz = SIZE
    faces = [((0, 1), 2, 0.0, 1.0, 0), ((0, 1), 2, lz, -1.0, 1),
             ((1, 2), 0, 0.0, 1.0, 2), ((1, 2), 0, lx, -1.0, 2),
             ((0, 2), 1, 0.0, 1.0, 2), ((0, 2), 1, ly, -1.0, 2)]
    cen, nrm, area, kind, sub = [], [], [], [], []
    g = (np.arange(4) + 0.5) / 4.0
    for (u, v), w, pos, sgn, k in faces:
        nu, nv = max(1, int(round(SIZE[u] / h))), max(1, int(round(SIZE[v] / h)))
        du, dv = SIZE[u] / nu, SIZE[v] / nv
        for i in range(nu):
            for j in range(nv):
                p = np.zeros(3)
                p[u], p[v], p[w] = (i + 0.5) * du, (j + 0.5) * dv, pos
                n = np.zeros(3)
                n[w] = sgn
                q = np.zeros((16, 3))
                q[:, u] = (i + np.repeat(g, 4)) * du
                q[:, v] = (j + np.tile(g, 4)) * dv
                q[:, w] = pos
                cen.append(p), nrm.append(n), area.append(du * dv), kind.append(k), sub.append(q)
    return np.array(cen), np.array(nrm), np.array(area), np.array(kind), np.array(sub)


def radiosity_hist(alpha, t_max, air_db_m, c, h=2.0):
    """Energy echogram of the same box with LAMBERTIAN walls (scattering 1): time-domain
    radiosity on patches of side h, form factors from 4 x 4 sub-points, 1 ms steps."""
    cen, nrm, area, kind, sub = _patches(h)
    rho = 1.0 - np.asarray(alpha)[kind]
    m_air = np.log(10.0) / 10.0 * air_db_m
    P = len(cen)
    F = np.zeros((P, P))
    delay = np.zeros((P, P), dtype=np.int64)
    for i in range(P):
        r = sub[i][:, None, None, :] - sub[None, :, :, :]          # (16, P, 16, 3): i -> j
        d2 = np.maximum((r ** 2).sum(-1), 1e-12)
        d = np.sqrt(d2)
        ci = np.clip(-(r @ nrm[i]) / d, 0.0, None)
        cj = np.clip(np.einsum("apbk,pk->apb", r, nrm) / d, 0.0, None)
        w = ci * cj / (np.pi * d2)
        F[i] = w.mean(axis=(0, 2)) * area
        d_eff = (w * d).sum(axis=(0, 2)) / np.maximum(w.sum(axis=(0, 2)), 1e-30)   # weighted path
        delay[i] = np.rint(d_eff / c / BIN).astype(np.int64)
        F[i] *= np.exp(-m_air * d_eff)
    F[np.arange(P), np.arange(P)] = 0.0
    # closed box: every patch sends ALL its power to the others; the sub-point quadrature
    # misses part of it near the edges (singular kernel), so the rows are renormalised
    geo = F * np.exp(m_air * delay * BIN * c)
    closure = geo.sum(axis=1)
    F /= closure[:, None]
    radiosity_hist.closure = (float(closure.min()), float(np.median(closure)), float(closure.max()))
    n_bins = int(np.ceil(t_max / BIN))
    k_max = int(delay.max()) + 1
    inc = np.zeros((n_bins + k_max + 2, P))
    # first irradiation by a source of unit intensity at 1 m (power 4 pi), sub-point average
    rs = sub - SRC
    ds = np.linalg.norm(rs, axis=-1)
    ws = np.clip(-(rs * nrm[:, None, :]).sum(-1) / ds, 0, None) / ds ** 2 * np.exp(-m_air * ds)
    e_in = ws.mean(axis=1) * area
    k_s = np.rint((ws * ds).sum(1) / np.maximum(ws.sum(1), 1e-30) / c / BIN).astype(np.int64)
    inc[k_s, np.arange(P)] += e_in
    rr = sub - RCV
    dr = np.linalg.norm(rr, axis=-1)
    wr = (np.clip(-(rr * nrm[:, None, :]).sum(-1) / dr, 0, None) / (np.pi * dr ** 2)
          * np.exp(-m_air * dr))
    g_r = wr.mean(axis=1)                                          # per unit radiated power
    k_r = np.rint((wr * dr).sum(1) / np.maximum(wr.sum(1), 1e-30) / c / BIN).astype(np.int64)
    hist = np.zeros(n_bins + k_max + 2)
    dd = np.linalg.norm(RCV - SRC)
    hist[int(dd / c / BIN)] += np.exp(-m_air * dd) / dd ** 2
    flat = (delay * P + np.arange(P)[None, :]).ravel()             # (delay, target) per pair
    for t in range(n_bins):
        out = rho * inc[t]                                         # radiated power per patch
        if out.max() <= 0:
            continue
        np.add.at(hist, t + k_r, out * g_r)
        add = np.bincount(flat, weights=(out[:, None] * F).ravel(), minlength=k_max * P)
        inc[t:t + k_max] += add.reshape(k_max, P)
    return hist[:n_bins]


def metrics(e):
    """T30, EDT (s), C50 (dB), Ts (ms) of an energy echogram in 1 ms bins."""
    k0 = int(np.argmax(e > 0))
    e = e[k0:]
    edc = 10.0 * np.log10(np.maximum(np.cumsum(e[::-1])[::-1] / e.sum(), 1e-30))
    fs = int(round(1.0 / BIN))
    return {"T30": rm._rt_from_edc(edc, fs, -5.0, -35.0), "EDT": rm._rt_from_edc(edc, fs, 0.0, -10.0),
            "C50": 10.0 * np.log10(e[:50].sum() / e[50:].sum()),
            "Ts": 1000.0 * (np.arange(len(e)) * BIN * e).sum() / e.sum(), "E": e.sum()}


def infinite_dims(scene, *args, **kwargs):
    """Surface sizes forced to 1e6 m: rbs_scattering then adds no finite-size scattering
    and only its two estimator corrections remain (BRAS/bras_estimator_only.py)."""
    n = len(scene.tri_verts)
    return np.full(n, 1e6, np.float32), np.full(n, 1e6, np.float32)


def engine_hist(alpha, scattering, t_max, rbs=None):
    """rbs: None = the engine default, else force rbs_scattering on / off."""
    mats = [{"absorption": np.full(7, a), "scattering": np.full(7, scattering),
             "transmission": np.zeros(7)} for a in alpha]
    sc = make_shoebox(SIZE, wall_material=mats[2], floor_material=mats[0], ceiling_material=mats[1])
    cfg = BakeConfig(n_rays=300_000, max_time=t_max, temperature_c=T_C, humidity_percent=HR,
                     complex_reflection=False, diffraction=False, normalize=False, seed=1234,
                     max_bounces=1000)
    if rbs is not None:
        cfg.rbs_scattering = bool(rbs)
    res = bake(sc, SourceDef(name="s", position=SRC), ReceiverDef(name="r", position=RCV), cfg)
    return res.hist_total[BAND]


def eyring(alpha, air_db_m):
    S = np.array([SIZE[0] * SIZE[1], SIZE[0] * SIZE[1],
                  2 * (SIZE[0] * SIZE[2] + SIZE[1] * SIZE[2])])
    V = SIZE[0] * SIZE[1] * SIZE[2]
    a = (S * np.asarray(alpha)).sum() / S.sum()
    m = np.log(10.0) / 10.0 * air_db_m
    return 0.161 * V / (-S.sum() * np.log(1 - a) + 4 * m * V)


def main():
    c = speed_of_sound(T_C)
    air = float(air_attenuation_db_per_m(temperature_c=T_C, humidity_percent=HR)[BAND])
    # surfaces of infinite size: the estimator alone, without finite-size scattering
    from auralis.engine import geometry
    geometry.surface_dims = infinite_dims
    rbs = False if "--old" in sys.argv else None
    print("estimator: %s" % ("OLD (image sources at full energy)" if rbs is False
                             else "engine default (image sources at their specular share)"))
    refs = "--no-refs" not in sys.argv
    scat = [float(x) for x in sys.argv[1:] if not x.startswith("--")] or [0.0, 0.1, 0.3, 1.0]
    print("box %g x %g x %g m, 1 kHz, air %.4f dB/m; T30 EDT (s), C50 (dB), Ts (ms), "
          "total energy (dB rel. exact)" % (SIZE + (air,)))
    for name, alpha in CASES.items():
        t_max = 6.0
        ex = metrics(exact_hist(alpha, t_max, air, c))
        print("\n%s (floor %.2f, ceiling %.2f, walls %.2f), Eyring %.2f s"
              % ((name,) + alpha + (eyring(alpha, air),)))
        print("  exact specular        T30 %.2f  EDT %.2f  C50 %+.2f  Ts %5.1f" % (
            ex["T30"], ex["EDT"], ex["C50"], ex["Ts"]))
        if refs:
            rd = metrics(radiosity_hist(alpha, t_max, air, c, h=1.34))
            print("  radiosity (Lambert)   T30 %.2f  EDT %.2f  C50 %+.2f  Ts %5.1f  E %+.2f dB" % (
                rd["T30"], rd["EDT"], rd["C50"], rd["Ts"], 10 * np.log10(rd["E"] / ex["E"])))
        for s in scat:
            en = metrics(engine_hist(alpha, s, t_max, rbs))
            print("  engine, scattering %.1f T30 %.2f  EDT %.2f  C50 %+.2f  Ts %5.1f  E %+.2f dB" % (
                s, en["T30"], en["EDT"], en["C50"], en["Ts"], 10 * np.log10(en["E"] / ex["E"])))


if __name__ == "__main__":
    main()
