"""Wave-band checks of the physical wall filters (GPU when available):
  stability  every surface of a 7 x 5.5 x 3.2 m box made of one physical model; wave-band
             energy per 200 ms window (it must keep decaying);
  truth      a furnished room whose materials are known physical models (felt floor, tile
             over a 200 mm plenum, curtain): T30 / EDT / C50 at 63-250 Hz from their alphas
             alone, alpha-only fit against matinfer inference, errors against the truth.
             Default IR length 4 s: the first modes (22-28 Hz) ring for 3-5 s behind walls
             that are rigid at DC, and cutting them at 1.6 s leaks into the 63 Hz band.
python tools/physical_wall_check.py stability [model ...] | truth [duration_s]"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine import matinfer, rirmetrics as rm  # noqa: E402
from auralis.engine.geometry import box_triangles  # noqa: E402
from auralis.engine.pipeline import ReceiverDef, SourceDef, bake  # noqa: E402
from auralis.engine.project import build_config, build_scene  # noqa: E402

SIZE = (7.0, 5.5, 3.2)
MODELS = {"felt": matinfer.porous(20000, 30, 0), "tile_plenum": matinfer.porous(25000, 20, 200),
          "curtain": matinfer.screen(800, 0.01, 120), "plasterboard": matinfer.panel(10.0, 70.0, 4.0)}
PAIRS = [([1.6, 1.3, 1.5], [5.1, 4.0, 1.2]), ([5.5, 1.1, 1.7], [2.0, 3.9, 1.3]),
         ([3.4, 2.6, 1.1], [6.2, 4.6, 1.6])]


def _manifest(mats, tri_mat, infer, duration):
    tris, _names = box_triangles(SIZE, inward=True)
    return {"fs": 48000, "duration": duration,
            "engine": {"fdtd": True, "fdtd_fcross": 300.0, "ambisonic_order": 1, "n_rays": 30000,
                       "infer_materials": infer},
            "materials": mats, "geometry": {"triangles": tris.astype(float).tolist(), "tri_mat": tri_mat}}


def _run(man, pair):
    sc, cfg = build_scene(man, ""), build_config(man)
    s, r = pair
    out = bake(sc, SourceDef(name="s", position=np.array(s)), ReceiverDef(name="r", position=np.array(r)), cfg)
    w = np.asarray(out.ambi[:, 0], float)
    if not np.isfinite(w).all():
        raise RuntimeError("non-finite output")
    return out, w


def stability(only=None):
    for name, spec in MODELS.items():
        if only and name not in only:
            continue
        mats = {"w": {"absorption": matinfer.band_alpha(spec).tolist(), "phys": spec}}
        out, w = _run(_manifest(mats, [0] * 12, False, 2.0), PAIRS[0])
        low = np.asarray(out.omni_parts["wave"], float) if out.omni_parts and "wave" in out.omni_parts else w
        e = [float(np.sum(low[int(t * out.fs):int((t + 0.2) * out.fs)] ** 2)) for t in (0.0, 0.4, 0.8, 1.2, 1.6)]
        print("%-12s wave-band energy per 200 ms (dB): %s" % (name, np.round(10 * np.log10(np.maximum(e, 1e-30) / e[0]), 1)), flush=True)


def truth(duration=4.0):
    tris, names = box_triangles(SIZE, inward=True)
    tri_mat = [{"floor": 0, "ceiling": 1}.get(n, 2) for n in names]
    tri_mat[8] = tri_mat[9] = 3                     # two wall triangles: plaster (hard)
    order = ["felt", "tile_plenum", "curtain", "plaster"]
    plaster = [0.013, 0.015, 0.02, 0.03, 0.04, 0.05, 0.05]
    res = {}
    for mode in ("truth", "alpha_fit", "inferred"):
        mats = {}
        for n in order:
            if n not in MODELS and n != "plaster":
                continue
            if n == "plaster":
                mats[n] = {"absorption": plaster}
            else:
                mats[n] = {"absorption": matinfer.band_alpha(MODELS[n]).tolist()}
                if mode == "truth":
                    mats[n]["phys"] = MODELS[n]
        man = _manifest(mats, tri_mat, mode == "inferred", duration)
        acc = []
        for pair in PAIRS:
            out, w = _run(man, pair)
            mm = rm.band_metrics_at(w, out.fs, [63.0, 125.0, 250.0])
            acc.append([mm["T30"], mm["EDT"], mm["C50"]])
        res[mode] = np.array(acc, float)
    for k, name in enumerate(("T30", "EDT", "C50")):
        t = res["truth"][:, k]
        for mode in ("alpha_fit", "inferred"):
            d = res[mode][:, k] - t
            err = np.mean(np.abs(d / t if name != "C50" else d), axis=0)
            print("%-4s %-9s mean error vs truth at 63/125/250 Hz: %s" % (name, mode, np.round(err, 3)))
    for mode in res:
        print("%-9s T30 63/125/250 per pair: %s" % (mode, np.round(res[mode][:, 0], 3).tolist()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "truth":
        truth(float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
    else:
        stability(sys.argv[2:] or None)
