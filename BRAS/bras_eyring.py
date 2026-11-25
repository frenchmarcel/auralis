#!/usr/bin/env python3
"""Sabine / Eyring reverberation time of a room bench manifest (closed OBJ, octave absorption,
ISO 9613-1 air at the manifest's temperature and humidity), against the simulated and
measured T30 of a bench report. Usage: bras_eyring.py manifest.json [report.csv]"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from auralis.engine.air import air_attenuation_db_per_m, speed_of_sound  # noqa: E402

OCTAVES = [125, 250, 500, 1000, 2000, 4000, 8000]
LABELS = ["125", "250", "500", "1k", "2k", "4k", "8k"]


def obj_areas_volume(path):
    """({material: area m2}, |enclosed volume| m3) of a triangulated OBJ."""
    v, tris, area, mat = [], [], {}, None
    for ln in open(path):
        if ln.startswith("v "):
            v.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("usemtl"):
            mat = ln.split()[1]
        elif ln.startswith("f "):
            idx = [int(x.split("/")[0]) - 1 for x in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                t = np.array([v[idx[0]], v[idx[k]], v[idx[k + 1]]])
                area[mat] = area.get(mat, 0.0) + 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
                tris.append(t)
    t = np.array(tris)
    return area, abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


def reverberation(man_path):
    """(V, S, mean alpha, Sabine, Eyring) per octave, air absorption included."""
    man = json.load(open(man_path))
    area, V = obj_areas_volume(os.path.join(os.path.dirname(man_path), man["geometry"]["obj"]))
    missing = sorted(set(area) - set(man["materials"]))
    if missing:
        print("WARNING: OBJ materials without absorption, counted rigid: %s" % ", ".join(missing))
    S = sum(area.values())
    e = man["engine"]
    t_c, h = e.get("temperature_c", 20.0), e.get("humidity_percent", 50.0)
    c = speed_of_sound(t_c)
    m = np.asarray(air_attenuation_db_per_m(np.array(OCTAVES, float), t_c, h)) / (10 * np.log10(np.e))
    A = np.array([sum(a * man["materials"][k]["absorption"][i] for k, a in area.items()
                      if k in man["materials"]) for i in range(len(OCTAVES))])
    k = 24 * np.log(10) * V / c
    alpha = A / S
    return V, S, alpha, k / (A + 4 * m * V), k / (-S * np.log(1 - alpha) + 4 * m * V)


def report_t30(path):
    """Mean simulated and measured T30 per octave over the pairs of a bench report."""
    rows = list(csv.DictReader(open(path)))
    out = []
    for key in ("T30_sim", "T30_ref"):
        out.append([np.mean([float(r[key]) for r in rows if r["bande"] == b and r[key]])
                    if any(r["bande"] == b for r in rows) else np.nan for b in LABELS])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("report", nargs="?")
    a = ap.parse_args()
    V, S, alpha, sab, eyr = reverberation(a.manifest)
    print("%s: V %.1f m3, S %.1f m2" % (os.path.basename(a.manifest), V, S))
    print("band        " + "".join("%8s" % b for b in LABELS))
    print("alpha       " + "".join("%8.3f" % x for x in alpha))
    print("Sabine      " + "".join("%8.2f" % x for x in sab))
    print("Eyring      " + "".join("%8.2f" % x for x in eyr))
    if a.report:
        sim, ref = report_t30(a.report)
        print("T30 meas.   " + "".join("%8.2f" % x for x in ref))
        print("T30 sim.    " + "".join("%8.2f" % x for x in sim))
        print("sim/Eyring  " + "".join("%+7.0f%%" % x for x in 100 * (sim / eyr - 1)))
        print("meas/Eyring " + "".join("%+7.0f%%" % x for x in 100 * (ref / eyr - 1)))


if __name__ == "__main__":
    main()
