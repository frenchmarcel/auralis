# -*- coding: utf-8 -*-
"""Tests of UTD edge diffraction: continuity at the shadow and
reflection boundaries, screen insertion loss, reciprocity, edge
extraction and sound bending in a scene."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import (  # noqa: E402
    BakeConfig, ReceiverDef, SourceDef, add_box, bake, compute_diffraction,
    extract_edges, fresnel_F, make_shoebox, utd_coefficient,
)
from auralis.engine import materials as mat_lib  # noqa: E402
from auralis.engine.air import air_attenuation_db_per_m, \
    speed_of_sound  # noqa: E402
from auralis.engine.bands import CENTERS  # noqa: E402
from auralis.engine.geometry import NumpyRaycaster, SceneData, \
    box_triangles  # noqa: E402
from auralis.engine.geometry import extract_planes  # noqa: E402
from auralis.engine.ism import compute_direct, \
    compute_image_sources  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s %s" % (name, extra))
    else:
        FAIL += 1
        print("  FAIL %s %s" % (name, extra))
        if "PYTEST_CURRENT_TEST" in os.environ:     # under pytest a failed check fails
            raise AssertionError("%s %s" % (name, extra))


C = speed_of_sound(20.0)
K = 2.0 * np.pi * np.asarray(CENTERS, float) / C


def test_coefficient():
    check("F -> 1 at large X", abs(abs(fresnel_F(25.0)) - 1.0) < 2e-3)
    check("F ~ sqrt(pi X) at small X",
          abs(abs(fresnel_F(1e-3)) / np.sqrt(np.pi * 1e-3) - 1.0) < 0.05)

    sp, s = 3.0, 4.0
    L = s * sp / (s + sp)
    GO = 1.0 / (sp + s)
    d = 1e-6
    for n, phi0, tag in ((2.0, 0.7, "half-plane"), (1.5, 0.7, "90 deg wedge")):
        jump = np.abs(
            utd_coefficient(phi0 + np.pi - d, phi0, n, K, L)
            - utd_coefficient(phi0 + np.pi + d, phi0, n, K, L)
        ) / np.sqrt(sp * s * (sp + s))
        check("jump = GO at the shadow boundary (%s)" % tag,
              np.allclose(jump / GO, 1.0, rtol=2e-3),
              "(max error %.1e)" % np.abs(jump / GO - 1).max())
    phiR = np.pi - 0.7
    jr = np.abs(utd_coefficient(phiR - d, 0.7, 1.5, K, L)
                - utd_coefficient(phiR + d, 0.7, 1.5, K, L)) \
        / np.sqrt(sp * s * (sp + s))
    check("jump = GO at the reflection boundary",
          np.allclose(jr / GO, 1.0, rtol=2e-3))
    D1 = utd_coefficient(2.5, 0.7, 1.7, K, L)
    D2 = utd_coefficient(0.7, 2.5, 1.7, K, L)
    check("coefficient reciprocity",
          np.allclose(np.abs(D1), np.abs(D2), rtol=1e-9))


def _screen_scene(quad=True, th=0.02):
    """8 m x 3 m screen standing in free space; quad=True: one rectangle
    (exact half-plane), otherwise a thin box th m thick."""
    mats = [mat_lib.get_preset("concrete")]
    if quad:
        v = np.array([[[-4, 0, 0], [4, 0, 0], [4, 0, 3]],
                      [[-4, 0, 0], [4, 0, 3], [-4, 0, 3]]], float)
        return SceneData.from_materials(v, np.zeros(2, np.int32),
                                        mats, ["concrete"])
    tv, _ = box_triangles((8.0, th, 3.0), (-4.0, -th / 2, 0.0),
                          inward=False)
    return SceneData.from_materials(tv, np.zeros(len(tv), np.int32),
                                    mats, ["concrete"])


def test_kurze_anderson():
    """Insertion loss above a screen vs Kurze-Anderson (+/-3 dB)."""
    sc = _screen_scene()
    rc = NumpyRaycaster(sc)
    air = np.zeros(7)                       # isolates pure diffraction
    edges = extract_edges(sc)
    src = np.array([0.0, -3.0, 1.2])
    worst = 0.0
    rows = []
    for hr in (1.0, 1.5, 2.0):
        rcv = np.array([0.0, 4.0, hr])
        evs = compute_diffraction(sc, rc, edges, src, rcv, air, C)
        check("R z=%.1f: paths found (%d)" % (hr, len(evs)),
              len(evs) >= 1)
        d_dir = float(np.linalg.norm(rcv - src))
        e_tot = np.zeros(7)
        t_top = None
        for ev in evs:
            e_tot += ev.amps ** 2
            t_top = min(t_top, ev.time) if t_top else ev.time
        il = -10.0 * np.log10(np.maximum(e_tot * d_dir * d_dir, 1e-30))
        delta = t_top * C - d_dir           # detour of the shortest path
        Nf = 2.0 * delta * np.asarray(CENTERS) / C
        ka = 5.0 + 20.0 * np.log10(np.sqrt(2 * np.pi * Nf)
                                   / np.tanh(np.sqrt(2 * np.pi * Nf)))
        m = Nf > 0.3
        werr = float(np.abs(il - ka)[m].max())
        worst = max(worst, werr)
        rows.append((hr, Nf[m][0], Nf[m][-1], werr))
    for hr, n0, n1, we in rows:
        print("       z=%.1f  N=%.1f..%.0f  max error %.1f dB" %
              (hr, n0, n1, we))
    check("Kurze-Anderson: max error %.1f dB (< 3 dB)" % worst, worst < 3.0)


def test_extraction():
    room = make_shoebox((8.0, 6.0, 3.0))
    check("room (concave corners): 0 edge",
          len(extract_edges(room)) == 0)
    sc = make_shoebox((8.0, 6.0, 3.0))
    sc = add_box(sc, (1.0, 1.0, 1.0), (3.0, 2.0, 0.0),
                 mat_lib.get_preset("wood_panel"))
    ed = extract_edges(sc)
    n15 = [e for e in ed if abs(e.n - 1.5) < 1e-6]
    check("solid box: 12 convex edges n=1.5 (%d)" % len(n15),
          len(n15) == 12)
    lens = sorted(round(e.length, 3) for e in n15)
    check("edge lengths = 1 m", lens == [1.0] * 12)
    scr = _screen_scene(quad=True)
    eds = extract_edges(scr)
    n2 = [e for e in eds if abs(e.n - 2.0) < 1e-6]
    check("quad screen: free borders n=2 (%d)" % len(n2), len(n2) >= 3)
    thin = _screen_scene(quad=False, th=0.02)
    edt = extract_edges(thin)
    merged = [e for e in edt if abs(e.n - 2.0) < 1e-6 and e.length > 1.0]
    check("2 cm thin box: pairs merged into half-planes (%d)"
          % len(merged), len(merged) >= 3)
    top = [e for e in merged if abs(e.length - 8.0) < 0.05
           and abs(e.a[2] - 3.0) < 0.05]
    check("including the 8 m top edge at z=3", len(top) == 1)


def test_scene_continuity():
    """Thin screen in a room: the total early field stays smooth at the
    shadow boundary while the direct sound alone dies out."""
    room = make_shoebox((20.0, 14.0, 6.0))
    quad = np.array([[[8.0, 6.0, 0.0], [8.0, 8.0, 0.0], [8.0, 8.0, 2.2]],
                     [[8.0, 6.0, 0.0], [8.0, 8.0, 2.2], [8.0, 6.0, 2.2]]])
    tv = np.concatenate([room.tri_verts, quad])
    tm = np.concatenate([room.tri_mat,
                         np.full(2, room.tri_mat.max() + 1, np.int32)])
    mats = [dict(absorption=room.absorption[i],
                 scattering=room.scattering[i],
                 transmission=room.transmission[i])
            for i in range(len(room.absorption))]
    mats.append(mat_lib.get_preset("concrete"))
    sc = SceneData.from_materials(
        tv, tm, mats, list(room.mat_names) + ["screen"])
    rc = NumpyRaycaster(sc)
    air = air_attenuation_db_per_m()
    edges = extract_edges(sc)
    check("screen in a room: n=2 edges present",
          sum(1 for e in edges if abs(e.n - 2.0) < 1e-6) >= 3)
    planes, tp = extract_planes(sc, 48)
    refl = np.sqrt(sc.reflectance())
    src = np.array([3.0, 7.0, 1.5])
    ys = np.linspace(7.4, 10.6, 65)
    e_dd, e_dir, n_ev, e_isk = [], [], [], []
    for y in ys:
        rcv = np.array([13.0, y, 1.5])
        ev_d = compute_direct(sc, rc, src, rcv, air, C)
        ism = compute_image_sources(sc, planes, tp, rc, src, rcv, 1, air, C)
        dif = compute_diffraction(sc, rc, edges, src, rcv, air, C,
                                  refl_amp=refl)
        e_dir.append(float((ev_d.amps ** 2).sum()))
        e_dd.append(e_dir[-1] + sum(float((e.amps ** 2).sum())
                                    for e in dif))
        e_isk.append(sum(float((e.amps ** 2).sum()) for e in ism))
        n_ev.append(len(dif))
    e_dd = np.array(e_dd)      # direct + diffracted
    e_dir = np.array(e_dir)
    shadowed = e_dir < 1e-12
    check("the screen casts a shadow (%d/%d points)"
          % (int(shadowed.sum()), len(ys)), 10 < shadowed.sum() < 55)
    check("diffracted paths everywhere, shadow included (min %d)"
          % min(n_ev), min(n_ev) >= 1)
    check("direct only: abrupt cutoff",
          e_dir[shadowed].max() == 0.0 and e_dir[~shadowed].min() > 1e-6)
    # Incoherent energies: strict continuity belongs to the COHERENT
    # field (proved analytically above). Here the GO cutoff becomes ONE
    # bounded step (~8 dB instead of -inf), and the field is smooth
    # everywhere else.
    r = e_dd[1:] / np.maximum(e_dd[:-1], 1e-20)
    r = np.maximum(r, 1.0 / np.maximum(r, 1e-20))
    n_big = int((r > 1.25).sum())
    check("shadow boundary: single bounded step "
          "(max %.1f dB < 8.2, %d steps > 25 %%)"
          % (10 * np.log10(r.max()), n_big),
          r.max() < 10 ** 0.82 and n_big <= 2)
    lit = float(np.median(e_dd[~shadowed]))
    deep = float(e_dd[shadowed].min())
    il_db = 10 * np.log10(lit / deep)
    check("no dead zone: shadow floor at %.1f dB (< 26 dB)"
          % il_db, il_db < 26.0)
    check("first-order reflections do exist elsewhere",
          min(e_isk) > 1e-6)

    rcv = np.array([13.0, 7.0, 1.5])
    e1 = sum(float((e.amps ** 2).sum()) for e in compute_diffraction(
        sc, rc, edges, src, rcv, air, C, refl_amp=refl))
    e2 = sum(float((e.amps ** 2).sum()) for e in compute_diffraction(
        sc, rc, edges, rcv, src, air, C, refl_amp=refl))
    check("reciprocity S <-> R", e2 > 0 and abs(e1 / e2 - 1.0) < 1e-6)

    # Known first-order limit: around a THICK obstacle reaching the
    # ceiling, bending around it needs two diffractions -> no path here.
    sc2 = make_shoebox((20.0, 14.0, 6.0))
    sc2 = add_box(sc2, (0.6, 0.6, 6.0), (9.7, 6.7, 0.0),
                  mat_lib.get_preset("concrete"))
    rc2 = NumpyRaycaster(sc2)
    ed2 = extract_edges(sc2)
    evs2 = compute_diffraction(sc2, rc2, ed2, np.array([4.0, 7.0, 1.7]),
                               np.array([16.0, 7.0, 1.7]), air, C)
    check("thick pillar up to the ceiling: deep shadow = 0 path "
          "at first order (documented limit)", len(evs2) == 0)


def test_open_doorway():
    """Brick building, open door, off-axis receiver: direct blocked,
    diffraction must fill the 24-35 ms window."""
    L, W, H, T = 8.0, 6.0, 3.0, 0.05
    DY0, DY1, DZ = 2.5, 3.5, 2.1
    names = ["brick", "concrete", "plaster"]
    mats = [mat_lib.get_preset(n) for n in names]
    BR, BE, PL = range(3)
    boxes = [((L + 4, W, T), (0, 0, -T), BE), ((L, W, T), (0, 0, H), PL),
             ((L, T, H), (0, -T, 0), BR), ((L, T, H), (0, W, 0), BR),
             ((T, W, H), (-T, 0, 0), BR),
             ((T, W, H - DZ), (L, 0, DZ), BR),
             ((T, DY0, DZ), (L, 0, 0), BR),
             ((T, W - DY1, DZ), (L, DY1, 0), BR)]
    tv, tm = [], []
    for size, orig, mi in boxes:
        t3, _ = box_triangles(size, orig, inward=False)
        tv.append(t3)
        tm.append(np.full(len(t3), mi, np.int32))
    sc = SceneData.from_materials(np.concatenate(tv), np.concatenate(tm),
                                  mats, names)
    src = SourceDef("outside", (L + 3.0, 3.0, 1.6))
    rcv = ReceiverDef("inside", (3.0, 0.9, 1.5), forward=(1, 0, 0))

    r0 = bake(sc, src, rcv,
              BakeConfig(n_rays=6000, seed=3, max_time=0.6,
                         diffraction=False))
    r1 = bake(sc, src, rcv,
              BakeConfig(n_rays=6000, seed=3, max_time=0.6))

    def window(res, tmax=0.035):
        return sum(float((e.amps ** 2).sum())
                   for e in res.events if e.time < tmax)
    nd = int(r1.stats["diff_events"])
    # first-order edge paths; an open scene also gets the paths that reflect on a plane
    # before or after the edge (spherical_reflection), later by construction
    diff_ev = [e for e in r1.events if e.kind == "diffracted" and e.order == 1]
    spec_t = min(e.time for e in r1.events
                 if e.kind == "specular" and (e.amps ** 2).sum() > 1e-6)
    check("open door: diffracted paths (%d, edges=%d)"
          % (nd, int(r1.stats["diff_edges"])), nd >= 4)
    check("window <35 ms empty without diffraction (%.1e)" % window(r0),
          window(r0) < 1e-9)
    check("sound bends around: window <35 ms filled (%.1e)" % window(r1),
          window(r1) > 1e-3)
    check("every diffracted path precedes the 1st reflection (%.1f ms)"
          % (1e3 * spec_t), all(e.time < spec_t for e in diff_ev))
    dmin = float(np.linalg.norm(np.asarray(src.position)
                                - np.asarray(rcv.position)))
    check("diffracted times > direct flight",
          all(e.time * C > dmin - 1e-6 for e in diff_ev))
    check("stable bake (finite IR, direct first)",
          np.isfinite(r1.ir).all() and r1.events[0].kind == "direct")


if __name__ == "__main__":
    print("== UTD coefficient ==")
    test_coefficient()
    print("== screen: Kurze-Anderson ==")
    test_kurze_anderson()
    print("== edge extraction ==")
    test_extraction()
    print("== in-scene continuity (pillar) ==")
    test_scene_continuity()
    print("== open door ==")
    test_open_doorway()
    print("\n%d ok, %d failures" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
