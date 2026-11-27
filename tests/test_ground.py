# -*- coding: utf-8 -*-
"""Outdoor ground: the Faddeeva function, the spherical-wave reflection coefficient, the
ground classes, and the engine's ground effect against the analytic solution."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine import ground, poromat, reflection                       # noqa: E402
from auralis.engine.air import config_speed_of_sound                        # noqa: E402
from auralis.engine.geometry import SceneData                               # noqa: E402
from auralis.engine.pipeline import BakeConfig, ReceiverDef, SourceDef, bake  # noqa: E402

FS = 48000


def _dawson(x):
    """Dawson's integral by the trapezoidal rule (reference for w on the real axis)."""
    t = np.linspace(0.0, x, 200001)
    return math.exp(-x * x) * np.trapz(np.exp(t * t), t)


def test_faddeeva_matches_closed_forms_on_both_axes():
    assert abs(ground.faddeeva(0.0) - 1.0) < 1e-13
    for y in (0.3, 1.0, 3.0):
        assert abs(ground.faddeeva(1j * y) - math.exp(y * y) * math.erfc(y)) < 1e-12
    for x in (0.5, 1.0, 2.5):
        ref = math.exp(-x * x) + 2j / math.sqrt(math.pi) * _dawson(x)
        assert abs(ground.faddeeva(x) - ref) < 1e-9


def test_faddeeva_symmetries_hold_in_the_lower_half_plane():
    z = np.array([0.7 + 0.2j, -1.3 + 2.0j, 2.0 - 0.5j, -0.4 - 1.1j])
    w = ground.faddeeva(z)
    assert np.allclose(ground.faddeeva(-np.conj(z)), np.conj(w), atol=1e-12)
    assert np.allclose(ground.faddeeva(-z), 2.0 * np.exp(-z * z) - w, atol=1e-11)


def test_spherical_coefficient_limits():
    f = np.geomspace(20.0, 10000.0, 50)
    beta = poromat.gamma_normal(ground.ground_spec("C"), f)
    assert np.allclose(ground.spherical_reflection(f, np.zeros_like(f), 0.2, 30.0, 343.0), 1.0)
    # far and steep: the plane-wave coefficient
    q = ground.spherical_reflection(f, beta, 0.9, 1e5, 343.0)
    assert np.abs(q - (0.9 - beta) / (0.9 + beta)).max() < 1e-3
    # numerical distance -> 0: pressure doubling
    q0 = ground.spherical_reflection(np.array([1e-5]), np.array([0.3 + 0.2j]), 0.1, 1.0, 343.0)
    assert abs(q0[0] - 1.0) < 1e-3


def test_ground_classes_are_passive_and_ordered():
    f = np.geomspace(50.0, 5000.0, 30)
    prev = None
    for key in "ABCDEFG":
        zs = poromat.surface_impedance(ground.ground_spec(key), f)
        assert np.all(zs.real > 0.0)
        a = poromat.alpha_bands(ground.ground_spec(key))
        if prev is not None:
            assert np.all(a < prev)            # a harder ground absorbs less
        prev = a
    with pytest.raises(ValueError):
        ground.ground_spec("Z")


def test_semi_infinite_layer_equals_a_deep_layer():
    f = np.geomspace(50.0, 4000.0, 30)
    deep = {"layers": [{"model": "miki", "sigma": 200e3, "d": 5.0}]}
    half = {"layers": [{"model": "miki", "sigma": 200e3, "semi_infinite": True}]}
    for mu in (1.0, 0.3):
        a, b = poromat.surface_impedance(deep, f, mu), poromat.surface_impedance(half, f, mu)
        assert np.abs(a / b - 1.0).max() < 1e-9


def test_long_event_filter_reproduces_the_coefficient():
    spec = ground.ground_spec("A")
    tab = reflection.build_admittance_table(poromat.alpha_bands(spec)[None], [spec])
    r2, mu, c = 50.0, 0.05, 343.0
    b = [(0, mu, r2, c)]
    h, pre = reflection.long_event_fir(reflection.band_R_mag(b, tab), b, tab, FS)
    f = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    H = np.array([np.sum(h * np.exp(-2j * np.pi * x * (np.arange(len(h)) - pre) / FS)) for x in f])
    q = ground.spherical_reflection(f, poromat.gamma_normal(spec, f), mu, r2, c)
    assert np.abs(H - q).max() < 1e-2


def _ground_scene(spec, d, z):
    a, b, cc, dd = [-3.0, -6.0, z], [d + 3.0, -6.0, z], [d + 3.0, 6.0, z], [-3.0, 6.0, z]
    tv = np.array([[a, b, cc], [a, cc, dd]], dtype=np.float64)
    alpha = poromat.alpha_bands(spec)[None]
    return SceneData(tv, np.zeros(2, np.int32), alpha, np.zeros_like(alpha),
                     np.zeros_like(alpha), mat_phys=[spec], mat_names=["ground"])


def _thirds(h, ref, fc):
    n = 1 << int(np.ceil(np.log2(max(len(h), len(ref)) * 2)))
    H, R = np.fft.rfft(h, n), np.fft.rfft(ref, n)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    out = []
    for x in fc:
        m = (f >= x * 2 ** (-1 / 6)) & (f < x * 2 ** (1 / 6))
        out.append(10 * np.log10(np.sum(np.abs(H[m]) ** 2) / np.sum(np.abs(R[m]) ** 2)))
    return np.array(out)


def test_engine_ground_effect_matches_the_analytic_solution():
    spec = ground.ground_spec("C")
    d, hs, hr = 25.0, 1.0, 1.5
    src, rcv = SourceDef("S", np.array([0.0, 0.0, hs])), ReceiverDef("R", np.array([d, 0.0, hr]))
    cfg = BakeConfig(fs=FS, n_rays=500, ism_order=1, diffraction=False, max_time=0.2,
                     normalize=False, ambisonic_order=1, rbs_scattering=False,
                     spherical_reflection=True, n_threads=2)
    c = config_speed_of_sound(cfg)
    free = np.asarray(bake(_ground_scene(spec, d, -500.0), src, rcv, cfg).ambi[:, 0])
    h = np.asarray(bake(_ground_scene(spec, d, 0.0), src, rcv, cfg).ambi[:, 0])
    fc = 1000.0 * 2.0 ** (np.arange(-13, 6) / 3.0)          # 50 Hz - 3.2 kHz
    eng = _thirds(h, free, fc)
    f = np.linspace(1.0, 6000.0, 60001)
    r1, r2 = np.hypot(d, hr - hs), np.hypot(d, hr + hs)
    k = 2 * np.pi * f / c
    q = ground.spherical_reflection(f, poromat.gamma_normal(spec, f), (hs + hr) / r2, r2, c)
    p = 1.0 + q * r1 / r2 * np.exp(-1j * k * (r2 - r1))
    ana = np.array([10 * np.log10(np.mean(np.abs(p[(f >= x * 2 ** (-1 / 6))
                                                   & (f < x * 2 ** (1 / 6))]) ** 2)) for x in fc])
    assert np.abs(eng - ana).max() < 0.3
    assert eng.min() < -8.0                                  # the ground dip is there


def _screen_scene(spec):
    def quad(a, b, c, d):
        return [[a, b, c], [a, c, d]]
    tv = np.array(quad([-30, -30, 0], [40, -30, 0], [40, 30, 0], [-30, 30, 0])
                  + quad([5, -20, 0], [5, 20, 0], [5, 20, 2], [5, -20, 2]), float)
    al = np.vstack([poromat.alpha_bands(spec), np.zeros(7)])
    return SceneData(tv, np.array([0, 0, 1, 1], np.int32), al, np.zeros_like(al),
                     np.zeros_like(al), mat_phys=[spec, None], mat_names=["ground", "screen"])


def test_a_screen_on_a_ground_gets_the_four_paths():
    from auralis.engine.diffraction import compute_diffraction, extract_edges
    from auralis.engine.geometry import NumpyRaycaster, extract_planes
    sc = _screen_scene(ground.ground_spec("D"))
    planes, tp = extract_planes(sc, 48)
    tab = reflection.build_admittance_table(sc.absorption, sc.mat_phys)
    src, rcv = np.array([0.0, 0.0, 0.5]), np.array([10.0, 0.0, 1.5])
    args = (sc, NumpyRaycaster(sc), extract_edges(sc), src, rcv, np.zeros(7), 343.0)
    one = [e for e in compute_diffraction(*args) if e.time * 343.0 < 12.0]
    four = [e for e in compute_diffraction(*args, planes=planes, tri_plane=tp, adm_table=tab,
                                           spherical=True) if e.time * 343.0 < 12.0]
    assert len(one) == 1 and all(not e.exact for e in one)
    top = np.array([5.0, 0.0, 2.0])
    want = sorted(np.linalg.norm(top - s) + np.linalg.norm(top - r)
                  for s in (src, src * [1, 1, -1]) for r in (rcv, rcv * [1, 1, -1]))
    got = sorted(e.time * 343.0 for e in four)
    assert np.allclose(got, want, atol=1e-6)
    assert sorted(e.order for e in four) == [1, 2, 2, 3] and all(e.exact for e in four)


def test_four_path_render_matches_the_closed_form_sum():
    """The rendered exact events (long filters, complex UTD, spherical-wave ground) give the
    sum of the four paths written in closed form."""
    from auralis.engine.bands import interp_band_gains
    from auralis.engine.reflection import _bounce_product, _wedge_spectrum, band_R_mag
    sc = _screen_scene(ground.ground_spec("C"))
    src, rcv = np.array([0.0, 0.0, 0.5]), np.array([10.0, 0.0, 1.5])
    cfg = BakeConfig(fs=FS, n_rays=200, ism_order=2, max_time=0.1, normalize=False,
                     ambisonic_order=1, rbs_scattering=False, spherical_reflection=True,
                     n_threads=2)
    res = bake(sc, SourceDef("S", src), ReceiverDef("R", rcv), cfg)
    h = np.asarray(res.ambi[:, 0])
    tab = reflection.build_admittance_table(sc.absorption, sc.mat_phys)
    n = 1 << 16
    f = np.fft.rfftfreq(n, 1.0 / FS)
    P = np.zeros(len(f), complex)
    paths = [e for e in res.events if float(np.max(e.amps)) > 0.0]
    assert {e.kind for e in paths} == {"diffracted"} and len(paths) == 4
    for e in paths:
        d, dc = _wedge_spectrum(np.maximum(f, 1.0), e.wedge)
        mags = (band_R_mag(e.bounces, tab) if e.bounces else np.ones(7)) * dc
        rest = interp_band_gains(np.maximum(f, 1.0), e.amps / mags)
        q = _bounce_product(np.maximum(f, 1.0), e.bounces, tab) if e.bounces else 1.0
        P += rest * d * q * np.exp(-2j * np.pi * f * e.time)
    H = np.fft.rfft(h, n)
    for x in (125.0, 250.0, 500.0, 1000.0, 2000.0):
        m = (f >= x * 2 ** (-1 / 6)) & (f < x * 2 ** (1 / 6))
        e_dB = 10 * np.log10(np.sum(np.abs(H[m]) ** 2) / np.sum(np.abs(P[m]) ** 2))
        assert abs(e_dB) < 0.3, (x, e_dB)


def test_spherical_reflection_turns_itself_on_in_open_scenes_only():
    from auralis.engine.geometry import NumpyRaycaster
    from auralis.engine.pipeline import resolve_spherical

    def box(lx, ly, lz):
        v = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0], [0, 0, lz], [lx, 0, lz],
                      [lx, ly, lz], [0, ly, lz]], float)
        t = []
        for a, b, c, d in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
                           (3, 0, 4, 7)]:
            t += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
        return np.array(t)
    al = np.full((1, 7), 0.1)
    room = SceneData(box(6, 5, 3), np.zeros(12, np.int32), al, np.zeros_like(al),
                     np.zeros_like(al), mat_names=["wall"])
    field = _ground_scene(ground.ground_spec("C"), 10.0, 0.0)
    src = np.array([2.0, 2.0, 1.5])
    auto = BakeConfig()
    assert auto.spherical_reflection is None
    assert not resolve_spherical(room, auto, NumpyRaycaster(room), src)
    assert resolve_spherical(field, auto, NumpyRaycaster(field), np.array([0.0, 0.0, 1.0]))
    assert resolve_spherical(room, BakeConfig(spherical_reflection=True), None, src)
    assert not resolve_spherical(field, BakeConfig(spherical_reflection=False), None, src)


def test_roughness_coherence_follows_nord2000():
    from auralis.engine.ground import roughness_coherence
    c = 343.0
    assert np.allclose(roughness_coherence([50.0, 5000.0], 0.0, 0.3, c), 1.0)
    # continuous at both joints of the piecewise g(X), X = k sigma_r sin(psi)
    for x0 in (0.026686, 0.115448):
        f = np.array([x0 * (1 - 1e-9), x0 * (1 + 1e-9)]) * c / (2 * np.pi)
        a, b = roughness_coherence(f, 1.0, 1.0, c)
        assert abs(a - b) < 1e-6
    fr = roughness_coherence(np.geomspace(10.0, 20000.0, 400), 0.5, 0.2, c)
    assert np.all(np.diff(fr) <= 1e-12) and fr[0] > 0.99 and fr[-1] < 1e-6


def test_variable_porosity_matches_its_usual_numerical_form():
    f = np.geomspace(50.0, 5000.0, 20)
    z = poromat.variable_porosity(f, 200e3, 10.0)
    usual = 0.436 * (1 - 1j) * np.sqrt(200e3 / f) - 19.48j * 10.0 / f
    assert np.abs(z / usual - 1.0).max() < 0.01
    spec = ground.ground_spec("D", alpha_e=10.0)
    assert np.all(poromat.surface_impedance(spec, f).real > 0.0)


def test_rough_ground_matches_nord2000_in_octaves():
    """Coherent part Fr Q, incoherent part (1 - Fr^2) R^2 (Nord2000 Part 1, Eq. 137), the
    engine's diffuse share averaged over noise draws."""
    from auralis.engine.reflection import alpha_random
    spec = ground.ground_spec("D", roughness="M")
    d, hs, hr = 25.0, 1.0, 1.5
    src, rcv = SourceDef("S", np.array([0.0, 0.0, hs])), ReceiverDef("R", np.array([d, 0.0, hr]))
    base = BakeConfig(fs=FS, n_rays=200, ism_order=1, diffraction=False, max_time=0.2,
                      normalize=False, ambisonic_order=1, rbs_scattering=False,
                      spherical_reflection=True, n_threads=2)
    c = config_speed_of_sound(base)
    free = np.asarray(bake(_ground_scene(spec, d, -500.0), src, rcv, base).ambi[:, 0])
    octs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    eng = []
    for seed in range(6):
        from dataclasses import replace
        h = np.asarray(bake(_ground_scene(spec, d, 0.0), src, rcv,
                            replace(base, seed=seed)).ambi[:, 0])
        n = 1 << 16
        H, H0 = np.fft.rfft(h, n), np.fft.rfft(free, n)
        f = np.fft.rfftfreq(n, 1.0 / FS)
        eng.append([np.sum(np.abs(H[(f >= x / np.sqrt(2)) & (f < x * np.sqrt(2))]) ** 2)
                    / np.sum(np.abs(H0[(f >= x / np.sqrt(2)) & (f < x * np.sqrt(2))]) ** 2)
                    for x in octs])
    eng = 10 * np.log10(np.mean(eng, axis=0))
    f = np.linspace(20.0, 6000.0, 60000)
    r1, r2 = np.hypot(d, hr - hs), np.hypot(d, hr + hs)
    mu = (hs + hr) / r2
    beta = poromat.gamma_normal(spec, f)
    q = ground.spherical_reflection(f, beta, mu, r2, c)
    fr = ground.roughness_coherence(f, spec["roughness"], mu, c)
    e = (np.abs(1.0 + fr * q * r1 / r2 * np.exp(-2j * np.pi * f * (r2 - r1) / c)) ** 2
         + (1.0 - fr ** 2) * (1.0 - alpha_random(beta)) * (r1 / r2) ** 2)
    ref = np.array([10 * np.log10(np.mean(e[(f >= x / np.sqrt(2)) & (f < x * np.sqrt(2))]))
                    for x in octs])
    assert np.abs(eng - ref).max() < 0.6, np.round(eng - ref, 2)


def test_a_ground_through_the_absorbing_layer_stays_stable():
    """A domain box cuts a large ground: it runs into the CPML, whose cells next to it get the
    layer's damping as a volume loss. The response stays stable and nearly all its energy is
    the direct sound and the ground reflection."""
    from auralis.engine.fdtd import run_low_band
    tv = np.array([[[-1e3, -1e3, 0.0], [1e3, -1e3, 0.0], [1e3, 1e3, 0.0]],
                   [[-1e3, -1e3, 0.0], [1e3, 1e3, 0.0], [-1e3, 1e3, 0.0]]])
    sc = SceneData(tv, np.zeros(2, np.int32), np.zeros((1, 7)), np.zeros((1, 7)),
                   np.zeros((1, 7)), mat_names=["rigid"])
    src, rcv = np.array([0.0, 0.0, 1.0]), np.array([4.0, 0.0, 1.5])
    h, info = run_low_band(sc, src, rcv, fs=8000, duration=0.4, f_cross=150.0,
                           speed_of_sound=343.0, max_cells=3_000_000,
                           domain_box=((-3.0, -3.0, -0.3), (7.0, 3.0, 4.0)), pml_cells=20)
    assert info.is_open and np.all(np.isfinite(h))
    t = np.arange(len(h)) / 8000.0
    late = np.sum(h[t > 0.06] ** 2) / np.sum(h ** 2)
    assert late < 0.02
