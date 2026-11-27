# -*- coding: utf-8 -*-
"""Physical material models: porous media (Delany-Bazley, Miki, JCA) and multilayer
stacks through transfer matrices, rigid backing, extended reaction.
e^{+jwt} convention; provides Zs, R(theta, f), diffuse alpha and gamma."""
from __future__ import annotations

import numpy as np

from .bands import CENTERS, N_BANDS

# Air at 20 degC, 1 atm
RHO0 = 1.204          # kg/m^3
C0 = 343.0            # m/s
ETA = 1.81e-5         # Pa.s (dynamic viscosity)
PR = 0.71             # Prandtl number
GAM = 1.4             # ratio of specific heats
P0 = 101325.0         # Pa
Z0 = RHO0 * C0

# Gauss-Legendre quadrature over mu = cos(theta) in (0, 1]
_MU, _WMU = np.polynomial.legendre.leggauss(32)
_MU = 0.5 * (_MU + 1.0)
_WMU = 0.5 * _WMU


# ---------------------------------------------------------------------------
# Equivalent fluids
# ---------------------------------------------------------------------------

def miki(f, sigma: float):
    """(Zc, k) of the Miki model. f: Hz (array), sigma: Pa.s/m^2."""
    f = np.asarray(f, dtype=np.float64)
    X = np.maximum(f, 1e-3) / float(sigma)
    Zc = Z0 * (1.0 + 0.070 * X ** -0.632 - 1j * 0.107 * X ** -0.632)
    k = (2.0 * np.pi * f / C0) * (1.0 + 0.109 * X ** -0.618
                                  - 1j * 0.160 * X ** -0.618)
    return Zc, k


def delany_bazley(f, sigma: float):
    """(Zc, k) of the Delany-Bazley model. f: Hz (array), sigma: Pa.s/m^2.

    The model that defines the effective flow resistivities of outdoor ground classes
    (Nord2000, Harmonoise)."""
    f = np.asarray(f, dtype=np.float64)
    X = RHO0 * np.maximum(f, 1e-3) / float(sigma)
    Zc = Z0 * (1.0 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732)
    k = (2.0 * np.pi * f / C0) * (1.0 + 0.0978 * X ** -0.700
                                  - 1j * 0.189 * X ** -0.595)
    return Zc, k


def jca(f, phi: float, sigma: float, alpha_inf: float,
        lam_v: float, lam_t: float):
    """(Zc, k) of the Johnson-Champoux-Allard model. phi: porosity;
    sigma: Pa.s/m^2; alpha_inf >= 1; lam_v, lam_t: lengths (m)."""
    f = np.asarray(f, dtype=np.float64)
    w = 2.0 * np.pi * np.maximum(f, 1e-3)
    jw = 1j * w
    # effective density (Johnson)
    G = np.sqrt(1.0 + jw * (4.0 * alpha_inf ** 2 * ETA * RHO0)
                / (sigma ** 2 * lam_v ** 2 * phi ** 2))
    rho = (alpha_inf * RHO0 / phi) * (1.0 + (sigma * phi)
                                      / (jw * RHO0 * alpha_inf) * G)
    # bulk modulus (Champoux-Allard)
    Gp = np.sqrt(1.0 + jw * (RHO0 * PR * lam_t ** 2) / (16.0 * ETA))
    K = (GAM * P0 / phi) / (
        GAM - (GAM - 1.0)
        / (1.0 + (8.0 * ETA) / (jw * RHO0 * PR * lam_t ** 2) * Gp))
    k = w * np.sqrt(rho / K)
    Zc = np.sqrt(rho * K)
    # physical branch: Re(Zc) > 0, Im(k) <= 0 (attenuation)
    Zc = np.where(Zc.real < 0.0, -Zc, Zc)
    k = np.where(k.imag > 0.0, -k, k)
    return Zc, k


def _layer_fluid(f, layer: dict):
    """(Zc, k) of one layer: {"model": "air" | "delany_bazley" | "miki" | "jca", ...}."""
    m = layer.get("model", "air")
    if m == "air":
        f = np.asarray(f, dtype=np.float64)
        return (np.full(f.shape, Z0, dtype=np.complex128),
                (2.0 * np.pi * f / C0).astype(np.complex128))
    if m == "miki":
        return miki(f, float(layer["sigma"]))
    if m == "delany_bazley":
        return delany_bazley(f, float(layer["sigma"]))
    if m == "jca":
        return jca(f, float(layer.get("phi", 0.98)),
                   float(layer["sigma"]),
                   float(layer.get("alpha_inf", 1.0)),
                   float(layer.get("lam_v", 100e-6)),
                   float(layer.get("lam_t", 200e-6)))
    raise ValueError("unknown layer model: %r" % m)


# ---------------------------------------------------------------------------
# Stack: oblique transfer matrices, rigid backing
# ---------------------------------------------------------------------------

def _safe_sin_over(x):
    """Complex sin(x)/x, stable at 0 (series up to x^4)."""
    x = np.asarray(x, dtype=np.complex128)
    small = np.abs(x) < 1e-4
    out = np.empty_like(x)
    xs = x[small]
    out[small] = 1.0 - xs ** 2 / 6.0 + xs ** 4 / 120.0
    out[~small] = np.sin(x[~small]) / x[~small]
    return out


def surface_impedance(spec: dict, f, mu: float = 1.0):
    """NORMAL surface impedance Zs(f) at incidence cos(theta) = mu, with
    rigid backing. spec = {"layers": [{"model", "d" (m), ...} front->back]}.
    A last layer with "semi_infinite": true has no backing (a deep ground)."""
    f = np.asarray(f, dtype=np.float64)
    w = 2.0 * np.pi * np.maximum(f, 1e-3)
    mu = float(np.clip(mu, 1e-6, 1.0))
    kx = (w / C0) * np.sqrt(max(0.0, 1.0 - mu * mu))
    # cumulative matrix T = T_1 ... T_n (front -> backing)
    A = np.ones(f.shape, dtype=np.complex128)
    B = np.zeros(f.shape, dtype=np.complex128)
    Cc = np.zeros(f.shape, dtype=np.complex128)
    D = np.ones(f.shape, dtype=np.complex128)
    for layer in spec.get("layers", []):
        m = str(layer.get("model", "air")).lower()
        if m in ("perforated", "slotted", "mpp"):
            Zp = sheet_impedance(layer, w)
            A, B, Cc, D = A, A * Zp + B, Cc, Cc * Zp + D
            continue
        if m == "screen":
            # thin rigid-frame resistive screen (film, fleece): series Zs = r + j w m
            # with r the flow resistance (Pa.s/m) and m the inertia of the air in its
            # pores; unlike "panel", never turned into a resonant FDTD section
            Zp = float(layer.get("r", 0.0)) + 1j * w * float(layer.get("m", 0.0))
            A, B, Cc, D = A, A * Zp + B, Cc, Cc * Zp + D
            continue
        if m == "panel":
            # airtight LIMP panel (plaster, glazing, door, wood):
            # SERIES impedance Zp = j w m + r + kk / (j w). Angle
            # independent (mass law), valid below the coincidence
            # frequency (~2.5-3 kHz for 12.5 mm plaster) - the wave
            # band stops well before it. The air cavity spring comes
            # from the "air" layer behind; kk (N/m3) models the
            # MOUNTING stiffness (framing, sealant) when known.
            pm = float(layer["m"])
            pr = float(layer.get("r", 0.0))
            kk = float(layer.get("k", 0.0))
            Zp = 1j * w * pm + pr + (kk / (1j * w) if kk > 0.0 else 0.0)
            A, B, Cc, D = A, A * Zp + B, Cc, Cc * Zp + D
            continue
        if layer.get("semi_infinite"):
            # the wave never comes back: the stack ends on the medium's own impedance
            Zc, k = _layer_fluid(f, layer)
            kz = np.sqrt(k * k - kx * kx)
            kz = np.where(kz.imag > 0.0, -kz, kz)
            Zt = Zc * k / kz
            with np.errstate(divide="ignore", invalid="ignore"):
                return (A * Zt + B) / (Cc * Zt + D)
        d = float(layer.get("d", 0.0))
        if d <= 0.0:
            continue
        Zc, k = _layer_fluid(f, layer)
        kz2 = k * k - kx * kx
        kz = np.sqrt(kz2)
        kz = np.where(kz.imag > 0.0, -kz, kz)     # damped evanescent wave
        rho_w = Zc * k                            # w * rho_eff
        cs = np.cos(kz * d)
        so = _safe_sin_over(kz * d) * d           # sin(kz d)/kz
        t11 = cs
        # forms stable as kz -> 0: sin(kz d)/kz = so
        t12 = 1j * rho_w * so
        t21 = 1j * (kz * kz) * so / rho_w
        t22 = cs
        A, B, Cc, D = (A * t11 + B * t21, A * t12 + B * t22,
                       Cc * t11 + D * t21, Cc * t12 + D * t22)
    # rigid backing: v = 0  ->  Zs = T11 / T21
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(np.abs(Cc) > 1e-300, Cc, 1e-300 + 0j)
        Zs = np.where(np.abs(Cc) > 1e-300, A / den,
                      np.inf * (1.0 + 0j))
    return Zs


def sheet_impedance(layer: dict, w):
    """Series impedance (Pa.s/m) of a rigid perforated, slotted or micro-perforated sheet.
    perforated: holes d (m) in a plate t (m), open area phi; Cox & D'Antonio, Acoustic
    Absorbers and Diffusers (2016), sec. 7.3.1, end correction of Table 7.1 (square
    pattern, Jaouen & Becot). slotted: slots of width "w_slot" (m); Kristiansen & Vigran,
    Appl. Acoust. 43 (1994). mpp: Maa, JASA 104 (1998), eqs. 5-6, sub-millimetre holes.
    Same forms as the MIT library rinaldipp/tmm (lumped "barrier" sheets)."""
    t = float(layer["t"])
    phi = float(np.clip(layer["phi"], 1e-4, 0.95))
    kind = str(layer["model"]).lower()
    if kind == "slotted":
        ws = float(layer["w_slot"])
        t_eff = t - 2.0 * ws / np.pi * np.log(np.sin(0.5 * np.pi * phi))
        rp = 0.5 * np.sqrt(2.0 * ETA * RHO0 * w) * (4.0 + 2.0 * t / ws)
        return (rp + 1j * w * RHO0 * t_eff) / phi
    d = float(layer["d"])
    if kind == "mpp":
        k = d * np.sqrt(w * RHO0 / (4.0 * ETA))
        r = 32.0 * ETA * t / (phi * d * d) * (np.sqrt(1.0 + k * k / 32.0)
                                             + np.sqrt(2.0) / 32.0 * k * d / t)
        x = w * RHO0 * t / phi * (1.0 + 1.0 / np.sqrt(9.0 + k * k / 2.0) + 0.85 * d / t)
        return r + 1j * x
    delta = 0.85 * (1.0 - 1.13 * np.sqrt(phi) - 0.09 * phi + 0.27 * phi ** 1.5)
    t_corr = t + delta * d
    rm = np.sqrt(8.0 * ETA * RHO0 * w) * (1.0 + t / d) / phi
    return rm + 1j * w * RHO0 * t_corr / phi


def reflection(spec: dict, f, mu: float = 1.0):
    """Exact complex R(theta, f) of the stack (extended reaction)."""
    mu = float(np.clip(mu, 1e-6, 1.0))
    Zs = surface_impedance(spec, f, mu)
    zn = Zs * mu / Z0
    finite = np.isfinite(zn.real) & np.isfinite(zn.imag)
    R = np.where(finite, (zn - 1.0) / (zn + 1.0), 1.0 + 0j)
    return R


def alpha_diffuse(spec: dict, f):
    """Diffuse-field absorption: Paris formula over the EXACT oblique R
    (Gauss-Legendre quadrature, 32 points over cos(theta))."""
    f = np.asarray(f, dtype=np.float64)
    acc = np.zeros(f.shape)
    for mu, wq in zip(_MU, _WMU):
        a = 1.0 - np.abs(reflection(spec, f, float(mu))) ** 2
        acc += wq * a * 2.0 * mu
    return np.clip(acc, 0.0, 1.0)


def alpha_bands(spec: dict, centers=None, pts_per_band: int = 7):
    """Diffuse-field alpha averaged per octave band (N_BANDS,)."""
    c = np.asarray(CENTERS if centers is None else centers,
                   dtype=np.float64)
    out = np.zeros(len(c))
    for b, fc in enumerate(c):
        fg = np.geomspace(fc / np.sqrt(2.0), fc * np.sqrt(2.0),
                          pts_per_band)
        out[b] = float(alpha_diffuse(spec, fg).mean())
    return out


def gamma_normal(spec: dict, f):
    """Normalized admittance target for the engine passive fitter:
    gamma(f) = rho0 c0 / Zs(f, normal incidence)."""
    Zs = surface_impedance(spec, f, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(np.isfinite(Zs) & (np.abs(Zs) > 0.0), Z0 / Zs, 0.0)
    return g


# ---------------------------------------------------------------------------
# User specifications
# ---------------------------------------------------------------------------

def panel_spec(m_kg_m2: float, cavity_mm: float, q: float = 2.0,
               r_pa_s_m: float = None, k_n_m3: float = 0.0,
               fill_sigma: float = 0.0, fill_mm: float = 0.0) -> dict:
    """Lightweight wall: limp panel (m in kg/m^2) over an air cavity
    (mm), optional Miki fill; resonance f0 ~ 60/sqrt(m d) Hz, quality q."""
    m = max(float(m_kg_m2), 0.05)
    d = max(float(cavity_mm), 1.0) * 1e-3
    k_cav = RHO0 * C0 * C0 / d
    w0 = np.sqrt((k_cav + max(float(k_n_m3), 0.0)) / m)
    r = float(r_pa_s_m) if r_pa_s_m is not None \
        else w0 * m / max(float(q), 0.2)
    layers = [{"model": "panel", "m": m, "r": r,
               "k": max(float(k_n_m3), 0.0)}]
    fd = max(float(fill_mm), 0.0) * 1e-3
    if fill_sigma > 0.0 and fd > 0.0:
        layers.append({"model": "miki", "d": min(fd, d),
                       "sigma": float(fill_sigma)})
        if d - fd > 1e-4:
            layers.append({"model": "air", "d": d - fd})
    else:
        layers.append({"model": "air", "d": d})
    return {"layers": layers, "f0_hz": float(w0 / (2.0 * np.pi))}


def spec_from_params(model: str, sigma: float, thickness_mm: float,
                     air_gap_mm: float = 0.0, phi: float = 0.98,
                     alpha_inf: float = 1.05, lam_v_um: float = 100.0,
                     lam_t_um: float = 200.0) -> dict:
    """Standard stack: [porous layer (thickness), optional air gap],
    rigid backing. User units: mm and um."""
    model = str(model).lower()
    layer = {"model": model, "d": max(float(thickness_mm), 0.1) * 1e-3,
             "sigma": float(np.clip(sigma, 200.0, 5e6))}
    if model == "jca":
        layer.update(phi=float(np.clip(phi, 0.05, 0.999)),
                     alpha_inf=float(np.clip(alpha_inf, 1.0, 4.0)),
                     lam_v=float(np.clip(lam_v_um, 5.0, 2000.0)) * 1e-6,
                     lam_t=float(np.clip(lam_t_um, 5.0, 4000.0)) * 1e-6)
    layers = [layer]
    if air_gap_mm > 0.0:
        layers.append({"model": "air", "d": float(air_gap_mm) * 1e-3})
    return {"layers": layers}


# Mineral wool: flow resistivity from the bulk density, from MATERIAL data only (never a room).
# glass: Bies & Hansen, sigma = 3.18e-9 rho^1.53 / d^2, with d = 5.34 um, the geometric mean
#   that reproduces the 10 Knauf glass mineral wools measured with an AcoustiFlow (Kovacic
#   et al., Forum Acusticum 2026, CC BY 3.0, Tables 4-5: 13-32 kg/m3) within 0.061 decade
#   rms (-23 / +24 % at worst); their Micronaire diameters are 4.4-5.4 um. At 80 kg/m3 the law
#   gives 91 kPa.s/m2 against ~70 for the round-robin glass wool (Bonfiglio 2018): +30 %.
# rock: "mineral fibre, random radii", sigma = 4.4 eta (rho/rho_f)^1.59 / a^2 (Cox & D'Antonio,
#   2nd ed., Table 5.1), rho_f = 2700 kg/m3 and an ASSUMED fibre diameter of 6 um: within
#   0.091 decade rms (bias -2 %) of 19 product values at 23-100 kg/m3, unverified datasheet
#   values relayed by one source. No primary measurement: the least sure of the two.
WOOL_LAWS = {
    "glass": {"range_kg_m3": (13.0, 32.0), "rms_decade": 0.061,
              "law": "Bies & Hansen, fibre diameter 5.34 um fitted on 10 measured Knauf glass wools",
              "source": "Kovacic, Poljansek & Puljizevic, Forum Acusticum 2026 (CC BY 3.0)"},
    "rock": {"range_kg_m3": (23.0, 100.0), "rms_decade": 0.091,
             "law": "mineral fibre, random radii (Cox & D'Antonio, Table 5.1), assumed 6 um fibres",
             "source": "checked against 19 product datasheet values (unverified, no primary "
                       "measurement)"},
}
_ETA_AIR = 1.84e-5


def wool_flow_resistivity(rho_kg_m3: float, kind: str) -> dict:
    """Flow resistivity (Pa.s/m2) of a glass or rock wool of bulk density rho, with its 1-sigma
    band (from the scatter of the law against its data) and whether rho lies outside the data
    the law was checked on (extrapolated: the band is then only a lower bound of the error)."""
    kind = str(kind).lower()
    if kind not in WOOL_LAWS:
        raise ValueError("wool must be one of %s" % ", ".join(WOOL_LAWS))
    rho = float(rho_kg_m3)
    if not (5.0 <= rho <= 250.0):
        raise ValueError("wool density must be within 5-250 kg/m3")
    if kind == "glass":
        sigma = 3.18e-9 * rho ** 1.53 / (5.34e-6) ** 2
    else:
        sigma = 4.4 * _ETA_AIR * (rho / 2700.0) ** 1.59 / (3.0e-6) ** 2
    law = WOOL_LAWS[kind]
    k = 10.0 ** law["rms_decade"]
    lo, hi = law["range_kg_m3"]
    return {"sigma": float(sigma), "sigma_low": float(sigma / k), "sigma_high": float(sigma * k),
            "uncertainty_pct": float(100.0 * (k - 1.0)), "extrapolated": not (lo <= rho <= hi),
            "range_kg_m3": [lo, hi], "law": law["law"], "source": law["source"]}


# ---------------------------------------------------------------------------
# Resonant sections (panels) for the 2nd-order FDTD faces
# ---------------------------------------------------------------------------

def normalize_spec(spec: dict) -> dict:
    """Normalizes the manifest sugar forms ({"panel"}, {"panels"},
    {"model": ...}) into a {"layers": [...]} stack."""
    if not isinstance(spec, dict):
        return spec
    if "layers" in spec:
        return spec
    if "gamma_table" in spec:
        return dict(spec)
    if "panels" in spec:
        # several resonances (e.g. low-frequency bending of the glazing
        # on its frame + mass-spring of the air gap): each entry is a
        # panel_spec; layers are concatenated, and the cavity of the
        # LAST panel closes the stack.
        subs = [normalize_spec({"panel": p}) for p in spec["panels"]]
        layers = []
        for sub in subs:
            layers.extend(sub["layers"])
        return {"layers": layers, "panels_multi": len(subs)}
    if "panel" in spec:
        p = dict(spec["panel"])
        base = panel_spec(
            float(p["m"]), float(p.get("cavity_mm", 50.0)),
            q=float(p.get("q", 2.0)),
            r_pa_s_m=p.get("r"),
            k_n_m3=float(p.get("k", 0.0)),
            fill_sigma=float(p.get("fill_sigma", 0.0)),
            fill_mm=float(p.get("fill_mm", 0.0)))
        if "extra_sections" in spec:
            base = dict(base)
            base["extra_sections"] = list(spec["extra_sections"])
        return base
    if "model" in spec:
        return spec_from_params(
            spec["model"], float(spec.get("sigma", 10000.0)),
            float(spec.get("thickness_mm", 50.0)),
            air_gap_mm=float(spec.get("air_gap_mm", 0.0)),
            phi=float(spec.get("phi", 0.98)),
            alpha_inf=float(spec.get("alpha_inf", 1.05)),
            lam_v_um=float(spec.get("lam_v_um", 100.0)),
            lam_t_um=float(spec.get("lam_t_um", 200.0)))
    return spec


def _gamma_sections(sections, f):
    """Sum of the admittances a*jw / ((jw)^2 + beta*jw + w0^2)."""
    w = 2j * np.pi * np.asarray(f, dtype=np.float64)
    g = np.zeros(w.shape, dtype=np.complex128)
    for (a, beta, w02) in sections:
        g = g + a * w / (w * w + beta * w + w02)
    return g


def _a_from_alpha_peak(alpha_pk, w0, q):
    """Inverts alpha_random(gamma) at the peak: gamma(f0) = a/beta with
    beta = w0/q. Returns a (bisection)."""
    from .reflection import alpha_random
    beta = w0 / max(q, 1e-6)
    lo, hi = 1e-4, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(alpha_random(np.array([mid + 0j]))[0]) < alpha_pk:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * beta


def _gamma_table(spec, f):
    """Measured gamma table {"f", "re", "im"} in the engine convention,
    interpolated in log frequency (flat extrapolation)."""
    tb = spec["gamma_table"]
    ft = np.asarray(tb["f"], float)
    lg = np.log(np.maximum(np.asarray(f, float), 1e-3))
    lt = np.log(ft)
    re = np.interp(lg, lt, np.asarray(tb["re"], float))
    im = np.interp(lg, lt, np.asarray(tb["im"], float))
    return re + 1j * im


def gamma_full(spec: dict, f):
    """FULL target admittance: layer transfer matrix + the declared
    additional sections (extra_sections)."""
    if "gamma_table" in spec:
        return _gamma_table(spec, f)
    g = gamma_normal(spec, f)
    ex = spec.get("extra_sections", [])
    if ex:
        secs = []
        for e in ex:
            w0 = 2.0 * np.pi * float(e["f0"])
            q = float(e.get("q", 1.5))
            a = _a_from_alpha_peak(float(e.get("alpha_peak", 0.1)), w0, q)
            secs.append((a, w0 / q, w0 * w0))
        g = g + _gamma_sections(secs, f)
    return g


def _fit_one_section(a0, b0, w0, target, f):
    """Fits ONE section (a, beta, w0^2) to a complex gamma(f) target.
    Hard bounds: f0 <= 3 f[-1], beta capped (else the descent diverges)."""
    w2_max = (2.0 * np.pi * 3.0 * float(f[-1])) ** 2
    b_max = 40.0 * np.sqrt(w2_max)
    a_max = 1e4

    def cost(p):
        a, b, w2 = p
        if a <= 0 or b <= 0 or w2 <= 0:
            return np.inf
        if w2 > w2_max or b > b_max or a > a_max:
            return np.inf
        g = _gamma_sections([(a, b, w2)], f)
        return float(np.mean(np.abs(g.real - target.real) ** 2)
                     + 0.25 * np.mean(np.abs(g.imag
                                             - target.imag) ** 2))
    best = np.array([a0, b0, w0], dtype=np.float64)
    c_best = cost(best)
    step = np.array([0.5, 0.5, 0.5])
    for _ in range(150):
        moved = False
        for i in range(3):
            for sgn in (1.0, -1.0):
                cand = best.copy()
                cand[i] *= (1.0 + sgn * step[i])
                c = cost(cand)
                if c < c_best:
                    best, c_best = cand, c
                    moved = True
        if not moved:
            step *= 0.5
            if step.max() < 1e-3:
                break
    return tuple(best)


def resonant_sections(spec: dict, f_lo: float = 30.0,
                      f_hi: float = 700.0):
    """Sections (a, beta, w02) of a 'panel', 'extra_sections' or
    'gamma_table' spec. Returns (secs, resid, f_fit) or ([], None, None)."""
    layers = spec.get("layers", [])
    panels = [la for la in layers if str(la.get("model", "")) == "panel"]
    extras = spec.get("extra_sections", [])
    if "gamma_table" in spec:
        # measured table: generic sections auto-initialized to follow
        # the COMPLEX gamma (real branches alone cannot track a
        # reactance that crosses zero).
        f = np.geomspace(float(f_lo), float(f_hi), 96)
        g_t = _gamma_table(spec, f)
        n_sec = int(spec.get("n_sections", 3))
        # init: k log-spaced sections, moderate Q (overdamped allowed:
        # one section with a large beta = two real relaxations),
        # amplitude ~ local Re; the sequential fit roughs it out, the
        # discrete polish (run_low_band) finishes it.
        f0s = np.geomspace(max(2.0*f_lo, f[2]), f_hi/1.3, n_sec)
        out = []
        g_left = g_t.copy()
        for f0 in f0s:
            w0 = 2*np.pi*float(f0)
            i0 = int(np.argmin(np.abs(f - f0)))
            a0 = max(abs(g_t.real[i0]) * w0 * 0.7, 1e-3)
            sec = _fit_one_section(a0, w0/0.6, w0*w0, g_left, f)
            f0_out = np.sqrt(sec[2]) / (2*np.pi)
            if not (0.3*f_lo <= f0_out <= 2.5*f_hi):
                continue          # section drifted out of band: useless
            g_sec = _gamma_sections([sec], f)
            if np.mean(np.abs(g_left - g_sec)**2)                     > 1.02 * np.mean(np.abs(g_left)**2):
                continue          # no improvement: reject
            out.append(sec)
            g_left = g_left - g_sec
        return out, g_left, f
    if not panels and not extras:
        return [], None, None
    ex_secs = []
    for e in extras:
        w0 = 2.0 * np.pi * float(e["f0"])
        q = float(e.get("q", 1.5))
        a = _a_from_alpha_peak(float(e.get("alpha_peak", 0.1)), w0, q)
        ex_secs.append((a, w0 / q, w0 * w0))
    if not panels:
        f = np.geomspace(float(f_lo), float(f_hi), 96)
        return ex_secs, np.zeros(f.shape, complex), f
    f = np.geomspace(float(f_lo), float(f_hi), 96)
    g_t = gamma_normal(spec, f)
    out = []
    g_left = g_t.copy()

    def fit_one(a0, b0, w0, target):
        return _fit_one_section(a0, b0, w0, target, f)

    def _unused(a0, b0, w0, target):
        def cost(p):
            a, b, w2 = p
            if a <= 0 or b <= 0 or w2 <= 0:
                return np.inf
            g = _gamma_sections([(a, b, w2)], f)
            # weighting: the REAL part (absorption) first, then the
            # reactance (it places the modes)
            return float(np.mean(np.abs(g.real - target.real) ** 2)
                         + 0.25 * np.mean(np.abs(g.imag
                                                 - target.imag) ** 2))
        best = np.array([a0, b0, w0], dtype=np.float64)
        c_best = cost(best)
        step = np.array([0.5, 0.5, 0.5])
        for _ in range(120):
            moved = False
            for i in range(3):
                for sgn in (1.0, -1.0):
                    cand = best.copy()
                    cand[i] *= (1.0 + sgn * step[i])
                    c = cost(cand)
                    if c < c_best:
                        best, c_best = cand, c
                        moved = True
            if not moved:
                step *= 0.5
                if step.max() < 1e-3:
                    break
        return tuple(best)

    d_below = 0.0
    for pa in reversed(panels):
        # cavity seen by THIS panel: whatever lies behind it
        m = float(pa["m"])
        r = float(pa.get("r", 0.0))
        kk = float(pa.get("k", 0.0))
        i_pa = layers.index(pa)
        d_cav = sum(float(la.get("d", 0.0)) for la in layers[i_pa + 1:]
                    if str(la.get("model", "")) != "panel")
        d_cav = max(d_cav + d_below, 1e-3)
        a0 = Z0 / m
        b0 = max(r / m, 1.0)
        w0 = (RHO0 * C0 * C0 / d_cav + kk) / m
        sec = fit_one(a0, b0, w0, g_left)
        out.insert(0, sec)
        g_left = g_left - _gamma_sections([sec], f)
        d_below = d_cav
    out = ex_secs + out
    return out, g_left, f
