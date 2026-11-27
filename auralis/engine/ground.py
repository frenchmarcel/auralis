"""Point source above a locally reacting plane: the spherical-wave reflection coefficient."""
from __future__ import annotations

import numpy as np

_N = 32
_M = 2 * _N
_L = np.sqrt(_N / np.sqrt(2.0))


def _weideman_coeffs():
    k = np.arange(-_M + 1, _M)
    t = _L * np.tan(k * np.pi / (2.0 * _M))
    f = np.concatenate([[0.0], np.exp(-t * t) * (_L * _L + t * t)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * _M)
    return np.flipud(a[1:_N + 1])


_A = _weideman_coeffs()


def faddeeva(z):
    """w(z) = exp(-z^2) erfc(-i z) for complex z, without scipy.

    Weideman's rational approximation (SIAM J. Numer. Anal. 31, 1994) with 32 terms in the
    upper half-plane, and w(z) = 2 exp(-z^2) - w(-z) below it.
    """
    z = np.asarray(z, dtype=np.complex128)
    lower = z.imag < 0.0
    zu = np.where(lower, -z, z)
    den = _L - 1j * zu
    p = np.polyval(_A, (_L + 1j * zu) / den)
    w = 2.0 * p / den ** 2 + 1.0 / (np.sqrt(np.pi) * den)
    return np.where(lower, 2.0 * np.exp(-z * z) - w, w)


def spherical_reflection(f, beta, cos_theta, r2, c):
    """Q(f) of a point source above a locally reacting plane (Weyl-Van der Pol).

    Q = Rp + (1 - Rp) F(w), with Rp the plane-wave coefficient, F the boundary loss factor and
    w the numerical distance. beta is the normalised surface admittance rho c / Zs at each
    frequency, cos_theta the cosine of the incidence angle from the normal, r2 the length of
    the reflected path (image source to receiver). Engine convention exp(+j w t): the
    textbook form (exp(-i w t), Attenborough et al., Predicting Outdoor Sound) is applied to
    the conjugate and conjugated back. Accurate for k r2 >> 1, the usual outdoor case.
    """
    f = np.asarray(f, dtype=np.float64)
    b = np.conj(np.asarray(beta, dtype=np.complex128))
    k = 2.0 * np.pi * f / float(c)
    mu = float(np.clip(cos_theta, 0.0, 1.0))
    rp = (mu - b) / (mu + b)
    w = np.sqrt(0.5j * k * float(r2)) * (mu + b)
    F = 1.0 + 1j * np.sqrt(np.pi) * w * faddeeva(w)
    return np.conj(rp + (1.0 - rp) * F)


# Impedance classes of ground surfaces in Nord2000: representative effective flow resistivity
# (Pa.s/m2) and description. Jonasson and Storeheier, "Nord 2000. New Nordic prediction method
# for rail traffic noise", SP Rapport 2001:11, Table 3.1, after Nordtest NT ACOU 104. They go
# with the Delany-Bazley model on a semi-infinite ground.
GROUND_CLASSES = {
    "A": (12.5e3, "Very soft (snow or moss-like)"),
    "B": (31.5e3, "Soft forest floor (short, dense, heather-like or thick moss)"),
    "C": (80e3, "Uncompacted, loose ground (turf, grass, loose soil)"),
    "D": (200e3, "Normal uncompacted ground (forest floors, pasture field)"),
    "E": (500e3, "Compacted field and gravel (compacted lawns, park area)"),
    "F": (2000e3, "Compacted dense ground (gravel road, parking lot)"),
    "G": (20000e3, "Hard surface (dense asphalt, concrete, water)"),
}


# Terrain roughness classes of Nord2000: standard deviation of the random height variations, m.
# Plovsing and Kragh, Nord2000 Part 1 (DELTA report AV 1849/00, revised 2006), Table 2.
ROUGHNESS_CLASSES = {"N": 0.0, "S": 0.25, "M": 0.5, "L": 1.0}


def roughness_coherence(f, sigma_r, sin_grazing, c):
    """Coherence coefficient Fr of a reflection on terrain with random height variations of
    standard deviation sigma_r (m), Nord2000 Part 1 Eqs. 148-150: X = k sigma_r sin(psi),
    Fr = exp(g(X) / 2). Fr = 1 for a smooth ground and falls towards 0 when X grows."""
    X = 2.0 * np.pi * np.asarray(f, dtype=np.float64) / float(c) * float(sigma_r) \
        * float(sin_grazing)
    g = np.zeros_like(X)
    mid = (X > 0.026686) & (X < 0.115448)
    g[mid] = 0.55988 * (0.115448 - X[mid]) - 0.049696
    hi = X >= 0.115448
    x = X[hi]
    g[hi] = -0.066 + 1.066 * x - 8.543 * x ** 2 + 4.71 * x ** 3 - 0.83 * x ** 4
    return np.exp(0.5 * g)


def _sigma_of(ground):
    if isinstance(ground, str):
        key = ground.strip().upper()
        if key not in GROUND_CLASSES:
            raise ValueError("unknown ground class %r (A-G)" % ground)
        return GROUND_CLASSES[key][0]
    sigma = float(ground)
    if not sigma > 0.0:
        raise ValueError("flow resistivity must be positive")
    return sigma


def ground_spec(ground, roughness=None, alpha_e=None) -> dict:
    """Physical material of a ground.

    ground: a Nord2000 class letter, or an effective flow resistivity in Pa.s/m2. Without
    alpha_e: the Delany-Bazley model on a semi-infinite ground (the Nord2000 classes). With
    alpha_e (1/m, the rate at which the porosity falls with depth): Attenborough's
    two-parameter variable porosity model. roughness: a Nord2000 roughness class (N, S, M, L)
    or the standard deviation of the height variations in m; it makes the reflection partly
    incoherent (spherical-wave reflections only)."""
    sigma = _sigma_of(ground)
    if alpha_e is None:
        layer = {"model": "delany_bazley", "sigma": sigma, "semi_infinite": True}
    else:
        layer = {"model": "variable_porosity", "sigma": sigma, "alpha_e": float(alpha_e)}
    spec = {"layers": [layer]}
    if roughness is not None:
        if isinstance(roughness, str):
            key = roughness.strip().upper()
            if key not in ROUGHNESS_CLASSES:
                raise ValueError("unknown roughness class %r (N, S, M, L)" % roughness)
            sr = ROUGHNESS_CLASSES[key]
        else:
            sr = float(roughness)
            if sr < 0.0:
                raise ValueError("roughness must be >= 0")
        if sr > 0.0:
            spec["roughness"] = sr
    return spec
