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


def ground_spec(ground) -> dict:
    """Physical material of a ground: a Nord2000 class letter, or an effective flow
    resistivity in Pa.s/m2. Delany-Bazley model, semi-infinite."""
    if isinstance(ground, str):
        key = ground.strip().upper()
        if key not in GROUND_CLASSES:
            raise ValueError("unknown ground class %r (A-G)" % ground)
        sigma = GROUND_CLASSES[key][0]
    else:
        sigma = float(ground)
        if not sigma > 0.0:
            raise ValueError("flow resistivity must be positive")
    return {"layers": [{"model": "delany_bazley", "sigma": sigma, "semi_infinite": True}]}
