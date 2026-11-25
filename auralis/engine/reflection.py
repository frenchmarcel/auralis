"""Complex angle-dependent specular reflection: R(theta, w) =
(cos theta - gamma) / (cos theta + gamma), gamma fitted on diffuse-field
alphas (Paris) then anchored per band on the tables."""
from __future__ import annotations

import numpy as np

from .bands import CENTERS, N_BANDS, interp_band_gains

# Gauss-Legendre quadrature over mu = cos(theta) in (0, 1].
_MU, _WMU = np.polynomial.legendre.leggauss(48)
_MU = 0.5 * (_MU + 1.0)
_WMU = 0.5 * _WMU

def _paris_argmax():
    g = np.geomspace(0.05, 3.0, 400)
    a = 8.0 * g * (1.0 + g / (1.0 + g) - 2.0 * g * np.log1p(1.0 / g))
    k = int(np.argmax(a))
    return float(g[k]), float(a[k])


G_ALPHA_MAX, _A_MAX = _paris_argmax()   # ~0.63, ~0.951
ALPHA_RAND_MAX = _A_MAX - 0.002


def paris_alpha_real(g):
    """Diffuse-field alpha of a REAL admittance g (closed form)."""
    g = np.asarray(g, dtype=np.float64)
    return 8.0 * g * (1.0 + g / (1.0 + g) - 2.0 * g * np.log1p(1.0 / g))


def alpha_random(gamma):
    """Diffuse-field alpha of a COMPLEX admittance (Paris quadrature).
    gamma: scalar or array (...,); returns the same shape."""
    gm = np.asarray(gamma, dtype=np.complex128)
    mu = _MU.reshape((1,) * gm.ndim + (-1,))
    g = gm[..., None]
    integ = np.sum(_WMU * mu ** 2 / np.abs(mu + g) ** 2, axis=-1)
    return 8.0 * np.real(gm) * integ


def invert_paris(alpha: float) -> float:
    """Real admittance g such that paris_alpha_real(g) = alpha
    (bisection; alpha saturates at ALPHA_RAND_MAX)."""
    a = float(np.clip(alpha, 0.0, ALPHA_RAND_MAX))
    if a <= 0.0:
        return 0.0
    lo, hi = 1e-6, G_ALPHA_MAX
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if paris_alpha_real(mid) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gamma_of(f, gi, A, l):
    """gamma(f) of the passive model: gi plus two relaxation branches A / (j w + l)."""
    w = 2.0 * np.pi * np.asarray(f, dtype=np.float64)
    return (gi + A[0] / (1j * w + l[0]) + A[1] / (1j * w + l[1]))


def invert_beta_multiangle(alphas_by_angle, angles_deg):
    """Per-band inversion of the complex admittance beta (locally
    reacting) from alphas (n_angles, n_bands). Returns beta, per-band rms."""
    A = np.asarray(alphas_by_angle, dtype=np.float64)
    cs = np.cos(np.radians(np.asarray(angles_deg, dtype=np.float64)))
    n_b = A.shape[1]
    betas = np.empty(n_b, dtype=np.complex128)
    rms = np.empty(n_b)

    def cost(b, av):
        R = (cs - b) / (cs + b)
        return float(np.mean((1.0 - np.abs(R) ** 2 - av) ** 2))

    for i in range(n_b):
        av = A[:, i]
        # coarse grid then descent with adaptive steps
        best, bb = np.inf, 0.2 + 0.0j
        for br in np.linspace(0.02, 2.0, 40):
            for bi in np.linspace(-1.5, 1.5, 41):
                c = cost(br + 1j * bi, av)
                if c < best:
                    best, bb = c, br + 1j * bi
        step = 0.05
        while step > 1e-4:
            moved = False
            for db in (step, -step, 1j * step, -1j * step):
                c = cost(bb + db, av)
                if c < best and (bb + db).real > 1e-3:
                    best, bb = c, bb + db
                    moved = True
            if not moved:
                step *= 0.5
        betas[i] = bb
        rms[i] = np.sqrt(best)
    return betas, rms


def fit_admittance_multiangle(alphas_by_angle, angles_deg, band_centers,
                              f_hi: float = 8000.0, f_lo: float = 100.0,
                              n_desc: int = 2, n_mont: int = 1):
    """Fit a passive admittance on multi-angle alphas (phase-blind).
    Returns (gi, A, l), decaying branches first."""
    Aa = np.asarray(alphas_by_angle, dtype=np.float64)
    fb = np.asarray(band_centers, dtype=np.float64)
    m = (fb >= f_lo) & (fb <= f_hi)
    f = np.geomspace(fb[m][0], fb[m][-1], 90)
    tgt = np.stack([np.interp(f, fb[m], Aa[k][m])
                    for k in range(Aa.shape[0])])
    cs = np.cos(np.radians(np.asarray(angles_deg, dtype=np.float64)))
    jw = 1j * 2.0 * np.pi * f
    nb = n_desc + n_mont

    def gamma_of(p):
        gi = p[0]
        g = np.full(f.shape, gi, dtype=np.complex128)
        for k in range(n_desc):
            g = g + p[1 + 2 * k] / (jw + p[2 + 2 * k])
        for k in range(n_mont):
            j0 = 1 + 2 * n_desc + 2 * k
            g = g + p[j0] * jw / (jw + p[j0 + 1])
        return g

    l_min = 2.0 * np.pi * 8.0     # floor: no pure integrator
                                   # (l=0 -> state with no decay,
                                   # DC drift in the grid)

    def cost(p):
        q = np.asarray(p)
        if np.any(q < 0):
            return np.inf
        for k in range(n_desc):
            if q[2 + 2 * k] < l_min:
                return np.inf
        base = 1 + 2 * n_desc
        for k in range(n_mont):
            if q[base + 2 * k + 1] < l_min:
                return np.inf
        g = gamma_of(p)
        e = 0.0
        for k, c in enumerate(cs):
            R = (c - g) / (c + g)
            e += float(np.mean((1.0 - np.abs(R) ** 2 - tgt[k]) ** 2))
        return e

    # init: gi = mean top-of-band beta, branches spread in log f
    hi_a = float(np.mean(tgt[:, -20:]))
    gi0 = max(0.05, 1.0 - np.sqrt(max(1.0 - hi_a, 1e-3)))
    p = [gi0]
    ls = np.geomspace(2 * np.pi * f[0] * 2, 2 * np.pi * f[-1] * 0.5,
                      nb)
    for k in range(n_desc):
        p += [gi0 * ls[k], ls[k]]
    for k in range(n_mont):
        p += [gi0, ls[n_desc + k]]
    p = np.asarray(p, dtype=np.float64)
    best = cost(p)
    step = np.full(len(p), 0.6)
    for _ in range(400):
        moved = False
        for i in range(len(p)):
            for s_ in (1.0, -1.0):
                q = p.copy()
                q[i] *= (1.0 + s_ * step[i])
                c = cost(q)
                if c < best:
                    p, best = q, c
                    moved = True
        if not moved:
            step *= 0.5
            if step.max() < 1e-3:
                break
    gi = float(p[0])
    A_d = [p[1 + 2 * k] for k in range(n_desc)]
    l_d = [p[2 + 2 * k] for k in range(n_desc)]
    base = 1 + 2 * n_desc
    A_m = [p[base + 2 * k] for k in range(n_mont)]
    l_m = [p[base + 2 * k + 1] for k in range(n_mont)]
    A = np.asarray(A_d + A_m, dtype=np.float64)
    l = np.asarray(l_d + l_m, dtype=np.float64)
    return gi, A, l


def _min_re_discrete(gi, a1, l1, b2, n2, dt, n=256):
    """Smallest real part of the admittance the grid realizes for the branch form
    (coarse fdtd.wall_passivity, kept here to avoid an import cycle)."""
    w = np.linspace(0.0, np.pi / dt, n + 2)[1:-1]
    z = np.exp(1j * w * dt)
    acc = np.zeros(len(w), dtype=np.complex128)
    for a, ll in ((a1, l1), (-b2 * n2, n2)):
        if a == 0.0:
            continue
        e = np.exp(-ll * dt)
        acc += a * (1.0 - (1.0 - e) * (z + 1.0) / (2.0 * (z - e)))
    return float(((gi + b2) + dt * acc / (1j * np.sin(w * dt))).real.min())


def fit_admittance_random(alpha_bands, band_centers, f_hi: float = 8000.0,
                          l_max: float = None, gamma_target=None, w_r: float = 0.2,
                          dt: float = None):
    """Fit a passive admittance so that alpha_rand(gamma(f)) follows the tabulated alphas.

    Returns the FDTD format (gi_eff, A(2,), l(2,)).
    l_max (rad/s) caps both poles: the grid's explicit branches stay passive
    only for l dt well below 1 (fdtd.wall_passivity).
    gamma_target(f) (optional): a physical model's normal admittance (matinfer
    materials). The fit is pulled toward its normal R with weight w_r, the alphas
    staying the anchor, and with dt the grid's passivity is enforced in the cost.
    Without gamma_target the fit is unchanged, bit for bit."""
    fb = np.asarray(band_centers, dtype=np.float64)
    ab = np.clip(np.asarray(alpha_bands, dtype=np.float64),
                 0.0, ALPHA_RAND_MAX)
    f = np.geomspace(max(50.0, 0.4 * fb[0]), min(1.2 * f_hi, 1.2 * fb[-1]),
                     48)
    a_t = np.interp(f, fb, ab)
    w = 2.0 * np.pi * f

    def gamma_p(gi, a1, l1, b2, n2):
        return (gi + a1 / (1j * w + l1)
                + b2 * (1j * w) / (1j * w + n2))

    if gamma_target is None:
        def cost(p):
            return float(np.mean(
                (alpha_random(gamma_p(*p)) - a_t) ** 2))
    else:
        gt = np.asarray(gamma_target(f), dtype=np.complex128)
        r_t = (1.0 - gt) / (1.0 + gt)

        def cost(p):
            g = gamma_p(*p)
            c = float(np.mean((alpha_random(g) - a_t) ** 2)
                      + w_r * np.mean(np.abs((1.0 - g) / (1.0 + g) - r_t) ** 2))
            if dt is not None:
                m = _min_re_discrete(*p, dt)
                if m < 0.0:
                    c += 1e3 * m * m + 1e-3
            return c

    g_lo = invert_paris(float(a_t[:8].mean()))
    g_hi = invert_paris(float(a_t[-8:].mean()))
    g_mid = invert_paris(float(np.median(a_t)))
    grid = 2.0 * np.pi * np.array([60.0, 150.0, 400.0, 1000.0,
                                   2500.0, 6000.0])
    if l_max is not None:
        grid = np.unique(np.minimum(grid, float(l_max)))
    best = (cost((g_mid, 0.0, 1.0, 0.0, 1.0)),
            np.array([g_mid, 0.0, 1.0, 0.0, 1.0]))
    for l1 in grid:
        for n2 in grid:
            for a1 in (0.0, g_lo * l1, 2.5 * g_lo * l1):
                for b2 in (0.0, max(g_hi - g_lo, 0.0),
                           1.6 * max(g_hi - g_lo, 0.0) + 0.01):
                    p = np.array([min(g_lo, g_hi), a1, l1, b2, n2])
                    c = cost(p)
                    if c < best[0]:
                        best = (c, p)
    x = best[1].copy()
    step = 0.3 * np.maximum(np.abs(x), [0.01, 30.0, 80.0, 0.02, 80.0])
    for _ in range(160):
        improved = False
        for i in range(5):
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] = max(0.0, y[i] + sgn * step[i])
                if i in (2, 4):
                    y[i] = max(y[i], 1.0)
                    if l_max is not None:
                        y[i] = min(y[i], float(l_max))
                c = cost(y)
                if c < best[0]:
                    best = (c, y.copy())
                    x = y
                    improved = True
        if not improved:
            step *= 0.5
            if step.max() < 1e-4:
                break
    gi, a1, l1, b2, n2 = best[1]
    # FDTD format: gi_eff, A = (a1, -b2*n2), l = (l1, n2)
    return (float(gi + b2), np.array([a1, -b2 * n2]),
            np.array([l1, n2]))


def _res_five(sec):
    """Normalize a section: (a, b, w2) -> (a, a w2, a b, b, w2);
    5-tuples (rA, rc1, rc2, b, w2) pass through unchanged."""
    if len(sec) == 3:
        a, b, w2 = sec
        return (a, a * w2, a * b, b, w2)
    return tuple(sec)


def _res_B_z(res, z, dt):
    """z-EXACT drive of the resonant sections, summed over sections:
    B_r/p = rA - (rc1 psi1 + rc2 psi2)/p."""
    from .fdtd import _soss_prop
    B = np.zeros(np.shape(z), dtype=np.complex128)
    for sec in res:
        rA, rc1, rc2, b, w2 = _res_five(sec)
        E, G = _soss_prop(float(b), float(w2), float(dt))
        det = (z - E[0, 0]) * (z - E[1, 1]) - E[0, 1] * E[1, 0]
        p1 = ((z - E[1, 1]) * G[0] + E[0, 1] * G[1]) / det
        p2 = (E[1, 0] * G[0] + (z - E[0, 0]) * G[1]) / det
        half = 0.5 * (z + 1.0)
        B = B + rA - (rc1 * p1 + rc2 * p2) * half
    return B


def wall_R_discrete(f, gi, A, l, dt, lam, res=None):
    """EXACT discrete NORMAL reflection of the filtering wall (axial
    plane wave). lam: Courant number of the RUN."""
    f = np.asarray(f, dtype=np.float64)
    w = 2.0 * np.pi * f
    z = np.exp(1j * w * dt)
    kdx = 2.0 * np.arcsin(np.clip(np.sin(0.5 * w * dt) / lam,
                                  -1.0, 1.0))
    A = np.asarray(A, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    B = np.full(z.shape, float(A.sum()), dtype=np.complex128)
    for A_, l_ in zip(A, l):
        fe = np.exp(-l_ * dt)
        fg = A_ / max(l_, 1e-30) * (1.0 - fe)
        B = B - l_ * fg * (z + 1.0) / (2.0 * (z - fe))
    if res:
        B = B + _res_B_z(res, z, dt)
    Gam = 0.5 * lam * gi
    den = 1.0 + Gam
    W = den * z - (2.0 - lam ** 2) - (Gam - 1.0) / z + lam * dt * B
    return (((lam ** 2) * np.exp(1j * kdx) - W)
            / (W - (lam ** 2) * np.exp(-1j * kdx)))


def fit_admittance_complex(f, gamma_target, l_max=None, dt=None,
                           lam=None, res_sections=None, pass_weight: float = 10.0):
    """Fit the passive form on a COMPLEX gamma(f) target; with dt/lam the
    cost applies to discrete R. Returns (gi_eff, A(2,), l(2,))."""
    f = np.asarray(f, dtype=np.float64)
    gt = np.asarray(gamma_target, dtype=np.complex128)
    w = 2.0 * np.pi * f
    l_cap = float(l_max) if l_max else np.inf
    R_t = (1.0 - gt) / (1.0 + gt)
    lam_run = float(lam) if lam else (1.0 / np.sqrt(3.0))
    if dt:
        f_ext = np.geomspace(f[0], 0.45 / dt, 40)
        # low-end weighting: the FDTD band exists for the bottom of the
        # spectrum, where the cost spends its budget (energy ~ sqrt(fmax/f))
        w_lo = np.clip(np.sqrt(f[-1] / np.maximum(f, 1.0)), 1.0, 4.0)
        w_lo = w_lo / w_lo.mean()
        # reference PLANE phase of the discrete rigid wall:
        # wall_R_discrete uses reference planes different from the
        # continuous target (half cell + dispersion). We remove that
        # convention phase before the cost, otherwise the high-frequency
        # tail dominates with an artifact error that free zeros can
        # exploit by over-absorbing.
        R0 = wall_R_discrete(f, 0.0, np.zeros(1), np.ones(1), dt,
                             lam_run)
        ph_ref = np.conj(R0 / np.maximum(np.abs(R0), 1e-12))

    def R_model(p, fr):
        gi, a1, l1, b2, n2 = p
        if dt is None:
            wv = 2.0 * np.pi * np.asarray(fr, dtype=np.float64)
            g = (gi + a1 / (1j * wv + l1)
                 + b2 * (1j * wv) / (1j * wv + n2))
            return (1.0 - g) / (1.0 + g)
        return wall_R_discrete(fr, gi + b2, (a1, -b2 * n2),
                               (l1, n2), dt, lam_run,
                               res=res_sections)

    def cost(p):
        # cost in the REFLECTION plane: what is heard is R (magnitude
        # AND phase); the gamma plane is ill-conditioned when gamma is
        # small (hard walls) or large (matched walls). In discrete form,
        # the phase is referred back to the rigid-wall plane.
        Rm = R_model(p, f)
        if dt is not None:
            Rm = Rm * ph_ref
        d = Rm - R_t
        d2 = d.real ** 2 + d.imag ** 2
        c = float(np.mean(d2 if dt is None else w_lo * d2))
        if dt is not None:
            # passivity of the REALIZED response up to near Nyquist
            ex = float(np.maximum(np.abs(R_model(p, f_ext)) - 1.0,
                                  0.0).max())
            c += pass_weight * ex * ex
        return c

    re = np.clip(gt.real, 0.0, None)
    q = max(len(f) // 4, 1)
    g_lo = float(np.median(re[:q]))
    g_hi = float(np.median(re[-q:]))
    g_mid = float(np.median(re))
    # slope of the reactive part: gamma ~ j w slope at low frequency
    # (air-gap stiffness) -> seed of the rising branch
    slope = float(np.median(np.clip(gt.imag, 0.0, None)[:2 * q]
                            / w[:2 * q]))
    grid = 2.0 * np.pi * np.array([40.0, 120.0, 350.0, 900.0,
                                   2200.0, 5500.0])
    grid = np.minimum(grid, l_cap)
    best = (cost((g_mid, 0.0, 1.0, 0.0, 1.0)),
            np.array([g_mid, 0.0, 1.0, 0.0, 1.0]))
    dn = max(g_lo - g_hi, 0.0)
    up = max(g_hi - g_lo, 0.0)
    for l1 in grid:
        for n2 in grid:
            for a1 in (0.0, dn * l1, 2.5 * dn * l1 + 0.01):
                for b2 in (0.0, up, 1.6 * up + 0.01,
                           max(slope, 0.0) * n2):
                    p = np.array([max(min(g_lo, g_hi), 0.0),
                                  a1, l1, b2, n2])
                    c = cost(p)
                    if c < best[0]:
                        best = (c, p)
    x = best[1].copy()
    step = 0.3 * np.maximum(np.abs(x), [0.01, 30.0, 80.0, 0.02, 80.0])
    for _ in range(200):
        improved = False
        for i in range(5):
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] = max(0.0, y[i] + sgn * step[i])
                if i in (2, 4):
                    y[i] = float(np.clip(y[i], 1.0, l_cap))
                c = cost(y)
                if c < best[0]:
                    best = (c, y.copy())
                    x = y
                    improved = True
        if not improved:
            step *= 0.5
            if step.max() < 1e-4:
                break
    gi, a1, l1, b2, n2 = best[1]
    return (float(gi + b2), np.array([a1, -b2 * n2]),
            np.array([l1, n2]))


_TABLE_CACHE = {}


def build_admittance_table(absorption, mat_phys=None) -> dict:
    """Bake admittance table: per-material fit plus per-band energy
    anchors. mat_phys: poromat specs -> complex fit."""
    ab = np.asarray(absorption, dtype=np.float64)
    key = (ab.tobytes(), repr(mat_phys))
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    nm = len(ab)
    gi = np.zeros(nm)
    A = np.zeros((nm, 2))
    l = np.ones((nm, 2))
    anchor = np.ones((nm, N_BANDS))
    for m in range(nm):
        spec = mat_phys[m] if (mat_phys is not None
                               and m < len(mat_phys)) else None
        if spec:
            from . import poromat
            fg = np.geomspace(40.0, 1.2 * CENTERS[-1], 60)
            target = (poromat._gamma_table(spec, fg) if "gamma_table" in spec
                      else poromat.gamma_normal(spec, fg))     # measured table / model
            gi[m], A[m], l[m] = fit_admittance_complex(fg, target)
        else:
            gi[m], A[m], l[m] = fit_admittance_random(ab[m], CENTERS)
        g_b = _gamma_of(CENTERS, gi[m], A[m], l[m])
        a_fit = np.clip(alpha_random(g_b), 0.0, 0.995)
        a_tab = np.clip(ab[m], 0.0, ALPHA_RAND_MAX)
        anchor[m] = np.sqrt((1.0 - a_tab) / np.maximum(1.0 - a_fit, 1e-3))
        anchor[m] = np.clip(anchor[m], 0.25, 4.0)
    out = {"gi": gi, "A": A, "l": l, "anchor": anchor}
    if len(_TABLE_CACHE) > 64:
        _TABLE_CACHE.clear()
    _TABLE_CACHE[key] = out
    return out


def refl_coeff(mu, mat: int, table: dict, f):
    """Complex R(theta, f) for cos(theta) = mu and material mat."""
    g = _gamma_of(f, table["gi"][mat], table["A"][mat], table["l"][mat])
    return (mu - g) / (mu + g)


def band_R_mag(bounces, table: dict) -> np.ndarray:
    """Per-band product of the ANCHORED |R| at the octave centers (for
    event amplitudes and the energy echogram)."""
    out = np.ones(N_BANDS)
    for mat, mu in bounces:
        out *= table["anchor"][mat] * np.abs(
            refl_coeff(mu, mat, table, CENTERS))
    return out


def event_reflection_spectrum(freqs, bounces, table: dict) -> np.ndarray:
    """Complex spectral correction U(f) of an event's smooth spectrum:
    exact anchored product / interpolation at centers (|U| = 1 there)."""
    f = np.asarray(freqs, dtype=np.float64)
    fq = np.maximum(f, 1.0)
    P = np.ones(len(f), dtype=np.complex128)
    for mat, mu in bounces:
        anc = interp_band_gains(fq, table["anchor"][mat])
        P *= anc * refl_coeff(mu, mat, table, fq)
    mags = band_R_mag(bounces, table)
    base = np.maximum(interp_band_gains(fq, mags), 1e-9)
    return P / base


def fit_wall_table(f_tgt, gamma_tgt, dt, lam, n_sections: int = 3,
                   f_pass_hi=None, iters: int = 400):
    """Passive vector fitting of the discrete wall on gamma(f), |R|<=1
    enforced. Returns (gi, A(2), l(2), second-order sections, rms)."""
    f_tgt = np.asarray(f_tgt, float)
    g_t = np.asarray(gamma_tgt, complex)
    R_t = (1.0 - g_t) / (1.0 + g_t)
    if f_pass_hi is None:
        f_pass_hi = 0.45 / dt
    f_pass = np.geomspace(max(f_tgt[0] * 0.5, 10.0), f_pass_hi, 220)
    # normalize by the plane phase of the discrete RIGID wall: without it, the
    # cost chases a grid phase that no passive parameter can compensate
    R_rig = wall_R_discrete(f_tgt, 0.0, np.zeros(2), np.ones(2), dt, lam)
    ph_rig = np.conj(R_rig / np.abs(R_rig))

    w1 = 2 * np.pi * f_tgt[len(f_tgt) // 3]
    w2 = 2 * np.pi * f_tgt[2 * len(f_tgt) // 3]
    f0s = np.geomspace(f_tgt[2], f_tgt[-3], max(n_sections, 1))
    # parameters: [gi, A1, A2, log l1, log l2,
    #               (rA, rc1n, rc2n, log beta, log w0)*k]
    x = [0.05, 0.05, -0.05, np.log(w1), np.log(w2)]
    for f0 in f0s:
        w0 = 2 * np.pi * float(f0)
        x += [0.3, 0.3, 0.3, np.log(w0 / 0.7), np.log(w0)]
    x = np.array(x, float)

    def unpack(x):
        gi = abs(x[0])
        A = np.array([x[1], x[2]])
        l = np.exp(np.clip(x[3:5], 2.0, np.log(0.6 / dt)))
        secs = []
        for k in range(n_sections):
            b = x[5 + 5 * k:10 + 5 * k]
            w0 = np.exp(np.clip(b[4], 3.0, np.log(2 * np.pi * f_pass_hi)))
            beta = np.exp(np.clip(b[3], 1.0, np.log(60.0 * w0)))
            w02 = w0 * w0
            secs.append((b[0] * w0, b[1] * w0 * w02 / max(beta, 1e-9),
                         b[2] * w0, beta, w02))
        return gi, A, l, secs

    def cost(x):
        gi, A, l, secs = unpack(x)
        try:
            R = wall_R_discrete(f_tgt, gi, A, l, dt, lam, res=secs)
            Rp = np.abs(wall_R_discrete(f_pass, gi, A, l, dt, lam,
                                        res=secs))
        except Exception:
            return 1e9
        if not (np.isfinite(R).all() and np.isfinite(Rp).all()):
            return 1e9
        e = float(np.mean(np.abs(R * ph_rig - R_t) ** 2))
        p = float(np.mean(np.maximum(Rp - 1.0, 0.0) ** 2))
        return e + 400.0 * p

    def residuals(x):
        gi, A, l, secs = unpack(x)
        try:
            R = wall_R_discrete(f_tgt, gi, A, l, dt, lam, res=secs)
            Rp = np.abs(wall_R_discrete(f_pass, gi, A, l, dt, lam,
                                        res=secs))
        except Exception:
            return np.full(2 * len(f_tgt) + len(f_pass), 1e3)
        if not (np.isfinite(R).all() and np.isfinite(Rp).all()):
            return np.full(2 * len(f_tgt) + len(f_pass), 1e3)
        d = R * ph_rig - R_t
        pen = 20.0 * np.maximum(Rp - 1.0, 0.0)
        return np.concatenate([d.real, d.imag, pen])

    try:
        from scipy.optimize import least_squares
        best_x, best_c = x, float(np.sum(residuals(x) ** 2))
        rng = np.random.default_rng(7)
        for trial in range(4):
            x0 = x if trial == 0 else                 x * (1.0 + 0.3 * rng.standard_normal(len(x)))
            try:
                sol = least_squares(residuals, x0, method="lm"
                                    if False else "trf",
                                    max_nfev=4000)
                if sol.cost * 2 < best_c:
                    best_x, best_c = sol.x, sol.cost * 2
            except Exception:
                continue
        x = best_x
        c_best = best_c / (2 * len(f_tgt))
    except ImportError:
        c_best = cost(x)
        step = 0.4 * np.ones_like(x)
        for it in range(iters):
            moved = False
            for i in range(len(x)):
                for sgn in (1.0, -1.0):
                    x2 = x.copy()
                    x2[i] += sgn * step[i] * max(abs(x[i]), 0.05)
                    c = cost(x2)
                    if c < c_best:
                        x, c_best = x2, c
                        moved = True
            if not moved:
                step *= 0.5
                if step.max() < 1e-4:
                    break
    gi, A, l, secs = unpack(x)
    return gi, A, l, secs, np.sqrt(max(c_best, 0.0))


def polish_resonant(f, gamma_target, secs, gi, A, l, dt, lam,
                    fixed=None):
    """Pre-compensate the resonant section for the GRID (free poles and
    zeros) on discrete R; |R| <= 1 enforced up to 0.45/dt."""
    f = np.asarray(f, dtype=np.float64)
    gt = np.asarray(gamma_target, dtype=np.complex128)
    R_t = (1.0 - gt) / (1.0 + gt)
    w_lo = np.clip(np.sqrt(f[-1] / np.maximum(f, 1.0)), 1.0, 4.0)
    w_lo = w_lo / w_lo.mean()
    f_pas = np.geomspace(max(f[0], 5.0), 0.45 / dt, 160)
    R0 = wall_R_discrete(f, 0.0, np.zeros(1), np.ones(1), dt, lam)
    ph_ref = np.conj(R0 / np.maximum(np.abs(R0), 1e-12))

    def cost(p):
        rA, rc1, rc2, b, w2 = p
        if min(rA, rc1, rc2, b, w2) <= 0:
            return np.inf
        sec = [(rA, rc1, rc2, b, w2)] + list(fixed or [])
        Rp = np.abs(wall_R_discrete(f_pas, gi, A, l, dt, lam, res=sec))
        over = float(np.maximum(Rp - 1.0, 0.0).max())
        if over > 0:
            return 1e3 + 1e5 * over
        R = wall_R_discrete(f, gi, A, l, dt, lam, res=sec) * ph_ref
        d = R - R_t
        return float(np.mean(w_lo * (d.real ** 2 + d.imag ** 2)))

    best = np.asarray(_res_five(secs[0]), dtype=np.float64)
    c_best = cost(best)
    step = np.full(5, 0.4)
    for _ in range(140):
        moved = False
        for i in range(5):
            for sgn in (1.0, -1.0):
                cand = best.copy()
                cand[i] *= (1.0 + sgn * step[i])
                c = cost(cand)
                if c < c_best:
                    best, c_best = cand, c
                    moved = True
        if not moved:
            step *= 0.5
            if step.max() < 5e-4:
                break
    return [tuple(best)]
