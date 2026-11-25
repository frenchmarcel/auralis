"""Measured-HRTF loader (.npz from tools/sofa_to_npz.py: ALIGNED ir,
delay, az_deg/el_deg in SOFA convention -- azimuth +90 = LEFT, fs).
Interpolation: 3 nearest neighbours, inverse-angle weights."""
from __future__ import annotations

import numpy as np


def _resample_fir(h: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample a FIR by spectral zero-padding: the frequency response
    is preserved over the common band, with no scale factor."""
    if abs(fs_in - fs_out) < 1e-6:
        return h.astype(np.float64)
    n_in = h.shape[-1]
    n_out = max(int(round(n_in * fs_out / fs_in)), 8)
    H = np.fft.rfft(h, axis=-1)
    n_bins = n_out // 2 + 1
    out = np.zeros(h.shape[:-1] + (n_bins,), dtype=np.complex128)
    k = min(H.shape[-1], n_bins)
    out[..., :k] = H[..., :k]
    return np.fft.irfft(out, n_out, axis=-1)


def solid_angle_weights(units: np.ndarray, k: int = 6, chunk: int = 512) -> np.ndarray:
    """Solid angle of each direction (sums to 1), from the mean angle to its k
    nearest neighbours: equal-angle grids crowd the poles, this undoes it."""
    n = len(units)
    k = min(k, n - 1)
    w = np.empty(n)
    for a in range(0, n, chunk):
        d = np.clip(units[a:a + chunk] @ units.T, -1.0, 1.0)
        near = np.sort(np.partition(d, n - k - 1, axis=1)[:, n - k - 1:], axis=1)
        w[a:a + chunk] = np.arccos(near[:, :-1]).mean(axis=1) ** 2   # self dropped
    return w / w.sum()


def min_phase_fir(mag: np.ndarray, n_taps: int) -> np.ndarray:
    """Minimum-phase FIR (real cepstrum) of a magnitude on an rfft grid,
    truncated to n_taps with a short fade-out."""
    n_fft = 2 * (len(mag) - 1)
    cep = np.fft.irfft(np.log(np.maximum(mag, 1e-9 * mag.max())), n_fft)
    cep[1:n_fft // 2] *= 2.0
    cep[n_fft // 2 + 1:] = 0.0
    h = np.fft.irfft(np.exp(np.fft.rfft(cep, n_fft)), n_fft)[:n_taps]
    nf = max(n_taps // 16, 1)
    h[-nf:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, nf)))
    return h


class HrtfSet:
    """Measured HRTF set, resampled to the bake rate when opened."""

    def __init__(self, path: str, fs: int):
        z = np.load(path, allow_pickle=False)
        fs_set = float(z["fs"])
        ir = np.asarray(z["ir"], dtype=np.float64)          # (M, 2, L)
        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError("HRTF npz: ir must have shape (M, 2, L)")
        self.fs = int(fs)
        self.name = str(z["name"]) if "name" in z.files else "hrtf"
        self.ir = _resample_fir(ir, fs_set, fs)
        self.delay = (np.asarray(z["delay"], dtype=np.float64)
                      * (fs / fs_set))                       # samples @ fs
        az = np.radians(np.asarray(z["az_deg"], dtype=np.float64))
        el = np.radians(np.asarray(z["el_deg"], dtype=np.float64))
        # listener frame: x = front, y = left, z = up (SOFA)
        self.units = np.stack([np.cos(el) * np.cos(az),
                               np.cos(el) * np.sin(az),
                               np.sin(el)], axis=1)          # (M, 3)
        self.n_taps = int(self.ir.shape[-1])
        self._cache = {}
        self._diffuse = None

    # ------------------------------------------------------------------
    def _diffuse_spectra(self, n_fft: int = 2048):
        """Area-weighted spectra over all directions, ear delays included:
        (freqs, S_LL, S_RR, S_LR) on the rfft grid of n_fft (cached)."""
        if self._diffuse is None or self._diffuse[0] != n_fft:
            w = solid_angle_weights(self.units)
            f = np.fft.rfftfreq(n_fft, 1.0 / self.fs)
            sll = np.zeros(len(f))
            srr = np.zeros(len(f))
            slr = np.zeros(len(f), dtype=np.complex128)
            for a in range(0, len(w), 1024):
                H = np.fft.rfft(self.ir[a:a + 1024], n_fft, axis=-1)
                H = H * np.exp(-2j * np.pi * f[None, None, :]
                               * self.delay[a:a + 1024, :, None] / self.fs)
                wa = w[a:a + 1024, None]
                sll += (wa * np.abs(H[:, 0]) ** 2).sum(axis=0)
                srr += (wa * np.abs(H[:, 1]) ** 2).sum(axis=0)
                slr += (wa * H[:, 0] * np.conj(H[:, 1])).sum(axis=0)
            self._diffuse = (n_fft, f, sll, srr, slr)
        return self._diffuse[1:]

    def diffuse_field_firs(self, n_taps: int = 512):
        """Minimum-phase FIR per ear (2, n_taps) with the set's diffuse-field
        magnitude: the colour the head and pinnae give a diffuse tail."""
        _f, sll, srr, _slr = self._diffuse_spectra()
        return np.stack([min_phase_fir(np.sqrt(s), n_taps) for s in (sll, srr)])

    def band_coherence(self):
        """Interaural coherence of a diffuse field per engine band, from the set:
        Re(S_LR) / sqrt(S_LL S_RR), each spectrum summed under the band mask."""
        from .bands import band_masks
        f, sll, srr, slr = self._diffuse_spectra()
        m2 = band_masks(f, self.fs) ** 2
        num = m2 @ slr.real
        den = np.sqrt((m2 @ sll) * (m2 @ srr))
        return np.clip(num / np.maximum(den, 1e-30), -0.999, 0.999)

    def wave_ear_half(self, f_lo: float = 100.0, f_hi: float = 600.0,
                      c: float = 343.0) -> float:
        """Half spacing (m) of two omni probes whose diffuse-field coherence
        sinc(k d) best matches the set's (least squares, f_lo-f_hi): the wave
        band has no head, its ears are probes (FABIAN: 0.167 m, rms 0.014)."""
        if getattr(self, "_ear_half", None) is None:
            f, sll, srr, slr = self._diffuse_spectra()
            s = (f >= f_lo) & (f <= f_hi)
            ic = slr.real[s] / np.sqrt(np.maximum(sll[s] * srr[s], 1e-300))
            d = np.linspace(0.05, 0.60, 1101)
            err = ((ic[None, :] - np.sinc(2.0 * f[s][None, :] * d[:, None] / c)) ** 2).mean(1)
            j = int(np.argmin(err))
            if j in (0, len(d) - 1):
                print("[Auralis] WARNING: HRTF set '%s': diffuse coherence fits no probe "
                      "spacing in %.2f-%.2f m: wave-band ears at %.3f m"
                      % (self.name, d[0], d[-1], d[j]))
            self._ear_half = 0.5 * float(d[j])
        return self._ear_half

    # ------------------------------------------------------------------
    def _local_unit(self, direction, basis):
        f, u, r = basis
        d = np.asarray(direction, dtype=np.float64)
        # local y = left = -right
        v = np.array([float(np.dot(d, f)), -float(np.dot(d, r)),
                      float(np.dot(d, u))])
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([1.0, 0.0, 0.0])

    def _interp(self, v):
        dots = self.units @ v
        k = np.argpartition(-dots, 2)[:3]
        k = k[np.argsort(-dots[k])]
        if dots[k[0]] > 0.999995:                # < ~0.2 degree: exact
            w = np.array([1.0])
            k = k[:1]
        else:
            ang = np.arccos(np.clip(dots[k], -1.0, 1.0))
            w = 1.0 / np.maximum(ang, 1e-6)
            w /= w.sum()
        fir = np.tensordot(w, self.ir[k], axes=(0, 0))       # (2, L)
        dly = w @ self.delay[k]                              # (2,)
        return fir, dly

    def ear_responses(self, direction, basis, n_fft: int):
        """(HL(f), HR(f), dL_s, dR_s) on the rfft grid of size n_fft.
        Delays in seconds, applied by the caller (aligned IRs)."""
        v = self._local_unit(direction, basis)
        # exact key: a rounded key would serve the first direction of its bin to the
        # later ones, so a pair would depend on the pairs baked before it in the process
        key = (float(v[0]), float(v[1]), float(v[2]), int(n_fft))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        fir, dly = self._interp(v)
        L = min(fir.shape[-1], n_fft)
        HL = np.fft.rfft(fir[0, :L], n_fft)
        HR = np.fft.rfft(fir[1, :L], n_fft)
        out = (HL, HR, float(dly[0] / self.fs), float(dly[1] / self.fs))
        if len(self._cache) > 1024:
            self._cache.clear()
        self._cache[key] = out
        return out


_SET_CACHE = {}


def load_hrtf(path: str, fs: int):
    """Cache of open sets (path, fs) -> HrtfSet; '' -> None."""
    if not path:
        return None
    key = (str(path), int(fs))
    hit = _SET_CACHE.get(key)
    if hit is None:
        hit = HrtfSet(path, fs)
        if len(_SET_CACHE) > 8:
            _SET_CACHE.clear()
        _SET_CACHE[key] = hit
    return hit
