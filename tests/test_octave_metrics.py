import numpy as np

from auralis.engine import rirmetrics as rm


def _decay(fs, lo, hi, t60, dur, rng):
    """Noise band-limited to [lo, hi] Hz with an exponential decay t60."""
    n = int(dur * fs)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(X, n)
    return x / np.std(x) * 10 ** (-3 * np.arange(n) / fs / t60)


def test_125_hz_octave_ignores_content_below_88_hz():
    """A long 30-50 Hz decay (like the low-frequency noise of a measured hall) must not leak
    into the 125 Hz T30. The engine's synthesis band reaches down to 22 Hz and would give
    ~5.9 s here instead of 1 s."""
    fs = 44100
    clean, mixed = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        ir = _decay(fs, 100, 160, 1.0, 3.5, rng)
        low = _decay(fs, 30, 50, 6.0, 3.5, rng)
        clean.append(rm.band_metrics(ir, fs)["T30"][0])
        mixed.append(rm.band_metrics(ir + 0.3 * low, fs)["T30"][0])
    assert abs(np.mean(clean) - 1.0) < 0.08, clean
    assert np.max(np.abs(np.subtract(mixed, clean))) < 0.02, (clean, mixed)


def test_levels_are_relative_to_the_seven_octaves():
    fs = 44100
    rng = np.random.default_rng(5)
    m = rm.band_metrics(rng.standard_normal(fs) * np.exp(-np.arange(fs) / fs * 6.9), fs)
    assert abs(np.sum(10 ** (np.asarray(m["level_db"]) / 10)) - 1.0) < 1e-9
