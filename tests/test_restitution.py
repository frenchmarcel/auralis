# -*- coding: utf-8 -*-
"""Restitution tests: stereo loudspeaker back end and measured HRTFs. The HRTF
fixture is pure numpy. The SOFA converter test needs netCDF4 or h5py.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auralis.engine import make_shoebox                      # noqa: E402
from auralis.engine.pipeline import (BakeConfig, SourceDef,  # noqa: E402
                                     ReceiverDef, bake)
from auralis.engine.hrtf import (FIR_TAPS, _FIR_NFFT,        # noqa: E402
                                 _shadow_magnitude, itd_seconds,
                                 pan_gains_stereo, event_fir_stereo,
                                 late_tail_stereo, event_fir_pair)
from auralis.engine.hrtf_sofa import HrtfSet, load_hrtf      # noqa: E402
from auralis.engine.ir_synthesis import (synthesize_stereo,  # noqa: E402
                                         synthesize_binaural)
from auralis.engine.motion import (render_moving, MotionConfig,  # noqa: E402
                                   MovingSource, Trajectory)

FS = 48000
_TMP = tempfile.mkdtemp(prefix="aur_resti_")


# --------------------------------------------------------------- helpers
def _causal_fir(mag):
    win = np.hanning(FIR_TAPS)
    return np.roll(np.fft.irfft(mag, 512), FIR_TAPS // 2)[:FIR_TAPS] * win


def _place(h, n_out, delay):
    H = np.fft.rfft(h, 2 * n_out)
    k = np.arange(len(H))
    y = np.fft.irfft(H * np.exp(-2j * np.pi * k * delay / (2 * n_out)),
                     2 * n_out)
    return y[:n_out]


def make_param_npz(path, fs=FS, az_step=30.0, el_vals=(-15.0, 0.0, 15.0,
                                                       30.0, 45.0),
                   taps=160, guard=4.0):
    """HRTF npz set sampling the parametric model (pure numpy), IRs
    aligned on the origin + separate delays, like the converter."""
    freqs = np.fft.rfftfreq(512, 1.0 / fs)
    right = np.array([0.0, -1.0, 0.0])
    az = np.arange(0.0, 360.0, az_step)
    AZ, EL = np.meshgrid(az, np.asarray(el_vals), indexing="ij")
    AZ, EL = AZ.ravel(), EL.ravel()
    M = len(AZ)
    ir = np.zeros((M, 2, taps), dtype=np.float32)
    dly = np.zeros((M, 2), dtype=np.float32)
    for m in range(M):
        a, e = np.radians(AZ[m]), np.radians(EL[m])
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                      np.sin(e)])
        tau, nr = itd_seconds(d, right, 343.0)
        dL, dR = (tau, 0.0) if nr else (0.0, tau)
        for ei, (ce, dd) in enumerate(((float(d[1]), dL),
                                       (float(-d[1]), dR))):
            h = _causal_fir(_shadow_magnitude(freqs, ce, 343.0))
            ir[m, ei] = np.concatenate(
                [h, np.zeros(taps - FIR_TAPS)]).astype(np.float32)
            dly[m, ei] = dd * fs
    np.savez_compressed(path, ir=ir, delay=dly,
                        az_deg=AZ.astype(np.float32),
                        el_deg=EL.astype(np.float32),
                        fs=fs, name="fixture-param")
    return path


_FIX = make_param_npz(os.path.join(_TMP, "fix.npz"))
_BASIS = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
          np.array([0.0, -1.0, 0.0]))


def _dead_box(dims=(14.0, 12.0, 6.0)):
    dead = {"name": "anecho", "absorption": [0.99] * 7,
            "scattering": [0.1] * 7, "transmission": [0.0] * 7}
    return make_shoebox(dims, wall_material=dead)


def _itd_xcorr_us(ir, fs=FS, up=8, span=96):
    n = len(ir)
    X = np.fft.rfft(ir[:, 0], 2 * n) * np.conj(np.fft.rfft(ir[:, 1], 2 * n))
    nu = 2 * n * up
    Xu = np.zeros(nu // 2 + 1, dtype=np.complex128)
    Xu[: len(X)] = X
    xc = np.fft.irfft(Xu, nu)
    sp = span * up
    xc = np.concatenate([xc[-sp:], xc[: sp + 1]])
    return (int(np.argmax(xc)) - sp) / up / fs * 1e6


def _band_e(ir, lo=500.0, hi=4000.0):
    H = np.abs(np.fft.rfft(ir, 1 << 15, axis=0)) ** 2
    f = np.fft.rfftfreq(1 << 15, 1.0 / FS)
    m = (f > lo) & (f < hi)
    return H[m].sum(axis=0)


# ----------------------------------------------------------------- tests
def test_constant_power_pan():
    for lat in np.linspace(-1.0, 1.0, 41):
        gL, gR = pan_gains_stereo(lat, 30.0)
        assert abs(gL * gL + gR * gR - 2.0) < 1e-12
    gL, gR = pan_gains_stereo(1.0, 30.0)
    assert abs(gL - np.sqrt(2.0)) < 1e-12 and abs(gR) < 1e-12
    gL, gR = pan_gains_stereo(-1.0, 30.0)
    assert abs(gR - np.sqrt(2.0)) < 1e-12 and abs(gL) < 1e-12
    gL, gR = pan_gains_stereo(0.0, 30.0)
    assert abs(gL - 1.0) < 1e-12 and abs(gR - 1.0) < 1e-12


def test_event_fir_stereo_lateralisation():
    amps = np.ones(7)
    right = np.array([1.0, 0.0, 0.0])
    fl, fr, dl, dr = event_fir_stereo(amps, np.array([1.0, 0.0, 0.0]),
                                      right, FS, 30.0)   # source at right
    assert dl == 0.0 and dr == 0.0
    assert (fr ** 2).sum() > 100.0 * (fl ** 2).sum()
    fl2, fr2, _, _ = event_fir_stereo(amps, np.array([-1.0, 0.0, 0.0]),
                                      right, FS, 30.0)
    assert (fl2 ** 2).sum() > 100.0 * (fr2 ** 2).sum()
    assert abs((fl2 ** 2).sum() - (fr ** 2).sum()) < 1e-9


def test_tail_stereo_energie():
    nb, n_bins = 7, 40
    hist = np.full((nb, n_bins), 2.5e-4)
    dirs = np.zeros((n_bins, 3))          # no laterality -> centered
    rng = np.random.default_rng(3)
    L, R = late_tail_stereo(hist, dirs, 0.004, FS, int(0.004 * n_bins * FS),
                            np.array([1.0, 0.0, 0.0]), 30.0, rng)
    tot = (L ** 2).sum() + (R ** 2).sum()
    # the tail converts INTEGRATED band energy into physical
    # amplitude via band_energy_fraction (same units as the
    # events; checked against the exact image lattice):
    from auralis.engine.hrtf import band_energy_fraction
    fac = band_energy_fraction(int(0.004 * n_bins * FS), FS)
    ref = 2.0 * float((hist.sum(axis=1) * fac).sum())
    assert abs(tot / ref - 1.0) < 0.08
    assert abs((L ** 2).sum() / (R ** 2).sum() - 1.0) < 0.12


def test_synthese_stereo_direct():
    class Ev:
        pass
    ev = Ev()
    ev.amps = np.ones(7)
    ev.time = 0.005
    ev.direction = np.array([np.sin(np.radians(20.0)),
                             np.cos(np.radians(20.0)), 0.0])  # 20 deg right
    ev.bounces = []
    hist = np.zeros((7, 4))
    dirs = np.zeros((4, 3))
    ir, info = synthesize_stereo([ev], hist, dirs, 0.004, FS, 343.0)
    assert abs(_itd_xcorr_us(ir)) < 3.0
    gL, gR = pan_gains_stereo(-float(ev.direction[0]), 30.0)
    mes = 10 * np.log10(_band_e(ir)[1] / _band_e(ir)[0])
    att = 20 * np.log10(gR / gL)
    assert abs(mes - att) < 0.3


def test_bake_stereo_stats_identiques():
    sc = _dead_box()
    src = SourceDef("S", np.array([5.0, 6.0, 1.6]))
    rcv = ReceiverDef("R", np.array([2.5, 6.0, 1.6]))
    kw = dict(fs=FS, n_rays=2000, normalize=False, seed=4, max_time=0.15)
    rb = bake(sc, src, rcv, BakeConfig(**kw))
    rs = bake(sc, src, rcv, BakeConfig(**kw, output_layout="stereo"))
    assert rs.stats["stereo"] == 1.0 and rb.stats["stereo"] == 0.0
    for key in ("spl_dba", "spl_db", "g_mid"):
        assert abs(rb.stats[key] - rs.stats[key]) < 1e-9
    assert rs.ir.shape[1] == 2


def test_bake_stereo_lateral():
    sc = _dead_box()
    src = SourceDef("S", np.array([5.0, 6.0, 1.6]))     # right of receiver
    rcv = ReceiverDef("R", np.array([2.5, 6.0, 1.6]))
    r = bake(sc, src, rcv, BakeConfig(fs=FS, n_rays=2000, normalize=False,
                                      seed=4, max_time=0.15,
                                      output_layout="stereo"))
    eL, eR = _band_e(r.ir)
    assert 10 * np.log10(eR / eL) > 12.0
    assert abs(_itd_xcorr_us(r.ir)) < 5.0


def test_loader_exact_grid():
    hs = HrtfSet(_FIX, FS)
    z = np.load(_FIX)
    m = int(np.where((z["az_deg"] == 90.0) & (z["el_deg"] == 0.0))[0][0])
    d = np.array([0.0, 1.0, 0.0])
    HL, HR, dl, dr = hs.ear_responses(d, _BASIS, _FIR_NFFT)
    refL = np.fft.rfft(z["ir"][m, 0].astype(np.float64), _FIR_NFFT)
    assert np.max(np.abs(HL - refL)) < 1e-6
    assert abs(dl - z["delay"][m, 0] / FS) < 1e-9
    assert abs(dr - z["delay"][m, 1] / FS) < 1e-9


def test_loader_interpolation():
    hs = HrtfSet(_FIX, FS)
    d = np.array([np.cos(np.radians(15.0)), np.sin(np.radians(15.0)), 0.0])
    HL, HR, dl, dr = hs.ear_responses(d, _BASIS, _FIR_NFFT)
    right = np.array([0.0, -1.0, 0.0])
    tA = itd_seconds(np.array([1.0, 0.0, 0.0]), right, 343.0)
    tB = itd_seconds(np.array([np.cos(np.radians(30.0)),
                               np.sin(np.radians(30.0)), 0.0]), right, 343.0)
    iA = tA[0] if tA[1] else -tA[0]
    iB = tB[0] if tB[1] else -tB[0]
    lo, hi = min(iA, iB), max(iA, iB)
    assert lo - 5e-6 <= (dl - dr) <= hi + 5e-6
    assert np.isfinite(HL).all() and np.isfinite(HR).all()


def test_loader_resampling():
    p44 = os.path.join(_TMP, "fix44.npz")
    make_param_npz(p44, fs=44100)
    h44 = HrtfSet(p44, FS)                 # 44.1 kHz opened at 48 kHz
    h48 = HrtfSet(_FIX, FS)
    d = np.array([0.0, 1.0, 0.0])
    HL44, _, dl44, dr44 = h44.ear_responses(d, _BASIS, _FIR_NFFT)
    HL48, _, dl48, dr48 = h48.ear_responses(d, _BASIS, _FIR_NFFT)
    f = np.fft.rfftfreq(_FIR_NFFT, 1.0 / FS)
    band = (f > 300) & (f < 8000)
    rel = np.abs(np.abs(HL44[band]) - np.abs(HL48[band])) \
        / np.maximum(np.abs(HL48[band]), 1e-6)
    assert float(np.median(rel)) < 0.05
    assert abs((dl44 - dr44) - (dl48 - dr48)) < 8e-6


def test_bake_sofa_boucle_fermee():
    sc = _dead_box()
    src = SourceDef("S", np.array([5.0, 6.0, 1.6]))
    rcv = ReceiverDef("R", np.array([2.5, 6.0, 1.6]))
    kw = dict(fs=FS, n_rays=2000, normalize=False, seed=4, max_time=0.15)
    rp = bake(sc, src, rcv, BakeConfig(**kw))
    rh = bake(sc, src, rcv, BakeConfig(**kw, hrtf_path=_FIX))
    itd_p, itd_h = _itd_xcorr_us(rp.ir), _itd_xcorr_us(rh.ir)
    assert abs(itd_p - itd_h) < 30.0
    ep, eh = _band_e(rp.ir), _band_e(rh.ir)
    ild_p = 10 * np.log10(ep[1] / ep[0])
    ild_h = 10 * np.log10(eh[1] / eh[0])
    assert abs(ild_p - ild_h) < 1.0
    assert np.all(np.abs(10 * np.log10(eh / ep)) < 1.5)


def test_contrat_placement_sofa():
    hs = HrtfSet(_FIX, FS)

    class Ev:
        pass
    ev = Ev()
    ev.amps = np.ones(7)
    ev.time = 0.005
    ev.direction = np.array([0.0, 1.0, 0.0])
    ev.bounces = []
    hist = np.zeros((7, 4))
    dirs = np.zeros((4, 3))
    ir, _ = synthesize_binaural([ev], hist, dirs, 0.004, FS, 343.0,
                                normalize=False, hrtf_set=hs)
    k = int(np.argmax(np.abs(ir[:, 0])))
    expected = ev.time * FS + hs.delay[0, 0] * 0.0  # onset ~ t + dL
    HLd = hs.ear_responses(ev.direction, _BASIS, _FIR_NFFT)[2]
    expected = ev.time * FS + HLd * FS
    assert abs(k - expected) < FIR_TAPS  # peak within the onset window


def test_motion_stereo_traverse():
    sc = _dead_box((12.0, 9.0, 3.2))
    dry = np.zeros(FS // 2)
    dry[::3000] = 1.0
    traj = Trajectory(times=[0.0, 0.5],
                      positions=[[3.0, 6.5, 1.5], [9.0, 6.5, 1.5]])
    msrc = MovingSource("t", traj)
    rcv = ReceiverDef("R", np.array([6.0, 2.0, 1.5]),
                      forward=(0.0, 1.0, 0.0))
    cfg = BakeConfig(fs=FS, n_rays=800, seed=7, max_time=0.1,
                     normalize=False, output_layout="stereo")
    out, _ = render_moving(sc, msrc, rcv, dry, MotionConfig(base=cfg))
    q = len(out) // 4

    def dom(seg):
        return 10 * np.log10(float((seg[:, 1] ** 2).sum())
                             / max(float((seg[:, 0] ** 2).sum()), 1e-30))
    d0, d1 = dom(out[:q]), dom(out[-q:])
    assert d0 < 0.0 < d1 and (d1 - d0) > 1.5


def test_motion_tail_by_keys_is_exact():
    """The per-key late-field convolution is an exact rewrite of the per-hop
    cross-fade (same output to rounding), moving source AND moving receiver."""
    from auralis.engine.motion import MovingReceiver
    sc = _dead_box((12.0, 9.0, 3.2))
    rng = np.random.default_rng(5)
    dry = rng.standard_normal(int(1.3 * FS)) * 0.1
    msrc = MovingSource("s", Trajectory(times=[0.0, 1.3],
                                        positions=[[3.0, 6.5, 1.5], [9.0, 6.0, 1.6]]))
    mrcv = MovingReceiver("r", Trajectory(times=[0.2, 1.1],
                                          positions=[[6.0, 2.0, 1.5], [7.0, 3.0, 1.5]],
                                          forwards=[[0, 1, 0], [1, 0, 0]]))
    # CPU tracer: the GPU one differs by ~1e-9 from one call to the next (float atomics)
    cfg = BakeConfig(fs=FS, n_rays=800, seed=7, max_time=0.4, normalize=False, engine="native")
    outs = [render_moving(sc, msrc, mrcv, dry,
                          MotionConfig(base=cfg, reverb_dt=0.3, tail_by_keys=b))[0]
            for b in (False, True)]
    assert outs[0].shape == outs[1].shape
    assert np.abs(outs[1] - outs[0]).max() < 1e-9 * np.abs(outs[0]).max()


def test_convertisseur_sofa():
    try:
        import netCDF4  # noqa: F401
        writer = "netcdf4"
    except ImportError:
        try:
            import h5py  # noqa: F401
            writer = "h5py"
        except ImportError:
            print("    (skipped: netCDF4 and h5py missing)")
            return
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "tools"))
    from sofa_to_npz import convert
    fs, N, T = FS, 320, FIR_TAPS
    freqs = np.fft.rfftfreq(512, 1.0 / fs)
    right = np.array([0.0, -1.0, 0.0])
    az = np.arange(0.0, 360.0, 30.0)
    el = np.array([-15.0, 0.0, 30.0])
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    AZ, EL = AZ.ravel(), EL.ravel()
    M = len(AZ)
    IR = np.zeros((M, 2, N))
    for m in range(M):
        a, e = np.radians(AZ[m]), np.radians(EL[m])
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                      np.sin(e)])
        tau, nr = itd_seconds(d, right, 343.0)
        dl, dr = (tau, 0.0) if nr else (0.0, tau)
        for ei, (ce, dd) in enumerate(((float(d[1]), dl),
                                       (float(-d[1]), dr))):
            h = _causal_fir(_shadow_magnitude(freqs, ce, 343.0))
            IR[m, ei] = _place(np.concatenate([h, np.zeros(N - T)]),
                               N, 40.0 + dd * fs)
    sofa = os.path.join(_TMP, "mini.sofa")
    if writer == "netcdf4":
        from netCDF4 import Dataset
        ds = Dataset(sofa, "w", format="NETCDF4")
        for dim, sz in (("M", M), ("R", 2), ("N", N), ("I", 1), ("C", 3)):
            ds.createDimension(dim, sz)
        v = ds.createVariable("Data.IR", "f8", ("M", "R", "N")); v[:] = IR
        v = ds.createVariable("Data.SamplingRate", "f8", ("I",)); v[:] = fs
        v = ds.createVariable("SourcePosition", "f8", ("M", "C"))
        v[:] = np.stack([AZ, EL, np.full(M, 1.5)], 1)
        v = ds.createVariable("ReceiverPosition", "f8", ("R", "C"))
        v[:] = [[0, 0.09, 0], [0, -0.09, 0]]
        ds.close()
    else:
        import h5py
        f = h5py.File(sofa, "w")
        f["Data.IR"] = IR
        f["Data.SamplingRate"] = np.array([fs], dtype=np.float64)
        f["SourcePosition"] = np.stack([AZ, EL, np.full(M, 1.5)], 1)
        f["ReceiverPosition"] = np.array([[0, 0.09, 0], [0, -0.09, 0]],
                                         dtype=np.float64)
        f.close()
    npz = os.path.join(_TMP, "mini.npz")
    convert(sofa, npz, taps=192)
    hs = HrtfSet(npz, fs)
    d = np.array([0.0, 1.0, 0.0])
    HL, HR, dl, dr = hs.ear_responses(d, _BASIS, _FIR_NFFT)
    tau = itd_seconds(d, right, 343.0)[0]
    assert abs((dl - dr) - (-tau)) < 5e-6
    ref = _shadow_magnitude(np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs),
                            1.0, 343.0)
    sel = ref > 0.05
    rel = np.abs(np.abs(HL[sel]) - ref[sel]) / ref[sel]
    f = np.fft.rfftfreq(_FIR_NFFT, 1.0 / fs)
    rel = rel[(f[sel] > 200) & (f[sel] < 9000)]
    assert float(rel.max()) < 0.08


TESTS = [
    test_constant_power_pan,
    test_event_fir_stereo_lateralisation,
    test_tail_stereo_energie,
    test_synthese_stereo_direct,
    test_bake_stereo_stats_identiques,
    test_bake_stereo_lateral,
    test_loader_exact_grid,
    test_loader_interpolation,
    test_loader_resampling,
    test_bake_sofa_boucle_fermee,
    test_contrat_placement_sofa,
    test_motion_stereo_traverse,
    test_motion_tail_by_keys_is_exact,
    test_convertisseur_sofa,
]

if __name__ == "__main__":
    ok = fail = 0
    for t in TESTS:
        try:
            t()
            print("ok    %s" % t.__name__)
            ok += 1
        except Exception as exc:
            print("FAIL  %s: %r" % (t.__name__, exc))
            fail += 1
    print("\n%d ok, %d failures" % (ok, fail))
    sys.exit(1 if fail else 0)
