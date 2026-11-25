# -*- coding: utf-8 -*-
"""Imported directivities: a cardioid given as a CSV or as an AES69 SOFA (synthetic,
skipped without h5py) comes back as the engine's cardioid table, used by the runs."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auralis.engine.directivity import Directivity  # noqa: E402
from webapp.server import directivities as dl  # noqa: E402


def _cardioid(theta_deg):
    return (0.5 * (1 + np.cos(np.radians(theta_deg)))) ** 2       # energy


def test_csv_cardioid_and_its_use_in_a_source():
    root = tempfile.mkdtemp()
    th = np.arange(0, 181, 10.0)
    db = 10 * np.log10(np.maximum(_cardioid(th), 1e-6))
    p = os.path.join(root, "c.csv")
    with open(p, "w") as fh:
        fh.write("theta_deg,125,250,500,1000,2000,4000,8000\n")
        for t, d in zip(th, db):
            fh.write("%g,%s\n" % (t, ",".join(["%.4f" % d] * 7)))
    rec = dl.import_file(root, p, "Cardioid", "c.csv")
    d = dl.engine_directivity(root, rec["id"])
    assert isinstance(d, Directivity)
    q = d.energy(np.cos(np.radians([0, 60, 90, 120])))[:, 3]
    assert np.allclose(q, _cardioid(np.array([0, 60, 90, 120])), atol=0.02)
    assert dl.known(root, rec["id"]) and not dl.known(root, "custom-deadbeef")
    from webapp.server import runner
    s = runner.source_def({"id": "S", "position": [0, 0, 1], "directivity": rec["id"]}, root)
    assert isinstance(s.directivity, Directivity)
    items = {x["id"]: x for x in dl.listing(root)}
    assert items[rec["id"]]["kind"] == "imported" and items["omni"]["kind"] == "preset"
    assert abs(items[rec["id"]]["di_db"][3] - 10 * np.log10(3.0)) < 0.2   # cardioid DI 4.8 dB


def test_sofa_free_field_directivity():
    h5py = pytest.importorskip("h5py")
    root = tempfile.mkdtemp()
    az = np.arange(0, 360, 15.0)
    el = np.arange(-75, 90, 15.0)
    A, E = np.meshgrid(az, el)
    pos = np.stack([A.ravel(), E.ravel(), np.ones(A.size)], 1)
    d = np.stack([np.cos(np.radians(pos[:, 1])) * np.cos(np.radians(pos[:, 0])),
                  np.cos(np.radians(pos[:, 1])) * np.sin(np.radians(pos[:, 0])),
                  np.sin(np.radians(pos[:, 1]))], 1)
    theta = np.degrees(np.arccos(d[:, 0]))                         # axis +x
    freqs = np.geomspace(100, 10000, 40)
    amp = np.sqrt(_cardioid(theta))[:, None, None] * np.ones((1, 1, len(freqs)))
    p = os.path.join(root, "c.sofa")
    with h5py.File(p, "w") as f:
        f.attrs["SOFAConventions"] = "FreeFieldDirectivityTF"
        f["Data.Real"] = amp
        f["Data.Imag"] = np.zeros_like(amp)
        f["N"] = freqs
        f["ReceiverPosition"] = pos[:, None, :]
        f["ReceiverPosition"].attrs["Type"] = "spherical"
        f["SourceView"] = np.array([[1.0, 0.0, 0.0]])
        f["SourceView"].attrs["Type"] = "cartesian"
    rec = dl.import_file(root, p, "", "c.sofa")
    dirv = dl.engine_directivity(root, rec["id"])
    q = dirv.energy(np.cos(np.radians([0, 45, 90, 135])))[:, 4]
    assert np.allclose(q, _cardioid(np.array([0, 45, 90, 135])), atol=0.03)
    assert rec["asymmetry_db"] < 1.0


def _so8(level_db):
    """Synthetic .SO8 bytes from level_db(theta_deg, roll_deg, band) (8 bands, 63 Hz-8 kHz),
    in the layout read by read_so8 (normalised to the total power like Odeon's)."""
    th = 180.0 - 10.0 * np.arange(19)
    roll = 10.0 * np.arange(37)
    g = np.array([[[level_db(t, r, b) for b in range(8)] for t in th] for r in roll])
    w = np.cos(np.radians(th - 90.0))[None, :] * np.ones((37, 1))
    w[-1] = 0.0
    tot = 10 * np.log10((10 ** (g / 10) * w[:, :, None]).sum(axis=(0, 1)) * np.radians(10) ** 2)
    recs = np.concatenate([np.zeros((1, 8)), np.full((1, 8), -31.0), (g - tot).reshape(-1, 8)])
    return b"\x00" * 10 + b"\x01\x00" + np.float32(1.0).tobytes() + recs.astype("<f4").tobytes()


def test_so8_layout_axis_and_rings():
    def cardioid(t, r, b):
        return 20 * np.log10(max(0.5 * (1 + np.cos(np.radians(t))), 1e-3)) * (b / 7.0)
    table, asym, spl = dl.read_so8(_so8(cardioid))
    assert spl is None and asym < 1e-6
    for t in (0, 40, 90, 130):
        want = 10 ** (cardioid(t, 0, np.arange(1, 8)) / 10) / 10 ** (cardioid(0, 0, np.arange(1, 8)) / 10)
        assert np.allclose(table[:, t], want, rtol=1e-4), t      # 63 Hz dropped, axis = 1
    table, asym, _ = dl.read_so8(_so8(lambda t, r, b: 6.0 * np.cos(np.radians(r)) * (0 < t < 180)))
    assert 11.5 < asym < 12.5 and np.allclose(table[:, 0], 1.0)   # +-6 dB around the ring


@pytest.mark.skipif(not dl.odeon_patterns(), reason="local Odeon archives absent")
def test_odeon_patterns_are_physical():
    pats = dl.odeon_patterns()
    flat = Directivity(np.asarray(pats["odeon-omni"]["table"]))
    assert np.allclose(flat.directivity_index_db(), 0.0, atol=0.05)
    talk = np.asarray(pats["odeon-tlkraise"]["table"])
    assert 10 * np.log10(talk[5, 180]) < -12.0                   # 4 kHz behind the head
    w = np.sin(np.radians(np.arange(181)))

    def rear_front_db(did):
        t = np.asarray(pats[did]["table"])
        return 10 * np.log10((t[:, 91:] * w[91:]).sum(1) / (t[:, :90] * w[:90]).sum(1))
    assert (rear_front_db("odeon-french-horn") > 0.5).all()       # the bell points backwards
    assert (rear_front_db("odeon-singer")[4:] < -5.0).all()       # voices face forward up high
    raised = np.asarray(pats["odeon-tlkraise"]["on_axis_spl_1m_db"])
    assert np.all(np.abs(raised[2:6] - np.array([66.0, 60.0, 54.0, 48.0])) < 4.0)   # IEC raised
    assert dl.known("/nonexistent", "odeon-violin")
    for name, want in (("1st violinA", "odeon-violin"), ("vla_1_2", "odeon-viola"),
                       ("4Seasons_Alto", "odeon-viola"), ("vl_2_1_3", "odeon-violin"),
                       ("cbass_1", None), ("double bass", None), ("tuba", None),
                       ("hornG", "odeon-french-horn"), ("BluesA_Voc", "odeon-singer"),
                       ("trp_1", "odeon-trumpet"), ("timpani", None)):
        assert dl.instrument_pattern(name) == want, name
