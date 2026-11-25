# -*- coding: utf-8 -*-
"""Custom materials: rename keeps the id, duplicate copies into a new id with the licence of
the original, and an id is never reused after a delete."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server.library import Library  # noqa: E402

ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _lib(tmp_path):
    return Library(str(tmp_path / "materials_custom.json"))


def test_rename_keeps_id_and_refuses_empty_or_taken(tmp_path):
    lib = _lib(tmp_path)
    a = lib.add_custom("Wool", "", "", ALPHA, 0.1)
    b = lib.add_custom("Board", "", "", ALPHA, 0.1)
    r = lib.rename_custom(a["id"], "  Rock   wool ")
    assert r["id"] == a["id"] and r["name"] == "Rock wool" and r["absorption"] == a["absorption"]
    assert _lib(tmp_path).get(a["id"])["name"] == "Rock wool"            # persisted
    for bad in ("", "   ", None, "board", " BOARD "):
        with pytest.raises(ValueError):
            lib.rename_custom(a["id"], bad)
    assert lib.rename_custom(b["id"], "board")["name"] == "board"        # its own name, recased
    with pytest.raises(ValueError):
        lib.rename_custom("auralis:concrete", "Mine")
    with pytest.raises(KeyError):
        lib.rename_custom("custom:999", "x")


def test_duplicate_builtin_and_custom(tmp_path):
    lib = _lib(tmp_path)
    c = lib.duplicate("auralis:concrete")
    src = lib.get("auralis:concrete")
    assert c["id"].startswith("custom:") and c["library"] == "custom"
    assert c["name"] == src["name"] + " (copy)" and c["derived_from"] == "auralis:concrete"
    assert c["absorption"] == src["absorption"] and c["scattering"] == src["scattering"]
    assert c["license"] == src["license"] and c["redistributable"] is True
    c2 = lib.duplicate("auralis:concrete")
    assert c2["name"] == src["name"] + " (copy 2)" and c2["id"] != c["id"]
    with pytest.raises(ValueError):
        lib.duplicate("auralis:concrete", c["name"].upper())
    d = lib.duplicate(c["id"], "My concrete")
    assert d["name"] == "My concrete" and d["derived_from"] == c["id"]
    with pytest.raises(KeyError):
        lib.duplicate("nope:1")


def test_duplicate_of_a_local_material_stays_local(tmp_path):
    lib = _lib(tmp_path)
    m = lib.add_local("odeon_rooms", "hall:1", "Seats", "Odeon", ALPHA, 0.3, "Odeon hall")
    c = lib.duplicate(m["id"])
    assert c["redistributable"] is False and c["license"] == m["license"]
    assert lib.duplicate(c["id"])["redistributable"] is False


def test_physical_and_clipped_flags_survive(tmp_path):
    lib = _lib(tmp_path)
    m = lib.add_custom("Hot", "", "", [0.99] * 7, 0.1,
                       physical={"type": "porous", "model": "miki", "sigma": 1e4,
                                 "thickness_mm": 50.0, "air_gap_mm": 0.0})
    c = lib.duplicate(m["id"])
    assert c["clipped"] and c["physical"] == m["physical"]


def test_ids_are_never_reused(tmp_path):
    lib = _lib(tmp_path)
    a = lib.add_custom("A", "", "", ALPHA, 0.1)
    b = lib.add_custom("B", "", "", ALPHA, 0.1)
    lib.delete_custom(b["id"])
    c = lib.add_custom("C", "", "", ALPHA, 0.1)
    assert c["id"] not in (a["id"], b["id"])
    lib.delete_custom(c["id"])
    d = _lib(tmp_path).duplicate(a["id"])                 # also across a restart
    assert d["id"] not in (a["id"], b["id"], c["id"])


def test_ids_not_reused_in_a_store_made_before_the_counter(tmp_path):
    lib = _lib(tmp_path)
    lib.add_custom("A", "", "", ALPHA, 0.1)
    b = lib.add_custom("B", "", "", ALPHA, 0.1)
    os.remove(str(tmp_path / "materials_custom_seq.json"))     # as an older version left it
    lib = _lib(tmp_path)
    lib.delete_custom(b["id"])
    assert lib.add_custom("C", "", "", ALPHA, 0.1)["id"] != b["id"]
