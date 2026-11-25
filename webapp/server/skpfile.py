# -*- coding: utf-8 -*-
"""SketchUp .skp import: BRAS/skp_to_obj.py (OpenSKP, MIT) run in its own environment, which
the app's Python does not carry. Metres, Z up, one layer per SketchUp material."""
from __future__ import annotations

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERTER = os.path.join(REPO, "BRAS", "skp_to_obj.py")
DEFAULT_PY = os.path.expanduser("~/auralis-tools/skp-venv/bin/python")
TIMEOUT_S = 600


def python():
    """The interpreter that has OpenSKP: AURALIS_SKP_PYTHON, else the default venv."""
    return os.environ.get("AURALIS_SKP_PYTHON") or DEFAULT_PY


def available():
    return os.path.isfile(python()) and os.path.isfile(CONVERTER)


def to_obj(skp_path, obj_path):
    """Convert; returns the converter's report lines (materials, areas, bounding box).
    ValueError with the reason when the converter is missing or fails."""
    if not available():
        raise ValueError(
            "SketchUp import needs OpenSKP in its own environment, not found at %s. Install it "
            "once: python3 -m venv ~/auralis-tools/skp-venv && ~/auralis-tools/skp-venv/bin/pip "
            "install openskp (or set AURALIS_SKP_PYTHON); or export the model to .obj from "
            "SketchUp." % python())
    try:
        out = subprocess.run([python(), CONVERTER, skp_path, obj_path], capture_output=True,
                             text=True, timeout=TIMEOUT_S, cwd=REPO)
    except subprocess.TimeoutExpired:
        raise ValueError("SketchUp conversion took more than %d s: export the model to .obj "
                         "from SketchUp instead" % TIMEOUT_S)
    if out.returncode != 0 or not os.path.exists(obj_path):
        tail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
        raise ValueError("SketchUp conversion failed (OpenSKP reads SketchUp 2013-2020 files): "
                         + " | ".join(tail))
    return [ln.rstrip() for ln in out.stdout.splitlines() if ln.strip()]
