# -*- coding: utf-8 -*-
"""SketchUp import: the bench converter run in its own environment, its areas kept, and a
clear refusal when OpenSKP is missing."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import meshcheck, skpfile  # noqa: E402

CR2_SKP = os.path.join(ROOT, "data_bras", "1 Scene descriptions", "CR2 small room (seminar room)",
                       "Geometry", "CR2_RIR_Dodecahedron.skp")


def test_missing_openskp_is_a_clear_refusal(monkeypatch):
    monkeypatch.setenv("AURALIS_SKP_PYTHON", "/nonexistent/python")
    with pytest.raises(ValueError, match="OpenSKP"):
        skpfile.to_obj("in.skp", "out.obj")


@pytest.mark.skipif(not (skpfile.available() and os.path.exists(CR2_SKP)),
                    reason="OpenSKP environment or BRAS CR2 model not installed")
def test_cr2_sketchup_model_imports_with_its_areas():
    out = os.path.join(tempfile.mkdtemp(), "cr2.obj")
    report = skpfile.to_obj(CR2_SKP, out)
    v, f, fl, names, _notes = meshcheck.import_model(out, unit="m", up_axis="z")
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    got = {n: float(area[fl == i].sum()) for i, n in enumerate(names)}
    # the converter's own report (BRAS/skp_to_obj.py, areas exact on CR2-CR4)
    for ln in report:
        parts = ln.split()
        if len(parts) == 3 and parts[0].startswith("mat_CR2_"):
            assert got[parts[0]] == pytest.approx(float(parts[1]), abs=0.01)
    assert got["mat_CR2_floor"] == pytest.approx(49.29, abs=0.01)
    assert np.ptp(v[:, 2]) == pytest.approx(2.988, abs=0.01)          # Z up, metres
