# -*- coding: utf-8 -*-
"""OBJ layers keep their whole usemtl name: names with spaces stay distinct layers."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auralis.engine.project import read_obj  # noqa: E402


def test_usemtl_names_with_spaces_stay_distinct():
    p = os.path.join(tempfile.mkdtemp(), "m.obj")
    with open(p, "w") as fh:
        fh.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                 "usemtl Wall A\nf 1 2 3\nusemtl Wall B  \nf 1 2 4\nusemtl floor\nf 1 3 4\n")
    _v, f, layer = read_obj(p)
    assert layer == ["Wall A", "Wall B", "floor"] and len(f) == 3
