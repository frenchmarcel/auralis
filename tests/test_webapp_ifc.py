# -*- coding: utf-8 -*-
"""IFC import (skipped without IfcOpenShell): building elements per class/material, and
an IfcSpace as a closed acoustic volume labelled by the nearest elements."""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.server import meshcheck  # noqa: E402


def _model(path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    import ifcopenshell.api as api
    m = ifcopenshell.file(schema="IFC4")
    proj = api.run("root.create_entity", m, ifc_class="IfcProject", name="P")
    api.run("unit.assign_unit", m)
    ctx = api.run("context.add_context", m, context_type="Model")
    body = api.run("context.add_context", m, context_type="Model", context_identifier="Body",
                   target_view="MODEL_VIEW", parent=ctx)
    site = api.run("root.create_entity", m, ifc_class="IfcSite")
    bld = api.run("root.create_entity", m, ifc_class="IfcBuilding")
    sto = api.run("root.create_entity", m, ifc_class="IfcBuildingStorey")
    api.run("aggregate.assign_object", m, relating_object=proj, products=[site])
    api.run("aggregate.assign_object", m, relating_object=site, products=[bld])
    api.run("aggregate.assign_object", m, relating_object=bld, products=[sto])
    concrete = api.run("material.add_material", m, name="Concrete")

    def box(cls, name, size, origin, material=None):
        el = api.run("root.create_entity", m, ifc_class=cls, name=name)
        rep = api.run("geometry.add_wall_representation", m, context=body, length=size[0],
                      thickness=size[1], height=size[2])
        api.run("geometry.assign_representation", m, product=el, representation=rep)
        mat = np.eye(4)
        mat[:3, 3] = origin
        api.run("geometry.edit_object_placement", m, product=el, matrix=mat)
        if cls == "IfcSpace":
            api.run("aggregate.assign_object", m, relating_object=sto, products=[el])
        else:
            api.run("spatial.assign_container", m, relating_structure=sto, products=[el])
        if material is not None:
            api.run("material.assign_material", m, products=[el], material=material)
        return el

    box("IfcSpace", "Room", (5.0, 4.0, 3.0), (0, 0, 0))
    box("IfcSlab", "Floor", (5.0, 4.0, 0.2), (0, 0, -0.2), concrete)
    box("IfcSlab", "Roof", (5.0, 4.0, 0.2), (0, 0, 3.0), concrete)
    box("IfcWall", "S", (5.0, 0.2, 3.0), (0, -0.2, 0), concrete)
    box("IfcWall", "N", (5.0, 0.2, 3.0), (0, 4.0, 0))
    box("IfcWall", "W", (0.2, 4.0, 3.0), (-0.2, 0, 0))
    box("IfcWall", "E", (0.2, 4.0, 3.0), (5.0, 0, 0))
    m.write(path)
    return m


def test_ifc_space_is_a_closed_room_with_element_layers():
    path = os.path.join(tempfile.mkdtemp(), "room.ifc")
    m = _model(path)
    from webapp.server import ifcfile
    inv = ifcfile.inventory(path)
    assert inv["classes"]["IfcWall"] == 4 and len(inv["spaces"]) == 1
    v, f, fl, names, rep = ifcfile.read_ifc(path, space=inv["spaces"][0]["id"])
    s, _closed = meshcheck.analyse(v, f, fl, names)
    assert s["check"]["watertight"] and abs(s["volume_m3"] - 60.0) < 1e-6
    assert set(names) == {"IfcSlab / Concrete", "IfcWall / Concrete", "IfcWall"}
    v, f, fl, names, rep = ifcfile.read_ifc(path)               # elements mode
    assert "IfcSpace" not in " ".join(names) and len(f) == 6 * 12
