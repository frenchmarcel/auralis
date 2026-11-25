# Manifests

A manifest is a JSON file that describes a whole simulation: the geometry,
the materials, the settings and the source and receiver pairs. The web app and
the BRAS benchmark both use this format. The same manifest always gives the
same scene and the same settings.

## Example

```json
{
  "name": "box_example",
  "fs": 48000,
  "duration": 1.2,
  "engine": {"n_rays": 200000, "ism_order": 2, "fdtd": true, "seed": 1234},
  "geometry": {"obj": "geometry.obj"},
  "materials": {
    "plaster_wall": {"absorption": [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06],
                     "scattering": 0.1, "transmission": 0.0},
    "wool_panel": {"phys": {"model": "miki", "sigma": 12000,
                            "thickness_mm": 50, "air_gap_mm": 100},
                   "scattering": 0.3}
  },
  "pairs": [
    {"label": "S1-R1", "source": [1.0, 2.0, 1.5], "receiver": [4.0, 3.0, 1.2]},
    {"label": "S1-R2", "source": [1.0, 2.0, 1.5], "receiver": [2.5, 4.5, 1.2]}
  ]
}
```

## Fields

| Field | Meaning |
|---|---|
| `name` | A name for the run |
| `fs` | Sample rate in Hz |
| `duration` | Length of the responses in seconds. Leave it out for an automatic length. |
| `engine` | Any field of `BakeConfig`, for example `n_rays`, `fdtd`, `fdtd_fcross`, `air_cramer` |
| `geometry.obj` | An OBJ file, relative to the manifest. `usemtl` names must match the material names. |
| `geometry.triangles` | Or the triangles inline, with `tri_mat` |
| `materials` | One entry per material name |
| `pairs` | Source and receiver positions, with a label |

An unknown `engine` field is an error. This catches typos.

### A material entry

| Key | Meaning |
|---|---|
| `absorption` | Seven α values, 125 Hz to 8 kHz |
| `scattering` | One value or seven values |
| `transmission` | One value or seven values (0 by default) |
| `phys` | A physical model, for example a Miki porous layer (`sigma` in Pa·s/m², `thickness_mm`, `air_gap_mm`) |
| `absorption_full` | α at more frequencies for the wave band, for example from 63 Hz: `{"f": [63, 125, ...], "alpha": [...]}` |

When a material has only `absorption`, Auralis looks for a physical model
that reproduces it for the wave band. See {doc}`../user-guide/materials`.

## Using a manifest from Python

```python
import json
from auralis.engine import project

man = json.load(open("box_example.json"))
scene = project.build_scene(man, base_dir=".")
config = project.build_config(man)
```

`build_scene` reads the geometry and the materials. `build_config` makes the
`BakeConfig`. You can then call `bake` for each pair.

## Running a manifest on the benchmark harness

The BRAS harness runs every pair of a manifest and compares the results with
measured responses when they are given:

```bash
python3 BRAS/CR2/cr2_run.py BRAS/CR2/cr2_dodecahedron_fitted31_dir_omni.json -o out_cr2
```

It writes a report (`report.txt`, `report.csv`) and the figures. See
{doc}`../validation/index`.
