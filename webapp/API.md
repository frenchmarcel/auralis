# Auralis web app: HTTP API contract

Server: FastAPI on `http://127.0.0.1:8765`. Every route is under `/api`.
In development, the Vite dev server (port 5173) proxies `/api` to it. In
production, the server also serves the built UI (`webapp/web/dist`) at `/`.

Conventions:
- JSON everywhere, except uploads (multipart) and audio (WAV).
- Errors: HTTP 4xx/5xx with `{"detail": "<message>"}`.
- Coordinates in metres, **Z up** (engine convention), right-handed.
- Octave bands, always in this order: `[125, 250, 500, 1000, 2000, 4000, 8000]` Hz.
  A metric that cannot be computed is `null`, never a fake number.
- Ids are short opaque strings. Timestamps are ISO 8601 strings.

## Health

`GET /api/health` → `{"native": bool, "gpu": bool, "version": "0.1"}`

## Projects, spaces, geometries (hierarchy)

`GET /api/projects` → `Project[]`, the whole tree for the home page:

```json
{"id": "p1", "name": "Demo", "description": "", "created": "...",
 "spaces": [
   {"id": "s1", "project_id": "p1", "name": "Club", "description": "", "created": "...",
    "geometries": [
      {"id": "g1", "space_id": "s1", "name": "Geometry 1", "created": "...",
       "n_triangles": 508, "n_layers": 5, "watertight": true,
       "simulations": [
         {"id": "m1", "geometry_id": "g1", "name": "Simulation 1", "created": "...",
          "status": "draft|queued|running|done|failed|cancelled",
          "last_run": "...|null", "results_stale": false, "has_results": true}
       ]}
    ]}
 ]}
```

`GET /api/recent-simulations` → up to 6 `SimulationSummary` (the shape above plus
`project_name`, `space_name`, `geometry_name`, `duration_s`), newest run first.

Create / edit / delete:

| Method | Route | Body | Returns |
|---|---|---|---|
| POST | `/api/projects` | `{"name", "description"}` | Project |
| PATCH | `/api/projects/{pid}` | `{"name"?, "description"?}` | Project |
| DELETE | `/api/projects/{pid}` | none | `{"ok": true}` (cascades) |
| POST | `/api/projects/{pid}/spaces` | `{"name", "description"}` | Space |
| PATCH / DELETE | `/api/spaces/{sid}` | as above | Space / ok |
| PATCH / DELETE | `/api/geometries/{gid}` | `{"name"?}` | Geometry / ok |
| GET | `/api/geometries/{gid}/source` | none | the imported file, unchanged (no placed objects), as an attachment named `source_filename` (`.obj`, `.Par`, `.dxf`, `.3dm`, `.ifc` or `.skp`, non-ASCII names in `filename*`). 404 with a message when the geometry has no stored file. |

### Import a geometry

`POST /api/spaces/{sid}/geometries`, multipart form:
- `file`, one of:
  - `.obj`: layers come from `usemtl`. Faces before any `usemtl` go to layer `"(default)"`.
  - **Odeon `.Par`** room model: layers are the Odeon surface names. Models built with
    Odeon's parametric commands (For loops, Cylinder, RevSurf...) are refused with a
    clear message. Metres, Z up. `import_notes` reports what was read and any free-text
    line that was ignored.
  - **`.dxf`**: 3DFACE, MESH, polyface and polygon meshes, also inside INSERTed blocks.
    Layers are the DXF layers. Other entities are counted in `import_notes`, with the
    unit the file declares.
  - Rhino **`.3dm`**: meshes, and the render meshes saved with Breps and extrusions.
    Layers are the Rhino layers.
  - **`.ifc`**: see the IFC fields below.
  - **SketchUp `.skp`** (2013-2020 files): converted by `BRAS/skp_to_obj.py` in a separate
    Python environment with the SketchUp reader (see `webapp/server/skpfile.py`, or set
    `AURALIS_SKP_PYTHON`). Metres, Z up, one layer per SketchUp material, groups and
    components flattened. `unit` and `up_axis` are ignored. Without that environment, the
    import is refused (400) with the install command.

  The source and receiver markers drawn in a model come in as their own small layers.
- `name`: string.
- `up_axis`: `"z"` (default) or `"y"`. `"y"` rotates Y-up files to Z-up.
- `unit`: `"m"` (default), `"cm"` or `"mm"`. Ignored for IFC (metres, Z up).
- IFC only:
  - `ifc_exclude`: comma-separated classes. Default: IfcSpace, IfcOpeningElement,
    IfcAnnotation, IfcGrid, IfcVirtualElement, IfcSite, IfcBuildingElementProxy.
  - `ifc_space`: an IfcSpace GlobalId. The closed shell of that space becomes the room,
    and each face takes the layer of the nearest included element. The model is closed
    by construction.

  Layers are named `"<IfcClass> / <material>"`.

`POST /api/import/ifc-inventory` (multipart `file`) → `{"classes": {"IfcWall": 12, ...},
"spaces": [{"id": "<GlobalId>", "name": "Room 1.01"}], "schema": "IFC4",
"default_exclude": [...]}`. Use it to choose the classes or a space before importing.

Returns `Geometry`:

```json
{"id": "g1", "space_id": "s1", "name": "Geometry 1", "created": "...",
 "source_filename": "CR2.obj", "unit_in": "m", "up_axis_in": "z",
 "n_triangles": 508,
 "bbox": {"min": [x, y, z], "max": [x, y, z]},
 "volume_m3": 145.2, "surface_m2": 172.4,
 "layers": [{"name": "mat_CR2_floor", "n_triangles": 40, "area_m2": 50.1}],
 "check": {
   "watertight": true,
   "counts": {"open_edges": 0, "nonmanifold_edges": 0, "short_edges": 3, "degenerate_triangles": 0,
              "overlapping_faces": 1, "small_gaps": 2},
   "excluded": {"open_edge": 0, "nonmanifold_edge": 0, "short_edge": 0, "degenerate_triangle": 0,
                "overlapping_faces": 1, "small_gap": 0},
   "findings": [
     {"id": "se-3fa2c1d4e5b6a7c8",
      "kind": "open_edge|nonmanifold_edge|short_edge|degenerate_triangle|overlapping_faces|small_gap",
      "severity": "error|warning|info",
      "message": "Open edge: the volume is not closed here",
      "segment": [[x, y, z], [x, y, z]],
      "layers": ["mat_CR2_floor"],
      "excluded": false, "excluded_by": "finding|layer"}
   ],
   "solvers": {"geometrical": true, "wave": true},
   "detail": {"rule": "lambda/4|fixed", "threshold_m": 0.0857, "transition_hz": 1000.0,
              "wave_dx_m": 0.0254, "source": "automatic transition of 'Simulation 1' (same rule as its run)"},
   "exclusions": {"findings": ["sg-..."], "layers": ["Chairs"], "unmatched": []},
   "blocking": [],
   "notes": []
 }}
```

Finding kinds. Each finding carries `layers`, the layers of the faces involved.
- `open_edge`, `nonmanifold_edge`, `degenerate_triangle`: basic mesh defects. Edge
  findings add `length_m`.
- `short_edge` (`info`): a FEATURE edge shorter than `detail.threshold_m`. A feature edge
  is open, non-manifold, between two layers, or a fold of at least 1°. Diagonals inside
  a flat single-layer surface are not counted.
- `overlapping_faces`: coplanar faces (planes within 1 mm and 1°) that share more than
  0.1 % of the smaller triangle. One finding per connected patch of the smaller side (a
  seat on the floor, a duplicated wall).
  - `overlap`: `"duplicate"` (exactly the same triangles) or `"partial"`.
  - `contact`: `true` for faces back to back (a solid resting on a surface), `info`.
    `false` when both faces are seen from the same air, `warning` (the material there is
    arbitrary).
  - `area_m2` (overlap), `n_triangles`, `centroid`, and `triangles` (up to 24 of the
    smaller side, `[[x,y,z] x 3]`, to highlight them). `segment` spans their bounding box.
- `small_gap`: two surfaces closer than `detail.threshold_m` and farther than 1 mm (closer
  than that, they touch).
  - `gap_type: "slit"`: an open edge facing another surface (`length_m`). `leak: true`
    when room air is on one side and outside air on the other, `null` when the voxel
    closure cannot tell.
  - `gap_type: "layer"`: a thin air layer between faces parallel within 10° (`area_m2`).
  - `gap_m` / `gap_max_m`: narrowest / widest gap. `segment` goes across it.
  - `info`, except a leaking slit at least 1.5 wave cells wide in a model closed only at
    the voxel scale (`warning`): the wave grid may leave it open.

`detail` is the threshold for short edges and gaps: λ/4 at the wave band's transition
(c = 343 m/s), with the wave grid's cell `wave_dx_m` = 343 / (13.5 fc). With no known
transition, it is a fixed 2 cm (`rule: "fixed"`, and `source` says why). An import is checked with the fixed threshold. `POST
/api/geometries/{gid}/check` (below) applies a transition.

`notes` lists the checks that could not run (e.g. "thin air layers NOT checked: ...
above the budget" for a model too dense for that threshold). `blocking` lists what the
model itself blocks: a model that is not closed turns the wave solver off, unless the
simulation declares an open scene. Each entry gives the number of excluded findings,
which do not lift the block.

**Re-check** `POST /api/geometries/{gid}/check` with `{"transition_hz": 480}` or
`{"simulation_id": "m1"}` or `{}` → Geometry with the new `check`, saved. `watertight`,
the volume and the solver mesh do not change.
- `transition_hz` (20–20 000 Hz) wins.
- Else the simulation's transition (the simulation must belong to this geometry): its
  fixed Advanced `transition_hz`, or the automatic value its run would use (same rule as
  `suggested_transition_hz`). A simulation without wave band (Survey mode, geometrical
  only, model neither closed nor an open scene) or with an unassigned layer gives the
  fixed threshold, and `detail.source` says why.
- `{}`: fixed threshold.

Placing objects (PUT `.../objects`) re-checks with the last transition and the
exclusions.

**Exclusions** `PUT /api/geometries/{gid}/exclusions` with `{"findings": ["<id>", ...],
"layers": ["<layer>", ...]}` → Geometry.
- Replaces the whole set (to undo, PUT again without the id). Saved in the geometry
  (`check_exclusions`, copied by a duplicate), then re-checked at the same transition.
- A finding is excluded by its id, or because one of its `layers` is excluded. It stays
  listed with `excluded: true` and `excluded_by`. `excluded` counts them per kind, and
  `counts` stay the totals.
- Ids are stable (derived from the place of the finding), so an exclusion survives
  re-checks and changes of transition. An id that the current check does not find is
  kept in `exclusions.unmatched`, and applies again if the finding comes back.
- Exclusions NEVER change `watertight`, `solvers`, the validation or what a run accepts
  (see `blocking`).
- 400 on a malformed id, more than 5000 ids, or an unknown layer.

`check.closure` tells how closure was established:
- `{"method": "topology"}`: edges match vertex to vertex.
- `{"method": "voxels", "dx_m": 0.17, "closed": true}`: the outside air, flooded on a
  voxel grid, cannot reach the inside (gaps below `dx_m` are sealed). The volume comes
  from the voxel interior.

A model counts as a room only if the enclosed volume fills at least 25 % of its bounding
box.

`volume_m3` is `null` when the mesh is not watertight. Once objects are placed, `check`,
`volume_m3`, `surface_m2`, `n_triangles` and `bbox` describe what the solvers see: the
room plus the object proxies.
- `watertight` refers to the OUTER SHELL. Inner open surfaces such as table tops are
  allowed and only add `info` open edges.
- The air volume excludes closed proxies.
- `bbox` equals the room's as long as objects stay inside it.

Geometries also carry `objects` (see Objects) and `layers[].kind` (`"room"|"object"`).
`findings` lists at most 200 active entries per kind (errors first), then every finding
excluded by id and up to 50 excluded by layer. `counts` and `excluded` are always exact.
A geometry checked by an older version has no ids, no `detail` and no `overlapping_faces`
or `small_gap` findings until it is re-checked. `GET /api/geometries/{gid}` returns the
same object.

`GET /api/geometries/{gid}/mesh` → the triangles for the viewer:

```json
{"positions": [x0, y0, z0, x1, y1, z1, ...],
 "layers": [{"name": "mat_CR2_floor", "indices": [0, 1, 2, 0, 2, 3, ...]}]}
```

`indices` are vertex-index triplets into `positions`, one triangle each.

`GET /api/geometries/{gid}/subsurfaces` → the sub-surfaces under each layer of the
material list. Display only: materials stay assigned PER LAYER, and nothing here enters
a simulation.

```json
{"method": "smooth_patches+obj_groups", "angle_deg": 20, "groups_from_source": true,
 "note": "Surfaces = triangles joined across edges flatter than 20 deg ...",
 "layers": [{"name": "mat_CR2_concrete", "by": "patches",
             "subsurfaces": [{"label": "Surface 1", "group": "Mesh3 Model",
                              "orientation": "vertical", "area_m2": 16.32, "n_triangles": 6,
                              "triangles": [12, 13, 40, 41, 42, 43]}]},
            {"name": "obj:Ottoman_01:body", "by": "instances",
             "subsurfaces": [{"label": "Cabinet left", "object_id": "O1", "group": null,
                              "orientation": null, "area_m2": 2.98, "n_triangles": 12,
                              "triangles": [0, 1, ...]}]}]}
```

The server derives them (the model files have none):
- **Room layers** (`by: "patches"`): a surface is a set of triangles that share edges,
  with planes within `angle_deg` (20°).
  - For an `.obj` source, surfaces are also split where its `g` / `o` names change, and
    `group` carries that name (`"o / g"` when both are set). Otherwise `group` is `null`
    and `note` says why.
  - Surfaces are sorted by area, largest first (`label` = rank). The collinear
    (zero-area) triangles of a layer that touch no surface come last, as one entry with
    `"degenerate": true`.
- **Object layers** (`by: "instances"`): one entry per placed object, `label` = its name.
- `triangles` index the layer's triangles in the order of `GET .../mesh`
  (`indices[3t..3t+2]`). The entries of a layer cover each of its triangles once.
  `orientation` is `horizontal`, `vertical`, `sloped`, `curved` or `null`. Read again
  after `PUT .../objects`.

### Visual model (.glb overlay)

Display only: the solvers never see it.
- `POST /api/geometries/{gid}/visual` (multipart `file`, a binary glTF `.glb`, ≤ 200 MB)
  → Geometry with `"visual": {"filename", "bytes", "uploaded", "transform": {"position":
  [0,0,0], "rotation_deg": [0,0,0], "scale": 1, "up_axis": "y"}, "lighting": null}`. A new
  file resets the alignment and keeps the lighting.
- `GET /api/geometries/{gid}/visual.glb` → the file.
- `PUT /api/geometries/{gid}/visual` with any of `position`, `rotation_deg`, `scale`,
  `up_axis`, `lighting` → Geometry (manual alignment, saved).
- `DELETE /api/geometries/{gid}/visual` → Geometry.
- These writes hold the store lock, so a concurrent `PUT .../objects` cannot lose one of
  them.
- `lighting` (display only): `null` = the default look (the UI then uses the look saved
  in this browser, if any). Otherwise `{"preset": "default|daylight|evening|
  overcast|studio|custom", "env_intensity": 0-3, "sun_intensity": 0-8, "sun_azimuth_deg":
  0-360 (counter-clockwise from +X), "sun_elevation_deg": -10-90, "sun_color": "#rrggbb",
  "view_lights": bool, "exposure_ev": -3-3}`. Every number is required. Out of range =
  400. A geometry stored by an older version reads `"lighting": null`.
- Apply in this order: up-axis conversion (glTF is Y-up: Y→Z), scale, rotation X then Y
  then Z, translation.

## Materials

`GET /api/materials` →

```json
{"bands_hz": [125, 250, 500, 1000, 2000, 4000, 8000],
 "libraries": [
   {"id": "auralis", "name": "Auralis presets", "redistributable": true, "license": "MIT"},
   {"id": "pyroomacoustics", "name": "pyroomacoustics (Vorländer tables)", "redistributable": true, "license": "MIT"},
   {"id": "odeon_local", "name": "Odeon library (local, research licence)", "redistributable": false, "license": "Odeon A/S — research use only, do not redistribute"},
   {"id": "custom", "name": "My materials", "redistributable": true, "license": "user"}
 ],
 "materials": [Material]}
```

`Material`:

```json
{"id": "auralis:concrete", "library": "auralis", "name": "Concrete",
 "category": "Hard surfaces", "description": "",
 "absorption": [0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04],
 "absorption_63": null, "scattering": 0.1,
 "source": "Auralis engine preset", "license": "MIT",
 "clipped": false, "physical": null}
```

- `clipped: true`: at least one value above 0.95 was clipped to 0.95. Show it.
- Wave band: an alpha-only material gets the physical model that reproduces its alpha
  (porous layer, resistive screen or panel), or keeps the alpha-only fit when none does
  (hard surfaces). `porous` and `perforated` materials keep their model's alphas and
  take its phase (`anchor_alpha`).
  `GET /api/materials/{id}/details` returns `wave_model` for alpha-only materials (the
  inferred model, or why none was kept), `null` otherwise.
- `physical` is `null` or a description of the physical model used by the wave band,
  e.g. `{"type": "porous", "model": "miki", "sigma": 15000, "thickness_mm": 50, "air_gap_mm": 100}`.
- The `auralis` library has the seven Nord2000 ground classes, category `Outdoor ground`
  (`auralis:ground_A` to `auralis:ground_G`, snow to dense asphalt), with
  `physical = {"type": "ground", "class", "sigma" (Pa.s/m2), "model": "delany_bazley"}`:
  Delany-Bazley on a semi-infinite ground, alphas from the model, scattering 0.1.
- A library may be absent from the list when its data is not on disk. The examples add
  the local libraries `odeon_rooms` (the Odeon halls' own materials) and `bras` (the
  BRAS rooms' materials), both `"redistributable": false`.
- `absorption_63` is `null` or the α of the 63 Hz octave (Odeon, room imports, custom
  materials). The rays never use it (they start at 125 Hz). The wave band uses it below
  125 Hz (engine manifest field `absorption_full`). Physical materials (porous, table)
  have their own curve and no separate 63 Hz value.

Custom materials:

| Method | Route | Body | Returns |
|---|---|---|---|
| POST | `/api/materials` | `{"name", "category", "description", "absorption": [7], "scattering", "absorption_63"?}` (`absorption_63` optional, null by default) | Material (`custom:…`) |
| POST | `/api/materials/wool-sigma` | `{"wool": "glass" \| "rock", "density_kg_m3"}` | `{"sigma", "sigma_low", "sigma_high" (1-sigma band of the law against its data), "uncertainty_pct", "extrapolated" (density outside the data the law was checked on: say so), "range_kg_m3", "law", "source", "wool", "density_kg_m3"}`. 400 outside 5-250 kg/m3. |
| POST | `/api/materials/porous-preview` (raw model output: may exceed 0.95) | `{"model": "miki", "sigma", "thickness_mm", "air_gap_mm"}`, or `"wool"` + `"density_kg_m3"` instead of `"sigma"` | `{"absorption": [7], "sigma"}`. From a wool, also `"wool"` (the wool-sigma answer) and `"absorption_band": [[7], [7]]` (alphas at `sigma_low` and `sigma_high`: show the spread). |
| POST | `/api/materials/porous` | preview body + `"name", "category", "description", "scattering"` | Material. From a wool, `physical.sigma_from` keeps the wool, density, band and law (shown in the details' `phase_source`). |
| POST | `/api/materials/perforated-preview` | `{"kind": "perforated" \| "slotted" \| "mpp", "size_mm" (hole diameter or slot width), "t_mm", "open_pct", "plenum_mm", "wool_mm"?, "wool_sigma"?, "fleece_rayl"?}` | `{"absorption": [7]}` |
| POST | `/api/materials/perforated` | preview body + `"name", "category", "description", "scattering"` | Material, `physical.type = "perforated"` |
| DELETE | `/api/materials/{id}` | none | ok (custom only, else 400) |
| PATCH | `/api/materials/{id}` | `{"name"}` | Material: renamed, **same id** (custom only, else 400, and 404 if unknown) |
| POST | `/api/materials/{id}/duplicate` | `{"name"?}` (body optional) | Material: a new `custom:…` copy of ANY material |

- **Ids are stable.** Simulations store `material_id`, never the name, so a rename
  changes nothing in them (no stale results). A custom id is never reused, even after a
  delete: a layer that names a deleted material stays "missing".
- Names of your materials: whitespace collapsed, 1-200 characters, unique among YOUR
  materials ignoring case (400 otherwise). Library names may repeat. Rename and
  duplicate enforce this. `POST /api/materials` does not.
- A duplicate copies absorption, `absorption_63`, scattering, `physical`, `clipped`,
  category, description and `source`, and adds `derived_from` (the original id). When the
  original is not a custom material, it also copies `license` and `redistributable` of
  its library: a copy of an Odeon or BRAS material stays `"redistributable": false`
  (show it as local). With no `name` (or a blank one), the copy is named
  `"<name> (copy)"`, then `(copy 2)`, ...

`GET /api/directivities` → `[{"id": "omni", "name": "Omnidirectional"}, ...]`. Full shape
in [Source directivities](#source-directivities).

### Complex materials, details, scattering per octave

- `POST /api/materials/table` with `{"name", "kind": "z" | "r", "freqs": [Hz], "re": [...],
  "im": [...], "scattering": 0.15}`: a material from a measured complex quantity at
  normal incidence (e.g. 1/3 octaves 50 Hz–10 kHz). `z` = surface impedance / (ρ0 c0),
  `r` = reflection factor. 400 if not passive (Re z ≤ 0, |R| ≥ 1). The rays use its
  random-incidence α per octave. The wave band and the complex reflections use the table
  itself (`physical.type = "table"`).
- `POST /api/materials/table-preview` (same body) → `{"absorption": [7]}`: the
  random-incidence octave α the table would give. Nothing is created.
- `GET /api/materials/{id}/details` → `{"bands_hz", "alpha_table", "alpha_table_63", "alpha_fit_random", "alpha_fit_63",
  "freqs_hz" (64, 50 Hz–10 kHz), "R_normal": {"re", "im", "mag", "phase_deg"},
  "alpha_normal", "phase_source"}`: what the engine uses.
  - `alpha_fit_random` is the admittance fit BEFORE the per-band anchoring to
    `alpha_table`.
  - `phase_source` says whether the phase is known (porous model, measured table) or
    assumed (alpha-only fit). Show it.
  - `alpha_fit_63` is the fit at 63 Hz, for every material: what the wave band sees
    there.
  - `alpha_table_63` is `null` when the material has no 63 Hz value. The wall then keeps
    its 125 Hz behaviour.
- A layer's `scattering` may be one value or **7 values** (one per octave, 125 Hz–8 kHz).
  A single value is used flat across the bands (not extrapolated into a curve).

## Objects (furniture library)

Each object has a realistic **visual model** (CC0, Poly Haven, display only) and a
simplified **acoustic proxy** that the solvers use. The proxy is what is simulated, so
the UI must be able to show it (view modes "Visual", "Acoustic", "Both").

`GET /api/objects/catalog` →

```json
{"license_note": "...",
 "objects": [
   {"id": "Sofa_01", "name": "Sofa", "category": "Armchairs & sofas",
    "source": "polyhaven|procedural", "license": "CC0 (Poly Haven)|Auralis",
    "proxy": "upholstered|chair|plane_top|box|bed|vertical_plane|none",
    "acoustic": true,
    "parts": {"upholstery": {"material": "pyroomacoustics:chairs_upholstered_cloth", "scattering": 0.6}},
    "dims": [1.57, 0.66, 0.80],
    "visual": true, "visual_url": "/api/objects/visual/Sofa_01/Sofa_01.gltf",
    "thumbnail_url": "/api/objects/visual/Sofa_01/thumbnail.png",
    "visual_offset": [0.0, 0.01, 0.0], "visual_scale": 1.0,
    "note": "(optional, absent when empty)"}
 ]}
```

- Object frame: x = width, y = depth (front towards −y), z = height. Origin on the floor
  at the centre of the footprint. `dims` = [width, depth, height] in metres.
- `proxy: "none"` (plants, lamps): visual only, no acoustic effect, `acoustic: false`.
- `visual: false` (procedural objects, or model not downloaded): no glTF. Draw the proxy
  itself, coloured by its material.
- **Placing the glTF** (glTF is Y-up). Build this three.js chain, outermost first:
  `objectGroup` (object `position`, `rotation.z = rotation_deg` in radians, `scale`)
  → `offsetGroup` (`position = visual_offset`)
  → `scaleGroup` (`scale = visual_scale`, uniform)
  → `rotGroup` (`rotation.x = +π/2`, i.e. glTF (x, y, z) → (x, −z, y))
  → `gltf.scene`.
  `visual_offset` is already multiplied by `visual_scale`. With this chain, the model sits
  on z = 0, centred on x = y = 0 of the object frame, exactly over its proxy.
- `GET /api/objects/visual/{id}/{file}` serves the glTF, its `.bin` and textures
  (relative URIs inside the glTF resolve against this route) and `thumbnail.png`.

Placing objects in a geometry: `PUT /api/geometries/{gid}/objects` with

```json
{"objects": [{"id": "O1", "catalog_id": "Sofa_01", "name": "Sofa",
              "position": [x, y, z], "rotation_deg": 180, "scale": [1, 1, 1]}]}
```

- `position` = the object origin (floor, footprint centre) in room coordinates.
  `rotation_deg` about +Z, counter-clockwise. `scale` per axis of the object frame
  (applied to visual and proxy alike). Ids are assigned by the client.
- Returns the updated Geometry (now with `objects`, and `layers[].kind` =
  `"room"|"object"`). The solver mesh, the geometry check and the volume are recomputed.
  Every simulation of the geometry gets the new object layers (catalog default material
  and scattering), loses the removed ones, and is marked `results_stale` if it had
  results. 409 while one of its simulations runs.
- Object layers are named `obj:<catalog_id>:<part>` and are **shared by all instances of
  the same catalog object**. Assigning a material to `obj:WoodenChair_01:seat` changes
  every such chair.

`GET /api/geometries/{gid}/mesh` returns room AND proxy triangles. Each layer has `kind`.
Object layers also carry `owners`: the object id of each of their triangles (same order
as `indices`), so a click on a proxy can select the object.

`POST /api/geometries/{gid}/duplicate` with `{"name"}` → a new Geometry in the same space
(same room, same objects), to compare "empty" and "furnished" rooms. No simulation is
copied: create one with `copy_from`.

`POST /api/objects/proxy` with `{"objects": [{"id", "catalog_id", "position",
"rotation_deg", "scale"}]}` (same item rules as the PUT) → `{"objects": [{"id",
"layers": {"obj:<catalog_id>:<part>": [[[x, y, z] × 3] per triangle]}}]}`: the acoustic
proxies at that placement. NOTHING is saved. A gizmo drag shows this (throttle to ~10
requests/s, several objects for a multi-selection). The saved mesh still comes from the
PUT on release.

## Surface textures (realistic view)

Display only: the solvers never see them. CC0 texture sets (Poly Haven, 1k JPG:
diffuse, OpenGL normal, roughness) and two 1k HDR environment maps, downloaded once by `python -m webapp.server.fetch_textures` (run by `start.sh`) into
`webapp/assets/textures/` (~41 MB, not in the repository).

- `GET /api/textures` → `{"available": bool, "textures": {"<key>": {"id", "name",
  "credit", "size_m": [u, v], "maps": {"diff": "/api/textures/<key>/diff.jpg", "nor"?,
  "rough"?}, "license", "url", "bytes"}}, "hdri": {"interior"|"exterior": {"id", "name",
  "credit", "url_file": "/api/textures/hdri/<key>.hdr", "license", "url", "bytes"}}}`.
  Keys: parquet, wood_floor, wood_panel, carpet, plaster, concrete, brick, stone, marble,
  tiles, metal_sheet, velvet, fabric, ceiling_tiles, linoleum, asphalt, paving, grass,
  gravel, render. Entries whose files are missing are left out. Nothing downloaded →
  `available: false` (the UI then shows plain colours and says so once).
- `size_m` is the scanned area of one texture repeat. The UI projects the texture in
  world space (triplanar) at that physical scale.
- `GET /api/textures/{key}/{file}` and `GET /api/textures/hdri/{key}.hdr` → the files
  (cached one day by the browser).

## Source directivities

`GET /api/directivities` → `[{"id": "omni" | "speaker_2way" | ... | "custom-1a2b3c4d" |
"odeon-violin", "name", "kind": "preset|imported|odeon_local", "di_db": [7], "polar":
{"theta_deg": [0, 5, ... 180], "db": [7][37]}, "on_axis_spl_1m_db": [7] | null, (imported,
odeon_local:) "source", "method", "notes", "asymmetry_db", (odeon_local:) "license"}]`.
- `odeon_local`: the 18 `.SO8` patterns of the local Odeon archives (instruments, singer,
  talkers, omni), read in place and never copied (research licence). They cannot be
  deleted. The talker files carry absolute levels, hence `on_axis_spl_1m_db`.
- The engine's model is AXISYMMETRIC about the source's front axis (energy per band
  against the angle θ from it).

`POST /api/directivities` (multipart `file`, `name`) imports:
- `.json`: the engine table `{"name", "table": [7][S] energy vs θ 0–180° (or [S]),
  "on_axis_spl_1m_db": [7] (optional)}`.
- `.csv`: header `theta_deg,125,250,500,1000,2000,4000,8000`, then one row per angle with
  the level in dB relative to the axis per octave. Several rows at one angle are
  averaged.
- `.sofa`: AES69 FreeFieldDirectivityTF, averaged per octave and in rings about
  `SourceView`. `asymmetry_db` = the largest spread between directions at the same angle
  (a note is added above 1 dB).
- `.so8`: an Odeon pattern, averaged in rings like the others, 63 Hz dropped. Binary
  layout: see `read_so8` in `webapp/server/directivities.py`.

Ensembles (`POST .../ensemble`, and the examples) give each player the Odeon pattern of
its instrument, facing the audience. An instrument without its own pattern (double bass,
tuba, harp, percussion) stays omni.

The CLF `.CF2` format is not read (no public specification).
`DELETE /api/directivities/{id}` (409 while a source uses it). A source's `directivity`
may be any listed id. An imported one with `on_axis_spl_1m_db` sets the source level used
for SPL.

Where the directivity applies:
- The geometrical solver uses the full pattern per octave.
- The wave solver (below the transition) uses a first-order pattern with the same power
  per octave. A source that is directional there costs a second wave run. A source that
  is nearly omni below ~2× the transition keeps one run.
- Moving sources in animated renders get the same pattern along the forward axis of their
  path (no extra run). A key where the source hugs a wall stays omni, with a warning in
  the log.

## Simulations

`POST /api/geometries/{gid}/simulations` with `{"name", "copy_from": "<simulation id>|null"}` → Simulation.
`copy_from` may be a simulation of ANOTHER geometry (e.g. after a duplicate). Sources,
receivers and settings are copied, layers are matched by name, and object layers that
did not exist get the catalog defaults.

`GET /api/simulations/{id}` → Simulation. `PUT /api/simulations/{id}` saves the editable
fields (`name`, `description`, `layers`, `sources`, `receivers`, `settings`) and returns
the stored Simulation. `DELETE` removes it.

```json
{"id": "m1", "geometry_id": "g1", "name": "Simulation 1", "description": "",
 "created": "...", "status": "draft", "last_run": null, "results_stale": false,
 "layers": {"mat_CR2_floor": {"material_id": "auralis:parquet", "scattering": null}},
 "sources": [{"id": "S1", "name": "Source 1", "position": [2.0, 3.0, 1.5],
              "directivity": "omni", "azimuth_deg": 0, "elevation_deg": 0, "gain_db": 0,
              "eq_db": [0, 0, 0, 0, 0, 0, 0]}],
 "receivers": [{"id": "R1", "name": "Receiver 1", "position": [5.0, 3.0, 1.2],
                "azimuth_deg": 0, "elevation_deg": 0}],
 "settings": {"mode": "survey|default|advanced",
              "transition_hz": null, "ir_length_s": null,
              "ism_order": 2, "n_rays": 200000,
              "solvers": "both|wave|geometrical",
              "temperature_c": 20.0, "humidity_percent": 50.0,
              "diffraction": true, "conformal": true, "open_scene": false,
              "speech_effort": "normal|raised", "background_noise": "none|NC-35|[7 dB values]",
              "background_recording": null,
              "head": "model|fabian"}}
```

- Source `eq_db` (optional, absent or all zeros = flat): 7 octave gains in dB, 125 Hz-8
  kHz, within ±24 dB (400 otherwise). A minimum-phase filter applies them to the IRs of
  that source (binaural, ambisonic, per solver). SPL follows the band gains. G
  (normalised by the source's own power) does not. T30, C50… are computed on the
  filtered IR. Animated renders filter the source's dry signal instead. Editing it makes
  the results stale. Reordering sources or receivers alone does not.
- `background_recording` (default `null`): `{"sound_id", "dba"}` makes a recorded
  ambience of the sound library (e.g. a city's traffic) the diffuse background of
  animated renders, in place of the octave levels of `background_noise`. It is played
  diffuse, looped over the render, at `dba` dB(A) at each ear (20-100). The server stores
  the file path, name and credit with it. Tracks are scaled as speech at the vocal
  effort, as with `background_noise`, before their gains.
- `stop_decay_db` (Advanced, 20–90, `null` = automatic): an IR length such that every
  band decays by that many dB + 10 dB (Eyring per band). `ir_length_s` wins when both are
  set.
- `grid_rays` (1000–2 000 000, default 50 000): rays per receiver-grid point.
- `spherical_reflection` (optional, default: the value of `open_scene`): spherical-wave
  reflection coefficients on the image sources, with a longer filter for them and for the
  direct sound. Over a ground near grazing incidence, plane-wave reflections miss the
  ground effect by up to 10 dB. Diffracting edges also get the paths that reflect on a
  plane before and/or after the edge (a screen on a ground: the four-path model), with the
  complex UTD coefficient. Materials with a physical model (`ground`, `porous`,
  `table`…) use the model itself; alpha-only ones their fitted admittance.
- `rbs_scattering` (default `true`, can be turned off in Advanced mode): adds scattering
  from the size of each surface hit, so a material's scattering should be its roughness
  only (e.g. 0.05 for a smooth wall).
- `head` (all modes, default `"model"`): the listener head of the binaural IRs and of the
  head-tracked decoder. `"model"` is the parametric sphere (no pinnae). `"fabian"` is the
  measured FABIAN head and torso (TU Berlin, CC BY 4.0, credit mandatory).
  `GET /api/heads` → `[{"id", "name", "available", "credit"}]`. A head whose data is not installed locally
  (`data_hrtf/`, see its README) is listed with `available: false` and refused (400).
  Changing it makes the binaural results stale.
- `speech_effort` and `background_noise` only feed the STI of the results (animated
  renders can also use them, see below): the speech level at 1 m (IEC 60268-16 male
  spectrum, 60 dB(A) normal / 66 dB(A) raised) and the background noise at the listeners
  (an NC curve, NC-15 … NC-70 in steps of 5, or 7 octave levels 125 Hz–8 kHz in dB).
  Changing them makes the results stale.
- Source and receiver `id`s are assigned by the client (e.g. `S1`, `R1`) and must be
  unique within the simulation. The server fills a missing one.
- Every layer of the geometry is present in `layers`. `material_id: null` = unassigned.
  `scattering: null` = use the material's default.
- Orientation: azimuth counter-clockwise from +X in the XY plane, elevation from −90
  (−Z) to +90 (+Z). Source front axis / receiver look direction.
- `transition_hz` / `ir_length_s`: `null` = automatic. `solvers` only matters in
  `advanced`.
- `solvers: "wave"` (wave solver only): the IR is the wave band alone, through the
  crossover's low-pass. Its pairs carry `"wave_only": true`. Their parameters exist in
  the octaves whose upper edge is at or below the transition (`null` above). STI, MTI,
  J_LF, L_J and the overall `SPL_dbz/dba/dbc` are `null`, with no `by_solver`. Receiver
  grids are refused (blocking) in this mode, and `validate` warns that the IR stops at
  the transition.
- A source may carry `"surface_layer": "<layer>"`: that layer vibrates as a whole (a
  panel, a flush-mounted loudspeaker) with the volume velocity of the unit point source,
  in place of the point. `position` then only marks the room's air. Needs
  `solvers: "wave"` in `advanced` and a closed model (blocking otherwise). Refused in
  animated renders. An unknown layer is a 400.
- `open_scene` matters when the model is NOT closed, or closed only at the voxel scale
  (an open-air theatre whose ground is a closed solid looks closed).
  - `true` declares an object in free field (panel, wedge, seats). It unlocks the wave
    solver (absorbing boundaries around the model), with the anisotropic conformal
    correction. Its image sources use spherical-wave reflections: the ground effect of a
    point source over a ground (see `spherical_reflection`).
  - `false` (default) means "a room with holes". The wave solver stays blocked, since
    energy would leak through them.
  - For a model that is not closed, any position at a valid distance from the surfaces
    is accepted (free field).
- Saving a simulation whose status is `done` keeps the old results but sets
  `results_stale: true` and the status back to `draft`. Editing only a `path` (below)
  does NOT make the results stale, unless its first waypoint moves.
- Any source or receiver may carry a **trajectory** in `path` (absent or `null` = static):

  ```json
  "path": {"waypoints": [{"position": [2.0, 3.0, 1.5], "t": 0.0},
                         {"position": [8.0, 3.0, 1.5], "t": 4.0}],
           "orientation": "fixed|follow|target", "target_id": "R1|null",
           "interpolation": "linear|smooth"}
  ```

  - 2 to 200 waypoints. Times in seconds, ≥ 0 and strictly increasing, the last ≤ 600 s.
    The point moves in straight lines between waypoints, at the speed set by the times
    (the UI may edit a segment's speed and recompute the times). Before the first time,
    it waits at the first waypoint. After the last, it stays at the last.
  - The server sets `position` to the first waypoint: the static position of a moving
    point (used by normal runs) is its start.
  - `interpolation` (default `linear`): `smooth` = a C1 cubic through every waypoint AT
    its time. It stops at holds, rounds the corners and can leave the polyline near them.
    The checks sample the TRUE curve (≤ 0.10 m). The validation's `paths[]` then carries `"curve": [[x, y, z]...]` to draw it, with
    `length_m` and `max_speed_mps` measured along it.
  - `orientation`: `fixed` = the point's azimuth/elevation. `follow` = azimuth along the
    direction of travel (corners smoothed over ±0.25 s, elevation kept). `target` =
    always aims at the source/receiver `target_id`, which may move too.
  - 400 with a message if the path is malformed.

`GET /api/simulations/{id}/validation` →

```json
{"ok_to_run": false,
 "blocking": ["1 layer(s) without material"],
 "unassigned_layers": ["mat_CR2_windows"],
 "sources": [{"id": "S1", "valid": true, "reasons": []}],
 "receivers": [{"id": "R1", "valid": false, "reasons": ["0.06 m from a surface (min 0.10 m)"]}],
 "rt_estimate": {"sabine": [7] , "eyring": [7]},
 "volume_m3": 145.2, "surface_m2": 172.4,
 "suggested_transition_hz": 480, "wave_available": true,
 "watertight": true, "open_scene": false, "closure_method": "topology|voxels",
 "paths": [{"id": "S1", "valid": false, "length_m": 6.0, "start_s": 0.0, "end_s": 4.0,
            "max_speed_mps": 1.5,
            "reasons": ["leaves the air volume 4.2 m along the path, at (6.20, 3.00, 1.50)"]}]}
```

`paths` has one entry per point with a trajectory, checked every 0.1 m along it (same
rules as a static point). It never blocks a normal run. Draw an invalid path in red and
disable the animated render (below).

`blocking` lists every reason that keeps `ok_to_run` false (human-readable). `warnings`
are not blocking (e.g. near-rigid materials in wave mode). Distance rules:
- ≥ 0.10 m from surfaces.
- With the wave solver, ≥ one FDTD cell at the transition frequency (343 / (13.5 fc) m,
  e.g. 0.18 m at 143 Hz) when that is more.
- A receiver closer to a surface than the ray receiver sphere (0.3 m) gets a sphere
  shrunk to that distance (logged).

`rt_estimate` is `null` while a layer is unassigned (statistical, no air absorption).
Show invalid sources and receivers in red. The Run button is disabled unless `ok_to_run`.

### Run

`GET /api/simulations/{id}/estimate` → `{"pairs_s", "wave_s", "grids_s", "total_s",
"transition_hz", "ir_length_s", "gpu", "note"}`: a rough run time from unit costs
measured on this machine. `ir_length_s` is the IR length the wave solver will run (fixed,
from `stop_decay_db`, or automatic). 409 while a layer is unassigned. Show it before Run.

`POST /api/simulations/{id}/run` → `Job` (409 if not `ok_to_run`).

```json
{"id": "j1", "simulation_id": "m1",
 "status": "queued|running|done|failed|cancelled",
 "progress": 0.42, "stage": "wave band (FDTD) S1",
 "sources": [{"id": "S1", "name": "Source 1", "progress": 0.6}],
 "started": "...|null", "finished": "...|null", "eta_s": 75.0, "error": null,
 "log": ["[Auralis] ...", "..."]}
```

`GET /api/jobs/{jid}` → Job (poll every ~500 ms). `GET /api/jobs` → active and recent
jobs. `POST /api/jobs/{jid}/cancel` → Job. One GPU job runs at a time. The others wait as
`queued`. `log` carries the engine's console lines, warnings and fallbacks included.

**Server restart.** At start-up (`python -m webapp.server`), jobs that a stopped server
left `queued` or `running` are queued again. `--no-resume` marks them `failed` instead,
as does a plain import of the app. A resumed job has a new id, and its `log` starts with
"resumed after a server restart". It keeps the sources (and receiver-grid runs) already
finished (`results.tmp/checkpoint.json`) if the model, materials, settings, sources,
receivers and grids are unchanged. Else the run starts over. Animated renders, modal and
field tasks start over from the beginning. Find the new job with `GET /api/jobs`.

## Results

`GET /api/simulations/{id}/results` (404 before the first run) →

```json
{"bands_hz": [125, 250, 500, 1000, 2000, 4000, 8000], "fs": 48000,
 "transition_hz": 480.0, "engine_used": "cuda", "ir_scale": 0.73, "stale": false,
 "duration_s": 53.2,
 "settings": {...settings used for this run...},
 "pairs": [
   {"source_id": "S1", "receiver_id": "R1",
    "metrics": {"T20": [7], "T30": [7], "EDT": [7], "C50": [7], "C80": [7],
                "D50": [7], "Ts_ms": [7], "SPL_db": [7], "G_db": [7],
                "STI": 0.49, "MTI": [7], "STI_signal_db": [7],
                "JLF": [7], "LJ_db": [7],
                "SPL_dbz": 105.9, "SPL_dba": 103.7, "SPL_dbc": 105.7},
    "timings": {"total_s": 12.3}}
 ]}
```

Units: T20/T30/EDT in s, C50/C80 in dB, D50 as a 0-1 ratio, Ts in ms. SPL in dB re
20 µPa for the source's `spl_1m` (94 dB at 1 m, omni) plus its `gain_db`. G in dB (re
the same source at 10 m in free field). `transition_hz` is `null` without wave band.
With the wave band, G and SPL of the bands it reaches (centre ≤ 2 × transition) come from
the omni IR delivered. Above, they come from the geometrical solver's echogram.

- `metrics_ir` (meta): `"omni"`: the ISO 3382 parameters (T20 … Ts, STI) are computed on
  the omnidirectional IR (wave band included). `"binaural mean"` when the run has no Ambisonics response.
- **STI** (IEC 60268-16:2020, indirect method): MTF of the full-band IR (wave band
  included). Speech at the listener = `speech_effort` spectrum at 1 m + G − 20 dB.
  Background noise from the settings. `MTI` per band, `STI_signal_db` the speech level
  used per band. Rate it as in the standard: < 0.30 bad, 0.30–0.45 poor, 0.45–0.60 fair,
  0.60–0.75 good, > 0.75 excellent.
- **J_LF** (early lateral energy fraction) and **L_J** (late lateral sound level, dB)
  per band (ISO 3382-1), from the geometrical solver. ISO single numbers: mean of
  125 Hz–1 kHz.
- `SPL_dbz/dba/dbc`: overall levels (energy sum of the octave bands, Z / A / C
  weighting).
- `by_solver` (runs with the wave solver, v0.6+): `{"geo": {T20, T30, EDT, C50, C80, D50,
  Ts_ms, G_db, SPL_db}, "wave": {...}}`. The same parameters on the omni IR of the
  geometrical solver alone (full band) and of the wave band alone (bands whose upper edge
  is above the transition are `null`). This is the "result type" (hybrid / geometrical /
  wave). The curves take `part=geo|wave` for the same split (404 for older runs).
- `band63` (only when the wave solver ran with a transition ≥ 125 Hz): T20, T30, EDT,
  C50, C80, D50, Ts_ms, `G_db` and `SPL_db` of the 63 Hz octave (44–88 Hz), from the wave
  band alone (the source's level at 63 Hz is its 125 Hz one). Absent otherwise.
- `decay_range_db` (per band): level reached at the end of the echogram, dB re its early
  peak. Above −45 dB, the T30 of that band is unreliable (a warning is logged).
- Any value may be `null` (undefined: e.g. a decay too short for T30).
- meta `wave_failed`: `[{"source_id", "error"}]`, the sources whose wave band failed (e.g.
  the FDTD watchdog on near-rigid materials). Their pairs use the geometrical solver
  only.

`GET /api/simulations/{id}/results/{source_id}/{receiver_id}/curves?fr_smoothing=3[&part=geo|wave]` →

```json
{"fs": 48000,
 "ir": {"t": [...], "left": [...], "right": [...]},
 "edc": {"125": {"t": [...], "db": [...]}, "250": {}, "500": {}, "1k": {}, "2k": {}, "4k": {}, "8k": {}},
 "fr": {"f": [...], "db": [...]}}
```

`ir.omni` (and `"source": "omni"`) is the omnidirectional IR when the run produced it.
The EDC and the frequency response are then computed on it (else on the mean of the
ears). `GET .../results/{source_id}/{receiver_id}/ambi.wav` → the 9-channel ambisonic IR
(2nd order, ACN/SN3D, world axes x, y, z, z up, float32, same scale as `ir.wav`).

The IR is decimated to at most ~4000 points (min/max per bucket, so peaks survive). EDC
per band in dB. FR = magnitude (dB) on log-spaced frequencies 20 Hz–20 kHz, smoothed at
1/N octave (`fr_smoothing` ∈ {0, 24, 12, 6, 3, 1}, 0 = raw).

`GET /api/simulations/{id}/results/{source_id}/{receiver_id}/ir.wav` → binaural IR (2
channels, float32 WAV). All IRs of one simulation share the same `ir_scale`, so relative
levels between pairs are preserved.

### Open-plan offices (ISO 3382-3:2022)

`GET /api/simulations/{id}/iso3382-3?source_id=S1&receiver_ids=R1,R2,R3,R4` (the
measurement path, any order, ≥ 2 receivers, results needed) →
`{"positions": [{"receiver_id", "distance_m", "Lp_A_S_db", "STI", "Lp_S_db": [7]}],
"n_in_range", "D2S_db", "Lp_A_S_4m_db", "rC_m", "rC_note", "rD_m", "rD_note", "Lp_A_B_db",
"decay_fit": {"intercept_db", "slope_db_per_decade"} (Lp,A,S = intercept + slope·log10 r),
"sti_fit": {"intercept", "slope_per_m"} (STI = intercept + slope·r), "directivity_note"}`.

Speech = the standard's normal-effort spectrum at 1 m (Table 1) + the simulated G. STI
with the simulation's background noise. D2,S and Lp,A,S,4m by regression over the
positions 2–16 m from the source (`n_in_range`). rC where the regression reaches
45 dB(A). rD where a linear fit of STI against distance reaches 0.50. `*_note` flags
extrapolations. The source should be omnidirectional (`directivity_note` otherwise).

### Forced-response (modal) maps

`POST /api/simulations/{id}/modal` with `{"source_id": "S1", "grid_id": "G1", "f_min": 20,
"f_max": 200, "step_hz": 1, "label": ""}` (or `"freqs": [...]`, 1–400 frequencies,
≤ 1000 Hz) → `{"modal": Task, "job": Job}`. The job (`kind: "modal"`) runs the wave
solver once, with a probe on every valid point of the grid. 409 without a closed model or
a declared open scene, or with unassigned layers. Points closer than one wave-solver cell
to a surface are left out (logged).
- `GET .../modal` → Task[].
- `GET .../modal/{rid}` → Task, plus `result`: `freqs_hz` actually used (nearest FFT
  bins), `dx_m`, `duration_s`, `engine`, `conformal`, `render_s`, `note`.
- `GET .../modal/{rid}/map?freq=63` → `{"points", "valid", "area_m2", "freqs_hz": [f],
  "level_db": [[per point]]}` (all frequencies without `freq`). Level in dB re the same
  source in free field at 10 m (G convention).
- `DELETE` (409 while running).

Mode frequencies carry the voxelisation bias of the wave solver: show `note`.

### Wave-field animation

`POST /api/simulations/{id}/field` with `{"source_id", "grid_id", "f_max_hz": 300,
"duration_s": 0.2, "fps": 1200, "label"}` → `{"field": Task, "job": Job}` (`kind:
"field"`, same 409 rules as the modal maps).
- `grid_id`: a plane grid is the usual choice.
- `f_max_hz` 50–1000, `duration_s` 0.02–1, `fps` ≥ 2.5 × f_max, and duration × fps
  ≤ 2000.

The wave solver records its impulse response (pressure, re 1 m, band-limited to ~f_max)
on every valid point.
- `GET .../field` → Task[].
- `GET .../field/{rid}` → Task, plus `result`: `fps`, `n_frames`, `duration_s`,
  `f_max_hz`, `dx_m`, `engine`, `note`.
- `GET .../field/{rid}/frames` → `{"points", "valid", "normals", "u_axes", "n_frames",
  "n_points", "scale", "data_b64"}`: int8 frames × points, row-major, value = byte / 127
  × scale. Use a diverging colour map around 0, and play it slowed down (e.g. 1/100 of
  real time).
- `DELETE` (409 while running).

### Coherent sum of sources

`POST /api/simulations/{id}/combine` with
`{"receiver_id": "R1", "sources": [{"source_id": "S1", "gain_db": 0, "delay_ms": 0, "polarity": 1},
{"source_id": "S2", "gain_db": -3, "delay_ms": 12.5, "polarity": -1}]}` →
`{"key", "receiver_id", "sources", "fs", "metrics": {same keys as a pair, minus JLF/LJ},
"curves_url", "wav_url"}`.

The pressure IRs of the run are added with each source's gain, delay (0–2000 ms) and
polarity (±1): a loudspeaker system or an ensemble heard as one source, interference
included. `curves_url` → same shape as the pair curves. `wav_url` → the summed binaural
IR. Results are cached until the next run.

### Receiver grids (surface receivers)

A simulation may carry `receiver_grids`, saved by `PUT /api/simulations/{id}` like
`sources`. Changing them makes the results stale.

```json
"receiver_grids": [
  {"id": "G1", "name": "Stalls", "kind": "surface", "layer": "Audience", "offset_m": 1.2,
   "spacing_m": 1.0},
  {"id": "G2", "name": "Desk plane", "kind": "plane", "center": [5, 4, 1.2],
   "width_m": 10, "length_m": 6, "azimuth_deg": 0, "tilt_deg": 0, "spacing_m": 0.5}]
```

- `surface`: a square lattice (step `spacing_m`, 0.1–20 m) on every triangle of `layer`,
  offset by `offset_m` along the normal, towards the air. `plane`: a free rectangle,
  `width_m` along the azimuth direction, `length_m` across it, tilted by `tilt_deg`.
- Max 2500 points per grid (400 otherwise, at validation: use a larger spacing). Points
  outside the air volume, < 0.10 m from a surface or < 0.50 m from a source are disabled
  (kept, shown greyed).
- Each point stands for `spacing_m²` of area (area-weighted statistics).
- `GET /api/simulations/{id}/grids/{grid_id}/points` → `{"points": [[x,y,z]], "valid": [bool],
  "normals": [[nx,ny,nz]], "u_axes": [[ux,uy,uz]], "reasons": {"<index>": "0.04 m from a
  surface"}, "area_m2": [...], "n_points", "n_valid"}`: preview in the editor, before any
  run. `normals` point into the air. `u_axes` is the first axis of the lattice (draw each
  point as a `spacing` square in the (u, n × u) plane).
- Validation adds `"grids": [{"id", "n_points", "n_valid", "error"}]`. A grid with too
  many points or no valid point blocks the run. A simulation may have grids and no point
  receiver.
- Runs compute, for every source, the parameters at every valid point from the
  energy-time curve of the geometrical solver (no IR): T20, T30, EDT, C50, C80, D50,
  Ts_ms, G_db, SPL_db, JLF, LJ_db, MTI per band, STI and SPL_dba per point. With the wave
  solver on, **G_db, SPL_db, SPL_dba and the speech levels of the STI** take the wave
  band in the octaves it reaches (centre ≤ 2 × transition), as for a point receiver. This
  costs one more wave run per grid and source. Points outside the air of the wave band
  keep the geometrical levels (logged). The estimate's `grids_s` includes it.
- `settings.grid_rays` (default 50 000): rays per point. The receiver sphere is half the
  spacing, within 0.3–1 m. T30 maps can show isolated outliers in large halls: offer a
  median filter or say so.
- `meta.grids` in the results: `[{"grid_id", "sources": [{"source_id", "n_points", "n_valid",
  "n_rays", "rcv_radius_m", "duration_s"}]}]`.
- `GET /api/simulations/{id}/grids/{grid_id}/results?source_id=S1[&param=C80]` →
  `{"points", "valid", "area_m2", "reasons", "normals", "u_axes", "bands_hz", "params":
  {"C80": [[7 per point]], "STI": [1 per point], "decay_range_db": [[7 per point]], ...}}`.
  `null` = disabled point or undefined value. `decay_range_db` above −45 dB = T30
  unreliable there, as for pairs. `reasons`, `normals` and `u_axes` are absent in results
  from older versions.

### Saved views

`GET` / `PUT /api/simulations/{id}/ui-state/{key}` with `key` ∈ `results`, `auralizer`,
`editor`: an opaque JSON object (≤ 512 kB) that the UI stores per simulation (comparison
cards, chosen plots, auralizer strips and A/B/C set-ups). `{}` when nothing was saved.

### Exports

- `GET /api/simulations/{id}/export.xlsx` → workbook with sheets Settings, Parameters
  (one row per pair and parameter, columns = bands), Single numbers (STI, SPL Z/A/C),
  Points.
- `GET /api/simulations/{id}/irs.zip` → per pair `<source>__<receiver>_binaural.wav`
  (2 ch), `_omni.wav` (1 ch) and `_ambix.wav` (9 ch, 2nd order ACN/SN3D, world axes),
  float32, one shared scale, plus `parameters.json` and `README.txt`. Runs made before
  v0.5 have the binaural files only.
- The UI makes graph data, PNG images and clipboard copies from the data above.

### Reflection tracking

`GET /api/simulations/{id}/reflections?source_id=S1&receiver_id=R1&max_order=3` →

```json
{"source_id": "S1", "receiver_id": "R1", "max_order": 3, "speed_of_sound": 343.4,
 "spl_1m_db": 94.0, "n_direct_occluded": 0,
 "events": [{"kind": "direct|specular|diffracted", "order": 1,
             "time_ms": 11.18, "delay_ms": 3.44, "distance_m": 3.84,
             "spl_db": [7], "level_db": 89.4, "level_rel_db": -4.3,
             "azimuth_deg": 287.7, "azimuth_rel_deg": 17.7, "elevation_deg": -46.9,
             "surfaces": ["mat_CR2_floor"],
             "path": [[x, y, z], ..., [x, y, z]]}]}
```

- The deterministic arrivals exactly as the runs compute them (direct sound, image
  sources, first-order edge diffraction), with the source's directivity and gain. Sorted
  by time. Needs every layer assigned (409 otherwise), not the results. Computed on
  demand on the CPU.
- `max_order` 0–4, default = the simulation's ISM order (order 4 can take several
  seconds). Optional `azimuth_deg` / `elevation_deg` replace the receiver's look
  direction for `azimuth_rel_deg` (the listener turns the head).
- `path`: source, reflection (or diffraction) points in order, receiver. Draw it in 3D.
  `surfaces`: the layer of each reflection, source side first.
- `azimuth_deg`: direction of arrival, world frame (CCW from +X). `azimuth_rel_deg`:
  relative to the receiver's look direction (+ = to its left). `elevation_deg` from the
  horizontal. `level_rel_db` is relative to the direct sound (`null` if it is occluded).
- `spl_db`/`level_db` for a source of 94 dB at 1 m (the results' convention).

## Sounds (auralizer)

`GET /api/sounds` → `[{"id": "gen:speechish", "name": "Speech-like (synthetic)", "kind": "generated", "duration_s": 6.0}, ...]`
(generated: `clicks`, `speechish`, `sweep`, `noise_bursts`, plus uploaded files).

`GET /api/sounds/{id}.wav` → the WAV (mono, except a few Odeon recordings).
`POST /api/sounds` (multipart `file`, `name`, optional `credit`: the licence attribution
of a recording, e.g. "Freesound #512142 by lupalenzuela (CC0)") → Sound. Any PCM or float
WAV is accepted, stored mono (channels averaged) at its own sample rate. A sound's
`credit` goes into the credits of every render that plays it.

Odeon anechoic recordings (only when `data_odeon/anechoic/index.json` exists):
`kind: "odeon_local"`, `group: "concerts"|"misc"`, plus `credit` (the notice that MUST
accompany any result where the recording is recognisable) and `license` (Odeon
Auditorium/Combined >= 16: internal use only, no redistribution, no modification). Served
untouched. A piece shipped with several rooms is listed once.

## Ensembles (one source per player) and server-side render

Only when the Odeon anechoic library is present (`data_odeon/anechoic/`). Without it,
`GET /api/ensembles` returns `[]` (not an error).

`GET /api/ensembles` →

```json
[{"id": "mozart40", "title": "Mozart – Symphony 40 (15 players)", "n_tracks": 15,
  "duration_s": 114.5, "credit": "Derived from ...", "license": "Odeon ... internal use only ...",
  "tracks": [{"sound_id": "odeon-412", "name": "1st violinA"}]}]
```

`POST /api/simulations/{id}/ensemble` with

```json
{"ensemble_id": "mozart40", "center": [x, y, z_floor], "facing_deg": 90,
 "width_m": 12, "depth_m": 8, "replace_sources": false}
```

→ `{"simulation": Simulation, "scale": 0.47, "n_sources": 15}`.
With `"preview": true`, nothing is saved → `{"preview": true, "sources": [Source + "valid",
"reason"], "scale", "n_sources", "n_invalid"}`: the players as they would be added, each
with its check (in the air volume, ≥ 0.10 m from surfaces). Draw them, and mark the
invalid ones before the user confirms.

- `center` = where the CONDUCTOR stands, on the floor. The players sit upstage of it
  (away from the audience), in an orchestra layout (small bands in one arc). `facing_deg` = direction of the audience (azimuth, CCW from +X).
- The layout shrinks to fit `width_m` × `depth_m`. `scale` < 1 tells by how much (show
  it: "orchestra compressed to 47 %").
- Each new source carries `sound_id` (its track), `ensemble` (the title), and as
  `directivity` the Odeon pattern of its instrument (see Source directivities) or
  `omni`. Max 100 sources per simulation.
- Cost: every source has its own run. With the wave solver, N sources = N wave grids.

`POST /api/simulations/{id}/render` with

```json
{"receiver_id": "R1", "master_db": 0,
 "strips": [{"source_id": "E1_1", "sound_id": "odeon-412", "gain_db": 0, "muted": false}]}
```

→ `{"url": "/api/simulations/{id}/renders/render_<key>.wav", "credits_url": ".../render_<key>_credits.txt",
"fs": 44100, "duration_s": 114.5, "normalization_db": -15.7, "credits": [...], "n_sources": 15,
"receiver_id": "R1"}`.

A binaural mix of every non-muted strip with a sound, computed on the server and
normalised to −1 dBFS. Muted strips and strips with no `sound_id` may be sent: they are
skipped. Synchronous, no progress (time grows with the number of tracks × the longest
track). The credits file lists the notices that the licences of the recordings require:
offer it next to the WAV. `GET` on `url` / `credits_url` downloads them.

The same body to `POST /api/simulations/{id}/mixes` runs it as a background task, in its
own thread, never in the GPU queue. Bad requests are still refused at once (400 / 409).
It returns the task `{"id": "x…", "status": "queued|running|done|failed|cancelled",
"progress": 0-1, "stage": "track 3 / 15", "result": null | <the synchronous response>,
"error"}`.
- Poll `GET .../mixes/{rid}`. `GET .../mixes` lists the tasks, newest first (a page
  reload picks up a running one).
- Cancel with `POST .../mixes/{rid}/cancel`. It returns the task as it is. The render
  stops at the next track, so keep polling until `cancelled`.
- A mixing task left running by a server restart reads `failed` ("interrupted (the
  server restarted)"). Simulations and their other tasks are queued again (see Run).

`"sample_format": "pcm16" | "float32"`: default float32 for binaural, pcm16 for
ambisonic (half the size).

`"format": "ambisonic"` (default `"binaural"`) mixes with the 9-channel ambisonic IRs
instead (runs from v0.5). The WAV then has 9 channels (2nd order, ACN/SN3D, world axes)
and `format` is echoed. It is the input of the head-tracked player:

`GET /api/ambisonic/decoder?fs=48000&head=model` (44100 or 48000, `head` = a `/heads` id)
→ `{"order": 2, "fs", "head", "n_taps", "latency_samples", "convention": "ACN/SN3D, x
front, y left, z up", "filters": [9][2][n_taps]}`.
- `n_taps` is 256 with the model head, 1024 with a measured one (its HRIRs are longer).
- The ambisonic mix does not depend on the head. The player picks the decoder of the
  simulation's `settings.head`, no new run needed.
- Binaural output: `ear_e = Σ_ch filters[ch][e] ⊛
  ambi_ch`, after rotating the field, with the head looking along +x. To render a
  listener whose head has yaw ψ (azimuth CCW from +X) and pitch θ, rotate the FIELD by
  −ψ, −θ (engine: `auralis.engine.ambidecode.rotation_matrix`. For yaw only, (Y, X) and
  (T, S) rotate by ψ, (V, U) by 2ψ). Initial yaw = the receiver's azimuth.
- For head-fixed listening, use the direct binaural IR (2nd order limits the accuracy).

## Animated renders (trajectories)

A render where the receiver, some sources, or both move along their `path`: one
receiver, the sources that have a sound. Doppler on the direct sound, and a late field
from key frames every `reverb_dt` seconds, cross-faded. When the simulation uses the
wave solver, **the wave band is animated too**: the geometric render is high-passed at
the crossover, and the low band comes from static wave-solver IRs at key frames (at most
λ/8 apart along the path), low-passed and cross-faded.

How many wave runs:
- Static receiver: by reciprocity, 2 runs (one per ear) for every source and key.
- Static sources: one run per source, probes at the ears along the receiver's path.
- A source and the receiver both moving: reciprocity at the receiver's key frames, two
  runs per key (one per ear), capped by `wave_max_runs` (below).
- A directional source keeps its first-order pattern (see
  [Source directivities](#source-directivities)).

`result.wave` = `{"animated", "note"}`. It does not need the static results: it uses the
scene, materials, settings and paths.

`POST /api/simulations/{id}/motion/check` with

```json
{"receiver_id": "R1", "duration_s": null,
 "strips": [{"source_id": "S1", "sound_id": "odeon-412", "gain_db": 0, "muted": false,
             "loop": false, "offset_s": 0}]}
```

- `wave_max_runs` (optional, default 64): the cap on wave-band runs when the listener and
  a source both move. Above it, the listener's key frames are spread, with a warning.
- `loop`: a recording shorter than the render is repeated with cross-fades, for
  continuous sounds such as traffic, crowds or engines.
- `offset_s`: the strip starts that far into its recording, so that several sources
  sharing one recording play different passages.

→

```json
{"ok": true, "blocking": [], "warnings": ["..."], "duration_s": 12.0, "path_end_s": 12.0,
 "paths": [{"id": "R1", "valid": true, "reasons": [], "length_m": 14.2, "start_s": 0,
            "end_s": 12.0, "max_speed_mps": 1.4}]}
```

- `duration_s: null` = until the last waypoint of the moving points involved
  (`path_end_s`). Otherwise 0.5–600 s. Tracks are cut or padded with silence to it. The
  WAV is longer: it keeps the room's whole decay after the last sample (the IR length),
  points staying where their paths end.
- Blocking: nothing moves, a path leaves the air volume or passes < 0.10 m from a
  surface, a source comes < 0.50 m from the receiver at some time (the message gives the
  time), no source with a sound, unassigned layers.

`POST /api/simulations/{id}/motion` with the same body plus
`{"doppler": true, "reverb_dt": 0.75, "master_db": 0, "label": "walk to the stage",
"turn_taking": null, "background_noise": false}` → `{"render": MotionRender, "job": Job}`
(409 with the blocking reasons if the check fails).
- `background_noise: true` (400 if the simulation's `settings.background_noise` is
  `"none"`) mixes that noise into the render at its physical level: diffuse, with the
  given octave levels at each ear. To put the recordings on the same scale, each track
  is taken as speech at the vocal effort of the settings (`speech_effort`:
  60 dB(A) at 1 m on axis for "normal", 66 for "raised"), measured over its utterances
  only. That is right for speech. For music, correct with the strip's `gain_db`.
- `turn_taking: {"gap_s": 0.45}` (0–5 s, at least two playing strips, 400 otherwise)
  makes a **conversation**. Each recording is cut at its own pauses, then the strips
  speak in turn, one utterance each in the order of `strips`, `gap_s` apart, from 0.5 s.
  A strip that runs out drops out. The first turn that would not end inside the render
  ends the conversation (no sentence cut short). The files are not modified.
- The job is in the same queue as the runs (`Job.kind` = `"motion"`, `Job.render_id`).
  Poll it with `GET /api/jobs/{jid}`, cancel with `POST /api/jobs/{jid}/cancel`. Run
  jobs have `kind: "simulation"`.

MotionRender:

```json
{"id": "a1b2c3", "simulation_id": "m1", "created": "...", "status": "queued|running|done|failed|cancelled",
 "receiver_id": "R1", "label": "", "n_sources": 3, "error": null, "job_id": "j9",
 "url": "/api/simulations/m1/motion/a1b2c3/render.wav",
 "credits_url": "/api/simulations/m1/motion/a1b2c3/credits.txt",
 "result": {"fs": 44100, "duration_s": 13.0, "render_s": 95.1, "normalization_db": -12.3,
            "credits": ["..."], "warnings": ["..."], "doppler": true,
            "turns": [{"source_id": "S1", "start_s": 0.5, "end_s": 3.12}]}}
```

`turns`: null without `turn_taking`. `background_noise`: null, or `{"curve", "dba",
"speech_dba_1m", "effort"}`. `url`, `credits_url` and `result` only once `done`.

`GET /api/simulations/{id}/motion` → MotionRender[] (newest first). `GET .../motion/{rid}`
→ MotionRender. `DELETE .../motion/{rid}` (409 while queued or running). The request is a
snapshot: editing the simulation afterwards does not change a queued render.

Rough cost: about 2.5 × real time per source with the GPU, 7 × on the CPU
(`GET /api/health` → `gpu`). Show the estimate before rendering.

## Example project

On first launch (no `examples.json` in the projects folder), the server builds an
"Examples" project in the background and queues the runs of its static examples
(version 7). There are three groups: the **concert halls** (Musikverein, Gothenburg
Konserthus, Royal Festival Hall, only with the local Odeon data), the **BRAS rooms**
(CR1 with two door openings, CR2, CR3, CR4, built from the files in `BRAS/`), and a
**Paris street** with real recordings and two finished animated renders (nothing queued,
`"queue": false`). A store made by an older version gets only the examples added after
its version. Examples that this version no longer ships stay in a store that has them.

`POST /api/examples/install` with `{"force": false, "run": true}` → `{"version",
"project_id", "examples": [{"example", "simulation", "ok", "blocking"}]}` (the Paris
street adds `"queue": false`). `force` builds a new copy.
