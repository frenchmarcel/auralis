# Models

A model is the 3D geometry of the space you want to simulate. Auralis works
with triangles. Each triangle belongs to a **layer**. You assign one material
per layer.

## Supported files

| Format | What Auralis reads | Layers come from |
|---|---|---|
| `.obj` | Triangles and polygons | `usemtl` names |
| Odeon `.Par` | Points, surfaces, constants and expressions | Odeon surface names |
| `.dxf` | 3DFACE, MESH, polyface and polygon meshes, also inside blocks | DXF layers |
| Rhino `.3dm` | Meshes, and the render meshes of Breps and extrusions | Rhino layers |
| `.ifc` | Building elements, or the closed shell of one IfcSpace | IFC class and material |
| SketchUp `.skp` | Faces of groups and components (SketchUp 2013 to 2020) | SketchUp materials |

When you import a file, you give two things:

- the **unit**: metres, centimetres or millimetres,
- the **up axis**: Z (the default) or Y. A Y-up file is turned to Z-up.

The app never guesses these values. A wrong unit gives a room 100 times too
big or too small, and a wrong axis puts the floor on a wall.

Some formats need extra notes:

- **Odeon models** are in metres with Z up. Models built with Odeon's
  parametric commands (loops, cylinders, revolved surfaces) are refused with a
  message.
- **IFC models** can be very detailed. By default, spaces, openings,
  annotations, grids and proxies are left out. You can also import one
  IfcSpace. Its closed shell then becomes the room, and each face takes the
  layer of the nearest element.
- **SketchUp files** are read by a separate tool. The app tells you how to
  install it if it is missing.

## The model check

After the import, the app checks the model. It lists findings with their
place in 3D:

| Finding | Meaning |
|---|---|
| Open edge | An edge with only one face. The volume is not closed there. |
| Non-manifold edge | An edge shared by more than two faces. |
| Degenerate triangle | A triangle with no area. |
| Short edge | A very short feature edge. It can force a fine grid. |
| Overlapping faces | Two faces on top of each other. The solvers see one surface with an unclear material. |
| Small gap | A thin gap between surfaces that should touch. |

A model is **watertight** when it has no open edge. The wave solver needs a
closed volume, or a model that you declare as an open scene (see below).

You can open the check again at any time from the **⋯** menu of the editor,
with **Geometry check**.

```{figure} ../images/model-check.jpg
:alt: The geometry check dialog of the BRAS seminar room
:width: 100%

The geometry check of the BRAS seminar room. The model is watertight, so both
solvers can run. Short edges and small gaps are judged against a quarter of
the wavelength at the transition frequency.
```

You can mark a finding as known, for example an opening that you want. A known
finding stops counting. It does not change what the solvers see.

## Closed rooms and open scenes

The wave solver simulates the air inside a box of cells. It needs to know
where the air ends.

- A **closed room** has walls all around. The air is what is inside.
- An **open scene** is an object in free field, a street, or an open-air
  theatre. Declare it in the settings. The solver then puts absorbing layers
  at the edges of its box (a perfectly matched layer, CPML). Sound leaves the
  model as it would outdoors.

If a model is not closed and not declared open, the wave solver is not
available. The geometrical solver always works.

## Visual model

You can add a `.glb` model on top of the acoustic model. It is for display
only. The solvers never see it. Use it for textures, furniture details and the
buildings around a street.

## Furniture

The app has a furniture library. Each object has two forms:

- a realistic **visual model** (free CC0 models from Poly Haven), for display,
- a simplified **acoustic proxy**, made of a few boxes, which the solvers use.

The proxy is what is simulated, never the detailed mesh. You can show the
visual model, the proxy, or both, with the **Visual**, **Acoustic** and
**Both** buttons of the editor. Place objects from the **Objects** tab.

## Tips for good models

- Model the surfaces that matter for sound. Small details below about 10 cm do
  not change the result much and slow the wave solver down.
- Give one layer to each material. Two layers can share a material later.
- Keep the source and the receivers at least one grid cell away from any
  surface. The app checks this before a run.
- Close the model if you want the wave solver in a room.
