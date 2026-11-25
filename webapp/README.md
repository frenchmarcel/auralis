# Auralis web app

A local room acoustics app: import a model, assign materials, place sources
and receivers, run the engine (GPU), read the results and listen. A Python
server on this machine drives the engine. The interface runs in the browser.
User guide: `docs/getting-started/web-app.md`.

## Start

```bash
bash webapp/start.sh
```

The first run creates `webapp/.venv` on top of the `python310` env (numpy
is shared, `python310` is not changed) and builds the interface. Then the
server starts and opens http://127.0.0.1:8765. `REBUILD=1 bash webapp/start.sh`
rebuilds the interface.

Development (hot reload of the interface):

```bash
webapp/.venv/bin/python -m webapp.server        # API on :8765
cd webapp/web && npm run dev                      # UI on :5173, proxies /api
```

## Examples

On the first start, the app builds an **Examples** project:

- three concert halls with the materials of the Odeon models: Mozart in the
  Musikverein, Mahler in Gothenburg, Bruckner at the Royal Festival Hall
  (they need the local Odeon data, `data_odeon/`),
- the BRAS rooms CR1 (door open 30.4° and 4.1°), CR2, CR3 and CR4, with the
  measured positions and materials,
- a Paris street with real recordings and two finished renders
  (map: OpenStreetMap, ODbL. Recordings: Freesound, CC0).

The halls and the BRAS rooms are computed in the background (seconds to
minutes each). Open one, go to the Auralizer and listen, or "Render to WAV".
Details: `docs/user-guide/examples.md`.

## What it does

- **Import**: `.obj`, Odeon `.Par`, `.dxf`, Rhino `.3dm`, `.ifc` (building
  elements, or one IfcSpace as a closed room), SketchUp `.skp`. You give the
  unit and the up axis. The app never guesses them. It checks that the model
  is closed, by topology or by voxels.
- **Model**: materials (presets, pyroomacoustics, BRAS, local Odeon library,
  or your own α, porous builder, complex Z or R tables, scattering per
  octave). Furniture, a `.glb` visual overlay, sources with directivities
  (presets, JSON, CSV, AES69 SOFA), receivers, receiver grids, trajectories.
- **Simulate**: geometrical (image sources and rays), wave (FDTD) or hybrid,
  with a run-time estimate.
- **Results**: ISO 3382-1 parameters on the omni IR (T20, T30, EDT, C50,
  C80, D50, Ts, G, SPL Z/A/C, J_LF, L_J, 63 Hz from the wave band), STI
  (IEC 60268-16:2020), ISO 3382-3 open-plan offices. Comparison cards,
  Report view, reflection tracking in 3D, grid maps, modal maps, wave-field
  animation, coherent sums of sources. Exports: xlsx, IR zip (binaural,
  omni, AmbiX), CSV, PNG.
- **Listen**: binaural (fixed head, exact IR) or head-tracked 2nd-order
  ambisonics, A/B/C switching, ensembles (one source per player), WAV
  renders with the recordings' credits, animated renders with Doppler.
- **3D tools**: particles, reflection density, absorption colours, PNG
  capture.

The user guide (`docs/user-guide/`) describes each feature.

## Extra dependencies

`start.sh` installs these into `webapp/.venv` only: fastapi, uvicorn,
python-multipart, openpyxl (xlsx), ezdxf (DXF), rhino3dm (3DM), ifcopenshell
(IFC, LGPL, used as a library), h5py (SOFA). The CUDA kernels need nvcc. If your system nvcc is too old, see
`docs/getting-started/installation.md`.

## Where things are

| Path | Content |
|---|---|
| `API.md` | HTTP contract between server and UI |
| `server/` | FastAPI app, storage, job queue, simulation runner |
| `web/` | browser UI (Vite + React + three.js) |
| `reference/` | inventory of Treble's public documentation (in French) |

- Projects are stored in `~/AuralisProjects`, outside the repository. Set
  `AURALIS_PROJECTS` to change it.
- Material libraries: engine presets, pyroomacoustics (MIT, `server/data/`),
  the Odeon library from `data_odeon/` **if present** (local research
  licence, never redistributed), and your own materials.

## How a run works

The server builds the same scene manifest as the BRAS harnesses and passes
it to `auralis.engine.project`. So the app and the validation bench share
one code path. Each run is a separate process (`server/runner.py`). One GPU
job runs at a time, with a 10 s pause between runs. If the GPU refuses new
CUDA contexts, the runner waits. It does not fall back to the CPU in
silence. The server process itself never opens a CUDA context.

## Furniture and textures

`server/data/objects_catalog.json` lists 56 objects. 45 are Poly Haven
models (CC0), for display only. 11 are acoustic objects without a model
(curtain, absorber panel, audience area, piano...). The solvers only see a
simple **acoustic proxy** (planes and blocks) sized on the real model.

The default 3D look, "Realistic", is also for display only. Each surface
gets a texture from its material. Light comes from a CC0 HDR map and a sun,
with soft shadows and ambient occlusion (N8AO). Closed rooms are cut at
1.2 m (or a third of their height) so you can see inside ("Cut-away",
"Iso"). Without textures, the view shows plain colours.

`start.sh` runs these downloads the first time:

- `python -m webapp.server.fetch_assets`: models into `assets/visual/`, then
  measures them,
- `python -m webapp.server.fetch_textures`: CC0 textures and HDRIs from Poly
  Haven into `assets/textures/` (~41 MB), with credits in `index.json`.

## Not packaged

Keep `webapp/.venv/`, `webapp/web/node_modules/`, `webapp/assets/visual/`
(75 MB) and `webapp/assets/textures/` (41 MB) out of shared archives. They
can all be downloaded or built again.
