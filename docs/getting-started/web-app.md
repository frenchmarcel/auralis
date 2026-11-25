# The web app

The web app is the easiest way to use Auralis. It runs on your own machine.
A Python server drives the engine. The interface runs in your browser.
Nothing leaves your computer.

## Start the app

```bash
bash webapp/start.sh
```

The first start takes a few minutes. The script:

1. creates a Python environment for the app (`webapp/.venv`),
2. installs the app packages (FastAPI, Uvicorn, readers for XLSX, DXF, Rhino,
   IFC and SOFA files),
3. downloads free (CC0) furniture models and textures from Poly Haven,
4. builds the interface with npm,
5. starts the server and opens <http://127.0.0.1:8765>.

The next starts are fast. To rebuild the interface after an update:

```bash
REBUILD=1 bash webapp/start.sh
```

To use another port:

```bash
PORT=8770 bash webapp/start.sh
```

Your projects are stored in `~/AuralisProjects`. Set `AURALIS_PROJECTS` to
use another folder.

## How a project is organised

| Level | What it holds |
|---|---|
| Project | A job, for example "Concert hall study" |
| Space | One place in the project, for example "Main hall" |
| Geometry | One imported model of that space |
| Simulation | Materials, sources, receivers and settings for one geometry |

You can have several simulations per geometry. For example, one with the
current materials and one with a new ceiling.

```{figure} ../images/app-home.png
:alt: The home page of the web app with the Examples project
:width: 100%

The home page. The Examples project holds five spaces. Recent runs are listed on the right.
```

## The three views

Each simulation has three views. The tabs at the top switch between them.

- **Editor**: the 3D model, the materials, the sources and receivers, and the settings.
- **Results**: parameters, curves, maps and reflections.
- **Auralizer**: listen to the results and render WAV files.

```{figure} ../images/editor-bras.jpg
:alt: The editor with the BRAS auditorium, two sources and five microphones
:width: 100%

The editor with the BRAS auditorium (CR4). The 3D view shows the sources
(green) and the receivers (blue). The tabs on the right hold the materials,
sources, receivers, settings, objects and receiver grids. Here the Materials
tab is open. The slider on the left cuts the walls at a given height.
```

## A first simulation, step by step

1. **Create a project and a space.** Use the buttons on the home page.
2. **Import a model.** Choose a file and give its unit (m, cm or mm) and its
   up axis (Z or Y). The app never guesses them. See {doc}`../user-guide/models`.
3. **Read the model check.** The app tells you if the model is closed and
   lists open edges, small gaps and overlapping faces. A closed model can use
   the wave solver.
4. **Create a simulation** on the geometry.
5. **Assign materials** to each layer. See {doc}`../user-guide/materials`.
6. **Place sources and receivers.** Type positions, click on surfaces, or use
   the batch tools. See {doc}`../user-guide/sources-receivers`.
7. **Choose the settings.** Start with the Survey mode for a fast first
   result. See {doc}`../user-guide/simulation-settings`.
8. **Run.** The app shows an estimate of the run time, then the progress.
   Only one run uses the GPU at a time. Other runs wait in a queue.
9. **Read the results.** See {doc}`../user-guide/results`.
10. **Listen.** Open the Auralizer and pick a sound. See
    {doc}`../user-guide/listening`.

## The example project

On the first start, the app builds an **Examples** project. It holds concert
halls, the BRAS rooms of TU Berlin and a Paris street with real recordings.
The concert halls need local data (see {doc}`../user-guide/examples`). The
examples are a good way to see what each feature does.
