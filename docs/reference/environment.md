# Environment variables

These variables change how Auralis runs. Most users never need them. Set them
in the shell before you start a script or the web app.

## Everyday use

| Variable | Default | Effect |
|---|---|---|
| `AURALIS_PROJECTS` | `~/AuralisProjects` | Folder where the web app stores your projects |
| `AURALIS_PYTHON` | `python3` | Python on which `webapp/start.sh` builds the app's environment, the first time. It needs NumPy. |
| `AURALIS_GPU` | (unset) | `0` turns the GPU off. The engine then uses the CPU library. |
| `AURALIS_FDTD_GPUS` | (unset) | GPUs for the wave solver, for example `0,1,2,3`. The grid is split between them. |
| `AURALIS_FDTD_MAX_CELLS` | 12 million | Largest wave grid in the web app. Raise it on large GPUs, for example `4e9`. |
| `AURALIS_PAIR_WORKERS` | automatic | Processes that compute the source and receiver pairs of a run side by side. With the GPU tracer, the default is a third of the CPU cores, at most 6. Without it, the default is 1. `1` computes the pairs one at a time. |
| `AURALIS_SUBGRID_MAX` | 450 | Memory for the conformal correction, in millions of sub-cells (about 4 bytes each). Raise it on machines with a lot of memory. |

## Libraries

| Variable | Effect |
|---|---|
| `AURALIS_NATIVE_LIB` | Path to the CPU library to use |
| `AURALIS_CUDA_LIB` | Path to the GPU library to use |
| `AURALIS_GRID_NATIVE` | `0` builds the wave grid in Python instead of C++ (slow, for checks) |
| `AURALIS_GRID_THREADS` | Threads used to build the wave grid. 0 means all. |
| `AURALIS_SKP_PYTHON` | The Python of the separate SketchUp reader environment |

## Wave solver

| Variable | Default | Effect |
|---|---|---|
| `AURALIS_FDTD_CONFORMAL` | 1 | `0` turns the conformal correction off (staircase walls) |
| `AURALIS_FDTD_CONFORMAL_ANISO` | 0 | `1` uses the anisotropic form of the correction |
| `AURALIS_FDTD_ALIGN` | 0 | `1` turns the grid to follow the scene |
| `AURALIS_FDTD_SUBCELL` | 1 | `0` turns the openings smaller than a cell off |
| `AURALIS_FDTD_SUBSONIC_HZ` | 16 | High-pass frequency of closed rooms. `0` turns it off. |
| `AURALIS_FDTD_DC_DAMP` | 0.01 | Small damping of the very low frequencies on physical walls. `0` turns it off. |
| `AURALIS_FDTD_RESONANT` | 1 | `0` turns the resonant panel model off |
| `AURALIS_FDTD_PHYS_FIT` | auto | How physical wall filters are fitted. `anchored` keeps the older method. |
| `AURALIS_FDTD_GRID_CACHE` | 1 | `0` turns off the reuse of grids between runs |
| `AURALIS_FDTD_EMPTY_DENSE` | 1 | `0` uses the indexed grid for the empty-box run of open scenes |
| `AURALIS_FDTD_SLABS` | 0 | `1` forces the slab-by-slab grid (for tests) |
| `AURALIS_FDTD_GRID_TIMING` | 0 | `1` prints the time of each step of the grid build |
| `AURALIS_HEAD2_CELLS` | 1 | Probe spacing of the second-order head, in cells |

## For tests and development

| Variable | Effect |
|---|---|
| `AURALIS_FDTD_SLABS_LIB` | The library for slab-by-slab grids (the emulation library in tests) |
| `AURALIS_FDTD_DENSE_LIB` | The library for the dense empty-box run |
| `AURALIS_CUDA_EMU_LIB` | The CPU emulation of the GPU library used by the tests |
| `AURALIS_CONFORMAL_AXIS_NUMPY` | `1` keeps the older, slower path of the conformal correction |
| `AURALIS_FDTD_SUBCELL_SCALE` | Scales the conductance of small openings (sensitivity studies) |
| `AURALIS_RESUME_JOBS` | Set by the server. `0` means interrupted runs are not restarted. |

## Build scripts

| Variable | Effect |
|---|---|
| `AUR_CUDA_ARCH` | GPU architecture for `build_cuda.sh`, for example `sm_89` |
| `PORT` | Port of the web app for `start.sh` (8765 by default) |
| `REBUILD` | `1` rebuilds the web interface in `start.sh` |
