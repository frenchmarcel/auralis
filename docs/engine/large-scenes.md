# GPUs and large scenes

## One GPU

On one GPU, the wave solver keeps the whole grid in GPU memory. The
geometrical solver traces its rays on the GPU too. On an RTX 2000 Ada laptop
GPU (70 W):

- the ray tracer is about 25 times faster than the multithreaded CPU code,
- the wave solver runs at about 2 billion cell updates per second.

By default, the app limits the wave grid to 12 million cells. This keeps a
run safe on a laptop GPU.

## Several GPUs

Set `AURALIS_FDTD_GPUS` to the list of GPUs to use, for example:

```bash
export AURALIS_FDTD_GPUS=0,1,2,3
```

The grid is then cut into slabs along x. Each GPU computes its slab and
exchanges two planes of cells with its neighbours at every time step. The
result is identical, bit for bit, to a run on one GPU.

Measured speed-ups: four H100 GPUs run 3.7 times faster than one. Four B200
GPUs run 3.6 times faster than one.

## Very large grids

Above about 2 billion air cells, one grid no longer fits the 32-bit indices
of the solver. Auralis then builds the grid **slab by slab**. Each slab is
built on the CPU, sent to its GPU and freed. No full grid ever sits in the
host memory.

Example: a 200 m street at 1 kHz on four B200 GPUs.

| Quantity | Value |
|---|---|
| Cells | 10.5 billion |
| Air cells | 5.4 billion |
| Total time | 1 h 13 |
| GPU memory | 433 GiB over four GPUs |

## Open scenes on large grids

An open scene needs a second run in an empty box (see
{doc}`wave-solver`). For large boxes, this run uses a dense kernel with only
12 bytes per cell. It is split over the same GPUs as the scene.

## Deterministic results

Two GPU runs of the same scene give the same numbers, bit for bit. The wall
filters are summed in a fixed order for each cell.

## Settings for big machines

| Variable | Use |
|---|---|
| `AURALIS_FDTD_GPUS` | GPUs to use, for example `0,1,2,3` |
| `AURALIS_FDTD_MAX_CELLS` | Raise the app's cell limit, for example `4e9` |
| `AURALIS_SUBGRID_MAX` | Memory for the conformal correction, in millions of sub-cells |
| `AURALIS_FDTD_ALIGN` | Turn the grid to follow a scene that is not aligned with the axes |

See {doc}`../reference/environment` for all variables.

## Running on a cluster

On a Slurm cluster, run each job inside your allocation. Keep the native
libraries built on the compute nodes, since the login node may have another
CPU architecture. A typical command looks like this:

```bash
srun --jobid=<job> --overlap -n 1 -c 48 bash my_run.sh
```

To check that a new GPU library is correct on several GPUs, run:

```bash
python3 tools/split_check.py 0,1 0,1,2,3
python3 tools/io_check.py
```
