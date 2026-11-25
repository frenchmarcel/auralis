# Installation

Auralis has three parts:

1. The **engine**, a Python package (`auralis/`). It needs only NumPy.
2. The **native libraries**, in C++ and CUDA. They make the engine much
   faster. They are optional, but you will want them.
3. The **web app** (`webapp/`), a local server and a browser interface.

## Requirements

| Part | Needs |
|---|---|
| Engine | Python 3.9 or later, NumPy |
| CPU library | A C++17 compiler (g++ or clang) |
| GPU library | An NVIDIA GPU and the CUDA Toolkit (nvcc 11.8 or later) |
| Web app | The engine, Node.js with npm, and an internet connection for the first start |

The web app installs its own Python packages (FastAPI, Uvicorn and a few file
readers). You do not need to install them by hand. It builds its environment
on `python3`. Set `AURALIS_PYTHON` to use another Python that has NumPy.

## Get the code

Copy or clone the repository, then go into it:

```bash
cd auralis
```

All commands on this page run from the repository root.

## Build the native libraries

```bash
bash native/build.sh
bash native/build_cuda.sh
bash native/build_cuda.sh --emulate
```

- `build.sh` builds the CPU library (`libauralis`).
- `build_cuda.sh` builds the GPU library (`libauralis_cuda`).
- `build_cuda.sh --emulate` builds a CPU copy of the GPU code. The tests use it.

The libraries go to `auralis/lib/<platform>/`, for example
`auralis/lib/linux-x86_64/`.

Build all three together. If one library is older than the others, the engine
sees a version mismatch. It then prints a warning and falls back to slower
code.

### Choosing the GPU architecture

`build_cuda.sh` asks nvcc for the architecture of the GPU it sees. If no GPU is
visible at build time, nvcc picks an old default and the script prints a
warning. Set the architecture yourself in that case:

```bash
AUR_CUDA_ARCH=sm_89 bash native/build_cuda.sh    # RTX 40 series, RTX 2000 Ada
AUR_CUDA_ARCH=sm_86 bash native/build_cuda.sh    # RTX 30 series
```

If your system nvcc is too old, install a recent one in a separate conda
environment and put it first in your `PATH`:

```bash
conda create -n auralis-cuda -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev
conda activate auralis-cuda
bash native/build_cuda.sh
```

## Check the installation

```bash
python3 - <<'EOF'
from auralis.engine import native_available, gpu_available
print(native_available(), gpu_available())
EOF
```

You should see `True True` on a machine with a GPU, and `True False` without
one. Auralis works in both cases. Without the CPU library it still works, in
pure Python, but slowly.

## Run the tests

```bash
python3 -m pytest tests/ -q
```

Some tests need data that is not in the repository (measurements, licensed
libraries). They are skipped when the data is missing. The web app tests need
the packages of the app environment:

```bash
webapp/.venv/bin/python -m pytest tests/test_webapp_*.py -q
```

## Optional data

Some features use data that you download separately. See
{doc}`../reference/data-licences` for what each folder holds and its licence.

| Folder | Content | Used for |
|---|---|---|
| `data_bras/` | BRAS measurements and models | Validation, the CR2 example |
| `data_hrtf/` | The FABIAN measured head | Binaural output with a real head |
| `data_odeon/` | Odeon material library and recordings | Local use only, with an Odeon licence |
