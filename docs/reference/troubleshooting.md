# Troubleshooting

Auralis never fails silently. When it falls back to slower code or changes
something on its own, it prints a message that starts with `[Auralis]`. Read
these messages first.

## The GPU is not used

Check what the engine sees:

```bash
python3 - <<'EOF'
from auralis.engine import native_available, gpu_available
print(native_available(), gpu_available())
EOF
```

| You see | Likely cause | What to do |
|---|---|---|
| `True False` | The GPU library is missing or was built for another GPU | Run `bash native/build_cuda.sh`. Set `AUR_CUDA_ARCH` if no GPU was visible at build time. |
| `False False` | The CPU library is missing | Run `bash native/build.sh` |
| A version warning | The three libraries come from different versions | Build all three again |

On a laptop, the GPU can refuse new work after a sleep or a long session,
even when `nvidia-smi` shows it. A restart of the machine fixes it.

## `nvcc` fails

- An old CUDA Toolkit does not know recent GPUs. Install a recent nvcc in a
  separate conda environment (see {doc}`../getting-started/installation`).
- An old nvcc can also refuse a recent gcc. The same fix applies.

## The wave solver is not available

- The model is not closed. Fix its open edges, or declare it an open scene if
  it is an outdoor space or an object in free field.
- A source or a receiver is too close to a surface. Keep at least one grid
  cell of air around it.

## "FDTD grid too large"

The grid needs more cells than the limit allows. You can:

- lower the transition frequency,
- simplify the model, for example remove far parts that do not matter,
- on a large GPU, raise the limit with `AURALIS_FDTD_MAX_CELLS`.

## "position outside the air"

A point is inside a solid cell of the wave grid. This happens when a point is
very close to a wall, because the grid turns walls into cells. Move the point
away from the wall by more than one cell.

## A wall filter "was ACTIVE"

A wall filter created a little energy on the grid. Auralis corrected it. The
message is a warning, not an error. Very rigid materials (α below 0.03 at low
frequencies) make this more likely. Realistic values, like 0.05 to 0.15 for a
furnished room, also sound more natural.

## "FDTD instability caught by the watchdog"

The wave solver stopped because the energy grew. Auralis then uses the
geometrical solver alone for that source and says so. Try:

- a shorter response,
- more absorption on near-rigid materials.

Please report the scene if it happens with normal materials.

## The run is slow

- Use the Survey mode first.
- Lower the number of rays.
- Lower the transition frequency. Halving it makes the wave solver about 16
  times faster.
- Check that the GPU is used.

## T30 is empty

The response does not decay by 35 dB within its length. Make the response
longer with the IR length or the stop decay setting.

## The app shows old results

Results become stale when you change materials, sources, receivers or
settings. Run the simulation again.
