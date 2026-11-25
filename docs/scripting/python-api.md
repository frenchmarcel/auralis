# Using Auralis from Python

The engine is the `auralis.engine` package. Everything the web app does, you
can do from a script.

## Build a scene

A scene is a list of triangles with one material each.

**A box room:**

```python
from auralis.engine import make_shoebox, add_box

scene = make_shoebox((12.0, 8.0, 4.0), wall_material="brick",
                     floor_material="parquet", ceiling_material="acoustic_tile")
scene = add_box(scene, (3.0, 1.0, 1.1), (7.5, 0.5, 0.0), "concrete", "bar")
```

`add_box(scene, size, origin, material, name)` adds a solid box, for example a
counter or a stage.

**Your own triangles:**

```python
import numpy as np
from auralis.engine import SceneData, get_preset

tri_verts = np.array(...)          # (n, 3, 3) triangle corners in metres
tri_mat = np.array(...)            # (n,) material index of each triangle
materials = [get_preset("plaster"), get_preset("carpet")]
scene = SceneData.from_materials(tri_verts, tri_mat, materials, ["walls", "floor"])
```

A material can be a preset name or a dictionary with `absorption` (seven
values from 125 Hz to 8 kHz) and `scattering`.

**An OBJ file:** use a manifest. See {doc}`manifests`.

## Sources and receivers

```python
from auralis.engine import SourceDef, ReceiverDef

source = SourceDef("S1", np.array([2.0, 3.0, 1.5]), directivity="speaker_2way",
                   forward=np.array([1.0, 0.0, 0.0]), gain_db=0.0)
listener = ReceiverDef("R1", np.array([8.0, 4.0, 1.2]),
                       forward=np.array([-1.0, 0.0, 0.0]))
```

- `directivity` is `None` (omni), a preset name or a `Directivity` object.
- `forward` is the front axis of the source, or the look direction of the
  listener.
- `spl_1m` is the source level at 1 m in dB (94 by default).

## Settings

`BakeConfig` holds all settings. The most useful ones:

```python
from auralis.engine import BakeConfig

config = BakeConfig(
    n_rays=200_000,         # rays per source
    ism_order=2,            # image source order
    fdtd=True,              # run the wave solver
    fdtd_fcross=0.0,        # transition frequency in Hz, 0 = automatic
    ambisonic_order=2,      # also make an Ambisonics response (0 = off)
    hrtf_path="",           # a measured head (.npz), "" = model head
    temperature_c=20.0,
    humidity_percent=50.0,
)
```

See {doc}`../reference/settings` for the full list.

## Run one pair

```python
from auralis.engine import bake

result = bake(scene, source, listener, config)
```

`result` is a `BakeResult`:

| Field | Content |
|---|---|
| `ir` | Binaural impulse response, shape (n, 2) |
| `fs` | Sample rate |
| `ambi` | Ambisonics response, shape (n, channels), when `ambisonic_order > 0` |
| `metrics` | Parameters from the echogram, levels G and SPL |
| `events` | The direct sound, image sources and diffraction paths |
| `hist_total` | The echogram, per octave band |
| `omni_parts` | The omni response of each solver before the crossover |
| `stats`, `timings` | What ran, and how long it took |

## Many receivers

The wave solver builds one grid per source. With several receivers, compute
the wave band once and reuse it:

```python
from auralis.engine import precompute_low_bands

receivers = [listener, ReceiverDef("R2", np.array([6.0, 2.0, 1.2]))]
low = precompute_low_bands(scene, source, receivers, config)
results = {r.name: bake(scene, source, r, config, low_band=low[r.name])
           for r in receivers}
```

## Room acoustic parameters

The web app computes the ISO 3382 parameters on the omni response, in true
octave bands. Do the same in a script:

```python
from auralis.engine import rirmetrics

omni = result.ambi[:, 0]              # the W channel (needs ambisonic_order > 0)
m = rirmetrics.band_metrics(omni, result.fs)
print(m["T30"], m["C80"], m["Ts"])    # one value per octave band, Ts in ms
```

## Listen

```python
from auralis.engine import fft_convolve, make_test_signal, write_wav

dry = make_test_signal(result.fs, "speechish", duration=4.0)   # or your own dry recording
wet = fft_convolve(dry, result.ir)       # a mono signal against the two ears: shape (n, 2)
write_wav("speech_in_room.wav", 0.9 * wet / abs(wet).max(), result.fs)
```

Your own recording must be dry, mono and at the sample rate `result.fs`
(48 kHz by default).

To decode an Ambisonics response for headphones:

```python
from auralis.engine.ambidecode import decode, decoder_filters

filters, latency = decoder_filters(2, result.fs)
binaural = decode(result.ambi, filters)[latency:]
```

## Moving sources

```python
from auralis.engine import (MotionConfig, MovingSource, Trajectory,
                            render_moving)

times = np.linspace(0.0, 6.0, 33)
path = np.stack([np.full(33, 4.3), np.linspace(0.6, 7.4, 33), np.full(33, 1.5)], axis=1)
car = MovingSource("car", Trajectory(times, path))
motion = MotionConfig(base=BakeConfig(fs=48_000), doppler=True)
out, info = render_moving(scene, car, listener, dry, motion)
```

- `Trajectory(times, positions)` holds positions at increasing times. It can
  also hold the front axis over time (`forwards=`).
- `render_scene_moving(scene, [(source, signal), ...], listener, motion)`
  renders several sources together.
- `MotionConfig(wave=True)` adds the wave band.

## Wave band only

```python
config = BakeConfig(fdtd=True, fdtd_fcross=300.0, wave_only=True, ambisonic_order=1)
```

The response is the wave band through the low-pass of the crossover. Bands
above the transition have no parameters.

## A vibrating surface

```python
source = SourceDef("panel", np.array([0.5, 1.1, 0.95]), surface="panel_material")
config = BakeConfig(fdtd=True, wave_only=True)
```

`surface` is the name of a material of the scene. Every surface with that
material vibrates. The position only marks the air of the room. This source
needs `wave_only=True` and a closed room.

## Hardware checks

```python
from auralis.engine import native_available, gpu_available
print(native_available(), gpu_available())
```

Set `AURALIS_GPU=0` to force the CPU, for example in tests.
