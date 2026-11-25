# Quick start

This page shows two ways to get a first result. The first one uses Python. The
second one uses the web app.

## A first room in Python

Save this script as `first_room.py` in the repository root and run it.

```python
import numpy as np
from auralis.engine import (BakeConfig, ReceiverDef, SourceDef, bake, format_report,
                            make_shoebox, write_wav)

# a 7 x 5 x 3 m room with plaster walls, a parquet floor and an absorbing ceiling
scene = make_shoebox((7.0, 5.0, 3.0), wall_material="plaster",
                     floor_material="parquet", ceiling_material="acoustic_tile")

source = SourceDef("S1", np.array([1.5, 2.0, 1.5]))
listener = ReceiverDef("R1", np.array([5.0, 3.0, 1.2]),
                       forward=np.array([-1.0, 0.0, 0.0]))

config = BakeConfig(n_rays=100_000, fdtd=True, fdtd_fcross=250.0)
result = bake(scene, source, listener, config)

print(format_report(result.metrics))
write_wav("room_ir.wav", result.ir, result.fs)
```

```bash
python3 first_room.py
```

What happens:

1. `make_shoebox` builds a box room with three materials.
2. `SourceDef` and `ReceiverDef` place a source and a listener. The listener
   looks along `forward`.
3. `BakeConfig` holds the settings. Here the wave solver runs below 250 Hz.
4. `bake` runs both solvers and joins them.
5. `format_report` prints the reverberation time, clarity and levels per
   octave band.
6. `room_ir.wav` is the binaural impulse response. Convolve it with a dry
   recording to hear the room.

The run takes a few seconds with the native libraries. It prints a table like
this one (values per octave band):

```text
Band       RT       EDT     C50     C80     D50     Ts
125        0.53(T20)    0.53   4.45   8.71  0.736  0.047
250        0.49(T20)    0.49   5.16   9.52  0.766  0.044
500        0.51(T20)    0.49   5.39   9.59  0.776  0.042
...
```

## Listen to a convolved signal

The repository has a ready demo. It builds a small club, bakes binaural
responses and convolves test signals with them:

```bash
python3 examples/standalone_demo.py demo_out
```

Open the WAV files in `demo_out/` with headphones.

A second demo renders a moving source with the Doppler effect:

```bash
python3 examples/demo_motion.py
```

## A first project in the web app

```bash
bash webapp/start.sh
```

Your browser opens <http://127.0.0.1:8765>. The next page,
{doc}`web-app`, walks through a first project.
