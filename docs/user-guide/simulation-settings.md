# Simulation settings

## Modes

| Mode | What it does |
|---|---|
| Survey | Geometrical solver only (image sources and rays). Fast. The response length comes from the estimated reverberation time. |
| Default | Hybrid. The wave solver runs below a proposed transition frequency, the geometrical solver above. Needs a closed model or an open scene. |
| Advanced | Full control over every setting below. |

Start with Survey to check your model and your materials. Then use Default or
Advanced for the final results.

```{figure} ../images/settings-tab.jpg
:alt: The Settings tab with the Survey, Default and Advanced modes
:width: 100%

The Settings tab of the BRAS seminar room. The engine suggests a transition
frequency of 451 Hz for this model. The air (here, as measured by BRAS), the
listener head and the speech level for the STI are set here too.
```

## Main settings

| Setting | Default | Meaning |
|---|---|---|
| Transition frequency | automatic | Below it, the wave solver. Above it, the geometrical solver. |
| IR length | automatic | Length of the impulse responses, in seconds |
| Stop decay | automatic | Or ask for a response long enough for every band to decay by this many dB |
| Solvers | both | Both (hybrid), wave solver only, or geometrical only |
| Image source order | 2 | Reflection order of the image sources |
| Rays | 200 000 | Number of rays per source |
| Diffraction | on | First-order edge diffraction (UTD) |
| Conformal correction | on | True wall positions in the wave grid (see {doc}`../engine/wave-solver`) |
| Open scene | off | Declare an object in free field, a street or an outdoor space |
| Temperature, humidity | 20 °C, 50 % | Air absorption and speed of sound |
| Head | Model | The head for binaural output (Model or FABIAN) |
| Speech effort, background noise | normal, none | Used for the STI |
| Size-dependent scattering | on | See {doc}`materials` |

## The transition frequency

In automatic mode, Auralis takes the highest transition frequency that a
compute budget allows, up to 1 kHz. The budget keeps the run time and the
memory reasonable. The transition never goes below the Schroeder frequency of
the room, where the room modes start to overlap.

The grid step of the wave solver is about c / (13.5 fc). Twice the transition
frequency means eight times more cells and twice as many time steps. The
cost grows fast.

## Wave solver only

In Advanced mode you can ask for the wave solver alone. The result is the wave
band through the low-pass of the crossover. This is useful for low-frequency
studies and modal analysis.

- Parameters exist only in the octave bands that lie fully below the
  transition frequency. The other bands are empty.
- There is no STI, no J_LF and no overall level, since they need every band.
- The impulse response stops at the transition frequency. It sounds muffled.
- Receiver grids are not available in this mode.

## Run time

Before a run, the app shows an estimate. It is based on measured costs of the
solvers on your machine. Only one run uses the GPU at a time. Other runs wait
in a queue. A run that stops in the middle resumes from the sources it already
finished.

Most of the time of a source and receiver pair is CPU work: the synthesis of
its responses. So the app computes several pairs at once, in separate
processes. The wave solver of the next source runs on the GPU meanwhile. The
results are the same as one pair at a time. A concert with 47 instruments and
2 seats runs about 3 times faster this way. See `AURALIS_PAIR_WORKERS` in
{doc}`../reference/environment`.

## Before a run

The app checks the simulation. It blocks the run if something is wrong, for
example:

- a layer without material,
- no source or no receiver,
- a point inside a wall or too close to a surface,
- the wave solver on a model that is not closed and not declared open.

It also gives warnings that do not block, for example very rigid materials in
the wave band.
