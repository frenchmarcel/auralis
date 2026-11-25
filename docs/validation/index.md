# Validation against BRAS

Auralis is checked against **BRAS**, the Benchmark for Room Acoustical
Simulation from TU Berlin. BRAS has measured impulse responses, room models
and measured material data for a set of scenes. Its data is free (CC BY-SA
4.0).

## The rules

- **No fitting to the target.** Materials come from the BRAS material data
  and documents. They are never tuned to get closer to the measured responses.
- **Report the strict reading too.** When a metric can be read in a kind way
  and a strict way, both are published.
- **Keep negative results.** What does not work is written down in the
  project history.

## Rooms (CR scenes)

Each room result is the mean error over the source and receiver pairs and the
octave bands, in just noticeable differences (JND):

| Parameter | One JND |
|---|---|
| T30 | 5 % |
| C50 | 1 dB |
| Ts | 10 ms |
| Level per band | 1 dB |

A value of 1 JND means the error is just at the limit of what a listener can
hear. The parameters are computed on the omnidirectional response, like the
omni microphone of BRAS, in true octave bands.

| Scene | Room | Mean error | Share within 1 JND |
|---|---|---|---|
| CR2 | Seminar room | 1.09 JND | 67 % |
| CR3 | Chamber music hall | 1.51 JND | 40 % |
| CR4 | Auditorium | 1.86 JND | 37 % |
| CR1 | Coupled rooms, door at 30.4° | 3.01 JND | 26 % |
| CR1 | Coupled rooms, door at 4.1° | 3.99 JND | 18 % |

What is known about the remaining errors:

- **CR3 and CR4:** the simulated T30 is 10 to 16 % too long between 250 Hz
  and 2 kHz. The BRAS "fitted" materials were tuned with the Eyring formula,
  which assumes a diffuse field. With little scattering on the walls, the real
  decay depends on scattering that the data does not describe. Auralis does not
  add scattering to hide this.
- **All rooms, 2 kHz:** the level is 3 dB too high. The source data sheet and
  the direct sound of the room sessions do not agree at 2 kHz. Auralis keeps
  the data sheet.
- **CR1:** the coupling between the two rooms is right when the grid resolves
  it. The low end of the reverberant room is too long, for the same reason as
  CR3 and CR4.

## Scenes with one object (RS scenes)

These scenes have a single condition. The metric is the **insertion loss**:
the response with the object against the free-field response. It removes the
response and the directivity of the loudspeaker. Auralis reports the Pearson
correlation with the measurement from 160 Hz to 2 kHz, in third-octave bands
and raw.

| Scene | Content | Insertion loss, third octaves | Raw transfer function |
|---|---|---|---|
| RS7 | Rows of seats (seat dip) | 0.890 | 0.766 |
| RS6 | A finite body (diffraction) | 0.836 | 0.607 |

## Reflection from a plane (RS1)

RS1 compares a rigid floor with an absorbing floor and with a diffuser. The
metric is the correlation of the transfer function differences between two
states.

| Difference | Correlation |
|---|---|
| Rigid against absorbing | 0.658 |
| Rigid against diffuser | 0.511 |

These numbers come from the published grid settings. A finer grid gives
different values (0.54 to 0.68 for the absorber, 0.22 to 0.26 for the
diffuser, depending on the wall correction). The finer values are the more
reliable ones. The diffuser at high frequencies is still not well reproduced.

## Exact references for the geometrical solver

In a box, the geometrical solver is compared with image sources of all orders
and with radiosity. C50 and EDT match these exact references to the second
decimal.

## Reproduce the results

The BRAS data goes in `data_bras/`. Each scene has a README next to its
harness, with the exact commands:

| Scene | Harness |
|---|---|
| CR1 to CR4 | `BRAS/CR2/cr2_run.py <manifest>` |
| RS1 | `BRAS/RS1/rs1_reflection.py` |
| RS5, RS6, RS7 | `BRAS/RS5/run.py`, `BRAS/RS6/run.py`, `BRAS/RS7/run.py` |

For example:

```bash
python3 BRAS/CR2/cr2_run.py BRAS/CR2/cr2_dodecahedron_fitted31_dir_omni.json -o out_cr2
python3 BRAS/bras_jnd.py out_cr2/report.csv
```

The harnesses keep their intermediate results. Running the same command again
resumes where it stopped.
