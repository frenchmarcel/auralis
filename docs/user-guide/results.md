# Results

After a run, each source and receiver pair has its impulse responses and its
parameters.

```{figure} ../images/results-model.jpg
:alt: The results page with T30 per octave band, single numbers and the 3D model
:width: 100%

The results of the BRAS seminar room. On the left are T30 per octave band for
one source and receiver pair, the single numbers and the decay curves. On the
right are the 3D model, where you click a receiver, and the details of the run.
```

The buttons at the top switch between the views: Model, Report,
Reflections, Grid maps, Modal maps, Wave field and ISO 3382-3. Add a
comparison to show several pairs or simulations in the same charts.

## Parameters

Auralis computes the ISO 3382-1 parameters on the **omnidirectional** impulse
response, as a measurement with an omni microphone would. The bands are true
octave bands from 125 Hz to 8 kHz.

| Parameter | Unit | Meaning |
|---|---|---|
| T20, T30 | s | Reverberation time from the decay between -5 and -25 dB (T20) or -5 and -35 dB (T30) |
| EDT | s | Early decay time, from the first 10 dB of the decay |
| C50, C80 | dB | Clarity: early energy (first 50 or 80 ms) against late energy |
| D50 | 0 to 1 | Definition: share of the energy in the first 50 ms |
| Ts | ms | Centre time of the energy |
| G | dB | Strength: level against the same source at 10 m in free field |
| SPL | dB | Sound pressure level for the source's power and gains |
| J_LF | 0 to 1 | Early lateral energy fraction |
| L_J | dB | Late lateral sound level |
| STI | 0 to 1 | Speech transmission index (IEC 60268-16) |

Some details:

- A value can be empty when it is not defined, for example a T30 when the
  response does not decay by 35 dB.
- **STI** uses the full response, the speech level at the listener and the
  background noise of the settings. The ratings follow the standard: below
  0.30 bad, 0.30 to 0.45 poor, 0.45 to 0.60 fair, 0.60 to 0.75 good, above
  0.75 excellent.
- **J_LF and L_J** come from the directions of the geometrical solver's
  energy.
- **G and SPL** in the low bands come from the wave band when it ran. Above,
  they come from the geometrical solver.
- With the wave solver, the **63 Hz band** is also given, from the wave band
  alone.
- For open-plan offices, the app gives the ISO 3382-3 single numbers (spatial
  decay of speech, distraction distance) along a line of receivers.

## Results per solver

With the hybrid mode, you can also look at each solver alone:

- **Hybrid**: the joined result. This is the main result.
- **Geometrical**: the geometrical solver over the full band.
- **Wave-based**: the wave band alone. Bands above the transition are empty.

This helps to see what each solver brings.

## Curves

For each pair you can see:

- the impulse response (binaural or omni),
- the energy decay curve per band,
- the frequency response, smoothed or raw.

## Maps

Receiver grids give maps of every parameter. The app shows statistics and
lets you set target ranges. Points that fall inside a wall or too close to a
surface are left out and listed.

```{figure} ../images/results-grid-map.jpg
:alt: A map of C80 at 1 kHz over a listening plane of the seminar room
:width: 100%

C80 at 1 kHz over a listening plane 1.2 m above the floor of the seminar room,
averaged over its two sources. The panel on the
right gives the statistics, a target and the distribution over the area.
```

## Modal maps and wave field

With the wave solver, the app can show:

- the pressure level over a plane at one frequency (a modal map),
- an animation of the wave field in time.

## Coherent sum of sources

You can add several sources together with gains, delays and polarities. The
app sums their responses as signals, so interference is kept. It then
computes the parameters of the sum.

## Reflection tracking

The Reflections view lists the early arrivals between a source and a
receiver: the direct sound, the specular reflections (image sources) and the
diffracted paths. You can filter them by order, type, level and time.

```{figure} ../images/results-reflections.jpg
:alt: Reflection tracking with a time chart, a polar plot and paths in 3D
:width: 100%

Reflection tracking in the seminar room. The chart shows each arrival by time
and level. The polar plot shows where it comes from. The 3D view draws the
paths, coloured by reflection order.
```

## Exports

| Export | Content |
|---|---|
| Workbook (`.xlsx`) | Settings, parameters per pair and band, single numbers, points |
| Impulse responses (`.zip`) | Per pair: binaural (2 channels), omni (1 channel) and Ambisonics (9 channels, ACN/SN3D), 32-bit float WAV, one shared scale |
| Graphs | Data, PNG images and clipboard copies from the interface |

All responses of one simulation share the same scale. Relative levels between
pairs are kept.

The Report view gathers the results on one page that you can print or save as
a PDF.

```{figure} ../images/results-report.png
:alt: The printable report with single numbers and T30 per octave band
:width: 100%

The report of the seminar room. The STI is rated with the scale of IEC
60268-16. The hatched 63 Hz bar comes from the wave band alone.
```
