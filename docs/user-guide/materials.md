# Materials

A material tells the solvers how a surface absorbs and scatters sound.

## What a material holds

| Value | Meaning |
|---|---|
| Absorption α | The share of energy absorbed, per octave band from 125 Hz to 8 kHz. From 0 to 0.95. |
| α at 63 Hz | Optional. Used only by the wave solver. |
| Scattering s | The share of reflected energy sent in random directions. One value, or one value per octave band. |
| Physical model | Optional. A porous layer, a perforated panel or a measured complex impedance. |

Absorption values above 0.95 are clipped to 0.95, the physical limit for
random incidence. The app shows it when this happens.

```{figure} ../images/materials-tab.jpg
:alt: The Materials tab of the BRAS seminar room with five layers
:width: 100%

The Materials tab of the seminar room. Each layer has a material and a
scattering value. Below the list is a statistical estimate of the
reverberation time (Sabine and Eyring).
```

To change a material, select one or more layers and click **Assign**. The
material library opens.

## Libraries

| Library | Content | Licence |
|---|---|---|
| Auralis presets | Common materials (concrete, plaster, glass, carpet, wood and more), and the seven Nord2000 ground classes (see {doc}`../engine/outdoor-ground`) | MIT |
| pyroomacoustics | Tables from the literature (Vorländer) | MIT |
| Odeon (local) | The Odeon library, with 63 Hz values | Local use with an Odeon licence only |
| My materials | Your own materials | Yours |

Some libraries only show when their data is on disk.

```{figure} ../images/material-library.jpg
:alt: The material library with the measured plaster of the BRAS seminar room
:width: 100%

The material library, with the measured plaster of the BRAS seminar room. The
panel shows the absorption per octave band, the scattering, the source of the
data and its licence.
```

## Your own materials

You can create a material in several ways:

- **From absorption values.** Type seven α values, and an optional 63 Hz value.
- **Porous absorber.** Give the flow resistivity, the thickness and the air
  gap behind it. Auralis uses the Miki model. You can also give a glass wool or
  rock wool density instead of the flow resistivity.
- **Perforated, slotted or micro-perforated panel.** Give the hole size or the
  slot width, the panel thickness, the open area and the cavity depth.
- **Measured complex data.** Import a table of normal-incidence impedance or
  reflection factor over frequency.

Your materials have a stable id. Renaming one does not change the simulations
that use it. You can duplicate any material, including library ones, and edit
the copy.

## How the two solvers use a material

The two solvers do not need the same data.

**The geometrical solver** uses α and s per octave band. With a physical model
or a measured table, it also uses the complex reflection factor on its image
sources. This gives the right phase for early reflections.

**The wave solver** works in the time domain. It needs a wall filter that
behaves like the material at every frequency, with the right phase.

- If the material has a physical model, Auralis uses it directly.
- If the material has only α values, Auralis looks for a physical model with
  the same α. It tries a porous layer, a resistive screen and a panel. The
  model must stay inside physical limits. If no model fits, Auralis keeps a plain
  fit of α. Hard surfaces (α at most 0.2) keep the plain fit.

In both cases the filter is fitted on the grid itself and checked for
passivity. A passive wall can absorb energy but never create it. Auralis
repairs a filter that is not passive and prints a message when it does.

An α value alone says nothing about the phase. When phase matters (a resonant
panel, a thin absorber), a physical model gives a better wave band.

## Scattering

Scattering sends part of the reflected energy in random directions. It matters
for the late part of the response.

Auralis computes the scattering due to the **size** of each surface by
default. A small surface scatters more at low frequencies than a large one. So
the scattering value of a material should only describe its roughness. Use
about 0.05 for a smooth wall and higher values for rough or structured
surfaces.

## Air

Air absorbs high frequencies over long distances. Auralis computes it from
the temperature and the humidity (ISO 9613-1). The speed of sound also comes
from the temperature, the humidity and the pressure (Cramer formula). Set them
in the simulation settings.
