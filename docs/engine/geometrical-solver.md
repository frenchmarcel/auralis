# Geometrical solver

The geometrical solver follows sound as rays and image sources. It covers the
whole audible range. In a hybrid run, its result is used above the transition
frequency.

## Image sources

An image source is the mirror image of the source in a wall. Each one gives
an exact specular reflection with its arrival time and direction.

- The default order is 2. Higher orders cost much more and gave no gain on
  the validation rooms.
- Each image source carries only the **specular part** of the reflection.
  For each wall it hits, it keeps (1 − s) of the energy, where s is the
  scattering. The rays carry the rest.
- With a physical or measured material, the reflection uses the complex
  reflection factor at the true angle of incidence. This gives the right phase
  for the early reflections.

## Edge diffraction

Sound bends around edges. Auralis computes first-order diffraction with the
Uniform Theory of Diffraction (UTD). It matters behind obstacles and near the
edges of balconies, screens and buildings.

## Stochastic ray tracing

Many rays leave the source in random directions. At each wall a ray loses the
absorbed energy and splits into a specular part and a scattered part. Each
time a ray passes near a receiver, it leaves energy in a histogram over time,
per octave band and with its direction. This histogram is the **echogram**.

- **Size-dependent scattering.** A small surface scatters more than a large
  one, mostly at low frequencies. Auralis adds this effect at each reflection
  from the size of the surface hit, the angle of incidence and the distances.
  The material's own scattering then only describes its roughness.
- **Air absorption** follows ISO 9613-1, from the temperature and the
  humidity.
- **Number of bounces.** Rays keep going as long as the response needs them.
  Long reverberation therefore keeps its tail.

## Synthesis

The solver gives two things: a list of **events** (direct sound, image
sources, diffraction paths) and the **echogram**. The synthesis turns them
into impulse responses.

**Events.** Each event is placed at its exact arrival time, with a
fractional-delay filter. It gets the HRTF of its direction for binaural
output, and its spherical harmonics for Ambisonics output.

**Late part.** For each octave band, Auralis makes a noise whose energy
follows the echogram, millisecond by millisecond. With a measured head, the
late part takes the colour of the head in a diffuse field and the interaural
coherence of the HRTF set. Without this, the reverberation sounds dull.

**Outputs.**

- Binaural, for headphones (two channels).
- Stereo, for a pair of loudspeakers.
- Ambisonics, ACN order with SN3D normalisation, from order 1 to 5.

## Checked against exact references

The geometrical solver is compared with two exact references in a box:
image sources of all orders, and radiosity. C50 and EDT match them to the
second decimal. See {doc}`../validation/index`.
