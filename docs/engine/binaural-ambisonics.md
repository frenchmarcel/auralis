# Binaural output and Ambisonics

## Heads and HRTFs

A head-related transfer function (HRTF) describes how the head, the ears and
the torso filter a sound from one direction. Auralis has two heads:

- **Model.** A spherical head without ears. It gives the time and level
  differences between the ears, but no colour from the ears.
- **FABIAN.** A measured head and torso from TU Berlin (CC BY 4.0). It is the
  head of the BRAS binaural measurements. Its data goes in `data_hrtf/`.

Each reflection gets the HRTF of its direction. Reflections are placed at an
exact fractional delay with a phase ramp, so high frequencies are not lost.

## Ambisonics

Ambisonics describes the sound field around the listener with spherical
harmonics. Auralis uses the ACN channel order and SN3D normalisation (the
AmbiX convention).

- The engine supports orders 1 to 5.
- The app uses order 2 (9 channels).
- The x axis points to the front, y to the left and z up.

The big advantage of Ambisonics is head tracking. To follow a head turn, the
field is rotated with a matrix on its harmonics. No acoustic computation runs
again.

## The binaural decoder

The decoder turns Ambisonics into two ear signals. It places virtual
loudspeakers on a sphere and gives each one the HRTF of its direction.

Two decoder designs are available:

- **Dual-band** (default). Below 700 Hz, mode matching keeps the exact phase.
  Above, the max-rE weights give a better energy distribution. A diffuse-field
  equaliser keeps the timbre.
- **MagLS** (option). Above N c / (2πa), where N is the order and a the head
  radius, a low order cannot follow the phase between the ears. MagLS then
  fits only the magnitude of the HRTFs. Below the cutoff it is the same as
  the dual-band decoder.

Mean level error with FABIAN at order 2, over 300 directions:

| Decoder | 2 kHz | 4 kHz | 8 kHz |
|---|---|---|---|
| Dual-band | 3.5 dB | 4.1 dB | 4.8 dB |
| MagLS | 1.0 dB | 1.3 dB | 2.5 dB |

Use `ambidecode.decoder_filters(order, fs, hrtf_set=..., method="magls")` to
choose MagLS in Python.

## Higher orders

Higher orders give a sharper image and keep the level differences between the
ears up to higher frequencies. With the model head at 45°, order 5 follows
the head within 0.3 dB up to 1.6 kHz. Order 2 is within 0.55 dB there.
