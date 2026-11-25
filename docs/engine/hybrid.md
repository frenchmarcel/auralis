# Joining the two bands

## The crossover

The wave band and the geometrical band meet at the transition frequency fc.

- The wave band goes through a low-pass filter at fc.
- The geometrical band goes through a high-pass filter at fc.
- The two filters have zero phase and add up to exactly one. Where both bands
  agree, the sum is unchanged.

Both bands use the same level convention. A direct sound at distance d has an
amplitude of 1/d and arrives after d/c. So they join without a level jump.

For binaural output, the wave band brings its own two ear signals. For
Ambisonics output, the wave band feeds the omnidirectional channel W.

## Choosing fc

In automatic mode, Auralis takes the highest transition frequency that the
compute budget allows, up to 1 kHz. The budget is 40 billion cell updates by
default. The transition never goes below the Schroeder frequency of the room. Below that frequency the
modes are separate, and only a wave solver gets them right.

You can also set fc yourself in Advanced mode.

## Levels: G and SPL

G and SPL need the energy of the response in each octave band. In a hybrid
run they come from two places:

- In the bands the wave band reaches (centre up to 2 fc), the energy has two
  parts. The first is the low-passed wave band. The second is the part of the
  echogram that the high-pass lets through.
- Above, the energy comes from the echogram.

This matters. Far from the walls, low-frequency energy in a real room is lower
than the ray model says (the Waterhouse effect). The wave band gets this
right. In a test box, the geometrical solver alone was 1.5 dB too high at
125 Hz, and the hybrid matched theory within 0.4 dB.

## Parameters

The parameters use the omnidirectional response (the W channel). They are
computed in true octave bands, with the same filters for simulations and for
measurements. See {doc}`../user-guide/results`.

## Results per solver

Auralis keeps the omni response of each band. You can read the parameters of
the hybrid, of the geometrical band over the full range, or of the wave band
alone.
