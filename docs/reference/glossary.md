# Glossary

**Ambisonics**
: A way to describe the sound field around a point with spherical harmonics.
  Auralis uses the ACN order and SN3D normalisation (AmbiX).

**Binaural**
: Two signals, one per ear, made for headphones. They contain the effect of
  the head and the ears.

**BRAS**
: Benchmark for Room Acoustical Simulation, a database of measurements from
  TU Berlin.

**Conformal correction**
: A change to the cells next to a wall, so that the grid has the true air
  volume and wall areas.

**CPML**
: Convolutional perfectly matched layer. An absorbing layer at the edge of the
  grid for open scenes.

**Crossover**
: A pair of filters, a low-pass and a high-pass, that join the two solvers at
  the transition frequency.

**Echogram**
: A histogram of the sound energy at the receiver over time, per octave band.

**FDTD**
: Finite-difference time-domain. The method of the wave solver.

**G (strength)**
: The level of a source in the room against the same source at 10 m in free
  field, in dB.

**HRTF**
: Head-related transfer function. How the head and the ears filter sound from
  one direction.

**Image source**
: The mirror image of a source in a wall. It gives one specular reflection.

**Impulse response (IR)**
: What a receiver records after a very short sound at the source. It holds the
  whole acoustic effect of the room between the two points.

**Insertion loss**
: The difference between a response with an object and the response without
  it, in free field.

**JND**
: Just noticeable difference. The smallest change a listener can hear.

**Schroeder frequency**
: The frequency above which the modes of a room overlap. Below it, modes are
  separate and a wave solver is needed.

**Scattering**
: The share of reflected energy that leaves in random directions.

**STI**
: Speech transmission index. A measure of speech intelligibility from 0 to 1.

**Transition frequency (fc)**
: The frequency where the wave band ends and the geometrical band starts.

**UTD**
: Uniform Theory of Diffraction. The model Auralis uses for sound that bends
  around edges.

**Voxel**
: A small cube of the grid. Each voxel is air or solid.
