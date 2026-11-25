# Wave-based solver (FDTD)

The wave solver computes the sound pressure in the air at many points in
space and many times. It uses the finite-difference time-domain method
(FDTD).

## The grid

The air is cut into small cubes (cells). Each cell holds one pressure value.

- The cell size is dx = c / (13.5 fc), where c is the speed of sound and fc
  the transition frequency. At fc = 500 Hz, dx is about 5 cm.
- The time step follows from dx and the stability limit of the scheme.
- The interior uses a fourth-order stencil. It has less numerical dispersion
  than the classic second-order scheme. Cells next to a wall use a
  second-order stencil.

### From triangles to cells

1. **Voxelisation.** Each cell is air or solid. A solid cell next to the air
   takes the material of the nearest triangle.
2. **Air of the source.** Auralis fills the air from the source position.
   Cells that the source cannot reach (inside walls, closed cavities) are left
   out.
3. **Openings smaller than a cell.** In a closed room, a gap thinner than the
   grid (a door left ajar) would close in the voxels. Auralis finds such
   pockets and joins them to the air through links. Each link carries the
   conductance of the real gap, computed on a fine local grid.

### Conformal boundaries

A plain voxel grid turns every wall into a staircase. Its walls sit up to one
cell away from the real ones, and its areas are wrong for slanted surfaces.

The **conformal correction** fixes this. Cells cut by a wall get their true
air volume and their true face areas. Auralis has two forms:

- The **default form** gives the cut cells their true volume. It is the
  default for closed rooms.
- The **anisotropic form** also gives the cut cells their true conductance
  along the wall. Room modes are then exact in an aligned box. It is the
  default for open scenes with sharp edges, where it gives better results.

## Walls

Each material becomes a **wall filter**. The filter gives the reflection of
the material at every frequency, with its phase.

- The filter is fitted on the discrete grid itself, not on the continuous
  theory. This matters because the grid changes how a wall reflects.
- It is checked for passivity. A wall must not create energy. If the fit is
  not passive, Auralis corrects it and says so.
- Resonant panels have their own exact model (mass, spring and damping).

## Sources in the grid

The source is a soft source. It adds a short pulse to the pressure of the
cells around its position.

- A point source spreads over the eight cells around its exact position.
- The injection is divided by the volume of each cell. A cell cut by a wall
  has a larger volume. Without this, a source close to a wall would be too
  loud.
- A **directive** source is a monopole plus a dipole. Auralis runs the grid
  twice and combines the two runs per frequency. This gives the pattern
  a + (1 − a) cos θ with the right power in each octave band.
- A **vibrating layer** injects on every air cell next to that layer, in
  proportion to the area each cell shares with it.

## Receivers in the grid

A receiver reads the pressure at its exact position, by interpolation between
the eight cells around it. For binaural output there are three choices:

- **Two ears** (default). Two probes, spaced so that their coherence in a
  diffuse field matches the chosen head.
- **Head, first order.** Seven probes around the head give first-order
  Ambisonics. The HRTF set then decodes them.
- **Head, second order.** Nineteen probes give second-order Ambisonics.

## Calibration

A run on an empty grid gives the response of the grid to the pulse in free
field. A Wiener filter turns this into a clean impulse with a 1/d level and a
d/c delay. Every run uses this calibration. So the wave band and the
geometrical band have the same level convention.

## Open scenes

In an open scene, the grid ends with absorbing layers (CPML). Their absorption
is good but not perfect. To remove the small echo from the edges, Auralis runs
the same source a second time in an empty box. The difference is the field
scattered by the scene. The ideal direct sound is then added back in closed
form.

## Closed rooms

A closed room with a pulse source slowly fills with pressure below its first
mode. A real loudspeaker does not do this. Auralis applies a fourth-order
high-pass filter at 16 Hz to the output of closed rooms.

## Accuracy notes

- The wave band is accurate up to about 1.5 times fc. Above, the grid
  disperses the waves. The crossover keeps only the part below fc.
- A finer grid is more accurate but much slower. Halving dx multiplies the
  number of cells by 8 and the number of time steps by 2.
