# Sources and receivers

## Sources

A source has a position, a direction and a sound power.

| Setting | Meaning |
|---|---|
| Position | x, y, z in metres |
| Azimuth, elevation | The direction the source faces. Azimuth turns from +X towards +Y. Elevation goes from -90 (down) to +90 (up). |
| Directivity | How the level changes with the angle from the front axis |
| Gain | A level change in dB |
| EQ | Optional gains per octave band, within ±24 dB |

By default a source gives 94 dB at 1 m in every direction. Gains and the
directivity change this level. Some imported directivities carry their own
level at 1 m.

```{figure} ../images/sources-bras.jpg
:alt: The Sources tab of the BRAS chamber music hall, with two dodecahedron positions
:width: 100%

The BRAS chamber music hall (CR3). The dodecahedron of the measurements is an
imported directivity. It stands at the two measured positions, LS1 and LS2,
and faces up, as in the measurements.
```

### Directivities

| Kind | Where it comes from |
|---|---|
| Presets | Built into Auralis (omni, loudspeakers, voice and more) |
| Imported | Your files: JSON tables, CSV tables or AES69 SOFA files |
| Odeon (local) | The instrument, singer and talker patterns of a local Odeon installation |

Auralis treats a directivity as symmetric around the front axis. It uses the
level per octave band as a function of the angle from that axis.

In the wave band, a directive source is a monopole plus a dipole. This gives
the right pattern to first order with the right power in each octave band.
A source that is almost omnidirectional at low frequencies stays a single
monopole.

### Source EQ

The EQ is a filter on the source signal. It changes the SPL in each band. It
does not change G, which is normalised by the source's own power. T30, C50 and
the other parameters are computed on the filtered response.

### A vibrating layer

A source can be a **whole layer that vibrates**, instead of a point. Use it
for a panel, a flush-mounted loudspeaker or a source glued to a wall. The
whole layer moves with one normal velocity. Its total volume velocity is the
one of the standard point source.

This source exists only in the wave band. Use it with the Advanced mode and
the wave solver only, in a closed room. The point you place then only marks
the air of the room.

## Receivers

A receiver has a position and a direction. The direction is where the
listener looks. It matters for binaural results and for J_LF.

### Heads

The settings choose the head used for binaural output:

- **Model**: a simple spherical head without ears.
- **FABIAN**: a measured head and torso from TU Berlin (CC BY 4.0). It is the
  head used for the BRAS binaural measurements.

## Receiver grids

A receiver grid gives maps of parameters over an area.

- A **surface grid** puts points on a layer, for example the audience, at a
  height above it.
- A **plane grid** is a free rectangle, for example a desk plane.

The spacing goes from 0.1 m to 20 m, with at most 2500 points per grid. Grids
use the geometrical solver. When the wave solver runs, G and SPL in the low
bands come from the wave band at every point.

## Trajectories

Sources and receivers can move. Draw a path in the 3D view or type its points
with their times. A path has an orientation mode. The object can face where it
goes, keep a fixed direction, or look at a target.

Trajectories are used by animated renders. See {doc}`listening`.

## Groups and batch tools

You can group sources or receivers, give a group a colour, and select it at
once. Batch tools help with large setups:

- **Multiply**: copy points with a step in x, y or z.
- **Offset from a surface**: click on surfaces to place points at a distance
  from them.
- **Import**: read positions from a table.
- **Ensembles**: place an orchestra or a band, one source per player, each with
  the directivity of its instrument.
