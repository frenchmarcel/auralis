# Outdoor ground

Outdoors, the ground is often the only surface near the sound path. The sound
reflected by the ground meets the direct sound and interferes with it. This is
the **ground effect**. It raises the low frequencies and digs a dip, often
between 200 Hz and 1 kHz over grass.

## Why a plane wave is not enough

A reflection factor for plane waves assumes a source far away. A real source is
a point. Near grazing incidence over a soft ground, the difference is large.
Over grass at 50 m, the plane-wave factor is 10 dB off at 50 Hz.

Auralis uses the **spherical-wave reflection coefficient** of Weyl and Van der
Pol. It depends on the ground, the angle of incidence and the path length. When
the ground is hard or the path is steep, it tends to the plane-wave factor.

## What the engine does

- Image sources use the spherical-wave coefficient when
  `BakeConfig.spherical_reflection` is on. By default (`None`) it turns itself
  on in an open scene, when test rays from the source escape the model, and
  off in a closed room. In a room it changes little: CR2 gives 1.10 JND with
  it, 1.09 without. The web app follows the scene's open flag.
- A material with a physical model uses the model itself. This covers the
  ground classes below, porous layers and measured tables. An α-only material
  uses its fitted admittance. Its phase is then an assumption.
- A higher-order reflection uses the full unfolded path length. This is exact
  when the other walls are rigid, for example a ground and a hard facade.
- These reflections and the direct sound get a long filter: 43 ms or more,
  until the reflection is within 0.1 % from 40 Hz to 1 kHz. The usual short
  event filters would smooth away the low-frequency detail.
- Behind a screen, edge diffraction also takes the paths that bounce on the
  ground before or after the edge. See {ref}`screens-on-a-ground`.

## Ground classes

The material library has the seven ground classes of Nord2000, in the category
**Outdoor ground**. Each one is an effective flow resistivity used with the
Delany-Bazley model on a semi-infinite ground.

| Class | Flow resistivity (kPa·s/m²) | Ground |
|---|---|---|
| A | 12.5 | Very soft (snow or moss-like) |
| B | 31.5 | Soft forest floor (short, dense, heather-like or thick moss) |
| C | 80 | Uncompacted, loose ground (turf, grass, loose soil) |
| D | 200 | Normal uncompacted ground (forest floors, pasture field) |
| E | 500 | Compacted field and gravel (compacted lawns, park area) |
| F | 2000 | Compacted dense ground (gravel road, parking lot) |
| G | 20000 | Hard surface (dense asphalt, concrete, water) |

Source: Jonasson and Storeheier, *Nord 2000. New Nordic prediction method for
rail traffic noise*, SP Rapport 2001:11, Table 3.1, after Nordtest NT ACOU 104.
The same table, and the Delany-Bazley coefficients used with it, are in the
Nord2000 model report (Plovsing and Kragh, DELTA AV 1849/00, Table 1 and
Appendix B).

### Two-parameter grounds

When a ground's parameters were measured with a two-parameter fit, give the
rate of porosity change with depth too (`alpha_e`, in 1/m). Auralis then uses
Attenborough's variable porosity model instead of Delany-Bazley. There is no
preset: use your own fitted values.

### Roughness

Terrain with random height variations makes part of each ground reflection
incoherent. Auralis follows Nord2000: the coherent part is multiplied by a
coherence coefficient Fr, which depends on k σr sin ψ. The share it loses,
(1 − Fr²) times the random-incidence reflection, becomes diffuse noise just
after the reflection. The roughness classes are those of Nord2000 (Table 2):

| Class | σr (m) | Height range |
|---|---|---|
| N | 0 | ±0.25 m |
| S | 0.25 | ±0.5 m |
| M | 0.5 | ±1 m |
| L | 1 | ±2 m |

Nord2000 notes that no outdoor measurement over rough terrain was available to
validate this part. Neither is one here.

In a script:

```python
from auralis.engine import ground, poromat

spec = ground.ground_spec("C")          # or a flow resistivity in Pa·s/m²
# ground.ground_spec("C", roughness="M", alpha_e=10.0) for a rough two-parameter ground
alpha = poromat.alpha_bands(spec)       # random-incidence α for the rays
# SceneData(..., absorption=alpha[None], mat_phys=[spec])
# BakeConfig(..., spherical_reflection=True)
```

## Checked against the exact solution

`tools/ground_effect_check.py` compares the engine with the analytic solution
for a point source over a flat ground, in third octaves from 50 Hz to 4 kHz.
The source is at 1 m and the receiver at 1.5 m, unless noted.

| Case | Deepest dip | Spherical (engine) | Plane wave (engine) |
|---|---|---|---|
| Class D, 10 m | −7.2 dB at 400 Hz | 0.1 dB | 2.3 dB |
| Class D, 25 m | −7.1 dB at 500 Hz | 0.0 dB | 4.0 dB |
| Class C, 50 m | −13.6 dB at 400 Hz | 0.1 dB | 9.9 dB |
| Class A, 200 m, source at 0.5 m | −31.6 dB at 200 Hz | 0.7 dB | 14.8 dB |
| Class G, 50 m | −9.9 dB at 2.5 kHz | 0.1 dB | 0.7 dB |

The last two columns give the largest deviation from the analytic solution.
A two-parameter ground (class D with αe = 5.5 m⁻¹) is within 0.1 dB too. A
rough ground (class D, roughness M, at 10 and 25 m) matches Nord2000's own
formula within 0.3 dB in octaves, as an average over eight noise draws.
Animated renders follow it too, Doppler path included: 0.1 dB over grass at
25 m, 0.4 dB over snow at 200 m.

(screens-on-a-ground)=
## Screens on a ground

Behind a screen, the sound reaches the receiver over the top edge by four
paths. The path can bounce on the ground before the edge, after it, both or
neither. This is the four-path model. With `spherical_reflection` on, Auralis
adds the three ground paths to the usual edge path. It uses any plane that the
image sources use, so a facade works too. Each ground bounce gets its
spherical-wave coefficient. Each path keeps the complex UTD coefficient, so the
four paths interfere with their true phases.

`tools/screen_ground_check.py` compares this with the wave solver. The screen
is 2 m high and 5 m from the source. The source is at 0.5 m and the receiver at
1.5 m, 10 m away. Levels are re free field in third octaves.

| Ground | Range | One edge path | Four paths |
|---|---|---|---|
| Class D | 100 to 400 Hz | up to 7.0 dB | up to 1.5 dB (mean 1.1 dB) |
| Class G | 100 to 400 Hz | up to 8.2 dB | up to 1.2 dB |

Above 400 Hz the wave solver is near the top of its band. A finer grid moves it
toward the four paths: at 630 Hz over class G, from −5.4 to −7.6 dB, with
−10.6 dB for the four paths.

## The wave solver outdoors

The wave solver handles the ground as a wall with the same model.

- **An infinite ground.** Give `BakeConfig.fdtd_box`, the corners of the wave
  domain, and a ground larger than the box. The ground then runs through the
  absorbing layer and has no edge to diffract. The cells of the layer that touch
  it get the layer's damping as a volume loss, which keeps it stable. The box
  needs a margin of about one wavelength at the lowest frequency, and a thicker
  absorbing layer helps (28 cells). With a 6 m margin, a rigid ground at 10 m is
  within 0.2 dB of the exact solution from 50 to 500 Hz.
- **Use the anisotropic conformal correction** (`fdtd_conformal_aniso`). The web
  app does it in open scenes, and a domain box turns it on. The default
  correction slows waves that graze a wall, and it is unstable where a ground
  runs into the absorbing layer.
- **Grazing paths need points per wavelength.** The cells next to a wall run at
  second order. At grazing incidence their reflection is too strong, and the
  error falls as dx². At 25 m (0.1 rad grazing) it is 9 % at 20 points per
  wavelength, 15 % at 18 and 25 % at 13. Raise `BakeConfig.fdtd_ppw` (9 by
  default, at 1.5 times the transition) for long grazing paths, or keep the
  transition low: the rays above it are exact.

## Limits

- The ground is flat and locally reacting.
- The air is still and uniform. There is no refraction by wind or temperature,
  and no turbulence. Both matter beyond about 100 m.
- Edge diffraction is first order: one edge per path, with the ground bounces
  around it.
