# Engine settings (BakeConfig)

`BakeConfig` holds every setting of a run. The web app fills it from the
simulation settings. In a script you set it yourself. See
{doc}`../scripting/python-api`.

## Output

| Field | Default | Meaning |
|---|---|---|
| `fs` | 48000 | Sample rate in Hz |
| `max_time` | None | Response length in seconds. None means automatic, from an Eyring estimate. |
| `output_layout` | `"binaural"` | `"binaural"` for headphones, `"stereo"` for two loudspeakers |
| `hrtf_path` | `""` | A measured head as a `.npz` file. Empty means the model head. |
| `stereo_base_deg` | 30.0 | Half-angle of the loudspeaker pair in stereo mode |
| `ambisonic_order` | 0 | 0 means no Ambisonics. 1 to 5 add an ACN/SN3D response. |
| `normalize` | True | Scale the binaural response to a peak of about 0 dB. The gain is logged. |

## Geometrical solver

| Field | Default | Meaning |
|---|---|---|
| `n_rays` | 200000 | Rays per source |
| `max_bounces` | 200 | Reflections per ray. It grows with the response length. |
| `ism_order` | 2 | Order of the image sources |
| `complex_reflection` | True | Use the complex reflection factor on image sources |
| `spherical_reflection` | None | Spherical-wave reflections on image sources, and ground paths around diffracting edges, for the ground effect and screens. None: on in an open scene (test rays escape), off in a closed room. |
| `diffraction` | True | First-order edge diffraction (UTD) |
| `max_edges` | 64 | Most edges used for diffraction |
| `max_planes` | 48 | Most planes used for image sources |
| `rbs_scattering` | True | Scattering from the size of each surface |
| `bin_dt` | 0.001 | Time step of the echogram in seconds |
| `energy_cap` | 1e-6 | A ray stops below this share of its starting energy |
| `rcv_radius` | 0.3 | Radius of the receiver sphere for rays, in metres |
| `seed` | 1234 | Random seed of the rays |

## Wave solver

| Field | Default | Meaning |
|---|---|---|
| `fdtd` | False | Run the wave solver |
| `fdtd_fcross` | 0.0 | Transition frequency in Hz. 0 means automatic. |
| `fdtd_fcross_max` | 1000.0 | Highest automatic transition frequency |
| `fdtd_budget_gcell` | 40.0 | Cell budget of the automatic choice, in billions of cell updates |
| `fdtd_max_cells` | 12000000 | Largest grid allowed |
| `fdtd_order` | 4 | Order of the interior stencil (2 or 4) |
| `fdtd_absorber` | `"cpml"` | Absorbing boundary of open scenes (`"cpml"` or `"sponge"`) |
| `fdtd_conformal` | True | Conformal wall correction |
| `fdtd_conformal_aniso` | False | The anisotropic form of the correction |
| `fdtd_source_directivity` | True | Directive sources in the wave band (monopole and dipole) |
| `fdtd_align` | False | Turn the grid to follow the scene |
| `fdtd_box` | None | Corners of the wave domain. The geometry runs on through the absorbing layer: a large ground acts as an infinite one. |
| `fdtd_ppw` | 9.0 | Grid points per wavelength at 1.5 times the transition. Long grazing ground paths need more. |
| `infer_materials` | True | Give α-only materials a physical model for the wave band |
| `wave_ear_half` | 0.0 | Half spacing of the two ear probes. 0 means from the head. |
| `wave_head` | `"ears"` | `"ears"`, `"hrtf"` (first order) or `"hrtf2"` (second order) |
| `wave_only` | False | Output the wave band alone |

## Air

| Field | Default | Meaning |
|---|---|---|
| `temperature_c` | 20.0 | Air temperature in °C |
| `humidity_percent` | 50.0 | Relative humidity in % |
| `pressure_kpa` | 101.325 | Air pressure in kPa |
| `air_cramer` | True | Speed of sound with humidity (Cramer). False uses the dry formula. |

## Hardware

| Field | Default | Meaning |
|---|---|---|
| `engine` | `"auto"` | `"auto"`, `"gpu"`, `"native"` (CPU) or `"python"` |
| `n_threads` | 0 | CPU threads. 0 means all. |
