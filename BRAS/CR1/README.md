# CR1: coupled rooms (laboratory and reverberation chamber)

A reverberation chamber (121 m³, painted concrete, alpha 0.01-0.02) opens through a door onto a
laboratory (102 m³). RWTH Aachen measured it with the door open by 4.1° (`DoorAngle1`) and by
30.4° (`DoorAngle3`). Dodecahedron source at LS1 (chamber) and LS2 (laboratory), microphones
MP3 and MP4 in the laboratory. RIRs of 4.5 s, air 18.2 °C and 47.6 %.

Same metric and JND as CR2 (`../CR2/README.md`), on the omni response. Data: BRAS v3 (TU Berlin,
CC BY-SA 4.0, `../LICENSE.md`), unpacked into `data_bras/1 Scene descriptions/`.

## Results

`reference_run/`: engine defaults, automatic crossover 352 Hz. BRAS fitted the absorption of
each room separately, with Eyring.

| Door | Materials | Mean | ≤ 1 JND | T30 | C50 | Ts | Level |
|---|---|---|---|---|---|---|---|
| 30.4° | fitted | **3.01** | **26 %** | 4.44 | 2.39 | 4.01 | 1.18 |
| 4.1° | fitted | **3.99** | **18 %** | 6.81 | 2.39 | 5.19 | 1.56 |
| 30.4° | initial (blind) | 3.91 | 26 % | 7.49 | 2.23 | 3.73 | 2.21 |
| 4.1° | initial (blind) | 4.46 | 23 % | 5.41 | 3.28 | 6.93 | 2.21 |

## What remains, and why

- **The coupling is right when the grid resolves it.** At 30.4°, LS2 hears the chamber's
  energy come back (a double-slope decay), with T30 3.08 s against 3.19 s measured at 1 kHz.
- **The chamber's bass decays too slowly.** The wave band uses Eyring-fitted alphas in a field
  that is not diffuse (T30 at 125 Hz: 13.3 JND at 30.4°). This is the CR2 low-end mechanism,
  stronger here.
- **At 4.1°, the door leaves a slot of about 4 cm**, thinner than a grid cell. It is a sub-cell
  opening (`auralis/engine/subcell.py`). Its conductance is uncertain by 10-15 %, which weighs
  about 0.05 JND. Nothing is tuned on it.
- **The laboratory's early decay is too slow** (EDT +16 to +40 % at 500 Hz-2 kHz, C50 2-3 dB
  low). Its absorption is concentrated and its scattering is low, as in CR3 and CR4. We do not
  add scattering to correct it.

## Geometry

`relabel_cr1_obj.py` gives concrete to a DoorAngle1 wall with no material (17.30 m²). It
removes the source markers and the loudspeaker bodies. The areas then match the official table
(`mat_CR1.txt`): absorber 12.70, concrete 150.41, painted concrete 179.58, tables and equipment
36.15 m².

The ray bounce limit now follows the IR length (`pipeline.effective_max_bounces`). The old fixed
limit of 200 reflections cut the chamber's tail (T30 2.8 s against 4.3 s measured).

## Reproduce

`<Geometry>` is the `Geometry/` folder of the CR1 scene in `data_bras/1 Scene descriptions/`.
The first command needs a Python environment with the OpenSKP reader (MIT).

```
python -I BRAS/skp_to_obj.py "<Geometry>/CR1_RIR_DoorAngle1_Dodecahedron.skp" full1.obj
python3 BRAS/CR1/relabel_cr1_obj.py full1.obj BRAS/CR1/CR1_RIR_DoorAngle1.obj
python3 BRAS/bras_room_manifest.py CR1 --variant DoorAngle1 --obj CR1_RIR_DoorAngle1.obj \
    -o BRAS/CR1/cr1_door1_dodecahedron_fitted31_dir.json
python3 BRAS/CR2/cr2_run.py BRAS/CR1/cr1_door1_dodecahedron_fitted31_dir.json -o out
python3 BRAS/bras_jnd.py out/report.csv
```

Door at 30.4°: use `DoorAngle3`. Blind materials: add `--estimates initial` to the manifest
command.

## Files

- `CR1_RIR_DoorAngle1.obj`, `CR1_RIR_DoorAngle3.obj`: models. `relabel_cr1_obj.py`: their fix.
- `cr1_door1_*.json` (4.1°), `cr1_door3_*.json` (30.4°): manifests, fitted and initial.
- `reference_run/`: the published runs. Earlier runs: `reference_run_2026-11-10_marker/` (a
  source marker left around LS1), `reference_run_2026-11-10_r25/` (4.1°, before a cache fix).
