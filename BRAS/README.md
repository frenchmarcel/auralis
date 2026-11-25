# BRAS (Benchmark for Room Acoustical Simulation)

This folder is the validation bench against BRAS, the measurement database
of TU Berlin. There is one folder per scene. The RS folders and CR2 follow
the Treble validation pages. Each folder has a `README.md` with the exact
commands, the experiment script, and the figures in `figures/`.

- Shared harnesses are at the root of this folder: `bras_simple.py`
  (fundamental scenes), `bras_bench.py` (rooms) and `manifests/`.
- The checks are in the repository's `tests/`.
- The repository's `tools/` holds engine utilities only.

## Results

- Rooms (CR): mean error in JND over T30, C50, Ts and level, all pairs and
  octaves 125 Hz-8 kHz, on the omni response. The share is the part of
  values within 1 JND. One JND is 5 % for T30, 1 dB for C50 and level,
  10 ms for Ts.
- Other scenes: Pearson correlation with the measurement. IL is the
  insertion loss, TF the transfer function.

| Scene | Folder | Result | Other readings |
|---|---|---|---|
| Reflection from an infinite plane | `RS1/` | 3 conditions. Absorber 0.658 (Treble: 0.63), diffuser 0.511 (published grid, not converged) | finer grid: absorber 0.54-0.68, diffuser 0.22-0.26 |
| Small room (seminar room) | `CR2/` | 10 pairs. 1.09 JND, 67 % | ears, older default: 1.15 JND, 61 % |
| Medium room (chamber music hall) | `CR3/` | 1.51 JND, 40 % | ears, older default: 1.85 JND, 38 %. Blind materials: 2.87 JND, 26 % (omni), 3.08 JND, 30 % (ears) |
| Large room (auditorium) | `CR4/` | 1.86 JND, 37 % | ears, older default: 2.06 JND, 26 %. Blind materials: 2.31 JND, 25 % (omni), 2.52 JND, 20 % (ears) |
| Coupled rooms | `CR1/` | door at 30.4°: 3.01 JND, 26 %. Door at 4.1°: 3.99 JND, 18 % | blind materials: 3.91 JND, 26 % and 4.46 JND, 23 % |
| Diffraction (infinite edge) | `RS5/` | hybrid FDTD + UTD: IL 0.71 | absolute TF 0.52 |
| Diffraction (finite object) | `RS6/` | FDTD fc=2000, anisotropic conformal: IL 0.84 in 1/3 oct | narrowband TF 0.61 |
| Seat-dip effect | `RS7/` | FDTD fc=2000, resonant walls, anisotropic conformal: IL 0.89 in 1/3 oct | narrowband TF 0.77 |

CR1, CR3 and CR4 also run with `CR2/cr2_run.py <manifest>`. The known
errors are summed up in `../docs/validation/index.md`.

## Checks

- `tests/test_open_domain.py`: the open-domain solver against an image
  solution (no data, ~2 min).
- `tests/test_bras_data_checks.py`: five checks against external data
  (Rockfon fit, Genelec phase, scene 01 SOFA erratum, dodecahedron Lvl
  note, 8k note). Each one SKIPS cleanly if its environment variable is not
  set. The variables are listed at the top of the file.

## Licence

The files derived from BRAS (geometry, manifests, reports, figures) are
under CC BY-SA 4.0. The scripts are under MIT. See `LICENSE.md`.
