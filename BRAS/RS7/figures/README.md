# RS7 figures (published configuration, narrowband)

Published configuration since 2026-11-10: fc=2000, resonant walls (default),
ANISOTROPIC conformal correction (`AURALIS_FDTD_CONFORMAL=1
AURALIS_FDTD_CONFORMAL_ANISO=1`), 80 ms window. The figures come from the
8-pair run of that day.

They show the NARROWBAND insertion loss, with no smoothing: the strictest
reading. Every tooth of the comb must fall in the right place. The
third-octave values (the reference metric of the READMEs) are higher. Both are
published.

| figure | pair | narrowband IL Pearson | (1/3 oct) | why this one |
|---|---|---|---|---|
| `IL_LS2-MP4.png` | LS2-MP4 | **0.94** | 0.99 | the best |
| `IL_LS2-MP3.png` | LS2-MP3 | 0.80 | 0.91 | typical (median 0.78) |
| `IL_LS1-MP1.png` | LS1-MP1 | **0.73** | 0.81 | the worst |
| `TF_LS2-MP4.png` | LS2-MP4 | n/a | n/a | absolute TF (seat-dip comb) |

`TF_LS1-MP3_avant_apres.png` is an older figure. It shows the geometry fix
(pitch 0.22 → 0.34 m, from the official OBJ) that took RS7 from 0.01 to 0.39.
It was computed at fc=1000 with the conformal correction on, so it is NOT
comparable to the others.
