# RS6 figures (published configuration, narrowband)

Published configuration since 2026-11-10: fc=2000, ANISOTROPIC conformal
correction (`AURALIS_FDTD_CONFORMAL=1 AURALIS_FDTD_CONFORMAL_ANISO=1`), 30 ms
window. The figures come from the 9-pair run of that day.

They show the NARROWBAND insertion loss, with no smoothing: the strictest
reading. The third-octave values (the reference metric of the READMEs) are
higher. Both are published.

| figure | pair | narrowband IL Pearson | (1/3 oct) | why this one |
|---|---|---|---|---|
| `IL_LS3-MP3.png` | LS3-MP3 | **0.96** | 0.97 | the best |
| `IL_LS3-MP2.png` | LS3-MP2 | 0.77 | 0.89 | typical, Treble showcase |
| `IL_LS2-MP3.png` | LS2-MP3 | **0.41** | 0.44 | the worst, still open |
| `TF_LS1-MP1.png` | LS1-MP1 | n/a | n/a | absolute TF |
