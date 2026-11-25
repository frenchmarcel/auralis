"""Read a BRAS loudspeaker directivity (third-octave MPS CSV, front pole)."""
import numpy as np

OCT = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]   # N_BANDS=7


def parse_complex(s):
    s = s.strip().replace(" ", "").replace("i", "j")
    try:
        return complex(s)
    except ValueError:
        return complex("nan")


def read_mps(csv):
    """(third-octave freqs, energy (181, F) per polar angle T, azimuth mean)."""
    lines = open(csv, encoding="utf-8-sig").read().splitlines()
    freqs = np.array([float(x) for x in lines[0].split(",")[1:] if x.strip()])
    nT = 181
    acc = np.zeros((nT, len(freqs)))
    cnt = np.zeros(nT)
    for ln in lines[1:]:
        p = ln.split(",")
        lab = p[0].strip()
        if not lab.startswith("P") or "T" not in lab:
            continue
        T = int(lab[lab.index("T") + 1:])
        v = np.array([parse_complex(x) for x in p[1:1 + len(freqs)]])
        # NaNs in the low third octaves (25 Hz...): NaN -> 0, keep the row
        acc[T] += np.nan_to_num(np.abs(v) ** 2, nan=0.0)
        cnt[T] += 1
    cnt[cnt == 0] = 1
    return freqs, acc / cnt[:, None]


def octave_table(freqs, E):
    """Q_b(theta) per octave (mean of the third octaves inside it), Q_b(0) = 1."""
    tbl = np.zeros((len(OCT), E.shape[0]))
    for b, fc in enumerate(OCT):
        m = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
        tbl[b] = E[:, m].mean(axis=1) if m.any() else E[:, np.argmin(
            np.abs(freqs - fc))]
    on = tbl[:, 0].copy()
    on[on <= 0] = 1.0
    return tbl / on[:, None]
