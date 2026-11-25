import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auralis.engine.wavio import write_wav  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp",
                   "server", "data", "examples", "paris", "sounds")
FS = 32000
SEGMENT_S = 60.0

# Freesound recordings, all CC0 1.0 (licence checked on each page): id -> (author, title, role)
RECORDINGS = {
    522221: ("Nox_Sound", "Vehicle_Car_Peugeot_308_Acceleration_Idle_Mono.wav", "car"),
    268622: ("EverydaySounds", "VW Golf R (Zoom H1, OST Lav).wav", "car"),
    451668: ("kyles", "school bus truck ext idle engine close2.flac", "bus"),
    635757: ("kyles", "traffic heavy boulevard passing by close cars trucks people fast "
                      "Montreal, Canada.flac", "flow"),
    451516: ("kyles", "traffic light mopeds scooters cars trucks motorcycles rickshaws pass by "
                      "good detail side street with walls echo in front of city gate2 like from "
                      "window India.flac", "flow"),
    512142: ("lupalenzuela", "Paris_traffic.wav", "hum"),
    566891: ("KikeVilaplana", "Medium Crowd Ambience Outdoor Talking Background Spanish "
                              "STR_Zoom_H4nPro", "crowd"),
    634880: ("brunoboselli", "Generic Exterior Walla", "crowd"),
    869048: ("SignatureSoundsOrg", "Walla_In_Madrid_Spain", "crowd"),
}


def decode(mp3):
    """Mono float samples at FS of an audio file (ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-ac", "1", "-ar", str(FS),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def busiest(x, seg):
    """Start (samples) of the seg-long window with the highest median 100 ms level: no silent
    intro or tail in the kept part."""
    n = len(x)
    if n <= seg:
        return 0
    hop = FS // 10
    e = np.array([np.mean(x[k:k + hop] ** 2) for k in range(0, n - hop, hop)])
    w = seg // hop
    best, arg = -1.0, 0
    for k in range(0, len(e) - w + 1, 5):
        m = float(np.median(e[k:k + w]))
        if m > best:
            best, arg = m, k
    return arg * hop


def main():
    """Convert the CC0 Freesound previews (HQ MP3, downloaded into SRC as fs_<id>.mp3) into the
    Paris example's sounds: mono 16-bit WAV at 32 kHz, the busiest 60 s, with sounds.json
    (credits, the kept segment)."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("src", help="folder holding fs_<id>.mp3")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = {}
    for sid, (author, title, role) in RECORDINGS.items():
        x = decode(os.path.join(a.src, "fs_%d.mp3" % sid))
        seg = int(SEGMENT_S * FS)
        k0 = busiest(x, seg)
        y = x[k0:k0 + seg]
        y = 0.89 * y / max(float(np.abs(y).max()), 1e-9)
        f = 0.05 * FS                                  # 50 ms fades: clean loop points
        r = np.minimum(np.arange(len(y)) / f, 1.0)
        y = y * np.minimum(r, r[::-1])
        write_wav(os.path.join(a.out, "fs_%d.wav" % sid), y, FS, float32=False)
        index[str(sid)] = {"file": "fs_%d.wav" % sid, "role": role, "title": title,
                           "author": author, "licence": "CC0 1.0",
                           "url": "https://freesound.org/s/%d/" % sid,
                           "credit": "%s by %s (freesound.org/s/%d/), CC0 1.0" % (title, author, sid),
                           "segment_s": [round(k0 / FS, 1), round((k0 + len(y)) / FS, 1)],
                           "source_duration_s": round(len(x) / FS, 1)}
        print("%d %-6s %5.1f s kept from %.1f s (of %.1f s)" % (sid, role, len(y) / FS, k0 / FS,
                                                                len(x) / FS))
    with open(os.path.join(a.out, "sounds.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
