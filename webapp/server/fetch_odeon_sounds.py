# -*- coding: utf-8 -*-
"""Download Odeon's anechoic recordings into data_odeon/anechoic/, extract the WAV files and
licence notes, and index them. Local use under an Odeon Auditorium/Combined licence only.
Usage: python -m webapp.server.fetch_odeon_sounds [--extract-only]"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
import wave
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "data_odeon", "anechoic")
RAW = os.path.join(ROOT, "raw")
PAGE = "https://odeon.dk/downloads/odeon-zip-archives/"
MISC = ["https://odeon.dk/download/Anechoic/NoiseSignals.zip",
        "https://odeon.dk/download/Anechoic/CrowdSounds.zip",
        "https://odeon.dk/download/Anechoic/BackgroundSpeech.zip",
        "https://odeon.dk/download/Anechoic/Harvard%20Word%20List%20Speech%20Inteligibility%20Test%20Recordings.zip",
        "https://odeon.dk/download/ImpulseResponses/Clapping.wav"]
UA = {"User-Agent": "Mozilla/5.0 (Auralis local research fetch)"}


def fetch(url, timeout=900, tries=4):
    """GET with retries: (bytes, filename from Content-Disposition or URL)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                cd = r.headers.get("Content-Disposition", "")
                m = re.search(r'filename="?([^";]+)"?', cd)
                name = m.group(1) if m else urllib.parse.unquote(url.rsplit("/", 1)[-1])
                return r.read(), os.path.basename(name)
        except Exception as exc:
            print("  retry %d/%d %s: %s" % (k + 1, tries, url, exc), flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("could not download " + url)


def download():
    os.makedirs(os.path.join(RAW, "concerts"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "misc"), exist_ok=True)
    page = fetch(PAGE, timeout=120)[0].decode("utf-8", "replace")
    ids = sorted(set(re.findall(r"\?wpdmdl=(\d+)", page)), key=int)
    failed = []
    jobs = [(PAGE + "?wpdmdl=" + i, "concerts") for i in ids] + [(u, "misc") for u in MISC]
    done_file = os.path.join(RAW, "downloaded.json")
    done = json.load(open(done_file)) if os.path.exists(done_file) else {}
    for url, sub in jobs:
        if url in done and os.path.exists(os.path.join(RAW, sub, done[url])):
            continue
        try:
            data, name = fetch(url)
            with open(os.path.join(RAW, sub, name), "wb") as fh:
                fh.write(data)
            done[url] = name
            json.dump(done, open(done_file, "w"), indent=1)
            print("ok   %-70s %7.1f MB" % (name, len(data) / 1e6), flush=True)
        except Exception as exc:
            failed.append(url)
            print("FAIL %s: %s" % (url, exc), flush=True)
    return failed


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80]


TECHNICAL = re.compile(r"^Subject_\d+Res\d+deg", re.I)     # HRTF diffuse-field files, not recordings


def _wav_info(data):
    """(channels, fs, duration) from the RIFF chunks; handles float WAV that `wave` refuses."""
    try:
        w = wave.open(io.BytesIO(data))
        return w.getnchannels(), w.getframerate(), round(w.getnframes() / w.getframerate(), 2)
    except Exception:
        pass
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None, None, None
    pos, ch, fs, block = 12, None, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if cid == b"fmt ":
            ch = int.from_bytes(data[pos + 10:pos + 12], "little")
            fs = int.from_bytes(data[pos + 12:pos + 16], "little")
            block = int.from_bytes(data[pos + 20:pos + 22], "little")
        elif cid == b"data" and fs and block:
            return ch, fs, round(min(size, len(data) - pos - 8) / block / fs, 2)
        pos += 8 + size + (size & 1)
    return ch, fs, None


def _text(raw):
    """Licence notes come in UTF-8 or UTF-16 (with BOM)."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", "replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _credit(notes, piece):
    """The source's own credit wording, verbatim (licences forbid altering credits)."""
    m = re.search(r'["\u201c](Derived from[^"\u201d]+)["\u201d]', notes)
    if m:
        return m.group(1).strip()
    keys = ("recorded by", "performed by", "produced by", "recorded in", "recorded at")
    sentences = [x.strip() for x in re.split(r"(?<=\.)\s+|\n", notes) if x.strip()]
    hits = [x for x in sentences if any(k in x for k in keys)]
    if hits:
        return " ".join(hits[:2])
    return "Odeon A/S anechoic recordings (%s)" % piece


def _license(notes):
    base = ("Odeon Auditorium/Combined >= 16: internal use only, no redistribution, "
            "no modification; credit required")
    if "BY-NC-SA" in notes or "by-nc-sa" in notes:
        return "CC BY-NC-SA 4.0 + " + base
    if "Attribution 4.0" in notes:
        return "CC BY 4.0 + " + base
    return base


def extract():
    """WAV + licence notes only, flat per piece (no paths taken from the archive)."""
    index = []
    for sub in ("concerts", "misc"):
        d = os.path.join(RAW, sub)
        for fname in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            src = os.path.join(d, fname)
            piece = os.path.splitext(fname)[0]
            out = os.path.join(ROOT, sub, _slug(piece))
            os.makedirs(out, exist_ok=True)
            members = []
            if fname.lower().endswith(".wav"):
                members = [(fname, open(src, "rb").read())]
                notes = ""
            elif zipfile.is_zipfile(src):
                z = zipfile.ZipFile(src)
                notes = ""
                for n in z.namelist():
                    low = n.lower()
                    if low.endswith(".wav"):
                        members.append((os.path.basename(n), z.read(n)))
                    elif low.endswith(".licenses") or "licen" in low or low.endswith(("readme.txt", ".pdf")):
                        base = os.path.basename(n)
                        with open(os.path.join(out, _slug(base)), "wb") as fh:
                            fh.write(z.read(n))
                        if low.endswith((".licenses", ".txt")):
                            notes += _text(z.read(n)) + "\n"
            else:
                continue
            credit, lic = _credit(notes, piece), _license(notes)
            for base, data in members:
                if TECHNICAL.match(base):
                    continue
                path = os.path.join(out, _slug(base))
                if not (os.path.exists(path) and os.path.getsize(path) == len(data)):
                    with open(path, "wb") as fh:
                        fh.write(data)
                ch, fs, dur = _wav_info(data)
                index.append({"id": "odeon-%d" % len(index), "piece": piece,
                              "track": os.path.splitext(base)[0], "group": sub,
                              "file": os.path.relpath(path, ROOT), "channels": ch, "fs": fs,
                              "duration_s": dur, "credit": credit, "license": lic})
    with open(os.path.join(ROOT, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    unread = [e["file"] for e in index if e["fs"] is None]
    print("%d recordings indexed in %d pieces; WAV headers not readable by `wave`: %d" % (
        len(index), len({e["piece"] for e in index}), len(unread)))
    return index


def main():
    ap = argparse.ArgumentParser(description="Fetch Odeon's anechoic recordings (local research use)")
    ap.add_argument("--extract-only", action="store_true", help="skip the download step")
    args = ap.parse_args()
    failed = [] if args.extract_only else download()
    extract()
    if failed:
        print("FAILED downloads (re-run to resume): %s" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
