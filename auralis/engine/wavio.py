# -*- coding: utf-8 -*-
"""Minimal WAV I/O (no external dependencies): read PCM 8/16/24/32-bit and
IEEE float 32/64, write float32 (default) or PCM16.
Arrays are float64, shaped (n, channels), in [-1, 1]."""

from __future__ import annotations

import struct

import numpy as np


def read_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Invalid WAV file: %s" % path)

    pos = 12
    fmt = None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
        if fmt is not None and raw is not None:
            break
    if fmt is None or raw is None:
        raise ValueError("Missing fmt/data chunks: %s" % path)

    audio_format, n_ch, fs, _br, _ba, bits = struct.unpack("<HHIIHH", fmt[:16])
    if audio_format == 0xFFFE and len(fmt) >= 40:   # WAVE_FORMAT_EXTENSIBLE
        audio_format = struct.unpack("<H", fmt[24:26])[0]

    if audio_format == 1:      # PCM
        if bits == 8:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bits == 16:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif bits == 24:
            b = np.frombuffer(raw, dtype=np.uint8)
            n = len(b) // 3
            b = b[:n * 3].reshape(n, 3)
            v = (b[:, 0].astype(np.int32)
                 | (b[:, 1].astype(np.int32) << 8)
                 | (b[:, 2].astype(np.int32) << 16))
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float64) / 8388608.0
        elif bits == 32:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError("Unsupported %d-bit PCM" % bits)
    elif audio_format == 3:    # IEEE float
        if bits == 32:
            x = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        elif bits == 64:
            x = np.frombuffer(raw, dtype="<f8").astype(np.float64)
        else:
            raise ValueError("Unsupported %d-bit float" % bits)
    else:
        raise ValueError("Unsupported WAV format %d" % audio_format)

    n = (len(x) // n_ch) * n_ch
    return x[:n].reshape(-1, n_ch), int(fs)


def write_wav(path, data, fs, float32: bool = True):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, ch = x.shape
    if float32:
        # float32 WAV legitimately stores |x| > 1: NEVER clip here, silent
        # clipping would turn unnormalised mixes into crackle
        fmt_tag, bits = 3, 32
        payload = x.astype("<f4").tobytes()
    else:
        fmt_tag, bits = 1, 16
        payload = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    block = ch * bits // 8
    hdr = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, fmt_tag, ch, fs,
                                 fs * block, block, bits)
    hdr += b"data" + struct.pack("<I", len(payload))
    with open(path, "wb") as f:
        f.write(hdr + payload)
