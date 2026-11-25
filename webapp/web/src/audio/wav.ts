export interface WavInfo {
  sampleRate: number;
  channels: number;
  frames: number;
  /** 1 = integer PCM, 3 = IEEE float. */
  format: number;
  bits: number;
  dataOffset: number;
}

/** Header of a RIFF/WAVE file (PCM 16/24/32 bit, float 32/64, plain or WAVE_FORMAT_EXTENSIBLE). */
export function parseWavHeader(buf: ArrayBuffer): WavInfo {
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (buf.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('not a WAV file');
  let o = 12;
  let fmt: Omit<WavInfo, 'frames' | 'dataOffset'> | null = null;
  while (o + 8 <= buf.byteLength) {
    const id = tag(o);
    const size = v.getUint32(o + 4, true);
    const body = o + 8;
    if (id === 'fmt ') {
      let format = v.getUint16(body, true);
      const channels = v.getUint16(body + 2, true);
      const sampleRate = v.getUint32(body + 4, true);
      const bits = v.getUint16(body + 14, true);
      if (format === 0xfffe && size >= 26) format = v.getUint16(body + 24, true);
      fmt = { format, channels, sampleRate, bits };
    } else if (id === 'data') {
      if (!fmt) throw new Error('WAV: data chunk before fmt chunk');
      const bytes = Math.min(size, buf.byteLength - body);
      const frames = Math.floor(bytes / (fmt.channels * (fmt.bits / 8)));
      return { ...fmt, frames, dataOffset: body };
    }
    o = body + size + (size % 2);
  }
  throw new Error('WAV: no data chunk');
}

/**
 * Decode a WAV into an AudioBuffer at the file's own rate and channel count, without
 * the browser's decoder (which resamples to the context rate and may refuse 9 channels).
 */
export function decodeWav(buf: ArrayBuffer): AudioBuffer {
  const info = parseWavHeader(buf);
  const { channels: nc, frames, format, bits, dataOffset } = info;
  if (!nc || !frames) throw new Error('WAV: empty');
  const out = new AudioBuffer({ numberOfChannels: nc, length: frames, sampleRate: info.sampleRate });
  const v = new DataView(buf);
  const bps = bits / 8;
  const read: (o: number) => number =
    format === 3 && bits === 32
      ? (o) => v.getFloat32(o, true)
      : format === 3 && bits === 64
        ? (o) => v.getFloat64(o, true)
        : format === 1 && bits === 16
          ? (o) => v.getInt16(o, true) / 32768
          : format === 1 && bits === 24
            ? (o) => ((v.getUint8(o) | (v.getUint8(o + 1) << 8) | (v.getInt8(o + 2) << 16)) / 8388608)
            : format === 1 && bits === 32
              ? (o) => v.getInt32(o, true) / 2147483648
              : (() => {
                  throw new Error(`WAV: unsupported sample format ${format}/${bits} bit`);
                })();
  for (let c = 0; c < nc; c++) {
    const ch = out.getChannelData(c);
    let o = dataOffset + c * bps;
    const stride = nc * bps;
    for (let n = 0; n < frames; n++, o += stride) ch[n] = read(o);
  }
  return out;
}
