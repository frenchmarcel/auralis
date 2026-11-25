import type { AmbisonicDecoder } from '../api';
import { headRotationMatrix, N_CH } from './ambiRotation';
import { dbToGain } from './engine';

/** Time constant of the rotation gains (s): smooth enough to avoid zipper noise while turning. */
const ROTATION_TAU_S = 0.02;
const DEGREE_OF = Array.from({ length: N_CH }, (_, c) => Math.floor(Math.sqrt(c)));

/** The 35 entries a rotation can make non-zero: it never mixes degrees (1 + 3x3 + 5x5). */
export const ROTATION_ENTRIES: [number, number][] = (() => {
  const out: [number, number][] = [];
  for (let i = 0; i < N_CH; i++) for (let j = 0; j < N_CH; j++) if (DEGREE_OF[i] === DEGREE_OF[j]) out.push([i, j]);
  return out;
})();

/**
 * Head-tracked binaural playback of a 9-channel ambisonic signal (ACN/SN3D, world axes):
 * source (9 ch) -> splitter -> rotation gains (field turned by the inverse of the head)
 * -> 9 convolvers (decoder FIRs, one stereo pair per channel) -> sum -> master -> output.
 */
export class AmbiHeadPlayer {
  readonly ctx: AudioContext;
  private splitter: ChannelSplitterNode;
  private gains = new Map<string, GainNode>();
  private convs: ConvolverNode[] = [];
  private sum: GainNode;
  private master: GainNode;
  private meters: AnalyserNode[];
  private meterBuf: Float32Array<ArrayBuffer>;
  private buffer: AudioBuffer | null = null;
  private node: AudioBufferSourceNode | null = null;
  private matrix: Float64Array = headRotationMatrix(0, 0);
  private gainDb = 0;
  private loop = true;
  private playing = false;
  private startedAt = 0;
  onPlayingChange?: (playing: boolean) => void;

  constructor(sampleRate: number) {
    let ctx: AudioContext;
    try {
      ctx = new AudioContext({ sampleRate, latencyHint: 'interactive' });
    } catch {
      ctx = new AudioContext({ latencyHint: 'interactive' });
    }
    this.ctx = ctx;
    this.splitter = ctx.createChannelSplitter(N_CH);
    this.sum = new GainNode(ctx, { channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers' });
    this.master = new GainNode(ctx, { gain: 0 });
    this.sum.connect(this.master);
    this.master.connect(ctx.destination);
    const split2 = ctx.createChannelSplitter(2);
    this.master.connect(split2);
    this.meters = [0, 1].map((e) => {
      const a = new AnalyserNode(ctx, { fftSize: 2048 });
      split2.connect(a, e);
      return a;
    });
    this.meterBuf = new Float32Array(2048);
    for (const [i, j] of ROTATION_ENTRIES) {
      const g = new GainNode(ctx, { gain: this.matrix[i * N_CH + j] });
      this.splitter.connect(g, j);
      this.gains.set(`${i},${j}`, g);
    }
  }

  get sampleRate(): number {
    return this.ctx.sampleRate;
  }

  /** One convolver per ambisonic channel; its stereo IR is the channel's pair of decoder FIRs. */
  setDecoder(dec: AmbisonicDecoder): void {
    if (dec.fs !== this.ctx.sampleRate) throw new Error(`decoder at ${dec.fs} Hz for an audio context at ${this.ctx.sampleRate} Hz`);
    if (dec.filters.length !== N_CH) throw new Error(`decoder has ${dec.filters.length} channels, expected ${N_CH}`);
    for (const c of this.convs) c.disconnect();
    this.convs = dec.filters.map((pair) => {
      const ir = new AudioBuffer({ numberOfChannels: 2, length: dec.n_taps, sampleRate: this.ctx.sampleRate });
      ir.copyToChannel(Float32Array.from(pair[0]), 0);
      ir.copyToChannel(Float32Array.from(pair[1]), 1);
      const conv = new ConvolverNode(this.ctx, { buffer: ir, disableNormalization: true });
      conv.connect(this.sum);
      return conv;
    });
    for (const [i, j] of ROTATION_ENTRIES) {
      const g = this.gains.get(`${i},${j}`)!;
      g.disconnect();
      g.connect(this.convs[i]);
    }
  }

  setBuffer(buf: AudioBuffer): void {
    if (buf.numberOfChannels !== N_CH) throw new Error(`the mix has ${buf.numberOfChannels} channels, expected ${N_CH} (2nd-order ambisonics)`);
    const wasPlaying = this.playing;
    this.stopNode(this.ctx.currentTime);
    this.buffer = buf;
    if (wasPlaying) this.startNode(this.ctx.currentTime + 0.02, 0);
  }

  get duration(): number {
    return this.buffer?.duration ?? 0;
  }

  /** Head looking along azimuth yaw (CCW from +X) and elevation pitch (+ up), degrees. */
  setOrientation(yawDeg: number, pitchDeg: number, immediate = false): void {
    this.matrix = headRotationMatrix(yawDeg, pitchDeg);
    const now = this.ctx.currentTime;
    for (const [i, j] of ROTATION_ENTRIES) {
      const p = this.gains.get(`${i},${j}`)!.gain;
      const v = this.matrix[i * N_CH + j];
      p.cancelScheduledValues(now);
      if (immediate || !this.playing) p.setValueAtTime(v, now);
      else p.setTargetAtTime(v, now, ROTATION_TAU_S);
    }
  }

  /** The 9x9 matrix last requested (row-major). */
  targetMatrix(): Float64Array {
    return this.matrix;
  }

  /** Current values of the rotation gains (they ramp towards targetMatrix). */
  gainValues(): Float64Array {
    const out = new Float64Array(N_CH * N_CH);
    for (const [i, j] of ROTATION_ENTRIES) out[i * N_CH + j] = this.gains.get(`${i},${j}`)!.gain.value;
    return out;
  }

  setGain(db: number): void {
    this.gainDb = db;
    if (this.playing) this.master.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, 0.015);
  }

  setLoop(loop: boolean): void {
    this.loop = loop;
    if (this.node) this.node.loop = loop;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  /** Playback position (s) within the mix. */
  position(): number {
    if (!this.playing || !this.buffer) return 0;
    const t = Math.max(0, this.ctx.currentTime - this.startedAt);
    return this.loop ? t % this.buffer.duration : Math.min(t, this.buffer.duration);
  }

  /** Peak of the output over the last ~43 ms, per ear (linear). */
  peaks(): [number, number] {
    return this.meters.map((a) => {
      a.getFloatTimeDomainData(this.meterBuf);
      let m = 0;
      for (let k = 0; k < this.meterBuf.length; k++) m = Math.max(m, Math.abs(this.meterBuf[k]));
      return m;
    }) as [number, number];
  }

  async play(): Promise<void> {
    if (!this.buffer) throw new Error('no ambisonic mix loaded');
    if (!this.convs.length) throw new Error('no binaural decoder loaded');
    await this.ctx.resume();
    const now = this.ctx.currentTime;
    this.stopNode(now);
    const t0 = now + 0.05;
    this.startNode(t0, 0);
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setValueAtTime(0, now);
    this.master.gain.linearRampToValueAtTime(dbToGain(this.gainDb), t0 + 0.02);
    this.setPlaying(true);
  }

  stop(): void {
    const now = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setValueAtTime(this.master.gain.value, now);
    this.master.gain.linearRampToValueAtTime(0, now + 0.03);
    this.stopNode(now + 0.04);
    this.setPlaying(false);
  }

  async close(): Promise<void> {
    this.stopNode(0);
    this.setPlaying(false);
    try {
      await this.ctx.close();
    } catch {
      /* already closed */
    }
  }

  private startNode(when: number, offset: number): void {
    if (!this.buffer) return;
    const node = new AudioBufferSourceNode(this.ctx, { buffer: this.buffer, loop: this.loop });
    node.connect(this.splitter);
    node.onended = () => {
      if (this.node === node) {
        this.node = null;
        this.setPlaying(false);
      }
    };
    node.start(when, offset);
    this.node = node;
    this.startedAt = when - offset;
  }

  private stopNode(when: number): void {
    const n = this.node;
    if (!n) return;
    this.node = null;
    try {
      n.stop(when);
    } catch {
      /* not started */
    }
  }

  private setPlaying(p: boolean): void {
    if (p === this.playing) return;
    this.playing = p;
    this.onPlayingChange?.(p);
  }
}
