const XFADE_S = 0.05;
const SMOOTH_S = 0.015;

interface Chain {
  conv: ConvolverNode | null;
  fade: GainNode;
}

interface Strip {
  id: string;
  input: GainNode;
  wet: GainNode;
  dry: GainNode;
  chains: Chain[];
  node: AudioBufferSourceNode | null;
  buffer: AudioBuffer | null;
  gainDb: number;
  muted: boolean;
}

export const dbToGain = (db: number) => Math.pow(10, db / 20);

/**
 * Per source: AudioBufferSourceNode -> GainNode -> ConvolverNode (binaural IR of
 * the current receiver, no normalisation) -> crossfade gain -> master -> output.
 * A receiver switch builds a new convolver chain and crossfades over 50 ms;
 * so does a switch of IR set (A/B/C comparison), keeping the strips as they are.
 * A parallel "dry" path bypasses the convolvers for the unconvolved preview.
 */
export class AuralEngine {
  readonly ctx: AudioContext;
  private master: GainNode;
  private strips = new Map<string, Strip>();
  private irs = new Map<string, AudioBuffer>();
  private receiverId: string | null = null;
  /** IR set in use: A/B/C comparison of simulations with the same points. */
  private variant = 'A';
  private masterDb = 0;
  private dryOn = false;
  private loop = true;
  private playing = false;
  private activeNodes = 0;
  onPlayingChange?: (playing: boolean) => void;

  constructor(sampleRate?: number) {
    let ctx: AudioContext;
    try {
      ctx = new AudioContext(sampleRate ? { sampleRate, latencyHint: 'playback' } : { latencyHint: 'playback' });
    } catch {
      ctx = new AudioContext();
    }
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.connect(ctx.destination);
  }

  async close(): Promise<void> {
    this.stopNodes(0);
    try {
      await this.ctx.close();
    } catch {
      /* already closed */
    }
  }

  async decode(data: ArrayBuffer): Promise<AudioBuffer> {
    return await this.ctx.decodeAudioData(data);
  }

  setIR(sourceId: string, receiverId: string, buf: AudioBuffer, variant = 'A'): void {
    this.irs.set(`${variant}|${sourceId}|${receiverId}`, buf);
  }

  hasIR(sourceId: string, receiverId: string, variant = this.variant): boolean {
    return this.irs.has(`${variant}|${sourceId}|${receiverId}`);
  }

  /** Forget every IR of a set (its simulation was changed). */
  clearVariant(variant: string): void {
    for (const k of [...this.irs.keys()]) if (k.startsWith(`${variant}|`)) this.irs.delete(k);
  }

  getVariant(): string {
    return this.variant;
  }

  /** Swap the IR set in place: crossfade while playing, same sounds, gains and receiver. */
  setVariant(variant: string): void {
    if (variant === this.variant) return;
    this.variant = variant;
    if (this.receiverId) for (const s of this.strips.values()) this.buildChain(s, this.receiverId, this.playing);
  }

  private strip(id: string): Strip {
    let s = this.strips.get(id);
    if (!s) {
      const input = this.ctx.createGain();
      const wet = this.ctx.createGain();
      const dry = this.ctx.createGain();
      wet.gain.value = this.dryOn ? 0 : 1;
      dry.gain.value = this.dryOn ? 1 : 0;
      wet.connect(this.master);
      input.connect(dry);
      dry.connect(this.master);
      s = { id, input, wet, dry, chains: [], node: null, buffer: null, gainDb: 0, muted: false };
      this.strips.set(id, s);
      if (this.receiverId) this.buildChain(s, this.receiverId, false);
    }
    return s;
  }

  setSources(ids: string[]): void {
    for (const id of ids) this.strip(id);
  }

  private buildChain(s: Strip, receiverId: string, crossfade: boolean): void {
    const now = this.ctx.currentTime;
    const ir = this.irs.get(`${this.variant}|${s.id}|${receiverId}`) ?? null;
    const fade = this.ctx.createGain();
    let conv: ConvolverNode | null = null;
    if (ir) {
      conv = new ConvolverNode(this.ctx, { buffer: ir, disableNormalization: true });
      s.input.connect(conv);
      conv.connect(fade);
    }
    fade.connect(s.wet);
    const old = s.chains;
    if (crossfade) {
      fade.gain.setValueAtTime(0, now);
      fade.gain.linearRampToValueAtTime(1, now + XFADE_S);
      for (const c of old) {
        c.fade.gain.cancelScheduledValues(now);
        c.fade.gain.setValueAtTime(c.fade.gain.value, now);
        c.fade.gain.linearRampToValueAtTime(0, now + XFADE_S);
      }
      window.setTimeout(() => old.forEach((c) => this.disposeChain(s, c)), (XFADE_S + 0.1) * 1000);
    } else {
      fade.gain.value = 1;
      old.forEach((c) => this.disposeChain(s, c));
    }
    s.chains = [{ conv, fade }];
  }

  private disposeChain(s: Strip, c: Chain): void {
    try {
      if (c.conv) {
        s.input.disconnect(c.conv);
        c.conv.disconnect();
      }
      c.fade.disconnect();
    } catch {
      /* already disconnected */
    }
  }

  setReceiver(receiverId: string): void {
    const changed = receiverId !== this.receiverId;
    this.receiverId = receiverId;
    if (!changed) return;
    for (const s of this.strips.values()) this.buildChain(s, receiverId, this.playing);
  }

  setBuffer(sourceId: string, buf: AudioBuffer | null): void {
    const s = this.strip(sourceId);
    s.buffer = buf;
    if (this.playing) {
      this.stopStripNode(s, this.ctx.currentTime);
      if (buf) this.startStripNode(s, this.ctx.currentTime + 0.02);
    }
  }

  setStripGain(sourceId: string, gainDb: number, muted: boolean): void {
    const s = this.strip(sourceId);
    s.gainDb = gainDb;
    s.muted = muted;
    s.input.gain.setTargetAtTime(muted ? 0 : dbToGain(gainDb), this.ctx.currentTime, SMOOTH_S);
  }

  setMaster(db: number): void {
    this.masterDb = db;
    if (this.playing) this.master.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, SMOOTH_S);
  }

  setDry(on: boolean): void {
    this.dryOn = on;
    const now = this.ctx.currentTime;
    for (const s of this.strips.values()) {
      s.wet.gain.setTargetAtTime(on ? 0 : 1, now, SMOOTH_S);
      s.dry.gain.setTargetAtTime(on ? 1 : 0, now, SMOOTH_S);
    }
  }

  setLoop(loop: boolean): void {
    this.loop = loop;
    for (const s of this.strips.values()) if (s.node) s.node.loop = loop;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  async play(): Promise<void> {
    await this.ctx.resume();
    const now = this.ctx.currentTime;
    this.stopNodes(now);
    // Fresh convolvers so no tail of a previous playback leaks in.
    if (this.receiverId) for (const s of this.strips.values()) this.buildChain(s, this.receiverId, false);
    const t0 = now + 0.06;
    for (const s of this.strips.values()) if (s.buffer) this.startStripNode(s, t0);
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setValueAtTime(0, now);
    this.master.gain.linearRampToValueAtTime(dbToGain(this.masterDb), t0 + 0.02);
    this.setPlaying(this.activeNodes > 0);
  }

  stop(): void {
    const now = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setValueAtTime(this.master.gain.value, now);
    this.master.gain.linearRampToValueAtTime(0, now + 0.03);
    this.stopNodes(now + 0.04);
    this.setPlaying(false);
  }

  private startStripNode(s: Strip, when: number): void {
    if (!s.buffer) return;
    const node = new AudioBufferSourceNode(this.ctx, { buffer: s.buffer, loop: this.loop });
    node.connect(s.input);
    this.activeNodes++;
    node.onended = () => {
      if (s.node === node) s.node = null;
      this.activeNodes = Math.max(0, this.activeNodes - 1);
      if (this.activeNodes === 0 && this.playing) this.setPlaying(false);
    };
    node.start(when);
    s.node = node;
  }

  private stopStripNode(s: Strip, when: number): void {
    if (!s.node) return;
    const n = s.node;
    s.node = null;
    try {
      n.stop(when);
    } catch {
      /* not started */
    }
  }

  private stopNodes(when: number): void {
    for (const s of this.strips.values()) this.stopStripNode(s, when);
  }

  private setPlaying(p: boolean): void {
    if (p === this.playing) return;
    this.playing = p;
    this.onPlayingChange?.(p);
  }
}
