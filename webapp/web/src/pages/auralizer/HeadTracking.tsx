import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError, errorMessage, type AmbisonicDecoder, type RenderBody, type RenderResult } from '../../api';
import { AmbiHeadPlayer } from '../../audio/ambiPlayer';
import { decodeWav } from '../../audio/wav';
import { Spinner, Toggle } from '../../components/ui';
import { fmtDuration, fmtNum } from '../../lib/format';
import type { HeadStore } from './headStore';
import { MixTaskProgress, useMixTask } from './MixTask';

const DECODER_RATES = [44100, 48000];
/** Above this size the 9-channel mix weighs on the browser's memory (decoded once more). */
const BIG_WAV_BYTES = 300e6;

export interface TrackedRender {
  result: RenderResult;
  /** Mix key at render time (simulation, receiver, master, strips): stale when it changes. */
  key: string;
  letter: string;
  simName: string;
  receiverId: string;
  bytes: number;
  wavFs: number;
  playFs: number;
}

type Phase = 'idle' | 'rendering' | 'loading' | 'ready';

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  return await res.arrayBuffer();
}

/**
 * Head-tracked listening: server-side ambisonic mix (background task, format ambisonic),
 * decoded in the browser with the rotation applied before the binaural decoder.
 */
export function useHeadTracked(head: HeadStore, active: boolean) {
  const playerRef = useRef<AmbiHeadPlayer | null>(null);
  const decoders = useRef(new Map<string, Promise<AmbisonicDecoder>>());
  const token = useRef(0);
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const [render, setRender] = useState<TrackedRender | null>(null);
  const mix = useMixTask();
  const runMix = mix.run;
  const [playing, setPlaying] = useState(false);
  const [gainDb, setGainDb] = useState(-6);
  const [loop, setLoop] = useState(true);

  // Orientation -> rotation gains, at most once per animation frame (the gains ramp in between).
  useEffect(() => {
    let raf = 0;
    const push = () => {
      raf = 0;
      const h = head.get();
      playerRef.current?.setOrientation(h.yaw, h.pitch);
    };
    const unsub = head.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(push);
    });
    return () => {
      unsub();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [head]);

  useEffect(() => playerRef.current?.setGain(gainDb), [gainDb]);
  useEffect(() => playerRef.current?.setLoop(loop), [loop]);
  useEffect(() => {
    if (!active) playerRef.current?.stop();
  }, [active]);
  useEffect(
    () => () => {
      token.current++;
      void playerRef.current?.close();
      playerRef.current = null;
    },
    [],
  );
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    (window as unknown as { __auralisAmbi?: unknown }).__auralisAmbi = { player: () => playerRef.current, head };
  }, [head]);

  // the ambisonic mix does not depend on the head: the decoder carries it
  const decoderFor = (fs: number, headId: string) => {
    const key = `${fs}:${headId}`;
    let p = decoders.current.get(key);
    if (!p) {
      p = api.getAmbisonicDecoder(fs, headId);
      decoders.current.set(key, p);
      p.catch(() => decoders.current.delete(key));
    }
    return p;
  };

  const renderMix = useCallback(
    async (simId: string, body: RenderBody, meta: { key: string; letter: string; simName: string }) => {
      const my = ++token.current;
      setError(null);
      setCancelled(false);
      setPhase('rendering');
      try {
        const out = await runMix(simId, { ...body, format: 'ambisonic' });
        if (my !== token.current || out.status === 'superseded') return;
        if (out.status === 'cancelled') {
          setCancelled(true);
          setPhase(playerRef.current?.duration ? 'ready' : 'idle');
          return;
        }
        if (out.status === 'failed') throw new Error(out.error);
        const r = out.result;
        if (r.format && r.format !== 'ambisonic') throw new Error(`the server returned a ${r.format} mix instead of an ambisonic one`);
        setPhase('loading');
        const data = await fetchArrayBuffer(r.url);
        if (my !== token.current) return;
        const buf = decodeWav(data);
        const playFs = DECODER_RATES.includes(buf.sampleRate) ? buf.sampleRate : 48000;
        let player = playerRef.current;
        if (!player || player.sampleRate !== playFs) {
          player?.stop();
          await player?.close();
          player = new AmbiHeadPlayer(playFs);
          player.onPlayingChange = setPlaying;
          playerRef.current = player;
        }
        if (!DECODER_RATES.includes(player.sampleRate)) {
          throw new Error(`the browser opened its audio output at ${player.sampleRate} Hz; the binaural decoder exists at 44.1 and 48 kHz only`);
        }
        const headId = (await api.getSimulation(simId)).settings?.head ?? 'model';
        if (my !== token.current) return;
        const dec = await decoderFor(player.sampleRate, headId);
        if (my !== token.current) return;
        player.setDecoder(dec);
        player.setBuffer(buf);
        player.setGain(gainDb);
        player.setLoop(loop);
        const h = head.get();
        player.setOrientation(h.yaw, h.pitch, true);
        setRender({ result: r, ...meta, receiverId: body.receiver_id, bytes: data.byteLength, wavFs: buf.sampleRate, playFs: player.sampleRate });
        setPhase('ready');
      } catch (e) {
        if (my !== token.current) return;
        setError(errorMessage(e));
        setPhase(playerRef.current?.duration ? 'ready' : 'idle');
      }
    },
    // gain/loop are pushed by their own effects afterwards
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [head, runMix],
  );

  const play = useCallback(async () => {
    setError(null);
    try {
      await playerRef.current?.play();
    } catch (e) {
      setError(errorMessage(e));
    }
  }, []);
  const stop = useCallback(() => playerRef.current?.stop(), []);

  return { phase, error, cancelled, render, playing, play, stop, renderMix, mix, gainDb, setGainDb, loop, setLoop, player: playerRef };
}

export type HeadTracked = ReturnType<typeof useHeadTracked>;

/** Transport of the head-tracked mode: render, play, level, accuracy and credits. */
export function HeadTrackingPanel({
  ht,
  canRender,
  whyNot,
  currentKey,
  onRender,
  receiverName,
  nTracks,
  showLetter,
}: {
  ht: HeadTracked;
  canRender: boolean;
  whyNot: string | null;
  currentKey: string;
  onRender: () => void;
  receiverName: (id: string) => string;
  nTracks: number;
  showLetter: boolean;
}) {
  const r = ht.render;
  const stale = !!r && r.key !== currentKey;
  const busy = ht.phase === 'rendering' || ht.phase === 'loading';
  const [peak, setPeak] = useState<[number, number] | null>(null);
  useEffect(() => {
    if (!ht.playing) {
      setPeak(null);
      return;
    }
    const id = window.setInterval(() => {
      const p = ht.player.current?.peaks();
      if (p) setPeak((old) => [Math.max(p[0], (old?.[0] ?? 0) * 0.85), Math.max(p[1], (old?.[1] ?? 0) * 0.85)]);
    }, 120);
    return () => window.clearInterval(id);
  }, [ht.playing, ht.player]);
  const clip = !!peak && Math.max(peak[0], peak[1]) >= 0.999;

  return (
    <div className="ht-panel">
      <button
        className={`btn btn-play ${ht.playing ? 'on' : ''}`}
        onClick={() => (ht.playing ? ht.stop() : void ht.play())}
        disabled={!r || busy}
        title={r ? 'Play / stop the head-tracked mix (Space)' : 'Render the ambisonic mix first'}
      >
        {ht.playing ? '■ Stop' : '▶ Play'}
      </button>
      <div className="row gap wrap">
        <button className={`btn btn-sm ${!r || stale ? 'btn-primary' : ''}`} disabled={!canRender || busy} onClick={onRender} title={whyNot ?? 'Mix every non-muted strip at this receiver in 2nd-order ambisonics, on the server'}>
          {busy ? (
            <>
              <Spinner size={11} /> {ht.phase === 'rendering' ? 'Mixing…' : 'Loading…'}
            </>
          ) : r ? (
            'Render again'
          ) : (
            'Render for head tracking'
          )}
        </button>
        <Toggle checked={ht.loop} onChange={ht.setLoop} label="Loop" />
      </div>
      {whyNot && !busy && <div className="small muted">{whyNot}</div>}
      {ht.phase === 'rendering' && <MixTaskProgress mix={ht.mix} what={`${nTracks} track${nTracks === 1 ? '' : 's'}, 9 channels`} />}
      {ht.error && <div className="small text-err">{ht.error}</div>}
      {ht.cancelled && !busy && <div className="small text-warn">Mix cancelled.</div>}
      {r && (
        <div className="small muted">
          {showLetter ? `${r.letter}: ${r.simName} · ` : ''}
          {r.result.n_sources} source{r.result.n_sources === 1 ? '' : 's'} at {receiverName(r.receiverId)} · {fmtDuration(r.result.duration_s)} · 9 ch, {r.wavFs} Hz · {fmtNum(r.bytes / 1e6, 0)} MB
        </div>
      )}
      {r && r.wavFs !== r.playFs && <div className="small text-warn">The mix is at {r.wavFs} Hz: the browser resamples it to {r.playFs} Hz (the decoder's rate).</div>}
      {r && r.bytes > BIG_WAV_BYTES && <div className="small text-warn">Large mix ({fmtNum(r.bytes / 1e6, 0)} MB): the browser holds it twice in memory while loading.</div>}
      {stale && <div className="small text-warn">The mix, receiver or simulation changed since this render: render again to hear it.</div>}
      {r && (
        <div className="small ht-credits">
          {r.result.credits.length > 0 && (
            <ul className="motion-credits muted">
              {r.result.credits.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          )}
          <div className="row gap">
            <a href={r.result.credits_url} target="_blank" rel="noreferrer">
              Credits
            </a>
            <a href={r.result.url} download>
              Download 9-ch WAV
            </a>
          </div>
        </div>
      )}
      <label className="gain-slider">
        <span className="small muted">Level</span>
        <input type="range" min={-30} max={12} step={0.5} value={ht.gainDb} onChange={(e) => ht.setGainDb(Number(e.target.value))} onDoubleClick={() => ht.setGainDb(-6)} title="Playback level (double-click: −6 dB)" />
        <span className="small mono gain-value">
          {ht.gainDb > 0 ? '+' : ''}
          {ht.gainDb.toFixed(1)} dB
        </span>
      </label>
      {peak && (
        <div className={`small mono ${clip ? 'text-err' : 'muted'}`}>
          peak L {fmtNum(20 * Math.log10(Math.max(peak[0], 1e-6)), 1)} · R {fmtNum(20 * Math.log10(Math.max(peak[1], 1e-6)), 1)} dBFS{clip ? ' · clipping: lower the level' : ''}
        </div>
      )}
      <div className="notice notice-info small ht-note">
        <div>
          <b>2nd-order ambisonics</b>: the direct sound&apos;s interaural cues (ILD/ITD) are less exact than in <b>Fixed head</b>, which plays the exact binaural IR.
        </div>
        <details>
          <summary>Measured</summary>
          The field is rotated by the inverse of the head, then decoded to binaural in the browser. Against the direct binaural synthesis (CR2): early and late levels within ±1 dB per octave; direct sound ILD up to +4 dB too large around 1–2 kHz for lateral sources, ITD below 1 kHz about halved — the usual limits of 2nd order.
        </details>
      </div>
    </div>
  );
}
