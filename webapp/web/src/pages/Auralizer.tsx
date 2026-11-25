import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, ApiError, errorMessage, type CatalogObject, type MeshData, type PathValidity, type PlacedObject, type RenderResult, type RenderStrip, type Results, type Simulation, type Sound } from '../api';
import { useApp } from '../AppContext';
import { AuralEngine } from '../audio/engine';
import { Timeline } from '../components/Timeline';
import { TopBar } from '../components/TopBar';
import { ErrorBox, ProgressBar, Segmented, Spinner, Toggle } from '../components/ui';
import { loadCatalog, loadGeometry, loadMesh } from '../lib/cache';
import { layerColor } from '../lib/colors';
import { fmtDuration, fmtNum } from '../lib/format';
import { boundsOf, dirFromAzEl, type Bounds } from '../lib/geom';
import { href } from '../lib/router';
import { useTimeStore } from '../lib/timeline';
import { endTime, type Movable } from '../lib/trajectory';
import { isSmooth, smoothCurve } from '../lib/smoothPath';
import { compatibility, type Compatibility } from '../lib/abCompare';
import { getSlice, setSlice, useSavedDoc } from '../lib/savedViews';
import { NamedConfigsButton } from '../components/NamedConfigs';
import { SimDataStore, useBundles } from '../lib/simdata';
import { locateSimulation } from '../lib/tree';
import { PlacedObjects } from '../three/Objects';
import { PathLines, TimedPointMarkers, type PathDraw } from '../three/Paths';
import { LayerMeshes, SceneCanvas, markerRadius, type PointMarker } from '../three/Scene';
import { MotionPanel } from './MotionPanel';
import { useHeadStore } from './auralizer/headStore';
import { HeadTrackingPanel, useHeadTracked } from './auralizer/HeadTracking';
import { MixTaskProgress, useMixTask } from './auralizer/MixTask';
import { FirstPersonView, HeadCone, HeadDial, useHeadKeys } from './auralizer/HeadViews';

interface StripState {
  soundId: string | null;
  gainDb: number;
  muted: boolean;
  /** true while the sound is the automatic default (the source's track); false once chosen by hand. */
  auto?: boolean;
}

/** Above this many sources with a sound, live in-browser convolution may glitch. */
const LIVE_LIMIT = 12;
const OLD_DEFAULT = 'gen:speechish';

/** fixed = exact binaural IRs, live convolution; tracked = ambisonic mix rotated by the head. */
type ListenMode = 'fixed' | 'tracked';
type MapView = 'map' | 'first';

interface Saved {
  strips: Record<string, StripState>;
  masterDb: number;
  receiverId: string | null;
  loop: boolean;
  /** A/B/C: simulation ids of B and C (A is this simulation). */
  variants?: (string | null)[];
  listenMode?: ListenMode;
  mapView?: MapView;
}

const LETTERS = ['A', 'B', 'C'] as const;

interface VariantLoad {
  simId: string;
  done: number;
  total: number;
  errors: string[];
}

/** The set-up is a saved view of the simulation (server, with this browser's copy as fallback). */
function loadSaved(simId: string): Partial<Saved> {
  return getSlice<Saved>(simId, 'auralizer', 'main') ?? {};
}

function storeSaved(simId: string, s: Saved): void {
  setSlice(simId, 'auralizer', 'main', s);
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (typeof j?.detail === 'string') detail = j.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return await res.arrayBuffer();
}

const SOUND_GROUPS: [string, (s: Sound) => boolean][] = [
  ['Test signals', (s) => s.kind === 'generated'],
  ['My sounds', (s) => s.kind === 'uploaded'],
  ['Anechoic music (Odeon, local licence)', (s) => s.kind === 'odeon_local' && s.group === 'concerts'],
  ['Anechoic speech, crowd & noise (Odeon, local licence)', (s) => s.kind === 'odeon_local' && s.group !== 'concerts'],
  ['Other', (s) => !['generated', 'uploaded', 'odeon_local'].includes(s.kind)],
];

/** Restores the saved set-up (strips, A/B/C) before the page starts. */
export function AuralizerPage({ simId }: { simId: string }) {
  const doc = useSavedDoc(simId, 'auralizer');
  if (!doc.ready)
    return (
      <div className="app-shell">
        <TopBar simId={simId} view="auralizer" />
        <div className="loading-row pad">
          <Spinner /> Loading…
        </div>
      </div>
    );
  return <AuralizerBody simId={simId} docError={doc.info?.error ?? null} saveError={doc.saveError} />;
}

function AuralizerBody({ simId, docError, saveError }: { simId: string; docError: string | null; saveError: string | null }) {
  const app = useApp();
  const [sim, setSim] = useState<Simulation | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [noResults, setNoResults] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mesh, setMesh] = useState<MeshData | null>(null);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  const [placed, setPlaced] = useState<PlacedObject[]>([]);
  const [catalogById, setCatalogById] = useState<Map<string, CatalogObject>>(new Map());
  const [sounds, setSounds] = useState<Sound[]>([]);
  const [soundsError, setSoundsError] = useState<string | null>(null);
  const saved = useMemo(() => loadSaved(simId), [simId]);
  const [strips, setStrips] = useState<Record<string, StripState>>(saved.strips ?? {});
  const [masterDb, setMasterDb] = useState(saved.masterDb ?? 0);
  const [receiverId, setReceiverId] = useState<string | null>(saved.receiverId ?? null);
  const [loop, setLoop] = useState(saved.loop ?? true);
  const [dryToggle, setDryToggle] = useState(false);
  const [dryHold, setDryHold] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [irProgress, setIrProgress] = useState<{ done: number; total: number; errors: string[] }>({ done: 0, total: 0, errors: [] });
  const [soundLoading, setSoundLoading] = useState<Set<string>>(new Set());
  const [audioError, setAudioError] = useState<string | null>(null);
  const engineRef = useRef<AuralEngine | null>(null);
  const soundCache = useRef(new Map<string, Promise<AudioBuffer>>());
  const [engineReady, setEngineReady] = useState(false);
  const [engineEpoch, setEngineEpoch] = useState(0);

  // A/B/C comparison: B and C are other simulations of the same space with the same points.
  const [variants, setVariants] = useState<(string | null)[]>(() => [simId, saved.variants?.[1] ?? null, saved.variants?.[2] ?? null]);
  const [activeVariant, setActiveVariant] = useState(0);
  const [variantLoads, setVariantLoads] = useState<(VariantLoad | null)[]>([null, null, null]);
  const dataStore = useMemo(() => new SimDataStore(), []);
  const loc = locateSimulation(app.tree, simId);
  const candKey = loc ? loc.space.geometries.flatMap((g) => g.simulations.map((m) => m.id)).filter((id) => id !== simId).join(',') : '';
  const candidateIds = useMemo(() => (candKey ? candKey.split(',') : []), [candKey]);
  const candidates = useBundles(dataStore, candidateIds);
  // Forget saved B/C picks that are no longer simulations of this space.
  const located = !!loc;
  useEffect(() => {
    if (!located) return;
    setVariants((v) => (v.some((id, i) => i > 0 && id && !candidateIds.includes(id)) ? v.map((id, i) => (i > 0 && id && !candidateIds.includes(id) ? null : id)) : v));
  }, [located, candidateIds]);
  const geometryNames = new Map((loc?.space.geometries ?? []).map((g) => [g.id, g.name]));
  const compat = useMemo(() => {
    const m = new Map<string, Compatibility>();
    if (!sim || !results) return m;
    for (const id of candidateIds) {
      const b = candidates.get(id)?.data;
      if (b) m.set(id, compatibility(sim, results, b.sim, b.results));
    }
    return m;
  }, [sim, results, candidateIds, candidates]);
  const variantSim = (i: number) => (i === 0 ? sim : (candidates.get(variants[i] ?? '')?.data?.sim ?? null));
  const variantReady = (i: number) => {
    if (i === 0) return true;
    const l = variantLoads[i];
    return !!variants[i] && !!l && l.simId === variants[i] && l.total > 0 && l.done >= l.total && l.errors.length === 0;
  };
  const activeSimId = (activeVariant > 0 && variants[activeVariant]) || simId;

  // The map shows the geometry of the simulation being heard (e.g. a furnished duplicate).
  const mapGid = variantSim(activeVariant)?.geometry_id ?? sim?.geometry_id ?? null;
  useEffect(() => {
    if (!mapGid) return;
    let cancelled = false;
    Promise.all([loadGeometry(mapGid), loadMesh(mapGid)])
      .then(([g, m]) => {
        if (cancelled) return;
        setBounds(boundsOf(g.bbox.min, g.bbox.max));
        setMesh(m);
        setPlaced(g.objects ?? []);
        if (g.objects?.length) {
          loadCatalog()
            .then((c) => !cancelled && setCatalogById(new Map(c.objects.map((o) => [o.id, o]))))
            .catch(() => undefined);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [mapGid]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await api.getSimulation(simId);
        if (cancelled) return;
        setSim(s);
        try {
          const r = await api.getResults(simId);
          if (!cancelled) setResults(r);
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) setNoResults(true);
          else throw e;
        }
      } catch (e) {
        if (!cancelled) setError(errorMessage(e));
      }
    })();
    api
      .listSounds()
      .then((s) => !cancelled && setSounds(s))
      .catch((e) => !cancelled && setSoundsError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [simId]);

  // Without static results (animated renders do not need them) the strips are the simulation's sources.
  const sourceIds = useMemo(
    () => (results ? Array.from(new Set(results.pairs.map((p) => p.source_id))) : noResults && sim ? sim.sources.map((s) => s.id) : []),
    [results, noResults, sim],
  );
  const receiverIds = useMemo(
    () => (results ? Array.from(new Set(results.pairs.map((p) => p.receiver_id))) : noResults && sim ? sim.receivers.map((r) => r.id) : []),
    [results, noResults, sim],
  );
  const srcName = (id: string) => sim?.sources.find((s) => s.id === id)?.name ?? id;
  const rcvName = (id: string) => sim?.receivers.find((r) => r.id === id)?.name ?? id;

  // Default strip state and receiver once results and sounds are known: a source's
  // own track (sound_id, ensembles) when it has one, else the speech-like signal.
  useEffect(() => {
    if (!sim || (!results && !noResults)) return;
    const known = (id: string) => sounds.some((s) => s.id === id);
    const generic = sounds.find((s) => s.id === OLD_DEFAULT)?.id ?? sounds[0]?.id ?? null;
    setStrips((prev) => {
      const n = { ...prev };
      for (const id of sourceIds) {
        const track = sim.sources.find((s) => s.id === id)?.sound_id ?? null;
        const def = track && (!sounds.length || known(track)) ? track : generic;
        const cur = n[id];
        const isAuto = cur ? cur.auto === true || (cur.auto === undefined && (!cur.soundId || cur.soundId === OLD_DEFAULT)) : true;
        if (!cur) n[id] = { soundId: def, gainDb: 0, muted: false, auto: true };
        else if (isAuto && cur.soundId !== def) n[id] = { ...cur, soundId: def, auto: true };
        else if (cur.soundId && sounds.length && !known(cur.soundId)) n[id] = { ...cur, soundId: def, auto: true };
      }
      return n;
    });
    setReceiverId((r) => (r && receiverIds.includes(r) ? r : (receiverIds[0] ?? null)));
  }, [results, noResults, sim, sounds, sourceIds, receiverIds]);

  const withSound = sourceIds.filter((id) => strips[id]?.soundId && !strips[id]?.muted).length;
  const bigMix = withSound > LIVE_LIMIT;

  // Audio engine + preload of every IR of the simulation.
  useEffect(() => {
    if (!results) return;
    const engine = new AuralEngine(results.fs);
    engine.onPlayingChange = setPlaying;
    engineRef.current = engine;
    soundCache.current = new Map();
    engine.setSources(sourceIds);
    setEngineReady(true);
    setEngineEpoch((n) => n + 1);
    let cancelled = false;
    const pairs = results.pairs;
    setIrProgress({ done: 0, total: pairs.length, errors: [] });
    (async () => {
      let next = 0;
      const worker = async () => {
        while (!cancelled && next < pairs.length) {
          const p = pairs[next++];
          try {
            const data = await fetchArrayBuffer(api.irWavUrl(simId, p.source_id, p.receiver_id));
            const buf = await engine.decode(data);
            if (cancelled) return;
            engine.setIR(p.source_id, p.receiver_id, buf);
            setIrProgress((s) => ({ ...s, done: s.done + 1 }));
          } catch (e) {
            if (cancelled) return;
            setIrProgress((s) => ({ ...s, done: s.done + 1, errors: [...s.errors, `${p.source_id} → ${p.receiver_id}: ${errorMessage(e)}`] }));
          }
        }
      };
      await Promise.all([worker(), worker(), worker(), worker()]);
    })();
    return () => {
      cancelled = true;
      setEngineReady(false);
      engineRef.current = null;
      void engine.close();
    };
  }, [results, simId, sourceIds]);

  const irsLoaded = irProgress.total > 0 && irProgress.done >= irProgress.total;

  // IRs of B and C, put on A's scale (each simulation's WAVs are normalised by its own ir_scale).
  // One token per slot: a newer pick (or a new engine) abandons the older load.
  const loadTokens = useRef<({ simId: string; engine: AuralEngine; token: number } | null)[]>([null, null, null]);
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engineReady || !results) return;
    [1, 2].forEach((i) => {
      const letter = LETTERS[i];
      const id = variants[i];
      const b = id ? candidates.get(id)?.data : undefined;
      const want = id && b?.results && compat.get(id)?.ok ? id : null;
      const cur = loadTokens.current[i];
      if (cur && cur.simId === want && cur.engine === engine) return;
      if (!want && !cur) return;
      engine.clearVariant(letter);
      if (!want || !b?.results) {
        loadTokens.current[i] = null;
        setVariantLoads((v) => v.map((x, k) => (k === i ? null : x)));
        return;
      }
      const token = (cur?.token ?? 0) + 1;
      loadTokens.current[i] = { simId: want, engine, token };
      const alive = () => loadTokens.current[i]?.token === token && engineRef.current === engine;
      const k = b.results.ir_scale > 0 ? results.ir_scale / b.results.ir_scale : 1;
      const pairs = results.pairs;
      setVariantLoads((v) => v.map((x, j) => (j === i ? { simId: want, done: 0, total: pairs.length, errors: [] } : x)));
      let next = 0;
      const worker = async () => {
        while (alive() && next < pairs.length) {
          const p = pairs[next++];
          try {
            const data = await fetchArrayBuffer(api.irWavUrl(want, p.source_id, p.receiver_id));
            const buf = await engine.decode(data);
            if (!alive()) return;
            if (Math.abs(k - 1) > 1e-9) {
              for (let c = 0; c < buf.numberOfChannels; c++) {
                const ch = buf.getChannelData(c);
                for (let n = 0; n < ch.length; n++) ch[n] *= k;
              }
            }
            engine.setIR(p.source_id, p.receiver_id, buf, letter);
            setVariantLoads((v) => v.map((x, j) => (j === i && x?.simId === want ? { ...x, done: x.done + 1 } : x)));
          } catch (e) {
            if (!alive()) return;
            setVariantLoads((v) => v.map((x, j) => (j === i && x?.simId === want ? { ...x, done: x.done + 1, errors: [...x.errors, `${p.source_id} → ${p.receiver_id}: ${errorMessage(e)}`] } : x)));
          }
        }
      };
      void Promise.all([worker(), worker(), worker()]);
    });
  }, [engineEpoch, engineReady, results, variants, candidates, compat]);

  // A variant that is not (or no longer) loaded cannot stay active.
  useEffect(() => {
    if (activeVariant > 0 && !variantReady(activeVariant)) setActiveVariant(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [variants, variantLoads, activeVariant]);

  useEffect(() => {
    engineRef.current?.setVariant(LETTERS[activeVariant]);
  }, [activeVariant, engineEpoch]);

  // Once all IRs are in, (re)build the chains for the current receiver.
  useEffect(() => {
    const e = engineRef.current;
    if (!e || !engineReady || !irsLoaded || !receiverId) return;
    e.setReceiver(receiverId);
  }, [engineReady, irsLoaded, receiverId]);

  const loadSound = useCallback((id: string): Promise<AudioBuffer> => {
    const engine = engineRef.current;
    if (!engine) return Promise.reject(new Error('Audio engine not ready'));
    let p = soundCache.current.get(id);
    if (!p) {
      setSoundLoading((s) => new Set(s).add(id));
      p = fetchArrayBuffer(api.soundWavUrl(id))
        .then((d) => engine.decode(d))
        .finally(() =>
          setSoundLoading((s) => {
            const n = new Set(s);
            n.delete(id);
            return n;
          }),
        );
      soundCache.current.set(id, p);
      p.catch(() => soundCache.current.delete(id));
    }
    return p;
  }, []);

  // Push sounds into the engine (eagerly only for small mixes: big ensembles load on Play).
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engineReady || bigMix) return;
    let cancelled = false;
    for (const id of sourceIds) {
      const st = strips[id];
      if (!st?.soundId) {
        engine.setBuffer(id, null);
        continue;
      }
      loadSound(st.soundId)
        .then((buf) => {
          if (!cancelled && engineRef.current === engine) engine.setBuffer(id, buf);
        })
        .catch((e) => !cancelled && setAudioError(`Sound “${st.soundId}”: ${errorMessage(e)}`));
    }
    return () => {
      cancelled = true;
    };
    // Only the sound assignment matters here, not gains.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineReady, sourceIds, loadSound, bigMix, JSON.stringify(sourceIds.map((id) => strips[id]?.soundId ?? null))]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engineReady) return;
    for (const id of sourceIds) {
      const st = strips[id];
      if (st) engine.setStripGain(id, st.gainDb, st.muted);
    }
  }, [strips, engineReady, sourceIds]);

  useEffect(() => {
    engineRef.current?.setMaster(masterDb);
  }, [masterDb, engineReady]);
  useEffect(() => {
    engineRef.current?.setLoop(loop);
  }, [loop, engineReady]);
  const dry = dryToggle || dryHold;
  useEffect(() => {
    engineRef.current?.setDry(dry);
  }, [dry, engineReady]);

  // Head-tracked listening (2nd-order ambisonics): the head starts looking along the receiver.
  const [listenMode, setListenMode] = useState<ListenMode>(saved.listenMode ?? 'fixed');
  const [mapView, setMapView] = useState<MapView>(saved.mapView ?? 'map');
  const tracked = listenMode === 'tracked';
  const head = useHeadStore();
  const ht = useHeadTracked(head, tracked);
  const curRcv = sim?.receivers.find((r) => r.id === receiverId) ?? null;
  const rcvAz = curRcv?.azimuth_deg ?? 0;
  const rcvEl = curRcv?.elevation_deg ?? 0;
  const resetHead = useCallback(() => head.set({ yaw: rcvAz, pitch: rcvEl }), [head, rcvAz, rcvEl]);
  useEffect(() => {
    resetHead();
  }, [receiverId, resetHead]);
  useHeadKeys(head, tracked, resetHead);
  useEffect(() => {
    if (tracked) engineRef.current?.stop();
  }, [tracked]);

  useEffect(() => {
    storeSaved(simId, { strips, masterDb, receiverId, loop, variants, listenMode, mapView });
  }, [simId, strips, masterDb, receiverId, loop, variants, listenMode, mapView]);

  // A restored named set-up is taken in place: the IRs stay loaded and playback goes on.
  const applySaved = () => {
    const s = loadSaved(simId);
    setStrips((cur) => ({ ...cur, ...(s.strips ?? {}) }));
    setMasterDb(s.masterDb ?? 0);
    setReceiverId((r) => (s.receiverId && receiverIds.includes(s.receiverId) ? s.receiverId : r));
    setLoop(s.loop ?? true);
    setVariants([simId, s.variants?.[1] ?? null, s.variants?.[2] ?? null]);
    setListenMode(s.listenMode ?? 'fixed');
    setMapView(s.mapView ?? 'map');
  };

  const play = async () => {
    const engine = engineRef.current;
    if (!engine) return;
    ht.stop();
    setAudioError(null);
    try {
      const pending = sourceIds.map((id) => strips[id]?.soundId).filter((x): x is string => !!x);
      const bufs = await Promise.all(pending.map((sid) => loadSound(sid)));
      sourceIds.forEach((id) => {
        const sid = strips[id]?.soundId;
        const i = sid ? pending.indexOf(sid) : -1;
        engine.setBuffer(id, i >= 0 ? bufs[i] : null);
      });
      if (receiverId) engine.setReceiver(receiverId);
      await engine.play();
      if (!engine.isPlaying()) setAudioError('Nothing to play: assign a sound to at least one source.');
    } catch (e) {
      setAudioError(errorMessage(e));
    }
  };

  const stop = () => engineRef.current?.stop();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.tagName === 'AUDIO')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (tracked) {
          if (ht.playing) ht.stop();
          else playTracked();
        } else if (engineRef.current?.isPlaying()) stop();
        else void play();
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey && (e.key === '1' || e.key === '2' || e.key === '3')) {
        const i = Number(e.key) - 1;
        if (variantReady(i)) setActiveVariant(i);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const upload = async (sourceId: string, file: File) => {
    try {
      const snd = await api.uploadSound(file, file.name.replace(/\.wav$/i, ''));
      setSounds((s) => [...s.filter((x) => x.id !== snd.id), snd]);
      setStrips((s) => ({ ...s, [sourceId]: { ...(s[sourceId] ?? { gainDb: 0, muted: false }), soundId: snd.id, auto: false } }));
      app.toast('success', `Sound “${snd.name}” added`);
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  const [render, setRender] = useState<{ result: RenderResult; key: string; letter: string; simName: string } | null>(null);
  const mix = useMixTask();
  const rendering = mix.active;
  const [renderWhat, setRenderWhat] = useState('');
  const [renderError, setRenderError] = useState<string | null>(null);
  const [renderCancelled, setRenderCancelled] = useState(false);
  const renderStrips: RenderStrip[] = sourceIds.flatMap((id) => {
    const st = strips[id];
    return st?.soundId ? [{ source_id: id, sound_id: st.soundId, gain_db: st.gainDb, muted: st.muted }] : [];
  });
  const renderKey = JSON.stringify([activeSimId, receiverId, masterDb, renderStrips]);
  const doRender = async () => {
    if (!receiverId) return;
    const meta = { key: renderKey, letter: LETTERS[activeVariant], simName: variantSim(activeVariant)?.name ?? '' };
    const n = renderStrips.filter((r) => !r.muted).length;
    setRenderWhat(`${n} track${n === 1 ? '' : 's'} at ${rcvName(receiverId)}`);
    setRenderError(null);
    setRenderCancelled(false);
    const out = await mix.run(activeSimId, { receiver_id: receiverId, master_db: masterDb, strips: renderStrips });
    if (out.status === 'done') setRender({ result: out.result, ...meta });
    else if (out.status === 'failed') setRenderError(out.error);
    else if (out.status === 'cancelled') setRenderCancelled(true);
  };

  const playTracked = () => {
    engineRef.current?.stop();
    void ht.play();
  };
  const trackedWhyNot = !results
    ? 'Needs the results of a run (from v0.5, with ambisonic IRs).'
    : !receiverId
      ? 'Pick a receiver.'
      : renderStrips.every((r) => r.muted)
        ? 'Assign a sound to at least one non-muted source.'
        : null;
  const doRenderTracked = () => {
    if (!receiverId || trackedWhyNot) return;
    void ht.renderMix(
      activeSimId,
      { receiver_id: receiverId, master_db: masterDb, strips: renderStrips },
      { key: renderKey, letter: LETTERS[activeVariant], simName: variantSim(activeVariant)?.name ?? '' },
    );
  };

  const timeStore = useTimeStore();
  const [listening, setListening] = useState<{ el: HTMLAudioElement; label: string } | null>(null);
  // While an animated render plays, the map follows its clock.
  useEffect(() => {
    if (!listening) return;
    let id = 0;
    const tick = () => {
      timeStore.set(listening.el.currentTime);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [listening, timeStore]);
  const movables = useMemo(() => new Map<string, Movable>(sim ? [...sim.sources, ...sim.receivers].map((x) => [x.id, x]) : []), [sim]);
  const tMax = useMemo(() => Math.max(0, ...[...movables.values()].map(endTime)), [movables]);

  const points = useMemo<PointMarker[]>(() => {
    if (!sim) return [];
    return [
      ...sim.sources
        .filter((s) => sourceIds.includes(s.id))
        .map((s) => {
          const st = strips[s.id];
          const active = !!st?.soundId && !st.muted;
          return {
            kind: 'source' as const,
            id: s.id,
            label: s.name,
            position: s.position,
            dir: dirFromAzEl(s.azimuth_deg, s.elevation_deg),
            dim: !active,
            badge: active ? (playing ? '▶' : '♪') : undefined,
          };
        }),
      ...sim.receivers
        .filter((r) => receiverIds.includes(r.id))
        .map((r) => ({
          kind: 'receiver' as const,
          id: r.id,
          label: r.name,
          position: r.position,
          dir: dirFromAzEl(r.azimuth_deg, r.elevation_deg),
          selected: r.id === receiverId,
          dim: r.id !== receiverId,
        })),
    ];
  }, [sim, strips, receiverId, playing, sourceIds, receiverIds]);

  // Smooth paths are drawn along the curve the server checked (the same curve is computed here otherwise).
  const [pathChecks, setPathChecks] = useState<Map<string, PathValidity>>(new Map());
  const hasSmooth = !!sim && [...sim.sources, ...sim.receivers].some((x) => isSmooth(x.path));
  useEffect(() => {
    if (!hasSmooth) return;
    let cancelled = false;
    api
      .getValidation(simId)
      .then((v) => !cancelled && setPathChecks(new Map((v.paths ?? []).map((p) => [p.id, p]))))
      .catch((e) => console.warn(`[Auralis] validation unavailable, smooth paths drawn from the local curve: ${errorMessage(e)}`));
    return () => {
      cancelled = true;
    };
  }, [simId, hasSmooth]);

  const pathDraws = useMemo<PathDraw[]>(() => {
    if (!sim) return [];
    const shown = new Set(points.map((p) => `${p.kind}:${p.id}`));
    const out: PathDraw[] = [];
    for (const [kind, list] of [['source', sim.sources], ['receiver', sim.receivers]] as const) {
      for (const x of list) {
        if (!x.path || !shown.has(`${kind}:${x.id}`)) continue;
        out.push({
          kind,
          id: x.id,
          waypoints: x.path.waypoints.map((w) => w.position),
          curve: isSmooth(x.path) ? smoothCurve(x.path, pathChecks.get(x.id)) : null,
          selected: kind === 'receiver' && x.id === receiverId,
          dim: points.find((p) => p.kind === kind && p.id === x.id)?.dim,
        });
      }
    }
    return out;
  }, [sim, points, receiverId, pathChecks]);

  const setStrip = (id: string, p: Partial<StripState>) =>
    setStrips((s) => ({ ...s, [id]: { ...(s[id] ?? { soundId: null, gainDb: 0, muted: false }), ...p } }));

  return (
    <div className="app-shell">
      <TopBar simId={simId} view="auralizer" />
      {error && (
        <div className="pad">
          <ErrorBox message={error} />
        </div>
      )}
      {(docError || saveError) && (
        <div className="notice notice-warn small">
          {saveError ? `This set-up could not be saved on the server (${saveError}): it is kept in this browser only.` : `The saved set-up could not be read from the server (${docError}): this browser's copy is used.`}
        </div>
      )}
      {!error && !results && !noResults && (
        <div className="loading-row pad">
          <Spinner /> Loading…
        </div>
      )}
      {(results || (noResults && sim)) && (
        <div className="aur-configs">
          <NamedConfigsButton
            simId={simId}
            docKey="auralizer"
            what="set-up"
            contents="the sound, gain and mute of every strip, the master gain, the receiver, Loop, the A/B/C simulations, the listening mode and the map view"
            restoreEffect="It applies at once, during playback too (the head-tracking mode stops playback, as when you switch to it by hand)."
            onRestored={applySaved}
          />
        </div>
      )}
      {(results || (noResults && sim)) && (
        <div className="auralizer">
          <div className="aur-top">
            <div className={`aur-map ${tMax > 0 ? 'has-timeline' : ''}`}>
              {tracked && mapView === 'first' && mesh && bounds && curRcv ? (
                <FirstPersonView
                  mesh={mesh}
                  objects={placed}
                  catalog={catalogById}
                  position={curRcv.position}
                  head={head}
                  markers={points.filter((p) => p.kind === 'source')}
                  markerRadius={markerRadius(bounds) * 1.3}
                />
              ) : mesh && bounds ? (
                <SceneCanvas bounds={bounds} topView showGizmo={false}>
                  <LayerMeshes mesh={mesh} shading="ghosted" pickable={false} styleOf={(_, i) => ({ color: layerColor(i) })} />
                  <PlacedObjects objects={placed} catalog={catalogById} showVisual pickable={false} />
                  <PathLines paths={pathDraws} radius={markerRadius(bounds) * 1.3} />
                  <TimedPointMarkers
                    time={timeStore}
                    items={movables}
                    points={points}
                    radius={markerRadius(bounds) * 1.3}
                    onPointClick={(p) => {
                      if (p.kind === 'receiver') setReceiverId(p.id);
                    }}
                  />
                  {tracked && curRcv && <HeadCone head={head} position={curRcv.position} radius={markerRadius(bounds) * 9} />}
                </SceneCanvas>
              ) : (
                <Spinner />
              )}
              {tracked ? (
                <div className="map-toolbar">
                  <Segmented<MapView>
                    value={mapView}
                    onChange={setMapView}
                    options={[
                      { value: 'map', label: 'Top view' },
                      { value: 'first', label: 'First person', disabled: !curRcv, title: 'The room seen from the receiver: drag to turn the head' },
                    ]}
                  />
                  <span className="small muted">
                    {mapView === 'first' ? 'Drag to look around' : 'Click a receiver to listen from it · drag to pan'} · ←/→ turn, ↑/↓ pitch (Shift: 15°) · H resets
                  </span>
                </div>
              ) : (
                <div className="map-hint small muted">Top view · click a receiver to listen from it · drag to pan, wheel to zoom</div>
              )}
              {tracked && curRcv && (
                <div className="head-overlay">
                  <HeadDial head={head} onReset={resetHead} resetTitle={`Look along the receiver again (azimuth ${fmtNum(rcvAz, 0)}°, elevation ${fmtNum(rcvEl, 0)}°)`} />
                </div>
              )}
              {results?.stale && <div className="notice notice-warn map-notice">Stale results: the simulation was edited after this run.</div>}
              {tMax > 0 && <Timeline time={timeStore} tMax={tMax} driver={listening?.label ?? null} className="map-timeline" />}
            </div>
            {sim && (
              <aside className="aur-motion">
                <MotionPanel
                  simId={simId}
                  sim={sim}
                  receiverId={receiverId}
                  strips={renderStrips}
                  masterDb={masterDb}
                  srcName={srcName}
                  rcvName={rcvName}
                  onListen={(el, label) => setListening(el ? { el, label } : null)}
                />
              </aside>
            )}
          </div>
          <div className="mixer">
            <div className="mixer-transport">
              <Segmented<ListenMode>
                value={listenMode}
                onChange={setListenMode}
                options={[
                  { value: 'fixed', label: 'Fixed head', title: 'Exact binaural IRs, convolved live in the browser' },
                  { value: 'tracked', label: 'Head tracking', title: '2nd-order ambisonic mix rotated by your head (drag, arrow keys), decoded to binaural in the browser' },
                ]}
              />
              {tracked ? (
                <HeadTrackingPanel
                  ht={ht}
                  canRender={!trackedWhyNot}
                  whyNot={trackedWhyNot}
                  currentKey={renderKey}
                  onRender={doRenderTracked}
                  receiverName={rcvName}
                  nTracks={renderStrips.filter((r) => !r.muted).length}
                  showLetter={variants.some((v, i) => i > 0 && v)}
                />
              ) : (
                <>
                  <button className={`btn btn-play ${playing ? 'on' : ''}`} onClick={() => (playing ? stop() : void play())} disabled={!irsLoaded} title="Play / stop (Space)">
                    {playing ? '■ Stop' : '▶ Play'}
                  </button>
                  <Toggle checked={loop} onChange={setLoop} label="Loop" />
                  <div className="dry-controls">
                    <button
                      className={`btn btn-sm ${dryHold ? 'btn-on' : ''}`}
                      onPointerDown={() => setDryHold(true)}
                      onPointerUp={() => setDryHold(false)}
                      onPointerLeave={() => setDryHold(false)}
                      title="Hold to hear the dry (unconvolved) sounds"
                    >
                      Hold for dry
                    </button>
                    <Toggle checked={dryToggle} onChange={setDryToggle} label="Dry" title="Toggle the dry (unconvolved) sounds" />
                  </div>
                  {dry && <div className="small text-warn">Dry: no room. Levels are not matched to the convolved signal.</div>}
                  <div className="ir-status small">
                    {!results ? (
                      <span className="text-warn">
                        No static results yet: live listening and Render to WAV need a run (<a href={href.editor(simId)}>editor</a>). Animated renders do not.
                      </span>
                    ) : irsLoaded ? (
                      <span className="muted">{irProgress.total} IRs loaded (binaural, fixed head)</span>
                    ) : (
                      <>
                        <span>
                          Loading IRs {irProgress.done}/{irProgress.total}
                        </span>
                        <ProgressBar value={irProgress.total ? irProgress.done / irProgress.total : 0} />
                      </>
                    )}
                  </div>
                  {irProgress.errors.length > 0 && (
                    <div className="small text-err">
                      {irProgress.errors.length} IR(s) failed: {irProgress.errors.slice(0, 2).join(' · ')}
                    </div>
                  )}
                  {audioError && <div className="small text-err">{audioError}</div>}
                  {bigMix && (
                    <div className="warn-box small">
                      {withSound} sources with sounds: live in-browser convolution may glitch (and loads every track on Play). Prefer <b>Render to WAV</b>.
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="mixer-receivers">
              {results && sim && (
                <AbcPanel
                  sim={sim}
                  variants={variants}
                  active={activeVariant}
                  onActivate={setActiveVariant}
                  onPick={(i, id) => {
                    setVariants((v) => v.map((x, k) => (k === i ? id : x)));
                    if (activeVariant === i) setActiveVariant(0);
                  }}
                  candidateIds={candidateIds}
                  candidates={candidates}
                  compat={compat}
                  loads={variantLoads}
                  ready={variantReady}
                  geometryNames={geometryNames}
                  playing={playing}
                />
              )}
              <div className="section-title">Receiver</div>
              <div className="receiver-buttons" role="radiogroup">
                {receiverIds.map((r) => (
                  <button key={r} role="radio" aria-checked={receiverId === r} className={`chip-btn rcv ${receiverId === r ? 'active' : ''}`} onClick={() => setReceiverId(r)}>
                    {rcvName(r)}
                  </button>
                ))}
              </div>
            </div>

            <div className="mixer-strips">
              {soundsError && <ErrorBox message={`Sounds unavailable: ${soundsError}`} />}
              {sourceIds.map((id) => {
                const st = strips[id] ?? { soundId: null, gainDb: 0, muted: false };
                const missingIr = irsLoaded && receiverId && !engineRef.current?.hasIR(id, receiverId);
                return (
                  <div key={id} className={`strip ${st.muted ? 'muted' : ''}`}>
                    <div className="strip-head">
                      <span className="point-dot source">{id}</span>
                      <span className="ellipsis strip-name">{srcName(id)}</span>
                      <button className={`btn btn-xs mute ${st.muted ? 'on' : ''}`} onClick={() => setStrip(id, { muted: !st.muted })} aria-pressed={st.muted} title="Mute">
                        M
                      </button>
                    </div>
                    <div className="row gap">
                      <select value={st.soundId ?? ''} onChange={(e) => setStrip(id, { soundId: e.target.value || null, auto: false })} aria-label={`Sound for ${srcName(id)}`}>
                        <option value="">(no sound)</option>
                        {SOUND_GROUPS.map(([label, keep]) => {
                          const list = sounds.filter(keep);
                          return list.length ? (
                            <optgroup key={label} label={label}>
                              {list.map((s) => (
                                <option key={s.id} value={s.id}>
                                  {s.name}
                                  {s.duration_s ? ` · ${s.duration_s.toFixed(1)} s` : ''}
                                </option>
                              ))}
                            </optgroup>
                          ) : null;
                        })}
                      </select>
                      <label className="btn btn-xs" title="Upload a .wav (added to the sound list)">
                        Upload
                        <input
                          type="file"
                          accept=".wav,audio/wav"
                          hidden
                          onChange={(e) => {
                            const f = e.target.files?.[0];
                            if (f) void upload(id, f);
                            e.target.value = '';
                          }}
                        />
                      </label>
                      {st.soundId && soundLoading.has(st.soundId) && <Spinner size={11} />}
                    </div>
                    {(() => {
                      const snd = sounds.find((s) => s.id === st.soundId);
                      return snd?.credit ? (
                        <div
                          className="small muted sound-credit"
                          title={[snd.credit, snd.kind === 'odeon_local' ? 'local licence, not redistributable' : '', snd.license ?? '']
                            .filter(Boolean)
                            .join(' · ')}
                        >
                          {snd.credit}
                        </div>
                      ) : null;
                    })()}
                    {(() => {
                      const track = sim?.sources.find((s) => s.id === id)?.sound_id;
                      return track && sounds.length > 0 && !sounds.some((s) => s.id === track) ? (
                        <div className="small text-warn">Track “{track}” of this source is not available here.</div>
                      ) : null;
                    })()}
                    <GainSlider value={st.gainDb} onChange={(v) => setStrip(id, { gainDb: v })} label="Gain" />
                    {missingIr && <div className="small text-err">No IR for this source at {rcvName(receiverId!)}</div>}
                  </div>
                );
              })}
            </div>

            <div className="mixer-master">
              <div className="section-title">Master</div>
              <GainSlider value={masterDb} onChange={setMasterDb} label="Overall" min={-40} max={12} />
              <div className="render-box">
                <button
                  className={`btn btn-sm ${bigMix ? 'btn-primary' : ''}`}
                  disabled={rendering || !results || !receiverId || renderStrips.every((r) => r.muted)}
                  onClick={() => void doRender()}
                  title="Binaural mix of every non-muted strip at the current receiver (of the simulation being heard: A, B or C), computed on the server"
                >
                  {rendering ? (
                    <>
                      <Spinner size={11} /> Rendering…
                    </>
                  ) : (
                    'Render to WAV'
                  )}
                </button>
                <div className="small muted">Rendered on the server, in the background — use it for large ensembles.</div>
                <MixTaskProgress mix={mix} what={renderWhat} />
                {!rendering && renderError && <div className="small text-err">Render failed: {renderError}</div>}
                {!rendering && renderCancelled && <div className="small text-warn">Render cancelled.</div>}
                {render && (
                  <div className="render-result">
                    {render.key !== renderKey && <div className="small text-warn">Mix changed since this render.</div>}
                    {variants.some((v, i) => i > 0 && v) && (
                      <div className="small muted">
                        {render.letter}: {render.simName}
                      </div>
                    )}
                    <audio controls preload="metadata" src={render.result.url} />
                    <div className="row gap small">
                      <a href={render.result.url} download>
                        Download WAV
                      </a>
                      <a href={render.result.credits_url} target="_blank" rel="noreferrer">
                        Credits
                      </a>
                    </div>
                    <div className="small muted">
                      {render.result.n_sources} sources at {rcvName(render.result.receiver_id ?? receiverId ?? '')} · {fmtDuration(render.result.duration_s)} · {render.result.fs} Hz ·
                      normalised {render.result.normalization_db > 0 ? '+' : ''}
                      {fmtNum(render.result.normalization_db, 1)} dB
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function GainSlider({ value, onChange, label, min = -40, max = 12 }: { value: number; onChange: (v: number) => void; label: string; min?: number; max?: number }) {
  return (
    <label className="gain-slider">
      <span className="small muted">{label}</span>
      <input type="range" min={min} max={max} step={0.5} value={value} onChange={(e) => onChange(Number(e.target.value))} onDoubleClick={() => onChange(0)} title="Double-click to reset to 0 dB" />
      <span className="small mono gain-value">
        {value > 0 ? '+' : ''}
        {value.toFixed(1)} dB
      </span>
    </label>
  );
}

function AbcPanel({
  sim,
  variants,
  active,
  onActivate,
  onPick,
  candidateIds,
  candidates,
  compat,
  loads,
  ready,
  geometryNames,
  playing,
}: {
  sim: Simulation;
  variants: (string | null)[];
  active: number;
  onActivate: (i: number) => void;
  onPick: (i: number, id: string | null) => void;
  candidateIds: string[];
  candidates: Map<string, { data?: { sim: Simulation; results: Results | null }; loading?: boolean; error?: string }>;
  compat: Map<string, Compatibility>;
  loads: ({ simId: string; done: number; total: number; errors: string[] } | null)[];
  ready: (i: number) => boolean;
  geometryNames: Map<string, string>;
  playing: boolean;
}) {
  const probing = candidateIds.some((id) => !candidates.get(id)?.data && !candidates.get(id)?.error);
  const incompatible = candidateIds.filter((id) => compat.get(id) && !compat.get(id)!.ok);
  const nameOf = (id: string) => candidates.get(id)?.data?.sim.name ?? id;
  const geomOf = (id: string) => {
    const g = candidates.get(id)?.data?.sim.geometry_id;
    return g && g !== sim.geometry_id ? geometryNames.get(g) : undefined;
  };
  const used = variants.filter((v): v is string => !!v);
  return (
    <div className="abc">
      <div className="section-title" title="Up to 3 simulations of the same space whose sources and receivers have the same ids and positions. Switch while playing (keys 1, 2, 3): the IRs are swapped in place, with the same sounds and gains.">
        Compare A/B/C
      </div>
      {LETTERS.map((L, i) => {
        const id = variants[i];
        const c = id && i > 0 ? compat.get(id) : undefined;
        const l = loads[i];
        const isReady = ready(i);
        return (
          <div key={L} className={`abc-row ${active === i ? 'active' : ''}`}>
            <button
              type="button"
              className={`abc-letter ${active === i ? 'on' : ''}`}
              disabled={!isReady}
              aria-pressed={active === i}
              onClick={() => onActivate(i)}
              title={isReady ? `Listen to ${L}${playing ? ' (swaps the IRs in place)' : ''} · key ${i + 1}` : 'Pick a compatible simulation first'}
            >
              {L}
            </button>
            {i === 0 ? (
              <span className="ellipsis abc-name" title={sim.name}>
                {sim.name}
              </span>
            ) : (
              <select value={id ?? ''} onChange={(e) => onPick(i, e.target.value || null)} aria-label={`Simulation ${L}`}>
                <option value="">{candidateIds.length ? '(none)' : '(no other simulation in this space)'}</option>
                {candidateIds.map((cid) => {
                  const cc = compat.get(cid);
                  const st = candidates.get(cid);
                  const off = !cc?.ok || (used.includes(cid) && cid !== id);
                  const why = st?.error ? 'unavailable' : !st?.data ? '…' : !st.data.results ? 'no results' : cc && !cc.ok ? 'positions differ' : used.includes(cid) && cid !== id ? 'already used' : '';
                  const g = geomOf(cid);
                  return (
                    <option key={cid} value={cid} disabled={off && cid !== id}>
                      {nameOf(cid)}
                      {g ? ` [${g}]` : ''}
                      {why ? ` — ${why}` : ''}
                    </option>
                  );
                })}
              </select>
            )}
            {i > 0 && id && l && l.simId === id && l.done < l.total && (
              <span className="small muted mono">
                {l.done}/{l.total}
              </span>
            )}
            {i > 0 && id && c && !c.ok && <span className="small text-err" title={c.reasons.join('\n')}>!</span>}
            {i > 0 && id && l && l.errors.length > 0 && (
              <span className="small text-err" title={l.errors.join('\n')}>
                IR error
              </span>
            )}
            {i > 0 && id && c?.warnings.length ? (
              <span className="small text-warn" title={c.warnings.join('\n')}>
                ⚠
              </span>
            ) : null}
          </div>
        );
      })}
      {probing && (
        <div className="small muted">
          <Spinner size={10} /> checking the other simulations of this space…
        </div>
      )}
      {variants.slice(1).map((id, k) => {
        const c = id ? compat.get(id) : undefined;
        if (!id || !c || (c.ok && !c.warnings.length)) return null;
        return (
          <div key={k} className={`small ${c.ok ? 'text-warn' : 'text-err'}`}>
            {LETTERS[k + 1]}: {[...c.reasons, ...c.warnings].join('; ')}.
          </div>
        );
      })}
      {incompatible.length > 0 && (
        <details className="small abc-why">
          <summary className="muted">
            {incompatible.length} simulation{incompatible.length > 1 ? 's' : ''} of this space cannot be compared
          </summary>
          <ul>
            {incompatible.map((id) => (
              <li key={id}>
                <b>{nameOf(id)}</b>: {compat.get(id)!.reasons.join('; ')}.
              </li>
            ))}
          </ul>
          <div className="muted">A/B/C needs the same source and receiver ids at the same positions (1 cm), so that only the room changes.</div>
        </details>
      )}
    </div>
  );
}
