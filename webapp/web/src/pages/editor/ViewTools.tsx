import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import * as THREE from 'three';
import { api, errorMessage, type Geometry, type Material, type MeshData, type Simulation, type Vec3 } from '../../api';
import { NumField, Segmented, Spinner, Toggle } from '../../components/ui';
import { cacheGeometry } from '../../lib/cache';
import { safeFilename } from '../../lib/export';
import { fmtNum } from '../../lib/format';
import { buildTraceMesh, type EmitPattern, type TraceMesh } from '../../lib/raytrace';
import {
  ABS_DB_MIN,
  layerAlphas,
  ORDER_MAX,
  rampCss,
  rampGradientCss,
  TOOL_BANDS,
  toolBandLabel,
  type LayerAlphas,
} from '../../lib/toolRamps';
import {
  DENSITY_ORDER_T,
  ParticleAnimation,
  ReflectionDensity,
  traceDensity,
  type ParticleOptions,
  type ParticleStats,
  type PlayState,
} from '../../three/AcousticTools';
import { CAPTURE_MAX, CAPTURE_MIN, CaptureBridge, captureFrame, type CaptureFn, type CaptureOptions } from '../../three/Capture';
import { LIGHTING_PATCH_OK, VisualModel, type VisualGizmoMode, type VisualTransform } from '../../three/VisualModel';
import {
  DEFAULT_LIGHTING,
  initialLighting,
  LIGHTING_LIMITS,
  LIGHTING_PRESETS,
  sanitizeLighting,
  serverStoresLighting,
  writeLocalLighting,
  type LightingPreset,
  type VisualLighting,
} from '../../lib/visualLighting';

export type ViewTool = 'particles' | 'density';
export type RoomDisplay = 'normal' | 'overlay' | 'invisible';

export interface DensityOptions {
  orders: boolean[];
  rays: number;
  onlySelected: boolean;
  hideRays: boolean;
  xray: boolean;
}

const DEFAULT_PARTICLES: ParticleOptions = {
  pattern: 'sphere',
  count: 500,
  speed: 5,
  diameterMm: 60,
  colorBy: 'order',
  trails: false,
  maxBounces: 50,
  loop: true,
  xray: true,
};

const DEFAULT_DENSITY: DensityOptions = { orders: [true, true, false, false], rays: 4000, onlySelected: false, hideRays: false, xray: true };

const DEFAULT_CAPTURE: CaptureOptions = {
  width: 1920,
  height: 1080,
  background: 'theme',
  customColor: '#ffffff',
  showSources: true,
  showReceivers: true,
  showLabels: true,
  showGrid: false,
  showGizmos: false,
};

const SPEEDS = [
  { v: 343, label: 'Real time (343 m/s)' },
  { v: 20, label: '20 m/s' },
  { v: 10, label: '10 m/s' },
  { v: 5, label: '5 m/s' },
  { v: 2.5, label: '2.5 m/s' },
  { v: 1, label: '1 m/s' },
];

const IDENTITY: VisualTransform = { position: [0, 0, 0], rotation_deg: [0, 0, 0], scale: 1, up_axis: 'y' };

/** State of the 3D tools of the editor: particles, reflection density, capture, visual model. */
export function useViewTools({
  mesh,
  sim,
  geometry,
  materialsById,
  selLayers,
  selectedSourceId,
  absOn,
  setAbsOn,
  onGeometry,
  toast,
}: {
  mesh: MeshData | null;
  sim: Simulation | null;
  geometry: Geometry | null;
  materialsById: Map<string, Material>;
  selLayers: Set<string>;
  selectedSourceId: string | null;
  absOn: boolean;
  setAbsOn: (v: boolean) => void;
  onGeometry: (g: Geometry) => void;
  toast: (kind: 'error' | 'warning' | 'info' | 'success', msg: string) => void;
}) {
  const [tool, setToolRaw] = useState<ViewTool | null>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [band, setBand] = useState(3);
  const [particles, setParticles] = useState<ParticleOptions>(DEFAULT_PARTICLES);
  const [play, setPlay] = useState<PlayState>('stopped');
  const [stats, setStats] = useState<ParticleStats | null>(null);
  const [density, setDensity] = useState<DensityOptions>(DEFAULT_DENSITY);
  const [captureOpen, setCaptureOpen] = useState(false);
  const [capture, setCapture] = useState<CaptureOptions>(DEFAULT_CAPTURE);
  const [capturing, setCapturing] = useState(false);
  const captureApi = useRef<CaptureFn | null>(null);

  // Mutually exclusive with the α colouring.
  const setTool = useCallback(
    (t: ViewTool | null) => {
      setToolRaw(t);
      if (t) setAbsOn(false);
      if (t !== 'particles') setPlay('stopped');
    },
    [setAbsOn],
  );
  useEffect(() => {
    if (absOn) {
      setToolRaw(null);
      setPlay('stopped');
    }
  }, [absOn]);

  // The traced mesh is built on first use, then kept for this mesh.
  const traceCache = useRef<{ mesh: MeshData; trace: TraceMesh } | null>(null);
  const trace = useMemo(() => {
    if (!tool || !mesh) return null;
    if (traceCache.current?.mesh === mesh) return traceCache.current.trace;
    const t = buildTraceMesh(mesh);
    traceCache.current = { mesh, trace: t };
    return t;
  }, [tool, mesh]);

  const sources = sim?.sources ?? [];
  const source = sources.find((s) => s.id === sourceId) ?? sources.find((s) => s.id === selectedSourceId) ?? sources[0] ?? null;
  const sourcePos: Vec3 | null = source ? source.position : null;

  const alphas = useMemo<LayerAlphas | null>(
    () => (trace && sim ? layerAlphas(trace.layerNames, sim.layers, materialsById, band) : null),
    [trace, sim?.layers, materialsById, band], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const bboxRadius = geometry ? Math.hypot(...geometry.bbox.max.map((v, i) => v - geometry.bbox.min[i])) / 2 : 10;
  const densityResult = useMemo(
    () => (tool === 'density' && trace && sourcePos ? traceDensity(trace, sourcePos, density.rays, bboxRadius) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tool, trace, sourcePos?.[0], sourcePos?.[1], sourcePos?.[2], density.rays, bboxRadius],
  );
  const densityFilter = useMemo(() => {
    if (!density.onlySelected || !trace) return null;
    const s = new Set<number>();
    trace.layerNames.forEach((n, i) => selLayers.has(n) && s.add(i));
    return s;
  }, [density.onlySelected, trace, selLayers]);

  // ---------- visual model ----------
  const visualInfo = geometry?.visual ?? null;
  const gid = geometry?.id ?? null;
  const [visualShown, setVisualShown] = useState(true);
  const [visualOpacity, setVisualOpacity] = useState(1);
  const [roomDisplay, setRoomDisplay] = useState<RoomDisplay>('overlay');
  const [align, setAlign] = useState<VisualGizmoMode | null>(null);
  const [draft, setDraft] = useState<VisualTransform | null>(visualInfo?.transform ?? null);
  const [visualState, setVisualState] = useState<'idle' | 'uploading' | 'saving' | 'error'>('idle');
  const [visualError, setVisualError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploadFrac, setUploadFrac] = useState(0);
  const [modelBox, setModelBox] = useState<THREE.Box3 | null>(null);
  const [visualPanel, setVisualPanel] = useState(false);
  const saveTimer = useRef<number | undefined>(undefined);
  const pending = useRef<VisualTransform | null>(null);

  useEffect(() => {
    setDraft(visualInfo?.transform ?? null);
    setLoadError(null);
    setModelBox(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gid, visualInfo?.uploaded]);

  const pushTransform = useCallback(async () => {
    const t = pending.current;
    if (!t || !gid) return;
    pending.current = null;
    setVisualState('saving');
    try {
      const g = await api.setVisualTransform(gid, t);
      cacheGeometry(g);
      onGeometry(g);
      setVisualState('idle');
      setVisualError(null);
    } catch (e) {
      setVisualState('error');
      setVisualError(errorMessage(e));
    }
  }, [gid, onGeometry]);

  /** final: save now (gizmo released, field committed); otherwise only the view follows. */
  const changeTransform = useCallback(
    (t: VisualTransform, final = true) => {
      setDraft(t);
      if (!final) return;
      pending.current = t;
      window.clearTimeout(saveTimer.current);
      saveTimer.current = window.setTimeout(() => void pushTransform(), 250);
    },
    [pushTransform],
  );
  useEffect(() => () => window.clearTimeout(saveTimer.current), []);

  const uploadVisual = async (file: File) => {
    if (!gid) return;
    if (!file.name.toLowerCase().endsWith('.glb')) {
      toast('warning', 'A binary glTF (.glb) file is expected.');
      return;
    }
    setVisualState('uploading');
    setUploadFrac(0);
    try {
      const g = await api.uploadVisual(gid, file, setUploadFrac);
      cacheGeometry(g);
      onGeometry(g);
      setVisualShown(true);
      setVisualState('idle');
      setVisualError(null);
      toast('success', `Visual model “${file.name}” uploaded (display only: the solvers never see it).`);
    } catch (e) {
      setVisualState('error');
      setVisualError(errorMessage(e));
      toast('error', `Visual model: ${errorMessage(e)}`);
    }
  };

  const removeVisual = async () => {
    if (!gid) return;
    try {
      const g = await api.deleteVisual(gid);
      cacheGeometry(g);
      onGeometry(g);
      setAlign(null);
      setVisualState('idle');
    } catch (e) {
      toast('error', `Remove the visual model: ${errorMessage(e)}`);
    }
  };

  /** Model centred on the room in plan, its lowest point on the room's floor level (scale and rotation kept). */
  const centreOnRoom = () => {
    if (!draft || !modelBox || !geometry) return;
    const m = placementMatrix({ ...draft, position: [0, 0, 0] });
    const b = modelBox.clone().applyMatrix4(m);
    const rb = geometry.bbox;
    const c = (i: number) => (rb.min[i] + rb.max[i]) / 2;
    const round = (v: number) => Math.round(v * 1e4) / 1e4;
    changeTransform({
      ...draft,
      position: [round(c(0) - (b.min.x + b.max.x) / 2), round(c(1) - (b.min.y + b.max.y) / 2), round(rb.min[2] - b.min.z)],
    });
  };

  const visualUrl = visualInfo && gid ? api.visualUrl(gid, visualInfo.uploaded) : null;
  const visualActive = !!visualUrl && visualShown && !loadError;

  // Lighting of the visual model: per geometry, on the server when it stores it, else in this browser.
  const [lighting, setLightingState] = useState<VisualLighting>(() => initialLighting(gid, visualInfo));
  const lightingRef = useRef(lighting);
  const [lightingSave, setLightingSave] = useState<{ where: 'server' | 'browser'; error: string | null }>({ where: 'browser', error: null });
  const lightTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    const l = initialLighting(gid, visualInfo);
    lightingRef.current = l;
    setLightingState(l);
    setLightingSave({ where: serverStoresLighting(visualInfo) ? 'server' : 'browser', error: null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gid, !!visualInfo]);
  useEffect(() => () => window.clearTimeout(lightTimer.current), []);
  const onServer = serverStoresLighting(visualInfo);
  const persistLighting = useCallback(
    (l: VisualLighting) => {
      if (!gid) return;
      window.clearTimeout(lightTimer.current);
      if (!onServer) {
        const ok = writeLocalLighting(gid, l);
        setLightingSave({ where: 'browser', error: ok ? null : 'the browser storage is unavailable: kept for this session only' });
        return;
      }
      lightTimer.current = window.setTimeout(() => {
        api
          .setVisualLighting(gid, l)
          .then((g) => {
            cacheGeometry(g);
            onGeometry(g);
            setLightingSave({ where: 'server', error: null });
          })
          .catch((e) => {
            writeLocalLighting(gid, l);
            setLightingSave({ where: 'server', error: `${errorMessage(e)}; kept in this browser` });
          });
      }, 400);
    },
    [gid, onServer, onGeometry],
  );
  /** A preset, or a change of one control (the lighting becomes custom). */
  const changeLighting = useCallback(
    (patch: Partial<VisualLighting>, preset: LightingPreset = 'custom') => {
      const next = sanitizeLighting({ ...lightingRef.current, ...patch, preset }) ?? DEFAULT_LIGHTING;
      lightingRef.current = next;
      setLightingState(next);
      persistLighting(next);
    },
    [persistLighting],
  );
  const applyLightingPreset = useCallback(
    (id: LightingPreset) => {
      const p = LIGHTING_PRESETS.find((x) => x.id === id);
      if (p) changeLighting(p.values, p.id);
    },
    [changeLighting],
  );

  // ---------- capture ----------
  const exportPng = async (simName: string) => {
    const fn = captureApi.current;
    if (!fn) return;
    setCapturing(true);
    try {
      const blob = await fn(capture);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${safeFilename(simName)}_${capture.width}x${capture.height}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 4000);
    } catch (e) {
      toast('error', `Capture: ${errorMessage(e)}`);
    } finally {
      setCapturing(false);
    }
  };

  return {
    tool,
    setTool,
    source,
    setSourceId,
    band,
    setBand,
    particles,
    setParticles,
    play,
    setPlay,
    stats,
    setStats,
    density,
    setDensity,
    densityResult,
    densityFilter,
    trace,
    alphas,
    bboxRadius,
    captureOpen,
    setCaptureOpen,
    capture,
    setCapture,
    capturing,
    captureApi,
    exportPng,
    visual: {
      info: visualInfo,
      url: visualUrl,
      active: visualActive,
      shown: visualShown,
      setShown: setVisualShown,
      opacity: visualOpacity,
      setOpacity: setVisualOpacity,
      roomDisplay,
      setRoomDisplay,
      align,
      setAlign,
      draft: draft ?? visualInfo?.transform ?? IDENTITY,
      changeTransform,
      state: visualState,
      error: visualError,
      loadError,
      setLoadError,
      uploadFrac,
      upload: uploadVisual,
      remove: removeVisual,
      modelBox,
      setModelBox,
      centreOnRoom,
      panel: visualPanel,
      setPanel: setVisualPanel,
      lighting,
      changeLighting,
      applyLightingPreset,
      lightingSave,
    },
  };
}

export type ViewTools = ReturnType<typeof useViewTools>;

const DEG = Math.PI / 180;

/** The placement chain of the visual model as one matrix (same as VisualModel's groups). */
function placementMatrix(t: VisualTransform): THREE.Matrix4 {
  const up = new THREE.Matrix4().makeRotationX(t.up_axis === 'y' ? Math.PI / 2 : 0);
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(t.rotation_deg[0] * DEG, t.rotation_deg[1] * DEG, t.rotation_deg[2] * DEG, 'ZYX'));
  const m = new THREE.Matrix4().compose(new THREE.Vector3(...t.position), q, new THREE.Vector3(t.scale, t.scale, t.scale));
  return m.multiply(up);
}

/** Inside the Canvas: what the tools draw, plus the capture bridge. */
export function ViewToolsScene({ vt, onGizmoDrag }: { vt: ViewTools; onGizmoDrag?: (dragging: boolean) => void }) {
  const runKey = `${vt.particles.pattern}|${vt.particles.count}|${vt.band}|${vt.particles.maxBounces}|${vt.particles.trails}|${vt.source?.id ?? ''}`;
  const v = vt.visual;
  return (
    <>
      <CaptureBridge api={vt.captureApi} />
      {vt.tool === 'particles' && vt.trace && vt.alphas && vt.source && vt.play !== 'stopped' && (
        <ParticleAnimation
          trace={vt.trace}
          alpha={vt.alphas.alpha}
          source={vt.source.position}
          options={vt.particles}
          play={vt.play}
          runKey={runKey}
          sceneRadius={vt.bboxRadius}
          onStats={vt.setStats}
        />
      )}
      {vt.tool === 'density' && vt.densityResult && (
        <ReflectionDensity result={vt.densityResult} orders={vt.density.orders} layerFilter={vt.densityFilter} hideRays={vt.density.hideRays} xray={vt.density.xray} />
      )}
      {v.url && v.shown && !v.loadError && (
        <VisualModel
          url={v.url}
          transform={v.draft}
          opacity={v.opacity}
          lighting={v.lighting}
          gizmo={v.align}
          onChange={v.changeTransform}
          onGizmoDrag={onGizmoDrag}
          onError={(msg) => v.setLoadError(msg)}
          onLoaded={v.setModelBox}
        />
      )}
    </>
  );
}

function Panel({ title, onClose, children, className = '' }: { title: ReactNode; onClose: () => void; children: ReactNode; className?: string }) {
  return (
    <div className={`tool-panel ${className}`} role="dialog" aria-label={typeof title === 'string' ? title : undefined}>
      <div className="tool-panel-head">
        <b>{title}</b>
        <div className="grow" />
        <button className="icon-btn" aria-label="Close" onClick={onClose}>
          ×
        </button>
      </div>
      <div className="tool-panel-body">{children}</div>
    </div>
  );
}

function SourceSelect({ vt, sim }: { vt: ViewTools; sim: Simulation }) {
  if (!sim.sources.length) return <div className="warn-box small">Add a source first (Sources tab).</div>;
  return (
    <label className="tool-row">
      <span>Source</span>
      <select value={vt.source?.id ?? ''} onChange={(e) => vt.setSourceId(e.target.value)} aria-label="Source">
        {sim.sources.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function BandPills({ value, onChange, with63 }: { value: number; onChange: (b: number) => void; with63: boolean }) {
  return (
    <div className="tool-pills" role="radiogroup" aria-label="Octave band">
      {TOOL_BANDS.filter((b) => with63 || b.index >= 0).map((b) => (
        <button key={b.index} role="radio" aria-checked={value === b.index} className={`chip-btn ${value === b.index ? 'active' : ''}`} onClick={() => onChange(b.index)}>
          {toolBandLabel(b.index)}
        </button>
      ))}
    </div>
  );
}

/** Notes shared by the tools that use the layer α: fallbacks are said out loud. */
function AlphaNotes({ alphas, band, labelOf }: { alphas: LayerAlphas | null; band: number; labelOf: (n: string) => string }) {
  if (!alphas) return null;
  const list = (names: string[]) => names.slice(0, 3).map(labelOf).join(', ') + (names.length > 3 ? ', …' : '');
  return (
    <>
      {alphas.unassigned.length > 0 && (
        <div className="warn-box small">
          {alphas.unassigned.length} layer{alphas.unassigned.length === 1 ? '' : 's'} without material, traced as rigid (α = 0): {list(alphas.unassigned)}
        </div>
      )}
      {band < 0 && alphas.fallback63.length > 0 && (
        <div className="warn-box small">
          63 Hz: {alphas.fallback63.length} layer{alphas.fallback63.length === 1 ? '' : 's'} whose material has no 63 Hz value use the 125 Hz value ({list(alphas.fallback63)}).
        </div>
      )}
    </>
  );
}

function RampLegend({ left, mid, right, title }: { left: string; mid?: string; right: string; title: string }) {
  return (
    <div className="tool-legend">
      <div className="small muted">{title}</div>
      <div className="abs-bar" style={{ background: rampGradientCss() }} />
      <div className="abs-ticks small">
        <span>{left}</span>
        {mid && <span>{mid}</span>}
        <span>{right}</span>
      </div>
    </div>
  );
}

export function ParticlesPanel({ vt, sim, labelOf }: { vt: ViewTools; sim: Simulation; labelOf: (n: string) => string }) {
  const p = vt.particles;
  const set = (patch: Partial<ParticleOptions>) => vt.setParticles((x) => ({ ...x, ...patch }));
  const moving = !!vt.source?.path;
  return (
    <Panel title="Particle animation" onClose={() => vt.setTool(null)}>
      <SourceSelect vt={vt} sim={sim} />
      <div className="row gap tool-play">
        <button className="btn btn-sm btn-primary" disabled={!vt.source || !vt.trace} onClick={() => vt.setPlay(vt.play === 'playing' ? 'paused' : 'playing')}>
          {vt.play === 'playing' ? 'Pause' : 'Play'}
        </button>
        <button className="btn btn-sm" disabled={vt.play === 'stopped'} onClick={() => vt.setPlay('stopped')} title="Stop and clear the particles">
          Stop
        </button>
        <span className="small muted grow">
          {vt.play !== 'stopped' && vt.stats ? `${vt.stats.alive} / ${vt.stats.total} alive · up to order ${vt.stats.maxOrder}` : ''}
        </span>
      </div>
      {vt.play !== 'stopped' && vt.stats && vt.stats.alive === 0 && !p.loop && <div className="small muted">All particles have died. Play again with Stop, then Play.</div>}
      {vt.stats?.trailsFull && vt.play !== 'stopped' && <div className="warn-box small">Path buffer full (250 000 segments): older paths stay, new ones are not drawn.</div>}
      <div className="tool-label">Octave (absorption at each hit)</div>
      <BandPills value={vt.band} onChange={vt.setBand} with63 />
      <label className="tool-row">
        <span>Emission</span>
        <Segmented<EmitPattern>
          value={p.pattern}
          onChange={(v) => set({ pattern: v })}
          options={[
            { value: 'sphere', label: 'Sphere', title: 'Every direction (omnidirectional)' },
            { value: 'xy', label: 'XY', title: 'A fan in the horizontal plane' },
            { value: 'xz', label: 'XZ' },
            { value: 'yz', label: 'YZ' },
          ]}
        />
      </label>
      <label className="tool-row">
        <span>Speed</span>
        <select value={p.speed} onChange={(e) => set({ speed: Number(e.target.value) })} aria-label="Speed">
          {SPEEDS.map((s) => (
            <option key={s.v} value={s.v}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
      <label className="tool-row">
        <span>Particles</span>
        <input type="range" min={50} max={2000} step={50} value={p.count} onChange={(e) => set({ count: Number(e.target.value) })} aria-label="Number of particles" />
        <NumField value={p.count} min={50} max={2000} step={50} digits={0} onChange={(v) => v !== null && set({ count: v })} width={56} ariaLabel="Number of particles" />
      </label>
      <label className="tool-row">
        <span>Diameter (mm)</span>
        <input type="range" min={2} max={200} step={1} value={p.diameterMm} onChange={(e) => set({ diameterMm: Number(e.target.value) })} aria-label="Particle diameter" />
        <NumField value={p.diameterMm} min={2} max={200} step={5} digits={0} onChange={(v) => v !== null && set({ diameterMm: v })} width={56} ariaLabel="Particle diameter (mm)" />
      </label>
      <label className="tool-row">
        <span>Max. reflections</span>
        <NumField value={p.maxBounces} min={1} max={500} step={5} digits={0} onChange={(v) => v !== null && set({ maxBounces: v })} width={56} ariaLabel="Maximum number of reflections" />
      </label>
      <label className="tool-row">
        <span>Colour</span>
        <Segmented<'order' | 'absorption'>
          value={p.colorBy}
          onChange={(v) => set({ colorBy: v })}
          options={[
            { value: 'order', label: 'Order', title: 'Number of reflections so far' },
            { value: 'absorption', label: 'Absorption', title: 'Energy lost at the hits so far (dB)' },
          ]}
        />
      </label>
      {p.colorBy === 'order' ? (
        <RampLegend title="Reflections" left="0" mid="5" right={`${ORDER_MAX}+`} />
      ) : (
        <RampLegend title={`Cumulative absorption at ${toolBandLabel(vt.band)} Hz`} left="0 dB" mid={`−${ABS_DB_MIN / 2}`} right={`−${ABS_DB_MIN} dB`} />
      )}
      <div className="tool-toggles">
        <Toggle checked={p.trails} onChange={(v) => set({ trails: v })} label="Path tracing" title="Draw the path of every particle, reflections included" />
        <Toggle checked={p.loop} onChange={(v) => set({ loop: v })} label="Repeat" title="Emit a new burst when every particle has died" />
        <Toggle checked={p.xray} onChange={(v) => set({ xray: v })} label="Through walls" title="Draw the particles over the surfaces in front of them" />
      </div>
      <AlphaNotes alphas={vt.alphas} band={vt.band} labelOf={labelOf} />
      <div className="small muted tool-note">
        Specular reflections on the solver mesh (objects included), energy × (1 − α) at each hit; a particle dies below −{ABS_DB_MIN} dB, after the maximum number of
        reflections, or when it leaves an open model. Omnidirectional emission, no scattering{moving ? '; a moving source emits from its start point' : ''}.
      </div>
    </Panel>
  );
}

export function DensityPanel({ vt, sim, selCount }: { vt: ViewTools; sim: Simulation; selCount: number }) {
  const d = vt.density;
  const set = (patch: Partial<DensityOptions>) => vt.setDensity((x) => ({ ...x, ...patch }));
  const counts = useMemo(() => {
    const r = vt.densityResult;
    if (!r) return null;
    return [0, 1, 2, 3].map((k) => {
      if (!vt.densityFilter) return r.layers[k].length;
      let n = 0;
      for (const l of r.layers[k]) if (vt.densityFilter.has(l)) n++;
      return n;
    });
  }, [vt.densityResult, vt.densityFilter]);
  return (
    <Panel title="Reflection density" onClose={() => vt.setTool(null)}>
      <SourceSelect vt={vt} sim={sim} />
      <div className="tool-label">Orders (cumulative)</div>
      <div className="tool-pills">
        {[0, 1, 2, 3].map((k) => (
          <button
            key={k}
            className={`chip-btn ${d.orders[k] ? 'active' : ''}`}
            aria-pressed={d.orders[k]}
            title={k === 0 ? 'Order 0: where the direct sound lands' : `Points reached after ${k} specular reflection${k === 1 ? '' : 's'}`}
            onClick={() => set({ orders: d.orders.map((x, i) => (i === k ? !x : x)) })}
          >
            <span className="swatch" style={{ background: rampCss(DENSITY_ORDER_T[k]) }} />
            {k}
            {counts && <span className="muted"> · {counts[k]}</span>}
          </button>
        ))}
      </div>
      <label className="tool-row">
        <span>Rays</span>
        <select value={d.rays} onChange={(e) => set({ rays: Number(e.target.value) })} aria-label="Number of rays">
          {[1000, 2000, 4000, 10000, 20000].map((n) => (
            <option key={n} value={n}>
              {n.toLocaleString('en-US')}
            </option>
          ))}
        </select>
      </label>
      <label className="tool-row">
        <span>Surfaces</span>
        <Segmented<'all' | 'sel'>
          value={d.onlySelected ? 'sel' : 'all'}
          onChange={(v) => set({ onlySelected: v === 'sel' })}
          options={[
            { value: 'all', label: 'All' },
            { value: 'sel', label: `Selected layers (${selCount})`, title: 'Select layers in the 3D view or the Materials tab (Ctrl-click for several)' },
          ]}
        />
      </label>
      {d.onlySelected && selCount === 0 && <div className="small muted">Select layers in the view or in the Materials tab (Ctrl-click adds).</div>}
      <div className="tool-toggles">
        <Toggle checked={d.hideRays} onChange={(v) => set({ hideRays: v })} label="Hide rays" title="Only the hit points" />
        <Toggle checked={d.xray} onChange={(v) => set({ xray: v })} label="Through walls" />
      </div>
      <div className="small muted tool-note">
        Specular rays from the source, evenly spread over the sphere. Each dot is where a ray lands: order 0 is the direct sound, 1 to 3 the reflections that follow
        (geometry only, absorption ignored). Dense spots show where a reflector would catch energy.
      </div>
    </Panel>
  );
}

/** Legend extras of the α colouring: the selected surfaces' exact values on the scale. */
export function AbsorptionMarks({
  selected,
  valueOf,
  labelOf,
  band,
  fallback63,
}: {
  selected: string[];
  valueOf: (n: string) => number | null;
  labelOf: (n: string) => string;
  band: number;
  fallback63: number;
}) {
  const shown = selected.slice(0, 4);
  return (
    <>
      {shown.length > 0 && (
        <div className="abs-marks">
          {shown.map((n) => {
            const v = valueOf(n);
            return (
              <div key={n} className="abs-mark-row small">
                <span className="abs-mark-name" title={labelOf(n)}>
                  {labelOf(n)}
                </span>
                <b>{v === null ? 'unassigned' : `α = ${fmtNum(v, 2)}`}</b>
              </div>
            );
          })}
          {selected.length > shown.length && <div className="small muted">+ {selected.length - shown.length} more</div>}
        </div>
      )}
      {band < 0 && fallback63 > 0 && (
        <div className="small text-warn">
          {fallback63} material{fallback63 === 1 ? '' : 's'} without a 63 Hz value: 125 Hz shown
        </div>
      )}
    </>
  );
}

/** Ticks on the α bar for the selected surfaces. */
export function AbsorptionTicks({ values, max }: { values: number[]; max: number }) {
  return (
    <div className="abs-tick-layer">
      {values.slice(0, 4).map((v, i) => (
        <span key={i} className="abs-tick" style={{ left: `${Math.min(100, Math.max(0, (v / max) * 100))}%` }} />
      ))}
    </div>
  );
}

/** The exported region over the viewport, with its output size. */
export function CaptureFrame({ vt }: { vt: ViewTools }) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  const c = vt.capture;
  const f = size ? captureFrame(size.w, size.h, c.width, c.height) : null;
  return (
    <div className="capture-frame-layer" ref={ref}>
      {f && size && (
        <>
          <div className="capture-dim" style={{ left: 0, top: 0, width: size.w, height: f.y }} />
          <div className="capture-dim" style={{ left: 0, top: f.y + f.h, width: size.w, height: Math.max(0, size.h - f.y - f.h) }} />
          <div className="capture-dim" style={{ left: 0, top: f.y, width: f.x, height: f.h }} />
          <div className="capture-dim" style={{ left: f.x + f.w, top: f.y, width: Math.max(0, size.w - f.x - f.w), height: f.h }} />
          <div className="capture-frame" style={{ left: f.x, top: f.y, width: f.w, height: f.h }}>
            <span className="capture-size">
              {c.width} × {c.height} px
            </span>
          </div>
        </>
      )}
    </div>
  );
}

const PRESETS: { label: string; w: number; h: number }[] = [
  { label: 'Full HD 1920 × 1080', w: 1920, h: 1080 },
  { label: '4K 3840 × 2160', w: 3840, h: 2160 },
  { label: 'HD 1280 × 720', w: 1280, h: 720 },
  { label: 'Square 2048 × 2048', w: 2048, h: 2048 },
  { label: 'A4 landscape 3508 × 2480', w: 3508, h: 2480 },
];

/** Side panel of the capture mode (replaces the editor panel while open). */
export function CapturePanel({ vt, simName }: { vt: ViewTools; simName: string }) {
  const c = vt.capture;
  const [lock, setLock] = useState(true);
  const set = (patch: Partial<CaptureOptions>) => vt.setCapture((x) => ({ ...x, ...patch }));
  const clamp = (v: number) => Math.round(Math.min(CAPTURE_MAX, Math.max(CAPTURE_MIN, v)));
  const setW = (w: number) => {
    if (!lock) return set({ width: clamp(w) });
    const r = c.height / c.width;
    const w2 = clamp(Math.min(w, CAPTURE_MAX / Math.max(r, 1e-6)));
    set({ width: w2, height: clamp(w2 * r) });
  };
  const setOpen = vt.setCaptureOpen;
  const setH = (h: number) => {
    if (!lock) return set({ height: clamp(h) });
    const r = c.width / c.height;
    const h2 = clamp(Math.min(h, CAPTURE_MAX / Math.max(r, 1e-6)));
    set({ height: h2, width: clamp(h2 * r) });
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !document.querySelector('.modal-overlay')) {
        e.preventDefault();
        e.stopImmediatePropagation();
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [setOpen]);
  return (
    <aside className="side-panel capture-side" aria-label="Capture mode">
      <div className="capture-head">
        <b>Capture mode</b>
        <div className="grow" />
        <button className="btn btn-xs" onClick={() => vt.setCaptureOpen(false)} title="Leave the capture mode (Esc)">
          Close
        </button>
      </div>
      <div className="panel-scroll capture-body">
        <div className="small muted">Orbit, pan and zoom the view: the frame shows what is exported.</div>
        <div className="tool-label">Output size (px)</div>
        <div className="row gap">
          <NumField value={c.width} min={CAPTURE_MIN} max={CAPTURE_MAX} step={10} digits={0} onChange={(v) => v !== null && setW(v)} width={70} ariaLabel="Width (px)" />
          <span className="muted">×</span>
          <NumField value={c.height} min={CAPTURE_MIN} max={CAPTURE_MAX} step={10} digits={0} onChange={(v) => v !== null && setH(v)} width={70} ariaLabel="Height (px)" />
          <Toggle checked={lock} onChange={setLock} label="Lock ratio" />
        </div>
        <select
          className="capture-presets"
          value=""
          aria-label="Size presets"
          onChange={(e) => {
            const p = PRESETS[Number(e.target.value)];
            if (p) set({ width: p.w, height: p.h });
          }}
        >
          <option value="">Presets…</option>
          {PRESETS.map((p, i) => (
            <option key={p.label} value={i}>
              {p.label}
            </option>
          ))}
        </select>
        <div className="small muted">Each side from {CAPTURE_MIN} to {CAPTURE_MAX} px.</div>
        <div className="tool-label">Background</div>
        <div className="row gap">
          <Segmented<CaptureOptions['background']>
            value={c.background}
            onChange={(v) => set({ background: v })}
            options={[
              { value: 'theme', label: 'Theme' },
              { value: 'white', label: 'White' },
              { value: 'transparent', label: 'Transparent', title: 'PNG with an alpha channel' },
              { value: 'custom', label: 'Colour' },
            ]}
          />
          {c.background === 'custom' && <input type="color" value={c.customColor} onChange={(e) => set({ customColor: e.target.value })} aria-label="Background colour" />}
        </div>
        <div className="tool-label">Show</div>
        <div className="tool-toggles column">
          <Toggle checked={c.showSources} onChange={(v) => set({ showSources: v })} label="Sources" />
          <Toggle checked={c.showReceivers} onChange={(v) => set({ showReceivers: v })} label="Receivers" />
          <Toggle checked={c.showLabels} onChange={(v) => set({ showLabels: v })} label="Labels" />
          <Toggle checked={c.showGrid} onChange={(v) => set({ showGrid: v })} label="Floor grid" />
          <Toggle checked={c.showGizmos} onChange={(v) => set({ showGizmos: v })} label="Move / rotate handles" title="The transform gizmo of the selection. The axes gizmo is never exported." />
        </div>
        <div className="small muted">What the view shows otherwise (shading, α colouring, particles, reflection dots, visual model) is exported as is.</div>
        <button className="btn btn-primary btn-block" disabled={vt.capturing} onClick={() => void vt.exportPng(simName)}>
          {vt.capturing ? <Spinner /> : null} Save PNG
        </button>
      </div>
    </aside>
  );
}

/** Upload, show and align the display-only .glb of the geometry. */
export function VisualPanel({ vt, confirm }: { vt: ViewTools; confirm: (title: string, message: ReactNode) => Promise<boolean> }) {
  const v = vt.visual;
  const file = useRef<HTMLInputElement>(null);
  const t = v.draft;
  const setT = (patch: Partial<VisualTransform>) => v.changeTransform({ ...t, ...patch });
  const vec = (key: 'position' | 'rotation_deg', i: number, x: number) => {
    const a = [...t[key]] as Vec3;
    a[i] = x;
    setT({ [key]: a } as Partial<VisualTransform>);
  };
  const busy = v.state === 'uploading';
  return (
    <Panel title="Visual model (.glb)" onClose={() => {
      v.setPanel(false);
      v.setAlign(null);
    }} className="visual-panel">
      <input
        ref={file}
        type="file"
        accept=".glb,model/gltf-binary"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (f) void v.upload(f);
        }}
      />
      {!v.info ? (
        <>
          <div className="small muted">
            A binary glTF of the room (≤ 200 MB), shown over the acoustic model for display only: the solvers never see it.
          </div>
          <button className="btn btn-sm btn-primary" disabled={busy} onClick={() => file.current?.click()}>
            {busy ? `Uploading ${Math.round(v.uploadFrac * 100)} %…` : 'Upload .glb…'}
          </button>
        </>
      ) : (
        <>
          <div className="small">
            <b>{v.info.filename}</b> <span className="muted">· {fmtNum(v.info.bytes / 1e6, 1)} MB</span>
          </div>
          {v.loadError && <div className="warn-box small">The model could not be displayed: {v.loadError}</div>}
          <div className="tool-toggles">
            <Toggle checked={v.shown} onChange={v.setShown} label="Show" />
          </div>
          <label className="tool-row">
            <span>Opacity</span>
            <input type="range" min={0.1} max={1} step={0.05} value={v.opacity} onChange={(e) => v.setOpacity(Number(e.target.value))} aria-label="Opacity" />
            <span className="small muted">{Math.round(v.opacity * 100)} %</span>
          </label>
          <LightingControls vt={vt} />
          <div className="tool-label">Acoustic model (room surfaces)</div>
          <div className="row">
            <Segmented<RoomDisplay>
              value={v.roomDisplay}
              onChange={v.setRoomDisplay}
              options={[
                { value: 'normal', label: 'Shown' },
                { value: 'overlay', label: 'Translucent' },
                { value: 'invisible', label: 'Hidden', title: 'Not drawn, still clickable' },
              ]}
            />
          </div>
          <div className="tool-label">Alignment</div>
          <Segmented<'off' | VisualGizmoMode>
            value={v.align ?? 'off'}
            onChange={(m) => v.setAlign(m === 'off' ? null : m)}
            disabled={!v.shown}
            options={[
              { value: 'off', label: 'Off' },
              { value: 'translate', label: 'Move' },
              { value: 'rotate', label: 'Rotate' },
              { value: 'scale', label: 'Scale', title: 'Uniform scale' },
            ]}
          />
          <div className="visual-grid">
            <span className="small muted">Position (m)</span>
            {[0, 1, 2].map((i) => (
              <NumField key={`p${i}`} value={t.position[i]} step={0.05} digits={3} onChange={(x) => x !== null && vec('position', i, x)} ariaLabel={`Position ${'XYZ'[i]}`} />
            ))}
            <span className="small muted">Rotation (°)</span>
            {[0, 1, 2].map((i) => (
              <NumField key={`r${i}`} value={t.rotation_deg[i]} step={1} digits={2} onChange={(x) => x !== null && vec('rotation_deg', i, x)} ariaLabel={`Rotation ${'XYZ'[i]}`} />
            ))}
            <span className="small muted">Scale</span>
            <NumField value={t.scale} min={1e-4} max={1e4} step={0.01} digits={5} onChange={(x) => x !== null && setT({ scale: x })} ariaLabel="Scale" />
            <span className="small muted visual-up">Up axis</span>
            <Segmented<'y' | 'z'>
              value={t.up_axis}
              onChange={(a) => setT({ up_axis: a })}
              options={[
                { value: 'y', label: 'Y', title: 'glTF convention (Y up), converted to Z up' },
                { value: 'z', label: 'Z' },
              ]}
            />
          </div>
          <div className="small muted">Order: up axis, scale, rotation X then Y then Z, translation. Saved with the geometry.</div>
          <div className="row gap wrap">
            <button className="btn btn-xs" disabled={!v.modelBox} onClick={v.centreOnRoom} title="Centre the model on the room in plan and put its lowest point on the room's floor level">
              Centre on the room
            </button>
            <button className="btn btn-xs" onClick={() => v.changeTransform({ position: [0, 0, 0], rotation_deg: [0, 0, 0], scale: 1, up_axis: t.up_axis })}>
              Reset
            </button>
            <button className="btn btn-xs" disabled={busy} onClick={() => file.current?.click()}>
              {busy ? `Uploading ${Math.round(v.uploadFrac * 100)} %…` : 'Replace…'}
            </button>
            <button
              className="btn btn-xs btn-danger-ghost"
              onClick={async () => {
                if (await confirm('Remove the visual model', <>Remove <b>{v.info?.filename}</b> from this geometry?</>)) void v.remove();
              }}
            >
              Remove
            </button>
          </div>
          <div className={`small ${v.state === 'error' ? 'text-warn' : 'muted'}`}>
            {v.state === 'saving' ? 'Saving…' : v.state === 'error' ? `Not saved: ${v.error}` : 'Alignment saved'}
          </div>
        </>
      )}
    </Panel>
  );
}

function LightSlider({
  label,
  value,
  limits,
  step,
  unit,
  digits,
  onChange,
  disabled,
  title,
}: {
  label: string;
  value: number;
  limits: readonly [number, number];
  step: number;
  unit?: string;
  digits: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <label className="tool-row" title={title}>
      <span>{label}</span>
      <input type="range" min={limits[0]} max={limits[1]} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} aria-label={label} />
      <NumField value={value} min={limits[0]} max={limits[1]} step={step} digits={digits} onChange={(x) => x !== null && onChange(x)} width={54} ariaLabel={`${label}${unit ? ` (${unit})` : ''}`} />
    </label>
  );
}

/** Lighting of the visual model: presets, environment, a sun on the model only, the view's lights, exposure. */
function LightingControls({ vt }: { vt: ViewTools }) {
  const v = vt.visual;
  const l = v.lighting;
  const set = (patch: Partial<VisualLighting>) => v.changeLighting(patch);
  const noSun = l.sun_intensity === 0;
  const s = v.lightingSave;
  return (
    <>
      <div className="tool-label">Lighting of the model</div>
      <label className="tool-row">
        <span>Preset</span>
        <select value={l.preset} onChange={(e) => v.applyLightingPreset(e.target.value as LightingPreset)} aria-label="Lighting preset">
          {LIGHTING_PRESETS.map((p) => (
            <option key={p.id} value={p.id} title={p.title}>
              {p.label}
            </option>
          ))}
          {l.preset === 'custom' && <option value="custom">Custom</option>}
        </select>
      </label>
      <LightSlider label="Environment" value={l.env_intensity} limits={LIGHTING_LIMITS.env_intensity} step={0.05} digits={2} onChange={(x) => set({ env_intensity: x })} title="Soft image-based light (0.8 = the lighting of before)" />
      <LightSlider label="Sun" value={l.sun_intensity} limits={LIGHTING_LIMITS.sun_intensity} step={0.1} digits={1} onChange={(x) => set({ sun_intensity: x })} title="A directional light on the model only (0 = none)" />
      <LightSlider label="Sun azimuth" value={l.sun_azimuth_deg} limits={LIGHTING_LIMITS.sun_azimuth_deg} step={1} unit="°" digits={0} disabled={noSun} onChange={(x) => set({ sun_azimuth_deg: x })} title="Where the sunlight comes from, in degrees counter-clockwise from +X" />
      <LightSlider label="Sun elevation" value={l.sun_elevation_deg} limits={LIGHTING_LIMITS.sun_elevation_deg} step={1} unit="°" digits={0} disabled={noSun} onChange={(x) => set({ sun_elevation_deg: x })} title="Height of the sun above the horizontal, in degrees" />
      <label className="tool-row">
        <span>Sun colour</span>
        <input type="color" value={l.sun_color} disabled={noSun} onChange={(e) => set({ sun_color: e.target.value })} aria-label="Sun colour" />
      </label>
      <LightSlider label="Exposure" value={l.exposure_ev} limits={LIGHTING_LIMITS.exposure_ev} step={0.1} unit="EV" digits={1} onChange={(x) => set({ exposure_ev: x })} title="Brightness of the model in stops (+1 = twice as bright)" />
      <div className="tool-toggles">
        <Toggle checked={l.view_lights} onChange={(x) => set({ view_lights: x })} label="View lights" title="The fixed lights of the 3D view (the ones on the room surfaces) also light the model" />
        <button className="btn btn-xs" disabled={l.preset === 'default'} onClick={() => v.applyLightingPreset('default')} title="The fixed lighting of before">
          Reset
        </button>
      </div>
      {!LIGHTING_PATCH_OK && <div className="warn-box small">This three.js version is not supported by the sun and View lights controls: only the environment and the exposure apply.</div>}
      <div className={`small ${s.error ? 'text-warn' : 'muted'}`}>
        Display only: the room surfaces, the markers and the solvers are not affected.{' '}
        {s.error
          ? `Lighting not saved: ${s.error}.`
          : s.where === 'server'
            ? 'Saved with the geometry.'
            : 'Saved in this browser only, for this geometry (the server does not store the lighting yet).'}
      </div>
    </>
  );
}

/** The "S" key opens the capture mode (outside text fields and dialogs). */
export function useCaptureShortcut(vt: ViewTools, enabled: boolean) {
  const setOpen = vt.setCaptureOpen;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.key.toLowerCase() !== 's') return;
      if (t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName))) return;
      if (document.querySelector('.modal-overlay')) return;
      setOpen(true);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setOpen, enabled]);
}
