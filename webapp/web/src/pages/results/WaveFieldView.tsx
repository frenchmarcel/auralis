import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { Html } from '@react-three/drei';
import { api, editableOf, errorMessage, type CatalogObject, type FieldFrames, type FieldTask, type GridPoints, type MeshData, type PlacedObject, type ReceiverGrid, type Simulation, type Vec3 } from '../../api';
import { useApp } from '../../AppContext';
import { Plot } from '../../components/Plot';
import { EmptyState, ErrorBox, NumField, ProgressBar, Segmented, Spinner, StatusBadge, Toggle } from '../../components/ui';
import { loadCatalog, loadGeometry, loadMesh } from '../../lib/cache';
import { layerColor, seriesColor } from '../../lib/colors';
import { safeFilename, type DataTable } from '../../lib/export';
import { fmtDate, fmtDuration, fmtNum } from '../../lib/format';
import { boundsOf, boundsOfPositions, dirFromAzEl, type Bounds } from '../../lib/geom';
import { DISABLED_COLOR, newPlaneGrid, nextGridId, rampCss } from '../../lib/grids';
import { href } from '../../lib/router';
import { getSlice, setSlice } from '../../lib/savedViews';
import { FieldMap } from '../../three/FieldMap';
import { CaptureBridge, type CaptureFn } from '../../three/GridMap';
import { PlacedObjects } from '../../three/Objects';
import { LayerMeshes, PointMarkers, SceneCanvas, markerRadius, type FrameRequest, type PointMarker } from '../../three/Scene';
import { mapPng } from './mapPng';
import { ACTIVE_STATUSES, useTaskJobs } from './taskJobs';

/** Server rules (API "Wave-field animation"). */
export const FIELD_F_MIN = 50;
export const FIELD_F_MAX = 1000;
export const FIELD_D_MIN = 0.02;
export const FIELD_D_MAX = 1;
export const FIELD_MAX_FRAMES = 2000;
export const FIELD_FPS_FACTOR = 2.5;
/** Wave-solver cell for a given highest frequency (server: dx = c / (13.5 f_max)). */
const CELLS_PER_WAVELENGTH = 13.5;
const C_AIR = 343;
/** Playback speeds (fraction of real time). */
const SPEEDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25];
const DEFAULT_SPEED = 0.01;
const MAX_TRACES = 4;
/** Diverging ramp around 0: negative blue, positive red, silence dark (distinct from the grey of left-out points). */
export const FIELD_RAMP = ['#9ec5f4', '#3987e5', '#1c5cab', '#25282c', '#a83a3b', '#e34948', '#f2a09d'];
/** Default gain: the full scale is the largest pressure of the animation (next to the source), so the room's field needs some. */
const DEFAULT_GAIN = 4;
/** Points of a suggested ear-height plane (enough for the field, cheap enough for the wave solver). */
const EAR_PLANE_POINTS = 600;

interface FieldView {
  taskId?: string | null;
  gain?: number;
  speed?: number;
  loop?: boolean;
  interp?: boolean;
  relief?: boolean;
  top?: boolean;
  form?: Partial<FormState>;
}

interface FormState {
  sourceId: string;
  gridId: string;
  fMax: number;
  duration: number;
  fpsAuto: boolean;
  fps: number;
  label: string;
}

/** Frames per second the form sends, the number of frames, or why the server would refuse. */
export function fieldForm(f: Pick<FormState, 'fMax' | 'duration' | 'fpsAuto' | 'fps'>): { fps: number; frames: number; minFps: number; maxFps: number; error: string | null } {
  const minFps = Math.ceil(FIELD_FPS_FACTOR * f.fMax - 1e-9);
  const maxFps = f.duration > 0 ? Math.floor(FIELD_MAX_FRAMES / f.duration + 1e-9) : 0;
  const fps = f.fpsAuto ? Math.min(Math.max(400, Math.ceil(4 * f.fMax)), maxFps) : Math.round(f.fps);
  const frames = Math.round(f.duration * fps);
  const out = { fps, frames, minFps, maxFps };
  if (!(f.fMax >= FIELD_F_MIN && f.fMax <= FIELD_F_MAX)) return { ...out, error: `f max must be within ${FIELD_F_MIN}–${FIELD_F_MAX} Hz.` };
  if (!(f.duration >= FIELD_D_MIN && f.duration <= FIELD_D_MAX)) return { ...out, error: `The duration must be within ${FIELD_D_MIN}–${FIELD_D_MAX} s.` };
  if (minFps > maxFps)
    return { ...out, error: `${fmtNum(f.fMax, 0)} Hz needs ${minFps} frames/s, but ${fmtNum(f.duration, 2)} s allows ${maxFps} at most (${FIELD_MAX_FRAMES} frames): shorten the duration or lower f max.` };
  if (fps < minFps) return { ...out, error: `At least ${minFps} frames/s (${FIELD_FPS_FACTOR} × f max).` };
  if (f.duration * fps > FIELD_MAX_FRAMES + 1e-9) return { ...out, error: `${frames} frames: at most ${FIELD_MAX_FRAMES} (duration × frames/s).` };
  return { ...out, error: null };
}

export function fmtSpeed(s: number): string {
  return `×${s}`;
}

interface GeomView {
  mesh: MeshData;
  bounds: Bounds;
  placed: PlacedObject[];
}

function useGeom(gid: string | null): GeomView | null {
  const [v, setV] = useState<GeomView | null>(null);
  useEffect(() => {
    if (!gid) return;
    let cancelled = false;
    Promise.all([loadGeometry(gid), loadMesh(gid)])
      .then(([g, m]) => !cancelled && setV({ mesh: m, bounds: boundsOf(g.bbox.min, g.bbox.max), placed: g.objects ?? [] }))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [gid]);
  return v;
}

/** Decoded frames of one task. */
interface FieldData {
  rid: string;
  points: Vec3[];
  valid: boolean[];
  normals: Vec3[];
  uAxes: Vec3[];
  nFrames: number;
  nPoints: number;
  scale: number;
  data: Int8Array;
}

function decodeFrames(rid: string, f: FieldFrames): FieldData {
  const bin = atob(f.data_b64);
  if (bin.length !== f.n_frames * f.n_points) throw new Error(`frames: ${bin.length} bytes for ${f.n_frames} frames × ${f.n_points} points`);
  const data = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const c = bin.charCodeAt(i);
    data[i] = c > 127 ? c - 256 : c;
  }
  return { rid, points: f.points, valid: f.valid, normals: f.normals, uAxes: f.u_axes, nFrames: f.n_frames, nPoints: f.n_points, scale: f.scale, data };
}

const framesCache = new Map<string, Promise<FieldData>>();
function loadFrames(simId: string, rid: string): Promise<FieldData> {
  const k = `${simId}|${rid}`;
  let p = framesCache.get(k);
  if (!p) {
    p = api.getFieldFrames(simId, rid).then((f) => decodeFrames(rid, f));
    framesCache.set(k, p);
    p.catch(() => framesCache.delete(k));
    // Up to 2000 frames x 2500 points each: keep the last few only.
    while (framesCache.size > 3) framesCache.delete(framesCache.keys().next().value!);
  }
  return p;
}

/** Lattice step from the points themselves (grid deleted or edited since the task). */
function estimateSpacing(points: Vec3[]): number {
  let best = Infinity;
  const n = Math.min(points.length, 60);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const d = Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1], points[i][2] - points[j][2]);
      if (d > 1e-6 && d < best) best = d;
    }
  return Number.isFinite(best) ? best : 1;
}

const planeCount = (w: number, l: number, s: number) => Math.max(1, Math.floor(w / s + 1e-9)) * Math.max(1, Math.floor(l / s + 1e-9));
const autoEarSpacing = (w: number, l: number) => Math.max(0.1, Math.ceil(Math.sqrt((w * l) / EAR_PLANE_POINTS) * 20) / 20);

/** Wave-field animation: launch, list, and play a task's pressure frames on its grid points. */
export function WaveFieldView({ simId, sim, onSimulationChanged }: { simId: string; sim: Simulation; onSimulationChanged?: () => void }) {
  const app = useApp();
  const saved = useMemo(() => getSlice<FieldView>(simId, 'results', 'field') ?? {}, [simId]);
  const load = useCallback(() => api.listField(simId), [simId]);
  const { tasks, listError, jobs, lost, refresh, addQueued, cancel } = useTaskJobs<FieldTask>(load);
  const [taskId, setTaskId] = useState<string | null>(saved.taskId ?? null);
  const [gain, setGain] = useState(saved.gain ?? DEFAULT_GAIN);
  const [speed, setSpeed] = useState(saved.speed ?? DEFAULT_SPEED);
  const [loop, setLoop] = useState(saved.loop ?? true);
  const [interp, setInterp] = useState(saved.interp ?? true);
  const [relief, setRelief] = useState(saved.relief ?? false);
  const [top, setTop] = useState(saved.top ?? false);
  const [grids, setGrids] = useState<ReceiverGrid[]>(sim.receiver_grids ?? []);
  useEffect(() => setGrids(sim.receiver_grids ?? []), [sim]);
  const firstPlane = (l: ReceiverGrid[]) => l.find((g) => g.kind === 'plane')?.id ?? l[0]?.id ?? '';
  const [form, setForm] = useState<FormState>(() => ({
    sourceId: sim.sources[0]?.id ?? '',
    gridId: firstPlane(sim.receiver_grids ?? []),
    fMax: 250,
    duration: 0.2,
    fpsAuto: true,
    fps: 1000,
    label: '',
    ...(saved.form ?? {}),
  }));
  const setFormPatch = (p: Partial<FormState>) => setForm((f) => ({ ...f, ...p }));
  useEffect(() => {
    const fix: Partial<FormState> = {};
    if (!sim.sources.some((x) => x.id === form.sourceId)) fix.sourceId = sim.sources[0]?.id ?? '';
    if (!grids.some((g) => g.id === form.gridId)) fix.gridId = firstPlane(grids);
    if (Object.keys(fix).length) setForm((f) => ({ ...f, ...fix }));
  }, [sim, grids, form.sourceId, form.gridId]);
  const [awaited, setAwaited] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(
    () => setSlice(simId, 'results', 'field', { taskId, gain, speed, loop, interp, relief, top, form: { ...form, label: '' } }),
    [simId, taskId, gain, speed, loop, interp, relief, top, form],
  );

  const doneTasks = useMemo(() => (tasks ?? []).filter((t) => t.status === 'done' && t.result), [tasks]);
  useEffect(() => {
    if (awaited && doneTasks.some((t) => t.id === awaited)) {
      setTaskId(awaited);
      setAwaited(null);
    }
  }, [awaited, doneTasks]);
  const task = doneTasks.find((t) => t.id === taskId) ?? doneTasks[0] ?? null;
  useEffect(() => {
    if (task && task.id !== taskId) setTaskId(task.id);
  }, [task, taskId]);
  const meta = task?.result ?? null;
  const fps = meta?.fps ?? 1;

  // Frames of the shown task.
  const [field, setField] = useState<{ rid: string; data: FieldData | null; error: string | null } | null>(null);
  useEffect(() => {
    if (!task) return;
    let cancelled = false;
    const rid = task.id;
    loadFrames(simId, rid)
      .then((d) => !cancelled && setField({ rid, data: d, error: null }))
      .catch((e) => !cancelled && setField({ rid, data: null, error: errorMessage(e) }));
    return () => {
      cancelled = true;
    };
  }, [simId, task]);
  const fd = field && task && field.rid === task.id ? field.data : null;
  const fieldError = field && task && field.rid === task.id ? field.error : null;
  const nFrames = fd?.nFrames ?? 0;

  // Player: the time base advances at `speed` x real time; the frame shown is the one at that time.
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const frameRef = useRef(0);
  frameRef.current = frame;
  useEffect(() => {
    setFrame(0);
    setPlaying(false);
  }, [fd]);
  useEffect(() => {
    if (!playing || !nFrames) return;
    let raf = 0;
    let last = performance.now();
    let simT = (frameRef.current >= nFrames - 1 && !loop ? 0 : frameRef.current) / fps;
    const span = nFrames / fps;
    const tick = (now: number) => {
      simT += ((now - last) / 1000) * speed;
      last = now;
      if (simT >= span) {
        if (!loop) {
          setFrame(nFrames - 1);
          setPlaying(false);
          return;
        }
        simT %= span;
      }
      setFrame(Math.min(nFrames - 1, Math.floor(simT * fps + 1e-9)));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, nFrames, fps, speed, loop]);
  const step = (d: number) => {
    setPlaying(false);
    setFrame((f) => Math.max(0, Math.min(nFrames - 1, f + d)));
  };
  const tMs = (frame / fps) * 1000;
  const limit = (fd?.scale ?? 1) / gain;
  const valueAt = (i: number, f = frame) => (fd ? (fd.data[f * fd.nPoints + i] / 127) * fd.scale : 0);

  const gridDef = grids.find((g) => g.id === task?.grid_id) ?? null;
  const points = useMemo(() => fd?.points ?? [], [fd]);
  const spacing = useMemo(() => (gridDef?.spacing_m ? gridDef.spacing_m : estimateSpacing(points)), [gridDef, points]);
  const geom = useGeom(sim.geometry_id);
  const [catalog, setCatalog] = useState<Map<string, CatalogObject>>(new Map());
  useEffect(() => {
    if (!geom?.placed.length) return;
    let cancelled = false;
    loadCatalog()
      .then((c) => !cancelled && setCatalog(new Map(c.objects.map((o) => [o.id, o]))))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [geom]);

  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  useEffect(() => {
    setPicked([]);
    setHover(null);
  }, [task?.id]);
  const pick = (i: number, add: boolean) => setPicked((p) => (p.includes(i) ? p.filter((x) => x !== i) : add ? [...p, i].slice(-MAX_TRACES) : [i]));

  const [frameReq, setFrameReq] = useState<FrameRequest | null>(null);
  const frameKey = useRef(0);
  const capture = useRef<CaptureFn | null>(null);
  const frameGrid = () => {
    const b = points.length ? boundsOfPositions(points.flat()) : null;
    if (b) setFrameReq({ center: b.center, radius: Math.max(1, b.radius * 1.05), key: ++frameKey.current });
  };

  const srcName = (id: string) => sim.sources.find((s) => s.id === id)?.name ?? id;
  const gridName = (id: string) => grids.find((g) => g.id === id)?.name ?? `${id} (grid removed since)`;
  const taskTitle = (t: FieldTask) => t.label || `${srcName(t.source_id)} · ${gridName(t.grid_id)}`;
  const sourceMarkers = useMemo<PointMarker[]>(
    () =>
      sim.sources.map((s) => ({
        kind: 'source' as const,
        id: s.id,
        label: s.name,
        position: s.position,
        dir: dirFromAzEl(s.azimuth_deg, s.elevation_deg),
        selected: s.id === task?.source_id,
        dim: s.id !== task?.source_id,
      })),
    [sim, task],
  );
  // The static part of the scene is the same element from frame to frame: React skips it.
  const staticScene = useMemo(
    () =>
      geom ? (
        <>
          <LayerMeshes mesh={geom.mesh} shading="ghosted" pickable={false} styleOf={(_, i) => ({ color: layerColor(i) })} />
          <PlacedObjects objects={geom.placed} catalog={catalog} showVisual pickable={false} />
          <PointMarkers points={sourceMarkers} radius={markerRadius(geom.bounds)} />
          <CaptureBridge capture={capture} />
        </>
      ) : null,
    [geom, catalog, sourceMarkers],
  );
  const reliefM = relief && points.length ? Math.max(0.15, 0.06 * (boundsOfPositions(points.flat())?.radius ?? 1) * 2) : 0;

  // Pressure traces of the picked points, the cursor following the frame (throttled while playing).
  const [cursorFrame, setCursorFrame] = useState(0);
  const lastCursor = useRef(0);
  useEffect(() => {
    const now = performance.now();
    const wait = playing ? Math.max(0, 120 - (now - lastCursor.current)) : 0;
    const t = window.setTimeout(() => {
      lastCursor.current = performance.now();
      setCursorFrame(frame);
    }, wait);
    return () => window.clearTimeout(t);
  }, [frame, playing]);
  const traces = useMemo<Data[]>(() => {
    if (!fd) return [];
    const t = Array.from({ length: fd.nFrames }, (_, k) => (k / fps) * 1000);
    return picked.map((i, k) => ({
      type: 'scatter',
      mode: 'lines',
      x: t,
      y: t.map((_, f) => (fd.data[f * fd.nPoints + i] / 127) * fd.scale),
      name: `Point ${i + 1}`,
      line: { color: seriesColor(k), width: 1.6 },
      hovertemplate: `Point ${i + 1}<br>%{x:.2f} ms: %{y:.4f}<extra></extra>`,
    }));
  }, [fd, picked, fps]);
  const traceLayout = useMemo<Partial<Layout>>(() => {
    const x = (cursorFrame / fps) * 1000;
    return {
      height: 230,
      margin: { l: 58, r: 12, t: 28, b: 40 },
      xaxis: { title: { text: 'Time after emission (ms) · click to show that frame' } },
      yaxis: { title: { text: 'Pressure (re 1 m)' }, zeroline: true },
      shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: '#f0a83a', width: 1.5, dash: 'dash' } }],
      showlegend: true,
    };
  }, [cursorFrame, fps]);
  const traceTable = (): DataTable => ({
    columns: ['time (ms)', ...picked.map((i) => `point ${i + 1} (pressure re 1 m)`)],
    rows: Array.from({ length: nFrames }, (_, f) => [(f / fps) * 1000, ...picked.map((i) => valueAt(i, f))]),
  });

  const fileBase = () => safeFilename(`${sim.name}_wavefield_${task ? taskTitle(task) : ''}_${fmtNum(tMs, 2)}ms`);
  const exportPng = async () => {
    const cap = capture.current;
    if (!cap || !task) return;
    try {
      await mapPng(cap('#ffffff'), {
        title: `Wave field at t = ${fmtNum(tMs, 2)} ms (frame ${frame + 1} of ${nFrames}) · pressure re 1 m`,
        subtitle: `${sim.name} · ${taskTitle(task)} · wave band up to ${fmtNum(meta?.f_max_hz, 0)} Hz, dx ${fmtNum((meta?.dx_m ?? 0) * 100, 1)} cm, ${meta?.engine ?? ''} · gain ×${fmtNum(gain, 1)}`,
        ramp: FIELD_RAMP,
        lo: `−${fmtLim(limit)}`,
        mid: '0',
        hi: `+${fmtLim(limit)}`,
        filename: `${fileBase()}.png`,
      });
    } catch (e) {
      app.toast('error', `PNG export failed: ${errorMessage(e)}`);
    }
  };

  // Launch form.
  const ff = fieldForm(form);
  const [gridPreview, setGridPreview] = useState<{ id: string; info: GridPoints | null; error: string | null } | null>(null);
  const formGrid = grids.find((g) => g.id === form.gridId) ?? null;
  const formGridSig = formGrid ? JSON.stringify(formGrid) : '';
  useEffect(() => {
    if (!form.gridId) return;
    let cancelled = false;
    api
      .getGridPoints(simId, form.gridId)
      .then((g) => !cancelled && setGridPreview({ id: form.gridId, info: g, error: null }))
      .catch((e) => !cancelled && setGridPreview({ id: form.gridId, info: null, error: errorMessage(e) }));
    return () => {
      cancelled = true;
    };
  }, [simId, form.gridId, formGridSig]);
  const gp = gridPreview?.id === form.gridId ? gridPreview : null;
  const canSubmit = !!form.sourceId && !!form.gridId && !ff.error && !submitting;
  const submit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const r = await api.submitField(simId, { source_id: form.sourceId, grid_id: form.gridId, f_max_hz: form.fMax, duration_s: form.duration, fps: ff.fps, label: form.label.trim() });
      addQueued(r.field, r.job);
      setAwaited(r.field.id);
      setFormPatch({ label: '' });
    } catch (e) {
      setSubmitError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  };
  const onCancel = async (jid: string) => {
    try {
      await cancel(jid);
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };
  const remove = async (t: FieldTask) => {
    const ok = await app.confirm({ title: 'Delete wave-field animation', message: <>Delete the animation <b>{taskTitle(t)}</b>?</>, confirmLabel: 'Delete', danger: true });
    if (!ok) return;
    try {
      await api.deleteField(simId, t.id);
      framesCache.delete(`${simId}|${t.id}`);
      await refresh();
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  // Ear-height plane grid, added to the simulation on request.
  const [earHeight, setEarHeight] = useState(1.2);
  const [earSpacing, setEarSpacing] = useState<number | null>(null);
  const [addingGrid, setAddingGrid] = useState(false);
  const earDims = geom ? { w: Math.max(1, Math.round(geom.bounds.size[0] * 0.8 * 10) / 10), l: Math.max(1, Math.round(geom.bounds.size[1] * 0.8 * 10) / 10) } : null;
  const earS = earSpacing ?? (earDims ? autoEarSpacing(earDims.w, earDims.l) : 0.5);
  const earN = earDims ? planeCount(earDims.w, earDims.l, earS) : 0;
  const addEarPlane = async () => {
    if (!geom || !earDims) return;
    const ok = await app.confirm({
      title: 'Add a plane grid at ear height',
      message: (
        <>
          <p>
            Adds a horizontal plane grid to <b>{sim.name}</b>: {fmtNum(earDims.w, 1)} × {fmtNum(earDims.l, 1)} m centred on the model, {fmtNum(earHeight, 2)} m above its lowest point, spacing {fmtNum(earS, 2)} m (about {earN} points).
          </p>
          <p>Changing the grids marks the results of this simulation as stale: they stay readable, and the next run also computes the room parameters on the new points (about 0.15 s per point and source). Edit or delete the grid later in the editor&apos;s Grids tab.</p>
        </>
      ),
      confirmLabel: 'Add the grid',
    });
    if (!ok) return;
    setAddingGrid(true);
    try {
      const fresh = await api.getSimulation(simId);
      const list = fresh.receiver_grids ?? [];
      const g0 = newPlaneGrid(nextGridId(list), `Ear height ${fmtNum(earHeight, 2)} m`, geom.bounds);
      const grid: ReceiverGrid = {
        ...g0,
        center: [g0.center![0], g0.center![1], Math.round((geom.bounds.min[2] + earHeight) * 100) / 100],
        width_m: earDims.w,
        length_m: earDims.l,
        spacing_m: earS,
      };
      const updated = await api.saveSimulation(simId, { ...editableOf(fresh), receiver_grids: [...list, grid] });
      setGrids(updated.receiver_grids ?? [...list, grid]);
      setFormPatch({ gridId: grid.id });
      app.toast('success', `Grid “${grid.name}” added (${earN} points).`);
      onSimulationChanged?.();
    } catch (e) {
      app.toast('error', `The grid could not be added: ${errorMessage(e)}`);
    } finally {
      setAddingGrid(false);
    }
  };

  const hoverPoint = hover && points[hover.i] ? hover : null;
  const onGpu = app.health?.gpu === true;
  const hasPlane = grids.some((g) => g.kind === 'plane');

  const earCard = (
    <div className="field-ear">
      <div className="small">
        {hasPlane ? 'Another plane at ear height:' : <b>No plane grid yet.</b>} {hasPlane ? '' : 'A horizontal plane at ear height is the usual choice for an animation.'}
      </div>
      <div className="row gap wrap">
        <label className="inline-field" title="Height above the lowest point of the model">
          <span>height</span>
          <NumField value={earHeight} min={0.2} max={10} step={0.1} digits={2} width={52} onChange={(v) => setEarHeight(v ?? 1.2)} ariaLabel="Height (m)" />
          <span className="unit">m</span>
        </label>
        <label className="inline-field" title={`Empty = automatic (about ${EAR_PLANE_POINTS} points)`}>
          <span>spacing</span>
          <NumField value={earSpacing} allowNull placeholder={fmtNum(earS, 2)} min={0.1} max={5} step={0.05} digits={2} width={52} onChange={setEarSpacing} ariaLabel="Spacing (m)" />
          <span className="unit">m</span>
        </label>
        <span className={`small ${earN > 2500 ? 'text-err' : 'muted'}`}>≈ {earN} points</span>
        <button type="button" className="btn btn-xs" disabled={!geom || addingGrid || earN > 2500} onClick={() => void addEarPlane()} title="Adds the grid to this simulation (asks first: its results become stale)">
          {addingGrid ? <Spinner size={10} /> : null} Add to the simulation…
        </button>
      </div>
      <div className="small muted">
        Or draw any grid in the editor (<a href={href.editor(simId)}>Grids tab</a>, “Plane”).
      </div>
    </div>
  );

  const launchCard = (
    <section className="card field-launch">
      <h3>New wave-field animation</h3>
      {!grids.length ? (
        <>
          <div className="small muted">An animation needs a receiver grid: the wave solver records the pressure on every point.</div>
          {earCard}
        </>
      ) : (
        <>
          <div className="form-grid compact">
            <label>Source</label>
            <select value={form.sourceId} onChange={(e) => setFormPatch({ sourceId: e.target.value })}>
              {sim.sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <label>Grid</label>
            <div>
              <select value={form.gridId} onChange={(e) => setFormPatch({ gridId: e.target.value })}>
                {grids.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.name} ({g.kind})
                  </option>
                ))}
              </select>
              <div className="small muted">{gp?.info ? `${gp.info.n_valid} valid of ${gp.info.n_points} points` : gp?.error ? <span className="text-err">{gp.error}</span> : '…'}</div>
              {formGrid?.kind === 'surface' && <div className="small text-warn">A surface grid follows its layer; a plane grid (e.g. at ear height) is the usual choice for an animation.</div>}
            </div>
            <label title={`${FIELD_F_MIN}–${FIELD_F_MAX} Hz: the wave solver is band-limited to about this frequency`}>Up to</label>
            <div className="inline-field">
              <NumField value={form.fMax} min={FIELD_F_MIN} max={FIELD_F_MAX} step={10} digits={0} width={62} onChange={(v) => setFormPatch({ fMax: v ?? 250 })} ariaLabel="Highest frequency (Hz)" />
              <span className="unit">Hz</span>
              <span className="small muted">cell ≈ {fmtNum((C_AIR / (CELLS_PER_WAVELENGTH * form.fMax)) * 100, 1)} cm</span>
            </div>
            <label title={`${FIELD_D_MIN}–${FIELD_D_MAX} s of simulated sound after the emission`}>Duration</label>
            <div className="inline-field">
              <NumField value={form.duration} min={FIELD_D_MIN} max={FIELD_D_MAX} step={0.05} digits={3} width={62} onChange={(v) => setFormPatch({ duration: v ?? 0.2 })} ariaLabel="Duration (s)" />
              <span className="unit">s</span>
            </div>
            <label title={`At least ${FIELD_FPS_FACTOR} × the highest frequency, and duration × frames/s ≤ ${FIELD_MAX_FRAMES}`}>Frames/s</label>
            <div className="inline-field">
              <Toggle checked={form.fpsAuto} onChange={(v) => setFormPatch({ fpsAuto: v, fps: ff.fps })} label="auto" title="4 × f max (at least 400), within the frame limit" />
              <NumField value={form.fpsAuto ? ff.fps : form.fps} disabled={form.fpsAuto} min={1} max={100000} step={50} digits={0} width={66} onChange={(v) => setFormPatch({ fps: v ?? ff.fps })} ariaLabel="Frames per second" />
            </div>
          </div>
          <div className={`small ${ff.error ? 'text-err' : 'muted'}`}>
            {ff.error ?? `${ff.frames} frames at ${ff.fps} frames/s; at ${fmtSpeed(DEFAULT_SPEED)} they play in ${fmtNum(form.duration / DEFAULT_SPEED, 0)} s.`}
          </div>
          <div className="small muted">
            Rules: {FIELD_F_MIN}–{FIELD_F_MAX} Hz, {FIELD_D_MIN}–{FIELD_D_MAX} s, frames/s ≥ {FIELD_FPS_FACTOR} × f max ({ff.minFps}), duration × frames/s ≤ {FIELD_MAX_FRAMES} (at {fmtNum(form.duration, 3)} s: ≤ {ff.maxFps} frames/s). The wave solver&apos;s cost grows about as f max⁴.
          </div>
          <input value={form.label} onChange={(e) => setFormPatch({ label: e.target.value })} placeholder="Label (optional)" aria-label="Label" maxLength={80} />
          <div className="row gap">
            <button className="btn btn-sm btn-primary" disabled={!canSubmit} onClick={() => void submit()} title="Queue the computation (one wave-solver run recording the pressure on every valid grid point)">
              {submitting ? <Spinner size={11} /> : null} Compute
            </button>
            <span className="small muted">One wave-solver run{onGpu ? ' on the GPU' : ''}, through the job queue.</span>
          </div>
          {!onGpu && <div className="small text-warn">No GPU: the wave solver runs on the CPU (much slower). Keep the grid, f max and the duration small.</div>}
          {submitError && <div className="small text-err">{submitError}</div>}
          <details className="field-ear-details" open={!hasPlane}>
            <summary className="small">Plane grid at ear height…</summary>
            {earCard}
          </details>
        </>
      )}
    </section>
  );

  const listCard = (
    <section className="card">
      <div className="card-head">
        <h3>Animations of this simulation</h3>
        <div className="grow" />
        <button className="link small" onClick={() => void refresh()}>
          Refresh
        </button>
      </div>
      {listError && <div className="small text-err">{listError}</div>}
      {!tasks ? (
        <Spinner />
      ) : !tasks.length ? (
        <div className="small muted">None yet.</div>
      ) : (
        <div className="motion-list">
          {tasks.map((t) => {
            const job = t.job_id ? jobs[t.job_id] : undefined;
            const active = ACTIVE_STATUSES.has(t.status);
            const r = t.result;
            return (
              <div key={t.id} className={`motion-item modal-item ${t.id === task?.id ? 'active' : ''}`}>
                <div className="row gap">
                  <StatusBadge status={job && active ? job.status : t.status} />
                  {t.status === 'done' ? (
                    <button className="link ellipsis grow left" onClick={() => setTaskId(t.id)} title="Show this animation">
                      {taskTitle(t)}
                    </button>
                  ) : (
                    <b className="ellipsis grow">{taskTitle(t)}</b>
                  )}
                  {active && t.job_id ? (
                    <button className="btn btn-xs btn-danger-ghost" onClick={() => void onCancel(t.job_id!)} disabled={job?.status === 'cancelled'}>
                      Cancel
                    </button>
                  ) : (
                    <button className="btn btn-xs btn-ghost" onClick={() => void remove(t)} title="Delete this animation">
                      Delete
                    </button>
                  )}
                </div>
                <div className="small muted">
                  {srcName(t.source_id)} · {gridName(t.grid_id)}
                  {r ? ` · ${fmtNum(r.duration_s * 1000, 0)} ms up to ${fmtNum(r.f_max_hz, 0)} Hz, ${r.n_frames} frames` : ''} · {fmtDate(t.created)}
                </div>
                {active && (
                  <>
                    <ProgressBar value={job?.progress ?? 0} status={job?.status ?? t.status} />
                    <div className="small muted ellipsis" title={job?.stage}>
                      {t.job_id && lost.has(t.job_id)
                        ? 'The server no longer knows this job (restarted?).'
                        : job
                          ? `${job.status === 'queued' ? 'Waiting in the queue' : job.stage || 'running'} · ${Math.round(job.progress * 100)} %${job.eta_s ? ` · about ${fmtDuration(job.eta_s)} left` : ''}`
                          : 'Waiting in the queue'}
                    </div>
                  </>
                )}
                {(t.status === 'failed' || t.status === 'cancelled') && <div className="small text-err">{t.error || job?.error || (t.status === 'cancelled' ? 'Cancelled.' : 'Failed.')}</div>}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );

  const gainPos = Math.log10(gain);

  return (
    <div className="results grid-maps modal-maps field-view">
      <div className="results-main">
        {!task ? (
          <section className="card">
            <EmptyState title={tasks === null ? 'Loading…' : 'No wave-field animation yet'}>
              The wave solver records the pressure on every point of a receiver grid after an impulse from one source: the direct wave, its reflections and the build-up of the room&apos;s modes, below the
              transition. Compute one with the form on the right, then play it slowed down.
            </EmptyState>
          </section>
        ) : (
          <>
            <section className="card gm-controls">
              <div className="gm-row">
                <b className="ellipsis" title={taskTitle(task)}>
                  {taskTitle(task)}
                </b>
                {task.label && (
                  <span className="small muted">
                    {srcName(task.source_id)} · {gridName(task.grid_id)}
                  </span>
                )}
              </div>
              <div className="gm-row modal-freq field-transport">
                <button className={`btn btn-sm ${playing ? 'active' : ''}`} onClick={() => setPlaying((p) => !p)} title={playing ? 'Pause' : 'Play'} aria-label={playing ? 'Pause' : 'Play'} disabled={nFrames < 2}>
                  {playing ? '❚❚' : '▶'}
                </button>
                <button className="btn btn-sm" onClick={() => step(-1)} disabled={frame <= 0} aria-label="Previous frame" title="Previous frame">
                  ‹
                </button>
                <input
                  type="range"
                  className="grow"
                  min={0}
                  max={Math.max(0, nFrames - 1)}
                  step={1}
                  value={frame}
                  onChange={(e) => {
                    setPlaying(false);
                    setFrame(Number(e.target.value));
                  }}
                  aria-label="Frame"
                  disabled={!nFrames}
                />
                <button className="btn btn-sm" onClick={() => step(1)} disabled={frame >= nFrames - 1} aria-label="Next frame" title="Next frame">
                  ›
                </button>
                <span className="small mono field-time" title="Time after the emission of the impulse">
                  t = {fmtNum(tMs, 2)} ms
                </span>
                <span className="small muted">
                  frame {nFrames ? frame + 1 : 0} / {nFrames}
                </span>
              </div>
              <div className="gm-row">
                <label className="inline-field" title="Playback speed, as a fraction of real time">
                  <span>Speed</span>
                  <select value={String(speed)} onChange={(e) => setSpeed(Number(e.target.value))}>
                    {SPEEDS.map((s) => (
                      <option key={s} value={s}>
                        {fmtSpeed(s)} (1/{fmtNum(1 / s, 0)} of real time)
                      </option>
                    ))}
                  </select>
                </label>
                <Toggle checked={loop} onChange={setLoop} label="Loop" />
                <span className="small muted">
                  {meta ? `${fmtNum(meta.duration_s * 1000, 0)} ms of sound play in ${fmtNum(meta.duration_s / speed, 1)} s` : ''}
                </span>
              </div>
            </section>

            <section className="card gm-map">
              {fieldError && <ErrorBox message={`Frames: ${fieldError}`} />}
              <div className="gm-canvas">
                {geom ? (
                  <SceneCanvas key={top ? 'top' : 'persp'} bounds={geom.bounds} topView={top} frame={frameReq} showGizmo={false}>
                    {staticScene}
                    {fd && (
                      <FieldMap
                        points={fd.points}
                        valid={fd.valid}
                        normals={fd.normals}
                        uAxes={fd.uAxes}
                        spacing={spacing}
                        data={fd.data}
                        nPoints={fd.nPoints}
                        frame={Math.min(frame, fd.nFrames - 1)}
                        scale={fd.scale}
                        limit={limit}
                        ramp={FIELD_RAMP}
                        interpolate={interp}
                        relief={reliefM}
                        onHover={(i, x, y) => setHover(i === null ? null : { i, x, y })}
                        onClick={pick}
                      />
                    )}
                    {fd &&
                      picked.map((i, k) =>
                        points[i] ? (
                          <Html key={i} position={points[i]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
                            <div className="gm-pin" style={{ borderColor: seriesColor(k) }}>
                              {i + 1}
                            </div>
                          </Html>
                        ) : null,
                      )}
                  </SceneCanvas>
                ) : (
                  <div className="loading-row pad">
                    <Spinner /> Loading the model…
                  </div>
                )}
                {!fd && !fieldError && (
                  <div className="gm-loading small">
                    <Spinner size={11} /> loading the frames…
                  </div>
                )}
                <div className="viewport-toolbar">
                  <Segmented<'persp' | 'top'>
                    value={top ? 'top' : 'persp'}
                    onChange={(v) => setTop(v === 'top')}
                    options={[
                      { value: 'persp', label: '3D', title: 'Perspective view (drag to orbit)' },
                      { value: 'top', label: 'Top', title: 'Orthographic view from above' },
                    ]}
                  />
                  <Segmented<'squares' | 'interp'>
                    value={interp ? 'interp' : 'squares'}
                    onChange={(v) => setInterp(v === 'interp')}
                    options={[
                      { value: 'interp', label: 'Interpolated', title: 'Colours interpolated between neighbouring points' },
                      { value: 'squares', label: 'Squares', title: 'One square per point, coloured by its own value' },
                    ]}
                  />
                  <button type="button" className={`btn btn-sm ${relief ? 'active' : ''}`} onClick={() => setRelief((r) => !r)} title="Deform the grid along its normal by the pressure (exaggerated)" aria-pressed={relief}>
                    Relief
                  </button>
                  <button type="button" className="btn btn-sm" onClick={frameGrid} disabled={!points.length} title="Frame the grid">
                    Frame grid
                  </button>
                </div>
                {fd && (
                  <div className="gm-legend">
                    <div className="gm-legend-title">Pressure at t = {fmtNum(tMs, 2)} ms</div>
                    <div className="small muted">wave band, re 1 m · colours clip beyond ±{fmtLim(limit)}</div>
                    <div className="abs-bar" style={{ background: rampCss(FIELD_RAMP) }} />
                    <div className="abs-ticks small">
                      <span>−{fmtLim(limit)}</span>
                      <span>0</span>
                      <span>+{fmtLim(limit)}</span>
                    </div>
                    <div className="small muted">
                      <span className="swatch" style={{ background: DISABLED_COLOR }} /> left out (closer than one cell to a surface, or disabled)
                    </div>
                  </div>
                )}
                {hoverPoint && fd && (
                  <div className="grid-hover small" style={{ left: hoverPoint.x + 14, top: hoverPoint.y + 12 }}>
                    <b>Point {hoverPoint.i + 1}</b>
                    <span className="muted"> · ({points[hoverPoint.i].map((v) => fmtNum(v, 2)).join(', ')})</span>
                    <div>
                      <b>{fmtNum(valueAt(hoverPoint.i), 4)}</b> at {fmtNum(tMs, 2)} ms
                    </div>
                  </div>
                )}
              </div>
              <div className="row gap wrap gm-foot">
                <span className="muted small">Click a square for its pressure over time (Ctrl-click to compare up to {MAX_TRACES} points).</span>
                <div className="grow" />
                <button type="button" className="btn btn-xs" disabled={!fd || !geom} onClick={() => void exportPng()} title="The current frame on a white background, with its legend">
                  Frame .png
                </button>
              </div>
            </section>

            <section className="card">
              <div className="card-head">
                <h3>Pressure over time</h3>
                <div className="grow" />
                {picked.length > 0 && (
                  <button className="link small" onClick={() => setPicked([])}>
                    Clear points
                  </button>
                )}
              </div>
              {!picked.length ? (
                <div className="small muted">Click a point of the map: its pressure over the whole animation appears here, with the current time.</div>
              ) : (
                <Plot
                  data={traces}
                  layout={traceLayout}
                  tools
                  filename={`${fileBase()}_pressure`}
                  table={traceTable}
                  onClick={(e) => {
                    const x = e.points?.[0]?.x;
                    if (typeof x === 'number') {
                      setPlaying(false);
                      setFrame(Math.max(0, Math.min(nFrames - 1, Math.round((x / 1000) * fps))));
                    }
                  }}
                />
              )}
            </section>
          </>
        )}
      </div>

      <aside className="results-side wide">
        {meta && task && (
          <section className="card">
            <h3>Wave-solver run</h3>
            <div className="warn-box small modal-note">{meta.note}</div>
            <dl className="kv small">
              <dt>Cell size dx</dt>
              <dd>{fmtNum(meta.dx_m * 100, 2)} cm</dd>
              <dt>Band</dt>
              <dd>up to about {fmtNum(meta.f_max_hz, 0)} Hz</dd>
              <dt>Duration</dt>
              <dd>
                {fmtNum(meta.duration_s * 1000, 0)} ms, {meta.n_frames} frames at {meta.fps} frames/s
              </dd>
              <dt>Engine</dt>
              <dd className={meta.engine === 'gpu' || meta.engine === 'cuda' ? undefined : 'text-warn'}>{meta.engine}</dd>
              {meta.conformal !== undefined && (
                <>
                  <dt>Conformal correction</dt>
                  <dd>{meta.conformal ? 'on' : 'off'}</dd>
                </>
              )}
              <dt>Points</dt>
              <dd>
                {meta.n_valid} with a value of {meta.n_points}
              </dd>
              {meta.render_s !== undefined && (
                <>
                  <dt>Computed in</dt>
                  <dd>
                    {fmtDuration(meta.render_s)} · {fmtDate(task.created)}
                  </dd>
                </>
              )}
            </dl>
            <div className="small muted">Uses the simulation as it was when the animation was queued. Display only: the results and the IRs do not depend on it.</div>
          </section>
        )}
        {task && (
          <section className="card">
            <h3>Colour scale</h3>
            <label className="inline-field field-gain" title="Fixed symmetric scale around 0: full scale = the largest pressure of the whole animation, divided by the gain">
              <span>Gain</span>
              <input type="range" min={0} max={2} step={0.05} value={gainPos} onChange={(e) => setGain(Math.round(10 ** Number(e.target.value) * 100) / 100)} aria-label="Gain" className="grow" />
              <span className="mono small">×{fmtNum(gain, gain < 10 ? 1 : 0)}</span>
            </label>
            <div className="small muted">
              Diverging scale fixed over the animation: ±{fmtLim(limit)} (largest pressure {fmtLim(fd?.scale ?? 0)} ÷ gain). Raise the gain to follow the weaker reflections and the reverberation; the strongest values then clip.
            </div>
          </section>
        )}
        {launchCard}
        {listCard}
      </aside>
    </div>
  );
}

function fmtLim(v: number): string {
  if (!Number.isFinite(v) || v === 0) return '0';
  const a = Math.abs(v);
  return a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
}
