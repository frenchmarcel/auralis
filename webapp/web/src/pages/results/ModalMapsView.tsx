import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { Html } from '@react-three/drei';
import { api, errorMessage, type CatalogObject, type GridPoints, type MeshData, type ModalMap, type ModalTask, type PlacedObject, type Simulation } from '../../api';
import { useApp } from '../../AppContext';
import { Plot } from '../../components/Plot';
import { EmptyState, ErrorBox, NumField, ProgressBar, Segmented, Spinner, StatusBadge } from '../../components/ui';
import { loadCatalog, loadGeometry, loadMesh } from '../../lib/cache';
import { layerColor, MEAN_COLOR, seriesColor } from '../../lib/colors';
import { downloadText, safeFilename, toCsv, type DataTable } from '../../lib/export';
import { fmtDate, fmtDuration, fmtNum } from '../../lib/format';
import { boundsOf, boundsOfPositions, dirFromAzEl, type Bounds } from '../../lib/geom';
import { niceRange, pointFrames, rampCss, SEQ_RAMP, DISABLED_COLOR, type Num } from '../../lib/grids';
import { href } from '../../lib/router';
import { getSlice, setSlice } from '../../lib/savedViews';
import { PlacedObjects } from '../../three/Objects';
import { CaptureBridge, GridMap, type CaptureFn } from '../../three/GridMap';
import { LayerMeshes, PointMarkers, SceneCanvas, markerRadius, type FrameRequest, type PointMarker } from '../../three/Scene';
import { mapPng } from './mapPng';
import { ACTIVE_STATUSES as ACTIVE, useTaskJobs } from './taskJobs';

/** Server limits (API "Forced-response (modal) maps"). */
export const MODAL_MAX_FREQS = 400;
export const MODAL_MAX_HZ = 1000;
const MAX_SPECTRA = 6;
const PLAY_MS = 350;

interface ModalView {
  taskId?: string | null;
  freq?: number | null;
  fixed?: { min: number; max: number } | null;
  interp?: boolean;
  top?: boolean;
  /** Launch form. */
  form?: Partial<FormState>;
}

interface FormState {
  sourceId: string;
  gridId: string;
  mode: 'range' | 'list';
  fMin: number;
  fMax: number;
  step: number;
  list: string;
  label: string;
}

const finite = (v: Num | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

/** Frequencies the form asks for (as the server builds them), or why it cannot. */
export function formFreqs(f: Pick<FormState, 'mode' | 'fMin' | 'fMax' | 'step' | 'list'>): { freqs: number[]; error: string | null } {
  let freqs: number[] = [];
  if (f.mode === 'list') {
    const toks = f.list.split(/[\s,;]+/).filter(Boolean);
    const bad = toks.filter((t) => !Number.isFinite(Number(t)));
    if (bad.length) return { freqs: [], error: `Not a number: ${bad.slice(0, 3).join(', ')}` };
    freqs = Array.from(new Set(toks.map((t) => Math.round(Number(t) * 1000) / 1000))).sort((a, b) => a - b);
  } else {
    if (!(f.step > 0) || f.fMax < f.fMin) return { freqs: [], error: 'Need f min ≤ f max and a step above 0.' };
    const n = Math.floor((f.fMax - f.fMin) / f.step + 1e-9) + 1;
    if (n > MODAL_MAX_FREQS) return { freqs: [], error: `${n} frequencies: at most ${MODAL_MAX_FREQS} (widen the step).` };
    for (let i = 0; i < n; i++) freqs.push(Math.round((f.fMin + i * f.step) * 1000) / 1000);
  }
  if (!freqs.length) return { freqs, error: 'No frequency.' };
  if (freqs.length > MODAL_MAX_FREQS) return { freqs, error: `${freqs.length} frequencies: at most ${MODAL_MAX_FREQS}.` };
  if (freqs[0] <= 0 || freqs[freqs.length - 1] > MODAL_MAX_HZ) return { freqs, error: `Frequencies must be within 0–${MODAL_MAX_HZ} Hz.` };
  return { freqs, error: null };
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

const mapCache = new Map<string, Promise<ModalMap>>();
function fetchMap(simId: string, rid: string, freq?: number): Promise<ModalMap> {
  const k = `${simId}|${rid}|${freq ?? 'all'}`;
  let p = mapCache.get(k);
  if (!p) {
    p = api.getModalMap(simId, rid, freq);
    mapCache.set(k, p);
    p.catch(() => mapCache.delete(k));
  }
  return p;
}

/** Forced-response (modal) maps: launch, list, and show a task's level map and point spectra. */
export function ModalMapsView({ simId, sim }: { simId: string; sim: Simulation }) {
  const app = useApp();
  const saved = useMemo(() => getSlice<ModalView>(simId, 'results', 'modal') ?? {}, [simId]);
  const loadTasks = useCallback(() => api.listModal(simId), [simId]);
  const { tasks, listError, jobs, lost, refresh, addQueued, cancel: cancelJob } = useTaskJobs<ModalTask>(loadTasks);
  const [taskId, setTaskId] = useState<string | null>(saved.taskId ?? null);
  const [freq, setFreq] = useState<number | null>(saved.freq ?? null);
  const [fixed, setFixed] = useState<{ min: number; max: number } | null>(saved.fixed ?? null);
  const [interp, setInterp] = useState(saved.interp ?? true);
  const [top, setTop] = useState(saved.top ?? false);
  const grids = useMemo(() => sim.receiver_grids ?? [], [sim]);
  const [form, setForm] = useState<FormState>(() => ({
    sourceId: sim.sources[0]?.id ?? '',
    gridId: grids[0]?.id ?? '',
    mode: 'range',
    fMin: 20,
    fMax: 200,
    step: 1,
    list: '',
    label: '',
    ...(saved.form ?? {}),
  }));
  const setFormPatch = (p: Partial<FormState>) => setForm((f) => ({ ...f, ...p }));
  // A saved form may name a source or grid removed since.
  useEffect(() => {
    const fix: Partial<FormState> = {};
    if (!sim.sources.some((x) => x.id === form.sourceId)) fix.sourceId = sim.sources[0]?.id ?? '';
    if (!grids.some((g) => g.id === form.gridId)) fix.gridId = grids[0]?.id ?? '';
    if (Object.keys(fix).length) setForm((f) => ({ ...f, ...fix }));
  }, [sim, grids, form.sourceId, form.gridId]);
  /** A map queued from this page: shown as soon as it is done. */
  const [awaited, setAwaited] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => setSlice(simId, 'results', 'modal', { taskId, freq, fixed, interp, top, form }), [simId, taskId, freq, fixed, interp, top, form]);

  const doneTasks = useMemo(() => (tasks ?? []).filter((t) => t.status === 'done' && t.result), [tasks]);
  useEffect(() => {
    if (awaited && doneTasks.some((t) => t.id === awaited)) {
      setTaskId(awaited);
      setFreq(null);
      setAwaited(null);
    }
  }, [awaited, doneTasks]);
  const task = doneTasks.find((t) => t.id === taskId) ?? doneTasks[0] ?? null;
  useEffect(() => {
    if (task && task.id !== taskId) setTaskId(task.id);
  }, [task, taskId]);
  const meta = task?.result ?? null;
  const freqs = useMemo(() => meta?.freqs_hz ?? [], [meta]);
  const fi = useMemo(() => {
    if (!freqs.length) return -1;
    if (freq === null) return 0;
    let best = 0;
    freqs.forEach((f, i) => {
      if (Math.abs(f - freq) < Math.abs(freqs[best] - freq)) best = i;
    });
    return best;
  }, [freqs, freq]);
  const fNow = fi >= 0 ? freqs[fi] : null;
  const setIndex = (i: number) => freqs.length && setFreq(freqs[Math.max(0, Math.min(freqs.length - 1, i))]);

  // Map at the current frequency (one request each, cached), or from the full set once loaded.
  const [all, setAll] = useState<{ rid: string; map: ModalMap } | null>(null);
  const [allLoading, setAllLoading] = useState(false);
  const [allError, setAllError] = useState<string | null>(null);
  const allMap = all && task && all.rid === task.id ? all.map : null;
  const loadAll = useCallback(() => {
    if (!task || allMap || allLoading) return;
    setAllLoading(true);
    setAllError(null);
    const rid = task.id;
    fetchMap(simId, rid)
      .then((m) => setAll({ rid, map: m }))
      .catch((e) => setAllError(errorMessage(e)))
      .finally(() => setAllLoading(false));
  }, [simId, task, allMap, allLoading]);

  const [one, setOne] = useState<{ key: string; map: ModalMap | null; error: string | null }>({ key: '', map: null, error: null });
  const oneKey = task && fNow !== null && !allMap ? `${task.id}|${fNow}` : '';
  useEffect(() => {
    if (!oneKey || !task || fNow === null) return;
    let cancelled = false;
    const t = window.setTimeout(() => {
      fetchMap(simId, task.id, fNow)
        .then((m) => !cancelled && setOne({ key: oneKey, map: m, error: null }))
        .catch((e) => !cancelled && setOne({ key: oneKey, map: null, error: errorMessage(e) }));
    }, 80);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [oneKey]);
  const oneFresh = one.key === oneKey ? one : null;
  // Keep the last map on screen while the next frequency loads.
  const lastMap = useRef<{ rid: string; map: ModalMap } | null>(null);
  const base: ModalMap | null = allMap ?? oneFresh?.map ?? (lastMap.current && lastMap.current.rid === task?.id ? lastMap.current.map : null);
  if (base && task) lastMap.current = { rid: task.id, map: base };
  const mapLoading = !allMap && !!oneKey && !oneFresh;
  const mapError = allMap ? null : (oneFresh?.error ?? null);

  const values = useMemo<Num[] | null>(() => {
    if (!base) return null;
    const k = allMap ? fi : 0;
    const row = base.level_db[k] ?? [];
    return base.points.map((_, i) => (base.valid[i] ? (row[i] ?? null) : null));
  }, [base, allMap, fi]);
  const points = useMemo(() => base?.points ?? [], [base]);

  const gridDef = grids.find((g) => g.id === task?.grid_id) ?? null;
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
  const areas = base?.area_m2 ?? [];
  const spacing = gridDef?.spacing_m ?? (areas[0] ? Math.sqrt(areas[0]) : 1);
  const frames = useMemo(() => pointFrames(gridDef, points, geom?.mesh ?? null), [gridDef, points, geom]);

  const autoRange = useMemo<[number, number]>(() => {
    const v = (values ?? []).filter(finite);
    if (!v.length) return [0, 1];
    return niceRange(Math.min(...v), Math.max(...v));
  }, [values]);
  const [lo, hi] = fixed ? [fixed.min, fixed.max] : autoRange;

  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  useEffect(() => {
    setPicked([]);
    setHover(null);
  }, [task?.id]);
  const pick = (i: number, add: boolean) => {
    loadAll();
    setPicked((p) => (p.includes(i) ? p.filter((x) => x !== i) : add ? [...p, i].slice(-MAX_SPECTRA) : [i]));
  };

  // Play: steps through the frequencies once the full set is loaded.
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    if (!allMap) {
      loadAll();
      return;
    }
    const nearest = (f: number | null) => (f === null ? 0 : freqs.reduce((b, x, i) => (Math.abs(x - f) < Math.abs(freqs[b] - f) ? i : b), 0));
    const t = window.setInterval(() => setFreq((f) => freqs[(nearest(f) + 1) % freqs.length]), PLAY_MS);
    return () => window.clearInterval(t);
  }, [playing, allMap, freqs, loadAll]);
  useEffect(() => {
    if (allError) setPlaying(false);
  }, [allError]);

  const [frame, setFrame] = useState<FrameRequest | null>(null);
  const frameKey = useRef(0);
  const capture = useRef<CaptureFn | null>(null);
  const frameGrid = () => {
    const b = points.length ? boundsOfPositions(points.flat()) : null;
    if (b) setFrame({ center: b.center, radius: Math.max(1, b.radius * 1.05), key: ++frameKey.current });
  };

  const srcName = (id: string) => sim.sources.find((s) => s.id === id)?.name ?? id;
  const gridName = (id: string) => grids.find((g) => g.id === id)?.name ?? id;
  const taskTitle = (t: ModalTask) => t.label || `${srcName(t.source_id)} · ${gridName(t.grid_id)}`;
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

  // Spectra of the picked points, and the energy mean of the grid.
  const spectrum = useMemo<Data[]>(() => {
    if (!allMap) return [];
    const out: Data[] = picked.map((i, k) => ({
      type: 'scatter',
      mode: 'lines',
      x: allMap.freqs_hz,
      y: allMap.level_db.map((row) => (allMap.valid[i] ? row[i] : null)),
      name: `Point ${i + 1}`,
      line: { color: seriesColor(k), width: 2 },
      hovertemplate: `Point ${i + 1}<br>%{x:.1f} Hz: %{y:.1f} dB<extra></extra>`,
    }));
    const mean = allMap.level_db.map((row) => {
      const v = row.filter((x, i) => allMap.valid[i] && finite(x)) as number[];
      return v.length ? 10 * Math.log10(v.reduce((a, x) => a + 10 ** (x / 10), 0) / v.length) : null;
    });
    out.push({ type: 'scatter', mode: 'lines', x: allMap.freqs_hz, y: mean, name: 'Grid mean (energy)', line: { color: MEAN_COLOR, width: 1.5, dash: 'dot' }, hovertemplate: 'Grid mean<br>%{x:.1f} Hz: %{y:.1f} dB<extra></extra>' });
    return out;
  }, [allMap, picked]);
  const spectrumLayout = useMemo<Partial<Layout>>(
    () => ({
      height: 250,
      margin: { l: 52, r: 12, t: 28, b: 40 },
      xaxis: { title: { text: 'Frequency (Hz) · click to show that frequency on the map' } },
      yaxis: { title: { text: 'Level (dB re free field at 10 m)' } },
      shapes: fNow !== null ? [{ type: 'line', xref: 'x', yref: 'paper', x0: fNow, x1: fNow, y0: 0, y1: 1, line: { color: '#f0a83a', width: 1.5, dash: 'dash' } }] : [],
      showlegend: true,
    }),
    [fNow],
  );
  const spectrumTable = (): DataTable => ({
    columns: ['frequency (Hz)', ...picked.map((i) => `point ${i + 1} (dB)`), 'grid energy mean (dB)'],
    rows: (allMap?.freqs_hz ?? []).map((f, k) => [f, ...spectrum.map((d) => ((d as { y: (number | null)[] }).y[k] ?? null))]),
  });

  const fileBase = () => safeFilename(`${sim.name}_modal_${task ? taskTitle(task) : ''}_${fNow !== null ? `${fmtNum(fNow, 1)}Hz` : ''}`);
  const exportCsv = () => {
    if (!values) return;
    downloadText(
      toCsv({
        columns: ['point', 'x (m)', 'y (m)', 'z (m)', 'valid', 'area (m²)', `level at ${fmtNum(fNow, 3)} Hz (dB re free field at 10 m)`],
        rows: points.map((p, i) => [i + 1, p[0], p[1], p[2], base?.valid[i] ? 1 : 0, areas[i], values[i]]),
      }),
      `${fileBase()}.csv`,
    );
  };
  const exportPng = async () => {
    const cap = capture.current;
    if (!cap) return;
    try {
      await mapPng(cap('#ffffff'), {
        title: `Forced response at ${fmtNum(fNow, 1)} Hz · level (dB re free field at 10 m)`,
        subtitle: `${sim.name} · ${task ? taskTitle(task) : ''} · wave solver, dx ${fmtNum((meta?.dx_m ?? 0) * 100, 1)} cm`,
        ramp: SEQ_RAMP,
        lo: fmtNum(lo, 1),
        mid: fmtNum((lo + hi) / 2, 1),
        hi: fmtNum(hi, 1),
        filename: `${fileBase()}.png`,
      });
    } catch (e) {
      app.toast('error', `PNG export failed: ${errorMessage(e)}`);
    }
  };

  // Launch form.
  const ff = formFreqs(form);
  const [gridPreview, setGridPreview] = useState<{ id: string; info: GridPoints | null; error: string | null } | null>(null);
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
  }, [simId, form.gridId]);
  const gp = gridPreview?.id === form.gridId ? gridPreview : null;
  const canSubmit = !!form.sourceId && !!form.gridId && !ff.error && !submitting;
  const submit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body =
        form.mode === 'list'
          ? { source_id: form.sourceId, grid_id: form.gridId, freqs: ff.freqs, label: form.label.trim() }
          : { source_id: form.sourceId, grid_id: form.gridId, f_min: form.fMin, f_max: form.fMax, step_hz: form.step, label: form.label.trim() };
      const r = await api.submitModal(simId, body);
      addQueued(r.modal, r.job);
      setAwaited(r.modal.id);
      setFormPatch({ label: '' });
    } catch (e) {
      setSubmitError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  };
  const cancel = async (jid: string) => {
    try {
      await cancelJob(jid);
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };
  const remove = async (t: ModalTask) => {
    const ok = await app.confirm({ title: 'Delete modal map', message: <>Delete the modal map <b>{taskTitle(t)}</b>?</>, confirmLabel: 'Delete', danger: true });
    if (!ok) return;
    try {
      await api.deleteModal(simId, t.id);
      await refresh();
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  const hoverPoint = hover && points[hover.i] ? hover : null;
  const fmtV = (v: Num | undefined) => fmtNum(v ?? null, 1);
  const onGpu = app.health?.gpu === true;

  const launchCard = (
    <section className="card">
      <h3>New modal map</h3>
      {!grids.length ? (
        <div className="small muted">
          A modal map needs a receiver grid: add one in the editor (<a href={href.editor(simId)}>Grids tab</a>).
        </div>
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
                    {g.name}
                  </option>
                ))}
              </select>
              <div className="small muted">{gp?.info ? `${gp.info.n_valid} valid of ${gp.info.n_points} points` : gp?.error ? <span className="text-err">{gp.error}</span> : '…'}</div>
            </div>
            <label>Frequencies</label>
            <Segmented<'range' | 'list'>
              value={form.mode}
              onChange={(v) => setFormPatch({ mode: v })}
              options={[
                { value: 'range', label: 'Range', title: 'From f min to f max in steps' },
                { value: 'list', label: 'List', title: 'Chosen frequencies' },
              ]}
            />
          </div>
          {form.mode === 'range' ? (
            <div className="row gap wrap">
              <label className="inline-field">
                <span>from</span>
                <NumField value={form.fMin} min={1} max={MODAL_MAX_HZ} step={1} digits={1} width={58} onChange={(v) => setFormPatch({ fMin: v ?? 20 })} ariaLabel="Lowest frequency (Hz)" />
              </label>
              <label className="inline-field">
                <span>to</span>
                <NumField value={form.fMax} min={1} max={MODAL_MAX_HZ} step={1} digits={1} width={58} onChange={(v) => setFormPatch({ fMax: v ?? 200 })} ariaLabel="Highest frequency (Hz)" />
              </label>
              <label className="inline-field">
                <span>step</span>
                <NumField value={form.step} min={0.1} max={100} step={0.5} digits={2} width={52} onChange={(v) => setFormPatch({ step: v ?? 1 })} ariaLabel="Step (Hz)" />
              </label>
              <span className="unit">Hz</span>
            </div>
          ) : (
            <input value={form.list} onChange={(e) => setFormPatch({ list: e.target.value })} placeholder="e.g. 43 57 63 71.5 (Hz)" aria-label="Frequencies (Hz)" spellCheck={false} />
          )}
          <div className={`small ${ff.error ? 'text-err' : 'muted'}`}>
            {ff.error ?? `${ff.freqs.length} frequenc${ff.freqs.length === 1 ? 'y' : 'ies'}, ${fmtNum(ff.freqs[0], 1)}–${fmtNum(ff.freqs[ff.freqs.length - 1], 1)} Hz`} · limits: {MODAL_MAX_FREQS} frequencies, {MODAL_MAX_HZ} Hz
          </div>
          <input value={form.label} onChange={(e) => setFormPatch({ label: e.target.value })} placeholder="Label (optional)" aria-label="Label" maxLength={80} />
          <div className="row gap">
            <button className="btn btn-sm btn-primary" disabled={!canSubmit} onClick={() => void submit()} title="Queue the computation (one wave-solver run with a probe on every valid grid point)">
              {submitting ? <Spinner size={11} /> : null} Compute
            </button>
            <span className="small muted">One wave-solver run{onGpu ? ' on the GPU' : ''}, through the job queue.</span>
          </div>
          {!onGpu && <div className="small text-warn">No GPU: the wave solver runs on the CPU (much slower). Keep the grid and the highest frequency small.</div>}
          {submitError && <div className="small text-err">{submitError}</div>}
        </>
      )}
    </section>
  );

  const listCard = (
    <section className="card">
      <div className="card-head">
        <h3>Modal maps of this simulation</h3>
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
            const active = ACTIVE.has(t.status);
            return (
              <div key={t.id} className={`motion-item modal-item ${t.id === task?.id ? 'active' : ''}`}>
                <div className="row gap">
                  <StatusBadge status={job && active ? job.status : t.status} />
                  {t.status === 'done' ? (
                    <button className="link ellipsis grow left" onClick={() => setTaskId(t.id)} title="Show this map">
                      {taskTitle(t)}
                    </button>
                  ) : (
                    <b className="ellipsis grow">{taskTitle(t)}</b>
                  )}
                  {active && t.job_id ? (
                    <button className="btn btn-xs btn-danger-ghost" onClick={() => void cancel(t.job_id!)} disabled={job?.status === 'cancelled'}>
                      Cancel
                    </button>
                  ) : (
                    <button className="btn btn-xs btn-ghost" onClick={() => void remove(t)} title="Delete this modal map">
                      Delete
                    </button>
                  )}
                </div>
                <div className="small muted">
                  {srcName(t.source_id)} · {gridName(t.grid_id)} · {t.n_freqs} frequencies · {fmtDate(t.created)}
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

  return (
    <div className="results grid-maps modal-maps">
      <div className="results-main">
        {!task ? (
          <section className="card">
            <EmptyState title={tasks === null ? 'Loading…' : 'No modal map yet'}>
              A forced-response map shows the level of the wave solver at one frequency on every point of a receiver grid, for one source: modes, nodes and antinodes below the transition. Compute one with the form on the right.
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
              <div className="gm-row modal-freq">
                <button className={`btn btn-sm ${playing ? 'active' : ''}`} onClick={() => setPlaying((p) => !p)} title={playing ? 'Stop' : 'Step through the frequencies (loads them all once)'} aria-label={playing ? 'Stop' : 'Play'} disabled={freqs.length < 2}>
                  {playing ? '■' : '▶'}
                </button>
                <button className="btn btn-sm" onClick={() => setIndex(fi - 1)} disabled={fi <= 0} aria-label="Previous frequency">
                  ‹
                </button>
                <input
                  type="range"
                  className="grow"
                  min={0}
                  max={Math.max(0, freqs.length - 1)}
                  step={1}
                  value={Math.max(0, fi)}
                  onChange={(e) => setIndex(Number(e.target.value))}
                  aria-label="Frequency"
                />
                <button className="btn btn-sm" onClick={() => setIndex(fi + 1)} disabled={fi < 0 || fi >= freqs.length - 1} aria-label="Next frequency">
                  ›
                </button>
                <label className="inline-field" title="Type a frequency: the nearest computed one is shown">
                  <NumField value={fNow} min={0} max={MODAL_MAX_HZ} step={1} digits={2} width={70} onChange={(v) => v !== null && setFreq(v)} ariaLabel="Frequency (Hz)" />
                  <span className="unit">Hz</span>
                </label>
                <span className="small muted">
                  {fi + 1} / {freqs.length}
                </span>
              </div>
            </section>

            <section className="card gm-map">
              {mapError && <ErrorBox message={mapError} />}
              {allError && <ErrorBox message={`All frequencies: ${allError}`} />}
              <div className="gm-canvas">
                {geom ? (
                  <SceneCanvas key={top ? 'top' : 'persp'} bounds={geom.bounds} topView={top} frame={frame} showGizmo={false}>
                    <LayerMeshes mesh={geom.mesh} shading="ghosted" pickable={false} styleOf={(_, i) => ({ color: layerColor(i) })} />
                    <PlacedObjects objects={geom.placed} catalog={catalog} showVisual pickable={false} />
                    {values && points.length > 0 && (
                      <GridMap
                        points={points}
                        frames={frames}
                        spacing={spacing}
                        values={values}
                        lo={lo}
                        hi={hi}
                        ramp={SEQ_RAMP}
                        interpolate={interp}
                        onHover={(i, x, y) => setHover(i === null ? null : { i, x, y })}
                        onClick={pick}
                      />
                    )}
                    <PointMarkers points={sourceMarkers} radius={markerRadius(geom.bounds)} />
                    {picked.map((i, k) =>
                      points[i] ? (
                        <Html key={i} position={points[i]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
                          <div className="gm-pin" style={{ borderColor: seriesColor(k) }}>
                            {i + 1}: {fmtV(values?.[i])}
                          </div>
                        </Html>
                      ) : null,
                    )}
                    <CaptureBridge capture={capture} />
                  </SceneCanvas>
                ) : (
                  <div className="loading-row pad">
                    <Spinner /> Loading the model…
                  </div>
                )}
                {(mapLoading || allLoading) && (
                  <div className="gm-loading small">
                    <Spinner size={11} /> {allLoading ? 'loading all frequencies…' : 'loading the map…'}
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
                  <button type="button" className="btn btn-sm" onClick={frameGrid} disabled={!points.length} title="Frame the grid">
                    Frame grid
                  </button>
                </div>
                {values && (
                  <div className="gm-legend">
                    <div className="gm-legend-title">Level at {fmtNum(fNow, 1)} Hz (dB)</div>
                    <div className="small muted">re the same source in free field at 10 m</div>
                    <div className="abs-bar" style={{ background: rampCss(SEQ_RAMP) }} />
                    <div className="abs-ticks small">
                      <span>{fmtV(lo)}</span>
                      <span>{fmtV((lo + hi) / 2)}</span>
                      <span>{fmtV(hi)}</span>
                    </div>
                    <div className="small muted">
                      <span className="swatch" style={{ background: DISABLED_COLOR }} /> left out (closer than one wave-solver cell to a surface, or disabled)
                    </div>
                  </div>
                )}
                {hoverPoint && values && (
                  <div className="grid-hover small" style={{ left: hoverPoint.x + 14, top: hoverPoint.y + 12 }}>
                    <b>Point {hoverPoint.i + 1}</b>
                    <span className="muted"> · ({points[hoverPoint.i].map((v) => fmtNum(v, 2)).join(', ')})</span>
                    {base?.valid[hoverPoint.i] && finite(values[hoverPoint.i]) ? (
                      <div>
                        <b>{fmtV(values[hoverPoint.i])} dB</b> at {fmtNum(fNow, 1)} Hz
                      </div>
                    ) : (
                      <div className="text-warn">no value (left out of the wave-solver run)</div>
                    )}
                  </div>
                )}
              </div>
              <div className="row gap wrap gm-foot">
                <span className="muted small">Click a square for its spectrum (Ctrl-click to compare up to {MAX_SPECTRA} points).</span>
                <div className="grow" />
                <button type="button" className="btn btn-xs" disabled={!values} onClick={exportCsv} title="One row per grid point: position, validity, area and level at this frequency">
                  Map data .csv
                </button>
                <button type="button" className="btn btn-xs" disabled={!values || !geom} onClick={() => void exportPng()} title="The current view on a white background, with its legend">
                  View .png
                </button>
              </div>
            </section>

            <section className="card">
              <div className="card-head">
                <h3>Spectrum</h3>
                <div className="grow" />
                {picked.length > 0 && (
                  <button className="link small" onClick={() => setPicked([])}>
                    Clear points
                  </button>
                )}
              </div>
              {!picked.length ? (
                <div className="small muted">Click a point of the map: its level vs frequency appears here, with the energy mean of the grid.</div>
              ) : !allMap ? (
                <div className="loading-row">{allError ? <span className="text-err">{allError}</span> : <><Spinner /> Loading all frequencies…</>}</div>
              ) : (
                <Plot
                  data={spectrum}
                  layout={spectrumLayout}
                  tools
                  filename={`${fileBase()}_spectrum`}
                  table={spectrumTable}
                  onClick={(e) => {
                    const x = e.points?.[0]?.x;
                    if (typeof x === 'number') setFreq(x);
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
              <dt>Duration</dt>
              <dd title="Length of the simulated response; the frequency resolution is about 1/duration">
                {fmtNum(meta.duration_s, 2)} s (resolution ≈ {fmtNum(1 / meta.duration_s, 2)} Hz)
              </dd>
              <dt>Engine</dt>
              <dd className={meta.engine === 'gpu' || meta.engine === 'cuda' ? undefined : 'text-warn'}>{meta.engine}</dd>
              <dt>Conformal correction</dt>
              <dd>{meta.conformal ? 'on' : 'off'}</dd>
              <dt>Points</dt>
              <dd>
                {meta.n_valid} with a value of {meta.n_points}
              </dd>
              <dt>Frequencies</dt>
              <dd>
                {freqs.length} ({fmtNum(freqs[0], 1)}–{fmtNum(freqs[freqs.length - 1], 1)} Hz, nearest FFT bins)
              </dd>
              <dt>Computed in</dt>
              <dd>
                {fmtDuration(meta.render_s)} · {fmtDate(task.created)}
              </dd>
            </dl>
            <div className="small muted">Uses the simulation as it was when the map was queued. Below about 50 Hz the materials are extrapolated.</div>
          </section>
        )}
        {task && (
          <section className="card">
            <h3>Colour scale</h3>
            <div className="row gap wrap">
              <Segmented<'auto' | 'fixed'>
                value={fixed ? 'fixed' : 'auto'}
                onChange={(v) => setFixed(v === 'fixed' ? { min: lo, max: hi } : null)}
                options={[
                  { value: 'auto', label: 'Auto', title: 'From the values at this frequency' },
                  { value: 'fixed', label: 'Fixed', title: 'Kept when the frequency or the map changes' },
                ]}
              />
              <label className="inline-field">
                <span>Min</span>
                <NumField value={lo} step={1} digits={1} width={60} onChange={(v) => v !== null && setFixed({ min: v, max: Math.max(v + 0.1, hi) })} ariaLabel="Scale minimum (dB)" />
              </label>
              <label className="inline-field">
                <span>Max</span>
                <NumField value={hi} step={1} digits={1} width={60} onChange={(v) => v !== null && setFixed({ min: Math.min(lo, v - 0.1), max: v })} ariaLabel="Scale maximum (dB)" />
              </label>
            </div>
            <div className="small muted">{fixed ? 'Fixed: compare frequencies on one scale.' : 'Automatic: rounded range at this frequency. Edit min or max to fix it.'}</div>
          </section>
        )}
        {launchCard}
        {listCard}
      </aside>
    </div>
  );
}
