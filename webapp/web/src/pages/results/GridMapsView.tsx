import { useEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { Html } from '@react-three/drei';
import { api, BANDS_HZ, errorMessage, type CatalogObject, type GridResults, type MeshData, type PlacedObject } from '../../api';
import { useApp } from '../../AppContext';
import { Plot } from '../../components/Plot';
import { EmptyState, ErrorBox, NumField, Segmented, Spinner, Toggle } from '../../components/ui';
import { loadCatalog, loadGeometry, loadMesh } from '../../lib/cache';
import { layerColor } from '../../lib/colors';
import { downloadText, safeFilename, toCsv, type DataTable } from '../../lib/export';
import { bandLabel, fmtNum } from '../../lib/format';
import { boundsOf, boundsOfPositions, dirFromAzEl, type Bounds } from '../../lib/geom';
import {
  areaHistogram,
  areaShare,
  areaStats,
  BAND_MEANS,
  bandValue,
  DISABLED_COLOR,
  DIV_RAMP,
  GRID_PARAMS,
  gridParam,
  meanOverSources,
  median3x3,
  niceRange,
  pointFrames,
  rampColor,
  rampCss,
  SEQ_RAMP,
  TARGET_FAIL,
  TARGET_OK,
  type BandChoice,
  type Num,
  type TargetOp,
} from '../../lib/grids';
import { href } from '../../lib/router';
import { getSlice, setSlice } from '../../lib/savedViews';
import { useBundles, type SimDataStore, type SimEntry } from '../../lib/simdata';
import { PlacedObjects } from '../../three/Objects';
import { CaptureBridge, GridMap, type CaptureFn } from '../../three/GridMap';
import { LayerMeshes, PointMarkers, SceneCanvas, markerRadius, type FrameRequest, type PointMarker } from '../../three/Scene';

const ALL = '*';

interface MapView {
  simId?: string;
  gridId?: string;
  source?: string;
  param?: string;
  band?: BandChoice;
  interp?: boolean;
  median?: boolean;
  diffSimId?: string | null;
  /** Target per parameter. */
  targets?: Record<string, { op: TargetOp; value: number | null }>;
  showTarget?: boolean;
  top?: boolean;
}

/** Part of the Results page's saved view (server, with a browser copy as fallback). */
function loadMapView(pageSimId: string): MapView {
  return getSlice<MapView>(pageSimId, 'results', 'gridmaps') ?? {};
}

function storeMapView(pageSimId: string, v: MapView): void {
  setSlice(pageSimId, 'results', 'gridmaps', v);
}

type Scales = Record<string, { min: number; max: number }>;
const SCALES_KEY = 'auralis.gridmaps.scales';

/** Fixed colour scales, per parameter, band and map kind; shared by every simulation and source. */
function loadScales(): Scales {
  try {
    return (JSON.parse(window.localStorage.getItem(SCALES_KEY) ?? '{}') as Scales) ?? {};
  } catch {
    return {};
  }
}

function storeScales(s: Scales): void {
  try {
    window.localStorage.setItem(SCALES_KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable: the scale stays fixed for this page only */
  }
}

const gridCache = new Map<string, Promise<GridResults>>();

function fetchGrid(simId: string, gridId: string, sourceId: string, param: string): Promise<GridResults> {
  const k = `${simId}|${gridId}|${sourceId}|${param}`;
  let p = gridCache.get(k);
  if (!p) {
    p = api.getGridResults(simId, gridId, sourceId, param);
    gridCache.set(k, p);
    p.catch(() => gridCache.delete(k));
  }
  return p;
}

/** Grid results of several sources for one parameter. */
function useGridResults(simId: string | null, gridId: string | null, sources: string[], param: string) {
  const [state, setState] = useState<{ key: string; data: GridResults[] | null; error: string | null }>({ key: '', data: null, error: null });
  const key = simId && gridId && sources.length ? `${simId}|${gridId}|${sources.join(',')}|${param}` : '';
  useEffect(() => {
    if (!key || !simId || !gridId) return;
    let cancelled = false;
    Promise.all(sources.map((s) => fetchGrid(simId, gridId, s, param)))
      .then((data) => !cancelled && setState({ key, data, error: null }))
      .catch((e) => !cancelled && setState({ key, data: null, error: errorMessage(e) }));
    return () => {
      cancelled = true;
    };
    // `sources` is described by `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const fresh = state.key === key;
  return { data: fresh ? state.data : null, error: fresh ? state.error : null, loading: !!key && !fresh };
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

function useCatalogMap(needed: boolean): Map<string, CatalogObject> {
  const [c, setC] = useState<Map<string, CatalogObject>>(new Map());
  useEffect(() => {
    if (!needed) return;
    let cancelled = false;
    loadCatalog()
      .then((cat) => !cancelled && setC(new Map(cat.objects.map((o) => [o.id, o]))))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [needed]);
  return c;
}

function bandText(b: BandChoice): string {
  if (typeof b === 'number') return `${bandLabel(BANDS_HZ[b])} Hz`;
  return BAND_MEANS.find((m) => m.value === b)?.label ?? String(b);
}

const finite = (v: Num | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

export function GridMapsView({ pageSimId, store, entries, index }: { pageSimId: string; store: SimDataStore; entries: SimEntry[]; index: Map<string, boolean | 'error'> }) {
  const app = useApp();
  const saved = useMemo(() => loadMapView(pageSimId), [pageSimId]);
  const [simId, setSimId] = useState<string>(saved.simId ?? pageSimId);
  const [gridId, setGridId] = useState<string | null>(saved.gridId ?? null);
  const [source, setSource] = useState<string>(saved.source ?? ALL);
  const [paramKey, setParamKey] = useState<string>(GRID_PARAMS.some((p) => p.key === saved.param) ? saved.param! : 'C80');
  const [band, setBand] = useState<BandChoice>(saved.band ?? 3);
  const [interp, setInterp] = useState(saved.interp ?? true);
  const [median, setMedian] = useState(saved.median ?? false);
  const [diffSimId, setDiffSimId] = useState<string | null>(saved.diffSimId ?? null);
  const [targets, setTargets] = useState<Record<string, { op: TargetOp; value: number | null }>>(saved.targets ?? {});
  const targetOp: TargetOp = targets[paramKey]?.op ?? 'ge';
  const target: number | null = targets[paramKey]?.value ?? null;
  const setTargetOp = (op: TargetOp) => setTargets((t) => ({ ...t, [paramKey]: { op, value: target } }));
  const setTarget = (value: number | null) => setTargets((t) => ({ ...t, [paramKey]: { op: targetOp, value } }));
  const [showTarget, setShowTarget] = useState(saved.showTarget ?? false);
  const [top, setTop] = useState(saved.top ?? false);
  const [scales, setScales] = useState<Scales>(() => loadScales());
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const [pinned, setPinned] = useState<number[]>([]);
  const [frame, setFrame] = useState<FrameRequest | null>(null);
  const frameKey = useRef(0);
  const capture = useRef<CaptureFn | null>(null);

  useEffect(() => {
    storeMapView(pageSimId, { simId, gridId: gridId ?? undefined, source, param: paramKey, band, interp, median, diffSimId, targets, showTarget, top });
  }, [pageSimId, simId, gridId, source, paramKey, band, interp, median, diffSimId, targets, showTarget, top]);

  // Every simulation with results (memoised in the store): which have grid results.
  const withResults = useMemo(() => entries.filter((e) => index.get(e.id) === true).map((e) => e.id), [entries, index]);
  const bundles = useBundles(store, [simId, pageSimId, ...(diffSimId ? [diffSimId] : []), ...withResults]);
  const gridSims = useMemo(() => entries.filter((e) => (bundles.get(e.id)?.data?.results?.grids?.length ?? 0) > 0), [entries, bundles]);
  const probing = entries.some((e) => !index.has(e.id));

  // Default to a simulation that has grid results.
  useEffect(() => {
    const has = (bundles.get(simId)?.data?.results?.grids?.length ?? 0) > 0;
    const known = bundles.get(simId)?.data || bundles.get(simId)?.error;
    if (!has && known && gridSims.length && !gridSims.some((e) => e.id === simId)) setSimId(gridSims[0].id);
  }, [bundles, simId, gridSims]);

  const bundle = bundles.get(simId)?.data ?? null;
  const sim = bundle?.sim ?? null;
  const results = bundle?.results ?? null;
  const gridInfos = useMemo(() => results?.grids ?? [], [results]);
  const gridInfo = gridInfos.find((g) => g.grid_id === gridId) ?? gridInfos[0] ?? null;
  const gid = gridInfo?.grid_id ?? null;
  useEffect(() => {
    if (gridInfo && gridInfo.grid_id !== gridId) setGridId(gridInfo.grid_id);
  }, [gridInfo, gridId]);
  const gridDef = sim?.receiver_grids?.find((g) => g.id === gid) ?? null;
  const gridName = (id: string) => sim?.receiver_grids?.find((g) => g.id === id)?.name ?? id;
  const srcIds = useMemo(() => gridInfo?.sources.map((s) => s.source_id) ?? [], [gridInfo]);
  useEffect(() => {
    if (source !== ALL && srcIds.length && !srcIds.includes(source)) setSource(srcIds.length > 1 ? ALL : srcIds[0]);
  }, [srcIds, source]);
  const chosenSources = source === ALL ? srcIds : srcIds.filter((s) => s === source);
  const srcName = (id: string) => sim?.sources.find((s) => s.id === id)?.name ?? id;
  const param = gridParam(paramKey);
  const nPoints = gridInfo?.sources[0]?.n_points ?? null;

  // Difference: another simulation with the same grid (id and point count).
  const diffCandidates = useMemo(
    () =>
      gridSims.filter((e) => {
        if (e.id === simId || !gid) return false;
        const g = bundles.get(e.id)?.data?.results?.grids?.find((x) => x.grid_id === gid);
        return !!g && g.sources[0]?.n_points === nPoints;
      }),
    [gridSims, simId, gid, bundles, nPoints],
  );
  const diffOn = !!diffSimId && diffCandidates.some((e) => e.id === diffSimId);
  const diffBundle = diffOn ? (bundles.get(diffSimId!)?.data ?? null) : null;
  const diffInfo = diffBundle?.results?.grids?.find((g) => g.grid_id === gid) ?? null;
  const diffSrcIds = diffInfo?.sources.map((s) => s.source_id) ?? [];
  const diffSources = source === ALL ? diffSrcIds : diffSrcIds.filter((s) => s === source);
  const diffMissingSource = diffOn && !!diffInfo && source !== ALL && !diffSources.length;

  const main = useGridResults(results ? simId : null, gid, chosenSources, paramKey);
  const other = useGridResults(diffOn && diffInfo ? diffSimId : null, gid, diffSources, paramKey);

  const geom = useGeom(sim?.geometry_id ?? null);
  const catalog = useCatalogMap(!!geom?.placed.length);

  const base = main.data?.[0] ?? null;
  const points = useMemo(() => base?.points ?? [], [base]);
  const areas = useMemo(() => base?.area_m2 ?? [], [base]);
  const valid = useMemo(() => base?.valid ?? [], [base]);
  const spacing = gridDef?.spacing_m ?? (areas[0] ? Math.sqrt(areas[0]) : 1);
  const frames = useMemo(() => pointFrames(gridDef, points, geom?.mesh ?? null), [gridDef, points, geom]);

  const valuesOf = (list: GridResults[] | null): Num[] | null => {
    if (!list || !list.length) return null;
    const arrays = list.map((r) => {
      const col = r.params[paramKey];
      return col ? r.points.map((_, i) => (r.valid[i] ? bandValue(col[i], band, param.energetic) : null)) : r.points.map(() => null);
    });
    return meanOverSources(arrays, param.energetic);
  };
  const missingParam = !!main.data && main.data.some((r) => !r.params[paramKey]);
  const valuesA = useMemo(() => valuesOf(main.data), [main.data, paramKey, band, param.energetic]); // eslint-disable-line react-hooks/exhaustive-deps
  const valuesB = useMemo(() => (diffOn ? valuesOf(other.data) : null), [diffOn, other.data, paramKey, band, param.energetic]); // eslint-disable-line react-hooks/exhaustive-deps
  const sizeMismatch = !!valuesA && !!valuesB && valuesA.length !== valuesB.length;
  const raw = useMemo<Num[] | null>(() => {
    if (!valuesA) return null;
    if (!diffOn) return valuesA;
    if (!valuesB || valuesB.length !== valuesA.length) return null;
    return valuesA.map((a, i) => (finite(a) && finite(valuesB[i]) ? a - (valuesB[i] as number) : null));
  }, [valuesA, valuesB, diffOn]);
  const shown = useMemo(() => (raw && median ? median3x3(raw, points, spacing) : raw), [raw, median, points, spacing]);
  const stats = useMemo(() => (raw ? areaStats(raw, areas) : null), [raw, areas]);

  // Colour scale: automatic (rounded data range; symmetric for a difference) or fixed.
  const scaleKey = `${paramKey}|${band}|${diffOn ? 'diff' : 'abs'}`;
  const fixed = scales[scaleKey] ?? null;
  const autoRange = useMemo<[number, number]>(() => {
    const v = (shown ?? []).filter(finite);
    if (!v.length) return [0, 1];
    return niceRange(Math.min(...v), Math.max(...v), diffOn);
  }, [shown, diffOn]);
  const [lo, hi] = fixed ? [fixed.min, fixed.max] : autoRange;
  const ramp = diffOn ? DIV_RAMP : SEQ_RAMP;
  const setFixed = (r: { min: number; max: number } | null) => {
    const n = { ...scales };
    if (r) n[scaleKey] = r;
    else delete n[scaleKey];
    setScales(n);
    storeScales(n);
  };

  const share = raw && target !== null ? areaShare(raw, areas, targetOp, target) : null;
  const nMeet = raw && target !== null ? raw.filter((v) => finite(v) && (targetOp === 'ge' ? v >= target : v <= target)).length : 0;
  const targetColors = useMemo(() => {
    if (!showTarget || !raw || target === null) return null;
    return raw.map((v) => (finite(v) ? ((targetOp === 'ge' ? v >= target : v <= target) ? TARGET_OK : TARGET_FAIL) : null));
  }, [showTarget, raw, target, targetOp]);

  useEffect(() => {
    setPinned([]);
    setHover(null);
  }, [simId, gid]);

  const unit = param.unit ? ` (${param.unit})` : '';
  const plainLabel = param.label;
  const sourceText = source === ALL ? (srcIds.length > 1 ? `all ${srcIds.length} sources (${param.energetic ? 'energy' : 'arithmetic'} mean)` : srcName(srcIds[0] ?? '')) : srcName(source);
  const bandPart = param.scalar ? '' : ` · ${bandText(band)}`;
  const diffName = diffOn ? (diffBundle?.sim.name ?? diffSimId) : '';
  const title = `${diffOn ? 'Δ ' : ''}${plainLabel}${unit}${bandPart} · ${sourceText}`;
  const subtitle = `${sim?.name ?? ''} · ${gid ? gridName(gid) : ''}${diffOn ? ` − ${diffName}` : ''}`;

  const sourceMarkers = useMemo<PointMarker[]>(
    () =>
      (sim?.sources ?? []).map((s) => ({
        kind: 'source' as const,
        id: s.id,
        label: s.name,
        position: s.position,
        dir: dirFromAzEl(s.azimuth_deg, s.elevation_deg),
        selected: chosenSources.includes(s.id),
        dim: !chosenSources.includes(s.id),
      })),
    [sim, chosenSources],
  );

  const frameGrid = () => {
    const b = points.length ? boundsOfPositions(points.flat()) : null;
    if (b) setFrame({ center: b.center, radius: Math.max(1, b.radius * 1.05), key: ++frameKey.current });
  };

  const fmtV = (v: Num | undefined) => fmtNum(v ?? null, param.digits);

  const csvTable = (): DataTable => {
    const cols = ['point', 'x (m)', 'y (m)', 'z (m)', 'valid', 'area (m²)'];
    const label = `${plainLabel}${param.scalar ? '' : ` ${bandText(band)}`}${unit}`;
    if (diffOn) cols.push(`${sim?.name ?? 'A'}: ${label}`, `${diffName}: ${label}`, `difference${unit}`);
    else cols.push(label);
    if (median) cols.push(`${diffOn ? 'difference' : label}, 3x3 median (display only)`);
    return {
      columns: cols,
      rows: points.map((p, i) => [
        i + 1,
        p[0],
        p[1],
        p[2],
        valid[i] ? 1 : 0,
        areas[i],
        ...(diffOn ? [valuesA?.[i] ?? null, valuesB?.[i] ?? null, raw?.[i] ?? null] : [raw?.[i] ?? null]),
        ...(median ? [shown?.[i] ?? null] : []),
      ]),
    };
  };
  const fileBase = () => safeFilename(`${sim?.name ?? 'grid'}_${gid ? gridName(gid) : ''}_${diffOn ? 'delta_' : ''}${plainLabel}${param.scalar ? '' : `_${bandText(band)}`}_${source === ALL ? 'all' : source}`);

  const exportCsv = () => {
    if (!raw) return;
    downloadText(toCsv(csvTable()), `${fileBase()}.csv`);
  };

  const exportPng = async () => {
    const cap = capture.current;
    if (!cap) return;
    try {
      const url = cap('#ffffff');
      const img = new Image();
      await new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('the 3D view could not be read'));
        img.src = url;
      });
      const pad = 16;
      const legendH = 92;
      const c = document.createElement('canvas');
      c.width = img.width;
      c.height = img.height + legendH;
      const g = c.getContext('2d')!;
      g.fillStyle = '#ffffff';
      g.fillRect(0, 0, c.width, c.height);
      g.drawImage(img, 0, 0);
      const y0 = img.height;
      g.fillStyle = '#111417';
      g.font = '600 15px system-ui, sans-serif';
      g.fillText(title.replace(/<[^>]+>/g, ''), pad, y0 + 22);
      g.font = '12px system-ui, sans-serif';
      g.fillStyle = '#4a4f55';
      g.fillText(`${subtitle}${median ? ' · 3×3 median filter (display only)' : ''}${targetColors ? ` · target ${targetOp === 'ge' ? '≥' : '≤'} ${target}${param.unit ? ` ${param.unit}` : ''}` : ''}`, pad, y0 + 40);
      const bw = Math.min(420, c.width - 2 * pad - 140);
      const by = y0 + 52;
      if (targetColors) {
        g.fillStyle = TARGET_OK;
        g.fillRect(pad, by, 14, 14);
        g.fillStyle = '#111417';
        g.fillText(`meets the target: ${fmtNum((share ?? 0) * 100, 0)} % of the area`, pad + 20, by + 12);
        g.fillStyle = TARGET_FAIL;
        g.fillRect(pad + 260, by, 14, 14);
        g.fillStyle = '#111417';
        g.fillText('does not', pad + 280, by + 12);
      } else {
        const grad = g.createLinearGradient(pad, 0, pad + bw, 0);
        ramp.forEach((col, k) => grad.addColorStop(k / (ramp.length - 1), col));
        g.fillStyle = grad;
        g.fillRect(pad, by, bw, 14);
        g.fillStyle = '#111417';
        g.textAlign = 'left';
        g.fillText(fmtV(lo), pad, by + 30);
        g.textAlign = 'center';
        g.fillText(fmtV((lo + hi) / 2), pad + bw / 2, by + 30);
        g.textAlign = 'right';
        g.fillText(fmtV(hi), pad + bw, by + 30);
        g.textAlign = 'left';
      }
      g.fillStyle = DISABLED_COLOR;
      g.fillRect(c.width - pad - 120, by, 14, 14);
      g.fillStyle = '#111417';
      g.fillText('no value', c.width - pad - 100, by + 12);
      const a = document.createElement('a');
      a.href = c.toDataURL('image/png');
      a.download = `${fileBase()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      app.toast('error', `PNG export failed: ${errorMessage(e)}`);
    }
  };

  // Histogram of the unfiltered values, share of the area per bin.
  const hist = useMemo(() => {
    if (!raw || !stats) return null;
    const nb = Math.min(20, Math.max(5, Math.ceil(Math.sqrt(stats.n))));
    const [hl, hh] = stats.max > stats.min ? [stats.min, stats.max] : [stats.min - 0.5, stats.max + 0.5];
    return areaHistogram(raw, areas, hl, hh, nb);
  }, [raw, stats, areas]);
  const histData = useMemo<Data[]>(() => {
    if (!hist) return [];
    const centres = hist.share.map((_, k) => (hist.edges[k] + hist.edges[k + 1]) / 2);
    const w = hist.edges[1] - hist.edges[0];
    return [
      {
        type: 'bar',
        x: centres,
        y: hist.share.map((s) => s * 100),
        width: centres.map(() => w * 0.9),
        marker: { color: centres.map((c) => rampColor(ramp, (c - lo) / (hi - lo || 1))), line: { width: 0 } },
        customdata: hist.share.map((_, k) => [hist.edges[k], hist.edges[k + 1], hist.area[k]]),
        hovertemplate: `%{customdata[0]:.${param.digits}f} to %{customdata[1]:.${param.digits}f}${param.unit ? ` ${param.unit}` : ''}<br>%{y:.1f} % of the area (%{customdata[2]:.1f} m²)<extra></extra>`,
        name: 'share of the area',
      } as Data,
    ];
  }, [hist, ramp, lo, hi, param.digits, param.unit]);
  const histLayout = useMemo<Partial<Layout>>(() => {
    const e = hist?.edges ?? [0, 1];
    const w = e.length > 1 ? e[1] - e[0] : 1;
    const range: [number, number] = [e[0] - w / 2, e[e.length - 1] + w / 2];
    const tIn = target !== null && target >= range[0] && target <= range[1];
    return {
      height: 210,
      margin: { l: 44, r: 10, t: 10, b: 40 },
      bargap: 0.05,
      xaxis: { title: { text: `${diffOn ? 'Δ ' : ''}${param.axis}${unit}` }, range },
      yaxis: { title: { text: '% of the area' }, rangemode: 'tozero' },
      showlegend: false,
      shapes: tIn ? [{ type: 'line', xref: 'x', yref: 'paper', x0: target, x1: target, y0: 0, y1: 1, line: { color: '#f0a83a', width: 1.5, dash: 'dash' } }] : [],
    };
  }, [hist, diffOn, param.axis, unit, target]);
  const histTable = (): DataTable => ({
    columns: [`from${unit}`, `to${unit}`, 'area (m²)', 'share of the area (%)'],
    rows: (hist?.share ?? []).map((s, k) => [hist!.edges[k], hist!.edges[k + 1], hist!.area[k], s * 100]),
  });

  if (!entries.length) return <EmptyState title="No simulation" />;
  const noGrids = !gridSims.length && !probing && !(bundles.get(simId)?.loading ?? false);
  if (noGrids && !results?.grids?.length) {
    return (
      <EmptyState title="No receiver grid results">
        Add a surface or plane grid in the editor (<a href={href.editor(pageSimId)}>Grids tab</a>) and run the simulation: every valid grid point gets the room parameters, mapped here.
      </EmptyState>
    );
  }

  const hoverPoint = hover && points[hover.i] ? hover : null;
  const groups = new Map<string, SimEntry[]>();
  for (const e of gridSims) groups.set(e.path, [...(groups.get(e.path) ?? []), e]);

  return (
    <div className="results grid-maps">
      <div className="results-main">
        <section className="card gm-controls">
          <div className="gm-row">
            <label className="inline-field">
              <span>Simulation</span>
              <select value={simId} onChange={(e) => setSimId(e.target.value)}>
                {!gridSims.some((e) => e.id === simId) && <option value={simId}>{sim?.name ?? simId}</option>}
                {[...groups.entries()].map(([path, list]) => (
                  <optgroup key={path} label={path}>
                    {list.map((e) => (
                      <option key={e.id} value={e.id}>
                        {e.name}
                        {e.id === pageSimId ? ' (this one)' : ''}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>
            <label className="inline-field">
              <span>Grid</span>
              <select value={gid ?? ''} onChange={(e) => setGridId(e.target.value)} disabled={!gridInfos.length}>
                {gridInfos.map((g) => (
                  <option key={g.grid_id} value={g.grid_id}>
                    {gridName(g.grid_id)} ({g.sources[0]?.n_valid ?? 0}/{g.sources[0]?.n_points ?? 0} points)
                  </option>
                ))}
              </select>
            </label>
            <label className="inline-field">
              <span>Source</span>
              <select value={source} onChange={(e) => setSource(e.target.value)} disabled={!srcIds.length}>
                {srcIds.length > 1 && <option value={ALL}>All sources (average)</option>}
                {srcIds.map((s) => (
                  <option key={s} value={s}>
                    {srcName(s)}
                  </option>
                ))}
              </select>
            </label>
            <label
              className="inline-field"
              title={
                diffCandidates.length
                  ? 'Map the difference with another simulation that has the same grid (same id and point count)'
                  : 'No other simulation has results for this grid (same id and point count): duplicate this simulation, change it and run it to compare'
              }
            >
              <span>Difference with</span>
              <select value={diffOn ? diffSimId! : ''} onChange={(e) => setDiffSimId(e.target.value || null)} disabled={!diffCandidates.length}>
                <option value="">none</option>
                {diffCandidates.map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.name} — {e.geometryName}
                  </option>
                ))}
              </select>
            </label>
            {probing && (
              <span className="small muted">
                <Spinner size={10} /> finding results…
              </span>
            )}
          </div>
          <div className="gm-row">
            <div className="param-buttons" role="radiogroup" aria-label="Parameter">
              {GRID_PARAMS.map((p) => (
                <button key={p.key} type="button" role="radio" aria-checked={paramKey === p.key} className={`chip-btn ${paramKey === p.key ? 'active' : ''}`} title={p.title} onClick={() => setParamKey(p.key)}>
                  <span dangerouslySetInnerHTML={{ __html: p.axis }} />
                </button>
              ))}
            </div>
          </div>
          <div className="gm-row">
            <div className={`band-picks ${param.scalar ? 'disabled' : ''}`} role="radiogroup" aria-label="Band">
              {BANDS_HZ.map((f, i) => (
                <button key={f} type="button" role="radio" disabled={param.scalar} aria-checked={band === i} className={`chip-btn ${band === i && !param.scalar ? 'active' : ''}`} onClick={() => setBand(i)}>
                  {bandLabel(f)}
                </button>
              ))}
              {BAND_MEANS.map((m) => (
                <button
                  key={String(m.value)}
                  type="button"
                  role="radio"
                  disabled={param.scalar}
                  aria-checked={band === m.value}
                  className={`chip-btn ${band === m.value && !param.scalar ? 'active' : ''}`}
                  title={`${m.label} (${param.energetic ? 'energy' : 'arithmetic'} mean)`}
                  onClick={() => setBand(m.value)}
                >
                  {m.label.replace('mean ', '')}
                </button>
              ))}
              {param.scalar && <span className="small muted">single number (no band)</span>}
            </div>
            <div className="grow" />
            <Segmented<'squares' | 'interp'>
              value={interp ? 'interp' : 'squares'}
              onChange={(v) => setInterp(v === 'interp')}
              options={[
                { value: 'interp', label: 'Interpolated', title: 'Colours interpolated between neighbouring points' },
                { value: 'squares', label: 'Squares', title: 'One square per point, coloured by its own value' },
              ]}
            />
            <Toggle checked={median} onChange={setMedian} label="3×3 median (display only)" title="Smooths isolated outliers on the map only (each point: median of itself and its neighbours). Statistics, histogram, target and hover values stay unfiltered. T30 maps of large halls are noisy at 50 000 rays per point." />
          </div>
        </section>

        <section className="card gm-map">
          {main.error && <ErrorBox message={main.error} />}
          {other.error && <ErrorBox message={`Difference: ${other.error}`} />}
          {missingParam && <div className="warn-box small">{plainLabel} is not in these grid results (older run): run the simulation again.</div>}
          {diffMissingSource && <div className="warn-box small">The other simulation has no grid results for {srcName(source)}: choose “All sources” or the same source.</div>}
          {sizeMismatch && <div className="warn-box small">The two grids do not have the same points: no difference.</div>}
          <div className="gm-canvas">
            {geom ? (
              <SceneCanvas key={top ? 'top' : 'persp'} bounds={geom.bounds} topView={top} frame={frame} showGizmo={false}>
                <LayerMeshes mesh={geom.mesh} shading="ghosted" pickable={false} styleOf={(_, i) => ({ color: layerColor(i) })} />
                <PlacedObjects objects={geom.placed} catalog={catalog} showVisual pickable={false} />
                {shown && points.length > 0 && (
                  <GridMap
                    points={points}
                    frames={frames}
                    spacing={spacing}
                    values={shown}
                    lo={lo}
                    hi={hi}
                    ramp={ramp}
                    interpolate={interp}
                    colors={targetColors}
                    onHover={(i, x, y) => setHover(i === null ? null : { i, x, y })}
                    onClick={(i, add) => setPinned((p) => (p.includes(i) ? p.filter((x) => x !== i) : add ? [...p, i] : [i]))}
                  />
                )}
                <PointMarkers points={sourceMarkers} radius={markerRadius(geom.bounds)} />
                {pinned.map((i) =>
                  points[i] ? (
                    <Html key={i} position={points[i]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
                      <div className="gm-pin">{fmtV(raw?.[i])}</div>
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
            {(main.loading || other.loading) && (
              <div className="gm-loading small">
                <Spinner size={11} /> loading the grid…
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
              <button type="button" className="btn btn-sm" onClick={frameGrid} disabled={!points.length} title="Frame the grid">
                Frame grid
              </button>
            </div>
            {shown && (
              <div className="gm-legend">
                <div className="gm-legend-title" dangerouslySetInnerHTML={{ __html: `${diffOn ? 'Δ ' : ''}${param.axis}${unit}${bandPart}` }} />
                <div className="small muted ellipsis" title={`${sourceText} · ${subtitle}`}>
                  {sourceText}
                  {diffOn ? ` · ${sim?.name} − ${diffName}` : ''}
                </div>
                {targetColors ? (
                  <div className="gm-target-legend small">
                    <span>
                      <span className="swatch" style={{ background: TARGET_OK }} />✓ {targetOp === 'ge' ? '≥' : '≤'} {fmtV(target)}
                      {param.unit ? ` ${param.unit}` : ''}
                    </span>
                    <span>
                      <span className="swatch" style={{ background: TARGET_FAIL }} />✗ not met
                    </span>
                  </div>
                ) : (
                  <>
                    <div className="abs-bar" style={{ background: rampCss(ramp) }} />
                    <div className="abs-ticks small">
                      <span>{fmtV(lo)}</span>
                      <span>{fmtV((lo + hi) / 2)}</span>
                      <span>{fmtV(hi)}</span>
                    </div>
                  </>
                )}
                <div className="small muted">
                  <span className="swatch" style={{ background: DISABLED_COLOR }} /> disabled point or no value
                </div>
                {median && <div className="small text-warn">3×3 median filter (display only)</div>}
              </div>
            )}
            {hoverPoint && raw && (
              <div className="grid-hover small" style={{ left: hoverPoint.x + 14, top: hoverPoint.y + 12 }}>
                <b>Point {hoverPoint.i + 1}</b>
                <span className="muted"> · ({points[hoverPoint.i].map((v) => fmtNum(v, 2)).join(', ')})</span>
                {!valid[hoverPoint.i] ? (
                  <div className="text-warn">disabled (outside the air volume, or too close to a surface or a source)</div>
                ) : diffOn ? (
                  <div>
                    {sim?.name}: {fmtV(valuesA?.[hoverPoint.i])} · {diffName}: {fmtV(valuesB?.[hoverPoint.i])} · Δ <b>{fmtV(raw[hoverPoint.i])}</b>
                    {param.unit ? ` ${param.unit}` : ''}
                  </div>
                ) : (
                  <div>
                    {plainLabel}: <b>{fmtV(raw[hoverPoint.i])}</b>
                    {param.unit ? ` ${param.unit}` : ''}
                  </div>
                )}
                {median && valid[hoverPoint.i] && <div className="muted">filtered (display): {fmtV(shown?.[hoverPoint.i])}</div>}
              </div>
            )}
          </div>
          <div className="row gap wrap gm-foot">
            <span className="muted small">Hover a square for its value; click to label it (Ctrl-click several).</span>
            <div className="grow" />
            <button type="button" className="btn btn-xs" disabled={!raw} onClick={exportCsv} title="One row per grid point: position, validity, area and value (unfiltered; the filtered value too when the median filter is on)">
              Map data .csv
            </button>
            <button type="button" className="btn btn-xs" disabled={!raw || !geom} onClick={() => void exportPng()} title="The current view on a white background, with its legend">
              View .png
            </button>
          </div>
        </section>
      </div>

      <aside className="results-side wide">
        <section className="card">
          <h3>Colour scale</h3>
          <div className="row gap wrap">
            <Segmented<'auto' | 'fixed'>
              value={fixed ? 'fixed' : 'auto'}
              onChange={(v) => setFixed(v === 'fixed' ? { min: lo, max: hi } : null)}
              options={[
                { value: 'auto', label: 'Auto', title: 'From the values on the map' },
                { value: 'fixed', label: 'Fixed', title: 'Kept when switching simulations, sources or grids (per parameter and band)' },
              ]}
            />
            <label className="inline-field">
              <span>Min</span>
              <NumField value={lo} step={0.1} digits={param.digits} width={64} onChange={(v) => v !== null && setFixed({ min: v, max: Math.max(v + 1e-6, hi) })} ariaLabel="Scale minimum" />
            </label>
            <label className="inline-field">
              <span>Max</span>
              <NumField value={hi} step={0.1} digits={param.digits} width={64} onChange={(v) => v !== null && setFixed({ min: Math.min(lo, v - 1e-6), max: v })} ariaLabel="Scale maximum" />
            </label>
          </div>
          <div className="small muted">{fixed ? `Fixed for ${diffOn ? 'differences of ' : ''}${plainLabel}${param.scalar ? '' : ` at ${bandText(band)}`}: it stays when you switch simulations, sources or grids.` : 'Automatic: rounded range of the values shown. Edit min or max to fix it.'}</div>
        </section>

        <section className="card">
          <h3>Statistics</h3>
          {stats ? (
            <>
              <table className="table compact gm-stats">
                <tbody>
                  <tr>
                    <td>Minimum</td>
                    <td className="num">{fmtV(stats.min)}</td>
                    <td>Maximum</td>
                    <td className="num">{fmtV(stats.max)}</td>
                  </tr>
                  <tr>
                    <td>Mean</td>
                    <td className="num strong">{fmtV(stats.mean)}</td>
                    <td>Median</td>
                    <td className="num strong">{fmtV(stats.median)}</td>
                  </tr>
                  <tr>
                    <td>Std. dev.</td>
                    <td className="num">{fmtV(stats.std)}</td>
                    <td>Area</td>
                    <td className="num">
                      {fmtNum(stats.area, stats.area >= 100 ? 0 : 1)} m²
                    </td>
                  </tr>
                </tbody>
              </table>
              <div className="small muted">
                {diffOn ? 'Of the difference, ' : ''}area-weighted over {stats.n} point{stats.n === 1 ? '' : 's'} with a value{param.unit ? `, in ${param.unit}` : ''}; never filtered.
              </div>
            </>
          ) : main.loading ? (
            <Spinner />
          ) : (
            <div className="small muted">No value on this map.</div>
          )}
        </section>

        <section className="card">
          <h3>Target</h3>
          <div className="row gap wrap">
            <span className="small" dangerouslySetInnerHTML={{ __html: `${diffOn ? 'Δ ' : ''}${param.axis}` }} />
            <select value={targetOp} onChange={(e) => setTargetOp(e.target.value as TargetOp)} aria-label="Target condition">
              <option value="ge">≥ (at least)</option>
              <option value="le">≤ (at most)</option>
            </select>
            <NumField value={target} allowNull placeholder="value" step={0.1} digits={param.digits} width={70} onChange={setTarget} ariaLabel="Target value" />
            <span className="unit">{param.unit}</span>
          </div>
          {share !== null && target !== null ? (
            <div className="gm-share">
              <b>{fmtNum(share * 100, 0)} %</b> of the area meets it <span className="muted small">({nMeet} of {stats?.n ?? 0} points, area-weighted)</span>
            </div>
          ) : (
            <div className="small muted">Enter a value: the share of the area (weighted by each point&apos;s area) that meets it.</div>
          )}
          <Toggle checked={showTarget} onChange={setShowTarget} disabled={target === null} label="Show on the map (pass / fail)" />
        </section>

        <section className="card">
          <h3>Distribution</h3>
          {hist ? (
            <Plot data={histData} layout={histLayout} tools filename={`${fileBase()}_histogram`} table={histTable} />
          ) : (
            <div className="small muted">No value on this map.</div>
          )}
          <div className="small muted">Share of the area per bin, unfiltered values{target !== null ? '; dashed: the target (when inside the range)' : ''}.</div>
        </section>

        {pinned.length > 0 && raw && (
          <section className="card">
            <div className="card-head">
              <h3>Labelled points</h3>
              <div className="grow" />
              <button type="button" className="link small" onClick={() => setPinned([])}>
                Clear
              </button>
            </div>
            <table className="table compact">
              <thead>
                <tr>
                  <th>#</th>
                  <th className="num">x</th>
                  <th className="num">y</th>
                  <th className="num">z</th>
                  <th className="num">{diffOn ? 'Δ' : plainLabel}</th>
                </tr>
              </thead>
              <tbody>
                {pinned.map((i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    {points[i].map((v, k) => (
                      <td key={k} className="num">
                        {fmtNum(v, 2)}
                      </td>
                    ))}
                    <td className="num strong">{valid[i] ? fmtV(raw[i]) : 'disabled'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {gridInfo && (
          <section className="card">
            <h3>Grid run</h3>
            <dl className="kv small">
              <dt>Points</dt>
              <dd>
                {gridInfo.sources[0]?.n_valid ?? 0} valid of {gridInfo.sources[0]?.n_points ?? 0}, spacing {fmtNum(spacing, 1)} m
              </dd>
              <dt>Solver</dt>
              <dd>geometrical only (energy-time curve)</dd>
              <dt>Rays</dt>
              <dd>{(gridInfo.sources[0]?.n_rays ?? 0).toLocaleString()} per point</dd>
              <dt>Receiver sphere</dt>
              <dd>{fmtNum(gridInfo.sources[0]?.rcv_radius_m ?? null, 2)} m radius</dd>
            </dl>
            {results?.stale && <div className="small text-warn">Stale: the simulation was edited after this run (the grid may have changed since).</div>}
            {(paramKey === 'T30' || paramKey === 'T20' || paramKey === 'EDT') && (
              <div className="small text-warn">Decay times on grids are the noisiest parameter (T30 up to ~2 JND at isolated points in a large hall at 50 000 rays): read the median, or turn the median filter on.</div>
            )}
          </section>
        )}
      </aside>
    </div>
  );
}
