import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout, PlotMouseEvent } from 'plotly.js';
import { api, ApiError, BANDS_HZ, errorMessage, type Arrival, type ArrivalKind, type CatalogObject, type MeshData, type PlacedObject, type Reflections, type Simulation } from '../../api';
import { Plot } from '../../components/Plot';
import { EmptyState, ErrorBox, NumField, Segmented, Spinner, Toggle } from '../../components/ui';
import { mixColor } from '../../lib/colors';
import type { DataTable } from '../../lib/export';
import { bandLabel, fmtNum } from '../../lib/format';
import { azElTowards, dirFromAzEl, type Bounds } from '../../lib/geom';
import { weightedTotal, WEIGHTING_DB } from '../../lib/metrics';
import { href } from '../../lib/router';
import { PlacedObjects } from '../../three/Objects';
import { ReflectionPaths, type ArrivalPath, type PathAnimation } from '../../three/ReflectionPaths';
import { LayerMeshes, PointMarkers, SceneCanvas, markerRadius, type FrameRequest, type PointMarker } from '../../three/Scene';

/**
 * Colour per reflection order (0 = direct sound, 4 = 4 and more): luminous hues that
 * glow on the dark 3D view and stay apart for colour-blind readers (validated
 * adjacent-pair CVD ΔE ≥ 11), away from the green source and blue receiver markers.
 */
export const ORDER_COLORS = ['#f4f6f8', '#ffc24a', '#2ee6d0', '#ff5fa2', '#9b8cff'];
const ORDER_NAMES = ['Direct', '1st order', '2nd order', '3rd order', '4th order'];
/** Neutral room under the coloured paths. */
const ROOM_COLOR = '#7f8a94';
/** Default number of paths drawn in 3D (the strongest). */
const DEFAULT_MAX_PATHS = 60;
/** Slow-motion choices of the pulse animation: ms of sound per second. */
const RATES = [0.5, 1, 2, 5, 10, 20, 50];
const KIND_SYMBOL: Record<ArrivalKind, string> = { direct: 'diamond', specular: 'circle', diffracted: 'triangle-up' };
const KINDS: { value: ArrivalKind; label: string }[] = [
  { value: 'direct', label: 'Direct' },
  { value: 'specular', label: 'Specular' },
  { value: 'diffracted', label: 'Diffracted' },
];
const ORDERS = [0, 1, 2, 3, 4];
const MAX_ROWS = 400;

type Band = 'total' | number;
type Weighting = 'Z' | 'A' | 'C';
type LookMode = 'own' | 'x' | 'y' | 'source' | 'custom';
type ScaleKind = 'none' | 'speech' | 'music';
type DiscRadius = 'elevation' | 'level';

/** Arrival-time window, ms re the direct sound: [from, to). */
interface TimeGroup {
  key: string;
  label: string;
  from: number;
  to: number | null;
  color: string;
  title: string;
}

const DIRECT_GROUP: TimeGroup = { key: 'direct', label: 'Direct', from: -Infinity, to: 2, color: '#d8dde2', title: 'The direct sound and what arrives within 2 ms of it (heard as one with it)' };

/**
 * Arrival-time groups. Speech: the 50 ms limit of C50 and D50; music: the 80 ms limit of C80
 * (ISO 3382-1); both start with the 2 ms of the direct sound.
 */
export const TIME_GROUPS: Record<Exclude<ScaleKind, 'none'>, TimeGroup[]> = {
  speech: [
    DIRECT_GROUP,
    { key: 'useful', label: 'Useful', from: 2, to: 50, color: '#3ecf6e', title: 'Early reflections that support speech: the 50 ms window of C50 and D50 (ISO 3382-1)' },
    { key: 'mid', label: '50–80 ms', from: 50, to: 80, color: '#f0a83a', title: 'Past the speech limit (50 ms), within the music one (80 ms)' },
    { key: 'late', label: 'Late', from: 80, to: null, color: '#e5484d', title: 'Late arrivals: they blur speech; a strong isolated one is heard as an echo' },
  ],
  music: [
    DIRECT_GROUP,
    { key: 'early', label: 'Early', from: 2, to: 80, color: '#3ecf6e', title: 'Early reflections that support music: the 80 ms window of C80 (ISO 3382-1)' },
    { key: 'late', label: 'Late', from: 80, to: null, color: '#e5484d', title: 'Late arrivals (reverberance); a strong isolated one is heard as an echo' },
  ],
};

const inGroup = (g: TimeGroup, delay: number) => delay >= g.from && (g.to === null || delay < g.to);
const groupWindow = (g: TimeGroup) => (g.from === -Infinity ? `< ${g.to}` : g.to === null ? `≥ ${g.from}` : `${g.from}–${g.to}`);
const REF_SPREAD_COLOR = '#9ec5f4';
const REF_SLOPE_COLOR = '#c9b2f0';
type SortKey = 'kind' | 'order' | 'time' | 'delay' | 'distance' | 'level' | 'rel' | 'az' | 'el' | 'surfaces';

interface Row {
  i: number;
  e: Arrival;
  level: number | null;
  rel: number | null;
  azRel: number;
}

const orderColor = (o: number) => ORDER_COLORS[Math.min(o, ORDER_COLORS.length - 1)];

/** Signed angle in (-180, 180]. */
function signedDeg(a: number): number {
  const x = (((a + 180) % 360) + 360) % 360 - 180;
  return x === -180 ? 180 : x;
}

function levelOf(e: Arrival, band: Band, w: Weighting): number | null {
  if (band === 'total') return w === 'Z' && e.level_db !== null ? e.level_db : weightedTotal(e.spl_db, w);
  const v = e.spl_db[band];
  return v === null || v === undefined ? null : v;
}

/** Layer name for display; an object part ('obj:<catalog id>:<part>') gets its catalog name. */
function prettySurface(s: string, catalog?: Map<string, CatalogObject>): string {
  const m = /^obj:([^:]+):(.+)$/.exec(s);
  if (m) return `${catalog?.get(m[1])?.name ?? m[1]} (${m[2]})`;
  return s.replace(/^mat_/, '').replace(/^obj:/, '');
}

const SORTS: Record<SortKey, (r: Row) => number | string> = {
  kind: (r) => r.e.kind,
  order: (r) => r.e.order,
  time: (r) => r.e.time_ms,
  delay: (r) => r.e.delay_ms,
  distance: (r) => r.e.distance_m,
  level: (r) => r.level ?? -Infinity,
  rel: (r) => r.rel ?? -Infinity,
  az: (r) => r.azRel,
  el: (r) => r.e.elevation_deg,
  surfaces: (r) => r.e.surfaces.join(' '),
};

/** Time slider and read-out of the pulse animation, updated by the animation clock. */
function PulseClock({ subs, initial, endMs, tDirect, onSeek }: { subs: Set<(ms: number) => void>; initial: number; endMs: number; tDirect: number | null; onSeek: (ms: number) => void }) {
  const [ms, setMs] = useState(initial);
  useEffect(() => {
    subs.add(setMs);
    return () => {
      subs.delete(setMs);
    };
  }, [subs]);
  return (
    <>
      <input type="range" className="grow" min={0} max={endMs} step={0.1} value={Math.min(ms, endMs)} onChange={(e) => onSeek(Number(e.target.value))} aria-label="Time after emission (ms)" />
      <span className="small mono path-clock" title="Time after emission (and re the direct sound)">
        {fmtNum(ms, 1)} ms{tDirect !== null ? ` (${ms >= tDirect ? '+' : '−'}${fmtNum(Math.abs(ms - tDirect), 1)})` : ''}
      </span>
    </>
  );
}

export function ReflectionsView({
  simId,
  sim,
  mesh,
  bounds,
  placed,
  catalog,
  initialSource,
  initialReceiver,
}: {
  simId: string;
  sim: Simulation;
  mesh: MeshData | null;
  bounds: Bounds | null;
  placed: PlacedObject[];
  catalog: Map<string, CatalogObject>;
  initialSource: string | null;
  initialReceiver: string | null;
}) {
  const [sourceId, setSourceId] = useState<string | null>(initialSource ?? sim.sources[0]?.id ?? null);
  const [receiverId, setReceiverId] = useState<string | null>(initialReceiver ?? sim.receivers[0]?.id ?? null);
  const [maxOrder, setMaxOrder] = useState(() => Math.max(0, Math.min(4, Math.round(sim.settings.ism_order ?? 2))));
  const [data, setData] = useState<Reflections | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ msg: string; status: number } | null>(null);
  const [band, setBand] = useState<Band>('total');
  const [weighting, setWeighting] = useState<Weighting>('Z');
  const [kinds, setKinds] = useState<Set<ArrivalKind>>(new Set(['direct', 'specular', 'diffracted']));
  const [orders, setOrders] = useState<Set<number>>(new Set(ORDERS));
  const [minRel, setMinRel] = useState<number | null>(null);
  const [tFrom, setTFrom] = useState<number | null>(null);
  const [tTo, setTTo] = useState<number | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'time', dir: 1 });
  const [selected, setSelected] = useState<number[]>([]);
  const [hovered, setHovered] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(true);
  const [maxPaths, setMaxPaths] = useState<number | null>(DEFAULT_MAX_PATHS);
  const [throughWalls, setThroughWalls] = useState(true);
  const [animOn, setAnimOn] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [rate, setRate] = useState(5);
  const [big, setBig] = useState(false);
  const [frameReq, setFrameReq] = useState<FrameRequest | null>(null);
  const clock = useRef(0);
  const [seekKey, setSeekKey] = useState(0);
  // The clock display subscribes to the animation itself: the view does not re-render per frame.
  const timeSubs = useRef(new Set<(ms: number) => void>());
  const onTime = useCallback((ms: number) => timeSubs.current.forEach((f) => f(ms)), []);
  const [look, setLook] = useState<LookMode>('own');
  const [customLook, setCustomLook] = useState({ az: 0, el: 0 });
  const [timeScale, setTimeScale] = useState<ScaleKind>('none');
  const [refSpread, setRefSpread] = useState(false);
  const [refSlope, setRefSlope] = useState(false);
  const [slope, setSlope] = useState(0.1);
  const [discRadius, setDiscRadius] = useState<DiscRadius>('elevation');
  const cache = useRef(new Map<string, Reflections>());

  const rcv = sim.receivers.find((r) => r.id === receiverId) ?? null;
  const src = sim.sources.find((s) => s.id === sourceId) ?? null;
  /** Look direction sent to the server (null = the receiver's own). */
  const lookDir = useMemo<{ azimuth_deg: number; elevation_deg: number } | null>(() => {
    if (look === 'x') return { azimuth_deg: 0, elevation_deg: 0 };
    if (look === 'y') return { azimuth_deg: 90, elevation_deg: 0 };
    if (look === 'source' && rcv && src) return azElTowards(rcv.position, src.position);
    if (look === 'custom') return { azimuth_deg: customLook.az, elevation_deg: customLook.el };
    return null;
  }, [look, customLook, rcv, src]);
  const lookKey = lookDir ? `${lookDir.azimuth_deg.toFixed(1)},${lookDir.elevation_deg.toFixed(1)}` : 'own';
  const lookShown = lookDir ?? (rcv ? { azimuth_deg: rcv.azimuth_deg, elevation_deg: rcv.elevation_deg } : null);

  // A new pair or order clears the selection; a new look direction keeps it (same arrivals).
  useEffect(() => {
    setSelected([]);
    setHovered(null);
  }, [sourceId, receiverId, maxOrder]);

  useEffect(() => {
    if (!sourceId || !receiverId) return;
    const key = `${sourceId}|${receiverId}|${maxOrder}|${lookKey}`;
    const hit = cache.current.get(key);
    if (hit) {
      setData(hit);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .getReflections(simId, sourceId, receiverId, maxOrder, lookDir)
      .then((r) => {
        if (cancelled) return;
        cache.current.set(key, r);
        setData(r);
        setError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setData(null);
        setError({ msg: errorMessage(e), status: e instanceof ApiError ? e.status : 0 });
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // lookDir is described by lookKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simId, sourceId, receiverId, maxOrder, lookKey]);

  const rows = useMemo<Row[]>(() => {
    if (!data) return [];
    const direct = data.events.find((e) => e.kind === 'direct');
    const directOk = !!direct && direct.level_rel_db !== null;
    const dLevel = directOk ? levelOf(direct!, band, weighting) : null;
    return data.events.map((e, i) => {
      const level = levelOf(e, band, weighting);
      let rel: number | null;
      if (band === 'total' && weighting === 'Z') rel = e.level_rel_db;
      else rel = level !== null && dLevel !== null ? level - dLevel : null;
      return { i, e, level, rel, azRel: signedDeg(e.azimuth_rel_deg) };
    });
  }, [data, band, weighting]);

  const filtered = useMemo(() => {
    const out = rows.filter((r) => {
      if (!kinds.has(r.e.kind)) return false;
      if (!orders.has(Math.min(r.e.order, 4))) return false;
      if (minRel !== null && r.rel !== null && r.rel < minRel) return false;
      if (tFrom !== null && r.e.delay_ms < tFrom) return false;
      if (tTo !== null && r.e.delay_ms > tTo) return false;
      return true;
    });
    const f = SORTS[sort.key];
    return out.sort((a, b) => {
      const x = f(a);
      const y = f(b);
      const c = typeof x === 'string' || typeof y === 'string' ? String(x).localeCompare(String(y)) : x - y;
      return c !== 0 ? c * sort.dir : a.e.time_ms - b.e.time_ms;
    });
  }, [rows, kinds, orders, minRel, tFrom, tTo, sort]);

  const filtersActive = kinds.size < 3 || orders.size < ORDERS.length || minRel !== null || tFrom !== null || tTo !== null;
  const resetFilters = () => {
    setKinds(new Set(['direct', 'specular', 'diffracted']));
    setOrders(new Set(ORDERS));
    setMinRel(null);
    setTFrom(null);
    setTTo(null);
  };

  const select = (i: number, additive: boolean) => {
    setSelected((prev) => {
      if (additive) return prev.includes(i) ? prev.filter((x) => x !== i) : [...prev, i];
      return prev.length === 1 && prev[0] === i ? [] : [i];
    });
  };

  const levelTitle = band === 'total' ? `Level, total${weighting !== 'Z' ? ` dB(${weighting})` : ''}` : `Level, ${bandLabel(BANDS_HZ[band])} Hz band`;

  const table = useMemo<DataTable>(
    () => ({
      columns: ['kind', 'order', 'time_ms', 'delay_ms', 'distance_m', `level_db (${band === 'total' ? `total ${weighting}` : `${BANDS_HZ[band]} Hz`})`, 'level_re_direct_db', 'azimuth_rel_deg', 'azimuth_world_deg', 'elevation_deg', 'surfaces'],
      rows: filtered.map((r) => [r.e.kind, r.e.order, r.e.time_ms, r.e.delay_ms, r.e.distance_m, r.level, r.rel, r.azRel, r.e.azimuth_deg, r.e.elevation_deg, r.e.surfaces.join(' > ')]),
    }),
    [filtered, band, weighting],
  );

  const directRow = useMemo(() => rows.find((r) => r.e.kind === 'direct' && r.level !== null && r.e.level_rel_db !== null) ?? null, [rows]);
  const xMax = useMemo(() => Math.max(10, ...filtered.map((r) => r.e.delay_ms)), [filtered]);

  const plotData = useMemo<Data[]>(() => {
    const out: Data[] = [];
    const lv = filtered.filter((r) => r.level !== null);
    if (!lv.length) return out;
    const floor = Math.min(...lv.map((r) => r.level!)) - 6;
    // Reference curves from the direct sound (drawn under the stems).
    if (directRow && data) {
      const ld = directRow.level!;
      const d0 = Math.max(0.01, directRow.e.distance_m);
      const n = 160;
      if (refSpread) {
        const t = Array.from({ length: n + 1 }, (_, k) => (xMax * 1.03 * k) / n);
        out.push({
          type: 'scatter',
          mode: 'lines',
          x: t,
          y: t.map((ms) => ld - 20 * Math.log10(1 + (data.speed_of_sound * ms) / 1000 / d0)),
          name: '1/r from the direct sound',
          line: { color: REF_SPREAD_COLOR, width: 1.5, dash: 'dash' },
          hovertemplate: 'Spherical spreading only (lossless surfaces)<br>+%{x:.1f} ms: %{y:.1f} dB<extra></extra>',
          meta: 'decor',
        } as Data);
      }
      if (refSlope && slope > 0) {
        const tEnd = Math.min(xMax * 1.03, (ld - floor) / slope);
        out.push({
          type: 'scatter',
          mode: 'lines',
          x: [0, tEnd],
          y: [ld, ld - slope * tEnd],
          name: `−${fmtNum(slope, 3)} dB/ms`,
          line: { color: REF_SLOPE_COLOR, width: 1.5, dash: 'dot' },
          hovertemplate: `Decay of ${fmtNum(slope, 3)} dB/ms (T = ${fmtNum(60 / slope / 1000, 2)} s)<br>+%{x:.1f} ms: %{y:.1f} dB<extra></extra>`,
          meta: 'decor',
        } as Data);
      }
    }
    for (const o of ORDERS) {
      const list = lv.filter((r) => Math.min(r.e.order, 4) === o);
      if (!list.length) continue;
      const color = orderColor(o);
      out.push({
        type: 'scatter',
        mode: 'lines',
        x: list.flatMap((r) => [r.e.delay_ms, r.e.delay_ms, null]),
        y: list.flatMap((r) => [floor, r.level, null]),
        line: { color, width: 1.2 },
        hoverinfo: 'skip',
        showlegend: false,
        meta: 'decor',
      } as Data);
      out.push({
        type: 'scatter',
        mode: 'markers',
        name: o === 0 ? 'direct' : `order ${o}`,
        x: list.map((r) => r.e.delay_ms),
        y: list.map((r) => r.level),
        customdata: list.map((r) => r.i),
        text: list.map((r) => `${r.e.kind}${r.e.surfaces.length ? `: ${r.e.surfaces.map((x) => prettySurface(x)).join(' > ')}` : ''}`),
        marker: { color, size: list.map((r) => (r.e.kind === 'direct' ? 11 : 7)), symbol: list.map((r) => KIND_SYMBOL[r.e.kind]), line: { color: '#111417', width: 0.5 } },
        hovertemplate: '%{text}<br>+%{x:.2f} ms: %{y:.1f} dB<extra></extra>',
      } as Data);
    }
    const sel = lv.filter((r) => selected.includes(r.i));
    if (sel.length) {
      out.push({
        type: 'scatter',
        mode: 'markers',
        x: sel.map((r) => r.e.delay_ms),
        y: sel.map((r) => r.level),
        marker: { size: 17, symbol: 'circle-open', color: '#ffffff', line: { color: '#ffffff', width: 2 } },
        hoverinfo: 'skip',
        showlegend: false,
        meta: 'decor',
      } as Data);
    }
    return out;
  }, [filtered, selected, directRow, data, refSpread, refSlope, slope, xMax]);

  const groups = timeScale === 'none' ? null : TIME_GROUPS[timeScale];
  /** Energy of the filtered arrivals per time group (band and weighting as shown). */
  const groupStats = useMemo(() => {
    if (!groups) return null;
    const lv = filtered.filter((r) => r.level !== null);
    const energy = (l: Row[]) => l.reduce((a, r) => a + 10 ** (r.level! / 10), 0);
    const total = energy(lv);
    return groups.map((g) => {
      const inG = lv.filter((r) => inGroup(g, r.e.delay_ms));
      const e = energy(inG);
      return { g, n: inG.length, share: total > 0 ? e / total : null, level: e > 0 ? 10 * Math.log10(e) : null };
    });
  }, [groups, filtered]);

  const plotLayout = useMemo<Partial<Layout>>(() => {
    const direct = filtered.find((r) => r.e.kind === 'direct' && r.level !== null);
    const x0 = Math.min(-1, ...filtered.map((r) => r.e.delay_ms)) - 1;
    const x1 = xMax * 1.03 + 1;
    const shapes: Layout['shapes'] = (groups ?? []).map((g) => ({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: Math.max(x0, g.from),
      x1: g.to === null ? x1 : g.to,
      y0: 0,
      y1: 1,
      fillcolor: g.color,
      opacity: 0.1,
      line: { width: 0 },
      layer: 'below',
    }));
    const groupLabels: Layout['annotations'] = (groupStats ?? [])
      .filter((x) => x.g.key !== 'direct')
      .map((x) => ({
        x: x.g.from,
        y: 1,
        xref: 'x',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'top',
        showarrow: false,
        text: `${x.g.label}${x.share !== null ? ` ${fmtNum(x.share * 100, 0)} %` : ''}`,
        font: { size: 10, color: x.g.color },
      }));
    return {
      height: 280,
      margin: { l: 52, r: 12, t: 28, b: 40 },
      xaxis: { title: { text: 'Arrival time re. direct sound (ms)' }, zeroline: true, range: [x0, x1] },
      yaxis: { title: { text: `${levelTitle} (dB)` } },
      shapes,
      // With time groups their labels take the top edge: the direct sound is labelled from below.
      annotations: [...(direct ? [{ x: direct.e.delay_ms, y: direct.level!, text: 'direct', showarrow: true, arrowhead: 0, ax: 24, ay: groups ? 22 : -18, font: { size: 10 } }] : []), ...groupLabels],
      showlegend: true,
    };
  }, [filtered, levelTitle, groups, groupStats, xMax]);

  // Directions of arrival: angle = azimuth re the look direction (front up, + = left).
  const discData = useMemo<Data[]>(() => {
    const lv = filtered.filter((r) => r.level !== null);
    if (!lv.length) return [];
    const maxLevel = Math.max(...lv.map((r) => r.level!));
    const relOf = (r: Row) => (r.rel !== null ? r.rel : r.level! - maxLevel);
    const size = (r: Row) => 5 + 13 * Math.min(1, Math.max(0, (relOf(r) + 30) / 30));
    const radius = (r: Row) => (discRadius === 'elevation' ? 90 - r.e.elevation_deg : relOf(r));
    const out: Data[] = [];
    if (discRadius === 'elevation') {
      const th = Array.from({ length: 73 }, (_, k) => k * 5);
      out.push({ type: 'scatterpolar', mode: 'lines', r: th.map(() => 90), theta: th, line: { color: 'rgba(158,197,244,0.55)', width: 1.2, dash: 'dash' }, hoverinfo: 'skip', showlegend: false, meta: 'decor' } as Data);
    }
    for (const o of ORDERS) {
      const list = lv.filter((r) => Math.min(r.e.order, 4) === o).sort((a, b) => a.level! - b.level!);
      if (!list.length) continue;
      out.push({
        type: 'scatterpolar',
        mode: 'markers',
        name: o === 0 ? 'direct' : `order ${o}`,
        r: list.map(radius),
        theta: list.map((r) => r.azRel),
        customdata: list.map((r) => r.i),
        text: list.map(
          (r) =>
            `${r.e.kind}${r.e.surfaces.length ? `: ${r.e.surfaces.map((x) => prettySurface(x)).join(' > ')}` : ''}<br>azimuth ${fmtNum(r.azRel, 0)}°, elevation ${fmtNum(r.e.elevation_deg, 0)}°<br>${fmtNum(r.level, 1)} dB${r.rel !== null ? ` (${fmtNum(r.rel, 1)} re direct)` : ''}, +${fmtNum(r.e.delay_ms, 1)} ms`,
        ),
        marker: { color: orderColor(o), size: list.map(size), symbol: list.map((r) => KIND_SYMBOL[r.e.kind]), line: { color: '#111417', width: 0.6 }, opacity: 0.9 },
        hovertemplate: '%{text}<extra></extra>',
      } as Data);
    }
    const sel = lv.filter((r) => selected.includes(r.i));
    if (sel.length)
      out.push({
        type: 'scatterpolar',
        mode: 'markers',
        r: sel.map(radius),
        theta: sel.map((r) => r.azRel),
        marker: { size: sel.map((r) => size(r) + 8), symbol: 'circle-open', color: '#ffffff', line: { color: '#ffffff', width: 2 } },
        hoverinfo: 'skip',
        showlegend: false,
        meta: 'decor',
      } as Data);
    return out;
  }, [filtered, selected, discRadius]);

  const discLayout = useMemo<Partial<Layout>>(() => {
    const lv = filtered.filter((r) => r.level !== null);
    const rels = lv.map((r) => r.rel).filter((v): v is number => v !== null);
    const lo = Math.max(-60, Math.floor(((rels.length ? Math.min(...rels) : -30) - 3) / 5) * 5);
    const hi = Math.max(0, Math.ceil((rels.length ? Math.max(...rels) : 0) + 1));
    const radial =
      discRadius === 'elevation'
        ? { range: [0, 180], tickvals: [0, 45, 90, 135, 180], ticktext: ['+90°', '+45°', '0°', '−45°', '−90°'], angle: 68, tickangle: 68 }
        : { range: [lo, hi], ticksuffix: ' dB', angle: 68, tickangle: 68 };
    return {
      height: 410,
      margin: { l: 36, r: 36, t: 62, b: 30 },
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.16, yanchor: 'bottom' },
      polar: {
        domain: { x: [0, 1], y: [0, 1] },
        angularaxis: { rotation: 90, direction: 'counterclockwise', tickvals: [0, 45, 90, 135, 180, 225, 270, 315], ticktext: ['front', '45°', 'left 90°', '135°', 'back', '−135°', 'right −90°', '−45°'] },
        radialaxis: radial,
      } as Layout['polar'],
    };
  }, [filtered, discRadius]);

  const onDiscClick = (ev: PlotMouseEvent) => {
    const pt = ev.points?.[0] as unknown as { customdata?: unknown } | undefined;
    if (typeof pt?.customdata !== 'number') return;
    const me = ev.event as MouseEvent | undefined;
    select(pt.customdata, !!(me && (me.ctrlKey || me.metaKey || me.shiftKey)));
  };

  const onPlotClick = (ev: PlotMouseEvent) => {
    const pt = ev.points?.[0] as unknown as { customdata?: unknown } | undefined;
    const i = typeof pt?.customdata === 'number' ? pt.customdata : null;
    if (i === null) return;
    const me = ev.event as MouseEvent | undefined;
    select(i, !!(me && (me.ctrlKey || me.metaKey || me.shiftKey)));
  };

  // 3D: the strongest filtered paths, fainter when weaker (0 dB re direct = full, -30 dB = faintest).
  const maxLevel = useMemo(() => Math.max(-Infinity, ...rows.map((r) => r.level ?? -Infinity)), [rows]);
  const byId = useMemo(
    () =>
      new Map(
        rows.map((r) => {
          const rel = r.rel ?? (r.level !== null ? r.level - maxLevel : -30);
          const p: ArrivalPath = { id: r.i, path: r.e.path, color: orderColor(r.e.order), weight: Math.min(1, Math.max(0, 1 + rel / 30)), diffracted: r.e.kind === 'diffracted' };
          return [r.i, p];
        }),
      ),
    [rows, maxLevel],
  );
  const strongest = useMemo(() => [...filtered].sort((a, b) => (b.level ?? -Infinity) - (a.level ?? -Infinity)), [filtered]);
  const shown = useMemo(() => (showAll ? strongest.slice(0, maxPaths ?? strongest.length) : []), [showAll, strongest, maxPaths]);
  const paths = useMemo(() => shown.map((r) => byId.get(r.i)!).filter(Boolean), [shown, byId]);
  const focusIds = useMemo(() => {
    const ids = [...selected];
    if (hovered !== null && !ids.includes(hovered)) ids.push(hovered);
    return ids;
  }, [selected, hovered]);
  const focusPaths = useMemo(() => focusIds.map((i) => byId.get(i)).filter((x): x is ArrivalPath => !!x), [focusIds, byId]);
  const rowById = useMemo(() => new Map(rows.map((r) => [r.i, r])), [rows]);
  // Surfaces of the hovered (else the last selected) arrival, tinted with its colour.
  const hitLayers = useMemo(() => {
    const id = hovered ?? selected[selected.length - 1] ?? null;
    const r = id !== null ? rowById.get(id) : undefined;
    return new Map((r?.e.surfaces ?? []).map((sname) => [sname, orderColor(r!.e.order)]));
  }, [hovered, selected, rowById]);
  const layerStyle = useCallback(
    (name: string) => {
      const c = hitLayers.get(name);
      return c ? { color: mixColor(ROOM_COLOR, c, 0.6), hovered: true } : { color: ROOM_COLOR };
    },
    [hitLayers],
  );
  const animEnd = useMemo(() => {
    const ts = [...shown.map((r) => r.e.time_ms), ...focusIds.map((i) => rowById.get(i)?.e.time_ms ?? 0)];
    return Math.max(5, ...ts) * 1.12 + 4;
  }, [shown, focusIds, rowById]);
  const anim = useMemo<PathAnimation>(() => ({ on: animOn, playing, slow: 1000 / rate, endMs: animEnd, c: data?.speed_of_sound ?? 343 }), [animOn, playing, rate, animEnd, data]);
  const seek = useCallback(
    (ms: number) => {
      clock.current = ms;
      onTime(ms);
      setSeekKey((k) => k + 1);
    },
    [onTime],
  );
  const tDirect = rows.find((r) => r.e.kind === 'direct')?.e.time_ms ?? null;
  const pathTip = useCallback(
    (id: number) => {
      const r = rowById.get(id);
      if (!r) return null;
      const e = r.e;
      const head = e.kind === 'direct' ? 'Direct sound' : e.kind === 'diffracted' ? `Edge diffraction · order ${e.order}` : `${ORDER_NAMES[Math.min(e.order, 4)].replace('4th', `${e.order}th`)} reflection`;
      return (
        <>
          <div className="path-tip-head">
            <span className="swatch" style={{ background: orderColor(e.order) }} />
            {head}
          </div>
          {e.surfaces.length > 0 && <div className="path-tip-surf">{e.surfaces.map((x) => prettySurface(x, catalog)).join(' › ')}</div>}
          <div className="path-tip-grid">
            <span>Arrival</span>
            <b>{fmtNum(e.time_ms, 1)} ms</b>
            <span>Delay</span>
            <b>{e.kind === 'direct' ? '–' : `+${fmtNum(e.delay_ms, 1)} ms`}</b>
            <span>Level</span>
            <b>{fmtNum(r.level, 1)} dB</b>
            <span>Re direct</span>
            <b>{r.rel === null ? '–' : `${r.rel > 0 ? '+' : ''}${fmtNum(r.rel, 1)} dB`}</b>
          </div>
        </>
      );
    },
    [rowById, catalog],
  );
  const shownOrders = useMemo(() => [...new Set([...paths, ...focusPaths].map((p) => Math.min(rowById.get(p.id)?.e.order ?? 0, 4)))].sort((a, b) => a - b), [paths, focusPaths, rowById]);

  const points = useMemo<PointMarker[]>(
    () => [
      ...sim.sources.map((s) => ({ kind: 'source' as const, id: s.id, label: s.name, position: s.position, dir: dirFromAzEl(s.azimuth_deg, s.elevation_deg), selected: s.id === sourceId, dim: s.id !== sourceId })),
      ...sim.receivers.map((r) => {
        const d = r.id === receiverId && lookDir ? lookDir : r;
        return { kind: 'receiver' as const, id: r.id, label: r.name, position: r.position, dir: dirFromAzEl(d.azimuth_deg, d.elevation_deg), selected: r.id === receiverId, dim: r.id !== receiverId };
      }),
    ],
    [sim, sourceId, receiverId, lookDir],
  );

  const tableRef = useRef<HTMLDivElement>(null);
  const onTableKey = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    if (!filtered.length) return;
    const last = selected[selected.length - 1];
    const pos = filtered.findIndex((r) => r.i === last);
    const next = pos < 0 ? 0 : Math.max(0, Math.min(filtered.length - 1, pos + (e.key === 'ArrowDown' ? 1 : -1)));
    setSelected([filtered[next].i]);
    tableRef.current?.querySelector(`[data-row="${filtered[next].i}"]`)?.scrollIntoView({ block: 'nearest' });
  };

  const detail = selected.length ? rows.find((r) => r.i === selected[selected.length - 1]) : null;
  const toggleSet = <T,>(s: Set<T>, v: T) => {
    const n = new Set(s);
    if (n.has(v)) n.delete(v);
    else n.add(v);
    return n;
  };
  const th = (key: SortKey, label: string, num = true, title?: string) => (
    <th
      className={`${num ? 'num' : ''} sortable ${sort.key === key ? 'sorted' : ''}`}
      title={title ?? 'Sort'}
      aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}
      onClick={() => setSort((s) => (s.key === key ? { key, dir: (s.dir === 1 ? -1 : 1) as 1 | -1 } : { key, dir: key === 'level' || key === 'rel' ? -1 : 1 }))}
    >
      {label}
      {sort.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
    </th>
  );

  return (
    <div className="results refl-view">
      <div className="results-main">
        <section className="card">
          <div className="card-head">
            <h3>Reflection tracking</h3>
            <label className="inline-field">
              <span>Source</span>
              <select value={sourceId ?? ''} onChange={(e) => setSourceId(e.target.value)}>
                {sim.sources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="inline-field">
              <span>Receiver</span>
              <select value={receiverId ?? ''} onChange={(e) => setReceiverId(e.target.value)}>
                {sim.receivers.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="inline-field">
              <span>Max order</span>
              <Segmented<string> value={String(maxOrder)} onChange={(v) => setMaxOrder(Number(v))} options={ORDERS.map((o) => ({ value: String(o), label: String(o), title: o === 0 ? 'Direct sound and edge diffraction only' : `Image sources up to order ${o}` }))} />
            </div>
            {loading && <Spinner />}
          </div>
          <div className="card-head">
            <label className="inline-field">
              <span>Level</span>
              <select value={String(band)} onChange={(e) => setBand(e.target.value === 'total' ? 'total' : Number(e.target.value))}>
                <option value="total">Total (all bands)</option>
                {BANDS_HZ.map((f, i) => (
                  <option key={f} value={i}>
                    {bandLabel(f)} Hz
                  </option>
                ))}
              </select>
            </label>
            <div className="inline-field" title={band === 'total' ? 'Frequency weighting of the total level' : 'Weighting applies to the total only'}>
              <span>Weighting</span>
              <Segmented<Weighting>
                value={weighting}
                disabled={band !== 'total'}
                onChange={setWeighting}
                options={(['Z', 'A', 'C'] as Weighting[]).map((w) => ({ value: w, label: w, title: w === 'Z' ? 'No weighting' : `${w}-weighting: ${WEIGHTING_DB[w].map((x) => (x > 0 ? `+${x}` : x)).join(' / ')} dB` }))}
              />
            </div>
            <div
              className="inline-field"
              title="Look direction of the receiver for the relative azimuths (the listener turns the head). Only this view changes: the results and the downloaded IRs keep their own orientation."
            >
              <span>Look</span>
              <select value={look} onChange={(e) => setLook(e.target.value as LookMode)} aria-label="Receiver look direction">
                <option value="own">Receiver&apos;s own{rcv ? ` (${fmtNum(rcv.azimuth_deg, 0)}°, ${fmtNum(rcv.elevation_deg, 0)}°)` : ''}</option>
                <option value="x">Along +X</option>
                <option value="y">Along +Y</option>
                <option value="source">Towards the source</option>
                <option value="custom">Custom</option>
              </select>
              {look === 'custom' && (
                <>
                  <NumField value={customLook.az} min={0} max={360} step={5} digits={1} width={50} onChange={(v) => setCustomLook((c) => ({ ...c, az: v ?? 0 }))} ariaLabel="Look azimuth (degrees, CCW from +X)" title="Azimuth, 0–360°, counter-clockwise from +X" />
                  <span className="unit">az°</span>
                  <NumField value={customLook.el} min={-90} max={90} step={5} digits={1} width={46} onChange={(v) => setCustomLook((c) => ({ ...c, el: v ?? 0 }))} ariaLabel="Look elevation (degrees)" title="Elevation, −90…90°" />
                  <span className="unit">el°</span>
                </>
              )}
              {look === 'source' && lookDir && <span className="unit">({fmtNum(lookDir.azimuth_deg, 0)}°, {fmtNum(lookDir.elevation_deg, 0)}°)</span>}
            </div>
            <div className="grow" />
            <span className="muted small">
              {data ? `${filtered.length} of ${data.events.length} arrivals` : ''}
              {data ? ` · c = ${fmtNum(data.speed_of_sound, 1)} m/s` : ''}
            </span>
          </div>
          <div className="refl-filters">
            <div className="band-picks" aria-label="Kinds">
              {KINDS.map((k) => (
                <label key={k.value} className={`chip-btn ${kinds.has(k.value) ? 'active' : ''}`}>
                  <input type="checkbox" checked={kinds.has(k.value)} onChange={() => setKinds((s) => toggleSet(s, k.value))} />
                  {k.label}
                </label>
              ))}
            </div>
            <div className="band-picks" aria-label="Orders">
              {ORDERS.map((o) => (
                <label key={o} className={`chip-btn ${orders.has(o) ? 'active' : ''}`} title={o === 0 ? 'Order 0: the direct sound' : o === 1 ? 'Order 1: first reflections and first-order diffraction' : `Order ${o} reflections`}>
                  <input type="checkbox" checked={orders.has(o)} onChange={() => setOrders((s) => toggleSet(s, o))} />
                  <span className="swatch" style={{ background: orderColor(o) }} />
                  {o}
                </label>
              ))}
            </div>
            <label className="inline-field" title="Hide arrivals weaker than this, relative to the direct sound">
              <span>Min. re direct</span>
              <NumField value={minRel} allowNull min={-120} max={20} step={1} digits={1} placeholder="any" onChange={setMinRel} width={58} />
              <span className="unit">dB</span>
            </label>
            <label className="inline-field" title="Arrival time window, relative to the direct sound">
              <span>Time</span>
              <NumField value={tFrom} allowNull min={-10} max={5000} step={1} digits={1} placeholder="from" onChange={setTFrom} width={54} />
              <span className="unit">–</span>
              <NumField value={tTo} allowNull min={-10} max={5000} step={1} digits={1} placeholder="to" onChange={setTTo} width={54} />
              <span className="unit">ms</span>
            </label>
            <button type="button" className="link small" disabled={!filtersActive} onClick={resetFilters}>
              Reset filters
            </button>
          </div>
          <div className="refl-filters">
            <div className="inline-field" title="Arrival-time groups, re the direct sound: shaded on the chart, with their share of the energy of the filtered arrivals">
              <span>Scale</span>
              <Segmented<ScaleKind>
                value={timeScale}
                onChange={setTimeScale}
                options={[
                  { value: 'none', label: 'None' },
                  { value: 'speech', label: 'Speech', title: 'Direct < 2 ms · useful 2–50 ms (C50, D50) · 50–80 ms · late ≥ 80 ms' },
                  { value: 'music', label: 'Music', title: 'Direct < 2 ms · early 2–80 ms (C80) · late ≥ 80 ms' },
                ]}
              />
            </div>
            <div className="inline-field" title="Reference curves, started at the direct sound">
              <span>Reference</span>
              <label className="check inline" title="Level a reflection would have with spherical spreading only: lossless surfaces, omnidirectional source, no air absorption. Arrivals below it lost energy at their surfaces (or to the directivity).">
                <input type="checkbox" checked={refSpread} onChange={(e) => setRefSpread(e.target.checked)} disabled={!directRow} />
                <span className="swatch" style={{ background: REF_SPREAD_COLOR }} />
                1/r
              </label>
              <label className="check inline" title="A straight decay from the direct level: 60 dB over the reverberation time T">
                <input type="checkbox" checked={refSlope} onChange={(e) => setRefSlope(e.target.checked)} disabled={!directRow} />
                <span className="swatch" style={{ background: REF_SLOPE_COLOR }} />
                slope
              </label>
              <NumField value={slope} min={0.005} max={5} step={0.01} digits={3} width={56} disabled={!refSlope} onChange={(v) => setSlope(v ?? 0.1)} ariaLabel="Decay slope (dB/ms)" />
              <span className="unit">dB/ms (T = {fmtNum(60 / slope / 1000, 2)} s)</span>
            </div>
            {!directRow && data && <span className="small muted">(references need a direct sound)</span>}
          </div>
          {error && (
            <ErrorBox message={error.status === 409 ? `${error.msg}. Open the editor to assign the missing materials.` : error.msg} />
          )}
          {error?.status === 409 && (
            <div className="small">
              <a href={href.editor(simId)}>Open the editor</a>
            </div>
          )}
          {data && data.n_direct_occluded > 0 && <div className="warn-box small">The direct sound is occluded: levels relative to it are undefined.</div>}
          {data && plotData.length > 0 && (
            <Plot data={plotData} layout={plotLayout} tools filename={`${sim.name}_reflections_${sourceId}_${receiverId}`} table={table} onClick={onPlotClick} />
          )}
          {groupStats && data && filtered.length > 0 && (
            <>
              <table className="table compact refl-groups">
                <thead>
                  <tr>
                    <th>{timeScale === 'speech' ? 'Speech' : 'Music'}</th>
                    {groupStats.map((x) => (
                      <th key={x.g.key} className="num" title={x.g.title}>
                        <span className="swatch" style={{ background: x.g.color }} />
                        {x.g.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="muted">Window (ms re direct)</td>
                    {groupStats.map((x) => (
                      <td key={x.g.key} className="num">
                        {groupWindow(x.g)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="muted">Arrivals</td>
                    {groupStats.map((x) => (
                      <td key={x.g.key} className="num">
                        {x.n}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="muted">Energy share</td>
                    {groupStats.map((x) => (
                      <td key={x.g.key} className="num strong">
                        {x.share !== null ? `${fmtNum(x.share * 100, 1)} %` : '–'}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="muted">Level, energy sum (dB)</td>
                    {groupStats.map((x) => (
                      <td key={x.g.key} className="num">
                        {fmtNum(x.level, 1)}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
              <div className="small muted">
                Shares of the energy of the filtered deterministic arrivals in the chosen band ({band === 'total' ? `total, ${weighting}` : `${bandLabel(BANDS_HZ[band])} Hz`}): the diffuse reverberation of the run is not in them, so they are not the C50, D50 or C80
                of the IR. Windows from ISO 3382-1 (50 ms for speech, 80 ms for music) after a 2 ms direct-sound window.
              </div>
            </>
          )}
          {(refSpread || refSlope) && directRow && (
            <div className="small muted">
              {refSpread && (
                <>
                  <b>1/r</b>: the level of a reflection with spherical spreading only (lossless surfaces, omnidirectional source, no air absorption), L = L<sub>direct</sub> − 20 log<sub>10</sub>(1 + c·t / d<sub>direct</sub>).{' '}
                </>
              )}
              {refSlope && (
                <>
                  <b>Slope</b>: a straight decay of {fmtNum(slope, 3)} dB/ms from the direct level, i.e. 60 dB in T = {fmtNum(60 / slope / 1000, 2)} s.{' '}
                </>
              )}
              The perception thresholds of Treble&apos;s reference curves (Barron; Everest &amp; Pohlmann) are not included: their points are not published as tables.
            </div>
          )}
          {data && !filtered.length && <EmptyState title="No arrival matches the filters" />}
          <div className="muted small">
            ◆ direct · ● specular (image source) · ▲ diffracted (first-order edge). Deterministic arrivals exactly as the runs compute them, with the source&apos;s directivity and gain; computed on demand
            from the scene, not from stored results. Click a stem or a row to show its path; Ctrl-click adds to the selection.
          </div>
        </section>
        {data && filtered.length > 0 && (
          <section className="card">
            <div className="card-head">
              <h3>Directions of arrival</h3>
              <Segmented<DiscRadius>
                value={discRadius}
                onChange={setDiscRadius}
                options={[
                  { value: 'elevation', label: 'Radius: elevation', title: 'Centre = straight up, dashed circle = horizon, rim = straight down' },
                  { value: 'level', label: 'Radius: level', title: 'Level re the direct sound (dB): the rim is the loudest' },
                ]}
              />
              <div className="grow" />
              <span className="muted small">{lookShown ? `look direction ${fmtNum(lookShown.azimuth_deg, 0)}° az, ${fmtNum(lookShown.elevation_deg, 0)}° el (world)` : ''}</span>
            </div>
            <div className="refl-disc-row">
              <div className="refl-disc">
                <Plot data={discData} layout={discLayout} tools filename={`${sim.name}_directions_${sourceId}_${receiverId}`} table={table} onClick={onDiscClick} />
              </div>
              <div className="small muted refl-disc-legend">
                <p>
                  Angle: azimuth of arrival relative to the receiver&apos;s look direction (front at the top, left to the left), as in the table; change it with <b>Look</b> above.
                </p>
                <p>
                  {discRadius === 'elevation' ? (
                    <>Radius: elevation. The centre is straight up, the dashed circle the horizon, the rim straight down (floor reflections fall outside the circle).</>
                  ) : (
                    <>Radius: level relative to the direct sound, in dB (the direct sound is on the 0 dB circle).</>
                  )}
                </p>
                <p>Size: level re the direct sound (largest at 0 dB, smallest at −30 dB and below). Colour: order. ◆ direct · ● specular · ▲ diffracted. Same filters as the chart; click a marker to select it.</p>
              </div>
            </div>
          </section>
        )}
        {data && filtered.length > 0 && (
          <section className="card">
            <div className="table-scroll refl-table" ref={tableRef} tabIndex={0} onKeyDown={onTableKey} aria-label="Arrivals (arrow keys move the selection)">
              <table className="table compact">
                <thead>
                  <tr>
                    {th('kind', 'Kind', false)}
                    {th('order', 'Order')}
                    {th('time', 'Time (ms)', true, 'Arrival time after emission')}
                    {th('delay', 'Delay (ms)', true, 'Arrival time re. the direct sound')}
                    {th('distance', 'Dist. (m)', true, 'Path length')}
                    {th('level', 'Level (dB)', true, levelTitle)}
                    {th('rel', 'Re direct (dB)', true, 'Level relative to the direct sound')}
                    {th('az', 'Azimuth (°)', true, "Direction of arrival relative to the receiver's look direction, + = to its left")}
                    {th('el', 'Elev. (°)', true, 'Elevation of arrival above the horizontal')}
                    {th('surfaces', 'Surfaces', false, 'Layer of each reflection, source side first')}
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, MAX_ROWS).map((r) => (
                    <tr
                      key={r.i}
                      data-row={r.i}
                      className={`clickable ${selected.includes(r.i) ? 'selected' : ''}`}
                      onClick={(e) => select(r.i, e.ctrlKey || e.metaKey || e.shiftKey)}
                      onMouseEnter={() => setHovered(r.i)}
                      onMouseLeave={() => setHovered((h) => (h === r.i ? null : h))}
                    >
                      <td>
                        <span className="swatch" style={{ background: orderColor(r.e.order) }} />
                        {r.e.kind}
                      </td>
                      <td className="num">{r.e.order}</td>
                      <td className="num">{fmtNum(r.e.time_ms, 2)}</td>
                      <td className="num">{fmtNum(r.e.delay_ms, 2)}</td>
                      <td className="num">{fmtNum(r.e.distance_m, 2)}</td>
                      <td className="num">{fmtNum(r.level, 1)}</td>
                      <td className="num">{fmtNum(r.rel, 1)}</td>
                      <td className="num">{fmtNum(r.azRel, 0)}</td>
                      <td className="num">{fmtNum(r.e.elevation_deg, 0)}</td>
                      <td className="ellipsis surfaces" title={r.e.surfaces.join(' > ')}>
                        {r.e.surfaces.map((x) => prettySurface(x, catalog)).join(' › ') || <span className="muted">–</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filtered.length > MAX_ROWS && <div className="small muted">Showing the first {MAX_ROWS} of {filtered.length} rows (the CSV of the chart has them all). Narrow the filters to see the rest.</div>}
          </section>
        )}
      </div>
      <aside className={`results-side wide ${big ? 'refl-big' : ''}`}>
        <section className="card results-3d tall refl-3d">
          {mesh && bounds ? (
            <SceneCanvas bounds={bounds} showGizmo={false} frame={frameReq}>
              <LayerMeshes mesh={mesh} shading="ghosted" pickable={false} styleOf={layerStyle} />
              <PlacedObjects objects={placed} catalog={catalog} showVisual={false} pickable={false} />
              <ReflectionPaths
                paths={paths}
                focus={focusPaths}
                dim={focusPaths.length > 0}
                throughWalls={throughWalls}
                anim={anim}
                timeRef={clock}
                seekKey={seekKey}
                onTime={onTime}
                source={src?.position ?? null}
                receiver={rcv?.position ?? null}
                radius={markerRadius(bounds)}
                onHover={setHovered}
                onPick={(id, additive) => (id === null ? !additive && setSelected([]) : select(id, additive))}
                tip={pathTip}
              />
              <PointMarkers
                points={points}
                radius={markerRadius(bounds)}
                onPointClick={(p) => {
                  if (p.kind === 'source') setSourceId(p.id);
                  else setReceiverId(p.id);
                }}
              />
            </SceneCanvas>
          ) : (
            <Spinner />
          )}
          {shownOrders.length > 0 && (
            <div className="path-legend" aria-label="Path colours">
              {shownOrders.map((o) => (
                <span key={o}>
                  <i style={{ background: ORDER_COLORS[o], boxShadow: `0 0 6px ${ORDER_COLORS[o]}` }} />
                  {o === 4 ? '4th+' : ORDER_NAMES[o].replace(' order', '')}
                </span>
              ))}
              <span className="muted">fainter = weaker</span>
            </div>
          )}
          <button
            type="button"
            className="btn btn-sm path-expand"
            onClick={() => {
              setBig((b) => !b);
              // Re-frame the model once the canvas has its new size.
              if (bounds) window.setTimeout(() => setFrameReq({ center: bounds.center, radius: bounds.radius, key: Date.now() }), 150);
            }}
            title={big ? 'Back to the normal layout' : 'Enlarge the 3D view'}
            aria-pressed={big}
          >
            {big ? 'Shrink' : 'Enlarge'}
          </button>
          <div className="refl-3d-bar">
            <Toggle checked={showAll} onChange={setShowAll} label="Paths" title="Draw the filtered paths (otherwise only the selected ones)" />
            <label className="inline-field" title="Draw only the strongest paths of the filtered list (empty = all of them)">
              <span>strongest</span>
              <NumField value={maxPaths} allowNull min={1} max={5000} step={10} digits={0} placeholder="all" onChange={setMaxPaths} width={50} disabled={!showAll} ariaLabel="Number of paths drawn" />
              <span className="unit">of {filtered.length}</span>
            </label>
            <Toggle checked={throughWalls} onChange={setThroughWalls} label="Through walls" title="Paths behind opaque surfaces stay visible, dashed and faint" />
            <div className="grow" />
            <span className="muted small">hover a path: details · click: select</span>
          </div>
          <div className="refl-3d-bar">
            <Toggle
              checked={animOn}
              onChange={(v) => {
                setAnimOn(v);
                if (v) {
                  seek(0);
                  setPlaying(true);
                }
              }}
              label="Pulse"
              title="The sound leaves the source and travels every path, slowed down: the paths appear in their order of arrival"
            />
            {animOn && (
              <>
                <button type="button" className="btn btn-sm" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause' : 'Play'}>
                  {playing ? 'Pause' : 'Play'}
                </button>
                <PulseClock subs={timeSubs.current} initial={clock.current} endMs={animEnd} tDirect={tDirect} onSeek={seek} />
                <label className="inline-field" title="Slow motion: milliseconds of sound shown per second">
                  <input type="range" min={0} max={RATES.length - 1} step={1} value={Math.max(0, RATES.indexOf(rate))} onChange={(e) => setRate(RATES[Number(e.target.value)])} aria-label="Animation speed" style={{ width: 70 }} />
                  <span className="unit">{rate} ms/s</span>
                </label>
              </>
            )}
            {!animOn && <span className="muted small">{shown.length < filtered.length && showAll ? `${shown.length} strongest of ${filtered.length} filtered paths` : ''}</span>}
          </div>
        </section>
        <section className="card">
          <h3>Selected arrival</h3>
          {detail ? (
            <dl className="kv small">
              <dt>Kind</dt>
              <dd>
                <span className="swatch" style={{ background: orderColor(detail.e.order) }} />
                {detail.e.kind}, order {detail.e.order}
              </dd>
              <dt>Time</dt>
              <dd>
                {fmtNum(detail.e.time_ms, 2)} ms (+{fmtNum(detail.e.delay_ms, 2)} ms re direct)
              </dd>
              <dt>Path length</dt>
              <dd>{fmtNum(detail.e.distance_m, 2)} m</dd>
              <dt>Level</dt>
              <dd>
                {fmtNum(detail.level, 1)} dB{detail.rel !== null ? ` (${detail.rel >= 0 ? '+' : ''}${fmtNum(detail.rel, 1)} dB re direct)` : ''}
              </dd>
              <dt>Per band</dt>
              <dd className="mono">{detail.e.spl_db.map((v, i) => `${bandLabel(BANDS_HZ[i])} ${fmtNum(v, 1)}`).join(' · ')}</dd>
              <dt>Direction</dt>
              <dd>
                azimuth {fmtNum(detail.azRel, 0)}° re look direction ({fmtNum(detail.e.azimuth_deg, 0)}° world), elevation {fmtNum(detail.e.elevation_deg, 0)}°
              </dd>
              <dt>Surfaces</dt>
              <dd>{detail.e.surfaces.length ? detail.e.surfaces.join(' › ') : '–'}</dd>
              {selected.length > 1 && (
                <>
                  <dt>Selection</dt>
                  <dd>
                    {selected.length} arrivals{' '}
                    <button type="button" className="link small" onClick={() => setSelected([])}>
                      clear
                    </button>
                  </dd>
                </>
              )}
            </dl>
          ) : (
            <div className="muted small">Click a stem in the chart or a row of the table.</div>
          )}
        </section>
      </aside>
    </div>
  );
}
