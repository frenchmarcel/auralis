import * as THREE from 'three';
import type { MeshData, ReceiverGrid, Vec3 } from '../api';
import type { Bounds } from './geom';
import { PARAMS } from './metrics';

/** Server limit per grid (API "Receiver grids"). */
export const GRID_MAX_POINTS = 2500;
/** Measured cost of the geometrical solver per grid point and source (GPU, 50 000 rays). */
export const GRID_POINT_S = 0.15;
export const DEFAULT_GRID_RAYS = 50000;

export function nextGridId(grids: ReceiverGrid[]): string {
  const used = new Set(grids.map((g) => g.id));
  for (let i = 1; ; i++) if (!used.has(`G${i}`)) return `G${i}`;
}

/** A floor-like layer to start a surface grid on. */
export function defaultGridLayer(layers: string[], layerBounds: Map<string, Bounds>): string | null {
  const named = layers.find((n) => /audience|stalls|seat|chair|parterre|parquet/i.test(n)) ?? layers.find((n) => /floor|ground/i.test(n));
  if (named) return named;
  // Else the flattest, lowest large layer.
  let best: string | null = null;
  let score = Infinity;
  for (const n of layers) {
    const b = layerBounds.get(n);
    if (!b) continue;
    const s = b.size[2] / Math.max(0.1, Math.hypot(b.size[0], b.size[1])) + b.min[2] * 0.01;
    if (s < score) {
      score = s;
      best = n;
    }
  }
  return best ?? layers[0] ?? null;
}

/** Spacing (0.1 m steps) that keeps a w x l rectangle under `maxPoints`. */
export function spacingFor(w: number, l: number, maxPoints = 400): number {
  const s = Math.sqrt((w * l) / maxPoints);
  return Math.min(20, Math.max(0.5, Math.ceil(s * 10) / 10));
}

export function newSurfaceGrid(id: string, name: string, layer: string): ReceiverGrid {
  return { id, name, kind: 'surface', layer, offset_m: 1.2, spacing_m: 1.0 };
}

export function newPlaneGrid(id: string, name: string, bounds: Bounds | null): ReceiverGrid {
  const c: Vec3 = bounds ? [round2(bounds.center[0]), round2(bounds.center[1]), round2(bounds.min[2] + 1.2)] : [0, 0, 1.2];
  const w = bounds ? Math.max(1, Math.round(bounds.size[0] * 0.7 * 10) / 10) : 4;
  const l = bounds ? Math.max(1, Math.round(bounds.size[1] * 0.7 * 10) / 10) : 4;
  return { id, name, kind: 'plane', center: c, width_m: w, length_m: l, azimuth_deg: 0, tilt_deg: 0, spacing_m: spacingFor(w, l) };
}

const round2 = (v: number) => Math.round(v * 100) / 100;

/** The server's canonical face normal (shared by coplanar faces) and lattice axes. */
function canonical(n: THREE.Vector3): THREE.Vector3 {
  const e = 1e-6;
  if (n.z < -e || (Math.abs(n.z) <= e && (n.y < -e || (Math.abs(n.y) <= e && n.x < 0)))) n.negate();
  return n;
}

function latticeBasis(n: THREE.Vector3): { u: THREE.Vector3; v: THREE.Vector3 } {
  const ref = Math.abs(n.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const u = ref.clone().addScaledVector(n, -ref.dot(n)).normalize();
  return { u, v: new THREE.Vector3().crossVectors(n, u) };
}

/** In-plane axes of one grid point (one square per point is drawn along u and v). */
export interface PointFrame {
  u: Vec3;
  v: Vec3;
  n: Vec3;
}

const vec = (x: THREE.Vector3): Vec3 => [x.x, x.y, x.z];

export function planeFrame(g: ReceiverGrid): PointFrame {
  const az = ((g.azimuth_deg ?? 0) * Math.PI) / 180;
  const tl = ((g.tilt_deg ?? 0) * Math.PI) / 180;
  const u = new THREE.Vector3(Math.cos(az), Math.sin(az), 0);
  const v = new THREE.Vector3(-Math.sin(az) * Math.cos(tl), Math.cos(az) * Math.cos(tl), Math.sin(tl));
  return { u: vec(u), v: vec(v), n: vec(new THREE.Vector3().crossVectors(u, v)) };
}

const FLAT: PointFrame = { u: [1, 0, 0], v: [0, 1, 0], n: [0, 0, 1] };

/**
 * Frame of every point. Plane grids: the rectangle's axes. Surface grids: the axes of
 * the layer triangle the point was generated from (found again: at `offset_m` from its
 * plane, projecting inside it), as the server builds its lattice. Points whose triangle
 * is not found (the grid was edited after the run) take their nearest found neighbour's.
 */
export function pointFrames(grid: ReceiverGrid | null, points: Vec3[], mesh: MeshData | null): PointFrame[] {
  if (grid?.kind === 'plane') {
    const f = planeFrame(grid);
    return points.map(() => f);
  }
  const out: (PointFrame | null)[] = points.map(() => null);
  const layer = grid?.kind === 'surface' && mesh ? mesh.layers.find((l) => l.name === grid.layer) : undefined;
  if (layer && mesh && grid) {
    const P = mesh.positions;
    const off = grid.offset_m ?? 1.2;
    const tol = 2e-3 * Math.max(1, off);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const q = new THREE.Vector3();
    const tri = new THREE.Triangle();
    const bary = new THREE.Vector3();
    const tris: { n: THREE.Vector3; d: number; t: THREE.Triangle; f: PointFrame }[] = [];
    for (let k = 0; k + 2 < layer.indices.length; k += 3) {
      const [i0, i1, i2] = [layer.indices[k], layer.indices[k + 1], layer.indices[k + 2]];
      a.set(P[3 * i0], P[3 * i0 + 1], P[3 * i0 + 2]);
      b.set(P[3 * i1], P[3 * i1 + 1], P[3 * i1 + 2]);
      c.set(P[3 * i2], P[3 * i2 + 1], P[3 * i2 + 2]);
      const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a));
      if (n.lengthSq() < 1e-20) continue;
      canonical(n.normalize());
      const { u, v } = latticeBasis(n);
      tris.push({ n, d: n.dot(a), t: new THREE.Triangle(a.clone(), b.clone(), c.clone()), f: { u: vec(u), v: vec(v), n: vec(n) } });
    }
    points.forEach((p, i) => {
      q.set(p[0], p[1], p[2]);
      for (const t of tris) {
        const h = t.n.dot(q) - t.d;
        if (Math.abs(Math.abs(h) - off) > tol) continue;
        const on = q.clone().addScaledVector(t.n, -h);
        tri.copy(t.t);
        tri.getBarycoord(on, bary);
        if (bary.x >= -1e-6 && bary.y >= -1e-6 && bary.z >= -1e-6) {
          out[i] = t.f;
          break;
        }
      }
    });
  }
  const found = out.map((f, i) => (f ? i : -1)).filter((i) => i >= 0);
  return out.map((f, i) => {
    if (f) return f;
    let best = -1;
    let bd = Infinity;
    for (const j of found) {
      const d = dist2(points[i], points[j]);
      if (d < bd) {
        bd = d;
        best = j;
      }
    }
    return best >= 0 ? out[best]! : FLAT;
  });
}

function dist2(a: Vec3, b: Vec3): number {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

/** Uniform-cell hash of points for radius queries. */
export class PointIndex {
  private cells = new Map<string, number[]>();
  constructor(
    private points: Vec3[],
    private cell: number,
  ) {
    points.forEach((p, i) => {
      const k = this.key(p);
      const l = this.cells.get(k);
      if (l) l.push(i);
      else this.cells.set(k, [i]);
    });
  }
  private key(p: Vec3): string {
    return `${Math.floor(p[0] / this.cell)},${Math.floor(p[1] / this.cell)},${Math.floor(p[2] / this.cell)}`;
  }
  /** Indices of the points within `r` (≤ cell) of p. */
  within(p: Vec3, r: number): number[] {
    const out: number[] = [];
    const cx = Math.floor(p[0] / this.cell);
    const cy = Math.floor(p[1] / this.cell);
    const cz = Math.floor(p[2] / this.cell);
    const r2 = r * r;
    for (let x = cx - 1; x <= cx + 1; x++)
      for (let y = cy - 1; y <= cy + 1; y++)
        for (let z = cz - 1; z <= cz + 1; z++) {
          const l = this.cells.get(`${x},${y},${z}`);
          if (!l) continue;
          for (const i of l) if (dist2(this.points[i], p) <= r2) out.push(i);
        }
    return out;
  }
}

export type Num = number | null;
const finite = (v: Num | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

function medianOf(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Display-only 3 x 3 median: each point takes the median of itself and its lattice
 * neighbours (within 1.5 spacings, so diagonals count) that have a value.
 */
export function median3x3(values: Num[], points: Vec3[], spacing: number): Num[] {
  const idx = new PointIndex(points, 1.5 * spacing);
  return values.map((v, i) => {
    if (!finite(v)) return null;
    const nb = idx.within(points[i], 1.5 * spacing * 1.001).map((j) => values[j]).filter(finite);
    return nb.length ? medianOf(nb) : v;
  });
}

/** Per point: the mean over sources (energetic for levels in dB); null where none has a value. */
export function meanOverSources(arrays: Num[][], energetic: boolean): Num[] {
  if (!arrays.length) return [];
  if (arrays.length === 1) return arrays[0];
  return arrays[0].map((_, i) => {
    const v = arrays.map((a) => a[i]).filter(finite);
    if (!v.length) return null;
    if (energetic) return 10 * Math.log10(v.reduce((s, x) => s + 10 ** (x / 10), 0) / v.length);
    return v.reduce((s, x) => s + x, 0) / v.length;
  });
}

export interface MapStats {
  n: number;
  area: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
}

/** Area-weighted statistics of the points that have a value. */
export function areaStats(values: Num[], areas: number[]): MapStats | null {
  const items: { v: number; a: number }[] = [];
  values.forEach((v, i) => {
    if (finite(v)) items.push({ v, a: areas[i] ?? 1 });
  });
  if (!items.length) return null;
  const A = items.reduce((s, x) => s + x.a, 0);
  const mean = items.reduce((s, x) => s + x.a * x.v, 0) / A;
  const varc = items.reduce((s, x) => s + x.a * (x.v - mean) ** 2, 0) / A;
  items.sort((p, q) => p.v - q.v);
  let acc = 0;
  let median = items[items.length - 1].v;
  for (let k = 0; k < items.length; k++) {
    acc += items[k].a;
    if (acc >= A / 2 - 1e-12) {
      // Exactly half the area below: midway to the next value (unweighted-median convention).
      median = Math.abs(acc - A / 2) < 1e-9 * A && k + 1 < items.length ? (items[k].v + items[k + 1].v) / 2 : items[k].v;
      break;
    }
  }
  return { n: items.length, area: A, min: items[0].v, max: items[items.length - 1].v, mean, median, std: Math.sqrt(varc) };
}

export type TargetOp = 'ge' | 'le';

/** Share (0-1) of the area whose value meets `op target`; null when no point has a value. */
export function areaShare(values: Num[], areas: number[], op: TargetOp, target: number): number | null {
  let all = 0;
  let ok = 0;
  values.forEach((v, i) => {
    if (!finite(v)) return;
    const a = areas[i] ?? 1;
    all += a;
    if (op === 'ge' ? v >= target : v <= target) ok += a;
  });
  return all > 0 ? ok / all : null;
}

export interface Histogram {
  edges: number[];
  /** Share of the area (0-1) per bin. */
  share: number[];
  area: number[];
}

export function areaHistogram(values: Num[], areas: number[], lo: number, hi: number, nBins: number): Histogram {
  const span = hi - lo || 1;
  const edges = Array.from({ length: nBins + 1 }, (_, k) => lo + (span * k) / nBins);
  const area = new Array(nBins).fill(0);
  let all = 0;
  values.forEach((v, i) => {
    if (!finite(v)) return;
    const a = areas[i] ?? 1;
    all += a;
    const k = Math.min(nBins - 1, Math.max(0, Math.floor(((v - lo) / span) * nBins)));
    area[k] += a;
  });
  return { edges, area, share: area.map((x) => (all > 0 ? x / all : 0)) };
}

/** Sequential ramp (one hue, dark to light: low values recede into the dark view). */
export const SEQ_RAMP = ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'];
/** Diverging ramp for differences: blue (negative) - grey (0) - red (positive). */
export const DIV_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#45464a', '#a83a3b', '#e34948', '#f2a09d'];
export const DISABLED_COLOR = '#4f565d';
export const TARGET_OK = '#0ca30c';
export const TARGET_FAIL = '#e5484d';

export function rampColor(ramp: string[], t: number): string {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const c = new THREE.Color(ramp[i]).lerp(new THREE.Color(ramp[i + 1]), x - i);
  return `#${c.getHexString()}`;
}

export function rampCss(ramp: string[]): string {
  return `linear-gradient(to right, ${ramp.join(', ')})`;
}

/** 256 x 1 texture of a ramp: values interpolated across a face are looked up in it. */
export function rampTexture(ramp: string[]): THREE.DataTexture {
  const n = 256;
  const data = new Uint8Array(n * 4);
  for (let k = 0; k < n; k++) {
    const c = new THREE.Color(rampColor(ramp, k / (n - 1)));
    data[4 * k] = Math.round(c.r * 255);
    data[4 * k + 1] = Math.round(c.g * 255);
    data[4 * k + 2] = Math.round(c.b * 255);
    data[4 * k + 3] = 255;
  }
  const t = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}

/** "Nice" rounded bounds for an automatic colour scale. */
export function niceRange(lo: number, hi: number, symmetric = false): [number, number] {
  if (symmetric) {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const s = niceStep(m / 4);
    const r = Math.ceil(m / s) * s;
    return [-r, r];
  }
  if (!(hi > lo)) {
    const d = Math.abs(lo) * 0.05 || 0.5;
    return [lo - d, hi + d];
  }
  const s = niceStep((hi - lo) / 6);
  return [Math.floor(lo / s) * s, Math.ceil(hi / s) * s];
}

function niceStep(x: number): number {
  const p = 10 ** Math.floor(Math.log10(x));
  const m = x / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

/** Parameters computed on receiver grids (API: geometrical solver, energy-time curve). */
export interface GridParam {
  key: string;
  label: string;
  /** Plotly / HTML label (may carry <sub>). */
  axis: string;
  unit: string;
  title: string;
  digits: number;
  /** Levels in dB: averaged on energy over sources. */
  energetic: boolean;
  /** One value per point (no band). */
  scalar: boolean;
}

const fromMetrics = (key: string, energetic?: boolean): GridParam => {
  const p = PARAMS.find((x) => x.key === key)!;
  return { key, label: p.label, axis: p.axis, unit: p.unit, title: p.title, digits: p.digits, energetic: energetic ?? !!p.energetic, scalar: !!p.scalar };
};

export const GRID_PARAMS: GridParam[] = [
  fromMetrics('T20'),
  fromMetrics('T30'),
  fromMetrics('EDT'),
  fromMetrics('C50'),
  fromMetrics('C80'),
  fromMetrics('D50'),
  fromMetrics('Ts_ms'),
  fromMetrics('G_db'),
  fromMetrics('SPL_db'),
  { key: 'SPL_dba', label: 'SPL (A)', axis: 'SPL<sub>A</sub>', unit: 'dB(A)', title: 'Overall A-weighted sound pressure level', digits: 1, energetic: true, scalar: true },
  fromMetrics('STI'),
  fromMetrics('MTI'),
  fromMetrics('JLF'),
  fromMetrics('LJ_db'),
];

export function gridParam(key: string): GridParam {
  return GRID_PARAMS.find((p) => p.key === key) ?? GRID_PARAMS[1];
}

/** Octave band index, or the mean of several bands (T20/T30 style averages). */
export type BandChoice = number | 'm500_1k' | 'm500_2k';

export const BAND_MEANS: { value: BandChoice; label: string; bands: number[] }[] = [
  { value: 'm500_1k', label: 'mean 500 Hz–1 kHz', bands: [2, 3] },
  { value: 'm500_2k', label: 'mean 500 Hz–2 kHz', bands: [2, 3, 4] },
];

/** Value of one point for a band choice: a band, or the mean of bands (energetic for levels). */
export function bandValue(row: Num[] | Num | undefined, band: BandChoice, energetic: boolean): Num {
  if (row === null || row === undefined) return null;
  if (!Array.isArray(row)) return finite(row) ? row : null;
  if (typeof band === 'number') return finite(row[band]) ? row[band] : null;
  const idx = BAND_MEANS.find((b) => b.value === band)?.bands ?? [];
  const v = idx.map((i) => row[i]);
  if (!v.every(finite)) return null;
  const xs = v as number[];
  if (energetic) return 10 * Math.log10(xs.reduce((s, x) => s + 10 ** (x / 10), 0) / xs.length);
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}
