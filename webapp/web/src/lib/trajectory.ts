import type { Path, Vec3, Waypoint } from '../api';
import { dirFromAzEl, distance, round, roundVec } from './geom';
import { smoothPositionAt } from './smoothPath';

/** Same constants as webapp/server/trajectory.py. */
export const MAX_WAYPOINTS = 200;
export const MAX_PATH_S = 600;
export const FOLLOW_TAU_S = 0.25;
const SAMPLE_DT = 0.05;

/** Default walking speed for new segments (m/s). */
export const WALK_SPEED_MPS = 1.4;
/** Smallest gap kept between two waypoint times (s). */
export const MIN_DT_S = 0.01;

/** Anything that may move: a source or a receiver. */
export interface Movable {
  id: string;
  position: Vec3;
  azimuth_deg: number;
  elevation_deg: number;
  path?: Path | null;
}

export function endTime(item: Movable): number {
  const w = item.path?.waypoints;
  return w && w.length ? w[w.length - 1].t : 0;
}

/** Position at time t: linear (np.interp) or smooth between waypoints, held before the first and after the last. */
export function positionAt(item: Movable, t: number): Vec3 {
  const w = item.path?.waypoints;
  if (!w || !w.length) return item.position;
  if (item.path?.interpolation === 'smooth') return smoothPositionAt(w, t);
  const n = w.length;
  if (!(t > w[0].t)) return w[0].position;
  if (t >= w[n - 1].t) return w[n - 1].position;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (w[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = w[lo].position;
  const b = w[hi].position;
  const f = (t - w[lo].t) / (w[hi].t - w[lo].t);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

type Heading = [number, number] | null;

function headingAt(item: Movable, t: number): Heading {
  const a = positionAt(item, t - FOLLOW_TAU_S);
  const b = positionAt(item, t + FOLLOW_TAU_S);
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const n = Math.hypot(dx, dy);
  return n > 1e-6 ? [dx / n, dy / n] : null;
}

interface HeldTable {
  times: number[];
  held: Heading[];
}

const heldCache = new WeakMap<Path, HeldTable>();

/**
 * Heading held while stationary, as the server samples it: on a 0.05 s grid plus
 * the waypoint times, the last moving heading, or the first one before any motion.
 */
function heldTable(item: Movable, path: Path): HeldTable {
  const cached = heldCache.get(path);
  if (cached) return cached;
  const tEnd = path.waypoints[path.waypoints.length - 1].t + FOLLOW_TAU_S;
  const set = new Set<number>();
  for (let k = 0; k * SAMPLE_DT <= tEnd + SAMPLE_DT; k++) set.add(k * SAMPLE_DT);
  for (const w of path.waypoints) set.add(w.t);
  const times = [...set].sort((a, b) => a - b);
  const held: Heading[] = [];
  let last: Heading = null;
  let first: Heading = null;
  for (const t of times) {
    const h = headingAt(item, t);
    if (h) {
      last = h;
      first = first ?? h;
    }
    held.push(last);
  }
  for (let i = 0; i < held.length && !held[i]; i++) held[i] = first;
  const table = { times, held };
  heldCache.set(path, table);
  return table;
}

/**
 * Unit front axis at time t (port of trajectory.py forward_at): fixed = azimuth /
 * elevation; follow = horizontal heading from p(t + 0.25) - p(t - 0.25), held when
 * stationary, elevation kept; target = towards the target's position at t.
 */
export function forwardAt(item: Movable, t: number, byId: Map<string, Movable>): Vec3 {
  const fixed = dirFromAzEl(item.azimuth_deg ?? 0, item.elevation_deg ?? 0);
  const path = item.path;
  if (!path || path.orientation === 'fixed') return fixed;
  if (path.orientation === 'follow') {
    let h = headingAt(item, t);
    if (!h) {
      const tab = heldTable(item, path);
      let lo = 0;
      let hi = tab.times.length - 1;
      if (t >= tab.times[hi]) lo = hi;
      else
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (tab.times[mid] <= t) lo = mid;
          else hi = mid;
        }
      h = tab.held[lo];
    }
    if (!h) return fixed;
    const el = ((item.elevation_deg ?? 0) * Math.PI) / 180;
    return [Math.cos(el) * h[0], Math.cos(el) * h[1], Math.sin(el)];
  }
  const tgt = path.target_id ? byId.get(path.target_id) : undefined;
  if (!tgt) return fixed;
  const a = positionAt(item, t);
  const b = positionAt(tgt, t);
  const d: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const n = Math.hypot(d[0], d[1], d[2]);
  return n > 1e-6 ? [d[0] / n, d[1] / n, d[2] / n] : fixed;
}

export interface PathStats {
  length: number;
  start: number;
  end: number;
  maxSpeed: number;
  /** Speed of the segment arriving at each waypoint (NaN for the first). */
  speeds: number[];
}

export function pathStats(path: Path): PathStats {
  const w = path.waypoints;
  const speeds = [NaN];
  let length = 0;
  let maxSpeed = 0;
  for (let k = 1; k < w.length; k++) {
    const d = distance(w[k - 1].position, w[k].position);
    const v = d / Math.max(1e-9, w[k].t - w[k - 1].t);
    length += d;
    maxSpeed = Math.max(maxSpeed, v);
    speeds.push(v);
  }
  return { length, start: w[0]?.t ?? 0, end: w[w.length - 1]?.t ?? 0, maxSpeed, speeds };
}

/** The item with a new path; like the server, a moving point's position is its first waypoint. */
export function withPath<T extends Movable>(item: T, path: Path | null): T {
  return path ? { ...item, path, position: [...path.waypoints[0].position] as Vec3 } : { ...item, path: null };
}

const roundT = (t: number) => round(t, 3);

/** "Make it move": the current position at t = 0, then 2 m ahead along the azimuth at walking speed. */
export function makePath(item: Movable): Path {
  const az = ((item.azimuth_deg ?? 0) * Math.PI) / 180;
  const p = item.position;
  const q = roundVec([p[0] + 2 * Math.cos(az), p[1] + 2 * Math.sin(az), p[2]]);
  return { waypoints: [{ position: [...p], t: 0 }, { position: q, t: roundT(2 / WALK_SPEED_MPS) }], orientation: 'fixed', target_id: null };
}

function mapWaypoints(path: Path, fn: (w: Waypoint, k: number) => Waypoint): Path {
  return { ...path, waypoints: path.waypoints.map(fn) };
}

export function setWaypointPosition(path: Path, k: number, pos: Vec3): Path {
  return mapWaypoints(path, (w, i) => (i === k ? { ...w, position: roundVec(pos) } : w));
}

/** Bounds of a waypoint's time that keep the times strictly increasing. */
export function timeBounds(path: Path, k: number): { min: number; max: number } {
  const w = path.waypoints;
  return {
    min: k > 0 ? roundT(w[k - 1].t + MIN_DT_S) : 0,
    max: k < w.length - 1 ? roundT(w[k + 1].t - MIN_DT_S) : MAX_PATH_S,
  };
}

export function setWaypointTime(path: Path, k: number, t: number): Path {
  const b = timeBounds(path, k);
  const v = roundT(Math.min(b.max, Math.max(b.min, t)));
  return mapWaypoints(path, (w, i) => (i === k ? { ...w, t: v } : w));
}

/**
 * Speed of the segment arriving at waypoint k: its end time is recomputed and every
 * later time shifts by the same amount, so later segments keep their durations.
 */
export function setSegmentSpeed(path: Path, k: number, speed: number): Path {
  const w = path.waypoints;
  if (k < 1 || k >= w.length) return path;
  const d = distance(w[k - 1].position, w[k].position);
  if (d < 1e-9 || !(speed > 0)) return path;
  const room = MAX_PATH_S - w[w.length - 1].t + (w[k].t - w[k - 1].t);
  const dur = Math.max(MIN_DT_S, Math.min(room, d / speed));
  const delta = roundT(w[k - 1].t + dur) - w[k].t;
  return mapWaypoints(path, (x, i) => (i >= k ? { ...x, t: roundT(x.t + delta) } : x));
}

/** A new waypoint after k: halfway to the next one, or 2 m further on (walking speed) after the last. */
export function insertWaypointAfter(path: Path, k: number, azimuthDeg = 0): Path {
  const w = path.waypoints;
  if (w.length >= MAX_WAYPOINTS) return path;
  let nw: Waypoint;
  if (k < w.length - 1) {
    const a = w[k];
    const b = w[k + 1];
    nw = { position: roundVec([(a.position[0] + b.position[0]) / 2, (a.position[1] + b.position[1]) / 2, (a.position[2] + b.position[2]) / 2]), t: roundT((a.t + b.t) / 2) };
    if (!(nw.t > a.t && nw.t < b.t)) return path;
  } else {
    const a = w[k];
    const prev = w[k - 1];
    let dx = prev ? a.position[0] - prev.position[0] : 0;
    let dy = prev ? a.position[1] - prev.position[1] : 0;
    let n = Math.hypot(dx, dy);
    if (n < 1e-6) {
      const az = (azimuthDeg * Math.PI) / 180;
      dx = Math.cos(az);
      dy = Math.sin(az);
      n = 1;
    }
    const t = roundT(a.t + 2 / WALK_SPEED_MPS);
    if (t > MAX_PATH_S) return path;
    nw = { position: roundVec([a.position[0] + (2 * dx) / n, a.position[1] + (2 * dy) / n, a.position[2]]), t };
  }
  return { ...path, waypoints: [...w.slice(0, k + 1), nw, ...w.slice(k + 1)] };
}

export function deleteWaypoint(path: Path, k: number): Path {
  if (path.waypoints.length <= 2) return path;
  return { ...path, waypoints: path.waypoints.filter((_, i) => i !== k) };
}

/** Append a waypoint at `pos`, reached at walking speed from the last one. Null when full. */
export function appendWaypoint(path: Path, pos: Vec3): Path | null {
  const w = path.waypoints;
  const last = w[w.length - 1];
  const t = roundT(last.t + Math.max(0.05, distance(last.position, pos) / WALK_SPEED_MPS));
  if (w.length >= MAX_WAYPOINTS || t > MAX_PATH_S) return null;
  return { ...path, waypoints: [...w, { position: roundVec(pos), t }] };
}

/** Translate the whole path (keeps its shape and times). */
export function translatePath(path: Path, delta: Vec3): Path {
  return mapWaypoints(path, (w) => ({ ...w, position: roundVec([w.position[0] + delta[0], w.position[1] + delta[1], w.position[2] + delta[2]]) }));
}
