import type { Path, PathValidity, Vec3, Waypoint } from '../api';
import { distance } from './geom';

/** Same constant as webapp/server/trajectory.py (spacing of the checks along a path). */
const CHECK_STEP_M = 0.1;

export const isSmooth = (path: Path | null | undefined): boolean => path?.interpolation === 'smooth';

const tangentCache = new WeakMap<Waypoint[], Vec3[]>();

/**
 * Catmull-Rom tangents in time (port of trajectory.py _tangents): one-sided at the
 * ends, zero at either end of a hold (two equal waypoints: the point really stops).
 */
function tangents(w: Waypoint[]): Vec3[] {
  const cached = tangentCache.get(w);
  if (cached) return cached;
  const n = w.length;
  const diff = (a: number, b: number): Vec3 => {
    const dt = w[b].t - w[a].t;
    const p = w[a].position;
    const q = w[b].position;
    return [(q[0] - p[0]) / dt, (q[1] - p[1]) / dt, (q[2] - p[2]) / dt];
  };
  const m: Vec3[] = w.map(() => [0, 0, 0]);
  for (let k = 1; k < n - 1; k++) m[k] = diff(k - 1, k + 1);
  if (n >= 2) {
    m[0] = diff(0, 1);
    m[n - 1] = diff(n - 2, n - 1);
  }
  for (let k = 0; k < n - 1; k++) {
    const p = w[k].position;
    const q = w[k + 1].position;
    if (Math.abs(q[0] - p[0]) < 1e-9 && Math.abs(q[1] - p[1]) < 1e-9 && Math.abs(q[2] - p[2]) < 1e-9) {
      m[k] = [0, 0, 0];
      m[k + 1] = [0, 0, 0];
    }
  }
  tangentCache.set(w, m);
  return m;
}

/** Cubic Hermite through the waypoints at their times (C1), held outside the path's time span. */
export function smoothPositionAt(w: Waypoint[], t: number): Vec3 {
  const n = w.length;
  if (n === 0) return [0, 0, 0];
  if (n === 1 || !(t > w[0].t)) return w[0].position;
  if (t >= w[n - 1].t) return w[n - 1].position;
  let k = 0;
  let hi = n - 1;
  while (hi - k > 1) {
    const mid = (k + hi) >> 1;
    if (w[mid].t <= t) k = mid;
    else hi = mid;
  }
  const m = tangents(w);
  const h = w[k + 1].t - w[k].t;
  const s = (t - w[k].t) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = (s3 - 2 * s2 + s) * h;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = (s3 - s2) * h;
  const a = w[k].position;
  const b = w[k + 1].position;
  const ma = m[k];
  const mb = m[k + 1];
  return [0, 1, 2].map((i) => h00 * a[i] + h10 * ma[i] + h01 * b[i] + h11 * mb[i]) as Vec3;
}

export interface SmoothSample {
  points: Vec3[];
  times: number[];
  length: number;
  maxSpeed: number;
}

const sampleCache = new WeakMap<Waypoint[], SmoothSample>();

/** The curve sampled at most 0.10 m apart, waypoints included (port of trajectory.py _curve_times). */
export function sampleSmooth(w: Waypoint[]): SmoothSample {
  const cached = sampleCache.get(w);
  if (cached) return cached;
  const m = tangents(w);
  const norm = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
  const times = w.length ? [w[0].t] : [];
  for (let k = 0; k < w.length - 1; k++) {
    const h = w[k + 1].t - w[k].t;
    const est = distance(w[k].position, w[k + 1].position) + 0.5 * h * (norm(m[k]) + norm(m[k + 1]));
    let n = Math.max(1, Math.ceil(est / CHECK_STEP_M));
    let tk: number[] = [];
    for (let iter = 0; iter < 6; iter++) {
      tk = Array.from({ length: n + 1 }, (_, j) => w[k].t + (h * j) / n);
      let worst = 0;
      let prev = smoothPositionAt(w, tk[0]);
      for (let j = 1; j <= n; j++) {
        const p = smoothPositionAt(w, tk[j]);
        worst = Math.max(worst, distance(prev, p));
        prev = p;
      }
      if (worst <= CHECK_STEP_M) break;
      n *= 2;
    }
    times.push(...tk.slice(1));
  }
  const points = times.map((t) => smoothPositionAt(w, t));
  let length = 0;
  let maxSpeed = 0;
  for (let j = 1; j < points.length; j++) {
    const d = distance(points[j - 1], points[j]);
    length += d;
    maxSpeed = Math.max(maxSpeed, d / Math.max(times[j] - times[j - 1], 1e-12));
  }
  const out = { points, times, length, maxSpeed };
  sampleCache.set(w, out);
  return out;
}

/** Length and top speed as the server measures them: along the curve when smooth, else along the polyline. */
export function localPathInfo(path: Path): { length: number; maxSpeed: number } {
  if (isSmooth(path)) {
    const s = sampleSmooth(path.waypoints);
    return { length: s.length, maxSpeed: s.maxSpeed };
  }
  const w = path.waypoints;
  let length = 0;
  let maxSpeed = 0;
  for (let k = 1; k < w.length; k++) {
    const d = distance(w[k - 1].position, w[k].position);
    length += d;
    maxSpeed = Math.max(maxSpeed, d / Math.max(w[k].t - w[k - 1].t, 1e-12));
  }
  return { length, maxSpeed };
}

/**
 * True when a validation entry was computed for this very path: same time span, same
 * interpolation (a curve only for smooth paths, through both ends), and the same
 * length to 2 mm. Until the validation catches up with an edit, it is not.
 */
export function validityMatches(path: Path, v: PathValidity | undefined): v is PathValidity {
  const w = path.waypoints;
  if (!v || !w.length) return false;
  if (Math.abs(v.start_s - w[0].t) > 1e-6 || Math.abs(v.end_s - w[w.length - 1].t) > 1e-6) return false;
  if (isSmooth(path)) {
    const c = v.curve;
    if (!c || c.length < 2) return false;
    if (distance(c[0], w[0].position) > 2e-3 || distance(c[c.length - 1], w[w.length - 1].position) > 2e-3) return false;
  } else if (v.curve) return false;
  return Math.abs(localPathInfo(path).length - v.length_m) <= 2e-3;
}

/** The curve to draw for a smooth path: the validated one when it matches the path, else the same curve computed here. */
export function smoothCurve(path: Path, v: PathValidity | undefined): Vec3[] {
  return validityMatches(path, v) && v.curve ? v.curve : sampleSmooth(path.waypoints).points;
}
