import type { Vec3 } from '../api';

export interface Bounds {
  min: Vec3;
  max: Vec3;
  center: Vec3;
  size: Vec3;
  radius: number;
}

export function boundsOf(min: Vec3, max: Vec3): Bounds {
  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const size: Vec3 = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const radius = Math.max(0.5, Math.hypot(size[0], size[1], size[2]) / 2);
  return { min, max, center, size, radius };
}

export function boundsOfPositions(positions: ArrayLike<number>): Bounds | null {
  if (positions.length < 3) return null;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return boundsOf(min, max);
}

const DEG = Math.PI / 180;

/** Unit vector for azimuth (CCW from +X in XY) and elevation (from XY plane toward +Z). */
export function dirFromAzEl(azDeg: number, elDeg: number): Vec3 {
  const az = azDeg * DEG;
  const el = elDeg * DEG;
  return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
}

export function azElTowards(from: Vec3, to: Vec3): { azimuth_deg: number; elevation_deg: number } {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const dz = to[2] - from[2];
  const h = Math.hypot(dx, dy);
  const az = h < 1e-9 ? 0 : Math.atan2(dy, dx) / DEG;
  const el = Math.atan2(dz, h) / DEG;
  return { azimuth_deg: round(az, 1), elevation_deg: round(el, 1) };
}

export function distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function round(v: number, digits = 3): number {
  const k = 10 ** digits;
  return Math.round(v * k) / k;
}

export function roundVec(v: Vec3, digits = 3): Vec3 {
  return [round(v[0], digits), round(v[1], digits), round(v[2], digits)];
}
