import type { ThreeEvent } from '@react-three/fiber';
import type { MeshData, Vec3 } from '../../api';
import { roundVec } from '../../lib/geom';
import { MIN_SURFACE_OFFSET } from './PointsTab';

/** Fixed, non axis-aligned directions for the parity votes. */
const RAYS: Vec3[] = [
  [0.267, 0.534, 0.802],
  [-0.619, 0.311, 0.721],
  [0.408, -0.816, -0.408],
];

/** Number of triangles crossed by the ray p + t d, t > 0 (Moller-Trumbore). */
function crossings(mesh: MeshData, p: Vec3, d: Vec3): number {
  const P = mesh.positions;
  let n = 0;
  for (const l of mesh.layers) {
    const idx = l.indices;
    for (let k = 0; k + 2 < idx.length; k += 3) {
      const a = 3 * idx[k];
      const b = 3 * idx[k + 1];
      const c = 3 * idx[k + 2];
      const e1x = P[b] - P[a];
      const e1y = P[b + 1] - P[a + 1];
      const e1z = P[b + 2] - P[a + 2];
      const e2x = P[c] - P[a];
      const e2y = P[c + 1] - P[a + 1];
      const e2z = P[c + 2] - P[a + 2];
      const hx = d[1] * e2z - d[2] * e2y;
      const hy = d[2] * e2x - d[0] * e2z;
      const hz = d[0] * e2y - d[1] * e2x;
      const det = e1x * hx + e1y * hy + e1z * hz;
      if (Math.abs(det) < 1e-12) continue;
      const f = 1 / det;
      const sx = p[0] - P[a];
      const sy = p[1] - P[a + 1];
      const sz = p[2] - P[a + 2];
      const u = f * (sx * hx + sy * hy + sz * hz);
      if (u < 0 || u > 1) continue;
      const qx = sy * e1z - sz * e1y;
      const qy = sz * e1x - sx * e1z;
      const qz = sx * e1y - sy * e1x;
      const v = f * (d[0] * qx + d[1] * qy + d[2] * qz);
      if (v < 0 || u + v > 1) continue;
      if (f * (e2x * qx + e2y * qy + e2z * qz) > 1e-9) n++;
    }
  }
  return n;
}

/** Inside a closed mesh (and outside the closed objects in it): odd crossings, majority of 3 rays. */
export function insideClosedMesh(mesh: MeshData, p: Vec3): boolean {
  let votes = 0;
  for (const d of RAYS) if (crossings(mesh, p, d) % 2 === 1) votes++;
  return votes >= 2;
}

/**
 * "Offset from a surface": the clicked point moved `distance` metres along the face normal
 * (min MIN_SURFACE_OFFSET). In a closed model the side inside the room wins (a wall clicked
 * from outside still gets its point indoors); otherwise the side the camera looks from.
 */
export function pointFromSurfaceClick(e: ThreeEvent<MouseEvent>, distance: number, closedMesh: MeshData | null): Vec3 | null {
  if (!e.face) return null;
  const n = e.face.normal.clone().transformDirection(e.object.matrixWorld).normalize();
  if (n.dot(e.ray.direction) > 0) n.negate();
  const d = Math.max(MIN_SURFACE_OFFSET, distance);
  const front = e.point.clone().addScaledVector(n, d);
  const a: Vec3 = [front.x, front.y, front.z];
  if (closedMesh && !insideClosedMesh(closedMesh, a)) {
    const back = e.point.clone().addScaledVector(n, -d);
    const b: Vec3 = [back.x, back.y, back.z];
    if (insideClosedMesh(closedMesh, b)) return roundVec(b, 3);
  }
  return roundVec(a, 3);
}
