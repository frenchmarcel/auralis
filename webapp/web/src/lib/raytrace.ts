import type { MeshData, Vec3 } from '../api';

/** Closest hit of a ray on the traced mesh. */
export interface RayHit {
  t: number;
  /** Layer index into TraceMesh.layerNames. */
  layer: number;
  point: Vec3;
  /** Unit geometric normal of the hit triangle (not oriented). */
  normal: Vec3;
}

/** Triangles of the solver mesh in a bounding-volume hierarchy, for specular ray tracing. */
export interface TraceMesh {
  layerNames: string[];
  nTriangles: number;
  bounds: { min: Vec3; max: Vec3 };
  intersect: (o: Vec3, d: Vec3, tMax?: number) => RayHit | null;
}

const LEAF = 4;
const EPS_T = 1e-6;

/**
 * Build a BVH (centroid median split on the longest axis, 4 triangles per leaf) over
 * every triangle of the mesh. About 0.1-0.3 s for 200k triangles.
 */
export function buildTraceMesh(mesh: MeshData): TraceMesh {
  const P = mesh.positions;
  let n = 0;
  for (const l of mesh.layers) n += Math.floor(l.indices.length / 3);
  const tri0 = new Float32Array(n * 9);
  const lay0 = new Int32Array(n);
  let k = 0;
  mesh.layers.forEach((l, li) => {
    const I = l.indices;
    for (let i = 0; i + 2 < I.length; i += 3) {
      const a = 3 * I[i];
      const b = 3 * I[i + 1];
      const c = 3 * I[i + 2];
      const o = 9 * k;
      tri0[o] = P[a];
      tri0[o + 1] = P[a + 1];
      tri0[o + 2] = P[a + 2];
      tri0[o + 3] = P[b];
      tri0[o + 4] = P[b + 1];
      tri0[o + 5] = P[b + 2];
      tri0[o + 6] = P[c];
      tri0[o + 7] = P[c + 1];
      tri0[o + 8] = P[c + 2];
      lay0[k] = li;
      k++;
    }
  });
  n = k;

  const cen = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const o = 9 * i;
    for (let a = 0; a < 3; a++) cen[3 * i + a] = (tri0[o + a] + tri0[o + 3 + a] + tri0[o + 6 + a]) / 3;
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;

  const maxNodes = 2 * n + 1;
  const bmin = new Float32Array(maxNodes * 3);
  const bmax = new Float32Array(maxNodes * 3);
  const left = new Int32Array(maxNodes); // first child (second = left + 1); -1 for a leaf
  const start = new Int32Array(maxNodes);
  const count = new Int32Array(maxNodes);
  let nNodes = 1;
  let maxDepth = 0;

  const stack: number[] = [0, 0];
  start[0] = 0;
  count[0] = n;
  while (stack.length) {
    const depth = stack.pop()!;
    const node = stack.pop()!;
    if (depth > maxDepth) maxDepth = depth;
    const s = start[node];
    const c = count[node];
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    let cx0 = Infinity, cy0 = Infinity, cz0 = Infinity, cx1 = -Infinity, cy1 = -Infinity, cz1 = -Infinity;
    for (let i = s; i < s + c; i++) {
      const t = order[i];
      const o = 9 * t;
      for (let v = 0; v < 9; v += 3) {
        const px = tri0[o + v], py = tri0[o + v + 1], pz = tri0[o + v + 2];
        if (px < x0) x0 = px;
        if (px > x1) x1 = px;
        if (py < y0) y0 = py;
        if (py > y1) y1 = py;
        if (pz < z0) z0 = pz;
        if (pz > z1) z1 = pz;
      }
      const qx = cen[3 * t], qy = cen[3 * t + 1], qz = cen[3 * t + 2];
      if (qx < cx0) cx0 = qx;
      if (qx > cx1) cx1 = qx;
      if (qy < cy0) cy0 = qy;
      if (qy > cy1) cy1 = qy;
      if (qz < cz0) cz0 = qz;
      if (qz > cz1) cz1 = qz;
    }
    bmin[3 * node] = x0;
    bmin[3 * node + 1] = y0;
    bmin[3 * node + 2] = z0;
    bmax[3 * node] = x1;
    bmax[3 * node + 1] = y1;
    bmax[3 * node + 2] = z1;
    left[node] = -1;
    if (c <= LEAF || nNodes + 2 > maxNodes) continue;
    const ex = cx1 - cx0, ey = cy1 - cy0, ez = cz1 - cz0;
    const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
    const extent = axis === 0 ? ex : axis === 1 ? ey : ez;
    let mid = s;
    if (extent > 1e-9) {
      const split = (axis === 0 ? cx0 : axis === 1 ? cy0 : cz0) + extent / 2;
      let lo = s;
      let hi = s + c - 1;
      while (lo <= hi) {
        if (cen[3 * order[lo] + axis] < split) lo++;
        else {
          const tmp = order[lo];
          order[lo] = order[hi];
          order[hi] = tmp;
          hi--;
        }
      }
      mid = lo;
    }
    if (mid === s || mid === s + c) {
      // All centroids on one side (or coincident): split the range in two halves.
      const seg = Array.from(order.subarray(s, s + c)).sort((a, b) => cen[3 * a + axis] - cen[3 * b + axis]);
      order.set(seg, s);
      mid = s + (c >> 1);
    }
    const l = nNodes;
    nNodes += 2;
    left[node] = l;
    start[l] = s;
    count[l] = mid - s;
    start[l + 1] = mid;
    count[l + 1] = s + c - mid;
    stack.push(l, depth + 1, l + 1, depth + 1);
  }

  // Triangles in leaf order: contiguous memory for each leaf.
  const tri = new Float32Array(n * 9);
  const lay = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    tri.set(tri0.subarray(9 * order[i], 9 * order[i] + 9), 9 * i);
    lay[i] = lay0[order[i]];
  }

  // Depth-first traversal holds at most one pending sibling per level.
  const st = new Int32Array(2 * maxDepth + 8);
  const intersect = (o: Vec3, d: Vec3, tMax = Infinity): RayHit | null => {
    if (n === 0) return null;
    const ox = o[0], oy = o[1], oz = o[2];
    const dx = d[0], dy = d[1], dz = d[2];
    const ix = 1 / dx, iy = 1 / dy, iz = 1 / dz;
    let best = tMax;
    let bestTri = -1;
    let sp = 0;
    st[sp++] = 0;
    while (sp > 0) {
      const node = st[--sp];
      // slab test
      let t0 = (bmin[3 * node] - ox) * ix;
      let t1 = (bmax[3 * node] - ox) * ix;
      let tn = Math.min(t0, t1);
      let tf = Math.max(t0, t1);
      t0 = (bmin[3 * node + 1] - oy) * iy;
      t1 = (bmax[3 * node + 1] - oy) * iy;
      tn = Math.max(tn, Math.min(t0, t1));
      tf = Math.min(tf, Math.max(t0, t1));
      t0 = (bmin[3 * node + 2] - oz) * iz;
      t1 = (bmax[3 * node + 2] - oz) * iz;
      tn = Math.max(tn, Math.min(t0, t1));
      tf = Math.min(tf, Math.max(t0, t1));
      // NaN (ray parallel to a slab face, origin on it) keeps the node: conservative.
      if (tn > tf || tf < 0 || tn > best) continue;
      const l = left[node];
      if (l >= 0) {
        st[sp++] = l;
        st[sp++] = l + 1;
        continue;
      }
      const s = start[node];
      const e = s + count[node];
      for (let i = s; i < e; i++) {
        const q = 9 * i;
        const ax = tri[q], ay = tri[q + 1], az = tri[q + 2];
        const e1x = tri[q + 3] - ax, e1y = tri[q + 4] - ay, e1z = tri[q + 5] - az;
        const e2x = tri[q + 6] - ax, e2y = tri[q + 7] - ay, e2z = tri[q + 8] - az;
        const px = dy * e2z - dz * e2y;
        const py = dz * e2x - dx * e2z;
        const pz = dx * e2y - dy * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (det > -1e-12 && det < 1e-12) continue;
        const inv = 1 / det;
        const sx = ox - ax, sy = oy - ay, sz = oz - az;
        const u = (sx * px + sy * py + sz * pz) * inv;
        if (u < 0 || u > 1) continue;
        const qx = sy * e1z - sz * e1y;
        const qy = sz * e1x - sx * e1z;
        const qz = sx * e1y - sy * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (t > EPS_T && t < best) {
          best = t;
          bestTri = i;
        }
      }
    }
    if (bestTri < 0) return null;
    const q = 9 * bestTri;
    const e1x = tri[q + 3] - tri[q], e1y = tri[q + 4] - tri[q + 1], e1z = tri[q + 5] - tri[q + 2];
    const e2x = tri[q + 6] - tri[q], e2y = tri[q + 7] - tri[q + 1], e2z = tri[q + 8] - tri[q + 2];
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    return { t: best, layer: lay[bestTri], point: [ox + dx * best, oy + dy * best, oz + dz * best], normal: [nx, ny, nz] };
  };

  const bounds = n
    ? { min: [bmin[0], bmin[1], bmin[2]] as Vec3, max: [bmax[0], bmax[1], bmax[2]] as Vec3 }
    : { min: [0, 0, 0] as Vec3, max: [0, 0, 0] as Vec3 };
  return { layerNames: mesh.layers.map((l) => l.name), nTriangles: n, bounds, intersect };
}

/** Specular reflection of unit direction d on a surface of unit normal nrm. */
export function reflect(d: Vec3, nrm: Vec3): Vec3 {
  const k = 2 * (d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2]);
  return [d[0] - k * nrm[0], d[1] - k * nrm[1], d[2] - k * nrm[2]];
}

/** Start point of the next leg: the hit point nudged to the side the ray leaves towards. */
export function nudge(p: Vec3, nrm: Vec3, dOut: Vec3, eps = 1e-4): Vec3 {
  const s = nrm[0] * dOut[0] + nrm[1] * dOut[1] + nrm[2] * dOut[2] >= 0 ? eps : -eps;
  return [p[0] + s * nrm[0], p[1] + s * nrm[1], p[2] + s * nrm[2]];
}

export type EmitPattern = 'sphere' | 'xy' | 'xz' | 'yz';

/** n unit directions: an even sphere (Fibonacci lattice) or a fan in a coordinate plane. */
export function emitDirections(pattern: EmitPattern, n: number): Vec3[] {
  const out: Vec3[] = [];
  if (pattern === 'sphere') {
    const ga = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const z = 1 - (2 * (i + 0.5)) / n;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      const a = ga * i;
      out.push([r * Math.cos(a), r * Math.sin(a), z]);
    }
    return out;
  }
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * (i + 0.5)) / n;
    const c = Math.cos(a);
    const s = Math.sin(a);
    out.push(pattern === 'xy' ? [c, s, 0] : pattern === 'xz' ? [c, 0, s] : [0, c, s]);
  }
  return out;
}
