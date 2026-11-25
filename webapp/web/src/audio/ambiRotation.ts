export const AMBI_ORDER = 2;
export const N_CH = 9;
const N_DIRS = 64;
const DEG = Math.PI / 180;

export type Mat3 = [number, number, number, number, number, number, number, number, number];

/** Real spherical harmonics, ACN order, SN3D, for a unit direction (engine: ir_synthesis.sh_sn3d). */
export function shSn3d(x: number, y: number, z: number): number[] {
  const s3 = Math.sqrt(3);
  return [1, y, z, x, s3 * x * y, s3 * y * z, 0.5 * (3 * z * z - 1), s3 * x * z, 0.5 * s3 * (x * x - y * y)];
}

/** Near-uniform unit directions (engine: ambidecode.virtual_layout). */
export function fibonacciSphere(n: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  const golden = Math.PI * (1 + Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const k = i + 0.5;
    const z = 1 - (2 * k) / n;
    const phi = golden * k;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    out.push([r * Math.cos(phi), r * Math.sin(phi), z]);
  }
  return out;
}

/** Row-major 3x3 product a @ b. */
function mul3(a: Mat3, b: Mat3): Mat3 {
  const o = new Array(9).fill(0) as Mat3;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return o;
}

export function rotZ(deg: number): Mat3 {
  const c = Math.cos(deg * DEG);
  const s = Math.sin(deg * DEG);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

export function rotY(deg: number): Mat3 {
  const c = Math.cos(deg * DEG);
  const s = Math.sin(deg * DEG);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}

/**
 * The field rotation that renders a listener whose head looks along azimuth `yawDeg`
 * (CCW from +X) and elevation `pitchDeg` (+ up): the inverse of the head orientation
 * Rz(yaw) Ry(-pitch), i.e. Ry(pitch) Rz(-yaw). A source straight ahead of the head
 * ends up at +x (the decoder's front).
 */
export function headToField(yawDeg: number, pitchDeg: number): Mat3 {
  return mul3(rotY(pitchDeg), rotZ(-yawDeg));
}

/** Gauss-Jordan inverse of an n x n row-major matrix. */
function invert(a: number[], n: number): number[] {
  const m = a.slice();
  const inv = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) inv[i * n + i] = 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r * n + c]) > Math.abs(m[p * n + c])) p = r;
    if (Math.abs(m[p * n + c]) < 1e-14) throw new Error('singular matrix');
    if (p !== c) {
      for (let k = 0; k < n; k++) {
        [m[c * n + k], m[p * n + k]] = [m[p * n + k], m[c * n + k]];
        [inv[c * n + k], inv[p * n + k]] = [inv[p * n + k], inv[c * n + k]];
      }
    }
    const d = m[c * n + c];
    for (let k = 0; k < n; k++) {
      m[c * n + k] /= d;
      inv[c * n + k] /= d;
    }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r * n + c];
      if (f === 0) continue;
      for (let k = 0; k < n; k++) {
        m[r * n + k] -= f * m[c * n + k];
        inv[r * n + k] -= f * inv[c * n + k];
      }
    }
  }
  return inv;
}

let cache: { dirs: [number, number, number][]; pinv: number[] } | null = null;

/** pinv(Y) = (Y^T Y)^-1 Y^T, (9 x 64), for the fixed layout. */
function layout() {
  if (cache) return cache;
  const dirs = fibonacciSphere(N_DIRS);
  const Y = dirs.map((d) => shSn3d(d[0], d[1], d[2]));
  const yty = new Array(N_CH * N_CH).fill(0);
  for (const row of Y) for (let i = 0; i < N_CH; i++) for (let j = 0; j < N_CH; j++) yty[i * N_CH + j] += row[i] * row[j];
  const inv = invert(yty, N_CH);
  const pinv = new Array(N_CH * N_DIRS).fill(0);
  for (let i = 0; i < N_CH; i++)
    for (let s = 0; s < N_DIRS; s++) {
      let v = 0;
      for (let k = 0; k < N_CH; k++) v += inv[i * N_CH + k] * Y[s][k];
      pinv[i * N_DIRS + s] = v;
    }
  cache = { dirs, pinv };
  return cache;
}

/**
 * (9 x 9) row-major matrix M with ambi' = M ambi: the field turned by the 3x3 rotation R
 * (a plane wave from d then comes from R d). Same construction as the engine.
 */
export function shRotation(R: Mat3): Float64Array {
  const { dirs, pinv } = layout();
  // X = pinv(Y) Yr solves Y X = Yr; M = X^T.
  const Yr = dirs.map((d) =>
    shSn3d(R[0] * d[0] + R[1] * d[1] + R[2] * d[2], R[3] * d[0] + R[4] * d[1] + R[5] * d[2], R[6] * d[0] + R[7] * d[1] + R[8] * d[2]),
  );
  const M = new Float64Array(N_CH * N_CH);
  for (let i = 0; i < N_CH; i++)
    for (let j = 0; j < N_CH; j++) {
      let v = 0;
      for (let s = 0; s < N_DIRS; s++) v += pinv[i * N_DIRS + s] * Yr[s][j];
      // X[i][j] -> M[j][i]
      M[j * N_CH + i] = Math.abs(v) < 1e-12 ? 0 : v;
    }
  return M;
}

/** Field matrix for a head orientation (yaw, pitch in degrees). */
export function headRotationMatrix(yawDeg: number, pitchDeg = 0): Float64Array {
  return shRotation(headToField(yawDeg, pitchDeg));
}

/**
 * Closed form for a field rotation about +z by `alphaDeg` (yaw only): (Y, X) and (T, S)
 * turn by alpha, (V, U) by 2 alpha, W/Z/R unchanged. Used to cross-check shRotation.
 */
export function yawRotationClosedForm(alphaDeg: number): Float64Array {
  const M = new Float64Array(N_CH * N_CH);
  const c = Math.cos(alphaDeg * DEG);
  const s = Math.sin(alphaDeg * DEG);
  const c2 = Math.cos(2 * alphaDeg * DEG);
  const s2 = Math.sin(2 * alphaDeg * DEG);
  const set = (i: number, j: number, v: number) => (M[i * N_CH + j] = v);
  set(0, 0, 1);
  set(2, 2, 1);
  set(6, 6, 1);
  // Y' = c Y + s X ; X' = c X - s Y (ACN 1 = Y, 3 = X)
  set(1, 1, c);
  set(1, 3, s);
  set(3, 3, c);
  set(3, 1, -s);
  // T' = c T + s S ; S' = c S - s T (ACN 5 = T, 7 = S)
  set(5, 5, c);
  set(5, 7, s);
  set(7, 7, c);
  set(7, 5, -s);
  // V' = c2 V + s2 U ; U' = c2 U - s2 V (ACN 4 = V, 8 = U)
  set(4, 4, c2);
  set(4, 8, s2);
  set(8, 8, c2);
  set(8, 4, -s2);
  return M;
}
