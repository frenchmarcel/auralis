import * as THREE from 'three';
import type { Vec3 } from '../api';

/** One polyline drawn as a glowing ribbon. */
export interface GlowPath {
  id: number;
  points: Vec3[];
  color: string;
  /** 0..1: weaker paths are drawn fainter. */
  weight: number;
  /** Kind of the inner nodes: 0 reflection (disc), 1 diffraction (diamond). */
  nodeKind?: 0 | 1;
}

/** Node kinds of the point material. */
export const NODE_REFLECTION = 0;
export const NODE_DIFFRACTION = 1;
export const NODE_ARRIVAL = 2;
export const NODE_HALO = 3;
export const NODE_HEAD = 4;

const SEG_VS = /* glsl */ `
attribute vec3 iA;
attribute vec3 iB;
attribute vec3 iColor;
attribute vec2 iS;
attribute float iWeight;
uniform vec4 uViewport;
uniform float uPx;
uniform float uCore;
uniform float uCasing;
uniform float uGlow;
uniform float uFront;
varying vec2 vA;
varying vec2 vB;
varying vec2 vW;
varying vec2 vS;
varying vec3 vColor;
varying float vWeight;
void main() {
  mat4 pmv = projectionMatrix * modelViewMatrix;
  vec4 ca = pmv * vec4(iA, 1.0);
  vec4 cb = pmv * vec4(iB, 1.0);
  vec2 s = iS;
  // Clip against the near plane (z = -w) so both ends project.
  float da = ca.z + ca.w;
  float db = cb.z + cb.w;
  if (da < 0.0 && db < 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  if (da < 0.0) {
    float t = da / (da - db) + 1e-4;
    ca = mix(ca, cb, t);
    s.x = mix(s.x, s.y, t);
  } else if (db < 0.0) {
    float t = db / (db - da) + 1e-4;
    cb = mix(cb, ca, t);
    s.y = mix(s.y, s.x, t);
  }
  vec2 sa = uViewport.xy + (ca.xy / ca.w * 0.5 + 0.5) * uViewport.zw;
  vec2 sb = uViewport.xy + (cb.xy / cb.w * 0.5 + 0.5) * uViewport.zw;
  vec2 d = sb - sa;
  float len = length(d);
  vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  // Just wide enough for the glow (exp falloff < 2 % beyond 1.25 radii; wider pulse heads).
  float R = (uCore * 1.6 + uCasing + uGlow * (uFront >= 0.0 ? 1.4 : 1.0) * 1.25) * uPx + 1.0;
  bool atA = position.x < 0.5;
  vec4 c = atA ? ca : cb;
  vec2 sp = (atA ? sa : sb) + nrm * position.y * R + dir * (position.x * 2.0 - 1.0) * R;
  vec2 ndc = (sp - uViewport.xy) / uViewport.zw * 2.0 - 1.0;
  gl_Position = vec4(ndc * c.w, c.z, c.w);
  vA = sa;
  vB = sb;
  vW = vec2(ca.w, cb.w);
  vS = s;
  vColor = iColor;
  vWeight = iWeight;
}`;

const SEG_FS = /* glsl */ `
uniform float uPx;
uniform float uCore;
uniform float uCasing;
uniform float uCasingAlpha;
uniform float uGlow;
uniform float uGlowAlpha;
uniform float uHot;
uniform float uOpacity;
uniform float uFront;
uniform float uTail;
uniform float uAhead;
uniform float uDash;
uniform float uFade;
varying vec2 vA;
varying vec2 vB;
varying vec2 vW;
varying vec2 vS;
varying vec3 vColor;
varying float vWeight;
void main() {
  vec2 p = gl_FragCoord.xy;
  vec2 ab = vB - vA;
  float L2 = dot(ab, ab);
  float u = L2 > 1e-6 ? clamp(dot(p - vA, ab) / L2, 0.0, 1.0) : 0.0;
  float d = length(p - (vA + ab * u));
  // Distance along the path (m), perspective-correct.
  float t = u * vW.x / max(1e-9, u * vW.x + (1.0 - u) * vW.y);
  float s = mix(vS.x, vS.y, t);
  if (uDash > 0.0 && fract(s / uDash) > 0.5) discard;
  float vis = 1.0;
  float head = 0.0;
  if (uFront >= 0.0) {
    if (s > uFront) vis = uAhead;
    else {
      float x = (uFront - s) / max(uTail, 1e-4);
      head = exp(-3.0 * x * x);
    }
  }
  if (vis <= 0.0) discard;
  float core = uCore * uPx * (1.0 + 0.6 * head);
  float cas = uCasing * uPx;
  float glow = uGlow * uPx * (1.0 + 0.4 * head);
  float aa = 0.75;
  float aCore = 1.0 - smoothstep(core - aa, core + aa, d);
  float aCas = (1.0 - smoothstep(core + cas - aa, core + cas + aa, d)) * uCasingAlpha;
  float g = max(0.0, d - core) / max(glow, 1e-3);
  float w = mix(0.4, 1.0, vWeight);
  float aGlow = min(1.0, uGlowAlpha * w * (1.0 + 2.5 * head)) * exp(-2.6 * g * g);
  vec3 hot = mix(vColor, vec3(1.0), clamp(uHot + (uFade > 0.5 ? 0.0 : 0.7) * head, 0.0, 1.0));
  vec4 P = vec4(vColor * aGlow, aGlow);
  P = vec4(0.0, 0.0, 0.0, aCas) + P * (1.0 - aCas);
  P = vec4(hot * aCore, aCore) + P * (1.0 - aCore);
  float A = P.a * uOpacity * vis * mix(0.5, 1.0, vWeight);
  // Comet: only the head and its fading tail.
  if (uFade > 0.5) A *= head;
  if (A < 0.004) discard;
  gl_FragColor = vec4(P.rgb / max(P.a, 1e-4), A);
  #include <colorspace_fragment>
}`;

export interface SegmentStyle {
  /** Half-width of the bright core, CSS px. */
  core: number;
  casing: number;
  casingAlpha: number;
  /** Glow radius beyond the core, CSS px. */
  glow: number;
  glowAlpha: number;
  /** Whitening of the core, 0..1. */
  hot: number;
  opacity: number;
}

export const STYLE_BASE: SegmentStyle = { core: 1.3, casing: 1.1, casingAlpha: 0.55, glow: 6, glowAlpha: 0.32, hot: 0.22, opacity: 1 };
export const STYLE_FOCUS: SegmentStyle = { core: 2.2, casing: 1.2, casingAlpha: 0.7, glow: 14, glowAlpha: 0.62, hot: 0.42, opacity: 1 };
export const STYLE_DIM: SegmentStyle = { core: 0.9, casing: 0, casingAlpha: 0, glow: 2, glowAlpha: 0.0, hot: 0, opacity: 0.16 };

/** Material of the glowing ribbons (screen-space width, depth-tested against the scene). */
export function makeSegmentMaterial(): THREE.ShaderMaterial {
  const m = new THREE.ShaderMaterial({
    vertexShader: SEG_VS,
    fragmentShader: SEG_FS,
    uniforms: {
      uViewport: { value: new THREE.Vector4(0, 0, 1, 1) },
      uPx: { value: 1 },
      uCore: { value: 1.3 },
      uCasing: { value: 1 },
      uCasingAlpha: { value: 0.5 },
      uGlow: { value: 6 },
      uGlowAlpha: { value: 0.3 },
      uHot: { value: 0.2 },
      uOpacity: { value: 1 },
      uFront: { value: -1 },
      uTail: { value: 1 },
      uAhead: { value: 0 },
      uDash: { value: 0 },
      uFade: { value: 0 },
    },
    transparent: true,
    depthWrite: false,
    // Reflection points lie on the walls: keep the ribbons in front of them.
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -4,
  });
  return m;
}

export function applySegmentStyle(m: THREE.ShaderMaterial, st: SegmentStyle) {
  const u = m.uniforms;
  u.uCore.value = st.core;
  u.uCasing.value = st.casing;
  u.uCasingAlpha.value = st.casingAlpha;
  u.uGlow.value = st.glow;
  u.uGlowAlpha.value = st.glowAlpha;
  u.uHot.value = st.hot;
  u.uOpacity.value = st.opacity;
}

const QUAD = new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0]);

/** Every segment of every path as one instanced quad: one draw call for all of them. */
export function buildSegments(paths: GlowPath[]): THREE.InstancedBufferGeometry {
  let n = 0;
  for (const p of paths) n += Math.max(0, p.points.length - 1);
  const A = new Float32Array(3 * n);
  const B = new Float32Array(3 * n);
  const C = new Float32Array(3 * n);
  const S = new Float32Array(2 * n);
  const W = new Float32Array(n);
  const col = new THREE.Color();
  let k = 0;
  for (const p of paths) {
    col.set(p.color);
    let s = 0;
    for (let i = 1; i < p.points.length; i++) {
      const a = p.points[i - 1];
      const b = p.points[i];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      A.set(a, 3 * k);
      B.set(b, 3 * k);
      C[3 * k] = col.r;
      C[3 * k + 1] = col.g;
      C[3 * k + 2] = col.b;
      S[2 * k] = s;
      S[2 * k + 1] = s + len;
      W[k] = p.weight;
      s += len;
      k++;
    }
  }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(QUAD, 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  g.setAttribute('iA', new THREE.InstancedBufferAttribute(A, 3));
  g.setAttribute('iB', new THREE.InstancedBufferAttribute(B, 3));
  g.setAttribute('iColor', new THREE.InstancedBufferAttribute(C, 3));
  g.setAttribute('iS', new THREE.InstancedBufferAttribute(S, 2));
  g.setAttribute('iWeight', new THREE.InstancedBufferAttribute(W, 1));
  g.instanceCount = n;
  return g;
}

const NODE_VS = /* glsl */ `
attribute vec3 aColor;
attribute vec4 aInfo;
attribute vec3 aNormal;
uniform float uSize;
uniform float uBackAlpha;
uniform float uPx;
uniform float uFront;
uniform float uTail;
uniform float uLift;
varying vec3 vColor;
varying float vKind;
varying float vFlash;
varying float vVis;
varying float vWeight;
varying float vSize;
varying float vBack;
void main() {
  // A reflection point seen from behind its surface (the bisector of its two legs).
  vec3 wp = (modelMatrix * vec4(position, 1.0)).xyz;
  vBack = dot(aNormal, aNormal) > 0.0 && dot(cameraPosition - wp, aNormal) < 0.0 ? uBackAlpha : 1.0;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // Lifted towards the camera: nodes on a wall stay in front of it.
  mv.xyz -= normalize(mv.xyz) * uLift;
  gl_Position = projectionMatrix * mv;
  float kind = aInfo.y;
  float flash = 0.0;
  float vis = 1.0;
  if (uFront >= 0.0 && kind < 2.5) {
    float x = (uFront - aInfo.x) / max(uTail, 1e-4);
    vis = aInfo.x <= uFront ? 1.0 : 0.0;
    flash = x >= 0.0 ? exp(-2.0 * x * x) : 0.0;
  }
  if (kind > 1.5 && kind < 2.5) vis = flash;
  vColor = aColor;
  vKind = kind;
  vFlash = flash;
  vVis = vis;
  vWeight = aInfo.z;
  float size = uSize * aInfo.w * uPx * (1.0 + 0.9 * flash);
  vSize = size;
  gl_PointSize = vis > 0.0 && vBack > 0.0 ? size : 0.0;
}`;

const NODE_FS = /* glsl */ `
uniform float uOpacity;
varying vec3 vColor;
varying float vKind;
varying float vFlash;
varying float vVis;
varying float vWeight;
varying float vSize;
varying float vBack;
void main() {
  if (vVis <= 0.0 || vBack <= 0.0) discard;
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  bool diamond = vKind > 0.5 && vKind < 1.5;
  float r = diamond ? (abs(q.x) + abs(q.y)) * 0.8 : length(q);
  if (r > 1.0) discard;
  float aa = 2.0 / max(vSize, 1.0);
  vec4 P;
  if (vKind > 1.5 && vKind < 2.5) {
    // Arrival flash: an expanding ring in the path colour.
    float ring = 1.0 - smoothstep(0.1, 0.1 + aa * 1.5, abs(r - 0.62));
    float glow = 0.5 * exp(-8.0 * (r - 0.62) * (r - 0.62));
    float a = max(ring, glow);
    P = vec4(mix(vColor, vec3(1.0), 0.3), a * vFlash);
  } else if (vKind > 2.5 && vKind < 3.5) {
    // Halo around a source or receiver marker: a thin ring and a soft glow.
    float ring = (1.0 - smoothstep(0.035, 0.035 + aa, abs(r - 0.5))) * 0.85;
    float glow = 0.4 * exp(-5.0 * r * r) * (1.0 - smoothstep(0.85, 1.0, r));
    P = vec4(mix(vColor, vec3(1.0), 0.2 * ring), max(ring, glow));
  } else {
    float core = vKind > 3.5 ? 0.3 : 0.38;
    float aCore = 1.0 - smoothstep(core - aa, core + aa, r);
    float aCas = (1.0 - smoothstep(core + 0.1 - aa, core + 0.1 + aa, r)) * (vKind > 3.5 ? 0.0 : 0.75);
    float g = max(0.0, r - core) / (1.0 - core);
    float aGlow = (vKind > 3.5 ? 0.85 : 0.45 + 0.4 * vFlash) * exp(-4.0 * g * g) * (1.0 - smoothstep(0.9, 1.0, r));
    vec3 hot = mix(vColor, vec3(1.0), vKind > 3.5 ? 0.75 : 0.35 + 0.5 * vFlash);
    vec4 C = vec4(vColor * aGlow, aGlow);
    C = vec4(0.0, 0.0, 0.0, aCas) + C * (1.0 - aCas);
    C = vec4(hot * aCore, aCore) + C * (1.0 - aCore);
    P = vec4(C.rgb / max(C.a, 1e-4), C.a * mix(0.55, 1.0, vWeight));
  }
  float A = P.a * uOpacity * vBack;
  if (A < 0.004) discard;
  gl_FragColor = vec4(P.rgb, A);
  #include <colorspace_fragment>
}`;

/** Round (or diamond) glowing sprites whose size is in CSS px. */
export function makeNodeMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: NODE_VS,
    fragmentShader: NODE_FS,
    uniforms: {
      uSize: { value: 16 },
      uPx: { value: 1 },
      uFront: { value: -1 },
      uTail: { value: 1 },
      uLift: { value: 0 },
      uBackAlpha: { value: 1 },
      uOpacity: { value: 1 },
    },
    transparent: true,
    depthWrite: false,
  });
}

export interface NodeSpec {
  position: Vec3;
  color: string;
  /** Distance along the path (m) at which the node is reached. */
  s: number;
  kind: number;
  weight: number;
  /** Size multiplier of the sprite. */
  size?: number;
  /** Side of the surface the node is seen from (hidden or faint from behind). */
  normal?: Vec3;
}

export function buildNodes(nodes: NodeSpec[]): THREE.BufferGeometry {
  const n = nodes.length;
  const P = new Float32Array(3 * n);
  const C = new Float32Array(3 * n);
  const I = new Float32Array(4 * n);
  const N = new Float32Array(3 * n);
  const col = new THREE.Color();
  nodes.forEach((nd, k) => {
    P.set(nd.position, 3 * k);
    col.set(nd.color);
    C[3 * k] = col.r;
    C[3 * k + 1] = col.g;
    C[3 * k + 2] = col.b;
    I[4 * k] = nd.s;
    I[4 * k + 1] = nd.kind;
    I[4 * k + 2] = nd.weight;
    I[4 * k + 3] = nd.size ?? 1;
    if (nd.normal) N.set(nd.normal, 3 * k);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(P, 3));
  g.setAttribute('aColor', new THREE.BufferAttribute(C, 3));
  g.setAttribute('aInfo', new THREE.BufferAttribute(I, 4));
  g.setAttribute('aNormal', new THREE.BufferAttribute(N, 3));
  return g;
}

/** Unit bisector at b of the legs towards a and c: the normal of a specular reflector. */
function bisector(a: Vec3, b: Vec3, c: Vec3): Vec3 | undefined {
  const u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
  const lu = Math.hypot(u[0], u[1], u[2]);
  const lv = Math.hypot(v[0], v[1], v[2]);
  if (lu < 1e-9 || lv < 1e-9) return undefined;
  const n: Vec3 = [u[0] / lu + v[0] / lv, u[1] / lu + v[1] / lv, u[2] / lu + v[2] / lv];
  const ln = Math.hypot(n[0], n[1], n[2]);
  return ln < 1e-6 ? undefined : [n[0] / ln, n[1] / ln, n[2] / ln];
}

/** Inner nodes of every path, plus an arrival flash at its end (seen while animating). */
export function pathNodes(paths: GlowPath[], size = 1): NodeSpec[] {
  const out: NodeSpec[] = [];
  for (const p of paths) {
    let s = 0;
    for (let i = 1; i < p.points.length; i++) {
      const a = p.points[i - 1];
      const b = p.points[i];
      s += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      const last = i === p.points.length - 1;
      const kind = last ? NODE_ARRIVAL : (p.nodeKind ?? NODE_REFLECTION);
      out.push({ position: b, color: p.color, s, kind, weight: p.weight, size: last ? size * 2.4 : size, normal: kind === NODE_REFLECTION ? bisector(a, b, p.points[i + 1]) : undefined });
    }
  }
  return out;
}

/** Keeps the viewport and pixel ratio uniforms of a material in step with the renderer. */
export function syncViewport(renderer: THREE.WebGLRenderer, m: THREE.ShaderMaterial, v: THREE.Vector4) {
  renderer.getCurrentViewport(v);
  if (m.uniforms.uViewport) m.uniforms.uViewport.value.copy(v);
  m.uniforms.uPx.value = renderer.getPixelRatio();
}

/** Point at distance s along a polyline (clamped), written into `out`. */
export function pointAlong(points: Vec3[], s: number, out: Vec3): Vec3 {
  let acc = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    if (acc + len >= s || i === points.length - 1) {
      const t = len > 0 ? Math.min(1, Math.max(0, (s - acc) / len)) : 0;
      out[0] = a[0] + (b[0] - a[0]) * t;
      out[1] = a[1] + (b[1] - a[1]) * t;
      out[2] = a[2] + (b[2] - a[2]) * t;
      return out;
    }
    acc += len;
  }
  out[0] = points[0][0];
  out[1] = points[0][1];
  out[2] = points[0][2];
  return out;
}

export function pathLength(points: Vec3[]): number {
  let s = 0;
  for (let i = 1; i < points.length; i++) s += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1], points[i][2] - points[i - 1][2]);
  return s;
}

export interface PickHit {
  id: number;
  /** World point on the path nearest to the cursor. */
  point: Vec3;
  /** Distance along the path (m) of that point. */
  s: number;
  px: number;
}

/**
 * Nearest path to a cursor (CSS px in the canvas), in screen space; parts beyond
 * `maxS(id)` (not yet reached by the animation) are ignored.
 */
export function pickPath(paths: GlowPath[], camera: THREE.Camera, width: number, height: number, mx: number, my: number, maxPx: number, maxS?: (id: number) => number): PickHit | null {
  const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = m.elements;
  const proj = (p: Vec3) => {
    const x = e[0] * p[0] + e[4] * p[1] + e[8] * p[2] + e[12];
    const y = e[1] * p[0] + e[5] * p[1] + e[9] * p[2] + e[13];
    const z = e[2] * p[0] + e[6] * p[1] + e[10] * p[2] + e[14];
    const w = e[3] * p[0] + e[7] * p[1] + e[11] * p[2] + e[15];
    return { x, y, z, w };
  };
  let best: PickHit | null = null;
  let bestWeight = -1;
  for (const p of paths) {
    const lim = maxS ? maxS(p.id) : Infinity;
    let s0 = 0;
    for (let i = 1; i < p.points.length; i++) {
      const a3 = p.points[i - 1];
      const b3 = p.points[i];
      const len = Math.hypot(b3[0] - a3[0], b3[1] - a3[1], b3[2] - a3[2]);
      const ca = proj(a3);
      const cb = proj(b3);
      const da = ca.z + ca.w;
      const db = cb.z + cb.w;
      if (s0 >= lim || (da < 0 && db < 0)) {
        s0 += len;
        continue;
      }
      // Visible part [ta, tb] of the segment: in front of the near plane, behind the animation front.
      let ta = 0;
      let tb = 1;
      if (da < 0) ta = da / (da - db);
      if (db < 0) tb = 1 - db / (db - da);
      if (s0 + len > lim && len > 0) tb = Math.min(tb, (lim - s0) / len);
      const at = (u: number) => ({ x: ca.x + (cb.x - ca.x) * u, y: ca.y + (cb.y - ca.y) * u, w: ca.w + (cb.w - ca.w) * u });
      const qa = at(ta);
      const qb = at(tb);
      const ax = ((qa.x / qa.w) * 0.5 + 0.5) * width;
      const ay = (0.5 - (qa.y / qa.w) * 0.5) * height;
      const bx = ((qb.x / qb.w) * 0.5 + 0.5) * width;
      const by = (0.5 - (qb.y / qb.w) * 0.5) * height;
      const abx = bx - ax;
      const aby = by - ay;
      const L2 = abx * abx + aby * aby;
      const u = L2 > 1e-9 ? Math.min(1, Math.max(0, ((mx - ax) * abx + (my - ay) * aby) / L2)) : 0;
      const d = Math.hypot(ax + abx * u - mx, ay + aby * u - my);
      const better = !best || d < best.px - 0.5 || (Math.abs(d - best.px) <= 0.5 && p.weight > bestWeight);
      if (d <= maxPx && better) {
        // Screen parameter to world parameter (perspective-correct).
        const v = (u * qa.w) / Math.max(1e-12, u * qa.w + (1 - u) * qb.w);
        const t = ta + (tb - ta) * v;
        best = { id: p.id, point: [a3[0] + (b3[0] - a3[0]) * t, a3[1] + (b3[1] - a3[1]) * t, a3[2] + (b3[2] - a3[2]) * t], s: s0 + len * t, px: d };
        bestWeight = p.weight;
      }
      s0 += len;
    }
  }
  return best;
}
