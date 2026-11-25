import { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { emitDirections, nudge, reflect, type EmitPattern, type TraceMesh } from '../lib/raytrace';
import { ABS_DB_MIN, orderT, rampLinear } from '../lib/toolRamps';
import { applySegmentStyle, makeSegmentMaterial, syncViewport } from './glowLines';

const noRaycast = () => null;

export type ParticleColorMode = 'order' | 'absorption';
export type PlayState = 'playing' | 'paused' | 'stopped';

export interface ParticleOptions {
  pattern: EmitPattern;
  count: number;
  /** m/s; 343 = real time. */
  speed: number;
  diameterMm: number;
  colorBy: ParticleColorMode;
  trails: boolean;
  maxBounces: number;
  loop: boolean;
  /** Drawn over the walls (depth test off). */
  xray: boolean;
}

/** Stats reported by the animation, a few times per second. */
export interface ParticleStats {
  alive: number;
  total: number;
  maxOrder: number;
  trailsFull: boolean;
}

const DEAD_DB = -ABS_DB_MIN;
const TRAIL_CAP = 250_000;

/**
 * Specular particles: straight legs between hits found by ray casting on the solver
 * mesh, energy x (1 - α) at each hit; a particle dies below -60 dB, after
 * `maxBounces` reflections, or when it leaves an open model.
 */
class ParticleSystem {
  n: number;
  o: Float32Array;
  d: Float32Array;
  legStart: Float64Array;
  legLen: Float32Array;
  db: Float32Array;
  order: Int16Array;
  hitN: Float32Array;
  hitLayer: Int32Array;
  alive: Uint8Array;
  nAlive: number;
  maxOrder = 0;
  trailPos: Float32Array | null;
  trailOrder: Int16Array | null;
  trailDb: Float32Array | null;
  nTrail = 0;
  trailsFull = false;

  constructor(
    private trace: TraceMesh,
    private alpha: Float32Array,
    src: Vec3,
    dirs: Vec3[],
    private maxBounces: number,
    private escape: number,
    trails: boolean,
  ) {
    const n = dirs.length;
    this.n = n;
    this.o = new Float32Array(3 * n);
    this.d = new Float32Array(3 * n);
    this.legStart = new Float64Array(n);
    this.legLen = new Float32Array(n);
    this.db = new Float32Array(n);
    this.order = new Int16Array(n);
    this.hitN = new Float32Array(3 * n);
    this.hitLayer = new Int32Array(n);
    this.alive = new Uint8Array(n).fill(1);
    this.nAlive = n;
    this.trailPos = trails ? new Float32Array(6 * TRAIL_CAP) : null;
    this.trailOrder = trails ? new Int16Array(TRAIL_CAP) : null;
    this.trailDb = trails ? new Float32Array(TRAIL_CAP) : null;
    for (let i = 0; i < n; i++) {
      this.o.set(src, 3 * i);
      this.d.set(dirs[i], 3 * i);
      this.cast(i);
    }
  }

  private cast(i: number) {
    const o: Vec3 = [this.o[3 * i], this.o[3 * i + 1], this.o[3 * i + 2]];
    const d: Vec3 = [this.d[3 * i], this.d[3 * i + 1], this.d[3 * i + 2]];
    const h = this.trace.intersect(o, d, this.escape);
    if (h) {
      this.legLen[i] = h.t;
      this.hitN.set(h.normal, 3 * i);
      this.hitLayer[i] = h.layer;
    } else {
      this.legLen[i] = -this.escape; // no surface ahead: dies at the end of the leg
    }
  }

  private addTrail(i: number, end: Vec3) {
    if (!this.trailPos || !this.trailOrder || !this.trailDb) return;
    if (this.nTrail >= TRAIL_CAP) {
      this.trailsFull = true;
      return;
    }
    const k = this.nTrail++;
    this.trailPos.set([this.o[3 * i], this.o[3 * i + 1], this.o[3 * i + 2], end[0], end[1], end[2]], 6 * k);
    this.trailOrder[k] = this.order[i];
    this.trailDb[k] = this.db[i];
  }

  /** Advance every particle to travelled distance s (m). */
  advance(s: number) {
    for (let i = 0; i < this.n; i++) {
      if (!this.alive[i]) continue;
      // A particle may bounce several times in one frame at real-time speed.
      for (let guard = 0; guard < 64; guard++) {
        const len = this.legLen[i];
        if (s - this.legStart[i] < Math.abs(len)) break;
        const end = this.pointAt(i, Math.abs(len));
        this.addTrail(i, end);
        if (len < 0) {
          this.kill(i);
          break;
        }
        const nrm: Vec3 = [this.hitN[3 * i], this.hitN[3 * i + 1], this.hitN[3 * i + 2]];
        const dNew = reflect([this.d[3 * i], this.d[3 * i + 1], this.d[3 * i + 2]], nrm);
        const a = this.alpha[this.hitLayer[i]] ?? 0;
        this.db[i] += a >= 1 ? -1000 : 10 * Math.log10(1 - a);
        this.order[i]++;
        if (this.order[i] > this.maxOrder) this.maxOrder = this.order[i];
        this.legStart[i] += len;
        if (this.db[i] < DEAD_DB || this.order[i] > this.maxBounces) {
          this.kill(i);
          break;
        }
        this.o.set(nudge(end, nrm, dNew), 3 * i);
        this.d.set(dNew, 3 * i);
        this.cast(i);
      }
    }
  }

  private kill(i: number) {
    this.alive[i] = 0;
    this.nAlive--;
  }

  pointAt(i: number, along: number): Vec3 {
    return [this.o[3 * i] + this.d[3 * i] * along, this.o[3 * i + 1] + this.d[3 * i + 1] * along, this.o[3 * i + 2] + this.d[3 * i + 2] * along];
  }
}

const POINT_VS = /* glsl */ `
attribute vec3 aColor;
uniform float uSize;
uniform float uScale;
varying vec3 vColor;
void main() {
  vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float px = projectionMatrix[3][3] > 0.5 ? uSize * uScale : uSize * uScale / max(1e-4, -mv.z);
  // The sprite is 2.4 diameters wide: the particle plus its glow.
  gl_PointSize = max(5.0, px * 2.4);
}`;

const POINT_FS = /* glsl */ `
varying vec3 vColor;
void main() {
  float r = length(gl_PointCoord - 0.5) * 2.0;
  if (r > 1.0) discard;
  // Bright core (one diameter = r < 0.42), thin dark rim, soft glow.
  float core = 1.0 - smoothstep(0.34, 0.42, r);
  float rim = (1.0 - smoothstep(0.42, 0.5, r)) * 0.6;
  float glow = 0.5 * exp(-7.0 * (r - 0.4) * (r - 0.4)) * (1.0 - smoothstep(0.85, 1.0, r));
  vec4 P = vec4(vColor * glow, glow);
  P = vec4(0.0, 0.0, 0.0, rim) + P * (1.0 - rim);
  P = vec4(mix(vColor, vec3(1.0), 0.45 * (1.0 - r / 0.42)) * core, core) + P * (1.0 - core);
  if (P.a < 0.01) discard;
  gl_FragColor = vec4(P.rgb / P.a, P.a);
  #include <colorspace_fragment>
}`;

/** Round points whose diameter is in metres (perspective) with a 2 px floor. */
function useRoundPointMaterial(diameterM: number, xray: boolean) {
  const camera = useThree((s) => s.camera);
  const height = useThree((s) => s.size.height);
  const dpr = useThree((s) => s.viewport.dpr);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: POINT_VS,
        fragmentShader: POINT_FS,
        uniforms: { uSize: { value: 0.05 }, uScale: { value: 500 } },
        transparent: true,
        depthWrite: false,
      }),
    [],
  );
  useEffect(() => () => mat.dispose(), [mat]);
  const persp = camera instanceof THREE.PerspectiveCamera;
  const fov = persp ? (camera as THREE.PerspectiveCamera).fov : 0;
  const zoom = camera instanceof THREE.OrthographicCamera ? camera.zoom : 1;
  mat.uniforms.uSize.value = diameterM;
  mat.uniforms.uScale.value = persp ? (height * dpr) / (2 * Math.tan((fov * Math.PI) / 360)) : zoom * dpr;
  mat.depthTest = !xray;
  return mat;
}

/** Particle animation from one source (inside the Canvas). */
export function ParticleAnimation({
  trace,
  alpha,
  source,
  options,
  play,
  runKey,
  sceneRadius,
  onStats,
}: {
  trace: TraceMesh;
  alpha: Float32Array;
  source: Vec3;
  options: ParticleOptions;
  play: PlayState;
  /** Changes restart the emission. */
  runKey: string;
  sceneRadius: number;
  onStats?: (s: ParticleStats) => void;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const sys = useRef<ParticleSystem | null>(null);
  const dist = useRef(0);
  const lastStats = useRef(0);
  const nTrailDrawn = useRef(0);
  const opts = useRef(options);
  opts.current = options;
  const statsCb = useRef(onStats);
  statsCb.current = onStats;

  const n = Math.max(1, Math.round(options.count));
  const pts = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    return g;
  }, [n]);
  useEffect(() => () => pts.dispose(), [pts]);
  const trailGeom = useMemo(() => {
    if (!options.trails) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6 * TRAIL_CAP), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(6 * TRAIL_CAP), 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    return g;
  }, [options.trails]);
  useEffect(() => () => trailGeom?.dispose(), [trailGeom]);
  // Legs in progress (last hit to the particle), rebuilt every frame.
  const liveGeom = useMemo(() => {
    if (!options.trails) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6 * n), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(6 * n), 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    return g;
  }, [options.trails, n]);
  useEffect(() => () => liveGeom?.dispose(), [liveGeom]);

  const pointMat = useRoundPointMaterial(options.diameterMm / 1000, options.xray);
  // Comet tails: the last stretch of every particle's current leg, fading out.
  const tails = useMemo(() => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0]), 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    const dyn = (k: number) => new THREE.InstancedBufferAttribute(new Float32Array(k * n), k).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('iA', dyn(3));
    g.setAttribute('iB', dyn(3));
    g.setAttribute('iColor', dyn(3));
    g.setAttribute('iS', dyn(2));
    const w = new THREE.InstancedBufferAttribute(new Float32Array(n).fill(1), 1);
    g.setAttribute('iWeight', w);
    g.instanceCount = 0;
    return g;
  }, [n]);
  useEffect(() => () => tails.dispose(), [tails]);
  const tailMat = useMemo(() => {
    const m = makeSegmentMaterial();
    applySegmentStyle(m, { core: 1.1, casing: 0.8, casingAlpha: 0.3, glow: 4, glowAlpha: 0.4, hot: 0.1, opacity: 0.9 });
    m.uniforms.uFront.value = 0;
    m.uniforms.uFade.value = 1;
    return m;
  }, []);
  useEffect(() => () => tailMat.dispose(), [tailMat]);
  tailMat.depthTest = !options.xray;
  const tailVp = useMemo(() => new THREE.Vector4(), []);
  const escape = Math.max(10, sceneRadius * 6);

  const emit = () => {
    const o = opts.current;
    sys.current = new ParticleSystem(trace, alpha, source, emitDirections(o.pattern, n), o.maxBounces, escape, o.trails);
    dist.current = 0;
    nTrailDrawn.current = 0;
    if (trailGeom) trailGeom.setDrawRange(0, 0);
  };

  // New emission on any change of what is emitted; Stop clears.
  useEffect(() => {
    if (play === 'stopped') {
      sys.current = null;
      pts.setDrawRange(0, 0);
      trailGeom?.setDrawRange(0, 0);
      liveGeom?.setDrawRange(0, 0);
      tails.instanceCount = 0;
      statsCb.current?.({ alive: 0, total: 0, maxOrder: 0, trailsFull: false });
      invalidate();
      return;
    }
    if (!sys.current) emit();
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [play]);
  useEffect(() => {
    if (play === 'stopped') return;
    emit();
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey, trace, alpha, source[0], source[1], source[2], n, trailGeom]);

  const col = useMemo(() => new THREE.Color(), []);
  const colorFor = (order: number, db: number) => {
    const o = opts.current;
    return rampLinear(o.colorBy === 'order' ? orderT(order) : Math.min(1, -db / ABS_DB_MIN), col);
  };

  useFrame((_, delta) => {
    const s = sys.current;
    if (!s) return;
    const o = opts.current;
    if (play === 'playing') {
      dist.current += o.speed * Math.min(delta, 0.05);
      s.advance(dist.current);
      if (s.nAlive === 0 && o.loop) emit();
    }
    const S = sys.current!;
    // Live particles, compacted at the front of the buffers.
    const P = pts.attributes.position.array as Float32Array;
    const C = pts.attributes.aColor.array as Float32Array;
    const LP = liveGeom ? (liveGeom.attributes.position.array as Float32Array) : null;
    const LC = liveGeom ? (liveGeom.attributes.color.array as Float32Array) : null;
    const TA = tails.attributes.iA.array as Float32Array;
    const TB = tails.attributes.iB.array as Float32Array;
    const TCo = tails.attributes.iColor.array as Float32Array;
    const TS = tails.attributes.iS.array as Float32Array;
    // About a tenth of a second of travel, at least 15 cm.
    const tailLen = Math.max(0.15, Math.min(sceneRadius * 0.4, o.speed * 0.12));
    tailMat.uniforms.uTail.value = tailLen * 0.45;
    let k = 0;
    for (let i = 0; i < S.n; i++) {
      if (!S.alive[i]) continue;
      const along = Math.min(dist.current - S.legStart[i], Math.abs(S.legLen[i]));
      const p = S.pointAt(i, along);
      P[3 * k] = p[0];
      P[3 * k + 1] = p[1];
      P[3 * k + 2] = p[2];
      colorFor(S.order[i], S.db[i]);
      C[3 * k] = col.r;
      C[3 * k + 1] = col.g;
      C[3 * k + 2] = col.b;
      if (LP && LC) {
        LP.set([S.o[3 * i], S.o[3 * i + 1], S.o[3 * i + 2], p[0], p[1], p[2]], 6 * k);
        LC.set([col.r, col.g, col.b, col.r, col.g, col.b], 6 * k);
      }
      const tl = Math.min(along, tailLen);
      const q = S.pointAt(i, along - tl);
      TA.set(q, 3 * k);
      TB.set(p, 3 * k);
      TCo[3 * k] = col.r;
      TCo[3 * k + 1] = col.g;
      TCo[3 * k + 2] = col.b;
      TS[2 * k] = -tl;
      TS[2 * k + 1] = 0;
      k++;
    }
    tails.instanceCount = k;
    for (const name of ['iA', 'iB', 'iColor', 'iS']) tails.attributes[name].needsUpdate = true;
    pts.setDrawRange(0, k);
    pts.attributes.position.needsUpdate = true;
    pts.attributes.aColor.needsUpdate = true;
    if (liveGeom) {
      liveGeom.setDrawRange(0, 2 * k);
      liveGeom.attributes.position.needsUpdate = true;
      liveGeom.attributes.color.needsUpdate = true;
    }
    if (trailGeom && S.trailPos && S.trailOrder && S.trailDb) {
      const TP = trailGeom.attributes.position as THREE.BufferAttribute;
      const TC = trailGeom.attributes.color as THREE.BufferAttribute;
      const from = nTrailDrawn.current;
      if (S.nTrail > from) {
        (TP.array as Float32Array).set(S.trailPos.subarray(6 * from, 6 * S.nTrail), 6 * from);
        const tc = TC.array as Float32Array;
        for (let j = from; j < S.nTrail; j++) {
          colorFor(S.trailOrder[j], S.trailDb[j]);
          tc.set([col.r, col.g, col.b, col.r, col.g, col.b], 6 * j);
        }
        TP.addUpdateRange(6 * from, 6 * (S.nTrail - from));
        TC.addUpdateRange(6 * from, 6 * (S.nTrail - from));
        TP.needsUpdate = true;
        TC.needsUpdate = true;
        nTrailDrawn.current = S.nTrail;
        trailGeom.setDrawRange(0, 2 * S.nTrail);
      }
    }
    const now = performance.now();
    if (statsCb.current && now - lastStats.current > 250) {
      lastStats.current = now;
      statsCb.current({ alive: S.nAlive, total: S.n, maxOrder: S.maxOrder, trailsFull: S.trailsFull });
    }
    if (play === 'playing' && (S.nAlive > 0 || o.loop)) invalidate();
  });

  return (
    <group>
      {trailGeom && (
        <lineSegments geometry={trailGeom} raycast={noRaycast} renderOrder={11} frustumCulled={false}>
          <lineBasicMaterial vertexColors transparent opacity={0.28} depthWrite={false} depthTest={!options.xray} />
        </lineSegments>
      )}
      {liveGeom && (
        <lineSegments geometry={liveGeom} raycast={noRaycast} renderOrder={11} frustumCulled={false}>
          <lineBasicMaterial vertexColors transparent opacity={0.28} depthWrite={false} depthTest={!options.xray} />
        </lineSegments>
      )}
      <mesh geometry={tails} material={tailMat} raycast={noRaycast} renderOrder={12} frustumCulled={false} onBeforeRender={(r) => syncViewport(r, tailMat, tailVp)} />
      <points geometry={pts} material={pointMat} raycast={noRaycast} renderOrder={13} frustumCulled={false} />
    </group>
  );
}

export interface DensityResult {
  /** Hit points of each order 0..3 (flat xyz), with their layer index. */
  points: Float32Array[];
  layers: Int32Array[];
  /** Leg ending at each hit point (flat xyz pairs), same order as `points`. */
  legs: Float32Array[];
  nRays: number;
}

/** Specular rays from the source: where the direct sound (order 0) and reflections 1..3 land. */
export function traceDensity(trace: TraceMesh, source: Vec3, nRays: number, sceneRadius: number): DensityResult {
  const pts: number[][] = [[], [], [], []];
  const lay: number[][] = [[], [], [], []];
  const legs: number[][] = [[], [], [], []];
  const tMax = Math.max(10, sceneRadius * 6);
  for (const d0 of emitDirections('sphere', nRays)) {
    let o: Vec3 = source;
    let d: Vec3 = d0;
    for (let k = 0; k < 4; k++) {
      const h = trace.intersect(o, d, tMax);
      if (!h) break;
      pts[k].push(h.point[0], h.point[1], h.point[2]);
      lay[k].push(h.layer);
      legs[k].push(o[0], o[1], o[2], h.point[0], h.point[1], h.point[2]);
      const dn = reflect(d, h.normal);
      o = nudge(h.point, h.normal, dn);
      d = dn;
    }
  }
  return {
    points: pts.map((a) => new Float32Array(a)),
    layers: lay.map((a) => new Int32Array(a)),
    legs: legs.map((a) => new Float32Array(a)),
    nRays,
  };
}

export const DENSITY_ORDER_T = [0, 0.3, 0.62, 1];

/** Hit points (and optionally the rays) of the reflection-density tool. */
export function ReflectionDensity({
  result,
  orders,
  layerFilter,
  hideRays,
  xray,
  dotPx = 4,
}: {
  result: DensityResult;
  orders: boolean[];
  /** Layer indices to keep; null = every surface. */
  layerFilter: Set<number> | null;
  hideRays: boolean;
  xray: boolean;
  dotPx?: number;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const geoms = useMemo(() => {
    const col = new THREE.Color();
    return [0, 1, 2, 3].map((k) => {
      if (!orders[k]) return null;
      const P = result.points[k];
      const L = result.layers[k];
      const G = result.legs[k];
      const keep: number[] = [];
      for (let i = 0; i < L.length; i++) if (!layerFilter || layerFilter.has(L[i])) keep.push(i);
      rampLinear(DENSITY_ORDER_T[k], col);
      const pos = new Float32Array(3 * keep.length);
      const leg = new Float32Array(6 * keep.length);
      keep.forEach((i, j) => {
        pos.set(P.subarray(3 * i, 3 * i + 3), 3 * j);
        leg.set(G.subarray(6 * i, 6 * i + 6), 6 * j);
      });
      const pg = new THREE.BufferGeometry();
      pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.BufferAttribute(leg, 3));
      return { pg, lg, color: `#${col.getHexString()}`, n: keep.length };
    });
  }, [result, orders, layerFilter]);
  useEffect(
    () => () =>
      geoms.forEach((g) => {
        g?.pg.dispose();
        g?.lg.dispose();
      }),
    [geoms],
  );
  useEffect(() => invalidate(), [geoms, hideRays, xray, invalidate]);
  return (
    <group>
      {geoms.map((g, k) =>
        g ? (
          <group key={k}>
            {!hideRays && (
              <lineSegments geometry={g.lg} raycast={noRaycast} renderOrder={11} frustumCulled={false}>
                <lineBasicMaterial color={g.color} transparent opacity={0.16} depthWrite={false} depthTest={!xray} />
              </lineSegments>
            )}
            <points geometry={g.pg} raycast={noRaycast} renderOrder={12 + k} frustumCulled={false}>
              <pointsMaterial color={g.color} size={dotPx} sizeAttenuation={false} transparent opacity={0.95} depthWrite={false} depthTest={!xray} />
            </points>
          </group>
        ) : null,
      )}
    </group>
  );
}
