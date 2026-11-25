import { useEffect, useMemo, useRef, useState, type MutableRefObject, type ReactNode } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { COLORS } from '../lib/colors';
import {
  applySegmentStyle,
  buildNodes,
  buildSegments,
  makeNodeMaterial,
  makeSegmentMaterial,
  NODE_HALO,
  NODE_HEAD,
  pathLength,
  pathNodes,
  pickPath,
  pointAlong,
  STYLE_BASE,
  STYLE_DIM,
  STYLE_FOCUS,
  syncViewport,
  type GlowPath,
} from './glowLines';

const noRaycast = () => null;

export interface ArrivalPath {
  id: number;
  path: Vec3[];
  color: string;
  /** 0..1, from the level: weaker paths are fainter. */
  weight: number;
  diffracted?: boolean;
}

/** Pulse animation: the sound leaves the source at t = 0 and travels every path at c. */
export interface PathAnimation {
  on: boolean;
  playing: boolean;
  /** Slow-motion factor (100 = 100 times slower than real time). */
  slow: number;
  /** The clock loops after this time (ms after emission). */
  endMs: number;
  /** Speed of sound, m/s. */
  c: number;
}

const toGlow = (a: ArrivalPath): GlowPath => ({ id: a.id, points: a.path, color: a.color, weight: a.weight, nodeKind: a.diffracted ? 1 : 0 });

type Mat = THREE.ShaderMaterial;

/**
 * Reflection paths as glowing ribbons with a node at every reflection point. `paths`
 * are drawn normally (dimmed when `dim`), `focus` (selected or hovered) bright on top.
 * Hover and click are picked in screen space; `tip` gives the tooltip of a path.
 */
export function ReflectionPaths({
  paths,
  focus,
  dim,
  throughWalls,
  solidRoom = false,
  anim,
  timeRef,
  seekKey,
  onTime,
  source,
  receiver,
  radius,
  onHover,
  onPick,
  tip,
}: {
  paths: ArrivalPath[];
  focus: ArrivalPath[];
  dim: boolean;
  throughWalls: boolean;
  /** Opaque room: reflection points seen from behind their surface are hidden (faint through walls). */
  solidRoom?: boolean;
  anim: PathAnimation;
  /** Animation clock, ms after emission (shared with the playback controls). */
  timeRef: MutableRefObject<number>;
  /** Changes when the clock is moved from outside (scrubbing). */
  seekKey: number;
  onTime?: (ms: number) => void;
  source: Vec3 | null;
  receiver: Vec3 | null;
  radius: number;
  onHover?: (id: number | null) => void;
  onPick?: (id: number | null, additive: boolean) => void;
  tip?: (id: number) => ReactNode;
}) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const invalidate = useThree((s) => s.invalidate);

  // Weaker first, so the strong paths are drawn on top.
  const base = useMemo(() => [...paths].sort((a, b) => a.weight - b.weight).map(toGlow), [paths]);
  const foc = useMemo(() => focus.map(toGlow), [focus]);
  const lengths = useMemo(() => new Map([...base, ...foc].map((p) => [p.id, pathLength(p.points)])), [base, foc]);

  const baseSeg = useMemo(() => buildSegments(base), [base]);
  const focSeg = useMemo(() => buildSegments(foc), [foc]);
  const baseNodes = useMemo(() => buildNodes(pathNodes(base)), [base]);
  const focNodes = useMemo(() => buildNodes(pathNodes(foc, 1.35)), [foc]);
  const halos = useMemo(
    () =>
      buildNodes([
        ...(source ? [{ position: source, color: COLORS.source, s: 0, kind: NODE_HALO, weight: 1, size: 2.6 }] : []),
        ...(receiver ? [{ position: receiver, color: COLORS.receiver, s: 0, kind: NODE_HALO, weight: 1, size: 2.6 }] : []),
      ]),
    [source, receiver],
  );
  // Pulse heads: one sprite per path, moved every frame.
  const heads = useMemo(() => {
    const g = buildNodes(base.map((p) => ({ position: p.points[0], color: p.color, s: 0, kind: NODE_HEAD, weight: 1, size: 0 })));
    (g.attributes.position as THREE.BufferAttribute).setUsage(THREE.DynamicDrawUsage);
    (g.attributes.aInfo as THREE.BufferAttribute).setUsage(THREE.DynamicDrawUsage);
    return g;
  }, [base]);
  useEffect(() => () => baseSeg.dispose(), [baseSeg]);
  useEffect(() => () => focSeg.dispose(), [focSeg]);
  useEffect(() => () => baseNodes.dispose(), [baseNodes]);
  useEffect(() => () => focNodes.dispose(), [focNodes]);
  useEffect(() => () => halos.dispose(), [halos]);
  useEffect(() => () => heads.dispose(), [heads]);

  const mats = useMemo(() => {
    const segBase = makeSegmentMaterial();
    const segBaseOcc = makeSegmentMaterial();
    const segFoc = makeSegmentMaterial();
    const segFocOcc = makeSegmentMaterial();
    segBaseOcc.depthFunc = THREE.GreaterDepth;
    segFocOcc.depthFunc = THREE.GreaterDepth;
    const nodeBase = makeNodeMaterial();
    const nodeFoc = makeNodeMaterial();
    const nodeMisc = makeNodeMaterial();
    return { segBase, segBaseOcc, segFoc, segFocOcc, nodeBase, nodeFoc, nodeMisc };
  }, []);
  useEffect(() => () => Object.values(mats).forEach((m) => m.dispose()), [mats]);

  // Styles (cheap: uniforms only). Many paths: less glow, or they merge into a haze.
  const dash = Math.max(0.05, radius * 1.4);
  const crowd = Math.min(1, Math.max(0.3, Math.sqrt(80 / Math.max(1, base.length))));
  const baseStyle = { ...STYLE_BASE, glow: STYLE_BASE.glow * crowd, glowAlpha: STYLE_BASE.glowAlpha * (0.5 + 0.5 * crowd), core: STYLE_BASE.core * (0.75 + 0.25 * crowd) };
  applySegmentStyle(mats.segBase, dim ? STYLE_DIM : baseStyle);
  // Hidden parts: dashed, faint, a narrow glow (cheap: few fragments).
  applySegmentStyle(mats.segBaseOcc, dim ? STYLE_DIM : { ...baseStyle, glow: 2, glowAlpha: 0.1, opacity: 0.45 });
  mats.segBaseOcc.uniforms.uOpacity.value = dim ? 0.08 : 0.45;
  mats.segBaseOcc.uniforms.uDash.value = dash;
  applySegmentStyle(mats.segFoc, STYLE_FOCUS);
  applySegmentStyle(mats.segFocOcc, { ...STYLE_FOCUS, glow: 4, glowAlpha: 0.15, opacity: 0.55 });
  mats.segFocOcc.uniforms.uDash.value = dash;
  mats.nodeBase.uniforms.uSize.value = 15 * (0.6 + 0.4 * crowd);
  mats.nodeBase.uniforms.uOpacity.value = dim ? 0.12 : 1;
  mats.nodeFoc.uniforms.uSize.value = 17;
  mats.nodeMisc.uniforms.uSize.value = 18 * (0.6 + 0.4 * crowd);
  for (const m of [mats.nodeBase, mats.nodeFoc, mats.nodeMisc]) {
    m.depthTest = !throughWalls;
    m.uniforms.uLift.value = radius * 0.5;
    m.uniforms.uBackAlpha.value = !solidRoom ? 1 : throughWalls ? 0.4 : 0;
  }
  useEffect(() => invalidate(), [dim, throughWalls, solidRoom, paths, focus, invalidate]);

  const vp = useMemo(() => new THREE.Vector4(), []);
  const sync = (m: Mat) => (r: THREE.WebGLRenderer) => syncViewport(r, m, vp);

  // Animation clock.
  const frontRef = useRef(-1);
  const lastReport = useRef(0);
  const tailM = Math.max(0.25, (0.045 * anim.c * anim.endMs) / 1000);
  const setFront = (front: number) => {
    frontRef.current = front;
    for (const m of [mats.segBase, mats.segBaseOcc, mats.segFoc, mats.segFocOcc, mats.nodeBase, mats.nodeFoc]) {
      m.uniforms.uFront.value = front;
      m.uniforms.uTail.value = tailM;
    }
  };
  const updateHeads = (front: number) => {
    const P = heads.attributes.position as THREE.BufferAttribute;
    const I = heads.attributes.aInfo as THREE.BufferAttribute;
    const pa = P.array as Float32Array;
    const ia = I.array as Float32Array;
    const q: Vec3 = [0, 0, 0];
    base.forEach((p, k) => {
      const L = lengths.get(p.id) ?? 0;
      const on = front >= 0 && front < L;
      ia[4 * k + 3] = on ? 1 : 0;
      if (on) {
        pointAlong(p.points, front, q);
        pa[3 * k] = q[0];
        pa[3 * k + 1] = q[1];
        pa[3 * k + 2] = q[2];
      }
    });
    P.needsUpdate = true;
    I.needsUpdate = true;
  };
  useEffect(() => {
    const front = anim.on ? (anim.c * timeRef.current) / 1000 : -1;
    setFront(front);
    updateHeads(front);
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anim.on, anim.c, seekKey, heads, mats, tailM]);

  useFrame((_, delta) => {
    if (!anim.on) return;
    if (anim.playing) {
      let t = timeRef.current + (Math.min(delta, 0.1) * 1000) / Math.max(1, anim.slow);
      if (t > anim.endMs) t = 0;
      timeRef.current = t;
      const now = performance.now();
      if (onTime && now - lastReport.current > 80) {
        lastReport.current = now;
        onTime(t);
      }
      invalidate();
    }
    const front = (anim.c * timeRef.current) / 1000;
    if (front !== frontRef.current) {
      setFront(front);
      updateHeads(front);
    }
  });

  // Picking in screen space.
  /** Hovered path; `side`: where the tooltip goes so it stays inside the canvas. */
  const [hit, setHit] = useState<{ id: number; point: Vec3; side: string } | null>(null);
  const live = useRef({ base, foc, source, receiver, anim, onHover, onPick, hitId: null as number | null });
  live.current = { ...live.current, base, foc, source, receiver, anim, onHover, onPick };
  useEffect(() => {
    const el = gl.domElement;
    let down: { x: number; y: number } | null = null;
    const nearMarker = (mx: number, my: number, w: number, h: number) => {
      const L = live.current;
      for (const p of [L.source, L.receiver]) {
        if (!p) continue;
        const v = new THREE.Vector3(p[0], p[1], p[2]).project(camera);
        if (Math.hypot((v.x * 0.5 + 0.5) * w - mx, (0.5 - v.y * 0.5) * h - my) < 16) return true;
      }
      return false;
    };
    const pick = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const mx = e.clientX - r.left;
      const my = e.clientY - r.top;
      if (nearMarker(mx, my, r.width, r.height)) return null;
      const L = live.current;
      const ids = new Set(L.foc.map((p) => p.id));
      const all = [...L.foc, ...L.base.filter((p) => !ids.has(p.id))];
      const front = L.anim.on ? (L.anim.c * timeRef.current) / 1000 : Infinity;
      return pickPath(all, camera, r.width, r.height, mx, my, 9, () => front);
    };
    const setHover = (h: { id: number; point: Vec3; side: string } | null) => {
      const L = live.current;
      setHit(h);
      const id = h?.id ?? null;
      if (id !== L.hitId) {
        L.hitId = id;
        el.style.cursor = id !== null ? 'pointer' : '';
        L.onHover?.(id);
      }
    };
    const onMove = (e: PointerEvent) => {
      if (e.buttons) {
        if (live.current.hitId !== null) setHover(null);
        return;
      }
      const h = pick(e);
      const r = el.getBoundingClientRect();
      const side = `${e.clientX - r.left > r.width * 0.6 ? 'left' : ''} ${e.clientY - r.top < r.height * 0.3 ? 'below' : ''}`;
      setHover(h ? { id: h.id, point: h.point, side } : null);
    };
    const onLeave = () => setHover(null);
    const onDown = (e: PointerEvent) => {
      down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
    };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      down = null;
      const r = el.getBoundingClientRect();
      if (nearMarker(e.clientX - r.left, e.clientY - r.top, r.width, r.height)) return;
      const h = pick(e);
      live.current.onPick?.(h ? h.id : null, e.ctrlKey || e.metaKey || e.shiftKey);
    };
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerleave', onLeave);
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointerup', onUp);
    return () => {
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerleave', onLeave);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointerup', onUp);
      el.style.cursor = '';
    };
  }, [gl, camera, timeRef]);
  // A path that leaves the list loses its tooltip.
  useEffect(() => {
    if (hit && !lengths.has(hit.id)) setHit(null);
  }, [hit, lengths]);

  const nBase = base.length;
  const nFoc = foc.length;
  return (
    <group>
      {nBase > 0 && throughWalls && (
        <mesh geometry={baseSeg} material={mats.segBaseOcc} raycast={noRaycast} frustumCulled={false} renderOrder={5} onBeforeRender={sync(mats.segBaseOcc)} />
      )}
      {nBase > 0 && <mesh geometry={baseSeg} material={mats.segBase} raycast={noRaycast} frustumCulled={false} renderOrder={6} onBeforeRender={sync(mats.segBase)} />}
      {nFoc > 0 && throughWalls && (
        <mesh geometry={focSeg} material={mats.segFocOcc} raycast={noRaycast} frustumCulled={false} renderOrder={6} onBeforeRender={sync(mats.segFocOcc)} />
      )}
      {nFoc > 0 && <mesh geometry={focSeg} material={mats.segFoc} raycast={noRaycast} frustumCulled={false} renderOrder={7} onBeforeRender={sync(mats.segFoc)} />}
      <points geometry={halos} material={mats.nodeMisc} raycast={noRaycast} frustumCulled={false} renderOrder={5} onBeforeRender={sync(mats.nodeMisc)} />
      {nBase > 0 && <points geometry={baseNodes} material={mats.nodeBase} raycast={noRaycast} frustumCulled={false} renderOrder={8} onBeforeRender={sync(mats.nodeBase)} />}
      {nFoc > 0 && <points geometry={focNodes} material={mats.nodeFoc} raycast={noRaycast} frustumCulled={false} renderOrder={8} onBeforeRender={sync(mats.nodeFoc)} />}
      {anim.on && nBase > 0 && <points geometry={heads} material={mats.nodeMisc} raycast={noRaycast} frustumCulled={false} renderOrder={9} onBeforeRender={sync(mats.nodeMisc)} />}
      {hit && tip && (
        <Html position={hit.point} zIndexRange={[30, 20]} style={{ pointerEvents: 'none' }}>
          <div className={`path-tip ${hit.side}`}>{tip(hit.id)}</div>
        </Html>
      )}
    </group>
  );
}
