import { useEffect, useMemo, type MutableRefObject } from 'react';
import { useThree, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { DISABLED_COLOR, PointIndex, rampTexture, type Num, type PointFrame } from '../lib/grids';

const finite = (v: Num | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

interface Built {
  geometry: THREE.BufferGeometry;
  /** Grid point of each triangle (hover / click). */
  faceToPoint: Int32Array;
}

function square(p: Vec3, f: PointFrame, h: number): Vec3[] {
  const c = (su: number, sv: number): Vec3 => [p[0] + (f.u[0] * su + f.v[0] * sv) * h, p[1] + (f.u[1] * su + f.v[1] * sv) * h, p[2] + (f.u[2] * su + f.v[2] * sv) * h];
  return [c(-1, -1), c(1, -1), c(1, 1), c(-1, 1)];
}

/**
 * One square per point (side = spacing, along the point's lattice axes). Colours come
 * from a ramp texture looked up with uv.x, so values interpolated across a face map
 * exactly onto the scale. `interpolate`: each square is split into 4 triangles around
 * its centre (own value) and its corners (mean of the points sharing the corner).
 */
function buildValued(idx: number[], points: Vec3[], frames: PointFrame[], values: Num[], spacing: number, lo: number, hi: number, interpolate: boolean, colors: (string | null)[] | null): Built {
  const span = hi - lo || 1;
  const t = (v: number) => (v - lo) / span;
  const pos: number[] = [];
  const uv: number[] = [];
  const col: number[] = [];
  const faces: number[] = [];
  const index = interpolate ? new PointIndex(points, spacing) : null;
  const tmp = new THREE.Color();
  const cornerValue = (c: Vec3, own: number) => {
    const nb = index!.within(c, 0.8 * spacing).map((j) => values[j]).filter(finite);
    return nb.length ? nb.reduce((s, x) => s + x, 0) / nb.length : own;
  };
  for (const i of idx) {
    const v = values[i] as number;
    const f = frames[i];
    const color = colors?.[i] ?? null;
    if (color) tmp.set(color);
    const push = (p: Vec3, val: number) => {
      pos.push(p[0], p[1], p[2]);
      uv.push(t(val), 0.5);
      if (colors) col.push(tmp.r, tmp.g, tmp.b);
    };
    if (interpolate && !colors) {
      const sq = square(points[i], f, spacing / 2);
      const cv = sq.map((c) => cornerValue(c, v));
      for (let k = 0; k < 4; k++) {
        push(points[i], v);
        push(sq[k], cv[k]);
        push(sq[(k + 1) % 4], cv[(k + 1) % 4]);
        faces.push(i);
      }
    } else {
      const sq = square(points[i], f, (spacing / 2) * 0.92);
      for (const tri of [
        [0, 1, 2],
        [0, 2, 3],
      ]) {
        for (const k of tri) push(sq[k], v);
        faces.push(i);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  if (colors) g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.computeBoundingSphere();
  return { geometry: g, faceToPoint: Int32Array.from(faces) };
}

function buildFlat(idx: number[], points: Vec3[], frames: PointFrame[], spacing: number): Built {
  const pos: number[] = [];
  const faces: number[] = [];
  for (const i of idx) {
    const sq = square(points[i], frames[i], (spacing / 2) * 0.86);
    for (const k of [0, 1, 2, 0, 2, 3]) pos.push(...sq[k]);
    faces.push(i, i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeBoundingSphere();
  return { geometry: g, faceToPoint: Int32Array.from(faces) };
}

export interface GridMapProps {
  points: Vec3[];
  frames: PointFrame[];
  spacing: number;
  /** Displayed value per point (null: no value; drawn grey). */
  values: Num[];
  lo: number;
  hi: number;
  ramp: string[];
  interpolate: boolean;
  /** Per-point colours that replace the scale (target pass / fail); null = use the scale. */
  colors?: (string | null)[] | null;
  onHover?: (i: number | null, x: number, y: number) => void;
  onClick?: (i: number, additive: boolean) => void;
}

export function GridMap({ points, frames, spacing, values, lo, hi, ramp, interpolate, colors = null, onHover, onClick }: GridMapProps) {
  const invalidate = useThree((s) => s.invalidate);
  const texture = useMemo(() => rampTexture(ramp), [ramp]);
  useEffect(() => () => texture.dispose(), [texture]);
  const valued = useMemo(() => {
    const idx = values.flatMap((v, i) => (finite(v) && points[i] ? [i] : []));
    return buildValued(idx, points, frames, values, spacing, lo, hi, interpolate, colors);
  }, [points, frames, values, spacing, lo, hi, interpolate, colors]);
  const grey = useMemo(() => {
    const idx = values.flatMap((v, i) => (!finite(v) && points[i] ? [i] : []));
    return buildFlat(idx, points, frames, spacing);
  }, [points, frames, values, spacing]);
  useEffect(() => () => valued.geometry.dispose(), [valued]);
  useEffect(() => () => grey.geometry.dispose(), [grey]);
  useEffect(() => invalidate(), [valued, grey, texture, invalidate]);

  const handlers = (b: Built) => ({
    onPointerMove: onHover
      ? (e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation();
          const i = e.faceIndex !== undefined && e.faceIndex !== null ? b.faceToPoint[e.faceIndex] : undefined;
          onHover(i ?? null, e.nativeEvent.offsetX, e.nativeEvent.offsetY);
        }
      : undefined,
    onPointerOut: onHover ? () => onHover(null, 0, 0) : undefined,
    onClick: onClick
      ? (e: ThreeEvent<MouseEvent>) => {
          if (e.delta > 4) return;
          e.stopPropagation();
          const i = e.faceIndex !== undefined && e.faceIndex !== null ? b.faceToPoint[e.faceIndex] : undefined;
          if (i !== undefined) onClick(i, e.nativeEvent.ctrlKey || e.nativeEvent.metaKey || e.nativeEvent.shiftKey);
        }
      : undefined,
  });

  return (
    <group>
      <mesh geometry={valued.geometry} renderOrder={4} {...handlers(valued)}>
        {colors ? (
          <meshBasicMaterial key="vc" vertexColors side={THREE.DoubleSide} toneMapped={false} />
        ) : (
          <meshBasicMaterial key="map" map={texture} side={THREE.DoubleSide} toneMapped={false} />
        )}
      </mesh>
      <mesh geometry={grey.geometry} renderOrder={4} {...handlers(grey)}>
        <meshBasicMaterial color={DISABLED_COLOR} side={THREE.DoubleSide} toneMapped={false} transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

export type CaptureFn = (background: string | null) => string;

/** Lets the page render the canvas on demand and read it as a PNG data URL. */
export function CaptureBridge({ capture }: { capture: MutableRefObject<CaptureFn | null> }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    capture.current = (background) => {
      const old = scene.background;
      if (background) scene.background = new THREE.Color(background);
      gl.render(scene, camera);
      const url = gl.domElement.toDataURL('image/png');
      scene.background = old;
      gl.render(scene, camera);
      return url;
    };
    return () => {
      capture.current = null;
    };
  }, [gl, scene, camera, capture]);
  return null;
}
