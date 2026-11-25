import { useEffect, useMemo, useRef } from 'react';
import { useThree, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { DISABLED_COLOR, PointIndex, rampTexture } from '../lib/grids';

/** Static part of an animated grid map: built once, then only the values change. */
interface FieldGeometry {
  geometry: THREE.BufferGeometry;
  /** Rest positions (relief displaces them along the normals). */
  base: Float32Array;
  /** Unit normal of each vertex. */
  normal: Float32Array;
  /** Value of each vertex = mean of the frame values of its sources (CSR: start[v]..start[v+1]). */
  start: Int32Array;
  src: Int32Array;
  /** Grid point of each triangle (hover / click). */
  faceToPoint: Int32Array;
}

function unit(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * One square per valid point (side = spacing, in the plane of its lattice axes).
 * `interpolate`: four triangles around the centre (own value) and the corners (mean of
 * the valid points sharing the corner), as the static grid maps draw them.
 */
function buildField(points: Vec3[], valid: boolean[], normals: Vec3[], uAxes: Vec3[], spacing: number, interpolate: boolean): FieldGeometry {
  const idx = points.flatMap((_, i) => (valid[i] ? [i] : []));
  const index = interpolate ? new PointIndex(idx.map((i) => points[i]), spacing) : null;
  const pos: number[] = [];
  const nrm: number[] = [];
  const start: number[] = [0];
  const src: number[] = [];
  const tris: number[] = [];
  const faces: number[] = [];
  const h = (spacing / 2) * (interpolate ? 1 : 0.92);
  for (const i of idx) {
    const p = points[i];
    const n = unit(normals[i] ?? [0, 0, 1]);
    const u = unit(uAxes[i] ?? [1, 0, 0]);
    const v = cross(n, u);
    const corner = (su: number, sv: number): Vec3 => [p[0] + (u[0] * su + v[0] * sv) * h, p[1] + (u[1] * su + v[1] * sv) * h, p[2] + (u[2] * su + v[2] * sv) * h];
    const corners = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    const first = pos.length / 3;
    const vertex = (q: Vec3, from: number[]) => {
      pos.push(q[0], q[1], q[2]);
      nrm.push(n[0], n[1], n[2]);
      src.push(...from);
      start.push(src.length);
    };
    if (interpolate) {
      vertex(p, [i]);
      for (const c of corners) {
        const nb = index!.within(c, 0.8 * spacing).map((k) => idx[k]);
        vertex(c, nb.length ? nb : [i]);
      }
      for (let k = 0; k < 4; k++) {
        tris.push(first, first + 1 + k, first + 1 + ((k + 1) % 4));
        faces.push(i);
      }
    } else {
      for (const c of corners) vertex(c, [i]);
      tris.push(first, first + 1, first + 2, first, first + 2, first + 3);
      faces.push(i, i);
    }
  }
  const g = new THREE.BufferGeometry();
  const base = Float32Array.from(pos);
  g.setAttribute('position', new THREE.BufferAttribute(base.slice(), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array((pos.length / 3) * 2).fill(0.5), 2));
  g.setIndex(tris);
  g.computeBoundingSphere();
  return { geometry: g, base, normal: Float32Array.from(nrm), start: Int32Array.from(start), src: Int32Array.from(src), faceToPoint: Int32Array.from(faces) };
}

function buildGrey(points: Vec3[], valid: boolean[], normals: Vec3[], uAxes: Vec3[], spacing: number): THREE.BufferGeometry {
  const pos: number[] = [];
  const h = (spacing / 2) * 0.86;
  points.forEach((p, i) => {
    if (valid[i]) return;
    const n = unit(normals[i] ?? [0, 0, 1]);
    const u = unit(uAxes[i] ?? [1, 0, 0]);
    const v = cross(n, u);
    const c = (su: number, sv: number) => [p[0] + (u[0] * su + v[0] * sv) * h, p[1] + (u[1] * su + v[1] * sv) * h, p[2] + (u[2] * su + v[2] * sv) * h];
    const q = [c(-1, -1), c(1, -1), c(1, 1), c(-1, 1)];
    for (const k of [0, 1, 2, 0, 2, 3]) pos.push(...q[k]);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeBoundingSphere();
  return g;
}

export interface FieldMapProps {
  points: Vec3[];
  valid: boolean[];
  normals: Vec3[];
  uAxes: Vec3[];
  spacing: number;
  /** Quantised frames (frames x points, row-major); value = q / 127 x scale. */
  data: Int8Array;
  nPoints: number;
  frame: number;
  scale: number;
  /** Colour limit: values in [-limit, +limit] span the ramp, beyond it they clip. */
  limit: number;
  ramp: string[];
  interpolate: boolean;
  /** Displacement along the normal at +limit (m); 0 = flat. */
  relief: number;
  onHover?: (i: number | null, x: number, y: number) => void;
  onClick?: (i: number, additive: boolean) => void;
}

/** Animated wave field on grid points: the geometry is built once, each frame only rewrites the colours (and the relief). */
export function FieldMap({ points, valid, normals, uAxes, spacing, data, nPoints, frame, scale, limit, ramp, interpolate, relief, onHover, onClick }: FieldMapProps) {
  const invalidate = useThree((s) => s.invalidate);
  const texture = useMemo(() => rampTexture(ramp), [ramp]);
  useEffect(() => () => texture.dispose(), [texture]);
  const field = useMemo(() => buildField(points, valid, normals, uAxes, spacing, interpolate), [points, valid, normals, uAxes, spacing, interpolate]);
  const grey = useMemo(() => buildGrey(points, valid, normals, uAxes, spacing), [points, valid, normals, uAxes, spacing]);
  useEffect(() => () => field.geometry.dispose(), [field]);
  useEffect(() => () => grey.dispose(), [grey]);
  const displaced = useRef<FieldGeometry | null>(null);

  useEffect(() => {
    const uv = field.geometry.getAttribute('uv') as THREE.BufferAttribute;
    const pos = field.geometry.getAttribute('position') as THREE.BufferAttribute;
    const uvA = uv.array as Float32Array;
    const posA = pos.array as Float32Array;
    const off = frame * nPoints;
    const k = scale / 127 / (limit || 1);
    const nv = field.start.length - 1;
    for (let v = 0; v < nv; v++) {
      const a = field.start[v];
      const b = field.start[v + 1];
      let s = 0;
      for (let j = a; j < b; j++) s += data[off + field.src[j]];
      const x = b > a ? (s / (b - a)) * k : 0;
      uvA[2 * v] = (x + 1) / 2;
      if (relief) {
        const d = Math.max(-1.5, Math.min(1.5, x)) * relief;
        for (let c = 0; c < 3; c++) posA[3 * v + c] = field.base[3 * v + c] + field.normal[3 * v + c] * d;
      }
    }
    uv.needsUpdate = true;
    if (relief || displaced.current === field) {
      if (!relief) posA.set(field.base);
      displaced.current = relief ? field : null;
      pos.needsUpdate = true;
      field.geometry.computeBoundingSphere();
    }
    invalidate();
  }, [field, data, nPoints, frame, scale, limit, relief, invalidate]);
  useEffect(() => invalidate(), [grey, texture, invalidate]);

  const pointOf = (e: ThreeEvent<PointerEvent | MouseEvent>, faces: Int32Array | null) => {
    if (e.faceIndex === undefined || e.faceIndex === null) return undefined;
    return faces ? faces[e.faceIndex] : undefined;
  };

  return (
    <group>
      <mesh
        geometry={field.geometry}
        renderOrder={4}
        onPointerMove={
          onHover
            ? (e) => {
                e.stopPropagation();
                onHover(pointOf(e, field.faceToPoint) ?? null, e.nativeEvent.offsetX, e.nativeEvent.offsetY);
              }
            : undefined
        }
        onPointerOut={onHover ? () => onHover(null, 0, 0) : undefined}
        onClick={
          onClick
            ? (e) => {
                if (e.delta > 4) return;
                e.stopPropagation();
                const i = pointOf(e, field.faceToPoint);
                if (i !== undefined) onClick(i, e.nativeEvent.ctrlKey || e.nativeEvent.metaKey || e.nativeEvent.shiftKey);
              }
            : undefined
        }
      >
        <meshBasicMaterial map={texture} side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
      <mesh geometry={grey} renderOrder={4}>
        <meshBasicMaterial color={DISABLED_COLOR} side={THREE.DoubleSide} toneMapped={false} transparent opacity={0.8} />
      </mesh>
    </group>
  );
}
