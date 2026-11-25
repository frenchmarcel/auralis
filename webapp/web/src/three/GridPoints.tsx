import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import { useThree, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { Vec3 } from '../api';

export interface GridPointsDraw {
  id: string;
  points: Vec3[];
  valid: boolean[];
  selected: boolean;
}

export interface GridPointHover {
  gridId: string;
  index: number;
  /** Pixel position in the canvas. */
  x: number;
  y: number;
}

const VALID = new THREE.Color('#2bb3a3');
const VALID_SEL = new THREE.Color('#7fe3d7');
const DISABLED = new THREE.Color('#6e7881');

/** Receiver-grid preview: one small sphere per point (valid: teal; disabled: grey). */
export function GridPointsPreview({ grids, radius, onHover }: { grids: GridPointsDraw[]; radius: number; onHover?: (h: GridPointHover | null) => void }) {
  return (
    <group>
      {grids.map((g) => (
        <GridPointCloud key={g.id} g={g} radius={g.selected ? radius * 1.35 : radius} onHover={onHover} />
      ))}
    </group>
  );
}

function GridPointCloud({ g, radius, onHover }: { g: GridPointsDraw; radius: number; onHover?: (h: GridPointHover | null) => void }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const invalidate = useThree((s) => s.invalidate);
  const geom = useMemo(() => new THREE.IcosahedronGeometry(1, 1), []);
  useEffect(() => () => geom.dispose(), [geom]);
  const n = g.points.length;
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    const mat = new THREE.Matrix4();
    const s = new THREE.Vector3(radius, radius, radius);
    const q = new THREE.Quaternion();
    g.points.forEach((p, i) => {
      mat.compose(new THREE.Vector3(p[0], p[1], p[2]), q, g.valid[i] ? s : s.clone().multiplyScalar(0.8));
      m.setMatrixAt(i, mat);
      m.setColorAt(i, g.valid[i] ? (g.selected ? VALID_SEL : VALID) : DISABLED);
    });
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
    invalidate();
  }, [g, radius, invalidate]);
  if (!n) return null;
  const hover = (e: ThreeEvent<PointerEvent>) => {
    if (e.instanceId === undefined) return;
    onHover?.({ gridId: g.id, index: e.instanceId, x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY });
  };
  return (
    <instancedMesh
      key={n}
      ref={ref}
      args={[geom, undefined, n]}
      renderOrder={8}
      userData={{ hoverOnly: true }}
      onPointerMove={onHover ? hover : undefined}
      onPointerOut={onHover ? () => onHover(null) : undefined}
    >
      <meshBasicMaterial transparent opacity={g.selected ? 0.95 : 0.75} depthTest={false} depthWrite={false} toneMapped={false} />
    </instancedMesh>
  );
}
