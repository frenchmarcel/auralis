import { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { COLORS } from '../lib/colors';

const noRaycast = () => null;

function trianglesGeometry(tris: [Vec3, Vec3, Vec3][]): THREE.BufferGeometry {
  const arr = new Float32Array(tris.length * 9);
  tris.forEach((t, i) => {
    arr.set(t[0], i * 9);
    arr.set(t[1], i * 9 + 3);
    arr.set(t[2], i * 9 + 6);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  return g;
}

/** Filled triangles of findings (overlapping faces): all of a kind faint, the current one bright. */
export function TriangleHighlights({ all, current }: { all: [Vec3, Vec3, Vec3][]; current: [Vec3, Vec3, Vec3][] }) {
  const allGeom = useMemo(() => trianglesGeometry(all), [all]);
  const curGeom = useMemo(() => trianglesGeometry(current), [current]);
  useEffect(() => () => allGeom.dispose(), [allGeom]);
  useEffect(() => () => curGeom.dispose(), [curGeom]);
  return (
    <group>
      {all.length > 0 && (
        <mesh geometry={allGeom} raycast={noRaycast} renderOrder={5}>
          <meshBasicMaterial color={COLORS.finding} transparent opacity={0.25} side={THREE.DoubleSide} depthTest={false} />
        </mesh>
      )}
      {current.length > 0 && (
        <mesh geometry={curGeom} raycast={noRaycast} renderOrder={6}>
          <meshBasicMaterial color="#ffd166" transparent opacity={0.6} side={THREE.DoubleSide} depthTest={false} />
        </mesh>
      )}
    </group>
  );
}
