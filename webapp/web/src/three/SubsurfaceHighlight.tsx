import { useEffect, useMemo } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { MeshData } from '../api';

const HIGHLIGHT = '#ffd23f';
const noRaycast = () => null;

/** A sub-surface to show: triangles of one layer, indexed as in GET .../mesh. */
export interface SubsurfaceFocus {
  layer: string;
  triangles: number[];
  key: string;
}

/** Positions (9 per triangle) of a sub-surface, or null when the layer is not in the mesh. */
export function subsurfacePositions(mesh: MeshData, focus: Pick<SubsurfaceFocus, 'layer' | 'triangles'>): Float32Array | null {
  const layer = mesh.layers.find((l) => l.name === focus.layer);
  if (!layer) return null;
  const P = mesh.positions;
  const out = new Float32Array(focus.triangles.length * 9);
  let k = 0;
  for (const t of focus.triangles) {
    for (let c = 0; c < 3; c++) {
      const v = layer.indices[3 * t + c];
      if (v === undefined) continue;
      out[k++] = P[3 * v];
      out[k++] = P[3 * v + 1];
      out[k++] = P[3 * v + 2];
    }
  }
  return k === out.length ? out : out.slice(0, k - (k % 9));
}

/** The focused sub-surface drawn over the layers, and faintly through what hides it. */
export function SubsurfaceHighlight({ mesh, focus }: { mesh: MeshData; focus: SubsurfaceFocus }) {
  const invalidate = useThree((s) => s.invalidate);
  const geo = useMemo(() => {
    const pos = subsurfacePositions(mesh, focus);
    if (!pos || pos.length === 0) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return { fill: g, edges: new THREE.EdgesGeometry(g, 20) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mesh, focus.key]);
  useEffect(() => {
    invalidate();
    return () => {
      geo?.fill.dispose();
      geo?.edges.dispose();
      invalidate();
    };
  }, [geo, invalidate]);
  if (!geo) return null;
  return (
    <group>
      <mesh geometry={geo.fill} raycast={noRaycast} renderOrder={10}>
        <meshBasicMaterial color={HIGHLIGHT} transparent opacity={0.6} side={THREE.DoubleSide} depthWrite={false} polygonOffset polygonOffsetFactor={-4} polygonOffsetUnits={-4} />
      </mesh>
      <mesh geometry={geo.fill} raycast={noRaycast} renderOrder={11}>
        <meshBasicMaterial color={HIGHLIGHT} transparent opacity={0.16} side={THREE.DoubleSide} depthWrite={false} depthTest={false} />
      </mesh>
      <lineSegments geometry={geo.edges} raycast={noRaycast} renderOrder={12}>
        <lineBasicMaterial color={HIGHLIGHT} transparent opacity={0.95} depthTest={false} depthWrite={false} />
      </lineSegments>
    </group>
  );
}
