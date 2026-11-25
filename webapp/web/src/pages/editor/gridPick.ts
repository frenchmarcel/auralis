import type { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { Vec3 } from '../../api';
import { roundVec } from '../../lib/geom';

/** A plane-grid centre clicked on a wall or a ceiling is set back this far from it (m). */
const WALL_STEP_M = 0.5;
const DEFAULT_HEIGHT_M = 1.2;

function roomMeshes(e: ThreeEvent<MouseEvent>, isRoom: (layer: string) => boolean): THREE.Object3D[] {
  let root: THREE.Object3D = e.object;
  while (root.parent) root = root.parent;
  const out: THREE.Object3D[] = [];
  root.traverse((o) => {
    const ln = o.userData?.layer as string | undefined;
    if (ln && (o as THREE.Mesh).isMesh && isRoom(ln)) out.push(o);
  });
  return out;
}

function floorBelow(meshes: THREE.Object3D[], p: Vec3): number | null {
  const ray = new THREE.Raycaster(new THREE.Vector3(p[0], p[1], p[2] + 0.01), new THREE.Vector3(0, 0, -1));
  for (const h of ray.intersectObjects(meshes, false)) {
    if (!h.face) continue;
    const n = h.face.normal.clone().transformDirection(h.object.matrixWorld);
    if (Math.abs(n.z) > 0.3) return h.point.z;
  }
  return null;
}

/**
 * Centre of a plane grid from a click: on a floor, at the grid's current height above
 * its floor (1.2 m by default); on a wall or a ceiling, stepped back from it. Seen from
 * outside a closed room the roof comes first: the first floor under a ceiling wins.
 */
export function gridCenterFromClick(e: ThreeEvent<MouseEvent>, isRoom: (layer: string) => boolean, current: Vec3 | undefined): Vec3 | null {
  const meshes = roomMeshes(e, isRoom);
  const facing = (i: THREE.Intersection) => {
    const n = i.face ? i.face.normal.clone().transformDirection(i.object.matrixWorld) : new THREE.Vector3(0, 0, 1);
    return n.dot(e.ray.direction) > 0 ? n.negate() : n;
  };
  const hits = e.intersections.filter((i) => {
    const ln = i.object.userData?.layer as string | undefined;
    return !!ln && isRoom(ln);
  });
  const covered = (p: THREE.Vector3) => new THREE.Raycaster(new THREE.Vector3(p.x, p.y, p.z + 0.02), new THREE.Vector3(0, 0, 1)).intersectObjects(meshes, false).length > 0;
  const hit = hits.find((i) => facing(i).z > 0.7 && covered(i.point)) ?? hits[0];
  if (!hit) return null;
  const n = facing(hit);
  if (n.z > 0.7) {
    let h = DEFAULT_HEIGHT_M;
    if (current) {
      const f = floorBelow(meshes, current);
      if (f !== null && current[2] - f > 0.05) h = current[2] - f;
    }
    return roundVec([hit.point.x, hit.point.y, hit.point.z + h], 2);
  }
  return roundVec([hit.point.x + n.x * WALL_STEP_M, hit.point.y + n.y * WALL_STEP_M, hit.point.z + n.z * WALL_STEP_M], 2);
}
