import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useThree, type ThreeEvent } from '@react-three/fiber';
import { Html, TransformControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import type { CatalogObject, PlacedObject, Vec3 } from '../api';
import { COLORS } from '../lib/colors';

export type ObjectViewMode = 'visual' | 'acoustic' | 'both';
export type GizmoMode = 'translate' | 'rotate';

const noRaycast = () => null;
const DEG = Math.PI / 180;

/** Wrap an angle in degrees to [0, 360). */
export function normDeg(d: number): number {
  const r = ((d % 360) + 360) % 360;
  return Math.round(r * 100) / 100;
}

class VisualBoundary extends Component<{ fallback: ReactNode; children: ReactNode; onError: (msg: string) => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    this.props.onError(error instanceof Error ? error.message : String(error));
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

/**
 * glTF placed with the documented chain: offsetGroup (visual_offset) ->
 * scaleGroup (visual_scale) -> rotGroup (rotation.x = +pi/2, glTF Y-up to Z-up) -> scene.
 * The outer objectGroup (position, rotation.z, scale) is the caller's.
 */
function GltfVisual({ url, offset, vscale }: { url: string; offset: Vec3; vscale: number }) {
  const gltf = useGLTF(url, false, true);
  const scene = useMemo(() => {
    const s = gltf.scene.clone(true);
    // Shadows of the realistic view (ignored while the renderer's shadow map is off).
    s.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) o.castShadow = o.receiveShadow = true;
    });
    return s;
  }, [gltf.scene]);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => invalidate(), [scene, invalidate]);
  return (
    <group position={offset}>
      <group scale={vscale}>
        <group rotation={[Math.PI / 2, 0, 0]}>
          <primitive object={scene} />
        </group>
      </group>
    </group>
  );
}

function BoxOutline({ dims, color, opacity = 1, depthTest = false }: { dims: Vec3; color: string; opacity?: number; depthTest?: boolean }) {
  const geom = useMemo(() => {
    const b = new THREE.BoxGeometry(Math.max(dims[0], 0.01), Math.max(dims[1], 0.01), Math.max(dims[2], 0.01));
    const e = new THREE.EdgesGeometry(b);
    b.dispose();
    return e;
  }, [dims]);
  useEffect(() => () => geom.dispose(), [geom]);
  return (
    <lineSegments geometry={geom} position={[0, 0, dims[2] / 2]} raycast={noRaycast} renderOrder={8}>
      <lineBasicMaterial color={color} transparent opacity={opacity} depthTest={depthTest} />
    </lineSegments>
  );
}

function LoadingMarker({ dims }: { dims: Vec3 }) {
  return (
    <group>
      <BoxOutline dims={dims} color="#9aa3ab" opacity={0.6} />
      <Html position={[0, 0, dims[2] / 2]} center zIndexRange={[15, 0]} style={{ pointerEvents: 'none' }}>
        <div className="obj-loading" title="Loading the visual model">
          <span className="spinner" style={{ width: 14, height: 14 }} />
        </div>
      </Html>
    </group>
  );
}

function FailedMarker({ dims }: { dims: Vec3 }) {
  return (
    <group>
      <BoxOutline dims={dims} color={COLORS.warning} opacity={0.9} />
      <Html position={[0, 0, dims[2] + 0.05]} center zIndexRange={[15, 0]} style={{ pointerEvents: 'none' }}>
        <div className="obj-failed" title="The visual model could not be loaded: the acoustic proxy is shown instead">
          !
        </div>
      </Html>
    </group>
  );
}

/** Footprint centre of the objects, on the lowest base: the pivot of a group rotation. */
export function selectionCentre(objects: PlacedObject[]): Vec3 | null {
  if (!objects.length) return null;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const o of objects) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], o.position[k]);
      max[k] = Math.max(max[k], o.position[k]);
    }
  }
  return [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, min[2]];
}

/** Placement of one object emitted by the gizmo. */
export interface ObjectPlacement {
  id: string;
  position: Vec3;
  rotation_deg: number;
}

interface DragStart {
  centre: THREE.Vector3;
  rot: number;
  items: { id: string; position: Vec3; rotation_deg: number }[];
  moved: boolean;
}

/** Display-only objects (no acoustic proxy): an invisible box from the catalog dims takes the clicks in every view mode. */
function PickBox({ dims, outline, onClick }: { dims: Vec3; outline: boolean; onClick?: (e: ThreeEvent<MouseEvent>) => void }) {
  const [hover, setHover] = useState(false);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    if (!hover) return;
    document.body.style.cursor = 'pointer';
    return () => {
      document.body.style.cursor = '';
    };
  }, [hover]);
  useEffect(() => invalidate(), [hover, invalidate]);
  return (
    <>
      {onClick && (
        <mesh
          position={[0, 0, dims[2] / 2]}
          visible={false}
          onClick={onClick}
          onPointerOver={(e) => {
            if (e.intersections.some((i) => i.object.userData?.isMarker)) return;
            e.stopPropagation();
            setHover(true);
          }}
          onPointerOut={() => setHover(false)}
        >
          <boxGeometry args={[Math.max(dims[0], 0.01), Math.max(dims[1], 0.01), Math.max(dims[2], 0.01)]} />
        </mesh>
      )}
      {(outline || hover) && <BoxOutline dims={dims} color={hover ? '#d8dee4' : '#9aa3ab'} opacity={hover ? 0.7 : 0.35} depthTest />}
    </>
  );
}

export function PlacedObjects({
  objects,
  catalog,
  showVisual,
  selectedIds,
  onObjectClick,
  gizmo,
  onTransform,
  onGizmoDrag,
  onVisualError,
  failed,
  pickable = true,
}: {
  objects: PlacedObject[];
  catalog: Map<string, CatalogObject>;
  showVisual: boolean;
  selectedIds?: readonly string[];
  onObjectClick?: (id: string, e: ThreeEvent<MouseEvent>) => boolean | void;
  /** One gizmo for the whole selection: translation, or rotation about the selection centre. */
  gizmo?: { ids: readonly string[]; mode: GizmoMode } | null;
  onTransform?: (items: ObjectPlacement[], final: boolean) => void;
  onGizmoDrag?: (dragging: boolean) => void;
  onVisualError?: (catalogId: string, msg: string) => void;
  failed?: Record<string, string>;
  pickable?: boolean;
}) {
  const pivot = useMemo(() => new THREE.Object3D(), []);
  const drag = useRef<DragStart | null>(null);
  const invalidate = useThree((s) => s.invalidate);
  const selected = useMemo(() => new Set(selectedIds ?? []), [selectedIds]);
  const gizmoKey = gizmo ? gizmo.ids.join('\n') : '';
  const gizmoItems = useMemo(() => {
    const ids = new Set(gizmoKey ? gizmoKey.split('\n') : []);
    return objects.filter((o) => ids.has(o.id));
  }, [objects, gizmoKey]);

  // The pivot sits at the selection centre, unrotated, whenever no drag is going on.
  const recentre = () => {
    if (drag.current) return;
    const c = selectionCentre(gizmoItems);
    if (!c) return;
    const p = pivot.position;
    if (p.x === c[0] && p.y === c[1] && p.z === c[2] && pivot.rotation.z === 0) return;
    p.set(c[0], c[1], c[2]);
    pivot.rotation.set(0, 0, 0);
    pivot.updateMatrixWorld();
    invalidate();
  };
  useLayoutEffect(recentre);

  const placements = (): ObjectPlacement[] => {
    const d = drag.current;
    if (!d) return [];
    const dRot = pivot.rotation.z - d.rot;
    const cos = Math.cos(dRot);
    const sin = Math.sin(dRot);
    const c = d.centre;
    const t = pivot.position;
    return d.items.map((it) => {
      const vx = it.position[0] - c.x;
      const vy = it.position[1] - c.y;
      return {
        id: it.id,
        position: [t.x + cos * vx - sin * vy, t.y + sin * vx + cos * vy, it.position[2] + (t.z - c.z)],
        rotation_deg: normDeg(it.rotation_deg + dRot / DEG),
      };
    });
  };
  const begin = () => {
    drag.current = {
      centre: pivot.position.clone(),
      rot: pivot.rotation.z,
      items: gizmoItems.map((o) => ({ id: o.id, position: [...o.position], rotation_deg: o.rotation_deg })),
      moved: false,
    };
  };

  return (
    <group>
      {objects.map((o) => {
        const cat = catalog.get(o.catalog_id);
        if (!cat) return null;
        const hasVisual = cat.visual && !!cat.visual_url && !failed?.[cat.id];
        const click =
          pickable && onObjectClick
            ? (e: ThreeEvent<MouseEvent>) => {
                if (e.delta > 4) return;
                if (onObjectClick(o.id, e) !== false) e.stopPropagation();
              }
            : undefined;
        const displayOnly = cat.proxy === 'none';
        return (
          <group key={o.id} position={o.position} rotation={[0, 0, o.rotation_deg * DEG]} scale={o.scale}>
            {hasVisual && showVisual && (
              <group userData={{ objectId: o.id }} onClick={displayOnly ? undefined : click}>
                <VisualBoundary key={cat.visual_url!} fallback={<FailedMarker dims={cat.dims} />} onError={(msg) => onVisualError?.(cat.id, msg)}>
                  <Suspense fallback={<LoadingMarker dims={cat.dims} />}>
                    <GltfVisual url={cat.visual_url!} offset={cat.visual_offset} vscale={cat.visual_scale} />
                  </Suspense>
                </VisualBoundary>
              </group>
            )}
            {failed?.[cat.id] && showVisual && <FailedMarker dims={cat.dims} />}
            {displayOnly && (
              <PickBox
                dims={cat.dims}
                outline={!selected.has(o.id) && !(hasVisual && showVisual)}
                onClick={
                  click &&
                  ((e) => {
                    if (e.intersections.some((i) => i.object.userData?.isMarker)) return;
                    click(e);
                  })
                }
              />
            )}
            {selected.has(o.id) && <BoxOutline dims={cat.dims} color={COLORS.accent} />}
          </group>
        );
      })}
      <primitive object={pivot} />
      {gizmo && gizmoItems.length > 0 && onTransform && (
        <TransformControls
          object={pivot}
          mode={gizmo.mode}
          space="world"
          size={0.8}
          showX={gizmo.mode === 'translate'}
          showY={gizmo.mode === 'translate'}
          showZ
          onMouseDown={() => {
            begin();
            onGizmoDrag?.(true);
          }}
          onMouseUp={() => {
            const d = drag.current;
            if (d?.moved) onTransform(placements(), true);
            drag.current = null;
            recentre();
            onGizmoDrag?.(false);
          }}
          onObjectChange={() => {
            if (!drag.current) begin();
            drag.current!.moved = true;
            onTransform(placements(), false);
          }}
        />
      )}
    </group>
  );
}

/** Translucent footprint box following the pointer while placing an object. */
export function PlaceGhost({ dims, bus }: { dims: Vec3 | null; bus: React.MutableRefObject<(p: Vec3 | null) => void> }) {
  const ref = useRef<THREE.Group>(null);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    bus.current = (p) => {
      const g = ref.current;
      if (!g) return;
      if (p) g.position.set(p[0], p[1], p[2]);
      g.visible = !!p;
      invalidate();
    };
    return () => {
      bus.current = () => undefined;
    };
  }, [bus, invalidate]);
  if (!dims) return null;
  return (
    <group ref={ref} visible={false}>
      <mesh position={[0, 0, dims[2] / 2]} raycast={noRaycast} renderOrder={7}>
        <boxGeometry args={[Math.max(dims[0], 0.01), Math.max(dims[1], 0.01), Math.max(dims[2], 0.01)]} />
        <meshBasicMaterial color={COLORS.accent} transparent opacity={0.18} depthWrite={false} />
      </mesh>
      <BoxOutline dims={dims} color={COLORS.accent} />
    </group>
  );
}
