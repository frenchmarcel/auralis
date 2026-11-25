import { useLayoutEffect, useMemo, useRef, useState, type ComponentProps } from 'react';
import { Html, Line, TransformControls } from '@react-three/drei';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { COLORS } from '../lib/colors';
import { useTime, type TimeStore } from '../lib/timeline';
import { forwardAt, positionAt, type Movable } from '../lib/trajectory';
import { PointMarkers, type PointMarker } from './Scene';

const noRaycast = () => null;
const UP = new THREE.Vector3(0, 1, 0);

export interface PathDraw {
  kind: 'source' | 'receiver';
  id: string;
  waypoints: Vec3[];
  /** Smooth paths: points along the true curve (waypoints included), drawn instead of the polyline. */
  curve?: Vec3[] | null;
  invalid?: boolean;
  selected?: boolean;
  dim?: boolean;
}

export interface WaypointRef {
  kind: 'source' | 'receiver';
  id: string;
  index: number;
}

/** Trajectories as polylines (or smooth curves) with clickable waypoints; the selected waypoint can carry the gizmo. */
export function PathLines({
  paths,
  radius,
  selected = null,
  onWaypointClick,
  gizmo = false,
  onWaypointMove,
  onGizmoDrag,
}: {
  paths: PathDraw[];
  radius: number;
  selected?: WaypointRef | null;
  onWaypointClick?: (w: WaypointRef) => void;
  gizmo?: boolean;
  onWaypointMove?: (w: WaypointRef, pos: Vec3, final: boolean) => void;
  onGizmoDrag?: (dragging: boolean) => void;
}) {
  const refs = useRef(new Map<string, THREE.Object3D>());
  const [gizmoObj, setGizmoObj] = useState<THREE.Object3D | null>(null);
  const selKey = selected ? `${selected.kind}:${selected.id}:${selected.index}` : null;
  useLayoutEffect(() => {
    setGizmoObj(gizmo && selKey ? (refs.current.get(selKey) ?? null) : null);
  }, [gizmo, selKey, paths]);

  const emit = (final: boolean) => {
    if (!selected || !gizmoObj || !onWaypointMove) return;
    const p = gizmoObj.position;
    onWaypointMove(selected, [p.x, p.y, p.z], final);
  };

  return (
    <group>
      {paths.map((p) => (
        <PathLine
          key={`${p.kind}:${p.id}`}
          p={p}
          radius={radius}
          selectedIndex={selected && selected.kind === p.kind && selected.id === p.id ? selected.index : null}
          onWaypointClick={onWaypointClick}
          setRef={(i, o) => {
            const k = `${p.kind}:${p.id}:${i}`;
            if (o) refs.current.set(k, o);
            else refs.current.delete(k);
          }}
        />
      ))}
      {gizmoObj && selected && onWaypointMove && (
        <TransformControls
          object={gizmoObj}
          mode="translate"
          size={0.6}
          onMouseDown={() => onGizmoDrag?.(true)}
          onMouseUp={() => {
            emit(true);
            onGizmoDrag?.(false);
          }}
          onObjectChange={() => emit(false)}
        />
      )}
    </group>
  );
}

function PathLine({
  p,
  radius,
  selectedIndex,
  onWaypointClick,
  setRef,
}: {
  p: PathDraw;
  radius: number;
  selectedIndex: number | null;
  onWaypointClick?: (w: WaypointRef) => void;
  setRef: (i: number, o: THREE.Object3D | null) => void;
}) {
  const color = p.invalid ? COLORS.invalid : p.kind === 'source' ? COLORS.source : COLORS.receiver;
  const opacity = p.dim ? 0.35 : p.selected ? 1 : 0.7;
  const line = p.curve && p.curve.length >= 2 ? p.curve : p.waypoints;
  // Small cones at segment midpoints show the direction of travel.
  const arrows = useMemo(() => (p.curve && p.curve.length >= 2 ? curveArrows(p.curve, p.waypoints, radius) : polylineArrows(p.waypoints, radius)), [p.curve, p.waypoints, radius]);
  return (
    <group>
      <Line points={line} color={color} lineWidth={p.selected ? 3 : 1.6} transparent opacity={opacity} depthTest={false} renderOrder={6} raycast={noRaycast} />
      {arrows.map((a, i) => (
        <mesh key={`a${i}`} position={a.pos} quaternion={a.q} raycast={noRaycast} renderOrder={6}>
          <coneGeometry args={[radius * 0.22, radius * 0.6, 10]} />
          <meshBasicMaterial color={color} depthTest={false} transparent opacity={opacity} />
        </mesh>
      ))}
      {p.waypoints.map((w, i) => {
        const sel = selectedIndex === i;
        return (
          <mesh
            key={`w${i}`}
            ref={(m) => setRef(i, m)}
            position={w}
            renderOrder={7}
            userData={{ isMarker: true }}
            raycast={onWaypointClick ? undefined : noRaycast}
            onClick={
              onWaypointClick
                ? (e) => {
                    if (e.delta > 4) return;
                    e.stopPropagation();
                    onWaypointClick({ kind: p.kind, id: p.id, index: i });
                  }
                : undefined
            }
            onPointerOver={
              onWaypointClick
                ? (e) => {
                    e.stopPropagation();
                    document.body.style.cursor = 'pointer';
                  }
                : undefined
            }
            onPointerOut={onWaypointClick ? () => (document.body.style.cursor = '') : undefined}
          >
            <sphereGeometry args={[radius * (sel ? 0.55 : 0.38), 16, 12]} />
            <meshBasicMaterial color={sel ? '#ffffff' : color} depthTest={false} transparent opacity={opacity} />
          </mesh>
        );
      })}
      {p.selected &&
        p.waypoints.map((w, i) => (
          <Html key={`l${i}`} position={[w[0], w[1], w[2] + radius * 1.1]} center zIndexRange={[18, 0]} style={{ pointerEvents: 'none' }}>
            <div className={`wp-label ${p.kind} ${selectedIndex === i ? 'selected' : ''}`}>{i + 1}</div>
          </Html>
        ))}
    </group>
  );
}

interface Arrow {
  pos: Vec3;
  q: THREE.Quaternion;
}

function arrowAt(pos: Vec3, a: Vec3, b: Vec3): Arrow {
  const d = new THREE.Vector3(b[0] - a[0], b[1] - a[1], b[2] - a[2]).normalize();
  return { pos, q: new THREE.Quaternion().setFromUnitVectors(UP, d) };
}

function polylineArrows(w: Vec3[], radius: number): Arrow[] {
  const out: Arrow[] = [];
  for (let i = 1; i < w.length; i++) {
    const a = w[i - 1];
    const b = w[i];
    if (Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) < radius * 3) continue;
    out.push(arrowAt([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], a, b));
  }
  return out;
}

const near = (a: Vec3, b: Vec3, tol: number) => Math.abs(a[0] - b[0]) <= tol && Math.abs(a[1] - b[1]) <= tol && Math.abs(a[2] - b[2]) <= tol;

/** One cone per waypoint segment, halfway along the curve between the two waypoints (the curve passes through them in order). */
function curveArrows(curve: Vec3[], w: Vec3[], radius: number): Arrow[] {
  const out: Arrow[] = [];
  const seg = (j: number) => Math.hypot(curve[j + 1][0] - curve[j][0], curve[j + 1][1] - curve[j][1], curve[j + 1][2] - curve[j][2]);
  let start = 0;
  for (let i = 1; i < w.length; i++) {
    let end = -1;
    for (let j = start + 1; j < curve.length; j++) {
      if (near(curve[j], w[i], 2e-3)) {
        end = j;
        break;
      }
    }
    if (end < 0) break;
    let len = 0;
    for (let j = start; j < end; j++) len += seg(j);
    if (len >= radius * 3) {
      let acc = 0;
      for (let j = start; j < end; j++) {
        const d = seg(j);
        if (acc + d >= len / 2 && d > 1e-9) {
          const f = (len / 2 - acc) / d;
          const a = curve[j];
          const b = curve[j + 1];
          out.push(arrowAt([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f], a, b));
          break;
        }
        acc += d;
      }
    }
    start = end;
  }
  return out;
}

/** PointMarkers whose moving points follow their trajectory at the store's time. */
export function TimedPointMarkers({
  time,
  items,
  points,
  ...rest
}: Omit<ComponentProps<typeof PointMarkers>, 'points'> & { time: TimeStore; items: Map<string, Movable>; points: PointMarker[] }) {
  const t = useTime(time);
  const shown = useMemo(
    () =>
      points.map((p) => {
        const it = items.get(p.id);
        if (!it?.path) return p;
        return { ...p, position: positionAt(it, t), dir: p.dir === null ? null : forwardAt(it, t, items) };
      }),
    [points, items, t],
  );
  return <PointMarkers points={shown} {...rest} />;
}
