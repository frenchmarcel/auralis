import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { CatalogObject, MeshData, PlacedObject, Vec3 } from '../../api';
import { isTypingTarget } from '../../components/ui';
import { COLORS, layerColor } from '../../lib/colors';
import { dirFromAzEl } from '../../lib/geom';
import { PlacedObjects } from '../../three/Objects';
import { LayerMeshes, PointMarkers, type PointMarker } from '../../three/Scene';
import { PITCH_LIMIT, useHead, wrapYaw, type HeadStore } from './headStore';

const DEG = Math.PI / 180;
const noRaycast = () => null;
/** Degrees of head turn per pixel of drag in the first-person view. */
const DRAG_DEG_PER_PX = 0.25;

/**
 * Arrow keys turn the head while `enabled`: left/right = yaw ±5° (Shift ±15°), up/down = pitch.
 * Left turns to the left (yaw increases counter-clockwise).
 */
export function useHeadKeys(head: HeadStore, enabled: boolean, onReset?: () => void) {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || isTypingTarget(e.target)) return;
      const step = e.shiftKey ? 15 : 5;
      const h = head.get();
      if (e.key === 'ArrowLeft') head.set({ yaw: h.yaw + step });
      else if (e.key === 'ArrowRight') head.set({ yaw: h.yaw - step });
      else if (e.key === 'ArrowUp') head.set({ pitch: h.pitch + step });
      else if (e.key === 'ArrowDown') head.set({ pitch: h.pitch - step });
      else if ((e.key === 'h' || e.key === 'H') && onReset) onReset();
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [head, enabled, onReset]);
}

/**
 * Top-view dial (same orientation as the map: +X right, +Y up): drag the head round,
 * pitch on the slider.
 */
export function HeadDial({ head, onReset, resetTitle }: { head: HeadStore; onReset: () => void; resetTitle: string }) {
  const h = useHead(head);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const R = 46;
  const setFromPointer = (e: React.PointerEvent) => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r) return;
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    if (Math.hypot(dx, dy) < 4) return;
    head.set({ yaw: Math.round(Math.atan2(-dy, dx) / DEG) });
  };
  const a = h.yaw * DEG;
  const nose: [number, number] = [Math.cos(a), -Math.sin(a)];
  const side: [number, number] = [-nose[1], nose[0]];
  const cone = (k: number) => {
    const b = a + k * 35 * DEG;
    return `${(R - 2) * Math.cos(b)},${-(R - 2) * Math.sin(b)}`;
  };
  return (
    <div className="head-dial" onPointerDown={(e) => e.stopPropagation()}>
      <svg
        ref={svgRef}
        viewBox={`${-R - 4} ${-R - 4} ${2 * R + 8} ${2 * R + 8}`}
        width={2 * R + 8}
        height={2 * R + 8}
        role="slider"
        aria-label="Head yaw (drag to turn)"
        aria-valuemin={0}
        aria-valuemax={359}
        aria-valuenow={Math.round(h.yaw)}
        tabIndex={0}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          setFromPointer(e);
        }}
        onPointerMove={(e) => dragging.current && setFromPointer(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        <circle r={R} className="head-dial-ring" />
        {[0, 90, 180, 270].map((t) => (
          <text key={t} x={(R - 9) * Math.cos(t * DEG)} y={-(R - 9) * Math.sin(t * DEG)} className="head-dial-tick" textAnchor="middle" dominantBaseline="central">
            {t === 0 ? '+X' : t === 90 ? '+Y' : t === 180 ? '−X' : '−Y'}
          </text>
        ))}
        <polygon points={`0,0 ${cone(-1)} ${cone(1)}`} className="head-dial-cone" />
        <circle r={11} className="head-dial-head" />
        <polygon points={`${nose[0] * 16},${nose[1] * 16} ${side[0] * 6 + nose[0] * 9},${side[1] * 6 + nose[1] * 9} ${-side[0] * 6 + nose[0] * 9},${-side[1] * 6 + nose[1] * 9}`} className="head-dial-nose" />
      </svg>
      <input
        className="head-pitch"
        type="range"
        min={-PITCH_LIMIT}
        max={PITCH_LIMIT}
        step={1}
        value={h.pitch}
        aria-label="Head pitch"
        title="Head pitch (elevation of the gaze): ↑/↓ · double-click: level"
        onChange={(e) => head.set({ pitch: Number(e.target.value) })}
        onDoubleClick={() => head.set({ pitch: 0 })}
      />
      <div className="head-dial-side">
        <div className="small mono">yaw {Math.round(wrapYaw(h.yaw))}°</div>
        <div className="small mono">pitch {Math.round(h.pitch)}°</div>
        <button className="btn btn-xs" onClick={onReset} title={resetTitle}>
          Reset (H)
        </button>
      </div>
    </div>
  );
}

/** Head direction as a translucent sector on the top-view map, at the receiver. */
export function HeadCone({ head, position, radius }: { head: HeadStore; position: Vec3; radius: number }) {
  const ref = useRef<THREE.Group>(null);
  const invalidate = useThree((s) => s.invalidate);
  const geom = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(0, 0);
    s.absarc(0, 0, radius, -35 * DEG, 35 * DEG, false);
    s.lineTo(0, 0);
    return new THREE.ShapeGeometry(s, 24);
  }, [radius]);
  useEffect(() => () => geom.dispose(), [geom]);
  useEffect(() => {
    const apply = () => {
      if (ref.current) ref.current.rotation.set(0, 0, head.get().yaw * DEG);
      invalidate();
    };
    apply();
    return head.subscribe(apply);
  }, [head, invalidate, geom]);
  return (
    <group ref={ref} position={position}>
      <mesh geometry={geom} raycast={noRaycast} renderOrder={8}>
        <meshBasicMaterial color={COLORS.accent} transparent opacity={0.38} depthTest={false} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function HeadCamera({ head, position }: { head: HeadStore; position: Vec3 }) {
  const camera = useThree((s) => s.camera);
  const invalidate = useThree((s) => s.invalidate);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  // On demand rendering, HTML labels lag one frame behind a resize: render a few more.
  useEffect(() => {
    let n = 0;
    let id = 0;
    const tick = () => {
      invalidate();
      if (++n < 3) id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [width, height, invalidate]);
  useEffect(() => {
    const apply = () => {
      const { yaw, pitch } = head.get();
      const d = dirFromAzEl(yaw, pitch);
      camera.up.set(0, 0, 1);
      camera.position.set(position[0], position[1], position[2]);
      camera.lookAt(position[0] + d[0], position[1] + d[1], position[2] + d[2]);
      camera.updateMatrixWorld();
      invalidate();
    };
    apply();
    return head.subscribe(apply);
  }, [head, camera, invalidate, position]);
  return null;
}

/** The room seen from the receiver; drag to look around (the world follows the pointer). */
export function FirstPersonView({
  mesh,
  objects,
  catalog,
  position,
  head,
  markers,
  markerRadius,
}: {
  mesh: MeshData;
  objects: PlacedObject[];
  catalog: Map<string, CatalogObject>;
  position: Vec3;
  head: HeadStore;
  markers: PointMarker[];
  markerRadius: number;
}) {
  const last = useRef<{ x: number; y: number } | null>(null);
  return (
    <div
      className="scene first-person"
      onPointerDown={(e) => {
        last.current = { x: e.clientX, y: e.clientY };
        (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!last.current) return;
        const dx = e.clientX - last.current.x;
        const dy = e.clientY - last.current.y;
        last.current = { x: e.clientX, y: e.clientY };
        const h = head.get();
        head.set({ yaw: h.yaw + dx * DRAG_DEG_PER_PX, pitch: h.pitch + dy * DRAG_DEG_PER_PX });
      }}
      onPointerUp={() => (last.current = null)}
      onPointerCancel={() => (last.current = null)}
    >
      <Canvas frameloop="demand" dpr={[1, 2]} gl={{ antialias: true }} camera={{ fov: 80, near: 0.02, far: 5000, up: [0, 0, 1], position }}>
        <color attach="background" args={['#111417']} />
        <ambientLight intensity={0.55} />
        <hemisphereLight args={['#e3eaf0', '#262b30', 0.7]} />
        <directionalLight position={[1, -1.6, 2.6]} intensity={1.0} />
        <directionalLight position={[-2, 1.2, 0.8]} intensity={0.35} />
        <HeadCamera head={head} position={position} />
        <LayerMeshes mesh={mesh} shading="shaded" pickable={false} styleOf={(_, i) => ({ color: layerColor(i) })} />
        <PlacedObjects objects={objects} catalog={catalog} showVisual pickable={false} />
        <PointMarkers points={markers} radius={markerRadius} />
      </Canvas>
    </div>
  );
}
