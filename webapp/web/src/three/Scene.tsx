import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber';
import { Billboard, GizmoHelper, GizmoViewport, Html, Line, OrbitControls, TransformControls } from '@react-three/drei';
import * as THREE from 'three';
import type { MeshData, Vec3 } from '../api';
import { COLORS, mixColor } from '../lib/colors';
import type { Bounds } from '../lib/geom';
import { SURFACES, type SurfaceKind } from '../lib/surfaces';
import { SurfaceMaterial, useViewLook } from './Realistic';

/** realistic: textured surfaces by material (Realistic.tsx); shaded: flat colour per layer. */
export type Shading = 'realistic' | 'shaded' | 'wireframe' | 'ghosted';

export interface FrameRequest {
  center: Vec3;
  radius: number;
  key: number;
  /** View direction from the target (e.g. the isometric preset); default: keep the current one. */
  dir?: Vec3;
}

const noRaycast = () => null;

interface OrbitLike {
  target: THREE.Vector3;
  update: () => void;
}

export function SceneCanvas({
  bounds,
  topView = false,
  frame,
  children,
  onPointerMissed,
  className = '',
  showGrid = true,
  showGizmo = true,
  realistic = false,
  postFx = false,
  shadows = false,
}: {
  bounds: Bounds | null;
  topView?: boolean;
  frame?: FrameRequest | null;
  children?: ReactNode;
  onPointerMissed?: (e: MouseEvent) => void;
  className?: string;
  showGrid?: boolean;
  showGizmo?: boolean;
  /** The children bring their own lights (RealisticStage): no default lights. */
  realistic?: boolean;
  /** A child renders the frame at useFrame priority 1 (post-processing): the gizmo draws after it. */
  postFx?: boolean;
  /** Soft shadow maps (r3f resets the renderer's shadow flag to this prop on every render). */
  shadows?: boolean;
}) {
  return (
    <div className={`scene ${className}`}>
      <Canvas
        frameloop="demand"
        shadows={shadows ? 'soft' : false}
        orthographic={topView}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        camera={
          topView
            ? { position: [0, 0, 100], up: [0, 1, 0], zoom: 40, near: 0.01, far: 10000 }
            : { position: [10, -12, 8], up: [0, 0, 1], fov: 42, near: 0.01, far: 10000 }
        }
        onPointerMissed={onPointerMissed}
      >
        <color attach="background" args={['#111417']} />
        {!realistic && (
          <>
            <ambientLight intensity={0.5} />
            <hemisphereLight args={['#e3eaf0', '#262b30', 0.7]} />
            <directionalLight position={[1, -1.6, 2.6]} intensity={1.15} />
            <directionalLight position={[-2, 1.2, 0.8]} intensity={0.35} />
          </>
        )}
        <OrbitControls
          makeDefault
          enableDamping={false}
          enableRotate={!topView}
          screenSpacePanning
          mouseButtons={
            topView
              ? { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }
              : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }
          }
        />
        <CameraRig bounds={bounds} topView={topView} frame={frame ?? null} />
        {import.meta.env.DEV && <DevProbe />}
        <ResizeRefresh />
        {showGrid && bounds && <FloorGrid bounds={bounds} />}
        {children}
        {showGizmo && !topView && (
          <GizmoHelper alignment="bottom-right" margin={[56, 56]} renderPriority={postFx ? 2 : 1}>
            <GizmoViewport axisColors={['#e5484d', '#3ecf6e', '#4b8df0']} labelColor="#0b0d0f" />
          </GizmoHelper>
        )}
      </Canvas>
    </div>
  );
}

/**
 * With frameloop="demand", HTML labels are positioned one frame late after a
 * resize: render a few more frames once the canvas size changes.
 */
function ResizeRefresh() {
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const invalidate = useThree((s) => s.invalidate);
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
  return null;
}

/** Dev builds only: lets headless smoke tests project world points to screen pixels. */
function DevProbe() {
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const invalidate = useThree((s) => s.invalidate);
  const scene = useThree((s) => s.scene);
  useEffect(() => {
    const probe = {
      scene,
      gl,
      project(p: Vec3) {
        const v = new THREE.Vector3(p[0], p[1], p[2]).project(camera);
        const r = gl.domElement.getBoundingClientRect();
        return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height };
      },
      /** Frames rendered per second while the view is redrawn continuously for `ms`. */
      fps(ms = 3000): Promise<{ fps: number; frames: number; callsPerFrame: number; trianglesPerFrame: number }> {
        return new Promise((resolve) => {
          let frames = 0;
          const auto = gl.info.autoReset;
          gl.info.autoReset = false;
          gl.info.reset();
          const t0 = performance.now();
          const tick = () => {
            frames++;
            if (performance.now() - t0 < ms) {
              invalidate();
              requestAnimationFrame(tick);
            } else {
              const n = Math.max(1, gl.info.render.frame);
              const out = { fps: (frames * 1000) / (performance.now() - t0), frames, callsPerFrame: gl.info.render.calls / frames, trianglesPerFrame: gl.info.render.triangles / frames, renderCalls: n };
              gl.info.autoReset = auto;
              resolve(out);
            }
          };
          invalidate();
          requestAnimationFrame(tick);
        });
      },
      gizmoRadius(p: Vec3, size = 0.8) {
        if (!(camera instanceof THREE.PerspectiveCamera)) return 1;
        const d = new THREE.Vector3(p[0], p[1], p[2]).distanceTo(camera.position);
        return (d * Math.min((1.9 * Math.tan((Math.PI * camera.fov) / 360)) / camera.zoom, 7) * size) / 7;
      },
    };
    (window as unknown as { __auralisProbe?: typeof probe }).__auralisProbe = probe;
  }, [camera, gl, invalidate, scene]);
  return null;
}

function CameraRig({ bounds, topView, frame }: { bounds: Bounds | null; topView: boolean; frame: FrameRequest | null }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as OrbitLike | null;
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const invalidate = useThree((s) => s.invalidate);
  const fitted = useRef<string | null>(null);

  const fitTo = (center: Vec3, radius: number, initial: boolean, extent?: { w: number; h: number }, viewDir?: Vec3) => {
    if (!controls) return;
    const c = new THREE.Vector3(...center);
    if (topView && camera instanceof THREE.OrthographicCamera) {
      const w = Math.max(0.5, extent?.w ?? radius * 2);
      const h = Math.max(0.5, extent?.h ?? radius * 2);
      camera.zoom = Math.max(0.01, Math.min(width / (w * 1.15), height / (h * 1.15)));
      const top = (bounds ? bounds.max[2] : center[2]) + radius * 2 + 10;
      camera.position.set(c.x, c.y, top);
      camera.near = 0.01;
      camera.far = top - (bounds ? bounds.min[2] : center[2] - radius) + 100;
      camera.up.set(0, 1, 0);
      camera.lookAt(c);
      camera.updateProjectionMatrix();
    } else if (camera instanceof THREE.PerspectiveCamera) {
      const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
      if (initial || dir.lengthSq() < 1e-9) dir.set(1, -1.35, 0.95);
      if (viewDir) dir.set(viewDir[0], viewDir[1], viewDir[2]);
      dir.normalize();
      const vfov = (camera.fov * Math.PI) / 180;
      const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
      const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.05;
      camera.position.copy(c).addScaledVector(dir, dist);
      camera.near = Math.max(0.005, dist / 2000);
      camera.far = dist * 60 + radius * 10;
      camera.updateProjectionMatrix();
    }
    controls.target.copy(c);
    controls.update();
    invalidate();
  };

  useEffect(() => {
    if (!bounds || !controls || width === 0 || height === 0) return;
    const key = `${bounds.center.join(',')}|${bounds.radius}`;
    if (fitted.current === key) return;
    fitted.current = key;
    fitTo(bounds.center, bounds.radius, true, { w: bounds.size[0], h: bounds.size[1] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds, controls, width, height]);

  useEffect(() => {
    if (frame && controls) fitTo(frame.center, frame.radius, false, undefined, frame.dir);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame?.key]);

  return null;
}

function FloorGrid({ bounds }: { bounds: Bounds }) {
  const grid = useMemo(() => {
    const extent = Math.max(bounds.size[0], bounds.size[1], 2) * 1.6;
    let step = 1;
    if (extent > 200) step = 10;
    else if (extent > 60) step = 5;
    else if (extent < 4) step = 0.25;
    const size = Math.ceil(extent / step) * step;
    const g = new THREE.GridHelper(size, Math.round(size / step), '#353c43', '#22272c');
    g.rotation.x = Math.PI / 2;
    // Well below the floor: the floor's polygon offset (about one pixel of depth slope) let the lines through.
    g.position.set(bounds.center[0], bounds.center[1], bounds.min[2] - Math.max(0.05, bounds.radius * 0.01));
    (g.material as THREE.Material).transparent = true;
    (g.material as THREE.Material).opacity = 0.8;
    g.raycast = noRaycast;
    return g;
  }, [bounds]);
  useEffect(() => () => grid.dispose(), [grid]);
  return <primitive object={grid} />;
}

interface LayerGeometry {
  name: string;
  geometry: THREE.BufferGeometry;
  edges: THREE.EdgesGeometry;
}

function buildLayerGeometries(mesh: MeshData): LayerGeometry[] {
  const P = mesh.positions;
  const nVerts = Math.floor(P.length / 3);
  const remap = new Int32Array(nVerts);
  return mesh.layers.map((l) => {
    remap.fill(-1);
    const idx = new Uint32Array(l.indices.length);
    const pos: number[] = [];
    let n = 0;
    for (let i = 0; i < l.indices.length; i++) {
      const v = l.indices[i];
      let r = remap[v];
      if (r === -1) {
        r = n++;
        remap[v] = r;
        pos.push(P[3 * v], P[3 * v + 1], P[3 * v + 2]);
      }
      idx[i] = r;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    g.computeBoundingBox();
    g.computeBoundingSphere();
    const edges = new THREE.EdgesGeometry(g, 20);
    return { name: l.name, geometry: g, edges };
  });
}

export interface LayerStyle {
  color: string;
  selected?: boolean;
  hovered?: boolean;
  /** Not rendered and not pickable (eye toggled off). */
  hidden?: boolean;
  /** normal; invisible = not drawn but still pickable; overlay = translucent over a visual model. */
  display?: 'normal' | 'invisible' | 'overlay';
  /** Default true. */
  pickable?: boolean;
  /** Look in the realistic shading (default plaster). */
  surface?: SurfaceKind;
  /** Clipped by the cut-away plane (room surfaces). */
  clip?: boolean;
}

/** Return false from a click handler to let the event reach the object behind. */
export type LayerClick = (name: string, e: ThreeEvent<MouseEvent>) => boolean | void;

/** `hover`: also objects that only take the hover (userData.hoverOnly, e.g. grid points), not clicks. */
function hitsMarker(e: ThreeEvent<PointerEvent | MouseEvent>, hover = false): boolean {
  return e.intersections.some((i) => i.object.userData?.isMarker || (hover && i.object.userData?.hoverOnly));
}

export function LayerMeshes({
  mesh,
  styleOf,
  shading,
  onHover,
  onClick,
  onMove,
  pickable = true,
}: {
  mesh: MeshData;
  styleOf: (name: string, index: number) => LayerStyle;
  shading: Shading;
  onHover?: (name: string | null) => void;
  onClick?: LayerClick;
  onMove?: (name: string, e: ThreeEvent<PointerEvent>) => void;
  pickable?: boolean;
}) {
  const layers = useMemo(() => buildLayerGeometries(mesh), [mesh]);
  useEffect(
    () => () =>
      layers.forEach((l) => {
        l.geometry.dispose();
        l.edges.dispose();
      }),
    [layers],
  );
  return (
    <group>
      {layers.map((l, i) => {
        const st = styleOf(l.name, i);
        if (st.hidden) return null;
        const pick = pickable && st.pickable !== false;
        return (
          <LayerMesh
            key={l.name}
            layer={l}
            st={st}
            shading={shading}
            onHover={pick ? onHover : undefined}
            onClick={pick ? onClick : undefined}
            onMove={pick ? onMove : undefined}
          />
        );
      })}
    </group>
  );
}

function LayerMesh({
  layer,
  st,
  shading,
  onHover,
  onClick,
  onMove,
}: {
  layer: LayerGeometry;
  st: LayerStyle;
  shading: Shading;
  onHover?: (name: string | null) => void;
  onClick?: LayerClick;
  onMove?: (name: string, e: ThreeEvent<PointerEvent>) => void;
}) {
  const display = st.display ?? 'normal';
  const overlay = display === 'overlay';
  const invisible = display === 'invisible';
  const ghost = shading === 'ghosted' || overlay;
  const wire = shading === 'wireframe' && !overlay;
  const look = useViewLook();
  const clip = st.clip && look?.cut ? look.cut : null;
  const clipPlanes = useMemo(() => (clip ? [clip] : []), [clip]);
  const realistic = shading === 'realistic' && !overlay && !invisible;
  const color = st.selected
    ? mixColor(st.color, COLORS.accent, 0.72)
    : st.hovered
      ? mixColor(st.color, '#ffffff', 0.32)
      : st.color;
  const opacity = overlay
    ? st.selected
      ? 0.65
      : st.hovered
        ? 0.55
        : 0.4
    : ghost
      ? st.selected
        ? 0.5
        : st.hovered
          ? 0.3
          : 0.1
      : 1;
  const interactive = !!(onHover || onClick || onMove);
  return (
    <group>
      <mesh
        geometry={layer.geometry}
        visible={!invisible}
        raycast={interactive ? undefined : noRaycast}
        renderOrder={overlay ? 2 : 0}
        castShadow={realistic && !SURFACES[st.surface ?? 'plaster'].seeThrough}
        receiveShadow={realistic && !SURFACES[st.surface ?? 'plaster'].smooth}
        userData={{ layer: layer.name }}
        onPointerOver={
          onHover
            ? (e) => {
                if (hitsMarker(e, true)) return;
                e.stopPropagation();
                onHover(layer.name);
              }
            : undefined
        }
        onPointerOut={onHover ? () => onHover(null) : undefined}
        onPointerMove={onMove ? (e) => onMove(layer.name, e) : undefined}
        onClick={
          onClick
            ? (e) => {
                if (e.delta > 4 || hitsMarker(e)) return;
                if (onClick(layer.name, e) !== false) e.stopPropagation();
              }
            : undefined
        }
      >
        {realistic ? (
          <SurfaceMaterial kind={st.surface ?? 'plaster'} layerColor={st.color} selected={st.selected} hovered={st.hovered} clip={clip} />
        ) : (
          <meshStandardMaterial
            key={`${shading}|${display}|${clip ? 'cut' : ''}`}
            clippingPlanes={clipPlanes}
            color={color}
            emissive={st.selected ? COLORS.accent : '#000000'}
            emissiveIntensity={st.selected ? 0.22 : 0}
            flatShading
            side={THREE.DoubleSide}
            wireframe={wire}
            transparent={ghost}
            opacity={opacity}
            depthWrite={!ghost}
            roughness={0.9}
            metalness={0}
            polygonOffset
            polygonOffsetFactor={overlay ? -1 : 1}
            polygonOffsetUnits={overlay ? -1 : 1}
          />
        )}
      </mesh>
      {!wire && !invisible && (!realistic || st.selected || st.hovered) && (
        <lineSegments geometry={layer.edges} raycast={noRaycast} renderOrder={overlay ? 3 : 0}>
          <lineBasicMaterial
            key={clip ? 'cut' : ''}
            clippingPlanes={clipPlanes}
            color={
              overlay
                ? st.selected
                  ? COLORS.accent
                  : mixColor(st.color, '#ffffff', 0.45)
                : ghost
                  ? st.selected
                    ? COLORS.accent
                    : '#aab3bb'
                  : st.selected
                    ? '#7fe3d7'
                    : '#07090b'
            }
            transparent
            opacity={overlay ? 0.95 : ghost ? (st.selected ? 0.9 : 0.4) : st.selected ? 0.9 : realistic ? 0.14 : 0.5}
            depthWrite={false}
          />
        </lineSegments>
      )}
    </group>
  );
}

export interface PointMarker {
  kind: 'source' | 'receiver';
  id: string;
  label: string;
  position: Vec3;
  dir?: Vec3 | null;
  selected?: boolean;
  invalid?: boolean;
  dim?: boolean;
  badge?: string;
  /** Colour of the point's user group: drawn as a ring around the marker. */
  groupColor?: string | null;
}

export function PointMarkers({
  points,
  radius,
  onPointClick,
  gizmo,
  onMove,
  onGizmoDrag,
  showLabels = true,
}: {
  points: PointMarker[];
  radius: number;
  onPointClick?: (p: PointMarker, e: ThreeEvent<MouseEvent>) => void;
  gizmo?: { kind: 'source' | 'receiver'; id: string } | null;
  onMove?: (kind: 'source' | 'receiver', id: string, pos: Vec3, final: boolean) => void;
  onGizmoDrag?: (dragging: boolean) => void;
  showLabels?: boolean;
}) {
  const refs = useRef(new Map<string, THREE.Group>());
  const [gizmoObj, setGizmoObj] = useState<THREE.Object3D | null>(null);
  const gizmoKey = gizmo ? `${gizmo.kind}:${gizmo.id}` : null;
  useLayoutEffect(() => {
    setGizmoObj(gizmoKey ? (refs.current.get(gizmoKey) ?? null) : null);
  }, [gizmoKey, points]);

  const emit = (final: boolean) => {
    if (!gizmo || !gizmoObj || !onMove) return;
    const p = gizmoObj.position;
    onMove(gizmo.kind, gizmo.id, [p.x, p.y, p.z], final);
  };

  return (
    <group>
      {points.map((p) => (
        <Marker
          key={`${p.kind}:${p.id}`}
          p={p}
          radius={radius}
          onClick={onPointClick}
          showLabel={showLabels}
          groupRef={(g) => {
            const k = `${p.kind}:${p.id}`;
            if (g) refs.current.set(k, g);
            else refs.current.delete(k);
          }}
        />
      ))}
      {gizmoObj && gizmo && onMove && (
        <TransformControls
          object={gizmoObj}
          mode="translate"
          size={0.75}
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

function Marker({
  p,
  radius,
  onClick,
  showLabel,
  groupRef,
}: {
  p: PointMarker;
  radius: number;
  onClick?: (p: PointMarker, e: ThreeEvent<MouseEvent>) => void;
  showLabel: boolean;
  groupRef: (g: THREE.Group | null) => void;
}) {
  const base = p.kind === 'source' ? COLORS.source : COLORS.receiver;
  const color = p.invalid ? COLORS.invalid : base;
  // Read by the capture mode (marker visibility, labels drawn into the PNG).
  const capture = useMemo(
    () => ({ kind: p.kind, label: p.label, invalid: !!p.invalid, group: p.groupColor ?? null }),
    [p.kind, p.label, p.invalid, p.groupColor],
  );
  return (
    <group ref={groupRef} position={p.position} userData={{ capture }}>
      <mesh
        renderOrder={10}
        userData={{ isMarker: true }}
        onClick={
          onClick
            ? (e) => {
                if (e.delta > 4) return;
                e.stopPropagation();
                onClick(p, e);
              }
            : undefined
        }
        onPointerOver={
          onClick
            ? (e) => {
                e.stopPropagation();
                document.body.style.cursor = 'pointer';
              }
            : undefined
        }
        onPointerOut={onClick ? () => (document.body.style.cursor = '') : undefined}
      >
        <sphereGeometry args={[radius, 24, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={p.selected ? 0.6 : 0.28}
          roughness={0.45}
          transparent
          opacity={p.dim ? 0.35 : 1}
          depthTest={false}
        />
      </mesh>
      {p.selected && (
        <mesh raycast={noRaycast} renderOrder={9}>
          <sphereGeometry args={[radius * 1.7, 24, 16]} />
          <meshBasicMaterial color={color} transparent opacity={0.2} depthTest={false} depthWrite={false} />
        </mesh>
      )}
      {p.groupColor && (
        <Billboard>
          <mesh raycast={noRaycast} renderOrder={10}>
            <ringGeometry args={[radius * 1.3, radius * 1.65, 32]} />
            <meshBasicMaterial color={p.groupColor} transparent opacity={p.dim ? 0.35 : 0.95} depthTest={false} depthWrite={false} side={THREE.DoubleSide} />
          </mesh>
        </Billboard>
      )}
      {p.dir && <DirArrow dir={p.dir} radius={radius} color={color} dim={!!p.dim} />}
      {showLabel && (
        <Html position={[0, 0, radius * 2.3]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div
            className={`marker-label ${p.kind} ${p.invalid ? 'invalid' : ''} ${p.selected ? 'selected' : ''} ${p.dim ? 'dim' : ''}`}
            style={p.groupColor ? { boxShadow: `inset 4px 0 0 ${p.groupColor}, 0 1px 4px rgba(0, 0, 0, 0.5)`, paddingLeft: 8 } : undefined}
          >
            {p.label}
            {p.badge && <span className="marker-badge">{p.badge}</span>}
          </div>
        </Html>
      )}
    </group>
  );
}

function DirArrow({ dir, radius, color, dim }: { dir: Vec3; radius: number; color: string; dim: boolean }) {
  const q = useMemo(() => {
    const v = new THREE.Vector3(dir[0], dir[1], dir[2]);
    if (v.lengthSq() < 1e-12) v.set(1, 0, 0);
    return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), v.normalize());
  }, [dir]);
  const L = radius * 2.4;
  const coneH = radius * 1.1;
  return (
    <group quaternion={q}>
      <mesh position={[0, radius + L / 2, 0]} raycast={noRaycast} renderOrder={10}>
        <cylinderGeometry args={[radius * 0.13, radius * 0.13, L, 8]} />
        <meshBasicMaterial color={color} depthTest={false} transparent opacity={dim ? 0.35 : 0.95} />
      </mesh>
      <mesh position={[0, radius + L + coneH / 2, 0]} raycast={noRaycast} renderOrder={10}>
        <coneGeometry args={[radius * 0.4, coneH, 14]} />
        <meshBasicMaterial color={color} depthTest={false} transparent opacity={dim ? 0.35 : 0.95} />
      </mesh>
    </group>
  );
}

export function SegmentHighlights({ all, current }: { all: [Vec3, Vec3][]; current: [Vec3, Vec3] | null }) {
  const allGeom = useMemo(() => {
    const arr = new Float32Array(all.length * 6);
    all.forEach((s, i) => {
      arr.set(s[0], i * 6);
      arr.set(s[1], i * 6 + 3);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    return g;
  }, [all]);
  useEffect(() => () => allGeom.dispose(), [allGeom]);
  const endGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const arr = new Float32Array(6);
    if (current) {
      arr.set(current[0], 0);
      arr.set(current[1], 3);
    }
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    return g;
  }, [current]);
  useEffect(() => () => endGeom.dispose(), [endGeom]);
  return (
    <group>
      {all.length > 0 && (
        <lineSegments geometry={allGeom} raycast={noRaycast} renderOrder={5}>
          <lineBasicMaterial color={COLORS.finding} transparent opacity={0.55} depthTest={false} />
        </lineSegments>
      )}
      {current && (
        <>
          <Line points={current} color="#ffd166" lineWidth={4} depthTest={false} renderOrder={6} />
          <points geometry={endGeom} raycast={noRaycast} renderOrder={7}>
            <pointsMaterial color="#ffd166" size={9} sizeAttenuation={false} depthTest={false} transparent />
          </points>
        </>
      )}
    </group>
  );
}

/** Marker radius scaled to the model, clamped to a readable range. */
export function markerRadius(bounds: Bounds | null): number {
  if (!bounds) return 0.1;
  return Math.max(0.04, Math.min(0.25, bounds.radius * 0.018));
}
