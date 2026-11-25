import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useThree } from '@react-three/fiber';
import { TransformControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { VisualModelTransform } from '../api';
import { DEFAULT_LIGHTING, sunDirection, type VisualLighting } from '../lib/visualLighting';
import { useViewLook } from './Realistic';

export type VisualTransform = VisualModelTransform;

export type VisualGizmoMode = 'translate' | 'rotate' | 'scale';

const DEG = Math.PI / 180;
const noRaycast = () => null;

class Boundary extends Component<{ children: ReactNode; onError: (msg: string) => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    this.props.onError(error instanceof Error ? error.message : String(error));
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

/** Shared PMREM of three's RoomEnvironment: soft image-based light for PBR glTF materials only. */
function useEnvironment() {
  const gl = useThree((s) => s.gl);
  const env = useMemo(() => {
    const pm = new THREE.PMREMGenerator(gl);
    const room = new RoomEnvironment();
    const tex = pm.fromScene(room, 0.04).texture;
    room.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
    });
    pm.dispose();
    return tex;
  }, [gl]);
  useEffect(() => () => env.dispose(), [env]);
  return env;
}

interface LightUniforms {
  aurSunDir: { value: THREE.Vector3 };
  aurSunColor: { value: THREE.Color };
  aurViewLights: { value: number };
  aurExposure: { value: number };
}

const LIGHT_DECLS = 'uniform vec3 aurSunDir;\nuniform vec3 aurSunColor;\nuniform float aurViewLights;\nuniform float aurExposure;\n';
const DIFFUSE_ANCHOR = '#if defined( RE_IndirectDiffuse )';
const SPECULAR_ANCHOR = '#if defined( RE_IndirectSpecular )';
const BEGIN = THREE.ShaderChunk.lights_fragment_begin;

/**
 * three's light accumulation with the model's own controls: the view's lights (direct and
 * ambient/hemisphere) scaled by aurViewLights, plus a sun that lights the model only (the
 * room surfaces and markers keep the view's fixed lights). Null if the chunk has changed.
 */
const PATCHED_BEGIN: string | null =
  BEGIN.includes(DIFFUSE_ANCHOR) && BEGIN.includes(SPECULAR_ANCHOR)
    ? BEGIN.replace(
        DIFFUSE_ANCHOR,
        `reflectedLight.directDiffuse *= aurViewLights;
reflectedLight.directSpecular *= aurViewLights;
#if defined( RE_Direct )
directLight.color = aurSunColor;
directLight.direction = normalize( ( viewMatrix * vec4( aurSunDir, 0.0 ) ).xyz );
directLight.visible = true;
RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
${DIFFUSE_ANCHOR}`,
      ).replace(
        SPECULAR_ANCHOR,
        `${DIFFUSE_ANCHOR}
irradiance *= aurViewLights;
#endif
${SPECULAR_ANCHOR}`,
      )
    : null;

/** True when the sun and the view-lights switch can be applied (exposure and environment always can). */
export const LIGHTING_PATCH_OK = PATCHED_BEGIN !== null;
if (!LIGHTING_PATCH_OK) console.warn('[Auralis] three.js light chunk not recognised: the sun and the view-lights switch of the visual model are disabled.');

/** Hooks the model's lighting controls into a material's shader (uniforms shared by the whole model). */
function patchLighting(m: THREE.Material, u: LightUniforms): void {
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u);
    let fs = shader.fragmentShader;
    if (PATCHED_BEGIN) fs = fs.replace('#include <lights_fragment_begin>', PATCHED_BEGIN);
    fs = fs.replace('#include <tonemapping_fragment>', 'gl_FragColor.rgb *= aurExposure;\n#include <tonemapping_fragment>');
    shader.fragmentShader = LIGHT_DECLS + fs;
  };
  m.customProgramCacheKey = () => 'auralis-visual-lighting-1';
}

function Model({ url, opacity, lighting, onLoaded }: { url: string; opacity: number; lighting: VisualLighting; onLoaded: (box: THREE.Box3) => void }) {
  const gltf = useGLTF(url, false, true);
  const env = useEnvironment();
  const invalidate = useThree((s) => s.invalidate);
  // Cut-away of the editor: the model is clipped like the room surfaces; the realistic view's own
  // environment light (Z-up HDRI) replaces the room environment while it is shown.
  const look = useViewLook();
  const clip = look?.cut ?? null;
  const viewEnv = look?.environment ?? null;
  const uniforms = useMemo<LightUniforms>(
    () => ({
      aurSunDir: { value: new THREE.Vector3(0, 0, 1) },
      aurSunColor: { value: new THREE.Color(0, 0, 0) },
      aurViewLights: { value: 1 },
      aurExposure: { value: 1 },
    }),
    [],
  );
  const scene = useMemo(() => {
    const s = gltf.scene.clone(true);
    s.traverse((o) => {
      o.raycast = noRaycast;
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = mesh.receiveShadow = true;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mesh.material = (Array.isArray(mesh.material) ? mats.map((m) => m.clone()) : mats[0].clone()) as THREE.Material;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) patchLighting(m, uniforms);
    });
    return s;
  }, [gltf.scene, uniforms]);
  // Live: uniforms and intensities only, no shader recompilation.
  useEffect(() => {
    const [x, y, z] = sunDirection(lighting);
    uniforms.aurSunDir.value.set(x, y, z);
    uniforms.aurSunColor.value.set(lighting.sun_color).multiplyScalar(lighting.sun_intensity);
    uniforms.aurViewLights.value = lighting.view_lights ? 1 : 0;
    uniforms.aurExposure.value = 2 ** lighting.exposure_ev;
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        const std = m as THREE.MeshStandardMaterial;
        if (std.isMeshStandardMaterial) std.envMapIntensity = lighting.env_intensity;
      }
    });
    invalidate();
  }, [scene, uniforms, lighting, invalidate]);
  useEffect(() => {
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        const std = m as THREE.MeshStandardMaterial;
        if ('envMap' in std && std.isMeshStandardMaterial) {
          std.envMap = viewEnv ?? env;
          std.envMapRotation.set(viewEnv ? Math.PI / 2 : 0, 0, 0);
        }
        m.transparent = opacity < 1;
        m.opacity = opacity;
        m.depthWrite = opacity >= 1;
        m.side = THREE.DoubleSide;
        m.needsUpdate = true;
      }
    });
    invalidate();
  }, [scene, env, viewEnv, opacity, invalidate]);
  useEffect(() => {
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        if (!!m.clippingPlanes?.length !== !!clip) m.needsUpdate = true;
        m.clippingPlanes = clip ? [clip] : null;
        m.clipShadows = true;
      }
    });
    invalidate();
  }, [scene, clip, invalidate]);
  useLayoutEffect(() => {
    // Box in the file's own frame (before the placement chain of the parent groups).
    scene.updateWorldMatrix(true, true);
    const base = scene.parent ? scene.parent.matrixWorld.clone().invert() : new THREE.Matrix4();
    const box = new THREE.Box3();
    const rel = new THREE.Matrix4();
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      const b = mesh.geometry.boundingBox;
      if (b) box.union(b.clone().applyMatrix4(rel.multiplyMatrices(base, mesh.matrixWorld)));
    });
    onLoaded(box);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);
  useEffect(
    () => () =>
      scene.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) m.dispose();
      }),
    [scene],
  );
  return <primitive object={scene} />;
}

/**
 * The geometry's display-only .glb, placed with the API chain: up-axis conversion
 * (Y-up glTF to Z-up), uniform scale, rotation X then Y then Z (world axes), translation.
 * With `gizmo` set, a transform gizmo edits the placement; `onChange` gets the new
 * transform (final = mouse released).
 */
export function VisualModel({
  url,
  transform,
  opacity,
  lighting = DEFAULT_LIGHTING,
  gizmo,
  onChange,
  onGizmoDrag,
  onError,
  onLoaded,
}: {
  url: string;
  transform: VisualTransform;
  opacity: number;
  /** Lighting of the model (applied live); default DEFAULT_LIGHTING. */
  lighting?: VisualLighting;
  gizmo: VisualGizmoMode | null;
  onChange?: (t: VisualTransform, final: boolean) => void;
  onGizmoDrag?: (dragging: boolean) => void;
  onError: (msg: string) => void;
  /** Bounding box of the model in its own frame, before the placement. */
  onLoaded?: (box: THREE.Box3) => void;
}) {
  const [outer, setOuter] = useState<THREE.Group | null>(null);
  // Uniform scale at the start of a gizmo drag: the dragged axis is the one that moved from it.
  const dragStart = useRef(transform.scale);
  const r = transform.rotation_deg;
  // Matrix Rz * Ry * Rx: X applied first, then Y, then Z.
  const euler = useMemo(() => new THREE.Euler(r[0] * DEG, r[1] * DEG, r[2] * DEG, 'ZYX'), [r]);

  const read = (): VisualTransform | null => {
    if (!outer) return null;
    const e = new THREE.Euler().setFromQuaternion(outer.quaternion, 'ZYX');
    const sc = outer.scale;
    // Uniform scale: keep the axis the user dragged (the one that moved most).
    const d = [sc.x, sc.y, sc.z].map((v) => Math.abs(v - dragStart.current));
    const s = d[0] >= d[1] && d[0] >= d[2] ? sc.x : d[1] >= d[2] ? sc.y : sc.z;
    const s2 = Math.min(1e4, Math.max(1e-4, Math.abs(s)));
    outer.scale.setScalar(s2);
    const round = (v: number, k = 4) => Math.round(v * 10 ** k) / 10 ** k;
    return {
      position: [round(outer.position.x), round(outer.position.y), round(outer.position.z)],
      rotation_deg: [round(e.x / DEG, 2), round(e.y / DEG, 2), round(e.z / DEG, 2)],
      scale: round(s2, 5),
      up_axis: transform.up_axis,
    };
  };

  return (
    <>
      <group ref={setOuter} position={transform.position} rotation={euler} scale={transform.scale}>
        <group rotation={transform.up_axis === 'y' ? [Math.PI / 2, 0, 0] : [0, 0, 0]}>
          <Boundary key={url} onError={onError}>
            <Suspense fallback={null}>
              <Model url={url} opacity={opacity} lighting={lighting} onLoaded={(b) => onLoaded?.(b)} />
            </Suspense>
          </Boundary>
        </group>
      </group>
      {gizmo && outer && onChange && (
        <TransformControls
          object={outer}
          mode={gizmo}
          space="world"
          size={0.9}
          onMouseDown={() => {
            dragStart.current = transform.scale;
            onGizmoDrag?.(true);
          }}
          onMouseUp={() => {
            const t = read();
            if (t) onChange(t, true);
            onGizmoDrag?.(false);
          }}
          onObjectChange={() => {
            const t = read();
            if (t) onChange(t, false);
          }}
        />
      )}
    </>
  );
}
