import { createContext, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { N8AOPass } from 'n8ao';
import type { MeshData, TextureLibrary, TextureSetInfo } from '../api';
import { COLORS, mixColor } from '../lib/colors';
import type { Bounds } from '../lib/geom';
import { SURFACES, type SurfaceKind, type SurfaceSpec } from '../lib/surfaces';

const noRaycast = () => null;

/** What the realistic look and the cut-away share with the scene components (inside the Canvas). */
export interface ViewLookContext {
  library: TextureLibrary | null;
  /** Section plane keeping what lies below it; null = no cut-away. */
  cut: THREE.Plane | null;
  /** Image-based light of the realistic stage (Z-up rotation), once loaded; the visual model uses it too. */
  environment?: THREE.Texture | null;
  setEnvironment?: (t: THREE.Texture | null) => void;
  onWarning?: (msg: string) => void;
}

const LookContext = createContext<ViewLookContext | null>(null);
export const ViewLookProvider = LookContext.Provider;
export function useViewLook(): ViewLookContext | null {
  return useContext(LookContext);
}

// ---------- textures ----------

interface LoadedSet {
  diff: THREE.Texture;
  nor: THREE.Texture | null;
  rough: THREE.Texture | null;
  /** 1 / repeat size (m). */
  scale: THREE.Vector2;
  /** Mean linear luminance of the diffuse map (for the detail-only mode). */
  meanLum: number;
}

const setCache = new Map<string, Promise<LoadedSet>>();

function loadTexture(url: string, srgb: boolean): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(
      url,
      (t) => {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        t.anisotropy = 8;
        t.needsUpdate = true;
        resolve(t);
      },
      undefined,
      () => reject(new Error(`could not load ${url}`)),
    );
  });
}

function meanLuminance(img: CanvasImageSource): number {
  try {
    const c = document.createElement('canvas');
    c.width = c.height = 16;
    const ctx = c.getContext('2d');
    if (!ctx) return 0.5;
    ctx.drawImage(img, 0, 0, 16, 16);
    const d = ctx.getImageData(0, 0, 16, 16).data;
    const lin = (v: number) => {
      const s = v / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    };
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.2126 * lin(d[i]) + 0.7152 * lin(d[i + 1]) + 0.0722 * lin(d[i + 2]);
    return Math.max(0.02, sum / (d.length / 4));
  } catch {
    return 0.5;
  }
}

function loadTextureSet(info: TextureSetInfo): Promise<LoadedSet> {
  const key = info.maps.diff;
  let p = setCache.get(key);
  if (!p) {
    p = Promise.all([
      loadTexture(info.maps.diff, true),
      info.maps.nor ? loadTexture(info.maps.nor, false).catch(() => null) : Promise.resolve(null),
      info.maps.rough ? loadTexture(info.maps.rough, false).catch(() => null) : Promise.resolve(null),
    ]).then(([diff, nor, rough]) => ({
      diff,
      nor,
      rough,
      scale: new THREE.Vector2(1 / Math.max(0.01, info.size_m[0]), 1 / Math.max(0.01, info.size_m[1])),
      meanLum: meanLuminance(diff.image as CanvasImageSource),
    }));
    p.catch(() => setCache.delete(key));
    setCache.set(key, p);
  }
  return p;
}

function useTextureSet(info: TextureSetInfo | undefined, onWarning?: (msg: string) => void): LoadedSet | null {
  const [set, setSet] = useState<LoadedSet | null>(null);
  useEffect(() => {
    setSet(null);
    if (!info) return;
    let alive = true;
    loadTextureSet(info)
      .then((s) => alive && setSet(s))
      .catch((e) => {
        console.warn('[Auralis] texture', info.id, e);
        onWarning?.(`Texture “${info.name}” could not be loaded: plain colour shown instead.`);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info]);
  return set;
}

// ---------- surface material (world-space triplanar mapping: imported meshes have no UVs) ----------

const VERT_DECL = 'varying vec3 vAurWorld;\n';
const FRAG_DECL = `varying vec3 vAurWorld;
uniform vec2 aurScale;
uniform float aurNormalScale;
uniform float aurDetail;
uniform float aurMean;
uniform float aurGrid;
#ifdef AUR_MAP
uniform sampler2D aurMap;
#endif
#ifdef AUR_NMAP
uniform sampler2D aurNormalMap;
#endif
#ifdef AUR_RMAP
uniform sampler2D aurRoughMap;
#endif
vec3 aurWN;
vec3 aurW;
vec2 aurUvX;
vec2 aurUvY;
vec2 aurUvZ;
`;

// World Z is up: walls map (horizontal, z), floors and ceilings map (x, y).
const FRAG_SETUP = `
{
  vec3 aurVN = normalize( cross( dFdx( vViewPosition ), dFdy( vViewPosition ) ) );
  aurWN = inverseTransformDirection( aurVN, viewMatrix );
  vec3 w = pow( abs( aurWN ), vec3( 8.0 ) );
  aurW = w / max( w.x + w.y + w.z, 1e-5 );
  aurUvX = vAurWorld.yz * aurScale;
  aurUvY = vAurWorld.xz * aurScale;
  aurUvZ = vAurWorld.xy * aurScale;
}
#ifdef AUR_MAP
{
  vec4 t = texture2D( aurMap, aurUvX ) * aurW.x + texture2D( aurMap, aurUvY ) * aurW.y + texture2D( aurMap, aurUvZ ) * aurW.z;
  float lum = dot( t.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
  diffuseColor.rgb *= mix( t.rgb, vec3( lum / aurMean ), aurDetail );
}
#endif
#ifdef AUR_GRID
{
  vec2 gp = aurW.z > 0.5 ? vAurWorld.xy : ( aurW.x > aurW.y ? vAurWorld.yz : vAurWorld.xz );
  vec2 g = abs( fract( gp / aurGrid + 0.5 ) - 0.5 ) * aurGrid;
  float d = min( g.x, g.y );
  float line = 1.0 - smoothstep( 0.004, 0.004 + 1.5 * fwidth( d ), d );
  diffuseColor.rgb *= 1.0 - 0.45 * line;
}
#endif
`;

const FRAG_ROUGH = `
float roughnessFactor = roughness;
#ifdef AUR_RMAP
roughnessFactor *= texture2D( aurRoughMap, aurUvX ).g * aurW.x + texture2D( aurRoughMap, aurUvY ).g * aurW.y + texture2D( aurRoughMap, aurUvZ ).g * aurW.z;
#endif
`;

const FRAG_NORMAL = `
#ifdef AUR_NMAP
{
  vec3 s = sign( aurWN );
  vec3 tx = texture2D( aurNormalMap, aurUvX ).xyz * 2.0 - 1.0;
  vec3 ty = texture2D( aurNormalMap, aurUvY ).xyz * 2.0 - 1.0;
  vec3 tz = texture2D( aurNormalMap, aurUvZ ).xyz * 2.0 - 1.0;
  tx.xy *= aurNormalScale;
  ty.xy *= aurNormalScale;
  tz.xy *= aurNormalScale;
  vec3 wn = normalize( vec3( tx.z * s.x, tx.x, tx.y ) * aurW.x + vec3( ty.x, ty.z * s.y, ty.y ) * aurW.y + vec3( tz.x, tz.y, tz.z * s.z ) * aurW.z );
  normal = normalize( ( viewMatrix * vec4( wn, 0.0 ) ).xyz );
}
#endif
`;

interface SurfaceUniforms {
  aurScale: { value: THREE.Vector2 };
  aurNormalScale: { value: number };
  aurDetail: { value: number };
  aurMean: { value: number };
  aurGrid: { value: number };
  aurMap: { value: THREE.Texture | null };
  aurNormalMap: { value: THREE.Texture | null };
  aurRoughMap: { value: THREE.Texture | null };
}

/** Material of a room surface: PBR, flat-shaded, textures projected in world space (no UVs needed). */
function createSurfaceMaterial(spec: SurfaceSpec, set: LoadedSet | null): THREE.MeshStandardMaterial {
  const opacity = spec.opacity ?? 1;
  const m = new THREE.MeshStandardMaterial({
    flatShading: !spec.smooth,
    side: THREE.DoubleSide,
    roughness: spec.roughness,
    metalness: spec.metalness ?? 0,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 1,
    envMapIntensity: spec.envIntensity ?? 1,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const u: SurfaceUniforms = {
    aurScale: { value: spec.repeat ? new THREE.Vector2(1 / spec.repeat, 1 / spec.repeat) : set ? set.scale.clone() : new THREE.Vector2(1, 1) },
    aurNormalScale: { value: spec.normalScale ?? 1 },
    aurDetail: { value: spec.color ? (spec.detail ?? 0.7) : 0 },
    aurMean: { value: set?.meanLum ?? 0.5 },
    aurGrid: { value: spec.grid ?? 0.6 },
    aurMap: { value: set?.diff ?? null },
    aurNormalMap: { value: set?.nor ?? null },
    aurRoughMap: { value: set?.rough ?? null },
  };
  // Keep three's own defines (STANDARD) and add ours.
  const mat = m as unknown as { defines: Record<string, string> };
  const defines: Record<string, string> = { ...mat.defines };
  if (set) defines.AUR_MAP = '';
  if (set?.nor && !spec.smooth) defines.AUR_NMAP = '';
  if (set?.rough) defines.AUR_RMAP = '';
  if (spec.grid) defines.AUR_GRID = '';
  mat.defines = defines;
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u);
    shader.vertexShader = VERT_DECL + shader.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\nvAurWorld = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;');
    shader.fragmentShader =
      FRAG_DECL +
      shader.fragmentShader
        .replace('#include <map_fragment>', `#include <map_fragment>\n${FRAG_SETUP}`)
        .replace('#include <roughnessmap_fragment>', FRAG_ROUGH)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${FRAG_NORMAL}`);
  };
  m.customProgramCacheKey = () => 'auralis-surface-1';
  return m;
}

/** Room surface look: texture by material kind, a light tint of the layer colour, selection on top. */
export function SurfaceMaterial({
  kind,
  layerColor,
  selected,
  hovered,
  clip,
}: {
  kind: SurfaceKind;
  layerColor: string;
  selected?: boolean;
  hovered?: boolean;
  clip: THREE.Plane | null;
}) {
  const look = useViewLook();
  const invalidate = useThree((s) => s.invalidate);
  const spec = SURFACES[kind];
  const info = spec.texture ? look?.library?.textures[spec.texture] : undefined;
  const set = useTextureSet(info, look?.onWarning);
  const material = useMemo(() => createSurfaceMaterial(spec, set), [spec, set]);
  useEffect(() => () => material.dispose(), [material]);
  useLayoutEffect(() => {
    // Without its texture (loading, or not downloaded) a surface shows the texture's average colour.
    const base = mixColor(spec.color ?? (set ? '#ffffff' : (spec.tone ?? '#d9d6d0')), layerColor, 0.14);
    material.color.set(selected ? mixColor(base, COLORS.accent, 0.4) : hovered ? mixColor(base, '#ffffff', 0.25) : base);
    material.emissive.set(selected ? COLORS.accent : '#ffffff');
    material.emissiveIntensity = selected ? 0.32 : hovered ? 0.06 : 0;
    const planes = clip ? [clip] : null;
    if ((material.clippingPlanes?.length ?? 0) !== (planes?.length ?? 0)) material.needsUpdate = true;
    material.clippingPlanes = planes;
    material.clipShadows = true;
    invalidate();
  }, [material, spec, set, layerColor, selected, hovered, clip, invalidate]);
  return <primitive object={material} attach="material" />;
}

// ---------- lighting ----------

const envCache = new WeakMap<THREE.WebGLRenderer, Map<string, Promise<THREE.Texture>>>();

function roomEnvironment(gl: THREE.WebGLRenderer): THREE.Texture {
  const pm = new THREE.PMREMGenerator(gl);
  const room = new RoomEnvironment();
  const tex = pm.fromScene(room, 0.04).texture;
  room.traverse((o) => (o as THREE.Mesh).geometry?.dispose?.());
  pm.dispose();
  return tex;
}

function loadEnvironment(gl: THREE.WebGLRenderer, url: string | null): Promise<THREE.Texture> {
  let byUrl = envCache.get(gl);
  if (!byUrl) envCache.set(gl, (byUrl = new Map()));
  const key = url ?? 'room';
  let p = byUrl.get(key);
  if (!p) {
    p = url
      ? new HDRLoader().loadAsync(url).then((hdr) => {
          const pm = new THREE.PMREMGenerator(gl);
          const tex = pm.fromEquirectangular(hdr).texture;
          hdr.dispose();
          pm.dispose();
          return tex;
        })
      : Promise.resolve(roomEnvironment(gl));
    p.catch(() => byUrl!.delete(key));
    byUrl.set(key, p);
  }
  return p;
}

/** Materials compiled before shadows were switched on or off must be rebuilt. */
function refreshMaterials(scene: THREE.Object3D) {
  scene.traverse((o) => {
    const mat = (o as THREE.Mesh).material;
    if (!mat) return;
    for (const m of Array.isArray(mat) ? mat : [mat]) m.needsUpdate = true;
  });
}

export interface LookOptions {
  shadows: boolean;
  ao: boolean;
  bloom: boolean;
  /** Sky (open scenes) or interior light. */
  outdoor: boolean;
}

/**
 * Lighting of the realistic look: image-based light (CC0 HDRI, or three's room
 * environment), one key light with soft shadows, ambient occlusion and optional bloom.
 */
export function RealisticStage({ bounds, options }: { bounds: Bounds; options: LookOptions }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);
  const look = useViewLook();
  const light = useRef<THREE.DirectionalLight>(null);

  // The Canvas `shadows` prop switches the shadow map (SceneCanvas); materials compiled before must follow.
  useEffect(() => {
    refreshMaterials(scene);
    invalidate();
    return () => {
      refreshMaterials(scene);
      invalidate();
    };
  }, [gl, scene, options.shadows, invalidate]);

  const hdriUrl = look?.library?.hdri[options.outdoor ? 'exterior' : 'interior']?.url_file ?? null;
  useEffect(() => {
    let alive = true;
    const apply = (tex: THREE.Texture) => {
      if (!alive) return;
      scene.environment = tex;
      scene.environmentIntensity = options.outdoor ? 0.9 : 0.75;
      scene.environmentRotation.set(Math.PI / 2, 0, 0);
      look?.setEnvironment?.(tex);
      invalidate();
    };
    loadEnvironment(gl, hdriUrl)
      .then(apply)
      .catch(() => {
        look?.onWarning?.('The environment map could not be loaded: a generic room light is used instead.');
        void loadEnvironment(gl, null).then(apply);
      });
    return () => {
      alive = false;
      scene.environment = null;
      look?.setEnvironment?.(null);
      invalidate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, scene, hdriUrl, options.outdoor, invalidate]);

  // Key light from behind the model, on the left of the default view: its shadows fall towards
  // the viewer (visible in streets and cut-away rooms). Shadow box fitted to the model.
  useLayoutEffect(() => {
    const l = light.current;
    if (!l) return;
    const r = Math.max(1, bounds.radius);
    const az = (150 * Math.PI) / 180;
    const el = (48 * Math.PI) / 180;
    const c = new THREE.Vector3(...bounds.center);
    l.position.set(c.x + Math.cos(el) * Math.cos(az) * 2 * r, c.y + Math.cos(el) * Math.sin(az) * 2 * r, c.z + Math.sin(el) * 2 * r);
    l.target.position.copy(c);
    l.target.updateMatrixWorld();
    const cam = l.shadow.camera;
    cam.left = cam.bottom = -r * 1.05;
    cam.right = cam.top = r * 1.05;
    cam.near = 0.05 * r;
    cam.far = 4 * r;
    cam.updateProjectionMatrix();
    l.shadow.mapSize.set(2048, 2048);
    l.shadow.bias = -0.0004;
    l.shadow.normalBias = (4 * r) / 2048;
    l.shadow.needsUpdate = true;
    invalidate();
  }, [bounds, invalidate]);

  return (
    <>
      <hemisphereLight args={['#e6edf3', '#4a443c', 0.35]} />
      <directionalLight ref={light} color="#fff3e2" intensity={2.1} castShadow={options.shadows} />
      {options.ao || options.bloom ? <PostEffects radius={bounds.radius} ao={options.ao} bloom={options.bloom} /> : null}
    </>
  );
}

/** Linear value whose ACES tone mapping (three's fit, neutral input) gives y. */
function inverseAces(y: number, exposure: number): number {
  const f = (x: number) => {
    const v = (x * exposure) / 0.6;
    return Math.max(0, (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.432951) + 0.238081));
  };
  let lo = 0;
  let hi = 16;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) < y) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function disposeTargets(pass: object) {
  for (const v of Object.values(pass)) {
    const t = v as { isWebGLRenderTarget?: boolean; dispose?: () => void };
    if (t && t.isWebGLRenderTarget && t.dispose) t.dispose();
  }
}

/**
 * Takes over the rendering (useFrame priority 1): N8AO renders the scene with screen-space
 * ambient occlusion (from the depth of the visible, clipped surfaces), then tone mapping.
 * The view gizmo must render after it (GizmoHelper renderPriority 2).
 */
function PostEffects({ radius, ao, bloom }: { radius: number; ao: boolean; bloom: boolean }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const look = useViewLook();
  const failed = useRef(false);

  const fx = useMemo(() => {
    const composer = new EffectComposer(gl);
    const aoPass = new N8AOPass(scene, camera, 512, 512);
    const c = aoPass.configuration;
    c.gammaCorrection = false;
    c.halfRes = true;
    c.depthAwareUpsampling = true;
    c.aoRadius = Math.min(4, Math.max(0.5, radius * 0.08));
    c.distanceFalloff = 1;
    c.intensity = ao ? 3 : 0;
    aoPass.setQualityMode('Medium');
    // MSAA on the beauty render (the composer's own targets are not multisampled).
    aoPass.beautyRenderTarget.samples = 4;
    composer.addPass(aoPass);
    const bloomPass = bloom ? new UnrealBloomPass(new THREE.Vector2(512, 512), 0.22, 0.5, 0.92) : null;
    if (bloomPass) composer.addPass(bloomPass);
    composer.addPass(new OutputPass());
    return { composer, aoPass, bloomPass };
  }, [gl, scene, camera, radius, ao, bloom]);

  useEffect(() => {
    fx.composer.setPixelRatio(gl.getPixelRatio());
    fx.composer.setSize(size.width, size.height);
    invalidate();
  }, [fx, gl, size.width, size.height, invalidate]);

  // The composer tone-maps the background too: pre-compensate it so the view keeps its colour.
  useEffect(() => {
    const bg = scene.background as THREE.Color | null;
    if (!bg || !(bg as THREE.Color).isColor) return;
    const prev = bg.clone();
    const e = gl.toneMapping === THREE.ACESFilmicToneMapping ? gl.toneMappingExposure : 0;
    if (e > 0) bg.setRGB(inverseAces(prev.r, e), inverseAces(prev.g, e), inverseAces(prev.b, e));
    invalidate();
    return () => {
      bg.copy(prev);
      invalidate();
    };
  }, [scene, gl, fx, invalidate]);

  useEffect(
    () => () => {
      disposeTargets(fx.aoPass);
      fx.bloomPass?.dispose();
      fx.composer.dispose();
      invalidate();
    },
    [fx, invalidate],
  );

  useFrame(() => {
    gl.autoClear = true;
    if (!failed.current) {
      try {
        fx.composer.render();
        return;
      } catch (e) {
        failed.current = true;
        console.warn('[Auralis] post-processing failed, plain rendering instead', e);
        look?.onWarning?.('Ambient occlusion is not available with this graphics driver: plain rendering instead.');
      }
    }
    gl.setRenderTarget(null);
    gl.render(scene, camera);
  }, 1);
  return null;
}

// ---------- cut-away ----------

/** Turns on per-material clipping (off by default in three). */
export function LocalClipping() {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    const prev = gl.localClippingEnabled;
    gl.localClippingEnabled = true;
    return () => {
      gl.localClippingEnabled = prev;
    };
  }, [gl]);
  return null;
}

/** Strips of width w centred on the sections of the surfaces by the plane z = h, so cut walls read as solid. */
function buildCaps(mesh: MeshData, include: (name: string) => boolean, h: number, w: number): THREE.BufferGeometry {
  const P = mesh.positions;
  const out: number[] = [];
  const z = h + 0.002;
  const pts: number[][] = [];
  for (const l of mesh.layers) {
    if (!include(l.name)) continue;
    const I = l.indices;
    for (let t = 0; t + 2 < I.length; t += 3) {
      pts.length = 0;
      for (let e = 0; e < 3; e++) {
        const a = I[t + e];
        const b = I[t + ((e + 1) % 3)];
        let sa = P[3 * a + 2] - h;
        let sb = P[3 * b + 2] - h;
        if (Math.abs(sa) < 1e-7) sa = 1e-7;
        if (Math.abs(sb) < 1e-7) sb = 1e-7;
        if (sa * sb >= 0) continue;
        const k = sa / (sa - sb);
        pts.push([P[3 * a] + k * (P[3 * b] - P[3 * a]), P[3 * a + 1] + k * (P[3 * b + 1] - P[3 * a + 1])]);
      }
      if (pts.length !== 2) continue;
      const [p, q] = pts;
      let dx = q[0] - p[0];
      let dy = q[1] - p[1];
      const len = Math.hypot(dx, dy);
      if (len < 1e-6) continue;
      dx /= len;
      dy /= len;
      const hw = w / 2;
      const ax = p[0] - dx * hw;
      const ay = p[1] - dy * hw;
      const bx = q[0] + dx * hw;
      const by = q[1] + dy * hw;
      const nx = -dy * hw;
      const ny = dx * hw;
      out.push(ax + nx, ay + ny, z, bx + nx, by + ny, z, bx - nx, by - ny, z);
      out.push(ax + nx, ay + ny, z, bx - nx, by - ny, z, ax - nx, ay - ny, z);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  return g;
}

export function SectionCaps({
  mesh,
  include,
  height,
  width,
  color,
}: {
  mesh: MeshData;
  include: (name: string) => boolean;
  height: number;
  width: number;
  color: string;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const geom = useMemo(() => buildCaps(mesh, include, height, width), [mesh, include, height, width]);
  useEffect(() => {
    invalidate();
    return () => geom.dispose();
  }, [geom, invalidate]);
  return (
    <mesh geometry={geom} raycast={noRaycast} renderOrder={1} userData={{ sectionCap: true }}>
      <meshBasicMaterial color={color} side={THREE.DoubleSide} toneMapped={false} />
    </mesh>
  );
}
