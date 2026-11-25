import type { VisualModelInfo } from '../api';

/**
 * Lighting of the display-only .glb model. Field names follow the server's `visual.lighting`
 * (snake_case), so the same object can be stored there.
 */
export interface VisualLighting {
  preset: LightingPreset;
  /** Multiplier of the environment (image-based) light, 0–3; 0.8 = the default. */
  env_intensity: number;
  /** Sun (a directional light on the model only), 0–8; 0 = no sun. */
  sun_intensity: number;
  /** Direction the sunlight comes from: azimuth CCW from +X, elevation above the horizontal. */
  sun_azimuth_deg: number;
  sun_elevation_deg: number;
  sun_color: string;
  /** The fixed lights of the 3D view also light the model. */
  view_lights: boolean;
  /** Exposure of the model, in stops (×2 per stop), applied before tone mapping. */
  exposure_ev: number;
}

export type LightingPreset = 'default' | 'daylight' | 'evening' | 'overcast' | 'studio' | 'custom';

export const LIGHTING_LIMITS = {
  env_intensity: [0, 3],
  sun_intensity: [0, 8],
  sun_azimuth_deg: [0, 360],
  sun_elevation_deg: [-10, 90],
  exposure_ev: [-3, 3],
} as const;

/** Default lighting: environment 0.8 plus the lights of the view. */
export const DEFAULT_LIGHTING: VisualLighting = {
  preset: 'default',
  env_intensity: 0.8,
  sun_intensity: 0,
  sun_azimuth_deg: 225,
  sun_elevation_deg: 45,
  sun_color: '#fff4e5',
  view_lights: true,
  exposure_ev: 0,
};

export const LIGHTING_PRESETS: { id: Exclude<LightingPreset, 'custom'>; label: string; title: string; values: Omit<VisualLighting, 'preset'> }[] = [
  { id: 'default', label: 'Default', title: 'The fixed lighting of before: soft environment plus the lights of the 3D view', values: { ...DEFAULT_LIGHTING } },
  {
    id: 'daylight',
    label: 'Daylight',
    title: 'A high sun and a bright environment; the view lights are off',
    values: { env_intensity: 1, sun_intensity: 3, sun_azimuth_deg: 135, sun_elevation_deg: 55, sun_color: '#fff4e5', view_lights: false, exposure_ev: 0 },
  },
  {
    id: 'evening',
    label: 'Evening',
    title: 'A low warm sun and a dim environment',
    values: { env_intensity: 0.3, sun_intensity: 2.5, sun_azimuth_deg: 250, sun_elevation_deg: 8, sun_color: '#ffb46b', view_lights: false, exposure_ev: 0 },
  },
  {
    id: 'overcast',
    label: 'Overcast',
    title: 'Diffuse light only, no sun',
    values: { env_intensity: 1.6, sun_intensity: 0, sun_azimuth_deg: 225, sun_elevation_deg: 45, sun_color: '#ffffff', view_lights: false, exposure_ev: 0.3 },
  },
  {
    id: 'studio',
    label: 'Studio',
    title: 'The view lights plus a key light, for a crisp look',
    values: { env_intensity: 0.8, sun_intensity: 1.5, sun_azimuth_deg: 300, sun_elevation_deg: 40, sun_color: '#ffffff', view_lights: true, exposure_ev: 0 },
  },
];

const clamp = (v: unknown, [lo, hi]: readonly [number, number], d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);

/** A stored lighting made valid (null if it is not one). */
export function sanitizeLighting(v: unknown): VisualLighting | null {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const d = DEFAULT_LIGHTING;
  const presets: LightingPreset[] = ['default', 'daylight', 'evening', 'overcast', 'studio', 'custom'];
  return {
    preset: presets.includes(o.preset as LightingPreset) ? (o.preset as LightingPreset) : 'custom',
    env_intensity: clamp(o.env_intensity, LIGHTING_LIMITS.env_intensity, d.env_intensity),
    sun_intensity: clamp(o.sun_intensity, LIGHTING_LIMITS.sun_intensity, d.sun_intensity),
    sun_azimuth_deg: clamp(o.sun_azimuth_deg, LIGHTING_LIMITS.sun_azimuth_deg, d.sun_azimuth_deg),
    sun_elevation_deg: clamp(o.sun_elevation_deg, LIGHTING_LIMITS.sun_elevation_deg, d.sun_elevation_deg),
    sun_color: typeof o.sun_color === 'string' && /^#[0-9a-f]{6}$/i.test(o.sun_color) ? o.sun_color : d.sun_color,
    view_lights: typeof o.view_lights === 'boolean' ? o.view_lights : d.view_lights,
    exposure_ev: clamp(o.exposure_ev, LIGHTING_LIMITS.exposure_ev, d.exposure_ev),
  };
}

/** World direction the sunlight comes from (Z up). */
export function sunDirection(l: VisualLighting): [number, number, number] {
  const az = (l.sun_azimuth_deg * Math.PI) / 180;
  const el = (l.sun_elevation_deg * Math.PI) / 180;
  return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
}

/**
 * Lighting kept per geometry in this browser, for a server whose PUT /visual ignores unknown
 * fields. A server that returns `visual.lighting` (null when unset) stores it, and then it is
 * read and written there.
 */
export function serverStoresLighting(info: VisualModelInfo | null | undefined): boolean {
  return !!info && 'lighting' in info;
}

const localKey = (gid: string) => `auralis.visual.lighting.${gid}`;

export function readLocalLighting(gid: string): VisualLighting | null {
  try {
    const raw = window.localStorage.getItem(localKey(gid));
    return raw ? sanitizeLighting(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

/** Returns false when the browser storage is unavailable (private window, quota). */
export function writeLocalLighting(gid: string, l: VisualLighting): boolean {
  try {
    window.localStorage.setItem(localKey(gid), JSON.stringify(l));
    return true;
  } catch {
    return false;
  }
}

/** The lighting to start from: the server's when it stores one, else this browser's, else the default. */
export function initialLighting(gid: string | null, info: VisualModelInfo | null | undefined): VisualLighting {
  const srv = serverStoresLighting(info) ? sanitizeLighting((info as unknown as { lighting: unknown }).lighting) : null;
  return srv ?? (gid ? readLocalLighting(gid) : null) ?? DEFAULT_LIGHTING;
}
