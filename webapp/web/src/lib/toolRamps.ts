import * as THREE from 'three';
import { BANDS_HZ, type Material } from '../api';
import { bandLabel } from './format';

/**
 * Sequential ramp for the 3D tools (reflection order, cumulative absorption):
 * light and warm for the first arrivals, cooler and darker later, all readable on
 * the dark viewport.
 */
const RAMP = ['#fff1b8', '#ffd166', '#ff9f43', '#f2683c', '#d9467a', '#ac3fb4', '#7d52d8', '#5468e0'];
const RAMP_LIN = RAMP.map((h) => new THREE.Color(h));

/** Linear-RGB colour at t in [0, 1] (for vertex colours). */
export function rampLinear(t: number, out: THREE.Color = new THREE.Color()): THREE.Color {
  const x = Math.min(1, Math.max(0, t)) * (RAMP_LIN.length - 1);
  const i = Math.min(RAMP_LIN.length - 2, Math.floor(x));
  return out.copy(RAMP_LIN[i]).lerp(RAMP_LIN[i + 1], x - i);
}

export function rampCss(t: number): string {
  return `#${rampLinear(t).getHexString()}`;
}

export function rampGradientCss(): string {
  return `linear-gradient(to right, ${RAMP.join(', ')})`;
}

/** Reflection orders 0..ORDER_MAX (the last one means "and more"). */
export const ORDER_MAX = 10;
export const orderT = (order: number, max = ORDER_MAX) => Math.min(order, max) / max;

/** Cumulative absorption scale of the particles, 0 to -ABS_DB_MIN dB. */
export const ABS_DB_MIN = 60;

/** Octaves of the tools: index -1 = 63 Hz (when the materials carry it), 0..6 = BANDS_HZ. */
export const TOOL_BANDS: { index: number; hz: number }[] = [{ index: -1, hz: 63 }, ...BANDS_HZ.map((hz, index) => ({ index, hz }))];

export function toolBandHz(index: number): number {
  return index < 0 ? 63 : BANDS_HZ[index];
}

export function toolBandLabel(index: number): string {
  return bandLabel(toolBandHz(index));
}

/**
 * Absorption of a material in a tool octave. At 63 Hz, a material without its own
 * value falls back to its 125 Hz value (`fallback: true`: the UI says so).
 */
export function alphaAt(m: Material, band: number): { alpha: number; fallback: boolean } {
  if (band < 0) {
    if (m.absorption_63 !== null && m.absorption_63 !== undefined) return { alpha: m.absorption_63, fallback: false };
    return { alpha: m.absorption[0] ?? 0, fallback: true };
  }
  return { alpha: m.absorption[band] ?? 0, fallback: false };
}

export interface LayerAlphas {
  /** α per layer index of the traced mesh. */
  alpha: Float32Array;
  /** Layers without a material (traced as rigid, α = 0). */
  unassigned: string[];
  /** Layers whose material has no 63 Hz value (125 Hz used). */
  fallback63: string[];
}

/** α of every mesh layer in one tool octave, from the simulation's assignments. */
export function layerAlphas(
  layerNames: string[],
  layers: Record<string, { material_id: string | null } | undefined>,
  materialsById: Map<string, Material>,
  band: number,
): LayerAlphas {
  const alpha = new Float32Array(layerNames.length);
  const unassigned: string[] = [];
  const fallback63: string[] = [];
  layerNames.forEach((n, i) => {
    const mid = layers[n]?.material_id;
    const m = mid ? materialsById.get(mid) : undefined;
    if (!m) {
      unassigned.push(n);
      return;
    }
    const a = alphaAt(m, band);
    alpha[i] = Math.min(1, Math.max(0, a.alpha));
    if (a.fallback) fallback63.push(n);
  });
  return { alpha, unassigned, fallback63 };
}
