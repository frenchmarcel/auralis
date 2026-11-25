import * as THREE from 'three';

export const COLORS = {
  accent: '#2bb3a3',
  source: '#3ecf6e',
  receiver: '#4b8df0',
  invalid: '#e5484d',
  warning: '#f0a83a',
  finding: '#ff9a3c',
  neutralLayer: '#8b949c',
  unassigned: '#3a4046',
};

/** Categorical series colours for charts (stepped for a dark surface, fixed order). */
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
export const MEAN_COLOR = '#d8dde2';

export function seriesColor(i: number): string {
  return SERIES[i % SERIES.length];
}

/** Distinct, muted colour per layer index (golden-angle hue walk). */
export function layerColor(index: number): string {
  const h = ((index * 137.508) % 360) / 360;
  const c = new THREE.Color();
  c.setHSL(h, 0.3, 0.6);
  return `#${c.getHexString()}`;
}

/** Single-hue sequential ramp (dark to light blue) for absorption 0..0.95. */
const ABS_RAMP = ['#184f95', '#256abf', '#3987e5', '#5598e7', '#86b6ef', '#b7d3f6', '#e4effc'];
export const ABS_MAX = 0.95;

export function absorptionColor(alpha: number): string {
  const t = Math.min(1, Math.max(0, alpha / ABS_MAX)) * (ABS_RAMP.length - 1);
  const i = Math.min(ABS_RAMP.length - 2, Math.floor(t));
  const a = new THREE.Color(ABS_RAMP[i]);
  const b = new THREE.Color(ABS_RAMP[i + 1]);
  a.lerp(b, t - i);
  return `#${a.getHexString()}`;
}

export function absorptionGradientCss(): string {
  return `linear-gradient(to right, ${ABS_RAMP.join(', ')})`;
}

export function mixColor(a: string, b: string, t: number): string {
  const c = new THREE.Color(a).lerp(new THREE.Color(b), t);
  return `#${c.getHexString()}`;
}
