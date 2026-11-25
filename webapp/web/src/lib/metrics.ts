import { BANDS_HZ, type Band63, type BandValues, type MetricKey, type PairMetrics, type ResultPair, type ScalarKey } from '../api';

export type ParamKey = MetricKey | 'STI';

export interface ParamInfo {
  key: ParamKey;
  /** Plain-text label (buttons, CSV headers). */
  label: string;
  /** Plotly label (may carry <sub>). */
  axis: string;
  unit: string;
  title: string;
  /** Levels in dB: averaged over receivers on energy. */
  energetic?: boolean;
  /** Single number (not per band). */
  scalar?: boolean;
  digits: number;
  /** Signed values: the y axis does not start at zero. */
  signed?: boolean;
  /** ISO single number = mean of 125 Hz–1 kHz. */
  isoLowMean?: boolean;
}

export const PARAMS: ParamInfo[] = [
  { key: 'T20', label: 'T20', axis: 'T<sub>20</sub>', unit: 's', title: 'Reverberation time from a 20 dB decay range', digits: 2 },
  { key: 'T30', label: 'T30', axis: 'T<sub>30</sub>', unit: 's', title: 'Reverberation time from a 30 dB decay range', digits: 2 },
  { key: 'EDT', label: 'EDT', axis: 'EDT', unit: 's', title: 'Early decay time', digits: 2 },
  { key: 'C50', label: 'C50', axis: 'C<sub>50</sub>', unit: 'dB', title: 'Clarity, 50 ms', digits: 2, signed: true },
  { key: 'C80', label: 'C80', axis: 'C<sub>80</sub>', unit: 'dB', title: 'Clarity, 80 ms', digits: 2, signed: true },
  { key: 'D50', label: 'D50', axis: 'D<sub>50</sub>', unit: '', title: 'Definition (early-to-total energy ratio, 50 ms)', digits: 3 },
  { key: 'Ts_ms', label: 'Ts', axis: 'T<sub>s</sub>', unit: 'ms', title: 'Centre time', digits: 1 },
  { key: 'SPL_db', label: 'SPL', axis: 'SPL', unit: 'dB', title: 'Sound pressure level (re 20 µPa, source 94 dB at 1 m + gain)', energetic: true, digits: 1, signed: true },
  { key: 'G_db', label: 'G', axis: 'G', unit: 'dB', title: 'Strength (re. the same source at 10 m in free field)', energetic: true, digits: 1, signed: true },
  { key: 'STI', label: 'STI', axis: 'STI', unit: '', title: 'Speech transmission index (IEC 60268-16, indirect method)', scalar: true, digits: 2 },
  { key: 'MTI', label: 'MTI', axis: 'MTI', unit: '', title: 'Modulation transfer index per octave band (the STI is their weighted sum)', digits: 2 },
  { key: 'JLF', label: 'JLF', axis: 'J<sub>LF</sub>', unit: '', title: 'Early lateral energy fraction (ISO 3382-1); ISO single number = mean of 125 Hz–1 kHz', digits: 3, isoLowMean: true },
  { key: 'LJ_db', label: 'LJ', axis: 'L<sub>J</sub>', unit: 'dB', title: 'Late lateral sound level (ISO 3382-1); ISO single number = mean of 125 Hz–1 kHz', energetic: true, digits: 1, signed: true, isoLowMean: true },
];

export function paramInfo(key: ParamKey): ParamInfo {
  return PARAMS.find((p) => p.key === key) ?? PARAMS[1];
}

export const NULL_BANDS: BandValues = BANDS_HZ.map(() => null);

const finite = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

/** Mean of the finite values (energetic for dB levels); null when none. */
export function meanScalar(values: (number | null | undefined)[], energetic = false): number | null {
  const v = values.filter(finite);
  if (!v.length) return null;
  if (energetic) return 10 * Math.log10(v.reduce((a, x) => a + 10 ** (x / 10), 0) / v.length);
  return v.reduce((a, x) => a + x, 0) / v.length;
}

/** Band-wise mean over receivers; `missing[i]` = how many receivers had no value in band i. */
export function meanBands(rows: (BandValues | undefined)[], energetic = false): { values: BandValues; missing: number[] } {
  const values = BANDS_HZ.map((_, i) => meanScalar(rows.map((r) => r?.[i]), energetic));
  const missing = BANDS_HZ.map((_, i) => rows.filter((r) => !finite(r?.[i])).length);
  return { values, missing };
}

/** ISO 3382-1 single number of J_LF / L_J: arithmetic mean of 125, 250, 500 and 1000 Hz; null if one is missing. */
export function isoLowMean(values: BandValues | undefined): number | null {
  if (!values) return null;
  const v = [0, 1, 2, 3].map((i) => values[i]);
  if (!v.every(finite)) return null;
  return (v as number[]).reduce((a, x) => a + x, 0) / 4;
}

export interface StiRating {
  label: string;
  cls: string;
}

/** IEC 60268-16 qualification bands. */
export const STI_BANDS: { min: number; max: number; label: string; cls: string; color: string }[] = [
  { min: 0, max: 0.3, label: 'bad', cls: 'sti-bad', color: '#e5484d' },
  { min: 0.3, max: 0.45, label: 'poor', cls: 'sti-poor', color: '#e8833a' },
  { min: 0.45, max: 0.6, label: 'fair', cls: 'sti-fair', color: '#d9b23a' },
  { min: 0.6, max: 0.75, label: 'good', cls: 'sti-good', color: '#6cbf5a' },
  { min: 0.75, max: 1.0001, label: 'excellent', cls: 'sti-excellent', color: '#2bb3a3' },
];

export function stiRating(sti: number | null | undefined): StiRating | null {
  if (!finite(sti)) return null;
  const b = STI_BANDS.find((x) => sti >= x.min && sti < x.max) ?? STI_BANDS[STI_BANDS.length - 1];
  return { label: b.label, cls: b.cls };
}

/** Metrics of a set of pairs (one receiver, or the average of several). */
export interface CardValues {
  bands: Partial<Record<MetricKey, BandValues>>;
  /** Receivers without a value, per metric and band (averages only). */
  missing: Partial<Record<MetricKey, number[]>>;
  scalars: Partial<Record<ScalarKey, number | null>>;
  n: number;
  /** 63 Hz octave from the wave band alone (null when no pair of the card has it). */
  band63: Band63 | null;
  /** Receivers without a 63 Hz value, per metric (averages only). */
  missing63: Partial<Record<Band63Key, number>>;
}

export type Band63Key = keyof Band63;
/** Parameters the server gives at 63 Hz (wave band only; older results lack G and SPL). */
export const BAND63_KEYS: Band63Key[] = ['T20', 'T30', 'EDT', 'C50', 'C80', 'D50', 'Ts_ms', 'G_db', 'SPL_db'];

export function hasBand63(key: string): key is Band63Key {
  return (BAND63_KEYS as string[]).includes(key);
}

const BAND_KEYS: MetricKey[] = ['T20', 'T30', 'EDT', 'C50', 'C80', 'D50', 'Ts_ms', 'SPL_db', 'G_db', 'MTI', 'STI_signal_db', 'JLF', 'LJ_db'];
const ENERGETIC = new Set<MetricKey>(['SPL_db', 'G_db', 'LJ_db', 'STI_signal_db']);

export function combinePairs(pairs: ResultPair[]): CardValues {
  const out: CardValues = { bands: {}, missing: {}, scalars: {}, n: pairs.length, band63: null, missing63: {} };
  if (!pairs.length) return out;
  const ms: PairMetrics[] = pairs.map((p) => p.metrics ?? {});
  if (ms.some((m) => m.band63)) {
    const b: Band63 = {};
    for (const k of BAND63_KEYS) {
      const vals = ms.map((m) => m.band63?.[k]);
      b[k] = meanScalar(vals, ENERGETIC.has(k as MetricKey));
      if (ms.length > 1) out.missing63[k] = vals.filter((v) => !finite(v)).length;
    }
    out.band63 = b;
  }
  for (const k of BAND_KEYS) {
    if (ms.length === 1) {
      out.bands[k] = ms[0][k] ?? NULL_BANDS;
      continue;
    }
    const r = meanBands(
      ms.map((m) => m[k]),
      ENERGETIC.has(k),
    );
    out.bands[k] = r.values;
    out.missing[k] = r.missing;
  }
  out.scalars.STI = meanScalar(ms.map((m) => m.STI));
  for (const k of ['SPL_dbz', 'SPL_dba', 'SPL_dbc'] as const) out.scalars[k] = meanScalar(ms.map((m) => m[k]), true);
  return out;
}

/** Energetic mean of dB curves sampled on the same grid (shorter curves stop contributing where they end). */
export function meanDbCurves(curves: { x: number[]; y: number[] }[]): { x: number[]; y: number[] } | null {
  if (!curves.length) return null;
  if (curves.length === 1) return curves[0];
  const ref = curves.reduce((a, c) => (c.x.length > a.x.length ? c : a), curves[0]);
  const y = ref.x.map((xv, i) => {
    let s = 0;
    let n = 0;
    for (const c of curves) {
      let v: number | undefined;
      if (c.x[i] === xv) v = c.y[i];
      else v = interp(c.x, c.y, xv);
      if (v !== undefined && Number.isFinite(v)) {
        s += 10 ** (v / 10);
        n++;
      }
    }
    return n ? 10 * Math.log10(s / n) : NaN;
  });
  return { x: ref.x, y };
}

/** Linear interpolation on a sorted grid; undefined outside it. */
function interp(xs: number[], ys: number[], x: number): number | undefined {
  const n = xs.length;
  if (!n || x < xs[0] || x > xs[n - 1]) return undefined;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  if (xs[hi] === xs[lo]) return ys[lo];
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

/** Octave-band A and C weightings (IEC 61672-1), 125 Hz–8 kHz. */
export const WEIGHTING_DB: Record<'Z' | 'A' | 'C', number[]> = {
  Z: [0, 0, 0, 0, 0, 0, 0],
  A: [-16.1, -8.6, -3.2, 0, 1.2, 1.0, -1.1],
  C: [-0.2, 0, 0, 0, -0.2, -0.8, -3.0],
};

/** Energy sum of weighted octave levels; null if every band is missing. */
export function weightedTotal(bands: BandValues, w: 'Z' | 'A' | 'C'): number | null {
  let s = 0;
  let n = 0;
  bands.forEach((v, i) => {
    if (finite(v)) {
      s += 10 ** ((v + WEIGHTING_DB[w][i]) / 10);
      n++;
    }
  });
  return n ? 10 * Math.log10(s) : null;
}
