import { useEffect, useState } from 'react';
import { api, BANDS_HZ, errorMessage, type GroundBody } from '../../api';
import { ErrorBox, NumField, Spinner } from '../../components/ui';
import { bandLabel, fmtNum } from '../../lib/format';
import { AlphaChart } from './MaterialLibrary';

export const DEFAULT_GROUND: GroundBody = { ground_class: 'C', alpha_e: null, roughness: 'N' };

/** Nord2000 impedance classes: effective flow resistivity (kPa·s/m²) and ground. */
const CLASSES: [string, number, string][] = [
  ['A', 12.5, 'Very soft (snow, moss)'],
  ['B', 31.5, 'Soft forest floor'],
  ['C', 80, 'Loose ground (turf, grass, loose soil)'],
  ['D', 200, 'Normal ground (forest floor, pasture)'],
  ['E', 500, 'Compacted field, gravel, park'],
  ['F', 2000, 'Compacted dense ground (gravel road)'],
  ['G', 20000, 'Hard surface (asphalt, concrete, water)'],
];

/** Nord2000 roughness classes: standard deviation of the height variations (m). */
const ROUGHNESS: [string, string][] = [
  ['N', 'Nil (flat, under ±0.25 m)'],
  ['S', 'Small (0.25 m, ±0.5 m)'],
  ['M', 'Medium (0.5 m, ±1 m)'],
  ['L', 'Large (1 m, ±2 m)'],
];

/**
 * Outdoor ground: a Nord2000 class or a flow resistivity, an optional porosity rate (two-parameter
 * model) and a roughness class, with the random-incidence α it gives. The preview is reported
 * upward so the dialog can enable "Create".
 */
export function GroundBuilder({ value, onChange, onPreview }: { value: GroundBody; onChange: (v: GroundBody) => void; onPreview: (alpha: number[] | null) => void }) {
  const [preview, setPreview] = useState<number[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<GroundBody>) => onChange({ ...value, ...patch });
  const custom = !value.ground_class;

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    const t = window.setTimeout(async () => {
      try {
        const r = await api.groundPreview(value);
        if (!cancelled) {
          setPreview(r.absorption);
          setError(null);
          onPreview(r.absorption);
        }
      } catch (e) {
        if (!cancelled) {
          setError(errorMessage(e));
          setPreview(null);
          onPreview(null);
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <div className="new-mat-grid">
      <div>
        <div className="section-title">Ground (semi-infinite, locally reacting)</div>
        <div className="form-grid">
          <label>Ground class</label>
          <select value={value.ground_class ?? ''} onChange={(e) => set(e.target.value ? { ground_class: e.target.value, sigma: undefined } : { ground_class: undefined, sigma: value.sigma ?? 200000 })}>
            {CLASSES.map(([k, s, d]) => (
              <option key={k} value={k}>
                {k}: {d}, {fmtNum(s, s < 100 ? 1 : 0)} kPa·s/m²
              </option>
            ))}
            <option value="">Custom flow resistivity</option>
          </select>
          {custom && (
            <>
              <label>Flow resistivity</label>
              <div className="row gap">
                <NumField value={(value.sigma ?? 200000) / 1000} min={1} max={200000} step={10} digits={1} onChange={(v) => set({ sigma: (v ?? 200) * 1000 })} width={110} ariaLabel="Effective flow resistivity (kPa·s/m²)" />
                <span className="muted small">kPa·s/m²</span>
              </div>
            </>
          )}
          <label title="Two-parameter model: the rate at which the porosity falls with depth. Empty: Delany-Bazley (one parameter, the Nord2000 classes).">Porosity rate (optional)</label>
          <div className="row gap">
            <NumField value={value.alpha_e ?? null} allowNull min={0} max={500} step={1} digits={1} onChange={(v) => set({ alpha_e: v })} width={110} ariaLabel="Rate of porosity change with depth (1/m)" />
            <span className="muted small">1/m</span>
          </div>
          <label>Roughness</label>
          <select value={value.roughness ?? 'N'} onChange={(e) => set({ roughness: e.target.value })}>
            {ROUGHNESS.map(([k, d]) => (
              <option key={k} value={k}>
                {k}: {d}
              </option>
            ))}
          </select>
        </div>
        <p className="muted small">
          {value.alpha_e == null ? 'Delany-Bazley with the effective flow resistivity' : 'Two-parameter variable porosity model'}. In an open scene the image sources use this model with the spherical-wave coefficient of a point source. A rough ground sends part of each reflection to the diffuse field (Nord2000). α is the random-incidence result, for the rays.
        </p>
      </div>
      <div>
        {error ? (
          <ErrorBox message={error} />
        ) : preview ? (
          <>
            <AlphaChart values={preview} height={190} />
            <table className="table compact">
              <tbody>
                <tr>
                  {BANDS_HZ.map((f) => (
                    <th key={f} className="num">
                      {bandLabel(f)}
                    </th>
                  ))}
                </tr>
                <tr>
                  {preview.map((v, i) => (
                    <td key={i} className="num">
                      {fmtNum(v, 2)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </>
        ) : (
          <div className="loading-row">
            <Spinner /> Computing…
          </div>
        )}
        {busy && preview && <div className="muted small">Updating…</div>}
      </div>
    </div>
  );
}
