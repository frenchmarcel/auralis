import { useEffect, useState } from 'react';
import { api, BANDS_HZ, errorMessage, type PerforatedBody, type PerforatedKind } from '../../api';
import { ErrorBox, NumField, Spinner } from '../../components/ui';
import { ABS_MAX } from '../../lib/colors';
import { bandLabel, fmtNum } from '../../lib/format';
import { AlphaChart } from './MaterialLibrary';

export const DEFAULT_PERFORATED: PerforatedBody = {
  kind: 'perforated',
  size_mm: 8,
  t_mm: 12.5,
  open_pct: 12,
  plenum_mm: 200,
  wool_mm: 40,
  wool_sigma: 10000,
  fleece_rayl: 0,
};

const KIND_LABELS: Record<PerforatedKind, string> = {
  perforated: 'Round holes',
  slotted: 'Slots (grooves)',
  mpp: 'Micro-perforated (holes < 1 mm)',
};

/**
 * Perforated, slotted or micro-perforated sheet over a plenum with optional fleece and wool:
 * the parameters, a schematic of the stack and the random-incidence α it gives. The preview is
 * reported upward so the dialog can enable "Create".
 */
export function PerforatedBuilder({ value, onChange, onPreview }: { value: PerforatedBody; onChange: (v: PerforatedBody) => void; onPreview: (alpha: number[] | null) => void }) {
  const [preview, setPreview] = useState<number[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<PerforatedBody>) => onChange({ ...value, ...patch });

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    const t = window.setTimeout(async () => {
      try {
        const r = await api.perforatedPreview(value);
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

  const slotted = value.kind === 'slotted';
  const air = Math.max(value.plenum_mm - (value.wool_mm ?? 0), 0);
  return (
    <div className="new-mat-grid">
      <div>
        <div className="section-title">Sheet over a plenum, rigid backing</div>
        <div className="form-grid">
          <label>Type</label>
          <select value={value.kind} onChange={(e) => set({ kind: e.target.value as PerforatedKind })}>
            {(Object.keys(KIND_LABELS) as PerforatedKind[]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k]}
              </option>
            ))}
          </select>
          <label>{slotted ? 'Slot width' : 'Hole diameter'}</label>
          <div className="row gap">
            <NumField value={value.size_mm} min={0.05} max={50} step={value.kind === 'mpp' ? 0.05 : 0.5} digits={2} onChange={(v) => set({ size_mm: v ?? 8 })} width={110} />
            <span className="muted small">mm</span>
          </div>
          <label>Sheet thickness</label>
          <div className="row gap">
            <NumField value={value.t_mm} min={0.1} max={50} step={0.5} digits={1} onChange={(v) => set({ t_mm: v ?? 12.5 })} width={110} />
            <span className="muted small">mm</span>
          </div>
          <label title="Share of the sheet that is open (holes or slots), as published by manufacturers">Open area</label>
          <div className="row gap">
            <NumField value={value.open_pct} min={0.1} max={60} step={1} digits={1} onChange={(v) => set({ open_pct: v ?? 12 })} width={110} />
            <span className="muted small">%</span>
          </div>
          <label title="Acoustic fleece glued behind the sheet; 0 = none">Fleece flow resistance</label>
          <div className="row gap">
            <NumField value={value.fleece_rayl ?? 0} min={0} max={5000} step={50} digits={0} onChange={(v) => set({ fleece_rayl: v ?? 0 })} width={110} />
            <span className="muted small">Pa·s/m</span>
          </div>
          <label>Plenum depth</label>
          <div className="row gap">
            <NumField value={value.plenum_mm} min={5} max={1500} step={10} digits={0} onChange={(v) => set({ plenum_mm: v ?? 200 })} width={110} />
            <span className="muted small">mm</span>
          </div>
          <label title="Mineral wool laid in the plenum, behind the sheet; 0 = empty plenum">Wool in the plenum</label>
          <div className="row gap">
            <NumField value={value.wool_mm ?? 0} min={0} max={value.plenum_mm} step={10} digits={0} onChange={(v) => set({ wool_mm: v ?? 0 })} width={110} />
            <span className="muted small">mm</span>
          </div>
          {(value.wool_mm ?? 0) > 0 && (
            <>
              <label>Wool flow resistivity σ</label>
              <div className="row gap">
                <NumField value={value.wool_sigma ?? 10000} min={1000} max={1e6} step={1000} digits={0} onChange={(v) => set({ wool_sigma: v ?? 10000 })} width={110} />
                <span className="muted small">Pa·s/m²</span>
              </div>
            </>
          )}
        </div>
        <div className="stack-diagram" aria-hidden="true">
          <span className="stack-porous" style={{ flex: Math.max(value.t_mm, 4) }}>
            {slotted ? 'slots' : 'holes'} {fmtNum(value.open_pct, 1)} %
          </span>
          {(value.wool_mm ?? 0) > 0 && (
            <span className="stack-porous" style={{ flex: value.wool_mm }}>
              wool {fmtNum(value.wool_mm, 0)} mm
            </span>
          )}
          {air > 0 && (
            <span className="stack-air" style={{ flex: air }}>
              air {fmtNum(air, 0)} mm
            </span>
          )}
          <span className="stack-wall">rigid</span>
        </div>
        <p className="muted small">
          Textbook models (Cox &amp; D&apos;Antonio for holes, Kristiansen &amp; Vigran for slots, Maa for micro-perforations). The model is kept with the material: the wave band takes its phase and
          its α. Not yet validated against clean measurements: compare with the manufacturer&apos;s ISO 354 data.
        </p>
      </div>
      <div>
        {error ? (
          <ErrorBox message={error} />
        ) : preview ? (
          <>
            <AlphaChart values={preview} height={190} />
            {preview.some((v) => v > ABS_MAX) && <div className="warn-box small">Above 0.95 in some bands: stored values are clipped to 0.95.</div>}
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
