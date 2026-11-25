import { BANDS_HZ, type Source } from '../../api';
import { NumField } from '../../components/ui';
import { bandLabel } from '../../lib/format';

/** Server limit of one octave gain (dB). */
export const EQ_MAX_DB = 24;

export const EQ_TITLE =
  'Per-octave EQ of this source (125 Hz – 8 kHz, ±24 dB, octave-band energy gains). It filters this source’s results (impulse responses) exactly like a filter on its signal: ' +
  'SPL follows it, G (normalised by the source’s own power) does not; T30, C50… are computed on the filtered responses. Animated renders filter the source’s dry signal. ' +
  'Changing it makes the results stale.';

const flat = (): number[] => BANDS_HZ.map(() => 0);

/** Whether a source's EQ is anything but flat. */
export function eqActive(eq: readonly number[] | null | undefined): boolean {
  return !!eq && eq.some((g) => Number.isFinite(g) && g !== 0);
}

/** 7 finite gains within ±24 dB, rounded like the server (0.01 dB); undefined when flat. */
export function cleanEq(eq: readonly number[] | null | undefined): number[] | undefined {
  if (!eq) return undefined;
  const g = BANDS_HZ.map((_, i) => {
    const v = Number(eq[i]);
    return Number.isFinite(v) ? Math.round(Math.max(-EQ_MAX_DB, Math.min(EQ_MAX_DB, v)) * 100) / 100 : 0;
  });
  return g.some((v) => v !== 0) ? g : undefined;
}

/** The source with this EQ; the key is dropped when the EQ is flat (as the server stores it). */
export function withEq(src: Source, eq: readonly number[] | null | undefined): Source {
  const { eq_db: _old, ...rest } = src;
  const g = cleanEq(eq);
  return (g ? { ...rest, eq_db: g } : rest) as Source;
}

const fmtDb = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(Math.abs(v) % 1 ? 1 : 0)}`;

/** One line per band, for tooltips. */
export function eqSummary(eq: readonly number[] | null | undefined): string {
  const g = eq ?? flat();
  return BANDS_HZ.map((f, i) => `${bandLabel(f)}: ${fmtDb(g[i] ?? 0)} dB`).join(', ');
}

/** The 7 gains joined in log frequency (octave centres equally spaced), 0 dB line in the middle. */
export function EqCurve({ gains, width = 120, height = 36, compact = false }: { gains: readonly number[] | null | undefined; width?: number; height?: number; compact?: boolean }) {
  const g = BANDS_HZ.map((_, i) => gains?.[i] ?? 0);
  const peak = Math.max(...g.map(Math.abs));
  const range = Math.max(6, Math.ceil(peak / 6) * 6);
  const pad = compact ? 2 : 4;
  const x = (i: number) => pad + (i * (width - 2 * pad)) / (BANDS_HZ.length - 1);
  const y = (v: number) => height / 2 - (v / range) * (height / 2 - pad);
  const pts = g.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return (
    <svg className={`eq-curve ${compact ? 'compact' : ''}`} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Octave-band energy gains: ${eqSummary(g)}`}>
      <title>{`Octave-band energy gains (dB), scale ±${range} dB\n${eqSummary(g)}`}</title>
      {!compact && <line className="eq-grid" x1={pad} x2={width - pad} y1={y(range)} y2={y(range)} />}
      {!compact && <line className="eq-grid" x1={pad} x2={width - pad} y1={y(-range)} y2={y(-range)} />}
      <line className="eq-zero" x1={pad} x2={width - pad} y1={y(0)} y2={y(0)} />
      <polyline className="eq-line" points={pts} />
      {!compact && g.map((v, i) => <circle key={i} className="eq-dot" cx={x(i)} cy={y(v)} r={2.2} />)}
      {!compact && (
        <text className="eq-scale" x={width - pad} y={pad + 7} textAnchor="end">
          ±{range} dB
        </text>
      )}
    </svg>
  );
}

/** Compact 7-band editor: a vertical slider and a number per octave, a flat button and the response. */
export function SourceEqEditor({ value, onChange, error }: { value: readonly number[] | null | undefined; onChange: (eq: number[] | undefined) => void; error?: string | null }) {
  const g = BANDS_HZ.map((_, i) => value?.[i] ?? 0);
  const active = eqActive(g);
  const set = (i: number, v: number) => onChange(cleanEq(g.map((x, j) => (j === i ? v : x))));
  return (
    <div className="eq-editor" title={EQ_TITLE}>
      <div className="eq-head">
        <span className="small">
          <b>Octave EQ</b> <span className="muted">dB</span>
        </span>
        <EqCurve gains={g} width={132} height={38} />
        <div className="grow" />
        <button type="button" className="btn btn-xs" disabled={!active} onClick={() => onChange(undefined)} title="Set every band to 0 dB (flat: no filter)">
          Flat
        </button>
      </div>
      <div className="eq-bands">
        {BANDS_HZ.map((f, i) => (
          <div key={f} className="eq-band">
            <span className="eq-band-label">{bandLabel(f)}</span>
            <input
              type="range"
              className="eq-slider"
              min={-EQ_MAX_DB}
              max={EQ_MAX_DB}
              step={0.5}
              value={g[i]}
              aria-label={`EQ gain at ${f} Hz (dB)`}
              title={`${bandLabel(f)} Hz: ${fmtDb(g[i])} dB (double-click: 0 dB)`}
              onChange={(e) => set(i, Number(e.target.value))}
              onDoubleClick={() => set(i, 0)}
            />
            <NumField value={g[i]} min={-EQ_MAX_DB} max={EQ_MAX_DB} step={0.5} digits={1} width={40} ariaLabel={`EQ gain at ${f} Hz (dB)`} onChange={(v) => set(i, v ?? 0)} />
          </div>
        ))}
      </div>
      <div className="small muted">Octave-band energy gains. Filters this source’s impulse responses like a filter on its signal: SPL follows it, G does not.</div>
      {error && <div className="small text-err">{error}</div>}
    </div>
  );
}
