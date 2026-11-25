import { useMemo } from 'react';
import { BANDS_HZ, type FrSmoothing } from '../../api';
import { Plot, type PlotTheme } from '../../components/Plot';
import { EmptyState, ErrorBox, Segmented, Spinner } from '../../components/ui';
import { useApp } from '../../AppContext';
import { copyText, downloadText, safeFilename, toCsv, toTsv, type DataTable } from '../../lib/export';
import { bandLabel, fmtNum } from '../../lib/format';
import { hasBand63, isoLowMean, PARAMS, stiRating, type ParamInfo, type ParamKey } from '../../lib/metrics';
import { PART_LABELS, SOLVER_KEYS, type ResolvedCard } from './cards';
import { shortDecayBands } from './RunNotices';
import { drawable, paramLayout, paramTable, paramTraces, responseLayout, responseTraces, show63, singleNumberTable, waveBandOk, type Channel, type PlotType, type ResponseOpts } from './charts';

/** Note under a table or chart when some comparison shows the geometrical or wave result alone. */
function partNote(cards: ResolvedCard[], paramKey: ParamKey | null): string | null {
  const parts = Array.from(new Set(drawable(cards).map((c) => c.part).filter((p) => p !== 'hybrid')));
  if (!parts.length) return null;
  const which = parts.map((p) => PART_LABELS[p].toLowerCase()).join(' and ');
  if (paramKey && !(SOLVER_KEYS as string[]).includes(paramKey)) return `Only the hybrid result has this parameter: – for the ${which} result type.`;
  const out = [`Result types: ${which} shown where chosen on a comparison (T20, T30, EDT, C50, C80, D50 and Ts per solver).`];
  if (parts.includes('wave')) out.push('Wave-based: the wave band alone, so the octaves above the transition are –.');
  return out.join(' ');
}

export const SMOOTHING: { value: FrSmoothing; label: string }[] = [
  { value: 24, label: '1/24' },
  { value: 12, label: '1/12' },
  { value: 6, label: '1/6' },
  { value: 3, label: '1/3' },
  { value: 1, label: '1/1' },
  { value: 0, label: 'raw' },
];

export function ParamButtons({ value, onChange }: { value: ParamKey; onChange: (k: ParamKey) => void }) {
  return (
    <div className="param-buttons" role="radiogroup" aria-label="Parameter">
      {PARAMS.map((p) => (
        <button key={p.key} role="radio" aria-checked={value === p.key} className={`chip-btn ${value === p.key ? 'active' : ''}`} title={p.title} onClick={() => onChange(p.key)}>
          <span dangerouslySetInnerHTML={{ __html: p.axis }} />
        </button>
      ))}
    </div>
  );
}

/** Chart + table of one parameter for every comparison. */
export function ParamChart({
  cards,
  param,
  showTransition,
  theme = 'dark',
  height = 300,
  filename,
  showTable = true,
}: {
  cards: ResolvedCard[];
  param: ParamInfo;
  showTransition: boolean;
  theme?: PlotTheme;
  height?: number;
  filename: string;
  showTable?: boolean;
}) {
  const data = useMemo(() => paramTraces(cards, param), [cards, param]);
  const layout = useMemo(() => paramLayout(cards, param, { height, showTransition, light: theme === 'light' }), [cards, param, height, showTransition, theme]);
  const table = useMemo(() => paramTable(cards, param), [cards, param]);
  if (!drawable(cards).length) return <EmptyState title="Nothing to show yet" >Pick a simulation with results, a source and a receiver in a comparison.</EmptyState>;
  const note = partNote(cards, param.key);
  return (
    <>
      <Plot data={data} layout={layout} theme={theme} tools filename={filename} table={table} />
      {showTable && <ParamTable cards={cards} param={param} />}
      {note && <div className="small muted part-note">{note}</div>}
    </>
  );
}

function Missing({ n, total }: { n: number; total: number }) {
  if (!n) return null;
  return (
    <sup className="text-warn" title={`Mean of ${total - n} of ${total} receivers: undefined at the others`}>
      *
    </sup>
  );
}

export function ParamTable({ cards, param }: { cards: ResolvedCard[]; param: ParamInfo }) {
  const list = drawable(cards);
  if (!list.length) return null;
  if (param.scalar) {
    return (
      <table className="table compact param-table">
        <thead>
          <tr>
            <th />
            <th className="num">STI</th>
            <th className="col-rating">Rating</th>
            {BANDS_HZ.map((f) => (
              <th key={f} className="num" title={`Modulation transfer index, ${bandLabel(f)} Hz band`}>
                {bandLabel(f)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {list.map((c) => {
            const sti = c.values!.scalars.STI;
            const r = stiRating(sti);
            return (
              <tr key={c.card.key}>
                <td className="ellipsis" title={c.label}>
                  <span className="swatch" style={{ background: c.color }} />
                  {c.label}
                </td>
                <td className="num">{fmtNum(sti, 2)}</td>
                <td>{r ? <span className={`sti-chip ${r.cls}`}>{r.label}</span> : <span className="muted">–</span>}</td>
                {(c.values!.bands.MTI ?? BANDS_HZ.map(() => null)).map((v, i) => (
                  <td key={i} className="num">
                    {fmtNum(v, 2)}
                    <Missing n={c.values!.missing.MTI?.[i] ?? 0} total={c.values!.n} />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
        <caption className="muted small">Right: modulation transfer index (MTI) per octave band, from which the STI is weighted.</caption>
      </table>
    );
  }
  const key = param.key as Exclude<ParamKey, 'STI'>;
  const with63 = show63(cards, param);
  const k63 = hasBand63(key) ? key : null;
  return (
    <table className="table compact param-table">
      <thead>
        <tr>
          <th>{param.unit ? `(${param.unit})` : ''}</th>
          {with63 && (
            <th className="num band63" title="63 Hz octave (44–88 Hz), from the wave band alone: the geometrical solver has no data below 125 Hz">
              63<sup>*</sup>
            </th>
          )}
          {BANDS_HZ.map((f) => (
            <th key={f} className="num">
              {bandLabel(f)}
            </th>
          ))}
          {param.isoLowMean && (
            <th className="num" title="ISO 3382-1 single number: mean of 125 Hz–1 kHz">
              ISO
            </th>
          )}
        </tr>
      </thead>
      <tbody>
        {list.map((c) => {
          const vals = c.values!.bands[key] ?? BANDS_HZ.map(() => null);
          // T30 where some pair of the card ends before a 45 dB decay.
          const short = key === 'T30' ? new Set(c.pairs.flatMap((p) => shortDecayBands(p.metrics.decay_range_db))) : null;
          return (
            <tr key={c.card.key}>
              <td className="ellipsis" title={c.label}>
                <span className="swatch" style={{ background: c.color }} />
                {c.label}
              </td>
              {with63 && (
                <td className="num band63" title={c.values!.band63 ? '63 Hz, wave band only' : 'No 63 Hz value: the wave solver did not run with a transition of at least 125 Hz'}>
                  {fmtNum(k63 ? c.values!.band63?.[k63] : null, param.digits)}
                  {k63 && <Missing n={c.values!.missing63[k63] ?? 0} total={c.values!.n} />}
                </td>
              )}
              {vals.map((v, i) => (
                <td key={i} className="num" title={v === null ? 'Not computable' : undefined}>
                  {fmtNum(v, param.digits)}
                  <Missing n={c.values!.missing[key]?.[i] ?? 0} total={c.values!.n} />
                  {short?.has(i) && (
                    <sup className="text-warn" title="T30 unreliable: the IR ends before a 45 dB decay in this band (lengthen the IR)">
                      †
                    </sup>
                  )}
                </td>
              ))}
              {param.isoLowMean && <td className="num strong">{fmtNum(isoLowMean(vals), param.digits)}</td>}
            </tr>
          );
        })}
      </tbody>
      {with63 && (
        <caption className="muted small">
          {list.some((c) => c.part !== 'hybrid')
            ? '* 63 Hz octave (44–88 Hz) from the wave band alone (no G or SPL there; none for the geometrical result type); 125 Hz and above: the result type of each comparison.'
            : '* 63 Hz octave (44–88 Hz) from the wave band alone (no G or SPL there); 125 Hz and above: the full hybrid result.'}
        </caption>
      )}
    </table>
  );
}

function TableTools({ table, filename }: { table: DataTable; filename: string }) {
  const app = useApp();
  return (
    <div className="table-tools no-print">
      <button type="button" className="icon-link" onClick={() => downloadText(toCsv(table), `${safeFilename(filename)}.csv`)} title="Download the table (CSV)">
        CSV
      </button>
      <button
        type="button"
        className="icon-link"
        title="Copy the table to the clipboard (tab-separated, 3 decimals)"
        onClick={() =>
          copyText(toTsv(table))
            .then(() => app.toast('success', `Copied ${table.rows.length} rows`))
            .catch((e) => app.toast('error', `Copy failed: ${String(e)}`))
        }
      >
        Copy
      </button>
    </div>
  );
}

/** STI with its rating, J_LF and L_J ISO single numbers, overall SPL Z/A/C. */
export function SingleNumbers({ cards, filename }: { cards: ResolvedCard[]; filename: string }) {
  const list = drawable(cards);
  const table = useMemo(() => singleNumberTable(cards), [cards]);
  if (!list.length) return null;
  return (
    <div className="single-numbers">
      <div className="row gap">
        <h3>Single numbers</h3>
        <div className="grow" />
        <TableTools table={table} filename={filename} />
      </div>
      <table className="table compact">
        <thead>
          <tr>
            <th />
            <th className="num" title="Speech transmission index (IEC 60268-16)">
              STI
            </th>
            <th className="col-rating">Rating</th>
            <th className="num" title="Early lateral energy fraction, ISO single number (mean 125 Hz–1 kHz)">
              J<sub>LF</sub>
            </th>
            <th className="num" title="Late lateral sound level, ISO single number (mean 125 Hz–1 kHz)">
              L<sub>J</sub> (dB)
            </th>
            <th className="num" title="Overall SPL (dB), energy sum of the octave bands, no weighting">
              SPL Z
            </th>
            <th className="num" title="Overall SPL, A-weighted (dB)">
              SPL A
            </th>
            <th className="num" title="Overall SPL, C-weighted (dB)">
              SPL C
            </th>
          </tr>
        </thead>
        <tbody>
          {list.map((c) => {
            const v = c.values!;
            const r = stiRating(v.scalars.STI);
            return (
              <tr key={c.card.key}>
                <td className="ellipsis" title={c.label}>
                  <span className="swatch" style={{ background: c.color }} />
                  {c.label}
                </td>
                <td className="num strong">{fmtNum(v.scalars.STI, 2)}</td>
                <td>{r ? <span className={`sti-chip ${r.cls}`}>{r.label}</span> : <span className="muted">–</span>}</td>
                <td className="num">{fmtNum(isoLowMean(v.bands.JLF), 3)}</td>
                <td className="num">{fmtNum(isoLowMean(v.bands.LJ_db), 1)}</td>
                <td className="num">{fmtNum(v.scalars.SPL_dbz, 1)}</td>
                <td className="num">{fmtNum(v.scalars.SPL_dba, 1)}</td>
                <td className="num">{fmtNum(v.scalars.SPL_dbc, 1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {drawable(cards).some((c) => c.part !== 'hybrid') && <div className="small muted">Single numbers exist for the hybrid result only: – for the geometrical and wave result types.</div>}
      <div className="muted small">
        STI rating (IEC 60268-16): &lt; 0.30 bad · 0.30–0.45 poor · 0.45–0.60 fair · 0.60–0.75 good · &gt; 0.75 excellent. Overall SPL in dB(Z/A/C) for the source&apos;s level (94 dB at 1 m + gain).
      </div>
    </div>
  );
}

export interface ResponseState {
  plotType: PlotType;
  edcBands: number[];
  smoothing: FrSmoothing;
  irDb: boolean;
  channel: Channel;
}

export function ResponseControls({ s, set, loading }: { s: ResponseState; set: (p: Partial<ResponseState>) => void; loading: boolean }) {
  return (
    <>
      <Segmented<PlotType>
        value={s.plotType}
        onChange={(v) => set({ plotType: v })}
        options={[
          { value: 'edc', label: 'EDC' },
          { value: 'ir', label: 'Impulse response' },
          { value: 'fr', label: 'Frequency response' },
        ]}
      />
      <div className="grow" />
      {s.plotType === 'edc' && (
        <div className="band-picks">
          {BANDS_HZ.map((b) => (
            <label key={b} className={`chip-btn ${s.edcBands.includes(b) ? 'active' : ''}`}>
              <input
                type="checkbox"
                checked={s.edcBands.includes(b)}
                onChange={(e) => set({ edcBands: e.target.checked ? [...s.edcBands, b].sort((x, y) => x - y) : s.edcBands.filter((x) => x !== b) })}
              />
              {bandLabel(b)}
            </label>
          ))}
        </div>
      )}
      {s.plotType === 'fr' && (
        <label className="inline-field">
          <span>Smoothing</span>
          <select value={s.smoothing} onChange={(e) => set({ smoothing: Number(e.target.value) as FrSmoothing })}>
            {SMOOTHING.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label} {o.value ? 'octave' : ''}
              </option>
            ))}
          </select>
        </label>
      )}
      {s.plotType === 'ir' && (
        <>
          <Segmented<Channel>
            value={s.channel}
            onChange={(v) => set({ channel: v })}
            options={[
              { value: 'L', label: 'L', title: 'Left ear' },
              { value: 'R', label: 'R', title: 'Right ear' },
              { value: 'LR', label: 'L+R', title: 'Both ears' },
            ]}
          />
          <label className="check inline">
            <input type="checkbox" checked={s.irDb} onChange={(e) => set({ irDb: e.target.checked })} /> dB
          </label>
        </>
      )}
      {loading && <Spinner />}
    </>
  );
}

export function ResponseChart({
  cards,
  s,
  opts,
  error,
  loading,
  theme = 'dark',
  height = 300,
  filename,
}: {
  cards: ResolvedCard[];
  s: ResponseState;
  opts: Pick<ResponseOpts, 'curves' | 'showTransition'>;
  error: string | null;
  loading: boolean;
  theme?: PlotTheme;
  height?: number;
  filename: string;
}) {
  const data = useMemo(
    () => responseTraces(cards, { plotType: s.plotType, edcBands: s.edcBands, irDb: s.irDb, channel: s.channel, showTransition: opts.showTransition, curves: opts.curves }),
    [cards, s.plotType, s.edcBands, s.irDb, s.channel, opts.showTransition, opts.curves],
  );
  const multiSim = new Set(drawable(cards).map((c) => c.card.simId)).size > 1;
  const nSeries = data.filter((d) => (d as { meta?: unknown }).meta !== 'decor').length;
  const layout = useMemo(() => responseLayout({ plotType: s.plotType, irDb: s.irDb, height, multiSim, nSeries }), [s.plotType, s.irDb, height, multiSim, nSeries]);
  const waveCards = drawable(cards).filter((c) => c.part === 'wave');
  const hidden = s.plotType === 'edc' ? waveCards.filter((c) => s.edcBands.some((b) => !waveBandOk(b, c.results?.transition_hz))) : [];
  const parts = drawable(cards).some((c) => c.part !== 'hybrid');
  return (
    <>
      {error && <ErrorBox message={error} />}
      {hidden.length > 0 && (
        <div className="small muted part-note">
          Wave-based result: no decay above the transition ({hidden.map((c) => fmtNum(c.results?.transition_hz, 0)).filter((v, i, a) => a.indexOf(v) === i).join(', ')} Hz), so those bands are not drawn.
        </div>
      )}
      {parts && s.plotType === 'ir' && <div className="small muted part-note">Geometrical and wave-based results: the omnidirectional IR of that part (no binaural split).</div>}
      {data.length ? (
        <Plot data={data} layout={layout} theme={theme} tools filename={filename} />
      ) : (
        !loading && <EmptyState title={s.plotType === 'edc' && !s.edcBands.length ? 'Select at least one band' : 'No curve to show'} />
      )}
    </>
  );
}
