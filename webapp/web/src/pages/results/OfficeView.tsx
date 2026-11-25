import { useEffect, useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { api, BANDS_HZ, errorMessage, type OfficeResult, type Results, type Simulation } from '../../api';
import { useApp } from '../../AppContext';
import { Plot } from '../../components/Plot';
import { EmptyState, ErrorBox, Spinner } from '../../components/ui';
import { COLORS } from '../../lib/colors';
import { copyText, downloadText, safeFilename, toCsv, toTsv, type DataTable } from '../../lib/export';
import { bandLabel, fmtNum } from '../../lib/format';
import { distance } from '../../lib/geom';
import { getSlice, setSlice } from '../../lib/savedViews';
import { resultReceivers, resultSources } from './cards';

/** ISO 3382-3:2022 constants used for the drawing (the values come from the server). */
const COMFORT_DBA = 45;
const DISTRACTION_STI = 0.5;
const R_MIN = 2;
const R_MAX = 16;
const LINE = '#f0a83a';
const POINT = COLORS.receiver;

interface OfficeSaved {
  sourceId?: string | null;
  receiverIds?: string[];
}

function noiseText(n: unknown): string {
  if (n === undefined || n === null || n === 'none') return 'none (set one in the editor’s Settings, STI group)';
  if (Array.isArray(n)) return `custom levels (${n.map((v) => fmtNum(Number(v), 0)).join(' / ')} dB)`;
  return String(n);
}

/** Least-squares line y = k x + c (what the server fits for rD). */
function linFit(x: number[], y: number[]): { k: number; c: number } | null {
  const n = x.length;
  if (n < 2) return null;
  const sx = x.reduce((a, v) => a + v, 0);
  const sy = y.reduce((a, v) => a + v, 0);
  const sxx = x.reduce((a, v) => a + v * v, 0);
  const sxy = x.reduce((a, v, i) => a + v * y[i], 0);
  const d = n * sxx - sx * sx;
  if (Math.abs(d) < 1e-12) return null;
  const k = (n * sxy - sx * sy) / d;
  return { k, c: (sy - k * sx) / n };
}

/** Open-plan office indicators (ISO 3382-3) along a path of receivers for one source. */
export function OfficeView({ simId, sim, results }: { simId: string; sim: Simulation; results: Results }) {
  const app = useApp();
  const saved = useMemo(() => getSlice<OfficeSaved>(simId, 'results', 'office') ?? {}, [simId]);
  const sources = resultSources(results);
  const [sourceId, setSourceId] = useState<string | null>(saved.sourceId && sources.includes(saved.sourceId) ? saved.sourceId : (sim.sources.find((s) => sources.includes(s.id) && s.directivity === 'omni')?.id ?? sources[0] ?? null));
  const src = sim.sources.find((s) => s.id === sourceId) ?? null;
  const rcvIds = useMemo(() => resultReceivers(results, sourceId), [results, sourceId]);
  // Receivers of the path candidates, ordered by distance from the source.
  const candidates = useMemo(
    () =>
      rcvIds
        .map((id) => {
          const r = sim.receivers.find((x) => x.id === id);
          return { id, name: r?.name ?? id, d: r && src ? distance(r.position, src.position) : NaN };
        })
        .sort((a, b) => a.d - b.d),
    [rcvIds, sim, src],
  );
  const [selected, setSelected] = useState<string[]>(() => (saved.receiverIds ?? rcvIds).filter((x) => rcvIds.includes(x)));
  useEffect(() => {
    setSelected((s) => {
      const n = s.filter((x) => rcvIds.includes(x));
      return n.length === s.length ? s : n;
    });
  }, [rcvIds]);
  const ordered = candidates.filter((c) => selected.includes(c.id)).map((c) => c.id);
  useEffect(() => setSlice(simId, 'results', 'office', { sourceId, receiverIds: ordered }), [simId, sourceId, ordered.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const [state, setState] = useState<{ key: string; data: OfficeResult | null; error: string | null }>({ key: '', data: null, error: null });
  const key = sourceId && ordered.length >= 2 ? `${sourceId}|${ordered.join(',')}` : '';
  useEffect(() => {
    if (!key || !sourceId) return;
    let cancelled = false;
    const t = window.setTimeout(() => {
      api
        .getOffice(simId, sourceId, ordered)
        .then((d) => !cancelled && setState({ key, data: d, error: null }))
        .catch((e) => !cancelled && setState({ key, data: null, error: errorMessage(e) }));
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // `ordered` is described by `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simId, key]);
  const fresh = state.key === key ? state : null;
  const data = fresh?.data ?? null;
  const loading = !!key && !fresh;

  const rcvName = (id: string) => sim.receivers.find((r) => r.id === id)?.name ?? id;
  const pos = data?.positions ?? [];
  const dMin = pos.length ? Math.min(...pos.map((p) => p.distance_m)) : 1;
  const dMax = pos.length ? Math.max(...pos.map((p) => p.distance_m)) : 10;
  const inRange = (d: number) => d >= R_MIN && d <= R_MAX;
  const stiFit = useMemo(() => {
    const use = pos.filter((p) => p.distance_m > 1 + 1e-9);
    return linFit(
      use.map((p) => p.distance_m),
      use.map((p) => p.STI),
    );
  }, [pos]);

  // Plot ranges: the measured span, widened to rC / rD when they are close enough to show.
  const xLo = Math.max(0.5, dMin * 0.8);
  const near = (r: number | null) => r !== null && r > 0 && r <= dMax * 3 && r >= dMin / 3;
  const xHi = Math.max(dMax * 1.25, near(data?.rC_m ?? null) ? data!.rC_m! * 1.15 : 0, near(data?.rD_m ?? null) ? data!.rD_m! * 1.15 : 0);
  const xLo2 = Math.min(xLo, near(data?.rC_m ?? null) ? data!.rC_m! * 0.85 : xLo, near(data?.rD_m ?? null) ? data!.rD_m! * 0.85 : xLo);

  const splData = useMemo<Data[]>(() => {
    if (!data) return [];
    const out: Data[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x: pos.map((p) => p.distance_m),
        y: pos.map((p) => p.Lp_A_S_db),
        text: pos.map((p) => rcvName(p.receiver_id)),
        name: 'Lp,A,S at the positions',
        marker: { size: 10, color: pos.map((p) => (inRange(p.distance_m) ? POINT : 'rgba(0,0,0,0)')), line: { color: POINT, width: 2 } },
        hovertemplate: '%{text}<br>%{x:.2f} m: %{y:.1f} dB(A)<extra></extra>',
      },
    ];
    if (data.D2S_db !== null && data.Lp_A_S_4m_db !== null) {
      const xs = Array.from({ length: 40 }, (_, i) => xLo2 * (xHi / xLo2) ** (i / 39));
      out.push({
        type: 'scatter',
        mode: 'lines',
        x: xs,
        y: xs.map((r) => data.Lp_A_S_4m_db! - data.D2S_db! * Math.log2(r / 4)),
        name: `Regression (${data.n_in_range} positions 2–16 m): D2,S ${fmtNum(data.D2S_db, 1)} dB`,
        line: { color: LINE, width: 2 },
        hovertemplate: '%{x:.2f} m: %{y:.1f} dB(A)<extra>regression</extra>',
      });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, xLo2, xHi]);
  const splLayout = useMemo<Partial<Layout>>(() => {
    const shapes: Layout['shapes'] = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: COMFORT_DBA, y1: COMFORT_DBA, line: { color: '#9aa3ab', width: 1.2, dash: 'dash' } }];
    const annotations: Layout['annotations'] = [{ xref: 'paper', x: 1, y: COMFORT_DBA, yref: 'y', text: '45 dB(A)', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 10, color: '#9aa3ab' } }];
    if (data?.rC_m && near(data.rC_m)) {
      shapes.push({ type: 'line', xref: 'x', x0: data.rC_m, x1: data.rC_m, yref: 'paper', y0: 0, y1: 1, line: { color: LINE, width: 1.2, dash: 'dot' } });
      annotations.push({ xref: 'x', x: Math.log10(data.rC_m), yref: 'paper', y: 1, text: `rC ${fmtNum(data.rC_m, 1)} m`, showarrow: false, yanchor: 'bottom', font: { size: 10, color: LINE } });
    }
    const ys = pos.map((p) => p.Lp_A_S_db);
    const yMin = Math.min(COMFORT_DBA - 3, ...ys) - 2;
    const yMax = Math.max(COMFORT_DBA + 3, ...ys) + 2;
    return {
      height: 290,
      margin: { l: 52, r: 12, t: 30, b: 40 },
      xaxis: { type: 'log', range: [Math.log10(xLo2), Math.log10(xHi)], title: { text: 'Distance from the source (m, log scale)' } },
      yaxis: { title: { text: 'Lp,A,S (dB(A))' }, range: [yMin, yMax] },
      shapes,
      annotations,
      showlegend: true,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, xLo2, xHi]);
  const splTable = (): DataTable => ({
    columns: ['receiver', 'distance (m)', 'Lp,A,S (dB(A))', 'in 2-16 m', 'regression (dB(A))'],
    rows: pos.map((p) => [rcvName(p.receiver_id), p.distance_m, p.Lp_A_S_db, inRange(p.distance_m) ? 1 : 0, data?.D2S_db !== null && data?.Lp_A_S_4m_db !== null ? data!.Lp_A_S_4m_db! - data!.D2S_db! * Math.log2(p.distance_m / 4) : null]),
  });

  const stiData = useMemo<Data[]>(() => {
    if (!data) return [];
    const out: Data[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x: pos.map((p) => p.distance_m),
        y: pos.map((p) => p.STI),
        text: pos.map((p) => rcvName(p.receiver_id)),
        name: 'STI at the positions',
        marker: { size: 10, color: POINT, line: { color: POINT, width: 2 } },
        hovertemplate: '%{text}<br>%{x:.2f} m: STI %{y:.3f}<extra></extra>',
      },
    ];
    if (stiFit) {
      out.push({ type: 'scatter', mode: 'lines', x: [xLo2, xHi], y: [stiFit.k * xLo2 + stiFit.c, stiFit.k * xHi + stiFit.c], name: 'Linear fit (positions beyond 1 m)', line: { color: LINE, width: 2 }, hovertemplate: '%{x:.2f} m: STI %{y:.3f}<extra>fit</extra>' });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, stiFit, xLo2, xHi]);
  const stiLayout = useMemo<Partial<Layout>>(() => {
    const shapes: Layout['shapes'] = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: DISTRACTION_STI, y1: DISTRACTION_STI, line: { color: '#9aa3ab', width: 1.2, dash: 'dash' } }];
    const annotations: Layout['annotations'] = [{ xref: 'paper', x: 1, y: DISTRACTION_STI, yref: 'y', text: 'STI 0.50', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 10, color: '#9aa3ab' } }];
    if (data?.rD_m && near(data.rD_m)) {
      shapes.push({ type: 'line', xref: 'x', x0: data.rD_m, x1: data.rD_m, yref: 'paper', y0: 0, y1: 1, line: { color: LINE, width: 1.2, dash: 'dot' } });
      annotations.push({ xref: 'x', x: data.rD_m, yref: 'paper', y: 1, text: `rD ${fmtNum(data.rD_m, 1)} m`, showarrow: false, yanchor: 'bottom', font: { size: 10, color: LINE } });
    }
    return {
      height: 260,
      margin: { l: 52, r: 12, t: 30, b: 40 },
      xaxis: { range: [xLo2, xHi], title: { text: 'Distance from the source (m)' } },
      yaxis: { title: { text: 'STI' }, range: [0, 1], dtick: 0.1 },
      shapes,
      annotations,
      showlegend: true,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, xLo2, xHi]);
  const stiTable = (): DataTable => ({
    columns: ['receiver', 'distance (m)', 'STI', 'linear fit'],
    rows: pos.map((p) => [rcvName(p.receiver_id), p.distance_m, p.STI, stiFit ? stiFit.k * p.distance_m + stiFit.c : null]),
  });

  const posTable = (): DataTable => ({
    columns: ['receiver', 'distance (m)', 'in 2-16 m', 'Lp,A,S (dB(A))', 'STI', ...BANDS_HZ.map((f) => `Lp,S ${f} Hz (dB)`)],
    rows: pos.map((p) => [rcvName(p.receiver_id), p.distance_m, inRange(p.distance_m) ? 1 : 0, p.Lp_A_S_db, p.STI, ...p.Lp_S_db]),
  });
  const singleTable = (): DataTable => ({
    columns: ['quantity', 'value', 'unit', 'note'],
    rows: data
      ? [
          ['D2,S', data.D2S_db, 'dB', `regression over ${data.n_in_range} positions 2-16 m`],
          ['Lp,A,S,4m', data.Lp_A_S_4m_db, 'dB(A)', ''],
          ['rC', data.rC_m, 'm', data.rC_note ?? ''],
          ['rD', data.rD_m, 'm', data.rD_note ?? ''],
          ['Lp,A,B', data.Lp_A_B_db, 'dB(A)', data.Lp_A_B_db === null ? 'no background noise' : ''],
        ]
      : [],
  });
  const base = safeFilename(`${sim.name}_ISO3382-3_${src?.name ?? ''}`);

  if (!sources.length) return <EmptyState title="No results">Run the simulation first.</EmptyState>;

  // Remarks on what the numbers mean when they are extrapolated far out.
  const rcRemark = data?.rC_m !== null && data?.rC_m !== undefined && data.rC_m > 10 * dMax ? `far beyond the path (${fmtNum(dMax, 1)} m): the level hardly decays along it` : null;
  const rdRemark =
    data?.rD_m !== null && data?.rD_m !== undefined && data.rD_m < 0
      ? pos.every((p) => p.STI < DISTRACTION_STI)
        ? 'negative: the STI is below 0.50 at every position of the path'
        : 'negative: no crossing of 0.50 along the path'
      : null;

  return (
    <div className="results office">
      <div className="results-main">
        <section className="card">
          <div className="card-head">
            <h3>Open-plan office · ISO 3382-3:2022</h3>
            <div className="grow" />
            <button
              className="btn btn-xs"
              disabled={!data}
              onClick={() =>
                copyText(toTsv(singleTable()))
                  .then(() => app.toast('success', 'Copied the ISO 3382-3 values'))
                  .catch((e) => app.toast('error', `Copy failed: ${String(e)}`))
              }
              title="Copy the single numbers (tab-separated)"
            >
              Copy values
            </button>
          </div>
          {fresh?.error && <ErrorBox message={fresh.error} />}
          {ordered.length < 2 && <div className="warn-box small">Select at least 2 receivers along the path (the standard asks for at least 4, ideally 6 to 10, 2–16 m from the source).</div>}
          {loading && (
            <div className="loading-row">
              <Spinner /> Computing…
            </div>
          )}
          {data && (
            <>
              {data.directivity_note && <div className="warn-box small">{data.directivity_note}.</div>}
              <div className="office-numbers">
                <Num label="D2,S" title="Spatial decay rate of A-weighted speech per distance doubling (regression over the positions 2–16 m)" value={data.D2S_db} unit="dB" digits={1} note={data.n_in_range < 2 ? `needs ≥ 2 positions 2–16 m from the source (${data.n_in_range} here)` : `regression over ${data.n_in_range} position${data.n_in_range === 1 ? '' : 's'} 2–16 m`} />
                <Num label="Lp,A,S,4m" title="A-weighted speech level at 4 m, from the regression line" value={data.Lp_A_S_4m_db} unit="dB(A)" digits={1} />
                <Num label="rC" title="Comfort distance: where the regression line reaches 45 dB(A)" value={data.rC_m} unit="m" digits={1} note={data.rC_note} remark={rcRemark} />
                <Num label="rD" title="Distraction distance: where a linear fit of STI vs distance reaches 0.50" value={data.rD_m} unit="m" digits={1} note={data.rD_note} remark={rdRemark} />
                <Num label="Lp,A,B" title="A-weighted background noise level (from the simulation's settings)" value={data.Lp_A_B_db} unit="dB(A)" digits={1} note={data.Lp_A_B_db === null ? 'no background noise in the settings' : null} />
              </div>
              <div className="small muted">
                Speech: the standard&apos;s normal-effort spectrum at 1 m (Table 1) plus the simulated G; STI with the background noise of the run&apos;s settings: {noiseText(results.settings?.background_noise)}. Hollow markers: outside 2–16 m (not in the regression).
              </div>
            </>
          )}
        </section>
        {data && (
          <>
            <section className="card">
              <h3>Spatial decay of A-weighted speech</h3>
              <Plot data={splData} layout={splLayout} tools filename={`${base}_spatial_decay`} table={splTable} />
              {data.rC_m !== null && !near(data.rC_m) && <div className="small muted">rC = {fmtBig(data.rC_m)} m is outside the plotted range.</div>}
            </section>
            <section className="card">
              <h3>STI vs distance</h3>
              <Plot data={stiData} layout={stiLayout} tools filename={`${base}_sti`} table={stiTable} />
              {data.rD_m !== null && !near(data.rD_m) && <div className="small muted">rD = {fmtBig(data.rD_m)} m is outside the plotted range.</div>}
            </section>
            <section className="card">
              <div className="card-head">
                <h3>Positions</h3>
                <div className="grow" />
                <div className="table-tools">
                  <button type="button" className="icon-link" onClick={() => downloadText(toCsv(posTable()), `${base}_positions.csv`)} title="Download the table (CSV)">
                    CSV
                  </button>
                  <button
                    type="button"
                    className="icon-link"
                    onClick={() =>
                      copyText(toTsv(posTable()))
                        .then(() => app.toast('success', `Copied ${pos.length} rows`))
                        .catch((e) => app.toast('error', `Copy failed: ${String(e)}`))
                    }
                    title="Copy the table (tab-separated, 3 decimals)"
                  >
                    Copy
                  </button>
                </div>
              </div>
              <div className="table-scroll">
                <table className="table compact">
                  <thead>
                    <tr>
                      <th>Receiver</th>
                      <th className="num">Distance (m)</th>
                      <th className="num" title="A-weighted speech level">
                        Lp,A,S
                      </th>
                      <th className="num">STI</th>
                      {BANDS_HZ.map((f) => (
                        <th key={f} className="num" title={`Speech level in the ${bandLabel(f)} Hz octave (dB)`}>
                          {bandLabel(f)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {pos.map((p) => (
                      <tr key={p.receiver_id} className={inRange(p.distance_m) ? undefined : 'dimmed'} title={inRange(p.distance_m) ? undefined : 'Outside 2–16 m: not in the D2,S regression'}>
                        <td>{rcvName(p.receiver_id)}</td>
                        <td className="num">{fmtNum(p.distance_m, 2)}</td>
                        <td className="num strong">{fmtNum(p.Lp_A_S_db, 1)}</td>
                        <td className="num strong">{fmtNum(p.STI, 2)}</td>
                        {p.Lp_S_db.map((v, i) => (
                          <td key={i} className="num">
                            {fmtNum(v, 1)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="small muted">Columns 125 Hz–8 kHz: speech level per octave (dB) at each position. Dimmed rows lie outside 2–16 m.</div>
            </section>
          </>
        )}
      </div>
      <aside className="results-side">
        <section className="card">
          <h3>Measurement path</h3>
          <label className="field">
            <span>Source (talker)</span>
            <select value={sourceId ?? ''} onChange={(e) => setSourceId(e.target.value)}>
              {sources.map((id) => {
                const s = sim.sources.find((x) => x.id === id);
                return (
                  <option key={id} value={id}>
                    {s?.name ?? id}
                    {s && s.directivity !== 'omni' ? ` (${s.directivity})` : ''}
                  </option>
                );
              })}
            </select>
          </label>
          {src && src.directivity !== 'omni' && <div className="small text-warn">The standard assumes an omnidirectional source.</div>}
          <div className="row gap">
            <span className="small muted grow">Receivers, by distance</span>
            <button className="link small" onClick={() => setSelected(rcvIds)}>
              All
            </button>
            <button className="link small" onClick={() => setSelected([])}>
              None
            </button>
          </div>
          <div className="receiver-picks office-picks">
            {candidates.map((c) => (
              <label key={c.id} className={`check ${inRange(c.d) ? '' : 'muted'}`} title={inRange(c.d) ? undefined : 'Outside 2–16 m: used for rD only'}>
                <input type="checkbox" checked={selected.includes(c.id)} onChange={(e) => setSelected((s) => (e.target.checked ? [...s, c.id] : s.filter((x) => x !== c.id)))} />
                <span className="ellipsis grow">{c.name}</span>
                <span className="small mono">{fmtNum(c.d, 2)} m</span>
              </label>
            ))}
          </div>
          <div className="small muted">
            {ordered.length} selected. Lay the receivers along a path through the workstations, 1.2 m high, at least 1 m from walls; the talker at a workstation at the start of the path.
          </div>
        </section>
      </aside>
    </div>
  );
}

const fmtBig = (v: number) => (Math.abs(v) >= 1e4 ? v.toExponential(1) : fmtNum(v, 1));

function Num({ label, title, value, unit, digits, note, remark }: { label: string; title: string; value: number | null; unit: string; digits: number; note?: string | null; remark?: string | null }) {
  const big = value !== null && Math.abs(value) >= 1e4;
  return (
    <div className="office-num" title={title}>
      <div className="office-num-label">{label}</div>
      <div className="office-num-value">
        {value === null ? '–' : big ? fmtBig(value) : fmtNum(value, digits)} <span className="unit">{value === null ? '' : unit}</span>
      </div>
      {note && <div className={`small ${/extrapolated/.test(note) ? 'text-warn' : 'muted'}`}>{note}</div>}
      {remark && <div className="small text-warn">{remark}</div>}
    </div>
  );
}
