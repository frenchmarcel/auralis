import { useEffect, useMemo, useState } from 'react';
import { errorMessage, type CombineResult, type Results, type Simulation } from '../../api';
import { NumField, Spinner } from '../../components/ui';
import { fmtNum } from '../../lib/format';
import { resultReceivers } from './cards';
import type { ComboDef } from './cards';
import { autoComboName, combineCached } from './combos';

interface Row {
  on: boolean;
  gain_db: number;
  delay_ms: number;
  polarity: 1 | -1;
}

const MAX_DELAY_MS = 2000;

/**
 * Coherent sum of sources at one receiver (API "Coherent sum of sources"): each source
 * with a gain, a delay and a polarity; the sum becomes a comparison card "Σ name".
 */
export function CombinePanel({
  simId,
  sim,
  results,
  editing,
  onAdd,
  onUpdate,
  onCancelEdit,
}: {
  simId: string;
  sim: Simulation;
  results: Results;
  /** A summed card being edited (its definition fills the form). */
  editing: { key: string; def: ComboDef } | null;
  onAdd: (def: ComboDef) => void;
  onUpdate: (key: string, def: ComboDef) => void;
  onCancelEdit: () => void;
}) {
  const receivers = useMemo(() => resultReceivers(results), [results]);
  const [receiverId, setReceiverId] = useState<string>(receivers[0] ?? '');
  const [rows, setRows] = useState<Record<string, Row>>({});
  const [name, setName] = useState('');
  const [filter, setFilter] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<{ name: string; result: CombineResult } | null>(null);

  useEffect(() => {
    if (!editing) return;
    setReceiverId(editing.def.receiverId);
    const r: Record<string, Row> = {};
    for (const s of editing.def.sources) r[s.source_id] = { on: true, gain_db: s.gain_db, delay_ms: s.delay_ms, polarity: s.polarity };
    setRows(r);
    setName(editing.def.name);
    setError(null);
  }, [editing]);

  useEffect(() => {
    if (receiverId && !receivers.includes(receiverId)) setReceiverId(receivers[0] ?? '');
  }, [receivers, receiverId]);

  const sourceIds = useMemo(() => results.pairs.filter((p) => p.receiver_id === receiverId).map((p) => p.source_id), [results, receiverId]);
  const srcName = (id: string) => sim.sources.find((s) => s.id === id)?.name ?? id;
  const rcvName = (id: string) => sim.receivers.find((r) => r.id === id)?.name ?? id;
  const rowOf = (id: string): Row => rows[id] ?? { on: false, gain_db: 0, delay_ms: 0, polarity: 1 };
  const setRow = (id: string, p: Partial<Row>) => setRows((r) => ({ ...r, [id]: { ...rowOf(id), ...p } }));
  const chosen = sourceIds.filter((id) => rowOf(id).on);
  const autoName = autoComboName(chosen.map(srcName));
  const q = filter.trim().toLowerCase();
  const shown = q ? sourceIds.filter((id) => srcName(id).toLowerCase().includes(q) || id.toLowerCase().includes(q)) : sourceIds;
  const allOn = shown.length > 0 && shown.every((id) => rowOf(id).on);

  const submit = async () => {
    if (!chosen.length || !receiverId) return;
    const def: ComboDef = {
      receiverId,
      name: name.trim() || autoName,
      sources: chosen.map((id) => {
        const r = rowOf(id);
        return { source_id: id, gain_db: r.gain_db, delay_ms: r.delay_ms, polarity: r.polarity };
      }),
    };
    setBusy(true);
    setError(null);
    try {
      const result = await combineCached(simId, def);
      setLast({ name: def.name, result });
      if (editing) onUpdate(editing.key, def);
      else onAdd(def);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card combine-panel">
      <div className="card-head">
        <h3>{editing ? 'Edit sum of sources' : 'Combine sources'}</h3>
        <div className="grow" />
        {editing && (
          <button type="button" className="link small" onClick={onCancelEdit}>
            New sum instead
          </button>
        )}
      </div>
      <div className="small muted">
        Coherent sum of the pressure IRs at one receiver, interference included (a loudspeaker system, an ensemble heard as one source). Two identical sources give +6 dB and the same decay.
      </div>
      <div className="combine-fields">
        <label className="inline-field">
          <span>Receiver</span>
          <select value={receiverId} onChange={(e) => setReceiverId(e.target.value)} disabled={!!editing}>
            {receivers.map((r) => (
              <option key={r} value={r}>
                {rcvName(r)}
              </option>
            ))}
          </select>
        </label>
        <label className="inline-field combine-name">
          <span>Name</span>
          <input value={name} placeholder={autoName || 'S1 + S2'} onChange={(e) => setName(e.target.value)} aria-label="Name of the sum" />
        </label>
      </div>
      {sourceIds.length > 8 && <input className="combine-filter" placeholder="Filter sources…" value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter sources" />}
      <div className="table-scroll combine-table">
        <table className="table compact">
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  checked={allOn}
                  aria-label="Select all sources"
                  onChange={(e) => setRows((r) => {
                    const n = { ...r };
                    for (const id of shown) n[id] = { ...rowOf(id), on: e.target.checked };
                    return n;
                  })}
                />
              </th>
              <th>Source</th>
              <th className="num" title="Gain applied to this source (dB)">
                Gain dB
              </th>
              <th className="num" title={`Delay of this source, 0–${MAX_DELAY_MS} ms`}>
                Delay ms
              </th>
              <th title="Polarity: + as computed, − inverted">±</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((id) => {
              const r = rowOf(id);
              return (
                <tr key={id} className={r.on ? '' : 'dimmed'}>
                  <td>
                    <input type="checkbox" checked={r.on} onChange={(e) => setRow(id, { on: e.target.checked })} aria-label={`Include ${srcName(id)}`} />
                  </td>
                  <td className="ellipsis" title={id}>
                    {srcName(id)}
                  </td>
                  <td className="num">
                    <NumField value={r.gain_db} min={-60} max={40} step={1} digits={1} width={52} onChange={(v) => setRow(id, { gain_db: v ?? 0, on: true })} ariaLabel={`Gain of ${srcName(id)} (dB)`} />
                  </td>
                  <td className="num">
                    <NumField value={r.delay_ms} min={0} max={MAX_DELAY_MS} step={1} digits={2} width={58} onChange={(v) => setRow(id, { delay_ms: v ?? 0, on: true })} ariaLabel={`Delay of ${srcName(id)} (ms)`} />
                  </td>
                  <td>
                    <button
                      type="button"
                      className={`btn btn-xs polarity ${r.polarity < 0 ? 'btn-on' : ''}`}
                      title={r.polarity < 0 ? 'Inverted polarity (click for normal)' : 'Normal polarity (click to invert)'}
                      aria-pressed={r.polarity < 0}
                      onClick={() => setRow(id, { polarity: r.polarity < 0 ? 1 : -1, on: true })}
                    >
                      {r.polarity < 0 ? '−' : '+'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {sourceIds.length === 1 && <div className="small muted">This simulation has a single source at this receiver: the sum only applies its gain, delay and polarity.</div>}
      <div className="row gap">
        <button type="button" className="btn btn-sm btn-primary" disabled={!chosen.length || busy} onClick={() => void submit()} title="Sum the selected sources and add the result to the comparisons">
          {busy ? <Spinner size={11} /> : null} {editing ? 'Update the sum' : 'Sum sources'}
        </button>
        <span className="small muted">{chosen.length ? `${chosen.length} source${chosen.length === 1 ? '' : 's'}` : 'Tick at least one source'}</span>
      </div>
      {error && <div className="small text-err">{error}</div>}
      {last && !error && (
        <div className="small combine-last">
          <b>Σ {last.name}</b> {editing ? 'updated' : 'added to the comparisons'}: SPL {fmtNum(last.result.metrics.SPL_dbz ?? null, 1)} dB(Z), STI {fmtNum(last.result.metrics.STI ?? null, 2)}.{' '}
          <a href={last.result.wav_url} download>
            Summed IR (.wav)
          </a>
        </div>
      )}
    </section>
  );
}
