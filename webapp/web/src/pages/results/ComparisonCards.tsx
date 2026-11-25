import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type ResultPart } from '../../api';
import { Menu, Spinner } from '../../components/ui';
import { downloadUrl } from '../../lib/export';
import type { SimEntry } from '../../lib/simdata';
import { PART_LABELS, resultReceivers, resultSources, type ComboState, type CompCard, type ResolvedCard } from './cards';

const PARTS: ResultPart[] = ['hybrid', 'geo', 'wave'];

/** A run of the wave solver alone (Advanced settings): its one result is wave-based. */
function waveOnly(rc: ResolvedCard): boolean {
  const st = rc.results?.settings;
  return !!st && st.mode === 'advanced' && st.solvers === 'wave';
}

/** Why a card offers the hybrid result only (null when it has the three result types). */
function noSplitReason(rc: ResolvedCard): string | null {
  if (rc.card.combo) return 'A sum of sources has the hybrid result only.';
  if (!rc.results) return rc.loading ? 'Loading…' : 'No results.';
  if (!rc.pairs.length) return 'No result for this pair.';
  if (waveOnly(rc))
    return 'Wave solver only in this run: the result is the wave band alone (octaves above the transition are –; no STI, J_LF, L_J or overall level).';
  if (!rc.results.transition_hz)
    return rc.results.wave_failed?.length ? 'The wave solver failed in this run: the result is geometrical only.' : 'Geometrical solver only in this run: there is no wave band to separate.';
  if (rc.results.wave_failed?.some((f) => f.source_id === rc.card.sourceId)) return 'The wave solver failed for this source: its result is geometrical only.';
  if (!rc.split) return 'This run has no separate geometrical and wave results (wave-solver runs from v0.6 have them): run it again to compare them.';
  return null;
}

export function ComparisonCards({
  cards,
  activeKey,
  onActivate,
  onChange,
  onRemove,
  onMove,
  onAdd,
  entries,
  index,
  currentSimId,
  combos,
  onEditCombo,
}: {
  cards: ResolvedCard[];
  activeKey: string | null;
  onActivate: (key: string) => void;
  onChange: (key: string, patch: Partial<CompCard>) => void;
  onRemove: (key: string) => void;
  onMove: (key: string, dir: -1 | 1) => void;
  onAdd: () => void;
  entries: SimEntry[];
  index: Map<string, boolean | 'error'>;
  currentSimId: string;
  /** Results of the summed cards ("Σ"), by card key. */
  combos?: Map<string, ComboState>;
  onEditCombo?: (key: string) => void;
}) {
  const groups = useMemo(() => {
    const m = new Map<string, SimEntry[]>();
    for (const e of entries) {
      const l = m.get(e.path) ?? [];
      l.push(e);
      m.set(e.path, l);
    }
    return [...m.entries()];
  }, [entries]);
  const probing = entries.some((e) => !index.has(e.id));

  return (
    <div className="cards-bar no-print" role="list" aria-label="Comparisons">
      {cards.map((rc, i) => {
        const c = rc.card;
        const srcs = resultSources(rc.results);
        const active = c.key === activeKey;
        const combo = c.combo ?? null;
        const comboRes = combo ? (combos?.get(c.key)?.result ?? null) : null;
        return (
          <div
            key={c.key}
            role="listitem"
            className={`comp-card ${active ? 'active' : ''}`}
            style={{ borderTopColor: rc.color }}
            onMouseDown={() => onActivate(c.key)}
            onFocus={() => onActivate(c.key)}
          >
            <div className="comp-head">
              <span className="swatch" style={{ background: rc.color }} />
              <input
                className="comp-label"
                value={c.label}
                placeholder={rc.label}
                title={rc.label}
                aria-label="Comparison label"
                onChange={(e) => onChange(c.key, { label: e.target.value })}
              />
              <Menu
                items={[
                  { label: 'Move left', onClick: () => onMove(c.key, -1), disabled: i === 0 },
                  { label: 'Move right', onClick: () => onMove(c.key, 1), disabled: i === cards.length - 1 },
                  ...(combo
                    ? [
                        { label: 'Edit the sum…', onClick: () => onEditCombo?.(c.key), disabled: !onEditCombo },
                        { label: 'Download the summed IR (.wav)', onClick: () => comboRes && downloadUrl(comboRes.wav_url), disabled: !comboRes },
                      ]
                    : []),
                  { label: 'Download parameters (.xlsx)', onClick: () => downloadUrl(api.exportXlsxUrl(c.simId)), disabled: !rc.results },
                  { label: 'Download IRs (.zip)', onClick: () => downloadUrl(api.irsZipUrl(c.simId)), disabled: !rc.results },
                  { label: 'Remove', onClick: () => onRemove(c.key), danger: true, disabled: cards.length <= 1, title: cards.length <= 1 ? 'Keep at least one comparison' : undefined },
                ]}
              />
            </div>
            <label className="comp-field">
              <span>Simulation</span>
              <select
                value={c.simId}
                onChange={(e) => onChange(c.key, { simId: e.target.value })}
                title={combo ? 'A sum of sources belongs to its simulation' : entries.find((x) => x.id === c.simId)?.path}
                disabled={!!combo}
              >
                {!entries.some((e) => e.id === c.simId) && <option value={c.simId}>{rc.sim?.name ?? c.simId}</option>}
                {groups.map(([path, list]) => (
                  <optgroup key={path} label={path}>
                    {list.map((e) => {
                      const has = index.get(e.id);
                      const off = has === false || has === 'error';
                      return (
                        <option key={e.id} value={e.id} disabled={off && e.id !== c.simId}>
                          {e.name}
                          {e.id === currentSimId ? ' (this one)' : ''}
                          {has === false ? ' — no results' : has === 'error' ? ' — unavailable' : has === undefined ? ' …' : ''}
                        </option>
                      );
                    })}
                  </optgroup>
                ))}
              </select>
            </label>
            {combo ? (
              <>
                <div className="comp-field">
                  <span>Sum</span>
                  <div className="rcv-picker">
                    <button type="button" className="rcv-picker-btn" onClick={() => onEditCombo?.(c.key)} title={comboTitle(rc, combo.sources)}>
                      <span className="ellipsis">{rc.srcName}</span>
                      <span className="caret">✎</span>
                    </button>
                  </div>
                </div>
                <div className="comp-field">
                  <span>Receiver</span>
                  <span className="comp-fixed ellipsis">{rc.rcvLabel}</span>
                </div>
              </>
            ) : (
              <>
                <label className="comp-field">
                  <span>Source</span>
                  <select value={c.sourceId ?? ''} disabled={!srcs.length} onChange={(e) => onChange(c.key, { sourceId: e.target.value })}>
                    {srcs.map((s) => (
                      <option key={s} value={s}>
                        {rc.sim?.sources.find((x) => x.id === s)?.name ?? s}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="comp-field">
                  <span>Receiver</span>
                  <ReceiverPicker rc={rc} onChange={(ids) => onChange(c.key, { receiverIds: ids })} />
                </div>
              </>
            )}
            <ResultTypeField rc={rc} onChange={(part) => onChange(c.key, { part })} />
            {rc.loading && (
              <div className="small muted">
                <Spinner size={10} /> loading…
              </div>
            )}
            {rc.error && <div className="small text-err">{rc.error}</div>}
            {!rc.error && rc.results?.stale && <div className="small text-warn">Stale: edited after this run.</div>}
            {!rc.error && rc.results && !rc.pairs.length && <div className="small text-warn">No result for this pair.</div>}
          </div>
        );
      })}
      <button type="button" className="comp-add" onClick={onAdd} title="Add a comparison: any simulation of the projects that has results">
        <span className="comp-add-plus">+</span>
        Add comparison
        {probing && (
          <span className="small muted">
            <Spinner size={10} /> finding results…
          </span>
        )}
      </button>
    </div>
  );
}

function ResultTypeField({ rc, onChange }: { rc: ResolvedCard; onChange: (p: ResultPart) => void }) {
  const why = noSplitReason(rc);
  const wanted = rc.card.part ?? 'hybrid';
  const tr = rc.results?.transition_hz;
  const title =
    why ??
    `Hybrid: both solvers joined at the transition (${tr ? Math.round(tr) : '–'} Hz). Geometrical: the geometrical solver alone over the full band. Wave-based: the wave band alone, below the transition (bands above it are –). G, SPL, STI, J_LF and L_J exist for the hybrid result only.`;
  return (
    <>
      <label className="comp-field" title={title}>
        <span>Result type</span>
        <select value={rc.part} disabled={!!why} title={title} onChange={(e) => onChange(e.target.value as ResultPart)} aria-label="Result type">
          {PARTS.map((p) => (
            <option key={p} value={p}>
              {p === 'hybrid' && waveOnly(rc) ? PART_LABELS.wave : PART_LABELS[p]}
            </option>
          ))}
        </select>
      </label>
      {wanted !== rc.part && why && !rc.loading && !waveOnly(rc) && (
        <div className="small text-warn" title={why}>
          {PART_LABELS[wanted]} result not available for this run: hybrid shown.
        </div>
      )}
    </>
  );
}

function comboTitle(rc: ResolvedCard, sources: { source_id: string; gain_db: number; delay_ms: number; polarity: number }[]): string {
  const name = (id: string) => rc.sim?.sources.find((x) => x.id === id)?.name ?? id;
  return `Coherent sum (click to edit):\n${sources.map((s) => `${name(s.source_id)}: ${s.gain_db >= 0 ? '+' : ''}${s.gain_db} dB, ${s.delay_ms} ms, polarity ${s.polarity < 0 ? '−' : '+'}`).join('\n')}`;
}

function ReceiverPicker({ rc, onChange }: { rc: ResolvedCard; onChange: (ids: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const all = resultReceivers(rc.results, rc.card.sourceId);
  const sel = rc.card.receiverIds;
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopImmediatePropagation();
        setOpen(false);
      }
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [open]);
  const toggle = (id: string) => {
    if (sel.includes(id)) {
      if (sel.length > 1) onChange(sel.filter((x) => x !== id));
    } else onChange(all.filter((x) => x === id || sel.includes(x)));
  };
  return (
    <div className="rcv-picker" ref={ref}>
      <button type="button" className="rcv-picker-btn" disabled={!all.length} aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="ellipsis">{rc.rcvLabel}</span>
        <span className="caret">▾</span>
      </button>
      {open && (
        <div className="rcv-picker-list" role="listbox" aria-multiselectable="true">
          <div className="rcv-picker-actions small">
            <button type="button" className="link small" onClick={() => onChange(all)}>
              Average of all
            </button>
            <span className="muted">click a name for one receiver; tick several to average</span>
          </div>
          {all.map((id) => (
            <div key={id} className={`rcv-picker-row ${sel.includes(id) ? 'on' : ''}`} role="option" aria-selected={sel.includes(id)}>
              <input type="checkbox" checked={sel.includes(id)} onChange={() => toggle(id)} aria-label={`Include ${rc.rcvName(id)} in the average`} />
              <button
                type="button"
                className="rcv-picker-name"
                onClick={() => {
                  onChange([id]);
                  setOpen(false);
                }}
              >
                {rc.rcvName(id)}
              </button>
            </div>
          ))}
          <div className="small muted rcv-picker-note">Averages: arithmetic for times, ratios and STI; energetic for levels (SPL, G, L<sub>J</sub>).</div>
        </div>
      )}
    </div>
  );
}
