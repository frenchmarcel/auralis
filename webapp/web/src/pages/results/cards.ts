import type { CombineResult, CombineSource, PairMetrics, ResultPair, ResultPart, Results, Simulation, SolverMetricKey } from '../../api';
import { seriesColor } from '../../lib/colors';
import { combinePairs, NULL_BANDS, type CardValues } from '../../lib/metrics';
import type { BundleState } from '../../lib/simdata';

/** One comparison: a simulation, a source and one receiver (or the average of several). */
export interface CompCard {
  key: string;
  simId: string;
  sourceId: string | null;
  receiverIds: string[];
  /** Index into the series palette, kept when cards are reordered or removed. */
  color: number;
  /** User label; empty = automatic. */
  label: string;
  /** Coherent sum of sources at one receiver (the card then shows "Σ name"). */
  combo?: ComboDef;
  /** Result type: hybrid (default), geometrical solver alone or wave band alone (runs with `by_solver`). */
  part?: ResultPart;
}

export const PART_LABELS: Record<ResultPart, string> = { hybrid: 'Hybrid', geo: 'Geometrical', wave: 'Wave-based' };
/** The parameters the server gives per solver; the others exist for the hybrid result only. */
export const SOLVER_KEYS: SolverMetricKey[] = ['T20', 'T30', 'EDT', 'C50', 'C80', 'D50', 'Ts_ms'];

/** Whether every pair has the geometrical and wave parts (wave-solver runs, v0.6+). */
export function hasSplit(pairs: ResultPair[]): boolean {
  return pairs.length > 0 && pairs.every((p) => !!p.metrics?.by_solver?.geo && !!p.metrics?.by_solver?.wave);
}

/** A pair's metrics for one solver: the split parameters, 63 Hz for the wave part, nothing else. */
export function partMetrics(m: PairMetrics, part: Exclude<ResultPart, 'hybrid'>): PairMetrics {
  const b = m.by_solver?.[part] ?? {};
  const out: PairMetrics = {};
  for (const k of SOLVER_KEYS) out[k] = b[k] ?? NULL_BANDS;
  out.band63 = part === 'wave' ? (m.band63 ?? null) : null;
  return out;
}

/** Definition of a coherent sum: what is POSTed to /combine, plus its name. */
export interface ComboDef {
  receiverId: string;
  sources: CombineSource[];
  name: string;
}

export interface ComboState {
  result: CombineResult | null;
  error: string | null;
  loading: boolean;
}

/** Source id of a summed card's synthetic pair; the curves of `Σ:<key>` are the sum's. */
export const COMBO_PREFIX = 'Σ:';

export interface ResolvedCard {
  card: CompCard;
  color: string;
  label: string;
  sim: Simulation | null;
  results: Results | null;
  loading: boolean;
  error: string | null;
  pairs: ResultPair[];
  values: CardValues | null;
  srcName: string;
  rcvLabel: string;
  rcvName: (id: string) => string;
  /** Result type shown (hybrid when the run has no split). */
  part: ResultPart;
  /** The run has separate geometrical and wave results. */
  split: boolean;
}

let seq = 0;
export function newCardKey(): string {
  seq += 1;
  return `c${Date.now().toString(36)}${seq}`;
}

export function firstFreeColor(cards: CompCard[]): number {
  const used = new Set(cards.map((c) => c.color));
  for (let i = 0; ; i++) if (!used.has(i)) return i;
}

export function resultSources(r: Results | null): string[] {
  return Array.from(new Set(r?.pairs.map((p) => p.source_id) ?? []));
}

export function resultReceivers(r: Results | null, sourceId?: string | null): string[] {
  return Array.from(new Set(r?.pairs.filter((p) => !sourceId || p.source_id === sourceId).map((p) => p.receiver_id) ?? []));
}

/** Keep the card's source/receivers when they exist in its results, else fall back to the first ones. */
export function sanitizeCard(c: CompCard, r: Results | null): CompCard {
  if (!r || c.combo) return c;
  const srcs = resultSources(r);
  const sourceId = c.sourceId && srcs.includes(c.sourceId) ? c.sourceId : (srcs[0] ?? null);
  const rcvs = resultReceivers(r, sourceId);
  let receiverIds = c.receiverIds.filter((x) => rcvs.includes(x));
  if (!receiverIds.length && rcvs.length) receiverIds = [rcvs[0]];
  if (sourceId === c.sourceId && receiverIds.length === c.receiverIds.length && receiverIds.every((x, i) => x === c.receiverIds[i])) return c;
  return { ...c, sourceId, receiverIds };
}

export function resolveCards(cards: CompCard[], bundles: Map<string, BundleState>, combos?: Map<string, ComboState>): ResolvedCard[] {
  const multiSim = new Set(cards.map((c) => c.simId)).size > 1;
  return cards.map((card) => {
    const st = bundles.get(card.simId);
    const sim = st?.data?.sim ?? null;
    const results = st?.data?.results ?? null;
    const srcName = (id: string | null) => (id ? (sim?.sources.find((s) => s.id === id)?.name ?? id) : '–');
    const rcvName = (id: string) => sim?.receivers.find((r) => r.id === id)?.name ?? id;
    if (card.combo) {
      const cs = combos?.get(card.key);
      const res = cs?.result ?? null;
      const pairs: ResultPair[] = res && results ? [{ source_id: `${COMBO_PREFIX}${res.key}`, receiver_id: card.combo.receiverId, metrics: res.metrics, timings: {} }] : [];
      const sName = `Σ ${card.combo.name}`;
      const rLabel = rcvName(card.combo.receiverId);
      let error = st?.error ?? cs?.error ?? null;
      if (!error && st?.data && !results) error = 'This simulation has no results.';
      return {
        card,
        color: seriesColor(card.color),
        label: card.label.trim() || `${multiSim && sim ? `${sim.name} · ` : ''}${sName} → ${rLabel}`,
        sim,
        results,
        loading: !st || !!st.loading || !!cs?.loading || (!cs && !error),
        error,
        pairs,
        values: pairs.length ? combinePairs(pairs) : null,
        srcName: sName,
        rcvLabel: rLabel,
        rcvName,
        part: 'hybrid',
        split: false,
      };
    }
    const all = results
      ? card.receiverIds
          .map((r) => results.pairs.find((p) => p.source_id === card.sourceId && p.receiver_id === r))
          .filter((p): p is ResultPair => !!p)
      : [];
    const split = hasSplit(all);
    const part: ResultPart = card.part && card.part !== 'hybrid' && split ? card.part : 'hybrid';
    const pairs = part === 'hybrid' ? all : all.map((p) => ({ ...p, metrics: partMetrics(p.metrics, part) }));
    const nAll = resultReceivers(results, card.sourceId).length;
    const rcvLabel =
      card.receiverIds.length === 0
        ? '–'
        : card.receiverIds.length === 1
          ? rcvName(card.receiverIds[0])
          : card.receiverIds.length === nAll
            ? `average of all ${nAll} receivers`
            : `average of ${card.receiverIds.length} receivers`;
    const auto = `${multiSim && sim ? `${sim.name} · ` : ''}${srcName(card.sourceId)} → ${rcvLabel}${part !== 'hybrid' ? ` · ${PART_LABELS[part]}` : ''}`;
    let error = st?.error ?? null;
    if (!error && st?.data && !results) error = 'This simulation has no results.';
    return {
      card,
      color: seriesColor(card.color),
      label: card.label.trim() || auto,
      sim,
      results,
      loading: !st || !!st.loading,
      error,
      pairs,
      values: pairs.length ? combinePairs(pairs) : null,
      srcName: srcName(card.sourceId),
      rcvLabel,
      rcvName,
      part,
      split,
    };
  });
}

export interface SavedView {
  cards: CompCard[];
  activeKey?: string;
  param?: string;
  plotType?: string;
  edcBands?: number[];
  smoothing?: number;
  irDb?: boolean;
  channel?: string;
  showTransition?: boolean;
  view?: string;
}

/** A saved view as read back (server or browser copy): null unless it has a list of cards. */
export function validView(raw: unknown): SavedView | null {
  if (!raw || typeof raw !== 'object') return null;
  const v = { ...(raw as SavedView) };
  if (!Array.isArray(v.cards)) return null;
  v.cards = v.cards.filter((c) => c && typeof c.simId === 'string' && Array.isArray(c.receiverIds) && typeof c.key === 'string');
  return v;
}
