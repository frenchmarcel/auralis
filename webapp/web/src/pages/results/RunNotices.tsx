import type { ReactElement } from 'react';
import { bandLabel } from '../../lib/format';
import { href } from '../../lib/router';
import { COMBO_PREFIX, type ResolvedCard } from './cards';
import { BANDS_HZ, type Results, type Simulation } from '../../api';

/** Above this decay (dB) at the end of the IR, T30 is unreliable (API: decay_range_db). */
export const T30_DECAY_LIMIT_DB = -45;

/** Bands of a pair whose IR ends before a 45 dB decay. */
export function shortDecayBands(decay: (number | null)[] | undefined): number[] {
  return (decay ?? []).flatMap((v, i) => (v !== null && v !== undefined && v > T30_DECAY_LIMIT_DB ? [i] : []));
}

/**
 * Run-level warnings of every simulation shown: sources whose wave band failed, pairs
 * whose IR is too short for T30, and which IR the ISO parameters come from.
 */
export function RunNotices({ cards }: { cards: ResolvedCard[] }) {
  const seen = new Map<string, { sim: Simulation | null; results: Results }>();
  for (const c of cards) if (c.results && !seen.has(c.card.simId)) seen.set(c.card.simId, { sim: c.sim, results: c.results });
  const many = seen.size > 1;
  const out: ReactElement[] = [];
  for (const [simId, { sim, results }] of seen) {
    const prefix = many ? `${sim?.name ?? simId}: ` : '';
    const srcName = (id: string) => sim?.sources.find((s) => s.id === id)?.name ?? id;
    const rcvName = (id: string) => sim?.receivers.find((r) => r.id === id)?.name ?? id;
    const failed = results.wave_failed ?? [];
    if (failed.length) {
      const errors = Array.from(new Set(failed.map((f) => f.error)));
      out.push(
        <div key={`${simId}:wave`} className="notice notice-warn run-notice">
          <div>
            {prefix}
            <b>The wave solver failed for {failed.length === 1 ? '1 source' : `${failed.length} sources`}</b> ({failed.map((f) => srcName(f.source_id)).join(', ')}):{' '}
            {failed.length === 1 ? 'it uses' : 'they use'} the geometrical solver over the whole band, so below the transition frequency {failed.length === 1 ? 'its' : 'their'} results are geometrical, not wave-based.
          </div>
          <details className="small">
            <summary>Why</summary>
            {errors.map((e, i) => (
              <p key={i} className="mono-wrap">
                {e}
              </p>
            ))}
          </details>
        </div>,
      );
    }
    const short = results.pairs
      .filter((p) => !p.source_id.startsWith(COMBO_PREFIX))
      .map((p) => ({ p, bands: shortDecayBands(p.metrics.decay_range_db) }))
      .filter((x) => x.bands.length);
    if (short.length) {
      out.push(
        <div key={`${simId}:decay`} className="notice notice-warn run-notice">
          <div>
            {prefix}
            <b>T30 unreliable here: lengthen the IR.</b> The IR ends before a 45 dB decay in{' '}
            {short.slice(0, 6).map((x, i) => (
              <span key={i}>
                {i > 0 ? '; ' : ''}
                {srcName(x.p.source_id)} → {rcvName(x.p.receiver_id)} ({x.bands.map((b) => bandLabel(BANDS_HZ[b])).join(', ')} Hz)
              </span>
            ))}
            {short.length > 6 ? `; and ${short.length - 6} more pairs` : ''}.
          </div>
          <div className="small">
            In the <a href={href.editor(simId)}>editor</a>, Settings › Advanced: a longer IR length or a decay criterion, then run again. The T30 cells concerned are marked †.
          </div>
        </div>,
      );
    }
    if (results.metrics_ir !== 'omni') {
      out.push(
        <div key={`${simId}:ir`} className="notice notice-info run-notice small">
          {prefix}Parameters from the mean of the ears (run before v0.5). Run the simulation again to get them from the omnidirectional IR, as an omni measurement microphone would.
        </div>,
      );
    }
  }
  return out.length ? <>{out}</> : null;
}
