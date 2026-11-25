import type { Results, Simulation } from '../api';
import { distance } from './geom';

/** Positions closer than this count as identical. */
const POS_TOL_M = 0.01;
const ANGLE_TOL_DEG = 1;

export interface Compatibility {
  ok: boolean;
  /** Why the simulation cannot be swapped in (positions, missing points or pairs, no results). */
  reasons: string[];
  /** Swappable, but the listener should know (orientation, stale results). */
  warnings: string[];
}

const angleDiff = (a: number, b: number) => Math.abs((((a - b) % 360) + 540) % 360 - 180);

/**
 * Can simulation B replace A in the auralizer? Same space is checked by the caller;
 * here: same source and receiver ids at the same positions, and every pair of A computed in B.
 */
export function compatibility(a: Simulation, aRes: Results, b: Simulation, bRes: Results | null): Compatibility {
  const reasons: string[] = [];
  const warnings: string[] = [];
  if (!bRes) return { ok: false, reasons: ['no results: run it first'], warnings };
  const aSrc = new Set(aRes.pairs.map((p) => p.source_id));
  const aRcv = new Set(aRes.pairs.map((p) => p.receiver_id));
  const bSrc = new Set(bRes.pairs.map((p) => p.source_id));
  const bRcv = new Set(bRes.pairs.map((p) => p.receiver_id));
  const check = (kind: 'source' | 'receiver', ids: Set<string>, aList: Simulation['sources'] | Simulation['receivers'], bList: Simulation['sources'] | Simulation['receivers']) => {
    for (const id of ids) {
      const pa = aList.find((x) => x.id === id);
      const pb = bList.find((x) => x.id === id);
      const name = pa?.name ?? id;
      if (!pa) continue;
      if (!pb) {
        reasons.push(`${kind} “${name}” (${id}) does not exist in it`);
        continue;
      }
      const d = distance(pa.position, pb.position);
      if (d > POS_TOL_M) reasons.push(`${kind} “${name}” is ${d < 0.995 ? `${Math.round(d * 100)} cm` : `${d.toFixed(2)} m`} away from its position here`);
      if (kind === 'receiver' && (angleDiff(pa.azimuth_deg, pb.azimuth_deg) > ANGLE_TOL_DEG || Math.abs(pa.elevation_deg - pb.elevation_deg) > ANGLE_TOL_DEG)) {
        warnings.push(`receiver “${name}” looks another way (${Math.round(pb.azimuth_deg)}° vs ${Math.round(pa.azimuth_deg)}°): the binaural IRs differ by that turn too`);
      }
    }
  };
  check('source', aSrc, a.sources, b.sources);
  check('receiver', aRcv, a.receivers, b.receivers);
  const extraS = [...bSrc].filter((x) => !aSrc.has(x)).length;
  const extraR = [...bRcv].filter((x) => !aRcv.has(x)).length;
  if (extraS) reasons.push(`it has ${extraS} source(s) that this simulation does not have`);
  if (extraR) reasons.push(`it has ${extraR} receiver(s) that this simulation does not have`);
  if (!reasons.length) {
    const missing = aRes.pairs.filter((p) => !bRes.pairs.some((q) => q.source_id === p.source_id && q.receiver_id === p.receiver_id));
    if (missing.length) reasons.push(`${missing.length} source–receiver pair(s) were not computed in it`);
  }
  if (bRes.stale) warnings.push('its results are stale (edited after the run)');
  return { ok: reasons.length === 0, reasons, warnings };
}
