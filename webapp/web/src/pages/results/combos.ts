import { useEffect, useMemo, useState } from 'react';
import { api, errorMessage, type CombineResult } from '../../api';
import type { ComboDef, ComboState, CompCard } from './cards';

const cache = new Map<string, Promise<CombineResult>>();

function defKey(simId: string, def: ComboDef): string {
  return JSON.stringify([simId, def.receiverId, def.sources]);
}

/** POST /combine once per simulation and definition (the server caches the sum until the next run). */
export function combineCached(simId: string, def: ComboDef): Promise<CombineResult> {
  const k = defKey(simId, def);
  let p = cache.get(k);
  if (!p) {
    p = api.combineSources(simId, { receiver_id: def.receiverId, sources: def.sources });
    cache.set(k, p);
    p.catch(() => cache.delete(k));
  }
  return p;
}

/** Results of every summed card, by card key. */
export function useCombos(cards: CompCard[]): Map<string, ComboState> {
  const [state, setState] = useState<Map<string, ComboState>>(new Map());
  const list = useMemo(() => cards.filter((c) => c.combo).map((c) => ({ key: c.key, simId: c.simId, def: c.combo! })), [cards]);
  const sig = JSON.stringify(list.map((x) => [x.key, defKey(x.simId, x.def)]));
  useEffect(() => {
    let cancelled = false;
    const items = JSON.parse(sig) as [string, string][];
    setState((prev) => {
      const n = new Map<string, ComboState>();
      for (const [key] of items) {
        const old = prev.get(key);
        n.set(key, old ? { ...old, loading: true } : { result: null, error: null, loading: true });
      }
      return n;
    });
    for (const x of list) {
      combineCached(x.simId, x.def)
        .then((result) => !cancelled && setState((prev) => new Map(prev).set(x.key, { result, error: null, loading: false })))
        .catch((e) => !cancelled && setState((prev) => new Map(prev).set(x.key, { result: null, error: `Sum of sources: ${errorMessage(e)}`, loading: false })));
    }
    return () => {
      cancelled = true;
    };
    // `list` is described by `sig`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);
  return state;
}

/** "S1 + S2 + …" (shortened past three names). */
export function autoComboName(names: string[]): string {
  if (names.length <= 3) return names.join(' + ');
  return `${names.slice(0, 3).join(' + ')} + ${names.length - 3} more`;
}
