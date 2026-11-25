import { useEffect, useMemo, useRef, useState } from 'react';
import { api, errorMessage, type GridPoints, type Simulation } from '../../api';

export interface GridPreview {
  data: GridPoints | null;
  error: string | null;
  loading: boolean;
}

/**
 * Point previews of every grid (GET .../grids/{id}/points), fetched once the grid as
 * edited here is stored on the server. Re-fetched when the grid, the sources (points
 * < 0.5 m from a source are disabled) or the geometry change; the last points stay
 * shown while a new preview loads.
 */
export function useGridPreviews(sim: Simulation | null, saved: boolean, meshKey: unknown): Map<string, GridPreview> {
  const [state, setState] = useState<Map<string, GridPreview>>(new Map());
  /** Latest requested key per grid: answers to older requests are dropped. */
  const requested = useRef(new Map<string, string>());
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const meshSeq = useRef({ key: meshKey, n: 0 });
  if (meshSeq.current.key !== meshKey) meshSeq.current = { key: meshKey, n: meshSeq.current.n + 1 };
  const meshN = meshSeq.current.n;
  const simId = sim?.id ?? null;
  const grids = useMemo(() => sim?.receiver_grids ?? [], [sim?.receiver_grids]);
  const context = sim ? JSON.stringify([sim.sources.map((s) => s.position), !!sim.settings.open_scene, meshN]) : '';
  const keys = useMemo(() => new Map(grids.map((g) => [g.id, `${JSON.stringify(g)}|${context}`])), [grids, context]);

  useEffect(() => {
    setState((prev) => {
      if ([...prev.keys()].every((k) => keys.has(k))) return prev;
      const n = new Map(prev);
      for (const k of prev.keys()) if (!keys.has(k)) n.delete(k);
      return n;
    });
    for (const k of [...requested.current.keys()]) if (!keys.has(k)) requested.current.delete(k);
    if (!simId || !saved) return;
    for (const [gridId, key] of keys) {
      if (requested.current.get(gridId) === key) continue;
      requested.current.set(gridId, key);
      setState((prev) => new Map(prev).set(gridId, { data: prev.get(gridId)?.data ?? null, error: null, loading: true }));
      api
        .getGridPoints(simId, gridId)
        .then((data) => {
          if (!alive.current || requested.current.get(gridId) !== key) return;
          setState((prev) => new Map(prev).set(gridId, { data, error: null, loading: false }));
        })
        .catch((e) => {
          if (!alive.current || requested.current.get(gridId) !== key) return;
          setState((prev) => new Map(prev).set(gridId, { data: null, error: errorMessage(e), loading: false }));
        });
    }
  }, [simId, saved, keys]);

  return state;
}
