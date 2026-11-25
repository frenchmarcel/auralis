import { useCallback, useEffect, useRef, useState } from 'react';
import { api, errorMessage, type Geometry, type MeshData, type PlacedObject, type Simulation } from '../../api';
import { cacheGeometry, cacheMesh } from '../../lib/cache';
import { sameObjects } from './objectEditing';
import type { SimulationStore } from './useSimulationStore';

export type ObjectsSyncState = 'saved' | 'pending' | 'saving' | 'error';
export type EditMode = 'debounced' | 'live' | 'now';

const DELAY_MS = 300;
const HISTORY_MAX = 100;

interface History {
  gid: string | null;
  /** Object lists sent to the server, oldest first; `idx` is the one shown when nothing is pending. */
  steps: PlacedObject[][];
  idx: number;
}

/**
 * Local list of placed objects, saved as a whole with PUT /geometries/{gid}/objects.
 * Before each PUT the pending simulation autosave is flushed and further
 * simulation saves are paused; after it, geometry, mesh and simulation are
 * re-fetched (the server adds or removes object layers in every simulation).
 * Every list sent is an undo step; `step` moves through them (the caller re-sends
 * the list it returns with `edit`).
 */
export function useObjectsSync({
  geometry,
  simId,
  store,
  onServerGeometry,
  onServerMesh,
  onError,
}: {
  geometry: Geometry | null;
  simId: string;
  store: SimulationStore;
  onServerGeometry: (g: Geometry) => void;
  onServerMesh: (m: MeshData) => void;
  onError: (msg: string) => void;
}) {
  const [objects, setObjects] = useState<PlacedObject[]>([]);
  const [state, setState] = useState<ObjectsSyncState>('saved');
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<PlacedObject[]>([]);
  const server = useRef<PlacedObject[]>([]);
  const version = useRef(0);
  const saved = useRef(0);
  const live = useRef(false);
  const timer = useRef<number | undefined>(undefined);
  const inflight = useRef<Promise<unknown> | null>(null);
  const hist = useRef<History>({ gid: null, steps: [[]], idx: 0 });
  const [, setHistTick] = useState(0);
  const latest = useRef({ store, simId, onServerGeometry, onServerMesh, onError, gid: geometry?.id ?? null });
  latest.current = { store, simId, onServerGeometry, onServerMesh, onError, gid: geometry?.id ?? null };

  const commit = useCallback((list: PlacedObject[]) => {
    const h = hist.current;
    if (sameObjects(list, h.steps[h.idx])) return;
    h.steps = [...h.steps.slice(0, h.idx + 1), list].slice(-(HISTORY_MAX + 1));
    h.idx = h.steps.length - 1;
    setHistTick((t) => t + 1);
  }, []);

  /** After a revert to the server's list: point the history at the step showing it. */
  const resync = useCallback(() => {
    const h = hist.current;
    const cur = ref.current;
    if (sameObjects(cur, h.steps[h.idx])) return;
    for (let d = 1; d < h.steps.length; d++) {
      for (const j of [h.idx - d, h.idx + d]) {
        if (j >= 0 && j < h.steps.length && sameObjects(cur, h.steps[j])) {
          h.idx = j;
          setHistTick((t) => t + 1);
          return;
        }
      }
    }
    commit(cur);
  }, [commit]);

  useEffect(() => {
    if (!geometry) return;
    server.current = geometry.objects ?? [];
    if (hist.current.gid !== geometry.id) {
      hist.current = { gid: geometry.id, steps: [server.current], idx: 0 };
      setHistTick((t) => t + 1);
    }
    if (version.current === saved.current) {
      ref.current = server.current;
      setObjects(server.current);
    }
  }, [geometry]);

  /** `force` false (the debounce timer): leave changes of a gizmo drag still going on to its release. */
  const flush = useCallback(
    async (force = true): Promise<boolean> => {
      window.clearTimeout(timer.current);
      for (;;) {
        if (inflight.current) {
          await inflight.current;
          continue;
        }
        const { gid } = latest.current;
        if (!gid) return true;
        const v = version.current;
        if (v === saved.current) return true;
        if (force !== true && live.current) return true;
        const list = ref.current;
        if (sameObjects(list, server.current)) {
          // Back to what the server has (e.g. an undo before the change was sent): nothing to recompute.
          saved.current = v;
          setState(version.current === saved.current ? 'saved' : 'pending');
          continue;
        }
        commit(list);
        setState('saving');
        const run = (async () => {
          const { store: st, simId: sid } = latest.current;
          if (!(await st.flush())) throw new Error('the simulation could not be saved first, so the objects were not applied');
          st.pause();
          let fresh: Simulation | null = null;
          try {
            const g = await api.putObjects(gid, list);
            saved.current = v;
            server.current = g.objects ?? list;
            cacheGeometry(g);
            latest.current.onServerGeometry(g);
            const [m, sim] = await Promise.all([api.getMesh(gid), api.getSimulation(sid)]);
            cacheMesh(gid, m);
            latest.current.onServerMesh(m);
            fresh = sim;
          } finally {
            st.resume(fresh);
          }
        })();
        inflight.current = run.catch(() => undefined);
        try {
          await run;
          inflight.current = null;
          setError(null);
          setState(version.current === saved.current ? 'saved' : 'pending');
        } catch (e) {
          inflight.current = null;
          const msg = errorMessage(e);
          // Back to the server's state, so the viewport matches the solver mesh.
          if (saved.current !== v) {
            ref.current = server.current;
            setObjects(server.current);
            saved.current = version.current;
            live.current = false;
            resync();
          }
          setError(msg);
          setState('error');
          latest.current.onError(`Objects not saved: ${msg}`);
          return false;
        }
      }
    },
    [commit, resync],
  );

  const edit = useCallback(
    (fn: (list: PlacedObject[]) => PlacedObject[], mode: EditMode = 'debounced') => {
      const next = fn(ref.current);
      if (next === ref.current) return;
      ref.current = next;
      setObjects(next);
      version.current++;
      live.current = mode === 'live';
      setState('pending');
      window.clearTimeout(timer.current);
      if (mode === 'live') return;
      timer.current = window.setTimeout(() => void flush(false), mode === 'now' ? 0 : DELAY_MS);
    },
    [flush],
  );

  /** Drop the local edits not saved yet (back to the server's list). */
  const discard = useCallback(() => {
    window.clearTimeout(timer.current);
    if (inflight.current) return;
    ref.current = server.current;
    setObjects(server.current);
    saved.current = version.current;
    live.current = false;
    setState('saved');
    resync();
  }, [resync]);

  /**
   * The list one undo (-1) or redo (+1) step away, or null at either end. Local
   * changes not sent yet become a step first, so that a redo brings them back.
   */
  const step = useCallback(
    (dir: -1 | 1): PlacedObject[] | null => {
      const h = hist.current;
      if (!sameObjects(ref.current, h.steps[h.idx])) {
        if (dir > 0) return null;
        commit(ref.current);
      }
      const j = h.idx + dir;
      if (j < 0 || j >= h.steps.length) return null;
      h.idx = j;
      setHistTick((t) => t + 1);
      return h.steps[j];
    },
    [commit],
  );

  useEffect(
    () => () => {
      window.clearTimeout(timer.current);
      const gid = latest.current.gid;
      if (gid && version.current !== saved.current) api.putObjects(gid, ref.current).catch(() => undefined);
    },
    [],
  );

  const h = hist.current;
  const dirty = !sameObjects(objects, h.steps[h.idx]);
  const canUndo = h.idx > 0 || dirty;
  const canRedo = !dirty && h.idx < h.steps.length - 1;

  return { objects, edit, flush, discard, state, error, step, canUndo, canRedo };
}
