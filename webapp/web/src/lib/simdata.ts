import { useEffect, useRef, useState } from 'react';
import { api, ApiError, errorMessage, type Curves, type FrSmoothing, type Project, type Results, type ResultPart, type Simulation } from '../api';

export interface SimBundle {
  sim: Simulation;
  /** null when the simulation has no results (404). */
  results: Results | null;
}

export interface BundleState {
  data?: SimBundle;
  error?: string;
  loading?: boolean;
}

/**
 * Loads simulations with their results and curves, once per page: every
 * comparison card reads from here, so a simulation used twice is fetched once.
 */
export class SimDataStore {
  private bundles = new Map<string, Promise<SimBundle>>();
  private curves = new Map<string, Promise<Curves>>();

  bundle(simId: string): Promise<SimBundle> {
    let p = this.bundles.get(simId);
    if (!p) {
      p = (async () => {
        const sim = await api.getSimulation(simId);
        let results: Results | null = null;
        try {
          results = await api.getResults(simId);
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) throw e;
        }
        return { sim, results };
      })();
      this.bundles.set(simId, p);
      p.catch(() => this.bundles.delete(simId));
    }
    return p;
  }

  /** Forget a simulation (it was changed): the next bundle() reads it again. */
  invalidate(simId: string): void {
    this.bundles.delete(simId);
  }

  curve(simId: string, sourceId: string, receiverId: string, smoothing: FrSmoothing, part: ResultPart = 'hybrid'): Promise<Curves> {
    const key = `${simId}|${sourceId}|${receiverId}|${smoothing}|${part}`;
    let p = this.curves.get(key);
    if (!p) {
      // A coherent sum of sources (results card "Σ"): its curves live under /combined/<key>.
      p = sourceId.startsWith('Σ:')
        ? api.getCombinedCurves(`/api/simulations/${encodeURIComponent(simId)}/combined/${encodeURIComponent(sourceId.slice(2))}/curves`, smoothing)
        : api.getCurves(simId, sourceId, receiverId, smoothing, part);
      this.curves.set(key, p);
      p.catch(() => this.curves.delete(key));
    }
    return p;
  }
}

/** State of every requested simulation, refreshed as each one arrives (and again when `epoch` changes). */
export function useBundles(store: SimDataStore, simIds: string[], epoch = 0): Map<string, BundleState> {
  const [state, setState] = useState<Map<string, BundleState>>(new Map());
  const key = Array.from(new Set(simIds)).sort().join(',');
  useEffect(() => {
    let cancelled = false;
    const ids = key ? key.split(',') : [];
    for (const id of ids) {
      setState((prev) => {
        if (prev.get(id)?.data || prev.get(id)?.loading) return prev;
        const n = new Map(prev);
        n.set(id, { loading: true });
        return n;
      });
      store
        .bundle(id)
        .then((data) => {
          if (cancelled) return;
          setState((prev) => new Map(prev).set(id, { data }));
        })
        .catch((e) => {
          if (cancelled) return;
          setState((prev) => new Map(prev).set(id, { error: errorMessage(e) }));
        });
    }
    return () => {
      cancelled = true;
      // Requests still in flight resume on the next run (the store memoises them).
      setState((prev) => {
        let changed = false;
        const n = new Map(prev);
        for (const [k, v] of n) {
          if (v.loading) {
            n.delete(k);
            changed = true;
          }
        }
        return changed ? n : prev;
      });
    };
  }, [store, key, epoch]);
  return state;
}

export interface CurveRequest {
  simId: string;
  sourceId: string;
  receiverId: string;
  /** Result type (hybrid by default). */
  part?: ResultPart;
}

/** Key of a pair's curves in the map useCurves returns. */
export function curveKey(simId: string, sourceId: string, receiverId: string, part: ResultPart = 'hybrid'): string {
  return `${simId}|${sourceId}|${receiverId}|${part}`;
}

/** Curves of every requested pair at one smoothing; keyed by curveKey(). */
export function useCurves(store: SimDataStore, reqs: CurveRequest[], smoothing: FrSmoothing) {
  const [curves, setCurves] = useState<Map<string, Curves>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const smoothRef = useRef(smoothing);
  const key = Array.from(new Set(reqs.map((r) => curveKey(r.simId, r.sourceId, r.receiverId, r.part)))).join(',');
  useEffect(() => {
    let cancelled = false;
    if (smoothRef.current !== smoothing) {
      smoothRef.current = smoothing;
      setCurves(new Map());
    }
    const list = key ? key.split(',') : [];
    if (!list.length) {
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.allSettled(
      list.map((k) => {
        const [simId, src, rcv, part] = k.split('|');
        return store.curve(simId, src, rcv, smoothing, part as ResultPart).then((c) => [k, c] as const);
      }),
    ).then((res) => {
      if (cancelled) return;
      const n = new Map<string, Curves>();
      const errs: string[] = [];
      res.forEach((r, i) => {
        if (r.status === 'fulfilled') n.set(r.value[0], r.value[1]);
        else {
          const [, src, rcv, part] = list[i].split('|');
          errs.push(`${src} → ${rcv}${part !== 'hybrid' ? ` (${part === 'geo' ? 'geometrical' : 'wave'})` : ''}: ${errorMessage(r.reason)}`);
        }
      });
      setCurves(n);
      setError(errs.length ? errs.join(' · ') : null);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [store, key, smoothing]);
  return { curves, loading, error };
}

/** Every simulation of the tree, with where it lives (for pickers). */
export interface SimEntry {
  id: string;
  name: string;
  path: string;
  projectId: string;
  spaceId: string;
  geometryId: string;
  geometryName: string;
}

export function allSimulations(tree: Project[] | null): SimEntry[] {
  const out: SimEntry[] = [];
  for (const p of tree ?? []) {
    for (const s of p.spaces ?? []) {
      for (const g of s.geometries ?? []) {
        for (const m of g.simulations ?? []) {
          out.push({ id: m.id, name: m.name, path: `${p.name} › ${s.name} › ${g.name}`, projectId: p.id, spaceId: s.id, geometryId: g.id, geometryName: g.name });
        }
      }
    }
  }
  return out;
}

/**
 * Which simulations have results. The tree's `last_run` is not reliable for that
 * (copied stores keep results without it), so each one is probed once.
 */
export function useResultsIndex(store: SimDataStore, simIds: string[], enabled: boolean): Map<string, boolean | 'error'> {
  const [index, setIndex] = useState<Map<string, boolean | 'error'>>(new Map());
  const key = simIds.join(',');
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const ids = key ? key.split(',') : [];
    let next = 0;
    const worker = async () => {
      while (!cancelled && next < ids.length) {
        const id = ids[next++];
        try {
          const b = await store.bundle(id);
          if (!cancelled) setIndex((m) => new Map(m).set(id, !!b.results));
        } catch {
          if (!cancelled) setIndex((m) => new Map(m).set(id, 'error'));
        }
      }
    };
    void Promise.all([worker(), worker(), worker(), worker()]);
    return () => {
      cancelled = true;
    };
  }, [store, key, enabled]);
  return index;
}
