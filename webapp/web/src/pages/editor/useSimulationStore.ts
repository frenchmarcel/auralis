import { useCallback, useEffect, useRef, useState } from 'react';
import { api, editableOf, errorMessage, type Simulation, type Validation } from '../../api';

export type SaveState = 'saved' | 'pending' | 'saving' | 'error';

const SAVE_DELAY_MS = 700;

/**
 * Local copy of a simulation with debounced whole-object PUT saves.
 * Edits made while a save is in flight are kept and saved afterwards.
 */
export function useSimulationStore(simId: string) {
  const [sim, setSim] = useState<Simulation | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>('saved');
  const [saveError, setSaveError] = useState<string | null>(null);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const simRef = useRef<Simulation | null>(null);
  const version = useRef(0);
  const savedVersion = useRef(0);
  const timer = useRef<number | undefined>(undefined);
  const inflight = useRef<Promise<void> | null>(null);
  const alive = useRef(true);
  /** While paused (objects being re-assembled server-side) edits are queued, not saved. */
  const paused = useRef<((s: Simulation) => Simulation)[] | null>(null);
  const resumeWaiters = useRef<(() => void)[]>([]);
  /**
   * Edit guard (the "Modify / Duplicate / Revert" prompt): when it returns true for an
   * edit, that edit and the next ones are applied locally but not saved until
   * release() keeps or reverts them.
   */
  const guard = useRef<((before: Simulation, after: Simulation) => boolean) | null>(null);
  const held = useRef<{ snapshot: Simulation; version: number } | null>(null);
  const holdWaiters = useRef<(() => void)[]>([]);

  const refreshValidation = useCallback(async () => {
    try {
      const v = await api.getValidation(simId);
      if (alive.current) {
        setValidation(v);
        setValidationError(null);
      }
      return v;
    } catch (e) {
      if (alive.current) setValidationError(errorMessage(e));
      return null;
    }
  }, [simId]);

  const load = useCallback(async () => {
    try {
      const s = await api.getSimulation(simId);
      if (!alive.current) return;
      simRef.current = s;
      version.current = 0;
      savedVersion.current = 0;
      setSim(s);
      setLoadError(null);
      setSaveState('saved');
      void refreshValidation();
    } catch (e) {
      if (alive.current) setLoadError(errorMessage(e));
    }
  }, [simId, refreshValidation]);

  useEffect(() => {
    alive.current = true;
    void load();
    return () => {
      alive.current = false;
      window.clearTimeout(timer.current);
      // Unsaved edits when leaving: fire a last save.
      // Edits held by the prompt are dropped (= Revert); earlier unsaved ones are kept.
      const h = held.current;
      const s = h ? h.snapshot : simRef.current;
      const v = h ? h.version : version.current;
      if (s && v !== savedVersion.current) {
        api.saveSimulation(s.id, editableOf(s)).catch(() => undefined);
      }
    };
  }, [load]);

  /** Save now; resolves true once everything local is stored. */
  const flush = useCallback(async (): Promise<boolean> => {
    window.clearTimeout(timer.current);
    for (;;) {
      if (held.current) {
        await new Promise<void>((r) => holdWaiters.current.push(r));
        continue;
      }
      if (paused.current) {
        await new Promise<void>((r) => resumeWaiters.current.push(r));
        continue;
      }
      if (inflight.current) {
        await inflight.current;
        continue;
      }
      const s = simRef.current;
      if (!s) return true;
      const v = version.current;
      if (v === savedVersion.current) {
        if (alive.current) setSaveState('saved');
        return true;
      }
      if (alive.current) setSaveState('saving');
      let failed: unknown = null;
      const p = api
        .saveSimulation(s.id, editableOf(s))
        .then((stored) => {
          savedVersion.current = v;
          if (!alive.current) return;
          if (version.current === v) {
            simRef.current = stored;
            setSim(stored);
          } else if (simRef.current) {
            // Newer local edits exist: keep them, take only the server-owned status fields.
            const merged: Simulation = {
              ...simRef.current,
              status: stored.status,
              results_stale: stored.results_stale,
              last_run: stored.last_run,
            };
            simRef.current = merged;
            setSim(merged);
          }
          setSaveError(null);
          void refreshValidation();
        })
        .catch((e) => {
          failed = e;
        });
      inflight.current = p;
      await p;
      inflight.current = null;
      if (failed) {
        if (alive.current) {
          setSaveState('error');
          setSaveError(errorMessage(failed));
        }
        return false;
      }
    }
  }, [refreshValidation]);

  const update = useCallback(
    (fn: (s: Simulation) => Simulation) => {
      const cur = simRef.current;
      if (!cur) return;
      const next = fn(cur);
      if (next === cur) return;
      if (!held.current && guard.current?.(cur, next)) held.current = { snapshot: cur, version: version.current };
      simRef.current = next;
      setSim(next);
      version.current++;
      setSaveState('pending');
      if (held.current) return;
      if (paused.current) {
        paused.current.push(fn);
        return;
      }
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => void flush(), SAVE_DELAY_MS);
    },
    [flush],
  );

  /** Stop saving: call after flush(), before changing the geometry's layers server-side. */
  const pause = useCallback(() => {
    window.clearTimeout(timer.current);
    paused.current = [];
  }, []);

  /**
   * Resume with the server's fresh copy (null = keep the local one). Edits made
   * while paused are replayed on top of it, restricted to the layers it knows.
   */
  const resume = useCallback(
    (fresh: Simulation | null) => {
      const patches = paused.current ?? [];
      paused.current = null;
      if (fresh && alive.current) {
        let s = fresh;
        for (const p of patches) s = p(s);
        const known = Object.keys(fresh.layers);
        if (Object.keys(s.layers).some((k) => !(k in fresh.layers))) {
          s = { ...s, layers: Object.fromEntries(known.map((k) => [k, s.layers[k]])) };
        }
        simRef.current = s;
        setSim(s);
        if (!patches.length) {
          savedVersion.current = version.current;
          setSaveState('saved');
        }
      }
      const waiters = resumeWaiters.current;
      resumeWaiters.current = [];
      waiters.forEach((w) => w());
      if (version.current !== savedVersion.current) {
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => void flush(), SAVE_DELAY_MS);
      }
    },
    [flush],
  );

  /** Replace the local copy with a fresh server copy (after a run, for instance). */
  const reload = useCallback(async () => {
    if (version.current !== savedVersion.current) await flush();
    await load();
  }, [flush, load]);

  const isDirty = () => version.current !== savedVersion.current;

  const setGuard = useCallback((fn: ((before: Simulation, after: Simulation) => boolean) | null) => {
    guard.current = fn;
  }, []);

  /** End a hold: keep the held edits (they are saved now) or drop them. */
  const release = useCallback(
    (keep: boolean) => {
      const h = held.current;
      if (!h) return;
      held.current = null;
      if (!keep && alive.current) {
        simRef.current = h.snapshot;
        setSim(h.snapshot);
        version.current = h.version;
        setSaveState(h.version === savedVersion.current ? 'saved' : 'pending');
      }
      const waiters = holdWaiters.current;
      holdWaiters.current = [];
      waiters.forEach((w) => w());
      if (version.current !== savedVersion.current) {
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => void flush(), keep ? 0 : SAVE_DELAY_MS);
      }
    },
    [flush],
  );

  const isHeld = () => !!held.current;

  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (version.current !== savedVersion.current) {
        e.preventDefault();
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  return {
    sim,
    loadError,
    saveState,
    saveError,
    validation,
    validationError,
    update,
    flush,
    pause,
    resume,
    reload,
    refreshValidation,
    isDirty,
    setGuard,
    release,
    isHeld,
  };
}

export type SimulationStore = ReturnType<typeof useSimulationStore>;
