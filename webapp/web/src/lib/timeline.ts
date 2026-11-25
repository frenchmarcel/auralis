import { useState, useSyncExternalStore } from 'react';

/**
 * Current time of the trajectory preview, outside React state: only the
 * components that subscribe (markers, the timeline itself) re-render per frame.
 */
export interface TimeStore {
  get: () => number;
  set: (t: number) => void;
  subscribe: (fn: () => void) => () => void;
}

export function createTimeStore(t0 = 0): TimeStore {
  let t = t0;
  const subs = new Set<() => void>();
  return {
    get: () => t,
    set: (v) => {
      if (v === t) return;
      t = v;
      subs.forEach((f) => f());
    },
    subscribe: (fn) => {
      subs.add(fn);
      return () => subs.delete(fn);
    },
  };
}

export function useTimeStore(): TimeStore {
  const [s] = useState(() => createTimeStore());
  return s;
}

export function useTime(store: TimeStore): number {
  return useSyncExternalStore(store.subscribe, store.get);
}
