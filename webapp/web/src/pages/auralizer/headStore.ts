import { useState, useSyncExternalStore } from 'react';

/** Listener's head: yaw = azimuth CCW from +X, pitch = elevation (+ up), degrees. */
export interface Head {
  yaw: number;
  pitch: number;
}

export const PITCH_LIMIT = 80;

/** Head orientation outside React state: only its subscribers re-render while turning. */
export interface HeadStore {
  get: () => Head;
  set: (h: Partial<Head>) => void;
  subscribe: (fn: () => void) => () => void;
}

export const wrapYaw = (y: number) => ((y % 360) + 360) % 360;

export function createHeadStore(init: Head = { yaw: 0, pitch: 0 }): HeadStore {
  let h: Head = { yaw: wrapYaw(init.yaw), pitch: init.pitch };
  const subs = new Set<() => void>();
  return {
    get: () => h,
    set: (p) => {
      const yaw = p.yaw === undefined ? h.yaw : wrapYaw(p.yaw);
      const pitch = p.pitch === undefined ? h.pitch : Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, p.pitch));
      if (yaw === h.yaw && pitch === h.pitch) return;
      h = { yaw, pitch };
      subs.forEach((f) => f());
    },
    subscribe: (fn) => {
      subs.add(fn);
      return () => subs.delete(fn);
    },
  };
}

export function useHeadStore(): HeadStore {
  const [s] = useState(() => createHeadStore());
  return s;
}

export function useHead(store: HeadStore): Head {
  return useSyncExternalStore(store.subscribe, store.get);
}
