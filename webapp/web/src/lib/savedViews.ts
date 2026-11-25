import { useEffect, useState, useSyncExternalStore } from 'react';
import { api, errorMessage, type UiStateKey } from '../api';

/**
 * Saved views per simulation (GET/PUT /ui-state/{key}): one JSON document per page, made
 * of named slices (e.g. results: main, gridmaps, modal, office). The server is the store;
 * localStorage only keeps a fallback copy for when the server cannot be reached.
 */
type Doc = Record<string, unknown>;

export interface DocInfo {
  /** Where the restored view came from. */
  source: 'server' | 'browser' | 'none';
  /** Why the server copy could not be read (the browser copy is used then). */
  error: string | null;
}

interface Entry {
  data: Doc;
  /** The server copy could not be read: a save first reads it again and merges into it. */
  readFailed: boolean;
  /** Slices set since the last read of the server copy. */
  changed: Set<string>;
  timer: number | null;
  /** Last failed save, null once a save succeeds. */
  saveError: string | null;
  /** Called after every save attempt with its error (null = stored on the server). */
  listeners: Set<(err: string | null) => void>;
  /** Called after every change of the document (null slice = several slices replaced). */
  changeListeners: Set<DocChangeListener>;
  /** Bumped on every change: the snapshot of useDocVersion. */
  version: number;
}

/** A change of a saved document: the slice set (seeded = its first write, the page's initial state). */
export type DocChangeListener = (change: { slice: string; seeded: boolean } | null) => void;

/** Size limit of one document on the server (PUT /ui-state/{key}). */
export const UI_STATE_MAX_BYTES = 512 * 1024;

const DEBOUNCE_MS = 800;
const READ_TIMEOUT_MS = 4000;
const entries = new Map<string, Entry>();
const id = (simId: string, key: UiStateKey) => `${key}|${simId}`;
const localKey = (simId: string, key: UiStateKey) => `auralis.view.${key}.${simId}`;

function readJson(k: string): unknown {
  try {
    const raw = window.localStorage.getItem(k);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function isObject(v: unknown): v is Doc {
  return !!v && typeof v === 'object' && !Array.isArray(v);
}

/** Per-browser keys used before views were stored on the server: read once, as a fallback. */
function legacy(simId: string, key: UiStateKey): Doc | null {
  const out: Doc = {};
  if (key === 'results') {
    const main = readJson(`auralis.results.${simId}`);
    const maps = readJson(`auralis.gridmaps.${simId}`);
    if (isObject(main)) out.main = main;
    if (isObject(maps)) out.gridmaps = maps;
  } else if (key === 'auralizer') {
    const main = readJson(`auralis.auralizer.${simId}`);
    if (isObject(main)) out.main = main;
  }
  return Object.keys(out).length ? out : null;
}

function readLocal(simId: string, key: UiStateKey): Doc | null {
  const v = readJson(localKey(simId, key));
  return isObject(v) ? v : legacy(simId, key);
}

function writeLocal(simId: string, key: UiStateKey, doc: Doc): void {
  try {
    window.localStorage.setItem(localKey(simId, key), JSON.stringify(doc));
  } catch {
    /* storage unavailable: the server copy is the one that counts */
  }
}

function entry(simId: string, key: UiStateKey): Entry {
  let e = entries.get(id(simId, key));
  if (!e) {
    e = { data: {}, readFailed: false, changed: new Set(), timer: null, saveError: null, listeners: new Set(), changeListeners: new Set(), version: 0 };
    entries.set(id(simId, key), e);
  }
  return e;
}

function notifyChange(e: Entry, change: { slice: string; seeded: boolean } | null): void {
  e.version++;
  e.changeListeners.forEach((l) => l(change));
}

function setSaveError(e: Entry, err: string | null): void {
  e.saveError = err;
  e.listeners.forEach((l) => l(err));
}

/** After a failed read, never overwrite the server copy blindly: merge the changed slices into it. */
async function reconcile(simId: string, key: UiStateKey, e: Entry): Promise<void> {
  if (!e.readFailed) return;
  const srv = await withTimeout(api.getUiState(simId, key), READ_TIMEOUT_MS);
  const base = isObject(srv) ? srv : {};
  const mine = Object.fromEntries([...e.changed].map((k) => [k, e.data[k]]));
  e.data = { ...base, ...mine, ...(e.changed.size ? { saved: e.data.saved } : {}) };
  e.readFailed = false;
  writeLocal(simId, key, e.data);
  notifyChange(e, null);
}

async function put(simId: string, key: UiStateKey, e: Entry): Promise<void> {
  try {
    await reconcile(simId, key, e);
    await api.putUiState(simId, key, e.data);
    setSaveError(e, null);
  } catch (err) {
    setSaveError(e, errorMessage(err));
  }
}

/** Send a pending save now (page left, before reloading). */
export async function flushDoc(simId: string, key: UiStateKey, keepalive = false): Promise<void> {
  const e = entries.get(id(simId, key));
  if (!e || e.timer === null) return;
  window.clearTimeout(e.timer);
  e.timer = null;
  if (keepalive) {
    // The page is going away: a plain request could be cut off (and nothing to merge with).
    if (!e.readFailed) void api.putUiStateKeepalive(simId, key, e.data).catch(() => undefined);
    return;
  }
  await put(simId, key, e);
}

function schedule(simId: string, key: UiStateKey, e: Entry): void {
  if (e.timer !== null) window.clearTimeout(e.timer);
  e.timer = window.setTimeout(() => {
    e.timer = null;
    void put(simId, key, e);
  }, DEBOUNCE_MS);
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = window.setTimeout(() => reject(new Error(`no answer within ${ms / 1000} s`)), ms);
    p.then(
      (v) => {
        window.clearTimeout(t);
        resolve(v);
      },
      (err) => {
        window.clearTimeout(t);
        reject(err);
      },
    );
  });
}

/** Read the saved view from the server (after sending any pending save); falls back to the browser copy. */
export async function loadDoc(simId: string, key: UiStateKey): Promise<DocInfo> {
  const info = await readDoc(simId, key);
  notifyChange(entry(simId, key), null);
  return info;
}

async function readDoc(simId: string, key: UiStateKey): Promise<DocInfo> {
  await flushDoc(simId, key);
  const e = entry(simId, key);
  e.changed = new Set();
  e.readFailed = false;
  try {
    const srv = await withTimeout(api.getUiState(simId, key), READ_TIMEOUT_MS);
    if (isObject(srv) && Object.keys(srv).length) {
      e.data = srv;
      writeLocal(simId, key, srv);
      return { source: 'server', error: null };
    }
    // Nothing on the server yet: start from this browser's copy and store it there.
    const local = readLocal(simId, key);
    if (local) {
      e.data = local;
      schedule(simId, key, e);
      return { source: 'browser', error: null };
    }
    e.data = {};
    return { source: 'none', error: null };
  } catch (err) {
    const local = readLocal(simId, key);
    e.data = local ?? {};
    e.readFailed = true;
    return { source: local ? 'browser' : 'none', error: errorMessage(err) };
  }
}

export function getSlice<T>(simId: string, key: UiStateKey, slice: string): Partial<T> | null {
  const v = entries.get(id(simId, key))?.data[slice];
  return isObject(v) ? (v as Partial<T>) : null;
}

/** Store one slice: browser copy at once, server copy debounced. */
export function setSlice(simId: string, key: UiStateKey, slice: string, value: object): void {
  const e = entry(simId, key);
  const next = JSON.stringify(value);
  if (JSON.stringify(e.data[slice] ?? null) === next) return;
  // The first write of a slice the document did not have is the page's initial state, not an edit.
  const seed = !(slice in e.data);
  e.data = { ...e.data, [slice]: JSON.parse(next) as unknown, saved: new Date().toISOString() };
  if (!seed) e.changed.add(slice);
  writeLocal(simId, key, e.data);
  schedule(simId, key, e);
  notifyChange(e, { slice, seeded: seed });
}

/** The whole document as it stands (a copy of its top level). */
export function getDoc(simId: string, key: UiStateKey): Doc {
  return { ...(entries.get(id(simId, key))?.data ?? {}) };
}

/** Set several slices at once (null or undefined removes a slice): browser copy at once, server copy debounced. */
export function replaceSlices(simId: string, key: UiStateKey, slices: Record<string, unknown>): void {
  const e = entry(simId, key);
  const next: Doc = { ...e.data };
  for (const [k, v] of Object.entries(slices)) {
    if (v === null || v === undefined) delete next[k];
    else next[k] = JSON.parse(JSON.stringify(v)) as unknown;
    e.changed.add(k);
  }
  e.data = { ...next, saved: new Date().toISOString() };
  writeLocal(simId, key, e.data);
  schedule(simId, key, e);
  notifyChange(e, null);
}

/** Listen to every change of a document; returns the unsubscribe function. */
export function subscribeDoc(simId: string, key: UiStateKey, fn: DocChangeListener): () => void {
  const e = entry(simId, key);
  e.changeListeners.add(fn);
  return () => void e.changeListeners.delete(fn);
}

/** Re-renders the caller on every change of the document; returns its version number. */
export function useDocVersion(simId: string, key: UiStateKey): number {
  return useSyncExternalStore(
    (cb) => subscribeDoc(simId, key, cb),
    () => entry(simId, key).version,
  );
}

/**
 * Length of `json.dumps(v)` as the server measures it against its limit (Python defaults:
 * ", " and ": " separators, every non-ASCII character escaped as \uXXXX).
 */
export function serverJsonLength(v: unknown): number {
  if (v === null || v === undefined || typeof v === 'function') return 4;
  if (typeof v === 'string') {
    let n = JSON.stringify(v).length;
    for (let i = 0; i < v.length; i++) {
      const c = v.charCodeAt(i);
      // Lone surrogates are already escaped by JSON.stringify.
      const lone = c >= 0xd800 && c <= 0xdfff && !(c <= 0xdbff ? (v.charCodeAt(i + 1) & 0xfc00) === 0xdc00 : (v.charCodeAt(i - 1) & 0xfc00) === 0xd800);
      if (c > 0x7f && !lone) n += 5;
    }
    return n;
  }
  if (typeof v === 'number') {
    const t = Number.isFinite(v) ? JSON.stringify(v) : 'null';
    // Python writes 1e-07 where JavaScript writes 1e-7.
    return t.length + (/e[+-]\d$/.test(t) ? 1 : 0);
  }
  if (typeof v === 'boolean') return v ? 4 : 5;
  if (Array.isArray(v)) return v.length ? 2 + v.reduce<number>((a, x) => a + serverJsonLength(x), 0) + 2 * (v.length - 1) : 2;
  const items = Object.entries(v as Doc).filter(([, x]) => x !== undefined && typeof x !== 'function');
  if (!items.length) return 2;
  return 2 + items.reduce((a, [k, x]) => a + serverJsonLength(k) + 2 + serverJsonLength(x), 0) + 2 * (items.length - 1);
}

/**
 * Loads the saved view of a page before it renders (so its state starts from it), keeps
 * the last save error, and sends pending saves when the page closes.
 */
export function useSavedDoc(simId: string, key: UiStateKey): { ready: boolean; info: DocInfo | null; saveError: string | null } {
  const [state, setState] = useState<{ simId: string; info: DocInfo } | null>(null);
  const [saveError, setErr] = useState<string | null>(null);
  const [savedOnce, setSavedOnce] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setState(null);
    setSavedOnce(false);
    void loadDoc(simId, key).then((info) => !cancelled && setState({ simId, info }));
    const e = entry(simId, key);
    setErr(e.saveError);
    const onSave = (err: string | null) => {
      setErr(err);
      if (!err) setSavedOnce(true);
    };
    e.listeners.add(onSave);
    const onHide = () => void flushDoc(simId, key, true);
    window.addEventListener('pagehide', onHide);
    return () => {
      cancelled = true;
      e.listeners.delete(onSave);
      window.removeEventListener('pagehide', onHide);
      void flushDoc(simId, key);
    };
  }, [simId, key]);
  const ready = state?.simId === simId;
  // A save that reached the server makes an earlier read failure moot.
  const info = ready ? (savedOnce && state.info.error ? { ...state.info, error: null } : state.info) : null;
  return { ready, info, saveError };
}
