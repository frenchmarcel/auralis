import type { UiStateKey } from '../api';
import { getDoc, replaceSlices, serverJsonLength, setSlice, subscribeDoc, UI_STATE_MAX_BYTES, useDocVersion } from './savedViews';

/** A named copy of the slices of a page's saved view (results: main, gridmaps, …; auralizer: main). */
export interface NamedConfig {
  id: string;
  name: string;
  /** ISO time of the last save. */
  saved: string;
  slices: Record<string, unknown>;
}

/** The view that a restore replaced, kept so that the restore can be undone. */
export interface UndoState {
  slices: Record<string, unknown>;
  /** Configuration active before the restore. */
  active: string | null;
  /** Name of the configuration that was restored. */
  restored: string;
  at: string;
}

export interface NamedState {
  configs: NamedConfig[];
  active: string | null;
  undo: UndoState | null;
}

export type Outcome = { ok: true } | { ok: false; error: string; tooLarge?: boolean };

/** The slice of the saved document that holds the named configurations. */
const NAMED = 'named';
/** Top-level keys of the document that are not part of the view. */
const META = new Set([NAMED, 'saved']);

function isObject(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === 'object' && !Array.isArray(v);
}

/** The view part of a saved document: every slice but the named configurations. */
export function viewSlices(doc: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(doc).filter(([k, v]) => !META.has(k) && v !== undefined));
}

function readNamed(doc: Record<string, unknown>): NamedState {
  const n = doc[NAMED];
  if (!isObject(n)) return { configs: [], active: null, undo: null };
  const configs = (Array.isArray(n.configs) ? n.configs : []).filter(
    (c): c is NamedConfig => isObject(c) && typeof c.id === 'string' && typeof c.name === 'string' && isObject(c.slices),
  );
  const active = typeof n.active === 'string' && configs.some((c) => c.id === n.active) ? n.active : null;
  const u = n.undo;
  const undo = isObject(u) && isObject(u.slices) ? (u as unknown as UndoState) : null;
  return { configs, active, undo };
}

/** JSON with sorted keys: two views are the same whatever order their fields were written in. */
function stableJson(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableJson).join(',')}]`;
  if (isObject(v))
    return `{${Object.keys(v)
      .filter((k) => v[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableJson(v[k])}`)
      .join(',')}}`;
  return JSON.stringify(v ?? null);
}

export function sameView(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  return stableJson(a) === stableJson(b);
}

const newId = () => `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

/** What the server would measure for the document once `named` (and the view slices) are replaced. */
function sizeWith(doc: Record<string, unknown>, patch: Record<string, unknown>): number {
  const next: Record<string, unknown> = { ...doc, saved: new Date().toISOString() };
  for (const [k, v] of Object.entries(patch)) {
    if (v === null || v === undefined) delete next[k];
    else next[k] = v;
  }
  return serverJsonLength(next);
}

export const fmtKb = (bytes: number) => `${Math.max(0.1, Math.round(bytes / 102.4) / 10).toFixed(1)} kB`;

function tooLarge(bytes: number, what: string): Outcome {
  return {
    ok: false,
    tooLarge: true,
    error: `${what} would make the saved views of this simulation ${fmtKb(bytes)}, over the server's limit of ${fmtKb(UI_STATE_MAX_BYTES)}. Delete a configuration first.`,
  };
}

/**
 * A view slice the page writes for the first time (a part of the page opened for the first
 * time since the restore) is the default of that part: the active configuration, which did
 * not have it, gets it too, so it does not count as a change.
 */
const tracked = new Set<string>();
function trackSeeds(simId: string, key: UiStateKey): void {
  const k = `${key}|${simId}`;
  if (tracked.has(k)) return;
  tracked.add(k);
  subscribeDoc(simId, key, (change) => {
    if (!change?.seeded || META.has(change.slice)) return;
    const doc = getDoc(simId, key);
    const st = readNamed(doc);
    const cfg = st.configs.find((c) => c.id === st.active);
    if (!cfg || change.slice in cfg.slices) return;
    const configs = st.configs.map((c) => (c.id === cfg.id ? { ...c, slices: { ...c.slices, [change.slice]: doc[change.slice] } } : c));
    setSlice(simId, key, NAMED, { ...st, configs });
  });
}

/** What the last restore or undo did, shown next to the button until dismissed (survives the page remount). */
const restoredNotes = new Map<string, string>();
export function restoredNote(simId: string, key: UiStateKey): string | null {
  return restoredNotes.get(`${key}|${simId}`) ?? null;
}
export function clearRestoredNote(simId: string, key: UiStateKey): void {
  restoredNotes.delete(`${key}|${simId}`);
}

/** Named configurations of a page, stored in its saved document (slice `named`). */
export function useNamedConfigs(simId: string, key: UiStateKey, what = 'view') {
  useDocVersion(simId, key);
  trackSeeds(simId, key);
  const doc = getDoc(simId, key);
  const state = readNamed(doc);
  const current = viewSlices(doc);
  const active = state.configs.find((c) => c.id === state.active) ?? null;
  const modified = !!active && !sameView(current, active.slices);
  const bytes = serverJsonLength(doc);

  const write = (next: NamedState, view?: Record<string, unknown>): void => {
    replaceSlices(simId, key, { ...(view ?? {}), [NAMED]: next });
  };

  const save = (rawName: string): Outcome => {
    const name = rawName.trim();
    if (!name) return { ok: false, error: 'Give the configuration a name.' };
    if (state.configs.some((c) => c.name.toLowerCase() === name.toLowerCase()))
      return { ok: false, error: `A configuration named “${name}” exists: update it, or pick another name.` };
    const cfg: NamedConfig = { id: newId(), name, saved: new Date().toISOString(), slices: current };
    const next: NamedState = { ...state, configs: [...state.configs, cfg], active: cfg.id };
    const size = sizeWith(doc, { [NAMED]: next });
    if (size > UI_STATE_MAX_BYTES) return tooLarge(size, 'This configuration');
    write(next);
    return { ok: true };
  };

  const update = (cid: string): Outcome => {
    const next: NamedState = {
      ...state,
      configs: state.configs.map((c) => (c.id === cid ? { ...c, slices: current, saved: new Date().toISOString() } : c)),
      active: cid,
    };
    const size = sizeWith(doc, { [NAMED]: next });
    if (size > UI_STATE_MAX_BYTES) return tooLarge(size, 'This update');
    write(next);
    return { ok: true };
  };

  const rename = (cid: string, rawName: string): Outcome => {
    const name = rawName.trim();
    if (!name) return { ok: false, error: 'The name cannot be empty.' };
    if (state.configs.some((c) => c.id !== cid && c.name.toLowerCase() === name.toLowerCase()))
      return { ok: false, error: `A configuration named “${name}” exists.` };
    const next: NamedState = { ...state, configs: state.configs.map((c) => (c.id === cid ? { ...c, name } : c)) };
    if (sizeWith(doc, { [NAMED]: next }) > UI_STATE_MAX_BYTES) return tooLarge(sizeWith(doc, { [NAMED]: next }), 'This name');
    write(next);
    return { ok: true };
  };

  const remove = (cid: string): void => {
    write({ ...state, configs: state.configs.filter((c) => c.id !== cid), active: state.active === cid ? null : state.active });
  };

  /** Slices of `target` in place of the current view (slices it does not have are removed: their parts start from defaults). */
  const viewPatch = (target: Record<string, unknown>): Record<string, unknown> => ({
    ...Object.fromEntries(Object.keys(current).map((k) => [k, null])),
    ...target,
  });

  /**
   * Replaces the current (auto-saved) view by the configuration. The replaced view is kept
   * for `undo` unless it does not fit in the size limit, in which case `keepCurrent: false`
   * must be passed to restore without it.
   */
  const restore = (cid: string, keepCurrent = true): Outcome => {
    const cfg = state.configs.find((c) => c.id === cid);
    if (!cfg) return { ok: false, error: 'This configuration no longer exists.' };
    const undo: UndoState | null = keepCurrent ? { slices: current, active: state.active, restored: cfg.name, at: new Date().toISOString() } : null;
    const next: NamedState = { ...state, active: cid, undo };
    const patch = { ...viewPatch(cfg.slices), [NAMED]: next };
    const size = sizeWith(doc, patch);
    if (size > UI_STATE_MAX_BYTES)
      return keepCurrent
        ? { ok: false, tooLarge: true, error: `Keeping the current view for Undo would exceed the server's limit of ${fmtKb(UI_STATE_MAX_BYTES)}.` }
        : tooLarge(size, 'This configuration');
    replaceSlices(simId, key, patch);
    restoredNotes.set(`${key}|${simId}`, keepCurrent ? `Restored “${cfg.name}”; your previous ${what} is kept.` : `Restored “${cfg.name}”; your previous ${what} was not kept.`);
    return { ok: true };
  };

  /** Puts back the view that the last restore replaced. */
  const undo = (): Outcome => {
    const u = state.undo;
    if (!u) return { ok: false, error: 'Nothing to undo.' };
    const next: NamedState = { ...state, active: state.configs.some((c) => c.id === u.active) ? u.active : null, undo: null };
    replaceSlices(simId, key, { ...viewPatch(u.slices), [NAMED]: next });
    restoredNotes.set(`${key}|${simId}`, `Back to the ${what} you had before restoring “${u.restored}”.`);
    return { ok: true };
  };

  const dropUndo = (): void => write({ ...state, undo: null });

  return { state, active, modified, bytes, limit: UI_STATE_MAX_BYTES, save, update, rename, remove, restore, undo, dropUndo };
}

export type NamedConfigsApi = ReturnType<typeof useNamedConfigs>;
