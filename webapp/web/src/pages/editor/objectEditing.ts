import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CatalogObject, PlacedObject, Vec3 } from '../../api';
import { anyModalOpen, isTypingTarget } from '../../components/ui';
import { boundsOf, roundVec, type Bounds } from '../../lib/geom';
import { nextObjectId } from '../../lib/objects';

/** How a click changes the object selection: toggle one (Ctrl/Shift in 3D, Ctrl in the list), or a Shift range in `range` order. */
export interface SelectMods {
  toggle?: boolean;
  range?: string[];
}

/**
 * Selected placed objects, the last one clicked last (the primary). Ids of objects
 * that no longer exist are hidden, so an undone delete brings its selection back.
 */
export function useObjectSelection(objects: PlacedObject[]) {
  const [raw, setRaw] = useState<string[]>([]);
  const anchor = useRef<string | null>(null);
  const ids = useMemo(() => {
    const have = new Set(objects.map((o) => o.id));
    const out = raw.filter((id) => have.has(id));
    return out.length === raw.length ? raw : out;
  }, [raw, objects]);

  const select = useCallback((id: string | null, mods: SelectMods = {}) => {
    if (id === null) {
      setRaw([]);
      return;
    }
    const order = mods.range;
    const a = anchor.current;
    if (order && a && a !== id && order.includes(a) && order.includes(id)) {
      const i = order.indexOf(a);
      const j = order.indexOf(id);
      const span = order.slice(Math.min(i, j), Math.max(i, j) + 1).filter((x) => x !== id);
      setRaw([...span, id]);
      return;
    }
    anchor.current = id;
    if (mods.toggle || order) {
      setRaw((prev) => (prev.includes(id) ? (order ? [...prev.filter((x) => x !== id), id] : prev.filter((x) => x !== id)) : [...prev, id]));
      return;
    }
    setRaw([id]);
  }, []);
  const setAll = useCallback((list: string[]) => setRaw(list), []);
  const clear = useCallback((_id?: null) => setRaw([]), []);

  return { ids, primary: ids[ids.length - 1] ?? null, select, setAll, clear };
}

/** Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y (and Ctrl+A for all objects), never while typing in a field. */
export function useObjectKeys({
  enabled,
  onUndo,
  onRedo,
  onSelectAll,
}: {
  enabled: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onSelectAll?: () => void;
}) {
  const latest = useRef({ onUndo, onRedo, onSelectAll });
  latest.current = { onUndo, onRedo, onSelectAll };
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || !(e.ctrlKey || e.metaKey) || e.altKey || anyModalOpen() || isTypingTarget(e.target)) return;
      const k = e.key.toLowerCase();
      if (k === 'z' || k === 'y') {
        e.preventDefault();
        if (k === 'y' || e.shiftKey) latest.current.onRedo();
        else latest.current.onUndo();
      } else if (k === 'a' && !e.shiftKey && latest.current.onSelectAll) {
        e.preventDefault();
        latest.current.onSelectAll();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [enabled]);
}

/** World bounds of the placed objects (their rotated, scaled catalog boxes). */
export function objectsBounds(objects: PlacedObject[], catalog: Map<string, CatalogObject>): Bounds | null {
  if (!objects.length) return null;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const o of objects) {
    const d = catalog.get(o.catalog_id)?.dims ?? [1, 1, 1];
    const w = (d[0] * o.scale[0]) / 2;
    const h = (d[1] * o.scale[1]) / 2;
    const a = (o.rotation_deg * Math.PI) / 180;
    const ex = Math.abs(w * Math.cos(a)) + Math.abs(h * Math.sin(a));
    const ey = Math.abs(w * Math.sin(a)) + Math.abs(h * Math.cos(a));
    const lo: Vec3 = [o.position[0] - ex, o.position[1] - ey, o.position[2]];
    const hi: Vec3 = [o.position[0] + ex, o.position[1] + ey, o.position[2] + d[2] * o.scale[2]];
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], lo[k]);
      max[k] = Math.max(max[k], hi[k]);
    }
  }
  return boundsOf(min, max);
}

/** Copies of the objects `ids`, shifted together along +x by the selection width; returns the new ids. */
export function duplicateObjects(list: PlacedObject[], ids: string[], catalog: Map<string, CatalogObject>): { list: PlacedObject[]; added: string[] } {
  const src = list.filter((o) => ids.includes(o.id));
  const b = objectsBounds(src, catalog);
  if (!b) return { list, added: [] };
  const dx = b.size[0] + 0.1;
  const out = [...list];
  const added: string[] = [];
  for (const o of src) {
    const id = nextObjectId(out);
    out.push({ ...o, id, name: `${o.name} copy`, position: roundVec([o.position[0] + dx, o.position[1], o.position[2]], 3) });
    added.push(id);
  }
  return { list: out, added };
}

const EPS = 1e-6;
const sameVec = (a: Vec3, b: Vec3) => Math.abs(a[0] - b[0]) < EPS && Math.abs(a[1] - b[1]) < EPS && Math.abs(a[2] - b[2]) < EPS;

/** Same catalog object at the same position, rotation and scale. */
export function samePlacement(a: PlacedObject | undefined, b: PlacedObject | undefined): boolean {
  if (!a || !b) return a === b;
  return a.catalog_id === b.catalog_id && Math.abs(a.rotation_deg - b.rotation_deg) < EPS && sameVec(a.position, b.position) && sameVec(a.scale, b.scale);
}

/** Same objects, in the same order, with the same names and placements. */
export function sameObjects(a: PlacedObject[], b: PlacedObject[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  return a.every((o, i) => o.id === b[i].id && o.name === b[i].name && samePlacement(o, b[i]));
}
