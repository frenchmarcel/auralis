import { useCallback, useMemo, useRef, useState } from 'react';
import { api, errorMessage, type CatalogObject, type MeshData, type PlacedObject, type Vec3 } from '../../api';
import { samePlacement } from './objectEditing';

const INTERVAL_MS = 100;

type Tri = [Vec3, Vec3, Vec3];

interface LiveProxy {
  /** Placement last asked for (the release placement once the drag is over). */
  want: PlacedObject;
  /** Server mesh when `want` was set. */
  base: MeshData | null;
  /** Proxy layers of the latest answer, null before the first one. */
  layers: Record<string, Tri[]> | null;
}

/** Same layers without the triangles of the objects `hide` (their proxy is drawn by the preview instead). */
function maskMesh(mesh: MeshData, hide: Set<string>): MeshData {
  const layers: MeshData['layers'] = [];
  for (const l of mesh.layers) {
    const owners = l.owners;
    if (!owners || !owners.some((o) => hide.has(o))) {
      layers.push(l);
      continue;
    }
    const indices: number[] = [];
    const kept: string[] = [];
    owners.forEach((o, t) => {
      if (hide.has(o)) return;
      indices.push(l.indices[3 * t], l.indices[3 * t + 1], l.indices[3 * t + 2]);
      kept.push(o);
    });
    if (indices.length) layers.push({ ...l, indices, owners: kept });
  }
  return { positions: mesh.positions, layers };
}

function previewMesh(entries: [string, LiveProxy][]): MeshData | null {
  const positions: number[] = [];
  const byLayer = new Map<string, { indices: number[]; owners: string[] }>();
  for (const [id, e] of entries) {
    for (const [name, tris] of Object.entries(e.layers ?? {})) {
      let l = byLayer.get(name);
      if (!l) byLayer.set(name, (l = { indices: [], owners: [] }));
      for (const t of tris) {
        const v = positions.length / 3;
        positions.push(...t[0], ...t[1], ...t[2]);
        l.indices.push(v, v + 1, v + 2);
        l.owners.push(id);
      }
    }
  }
  if (!byLayer.size) return null;
  return { positions, layers: [...byLayer].map(([name, l]) => ({ name, kind: 'object' as const, ...l })) };
}

/**
 * Acoustic proxies at the placement of a gizmo drag (POST /objects/proxy, about
 * 10 requests/s, one at a time; nothing is saved). The dragged objects' triangles
 * are taken out of the server mesh (`displayMesh`) and drawn from `preview` at the
 * new placement, until the server mesh catches up after the PUT of the release.
 */
export function useLiveProxy({
  mesh,
  objects,
  serverObjects,
  catalogById,
  onError,
}: {
  mesh: MeshData | null;
  objects: PlacedObject[];
  /** The geometry's saved objects (absent before the first PUT). */
  serverObjects: PlacedObject[] | undefined;
  catalogById: Map<string, CatalogObject>;
  onError: (msg: string) => void;
}) {
  const [live, setLive] = useState<Map<string, LiveProxy>>(() => new Map());
  const [dragIds, setDragIds] = useState<ReadonlySet<string>>(() => new Set());
  const dragRef = useRef<ReadonlySet<string>>(dragIds);
  const activeRef = useRef<ReadonlySet<string>>(new Set());
  const queued = useRef<PlacedObject[] | null>(null);
  const busy = useRef(false);
  const lastAt = useRef(0);
  const timer = useRef<number | undefined>(undefined);
  const warned = useRef(false);
  const latest = useRef({ mesh, catalogById, onError });
  latest.current = { mesh, catalogById, onError };

  const send = useCallback(() => {
    window.clearTimeout(timer.current);
    if (!queued.current || busy.current) return;
    const wait = lastAt.current + INTERVAL_MS - performance.now();
    if (wait > 0) {
      timer.current = window.setTimeout(send, wait);
      return;
    }
    const list = queued.current;
    queued.current = null;
    busy.current = true;
    lastAt.current = performance.now();
    api
      .objectProxy(list.map(({ id, catalog_id, position, rotation_deg, scale }) => ({ id, catalog_id, position, rotation_deg, scale })))
      .then((r) =>
        setLive((prev) => {
          let next: Map<string, LiveProxy> | null = null;
          for (const o of r.objects) {
            const cur = prev.get(o.id);
            if (!cur) continue;
            next ??= new Map(prev);
            next.set(o.id, { ...cur, layers: o.layers });
          }
          return next ?? prev;
        }),
      )
      .catch((e) => {
        // The drag goes on with the proxy left at its saved place; say it once per drag.
        if (warned.current) return;
        warned.current = true;
        latest.current.onError(`Live proxy preview failed: ${errorMessage(e)}`);
      })
      .finally(() => {
        busy.current = false;
        if (queued.current) send();
      });
  }, []);

  /** Gizmo moves of `list` (the dragged objects at their new placement); `final` on release. */
  const track = useCallback(
    (list: PlacedObject[], final: boolean) => {
      const { catalogById: cat, mesh: m } = latest.current;
      const items = list.filter((o) => {
        const c = cat.get(o.catalog_id);
        return !!c && c.proxy !== 'none';
      });
      const prevDrag = dragRef.current;
      const start = !final && !prevDrag.size;
      if (start) warned.current = false;
      const ids = final ? new Set<string>() : new Set(items.map((o) => o.id));
      if (ids.size !== prevDrag.size || [...ids].some((id) => !prevDrag.has(id))) {
        dragRef.current = ids;
        setDragIds(ids);
      }
      if (!items.length) return;
      const keep = activeRef.current;
      setLive((prev) => {
        if (!final && !start && items.every((o) => prev.has(o.id))) return prev;
        // Entries no longer shown are dropped; one still shown keeps its triangles until the next answer.
        const next = new Map([...prev].filter(([id]) => keep.has(id)));
        for (const o of items) {
          const cur = next.get(o.id);
          if (final || !cur) next.set(o.id, { want: o, base: m, layers: cur?.layers ?? null });
        }
        return next;
      });
      queued.current = items;
      send();
    },
    [send],
  );

  const activeKey = useMemo(() => {
    const local = new Map(objects.map((o) => [o.id, o]));
    const srv = new Map((serverObjects ?? []).map((o) => [o.id, o]));
    const ids = [...live]
      .filter(([id, e]) => {
        if (!e.layers) return false;
        if (dragIds.has(id)) return true;
        // Released: shown while the object stays there and the server mesh still lags behind.
        if (!samePlacement(local.get(id), e.want)) return false;
        return e.base === mesh || !samePlacement(srv.get(id), e.want);
      })
      .map(([id]) => id);
    return ids.sort().join('\n');
  }, [live, dragIds, objects, serverObjects, mesh]);
  activeRef.current = useMemo(() => new Set(activeKey ? activeKey.split('\n') : []), [activeKey]);

  const displayMesh = useMemo(() => (mesh && activeKey ? maskMesh(mesh, new Set(activeKey.split('\n'))) : mesh), [mesh, activeKey]);
  const preview = useMemo(
    () => (activeKey ? previewMesh(activeKey.split('\n').map((id) => [id, live.get(id)!] as [string, LiveProxy])) : null),
    [live, activeKey],
  );

  return { displayMesh, preview, track };
}
