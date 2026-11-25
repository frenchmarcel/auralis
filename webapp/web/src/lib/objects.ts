import type { CatalogObject, Geometry, LayerKind, PlacedObject } from '../api';

export const OBJECT_LAYER_PREFIX = 'obj:';

/** obj:<catalog_id>:<part> -> parts; null for a room layer. */
export function parseObjectLayer(name: string): { catalogId: string; part: string } | null {
  if (!name.startsWith(OBJECT_LAYER_PREFIX)) return null;
  const rest = name.slice(OBJECT_LAYER_PREFIX.length);
  const i = rest.indexOf(':');
  if (i < 0) return { catalogId: rest, part: '' };
  return { catalogId: rest.slice(0, i), part: rest.slice(i + 1) };
}

export function layerKindOf(geometry: Geometry | null, name: string): LayerKind {
  const l = geometry?.layers.find((x) => x.name === name);
  if (l?.kind) return l.kind;
  return name.startsWith(OBJECT_LAYER_PREFIX) ? 'object' : 'room';
}

/** "<catalog name> · <part>" for object layers, the raw name otherwise. */
export function layerLabel(name: string, catalog: Map<string, CatalogObject>): string {
  const p = parseObjectLayer(name);
  if (!p) return name;
  const c = catalog.get(p.catalogId);
  return `${c?.name ?? p.catalogId}${p.part ? ` · ${p.part}` : ''}`;
}

export function instancesOf(name: string, objects: PlacedObject[]): number {
  const p = parseObjectLayer(name);
  if (!p) return 0;
  return objects.filter((o) => o.catalog_id === p.catalogId).length;
}

export function nextObjectId(list: PlacedObject[]): string {
  let max = 0;
  for (const o of list) {
    const m = /^O(\d+)$/.exec(o.id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `O${max + 1}`;
}
