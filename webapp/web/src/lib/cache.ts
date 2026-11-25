import { api, type Geometry, type MeshData, type ObjectCatalog } from '../api';

const geometries = new Map<string, Promise<Geometry>>();
const meshes = new Map<string, Promise<MeshData>>();
let catalog: Promise<ObjectCatalog> | null = null;

function memo<T>(map: Map<string, Promise<T>>, key: string, load: () => Promise<T>): Promise<T> {
  let p = map.get(key);
  if (!p) {
    p = load();
    map.set(key, p);
    p.catch(() => map.delete(key));
  }
  return p;
}

/** Geometries change only through this client (rename, objects): cache them per session. */
export function loadGeometry(gid: string): Promise<Geometry> {
  return memo(geometries, gid, () => api.getGeometry(gid));
}

export function loadMesh(gid: string): Promise<MeshData> {
  return memo(meshes, gid, () => api.getMesh(gid));
}

/** Store a fresh server copy (after an objects PUT). */
export function cacheGeometry(g: Geometry): void {
  geometries.set(g.id, Promise.resolve(g));
}

export function cacheMesh(gid: string, m: MeshData): void {
  meshes.set(gid, Promise.resolve(m));
}

export function forgetGeometry(gid: string): void {
  geometries.delete(gid);
  meshes.delete(gid);
}

export function loadCatalog(): Promise<ObjectCatalog> {
  if (!catalog) {
    catalog = api.getObjectCatalog();
    catalog.catch(() => {
      catalog = null;
    });
  }
  return catalog;
}
