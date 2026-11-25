import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import {
  api,
  errorMessage,
  type CatalogObject,
  type Geometry,
  type Material,
  type MeshData,
  type ObjectCatalog,
  type PathValidity,
  type PlacedObject,
  type PointValidity,
  type Receiver,
  type Simulation,
  type Source,
  type Validation,
  type Vec3,
} from '../../api';
import { useApp } from '../../AppContext';
import { Timeline } from '../../components/Timeline';
import { TopBar } from '../../components/TopBar';
import { anyModalOpen, EmptyState, ErrorBox, isTypingTarget, Menu, Segmented, Spinner, StatusBadge } from '../../components/ui';
import { cacheGeometry, loadCatalog, loadGeometry, loadMesh } from '../../lib/cache';
import { GeometryCheckDialog } from './GeometryCheckDialog';
import { ABS_MAX, absorptionColor, absorptionGradientCss, COLORS, layerColor } from '../../lib/colors';
import { fmtNum } from '../../lib/format';
import { boundsOf, boundsOfPositions, dirFromAzEl, roundVec, type Bounds } from '../../lib/geom';
import { href, navigate } from '../../lib/router';
import { instancesOf, layerLabel, nextObjectId, parseObjectLayer } from '../../lib/objects';
import { useTimeStore } from '../../lib/timeline';
import { appendWaypoint, deleteWaypoint, endTime, MAX_PATH_S, MAX_WAYPOINTS, positionAt, setWaypointPosition, translatePath, withPath, type Movable } from '../../lib/trajectory';
import { useLatest } from '../../lib/useLatest';
import { isSmooth, smoothCurve } from '../../lib/smoothPath';
import { locateSimulation, nextName } from '../../lib/tree';
import { PlaceGhost, PlacedObjects, type GizmoMode, type ObjectPlacement, type ObjectViewMode } from '../../three/Objects';
import { PathLines, TimedPointMarkers, type PathDraw, type WaypointRef } from '../../three/Paths';
import { LayerMeshes, SceneCanvas, markerRadius, type FrameRequest, type LayerStyle, type PointMarker } from '../../three/Scene';
import { ViewLookProvider } from '../../three/Realistic';
import { surfaceKindOf } from '../../lib/surfaces';
import { CutSlider, useViewLookState, ViewLookScene, ViewLookToolbar } from './ViewLook';
import { SubsurfaceHighlight, type SubsurfaceFocus } from '../../three/SubsurfaceHighlight';
import { isLocalOnly, MaterialLibrary } from './MaterialLibrary';
import { duplicateMaterial, renameMaterial } from './materialActions';
import { MaterialsTab } from './MaterialsTab';
import type { Scattering } from './ScatteringCell';
import { NewMaterialDialog } from './NewMaterialDialog';
import { EnsembleDialog, facingTowards, stageDefaults, type StagePlacement, type StagePreview } from './EnsembleDialog';
import { EnsembleStagePreview } from '../../three/EnsemblePreview';
import { ObjectsTab } from './ObjectsTab';
import { duplicateObjects, objectsBounds, useObjectKeys, useObjectSelection, type SelectMods } from './objectEditing';
import { useLiveProxy } from './useLiveProxy';
import { appendPoint, limitMessage, PointsTab, type PointKind } from './PointsTab';
import { pointFromSurfaceClick } from './surfacePlace';
import { isActiveJob, RunConfirmDialog, RunDialog, useSimulationJob } from './RunDialog';
import { SettingsTab } from './SettingsTab';
import { useObjectsSync } from './useObjectsSync';
import { useSimulationStore, type SaveState } from './useSimulationStore';
import { GridsTab, type GridPick } from './GridsTab';
import { gridCenterFromClick } from './gridPick';
import { useGridPreviews } from './useGridPreviews';
import { GridPointsPreview, type GridPointHover, type GridPointsDraw } from '../../three/GridPoints';
import type { ReceiverGrid } from '../../api';
import type { EditMode } from './useObjectsSync';
import { useModifyPrompt } from './ModifyPrompt';
import { AbsorptionMarks, AbsorptionTicks, CaptureFrame, CapturePanel, DensityPanel, ParticlesPanel, useCaptureShortcut, useViewTools, ViewToolsScene, VisualPanel } from './ViewTools';
import { alphaAt, TOOL_BANDS, toolBandLabel } from '../../lib/toolRamps';

type Tab = 'materials' | 'sources' | 'receivers' | 'settings' | 'objects' | 'grids';

const CONDUCTOR_DIMS: Vec3 = [0.5, 0.5, 1.8];
/** A waypoint clicked on a wall is set back this far from it (m). */
const WALL_OFFSET_M = 0.3;

function sceneRoot(o: THREE.Object3D): THREE.Object3D {
  let r = o;
  while (r.parent) r = r.parent;
  return r;
}

/** Height of the first floor-like surface straight below p, among `meshes`. */
function floorBelow(meshes: THREE.Object3D[], p: Vec3): number | null {
  const ray = new THREE.Raycaster(new THREE.Vector3(p[0], p[1], p[2] + 0.01), new THREE.Vector3(0, 0, -1));
  for (const h of ray.intersectObjects(meshes, false)) {
    if (!h.face) continue;
    const n = h.face.normal.clone().transformDirection(h.object.matrixWorld);
    if (Math.abs(n.z) > 0.3) return h.point.z;
  }
  return null;
}

/** True when some surface of `meshes` lies straight above p (p is under a ceiling, not on the roof). */
function covered(meshes: THREE.Object3D[], p: THREE.Vector3): boolean {
  return new THREE.Raycaster(new THREE.Vector3(p.x, p.y, p.z + 0.02), new THREE.Vector3(0, 0, 1)).intersectObjects(meshes, false).length > 0;
}

export function blockingReasons(sim: Simulation, v: Validation | null): string[] {
  if (!v) return ['Validation not loaded yet'];
  if (v.ok_to_run) return [];
  // The server's own wording wins when it sends it (field not in API.md).
  if (v.blocking && v.blocking.length) return v.blocking;
  const r: string[] = [];
  if (v.unassigned_layers.length) {
    const names = v.unassigned_layers.slice(0, 4).join(', ');
    r.push(`${v.unassigned_layers.length} unassigned layer${v.unassigned_layers.length === 1 ? '' : 's'}: ${names}${v.unassigned_layers.length > 4 ? ', …' : ''}`);
  }
  if (!sim.sources.length) r.push('No source');
  if (!sim.receivers.length) r.push('No receiver');
  const nameOf = (list: { id: string; name: string }[], id: string) => list.find((x) => x.id === id)?.name ?? id;
  for (const s of v.sources) if (!s.valid) r.push(`${nameOf(sim.sources, s.id)}: ${s.reasons.join('; ') || 'invalid'}`);
  for (const s of v.receivers) if (!s.valid) r.push(`${nameOf(sim.receivers, s.id)}: ${s.reasons.join('; ') || 'invalid'}`);
  const st = sim.settings;
  const usesWave = st.mode === 'default' || (st.mode === 'advanced' && st.solvers !== 'geometrical');
  if (usesWave && !v.wave_available) r.push('The model is not closed: fix its open edges, or declare it an open scene in Settings (or use the geometrical solver only)');
  if (r.length === 0) r.push('The server reports that the simulation cannot run');
  return r;
}

export function Editor({ simId }: { simId: string }) {
  const app = useApp();
  const store = useSimulationStore(simId);
  const { sim, validation } = store;
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [mesh, setMesh] = useState<MeshData | null>(null);
  const [geoError, setGeoError] = useState<string | null>(null);
  const gid = sim?.geometry_id;

  useEffect(() => {
    if (!gid) return;
    let cancelled = false;
    Promise.all([loadGeometry(gid), loadMesh(gid)])
      .then(([g, m]) => {
        if (cancelled) return;
        setGeometry(g);
        setMesh(m);
      })
      .catch((e) => !cancelled && setGeoError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [gid]);

  const [tab, setTab] = useState<Tab>('materials');
  const [selLayers, setSelLayers] = useState<Set<string>>(new Set());
  const anchor = useRef<string | null>(null);
  const [selPoint, setSelPoint] = useState<{ kind: PointKind; id: string } | null>(null);
  const [selWp, setSelWp] = useState<WaypointRef | null>(null);
  const [drawPath, setDrawPath] = useState<{ kind: PointKind; id: string } | null>(null);
  /** Sources/Receivers "from surface": each click on a room surface adds a point offset along its normal. */
  const [surfacePick, setSurfacePick] = useState<{ kind: PointKind; distance: number } | null>(null);
  /** Points selected together in the list (Ctrl/Shift-click), highlighted in 3D. */
  const [multiSel, setMultiSel] = useState<{ kind: PointKind; ids: string[] } | null>(null);
  const timeStore = useTimeStore();
  const [tlPlaying, setTlPlaying] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<string | null>(null);
  const [absOn, setAbsOn] = useState(false);
  const [absBand, setAbsBand] = useState(3);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [newMatOpen, setNewMatOpen] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [checkOpen, setCheckOpen] = useState(false);
  /** Run button: the validation at the time, shown with the estimate before launching. */
  const [confirmRun, setConfirmRun] = useState<Validation | null>(null);
  const [frame, setFrame] = useState<FrameRequest | null>(null);
  const frameKey = useRef(0);
  const gizmoAt = useRef(0);
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>('translate');
  const [placeId, setPlaceId] = useState<string | null>(null);
  const [stage, setStage] = useState<StagePlacement | null>(null);
  const [stagePick, setStagePick] = useState(false);
  const [stagePreview, setStagePreview] = useState<StagePreview | null>(null);
  const lastStage = useRef<StagePlacement | null>(null);
  const [ensembleNote, setEnsembleNote] = useState<{
    title: string;
    added: string[];
    replaced: Source[];
    scale: number;
    n: number;
  } | null>(null);
  const [viewMode, setViewMode] = useState<ObjectViewMode>('visual');
  const [visualErrors, setVisualErrors] = useState<Record<string, string>>({});
  const ghostBus = useRef<(p: Vec3 | null) => void>(() => undefined);
  const [catalog, setCatalog] = useState<ObjectCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  // Receiver grids: selection, 3D pick mode, hidden previews, hovered point.
  const [selGrid, setSelGrid] = useState<string | null>(null);
  const [gridPick, setGridPick] = useState<GridPick | null>(null);
  const [gridHidden, setGridHidden] = useState<Set<string>>(new Set());
  const [gridHover, setGridHover] = useState<GridPointHover | null>(null);
  const gridPreviews = useGridPreviews(sim, store.saveState === 'saved', mesh);

  useEffect(() => {
    loadCatalog()
      .then(setCatalog)
      .catch((e) => setCatalogError(errorMessage(e)));
  }, []);
  const catalogById = useMemo(() => new Map<string, CatalogObject>((catalog?.objects ?? []).map((c) => [c.id, c])), [catalog]);

  const objSyncRaw = useObjectsSync({
    geometry,
    simId,
    store,
    onServerGeometry: setGeometry,
    onServerMesh: setMesh,
    onError: (msg) => app.toast('error', msg),
  });
  // First object edit of a simulation with up-to-date results: held locally until the prompt answers.
  const objHold = useRef(false);
  const guardedObjEdit = useLatest((fn: (l: PlacedObject[]) => PlacedObject[], mode?: EditMode) => {
    if (objHold.current) {
      objSyncRaw.edit(fn, 'live');
      return;
    }
    if (modify.needsPrompt()) {
      objHold.current = true;
      objSyncRaw.edit(fn, 'live');
      void modify.choose('objects').then((c) => {
        objHold.current = false;
        if (c === 'modify') void objSyncRaw.flush();
        else objSyncRaw.discard();
      });
      return;
    }
    objSyncRaw.edit(fn, mode);
  });
  const objSync = { ...objSyncRaw, edit: guardedObjEdit };
  const objects = objSync.objects;
  // Selected objects (Ctrl/Shift-click adds), the last clicked being the primary one.
  const objSel = useObjectSelection(objects);
  const selObjects = objSel.ids;
  const selObject = objSel.primary;
  const setSelObject = objSel.clear;
  const liveProxy = useLiveProxy({ mesh, objects, serverObjects: geometry?.objects, catalogById, onError: (msg) => app.toast('warning', msg) });

  useEffect(
    () => () => {
      document.body.style.cursor = '';
    },
    [],
  );

  const jobState = useSimulationJob(simId, (j) => {
    void store.reload();
    void app.refreshTree();
    if (j.status === 'done') app.toast('success', 'Simulation finished');
    else if (j.status === 'failed') app.toast('error', `Simulation failed${j.error ? `: ${j.error}` : ''}`);
    else app.toast('warning', `Simulation ${j.status}`);
  });
  const job = jobState.job;

  const materials = app.materials;
  const materialsById = useMemo(() => new Map((materials?.materials ?? []).map((m) => [m.id, m])), [materials]);
  const librariesById = useMemo(() => new Map((materials?.libraries ?? []).map((l) => [l.id, l])), [materials]);
  const categories = useMemo(() => Array.from(new Set((materials?.materials ?? []).map((m) => m.category).filter(Boolean))), [materials]);
  // 3D tools (particles, reflection density, capture, visual model), exclusive with the α colouring.
  const vt = useViewTools({
    mesh,
    sim,
    geometry,
    materialsById,
    selLayers,
    selectedSourceId: selPoint?.kind === 'source' ? selPoint.id : null,
    absOn,
    setAbsOn,
    onGeometry: setGeometry,
    toast: app.toast,
  });
  useCaptureShortcut(vt, !!mesh);

  const layerKind = useMemo(
    () => new Map((geometry?.layers ?? []).map((l) => [l.name, l.kind ?? (l.name.startsWith('obj:') ? 'object' : 'room')])),
    [geometry],
  );
  // Room layers first, then object layers: the order of the Materials table (Shift ranges).
  const layerOrder = useMemo(() => {
    const names = geometry?.layers.map((l) => l.name) ?? [];
    return [...names.filter((n) => layerKind.get(n) !== 'object'), ...names.filter((n) => layerKind.get(n) === 'object')];
  }, [geometry, layerKind]);
  const ownersByLayer = useMemo(
    () => new Map((liveProxy.displayMesh?.layers ?? []).filter((l) => l.owners).map((l) => [l.name, l.owners!])),
    [liveProxy.displayMesh],
  );
  const labelOf = useCallback((n: string) => layerLabel(n, catalogById), [catalogById]);
  const colorByName = useMemo(() => new Map(layerOrder.map((n, i) => [n, layerColor(i)])), [layerOrder]);
  const bounds = useMemo<Bounds | null>(() => (geometry ? boundsOf(geometry.bbox.min, geometry.bbox.max) : null), [geometry]);
  // Display look: realistic (textures from the materials) by default; α colouring shows flat colours.
  const look = useViewLookState({ geometry, sim: sim ?? null, bounds, toast: app.toast });
  const shading = absOn && look.shading === 'realistic' ? 'shaded' : look.shading;
  const surfaceByLayer = useMemo(
    () => new Map(layerOrder.map((n) => [n, surfaceKindOf(materialsById.get(sim?.layers[n]?.material_id ?? ''), n)])),
    [layerOrder, sim?.layers, materialsById],
  );
  const layerBounds = useMemo(() => {
    const out = new Map<string, Bounds>();
    if (!mesh) return out;
    for (const l of mesh.layers) {
      const pts: number[] = [];
      for (const i of l.indices) pts.push(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]);
      const b = boundsOfPositions(pts);
      if (b) out.set(l.name, b);
    }
    return out;
  }, [mesh]);

  const srcValidity = useMemo(() => new Map<string, PointValidity>((validation?.sources ?? []).map((v) => [v.id, v])), [validation]);
  const rcvValidity = useMemo(() => new Map<string, PointValidity>((validation?.receivers ?? []).map((v) => [v.id, v])), [validation]);
  const pathValidity = useMemo(() => new Map<string, PathValidity>((validation?.paths ?? []).map((v) => [v.id, v])), [validation]);
  const movables = useMemo(() => new Map<string, Movable>(sim ? [...sim.sources, ...sim.receivers].map((x) => [x.id, x]) : []), [sim]);
  const tMax = useMemo(() => Math.max(0, ...[...movables.values()].map(endTime)), [movables]);
  // The selected waypoint, while it still exists and belongs to the selected point.
  const wpSel = useMemo(() => {
    if (!selWp || !selPoint || selWp.kind !== selPoint.kind || selWp.id !== selPoint.id || !sim) return null;
    const it = (selWp.kind === 'source' ? sim.sources : sim.receivers).find((x) => x.id === selWp.id);
    return it?.path && selWp.index < it.path.waypoints.length ? selWp : null;
  }, [selWp, selPoint, sim]);

  const locked = !!sim && (sim.status === 'queued' || sim.status === 'running' || isActiveJob(job));

  const selectLayer = useCallback(
    (name: string, mods: { ctrl: boolean; shift: boolean }) => {
      setSelPoint(null);
      setSelObject(null);
      setSelLayers((prev) => {
        if (mods.shift && anchor.current) {
          const a = layerOrder.indexOf(anchor.current);
          const b = layerOrder.indexOf(name);
          if (a >= 0 && b >= 0) {
            const [lo, hi] = a < b ? [a, b] : [b, a];
            const n = mods.ctrl ? new Set(prev) : new Set<string>();
            layerOrder.slice(lo, hi + 1).forEach((x) => n.add(x));
            return n;
          }
        }
        anchor.current = name;
        if (mods.ctrl) {
          const n = new Set(prev);
          if (n.has(name)) n.delete(name);
          else n.add(name);
          return n;
        }
        return new Set([name]);
      });
    },
    [layerOrder],
  );

  const clearSelection = useCallback(() => {
    setSelLayers(new Set());
    setSelPoint(null);
    setSelWp(null);
    setSelObject(null);
  }, []);

  const selectObjectIn = objSel.select;
  const selectObject = useCallback(
    (id: string | null, mods?: SelectMods) => {
      selectObjectIn(id, mods);
      setSelLayers(new Set());
      setSelPoint(null);
    },
    [selectObjectIn],
  );

  const openLibrary = (name?: string) => {
    if (locked || !sim) return;
    if (name && !selLayers.has(name)) {
      setSelLayers(new Set([name]));
      anchor.current = name;
    }
    const ref = name ?? [...selLayers][0];
    const mid = ref ? sim.layers[ref]?.material_id : null;
    if (mid) setPreviewId(mid);
    setLibraryOpen(true);
  };

  const assign = (materialId: string) => {
    const names = [...selLayers];
    if (!names.length) return;
    store.update((s) => {
      const layers = { ...s.layers };
      for (const n of names) layers[n] = { ...(layers[n] ?? { material_id: null, scattering: null }), material_id: materialId };
      return { ...s, layers };
    });
  };

  const setScattering = (name: string, v: number | number[] | null) =>
    store.update((s) => ({ ...s, layers: { ...s.layers, [name]: { ...(s.layers[name] ?? { material_id: null, scattering: null }), scattering: v } } }));
  /** The per-layer edit above, for several layers in one update (one save). */
  const setScatteringMany = (names: string[], v: Scattering) =>
    store.update((s) => {
      const layers = { ...s.layers };
      for (const n of names) layers[n] = { ...(layers[n] ?? { material_id: null, scattering: null }), scattering: Array.isArray(v) ? [...v] : v };
      return { ...s, layers };
    });
  const [subFocus, setSubFocus] = useState<SubsurfaceFocus | null>(null);

  const setSources =(items: (Source | Receiver)[]) => store.update((s) => ({ ...s, sources: items as Source[] }));
  const setReceivers = (items: (Source | Receiver)[]) => store.update((s) => ({ ...s, receivers: items as Receiver[] }));

  const updatePoint = (kind: PointKind, id: string, fn: (x: Source | Receiver) => Source | Receiver) =>
    store.update((s) =>
      kind === 'source'
        ? { ...s, sources: s.sources.map((x) => (x.id === id ? (fn(x) as Source) : x)) }
        : { ...s, receivers: s.receivers.map((x) => (x.id === id ? (fn(x) as Receiver) : x)) },
    );

  /** Dragging a moving point's marker translates its whole trajectory (shape and times kept). */
  const movePoint = (kind: PointKind, id: string, pos: Vec3) =>
    updatePoint(kind, id, (x) => {
      if (!x.path) return { ...x, position: roundVec(pos, 3) };
      const at = positionAt(x, timeStore.get());
      return withPath(x, translatePath(x.path, [pos[0] - at[0], pos[1] - at[1], pos[2] - at[2]]));
    });

  const moveWaypoint = (w: WaypointRef, pos: Vec3) =>
    updatePoint(w.kind, w.id, (x) => (x.path && w.index < x.path.waypoints.length ? withPath(x, setWaypointPosition(x.path, w.index, pos)) : x));

  const deleteSelectedWaypoint = useLatest(() => {
    if (!wpSel || locked || !sim) return;
    const it = (wpSel.kind === 'source' ? sim.sources : sim.receivers).find((x) => x.id === wpSel.id);
    if (!it?.path) return;
    if (it.path.waypoints.length <= 2) {
      app.toast('warning', 'A trajectory needs at least 2 waypoints (use “Remove” in the list to make the point static).');
      return;
    }
    const k = wpSel.index;
    updatePoint(wpSel.kind, wpSel.id, (x) => (x.path ? withPath(x, deleteWaypoint(x.path, k)) : x));
    setSelWp(null);
  });

  /** "Add waypoints" mode: the clicked surface point, at the previous waypoint's height above its floor. */
  const addWaypointFromClick = (e: ThreeEvent<MouseEvent>) => {
    if (!drawPath || !sim || !bounds || locked) return;
    const it = (drawPath.kind === 'source' ? sim.sources : sim.receivers).find((x) => x.id === drawPath.id);
    if (!it) return;
    const meshes: THREE.Object3D[] = [];
    sceneRoot(e.object).traverse((o) => {
      const ln = o.userData?.layer as string | undefined;
      if (ln && (o as THREE.Mesh).isMesh && layerKind.get(ln) !== 'object') meshes.push(o);
    });
    // Normals turned towards the camera.
    const facing = (i: THREE.Intersection) => {
      const n = i.face ? i.face.normal.clone().transformDirection(i.object.matrixWorld) : new THREE.Vector3(0, 0, 1);
      return n.dot(e.ray.direction) > 0 ? n.negate() : n;
    };
    const hits = e.intersections.filter((i) => {
      const ln = i.object.userData?.layer as string | undefined;
      return !!ln && layerKind.get(ln) !== 'object';
    });
    // Seen from outside a closed room the roof comes first: take the first floor under a ceiling.
    const hit = hits.find((i) => facing(i).z > 0.7 && covered(meshes, i.point)) ?? hits[0];
    if (!hit) return;
    const base = it.path ?? { waypoints: [{ position: it.position, t: 0 }], orientation: 'fixed' as const, target_id: null };
    const prev = base.waypoints[base.waypoints.length - 1].position;
    const height = Math.max(0.1, prev[2] - (floorBelow(meshes, prev) ?? bounds.min[2]));
    const n = facing(hit);
    let at: Vec3 = [hit.point.x, hit.point.y, hit.point.z];
    let floor: number | null = hit.point.z;
    if (n.z <= 0.7) {
      // A wall or a ceiling: step back from it, then down to the floor beneath.
      const h = Math.hypot(n.x, n.y);
      if (h > 0.3) at = [at[0] + (n.x / h) * WALL_OFFSET_M, at[1] + (n.y / h) * WALL_OFFSET_M, at[2]];
      floor = floorBelow(meshes, at);
    }
    const path = appendWaypoint(base, [at[0], at[1], (floor ?? hit.point.z) + height]);
    if (!path) {
      app.toast('warning', `A trajectory is limited to ${MAX_WAYPOINTS} waypoints and ${MAX_PATH_S} s.`);
      return;
    }
    updatePoint(drawPath.kind, drawPath.id, (x) => withPath(x, path));
    setSelWp({ kind: drawPath.kind, id: drawPath.id, index: path.waypoints.length - 1 });
  };

  const startSurfacePick = (kind: PointKind, distance: number | null) => {
    if (distance === null || locked) {
      setSurfacePick((cur) => (cur && cur.kind === kind ? null : cur));
      return;
    }
    setPlaceId(null);
    setStagePick(false);
    setDrawPath(null);
    setGridPick(null);
    ghostBus.current(null);
    setSurfacePick({ kind, distance });
  };

  const addPointFromSurface = (e: ThreeEvent<MouseEvent>) => {
    if (!surfacePick || !sim || locked) return;
    // closed room (not a declared open scene): keep the new point on the inner side
    const closed = !!geometry?.check.watertight && !validation?.open_scene && mesh ? mesh : null;
    const pos = pointFromSurfaceClick(e, surfacePick.distance, closed);
    if (!pos) return;
    const kind = surfacePick.kind;
    const r = appendPoint(kind, kind === 'source' ? sim.sources : sim.receivers, pos);
    if (!r) {
      app.toast('warning', limitMessage(kind, 1, 0));
      setSurfacePick(null);
      return;
    }
    if (kind === 'source') setSources(r.items);
    else setReceivers(r.items);
    setSelPoint({ kind, id: r.added.id });
  };

  const startDrawing = (kind: PointKind, id: string | null) => {
    if (!id || locked) {
      setDrawPath(null);
      return;
    }
    setPlaceId(null);
    setStagePick(false);
    setSurfacePick(null);
    ghostBus.current(null);
    setSelPoint({ kind, id });
    setDrawPath({ kind, id });
  };

  // The add-waypoints mode ends when its point is no longer selected, or editing locks.
  useEffect(() => {
    if (drawPath && (locked || !selPoint || selPoint.kind !== drawPath.kind || selPoint.id !== drawPath.id)) setDrawPath(null);
  }, [drawPath, selPoint, locked]);

  const deleteSelectedPoint = useCallback(() => {
    if (!selPoint || locked) return;
    store.update((s) =>
      selPoint.kind === 'source'
        ? { ...s, sources: s.sources.filter((x) => x.id !== selPoint.id) }
        : { ...s, receivers: s.receivers.filter((x) => x.id !== selPoint.id) },
    );
    setSelPoint(null);
  }, [selPoint, locked, store]);

  const addObject = (catalogId: string, at?: Vec3, keepPlacing = false) => {
    if (locked || !bounds) return;
    const cat = catalogById.get(catalogId);
    if (!cat) return;
    const pos: Vec3 = at ?? [bounds.center[0], bounds.center[1], bounds.min[2]];
    let newId = '';
    objSync.edit((list) => {
      newId = nextObjectId(list);
      const same = list.filter((o) => o.catalog_id === catalogId).length;
      const item: PlacedObject = {
        id: newId,
        catalog_id: catalogId,
        name: same ? `${cat.name} ${same + 1}` : cat.name,
        position: roundVec(pos, 3),
        rotation_deg: 0,
        scale: [1, 1, 1],
      };
      return [...list, item];
    }, 'now');
    selectObject(newId);
    setTab('objects');
    if (!keepPlacing) {
      setPlaceId(null);
      ghostBus.current(null);
    }
  };

  const patchObject = (id: string, patch: Partial<PlacedObject>) => {
    if (locked) return;
    objSync.edit((list) => list.map((o) => (o.id === id ? { ...o, ...patch } : o)));
  };

  /** Gizmo drag of the selection: live edits plus the live proxy preview, the PUT on release. */
  const transformObject = (items: ObjectPlacement[], final: boolean) => {
    if (locked) return;
    const by = new Map(items.map((t) => [t.id, t]));
    let moved: PlacedObject[] = [];
    objSync.edit(
      (list) => {
        const next = list.map((o) => {
          const t = by.get(o.id);
          return t ? { ...o, position: roundVec(t.position, 3), rotation_deg: t.rotation_deg } : o;
        });
        moved = next.filter((o) => by.has(o.id));
        return next;
      },
      final ? 'now' : 'live',
    );
    liveProxy.track(moved, final);
  };

  const duplicateSelection = () => {
    if (locked || !selObjects.length) return;
    let added: string[] = [];
    objSync.edit((list) => {
      const r = duplicateObjects(list, selObjects, catalogById);
      added = r.added;
      return r.list;
    }, 'now');
    if (added.length) objSel.setAll(added);
  };

  /** Undo (-1) or redo (+1) one object change: the list of that step is sent again. */
  const stepObjects = useLatest((dir: -1 | 1) => {
    if (locked) return;
    const list = objSync.step(dir);
    if (list) objSync.edit(() => list, 'now');
  });
  useObjectKeys({
    enabled: tab === 'objects' || selObjects.length > 0,
    onUndo: () => stepObjects(-1),
    onRedo: () => stepObjects(1),
    onSelectAll:
      tab === 'objects'
        ? () => {
            objSel.setAll(objects.map((o) => o.id));
            setSelLayers(new Set());
            setSelPoint(null);
          }
        : undefined,
  });

  const duplicateObject = (id: string) => {
    if (locked) return;
    let newId = '';
    objSync.edit((list) => {
      const o = list.find((x) => x.id === id);
      if (!o) return list;
      const cat = catalogById.get(o.catalog_id);
      newId = nextObjectId(list);
      const dx = (cat ? cat.dims[0] * o.scale[0] : 0.5) + 0.1;
      return [...list, { ...o, id: newId, name: `${o.name} copy`, position: roundVec([o.position[0] + dx, o.position[1], o.position[2]], 3) }];
    }, 'now');
    if (newId) selectObject(newId);
  };

  // The deleted ids drop out of the selection by themselves (and come back with an undo).
  const deleteObjects = useCallback(
    (ids: readonly string[]) => {
      if (locked || !ids.length) return;
      objSync.edit((list) => list.filter((o) => !ids.includes(o.id)), 'now');
    },
    [locked, objSync],
  );

  const focusObject = (id: string) => {
    const o = objects.find((x) => x.id === id);
    if (!o) return;
    const cat = catalogById.get(o.catalog_id);
    const d: Vec3 = cat ? [cat.dims[0] * o.scale[0], cat.dims[1] * o.scale[1], cat.dims[2] * o.scale[2]] : [1, 1, 1];
    selectObject(id);
    setFrame({ center: [o.position[0], o.position[1], o.position[2] + d[2] / 2], radius: Math.max(0.6, Math.hypot(d[0], d[1], d[2]) * 0.8), key: ++frameKey.current });
  };

  const frameSelection = useCallback(() => {
    if (!sim || !bounds) return;
    if (selObjects.length > 1) {
      const b = objectsBounds(objects.filter((o) => selObjects.includes(o.id)), catalogById);
      if (b) setFrame({ center: b.center, radius: Math.max(0.6, b.radius * 1.6), key: ++frameKey.current });
      return;
    }
    if (selObject) {
      focusObject(selObject);
      return;
    }
    if (selPoint) {
      const list = selPoint.kind === 'source' ? sim.sources : sim.receivers;
      const p = list.find((x) => x.id === selPoint.id);
      if (p) setFrame({ center: positionAt(p, timeStore.get()), radius: Math.max(1, bounds.radius * 0.2), key: ++frameKey.current });
      return;
    }
    if (selLayers.size) {
      const min: Vec3 = [Infinity, Infinity, Infinity];
      const max: Vec3 = [-Infinity, -Infinity, -Infinity];
      for (const n of selLayers) {
        const b = layerBounds.get(n);
        if (!b) continue;
        for (let k = 0; k < 3; k++) {
          min[k] = Math.min(min[k], b.min[k]);
          max[k] = Math.max(max[k], b.max[k]);
        }
      }
      if (Number.isFinite(min[0])) {
        const b = boundsOf(min, max);
        setFrame({ center: b.center, radius: b.radius, key: ++frameKey.current });
        return;
      }
    }
    setFrame({ center: bounds.center, radius: bounds.radius, key: ++frameKey.current });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sim, bounds, selPoint, selLayers, layerBounds, selObject, selObjects, objects, catalogById]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || anyModalOpen() || isTypingTarget(e.target)) return;
      if (e.key === 'Escape') {
        if (surfacePick) setSurfacePick(null);
        else if (gridPick) setGridPick(null);
        else if (drawPath) setDrawPath(null);
        else if (stagePick) {
          setStagePick(false);
          ghostBus.current(null);
        } else if (placeId) {
          setPlaceId(null);
          ghostBus.current(null);
        } else if (libraryOpen) setLibraryOpen(false);
        else clearSelection();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selObjects.length) {
          e.preventDefault();
          deleteObjects(selObjects);
        } else if (wpSel) {
          e.preventDefault();
          deleteSelectedWaypoint();
        } else if (selPoint) {
          e.preventDefault();
          deleteSelectedPoint();
        }
      } else if (selObject && !e.ctrlKey && !e.metaKey && (e.key === 't' || e.key === 'r')) {
        setGizmoMode(e.key === 't' ? 'translate' : 'rotate');
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a' && tab === 'materials') {
        e.preventDefault();
        setSelLayers(new Set(layerOrder));
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === 'f') {
        frameSelection();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [libraryOpen, clearSelection, selPoint, deleteSelectedPoint, tab, layerOrder, frameSelection, placeId, selObject, selObjects, deleteObjects, stagePick, drawPath, wpSel, deleteSelectedWaypoint, gridPick, surfacePick]);

  const siblings = useMemo(() => {
    const loc = locateSimulation(app.tree, simId);
    return loc?.geometry.simulations ?? (sim ? [{ ...sim }] : []);
  }, [app.tree, simId, sim]);
  const siblingNames = useMemo(() => siblings.map((s) => s.name), [siblings]);
  const toastError = useLatest((m: string) => app.toast('error', m));
  const modify = useModifyPrompt({ store, siblingNames, onError: toastError, onTreeChanged: app.refreshTree });

  const switchTo = async (id: string) => {
    if (id === simId) return;
    await store.flush();
    navigate(href.editor(id));
  };

  const newSimulation = async (copy: boolean) => {
    if (!sim) return;
    const name = await app.prompt({
      title: copy ? 'Duplicate simulation' : 'New simulation',
      label: 'Name',
      initial: copy ? `${sim.name} - Copy` : nextName('Simulation', siblings.map((s) => s.name)),
      confirmLabel: copy ? 'Duplicate' : 'Create',
    });
    if (!name) return;
    try {
      if (copy) await store.flush();
      const created = await api.createSimulation(sim.geometry_id, name, copy ? sim.id : null);
      await app.refreshTree();
      navigate(href.editor(created.id));
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  const renameSimulation = async () => {
    if (!sim) return;
    const name = await app.prompt({ title: 'Rename simulation', label: 'Name', initial: sim.name, confirmLabel: 'Rename' });
    if (!name || name === sim.name) return;
    store.update((s) => ({ ...s, name }));
    if (await store.flush()) void app.refreshTree();
  };

  const deleteSimulation = async () => {
    if (!sim) return;
    const ok = await app.confirm({
      title: 'Delete simulation',
      message: (
        <>
          Delete <b>{sim.name}</b> and its results? This cannot be undone.
        </>
      ),
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteSimulation(sim.id);
      const t = await app.refreshTree();
      const rest = (locateSimulation(t, simId)?.geometry.simulations ?? siblings).filter((s) => s.id !== sim.id);
      navigate(rest.length ? href.editor(rest[0].id) : href.home());
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  const duplicateGeometry = async () => {
    if (!sim || !geometry) return;
    const name = await app.prompt({
      title: 'Duplicate geometry',
      label: 'Name of the copy (same room, same objects; this simulation is copied into it)',
      initial: `${geometry.name} (copy)`,
      confirmLabel: 'Duplicate',
    });
    if (!name) return;
    try {
      if (!(await store.flush())) throw new Error('the simulation could not be saved first');
      if (!(await objSync.flush())) throw new Error('the objects could not be saved first');
      const g2 = await api.duplicateGeometry(geometry.id, name);
      const sim2 = await api.createSimulation(g2.id, sim.name, sim.id);
      await app.refreshTree();
      app.toast('success', `Geometry “${g2.name}” created with a copy of “${sim.name}”`);
      navigate(href.editor(sim2.id));
    } catch (e) {
      app.toast('error', `Duplicate geometry: ${errorMessage(e)}`);
    }
  };

  const startRun = async () => {
    if (!sim) return;
    if (!(await objSync.flush())) {
      app.toast('error', 'The objects could not be saved; the simulation was not started.');
      return;
    }
    if (!(await store.flush())) {
      app.toast('error', 'The simulation could not be saved; it was not started.');
      return;
    }
    const v = await store.refreshValidation();
    if (!v || !v.ok_to_run) {
      app.toast('warning', `Not ready to run: ${blockingReasons(sim, v).join(' · ')}`);
      return;
    }
    setConfirmRun(v);
  };

  const launchRun = async () => {
    if (!sim) return;
    setConfirmRun(null);
    try {
      const j = await api.runSimulation(sim.id);
      jobState.setJob(j);
      setRunOpen(true);
      void store.reload();
      void app.refreshTree();
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  const vtRoomDisplay = vt.visual.active && vt.visual.roomDisplay !== 'normal' ? vt.visual.roomDisplay : null;
  const styleOf = useCallback(
    (name: string) => {
      let color = colorByName.get(name) ?? COLORS.neutralLayer;
      if (absOn && sim) {
        const mid = sim.layers[name]?.material_id;
        const m = mid ? materialsById.get(mid) : undefined;
        color = m ? absorptionColor(alphaAt(m, absBand).alpha) : COLORS.unassigned;
      }
      const st: LayerStyle = {
        color,
        selected: selLayers.has(name),
        hovered: hover === name,
        hidden: hidden.has(name),
        surface: surfaceByLayer.get(name),
        clip: layerKind.get(name) !== 'object',
      };
      // Room surfaces under a shown visual model (.glb): as chosen in its panel.
      if (vtRoomDisplay && layerKind.get(name) !== 'object') st.display = vtRoomDisplay;
      if (layerKind.get(name) === 'object') {
        const cid = parseObjectLayer(name)?.catalogId;
        const cat = cid ? catalogById.get(cid) : undefined;
        // Objects without a (working) visual model always show their proxy.
        const visualShown = !!cat?.visual && !!cat.visual_url && !(cid && visualErrors[cid]);
        if (visualShown) st.display = viewMode === 'visual' ? 'invisible' : viewMode === 'both' ? 'overlay' : 'normal';
        // In placement mode only the room is a target.
        if (placeId || stagePick || drawPath || gridPick?.mode === 'center' || surfacePick) st.pickable = false;
      }
      return st;
    },
    [colorByName, absOn, sim, materialsById, absBand, selLayers, hover, hidden, layerKind, catalogById, visualErrors, viewMode, placeId, stagePick, drawPath, gridPick, surfacePick, vtRoomDisplay, surfaceByLayer],
  );
  // Section caps: the visible room surfaces that are solid (no openings), unless a visual model covers them.
  const capsInclude = useCallback(
    (n: string) => layerKind.get(n) !== 'object' && !hidden.has(n) && surfaceByLayer.get(n) !== 'opening' && !vtRoomDisplay,
    [layerKind, hidden, surfaceByLayer, vtRoomDisplay],
  );
  const isoView = () => {
    if (bounds) setFrame({ center: bounds.center, radius: bounds.radius, key: ++frameKey.current, dir: [1, -1, 1] });
  };

  const setGrids = (fn: (g: ReceiverGrid[]) => ReceiverGrid[]) => store.update((s) => ({ ...s, receiver_grids: fn(s.receiver_grids ?? []) }));
  /** Grids tab "Pick in 3D": the clicked layer, or a plane centre on the clicked surface. */
  const applyGridPick = (name: string, e: ThreeEvent<MouseEvent>) => {
    if (!gridPick || locked) return;
    const g = sim?.receiver_grids?.find((x) => x.id === gridPick.gridId);
    if (g && gridPick.mode === 'layer') setGrids((list) => list.map((x) => (x.id === g.id ? { ...x, layer: name } : x)));
    else if (g) {
      const c = gridCenterFromClick(e, (n) => layerKind.get(n) !== 'object', g.center);
      if (!c) return;
      setGrids((list) => list.map((x) => (x.id === g.id ? { ...x, center: c } : x)));
    }
    setGridPick(null);
  };

  const onLayerClick = useLatest((name: string, e: ThreeEvent<MouseEvent>): boolean | void => {
    if (Date.now() - gizmoAt.current < 400) return;
    const ne = e.nativeEvent;
    const isObj = layerKind.get(name) === 'object';
    if (surfacePick) {
      if (isObj) return false;
      addPointFromSurface(e);
      return;
    }
    if (gridPick) {
      if (gridPick.mode === 'center' && isObj) return false;
      applyGridPick(name, e);
      return;
    }
    if (drawPath) {
      if (isObj) return false;
      addWaypointFromClick(e);
      return;
    }
    if (stagePick) {
      if (isObj) return false;
      const p = floorHit(e);
      if (!p) {
        app.toast('warning', 'Click on the floor: the conductor stands on a horizontal surface in the lower part of the room.');
        return;
      }
      if (stage) setStage(withCenter(stage, p));
      setStagePick(false);
      ghostBus.current(null);
      return;
    }
    if (placeId) {
      if (isObj) return false;
      addObject(placeId, [e.point.x, e.point.y, e.point.z], ne.shiftKey);
      return;
    }
    if (isObj && tab !== 'materials') {
      const oid = ownersByLayer.get(name)?.[e.faceIndex ?? -1];
      if (oid) {
        selectObject(oid, { toggle: ne.shiftKey || ne.ctrlKey || ne.metaKey });
        setTab('objects');
        return;
      }
    }
    if (!isObj && tab === 'objects') {
      clearSelection();
      return;
    }
    selectLayer(name, { ctrl: ne.ctrlKey || ne.metaKey, shift: ne.shiftKey });
    setTab('materials');
  });

  const onObjectClick = useLatest((id: string, e: ThreeEvent<MouseEvent>): boolean | void => {
    if (placeId || stagePick || drawPath || gridPick || surfacePick) return false;
    if (Date.now() - gizmoAt.current < 400) return;
    if (tab === 'materials') {
      const ne = e.nativeEvent;
      // The proxy triangle of this object under the cursor, if any: select its layer.
      const hit = e.intersections.find((i) => {
        const ln = i.object.userData?.layer as string | undefined;
        return !!ln && layerKind.get(ln) === 'object' && ownersByLayer.get(ln)?.[i.faceIndex ?? -1] === id;
      });
      if (hit) {
        selectLayer(hit.object.userData.layer as string, { ctrl: ne.ctrlKey || ne.metaKey, shift: ne.shiftKey });
        return;
      }
      const cid = objects.find((o) => o.id === id)?.catalog_id;
      const names = layerOrder.filter((n) => parseObjectLayer(n)?.catalogId === cid);
      if (names.length) {
        setSelPoint(null);
        setSelObject(null);
        setSelLayers(new Set(names));
        return;
      }
    }
    selectObject(id, { toggle: e.nativeEvent.shiftKey || e.nativeEvent.ctrlKey || e.nativeEvent.metaKey });
    setTab('objects');
  });

  const onLayerHover = useLatest((n: string | null) => {
    setHover(n);
    document.body.style.cursor = n ? (placeId || stagePick || drawPath || gridPick || surfacePick ? 'crosshair' : 'pointer') : '';
    if (!n && (placeId || stagePick)) ghostBus.current(null);
  });
  const onLayerMove = useLatest((name: string, e: ThreeEvent<PointerEvent>) => {
    if (layerKind.get(name) === 'object') return;
    if (stagePick) ghostBus.current(floorHit(e));
    else if (placeId) ghostBus.current([e.point.x, e.point.y, e.point.z]);
  });

  /** First intersection along the ray on a floor-like face: horizontal, in the lower 40 % of the room. */
  const floorHit = (e: ThreeEvent<MouseEvent | PointerEvent>): Vec3 | null => {
    if (!bounds) return null;
    const zMax = bounds.min[2] + 0.4 * Math.max(bounds.size[2], 0.5);
    for (const i of e.intersections) {
      const ln = i.object.userData?.layer as string | undefined;
      if (!ln || layerKind.get(ln) === 'object' || !i.face) continue;
      const n = i.face.normal.clone().transformDirection(i.object.matrixWorld);
      if (Math.abs(n.z) > 0.7 && i.point.z <= zMax) return [i.point.x, i.point.y, i.point.z];
    }
    return null;
  };

  const withCenter = (p: StagePlacement, center: Vec3): StagePlacement => {
    const next = { ...p, center: roundVec(center, 2) };
    return next.sizeAuto && bounds ? { ...next, ...stageDefaults(bounds, next.center, next.facing_deg) } : next;
  };

  const openEnsembleDialog = () => {
    if (!bounds || !sim || locked) return;
    if (lastStage.current) {
      setStage(lastStage.current);
      return;
    }
    const center: Vec3 = [Math.round(bounds.center[0] * 100) / 100, Math.round(bounds.center[1] * 100) / 100, bounds.min[2]];
    const facing = sim.receivers[0] ? facingTowards(center, sim.receivers[0].position) : 90;
    setStage({ ensembleId: null, center, facing_deg: facing, ...stageDefaults(bounds, center, facing), replace: false, sizeAuto: true });
  };

  const usesWaveSolver = !!sim && (sim.settings.mode === 'default' || (sim.settings.mode === 'advanced' && sim.settings.solvers !== 'geometrical'));

  const applyEnsemble = async (body: Parameters<typeof api.addEnsemble>[1], title: string): Promise<boolean> => {
    if (!sim) return false;
    const before = sim.sources;
    if (modify.needsPrompt()) {
      const c = await modify.choose('server');
      if (c === 'revert') return false;
      if (c === 'duplicate') {
        if (!(await store.flush())) return false;
        return modify.duplicateWith(async (copy) => {
          await api.addEnsemble(copy.id, body);
        });
      }
    }
    try {
      if (!(await store.flush())) throw new Error('the simulation could not be saved first');
      const r = await api.addEnsemble(sim.id, body);
      await store.reload();
      const beforeIds = new Set(before.map((x) => x.id));
      const added = r.simulation.sources.filter((x) => !beforeIds.has(x.id)).map((x) => x.id);
      setEnsembleNote({ title, added, replaced: body.replace_sources ? before : [], scale: r.scale, n: r.n_sources });
      lastStage.current = stage;
      setStage(null);
      setTab('sources');
      selectObject(null);
      // Frame the new players.
      const pts = r.simulation.sources.filter((x) => added.includes(x.id)).map((x) => x.position);
      if (pts.length) {
        const b = boundsOfPositions(pts.flat());
        if (b) setFrame({ center: b.center, radius: Math.max(2, b.radius * 1.6), key: ++frameKey.current });
      }
      app.toast('success', `${r.n_sources} players of “${title}” added`);
      if (r.scale < 0.8) app.toast('warning', `Orchestra compressed to ${Math.round(r.scale * 100)} % to fit the stage`);
      return true;
    } catch (e) {
      app.toast('error', `Add ensemble: ${errorMessage(e)}`);
      return false;
    }
  };

  const undoEnsemble = () => {
    const note = ensembleNote;
    if (!note || locked) return;
    const added = new Set(note.added);
    store.update((s) => {
      const keep = s.sources.filter((x) => !added.has(x.id));
      const restored = note.replaced.filter((p) => !keep.some((k) => k.id === p.id));
      return { ...s, sources: [...restored, ...keep] };
    });
    setEnsembleNote(null);
  };

  const deleteEnsembleGroup = async (title: string) => {
    if (!sim) return;
    const n = sim.sources.filter((x) => x.ensemble === title).length;
    const ok = await app.confirm({
      title: 'Delete ensemble',
      message: (
        <>
          Delete the {n} sources of <b>{title}</b>?
        </>
      ),
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    store.update((s) => ({ ...s, sources: s.sources.filter((x) => x.ensemble !== title) }));
    setSelPoint(null);
    if (ensembleNote?.title === title) setEnsembleNote(null);
  };
  const onPointClick = useLatest((p: PointMarker) => {
    if (Date.now() - gizmoAt.current < 400) return;
    setSelLayers(new Set());
    setSelObject(null);
    setSelPoint({ kind: p.kind, id: p.id });
    setSelWp(null);
    setTab(p.kind === 'source' ? 'sources' : 'receivers');
  });
  const onWaypointClick = useLatest((w: WaypointRef) => {
    if (Date.now() - gizmoAt.current < 400 || drawPath) return;
    setSelLayers(new Set());
    setSelObject(null);
    setSelPoint({ kind: w.kind, id: w.id });
    setSelWp(w);
    setTab(w.kind === 'source' ? 'sources' : 'receivers');
  });
  const onPointMove = useLatest((kind: PointKind, id: string, pos: Vec3) => movePoint(kind, id, pos));
  const onWaypointMove = useLatest((w: WaypointRef, pos: Vec3) => moveWaypoint(w, pos));
  const onObjectTransform = useLatest(transformObject);
  const onVisualError = useLatest((cid: string, msg: string) => {
    setVisualErrors((v) => (v[cid] ? v : { ...v, [cid]: msg }));
    app.toast('warning', `Visual model of “${catalogById.get(cid)?.name ?? cid}” failed to load (${msg}); its acoustic proxy is shown instead.`);
  });
  const onGizmoDrag = useLatest(() => {
    gizmoAt.current = Date.now();
  });

  const points = useMemo<PointMarker[]>(() => {
    if (!sim) return [];
    const sel = (k: PointKind, id: string) => (!!selPoint && selPoint.kind === k && selPoint.id === id) || (!!multiSel && multiSel.kind === k && multiSel.ids.includes(id));
    return [
      ...sim.sources.map((s) => ({
        kind: 'source' as const,
        id: s.id,
        label: s.name,
        position: s.position,
        dir: dirFromAzEl(s.azimuth_deg, s.elevation_deg),
        selected: sel('source', s.id),
        invalid: srcValidity.get(s.id)?.valid === false,
        groupColor: s.group?.color ?? null,
      })),
      ...sim.receivers.map((r) => ({
        kind: 'receiver' as const,
        id: r.id,
        label: r.name,
        position: r.position,
        dir: dirFromAzEl(r.azimuth_deg, r.elevation_deg),
        selected: sel('receiver', r.id),
        invalid: rcvValidity.get(r.id)?.valid === false,
        groupColor: r.group?.color ?? null,
      })),
    ];
  }, [sim, selPoint, srcValidity, rcvValidity, multiSel]);

  const pathDraws = useMemo<PathDraw[]>(() => {
    if (!sim) return [];
    const out: PathDraw[] = [];
    const add = (kind: PointKind, x: Source | Receiver) => {
      if (!x.path) return;
      out.push({
        kind,
        id: x.id,
        waypoints: x.path.waypoints.map((w) => w.position),
        curve: isSmooth(x.path) ? smoothCurve(x.path, pathValidity.get(x.id)) : null,
        invalid: pathValidity.get(x.id)?.valid === false,
        selected: !!selPoint && selPoint.kind === kind && selPoint.id === x.id,
      });
    };
    sim.sources.forEach((x) => add('source', x));
    sim.receivers.forEach((x) => add('receiver', x));
    return out;
  }, [sim, selPoint, pathValidity]);

  const gridDraws = useMemo<GridPointsDraw[]>(() => {
    const out: GridPointsDraw[] = [];
    for (const g of sim?.receiver_grids ?? []) {
      const d = gridPreviews.get(g.id)?.data;
      if (d && !gridHidden.has(g.id)) out.push({ id: g.id, points: d.points, valid: d.valid, selected: g.id === selGrid && tab === 'grids' });
    }
    return out;
  }, [sim?.receiver_grids, gridPreviews, gridHidden, selGrid, tab]);
  const focusGrid = (id: string) => {
    const d = gridPreviews.get(id)?.data;
    const b = d && d.points.length ? boundsOfPositions(d.points.flat()) : null;
    if (b) setFrame({ center: b.center, radius: Math.max(1, b.radius * 1.1), key: ++frameKey.current });
  };
  // The pick mode ends when its tab is left or editing locks.
  useEffect(() => {
    if (gridPick && (locked || tab !== 'grids')) setGridPick(null);
  }, [gridPick, locked, tab]);

  if (store.loadError) {
    return (
      <div className="app-shell">
        <TopBar simId={simId} view="editor" />
        <div className="pad">
          <ErrorBox message={`Could not load the simulation: ${store.loadError}`} />
          <a href={href.home()}>Back to the home page</a>
        </div>
      </div>
    );
  }
  if (!sim) {
    return (
      <div className="app-shell">
        <TopBar simId={simId} view="editor" />
        <div className="loading-row pad">
          <Spinner /> Loading simulation…
        </div>
      </div>
    );
  }

  const reasons = blockingReasons(sim, validation);
  const canRun = !!validation?.ok_to_run && !locked && store.saveState !== 'error';
  const unassignedCount = geometry ? geometry.layers.filter((l) => !sim.layers[l.name]?.material_id).length : 0;
  const invalidSources = sim.sources.filter((s) => srcValidity.get(s.id)?.valid === false).length;
  const invalidReceivers = sim.receivers.filter((r) => rcvValidity.get(r.id)?.valid === false).length;
  const invalidPaths = (kind: PointKind) => (kind === 'source' ? sim.sources : sim.receivers).filter((x) => x.path && pathValidity.get(x.id)?.valid === false).length;
  const tabAlert = (nPoints: number, nPaths: number) =>
    [nPoints ? `${nPoints} invalid` : '', nPaths ? `${nPaths} invalid trajector${nPaths === 1 ? 'y' : 'ies'}` : ''].filter(Boolean).join(' · ') || undefined;
  const selItem = selPoint ? (selPoint.kind === 'source' ? sim.sources : sim.receivers).find((x) => x.id === selPoint.id) : undefined;
  const drawItem = drawPath ? (drawPath.kind === 'source' ? sim.sources : sim.receivers).find((x) => x.id === drawPath.id) : undefined;
  const gizmoFree = !locked && !tlPlaying && !drawPath;
  const hoverMat = hover ? materialsById.get(sim.layers[hover]?.material_id ?? '') : undefined;
  const hoverLayer = hover ? geometry?.layers.find((l) => l.name === hover) : undefined;
  const selArea = geometry ? geometry.layers.filter((l) => selLayers.has(l.name)).reduce((a, l) => a + l.area_m2, 0) : 0;
  const selAlphas = absOn
    ? [...selLayers].map((n) => {
        const m = materialsById.get(sim.layers[n]?.material_id ?? '');
        return m ? alphaAt(m, absBand).alpha : null;
      })
    : [];
  const selObjectItem = selObject ? objects.find((o) => o.id === selObject) : undefined;
  const placeCat = placeId ? catalogById.get(placeId) : undefined;
  const gridBad = (validation?.grids ?? []).filter((g) => g.error || g.n_valid === 0).length;
  const gridAlert = gridBad ? `${gridBad} grid${gridBad === 1 ? '' : 's'} blocking the run` : undefined;
  const gridHoverInfo = (() => {
    if (!gridHover) return null;
    const d = gridPreviews.get(gridHover.gridId)?.data;
    const p = d?.points[gridHover.index];
    if (!d || !p) return null;
    const name = sim.receiver_grids?.find((g) => g.id === gridHover.gridId)?.name ?? gridHover.gridId;
    return { ...gridHover, name, p, reason: d.valid[gridHover.index] ? null : (d.reasons[String(gridHover.index)] ?? 'disabled') };
  })();

  const jobChip = job && isActiveJob(job) && (
    <button className="job-chip" onClick={() => setRunOpen(true)} title="Show run progress">
      <Spinner size={11} /> {job.status === 'queued' ? 'Queued' : `Running ${Math.round(job.progress * 100)}%`}
    </button>
  );

  return (
    <div className="app-shell">
      <TopBar simId={simId} view="editor" right={jobChip} />
      <div className={`editor ${vt.captureOpen ? 'capture-mode' : ''}`}>
        <div className={`viewport-wrap ${tMax > 0 ? 'has-timeline' : ''}`}>
          {geoError ? (
            <div className="pad">
              <ErrorBox message={`Could not load the geometry: ${geoError}`} />
            </div>
          ) : geometry && mesh ? (
            <SceneCanvas
              bounds={bounds}
              frame={frame}
              realistic={look.realistic}
              postFx={look.postFx}
              shadows={look.realistic && look.options.shadows}
              onPointerMissed={() => {
                if (Date.now() - gizmoAt.current < 400 || drawPath || surfacePick) return;
                clearSelection();
              }}
            >
              <ViewLookProvider value={look.context}>
                <ViewLookScene look={look} mesh={liveProxy.displayMesh ?? mesh} bounds={bounds} capsInclude={capsInclude} />
                <LayerMeshes
                  mesh={liveProxy.displayMesh ?? mesh}
                  shading={shading}
                  styleOf={styleOf}
                  onHover={onLayerHover}
                  onClick={onLayerClick}
                  onMove={placeId || stagePick ? onLayerMove : undefined}
                />
                {liveProxy.preview && <LayerMeshes mesh={liveProxy.preview} shading={shading} styleOf={styleOf} pickable={false} />}
                {catalog && (
                  <PlacedObjects
                    objects={objects}
                    catalog={catalogById}
                    showVisual={viewMode !== 'acoustic'}
                    selectedIds={selObjects}
                    onObjectClick={onObjectClick}
                    gizmo={!locked && selObjects.length ? { ids: selObjects, mode: gizmoMode } : null}
                    onTransform={onObjectTransform}
                    onGizmoDrag={onGizmoDrag}
                    onVisualError={onVisualError}
                    failed={visualErrors}
                    pickable={!placeId && !stagePick && !drawPath && !gridPick && !surfacePick}
                  />
                )}
                <PlaceGhost dims={placeCat?.dims ?? (stagePick ? CONDUCTOR_DIMS : null)} bus={ghostBus} />
                {stage && (
                  <EnsembleStagePreview center={stage.center} facingDeg={stage.facing_deg} players={stagePreview?.players ?? null} highlight={stagePreview?.highlight ?? null} radius={markerRadius(bounds)} />
                )}
                <PathLines
                  paths={pathDraws}
                  radius={markerRadius(bounds)}
                  selected={wpSel}
                  onWaypointClick={onWaypointClick}
                  gizmo={gizmoFree && !!wpSel}
                  onWaypointMove={onWaypointMove}
                  onGizmoDrag={onGizmoDrag}
                />
                <TimedPointMarkers
                  time={timeStore}
                  items={movables}
                  points={points}
                  radius={markerRadius(bounds)}
                  gizmo={gizmoFree && selPoint && !wpSel ? selPoint : null}
                  onPointClick={onPointClick}
                  onMove={onPointMove}
                  onGizmoDrag={onGizmoDrag}
                />
                <GridPointsPreview grids={gridDraws} radius={markerRadius(bounds) * 0.45} onHover={setGridHover} />
                <ViewToolsScene vt={vt} onGizmoDrag={onGizmoDrag} />
                {tab === 'materials' && subFocus && <SubsurfaceHighlight mesh={mesh} focus={subFocus} />}
              </ViewLookProvider>
            </SceneCanvas>
          ) : (
            <div className="loading-row pad">
              <Spinner /> Loading geometry…
            </div>
          )}

          <div className="viewport-toolbar">
            <ViewLookToolbar look={look} onIso={isoView} />
            <button className={`btn btn-sm ${absOn ? 'btn-on' : ''}`} onClick={() => setAbsOn((o) => !o)} title="Colour each layer by the absorption of its material" aria-pressed={absOn}>
              α colouring
            </button>
            {absOn && (
              <select value={absBand} onChange={(e) => setAbsBand(Number(e.target.value))} aria-label="Octave band">
                {TOOL_BANDS.map((b) => (
                  <option key={b.index} value={b.index}>
                    {toolBandLabel(b.index)} Hz
                  </option>
                ))}
              </select>
            )}
            <button
              className={`btn btn-sm ${vt.tool === 'particles' ? 'btn-on' : ''}`}
              onClick={() => vt.setTool(vt.tool === 'particles' ? null : 'particles')}
              title="Particle animation: specular particles from a source, attenuated by the surfaces they hit"
              aria-pressed={vt.tool === 'particles'}
            >
              Particles
            </button>
            <button
              className={`btn btn-sm ${vt.tool === 'density' ? 'btn-on' : ''}`}
              onClick={() => vt.setTool(vt.tool === 'density' ? null : 'density')}
              title="Reflection density: where the direct sound and reflections of order 1 to 3 land"
              aria-pressed={vt.tool === 'density'}
            >
              Reflections
            </button>
            <button className="btn btn-sm" onClick={frameSelection} title="Frame the selection, or the whole model (F)">
              Frame
            </button>
            <button
              className={`btn btn-sm ${vt.visual.panel ? 'btn-on' : ''}`}
              onClick={() => vt.visual.setPanel(!vt.visual.panel)}
              title={vt.visual.info ? `Visual model: ${vt.visual.info.filename} (display only)` : 'Upload a display-only .glb model of the room'}
              aria-pressed={vt.visual.panel}
            >
              Model{vt.visual.info ? ' ●' : ''}
            </button>
            <button className="btn btn-sm" onClick={() => vt.setCaptureOpen(true)} title="Capture mode: export the view as a PNG (S)">
              Capture
            </button>
            {objects.length > 0 && (
              <Segmented<ObjectViewMode>
                value={viewMode}
                onChange={setViewMode}
                options={[
                  { value: 'visual', label: 'Visual', title: 'Realistic models; acoustic proxies hidden (still clickable)' },
                  { value: 'acoustic', label: 'Acoustic', title: 'What the solvers see: the simplified proxies, coloured like the room layers' },
                  { value: 'both', label: 'Both', title: 'Models with their proxies drawn translucent over them' },
                ]}
              />
            )}
          </div>

          {look.cutOn && bounds && <CutSlider look={look} bounds={bounds} />}
          {absOn && (
            <div className="abs-legend">
              <div className="small">α at {toolBandLabel(absBand)} Hz</div>
              <div className="abs-bar" style={{ background: absorptionGradientCss() }}>
                <AbsorptionTicks values={selAlphas.filter((x): x is number => x !== null)} max={ABS_MAX} />
              </div>
              <div className="abs-ticks small">
                <span>0</span>
                <span>0.5</span>
                <span>{ABS_MAX}</span>
              </div>
              <div className="small">
                <span className="swatch" style={{ background: COLORS.unassigned }} /> unassigned
              </div>
              <AbsorptionMarks
                selected={[...selLayers]}
                valueOf={(n) => {
                  const m = materialsById.get(sim.layers[n]?.material_id ?? '');
                  return m ? alphaAt(m, absBand).alpha : null;
                }}
                labelOf={labelOf}
                band={absBand}
                fallback63={absBand < 0 ? new Set(Object.values(sim.layers).map((l) => materialsById.get(l?.material_id ?? '')).filter((m) => m && alphaAt(m, -1).fallback)).size : 0}
              />
            </div>
          )}
          <div className="tool-stack">
            {vt.tool === 'particles' && <ParticlesPanel vt={vt} sim={sim} labelOf={labelOf} />}
            {vt.tool === 'density' && <DensityPanel vt={vt} sim={sim} selCount={selLayers.size} />}
            {vt.visual.panel && geometry && (
              <VisualPanel vt={vt} confirm={(title, message) => app.confirm({ title, message, confirmLabel: 'Remove', danger: true })} />
            )}
          </div>
          {vt.captureOpen && <CaptureFrame vt={vt} />}

          {hover && hoverLayer && (
            <div className="hover-info small">
              <b>{labelOf(hover)}</b>
              <span className="muted"> · {fmtNum(hoverLayer.area_m2, 2)} m²</span>
              <span> · {hoverMat ? hoverMat.name : <i className="muted">unassigned</i>}</span>
              {hoverMat && (
                <span className="muted">
                  {' '}
                  · α{toolBandLabel(absBand)} = {fmtNum(alphaAt(hoverMat, absBand).alpha, 2)}
                </span>
              )}
            </div>
          )}

          {gridHoverInfo && (
            <div className="grid-hover small" style={{ left: gridHoverInfo.x + 14, top: gridHoverInfo.y + 12 }}>
              <b>{gridHoverInfo.name}</b> · point {gridHoverInfo.index + 1}
              <span className="muted"> · ({gridHoverInfo.p.map((v) => fmtNum(v, 2)).join(', ')})</span>
              <div className={gridHoverInfo.reason ? 'text-warn' : 'text-ok'}>{gridHoverInfo.reason ? `Disabled: ${gridHoverInfo.reason}` : 'Valid'}</div>
            </div>
          )}
          {gridPick && (
            <div className="selection-bar place-bar">
              <span>
                {gridPick.mode === 'layer' ? 'Click a surface: its layer becomes the layer of ' : 'Click a surface to centre '}
                <b>{sim.receiver_grids?.find((g) => g.id === gridPick.gridId)?.name ?? gridPick.gridId}</b>
                {gridPick.mode === 'center' ? ' there (on a floor: same height above it)' : ''} · Esc cancels
              </span>
              <button className="btn btn-xs" onClick={() => setGridPick(null)}>
                Cancel
              </button>
            </div>
          )}
          {selLayers.size > 0 && (
            <div className="selection-bar">
              <span>
                {selLayers.size} layer{selLayers.size === 1 ? '' : 's'} · {fmtNum(selArea, 1)} m²
              </span>
              <button className="btn btn-xs btn-primary" disabled={locked} onClick={() => openLibrary()}>
                Assign material…
              </button>
              <button
                className="btn btn-xs"
                onClick={() =>
                  setHidden((h) => {
                    const n = new Set(h);
                    selLayers.forEach((x) => n.add(x));
                    return n;
                  })
                }
              >
                Hide
              </button>
              <button className="btn btn-xs" onClick={() => setHidden(new Set(layerOrder.filter((n) => !selLayers.has(n))))}>
                Isolate
              </button>
              <button className="btn btn-xs" onClick={clearSelection} title="Clear selection (Esc)">
                Clear
              </button>
            </div>
          )}
          {selPoint && !locked && !drawPath && (
            <div className="selection-bar">
              {wpSel ? (
                <span>
                  Waypoint {wpSel.index + 1} of <b>{selItem?.name ?? wpSel.id}</b> · drag the gizmo to move it · Delete removes it · Esc deselects
                </span>
              ) : selItem?.path ? (
                <span>Drag the gizmo to move the whole trajectory · click a waypoint to edit it · Esc deselects</span>
              ) : (
                <span>Drag the gizmo to move · Delete removes · Esc deselects</span>
              )}
            </div>
          )}
          {surfacePick && (
            <div className="selection-bar place-bar">
              <span>
                Click a room surface: a {surfacePick.kind} is placed {surfacePick.distance.toFixed(2)} m along its normal (inside the room when it is closed) · Esc ends
              </span>
              <button className="btn btn-xs" onClick={() => setSurfacePick(null)}>
                Done
              </button>
            </div>
          )}
          {drawPath && (
            <div className="selection-bar place-bar">
              <span>
                Click surfaces to add waypoints to <b>{drawItem?.name ?? drawPath.id}</b> (same height above the floor, 1.4 m/s) · Esc ends
              </span>
              <button className="btn btn-xs" onClick={() => setDrawPath(null)}>
                Done
              </button>
            </div>
          )}
          {tMax > 0 && <Timeline time={timeStore} tMax={tMax} onPlayingChange={setTlPlaying} className="viewport-timeline" />}
          {selObjectItem && !placeId && (
            <div className="selection-bar">
              <b>{selObjects.length > 1 ? `${selObjects.length} objects` : selObjectItem.name}</b>
              <Segmented<GizmoMode>
                value={gizmoMode}
                onChange={setGizmoMode}
                disabled={locked}
                options={[
                  { value: 'translate', label: 'Move (T)', title: 'Translate along X / Y / Z' },
                  { value: 'rotate', label: 'Rotate Z (R)', title: 'Rotate about the vertical axis (several objects: about the centre of the selection)' },
                ]}
              />
              <button className="btn btn-xs" onClick={() => (selObjects.length > 1 ? frameSelection() : focusObject(selObjectItem.id))}>
                Focus
              </button>
              <button className="btn btn-xs" disabled={locked} onClick={duplicateSelection}>
                Duplicate
              </button>
              <button className="btn btn-xs btn-danger-ghost" disabled={locked} onClick={() => deleteObjects(selObjects)} title="Delete the selected objects (Delete)">
                Delete
              </button>
            </div>
          )}
          {stagePick && (
            <div className="selection-bar place-bar">
              <span>Click on the floor where the conductor stands · Esc cancels</span>
              <button
                className="btn btn-xs"
                onClick={() => {
                  setStagePick(false);
                  ghostBus.current(null);
                }}
              >
                Cancel
              </button>
            </div>
          )}
          {placeCat && (
            <div className="selection-bar place-bar">
              <span>
                Click a room surface to place <b>{placeCat.name}</b> · Shift-click places several · Esc stops
              </span>
              <button
                className="btn btn-xs"
                onClick={() => {
                  setPlaceId(null);
                  ghostBus.current(null);
                }}
              >
                Stop
              </button>
            </div>
          )}

          {libraryOpen && materials && (
            <MaterialLibrary
              materials={materials}
              selectedCount={selLayers.size}
              currentIds={[...selLayers].map((n) => sim.layers[n]?.material_id).filter((x): x is string => !!x)}
              onAssign={assign}
              onClose={() => setLibraryOpen(false)}
              onNew={() => setNewMatOpen(true)}
              onDelete={async (m: Material) => {
                const ok = await app.confirm({
                  title: 'Delete material',
                  message: (
                    <>
                      Delete the custom material <b>{m.name}</b>? Layers that use it will show it as missing.
                    </>
                  ),
                  confirmLabel: 'Delete',
                  danger: true,
                });
                if (!ok) return;
                try {
                  await api.deleteMaterial(m.id);
                  setPreviewId(null);
                  await app.reloadMaterials();
                } catch (e) {
                  app.toast('error', errorMessage(e));
                }
              }}
              onRename={(m: Material) => void renameMaterial(app, m)}
              onDuplicate={async (m: Material) => {
                const copy = await duplicateMaterial(app, m, isLocalOnly(m, librariesById.get(m.library)));
                if (copy) setPreviewId(copy.id);
              }}
              locked={locked}
              previewId={previewId}
              setPreviewId={setPreviewId}
            />
          )}
          {libraryOpen && !materials && (
            <aside className="drawer library pad">
              {app.materialsError ? <ErrorBox message={app.materialsError} onRetry={() => void app.reloadMaterials()} /> : <Spinner />}
            </aside>
          )}
        </div>

        <aside className="side-panel">
          <div className="sim-selector">
            <select value={simId} onChange={(e) => void switchTo(e.target.value)} aria-label="Simulation">
              {siblings.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id === simId ? sim.name : s.name}
                </option>
              ))}
            </select>
            <StatusBadge status={sim.status} stale={sim.results_stale} />
            <Menu
              items={[
                { label: 'New simulation…', onClick: () => void newSimulation(false) },
                { label: 'Duplicate…', onClick: () => void newSimulation(true) },
                { label: 'Rename…', onClick: () => void renameSimulation(), disabled: locked },
                { label: 'Delete…', onClick: () => void deleteSimulation(), danger: true, disabled: locked },
                {
                  label: 'Geometry check…',
                  onClick: () => setCheckOpen(true),
                  title: 'Open edges, overlapping faces, small gaps and short edges against λ/4 at this simulation’s transition; mark findings as known',
                },
                {
                  label: 'Duplicate geometry…',
                  onClick: () => void duplicateGeometry(),
                  title: 'Copy the room and its objects into a new geometry (with a copy of this simulation), e.g. to compare empty and furnished',
                },
              ]}
            />
          </div>
          {locked && (
            <div className="notice notice-info">
              Running: editing is locked until the run finishes.{' '}
              <button className="link" onClick={() => setRunOpen(true)}>
                Show progress
              </button>
            </div>
          )}
          {!locked && sim.status === 'done' && !sim.results_stale && (
            <div className="notice notice-info">
              Results are up to date. The first edit asks whether to modify this simulation (its results become stale), duplicate it with the change, or revert;
              trajectory edits that keep the start point do not count.{' '}
              <button className="link" onClick={() => void newSimulation(true)}>
                Duplicate instead
              </button>
            </div>
          )}
          {!locked && sim.results_stale && (
            <div className="notice notice-warn">
              Edited since the last run: results are stale. <a href={href.results(sim.id)}>Old results</a>
            </div>
          )}
          {!locked && sim.status === 'failed' && <div className="notice notice-err">The last run failed{job?.error ? `: ${job.error}` : '.'}</div>}

          <div className="tabs" role="tablist">
            <TabButton id="materials" tab={tab} setTab={setTab} label="Materials" alert={unassignedCount > 0 ? `${unassignedCount} unassigned` : undefined} />
            <TabButton id="sources" tab={tab} setTab={setTab} label={`Sources ${sim.sources.length}`} alert={tabAlert(invalidSources, invalidPaths('source'))} />
            <TabButton id="receivers" tab={tab} setTab={setTab} label={`Receivers ${sim.receivers.length}`} alert={tabAlert(invalidReceivers, invalidPaths('receiver'))} />
            <TabButton id="settings" tab={tab} setTab={setTab} label="Settings" />
            <TabButton
              id="objects"
              tab={tab}
              setTab={setTab}
              label={`Objects ${objects.length}`}
              alert={objSync.state === 'error' ? 'Last objects change rejected' : undefined}
            />
            <TabButton id="grids" tab={tab} setTab={setTab} label={`Grids ${(sim.receiver_grids ?? []).length}`} alert={gridAlert} />
          </div>

          <div className="panel-scroll">
            {store.validationError && <div className="warn-box small">Validation unavailable: {store.validationError}</div>}
            {tab === 'materials' &&
              (geometry ? (
                <MaterialsTab
                  geometry={geometry}
                  sim={sim}
                  materialsById={materialsById}
                  librariesById={librariesById}
                  selected={selLayers}
                  onRowClick={(name, e) => selectLayer(name, { ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey })}
                  onOpenLibrary={openLibrary}
                  hidden={hidden}
                  onToggleHidden={(n) =>
                    setHidden((h) => {
                      const s = new Set(h);
                      if (s.has(n)) s.delete(n);
                      else s.add(n);
                      return s;
                    })
                  }
                  onShowAll={() => setHidden(new Set())}
                  onScattering={setScattering}
                  onSelectMany={(names) => {
                    setSelPoint(null);
                    setSelObject(null);
                    setSelLayers(new Set(names));
                  }}
                  labelOf={labelOf}
                  kindOf={(n) => layerKind.get(n) ?? 'room'}
                  instancesOf={(n) => instancesOf(n, objects)}
                  validation={validation}
                  locked={locked}
                  colorOf={(n) => styleOf(n).color}
                  hovered={hover}
                  onHoverRow={setHover}
                  mesh={mesh}
                  onScatteringMany={setScatteringMany}
                  onSubsurfaceFocus={setSubFocus}
                  onFrame={(center, radius) => setFrame({ center, radius, key: ++frameKey.current })}
                />
              ) : (
                <Spinner />
              ))}
            {tab === 'sources' && ensembleNote && (
              <div className={`notice ${ensembleNote.scale < 0.8 ? 'notice-warn' : 'notice-info'} ensemble-note`}>
                <div>
                  Added {ensembleNote.n} players of <b>{ensembleNote.title}</b>
                  {ensembleNote.scale < 0.8 ? (
                    <>
                      {' '}
                      — orchestra compressed to <b>{Math.round(ensembleNote.scale * 100)} %</b> to fit the stage.
                    </>
                  ) : (
                    ` (layout at ${Math.round(ensembleNote.scale * 100)} %).`
                  )}
                </div>
                {usesWaveSolver && <div className="small">{sim.sources.length} sources: {sim.sources.length} wave-solver runs.</div>}
                <div className="row gap">
                  <button className="link" disabled={locked} onClick={undoEnsemble}>
                    Undo
                  </button>
                  <button className="link" onClick={() => setEnsembleNote(null)}>
                    Dismiss
                  </button>
                </div>
              </div>
            )}
            {tab === 'sources' && (
              <PointsTab
                kind="source"
                items={sim.sources}
                others={sim.receivers}
                layers={geometry ? geometry.layers.map((l) => l.name) : undefined}
                directivities={app.directivities}
                validity={srcValidity}
                selectedId={selPoint?.kind === 'source' ? selPoint.id : null}
                onSelect={(id) => {
                  setSelLayers(new Set());
                  setSelObject(null);
                  setSelPoint(id ? { kind: 'source', id } : null);
                }}
                onChange={setSources}
                bounds={bounds}
                locked={locked}
                onAddEnsemble={openEnsembleDialog}
                onDeleteGroup={(t) => void deleteEnsembleGroup(t)}
                pathValidity={pathValidity}
                selectedWaypoint={wpSel?.kind === 'source' ? wpSel.index : null}
                onSelectWaypoint={(id, index) => {
                  setSelPoint({ kind: 'source', id });
                  setSelWp(index === null ? null : { kind: 'source', id, index });
                }}
                drawingId={drawPath?.kind === 'source' ? drawPath.id : null}
                onDraw={(id) => startDrawing('source', id)}
                onMultiSelect={(ids) => setMultiSel(ids.length > 1 ? { kind: 'source', ids } : null)}
                surfacePick={surfacePick?.kind === 'source' ? surfacePick.distance : null}
                onSurfacePick={(d) => startSurfacePick('source', d)}
                saveError={store.saveError}
              />
            )}
            {tab === 'receivers' && (
              <PointsTab
                kind="receiver"
                items={sim.receivers}
                others={sim.sources}
                directivities={app.directivities}
                validity={rcvValidity}
                selectedId={selPoint?.kind === 'receiver' ? selPoint.id : null}
                onSelect={(id) => {
                  setSelLayers(new Set());
                  setSelObject(null);
                  setSelPoint(id ? { kind: 'receiver', id } : null);
                }}
                onChange={setReceivers}
                bounds={bounds}
                locked={locked}
                pathValidity={pathValidity}
                selectedWaypoint={wpSel?.kind === 'receiver' ? wpSel.index : null}
                onSelectWaypoint={(id, index) => {
                  setSelPoint({ kind: 'receiver', id });
                  setSelWp(index === null ? null : { kind: 'receiver', id, index });
                }}
                drawingId={drawPath?.kind === 'receiver' ? drawPath.id : null}
                onDraw={(id) => startDrawing('receiver', id)}
                onMultiSelect={(ids) => setMultiSel(ids.length > 1 ? { kind: 'receiver', ids } : null)}
                surfacePick={surfacePick?.kind === 'receiver' ? surfacePick.distance : null}
                onSurfacePick={(d) => startSurfacePick('receiver', d)}
              />
            )}
            {tab === 'settings' && (
              <SettingsTab
                settings={sim.settings}
                validation={validation}
                locked={locked}
                onChange={(patch) => store.update((s) => ({ ...s, settings: { ...s.settings, ...patch } }))}
              />
            )}
            {tab === 'objects' && (
              <ObjectsTab
                catalog={catalog}
                catalogError={catalogError}
                catalogById={catalogById}
                objects={objects}
                selectedIds={selObjects}
                onSelect={selectObject}
                onAdd={(cid) => addObject(cid)}
                onPatch={patchObject}
                onDuplicate={duplicateObject}
                onDelete={(id) => deleteObjects([id])}
                onFocus={focusObject}
                canUndo={objSync.canUndo}
                canRedo={objSync.canRedo}
                onUndo={() => stepObjects(-1)}
                onRedo={() => stepObjects(1)}
                placeId={placeId}
                setPlaceId={(id) => {
                  setPlaceId(id);
                  if (!id) ghostBus.current(null);
                }}
                syncState={objSync.state}
                syncError={objSync.error}
                locked={locked}
                materialsById={materialsById}
                failed={visualErrors}
              />
            )}
            {tab === 'grids' && (
              <GridsTab
                sim={sim}
                layers={layerOrder}
                labelOf={labelOf}
                layerBounds={layerBounds}
                bounds={bounds}
                validation={validation}
                previews={gridPreviews}
                selectedId={selGrid}
                onSelect={setSelGrid}
                onChange={(grids) => store.update((s) => ({ ...s, receiver_grids: grids }))}
                onSettings={(patch) => store.update((s) => ({ ...s, settings: { ...s.settings, ...patch } }))}
                pick={gridPick}
                onPick={setGridPick}
                hidden={gridHidden}
                onToggleHidden={(id) =>
                  setGridHidden((h) => {
                    const n = new Set(h);
                    if (n.has(id)) n.delete(id);
                    else n.add(id);
                    return n;
                  })
                }
                onFocus={focusGrid}
                locked={locked}
              />
            )}
            {!geometry && !geoError && tab === 'materials' && <EmptyState title="Loading layers…" />}
          </div>

          <div className="panel-footer">
            <SaveIndicator state={store.saveState} error={store.saveError} onRetry={() => void store.flush()} />
            <div className="grow" />
            {sim.last_run && (
              <a className="btn btn-sm" href={href.results(sim.id)}>
                Results
              </a>
            )}
            {locked ? (
              <button className="btn btn-primary" onClick={() => setRunOpen(true)}>
                Progress…
              </button>
            ) : (
              <span className="run-wrap" title={canRun ? 'Run the simulation' : reasons.length ? `Cannot run:\n• ${reasons.join('\n• ')}` : 'Saving…'}>
                <button className="btn btn-primary btn-run" disabled={!canRun} onClick={() => void startRun()}>
                  Run
                </button>
              </span>
            )}
          </div>
          {!canRun && !locked && reasons.length > 0 && (
            <ul className="run-reasons small">
              {reasons.slice(0, 4).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
              {reasons.length > 4 && <li className="muted">… and {reasons.length - 4} more</li>}
            </ul>
          )}
          {!locked && (validation?.warnings?.length ?? 0) > 0 && (
            <div className="run-warnings small" role="note">
              <div className="run-warnings-title">Warning{validation!.warnings!.length === 1 ? '' : 's'} (the run is not blocked)</div>
              <ul>
                {validation!.warnings!.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </aside>
        {vt.captureOpen && <CapturePanel vt={vt} simName={sim.name} />}
      </div>

      {stage && !stagePick && bounds && (
        <EnsembleDialog
          simId={sim.id}
          placement={stage}
          setPlacement={setStage}
          bounds={bounds}
          receivers={sim.receivers}
          nSources={sim.sources.length}
          usesWave={usesWaveSolver}
          onPickFloor={() => {
            setStagePick(true);
            setTab('sources');
          }}
          onCancel={() => {
            lastStage.current = stage;
            setStage(null);
          }}
          onApply={(body, ens) => applyEnsemble(body, ens.title)}
          onPreview={setStagePreview}
          onFrame={(center, radius) => setFrame({ center, radius, key: ++frameKey.current })}
        />
      )}
      {newMatOpen && (
        <NewMaterialDialog
          categories={categories}
          onClose={() => setNewMatOpen(false)}
          onCreated={async (m) => {
            setNewMatOpen(false);
            await app.reloadMaterials();
            setPreviewId(m.id);
            app.toast('success', `Material “${m.name}” created`);
          }}
        />
      )}
      {modify.dialog}
      {confirmRun && !locked && <RunConfirmDialog sim={sim} validation={confirmRun} onRun={launchRun} onClose={() => setConfirmRun(null)} />}
      {checkOpen && geometry && mesh && (
        <GeometryCheckDialog
          geometry={geometry}
          mesh={mesh}
          simulationId={sim.id}
          onGeometry={(g) => {
            cacheGeometry(g);
            setGeometry(g);
          }}
          onClose={() => setCheckOpen(false)}
        />
      )}
      {runOpen && job && <RunDialog job={job} simName={sim.name} onClose={() => setRunOpen(false)} onCancel={() => void jobState.cancel()} pollError={jobState.pollError} />}
    </div>
  );
}

function TabButton({ id, tab, setTab, label, alert }: { id: Tab; tab: Tab; setTab: (t: Tab) => void; label: string; alert?: string }) {
  return (
    <button role="tab" aria-selected={tab === id} className={`tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)} title={alert}>
      {label}
      {alert && <span className="tab-alert" />}
    </button>
  );
}

function SaveIndicator({ state, error, onRetry }: { state: SaveState; error: string | null; onRetry: () => void }) {
  if (state === 'error') {
    return (
      <span className="save-state error" title={error ?? ''}>
        Save failed{error ? `: ${error}` : ''}{' '}
        <button className="link" onClick={onRetry}>
          Retry
        </button>
      </span>
    );
  }
  return (
    <span className={`save-state ${state}`}>
      {state === 'saved' ? 'All changes saved' : state === 'saving' ? 'Saving…' : 'Unsaved changes…'}
    </span>
  );
}
