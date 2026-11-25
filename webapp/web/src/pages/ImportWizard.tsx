import { useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  errorMessage,
  type CheckExclusions,
  type Finding,
  type FindingKind,
  type Geometry,
  type IfcInventory,
  type MeshData,
  type Unit,
  type UpAxis,
  type Vec3,
} from '../api';
import { useApp } from '../AppContext';
import { ErrorBox, EyeButton, Modal, ProgressBar, Segmented, Spinner } from '../components/ui';
import { forgetGeometry } from '../lib/cache';
import { layerColor } from '../lib/colors';
import { fmtNum } from '../lib/format';
import { boundsOf, distance } from '../lib/geom';
import { href, navigate } from '../lib/router';
import { locateGeometry } from '../lib/tree';
import { TriangleHighlights } from '../three/FindingHighlights';
import { LayerMeshes, SceneCanvas, SegmentHighlights, type FrameRequest, type Shading } from '../three/Scene';

export interface ImportPreset {
  projectId?: string;
  spaceId?: string;
}

const NEW = '__new__';
const MODEL_EXT = /\.(obj|par|dxf|3dm|ifc|skp)$/i;

const KIND_ORDER: FindingKind[] = ['open_edge', 'nonmanifold_edge', 'overlapping_faces', 'small_gap', 'short_edge', 'degenerate_triangle'];
const KIND_INFO: Record<FindingKind, { label: string; countKey: keyof Geometry['check']['counts']; help: string }> = {
  open_edge: {
    label: 'Open edges',
    countKey: 'open_edges',
    help: 'Edges used by a single triangle: the surface has a hole or a gap here. The wave solver needs a closed volume.',
  },
  nonmanifold_edge: {
    label: 'Non-manifold edges',
    countKey: 'nonmanifold_edges',
    help: 'Edges shared by more than two triangles, e.g. an internal panel touching the shell.',
  },
  overlapping_faces: {
    label: 'Overlapping faces',
    countKey: 'overlapping_faces',
    help: 'Coplanar faces covering each other: a wall exported twice, a rug drawn on the floor, a solid resting on a surface. Back to back (info) is harmless; faces seen from the same air make the material there arbitrary, or count the area twice.',
  },
  small_gap: {
    label: 'Small gaps',
    countKey: 'small_gaps',
    help: 'Surfaces closer than the detail threshold: a slit at an open edge, or a thin air layer between parallel faces. Close them unless the opening matters. A slit leading out of the room that the wave grid leaves open is a warning.',
  },
  short_edge: {
    label: 'Short edges',
    countKey: 'short_edges',
    help: 'Edges of the model (not diagonals inside flat faces) shorter than the detail threshold: finer than the wave band needs. Harmless for its fixed grid, but a sign of unneeded detail.',
  },
  degenerate_triangle: {
    label: 'Degenerate triangles',
    countKey: 'degenerate_triangles',
    help: 'Triangles with (near) zero area. They carry no acoustic surface.',
  },
};

export function ImportWizard({ preset, onClose }: { preset: ImportPreset; onClose: () => void }) {
  const app = useApp();
  const tree = app.tree ?? [];
  const [step, setStep] = useState<1 | 2>(1);
  const [file, setFile] = useState<File | null>(null);
  const [unit, setUnit] = useState<Unit>('m');
  const [upAxis, setUpAxis] = useState<UpAxis>('z');
  const [projectId, setProjectId] = useState<string>(() => preset.projectId ?? tree[0]?.id ?? NEW);
  const [newProjectName, setNewProjectName] = useState('New project');
  const [spaceId, setSpaceId] = useState<string>(() => {
    if (preset.spaceId) return preset.spaceId;
    const p = tree.find((x) => x.id === (preset.projectId ?? tree[0]?.id));
    return p?.spaces?.[0]?.id ?? NEW;
  });
  const [newSpaceName, setNewSpaceName] = useState('');
  const [geomName, setGeomName] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [mesh, setMesh] = useState<MeshData | null>(null);
  const [ifc, setIfc] = useState<IfcInventory | null>(null);
  const [ifcSpace, setIfcSpace] = useState('');
  const [ifcExclude, setIfcExclude] = useState<Set<string>>(new Set());
  const isIfc = !!file && /\.ifc$/i.test(file.name);

  const project = tree.find((p) => p.id === projectId) ?? null;
  const spaces = project?.spaces ?? [];

  useEffect(() => {
    if (projectId === NEW) {
      setSpaceId(NEW);
    } else if (spaceId !== NEW && !spaces.some((s) => s.id === spaceId)) {
      setSpaceId(spaces[0]?.id ?? NEW);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const chooseFile = (f: File | null) => {
    setError(null);
    if (!f) return;
    if (!MODEL_EXT.test(f.name)) {
      setError(`“${f.name}” is not supported. Use a Wavefront .obj, an Odeon .Par, a .dxf, a Rhino .3dm, an .ifc or a SketchUp .skp model.`);
      return;
    }
    setFile(f);
    setIfc(null);
    setIfcSpace('');
    if (/\.ifc$/i.test(f.name)) {
      setBusy(true);
      api
        .ifcInventory(f)
        .then((inv) => {
          setIfc(inv);
          setIfcExclude(new Set(inv.default_exclude.filter((c) => c in inv.classes)));
        })
        .catch((e) => setError(errorMessage(e)))
        .finally(() => setBusy(false));
    }
    const base = f.name.replace(MODEL_EXT, '');
    if (!geomName) setGeomName(base);
    if (!newSpaceName) setNewSpaceName(base);
  };

  const canUpload =
    !!file &&
    !!geomName.trim() &&
    (projectId !== NEW || !!newProjectName.trim()) &&
    (spaceId !== NEW || !!newSpaceName.trim()) &&
    !busy;

  const upload = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setProgress(0);
    try {
      let pid = projectId;
      if (pid === NEW) {
        const p = await api.createProject(newProjectName.trim(), '');
        pid = p.id;
        setProjectId(pid);
      }
      let sid = spaceId;
      if (sid === NEW) {
        const s = await api.createSpace(pid, newSpaceName.trim(), '');
        sid = s.id;
        setSpaceId(sid);
      }
      await app.refreshTree();
      const g = await api.importGeometry(
        sid,
        {
          file,
          name: geomName.trim(),
          up_axis: upAxis,
          unit,
          ...(isIfc ? { ifc_space: ifcSpace, ifc_exclude: [...ifcExclude] } : {}),
        },
        (f) => setProgress(f),
      );
      setProgress(null);
      const m = await api.getMesh(g.id);
      setGeometry(g);
      setMesh(m);
      setStep(2);
      void app.refreshTree();
    } catch (e) {
      setError(errorMessage(e));
      void app.refreshTree();
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const discardAndBack = async () => {
    if (!geometry) return;
    const ok = await app.confirm({
      title: 'Re-import',
      message: (
        <>
          Delete the imported geometry <b>{geometry.name}</b> and go back to the upload step (to change the unit or the up axis)?
        </>
      ),
      confirmLabel: 'Delete and go back',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteGeometry(geometry.id);
      forgetGeometry(geometry.id);
    } catch (e) {
      app.toast('error', errorMessage(e));
      return;
    }
    setGeometry(null);
    setMesh(null);
    setStep(1);
    void app.refreshTree();
  };

  const openEditor = async () => {
    if (!geometry) return;
    setBusy(true);
    try {
      const t = await app.refreshTree();
      const existing = locateGeometry(t, geometry.id)?.geometry.simulations ?? [];
      const simId = existing[0]?.id ?? (await api.createSimulation(geometry.id, 'Simulation 1', null)).id;
      await app.refreshTree();
      onClose();
      navigate(href.editor(simId));
    } catch (e) {
      setError(errorMessage(e));
      setBusy(false);
    }
  };

  return (
    <Modal
      title={
        <span className="wizard-title">
          Import model
          <span className="stepper">
            <span className={step === 1 ? 'active' : 'done'}>1 Upload</span>
            <span className="sep">—</span>
            <span className={step === 2 ? 'active' : ''}>2 Review</span>
          </span>
        </span>
      }
      onClose={onClose}
      width={step === 1 ? 640 : '94vw'}
      className={step === 2 ? 'modal-tall' : ''}
      closeOnBackdrop={false}
    >
      {step === 1 && (
        <>
          <div className="modal-body">
            <Dropzone file={file} onFile={chooseFile} disabled={busy} />
            {isIfc && ifc && <IfcOptions inv={ifc} space={ifcSpace} setSpace={setIfcSpace} exclude={ifcExclude} setExclude={setIfcExclude} />}
            <div className="form-grid">
              <label>Unit of the file</label>
              <div>
                <Segmented<Unit>
                  value={unit}
                  onChange={setUnit}
                  options={[
                    { value: 'm', label: 'm' },
                    { value: 'cm', label: 'cm' },
                    { value: 'mm', label: 'mm' },
                  ]}
                />
              </div>
              <label>Up axis</label>
              <div className="row gap">
                <Segmented<UpAxis>
                  value={upAxis}
                  onChange={setUpAxis}
                  options={[
                    { value: 'z', label: 'Z up' },
                    { value: 'y', label: 'Y up' },
                  ]}
                />
                <span className="muted small">Many OBJ exporters write Y up; the engine works in Z up.</span>
              </div>
              <label>Project</label>
              <div className="row gap">
                <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
                  {tree.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                  <option value={NEW}>+ New project…</option>
                </select>
                {projectId === NEW && (
                  <input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} placeholder="Project name" />
                )}
              </div>
              <label>Space</label>
              <div className="row gap">
                <select value={spaceId} onChange={(e) => setSpaceId(e.target.value)}>
                  {spaces.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                  <option value={NEW}>+ New space…</option>
                </select>
                {spaceId === NEW && <input value={newSpaceName} onChange={(e) => setNewSpaceName(e.target.value)} placeholder="Space name" />}
              </div>
              <label>Geometry name</label>
              <input value={geomName} onChange={(e) => setGeomName(e.target.value)} placeholder="e.g. Initial design" />
            </div>
            {progress !== null && (
              <div className="upload-progress">
                <span className="small muted">{progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : 'Checking geometry…'}</span>
                <ProgressBar value={progress} />
              </div>
            )}
            {error && <ErrorBox message={error} />}
          </div>
          <div className="modal-footer">
            <button className="btn" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" disabled={!canUpload} onClick={upload}>
              {busy ? <Spinner /> : null} Upload and check
            </button>
          </div>
        </>
      )}
      {step === 2 && geometry && mesh && (
        <>
          <ReviewStep geometry={geometry} mesh={mesh} onGeometry={setGeometry} />
          {error && (
            <div className="pad">
              <ErrorBox message={error} />
            </div>
          )}
          <div className="modal-footer">
            <button className="btn" onClick={discardAndBack} disabled={busy}>
              Back (re-import)
            </button>
            <div className="grow" />
            <button className="btn" onClick={onClose} disabled={busy}>
              Close
            </button>
            <button className="btn btn-primary" onClick={openEditor} disabled={busy}>
              {busy ? <Spinner /> : null} Open in editor
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}

function IfcOptions({
  inv,
  space,
  setSpace,
  exclude,
  setExclude,
}: {
  inv: IfcInventory;
  space: string;
  setSpace: (s: string) => void;
  exclude: Set<string>;
  setExclude: (s: Set<string>) => void;
}) {
  const classes = Object.entries(inv.classes).sort((a, b) => b[1] - a[1]);
  return (
    <div className="ifc-options">
      <div className="form-grid">
        <label>Acoustic volume</label>
        <div>
          <select value={space} onChange={(e) => setSpace(e.target.value)}>
            <option value="">Building elements (one layer per class and material)</option>
            {inv.spaces.map((s) => (
              <option key={s.id} value={s.id}>
                Space: {s.name}
              </option>
            ))}
          </select>
          <div className="muted small">
            {space
              ? 'The closed shell of this IfcSpace becomes the room; each face takes the layer of the nearest building element below.'
              : `${inv.schema} model with ${inv.spaces.length} space(s). Choosing a space gives a closed model by construction.`}
          </div>
        </div>
        <label>Elements used</label>
        <div className="ifc-classes">
          {classes.map(([c, n]) => (
            <label key={c} className="check">
              <input
                type="checkbox"
                checked={!exclude.has(c)}
                onChange={(e) => {
                  const next = new Set(exclude);
                  if (e.target.checked) next.delete(c);
                  else next.add(c);
                  setExclude(next);
                }}
              />{' '}
              {c} <span className="muted">×{n}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}

function Dropzone({ file, onFile, disabled }: { file: File | null; onFile: (f: File | null) => void; disabled: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      className={`dropzone ${over ? 'over' : ''} ${file ? 'has-file' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) onFile(e.dataTransfer.files?.[0] ?? null);
      }}
      onClick={() => !disabled && input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') input.current?.click();
      }}
    >
      <input ref={input} type="file" accept=".obj,.par,.Par,.dxf,.DXF,.3dm,.3DM,.ifc,.IFC,.skp,.SKP" hidden onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
      {file ? (
        <>
          <div className="dz-file">{file.name}</div>
          <div className="muted small">{(file.size / 1024).toFixed(1)} KiB · click or drop to replace</div>
        </>
      ) : (
        <>
          <div className="dz-title">Drop an .obj, Odeon .Par, .dxf, Rhino .3dm, .ifc or SketchUp .skp file here</div>
          <div className="muted small">
            or click to browse. Layers come from the OBJ “usemtl” groups, the Odeon surface names, the DXF / Rhino layers, or the IFC class and material. Odeon and IFC
            models are in metres, Z up.
          </div>
        </>
      )}
    </div>
  );
}

function fmtLen(m: number | null | undefined): string {
  if (m === null || m === undefined) return '–';
  return m < 0.1 ? `${fmtNum(m * 1000, 0)} mm` : `${fmtNum(m * 100, 1)} cm`;
}

function findingFacts(f: Finding): string {
  if (f.kind === 'overlapping_faces' && f.area_m2 !== undefined) {
    const c = f.centroid ? ` · centre (${f.centroid.map((v) => v.toFixed(2)).join(', ')})` : '';
    return `${fmtNum(f.area_m2, 3)} m² overlap · ${f.n_triangles ?? 0} triangles${c}`;
  }
  const seg = `${f.segment.map((p) => `(${p.map((v) => v.toFixed(3)).join(', ')})`).join(' → ')}`;
  if (f.kind === 'small_gap' && f.gap_m !== undefined) {
    const w = f.gap_max_m && f.gap_max_m > f.gap_m * 1.1 ? `${fmtLen(f.gap_m)} to ${fmtLen(f.gap_max_m)}` : fmtLen(f.gap_m);
    return `gap ${w}${f.gap_type === 'layer' && f.area_m2 ? ` over ${fmtNum(f.area_m2, 2)} m²` : ''} · ${seg}`;
  }
  return `${seg} · ${fmtNum(distance(f.segment[0], f.segment[1]), 4)} m`;
}

export function ReviewStep({
  geometry: initial,
  mesh,
  simulationId,
  autoCheck = false,
  onGeometry,
}: {
  geometry: Geometry;
  mesh: MeshData;
  /** The simulation whose transition the λ/4 rule uses (editor). */
  simulationId?: string;
  /** Re-check with that simulation's transition when opened. */
  autoCheck?: boolean;
  onGeometry?: (g: Geometry) => void;
}) {
  const [geometry, setGeo] = useState(initial);
  useEffect(() => setGeo(initial), [initial]);
  const bounds = useMemo(() => boundsOf(geometry.bbox.min, geometry.bbox.max), [geometry.bbox]);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [shading, setShading] = useState<Shading>('ghosted');
  const [activeKind, setActiveKind] = useState<FindingKind | null>(null);
  const [index, setIndex] = useState(0);
  const [frame, setFrame] = useState<FrameRequest | null>(null);
  const frameKey = useRef(0);
  const [busy, setBusy] = useState<'check' | 'exclude' | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [fcText, setFcText] = useState(() => (initial.check.detail?.transition_hz ? String(initial.check.detail.transition_hz) : ''));
  const [undo, setUndo] = useState<{ label: string; prev: CheckExclusions } | null>(null);
  const keepId = useRef<string | null>(null);

  const check = geometry.check;
  const exclusions: CheckExclusions = geometry.check_exclusions ?? { findings: [], layers: [] };
  const detail = check.detail;

  const byKind = useMemo(() => {
    const m = new Map<FindingKind, Finding[]>();
    for (const f of check.findings) {
      const list = m.get(f.kind) ?? [];
      list.push(f);
      m.set(f.kind, list);
    }
    return m;
  }, [check]);

  const list = activeKind ? (byKind.get(activeKind) ?? []) : [];
  const current = list[index] ?? null;

  // after a re-check or an exclusion the list is re-ordered: stay on the same finding
  useEffect(() => {
    if (!keepId.current) return;
    const j = list.findIndex((f) => f.id === keepId.current);
    keepId.current = null;
    if (j >= 0) setIndex(j);
    else setIndex((i) => Math.min(i, Math.max(list.length - 1, 0)));
  }, [list]);

  const apply = (g: Geometry) => {
    keepId.current = current?.id ?? null;
    setGeo(g);
    onGeometry?.(g);
  };

  const recheck = async (body: { transition_hz?: number | null; simulation_id?: string | null }) => {
    setBusy('check');
    setCheckError(null);
    try {
      const g = await api.checkGeometry(geometry.id, body);
      apply(g);
      setFcText(g.check.detail?.transition_hz ? String(g.check.detail.transition_hz) : '');
    } catch (e) {
      setCheckError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  useEffect(() => {
    if (autoCheck && simulationId) void recheck({ simulation_id: simulationId });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const saveExclusions = async (next: CheckExclusions, label: string | null) => {
    const prev = exclusions;
    setBusy('exclude');
    setCheckError(null);
    try {
      const g = await api.putExclusions(geometry.id, next);
      apply(g);
      setUndo(label ? { label, prev } : null);
    } catch (e) {
      setCheckError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const toggleFinding = (f: Finding) => {
    if (!f.id) return;
    const has = exclusions.findings.includes(f.id);
    void saveExclusions(
      { ...exclusions, findings: has ? exclusions.findings.filter((x) => x !== f.id) : [...exclusions.findings, f.id] },
      has ? 'Finding included again' : 'Finding excluded',
    );
  };

  const toggleLayer = (name: string) => {
    const has = exclusions.layers.includes(name);
    void saveExclusions(
      { ...exclusions, layers: has ? exclusions.layers.filter((x) => x !== name) : [...exclusions.layers, name] },
      has ? `Findings on “${name}” included again` : `Findings on “${name}” excluded`,
    );
  };

  const frameFinding = (f: Finding | null) => {
    if (!f) return;
    const [a, b] = f.segment;
    const center: Vec3 = f.centroid ?? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    const radius = Math.max(distance(a, b) * 1.5, bounds.radius * 0.08, 0.25);
    setFrame({ center, radius, key: ++frameKey.current });
  };

  const go = (i: number) => {
    if (!list.length) return;
    const j = (i + list.length) % list.length;
    setIndex(j);
    frameFinding(list[j]);
  };

  const layerIndex = useMemo(() => new Map(mesh.layers.map((l, i) => [l.name, i])), [mesh]);
  const size = bounds.size;
  const maxDim = Math.max(...size);
  const solvers = check.solvers;
  const tris = activeKind === 'overlapping_faces';
  const active = useMemo(() => list.filter((f) => !f.excluded), [list]);
  const segAll = useMemo(() => (tris ? [] : active.map((f) => f.segment)), [active, tris]);
  const triAll = useMemo(() => (tris ? active.flatMap((f) => f.triangles ?? []) : []), [active, tris]);
  const triCur = useMemo(() => (tris ? (current?.triangles ?? []) : []), [current, tris]);
  const fcValue = Number(fcText);
  const fcValid = fcText.trim() === '' || (Number.isFinite(fcValue) && fcValue >= 20 && fcValue <= 20000);
  const blocking = check.blocking ?? (check.watertight ? [] : null);

  return (
    <div className="review">
      <div className="review-view">
        <SceneCanvas bounds={bounds} frame={frame}>
          <LayerMeshes
            mesh={mesh}
            shading={shading}
            pickable={false}
            styleOf={(name, i) => ({ color: layerColor(i), hidden: hidden.has(name) })}
          />
          <SegmentHighlights all={segAll} current={tris ? null : (current?.segment ?? null)} />
          <TriangleHighlights all={triAll} current={triCur} />
        </SceneCanvas>
        <div className="viewport-toolbar">
          <Segmented<Shading>
            value={shading}
            onChange={setShading}
            options={[
              { value: 'shaded', label: 'Shaded' },
              { value: 'wireframe', label: 'Wireframe' },
              { value: 'ghosted', label: 'Ghosted' },
            ]}
          />
          <button className="btn btn-sm" onClick={() => setFrame({ center: bounds.center, radius: bounds.radius, key: ++frameKey.current })}>
            Frame all
          </button>
        </div>
      </div>
      <div className="review-side">
        <section>
          <h3>Model</h3>
          <dl className="kv">
            <dt>File</dt>
            <dd className="ellipsis" title={geometry.source_filename}>
              {geometry.source_filename}
            </dd>
            <dt>Imported as</dt>
            <dd>
              unit {geometry.unit_in}, {geometry.up_axis_in.toUpperCase()} up
            </dd>
            <dt>Size</dt>
            <dd>
              {fmtNum(size[0])} × {fmtNum(size[1])} × {fmtNum(size[2])} m
            </dd>
            <dt>Volume</dt>
            <dd>{geometry.volume_m3 !== null ? `${fmtNum(geometry.volume_m3, 1)} m³` : '– (not closed)'}</dd>
            <dt>Surface</dt>
            <dd>{fmtNum(geometry.surface_m2, 1)} m²</dd>
            <dt>Triangles</dt>
            <dd>{geometry.n_triangles.toLocaleString()}</dd>
          </dl>
          {(maxDim > 300 || maxDim < 0.3) && (
            <div className="warn-box">
              The model is {fmtNum(maxDim, 2)} m across. If that is not right, go back and change the unit.
            </div>
          )}
          {geometry.up_axis_in === 'z' && size[2] > 1.5 * Math.min(size[0], size[1]) && size[2] > 1.5 && (
            <div className="warn-box">
              The model is taller ({fmtNum(size[2], 2)} m along Z) than it is wide along {size[0] < size[1] ? 'X' : 'Y'} ({fmtNum(Math.min(size[0], size[1]), 2)} m). If it
              lies on its side, the file is probably Y up: go back and re-import with “Y up”. Keep Z up if your source and receiver coordinates are given in the
              file's own frame (BRAS scenes, for instance).
            </div>
          )}
          <div className="solver-badges">
            <span className={`chip ${solvers.geometrical ? 'chip-ok' : 'chip-err'}`}>Geometrical solver: {solvers.geometrical ? 'on' : 'off'}</span>
            <span className={`chip ${solvers.wave ? 'chip-ok' : 'chip-warn'}`}>
              Wave solver: {solvers.wave ? 'on' : `off${check.watertight ? '' : ' (not watertight)'}`}
            </span>
          </div>
        </section>
        <section>
          <h3>Geometry check</h3>
          <div className="check-detail small">
            {!detail ? (
              <span>Checked before the λ/4 rule existed: re-check to judge short edges and gaps against the transition, and to enable exclusions.</span>
            ) : detail.rule === 'lambda/4' ? (
              <span>
                Short edges and gaps judged against <b>λ/4 = {fmtLen(detail.threshold_m)}</b> at {fmtNum(detail.transition_hz ?? 0, 0)} Hz (wave grid dx ={' '}
                {fmtLen(detail.wave_dx_m)}).
              </span>
            ) : (
              <span>
                Short edges and gaps judged against a <b>fixed {fmtLen(detail.threshold_m)}</b>: give a transition frequency for the λ/4 rule.
              </span>
            )}
            {detail?.source && <div className="muted">{detail.source}</div>}
          </div>
          <div className="row gap check-recheck">
            <label className="small" htmlFor="check-fc">
              Transition
            </label>
            <input
              id="check-fc"
              type="number"
              min={20}
              max={20000}
              step={1}
              value={fcText}
              placeholder="none"
              onChange={(e) => setFcText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && fcValid && !busy) void recheck(fcText.trim() ? { transition_hz: fcValue } : {});
              }}
            />
            <span className="small muted">Hz</span>
            <button
              className="btn btn-sm"
              disabled={!!busy || !fcValid}
              title={fcText.trim() ? 'Re-check against λ/4 at this frequency' : 'Re-check with the fixed 2 cm threshold'}
              onClick={() => void recheck(fcText.trim() ? { transition_hz: fcValue } : {})}
            >
              Re-check
            </button>
            {simulationId && (
              <button className="btn btn-sm" disabled={!!busy} title="Use the transition of this simulation (fixed in Advanced settings, or the automatic one)" onClick={() => void recheck({ simulation_id: simulationId })}>
                Use simulation
              </button>
            )}
            {busy && <Spinner />}
          </div>
          {checkError && <ErrorBox message={checkError} />}
          {undo && (
            <div className="notice notice-info small">
              {undo.label}.{' '}
              <button className="link" disabled={!!busy} onClick={() => void saveExclusions(undo.prev, null)}>
                Undo
              </button>
            </div>
          )}
          {KIND_ORDER.map((kind) => {
            const info = KIND_INFO[kind];
            const count = check.counts[info.countKey] ?? 0;
            const excluded = check.excluded?.[kind] ?? 0;
            const items = byKind.get(kind) ?? [];
            const firstActive = items.find((f) => !f.excluded);
            const severity = count === 0 ? 'none' : firstActive ? firstActive.severity : excluded >= count ? 'excluded' : kind === 'open_edge' ? 'error' : 'warning';
            const open = activeKind === kind;
            return (
              <div key={kind} className={`check-group ${open ? 'open' : ''} ${count === 0 ? 'zero' : ''}`}>
                <button
                  className="check-head"
                  disabled={count === 0}
                  onClick={() => {
                    if (open) {
                      setActiveKind(null);
                      return;
                    }
                    setActiveKind(kind);
                    setIndex(0);
                    frameFinding(items[0] ?? null);
                  }}
                >
                  <span className="caret">{count === 0 ? '·' : open ? '▾' : '▸'}</span>
                  <span className={`sev sev-${severity}`} />
                  <span className="grow">{info.label}</span>
                  {excluded > 0 && <span className="small muted">{excluded.toLocaleString()} excluded ·</span>}
                  <span className="count">{count.toLocaleString()}</span>
                </button>
                {open && (
                  <div className="check-body">
                    <p className="muted small">{info.help}</p>
                    {items.length === 0 ? (
                      <p className="small">No individual entries listed (the server lists at most 200 per kind).</p>
                    ) : (
                      <>
                        <div className="finding-nav">
                          <button className="btn btn-sm" onClick={() => go(index - 1)} aria-label="Previous finding">
                            ‹
                          </button>
                          <span className="small">
                            {index + 1} / {items.length}
                            {items.length < count ? ` (of ${count.toLocaleString()})` : ''}
                          </span>
                          <button className="btn btn-sm" onClick={() => go(index + 1)} aria-label="Next finding">
                            ›
                          </button>
                          <button className="btn btn-sm" onClick={() => frameFinding(current)} title="Frame this finding">
                            Frame
                          </button>
                        </div>
                        {current && (
                          <div className={`finding ${current.excluded ? 'excluded' : ''}`}>
                            {current.excluded && (
                              <div className="small">
                                <span className="chip">excluded{current.excluded_by === 'layer' ? ' with its layer' : ''}</span>
                              </div>
                            )}
                            <div className="small">{current.message}</div>
                            {current.layers && current.layers.length > 0 && <div className="muted small">Layers: {current.layers.join(', ')}</div>}
                            <div className="muted mono small">{findingFacts(current)}</div>
                            <div className="finding-actions">
                              {!current.id ? (
                                <span className="muted small">Re-check to enable exclusions.</span>
                              ) : current.excluded_by === 'layer' ? (
                                (current.layers ?? [])
                                  .filter((n) => exclusions.layers.includes(n))
                                  .map((n) => (
                                    <button key={n} className="btn btn-xs" disabled={!!busy} onClick={() => toggleLayer(n)}>
                                      Include “{n}” again
                                    </button>
                                  ))
                              ) : (
                                <button
                                  className="btn btn-xs"
                                  disabled={!!busy}
                                  title={current.excluded ? 'Count this finding again' : 'Mark as known (an intended opening, a small detail): it stops counting. The solvers are not affected.'}
                                  onClick={() => toggleFinding(current)}
                                >
                                  {current.excluded ? 'Include again' : 'Exclude'}
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          {check.watertight ? (
            <div className="ok-box">Watertight: the model encloses a volume.</div>
          ) : (
            <div className="warn-box">
              {blocking && blocking.length ? blocking.join(' ') : 'Not watertight.'} If this is an open scene (an object in free field), declare it in the simulation
              Settings to use the wave solver; if it is a room with holes, fix the open edges and re-import.
              {(check.excluded?.open_edge ?? 0) > 0 && ' Excluding open edges does not close the model: the server still refuses the wave solver.'}
            </div>
          )}
          {(check.notes ?? []).map((n) => (
            <div key={n} className="warn-box small">
              {n}
            </div>
          ))}
          {exclusions.findings.length > 0 && check.exclusions && check.exclusions.unmatched.length > 0 && (
            <div className="muted small">
              {check.exclusions.unmatched.length} excluded finding(s) not found by this check (kept: they apply again if they come back).
            </div>
          )}
        </section>
        <section>
          <h3>Layers ({geometry.layers.length})</h3>
          <table className="table compact">
            <thead>
              <tr>
                <th />
                <th>Layer</th>
                <th className="num">Triangles</th>
                <th className="num">Area m²</th>
                <th title="Exclude every finding on this layer: they stop counting (the solvers are not affected)">Excl.</th>
              </tr>
            </thead>
            <tbody>
              {geometry.layers.map((l) => (
                <tr key={l.name} className={hidden.has(l.name) ? 'dimmed' : ''}>
                  <td>
                    <EyeButton
                      visible={!hidden.has(l.name)}
                      onToggle={() =>
                        setHidden((h) => {
                          const n = new Set(h);
                          if (n.has(l.name)) n.delete(l.name);
                          else n.add(l.name);
                          return n;
                        })
                      }
                    />
                  </td>
                  <td>
                    <span className="swatch" style={{ background: layerColor(layerIndex.get(l.name) ?? 0) }} />
                    <span className="ellipsis" title={l.name}>
                      {l.name}
                    </span>
                  </td>
                  <td className="num">{l.n_triangles.toLocaleString()}</td>
                  <td className="num">{fmtNum(l.area_m2, 2)}</td>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Exclude the findings on ${l.name}`}
                      title="Exclude every finding on this layer"
                      disabled={!!busy || !detail}
                      checked={exclusions.layers.includes(l.name)}
                      onChange={() => toggleLayer(l.name)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
