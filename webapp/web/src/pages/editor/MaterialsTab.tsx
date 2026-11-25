import type React from 'react';
import { useEffect, useRef, useState } from 'react';
import { api, BANDS_HZ, errorMessage, type Geometry, type Material, type MaterialLibrary, type MeshData, type Simulation, type Subsurface, type SubsurfacesResponse, type Validation, type Vec3 } from '../../api';
import { EyeButton, Spinner } from '../../components/ui';
import { bandLabel, fmtNum } from '../../lib/format';
import { boundsOfPositions } from '../../lib/geom';
import { subsurfacePositions, type SubsurfaceFocus } from '../../three/SubsurfaceHighlight';
import { isLocalOnly } from './MaterialLibrary';
import { BulkScatteringPopover, ScatteringCell, type Scattering } from './ScatteringCell';

/** Rows shown per expanded layer before "show all" (largest surfaces first). */
const SUB_ROWS = 40;

export function MaterialsTab({
  geometry,
  sim,
  materialsById,
  librariesById,
  selected,
  onRowClick,
  onOpenLibrary,
  hidden,
  onToggleHidden,
  onShowAll,
  onScattering,
  onSelectMany,
  validation,
  locked,
  colorOf,
  hovered,
  onHoverRow,
  labelOf = (n) => n,
  kindOf = () => 'room',
  instancesOf = () => 0,
  mesh = null,
  onScatteringMany,
  onSubsurfaceFocus,
  onFrame,
}: {
  geometry: Geometry;
  sim: Simulation;
  materialsById: Map<string, Material>;
  librariesById: Map<string, MaterialLibrary>;
  selected: Set<string>;
  onRowClick: (name: string, e: React.MouseEvent) => void;
  onOpenLibrary: (name?: string) => void;
  hidden: Set<string>;
  onToggleHidden: (name: string) => void;
  onShowAll: () => void;
  /** One value, 7 values per octave, or null (material default). Method syntax: callers typed for a single value still fit. */
  onScattering(name: string, v: number | number[] | null): void;
  onSelectMany: (names: string[]) => void;
  validation: Validation | null;
  locked: boolean;
  colorOf: (name: string) => string;
  hovered: string | null;
  onHoverRow: (name: string | null) => void;
  labelOf?: (name: string) => string;
  kindOf?: (name: string) => 'room' | 'object';
  instancesOf?: (name: string) => number;
  /** The mesh the 3D view shows: sub-surfaces are re-read when it changes. */
  mesh?: MeshData | null;
  /** One scattering for several layers, saved like a per-layer edit. */
  onScatteringMany?: (names: string[], v: Scattering) => void;
  /** The sub-surface to highlight in 3D (hovered, else pinned), or null. */
  onSubsurfaceFocus?: (f: SubsurfaceFocus | null) => void;
  onFrame?: (center: Vec3, radius: number) => void;
}) {
  const layers = geometry.layers;
  const subs = useSubsurfaces(geometry.id, mesh);
  const [hoverSub, setHoverSub] = useState<string | null>(null);
  const [pinSub, setPinSub] = useState<string | null>(null);
  const [showAll, setShowAll] = useState<Set<string>>(new Set());
  const [bulkOpen, setBulkOpen] = useState(false);
  const bulkBtn = useRef<HTMLButtonElement>(null);
  const focusKey = hoverSub ?? pinSub;
  const focusSub = focusKey ? subs.find(focusKey) : null;
  useEffect(() => {
    onSubsurfaceFocus?.(focusSub && focusKey ? { layer: focusSub.layer, triangles: focusSub.sub.triangles, key: focusKey } : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusKey, focusSub?.sub]);
  useEffect(() => () => onSubsurfaceFocus?.(null), [onSubsurfaceFocus]);
  const toggleExpand = (name: string) => {
    if (subs.expanded.has(name)) {
      if (pinSub?.startsWith(`${name}#`)) setPinSub(null);
      setHoverSub(null);
    }
    subs.toggle(name);
  };
  const frameSub = (layer: string, sub: Subsurface) => {
    const pos = mesh ? subsurfacePositions(mesh, { layer, triangles: sub.triangles }) : null;
    const b = pos ? boundsOfPositions(pos) : null;
    if (b && onFrame) onFrame(b.center, Math.max(0.6, b.radius * 1.4));
  };
  const selNames = layers.filter((l) => selected.has(l.name)).map((l) => l.name);
  const selValues = selNames.map((n) => sim.layers[n]?.scattering ?? null);
  const mixed = selValues.some((v) => JSON.stringify(v) !== JSON.stringify(selValues[0]));
  const roomLayers = layers.filter((l) => kindOf(l.name) !== 'object');
  const objectLayers = layers.filter((l) => kindOf(l.name) === 'object');
  const unassigned = layers.filter((l) => !sim.layers[l.name]?.material_id).map((l) => l.name);
  const renderRow = (l: (typeof layers)[number]) => {
    const a = sim.layers[l.name] ?? { material_id: null, scattering: null };
    const m = a.material_id ? materialsById.get(a.material_id) : undefined;
    const lib = m ? librariesById.get(m.library) : undefined;
    const isSel = selected.has(l.name);
    return (
      <tr
        key={l.name}
        className={`${isSel ? 'selected' : ''} ${hidden.has(l.name) ? 'dimmed' : ''} ${hovered === l.name ? 'hovered' : ''}`}
        onClick={(e) => onRowClick(l.name, e)}
        onDoubleClick={() => !locked && onOpenLibrary(l.name)}
        onMouseEnter={() => onHoverRow(l.name)}
        onMouseLeave={() => onHoverRow(null)}
      >
        <td className="col-eye">
          <EyeButton visible={!hidden.has(l.name)} onToggle={() => onToggleHidden(l.name)} />
        </td>
        <td
          className="col-layer"
          title={`${l.name}\n${l.n_triangles} triangles · ${fmtNum(l.area_m2, 2)} m²${
            kindOf(l.name) === 'object' ? `\nShared by the ${instancesOf(l.name)} instance(s) of this object in the geometry` : ''
          }`}
        >
          <button
            type="button"
            className={`icon-btn sub-caret ${subs.expanded.has(l.name) ? 'open' : ''}`}
            aria-expanded={subs.expanded.has(l.name)}
            aria-label={`Surfaces of ${labelOf(l.name)}`}
            title={kindOf(l.name) === 'object' ? 'Show the placed instances of this layer' : 'Show the surfaces of this layer (display only: materials are assigned per layer)'}
            onClick={(e) => {
              e.stopPropagation();
              toggleExpand(l.name);
            }}
            onDoubleClick={(e) => e.stopPropagation()}
          >
            ▸
          </button>
          <span className="swatch" style={{ background: colorOf(l.name) }} />
          <span className="ellipsis">{labelOf(l.name)}</span>
          {kindOf(l.name) === 'object' && instancesOf(l.name) > 1 && <span className="badge" title="Shared layer">×{instancesOf(l.name)}</span>}
        </td>
        <td className="col-mat">
          <button
            className={`mat-cell ${a.material_id ? '' : 'unassigned'}`}
            disabled={locked}
            onClick={(e) => {
              // Modifier clicks extend the row selection instead of opening the library.
              if (e.shiftKey || e.ctrlKey || e.metaKey) return;
              e.stopPropagation();
              onOpenLibrary(l.name);
            }}
            title={m ? `${m.name} (${lib?.name ?? m.library})` : a.material_id ? `Unknown material id ${a.material_id}` : 'Click to assign a material'}
          >
            {a.material_id ? (
              m ? (
                <>
                  <span className="ellipsis">{m.name}</span>
                  {m.clipped && (
                    <span className="badge badge-warn" title="At least one absorption value above 0.95 was clipped to 0.95">
                      clipped
                    </span>
                  )}
                  {isLocalOnly(m, lib) && (
                    <span className="badge badge-lic" title={m.license || lib?.license}>
                      local
                    </span>
                  )}
                </>
              ) : (
                <span className="text-err ellipsis">missing: {a.material_id}</span>
              )
            ) : (
              <span className="ellipsis">unassigned</span>
            )}
          </button>
        </td>
        <td className="col-scat" onClick={(e) => e.stopPropagation()}>
          <ScatteringCell layer={l.name} value={a.scattering} materialDefault={m ? m.scattering : null} disabled={locked} onChange={(v) => onScattering(l.name, v)} />
        </td>
      </tr>
    );
  };
  const renderSubRows = (l: (typeof layers)[number]) => {
    if (!subs.expanded.has(l.name)) return null;
    if (subs.error)
      return (
        <tr key={`${l.name}#err`} className="sub-row">
          <td />
          <td colSpan={3} className="small text-err">
            Surfaces unavailable: {subs.error}{' '}
            <button type="button" className="btn btn-xs" onClick={subs.retry}>
              Retry
            </button>
          </td>
        </tr>
      );
    const list = subs.byLayer.get(l.name);
    if (!list)
      return (
        <tr key={`${l.name}#load`} className="sub-row">
          <td />
          <td colSpan={3} className="small muted">
            {subs.loading ? 'Reading the surfaces…' : 'No surfaces for this layer'}
          </td>
        </tr>
      );
    const all = showAll.has(l.name) || list.length <= SUB_ROWS + 5;
    const shown = all ? list : list.slice(0, SUB_ROWS);
    const rest = list.slice(shown.length);
    const rows = shown.map((sub, i) => {
      const key = `${l.name}#${i}`;
      const extra = [sub.orientation, sub.group && sub.group !== l.name ? sub.group : null].filter(Boolean).join(' · ');
      return (
        <tr
          key={key}
          className={`sub-row ${pinSub === key ? 'pinned' : ''} ${hoverSub === key ? 'hovered' : ''}`}
          onMouseEnter={() => setHoverSub(key)}
          onMouseLeave={() => setHoverSub(null)}
          onClick={() => setPinSub((k) => (k === key ? null : key))}
          onDoubleClick={() => frameSub(l.name, sub)}
          title={`${sub.label}${extra ? ` (${extra})` : ''}\n${sub.n_triangles} triangle${sub.n_triangles === 1 ? '' : 's'} · ${fmtNum(sub.area_m2, 2)} m²\nClick: keep it highlighted · double-click: frame it in 3D\nThe material is the layer's: it applies to all its surfaces.`}
        >
          <td />
          <td colSpan={2} className="sub-label">
            <span className="ellipsis">{sub.label}</span>
            {extra && <span className="muted small ellipsis"> · {extra}</span>}
          </td>
          <td className="num small muted">{sub.degenerate ? `${sub.n_triangles} tri` : `${fmtNum(sub.area_m2, sub.area_m2 < 10 ? 2 : 1)} m²`}</td>
        </tr>
      );
    });
    if (rest.length)
      rows.push(
        <tr key={`${l.name}#more`} className="sub-row">
          <td />
          <td colSpan={3} className="small muted">
            {rest.length} smaller surface{rest.length === 1 ? '' : 's'} ({fmtNum(rest.reduce((x, s) => x + s.area_m2, 0), 2)} m²){' '}
            <button type="button" className="btn btn-xs btn-ghost" onClick={() => setShowAll((s) => new Set(s).add(l.name))}>
              Show all
            </button>
          </td>
        </tr>,
      );
    return rows;
  };
  const renderLayer = (l: (typeof layers)[number]) => [renderRow(l), renderSubRows(l)];

  return (
    <div className="tab-content materials-tab">
      <div className="tab-toolbar">
        <span className="small muted">
          {layers.length} layers{unassigned.length ? ` · ` : ''}
          {unassigned.length > 0 && <span className="text-warn">{unassigned.length} unassigned</span>}
        </span>
        <div className="grow" />
        <button className="btn btn-xs" onClick={() => onSelectMany(layers.map((l) => l.name))} title="Select all layers (Ctrl+A)">
          All
        </button>
        <button className="btn btn-xs" disabled={!unassigned.length} onClick={() => onSelectMany(unassigned)} title="Select the unassigned layers">
          Unassigned
        </button>
        {hidden.size > 0 && (
          <button className="btn btn-xs" onClick={onShowAll}>
            Show all
          </button>
        )}
        <span className="row tab-actions">
          {onScatteringMany && (
            <button
              ref={bulkBtn}
              className="btn btn-xs"
              disabled={locked || selNames.length === 0}
              onClick={() => setBulkOpen((o) => !o)}
              aria-expanded={bulkOpen}
              title="Set the scattering of all the selected layers: one value or one per octave"
            >
              Scattering…
            </button>
          )}
          <button className="btn btn-xs btn-primary" disabled={locked || selected.size === 0} onClick={() => onOpenLibrary()}>
            Assign…
          </button>
        </span>
        {bulkOpen && onScatteringMany && selNames.length > 0 && !locked && (
          <BulkScatteringPopover
            anchor={bulkBtn.current}
            count={selNames.length}
            initial={mixed ? null : selValues[0]}
            mixed={mixed}
            onApply={(v) => onScatteringMany(selNames, v)}
            onClose={() => setBulkOpen(false)}
          />
        )}
      </div>
      <div className="layer-table-wrap">
        <table className="table layer-table">
          <thead>
            <tr>
              <th className="col-eye" />
              <th>Layer</th>
              <th>Material</th>
              <th className="col-scat" title="Scattering coefficient (0–1), one value or one per octave (≡). Empty = material default.">
                Scatt.
              </th>
            </tr>
          </thead>
          <tbody>
            {objectLayers.length > 0 && (
              <tr className="group-row">
                <td colSpan={4}>Room</td>
              </tr>
            )}
            {roomLayers.map(renderLayer)}
            {objectLayers.length > 0 && (
              <tr className="group-row">
                <td colSpan={4}>
                  Objects <span className="muted">· a layer is shared by every instance of the same object</span>
                </td>
              </tr>
            )}
            {objectLayers.map(renderLayer)}
          </tbody>
        </table>
      </div>
      {subs.expanded.size > 0 && subs.data && <div className="small muted sub-note">{subs.data.note}</div>}
      <RtEstimate validation={validation} unassigned={unassigned.length} />
    </div>
  );
}

function RtEstimate({ validation, unassigned }: { validation: Validation | null; unassigned: number }) {
  return (
    <section className="rt-estimate">
      <div className="section-title">
        Reverberation time estimate
        <span className="muted small" title="Statistical estimate from the volume, the surface areas and the assigned absorption. Not a simulation result.">
          {' '}
          (statistical)
        </span>
      </div>
      {!validation ? (
        <Spinner />
      ) : validation.rt_estimate ? (
        <table className="table compact rt-table">
          <thead>
            <tr>
              <th />
              {BANDS_HZ.map((f) => (
                <th key={f} className="num">
                  {bandLabel(f)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Sabine</td>
              {validation.rt_estimate.sabine.map((v, i) => (
                <td key={i} className="num">
                  {fmtNum(v, 2)}
                </td>
              ))}
            </tr>
            <tr>
              <td>Eyring</td>
              {validation.rt_estimate.eyring.map((v, i) => (
                <td key={i} className="num">
                  {fmtNum(v, 2)}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      ) : (
        <div className="muted small">
          {validation.rt_note
            ? `${validation.rt_note[0].toUpperCase()}${validation.rt_note.slice(1)}.`
            : unassigned > 0
            ? `Assign all layers to see the estimate (${unassigned} unassigned).`
            : validation.volume_m3 === null
              ? 'No estimate: the volume is unknown (model not closed).'
              : 'No estimate available.'}
        </div>
      )}
      {validation && (
        <div className="muted small">
          {validation.volume_m3 !== null && validation.volume_m3 > 0 && !validation.rt_note ? `V = ${fmtNum(validation.volume_m3, 1)} m³` : 'V = –'}
          {validation.surface_m2 !== null ? ` · S = ${fmtNum(validation.surface_m2, 1)} m²` : ''} · values in s
        </div>
      )}
    </section>
  );
}

/** Sub-surfaces of the geometry, read on the first expanded layer and again when the mesh changes. */
function useSubsurfaces(gid: string, mesh: MeshData | null) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [data, setData] = useState<SubsurfacesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const wanted = expanded.size > 0;
  useEffect(() => {
    setData(null);
    setError(null);
  }, [gid, mesh]);
  useEffect(() => {
    if (!wanted || data) return;
    let cancelled = false;
    setLoading(true);
    api
      .getSubsurfaces(gid)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(errorMessage(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [gid, mesh, wanted, data, attempt]);
  const byLayer = new Map((data?.layers ?? []).map((l) => [l.name, l.subsurfaces]));
  return {
    expanded,
    data,
    error,
    loading,
    byLayer,
    toggle: (name: string) =>
      setExpanded((s) => {
        const n = new Set(s);
        if (n.has(name)) n.delete(name);
        else n.add(name);
        return n;
      }),
    retry: () => {
      setError(null);
      setAttempt((a) => a + 1);
    },
    /** "<layer>#<index>" -> the sub-surface. */
    find: (key: string) => {
      const at = key.lastIndexOf('#');
      const layer = key.slice(0, at);
      const sub = byLayer.get(layer)?.[Number(key.slice(at + 1))];
      return sub ? { layer, sub } : null;
    },
  };
}
