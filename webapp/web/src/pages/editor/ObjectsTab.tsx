import { useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from 'react';
import type { CatalogObject, Material, ObjectCatalog, PlacedObject, Vec3 } from '../../api';
import { ErrorBox, Menu, NumField, Spinner } from '../../components/ui';
import { fmtNum } from '../../lib/format';
import type { SelectMods } from './objectEditing';
import type { ObjectsSyncState } from './useObjectsSync';

const PROXY_LABEL: Record<string, string> = {
  upholstered: 'upholstered block',
  chair: 'seat + back planes',
  plane_top: 'top plane',
  box: 'box',
  bed: 'bed block',
  vertical_plane: 'vertical plane',
  none: 'none',
};

export function CatalogBadges({ c }: { c: CatalogObject }) {
  return (
    <>
      {c.proxy === 'none' && (
        <span className="badge badge-warn" title="Smaller than the modelled detail: drawn, but not simulated">
          visual only · no acoustic effect
        </span>
      )}
      {!c.visual && (
        <span className="badge" title="No realistic model: the acoustic proxy is drawn instead">
          no visual model
        </span>
      )}
    </>
  );
}

export function ObjectsTab({
  catalog,
  catalogError,
  catalogById,
  objects,
  selectedIds,
  onSelect,
  onAdd,
  onPatch,
  onDuplicate,
  onDelete,
  onFocus,
  placeId,
  setPlaceId,
  syncState,
  syncError,
  locked,
  materialsById,
  failed,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
}: {
  catalog: ObjectCatalog | null;
  catalogError: string | null;
  catalogById: Map<string, CatalogObject>;
  objects: PlacedObject[];
  selectedIds: readonly string[];
  /** Ctrl-click toggles a row, Shift-click selects the range from the last row clicked. */
  onSelect: (id: string, mods: SelectMods) => void;
  onAdd: (catalogId: string) => void;
  onPatch: (id: string, patch: Partial<PlacedObject>) => void;
  onDuplicate: (id: string) => void;
  onDelete: (id: string) => void;
  onFocus: (id: string) => void;
  placeId: string | null;
  setPlaceId: (id: string | null) => void;
  syncState: ObjectsSyncState;
  syncError: string | null;
  locked: boolean;
  materialsById: Map<string, Material>;
  failed: Record<string, string>;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [chosen, setChosen] = useState<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(true);

  const categories = useMemo(() => Array.from(new Set((catalog?.objects ?? []).map((o) => o.category))).sort(), [catalog]);
  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (catalog?.objects ?? []).filter(
      (o) =>
        (category === 'all' || o.category === category) &&
        (!q || o.name.toLowerCase().includes(q) || o.id.toLowerCase().includes(q) || o.category.toLowerCase().includes(q)),
    );
  }, [catalog, query, category]);
  const chosenItem = chosen ? (catalogById.get(chosen) ?? null) : null;

  const primary = selectedIds[selectedIds.length - 1] ?? null;
  useEffect(() => {
    if (!primary) return;
    document.querySelector(`.object-row[data-id="${CSS.escape(primary)}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [primary, objects.length]);

  const clickRow = (id: string, e: ReactMouseEvent) => {
    const onControl = !!(e.target as HTMLElement).closest('input, select, textarea, button, a, label');
    if (!onControl && (e.ctrlKey || e.metaKey)) onSelect(id, { toggle: true });
    else if (!onControl && e.shiftKey) onSelect(id, { range: objects.map((o) => o.id) });
    // Editing a field of a selected row keeps the multi-selection.
    else if (!(onControl && selectedIds.includes(id))) onSelect(id, {});
  };

  return (
    <div className="tab-content objects-tab">
      <div className="section-title">
        <button className="caret" onClick={() => setBrowserOpen((o) => !o)} aria-expanded={browserOpen}>
          {browserOpen ? '▾' : '▸'}
        </button>
        Object library
        {catalog && <span className="muted nocase small">({catalog.objects.length})</span>}
      </div>
      {browserOpen && (
        <>
          {catalogError && <ErrorBox message={`Object library unavailable: ${catalogError}`} />}
          {!catalog && !catalogError && <Spinner />}
          {catalog && (
            <>
              <div className="row gap">
                <input type="search" className="grow" placeholder="Search objects…" value={query} onChange={(e) => setQuery(e.target.value)} />
                <select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category">
                  <option value="all">All categories</option>
                  {categories.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
              <div className="catalog-grid" role="listbox" aria-label="Object library">
                {list.map((c) => (
                  <button
                    key={c.id}
                    role="option"
                    aria-selected={chosen === c.id}
                    className={`catalog-card ${chosen === c.id ? 'active' : ''} ${placeId === c.id ? 'placing' : ''}`}
                    onClick={() => setChosen(c.id)}
                    onDoubleClick={() => !locked && onAdd(c.id)}
                    title={`${c.name} · ${c.category}`}
                  >
                    {c.thumbnail_url ? (
                      <img src={c.thumbnail_url} alt="" loading="lazy" />
                    ) : (
                      <span className="thumb-placeholder" aria-hidden="true">
                        {c.proxy === 'none' ? '·' : '▭'}
                      </span>
                    )}
                    <span className="catalog-name">{c.name}</span>
                    {(c.proxy === 'none' || !c.visual) && <span className={`catalog-flag ${c.proxy === 'none' ? 'warn' : ''}`}>{c.proxy === 'none' ? 'visual only' : 'proxy only'}</span>}
                  </button>
                ))}
                {list.length === 0 && <div className="muted small">No object matches.</div>}
              </div>
              {chosenItem && (
                <div className="catalog-details">
                  <div className="row gap">
                    <b className="grow">{chosenItem.name}</b>
                    <span className="muted small">{chosenItem.category}</span>
                  </div>
                  <div className="badges-row">
                    <CatalogBadges c={chosenItem} />
                  </div>
                  <dl className="kv small">
                    <dt>Size (W×D×H)</dt>
                    <dd>
                      {fmtNum(chosenItem.dims[0])} × {fmtNum(chosenItem.dims[1])} × {fmtNum(chosenItem.dims[2])} m
                    </dd>
                    <dt>Acoustic proxy</dt>
                    <dd>{PROXY_LABEL[chosenItem.proxy] ?? chosenItem.proxy}</dd>
                    {Object.entries(chosenItem.parts).map(([part, v]) => (
                      <PartRow key={part} part={part} material={materialsById.get(v.material)} materialId={v.material} scattering={v.scattering} />
                    ))}
                    <dt>Visual model</dt>
                    <dd>{chosenItem.visual ? chosenItem.source : 'none (the proxy is drawn)'}</dd>
                    <dt>Licence</dt>
                    <dd>{chosenItem.license}</dd>
                    {chosenItem.note && (
                      <>
                        <dt>Note</dt>
                        <dd>{chosenItem.note}</dd>
                      </>
                    )}
                  </dl>
                  <div className="row gap">
                    <button className="btn btn-xs btn-primary" disabled={locked} onClick={() => onAdd(chosenItem.id)} title="Drop it on the floor at the room centre">
                      Add at centre
                    </button>
                    <button
                      className={`btn btn-xs ${placeId === chosenItem.id ? 'btn-on' : ''}`}
                      disabled={locked}
                      onClick={() => setPlaceId(placeId === chosenItem.id ? null : chosenItem.id)}
                      title="Then click a room surface in the 3D view (Shift-click to place several, Esc to stop)"
                    >
                      {placeId === chosenItem.id ? 'Placing… (Esc)' : 'Click to place'}
                    </button>
                  </div>
                </div>
              )}
              {catalog.license_note && <div className="muted small license-note">{catalog.license_note}</div>}
            </>
          )}
        </>
      )}

      <div className="section-title">
        Placed objects <span className="muted nocase small">({objects.length})</span>
        {selectedIds.length > 1 && <span className="muted nocase small">{selectedIds.length} selected</span>}
        <span className="grow" />
        <SyncIndicator state={syncState} error={syncError} />
        <span className="undo-buttons">
          <button className="btn btn-xs" disabled={locked || !canUndo} onClick={onUndo} title="Undo the last object change (Ctrl+Z)" aria-label="Undo">
            ↶
          </button>
          <button className="btn btn-xs" disabled={locked || !canRedo} onClick={onRedo} title="Redo (Ctrl+Shift+Z or Ctrl+Y)" aria-label="Redo">
            ↷
          </button>
        </span>
      </div>
      {objects.length > 1 && (
        <div className="muted small objects-hint">Ctrl-click or Shift-click (here or in 3D) to select several; the gizmo then moves or rotates them together.</div>
      )}
      {objects.length === 0 && <div className="empty-inline">No object in this geometry. Pick one above and add it.</div>}
      <div className="point-list">
        {objects.map((o) => {
          const c = catalogById.get(o.catalog_id);
          return (
            <div
              key={o.id}
              data-id={o.id}
              className={`point-row object-row ${selectedIds.includes(o.id) ? `selected${selectedIds.length > 1 ? ' multi' : ''}` : ''}`}
              onClick={(e) => clickRow(o.id, e)}
            >
              <fieldset disabled={locked}>
                <div className="point-head">
                  {c?.thumbnail_url ? <img className="obj-thumb" src={c.thumbnail_url} alt="" loading="lazy" /> : <span className="obj-thumb placeholder">▭</span>}
                  <input className="name-input" value={o.name} aria-label="Object name" onChange={(e) => onPatch(o.id, { name: e.target.value })} />
                  <button
                    className="btn btn-xs"
                    onClick={(e) => {
                      e.stopPropagation();
                      onFocus(o.id);
                    }}
                    title="Frame this object in the 3D view"
                  >
                    Focus
                  </button>
                  <Menu
                    items={[
                      { label: 'Duplicate', onClick: () => onDuplicate(o.id) },
                      { label: 'Delete', onClick: () => onDelete(o.id), danger: true },
                    ]}
                  />
                </div>
                <div className="muted small obj-sub">
                  {c ? `${c.name} · ${c.category}` : <span className="text-err">unknown catalog object “{o.catalog_id}”</span>}
                  {c && <CatalogBadges c={c} />}
                  {c && failed[c.id] && (
                    <span className="badge badge-warn" title={failed[c.id]}>
                      visual failed
                    </span>
                  )}
                </div>
                <div className="point-fields">
                  {(['X', 'Y', 'Z'] as const).map((axis, k) => (
                    <label key={axis} className="inline-field">
                      <span>{axis}</span>
                      <NumField
                        value={o.position[k]}
                        step={0.1}
                        digits={3}
                        ariaLabel={`${axis} (m)`}
                        onChange={(v) => {
                          const p: Vec3 = [...o.position];
                          p[k] = v ?? 0;
                          onPatch(o.id, { position: p });
                        }}
                      />
                    </label>
                  ))}
                  <span className="unit">m</span>
                </div>
                <div className="point-fields">
                  <label className="inline-field" title="Rotation about +Z, counter-clockwise (degrees)">
                    <span>Rot</span>
                    <NumField value={o.rotation_deg} step={15} digits={1} ariaLabel="Rotation (degrees)" onChange={(v) => onPatch(o.id, { rotation_deg: v ?? 0 })} />
                    <span className="unit">°</span>
                  </label>
                  <span className="small muted">Scale</span>
                  {(['W', 'D', 'H'] as const).map((axis, k) => (
                    <label key={axis} className="inline-field" title={`Scale along the object's ${axis === 'W' ? 'width (x)' : axis === 'D' ? 'depth (y)' : 'height (z)'}`}>
                      <span>{axis}</span>
                      <NumField
                        value={o.scale[k]}
                        min={0.05}
                        max={20}
                        step={0.05}
                        digits={3}
                        width={52}
                        ariaLabel={`Scale ${axis}`}
                        onChange={(v) => {
                          const sc: Vec3 = [...o.scale];
                          sc[k] = v ?? 1;
                          onPatch(o.id, { scale: sc });
                        }}
                      />
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PartRow({ part, material, materialId, scattering }: { part: string; material?: Material; materialId: string; scattering: number | null }) {
  return (
    <>
      <dt>Part “{part}”</dt>
      <dd>
        {material ? material.name : <span className="muted">{materialId}</span>}
        {scattering !== null && <span className="muted"> · scattering {fmtNum(scattering, 2)}</span>}
      </dd>
    </>
  );
}

function SyncIndicator({ state, error }: { state: ObjectsSyncState; error: string | null }) {
  if (state === 'error') {
    return (
      <span className="save-state error nocase" title={error ?? ''}>
        last change rejected (reverted)
      </span>
    );
  }
  return (
    <span className={`save-state nocase ${state}`}>
      {state === 'saving' ? (
        <>
          <Spinner size={10} /> updating solver mesh…
        </>
      ) : state === 'pending' ? (
        'unsaved…'
      ) : (
        'saved'
      )}
    </span>
  );
}
