import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { BANDS_HZ, type Material, type MaterialLibrary as MaterialLibraryInfo, type MaterialsResponse } from '../../api';
import { Plot } from '../../components/Plot';
import { absorptionColor, COLORS } from '../../lib/colors';
import { bandLabel, fmtNum } from '../../lib/format';
import { MaterialDetailsDialog } from './MaterialDetails';

const MAX_LIST = 400;

export function AlphaChart({
  values,
  compare,
  height = 150,
  compareLabel = 'Entered',
  label = 'α',
  band,
}: {
  values: (number | null)[];
  compare?: (number | null)[] | null;
  height?: number;
  compareLabel?: string;
  label?: string;
  /** Spread of each value (e.g. α at both ends of an input's uncertainty band), drawn as whiskers. */
  band?: { low: number[]; high: number[] } | null;
}) {
  const x = BANDS_HZ.map((f) => bandLabel(f));
  const data = useMemo<Data[]>(() => {
    const bar: Data = {
      type: 'bar',
      x,
      y: values,
      name: label,
      marker: { color: COLORS.accent },
      hovertemplate: '%{x} Hz: %{y:.3f}<extra></extra>',
    };
    if (band) {
      const v = values.map((y) => y ?? 0);
      Object.assign(bar, {
        error_y: {
          type: 'data',
          symmetric: false,
          array: band.high.map((h, i) => Math.max(0, h - v[i])),
          arrayminus: band.low.map((l, i) => Math.max(0, v[i] - l)),
          color: '#d8dde2',
          thickness: 1.5,
          width: 5,
        },
        customdata: band.low.map((l, i) => [l, band.high[i]]),
        hovertemplate: '%{x} Hz: %{y:.3f} (band %{customdata[0]:.3f} – %{customdata[1]:.3f})<extra></extra>',
      });
    }
    const d: Data[] = [bar];
    if (compare) {
      d.push({
        type: 'scatter',
        mode: 'lines+markers',
        x,
        y: compare,
        name: compareLabel,
        line: { color: '#d8dde2', width: 2, dash: 'dot' },
        marker: { size: 8 },
        hovertemplate: '%{x} Hz: %{y:.3f}<extra></extra>',
      });
    }
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, compare, label, compareLabel, band]);
  const layout = useMemo<Partial<Layout>>(
    () => ({
      height,
      margin: { l: 36, r: 8, t: 6, b: 28 },
      showlegend: !!compare,
      yaxis: { range: [0, 1.0], dtick: 0.2, title: { text: '' } },
      xaxis: { type: 'category', title: { text: '' } },
      shapes: [
        {
          type: 'line',
          xref: 'paper',
          x0: 0,
          x1: 1,
          y0: 0.95,
          y1: 0.95,
          line: { color: 'rgba(240,168,58,0.5)', width: 1, dash: 'dot' },
        },
      ],
    }),
    [height, compare],
  );
  return <Plot data={data} layout={layout} config={{ displayModeBar: false }} style={{ height }} />;
}

export function MaterialLibrary({
  materials,
  selectedCount,
  currentIds,
  onAssign,
  onClose,
  onNew,
  onDelete,
  onRename,
  onDuplicate,
  locked,
  previewId,
  setPreviewId,
}: {
  materials: MaterialsResponse;
  selectedCount: number;
  currentIds: string[];
  onAssign: (id: string) => void;
  onClose: () => void;
  onNew: () => void;
  onDelete: (m: Material) => void;
  /** Custom materials only (the id stays: simulations keep it). */
  onRename?: (m: Material) => void;
  /** Any material, copied into "My materials". */
  onDuplicate?: (m: Material) => void;
  locked: boolean;
  previewId: string | null;
  setPreviewId: (id: string | null) => void;
}) {
  const [query, setQuery] = useState('');
  const [library, setLibrary] = useState('all');
  const [category, setCategory] = useState('all');
  const [detailsOpen, setDetailsOpen] = useState(false);
  const libs = useMemo(() => new Map(materials.libraries.map((l) => [l.id, l])), [materials]);

  const inLibrary = useMemo(
    () => (library === 'all' ? materials.materials : materials.materials.filter((m) => m.library === library)),
    [materials, library],
  );
  const categories = useMemo(() => Array.from(new Set(inLibrary.map((m) => m.category || 'Uncategorised'))).sort(), [inLibrary]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return inLibrary.filter((m) => {
      if (category !== 'all' && (m.category || 'Uncategorised') !== category) return false;
      if (!q) return true;
      return (
        m.name.toLowerCase().includes(q) ||
        (m.category ?? '').toLowerCase().includes(q) ||
        (m.description ?? '').toLowerCase().includes(q) ||
        m.id.toLowerCase().includes(q)
      );
    });
  }, [inLibrary, category, query]);

  const preview = previewId ? materials.materials.find((m) => m.id === previewId) ?? null : null;
  const previewLib = preview ? libs.get(preview.library) : undefined;
  const current = new Set(currentIds);

  return (
    <aside className="drawer library" aria-label="Material library">
      <div className="drawer-head">
        <h3>Material library</h3>
        <div className="grow" />
        <button className="btn btn-xs" onClick={onNew} disabled={locked}>
          New material…
        </button>
        <button className="icon-btn" aria-label="Close library" onClick={onClose}>
          ×
        </button>
      </div>
      <div className="drawer-filters">
        <input type="search" placeholder="Search materials…" value={query} onChange={(e) => setQuery(e.target.value)} autoFocus />
        <div className="row gap">
          <select
            value={library}
            onChange={(e) => {
              setLibrary(e.target.value);
              setCategory('all');
            }}
            aria-label="Library"
          >
            <option value="all">All libraries</option>
            {materials.libraries.map((l) => (
              <option key={l.id} value={l.id}>
                {l.name}
              </option>
            ))}
          </select>
          <select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category">
            <option value="all">All categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <div className="muted small">
          {filtered.length} material{filtered.length === 1 ? '' : 's'}
          {filtered.length > MAX_LIST ? ` (showing ${MAX_LIST}; refine the search)` : ''}
        </div>
      </div>
      <ul className="mat-list" role="listbox">
        {filtered.slice(0, MAX_LIST).map((m) => {
          const lib = libs.get(m.library);
          return (
            <li
              key={m.id}
              role="option"
              aria-selected={m.id === previewId}
              className={`mat-item ${m.id === previewId ? 'active' : ''} ${current.has(m.id) ? 'current' : ''}`}
              onClick={() => setPreviewId(m.id)}
              onDoubleClick={() => !locked && selectedCount > 0 && onAssign(m.id)}
              title={current.has(m.id) ? 'Assigned to a selected layer' : undefined}
            >
              <div className="mat-item-main">
                <span className="mat-name ellipsis">{m.name}</span>
                <span className="mat-cat muted ellipsis">
                  {m.category || 'Uncategorised'}
                  {library === 'all' && lib ? ` · ${lib.name}` : ''}
                </span>
              </div>
              <Sparkline values={m.absorption} />
              <div className="mat-badges">
                {m.clipped && (
                  <span className="badge badge-warn" title="Clipped to 0.95">
                    !
                  </span>
                )}
                {isLocalOnly(m, lib) && (
                  <span className="badge badge-lic" title={m.license || lib?.license}>
                    local
                  </span>
                )}
                {m.physical && (
                  <span className="badge" title={`Physical model for the wave band: ${describePhysical(m.physical)}`}>
                    phys
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="mat-preview">
        {!preview ? (
          <div className="muted small pad">Select a material to preview it. Double-click to assign.</div>
        ) : (
          <>
            <div className="row gap">
              <div className="grow">
                <div className="mat-preview-name">{preview.name}</div>
                <div className="muted small">
                  {preview.category || 'Uncategorised'} · {previewLib?.name ?? preview.library}
                </div>
              </div>
              <button className="btn btn-xs" onClick={() => setDetailsOpen(true)} title="α of the table vs the fit, normal-incidence reflection factor and where its phase comes from">
                Details…
              </button>
              {onDuplicate && (
                <button className="btn btn-xs" onClick={() => onDuplicate(preview)} disabled={locked} title="Copy it into My materials (new id), keeping its source and licence">
                  Duplicate…
                </button>
              )}
              {preview.library === 'custom' && onRename && (
                <button className="btn btn-xs" onClick={() => onRename(preview)} disabled={locked} title="Change the name only: simulations refer to the material's id, which stays">
                  Rename…
                </button>
              )}
              {preview.library === 'custom' && (
                <button className="btn btn-xs btn-danger-ghost" onClick={() => onDelete(preview)} disabled={locked}>
                  Delete
                </button>
              )}
            </div>
            <AlphaChart values={preview.absorption} height={140} />
            <table className="table compact">
              <tbody>
                <tr>
                  {BANDS_HZ.map((f) => (
                    <th key={f} className="num">
                      {bandLabel(f)}
                    </th>
                  ))}
                </tr>
                <tr>
                  {preview.absorption.map((v, i) => (
                    <td key={i} className="num">
                      {fmtNum(v, 2)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
            {preview.clipped && <div className="warn-box small">At least one value above 0.95 was clipped to 0.95 (physical maximum for random incidence).</div>}
            {isLocalOnly(preview, previewLib) && <div className="lic-box small">Not redistributable: {preview.license || previewLib?.license}</div>}
            <dl className="kv small">
              <dt>Scattering</dt>
              <dd>{fmtNum(preview.scattering, 2)} (default)</dd>
              {preview.absorption_63 !== null && (
                <>
                  <dt>α 63 Hz</dt>
                  <dd>{fmtNum(preview.absorption_63, 2)} (wave band only: the rays start at 125 Hz)</dd>
                </>
              )}
              <dt>Wave model</dt>
              <dd>{preview.physical ? describePhysical(preview.physical) : 'none (fitted from α)'}</dd>
              <dt>Source</dt>
              <dd>{preview.source || '–'}</dd>
              {preview.derived_from && (
                <>
                  <dt>Copied from</dt>
                  <dd>{materials.materials.find((x) => x.id === preview.derived_from)?.name ?? preview.derived_from}</dd>
                </>
              )}
              <dt>Licence</dt>
              <dd>{preview.license || '–'}</dd>
              {preview.description && (
                <>
                  <dt>Notes</dt>
                  <dd>{preview.description}</dd>
                </>
              )}
            </dl>
            <button className="btn btn-primary btn-block" disabled={locked || selectedCount === 0} onClick={() => onAssign(preview.id)}>
              {selectedCount === 0 ? 'Select layers to assign' : `Assign to ${selectedCount} layer${selectedCount === 1 ? '' : 's'}`}
            </button>
            {detailsOpen && (
              <MaterialDetailsDialog
                material={preview}
                library={previewLib}
                onClose={() => setDetailsOpen(false)}
                onDelete={
                  locked
                    ? undefined
                    : () => {
                        setDetailsOpen(false);
                        onDelete(preview);
                      }
                }
                onRename={
                  locked || !onRename
                    ? undefined
                    : () => {
                        setDetailsOpen(false);
                        onRename(preview);
                      }
                }
                onDuplicate={
                  locked || !onDuplicate
                    ? undefined
                    : () => {
                        setDetailsOpen(false);
                        onDuplicate(preview);
                      }
                }
              />
            )}
          </>
        )}
      </div>
    </aside>
  );
}

/** Local data (Odeon, BRAS), or a copy of it in My materials: never redistributed. */
export function isLocalOnly(m: Material, lib: MaterialLibraryInfo | undefined): boolean {
  return m.redistributable === false || (!!lib && !lib.redistributable);
}

function Sparkline({ values }: { values: number[] }) {
  return (
    <span className="sparkline" aria-hidden="true">
      {values.map((v, i) => (
        <span key={i} style={{ height: `${Math.max(2, Math.min(1, v) * 100)}%`, background: absorptionColor(v) }} />
      ))}
    </span>
  );
}

export function describePhysical(p: Material['physical']): string {
  if (!p) return '';
  if (p.type === 'table') {
    const t = p.gamma_table as { f?: number[] } | undefined;
    const f = t?.f ?? [];
    const what = p.input === 'z' ? 'impedance Z/ρc' : p.input === 'r' ? 'reflection factor R' : 'complex';
    return `measured ${what} table${f.length ? `, ${f.length} frequencies, ${fmtNum(f[0], 0)}–${fmtNum(f[f.length - 1], 0)} Hz` : ''} (phase known)`;
  }
  const parts: string[] = [p.type];
  for (const [k, v] of Object.entries(p)) {
    if (k === 'type') continue;
    parts.push(`${k} ${typeof v === 'number' ? v : JSON.stringify(v)}`);
  }
  return parts.join(', ');
}
