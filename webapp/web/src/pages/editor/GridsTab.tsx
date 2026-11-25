import type { ReactNode } from 'react';
import type { ReceiverGrid, Simulation, SimulationSettings, Validation, Vec3 } from '../../api';
import { EyeButton, Menu, NumField, Spinner } from '../../components/ui';
import { fmtDuration, fmtNum } from '../../lib/format';
import type { Bounds } from '../../lib/geom';
import { DEFAULT_GRID_RAYS, defaultGridLayer, GRID_MAX_POINTS, GRID_POINT_S, newPlaneGrid, newSurfaceGrid, nextGridId } from '../../lib/grids';
import type { GridPreview } from './useGridPreviews';

export type GridPickMode = 'layer' | 'center';

export interface GridPick {
  gridId: string;
  mode: GridPickMode;
}

/** Disabled points by cause, from the server's reasons. */
function reasonCounts(reasons: Record<string, string>): { label: string; n: number }[] {
  let outside = 0;
  let surface = 0;
  let source = 0;
  for (const r of Object.values(reasons)) {
    if (r.includes('outside')) outside++;
    else if (r.includes('surface')) surface++;
    else source++;
  }
  return [
    { label: 'outside the air volume', n: outside },
    { label: '< 0.10 m from a surface', n: surface },
    { label: '< 0.50 m from a source', n: source },
  ].filter((x) => x.n > 0);
}

export function GridsTab({
  sim,
  layers,
  labelOf,
  layerBounds,
  bounds,
  validation,
  previews,
  selectedId,
  onSelect,
  onChange,
  onSettings,
  pick,
  onPick,
  hidden,
  onToggleHidden,
  onFocus,
  locked,
}: {
  sim: Simulation;
  /** Layer names, room layers first. */
  layers: string[];
  labelOf: (name: string) => string;
  layerBounds: Map<string, Bounds>;
  bounds: Bounds | null;
  validation: Validation | null;
  previews: Map<string, GridPreview>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onChange: (grids: ReceiverGrid[]) => void;
  onSettings: (patch: Partial<SimulationSettings>) => void;
  pick: GridPick | null;
  onPick: (p: GridPick | null) => void;
  hidden: Set<string>;
  onToggleHidden: (id: string) => void;
  onFocus: (id: string) => void;
  locked: boolean;
}) {
  const grids = sim.receiver_grids ?? [];
  const gridValidity = new Map((validation?.grids ?? []).map((g) => [g.id, g]));
  const nSources = Math.max(1, sim.sources.length);
  const rays = sim.settings.grid_rays ?? DEFAULT_GRID_RAYS;

  const patch = (id: string, p: Partial<ReceiverGrid>) => onChange(grids.map((g) => (g.id === id ? { ...g, ...p } : g)));
  const add = (kind: 'surface' | 'plane') => {
    const id = nextGridId(grids);
    const name = `${kind === 'surface' ? 'Surface' : 'Plane'} grid ${grids.length + 1}`;
    const roomLayers = layers.filter((n) => !n.startsWith('obj:'));
    const g =
      kind === 'surface' ? newSurfaceGrid(id, name, defaultGridLayer(roomLayers.length ? roomLayers : layers, layerBounds) ?? layers[0]) : newPlaneGrid(id, name, bounds);
    onChange([...grids, g]);
    onSelect(id);
  };
  const duplicate = (g: ReceiverGrid) => {
    const id = nextGridId(grids);
    onChange([...grids, { ...g, id, name: `${g.name} copy` }]);
    onSelect(id);
  };
  const remove = (id: string) => {
    onChange(grids.filter((g) => g.id !== id));
    if (selectedId === id) onSelect(null);
    if (pick?.gridId === id) onPick(null);
  };
  const totalValid = grids.reduce((s, g) => s + (previews.get(g.id)?.data?.n_valid ?? gridValidity.get(g.id)?.n_valid ?? 0), 0);
  const estimate = totalValid * nSources * GRID_POINT_S * Math.sqrt(rays / DEFAULT_GRID_RAYS);

  const pickButton = (g: ReceiverGrid, mode: GridPickMode, label: string, title: string) => {
    const on = pick?.gridId === g.id && pick.mode === mode;
    return (
      <button
        type="button"
        className={`btn btn-xs ${on ? 'btn-on' : ''}`}
        aria-pressed={on}
        title={title}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(g.id);
          onPick(on ? null : { gridId: g.id, mode });
        }}
      >
        {on ? 'Picking… (Esc)' : label}
      </button>
    );
  };

  const field = (label: string, title: string, node: ReactNode, unit?: string) => (
    <label className="inline-field" title={title}>
      <span>{label}</span>
      {node}
      {unit && <span className="unit">{unit}</span>}
    </label>
  );

  return (
    <div className="tab-content grids-tab">
      <div className="tab-toolbar">
        <span className="small muted">
          {grids.length} grid{grids.length === 1 ? '' : 's'}
        </span>
        <div className="grow" />
        <button className="btn btn-xs btn-primary" disabled={locked} onClick={() => add('surface')} title="A lattice of points over a layer (e.g. the audience area), offset along its normal">
          + Surface grid
        </button>
        <button className="btn btn-xs" disabled={locked} onClick={() => add('plane')} title="A free rectangle of points: centre, width, length, azimuth and tilt">
          + Plane grid
        </button>
      </div>
      <p className="small muted grids-intro">
        Surface receivers give maps of the room parameters. They are computed with the geometrical solver only (energy-based: T20, T30, EDT, C50, C80, D50, Ts, G, SPL, J<sub>LF</sub>, L<sub>J</sub>, STI), at
        most {GRID_MAX_POINTS} points per grid. Points outside the air volume, &lt; 0.10 m from a surface or &lt; 0.50 m from a source are disabled (grey).
      </p>
      {grids.length === 0 && (
        <div className="empty-inline">
          No grid yet.{' '}
          <button className="link" disabled={locked} onClick={() => add('surface')}>
            Add a surface grid
          </button>{' '}
          on a layer, or a plane grid.
        </div>
      )}
      <div className="point-list">
        {grids.map((g) => {
          const pv = previews.get(g.id);
          const gv = gridValidity.get(g.id);
          const error = pv?.error ?? gv?.error ?? null;
          const nPoints = pv?.data?.n_points ?? gv?.n_points ?? null;
          const nValid = pv?.data?.n_valid ?? gv?.n_valid ?? null;
          const bad = !!error || nValid === 0;
          const counts = pv?.data ? reasonCounts(pv.data.reasons) : [];
          const layerMissing = g.kind === 'surface' && !!g.layer && !layers.includes(g.layer);
          return (
            <div key={g.id} data-id={g.id} className={`point-row grid-row ${selectedId === g.id ? 'selected' : ''} ${bad ? 'invalid' : ''}`} onClick={() => onSelect(g.id)}>
              <fieldset disabled={locked}>
                <div className="point-head">
                  <span className={`grid-dot ${bad ? 'invalid' : ''}`} title={g.id} aria-hidden="true" />
                  <input className="name-input" value={g.name} aria-label="Grid name" onChange={(e) => patch(g.id, { name: e.target.value })} />
                  <span className="badge">{g.kind === 'surface' ? 'surface' : 'plane'}</span>
                  <EyeButton visible={!hidden.has(g.id)} onToggle={() => onToggleHidden(g.id)} title={hidden.has(g.id) ? 'Show its points in the 3D view' : 'Hide its points in the 3D view'} />
                  <Menu
                    items={[
                      { label: 'Frame in 3D', onClick: () => onFocus(g.id), disabled: !pv?.data?.points.length },
                      { label: 'Duplicate', onClick: () => duplicate(g) },
                      { label: 'Delete', onClick: () => remove(g.id), danger: true },
                    ]}
                  />
                </div>
                {g.kind === 'surface' ? (
                  <>
                    <div className="point-fields">
                      {field(
                        'Layer',
                        'The layer the grid follows (every triangle of it)',
                        <select className="grid-layer" value={g.layer ?? ''} onChange={(e) => patch(g.id, { layer: e.target.value })}>
                          {layerMissing && <option value={g.layer}>{g.layer} (missing)</option>}
                          {layers.map((n) => (
                            <option key={n} value={n}>
                              {labelOf(n)}
                            </option>
                          ))}
                        </select>,
                      )}
                      {pickButton(g, 'layer', 'Pick in 3D', 'Click a surface in the 3D view to use its layer')}
                    </div>
                    <div className="point-fields">
                      {field(
                        'Offset',
                        'Distance from the layer along its normal, towards the air (0.01–10 m); e.g. ear height above the floor',
                        <NumField value={g.offset_m ?? 1.2} min={0.01} max={10} step={0.1} digits={2} onChange={(v) => patch(g.id, { offset_m: v ?? 1.2 })} ariaLabel="Offset (m)" />,
                        'm',
                      )}
                      {field(
                        'Spacing',
                        'Distance between grid points (0.1–20 m, steps of 0.1 m); each point stands for spacing² of area',
                        <NumField value={g.spacing_m} min={0.1} max={20} step={0.1} digits={1} onChange={(v) => patch(g.id, { spacing_m: Math.round((v ?? 1) * 10) / 10 })} ariaLabel="Spacing (m)" />,
                        'm',
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="point-fields">
                      <span className="small muted">Centre</span>
                      {(['X', 'Y', 'Z'] as const).map((axis, k) => (
                        <label key={axis} className="inline-field">
                          <span>{axis}</span>
                          <NumField
                            value={g.center?.[k] ?? 0}
                            step={0.1}
                            digits={3}
                            ariaLabel={`Centre ${axis} (m)`}
                            onChange={(v) => {
                              const c: Vec3 = [...(g.center ?? [0, 0, 0])] as Vec3;
                              c[k] = v ?? 0;
                              patch(g.id, { center: c });
                            }}
                          />
                        </label>
                      ))}
                      <span className="unit">m</span>
                      {pickButton(g, 'center', 'Pick in 3D', 'Click a surface in the 3D view to move the centre there (on a floor: at the current height above it)')}
                    </div>
                    <div className="point-fields">
                      {field('Width', 'Size along the azimuth direction (0.1–500 m)', <NumField value={g.width_m ?? 4} min={0.1} max={500} step={0.5} digits={2} onChange={(v) => patch(g.id, { width_m: v ?? 4 })} ariaLabel="Width (m)" />, 'm')}
                      {field('Length', 'Size across the azimuth direction (0.1–500 m)', <NumField value={g.length_m ?? 4} min={0.1} max={500} step={0.5} digits={2} onChange={(v) => patch(g.id, { length_m: v ?? 4 })} ariaLabel="Length (m)" />, 'm')}
                    </div>
                    <div className="point-fields">
                      {field('Az', 'Azimuth of the width axis, counter-clockwise from +X (degrees)', <NumField value={g.azimuth_deg ?? 0} min={-360} max={360} step={5} digits={1} onChange={(v) => patch(g.id, { azimuth_deg: v ?? 0 })} ariaLabel="Azimuth (degrees)" />, '°')}
                      {field('Tilt', 'Rotation about the width axis: 0 = horizontal, 90 = vertical (degrees)', <NumField value={g.tilt_deg ?? 0} min={-90} max={90} step={5} digits={1} onChange={(v) => patch(g.id, { tilt_deg: v ?? 0 })} ariaLabel="Tilt (degrees)" />, '°')}
                      {field(
                        'Spacing',
                        'Distance between grid points (0.1–20 m, steps of 0.1 m)',
                        <NumField value={g.spacing_m} min={0.1} max={20} step={0.1} digits={1} onChange={(v) => patch(g.id, { spacing_m: Math.round((v ?? 1) * 10) / 10 })} ariaLabel="Spacing (m)" />,
                        'm',
                      )}
                    </div>
                  </>
                )}
                <div className="grid-status small">
                  {pv?.loading && <Spinner size={10} />}
                  {nPoints !== null ? (
                    <span>
                      <b>{nPoints}</b> point{nPoints === 1 ? '' : 's'} · <b className={nValid === 0 ? 'text-err' : ''}>{nValid ?? '–'}</b> valid
                      {nValid ? <span className="muted"> · {fmtNum(nValid * g.spacing_m ** 2, nValid * g.spacing_m ** 2 >= 100 ? 0 : 1)} m²</span> : null}
                    </span>
                  ) : (
                    !pv?.loading && !error && <span className="muted">{locked ? '' : 'Preview after saving…'}</span>
                  )}
                </div>
                {counts.length > 0 && <div className="small muted">Disabled: {counts.map((c) => `${c.n} ${c.label}`).join(' · ')}</div>}
                {error && <div className="small text-err">{error}</div>}
                {!error && nValid === 0 && nPoints !== null && <div className="small text-err">No valid point: this grid blocks the run.</div>}
              </fieldset>
            </div>
          );
        })}
      </div>
      {grids.length > 0 && (
        <div className="inline-panel grids-cost">
          <div className="point-fields">
            {field(
              'Rays per point',
              'Rays traced for every grid point (1000–2 000 000; default 50 000). Noise falls as 1/√rays; cost grows as √rays.',
              <NumField
                value={rays}
                min={1000}
                max={2000000}
                step={10000}
                digits={0}
                width={80}
                disabled={locked}
                onChange={(v) => onSettings({ grid_rays: Math.round(v ?? DEFAULT_GRID_RAYS) })}
                ariaLabel="Rays per grid point"
              />,
            )}
          </div>
          <div className="small muted">
            {totalValid} valid point{totalValid === 1 ? '' : 's'} × {sim.sources.length} source{sim.sources.length === 1 ? '' : 's'}: ≈ {fmtDuration(estimate)} with the GPU (measured {GRID_POINT_S} s per point and source at 50 000 rays; slower on the CPU).
            Measured noise at 50 000 rays: &lt; 0.2 JND in a 150 m³ room; in a 15 000 m³ hall C80 0.3, D50 0.1, G 0.1, STI 0.15, EDT 0.7 JND, but T30 ~2 JND at isolated points (the maps offer a
            display-only median filter).
          </div>
        </div>
      )}
    </div>
  );
}
