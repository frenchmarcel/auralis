import { useEffect, useMemo, useRef, useState } from 'react';
import { api, errorMessage, type Ensemble, type EnsembleBody, type EnsemblePreview, type Receiver, type Vec3 } from '../../api';
import { useApp } from '../../AppContext';
import { ErrorBox, Modal, NumField, Spinner } from '../../components/ui';
import { fmtDuration, fmtNum } from '../../lib/format';
import { boundsOfPositions, type Bounds } from '../../lib/geom';
import type { PreviewPlayer } from '../../three/EnsemblePreview';
import { MAX_POINTS } from './PointsTab';

export interface StagePlacement {
  ensembleId: string | null;
  center: Vec3;
  facing_deg: number;
  width_m: number;
  depth_m: number;
  replace: boolean;
  /** Width and depth follow the room and the conductor until edited by hand. */
  sizeAuto: boolean;
}

/** What the 3D view draws while the dialog is open (the players as they would be added). */
export interface StagePreview {
  players: PreviewPlayer[];
  /** Player hovered in the list. */
  highlight: string | null;
}

const PREVIEW_DEBOUNCE_MS = 300;
const DEG = Math.PI / 180;
const r1 = (v: number) => Math.round(v * 10) / 10;

export function facingTowards(from: Vec3, to: Vec3): number {
  const a = Math.atan2(to[1] - from[1], to[0] - from[0]) / DEG;
  return r1(((a % 360) + 360) % 360);
}

/**
 * Default stage: 60 % of the room extent across the audience direction, and 60 %
 * of its extent along it, limited to 90 % of the free distance upstage of the
 * conductor (players sit behind the conductor, away from the audience).
 */
export function stageDefaults(bounds: Bounds, center: Vec3, facingDeg: number): { width_m: number; depth_m: number } {
  const f = facingDeg * DEG;
  const c = Math.abs(Math.cos(f));
  const s = Math.abs(Math.sin(f));
  const across = s * bounds.size[0] + c * bounds.size[1];
  const along = c * bounds.size[0] + s * bounds.size[1];
  const dx = -Math.cos(f);
  const dy = -Math.sin(f);
  const reach = (d: number, lo: number, hi: number, p: number) => (d > 1e-6 ? (hi - p) / d : d < -1e-6 ? (lo - p) / d : Infinity);
  const upstage = Math.max(0.5, Math.min(reach(dx, bounds.min[0], bounds.max[0], center[0]), reach(dy, bounds.min[1], bounds.max[1], center[1])));
  return { width_m: Math.max(1, r1(0.6 * across)), depth_m: Math.max(1, r1(Math.min(0.6 * along, 0.9 * upstage))) };
}

let ensemblesCache: Promise<Ensemble[]> | null = null;
function loadEnsembles(): Promise<Ensemble[]> {
  if (!ensemblesCache) {
    ensemblesCache = api.listEnsembles();
    ensemblesCache.catch(() => {
      ensemblesCache = null;
    });
  }
  return ensemblesCache;
}

/**
 * Docked beside the 3D view: every change of the stage asks the server for a preview
 * (nothing saved), drawn in the scene through `onPreview` and listed here.
 */
export function EnsembleDialog({
  simId,
  placement,
  setPlacement,
  bounds,
  receivers,
  nSources,
  usesWave,
  onPickFloor,
  onCancel,
  onApply,
  onPreview,
  onFrame,
}: {
  simId: string;
  placement: StagePlacement;
  setPlacement: (p: StagePlacement) => void;
  bounds: Bounds;
  receivers: Receiver[];
  nSources: number;
  usesWave: boolean;
  onPickFloor: () => void;
  onCancel: () => void;
  onApply: (body: EnsembleBody, ensemble: Ensemble) => Promise<boolean>;
  onPreview: (p: StagePreview | null) => void;
  onFrame: (center: Vec3, radius: number) => void;
}) {
  const app = useApp();
  const [ensembles, setEnsembles] = useState<Ensemble[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{ key: string; center: Vec3; data: EnsemblePreview } | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [hoverRow, setHoverRow] = useState<string | null>(null);
  /** Conductor position last framed in the 3D view. */
  const framed = useRef<string | null>(null);

  useEffect(() => {
    loadEnsembles()
      .then(setEnsembles)
      .catch((e) => setError(errorMessage(e)));
  }, []);

  const set = (patch: Partial<StagePlacement>) => {
    const next = { ...placement, ...patch };
    if (next.sizeAuto && ('center' in patch || 'facing_deg' in patch || 'sizeAuto' in patch)) {
      Object.assign(next, stageDefaults(bounds, next.center, next.facing_deg));
    }
    setPlacement(next);
  };

  const chosen = ensembles?.find((e) => e.id === placement.ensembleId) ?? null;
  const total = chosen ? (placement.replace ? 0 : nSources) + chosen.n_tracks : nSources;
  const tooMany = total > MAX_POINTS;
  const c = placement.center;
  const outside = c[0] < bounds.min[0] || c[0] > bounds.max[0] || c[1] < bounds.min[1] || c[1] > bounds.max[1];
  const bodyOf = (ens: Ensemble): EnsembleBody => ({
    ensemble_id: ens.id,
    center: placement.center,
    facing_deg: placement.facing_deg,
    width_m: placement.width_m,
    depth_m: placement.depth_m,
    replace_sources: placement.replace,
  });

  // The positions do not depend on replace_sources: it is not part of the key.
  const previewKey = chosen ? JSON.stringify([chosen.id, placement.center, placement.facing_deg, placement.width_m, placement.depth_m]) : null;
  useEffect(() => {
    if (!chosen || !previewKey) return;
    let stale = false;
    setPreviewing(true);
    const timer = window.setTimeout(() => {
      const body = bodyOf(chosen);
      api
        .previewEnsemble(simId, body)
        .then((data) => {
          if (stale) return;
          setPreview({ key: previewKey, center: body.center, data });
          setPreviewError(null);
        })
        .catch((e) => !stale && setPreviewError(errorMessage(e)))
        .finally(() => !stale && setPreviewing(false));
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      stale = true;
      window.clearTimeout(timer);
    };
    // previewKey carries every field of the request body that matters
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simId, previewKey]);

  const patternName = useMemo(() => {
    const names = new Map(app.directivities.map((d) => [d.id, d.name]));
    return (id: string) => names.get(id) ?? id;
  }, [app.directivities]);
  const players = useMemo<PreviewPlayer[] | null>(
    () =>
      preview?.data.sources.map((s) => ({ id: s.id, name: s.name, position: s.position, valid: s.valid, reason: s.reason, pattern: patternName(s.directivity) })) ?? null,
    [preview, patternName],
  );
  const current = !!preview && preview.key === previewKey;
  const nInvalid = preview ? preview.data.sources.filter((s) => !s.valid).length : 0;

  // The scene keeps the last preview while a new one is computed (and while the floor is picked).
  useEffect(() => {
    if (players) onPreview({ players, highlight: hoverRow });
  }, [players, hoverRow, onPreview]);

  const frameStage = () => {
    if (!preview || !players) return;
    framed.current = JSON.stringify(preview.center);
    const b = boundsOfPositions([preview.center, ...players.map((p) => p.position)].flat());
    if (b) onFrame(b.center, Math.max(2, b.radius * 1.5));
  };
  // Frame the stage on the first preview, then whenever the conductor has moved.
  useEffect(() => {
    if (preview && players && framed.current !== JSON.stringify(preview.center)) frameStage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preview, players]);

  const apply = async () => {
    if (!chosen) return;
    setBusy(true);
    const ok = await onApply(bodyOf(chosen), chosen);
    if (!ok) setBusy(false);
  };

  return (
    <Modal title="Add an ensemble" onClose={onCancel} width={450} className="ensemble-dialog" closeOnBackdrop={false}>
      <div className="modal-body ensemble-body">
        <div className="ensemble-list-col">
          <div className="section-title">Ensemble (one source per player)</div>
          {error && <ErrorBox message={`Ensembles unavailable: ${error}`} />}
          {!ensembles && !error && <Spinner />}
          {ensembles && ensembles.length === 0 && (
            <div className="muted small">No ensemble available: the anechoic recordings library is not installed on the server.</div>
          )}
          <ul className="ensemble-list" role="listbox" aria-label="Ensembles">
            {ensembles?.map((e) => (
              <li
                key={e.id}
                role="option"
                aria-selected={placement.ensembleId === e.id}
                className={`ensemble-item ${placement.ensembleId === e.id ? 'active' : ''}`}
                onClick={() => set({ ensembleId: e.id })}
              >
                <div className="row gap">
                  <b className="grow">{e.title}</b>
                  <span className="muted small nowrap">
                    {e.n_tracks} players · {fmtDuration(e.duration_s)}
                  </span>
                </div>
                <div className="small muted ensemble-credit">{e.credit}</div>
                <div className="small lic-text">{e.license}</div>
              </li>
            ))}
          </ul>
        </div>
        <div className="ensemble-stage-col">
          <div className="section-title">Stage</div>
          <div className="form-grid compact">
            <label title="Where the conductor stands, on the floor; players sit upstage of it">Conductor</label>
            <div className="point-fields">
              {(['X', 'Y', 'Z'] as const).map((axis, k) => (
                <label key={axis} className="inline-field">
                  <span>{axis}</span>
                  <NumField
                    value={c[k]}
                    step={0.1}
                    digits={2}
                    ariaLabel={`Conductor ${axis}`}
                    onChange={(v) => {
                      const p: Vec3 = [...c];
                      p[k] = v ?? 0;
                      set({ center: p });
                    }}
                  />
                </label>
              ))}
            </div>
            <span />
            <div className="row gap">
              <button className="btn btn-xs" onClick={onPickFloor} title="Click on the floor in the 3D view">
                Click on the floor…
              </button>
              <button className="btn btn-xs" onClick={() => set({ center: [bounds.center[0], bounds.center[1], bounds.min[2]] })}>
                Room centre
              </button>
            </div>
            <label title="Direction of the audience, azimuth counter-clockwise from +X">Audience direction</label>
            <div className="row gap">
              <NumField value={placement.facing_deg} step={15} digits={1} ariaLabel="Audience direction (degrees)" onChange={(v) => set({ facing_deg: v ?? 0 })} />
              <span className="unit">°</span>
            </div>
            <span />
            <div className="row gap wrap">
              {receivers.map((r) => (
                <button key={r.id} className="btn btn-xs" onClick={() => set({ facing_deg: facingTowards(c, r.position) })} title={`Face the audience towards ${r.name}`}>
                  towards {r.name}
                </button>
              ))}
              {receivers.length === 0 && <span className="muted small">Add a receiver to aim the stage at it.</span>}
            </div>
            <label>Stage width × depth</label>
            <div className="row gap">
              <NumField value={placement.width_m} min={0.5} max={200} step={0.5} digits={1} ariaLabel="Stage width (m)" onChange={(v) => setPlacement({ ...placement, width_m: v ?? 1, sizeAuto: false })} />
              <span className="unit">×</span>
              <NumField value={placement.depth_m} min={0.5} max={200} step={0.5} digits={1} ariaLabel="Stage depth (m)" onChange={(v) => setPlacement({ ...placement, depth_m: v ?? 1, sizeAuto: false })} />
              <span className="unit">m</span>
              {!placement.sizeAuto && (
                <button className="link small" onClick={() => set({ sizeAuto: true })}>
                  from the room
                </button>
              )}
            </div>
          </div>
          <label className="check">
            <input type="checkbox" checked={placement.replace} onChange={(e) => set({ replace: e.target.checked })} /> Replace the existing sources ({nSources})
          </label>
          {outside && <div className="warn-box small">The conductor point is outside the room's footprint.</div>}
          {chosen && (
            <div className={`small ${tooMany ? 'text-err' : 'muted'}`}>
              {chosen.n_tracks} new sources → {total} in total{tooMany ? ` (max ${MAX_POINTS}: replace the existing ones or pick a smaller ensemble)` : ''}.
            </div>
          )}
          {chosen && (
            <div className={usesWave ? 'warn-box small' : 'muted small'}>
              Each source has its own run{usesWave ? `: ${total} sources = ${total} wave-solver runs.` : '.'} The layout shrinks to fit {fmtNum(placement.width_m, 1)} ×{' '}
              {fmtNum(placement.depth_m, 1)} m if needed.
            </div>
          )}
        </div>
        {chosen && (
          <div className="ensemble-preview-col">
            <div className="row gap">
              <div className="section-title grow">Preview</div>
              {previewing && <Spinner size={11} />}
              <button className="btn btn-xs" onClick={frameStage} disabled={!players} title="Frame the conductor and the players in the 3D view">
                Frame
              </button>
            </div>
            {previewError && <div className="small text-err">Preview unavailable: {previewError}</div>}
            {!preview && !previewError && <div className="small muted">Placing the players…</div>}
            {preview && (
              <>
                <div className={`small ${current ? '' : 'muted'}`}>
                  {preview.data.n_sources} players in the 3D view
                  {nInvalid > 0 ? (
                    <span className="text-err">
                      {' '}
                      · {nInvalid} invalid
                    </span>
                  ) : (
                    ' · all valid'
                  )}
                  {preview.data.scale < 1 && <> · orchestra compressed to {Math.round(preview.data.scale * 100)} %</>}
                  {!current && ' · updating…'}
                </div>
                {nInvalid > 0 && (
                  <div className="warn-box small">
                    {nInvalid} player{nInvalid === 1 ? '' : 's'} would be added in red (outside the air volume or closer than 0.10 m to a surface): move the conductor,
                    widen the stage or move them afterwards.
                  </div>
                )}
                <ul className="ens-players" aria-label="Players">
                  {players?.map((p) => (
                    <li
                      key={p.id}
                      className={`ens-player ${p.valid ? '' : 'invalid'} ${hoverRow === p.id ? 'hover' : ''}`}
                      onMouseEnter={() => setHoverRow(p.id)}
                      onMouseLeave={() => setHoverRow((h) => (h === p.id ? null : h))}
                      title={p.valid ? `${p.name}: ${p.pattern}` : `${p.name}: ${p.reason ?? 'invalid position'}`}
                    >
                      <span className="mono ens-player-id">{p.id}</span>
                      <span className="ellipsis grow">{p.name}</span>
                      <span className="small muted ellipsis ens-player-pattern">{p.pattern}</span>
                      {!p.valid && <span className="small text-err ens-player-reason">{p.reason ?? 'invalid position'}</span>}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
      <div className="modal-footer">
        <button className="btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          className="btn btn-primary"
          disabled={!chosen || tooMany || busy}
          onClick={() => void apply()}
          title={nInvalid > 0 && current ? `${nInvalid} of the players fail the check: they would be added in red` : undefined}
        >
          {busy ? <Spinner /> : null} Add {chosen ? `${chosen.n_tracks} players` : 'ensemble'}
          {nInvalid > 0 && current ? ` (${nInvalid} invalid)` : ''}
        </button>
      </div>
    </Modal>
  );
}
