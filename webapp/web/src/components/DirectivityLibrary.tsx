import { useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { api, BANDS_HZ, errorMessage, type Directivity } from '../api';
import { useApp } from '../AppContext';
import { seriesColor } from '../lib/colors';
import type { DataTable } from '../lib/export';
import { bandLabel, fmtNum } from '../lib/format';
import { Plot } from './Plot';
import { ErrorBox, Modal, Spinner } from './ui';

/** Polar plots stop here (dB re the axis); deeper nulls are drawn at the floor. */
const FLOOR_DB = -40;
const GRID = 'rgba(128,128,128,0.35)';
const TICK = '#8a939b';

/** θ 0-180° from the axis, mirrored to a full turn (the engine's model is axisymmetric). */
function mirrored(theta: number[], db: number[]): { t: number[]; r: number[] } {
  const back = theta.slice(0, -1).reverse();
  return { t: [...theta, ...back.map((x) => 360 - x)], r: [...db, ...db.slice(0, -1).reverse()].map((v) => Math.max(FLOOR_DB, v)) };
}

export function PolarChart({ d, bands, height = 320 }: { d: Directivity; bands: number[]; height?: number }) {
  const data = useMemo<Data[]>(() => {
    if (!d.polar) return [];
    return bands.map((b) => {
      const m = mirrored(d.polar!.theta_deg, d.polar!.db[b] ?? []);
      return {
        type: 'scatterpolar',
        mode: 'lines',
        theta: m.t,
        r: m.r,
        name: `${bandLabel(BANDS_HZ[b])} Hz`,
        line: { color: seriesColor(b), width: 2 },
        hovertemplate: `${bandLabel(BANDS_HZ[b])} Hz, %{theta:.0f}°: %{r:.1f} dB<extra></extra>`,
      } as Data;
    });
  }, [d, bands]);
  const layout = useMemo<Partial<Layout>>(
    () => ({
      height,
      margin: { l: 30, r: 30, t: 30, b: 30 },
      showlegend: true,
      polar: {
        bgcolor: 'rgba(0,0,0,0)',
        radialaxis: { range: [FLOOR_DB, 3], dtick: 10, gridcolor: GRID, linecolor: GRID, tickfont: { size: 9.5, color: TICK }, ticksuffix: ' dB', angle: 90, tickangle: 90 },
        angularaxis: { rotation: 90, direction: 'clockwise', dtick: 30, gridcolor: GRID, linecolor: GRID, tickfont: { size: 10, color: TICK }, ticksuffix: '°' },
      },
    }) as Partial<Layout>,
    [height],
  );
  const table = (): DataTable => ({
    columns: ['angle from the axis (deg)', ...bands.map((b) => `${BANDS_HZ[b]} Hz (dB re axis)`)],
    rows: (d.polar?.theta_deg ?? []).map((t, i) => [t, ...bands.map((b) => d.polar!.db[b]?.[i] ?? null)]),
  });
  if (!d.polar) return <div className="small muted">No polar data from the server.</div>;
  return <Plot data={data} layout={layout} tools filename={`directivity_${d.name}`} table={table} />;
}

/** Presets and imported directivities: polar plots, DI, notes; import (.json/.csv/.sofa) and delete. */
export function DirectivityLibrary({ onClose, initialId }: { onClose: () => void; initialId?: string | null }) {
  const app = useApp();
  const list = app.directivities;
  const [selId, setSelId] = useState<string | null>(initialId ?? list.find((d) => d.kind === 'imported')?.id ?? list[0]?.id ?? null);
  const [bands, setBands] = useState<number[]>([1, 3, 5]);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imported, setImported] = useState<Directivity | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const sel = list.find((d) => d.id === selId) ?? list[0] ?? null;
  const presets = list.filter((d) => d.kind !== 'imported' && d.kind !== 'odeon_local');
  const imports = list.filter((d) => d.kind === 'imported');
  const odeon = list.filter((d) => d.kind === 'odeon_local');

  const doImport = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const d = await api.importDirectivity(file, name.trim());
      setImported(d);
      setFile(null);
      setName('');
      await app.reloadDirectivities();
      setSelId(d.id);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (d: Directivity) => {
    const ok = await app.confirm({ title: 'Delete directivity', message: <>Delete the imported directivity <b>{d.name}</b>?</>, confirmLabel: 'Delete', danger: true });
    if (!ok) return;
    setError(null);
    try {
      await api.deleteDirectivity(d.id);
      await app.reloadDirectivities();
      setSelId(null);
      if (imported?.id === d.id) setImported(null);
    } catch (e) {
      // 409 while a source uses it: the server names the simulations.
      setError(`Cannot delete “${d.name}”: ${errorMessage(e)}`);
    }
  };

  const item = (d: Directivity) => (
    <li key={d.id} role="option" aria-selected={d.id === sel?.id} className={`mat-item ${d.id === sel?.id ? 'active' : ''}`} onClick={() => setSelId(d.id)}>
      <div className="mat-item-main">
        <span className="mat-name ellipsis">{d.name}</span>
        <span className="mat-cat muted ellipsis">
          {d.kind === 'imported' ? (d.source ?? 'imported') : d.kind === 'odeon_local' ? 'Odeon (local)' : 'preset'}
          {d.di_db ? ` · DI ${fmtNum(d.di_db[3], 1)} dB at 1 kHz` : ''}
        </span>
      </div>
    </li>
  );

  return (
    <Modal title="Source directivities" onClose={onClose} width={1000} className="dir-modal">
      <div className="modal-body dir-body">
        <aside className="dir-list">
          <div className="section-title">Presets</div>
          <ul className="mat-list" role="listbox" aria-label="Preset directivities">
            {presets.map(item)}
          </ul>
          <div className="section-title">Imported</div>
          {imports.length ? (
            <ul className="mat-list" role="listbox" aria-label="Imported directivities">
              {imports.map(item)}
            </ul>
          ) : (
            <div className="small muted">None yet: import a file below.</div>
          )}
          {odeon.length > 0 && (
            <>
              <div className="section-title" title="Read in place from the local Odeon archives: research licence, never copied or redistributed">
                Odeon (local licence)
              </div>
              <ul className="mat-list" role="listbox" aria-label="Odeon directivities">
                {odeon.map(item)}
              </ul>
            </>
          )}
          <div className="dir-import">
            <div className="section-title">Import</div>
            <div className="row gap">
              <button className="btn btn-xs" onClick={() => fileRef.current?.click()} disabled={busy}>
                Choose file…
              </button>
              <span className="small ellipsis grow" title={file?.name}>
                {file ? file.name : <span className="muted">.json, .csv, .sofa or .so8</span>}
              </span>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".json,.csv,.sofa,.so8"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                e.target.value = '';
                setFile(f);
                if (f && !name) setName(f.name.replace(/\.[^.]+$/, ''));
              }}
            />
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name (default: the file name)" aria-label="Directivity name" />
            <button className="btn btn-sm btn-primary" disabled={!file || busy} onClick={() => void doImport()}>
              {busy ? <Spinner size={11} /> : null} Import
            </button>
            <div className="small muted">
              <b>.json</b>: the engine table {'{name, table: [7][S] energy vs θ 0–180°, on_axis_spl_1m_db}'}. <b>.csv</b>: header <code>theta_deg,125,250,500,1000,2000,4000,8000</code>, levels in dB re the axis, one row per angle. <b>.sofa</b>: AES69 FreeFieldDirectivityTF. The CLF .CF2 binary is not read (no public specification).
            </div>
          </div>
        </aside>
        <section className="dir-detail">
          {error && <ErrorBox message={error} />}
          {imported && (
            <div className="notice notice-info small">
              Imported <b>{imported.name}</b>
              {imported.method ? ` (${imported.method})` : ''}.{imported.notes?.length ? ` ${imported.notes.join(' · ')}` : ''}
              {imported.asymmetry_db !== null && imported.asymmetry_db !== undefined ? ` Asymmetry about the axis: ${fmtNum(imported.asymmetry_db, 1)} dB.` : ''}
            </div>
          )}
          {!sel ? (
            <div className="muted small">No directivity.</div>
          ) : (
            <>
              <div className="row gap">
                <h3 className="grow">{sel.name}</h3>
                <span className="badge">{sel.kind === 'imported' ? 'imported' : sel.kind === 'odeon_local' ? 'Odeon (local)' : 'preset'}</span>
                {sel.kind === 'imported' && (
                  <button className="btn btn-xs btn-danger-ghost" onClick={() => void remove(sel)}>
                    Delete…
                  </button>
                )}
              </div>
              <div className="band-picks" role="group" aria-label="Octave bands shown">
                {BANDS_HZ.map((f, i) => (
                  <label key={f} className={`chip-btn ${bands.includes(i) ? 'active' : ''}`}>
                    <input type="checkbox" checked={bands.includes(i)} onChange={(e) => setBands((b) => (e.target.checked ? [...b, i].sort((x, y) => x - y) : b.filter((x) => x !== i)))} />
                    {bandLabel(f)}
                  </label>
                ))}
              </div>
              {bands.length ? <PolarChart d={sel} bands={bands} /> : <div className="small muted pad">Select at least one band.</div>}
              <div className="small muted">
                Energy vs the angle from the source&apos;s front axis (0° = front), dB re the axis, floor {FLOOR_DB} dB. The engine&apos;s model is axisymmetric about that axis: the curve is mirrored to a full turn.
              </div>
              <table className="table compact">
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
                    <td title="Directivity index: on-axis level re the level averaged over all directions">DI (dB)</td>
                    {BANDS_HZ.map((_, i) => (
                      <td key={i} className="num">
                        {fmtNum(sel.di_db?.[i], 1)}
                      </td>
                    ))}
                  </tr>
                  {sel.on_axis_spl_1m_db && (
                    <tr>
                      <td title="On-axis SPL at 1 m from the imported data: it sets the source level used for SPL">SPL 1 m (dB)</td>
                      {BANDS_HZ.map((_, i) => (
                        <td key={i} className="num">
                          {fmtNum(sel.on_axis_spl_1m_db?.[i], 1)}
                        </td>
                      ))}
                    </tr>
                  )}
                </tbody>
              </table>
              {(sel.kind === 'imported' || sel.kind === 'odeon_local') && (
                <dl className="kv small">
                  {sel.license && (
                    <>
                      <dt>Licence</dt>
                      <dd className="text-warn">{sel.license}</dd>
                    </>
                  )}
                  <dt>File</dt>
                  <dd>{sel.source ?? '–'}</dd>
                  <dt>Method</dt>
                  <dd>{sel.method ?? '–'}</dd>
                  <dt>Asymmetry</dt>
                  <dd className={(sel.asymmetry_db ?? 0) > 1 ? 'text-warn' : undefined}>
                    {sel.asymmetry_db !== null && sel.asymmetry_db !== undefined ? `${fmtNum(sel.asymmetry_db, 1)} dB between directions at the same angle (the engine uses their mean)` : '–'}
                  </dd>
                  <dt>Level</dt>
                  <dd>{sel.on_axis_spl_1m_db ? 'on-axis SPL at 1 m from the file (sets the source level for SPL)' : 'not in the file: the source keeps 94 dB at 1 m plus its gain'}</dd>
                  {sel.notes && sel.notes.length > 0 && (
                    <>
                      <dt>Notes</dt>
                      <dd>
                        {sel.notes.map((n, i) => (
                          <div key={i}>{n}</div>
                        ))}
                      </dd>
                    </>
                  )}
                </dl>
              )}
            </>
          )}
        </section>
      </div>
      <div className="modal-footer">
        <span className="small muted grow">A source&apos;s directivity is chosen in the editor (Sources tab).</span>
        <button className="btn" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}
