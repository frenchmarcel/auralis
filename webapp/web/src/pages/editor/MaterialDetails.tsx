import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import type { Data, Layout } from 'plotly.js';
import { api, errorMessage, type Material, type MaterialDetails, type MaterialLibrary } from '../../api';
import { Plot } from '../../components/Plot';
import { ErrorBox, Modal, Spinner } from '../../components/ui';
import { COLORS } from '../../lib/colors';
import type { DataTable } from '../../lib/export';
import { bandLabel, fmtNum } from '../../lib/format';
import { describePhysical, isLocalOnly } from './MaterialLibrary';

const FIT_COLOR = '#d8dde2';
const BAND63 = 63;
const NORMAL_COLOR = '#d95926';
const LOG_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000];
const LOG_X = { type: 'log' as const, range: [Math.log10(45), Math.log10(11000)], tickvals: LOG_TICKS, ticktext: LOG_TICKS.map(bandLabel), title: { text: 'Frequency (Hz)' } };

const detailsCache = new Map<string, Promise<MaterialDetails>>();

function loadDetails(id: string): Promise<MaterialDetails> {
  let p = detailsCache.get(id);
  if (!p) {
    p = api.getMaterialDetails(id);
    detailsCache.set(id, p);
    p.catch(() => detailsCache.delete(id));
  }
  return p;
}

/** Whether the server says the reflection phase is known (porous model, measured table). */
export function phaseKnown(source: string): boolean {
  return /phase known/i.test(source) && !/assumption/i.test(source);
}

/**
 * What the engine uses for a material: random-incidence α of the table vs the admittance
 * fit, and the normal-incidence reflection factor (magnitude and phase) 50 Hz–10 kHz.
 */
export function MaterialDetailsView({ material, compact = false }: { material: Material; compact?: boolean }) {
  const [d, setD] = useState<MaterialDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setD(null);
    setError(null);
    loadDetails(material.id)
      .then((x) => !cancelled && setD(x))
      .catch((e) => !cancelled && setError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [material.id]);

  const h = compact ? 230 : 270;
  const alphaData = useMemo<Data[]>(() => {
    if (!d) return [];
    const has63 = d.alpha_table_63 !== null;
    return [
      {
        type: 'bar',
        x: d.bands_hz,
        y: d.alpha_table,
        name: 'Table (what the rays use)',
        width: d.bands_hz.map((f) => f * 0.45),
        marker: { color: COLORS.accent, line: { width: 0 } },
        hovertemplate: '%{x} Hz: α %{y:.3f}<extra>table</extra>',
      } as Data,
      ...(has63
        ? [
            {
              type: 'bar',
              x: [BAND63],
              y: [d.alpha_table_63],
              name: 'Table, 63 Hz (wave band only)',
              width: [BAND63 * 0.45],
              marker: { color: COLORS.accent, line: { width: 0 }, pattern: { shape: '/', fillmode: 'overlay', fgcolor: 'rgba(17,20,23,0.7)', size: 6, solidity: 0.35 } },
              hovertemplate: '63 Hz: α %{y:.3f}<extra>table, wave band only</extra>',
            } as Data,
          ]
        : []),
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: [BAND63, ...d.bands_hz],
        y: [d.alpha_fit_63, ...d.alpha_fit_random],
        name: 'Admittance fit, before per-band anchoring',
        line: { color: FIT_COLOR, width: 2, dash: 'dot' },
        marker: { size: 8, color: FIT_COLOR },
        hovertemplate: '%{x} Hz: α %{y:.3f}<extra>fit</extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: d.freqs_hz,
        y: d.alpha_normal,
        name: 'Normal incidence, 1 − |R|² (fit)',
        line: { color: NORMAL_COLOR, width: 1.5, dash: 'dash' },
        hovertemplate: '%{x:.0f} Hz: α₀ %{y:.3f}<extra>normal incidence</extra>',
      },
    ];
  }, [d]);
  const alphaLayout = useMemo<Partial<Layout>>(
    () => ({
      height: h,
      margin: { l: 44, r: 10, t: 74, b: 38 },
      xaxis: { ...LOG_X, tickvals: d ? [BAND63, ...d.bands_hz] : [], ticktext: d ? ['63*', ...d.bands_hz.map(bandLabel)] : [] },
      yaxis: { title: { text: 'α' }, range: [0, 1.02], dtick: 0.2 },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.95, y1: 0.95, line: { color: 'rgba(240,168,58,0.5)', width: 1, dash: 'dot' } }],
      showlegend: true,
    }),
    [d, h],
  );
  const alphaTable = (): DataTable => ({
    columns: ['band (Hz)', 'alpha table (random incidence)', 'alpha fit (random incidence, before anchoring)'],
    rows: d ? [[BAND63, d.alpha_table_63 ?? '', d.alpha_fit_63], ...d.bands_hz.map((f, i) => [f, d.alpha_table[i], d.alpha_fit_random[i]])] : [],
  });
  const rTable = (): DataTable => ({
    columns: ['frequency (Hz)', 'Re R', 'Im R', '|R|', 'phase R (deg)', 'alpha normal incidence'],
    rows: (d?.freqs_hz ?? []).map((f, i) => [f, d!.R_normal.re[i], d!.R_normal.im[i], d!.R_normal.mag[i], d!.R_normal.phase_deg[i], d!.alpha_normal[i]]),
  });
  const magData = useMemo<Data[]>(
    () => (d ? [{ type: 'scatter', mode: 'lines', x: d.freqs_hz, y: d.R_normal.mag, name: '|R|', line: { color: COLORS.receiver, width: 2 }, hovertemplate: '%{x:.0f} Hz: |R| %{y:.4f}<extra></extra>' }] : []),
    [d],
  );
  const phaseData = useMemo<Data[]>(
    () => (d ? [{ type: 'scatter', mode: 'lines', x: d.freqs_hz, y: d.R_normal.phase_deg, name: 'phase', line: { color: COLORS.receiver, width: 2 }, hovertemplate: '%{x:.0f} Hz: %{y:.1f}°<extra></extra>' }] : []),
    [d],
  );
  const magLayout = useMemo<Partial<Layout>>(() => ({ height: h * 0.8, margin: { l: 44, r: 10, t: 28, b: 38 }, xaxis: LOG_X, yaxis: { title: { text: '|R|' }, range: [0, 1.02] }, showlegend: false }), [h]);
  const phaseLayout = useMemo<Partial<Layout>>(() => ({ height: h * 0.8, margin: { l: 44, r: 10, t: 28, b: 38 }, xaxis: LOG_X, yaxis: { title: { text: 'Phase of R (°)' }, range: [-180, 180], dtick: 90 }, showlegend: false }), [h]);

  if (error) return <ErrorBox message={`Details unavailable: ${error}`} />;
  if (!d)
    return (
      <div className="loading-row">
        <Spinner /> Loading what the engine uses…
      </div>
    );
  const known = phaseKnown(d.phase_source);
  const name = material.name.replace(/[^\w.-]+/g, '_');
  return (
    <div className="mat-details">
      <div className={`phase-source ${known ? 'known' : 'assumed'}`} role="note">
        <b>Phase of the reflection factor:</b> {d.phase_source}
        {!known && <div className="small">The complex reflections and the wave band use this phase; only α was measured.</div>}
        {d.wave_model && <div className="small">Wave band: {d.wave_model}</div>}
      </div>
      <div className={compact ? 'mat-details-grid one' : 'mat-details-grid'}>
        <section>
          <div className="section-title">
            Random-incidence <span className="nocase">α</span>
          </div>
          <Plot data={alphaData} layout={alphaLayout} tools filename={`${name}_alpha`} table={alphaTable} />
          <table className="table compact">
            <thead>
              <tr>
                <th />
                <th className="num band63" title="63 Hz octave: the wave band only (the rays start at 125 Hz)">
                  63*
                </th>
                {d.bands_hz.map((f) => (
                  <th key={f} className="num">
                    {bandLabel(f)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td title="The material's table: the rays and the complex reflections are anchored on it, band by band">Table</td>
                <td className="num band63" title={d.alpha_table_63 === null ? 'No 63 Hz value: below 125 Hz the wall holds its 125 Hz behaviour' : '63 Hz: followed by the wave band'}>
                  {d.alpha_table_63 === null ? '–' : fmtNum(d.alpha_table_63, 3)}
                </td>
                {d.alpha_table.map((v, i) => (
                  <td key={i} className="num">
                    {fmtNum(v, 3)}
                  </td>
                ))}
              </tr>
              <tr>
                <td title="Admittance fit before the per-band anchoring">Fit</td>
                <td className="num band63" title="What the wave band sees at 63 Hz (no anchoring below 125 Hz)">
                  {fmtNum(d.alpha_fit_63, 3)}
                </td>
                {d.alpha_fit_random.map((v, i) => (
                  <td key={i} className={`num ${Math.abs(v - d.alpha_table[i]) > 0.1 ? 'text-warn' : ''}`} title={Math.abs(v - d.alpha_table[i]) > 0.1 ? 'More than 0.1 from the table: the anchoring corrects it for the rays, the wave band sees the fit' : undefined}>
                    {fmtNum(v, 3)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </section>
        <section>
          <div className="section-title">Normal-incidence reflection factor R</div>
          <Plot data={magData} layout={magLayout} tools filename={`${name}_R_normal`} table={rTable} />
          <Plot data={phaseData} layout={phaseLayout} tools filename={`${name}_R_normal`} table={rTable} />
        </section>
      </div>
    </div>
  );
}

export function MaterialDetailsDialog({
  material,
  library,
  onClose,
  onDelete,
  onRename,
  onDuplicate,
}: {
  material: Material;
  library?: MaterialLibrary;
  onClose: () => void;
  onDelete?: () => void;
  onRename?: () => void;
  onDuplicate?: () => void;
}) {
  // A portal: the library drawer is its own stacking context, under the 3D labels.
  return createPortal(
    <Modal title={`Material details · ${material.name}`} onClose={onClose} width={960} className="mat-details-modal">
      <div className="modal-body">
        <dl className="kv small mat-details-head">
          <dt>Category</dt>
          <dd>{material.category || 'Uncategorised'}</dd>
          <dt>Library</dt>
          <dd>{library?.name ?? material.library}</dd>
          <dt>Default scattering</dt>
          <dd>{fmtNum(material.scattering, 2)}</dd>
          <dt>Wave model</dt>
          <dd>{material.physical ? describePhysical(material.physical) : 'none (fitted from α)'}</dd>
          {material.description && (
            <>
              <dt>Description</dt>
              <dd>{material.description}</dd>
            </>
          )}
          <dt>Source</dt>
          <dd>{material.source || '–'}</dd>
        </dl>
        {isLocalOnly(material, library) && <div className="lic-box small">Not redistributable: {material.license || library?.license}</div>}
        {material.clipped && <div className="warn-box small">At least one table value above 0.95 was clipped to 0.95 (physical maximum for random incidence).</div>}
        <MaterialDetailsView material={material} />
      </div>
      <div className="modal-footer">
        {onDelete && material.library === 'custom' && (
          <button className="btn btn-danger-ghost" onClick={onDelete}>
            Delete material…
          </button>
        )}
        {onRename && material.library === 'custom' && (
          <button className="btn" onClick={onRename} title="Change the name only: simulations refer to the material's id, which stays">
            Rename…
          </button>
        )}
        {onDuplicate && (
          <button className="btn" onClick={onDuplicate} title="Copy it into My materials (new id), keeping its source and licence">
            Duplicate…
          </button>
        )}
        <div className="grow" />
        <button className="btn" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>,
    document.body,
  );
}
