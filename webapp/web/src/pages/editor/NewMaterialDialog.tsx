import { useEffect, useMemo, useState } from 'react';
import { api, BANDS_HZ, errorMessage, type Material, type PorousBody, type PorousPreview, type TableKind, type WoolKind, type WoolSigma } from '../../api';
import { ErrorBox, Modal, NumField, Segmented, Spinner } from '../../components/ui';
import { ABS_MAX } from '../../lib/colors';
import { bandLabel, fmtNum } from '../../lib/format';
import { ComplexTableInput, parseComplexTable } from './ComplexTableInput';
import { AlphaChart } from './MaterialLibrary';
import { DEFAULT_GROUND, GroundBuilder } from './GroundBuilder';
import { DEFAULT_PERFORATED, PerforatedBuilder } from './PerforatedBuilder';

type Tab = 'alpha' | 'porous' | 'panel' | 'ground' | 'complex';
/** How the porous builder gets its flow resistivity: typed in, or from a mineral wool's type and density. */
type SigmaMode = 'sigma' | 'wool';

const WOOL_NAMES: Record<WoolKind, string> = { glass: 'Glass wool', rock: 'Rock wool' };

/** Flow resistivity to 3 significant digits. */
const fmtSigma = (v: number) => v.toLocaleString('en-US', { maximumSignificantDigits: 3 });

const A63_TITLE = '63 Hz, optional: the wave band follows it below 125 Hz (the rays start at 125 Hz). Empty: the wall holds its 125 Hz behaviour.';

function parseNumberToken(t: string): number | null {
  const s = t.trim().toLowerCase().replace(',', '.');
  const m = /^(-?\d+(?:\.\d+)?)(k)?(hz)?$/.exec(s);
  if (!m) return null;
  return Number(m[1]) * (m[2] ? 1000 : 1);
}

/**
 * Parse "freq value" lines (or one line of 7 values) into octave absorption values. With
 * `with63`, a 63 Hz line or a line of 8 values (63 Hz first) also gives `v63`.
 */
export function parseAlphaText(text: string, with63 = false): { values: (number | null)[]; v63: number | null; notes: string[] } {
  const values: (number | null)[] = BANDS_HZ.map(() => null);
  let v63: number | null = null;
  const notes: string[] = [];
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.replace(/#.*/, '').trim())
    .filter(Boolean);
  for (const line of lines) {
    // ";" or tab separated: commas are decimal marks. Whitespace separated: same,
    // trailing commas dropped. Otherwise comma separated with dot decimals.
    const tokens = /[;\t]/.test(line)
      ? line.split(/[;\t\s]+/)
      : /\s/.test(line)
        ? line.split(/\s+/).map((t) => t.replace(/,$/, ''))
        : line.split(',');
    tokens.splice(0, tokens.length, ...tokens.filter(Boolean));
    const nums = tokens.map(parseNumberToken);
    if (nums.some((n) => n === null)) {
      if (/[a-z]/i.test(line) && !/\d/.test(line)) continue; // header line
      notes.push(`Ignored line “${line}”: not numeric.`);
      continue;
    }
    const n = nums as number[];
    if (n.length === BANDS_HZ.length) {
      n.forEach((v, i) => (values[i] = v));
      continue;
    }
    if (with63 && n.length === BANDS_HZ.length + 1) {
      v63 = n[0];
      n.slice(1).forEach((v, i) => (values[i] = v));
      continue;
    }
    if (n.length !== 2) {
      notes.push(`Ignored line “${line}”: expected “frequency value”.`);
      continue;
    }
    const [f, v] = n;
    const idx = BANDS_HZ.findIndex((b) => Math.abs(f - b) / b < 0.04);
    if (idx < 0) {
      if (Math.abs(f - 63) < 3) {
        if (with63) v63 = v;
        else notes.push('63 Hz value ignored: the engine bands are 125 Hz – 8 kHz.');
      } else notes.push(`${f} Hz is not an octave centre (${with63 ? '63' : '125'} Hz – 8 kHz): ignored.`);
      continue;
    }
    values[idx] = v;
  }
  return { values, v63, notes };
}

export function NewMaterialDialog({
  categories,
  onClose,
  onCreated,
}: {
  categories: string[];
  onClose: () => void;
  onCreated: (m: Material) => void;
}) {
  const [tab, setTab] = useState<Tab>('alpha');
  const [name, setName] = useState('');
  const [category, setCategory] = useState('Custom');
  const [description, setDescription] = useState('');
  const [scattering, setScattering] = useState<number>(0.1);
  const [alpha, setAlpha] = useState<(number | null)[]>(BANDS_HZ.map(() => 0.1));
  /** Optional 63 Hz value: the wave band follows it, the rays start at 125 Hz. */
  const [a63, setA63] = useState<number | null>(null);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pasteNotes, setPasteNotes] = useState<string[]>([]);
  const [sigma, setSigma] = useState<number>(10000);
  const [thickness, setThickness] = useState<number>(50);
  const [airGap, setAirGap] = useState<number>(0);
  const [sigmaMode, setSigmaMode] = useState<SigmaMode>('sigma');
  const [wool, setWool] = useState<WoolKind>('glass');
  const [density, setDensity] = useState<number>(30);
  const [woolInfo, setWoolInfo] = useState<WoolSigma | null>(null);
  const [woolError, setWoolError] = useState<string | null>(null);
  const [porousResult, setPorousResult] = useState<PorousPreview | null>(null);
  const porousPreview = porousResult?.absorption ?? null;
  const [panel, setPanel] = useState(DEFAULT_PERFORATED);
  const [panelPreview, setPanelPreview] = useState<number[] | null>(null);
  const [groundBody, setGroundBody] = useState(DEFAULT_GROUND);
  const [groundPreview, setGroundPreview] = useState<number[] | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [complexKind, setComplexKind] = useState<TableKind>('z');
  const [complexText, setComplexText] = useState('');
  const complex = useMemo(() => parseComplexTable(complexText, complexKind), [complexText, complexKind]);
  /** A material created from a complex table, shown before the dialog closes (its α comes from the server). */
  const [created, setCreated] = useState<Material | null>(null);
  const close = () => (created ? onCreated(created) : onClose());

  /** Body of the porous preview and creation: σ, or the wool and its density. */
  const porousBody: PorousBody =
    sigmaMode === 'wool' ? { model: 'miki', wool, density_kg_m3: density, thickness_mm: thickness, air_gap_mm: airGap } : { model: 'miki', sigma, thickness_mm: thickness, air_gap_mm: airGap };

  useEffect(() => {
    if (tab !== 'porous' || sigmaMode !== 'wool') return;
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const r = await api.woolSigma({ wool, density_kg_m3: density });
        if (!cancelled) {
          setWoolInfo(r);
          setWoolError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setWoolInfo(null);
          setWoolError(errorMessage(e));
        }
      }
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [tab, sigmaMode, wool, density]);

  useEffect(() => {
    if (tab !== 'porous') return;
    let cancelled = false;
    setPreviewBusy(true);
    const t = window.setTimeout(async () => {
      try {
        const r = await api.porousPreview(porousBody);
        if (!cancelled) {
          setPorousResult(r);
          setPreviewError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setPreviewError(errorMessage(e));
          setPorousResult(null);
        }
      } finally {
        if (!cancelled) setPreviewBusy(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, sigmaMode, sigma, wool, density, thickness, airGap]);

  /** Per band, the spread of α over the wool's σ band (its two ends and the central value). */
  const alphaBand = useMemo(() => {
    const b = sigmaMode === 'wool' ? porousResult?.absorption_band : undefined;
    if (!b || !porousResult) return null;
    const a = porousResult.absorption;
    return { low: a.map((v, i) => Math.min(v, b[0][i], b[1][i])), high: a.map((v, i) => Math.max(v, b[0][i], b[1][i])) };
  }, [porousResult, sigmaMode]);

  const alphaComplete = alpha.every((v) => v !== null && Number.isFinite(v));
  const overMax = alpha.some((v) => v !== null && v > ABS_MAX) || (a63 !== null && a63 > ABS_MAX);
  const canCreate =
    !!name.trim() && !busy && (tab === 'alpha' ? alphaComplete : tab === 'porous' ? !!porousPreview : tab === 'panel' ? !!panelPreview : tab === 'ground' ? !!groundPreview : !complex.errors.length);

  const create = async () => {
    setBusy(true);
    setError(null);
    try {
      const common = { name: name.trim(), category: category.trim() || 'Custom', description: description.trim(), scattering };
      if (tab === 'complex') {
        setCreated(
          await api.createTableMaterial({ ...common, kind: complexKind, freqs: complex.rows.map((r) => r.f), re: complex.rows.map((r) => r.re), im: complex.rows.map((r) => r.im) }),
        );
        return;
      }
      const m =
        tab === 'alpha'
          ? await api.createMaterial({ ...common, absorption: alpha.map((v) => v ?? 0), absorption_63: a63 })
          : tab === 'panel'
            ? await api.createPerforatedMaterial({ ...common, ...panel })
            : tab === 'ground'
              ? await api.createGroundMaterial({ ...common, ...groundBody })
            : await api.createPorousMaterial({ ...common, ...porousBody });
      onCreated(m);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const datalist = useMemo(() => Array.from(new Set(['Custom', ...categories])).sort(), [categories]);

  const discard = async () => {
    if (!created) return;
    setBusy(true);
    try {
      await api.deleteMaterial(created.id);
      setCreated(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="New material" onClose={close} width={760}>
      <div className="modal-body">
        <div className="form-grid">
          <label>Name</label>
          <input autoFocus value={name} disabled={!!created} onChange={(e) => setName(e.target.value)} placeholder="e.g. Mineral wool 50 mm on wall" />
          <label>Category</label>
          <input list="mat-categories" value={category} onChange={(e) => setCategory(e.target.value)} />
          <label>Default scattering</label>
          <NumField value={scattering} min={0} max={1} step={0.05} digits={2} onChange={(v) => setScattering(v ?? 0)} width={80} />
          <label>Description</label>
          <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Construction and mounting (optional)" />
        </div>
        <datalist id="mat-categories">
          {datalist.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
        <div className="row gap tab-switch">
          <Segmented<Tab>
            value={tab}
            onChange={setTab}
            size="md"
            disabled={!!created}
            options={[
              { value: 'alpha', label: 'From octave absorption' },
              { value: 'porous', label: 'Porous builder' },
              { value: 'panel', label: 'Perforated / slotted panel', title: 'Perforated, slotted or micro-perforated sheet over a plenum, with optional fleece and wool' },
              { value: 'ground', label: 'Outdoor ground', title: 'A Nord2000 ground class or a flow resistivity, optional porosity rate and roughness' },
              { value: 'complex', label: 'From Z or R (complex)', title: 'Measured complex normal-incidence impedance or reflection factor: the phase is kept for the wave band' },
            ]}
          />
        </div>
        {tab === 'panel' ? (
          <PerforatedBuilder value={panel} onChange={setPanel} onPreview={setPanelPreview} />
        ) : tab === 'ground' ? (
          <GroundBuilder value={groundBody} onChange={setGroundBody} onPreview={setGroundPreview} />
        ) : tab === 'complex' ? (
          <ComplexTableInput kind={complexKind} setKind={setComplexKind} text={complexText} setText={setComplexText} parsed={complex} created={created} disabled={!!created || busy} />
        ) : tab === 'alpha' ? (
          <div className="new-mat-grid">
            <div>
              <div className="section-title">
                Random-incidence absorption <span className="nocase">α</span>
              </div>
              <table className="table compact alpha-input">
                <thead>
                  <tr>
                    <th className="num band63" title={A63_TITLE}>
                      63*
                    </th>
                    {BANDS_HZ.map((f) => (
                      <th key={f} className="num">
                        {bandLabel(f)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="band63" title={A63_TITLE}>
                      <NumField value={a63} allowNull min={0} max={1} step={0.01} digits={3} className={a63 !== null && a63 > ABS_MAX ? 'over' : ''} ariaLabel="α at 63 Hz (optional)" onChange={setA63} />
                    </td>
                    {alpha.map((v, i) => (
                      <td key={i}>
                        <NumField
                          value={v}
                          allowNull
                          min={0}
                          max={1}
                          step={0.01}
                          digits={3}
                          className={v !== null && v > ABS_MAX ? 'over' : ''}
                          ariaLabel={`α at ${BANDS_HZ[i]} Hz`}
                          onChange={(nv) => setAlpha((a) => a.map((x, j) => (j === i ? nv : x)))}
                        />
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
              {overMax && <div className="warn-box small">Values above 0.95 will be clipped to 0.95 by the server (physical maximum for random incidence).</div>}
              {!alphaComplete && <div className="muted small">Fill all seven bands from 125 Hz (63 Hz is optional).</div>}
              <button className="btn btn-xs" onClick={() => setPasteOpen((o) => !o)}>
                {pasteOpen ? 'Hide text input' : 'Paste as text…'}
              </button>
              {pasteOpen && (
                <div className="paste-box">
                  <textarea
                    rows={7}
                    value={pasteText}
                    onChange={(e) => setPasteText(e.target.value)}
                    placeholder={'63 0.03\n125 0.05\n250 0.12\n500 0.35\n1000 0.70\n2000 0.85\n4000 0.90\n8000 0.90'}
                    spellCheck={false}
                  />
                  <div className="row gap">
                    <button
                      className="btn btn-xs"
                      onClick={() => {
                        const r = parseAlphaText(pasteText, true);
                        setAlpha((a) => a.map((x, i) => r.values[i] ?? x));
                        if (r.v63 !== null) setA63(r.v63);
                        const missing = r.values.filter((v) => v === null).length;
                        setPasteNotes([...r.notes, ...(missing ? [`${missing} band(s) not found in the text: previous values kept.`] : [])]);
                      }}
                    >
                      Apply
                    </button>
                    <span className="muted small">One “frequency value” pair per line (63 … 8000, “1k” accepted, 63 Hz optional), or one line of 7 values (125 Hz – 8 kHz) or 8 (63 Hz first).</span>
                  </div>
                  {pasteNotes.map((n, i) => (
                    <div key={i} className="small text-warn">
                      {n}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div>
              <AlphaChart values={alpha.map((v) => (v === null ? null : Math.min(v, ABS_MAX)))} compare={overMax ? alpha : null} compareLabel="Entered" label="Stored" height={190} />
            </div>
          </div>
        ) : (
          <div className="new-mat-grid">
            <div>
              <div className="section-title">Porous layer on a rigid backing</div>
              <div className="form-grid">
                <label>Model</label>
                <select value="miki" disabled>
                  <option value="miki">Miki</option>
                </select>
                <label>Flow resistivity</label>
                <Segmented<SigmaMode>
                  value={sigmaMode}
                  onChange={setSigmaMode}
                  options={[
                    { value: 'sigma', label: 'σ', title: 'Type the flow resistivity σ (Pa·s/m²), e.g. from a datasheet' },
                    { value: 'wool', label: 'Mineral wool: type + density', title: 'Glass or rock wool: σ from its bulk density, with the uncertainty of the law' },
                  ]}
                />
                {sigmaMode === 'sigma' ? (
                  <>
                    <label>Flow resistivity σ</label>
                    <div className="row gap">
                      <NumField value={sigma} min={100} max={1e7} step={1000} digits={0} onChange={(v) => setSigma(v ?? 10000)} width={110} />
                      <span className="muted small">Pa·s/m²</span>
                    </div>
                  </>
                ) : (
                  <>
                    <label>Wool</label>
                    <Segmented<WoolKind>
                      value={wool}
                      onChange={setWool}
                      options={[
                        { value: 'glass', label: WOOL_NAMES.glass },
                        { value: 'rock', label: WOOL_NAMES.rock },
                      ]}
                    />
                    <label>Bulk density</label>
                    <div className="row gap">
                      <NumField value={density} min={5} max={250} step={5} digits={1} onChange={(v) => setDensity(v ?? 30)} width={110} ariaLabel="Bulk density (kg/m³)" />
                      <span className="muted small">kg/m³</span>
                    </div>
                    <label>σ from the density</label>
                    <div className="small wool-sigma" aria-live="polite">
                      {woolInfo ? (
                        <>
                          <b>{fmtSigma(woolInfo.sigma)}</b> Pa·s/m²{' '}
                          <span className="muted" title="1-sigma band: scatter of the law against the data it was checked on">
                            (1σ: {fmtSigma(woolInfo.sigma_low)} – {fmtSigma(woolInfo.sigma_high)}, ±{fmtNum(woolInfo.uncertainty_pct, 0)} %)
                          </span>
                        </>
                      ) : woolError ? (
                        <span className="text-err">{woolError}</span>
                      ) : (
                        <Spinner />
                      )}
                    </div>
                  </>
                )}
                <label>Thickness</label>
                <div className="row gap">
                  <NumField value={thickness} min={1} max={1000} step={5} digits={1} onChange={(v) => setThickness(v ?? 50)} width={110} />
                  <span className="muted small">mm</span>
                </div>
                <label>Air gap behind</label>
                <div className="row gap">
                  <NumField value={airGap} min={0} max={2000} step={10} digits={1} onChange={(v) => setAirGap(v ?? 0)} width={110} />
                  <span className="muted small">mm</span>
                </div>
              </div>
              {sigmaMode === 'wool' && woolInfo && (
                <div className={`wool-note small ${woolInfo.extrapolated ? 'warn-box' : ''}`}>
                  {woolInfo.extrapolated && (
                    <div>
                      <b>Extrapolated:</b> {fmtNum(woolInfo.density_kg_m3, woolInfo.density_kg_m3 % 1 ? 1 : 0)} kg/m³ is outside the {fmtNum(woolInfo.range_kg_m3[0], 0)}–{fmtNum(woolInfo.range_kg_m3[1], 0)} kg/m³ of the data this law
                      was checked on. The σ band then only bounds the error from below.
                    </div>
                  )}
                  <div>
                    <span className="muted">Law:</span> {woolInfo.law}
                    {!woolInfo.extrapolated && (
                      <span className="muted">
                        {' '}
                        (checked on {fmtNum(woolInfo.range_kg_m3[0], 0)}–{fmtNum(woolInfo.range_kg_m3[1], 0)} kg/m³)
                      </span>
                    )}
                  </div>
                  <div>
                    <span className="muted">Data:</span> {woolInfo.source}
                  </div>
                </div>
              )}
              <div className="stack-diagram" aria-hidden="true">
                <span className="stack-porous" style={{ flex: Math.max(thickness, 1) }}>
                  {sigmaMode === 'wool' ? WOOL_NAMES[wool].toLowerCase() : 'porous'} {fmtNum(thickness, 0)} mm
                </span>
                {airGap > 0 && (
                  <span className="stack-air" style={{ flex: airGap }}>
                    air {fmtNum(airGap, 0)} mm
                  </span>
                )}
                <span className="stack-wall">rigid</span>
              </div>
              <p className="muted small">The physical model is kept with the material and used by the wave band; α is its random-incidence result.</p>
            </div>
            <div>
              {previewError ? (
                <ErrorBox message={previewError} />
              ) : porousPreview ? (
                <>
                  <AlphaChart values={porousPreview} height={190} band={alphaBand} />
                  {alphaBand && <div className="muted small">Whiskers: α across the 1σ band of σ (the scatter of the density law against its data).</div>}
                  {porousPreview.some((v) => v > ABS_MAX) && (
                    <div className="warn-box small">
                      Above 0.95 in {porousPreview.filter((v) => v > ABS_MAX).length} band(s): the stored random-incidence values may be clipped to 0.95.
                    </div>
                  )}
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
                        {porousPreview.map((v, i) => (
                          <td key={i} className="num">
                            {fmtNum(v, 2)}
                          </td>
                        ))}
                      </tr>
                      {alphaBand && (
                        <tr className="muted" title="α at both ends of the σ band">
                          {alphaBand.low.map((l, i) => (
                            <td key={i} className="num small">
                              {fmtNum(l, 2)}–{fmtNum(alphaBand.high[i], 2)}
                            </td>
                          ))}
                        </tr>
                      )}
                    </tbody>
                  </table>
                </>
              ) : (
                <div className="loading-row">
                  <Spinner /> Computing…
                </div>
              )}
              {previewBusy && porousPreview && <div className="muted small">Updating…</div>}
            </div>
          </div>
        )}
        {error && <ErrorBox message={error} />}
      </div>
      <div className="modal-footer">
        {created ? (
          <>
            <button className="btn btn-danger-ghost" disabled={busy} onClick={() => void discard()} title="Delete this material and edit the table again">
              Delete and edit
            </button>
            <button className="btn btn-primary" disabled={busy} onClick={() => onCreated(created)}>
              Done
            </button>
          </>
        ) : (
          <>
            <button className="btn" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" disabled={!canCreate} onClick={create}>
              {busy ? <Spinner /> : null} Create material
            </button>
          </>
        )}
      </div>
    </Modal>
  );
}
