import { useMemo, useRef } from 'react';
import type { Data, Layout } from 'plotly.js';
import { BANDS_HZ, type Material, type TableKind } from '../../api';
import { Plot } from '../../components/Plot';
import { Segmented } from '../../components/ui';
import { COLORS } from '../../lib/colors';
import type { DataTable } from '../../lib/export';
import { bandLabel, fmtNum } from '../../lib/format';

export interface ComplexRow {
  f: number;
  re: number;
  im: number;
}

export interface ComplexParse {
  rows: ComplexRow[];
  /** Lines that were skipped or reordered. */
  notes: string[];
  /** What keeps the table from being sent (the server checks the same rules and has the last word). */
  errors: string[];
}

function num(t: string): number | null {
  const s = t.trim().toLowerCase().replace(',', '.');
  const m = /^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)(k)?(hz)?$/.exec(s);
  if (!m) return null;
  return Number(m[1]) * (m[2] ? 1000 : 1);
}

/** "frequency Re Im" per line; separators: spaces, tabs, ";" (then "," is a decimal mark) or ",". */
export function parseComplexTable(text: string, kind: TableKind): ComplexParse {
  const notes: string[] = [];
  const errors: string[] = [];
  const raw: ComplexRow[] = [];
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.replace(/#.*/, '').trim())
    .filter(Boolean);
  for (const line of lines) {
    const tokens = (/[;\t]/.test(line) ? line.split(/[;\t\s]+/) : /\s/.test(line) ? line.split(/\s+/).map((t) => t.replace(/,$/, '')) : line.split(',')).filter(Boolean);
    const n = tokens.map(num);
    if (n.some((v) => v === null)) {
      if (/[a-z]/i.test(line) && !/^\s*[-+\d.]/.test(line)) continue; // header line
      notes.push(`Ignored “${line}”: not numeric.`);
      continue;
    }
    if (n.length !== 3) {
      notes.push(`Ignored “${line}”: expected “frequency Re Im”.`);
      continue;
    }
    const [f, re, im] = n as number[];
    if (!(f > 0)) {
      notes.push(`Ignored “${line}”: the frequency must be positive.`);
      continue;
    }
    raw.push({ f, re, im });
  }
  const sorted = [...raw].sort((a, b) => a.f - b.f);
  if (sorted.some((r, i) => r !== raw[i])) notes.push('Rows sorted by frequency.');
  const rows = sorted.filter((r, i) => i === 0 || r.f !== sorted[i - 1].f);
  if (rows.length < sorted.length) notes.push(`${sorted.length - rows.length} duplicate frequenc${sorted.length - rows.length === 1 ? 'y' : 'ies'} dropped (first kept).`);
  if (rows.length < 3) errors.push('At least 3 frequencies are needed.');
  const bad =
    kind === 'z'
      ? rows.filter((r) => r.re <= 0).map((r) => `${fmtNum(r.f, 0)} Hz`)
      : rows.filter((r) => Math.hypot(r.re, r.im) >= 1).map((r) => `${fmtNum(r.f, 0)} Hz`);
  if (bad.length) errors.push(`Not passive (${kind === 'z' ? 'Re z ≤ 0' : '|R| ≥ 1'}) at ${bad.slice(0, 6).join(', ')}${bad.length > 6 ? ` and ${bad.length - 6} more` : ''}.`);
  return { rows, notes, errors };
}

/** Normal-incidence reflection factor of one row of the table. */
function reflection(r: ComplexRow, kind: TableKind): { re: number; im: number } {
  if (kind === 'r') return { re: r.re, im: r.im };
  // R = (z - 1) / (z + 1)
  const d = (r.re + 1) ** 2 + r.im ** 2;
  return { re: ((r.re - 1) * (r.re + 1) + r.im * r.im) / d, im: (2 * r.im) / d };
}

/** Placeholders: they show the format only (not data). */
const EXAMPLE: Record<TableKind, string> = {
  z: '# frequency (Hz)  Re(Z/ρc)  Im(Z/ρc)\n100  1.8  -6.5\n250  1.5  -2.4\n500  1.4  -0.9\n1000 1.3  -0.2\n2000 1.2   0.1\n4000 1.1   0.2\n8000 1.1   0.2',
  r: '# frequency (Hz)  Re(R)  Im(R)\n100  0.90  -0.20\n250  0.70  -0.35\n500  0.40  -0.30\n1000 0.15  -0.15\n2000 0.05  -0.05\n4000 0.05   0.00\n8000 0.05   0.00',
};

export function ComplexTableInput({
  kind,
  setKind,
  text,
  setText,
  parsed,
  created,
  disabled,
}: {
  kind: TableKind;
  setKind: (k: TableKind) => void;
  text: string;
  setText: (t: string) => void;
  parsed: ComplexParse;
  /** The material once created: its random-incidence octave α is the server's result. */
  created: Material | null;
  disabled: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const alpha0 = useMemo(
    () =>
      parsed.rows.map((r) => {
        const R = reflection(r, kind);
        return 1 - (R.re * R.re + R.im * R.im);
      }),
    [parsed, kind],
  );
  const data = useMemo<Data[]>(() => {
    const d: Data[] = [];
    if (created) {
      d.push({
        type: 'bar',
        x: [...BANDS_HZ],
        y: created.absorption,
        width: BANDS_HZ.map((f) => f * 0.45),
        name: 'Random incidence per octave (what the rays use)',
        marker: { color: COLORS.accent, line: { width: 0 } },
        hovertemplate: '%{x} Hz: α %{y:.3f}<extra>random incidence</extra>',
      } as Data);
    }
    if (parsed.rows.length) {
      d.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: parsed.rows.map((r) => r.f),
        y: alpha0,
        name: 'Normal incidence, 1 − |R|² of your table',
        line: { color: '#d8dde2', width: 1.5, dash: created ? 'dot' : 'solid' },
        marker: { size: 6, color: '#d8dde2' },
        hovertemplate: '%{x:.0f} Hz: α₀ %{y:.3f}<extra>your table</extra>',
      });
    }
    return d;
  }, [parsed, alpha0, created]);
  const layout = useMemo<Partial<Layout>>(
    () => ({
      height: 250,
      margin: { l: 40, r: 8, t: 60, b: 36 },
      xaxis: { type: 'log', range: [Math.log10(45), Math.log10(11000)], tickvals: [...BANDS_HZ], ticktext: BANDS_HZ.map(bandLabel), title: { text: 'Frequency (Hz)' } },
      yaxis: { title: { text: 'α' }, range: [0, 1.02], dtick: 0.2 },
      showlegend: true,
    }),
    [],
  );
  const table = (): DataTable => ({
    columns: ['frequency (Hz)', kind === 'z' ? 'Re z' : 'Re R', kind === 'z' ? 'Im z' : 'Im R', 'alpha normal incidence'],
    rows: parsed.rows.map((r, i) => [r.f, r.re, r.im, alpha0[i]]),
  });
  const span = parsed.rows.length ? `${parsed.rows.length} frequencies, ${fmtNum(parsed.rows[0].f, 0)}–${fmtNum(parsed.rows[parsed.rows.length - 1].f, 0)} Hz` : 'no row yet';

  return (
    <div className="new-mat-grid">
      <div>
        <div className="section-title">Complex normal-incidence table</div>
        <Segmented<TableKind>
          value={kind}
          onChange={setKind}
          disabled={disabled}
          options={[
            { value: 'z', label: 'Impedance Z / ρc', title: 'Surface impedance normalised by the characteristic impedance of air (specific, dimensionless)' },
            { value: 'r', label: 'Reflection factor R', title: 'Complex pressure reflection factor, normal incidence (|R| < 1)' },
          ]}
        />
        <textarea
          className="complex-text"
          rows={9}
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          placeholder={EXAMPLE[kind]}
          spellCheck={false}
          aria-label="Table: frequency, real part, imaginary part per line"
        />
        <div className="row gap wrap">
          <button className="btn btn-xs" disabled={disabled} onClick={() => fileRef.current?.click()}>
            Load .csv / .txt…
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.txt,text/plain,text/csv"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = '';
              if (f) void f.text().then(setText);
            }}
          />
          <span className="small muted">{span}</span>
        </div>
        <div className="muted small">
          One “frequency Re Im” line each (e.g. 1/3 octaves 50 Hz–10 kHz); spaces, tabs, “;” or “,” between columns. {kind === 'z' ? 'Specific impedance: divide Pa·s/m³ by ρc (≈ 413 Pa·s/m at 20 °C).' : 'R must be passive: |R| < 1.'}
        </div>
        {parsed.notes.map((n, i) => (
          <div key={`n${i}`} className="small muted">
            {n}
          </div>
        ))}
        {(text.trim() ? parsed.errors : []).map((n, i) => (
          <div key={`e${i}`} className="small text-err">
            {n}
          </div>
        ))}
      </div>
      <div>
        {data.length ? <Plot data={data} layout={layout} tools filename="complex_material_alpha" table={table} /> : <div className="muted small pad">Paste or type the table: its normal-incidence α appears here.</div>}
        {created ? (
          <>
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
                  {created.absorption.map((v, i) => (
                    <td key={i} className="num strong">
                      {fmtNum(v, 3)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
            <div className="small">
              Created. The rays use these random-incidence octave values (Paris integral over each band of your table); the wave band and the complex reflections use the table itself, phase included.
            </div>
          </>
        ) : (
          <div className="small muted">The random-incidence α per octave is computed by the server when the material is created (shown here, with an option to delete it).</div>
        )}
      </div>
    </div>
  );
}
