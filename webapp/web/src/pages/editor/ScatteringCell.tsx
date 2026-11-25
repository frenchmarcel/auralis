import type React from 'react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { BANDS_HZ } from '../../api';
import { NumField, Segmented } from '../../components/ui';
import { bandLabel, fmtNum } from '../../lib/format';
import { parseAlphaText } from './NewMaterialDialog';

export type Scattering = number | number[] | null;

/** Seven values 0-1 from pasted text: 7 numbers in any layout, or "frequency value" lines. */
export function parseSevenValues(text: string): { values: number[] | null; error: string | null } {
  const t = text.trim();
  if (!t) return { values: null, error: null };
  // ";" or tabs separate values and commas are decimal marks; otherwise commas separate too.
  const tokens = (/[;\t]/.test(t) ? t.split(/[;\t\s]+/).map((x) => x.replace(',', '.')) : /\s/.test(t) ? t.split(/[\s]+/).map((x) => x.replace(/,$/, '').replace(',', '.')) : t.split(',')).filter(Boolean);
  const nums = tokens.map(Number);
  let values: (number | null)[] | null = null;
  if (nums.length === 7 && nums.every(Number.isFinite)) values = nums;
  else {
    const r = parseAlphaText(text);
    if (r.values.every((v) => v !== null)) values = r.values;
  }
  if (!values) return { values: null, error: 'Expected 7 values (125 Hz to 8 kHz), or one “frequency value” pair per line.' };
  if (values.some((v) => v === null || v < 0 || v > 1)) return { values: null, error: 'Scattering values must be within 0–1.' };
  return { values: values as number[], error: null };
}

const mean = (v: number[]) => v.reduce((a, x) => a + x, 0) / v.length;

/** Scattering of one layer: one value (flat over the bands) or 7 values per octave, edited in a popover. */
export function ScatteringCell({ layer, value, materialDefault, disabled, onChange }: { layer: string; value: Scattering; materialDefault: number | null; disabled: boolean; onChange: (v: Scattering) => void }) {
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const perBand = Array.isArray(value);
  return (
    <div className="scat-cell">
      {perBand ? (
        <button
          type="button"
          className="scat-bands"
          disabled={disabled}
          onClick={() => setOpen(true)}
          title={`Per octave: ${BANDS_HZ.map((f, i) => `${bandLabel(f)} ${fmtNum(value[i], 2)}`).join(', ')}`}
          aria-label={`Scattering of ${layer}, per octave`}
        >
          <ScatSpark values={value} />
        </button>
      ) : (
        <NumField
          value={value}
          allowNull
          min={0}
          max={1}
          step={0.05}
          digits={2}
          placeholder={materialDefault !== null ? fmtNum(materialDefault, 2) : '–'}
          disabled={disabled}
          title={value === null ? `Material default (${materialDefault !== null ? fmtNum(materialDefault, 2) : 'n/a'}). Type a value to override.` : 'Override; clear to use the material default'}
          ariaLabel={`Scattering of ${layer}`}
          onChange={(v) => onChange(v)}
        />
      )}
      <button ref={btn} type="button" className="icon-btn scat-more" disabled={disabled} onClick={() => setOpen((o) => !o)} title="Scattering per octave band" aria-label={`Scattering per octave of ${layer}`} aria-expanded={open}>
        ≡
      </button>
      {open && <ScatteringPopover anchor={btn.current} layer={layer} value={value} materialDefault={materialDefault} onChange={onChange} onClose={() => setOpen(false)} />}
    </div>
  );
}

/** Position under (or above) the anchor, and close on an outside click or Escape. */
function usePopover(ref: React.RefObject<HTMLDivElement | null>, anchor: HTMLElement | null, onClose: () => void, layoutKey: unknown) {
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 });
  useLayoutEffect(() => {
    const r = anchor?.getBoundingClientRect();
    const w = 350;
    const h = ref.current?.offsetHeight ?? 220;
    if (!r) return;
    const left = Math.max(8, Math.min(window.innerWidth - w - 8, r.right - w));
    const below = r.bottom + 6;
    const top = below + h > window.innerHeight - 8 ? Math.max(8, r.top - h - 6) : below;
    setPos({ left, top });
  }, [anchor, layoutKey, ref]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && !anchor?.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopImmediatePropagation();
        onClose();
      }
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [anchor, onClose, ref]);
  return pos;
}

function ScatSpark({ values }: { values: number[] }) {
  return (
    <span className="scat-spark" aria-hidden="true">
      {values.map((v, i) => (
        <span key={i} style={{ height: `${Math.max(8, Math.min(1, v) * 100)}%` }} />
      ))}
    </span>
  );
}

function ScatteringPopover({
  anchor,
  layer,
  value,
  materialDefault,
  onChange,
  onClose,
}: {
  anchor: HTMLElement | null;
  layer: string;
  value: Scattering;
  materialDefault: number | null;
  onChange: (v: Scattering) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const start = value ?? materialDefault ?? 0.1;
  const [mode, setMode] = useState<'single' | 'bands'>(Array.isArray(value) ? 'bands' : 'single');
  const [bands, setBands] = useState<(number | null)[]>(Array.isArray(value) ? [...value] : BANDS_HZ.map(() => (typeof start === 'number' ? start : mean(start))));
  const [paste, setPaste] = useState('');
  const [pasteError, setPasteError] = useState<string | null>(null);
  const pos = usePopover(ref, anchor, onClose, mode);

  const commitBands = (b: (number | null)[]) => {
    setBands(b);
    if (b.every((v) => v !== null)) onChange(b as number[]);
  };
  const applyPaste = () => {
    const r = parseSevenValues(paste);
    if (r.values) {
      commitBands(r.values);
      setPaste('');
    } else setPasteError(r.error);
  };
  const toMode = (m: 'single' | 'bands') => {
    setMode(m);
    if (m === 'single') {
      const v = Array.isArray(value) ? Math.round(mean(value) * 100) / 100 : value;
      onChange(v);
    } else commitBands(bands);
  };

  return (
    <div ref={ref} className="scat-pop" style={{ left: pos.left, top: pos.top }} role="dialog" aria-label={`Scattering of ${layer}`} onClick={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
      <div className="row gap">
        <b className="ellipsis grow" title={layer}>
          Scattering · {layer}
        </b>
        <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
          ×
        </button>
      </div>
      <Segmented<'single' | 'bands'>
        value={mode}
        onChange={toMode}
        options={[
          { value: 'single', label: 'Single value', title: 'One value, used flat across the bands (not extrapolated into a curve)' },
          { value: 'bands', label: 'Per octave', title: 'One value per octave band, 125 Hz to 8 kHz' },
        ]}
      />
      {mode === 'single' ? (
        <div className="row gap">
          <NumField
            value={Array.isArray(value) ? null : value}
            allowNull
            min={0}
            max={1}
            step={0.05}
            digits={2}
            placeholder={materialDefault !== null ? fmtNum(materialDefault, 2) : '–'}
            onChange={(v) => onChange(v)}
            ariaLabel="Scattering (single value)"
            width={70}
          />
          <span className="small muted">{value === null ? `material default (${materialDefault !== null ? fmtNum(materialDefault, 2) : 'n/a'})` : 'flat across the bands'}</span>
        </div>
      ) : (
        <>
          <table className="table compact scat-table">
            <thead>
              <tr>
                {BANDS_HZ.map((f) => (
                  <th key={f} className="num">
                    {bandLabel(f)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {bands.map((v, i) => (
                  <td key={i}>
                    <NumField value={v} allowNull min={0} max={1} step={0.05} digits={2} ariaLabel={`Scattering at ${BANDS_HZ[i]} Hz`} onChange={(nv) => commitBands(bands.map((x, j) => (j === i ? nv : x)))} />
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
          {bands.some((v) => v === null) && <div className="small text-warn">Fill all seven bands: the layer keeps its previous value until then.</div>}
          <div className="row gap">
            <input
              className="grow"
              value={paste}
              onChange={(e) => {
                setPaste(e.target.value);
                setPasteError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && paste.trim()) applyPaste();
              }}
              placeholder="Paste 7 values: 0.05 0.1 0.2 0.3 0.4 0.5 0.6"
              aria-label="Paste 7 scattering values"
              spellCheck={false}
            />
            <button
              type="button"
              className="btn btn-xs"
              disabled={!paste.trim()}
              onClick={applyPaste}
            >
              Apply
            </button>
          </div>
          {pasteError && <div className="small text-err">{pasteError}</div>}
        </>
      )}
      <div className="row gap">
        <span className="small muted grow">Used by the geometrical solver only; the wave band scatters through the geometry.</span>
        <button
          type="button"
          className="btn btn-xs btn-ghost"
          disabled={value === null}
          onClick={() => {
            setMode('single');
            onChange(null);
          }}
          title="Use the material's default scattering"
        >
          Material default
        </button>
      </div>
    </div>
  );
}

/**
 * Scattering applied to several layers at once (one value, 7 octave values, or the
 * material default): `onApply` goes through the same save path as a per-layer edit.
 */
export function BulkScatteringPopover({
  anchor,
  count,
  initial,
  mixed,
  onApply,
  onClose,
}: {
  anchor: HTMLElement | null;
  count: number;
  /** The common value of the selected layers (null: all on their material default, or mixed). */
  initial: Scattering;
  /** The selected layers do not share one value. */
  mixed: boolean;
  onApply: (v: Scattering) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<'single' | 'bands'>(Array.isArray(initial) ? 'bands' : 'single');
  const [single, setSingle] = useState<number | null>(typeof initial === 'number' ? initial : null);
  const [bands, setBands] = useState<(number | null)[]>(Array.isArray(initial) ? [...initial] : BANDS_HZ.map(() => (typeof initial === 'number' ? initial : null)));
  const [paste, setPaste] = useState('');
  const [pasteError, setPasteError] = useState<string | null>(null);
  const pos = usePopover(ref, anchor, onClose, mode);
  const target = `${count} layer${count === 1 ? '' : 's'}`;
  const value: Scattering = mode === 'single' ? single : bands.every((v) => v !== null) ? (bands as number[]) : null;
  const apply = (v: Scattering) => {
    onApply(v);
    onClose();
  };
  const applyPaste = () => {
    const r = parseSevenValues(paste);
    if (r.values) {
      setBands(r.values);
      setPaste('');
    } else setPasteError(r.error);
  };
  return (
    <div ref={ref} className="scat-pop" style={{ left: pos.left, top: pos.top }} role="dialog" aria-label={`Scattering of ${target}`} onClick={(e) => e.stopPropagation()}>
      <div className="row gap">
        <b className="ellipsis grow">Scattering · {target}</b>
        <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
          ×
        </button>
      </div>
      {mixed && <div className="small muted">The selected layers have different values: the one applied replaces them all.</div>}
      <Segmented<'single' | 'bands'>
        value={mode}
        onChange={setMode}
        options={[
          { value: 'single', label: 'Single value', title: 'One value, used flat across the bands (not extrapolated into a curve)' },
          { value: 'bands', label: 'Per octave', title: 'One value per octave band, 125 Hz to 8 kHz' },
        ]}
      />
      {mode === 'single' ? (
        <div className="row gap">
          <NumField value={single} allowNull min={0} max={1} step={0.05} digits={2} placeholder="0–1" onChange={setSingle} ariaLabel="Scattering for the selected layers (single value)" width={70} />
          <span className="small muted">flat across the bands</span>
        </div>
      ) : (
        <>
          <table className="table compact scat-table">
            <thead>
              <tr>
                {BANDS_HZ.map((f) => (
                  <th key={f} className="num">
                    {bandLabel(f)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {bands.map((v, i) => (
                  <td key={i}>
                    <NumField value={v} allowNull min={0} max={1} step={0.05} digits={2} ariaLabel={`Scattering at ${BANDS_HZ[i]} Hz for the selected layers`} onChange={(nv) => setBands(bands.map((x, j) => (j === i ? nv : x)))} />
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
          <div className="row gap">
            <input
              className="grow"
              value={paste}
              onChange={(e) => {
                setPaste(e.target.value);
                setPasteError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && paste.trim()) applyPaste();
              }}
              placeholder="Paste 7 values: 0.05 0.1 0.2 0.3 0.4 0.5 0.6"
              aria-label="Paste 7 scattering values for the selected layers"
              spellCheck={false}
            />
            <button type="button" className="btn btn-xs" disabled={!paste.trim()} onClick={applyPaste}>
              Fill
            </button>
          </div>
          {pasteError && <div className="small text-err">{pasteError}</div>}
        </>
      )}
      <div className="row gap">
        <button type="button" className="btn btn-xs btn-ghost" onClick={() => apply(null)} title={`Clear the override of the ${target}: each uses its material's default scattering`}>
          Material default
        </button>
        <div className="grow" />
        <button type="button" className="btn btn-xs btn-primary" disabled={value === null} onClick={() => apply(value)} title={value === null ? (mode === 'single' ? 'Enter a value 0–1' : 'Fill all seven bands') : undefined}>
          Apply to {target}
        </button>
      </div>
      <div className="small muted">Used by the geometrical solver only; the wave band scatters through the geometry.</div>
    </div>
  );
}
