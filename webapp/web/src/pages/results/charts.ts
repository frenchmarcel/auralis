import type { Dash, Data, Layout } from 'plotly.js';
import { BANDS_HZ, type Curves } from '../../api';
import { seriesColor } from '../../lib/colors';
import type { DataTable } from '../../lib/export';
import { bandLabel, fmtHz } from '../../lib/format';
import { hasBand63, isoLowMean, meanDbCurves, STI_BANDS, stiRating, type ParamInfo } from '../../lib/metrics';
import { curveKey } from '../../lib/simdata';
import type { ResolvedCard } from './cards';

export type PlotType = 'edc' | 'ir' | 'fr';
export type Channel = 'L' | 'R' | 'LR';

const TRANSITION_COLOR = '#f0a83a';
const DASHES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot', '5px,3px,1px,3px'] as Dash[];
const BAND_X = BANDS_HZ.map((_, i) => i);
const BAND_TEXT = BANDS_HZ.map((f) => bandLabel(f));

/** x position of the 63 Hz octave (one octave below 125 Hz on the band axis). */
const X63 = -1;
const NOTE63 = '63 Hz: wave band only';

/** Whether a 63 Hz value (wave band only) is shown for this parameter: some drawn card has one. */
export function show63(cards: ResolvedCard[], p: ParamInfo): boolean {
  if (p.scalar || !hasBand63(p.key)) return false;
  const k = p.key;
  return drawable(cards).some((c) => {
    const v = c.values!.band63?.[k];
    return v !== null && v !== undefined && Number.isFinite(v);
  });
}

/** Cards that can be drawn (results and at least one pair). */
export function drawable(cards: ResolvedCard[]): ResolvedCard[] {
  return cards.filter((c) => c.values && c.results);
}

/** Octave bands of a wave-based result that lie below the transition (upper edge f·√2 at most). */
export function waveBandOk(bandHz: number, transitionHz: number | null | undefined): boolean {
  return !!transitionHz && bandHz * Math.SQRT2 <= transitionHz + 1e-6;
}

function transitions(cards: ResolvedCard[]): { hz: number; color: string }[] {
  const out: { hz: number; color: string }[] = [];
  const many = cards.length > 1;
  for (const c of cards) {
    const t = c.results?.transition_hz ?? null;
    if (!t || out.some((o) => Math.abs(o.hz - t) < 0.5)) continue;
    out.push({ hz: t, color: many ? c.color : TRANSITION_COLOR });
  }
  return out;
}

export function paramTraces(cards: ResolvedCard[], p: ParamInfo): Data[] {
  const list = drawable(cards);
  if (p.scalar) {
    return [
      {
        type: 'bar',
        x: list.map((c) => c.label.replace(/ · /g, '<br>')),
        y: list.map((c) => c.values!.scalars.STI ?? null),
        marker: { color: list.map((c) => c.color), line: { color: '#111417', width: 1 } },
        text: list.map((c) => {
          const v = c.values!.scalars.STI;
          const r = stiRating(v);
          return v === null || v === undefined ? 'n/a' : `${v.toFixed(2)} ${r?.label ?? ''}`;
        }),
        textposition: 'outside',
        cliponaxis: false,
        hovertemplate: 'STI %{y:.3f}<extra></extra>',
        name: 'STI',
        showlegend: false,
      } as Data,
    ];
  }
  const key = p.key as Exclude<typeof p.key, 'STI'>;
  const with63 = show63(cards, p);
  const k63 = hasBand63(key) ? key : null;
  const out: Data[] = list.map(
    (c) =>
      ({
        type: 'bar',
        x: with63 ? [X63, ...BAND_X] : BAND_X,
        y: [...(with63 ? [k63 ? (c.values!.band63?.[k63] ?? null) : null] : []), ...(c.values!.bands[key] ?? BANDS_HZ.map(() => null))],
        name: c.label,
        marker: {
          color: c.color,
          line: { color: '#111417', width: 1 },
          // The 63 Hz bar is hatched: it comes from the wave band alone.
          ...(with63 ? { pattern: { shape: ['/', ...BAND_X.map(() => '')], fillmode: 'overlay', fgcolor: 'rgba(17,20,23,0.7)', size: 6, solidity: 0.35 } } : {}),
        },
        customdata: [...(with63 ? ['63 Hz (wave band only)'] : []), ...BAND_TEXT.map((t) => `${t} Hz`)],
        hovertemplate: `${c.label.replace(/%/g, '%%')}<br>%{customdata}: %{y:.${p.digits}f} ${p.unit}<extra></extra>`,
      }) as Data,
  );
  if (p.isoLowMean) {
    for (const c of list) {
      const iso = isoLowMean(c.values!.bands[key]);
      if (iso === null) continue;
      out.push({
        type: 'scatter',
        mode: 'lines',
        x: [-0.45, 3.45],
        y: [iso, iso],
        line: { color: c.color, width: 2, dash: 'dot' },
        name: `${c.label} · ISO`,
        showlegend: false,
        meta: 'decor',
        hovertemplate: `${c.label.replace(/%/g, '%%')}<br>ISO single number (125 Hz–1 kHz): ${iso.toFixed(p.digits)} ${p.unit}<extra></extra>`,
      } as Data);
    }
  }
  return out;
}

export function paramLayout(cards: ResolvedCard[], p: ParamInfo, opts: { height: number; showTransition: boolean; light?: boolean }): Partial<Layout> {
  const list = drawable(cards);
  const yTitle = `${p.axis}${p.unit ? ` (${p.unit})` : ''}`;
  if (p.scalar) {
    const shapes: Layout['shapes'] = STI_BANDS.map((b) => ({
      type: 'rect',
      xref: 'paper',
      yref: 'y',
      x0: 0,
      x1: 1,
      y0: b.min,
      y1: Math.min(1, b.max),
      fillcolor: b.color,
      opacity: opts.light ? 0.13 : 0.1,
      line: { width: 0 },
      layer: 'below',
    }));
    const annotations: Layout['annotations'] = STI_BANDS.map((b) => ({
      xref: 'paper',
      yref: 'y',
      x: 1,
      y: (b.min + Math.min(1, b.max)) / 2,
      text: b.label,
      showarrow: false,
      xanchor: 'right',
      font: { size: 10, color: b.color },
    }));
    return {
      height: opts.height,
      margin: { l: 52, r: 12, t: 28, b: 40 },
      xaxis: { type: 'category', automargin: true, tickangle: 0, tickfont: { size: list.length > 3 ? 9.5 : 10.5 } },
      yaxis: { title: { text: 'STI' }, range: [0, 1], dtick: 0.15, tick0: 0 },
      shapes,
      annotations,
      showlegend: false,
      bargap: list.length > 3 ? 0.35 : 0.6,
    };
  }
  const shapes: Layout['shapes'] = [];
  const annotations: Layout['annotations'] = [];
  if (opts.showTransition) {
    for (const t of transitions(list)) {
      const xpos = Math.log2(t.hz / 125);
      if (xpos <= -0.5 || xpos >= 6.5) continue;
      shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: xpos, x1: xpos, y0: 0, y1: 1, line: { color: t.color, width: 1.5, dash: 'dash' } });
      annotations.push({ x: xpos, y: 1, xref: 'x', yref: 'paper', text: `transition ${fmtHz(t.hz)}`, showarrow: false, yanchor: 'bottom', font: { size: 10, color: t.color } });
    }
  }
  const with63 = show63(cards, p);
  const xTitle = ['Octave band (Hz)', ...(p.isoLowMean ? ['dotted: ISO single number (mean 125 Hz–1 kHz)'] : []), ...(with63 ? [`hatched ${NOTE63}`] : [])].join(' · ');
  return {
    height: opts.height,
    barmode: 'group',
    margin: { l: 52, r: 12, t: legendTop(list.length), b: 40 },
    xaxis: {
      tickvals: with63 ? [X63, ...BAND_X] : BAND_X,
      ticktext: with63 ? ['63*', ...BAND_TEXT] : BAND_TEXT,
      range: [with63 ? X63 - 0.6 : -0.6, 6.6],
      title: { text: xTitle },
    },
    yaxis: { title: { text: yTitle }, rangemode: p.signed ? 'normal' : 'tozero' },
    shapes,
    annotations,
    showlegend: list.length > 1,
  };
}

export function paramTable(cards: ResolvedCard[], p: ParamInfo): DataTable {
  const list = drawable(cards);
  const head = ['Comparison', 'Simulation', 'Source', 'Receiver'];
  if (p.scalar) {
    return {
      columns: [...head, 'STI', 'Rating', ...BANDS_HZ.map((f) => `MTI ${f} Hz`)],
      rows: list.map((c) => [
        c.label,
        c.sim?.name ?? c.card.simId,
        c.srcName,
        c.rcvLabel,
        c.values!.scalars.STI ?? null,
        stiRating(c.values!.scalars.STI)?.label ?? '',
        ...(c.values!.bands.MTI ?? BANDS_HZ.map(() => null)),
      ]),
    };
  }
  const key = p.key as Exclude<typeof p.key, 'STI'>;
  const unit = p.unit ? ` (${p.unit})` : '';
  const with63 = show63(cards, p);
  const k63 = hasBand63(key) ? key : null;
  return {
    columns: [...head, ...(with63 ? [`${p.label} 63 Hz, wave band only${unit}`] : []), ...BANDS_HZ.map((f) => `${p.label} ${f} Hz${unit}`), ...(p.isoLowMean ? [`${p.label} ISO 125-1k${unit}`] : [])],
    rows: list.map((c) => [
      c.label,
      c.sim?.name ?? c.card.simId,
      c.srcName,
      c.rcvLabel,
      ...(with63 ? [k63 ? (c.values!.band63?.[k63] ?? null) : null] : []),
      ...(c.values!.bands[key] ?? BANDS_HZ.map(() => null)),
      ...(p.isoLowMean ? [isoLowMean(c.values!.bands[key])] : []),
    ]),
  };
}

export function singleNumberTable(cards: ResolvedCard[]): DataTable {
  const list = drawable(cards);
  return {
    columns: ['Comparison', 'Simulation', 'Source', 'Receiver', 'STI', 'STI rating', 'JLF ISO (125-1k)', 'LJ ISO (125-1k) dB', 'SPL dB(Z)', 'SPL dB(A)', 'SPL dB(C)'],
    rows: list.map((c) => {
      const v = c.values!;
      return [
        c.label,
        c.sim?.name ?? c.card.simId,
        c.srcName,
        c.rcvLabel,
        v.scalars.STI ?? null,
        stiRating(v.scalars.STI)?.label ?? '',
        isoLowMean(v.bands.JLF),
        isoLowMean(v.bands.LJ_db),
        v.scalars.SPL_dbz ?? null,
        v.scalars.SPL_dba ?? null,
        v.scalars.SPL_dbc ?? null,
      ];
    }),
  };
}

export interface ResponseOpts {
  plotType: PlotType;
  edcBands: number[];
  irDb: boolean;
  channel: Channel;
  showTransition: boolean;
  /** curveKey(simId, src, rcv, part) → curves. */
  curves: Map<string, Curves>;
}

/** IRs and spectra of different simulations are put on the first card's scale (they are normalised per simulation). */
function scaleFactor(c: ResolvedCard, refScale: number): number {
  const s = c.results?.ir_scale;
  return s && Number.isFinite(s) && s > 0 ? refScale / s : 1;
}

export function responseTraces(cards: ResolvedCard[], o: ResponseOpts): Data[] {
  const list = drawable(cards);
  const out: Data[] = [];
  if (!list.length) return out;
  const refScale = list[0].results?.ir_scale || 1;
  const many = list.length > 1;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const c of list) {
    const got = c.pairs
      .map((p) => ({ rcv: p.receiver_id, cv: o.curves.get(curveKey(c.card.simId, p.source_id, p.receiver_id, c.part)) }))
      .filter((x): x is { rcv: string; cv: Curves } => !!x.cv);
    if (!got.length) continue;
    const esc = c.label.replace(/%/g, '%%');
    const partial = got.length < c.pairs.length ? ` (${got.length}/${c.pairs.length})` : '';
    if (o.plotType === 'edc') {
      o.edcBands.forEach((b, bi) => {
        // The wave band alone has no energy above the transition: no decay to show there.
        if (c.part === 'wave' && !waveBandOk(b, c.results?.transition_hz)) return;
        const per = got
          .map(({ cv }) => cv.edc[String(b)] ?? cv.edc[bandLabel(b)])
          .filter((e): e is { t: number[]; db: number[] } => !!e)
          .map((e) => ({ x: e.t, y: e.db }));
        const m = meanDbCurves(per);
        if (!m) return;
        out.push({
          type: 'scatter',
          mode: 'lines',
          x: m.x,
          y: m.y,
          name: `${many ? `${c.label} · ` : ''}${bandLabel(b)} Hz${partial}`,
          line: {
            color: many ? c.color : seriesColor(BANDS_HZ.indexOf(b as (typeof BANDS_HZ)[number])),
            width: 2,
            dash: many ? DASHES[bi % DASHES.length] : 'solid',
          },
          hovertemplate: `${esc} ${bandLabel(b)} Hz<br>%{x:.3f} s: %{y:.1f} dB<extra></extra>`,
        } as Data);
      });
    } else if (o.plotType === 'ir') {
      const k = scaleFactor(c, refScale);
      const conv = (v: number[]) => (o.irDb ? v.map((x) => Math.max(-120, 20 * Math.log10(Math.abs(x * k) + 1e-12))) : k === 1 ? v : v.map((x) => x * k));
      got.forEach(({ rcv, cv }, ri) => {
        const name = `${c.label}${got.length > 1 ? ` · ${c.rcvName(rcv)}` : ''}`;
        const dash = DASHES[ri % DASHES.length];
        // The geometrical and wave parts exist as omnidirectional IRs only.
        if (c.part !== 'hybrid') {
          out.push({ type: 'scatter', mode: 'lines', x: cv.ir.t, y: conv(cv.ir.omni ?? cv.ir.left), name: `${name} · omni`, line: { color: c.color, width: 1, dash } });
          return;
        }
        if (o.channel !== 'R') out.push({ type: 'scatter', mode: 'lines', x: cv.ir.t, y: conv(cv.ir.left), name: `${name} · L`, line: { color: c.color, width: 1, dash } });
        if (o.channel !== 'L')
          out.push({ type: 'scatter', mode: 'lines', x: cv.ir.t, y: conv(cv.ir.right), name: `${name} · R`, line: { color: c.color, width: 1, dash: o.channel === 'LR' ? 'dot' : dash }, opacity: o.channel === 'LR' ? 0.75 : 1 });
      });
    } else {
      const off = 20 * Math.log10(scaleFactor(c, refScale));
      const m = meanDbCurves(got.map(({ cv }) => ({ x: cv.fr.f, y: off ? cv.fr.db.map((v) => v + off) : cv.fr.db })));
      if (!m) continue;
      out.push({ type: 'scatter', mode: 'lines', x: m.x, y: m.y, name: `${c.label}${partial}`, line: { color: c.color, width: 1.6 }, hovertemplate: `${esc}<br>%{x:.0f} Hz: %{y:.1f} dB<extra></extra>` });
      for (const v of m.y) {
        if (Number.isFinite(v)) {
          yMin = Math.min(yMin, v);
          yMax = Math.max(yMax, v);
        }
      }
    }
  }
  if (o.plotType === 'fr' && o.showTransition && Number.isFinite(yMin)) {
    for (const t of transitions(list)) {
      out.push({ type: 'scatter', mode: 'lines', x: [t.hz, t.hz], y: [yMin, yMax], name: `transition ${fmtHz(t.hz)}`, line: { color: t.color, width: 1.5, dash: 'dash' }, hoverinfo: 'skip', meta: 'decor' } as Data);
    }
  }
  return out;
}

/** Room for a horizontal legend that may wrap (long comparison labels). */
export function legendTop(nSeries: number): number {
  return nSeries <= 1 ? 28 : Math.min(76, 28 + 14 * (nSeries - 1));
}

export function responseLayout(o: { plotType: PlotType; irDb: boolean; height: number; multiSim: boolean; nSeries: number }): Partial<Layout> {
  const margin = { l: 52, r: 12, t: legendTop(o.nSeries), b: 40 };
  if (o.plotType === 'edc') return { height: o.height, xaxis: { title: { text: 'Time (s)' } }, yaxis: { title: { text: 'Energy decay (dB)' } }, margin };
  if (o.plotType === 'ir') {
    const ref = o.multiSim ? ', scale of the first comparison' : '';
    return {
      height: o.height,
      xaxis: { title: { text: 'Time (s)' } },
      yaxis: { title: { text: o.irDb ? `Level (dB re. full scale${ref})` : `Amplitude${o.multiSim ? ' (common scale)' : ''}` }, range: o.irDb ? [-100, 0] : undefined },
      margin,
    };
  }
  return {
    height: o.height,
    xaxis: { type: 'log', title: { text: 'Frequency (Hz)' }, range: [Math.log10(20), Math.log10(20000)] },
    yaxis: { title: { text: `Magnitude (dB${o.multiSim ? ', common scale' : ''})` } },
    margin,
  };
}
