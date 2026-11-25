import { useEffect, useRef, useState } from 'react';
import type { Config, Data, Layout, PlotMouseEvent } from 'plotly.js';
import { useApp } from '../AppContext';
import { copyText, downloadText, safeFilename, tableFromTraces, toCsv, toTsv, type DataTable } from '../lib/export';
import { errorMessage } from '../api';

type PlotlyModule = typeof import('plotly.js-dist-min');

let plotlyPromise: Promise<PlotlyModule> | null = null;

/** Plotly is large: load it once, on first use, as its own chunk. */
function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min').then((m) => {
      const mod = m as unknown as { default?: PlotlyModule };
      return mod.default ?? (m as unknown as PlotlyModule);
    });
  }
  return plotlyPromise;
}

export type PlotTheme = 'dark' | 'light';

const THEMES = {
  dark: {
    grid: 'rgba(255,255,255,0.07)',
    zero: 'rgba(255,255,255,0.18)',
    line: 'rgba(255,255,255,0.18)',
    tick: '#9aa3ab',
    text: '#c9d0d6',
    bg: 'rgba(0,0,0,0)',
    hover: { bgcolor: '#23282d', bordercolor: '#3a4148', font: { color: '#e6e9ec', size: 11 } },
  },
  light: {
    grid: 'rgba(0,0,0,0.09)',
    zero: 'rgba(0,0,0,0.25)',
    line: 'rgba(0,0,0,0.3)',
    tick: '#4a4f55',
    text: '#22262a',
    bg: '#ffffff',
    hover: { bgcolor: '#ffffff', bordercolor: '#9aa3ab', font: { color: '#111417', size: 11 } },
  },
};

export function themedLayout(layout: Partial<Layout>, theme: PlotTheme = 'dark'): Partial<Layout> {
  const th = THEMES[theme];
  const axisBase = {
    gridcolor: th.grid,
    zerolinecolor: th.zero,
    linecolor: th.line,
    tickcolor: th.line,
    tickfont: { size: 10.5, color: th.tick },
    title: { font: { size: 11, color: th.tick } },
    automargin: true,
  };
  const xaxis = { ...axisBase, ...(layout.xaxis ?? {}), title: { ...axisBase.title, ...((layout.xaxis?.title as object) ?? {}) } } as Layout['xaxis'];
  const yaxis = { ...axisBase, ...(layout.yaxis ?? {}), title: { ...axisBase.title, ...((layout.yaxis?.title as object) ?? {}) } } as Layout['yaxis'];
  // Polar charts (reflection directions): same grid and tick colours as the cartesian axes.
  const polarAxis = { gridcolor: th.grid, linecolor: th.line, tickcolor: th.line, tickfont: { size: 10, color: th.tick } };
  const polar = layout.polar
    ? {
        polar: {
          bgcolor: th.bg,
          ...layout.polar,
          angularaxis: { ...polarAxis, ...(layout.polar.angularaxis ?? {}) },
          radialaxis: { ...polarAxis, ...(layout.polar.radialaxis ?? {}) },
        } as Layout['polar'],
      }
    : {};
  return {
    paper_bgcolor: th.bg,
    plot_bgcolor: th.bg,
    font: { family: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', size: 11, color: th.text },
    margin: { l: 48, r: 12, t: 12, b: 36 },
    hoverlabel: th.hover,
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 10.5, color: th.text }, bgcolor: 'rgba(0,0,0,0)' },
    bargap: 0.18,
    bargroupgap: 0.06,
    // Rounded bar ends (Plotly >= 2.29; not in the bundled typings).
    ...({ barcornerradius: 3 } as Partial<Layout>),
    ...layout,
    ...polar,
    xaxis,
    yaxis,
  };
}

/** Dark-theme shorthand of themedLayout. */
export function darkLayout(layout: Partial<Layout>): Partial<Layout> {
  return themedLayout(layout, 'dark');
}

export function Plot({
  data,
  layout,
  config,
  className = '',
  style,
  filename = 'auralis-plot',
  theme = 'dark',
  tools = false,
  table,
  xLabel,
  yLabel,
  onClick,
}: {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  style?: React.CSSProperties;
  filename?: string;
  theme?: PlotTheme;
  /** Show the CSV / PNG / copy buttons. */
  tools?: boolean;
  /** Exact data for the exports; default: built from the traces. */
  table?: DataTable | (() => DataTable);
  xLabel?: string;
  yLabel?: string;
  onClick?: (e: PlotMouseEvent) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  const app = useApp();

  useEffect(() => {
    let cancelled = false;
    loadPlotly()
      .then(async (Plotly) => {
        if (cancelled || !ref.current) return;
        await Plotly.react(ref.current, data, themedLayout(layout, theme), {
          displaylogo: false,
          responsive: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'],
          ...config,
        });
        const el = ref.current as unknown as { on?: (ev: string, cb: (e: PlotMouseEvent) => void) => void; __auralisClick?: boolean };
        if (el && el.on && !el.__auralisClick) {
          el.__auralisClick = true;
          el.on('plotly_click', (e) => clickRef.current?.(e));
        }
      })
      .catch((e) => setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [data, layout, config, theme]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      loadPlotly().then((Plotly) => {
        if (el.isConnected && (el as unknown as { _fullLayout?: unknown })._fullLayout) Plotly.Plots.resize(el);
      });
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      loadPlotly().then((Plotly) => Plotly.purge(el));
    };
  }, []);

  const exportTable = (): DataTable => {
    if (typeof table === 'function') return table();
    if (table) return table;
    const axTitle = (a: Partial<Layout>['xaxis']) => {
      const t = a?.title as { text?: string } | string | undefined;
      return typeof t === 'string' ? t : (t?.text ?? undefined);
    };
    return tableFromTraces(data, xLabel ?? axTitle(layout.xaxis) ?? 'x', yLabel ?? axTitle(layout.yaxis) ?? 'y');
  };
  const name = safeFilename(filename);

  const saveCsv = () => {
    try {
      downloadText(toCsv(exportTable()), `${name}.csv`);
    } catch (e) {
      app.toast('error', `CSV export failed: ${errorMessage(e)}`);
    }
  };
  const copy = async () => {
    try {
      const t = exportTable();
      await copyText(toTsv(t));
      app.toast('success', `Copied ${t.rows.length} rows (tab-separated, 3 decimals)`);
    } catch (e) {
      app.toast('error', `Copy failed: ${errorMessage(e)}`);
    }
  };
  const savePng = async () => {
    const el = ref.current;
    if (!el) return;
    try {
      const Plotly = await loadPlotly();
      const r = el.getBoundingClientRect();
      // Always a white background: the image goes into documents.
      await Plotly.downloadImage(
        { data, layout: { ...themedLayout(layout, 'light'), width: Math.max(480, Math.round(r.width)), height: Math.max(260, Math.round(r.height)) } } as unknown as HTMLElement,
        { format: 'png', filename: name, scale: 2, width: Math.max(480, Math.round(r.width)), height: Math.max(260, Math.round(r.height)) },
      );
    } catch (e) {
      app.toast('error', `PNG export failed: ${errorMessage(e)}`);
    }
  };

  return (
    <div className={`plot ${tools ? 'has-tools' : ''} ${className}`} style={style}>
      {error && <div className="plot-error">Chart failed to load: {error}</div>}
      {tools && (
        <div className="plot-tools no-print" role="toolbar" aria-label="Chart data">
          <button type="button" className="icon-link" onClick={saveCsv} title="Download the plotted data (CSV)">
            CSV
          </button>
          <button type="button" className="icon-link" onClick={() => void savePng()} title="Save the chart as a PNG image (white background)">
            PNG
          </button>
          <button type="button" className="icon-link" onClick={() => void copy()} title="Copy the plotted data to the clipboard (tab-separated, 3 decimals)">
            Copy
          </button>
        </div>
      )}
      <div ref={ref} className="plot-inner" />
    </div>
  );
}
