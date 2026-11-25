import type { Data } from 'plotly.js';

export type Cell = string | number | null | undefined;

/** A rectangular table of plotted data: what CSV, clipboard and data exports write. */
export interface DataTable {
  columns: string[];
  rows: Cell[][];
}

function fmtCell(v: Cell, digits: number | null, sep: string): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '';
    if (digits === null) return String(Number(v.toPrecision(7)));
    return String(Number(v.toFixed(digits)));
  }
  const s = String(v);
  if (sep === ',' && /[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  if (sep === '\t') return s.replace(/[\t\n]/g, ' ');
  return s;
}

export function toCsv(t: DataTable): string {
  return [t.columns, ...t.rows].map((r) => r.map((c) => fmtCell(c, null, ',')).join(',')).join('\n') + '\n';
}

/** Tab-separated, values rounded to 3 decimals (pastes into a spreadsheet). */
export function toTsv(t: DataTable, digits = 3): string {
  return [t.columns, ...t.rows].map((r) => r.map((c) => fmtCell(c, digits, '\t')).join('\t')).join('\n') + '\n';
}

export function safeFilename(name: string): string {
  return name.replace(/[^\w.-]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '') || 'auralis';
}

export function downloadText(text: string, filename: string, mime = 'text/csv'): void {
  // BOM: spreadsheet programs then read the UTF-8 labels (–, →, µ) correctly.
  const blob = new Blob([mime === 'text/csv' ? '\ufeff' : '', text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      /* fall back below (permission denied, insecure context) */
    }
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand('copy');
  ta.remove();
  if (!ok) throw new Error('the browser refused to copy to the clipboard');
}

type XY = { name?: string; x?: unknown; y?: unknown; meta?: unknown; visible?: unknown };

const plain = (s: string | undefined) => (s ?? '').replace(/<[^>]+>/g, '');

/**
 * Table of the plotted traces: one x column and one column per trace when they share
 * the same x, otherwise long format (series, x, y). Traces with meta "decor" are left out.
 */
export function tableFromTraces(data: Data[], xLabel = 'x', yLabel = 'y'): DataTable {
  const traces = (data as XY[]).filter(
    (d) => d.meta !== 'decor' && d.visible !== false && Array.isArray(d.x) && Array.isArray(d.y),
  ) as (XY & { x: Cell[]; y: Cell[] })[];
  if (!traces.length) return { columns: [xLabel, yLabel], rows: [] };
  const x0 = traces[0].x;
  const shared = traces.every((t) => t.x.length === x0.length && t.x.every((v, i) => v === x0[i]));
  if (shared) {
    return {
      columns: [plain(xLabel), ...traces.map((t, i) => plain(t.name) || `series ${i + 1}`)],
      rows: x0.map((xv, i) => [xv, ...traces.map((t) => t.y[i])]),
    };
  }
  const rows: Cell[][] = [];
  traces.forEach((t, k) => {
    const name = plain(t.name) || `series ${k + 1}`;
    t.x.forEach((xv, i) => {
      if (xv === null) return;
      rows.push([name, xv, t.y[i]]);
    });
  });
  return { columns: ['series', plain(xLabel), plain(yLabel)], rows };
}

/** Save a blob already fetched (e.g. after checking the server's answer). */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Download a server file (the server sends it as an attachment). */
export function downloadUrl(url: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}
