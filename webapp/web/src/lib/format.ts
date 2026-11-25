export function bandLabel(f: number): string {
  if (f >= 1000) return `${f % 1000 === 0 ? f / 1000 : (f / 1000).toFixed(1)}k`;
  return String(f);
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  return v.toFixed(digits);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '–';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return '–';
  if (s < 1) return '< 1 s';
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  if (m < 60) return r ? `${m} min ${r} s` : `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m - h * 60} min`;
}

export function fmtHz(f: number | null | undefined): string {
  if (f === null || f === undefined || !Number.isFinite(f)) return '–';
  return f >= 1000 ? `${(f / 1000).toFixed(f % 1000 === 0 ? 0 : 2)} kHz` : `${Math.round(f)} Hz`;
}

export function plural(n: number, word: string, pluralWord?: string): string {
  return `${n} ${n === 1 ? word : pluralWord ?? `${word}s`}`;
}
