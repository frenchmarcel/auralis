import { DISABLED_COLOR } from '../../lib/grids';

/** The captured 3D view on a white background with a title, a subtitle and the colour ramp. */
export async function mapPng(viewUrl: string, o: { title: string; subtitle: string; ramp: string[]; lo: string; mid: string; hi: string; filename: string }): Promise<void> {
  const img = new Image();
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error('the 3D view could not be read'));
    img.src = viewUrl;
  });
  const pad = 16;
  const legendH = 92;
  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height + legendH;
  const g = c.getContext('2d')!;
  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(img, 0, 0);
  const y0 = img.height;
  g.fillStyle = '#111417';
  g.font = '600 15px system-ui, sans-serif';
  g.fillText(o.title, pad, y0 + 22);
  g.font = '12px system-ui, sans-serif';
  g.fillStyle = '#4a4f55';
  g.fillText(o.subtitle, pad, y0 + 40);
  const bw = Math.min(420, c.width - 2 * pad - 140);
  const by = y0 + 52;
  const grad = g.createLinearGradient(pad, 0, pad + bw, 0);
  o.ramp.forEach((col, k) => grad.addColorStop(k / (o.ramp.length - 1), col));
  g.fillStyle = grad;
  g.fillRect(pad, by, bw, 14);
  g.fillStyle = '#111417';
  g.textAlign = 'left';
  g.fillText(o.lo, pad, by + 30);
  g.textAlign = 'center';
  g.fillText(o.mid, pad + bw / 2, by + 30);
  g.textAlign = 'right';
  g.fillText(o.hi, pad + bw, by + 30);
  g.textAlign = 'left';
  g.fillStyle = DISABLED_COLOR;
  g.fillRect(c.width - pad - 120, by, 14, 14);
  g.fillStyle = '#111417';
  g.fillText('no value', c.width - pad - 100, by + 12);
  const a = document.createElement('a');
  a.href = c.toDataURL('image/png');
  a.download = o.filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
