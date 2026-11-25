import type { Material } from '../api';

/** Look of a room surface in the realistic view, chosen from its material (display only). */
export type SurfaceKind =
  | 'plaster'
  | 'render'
  | 'concrete'
  | 'brick'
  | 'stone'
  | 'marble'
  | 'tiles'
  | 'parquet'
  | 'wood_floor'
  | 'wood_panel'
  | 'linoleum'
  | 'carpet'
  | 'curtain'
  | 'seating'
  | 'acoustic'
  | 'ceiling_tiles'
  | 'metal'
  | 'metal_sheet'
  | 'glass'
  | 'opening'
  | 'water'
  | 'asphalt'
  | 'paving'
  | 'grass'
  | 'gravel';

export interface SurfaceSpec {
  /** Texture set of the server (GET /api/textures); none = plain colour. */
  texture?: string;
  /** Base colour; with a texture it keeps only the texture's light/dark detail. */
  color?: string;
  /** Average colour of the texture: the plain colour shown while it loads or when textures are missing. */
  tone?: string;
  roughness: number;
  metalness?: number;
  normalScale?: number;
  /** Fraction of the texture's own contrast kept when `color` is set. */
  detail?: number;
  /** One texture repeat (m), instead of the scanned size: fine fabrics would shimmer (moiré) on large areas. */
  repeat?: number;
  /** Smooth normals, no normal map, no shadows received: stepped seating rows alias into moiré rings otherwise. */
  smooth?: boolean;
  opacity?: number;
  /** Suspended ceiling grid pitch (m). */
  grid?: number;
  envIntensity?: number;
  /** Casts no shadow and gets no section cap (glass, openings). */
  seeThrough?: boolean;
}

export const SURFACES: Record<SurfaceKind, SurfaceSpec> = {
  plaster: { texture: 'plaster', color: '#ebe7e1', roughness: 0.95, normalScale: 0.6, detail: 0.6 },
  render: { tone: '#d6c9b4', texture: 'render', roughness: 0.95, normalScale: 0.8 },
  concrete: { tone: '#a8a49d', texture: 'concrete', roughness: 0.95, normalScale: 0.8 },
  brick: { tone: '#8c4b37', texture: 'brick', roughness: 0.95, normalScale: 1 },
  stone: { tone: '#c8b48f', texture: 'stone', roughness: 0.9, normalScale: 0.9 },
  marble: { tone: '#dcd7ce', texture: 'marble', roughness: 0.35, normalScale: 0.5 },
  tiles: { tone: '#bcb9b3', texture: 'tiles', roughness: 0.5, normalScale: 0.8 },
  parquet: { tone: '#a06c3d', texture: 'parquet', roughness: 0.7, normalScale: 0.6 },
  wood_floor: { tone: '#8b5b34', texture: 'wood_floor', roughness: 0.75, normalScale: 0.6 },
  wood_panel: { tone: '#b88d5c', texture: 'wood_panel', roughness: 0.7, normalScale: 0.5 },
  linoleum: { tone: '#8b6b50', texture: 'linoleum', roughness: 0.6, normalScale: 0.4 },
  carpet: { texture: 'carpet', color: '#8b8579', roughness: 1, normalScale: 0.6, detail: 0.5, repeat: 0.9 },
  curtain: { texture: 'velvet', color: '#8a6d5b', roughness: 1, normalScale: 0.5, detail: 0.5, repeat: 1.2 },
  seating: { texture: 'velvet', color: '#7e1f29', roughness: 1, detail: 0.35, repeat: 1.5, smooth: true },
  acoustic: { texture: 'fabric', color: '#7d8085', roughness: 1, normalScale: 0.6, detail: 0.5, repeat: 0.9 },
  ceiling_tiles: { texture: 'ceiling_tiles', color: '#e9e7e2', roughness: 0.95, normalScale: 0.6, detail: 0.5, grid: 0.6 },
  metal: { color: '#b9bec3', roughness: 0.4, metalness: 0.85 },
  metal_sheet: { texture: 'metal_sheet', color: '#a3a9ae', roughness: 0.5, metalness: 0.6, normalScale: 1, detail: 0.5 },
  glass: { color: '#b4ccd6', roughness: 0.05, opacity: 0.22, envIntensity: 2.2, seeThrough: true },
  opening: { color: '#0d1013', roughness: 1, opacity: 0.12, seeThrough: true },
  water: { color: '#2f5f73', roughness: 0.08, opacity: 0.85, envIntensity: 1.6 },
  asphalt: { tone: '#4b4b4b', texture: 'asphalt', roughness: 0.95, normalScale: 0.8 },
  paving: { tone: '#9b9994', texture: 'paving', roughness: 0.9, normalScale: 0.9 },
  grass: { tone: '#5c7a36', texture: 'grass', roughness: 1, normalScale: 0.8 },
  gravel: { tone: '#9c958b', texture: 'gravel', roughness: 1, normalScale: 1 },
};

/** Ordered keyword rules (first match wins); the material text first, then the layer name. */
const RULES: [RegExp, SurfaceKind | ((t: string) => SurfaceKind)][] = [
  [/^opening|\bopenings\b|doorway|door opening|aperture|\bvoid\b/, 'opening'],
  [/water|swimming pool/, 'water'],
  [/audience|\bseats?\b|seating|chairs?\b|upholster|occupied|persons?\b|orchestra|choir/, 'seating'],
  [/carpet|\brug\b|needle felt|\bpile\b|tufted/, 'carpet'],
  [/curtain|drape|velour|velvet|blinds?\b|cloth\b|\bfabric\b(?!-covered)/, 'curtain'],
  [/parquet/, 'parquet'],
  [/plasterboard|gypsum|drywall/, 'plaster'],
  [/(ceiling|suspended).*(tile|board|panel|plate|mineral)|(tile|mineral fib(re|er)).*ceiling|fissured|dampa|gyptone|ecophon|armstrong/, 'ceiling_tiles'],
  [
    /absorb(er|ing)\b|fabric-covered|foam|mineral wool|rock ?wool|glass ?wool|min\. ?wool|fibre|fiber|porous|wood-?wool|acoustic|sonogamma|melamine|microsorber|basotect|resonator/,
    'acoustic',
  ],
  [/linoleum|vinyl|rubber|cork|pvc floor/, 'linoleum'],
  [
    /wood|timber|plywood|veneer|\bmdf\b|chipboard|plank|joist|tongue|\boak\b|\bpine\b|beech|\bdoor\b|sleepers|\bdeck/,
    (t) => (/floor|stage|sleepers|deck|joist/.test(t) ? 'wood_floor' : 'wood_panel'),
  ],
  [/marble|terrazzo|granite/, 'marble'],
  [/ceramic|\btiles?\b|mosaic/, 'tiles'],
  [/glass|glazing|window|\bpane\b/, 'glass'],
  [/brick/, 'brick'],
  [/gravel|ballast|crushed/, 'gravel'],
  [/asphalt|\broad\b|tarmac|carriageway/, 'asphalt'],
  [/paving|pavement|sidewalk|cobble|\bpavers?\b/, 'paving'],
  [/grass|lawn|turf|\bsoil\b|\bearth\b/, 'grass'],
  [/\bground\b/, 'asphalt'],
  [/stone|sandstone|limestone|ashlar|masonry/, 'stone'],
  [/trapez|corrugat|profiled? (steel|metal|sheet)|\broof/, 'metal_sheet'],
  [/steel|alumin|metal|\biron\b|\bzinc\b|lamell|\bsheet\b|\bcars?\b|vehicle/, 'metal'],
  [/concrete|cement|\bblocks?\b|reverberation chamber|screed|\bslab\b/, 'concrete'],
  [/facade|façade|exterior wall|\brender\b/, 'render'],
  [/plaster|gypsum|drywall|\blime\b|paint|wallpaper|stucco|hard surface|\bwalls?\b|ceiling/, 'plaster'],
  [/floor|stage/, 'wood_floor'],
];

/** Generic library entries say nothing about the look ("80% absorbent", "Default"). */
const GENERIC = /^\s*(\d+\s*%\s*absorbent|default|generic|general|special|custom|untitled|material\s*\d*)\b/;

function classify(text: string): SurfaceKind | null {
  const t = text.toLowerCase();
  for (const [re, kind] of RULES) if (re.test(t)) return typeof kind === 'function' ? kind(t) : kind;
  return null;
}

/** Surface kind of a layer: its material's name, category and id, then the layer name; plaster otherwise. */
export function surfaceKindOf(material: Material | undefined, layerName: string): SurfaceKind {
  if (material && !GENERIC.test(material.name)) {
    // The name alone first: categories are broad ("Floor;Wall;Ceiling", "Panel").
    const k = classify(material.name) ?? classify(`${material.category ?? ''} ${material.id.split(':').slice(1).join(' ')}`);
    if (k) return k;
  }
  return classify(layerName.replace(/^obj:/, '').replace(/[_\-.]+/g, ' ')) ?? 'plaster';
}
