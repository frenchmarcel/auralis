import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { api, type Geometry, type MeshData, type Simulation, type TextureLibrary } from '../../api';
import { Segmented } from '../../components/ui';
import type { Bounds } from '../../lib/geom';
import { LocalClipping, RealisticStage, SectionCaps, type LookOptions, type ViewLookContext } from '../../three/Realistic';
import type { Shading } from '../../three/Scene';

type ToastFn = (kind: 'info' | 'success' | 'warning' | 'error', message: string) => void;
type LightMode = 'auto' | 'interior' | 'exterior';

interface StoredLook {
  shading: Shading;
  shadows: boolean;
  ao: boolean;
  bloom: boolean;
  light: LightMode;
}

const STORE_KEY = 'auralis.viewLook.v1';
const DEFAULT_LOOK: StoredLook = { shading: 'realistic', shadows: true, ao: true, bloom: false, light: 'auto' };

function readStored(): StoredLook {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    const v = raw ? (JSON.parse(raw) as Partial<StoredLook>) : {};
    const shading = (['realistic', 'shaded', 'wireframe', 'ghosted'] as const).find((s) => s === v.shading);
    const light = (['auto', 'interior', 'exterior'] as const).find((s) => s === v.light);
    return {
      shading: shading ?? DEFAULT_LOOK.shading,
      shadows: typeof v.shadows === 'boolean' ? v.shadows : DEFAULT_LOOK.shadows,
      ao: typeof v.ao === 'boolean' ? v.ao : DEFAULT_LOOK.ao,
      bloom: typeof v.bloom === 'boolean' ? v.bloom : DEFAULT_LOOK.bloom,
      light: light ?? DEFAULT_LOOK.light,
    };
  } catch {
    return DEFAULT_LOOK;
  }
}

function writeStored(v: StoredLook) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(v));
  } catch {
    // Per-viewer convenience only.
  }
}

let libraryPromise: Promise<TextureLibrary | null> | null = null;
const warned = new Set<string>();

/** Texture index of the server, fetched once per page. */
function loadLibrary(): Promise<TextureLibrary | null> {
  if (!libraryPromise) {
    libraryPromise = api.getTextures().catch((e) => {
      console.warn('[Auralis] textures index', e);
      return null;
    });
  }
  return libraryPromise;
}

/** Default cut height: 1.2 m above the model's floor, or its lower third in tall rooms. */
export function defaultCutHeight(bounds: Bounds): number {
  const h = bounds.max[2] - bounds.min[2];
  return bounds.min[2] + Math.min(Math.max(1.2, h / 3), 0.9 * h);
}

/** Room-like model: closed, not declared an open scene. */
export function isIndoor(geometry: Geometry | null, sim: Simulation | null): boolean {
  return !!geometry && !!sim && !sim.settings.open_scene && !!geometry.check?.watertight;
}

/** Display look of the editor's 3D view: shading, realistic light options, cut-away. */
export function useViewLookState({ geometry, sim, bounds, toast }: { geometry: Geometry | null; sim: Simulation | null; bounds: Bounds | null; toast: ToastFn }) {
  const [stored, setStored] = useState<StoredLook>(readStored);
  const update = useCallback((patch: Partial<StoredLook>) => {
    setStored((s) => {
      const n = { ...s, ...patch };
      writeStored(n);
      return n;
    });
  }, []);
  const [library, setLibrary] = useState<TextureLibrary | null>(null);
  const toastRef = useRef(toast);
  toastRef.current = toast;
  const warnOnce = useCallback((msg: string) => {
    if (warned.has(msg)) return;
    warned.add(msg);
    toastRef.current('warning', msg);
  }, []);

  useEffect(() => {
    if (stored.shading !== 'realistic') return;
    let alive = true;
    void loadLibrary().then((lib) => {
      if (!alive) return;
      setLibrary(lib);
      if (!lib || !lib.available)
        warnOnce('Surface textures are not downloaded (python -m webapp.server.fetch_textures): the realistic view shows plain colours.');
    });
    return () => {
      alive = false;
    };
  }, [stored.shading, warnOnce]);

  // Cut-away: on by default for closed rooms, set once per geometry.
  const [cut, setCut] = useState<{ gid: string; on: boolean; z: number } | null>(null);
  const gid = geometry?.id ?? null;
  const indoor = isIndoor(geometry, sim);
  useEffect(() => {
    if (!gid || !bounds || !sim || (cut && cut.gid === gid)) return;
    setCut({ gid, on: indoor, z: defaultCutHeight(bounds) });
  }, [gid, bounds, sim, indoor, cut]);
  const cutOn = !!cut && cut.gid === gid && cut.on;
  const cutZ = cut?.z ?? 0;
  const plane = useMemo(() => (cutOn ? new THREE.Plane(new THREE.Vector3(0, 0, -1), cutZ) : null), [cutOn, cutZ]);

  const realistic = stored.shading === 'realistic';
  const outdoor = stored.light === 'auto' ? !indoor : stored.light === 'exterior';
  const options = useMemo<LookOptions>(
    () => ({ shadows: stored.shadows, ao: stored.ao, bloom: stored.bloom, outdoor }),
    [stored.shadows, stored.ao, stored.bloom, outdoor],
  );
  const [environment, setEnvironment] = useState<THREE.Texture | null>(null);
  const context = useMemo<ViewLookContext>(
    () => ({ library: realistic ? library : null, cut: plane, environment: realistic ? environment : null, setEnvironment, onWarning: warnOnce }),
    [realistic, library, plane, environment, warnOnce],
  );

  return {
    shading: stored.shading,
    setShading: (s: Shading) => update({ shading: s }),
    realistic,
    /** The realistic stage renders through the post-processing composer. */
    postFx: realistic && (stored.ao || stored.bloom),
    options,
    stored,
    update,
    library,
    cutOn,
    cutZ,
    setCutOn: (on: boolean) => {
      if (gid) setCut((c) => ({ gid, on, z: c?.gid === gid ? c.z : bounds ? defaultCutHeight(bounds) : 0 }));
    },
    setCutZ: (z: number) => {
      if (gid) setCut({ gid, on: true, z });
    },
    context,
  };
}

export type ViewLookState = ReturnType<typeof useViewLookState>;

/** Inside the Canvas: clipping, the realistic lights and post-processing, the section caps. */
export function ViewLookScene({
  look,
  mesh,
  bounds,
  capsInclude,
}: {
  look: ViewLookState;
  mesh: MeshData | null;
  bounds: Bounds | null;
  capsInclude: (name: string) => boolean;
}) {
  const capWidth = bounds ? Math.min(0.35, Math.max(0.1, bounds.radius * 0.015)) : 0.1;
  const showCaps = look.cutOn && !!mesh && (look.shading === 'realistic' || look.shading === 'shaded');
  return (
    <>
      <LocalClipping />
      {look.realistic && bounds && <RealisticStage bounds={bounds} options={look.options} />}
      {showCaps && <SectionCaps mesh={mesh!} include={capsInclude} height={look.cutZ} width={capWidth} color={look.realistic ? '#3b3f44' : '#2c3238'} />}
    </>
  );
}

/** Toolbar part: shading, cut-away, isometric preset, light options. */
export function ViewLookToolbar({ look, onIso }: { look: ViewLookState; onIso: () => void }) {
  return (
    <>
      <Segmented<Shading>
        value={look.shading}
        onChange={look.setShading}
        options={[
          { value: 'realistic', label: 'Realistic', title: 'Surfaces textured from their materials, soft light, shadows and ambient occlusion' },
          { value: 'shaded', label: 'Shaded', title: 'One flat colour per layer' },
          { value: 'wireframe', label: 'Wireframe' },
          { value: 'ghosted', label: 'Ghosted' },
        ]}
      />
      <button
        className={`btn btn-sm ${look.cutOn ? 'btn-on' : ''}`}
        onClick={() => look.setCutOn(!look.cutOn)}
        title="Cut-away: hide everything above a height (ceiling included), cut walls drawn solid"
        aria-pressed={look.cutOn}
      >
        Cut-away
      </button>
      <button className="btn btn-sm" onClick={onIso} title="Isometric view: the whole model, 3/4 from above">
        Iso
      </button>
      {look.realistic && <LightMenu look={look} />}
    </>
  );
}

function LightMenu({ look }: { look: ViewLookState }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopImmediatePropagation();
        setOpen(false);
      }
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [open]);
  const s = look.stored;
  const lib = look.library;
  return (
    <div className="look-menu" ref={ref}>
      <button className={`btn btn-sm ${open ? 'btn-on' : ''}`} onClick={() => setOpen((o) => !o)} aria-expanded={open} title="Light and effects of the realistic view">
        Light ▾
      </button>
      {open && (
        <div className="look-menu-panel" role="dialog" aria-label="Light and effects">
          <label className="check-row">
            <input type="checkbox" checked={s.shadows} onChange={(e) => look.update({ shadows: e.target.checked })} /> Shadows
          </label>
          <label className="check-row">
            <input type="checkbox" checked={s.ao} onChange={(e) => look.update({ ao: e.target.checked })} /> Ambient occlusion
          </label>
          <label className="check-row">
            <input type="checkbox" checked={s.bloom} onChange={(e) => look.update({ bloom: e.target.checked })} /> Bloom
          </label>
          <div className="small muted">Environment light</div>
          <Segmented<LightMode>
            value={s.light}
            onChange={(v) => look.update({ light: v })}
            options={[
              { value: 'auto', label: 'Auto', title: 'Interior for closed rooms, sky for open scenes' },
              { value: 'interior', label: 'Interior' },
              { value: 'exterior', label: 'Sky' },
            ]}
          />
          <div className="small muted look-credit">
            {lib?.available
              ? 'Display only. Textures and environment maps: Poly Haven, CC0.'
              : 'Textures not downloaded: run python -m webapp.server.fetch_textures (plain colours meanwhile).'}
          </div>
        </div>
      )}
    </div>
  );
}

/** Vertical slider at the left of the view: the cut height above the model's floor. */
export function CutSlider({ look, bounds }: { look: ViewLookState; bounds: Bounds }) {
  const z0 = bounds.min[2];
  const h = Math.max(0.1, bounds.max[2] - z0);
  const step = h > 20 ? 0.1 : 0.05;
  return (
    <div className="cut-slider" title="Cut-away height above the floor of the model">
      <span className="small">{(look.cutZ - z0).toFixed(2)} m</span>
      <input
        type="range"
        className="vertical"
        min={0.05}
        max={Math.round(h * 100) / 100}
        step={step}
        value={Math.round((look.cutZ - z0) * 100) / 100}
        onChange={(e) => look.setCutZ(z0 + Number(e.target.value))}
        aria-label="Cut-away height"
      />
      <span className="small muted">Cut</span>
    </div>
  );
}
