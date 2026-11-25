import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import type { Directivity, Path, PathValidity, PointGroup, PointValidity, Receiver, Source, Vec3 } from '../../api';
import { useApp } from '../../AppContext';
import { anyModalOpen, isTypingTarget, Menu, Modal, NumField } from '../../components/ui';
import { SERIES } from '../../lib/colors';
import { azElTowards, distance, roundVec, type Bounds } from '../../lib/geom';
import { translatePath, withPath } from '../../lib/trajectory';
import { EQ_TITLE, EqCurve, eqActive, eqSummary, SourceEqEditor, withEq } from './SourceEq';
import { TrajectorySection } from './TrajectorySection';

export type PointKind = 'source' | 'receiver';
type Item = Source | Receiver;

/** Server limit per simulation, for sources and for receivers. */
export const MAX_POINTS = 100;
/** Smallest offset of a point created from a surface (m). */
export const MIN_SURFACE_OFFSET = 0.01;
const GROUP_COLORS = SERIES;
/** Select entry that opens the directivity library instead of choosing a directivity. */
const DIR_LIBRARY = '__library__';

function prefixOf(kind: PointKind) {
  return kind === 'source' ? 'S' : 'R';
}

function nextIndex(kind: PointKind, items: Item[]): number {
  const re = new RegExp(`^${prefixOf(kind)}(\\d+)$`);
  let max = 0;
  for (const it of items) {
    const m = re.exec(it.id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return max + 1;
}

export function makeItem(kind: PointKind, n: number, position: Vec3, name?: string): Item {
  const base = { id: `${prefixOf(kind)}${n}`, name: name ?? `${kind === 'source' ? 'Source' : 'Receiver'} ${n}`, position: roundVec(position), azimuth_deg: 0, elevation_deg: 0 };
  return kind === 'source' ? { ...base, directivity: 'omni', gain_db: 0 } : base;
}

/** One more point of this kind at `position`, or null when the list is full. */
export function appendPoint(kind: PointKind, items: Item[], position: Vec3): { items: Item[]; added: Item } | null {
  if (items.length >= MAX_POINTS) return null;
  const added = makeItem(kind, nextIndex(kind, items), position);
  return { items: [...items, added], added };
}

export function limitMessage(kind: PointKind, wanted: number, room: number): string {
  const noun = kind === 'source' ? 'sources' : 'receivers';
  return room <= 0 ? `No more ${noun}: a simulation holds at most ${MAX_POINTS} ${noun}.` : `Only ${room} of ${wanted} new ${noun} fit (at most ${MAX_POINTS} ${noun} per simulation).`;
}

/** Step +0.5 m in x until the point is at least 0.5 m away from every existing point. */
function freePosition(start: Vec3, occupied: Vec3[]): Vec3 {
  let p: Vec3 = [...start];
  for (let i = 0; i < 40; i++) {
    if (occupied.every((o) => distance(o, p) >= 0.5)) return p;
    p = [p[0] + 0.5, p[1], p[2]];
  }
  return start;
}

/** Lines "x, y, z[, name]" (or "name, x, y, z"); comma, semicolon, tab or space separated. */
export function parsePointsText(text: string): { points: { name?: string; position: Vec3 }[]; errors: string[] } {
  const points: { name?: string; position: Vec3 }[] = [];
  const errors: string[] = [];
  text.split(/\r?\n/).forEach((raw, i) => {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) return;
    const tokens = /[;\t]/.test(line) ? line.split(/[;\t]+/) : line.includes(',') ? line.split(',') : line.split(/\s+/);
    const t = tokens.map((x) => x.trim()).filter(Boolean);
    const num = (x: string) => (/^[-+]?(\d+([.]\d*)?|[.]\d+)(e[-+]?\d+)?$/i.test(x) ? Number(x) : NaN);
    const nums = t.map(num);
    let name: string | undefined;
    let xyz: number[];
    if (t.length >= 4 && !Number.isFinite(nums[0])) {
      name = t[0];
      xyz = nums.slice(1, 4);
    } else {
      xyz = nums.slice(0, 3);
      if (t.length >= 4) name = t.slice(3).join(', ');
    }
    if (xyz.length < 3 || xyz.some((v) => !Number.isFinite(v))) {
      if (i === 0 && /[a-z]/i.test(line)) return; // header
      errors.push(`Line ${i + 1}: “${line}” is not “x, y, z[, name]”.`);
      return;
    }
    points.push({ name, position: [xyz[0], xyz[1], xyz[2]] });
  });
  return { points, errors };
}

function centroid(ps: Vec3[]): Vec3 {
  const c: Vec3 = [0, 0, 0];
  for (const p of ps) for (let k = 0; k < 3; k++) c[k] += p[k] / ps.length;
  return c;
}

function withoutGroup(it: Item): Item {
  const { group: _g, ...rest } = it as Item & { group?: PointGroup | null };
  return rest as Item;
}

interface Section {
  key: string;
  kind: 'none' | 'group' | 'ensemble';
  title: string;
  group?: PointGroup;
  members: { item: Item; index: number }[];
}

export function PointsTab({
  kind,
  items,
  others,
  directivities,
  validity,
  selectedId,
  onSelect,
  onChange,
  bounds,
  locked,
  onAddEnsemble,
  onDeleteGroup,
  pathValidity,
  selectedWaypoint,
  onSelectWaypoint,
  drawingId,
  onDraw,
  onMultiSelect,
  surfacePick,
  onSurfacePick,
  saveError,
  layers,
}: {
  kind: PointKind;
  items: Item[];
  others: Item[];
  directivities: Directivity[];
  validity: Map<string, PointValidity>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onChange: (items: Item[]) => void;
  bounds: Bounds | null;
  locked: boolean;
  /** Server checks of the trajectories, by point id. */
  pathValidity: Map<string, PathValidity>;
  /** Index of the selected waypoint of the selected point, if any. */
  selectedWaypoint: number | null;
  onSelectWaypoint: (id: string, index: number | null) => void;
  /** Point whose path is being drawn in the 3D view (click mode). */
  drawingId: string | null;
  onDraw: (id: string | null) => void;
  /** Sources only: open the ensemble dialog. */
  onAddEnsemble?: () => void;
  /** Sources only: delete every source of an ensemble (asks for confirmation). */
  onDeleteGroup?: (ensemble: string) => void;
  /** Every selected id (Ctrl/Shift-click in the list), for the 3D highlight. */
  onMultiSelect?: (ids: string[]) => void;
  /** Offset (m) while "place from a surface" is active in the 3D view, else null. */
  surfacePick?: number | null;
  /** Start (distance) or stop (null) the surface placement mode. */
  onSurfacePick?: (distance: number | null) => void;
  /** Last save error of the simulation, shown next to the EQ when it is about it. */
  saveError?: string | null;
  /** Sources only: the model's layers, for a vibrating-surface source. */
  layers?: string[];
}) {
  const app = useApp();
  const [multiOpen, setMultiOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [surfaceOpen, setSurfaceOpen] = useState(false);
  const [offset, setOffset] = useState(kind === 'source' ? 1.5 : 1.2);
  const [multiplyIds, setMultiplyIds] = useState<string[] | null>(null);
  const [groupEdit, setGroupEdit] = useState<{ ids: string[]; group: PointGroup; isNew: boolean } | null>(null);
  const [search, setSearch] = useState('');
  // Ensembles start collapsed (up to 63 players); user groups start open.
  const [openEnsembles, setOpenEnsembles] = useState<Set<string>>(new Set());
  const [closedGroups, setClosedGroups] = useState<Set<string>>(new Set());
  const [multi, setMulti] = useState<Set<string>>(new Set());
  /** Sources whose EQ editor is open. */
  const [eqOpen, setEqOpen] = useState<Set<string>>(new Set());
  const anchor = useRef<string | null>(null);

  const noun = kind === 'source' ? 'source' : 'receiver';
  const otherNoun = kind === 'source' ? 'receiver' : 'source';
  const room = MAX_POINTS - items.length;
  const full = room <= 0;
  const occupied = () => [...items, ...others].map((p) => p.position);
  const byId = useMemo(() => new Map(items.map((it) => [it.id, it])), [items]);

  // The list's selection follows the editor's (3D click, Esc, deletion).
  useEffect(() => {
    setMulti((m) => (selectedId === null ? (m.size ? new Set() : m) : m.has(selectedId) ? m : new Set([selectedId])));
  }, [selectedId]);
  const selected = useMemo(() => items.filter((it) => multi.has(it.id)), [items, multi]);
  const selKey = selected.map((x) => x.id).join(',');
  // Callbacks by ref: the editor passes inline functions, the effects must not re-run for them.
  const cb = useRef({ onMultiSelect, onSurfacePick });
  cb.current = { onMultiSelect, onSurfacePick };
  useEffect(() => {
    cb.current.onMultiSelect?.(selKey ? selKey.split(',') : []);
  }, [selKey]);
  useEffect(
    () => () => {
      cb.current.onMultiSelect?.([]);
      cb.current.onSurfacePick?.(null);
    },
    [],
  );

  const q = search.trim().toLowerCase();
  const matches = (it: Item) => !q || it.name.toLowerCase().includes(q) || it.id.toLowerCase().includes(q) || (it.group?.name ?? '').toLowerCase().includes(q);

  const sections = useMemo<Section[]>(() => {
    const order: string[] = [];
    const map = new Map<string, Section>();
    items.forEach((item, index) => {
      if (!matches(item)) return;
      const g = item.group ?? null;
      const ens = kind === 'source' ? ((item as Source).ensemble ?? null) || null : null;
      const key = g ? `g:${g.id}` : ens ? `e:${ens}` : 'none';
      let sec = map.get(key);
      if (!sec) {
        sec = g ? { key, kind: 'group', title: g.name, group: g, members: [] } : ens ? { key, kind: 'ensemble', title: ens, members: [] } : { key, kind: 'none', title: '', members: [] };
        map.set(key, sec);
        order.push(key);
      }
      sec.members.push({ item, index });
    });
    const rank = (k: string) => (k === 'none' ? 0 : k.startsWith('g:') ? 1 : 2);
    return order.sort((a, b) => rank(a) - rank(b)).map((k) => map.get(k)!);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, kind, q]);
  const isOpen = (s: Section) => !!q || (s.kind === 'group' ? !closedGroups.has(s.key) : s.kind === 'ensemble' ? openEnsembles.has(s.title) : true);
  const toggleOpen = (s: Section) => {
    if (s.kind === 'group')
      setClosedGroups((o) => {
        const n = new Set(o);
        if (n.has(s.key)) n.delete(s.key);
        else n.add(s.key);
        return n;
      });
    else if (s.kind === 'ensemble')
      setOpenEnsembles((o) => {
        const n = new Set(o);
        if (n.has(s.title)) n.delete(s.title);
        else n.add(s.title);
        return n;
      });
  };
  /** Ids in display order (for Shift ranges). */
  const visibleOrder = sections.flatMap((s) => (isOpen(s) ? s.members.map((m) => m.item.id) : []));
  /** Previous and next point of the same list section (group, ensemble or ungrouped), as displayed. */
  const neighbours = useMemo(() => {
    const m = new Map<string, { up: string | null; down: string | null }>();
    for (const s of sections) s.members.forEach((mb, k) => m.set(mb.item.id, { up: s.members[k - 1]?.item.id ?? null, down: s.members[k + 1]?.item.id ?? null }));
    return m;
  }, [sections]);
  /**
   * Swap a point with its neighbour in its section. Only the order changes: the server
   * compares sources and receivers by id, so the results stay valid.
   */
  const move = (id: string, dir: 'up' | 'down') => {
    const other = neighbours.get(id)?.[dir];
    if (!other) return;
    const a = items.findIndex((x) => x.id === id);
    const b = items.findIndex((x) => x.id === other);
    if (a < 0 || b < 0) return;
    const next = [...items];
    [next[a], next[b]] = [next[b], next[a]];
    onChange(next);
    window.requestAnimationFrame(() => document.querySelector(`.point-row[data-id="${CSS.escape(id)}"]`)?.scrollIntoView({ block: 'nearest' }));
  };

  // Open the section of a point selected elsewhere (3D view).
  useEffect(() => {
    const it = selectedId ? byId.get(selectedId) : undefined;
    if (!it) return;
    if (it.group) {
      const k = `g:${it.group.id}`;
      if (closedGroups.has(k))
        setClosedGroups((o) => {
          const n = new Set(o);
          n.delete(k);
          return n;
        });
    } else if (kind === 'source' && (it as Source).ensemble && !openEnsembles.has((it as Source).ensemble!)) {
      setOpenEnsembles((o) => new Set(o).add((it as Source).ensemble!));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);
  useEffect(() => {
    if (selectedId) document.querySelector(`.point-row[data-id="${CSS.escape(selectedId)}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [selectedId, openEnsembles, closedGroups]);

  const selectIds = (ids: string[], primary?: string) => {
    setMulti(new Set(ids));
    onSelect(primary ?? ids[0] ?? null);
  };

  const clickRow = (id: string, e: ReactMouseEvent) => {
    const target = e.target as HTMLElement;
    const onControl = !!target.closest('input, select, textarea, button, a, label');
    if (!onControl && (e.ctrlKey || e.metaKey)) {
      const n = new Set(multi);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      anchor.current = id;
      setMulti(n);
      onSelect(n.has(id) ? id : ([...n][n.size - 1] ?? null));
      return;
    }
    if (!onControl && e.shiftKey && anchor.current && visibleOrder.includes(anchor.current)) {
      const a = visibleOrder.indexOf(anchor.current);
      const b = visibleOrder.indexOf(id);
      const [lo, hi] = a < b ? [a, b] : [b, a];
      setMulti(new Set(visibleOrder.slice(lo, hi + 1)));
      onSelect(id);
      return;
    }
    anchor.current = id;
    if (!multi.has(id) || multi.size > 1 || selectedId !== id) {
      setMulti(new Set([id]));
      onSelect(id);
    }
  };

  const add = () => {
    if (full) {
      app.toast('warning', limitMessage(kind, 1, room));
      return;
    }
    const start: Vec3 = items.length
      ? (() => {
          const l = items[items.length - 1].position;
          return [l[0] + 0.5, l[1], l[2]] as Vec3;
        })()
      : bounds
        ? bounds.center
        : [0, 0, 1.2];
    const it = makeItem(kind, nextIndex(kind, items), freePosition(start, occupied()));
    onChange([...items, it]);
    onSelect(it.id);
  };

  const patch = (id: string, p: Partial<Source & Receiver>) => onChange(items.map((it) => (it.id === id ? ({ ...it, ...p } as Item) : it)));
  const setEq = (id: string, eq: number[] | undefined) => onChange(items.map((it) => (it.id === id ? withEq(it as Source, eq) : it)));
  const toggleEq = (id: string) =>
    setEqOpen((o) => {
      const n = new Set(o);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const setPath = (id: string, path: Path | null) => onChange(items.map((it) => (it.id === id ? withPath(it, path) : it)));
  /** A moving point's position is its first waypoint: moving it translates the whole path. */
  const moveTo = (it: Item, p: Vec3) => {
    if (!it.path) return patch(it.id, { position: p });
    setPath(it.id, translatePath(it.path, [p[0] - it.position[0], p[1] - it.position[1], p[2] - it.position[2]]));
  };
  const removeIds = (ids: string[]) => {
    const drop = new Set(ids);
    onChange(items.filter((it) => !drop.has(it.id)));
    if (selectedId && drop.has(selectedId)) onSelect(null);
    setMulti((m) => new Set([...m].filter((x) => !drop.has(x))));
  };
  const removeSelected = async () => {
    if (selected.length === 0 || locked) return;
    if (selected.length > 1) {
      const ok = await app.confirm({ title: `Delete ${noun}s`, message: `Delete the ${selected.length} selected ${noun}s?`, confirmLabel: 'Delete', danger: true });
      if (!ok) return;
    }
    removeIds(selected.map((x) => x.id));
  };

  /** Copies of `src` (all fields kept, group included), stepped by `step`: k = 1..count. */
  const multiply = (ids: string[], count: number, step: Vec3) => {
    const srcs = ids.map((id) => byId.get(id)).filter((x): x is Item => !!x);
    const wanted = srcs.length * count;
    const out: Item[] = [];
    let n = nextIndex(kind, items);
    for (const it of srcs) {
      for (let k = 1; k <= count && out.length < room; k++) {
        const d: Vec3 = [step[0] * k, step[1] * k, step[2] * k];
        const pos = roundVec([it.position[0] + d[0], it.position[1] + d[1], it.position[2] + d[2]]);
        let copy = { ...it, id: `${prefixOf(kind)}${n++}`, name: `${it.name} (${k})`, position: pos } as Item;
        if (it.path) copy = withPath(copy, translatePath(it.path, d));
        out.push(copy);
      }
    }
    if (out.length < wanted) app.toast('warning', limitMessage(kind, wanted, room));
    if (!out.length) return;
    onChange([...items, ...out]);
    selectIds(out.map((x) => x.id));
    setMultiplyIds(null);
  };

  // ---- groups ----
  const allGroups = useMemo(() => {
    const m = new Map<string, PointGroup>();
    for (const it of [...items, ...others]) if (it.group && !m.has(it.group.id)) m.set(it.group.id, it.group);
    return m;
  }, [items, others]);
  const newGroup = (): PointGroup => {
    let k = 1;
    while (allGroups.has(`G${k}`)) k++;
    const used = new Set([...allGroups.values()].map((g) => g.color));
    const color = GROUP_COLORS.find((c) => !used.has(c)) ?? GROUP_COLORS[(k - 1) % GROUP_COLORS.length];
    return { id: `G${k}`, name: `Group ${k}`, color };
  };
  const startGroup = (ids: string[]) => {
    if (locked || !ids.length) return;
    setGroupEdit({ ids, group: newGroup(), isNew: true });
  };
  const applyGroup = (ids: string[], g: PointGroup) => {
    const set = new Set(ids);
    onChange(items.map((it) => (set.has(it.id) ? ({ ...it, group: g } as Item) : it)));
    setClosedGroups((o) => {
      const n = new Set(o);
      n.delete(`g:${g.id}`);
      return n;
    });
  };
  const editGroup = (g: PointGroup) => setGroupEdit({ ids: items.filter((it) => it.group?.id === g.id).map((x) => x.id), group: g, isNew: false });
  const updateGroup = (g: PointGroup) => onChange(items.map((it) => (it.group?.id === g.id ? ({ ...it, group: g } as Item) : it)));
  const ungroup = (pred: (it: Item) => boolean) => onChange(items.map((it) => (pred(it) && it.group ? withoutGroup(it) : it)));
  const deleteGroup = async (g: PointGroup) => {
    const ids = items.filter((it) => it.group?.id === g.id).map((x) => x.id);
    const ok = await app.confirm({
      title: 'Delete group',
      message: (
        <>
          Delete the group <b>{g.name}</b> and its {ids.length} {ids.length === 1 ? noun : `${noun}s`}?
        </>
      ),
      confirmLabel: 'Delete',
      danger: true,
    });
    if (ok) removeIds(ids);
  };

  // ---- aiming ----
  const aimTargets = useMemo(() => {
    const list: { value: string; label: string; position: Vec3 }[] = others.map((o) => ({ value: o.id, label: o.name, position: o.position }));
    if (others.length > 1) list.push({ value: '@centre', label: `Centre of all ${otherNoun}s`, position: centroid(others.map((o) => o.position)) });
    const groups = new Map<string, { g: PointGroup; ps: Vec3[] }>();
    for (const o of others) if (o.group) (groups.get(o.group.id) ?? groups.set(o.group.id, { g: o.group, ps: [] }).get(o.group.id)!).ps.push(o.position);
    for (const { g, ps } of groups.values()) list.push({ value: `@group:${g.id}`, label: `Centre of group “${g.name}”`, position: centroid(ps) });
    const ensembles = new Map<string, Vec3[]>();
    for (const o of others) {
      const e = (o as Source).ensemble;
      if (e) (ensembles.get(e) ?? ensembles.set(e, []).get(e)!).push(o.position);
    }
    for (const [e, ps] of ensembles) list.push({ value: `@ensemble:${e}`, label: `Centre of ensemble “${e}”`, position: centroid(ps) });
    return list;
  }, [others, otherNoun]);
  const aim = (ids: string[], value: string) => {
    const t = aimTargets.find((x) => x.value === value);
    if (!t) return;
    const set = new Set(ids);
    onChange(items.map((it) => (set.has(it.id) && distance(it.position, t.position) > 1e-6 ? ({ ...it, ...azElTowards(it.position, t.position) } as Item) : it)));
  };
  const aimSelect = (ids: string[], title: string, className?: string) => (
    <select
      className={className}
      value=""
      disabled={others.length === 0 || locked}
      title={others.length === 0 ? `Add a ${otherNoun} first` : title}
      onChange={(e) => aim(ids, e.target.value)}
      aria-label={title}
    >
      <option value="">Point at…</option>
      {aimTargets.map((t) => (
        <option key={t.value} value={t.value}>
          {t.label}
        </option>
      ))}
    </select>
  );

  // ---- keyboard: Ctrl+G groups the selection, Ctrl+Shift+G ungroups it, Delete removes several, Alt+Up/Down moves one ----
  const keyState = useRef({ selected, locked, startGroup, ungroup, removeSelected, move });
  keyState.current = { selected, locked, startGroup, ungroup, removeSelected, move };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || anyModalOpen() || isTypingTarget(e.target)) return;
      const st = keyState.current;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'g') {
        e.preventDefault();
        if (st.locked || !st.selected.length) return;
        if (e.shiftKey) {
          const ids = new Set(st.selected.map((x) => x.id));
          st.ungroup((it) => ids.has(it.id));
        } else st.startGroup(st.selected.map((x) => x.id));
      } else if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown') && st.selected.length === 1 && !st.locked) {
        e.preventDefault();
        st.move(st.selected[0].id, e.key === 'ArrowUp' ? 'up' : 'down');
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && st.selected.length > 1 && !st.locked) {
        // capture phase: the editor's single-point Delete must not run as well
        e.preventDefault();
        void st.removeSelected();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, []);

  // The surface placement mode ends when this tab goes away or editing locks.
  const surfaceActive = surfacePick !== null && surfacePick !== undefined;
  useEffect(() => {
    if (locked && surfaceActive) cb.current.onSurfacePick?.(null);
  }, [locked, surfaceActive]);

  const renderRow = (it: Item, i: number) => {
    const v = validity.get(it.id);
    const invalid = v ? !v.valid : false;
    const src = kind === 'source' ? (it as Source) : null;
    const inMulti = multi.has(it.id);
    const eqOn = !!src && eqActive(src.eq_db);
    const nb = neighbours.get(it.id);
    return (
      <div
        key={it.id}
        data-id={it.id}
        className={`point-row ${selectedId === it.id ? 'selected' : inMulti ? 'multi' : ''} ${invalid ? 'invalid' : ''}`}
        onClick={(e) => clickRow(it.id, e)}
        style={it.group ? { borderLeftColor: it.group.color } : undefined}
      >
        <fieldset disabled={locked}>
          <div className="point-head">
            <span className={`point-dot ${kind} ${invalid ? 'invalid' : ''}`} title={src?.sound_id ? `${it.id} · track ${src.sound_id}` : it.id}>
              {i + 1}
            </span>
            <input className="name-input" value={it.name} aria-label={`${noun} name`} onChange={(e) => patch(it.id, { name: e.target.value })} />
            {it.path && (
              <span className={`badge ${pathValidity.get(it.id)?.valid === false ? 'badge-warn' : ''}`} title={pathValidity.get(it.id)?.valid === false ? 'Moves: its trajectory is invalid' : 'Moves along a trajectory'}>
                moves
              </span>
            )}
            {eqOn && (
              <span className="badge badge-eq" title={`EQ on: ${eqSummary(src!.eq_db)}`}>
                EQ
              </span>
            )}
            {aimSelect([it.id], `Aim this ${noun} at a ${otherNoun}`, 'point-to')}
            <span className="move-btns">
              <button type="button" className="icon-btn move-btn" disabled={!nb?.up} onClick={() => move(it.id, 'up')} title="Move up in the list (Alt+↑); the results stay valid" aria-label={`Move ${it.name} up`}>
                ▴
              </button>
              <button type="button" className="icon-btn move-btn" disabled={!nb?.down} onClick={() => move(it.id, 'down')} title="Move down in the list (Alt+↓); the results stay valid" aria-label={`Move ${it.name} down`}>
                ▾
              </button>
            </span>
            <Menu
              items={[
                { label: 'Move up', onClick: () => move(it.id, 'up'), disabled: !nb?.up },
                { label: 'Move down', onClick: () => move(it.id, 'down'), disabled: !nb?.down },
                { label: 'Duplicate', onClick: () => multiply([it.id], 1, [0.5, 0, 0]), disabled: full },
                { label: 'Multiply…', onClick: () => setMultiplyIds([it.id]), disabled: full },
                it.group ? { label: `Remove from “${it.group.name}”`, onClick: () => ungroup((x) => x.id === it.id) } : { label: 'Group… (Ctrl+G)', onClick: () => startGroup(inMulti && multi.size > 1 ? selected.map((x) => x.id) : [it.id]) },
                { label: 'Delete', onClick: () => removeIds([it.id]), danger: true },
              ]}
            />
          </div>
          <div className="point-fields" title={it.path ? 'Start of the trajectory: editing it moves the whole trajectory' : undefined}>
            {(['X', 'Y', 'Z'] as const).map((axis, k) => (
              <label key={axis} className="inline-field">
                <span>{axis}</span>
                <NumField
                  value={it.position[k]}
                  step={0.1}
                  digits={3}
                  ariaLabel={`${axis} (m)`}
                  onChange={(nv) => {
                    const p: Vec3 = [...it.position];
                    p[k] = nv ?? 0;
                    moveTo(it, p);
                  }}
                />
              </label>
            ))}
            <span className="unit">m</span>
          </div>
          <div className="point-fields">
            {src && (
              <select
                className="directivity"
                value={src.directivity}
                aria-label="Directivity"
                title="Directivity"
                onChange={(e) => (e.target.value === DIR_LIBRARY ? app.openDirectivities(src.directivity) : patch(it.id, { directivity: e.target.value }))}
              >
                {!directivities.some((d) => d.id === src.directivity) && <option value={src.directivity}>{src.directivity}</option>}
                <optgroup label="Presets">
                  {directivities
                    .filter((d) => d.kind !== 'imported' && d.kind !== 'odeon_local')
                    .map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                </optgroup>
                {directivities.some((d) => d.kind === 'imported') && (
                  <optgroup label="Imported">
                    {directivities
                      .filter((d) => d.kind === 'imported')
                      .map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                  </optgroup>
                )}
                {directivities.some((d) => d.kind === 'odeon_local') && (
                  <optgroup label="Odeon (local licence)">
                    {directivities
                      .filter((d) => d.kind === 'odeon_local')
                      .map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                  </optgroup>
                )}
                <option value={DIR_LIBRARY}>Directivity library…</option>
              </select>
            )}
            {src && layers && layers.length > 0 && (
              <select
                className="directivity"
                value={src.surface_layer ?? ''}
                aria-label="Radiates from"
                title="Radiates from: this point, or a layer that vibrates as a whole (a panel, a flush-mounted loudspeaker). A vibrating layer needs the wave solver only (Settings: Advanced, Wave solver only); the point then only marks the room."
                onChange={(e) => patch(it.id, { surface_layer: e.target.value || null })}
              >
                <option value="">Point source</option>
                <optgroup label="Vibrating layer (wave solver only)">
                  {layers.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </optgroup>
              </select>
            )}
            <label className="inline-field" title="Azimuth, counter-clockwise from +X (degrees)">
              <span>Az</span>
              <NumField value={it.azimuth_deg} min={-360} max={360} step={5} digits={1} onChange={(nv) => patch(it.id, { azimuth_deg: nv ?? 0 })} />
            </label>
            <label className="inline-field" title="Elevation from the XY plane, −90 (down) to +90 (up), degrees">
              <span>El</span>
              <NumField value={it.elevation_deg} min={-90} max={90} step={5} digits={1} onChange={(nv) => patch(it.id, { elevation_deg: nv ?? 0 })} />
            </label>
            {src && (
              <label className="inline-field" title="Source gain (dB)">
                <span>Gain</span>
                <NumField value={src.gain_db} min={-60} max={40} step={1} digits={1} onChange={(nv) => patch(it.id, { gain_db: nv ?? 0 })} />
                <span className="unit">dB</span>
              </label>
            )}
            {src && (
              <button
                type="button"
                className={`btn btn-xs eq-toggle ${eqOn ? 'btn-on' : ''}`}
                aria-expanded={eqOpen.has(it.id)}
                onClick={() => toggleEq(it.id)}
                title={`${eqOn ? `EQ on: ${eqSummary(src.eq_db)}` : 'EQ flat'}. ${EQ_TITLE}`}
              >
                EQ {eqOn ? <EqCurve gains={src.eq_db} width={34} height={14} compact /> : <span className="muted">flat</span>} {eqOpen.has(it.id) ? '▾' : '▸'}
              </button>
            )}
          </div>
          {src && eqOpen.has(it.id) && <SourceEqEditor value={src.eq_db} onChange={(eq) => setEq(it.id, eq)} error={saveError && /eq_db/i.test(saveError) ? saveError : null} />}
          {v && v.reasons.length > 0 && (
            <ul className="reasons">
              {v.reasons.map((r, j) => (
                <li key={j}>{r}</li>
              ))}
            </ul>
          )}
          <TrajectorySection
            item={it}
            targets={[...items, ...others].filter((x) => x.id !== it.id)}
            validity={pathValidity.get(it.id)}
            expanded={selectedId === it.id}
            selectedWaypoint={selectedId === it.id ? selectedWaypoint : null}
            onSelectWaypoint={(k) => onSelectWaypoint(it.id, k)}
            onPath={(path) => setPath(it.id, path)}
            drawing={drawingId === it.id}
            onDraw={(on) => onDraw(on ? it.id : null)}
            locked={locked}
          />
        </fieldset>
      </div>
    );
  };

  const renderSection = (s: Section) => {
    if (s.kind === 'none') return s.members.map(({ item, index }) => renderRow(item, index));
    const open = isOpen(s);
    const ids = s.members.map((m) => m.item.id);
    const invalidN = s.members.filter(({ item }) => validity.get(item.id)?.valid === false).length;
    const g = s.group;
    return (
      <div key={s.key} className={`point-group ${open ? 'open' : ''} ${g ? 'user-group' : ''}`} style={g ? { borderColor: g.color } : undefined}>
        <div className="point-group-head">
          <button className="caret" aria-expanded={open} onClick={() => toggleOpen(s)} disabled={!!q} title={open ? 'Collapse' : 'Expand'}>
            {open ? '▾' : '▸'}
          </button>
          {g ? <span className="group-swatch" style={{ background: g.color }} /> : null}
          <button className="group-title ellipsis link-like" title={g ? `${g.name}: click to select its ${noun}s` : `${s.title}: click to select its players`} onClick={() => selectIds(ids)} style={g ? { color: g.color } : undefined}>
            {g ? g.name : `♪ ${s.title}`}
          </button>
          <span className="muted small">{s.members.length}</span>
          {invalidN > 0 && (
            <span className="badge badge-warn" title={`${invalidN} invalid`}>
              invalid
            </span>
          )}
          <Menu
            items={
              g
                ? [
                    { label: 'Select members', onClick: () => selectIds(ids) },
                    { label: 'Rename / colour…', onClick: () => editGroup(g), disabled: locked },
                    { label: 'Multiply members…', onClick: () => setMultiplyIds(ids), disabled: locked || full },
                    { label: 'Ungroup (Ctrl+Shift+G)', onClick: () => ungroup((it) => it.group?.id === g.id), disabled: locked },
                    { label: 'Delete group and members…', danger: true, disabled: locked, onClick: () => void deleteGroup(g) },
                  ]
                : [
                    { label: 'Select players', onClick: () => selectIds(ids) },
                    { label: open ? 'Collapse' : 'Expand', onClick: () => toggleOpen(s) },
                    { label: 'Delete ensemble…', danger: true, disabled: locked, onClick: () => onDeleteGroup?.(s.title) },
                  ]
            }
          />
        </div>
        {open && <div className="point-group-body">{s.members.map(({ item, index }) => renderRow(item, index))}</div>}
      </div>
    );
  };

  return (
    <div className="tab-content points-tab">
      <div className="tab-toolbar">
        <span className={`small ${full ? 'text-warn' : 'muted'}`} title={`At most ${MAX_POINTS} ${noun}s per simulation`}>
          {items.length} / {MAX_POINTS} {noun}s
        </span>
        <div className="grow" />
        <button className="btn btn-xs btn-primary" disabled={locked || full} onClick={add} title={full ? limitMessage(kind, 1, room) : `Add a ${noun} (first at the model centre, next +0.5 m in x)`}>
          Add
        </button>
        <button className="btn btn-xs" disabled={locked || full} onClick={() => setMultiOpen((o) => !o)} title="A row of new points from the last one or the model centre">
          Add multiple…
        </button>
        <button className="btn btn-xs" disabled={locked || (full && !surfaceActive)} onClick={() => setSurfaceOpen((o) => !o)} title="Click surfaces in the 3D view: one point along each surface normal">
          From surface…
        </button>
        <button className="btn btn-xs" disabled={locked} onClick={() => setImportOpen(true)} title="Lines “x, y, z[, name]”">
          Import text…
        </button>
        {onAddEnsemble && (
          <button className="btn btn-xs" disabled={locked} onClick={onAddEnsemble} title="One source per player of a recorded ensemble, laid out on a stage">
            Add ensemble…
          </button>
        )}
      </div>
      {items.length > 6 && (
        <input className="points-search" type="search" placeholder={`Search ${noun}s or groups…`} value={search} onChange={(e) => setSearch(e.target.value)} aria-label={`Search ${noun}s`} />
      )}
      {full && !locked && <div className="small text-warn">{limitMessage(kind, 1, 0)}</div>}
      {selected.length > 1 && (
        <div className="selection-strip">
          <span className="small">
            <b>{selected.length}</b> selected
          </span>
          <button className="btn btn-xs" disabled={locked} onClick={() => startGroup(selected.map((x) => x.id))} title="Group the selection (Ctrl+G)">
            Group…
          </button>
          {selected.some((x) => x.group) && (
            <button className="btn btn-xs" disabled={locked} onClick={() => ungroup((it) => multi.has(it.id))} title="Ctrl+Shift+G">
              Ungroup
            </button>
          )}
          <button className="btn btn-xs" disabled={locked || full} onClick={() => setMultiplyIds(selected.map((x) => x.id))}>
            Multiply…
          </button>
          {aimSelect(
            selected.map((x) => x.id),
            `Aim the selected ${noun}s`,
          )}
          <button className="btn btn-xs btn-danger-ghost" disabled={locked} onClick={() => void removeSelected()}>
            Delete
          </button>
          <button className="btn btn-xs" onClick={() => selectIds([])} title="Clear the selection (Esc)">
            Clear
          </button>
        </div>
      )}
      {selected.length <= 1 && items.length > 1 && !locked && <div className="small muted points-hint">Ctrl-click or Shift-click rows to select several, then Group (Ctrl+G).</div>}
      {multiOpen && !locked && (
        <AddMultiple
          kind={kind}
          items={items}
          bounds={bounds}
          onCancel={() => setMultiOpen(false)}
          onAdd={(created, wanted) => {
            if (created.length < wanted) app.toast('warning', limitMessage(kind, wanted, room));
            onChange([...items, ...created]);
            setMultiOpen(false);
          }}
        />
      )}
      {multiplyIds && !locked && (
        <MultiplyPanel kind={kind} items={multiplyIds.map((id) => byId.get(id)).filter((x): x is Item => !!x)} room={room} onCancel={() => setMultiplyIds(null)} onApply={(n, step) => multiply(multiplyIds, n, step)} />
      )}
      {(surfaceOpen || surfaceActive) && !locked && (
        <div className="inline-panel">
          <div className="small">
            <b>Offset from a surface</b>: click surfaces in the 3D view; each click places a {noun} along the surface normal, inside the room when the model is closed (else on the side you look from). Room surfaces only; hide layers to reach inner ones.
          </div>
          <div className="point-fields">
            <label className="inline-field">
              <span>Offset</span>
              <NumField
                value={offset}
                min={MIN_SURFACE_OFFSET}
                max={50}
                step={0.1}
                digits={2}
                onChange={(v) => {
                  const d = Math.max(MIN_SURFACE_OFFSET, v ?? MIN_SURFACE_OFFSET);
                  setOffset(d);
                  if (surfaceActive) onSurfacePick?.(d);
                }}
                ariaLabel="Offset from the surface (m)"
              />
              <span className="unit">m (min {MIN_SURFACE_OFFSET})</span>
            </label>
          </div>
          <div className="row gap end">
            <button
              className="btn btn-xs"
              onClick={() => {
                onSurfacePick?.(null);
                setSurfaceOpen(false);
              }}
            >
              Close
            </button>
            <button className={`btn btn-xs ${surfaceActive ? 'btn-on' : 'btn-primary'}`} disabled={full && !surfaceActive} onClick={() => onSurfacePick?.(surfaceActive ? null : offset)}>
              {surfaceActive ? 'Stop picking' : 'Pick surfaces'}
            </button>
          </div>
        </div>
      )}
      {items.length === 0 && (
        <div className="empty-inline">
          No {noun} yet.{' '}
          <button className="link" disabled={locked} onClick={add}>
            Add one
          </button>{' '}
          at the model centre.
        </div>
      )}
      {q && sections.length === 0 && <div className="empty-inline">No {noun} matches “{search}”.</div>}
      <div className="point-list">{sections.map((s) => renderSection(s))}</div>
      {importOpen && (
        <ImportPointsDialog
          kind={kind}
          onClose={() => setImportOpen(false)}
          room={room}
          onImport={(pts, replace, groupName) => {
            const base = replace ? [] : items;
            let n = nextIndex(kind, base);
            const fit = MAX_POINTS - base.length;
            const group = groupName ? { ...newGroup(), name: groupName } : null;
            const created = pts.slice(0, fit).map((p) => {
              const it = makeItem(kind, n++, p.position, p.name);
              return group ? ({ ...it, group } as Item) : it;
            });
            if (created.length < pts.length) app.toast('warning', limitMessage(kind, pts.length, fit));
            onChange([...base, ...created]);
            if (created.length) selectIds(created.map((x) => x.id));
            setImportOpen(false);
          }}
        />
      )}
      {groupEdit && (
        <GroupDialog
          initial={groupEdit.group}
          count={groupEdit.ids.length}
          noun={noun}
          isNew={groupEdit.isNew}
          onClose={() => setGroupEdit(null)}
          onSave={(g) => {
            if (groupEdit.isNew) applyGroup(groupEdit.ids, g);
            else updateGroup(g);
            setGroupEdit(null);
          }}
        />
      )}
    </div>
  );
}

function AddMultiple({ kind, items, bounds, onAdd, onCancel }: { kind: PointKind; items: Item[]; bounds: Bounds | null; onAdd: (items: Item[], wanted: number) => void; onCancel: () => void }) {
  const [count, setCount] = useState(3);
  const [step, setStep] = useState<Vec3>([1, 0, 0]);
  const [from, setFrom] = useState<'last' | 'center'>(items.length ? 'last' : 'center');
  const room = MAX_POINTS - items.length;
  const create = () => {
    const n0 = nextIndex(kind, items);
    const origin: Vec3 = from === 'last' && items.length ? items[items.length - 1].position : bounds ? bounds.center : [0, 0, 1.2];
    const offset = from === 'last' && items.length ? 1 : 0;
    const out: Item[] = [];
    for (let i = 0; i < Math.min(count, room); i++) {
      const k = i + offset;
      out.push(makeItem(kind, n0 + i, [origin[0] + step[0] * k, origin[1] + step[1] * k, origin[2] + step[2] * k]));
    }
    onAdd(out, count);
  };
  return (
    <div className="inline-panel">
      <div className="small">
        <b>Add multiple</b> new {kind}s in a row
      </div>
      <div className="point-fields">
        <label className="inline-field">
          <span>Count</span>
          <NumField value={count} min={1} max={MAX_POINTS} step={1} digits={0} onChange={(v) => setCount(Math.round(v ?? 1))} />
        </label>
        <label className="inline-field">
          <span>From</span>
          <select value={from} onChange={(e) => setFrom(e.target.value as 'last' | 'center')}>
            <option value="last" disabled={!items.length}>
              last {kind}
            </option>
            <option value="center">model centre</option>
          </select>
        </label>
      </div>
      <StepFields step={step} setStep={setStep} label="Spacing" />
      {count > room && <div className="small text-warn">{limitMessage(kind, count, room)}</div>}
      <div className="row gap end">
        <button className="btn btn-xs" onClick={onCancel}>
          Cancel
        </button>
        <button className="btn btn-xs btn-primary" disabled={room <= 0} onClick={create}>
          Add {Math.min(count, room)}
        </button>
      </div>
    </div>
  );
}

function StepFields({ step, setStep, label }: { step: Vec3; setStep: (s: Vec3) => void; label: string }) {
  return (
    <div className="point-fields">
      <span className="small muted">{label}</span>
      {(['X', 'Y', 'Z'] as const).map((axis, k) => (
        <label key={axis} className="inline-field">
          <span>{axis}</span>
          <NumField
            value={step[k]}
            step={0.1}
            digits={3}
            ariaLabel={`${label} ${axis} (m)`}
            onChange={(v) => {
              const s: Vec3 = [...step];
              s[k] = v ?? 0;
              setStep(s);
            }}
          />
        </label>
      ))}
      <span className="unit">m</span>
    </div>
  );
}

/** N copies of each point (orientation, directivity, gain, EQ, group and trajectory kept), stepped in X/Y/Z. */
function MultiplyPanel({ kind, items, room, onApply, onCancel }: { kind: PointKind; items: Item[]; room: number; onApply: (n: number, step: Vec3) => void; onCancel: () => void }) {
  const [count, setCount] = useState(3);
  const [step, setStep] = useState<Vec3>([1, 0, 0]);
  const wanted = count * items.length;
  return (
    <div className="inline-panel">
      <div className="small">
        <b>Multiply</b> {items.length === 1 ? <>“{items[0].name}”</> : `${items.length} ${kind}s`}: copies keep the orientation{kind === 'source' ? ', directivity, gain, EQ' : ''}, group and trajectory.
      </div>
      <div className="point-fields">
        <label className="inline-field">
          <span>Copies</span>
          <NumField value={count} min={1} max={MAX_POINTS} step={1} digits={0} onChange={(v) => setCount(Math.max(1, Math.round(v ?? 1)))} ariaLabel="Number of copies" />
        </label>
        {items.length > 1 && <span className="small muted">each ({wanted} new)</span>}
      </div>
      <StepFields step={step} setStep={setStep} label="Step" />
      {wanted > room && <div className="small text-warn">{limitMessage(kind, wanted, room)}</div>}
      {step.every((v) => v === 0) && <div className="small text-warn">A zero step puts every copy on top of its original.</div>}
      <div className="row gap end">
        <button className="btn btn-xs" onClick={onCancel}>
          Cancel
        </button>
        <button className="btn btn-xs btn-primary" disabled={room <= 0 || !items.length} onClick={() => onApply(count, step)}>
          Create {Math.min(wanted, Math.max(room, 0))}
        </button>
      </div>
    </div>
  );
}

function GroupDialog({ initial, count, noun, isNew, onSave, onClose }: { initial: PointGroup; count: number; noun: string; isNew: boolean; onSave: (g: PointGroup) => void; onClose: () => void }) {
  const [name, setName] = useState(initial.name);
  const [color, setColor] = useState(initial.color);
  const ok = name.trim().length > 0;
  const save = () => ok && onSave({ ...initial, name: name.trim(), color });
  return (
    <Modal title={isNew ? `Group ${count} ${noun}${count === 1 ? '' : 's'}` : `Group “${initial.name}”`} onClose={onClose} width={400}>
      <div className="modal-body">
        <div className="form-grid compact">
          <label htmlFor="group-name">Name</label>
          <input
            id="group-name"
            value={name}
            autoFocus
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') save();
            }}
          />
          <label>Colour</label>
          <div className="row gap wrap group-colors" role="radiogroup" aria-label="Group colour">
            {GROUP_COLORS.map((c) => (
              <button key={c} type="button" role="radio" aria-checked={c === color} className={`group-color ${c === color ? 'on' : ''}`} style={{ background: c }} onClick={() => setColor(c)} title={c} />
            ))}
            <input type="color" value={color} onChange={(e) => setColor(e.target.value)} aria-label="Custom colour" title="Custom colour" />
          </div>
        </div>
        <div className="small muted">Groups are listed together and can be collapsed, aimed, multiplied or deleted at once.</div>
      </div>
      <div className="modal-footer">
        <div className="grow" />
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" disabled={!ok} onClick={save}>
          {isNew ? 'Group' : 'Save'}
        </button>
      </div>
    </Modal>
  );
}

function ImportPointsDialog({ kind, onClose, onImport, room }: { kind: PointKind; onClose: () => void; onImport: (pts: { name?: string; position: Vec3 }[], replace: boolean, groupName: string | null) => void; room: number }) {
  const [text, setText] = useState('');
  const [replace, setReplace] = useState(false);
  const [asGroup, setAsGroup] = useState(false);
  const [groupName, setGroupName] = useState('Imported');
  const parsed = parsePointsText(text);
  const limit = replace ? MAX_POINTS : room;
  return (
    <Modal title={`Import ${kind}s from text`} onClose={onClose} width={520}>
      <div className="modal-body">
        <p className="muted small">One point per line: “x, y, z” in metres, optionally followed by a name (“2.0, 3.5, 1.5, Stage left”) or preceded by one. Comma, semicolon, tab or space separated; # starts a comment.</p>
        <textarea rows={10} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} autoFocus placeholder={'2.0, 3.0, 1.5, Row A seat 1\n4.0, 3.0, 1.5, Row A seat 2'} />
        <label className="check">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} /> Replace the existing {kind}s
        </label>
        <label className="check">
          <input type="checkbox" checked={asGroup} onChange={(e) => setAsGroup(e.target.checked)} /> Put them in a new group
          {asGroup && <input className="group-name-inline" value={groupName} onChange={(e) => setGroupName(e.target.value)} aria-label="Group name" />}
        </label>
        <div className="small">
          {parsed.points.length} point{parsed.points.length === 1 ? '' : 's'} recognised
          {parsed.points.length > limit && <span className="text-warn"> · {limitMessage(kind, parsed.points.length, limit)}</span>}
        </div>
        {parsed.errors.slice(0, 6).map((e, i) => (
          <div key={i} className="small text-err">
            {e}
          </div>
        ))}
      </div>
      <div className="modal-footer">
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" disabled={!parsed.points.length || limit <= 0} onClick={() => onImport(parsed.points, replace, asGroup && groupName.trim() ? groupName.trim() : null)}>
          Import
        </button>
      </div>
    </Modal>
  );
}
