import { useCallback, useEffect, useRef, useState } from 'react';
import { api, editableOf, errorMessage, type Simulation } from '../../api';
import { Modal, Spinner } from '../../components/ui';
import { fmtDate } from '../../lib/format';
import { href, navigate } from '../../lib/router';
import type { SimulationStore } from './useSimulationStore';

export type EditChoice = 'modify' | 'duplicate' | 'revert';
/** sim: an edit of the simulation itself; objects: a geometry edit (no Duplicate); server: a server-side edit (ensemble). */
export type EditKind = 'sim' | 'objects' | 'server';

/** Same value whatever the key order (the server compares dicts). */
const sortedKeys = (v: unknown): unknown =>
  Array.isArray(v)
    ? v.map(sortedKeys)
    : v && typeof v === 'object'
      ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortedKeys((v as Record<string, unknown>)[k])]))
      : v;

/** Points as the server compares them: by id (order aside), without `path`, a flat `eq_db` dropped. */
const strip = (items: { id: string; eq_db?: unknown }[]) =>
  JSON.stringify(
    sortedKeys(
      [...items]
        .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
        .map((x) => ({ ...x, path: undefined, eq_db: Array.isArray(x.eq_db) && x.eq_db.some((g) => g !== 0) ? x.eq_db : undefined })),
    ),
  );

/**
 * Whether saving `after` would make the server mark the results stale (API "Simulations"):
 * layers, sources and receivers (their `path` aside: a trajectory edit keeps the results
 * unless the start point moves, which moves `position`; their order aside: the server
 * compares them by id), receiver grids and settings. The simulation's name and
 * description do not count.
 */
export function makesStale(before: Simulation, after: Simulation): boolean {
  if (before.layers !== after.layers && JSON.stringify(before.layers) !== JSON.stringify(after.layers)) return true;
  if (before.sources !== after.sources && strip(before.sources) !== strip(after.sources)) return true;
  if (before.receivers !== after.receivers && strip(before.receivers) !== strip(after.receivers)) return true;
  if (before.receiver_grids !== after.receiver_grids && JSON.stringify(before.receiver_grids ?? []) !== JSON.stringify(after.receiver_grids ?? [])) return true;
  if (before.settings !== after.settings && JSON.stringify(before.settings) !== JSON.stringify(after.settings)) return true;
  return false;
}

function copyName(name: string, existing: string[]): string {
  const used = new Set(existing);
  const base = `${name} - Copy`;
  if (!used.has(base)) return base;
  for (let k = 2; ; k++) if (!used.has(`${base} ${k}`)) return `${base} ${k}`;
}

/**
 * The "Modify / Duplicate / Revert" prompt: the first change to a simulation whose
 * results are up to date asks what it should do. Simulation edits are held by the
 * store meanwhile (applied on screen, not saved).
 */
export function useModifyPrompt({
  store,
  siblingNames,
  onError,
  onTreeChanged,
}: {
  store: SimulationStore;
  siblingNames: string[];
  onError: (msg: string) => void;
  onTreeChanged: () => Promise<unknown>;
}) {
  const [ask, setAsk] = useState<{ kind: EditKind; resolve: (c: EditChoice) => void } | null>(null);
  const [busy, setBusy] = useState(false);
  /** last_run for which "Modify" was chosen: no more prompts until the next run. */
  const accepted = useRef<string | null>(null);
  const simRef = useRef(store.sim);
  simRef.current = store.sim;
  const asking = useRef(false);

  const needsPrompt = useCallback((s: Simulation | null = simRef.current) => {
    return !!s && s.status === 'done' && !s.results_stale && !!s.last_run && accepted.current !== s.last_run;
  }, []);

  /** Ask once; resolves with the choice ('revert' if dismissed). */
  const choose = useCallback((kind: EditKind) => {
    if (asking.current) return Promise.resolve<EditChoice>('revert');
    asking.current = true;
    return new Promise<EditChoice>((resolve) =>
      setAsk({
        kind,
        resolve: (c) => {
          asking.current = false;
          setAsk(null);
          if (c === 'modify' && simRef.current) accepted.current = simRef.current.last_run;
          resolve(c);
        },
      }),
    );
  }, []);

  /** Create "<name> - Copy" from the stored simulation; `apply` changes the copy before it opens. */
  const duplicateWith = useCallback(
    async (apply: (copy: Simulation) => Promise<void>) => {
      const s = simRef.current;
      if (!s) return false;
      setBusy(true);
      try {
        const created = await api.createSimulation(s.geometry_id, copyName(s.name, siblingNames), s.id);
        await apply(created);
        await onTreeChanged();
        navigate(href.editor(created.id));
        return true;
      } catch (e) {
        onError(`Duplicate: ${errorMessage(e)}`);
        return false;
      } finally {
        setBusy(false);
      }
    },
    [siblingNames, onError, onTreeChanged],
  );

  const { setGuard, release } = store;
  useEffect(() => {
    setGuard((before, after) => {
      if (!needsPrompt(before) || !makesStale(before, after)) return false;
      void choose('sim').then(async (c) => {
        if (c === 'modify') release(true);
        else if (c === 'revert') release(false);
        else {
          // The copy gets the edited state; this simulation goes back to its stored one.
          const edited = simRef.current;
          await duplicateWith(async (copy) => {
            if (edited) await api.saveSimulation(copy.id, { ...editableOf(edited), name: copy.name, description: copy.description });
          });
          release(false);
        }
      });
      return true;
    });
    return () => setGuard(null);
  }, [setGuard, release, needsPrompt, choose, duplicateWith]);

  const dialog = ask && store.sim && (
    <ModifyDialog kind={ask.kind} sim={store.sim} busy={busy} onChoose={(c) => !busy && ask.resolve(c)} copyName={copyName(store.sim.name, siblingNames)} />
  );

  return { needsPrompt, choose, duplicateWith, dialog };
}

function ModifyDialog({
  kind,
  sim,
  busy,
  onChoose,
  copyName: copy,
}: {
  kind: EditKind;
  sim: Simulation;
  busy: boolean;
  onChoose: (c: EditChoice) => void;
  copyName: string;
}) {
  return (
    <Modal title="This simulation has results" onClose={() => onChoose('revert')} width={520} closeOnBackdrop={false}>
      <div className="modal-body modify-prompt">
        <p>
          <b>{sim.name}</b> was run {fmtDate(sim.last_run)} and its results are up to date. What should this change do?
        </p>
        {kind === 'objects' && (
          <p className="small muted">
            Objects belong to the geometry: every simulation of it that has results becomes stale. To compare with and without the change, revert it, then use
            “Duplicate geometry…” in the ⋯ menu.
          </p>
        )}
        <div className="modify-choices">
          <button className="modify-choice" disabled={busy} onClick={() => onChoose('modify')} autoFocus>
            <b>Modify</b>
            <span className="small muted">Keep editing this simulation. Its results are kept but marked stale until the next run.</span>
          </button>
          {kind !== 'objects' && (
            <button className="modify-choice" disabled={busy} onClick={() => onChoose('duplicate')}>
              <b>Duplicate {busy && <Spinner size={11} />}</b>
              <span className="small muted">
                Create “{copy}” with this change and open it. This simulation and its results stay as they are.
              </span>
            </button>
          )}
          <button className="modify-choice" disabled={busy} onClick={() => onChoose('revert')}>
            <b>{kind === 'server' ? 'Cancel' : 'Revert'}</b>
            <span className="small muted">{kind === 'server' ? 'Leave this simulation as it is.' : 'Undo the change.'}</span>
          </button>
        </div>
      </div>
    </Modal>
  );
}
