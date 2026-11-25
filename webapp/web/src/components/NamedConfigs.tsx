import { useState } from 'react';
import type { UiStateKey } from '../api';
import { fmtDate } from '../lib/format';
import { clearRestoredNote, fmtKb, restoredNote, useNamedConfigs, type NamedConfig, type Outcome } from '../lib/namedConfigs';
import { Modal } from './ui';

/**
 * Named configurations of a page (results view, auralizer set-up): a button showing the
 * active one, the note left by the last restore, and the dialog that manages them.
 * `onRestored` is called after the saved document changed (restore, undo): the page then
 * starts again from it.
 */
export function NamedConfigsButton({
  simId,
  docKey,
  what,
  contents,
  onRestored,
  restoreEffect,
}: {
  simId: string;
  docKey: UiStateKey;
  /** "view" or "set-up". */
  what: string;
  /** What a configuration holds, for the dialog. */
  contents: string;
  onRestored: () => void;
  /** What restoring does to the page besides its settings (e.g. data read again). */
  restoreEffect?: string;
}) {
  const nc = useNamedConfigs(simId, docKey, what);
  const [open, setOpen] = useState(false);
  const [, setTick] = useState(0);
  const note = restoredNote(simId, docKey);
  const dismiss = () => {
    clearRestoredNote(simId, docKey);
    setTick((t) => t + 1);
  };
  const doUndo = () => {
    const r = nc.undo();
    if (r.ok) onRestored();
    setTick((t) => t + 1);
  };
  const label = nc.active ? nc.active.name : `Unsaved ${what}`;
  return (
    <span className="named-configs">
      <button
        type="button"
        className="btn btn-sm named-configs-btn"
        onClick={() => setOpen(true)}
        title={
          nc.active
            ? `Configuration “${nc.active.name}”${nc.modified ? `, changed since it was saved or restored (the current ${what} is still saved automatically)` : ''}. Open the named configurations.`
            : `The current ${what} is saved automatically, under no name. Open the named configurations to save it under a name or restore one.`
        }
      >
        <span className="muted">Configuration:</span> <span className="named-configs-name">{label}</span>
        {nc.modified && <span className="badge badge-warn">modified</span>}
        {nc.state.configs.length > 0 && <span className="muted small"> · {nc.state.configs.length}</span>}
      </button>
      {note && (
        <span className="named-configs-note small" role="status">
          {note}
          {nc.state.undo && (
            <button type="button" className="btn btn-xs" onClick={doUndo} title={`Put back the ${what} you had before restoring “${nc.state.undo.restored}”`}>
              Undo
            </button>
          )}
          <button type="button" className="icon-btn" aria-label="Dismiss" onClick={dismiss}>
            ×
          </button>
        </span>
      )}
      {open && (
        <NamedConfigsDialog
          nc={nc}
          what={what}
          contents={contents}
          restoreEffect={restoreEffect}
          onClose={() => setOpen(false)}
          onRestored={() => {
            setOpen(false);
            setTick((t) => t + 1);
            onRestored();
          }}
        />
      )}
    </span>
  );
}

type Nc = ReturnType<typeof useNamedConfigs>;

function NamedConfigsDialog({
  nc,
  what,
  contents,
  restoreEffect,
  onClose,
  onRestored,
}: {
  nc: Nc;
  what: string;
  contents: string;
  restoreEffect?: string;
  onClose: () => void;
  onRestored: () => void;
}) {
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  /** Restore refused because the replaced view does not fit: offer to restore without keeping it. */
  const [noRoom, setNoRoom] = useState<string | null>(null);
  const report = (r: Outcome) => {
    setError(r.ok ? null : r.error);
    return r.ok;
  };
  const save = () => {
    if (report(nc.save(name))) setName('');
  };
  const restore = (c: NamedConfig, keep = true) => {
    const r = nc.restore(c.id, keep);
    if (!r.ok && r.tooLarge && keep) {
      setNoRoom(c.id);
      setError(r.error);
      return;
    }
    if (report(r)) onRestored();
  };
  const usage = nc.bytes / nc.limit;
  return (
    <Modal title="Named configurations" onClose={onClose} width={640}>
      <div className="modal-body named-configs-dialog">
        <div className="small muted">
          A configuration is a named copy of this page’s {what}: {contents}. The current {what} is saved automatically, apart from them. Restoring a configuration{' '}
          <b>replaces the current {what}</b>; the {what} it replaces is kept until the next restore, and <b>Undo</b> brings it back.
          {restoreEffect ? ` ${restoreEffect}` : ''}
        </div>
        <div className="row gap named-configs-save">
          <input
            type="text"
            value={name}
            placeholder="Name"
            maxLength={80}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && save()}
            aria-label="Name of the new configuration"
          />
          <button type="button" className="btn btn-sm btn-primary" disabled={!name.trim()} onClick={save}>
            Save the current {what}
          </button>
        </div>
        {error && (
          <div className="small text-warn" role="alert">
            {error}
          </div>
        )}
        {noRoom && (
          <div className="row gap">
            <button
              type="button"
              className="btn btn-xs btn-danger-ghost"
              onClick={() => {
                const c = nc.state.configs.find((x) => x.id === noRoom);
                setNoRoom(null);
                if (c) restore(c, false);
              }}
            >
              Restore without keeping the current {what}
            </button>
            <button type="button" className="btn btn-xs" onClick={() => {
                setNoRoom(null);
                setError(null);
              }}>
              Cancel
            </button>
          </div>
        )}
        {nc.state.configs.length === 0 ? (
          <div className="small muted named-configs-empty">No named configuration yet.</div>
        ) : (
          <table className="table compact named-configs-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Saved</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {nc.state.configs.map((c) => {
                const isActive = nc.active?.id === c.id;
                return (
                  <tr key={c.id} className={isActive ? 'active' : ''}>
                    <td>
                      {renaming?.id === c.id ? (
                        <input
                          type="text"
                          value={renaming.name}
                          maxLength={80}
                          autoFocus
                          onChange={(e) => setRenaming({ id: c.id, name: e.target.value })}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && report(nc.rename(c.id, renaming.name))) setRenaming(null);
                          }}
                          aria-label="New name"
                        />
                      ) : (
                        <>
                          <span className="named-configs-name">{c.name}</span>
                          {isActive && (
                            <span className={`badge ${nc.modified ? 'badge-warn' : ''}`} title={nc.modified ? `The current ${what} differs from this configuration` : `The current ${what} is this configuration`}>
                              {nc.modified ? 'active, modified' : 'active'}
                            </span>
                          )}
                        </>
                      )}
                    </td>
                    <td className="small muted nowrap">{fmtDate(c.saved)}</td>
                    <td className="named-configs-actions">
                      {renaming?.id === c.id ? (
                        <>
                          <button type="button" className="btn btn-xs" onClick={() => report(nc.rename(c.id, renaming.name)) && setRenaming(null)}>
                            OK
                          </button>
                          <button type="button" className="btn btn-xs" onClick={() => setRenaming(null)}>
                            Cancel
                          </button>
                        </>
                      ) : deleting === c.id ? (
                        <>
                          <span className="small">Delete?</span>
                          <button
                            type="button"
                            className="btn btn-xs btn-danger-ghost"
                            onClick={() => {
                              nc.remove(c.id);
                              setDeleting(null);
                            }}
                          >
                            Delete
                          </button>
                          <button type="button" className="btn btn-xs" onClick={() => setDeleting(null)}>
                            Keep
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            type="button"
                            className="btn btn-xs btn-primary"
                            onClick={() => restore(c)}
                            title={`Replace the current ${what} by “${c.name}” (the current ${what} is kept for Undo)`}
                          >
                            Restore
                          </button>
                          <button type="button" className="btn btn-xs" onClick={() => report(nc.update(c.id))} title={`Store the current ${what} in “${c.name}”, in place of what it held`}>
                            Update
                          </button>
                          <button type="button" className="btn btn-xs" onClick={() => setRenaming({ id: c.id, name: c.name })}>
                            Rename
                          </button>
                          <button type="button" className="btn btn-xs btn-danger-ghost" onClick={() => setDeleting(c.id)}>
                            Delete
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {nc.state.undo && (
          <div className="row gap small named-configs-undo">
            <span className="grow">
              Kept: the {what} you had before restoring “{nc.state.undo.restored}” ({fmtDate(nc.state.undo.at)}).
            </span>
            <button
              type="button"
              className="btn btn-xs"
              onClick={() => {
                if (report(nc.undo())) onRestored();
              }}
            >
              Undo the restore
            </button>
            <button type="button" className="btn btn-xs btn-ghost" onClick={nc.dropUndo} title="Forget the kept view (frees its room in the size limit)">
              Forget it
            </button>
          </div>
        )}
        <div className={`small ${usage > 0.9 ? 'text-warn' : 'muted'}`}>
          Stored on the server with this simulation: {fmtKb(nc.bytes)} of {fmtKb(nc.limit)}
          {usage > 0.9 ? ': nearly full, delete configurations to save new ones.' : '.'}
        </div>
      </div>
      <div className="modal-footer">
        <button type="button" className="btn btn-sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}
