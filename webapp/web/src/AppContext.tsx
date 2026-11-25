import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  api,
  errorMessage,
  type Directivity,
  type Health,
  type MaterialsResponse,
  type Project,
} from './api';
import { DirectivityLibrary } from './components/DirectivityLibrary';
import { Modal } from './components/ui';

type ToastKind = 'error' | 'warning' | 'info' | 'success';
interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ConfirmOptions {
  title: string;
  message: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
}

interface PromptOptions {
  title: string;
  label: string;
  initial?: string;
  confirmLabel?: string;
  allowEmpty?: boolean;
}

interface AppState {
  tree: Project[] | null;
  treeError: string | null;
  refreshTree: () => Promise<Project[] | null>;
  health: Health | null;
  healthError: string | null;
  materials: MaterialsResponse | null;
  materialsError: string | null;
  reloadMaterials: () => Promise<MaterialsResponse | null>;
  directivities: Directivity[];
  reloadDirectivities: () => Promise<Directivity[] | null>;
  /** Opens the directivity library (presets, imports), optionally on one entry. */
  openDirectivities: (id?: string | null) => void;
  toast: (kind: ToastKind, message: string) => void;
  confirm: (opts: ConfirmOptions) => Promise<boolean>;
  prompt: (opts: PromptOptions) => Promise<string | null>;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp outside AppProvider');
  return v;
}

type DialogState =
  | { type: 'confirm'; opts: ConfirmOptions; resolve: (v: boolean) => void }
  | { type: 'prompt'; opts: PromptOptions; resolve: (v: string | null) => void };

export function AppProvider({ children }: { children: ReactNode }) {
  const [tree, setTree] = useState<Project[] | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [materials, setMaterials] = useState<MaterialsResponse | null>(null);
  const [materialsError, setMaterialsError] = useState<string | null>(null);
  const [directivities, setDirectivities] = useState<Directivity[]>([]);
  const [dirLibrary, setDirLibrary] = useState<{ id: string | null } | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [dialog, setDialog] = useState<DialogState | null>(null);
  const toastId = useRef(0);

  const toast = useCallback((kind: ToastKind, message: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t.slice(-4), { id, kind, message }]);
    const ttl = kind === 'error' ? 9000 : 4500;
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), ttl);
  }, []);

  const refreshTree = useCallback(async () => {
    try {
      const t = await api.listProjects();
      setTree(t);
      setTreeError(null);
      return t;
    } catch (e) {
      setTreeError(errorMessage(e));
      return null;
    }
  }, []);

  const reloadMaterials = useCallback(async () => {
    try {
      const m = await api.getMaterials();
      setMaterials(m);
      setMaterialsError(null);
      return m;
    } catch (e) {
      setMaterialsError(errorMessage(e));
      return null;
    }
  }, []);

  const reloadDirectivities = useCallback(async () => {
    try {
      const d = await api.getDirectivities();
      setDirectivities(d);
      return d;
    } catch (e) {
      toast('warning', `Directivities unavailable: ${errorMessage(e)}`);
      return null;
    }
  }, [toast]);
  const openDirectivities = useCallback((id?: string | null) => setDirLibrary({ id: id ?? null }), []);

  useEffect(() => {
    void refreshTree();
    void reloadMaterials();
    void reloadDirectivities();
    const poll = async () => {
      try {
        setHealth(await api.health());
        setHealthError(null);
      } catch (e) {
        setHealthError(errorMessage(e));
      }
    };
    void poll();
    const t = window.setInterval(poll, 20000);
    return () => window.clearInterval(t);
  }, [refreshTree, reloadMaterials, reloadDirectivities]);

  const confirm = useCallback(
    (opts: ConfirmOptions) => new Promise<boolean>((resolve) => setDialog({ type: 'confirm', opts, resolve })),
    [],
  );
  const prompt = useCallback(
    (opts: PromptOptions) => new Promise<string | null>((resolve) => setDialog({ type: 'prompt', opts, resolve })),
    [],
  );

  const value = useMemo<AppState>(
    () => ({
      tree,
      treeError,
      refreshTree,
      health,
      healthError,
      materials,
      materialsError,
      reloadMaterials,
      directivities,
      reloadDirectivities,
      openDirectivities,
      toast,
      confirm,
      prompt,
    }),
    [tree, treeError, refreshTree, health, healthError, materials, materialsError, reloadMaterials, directivities, reloadDirectivities, openDirectivities, toast, confirm, prompt],
  );

  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            <span className="toast-msg">{t.message}</span>
            <button className="icon-btn" aria-label="Dismiss" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>
              ×
            </button>
          </div>
        ))}
      </div>
      {dirLibrary && <DirectivityLibrary initialId={dirLibrary.id} onClose={() => setDirLibrary(null)} />}
      {dialog?.type === 'confirm' && (
        <ConfirmDialog
          opts={dialog.opts}
          onDone={(v) => {
            dialog.resolve(v);
            setDialog(null);
          }}
        />
      )}
      {dialog?.type === 'prompt' && (
        <PromptDialog
          opts={dialog.opts}
          onDone={(v) => {
            dialog.resolve(v);
            setDialog(null);
          }}
        />
      )}
    </Ctx.Provider>
  );
}

function ConfirmDialog({ opts, onDone }: { opts: ConfirmOptions; onDone: (v: boolean) => void }) {
  return (
    <Modal title={opts.title} onClose={() => onDone(false)} width={420}>
      <div className="modal-body">{opts.message}</div>
      <div className="modal-footer">
        <button className="btn" onClick={() => onDone(false)}>
          Cancel
        </button>
        <button className={`btn ${opts.danger ? 'btn-danger' : 'btn-primary'}`} autoFocus onClick={() => onDone(true)}>
          {opts.confirmLabel ?? 'OK'}
        </button>
      </div>
    </Modal>
  );
}

function PromptDialog({ opts, onDone }: { opts: PromptOptions; onDone: (v: string | null) => void }) {
  const [value, setValue] = useState(opts.initial ?? '');
  const submit = () => {
    const v = value.trim();
    if (v || opts.allowEmpty) onDone(v);
  };
  return (
    <Modal title={opts.title} onClose={() => onDone(null)} width={420}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div className="modal-body">
          <label className="field">
            <span>{opts.label}</span>
            <input autoFocus value={value} onChange={(e) => setValue(e.target.value)} onFocus={(e) => e.target.select()} />
          </label>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={() => onDone(null)}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!opts.allowEmpty && !value.trim()}>
            {opts.confirmLabel ?? 'OK'}
          </button>
        </div>
      </form>
    </Modal>
  );
}
