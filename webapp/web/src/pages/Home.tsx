import { useEffect, useMemo, useState } from 'react';
import { api, errorMessage, type GeometrySummary, type Project, type RecentSimulation, type SimulationSummary, type Space } from '../api';
import { useApp } from '../AppContext';
import { TopBar } from '../components/TopBar';
import { EmptyState, ErrorBox, Menu, Spinner, StatusBadge } from '../components/ui';
import { forgetGeometry } from '../lib/cache';
import { downloadBlob } from '../lib/export';
import { fmtDate, fmtDuration, plural } from '../lib/format';
import { href, navigate } from '../lib/router';
import { nextName } from '../lib/tree';
import { ImportWizard, type ImportPreset } from './ImportWizard';

export function Home() {
  const app = useApp();
  const { tree, treeError, refreshTree } = app;
  const [query, setQuery] = useState('');
  const [importPreset, setImportPreset] = useState<ImportPreset | null>(null);
  const [recent, setRecent] = useState<RecentSimulation[] | null>(null);
  const [recentError, setRecentError] = useState<string | null>(null);

  const loadRecent = async () => {
    try {
      setRecent(await api.recentSimulations());
      setRecentError(null);
    } catch (e) {
      setRecentError(errorMessage(e));
    }
  };

  useEffect(() => {
    void refreshTree();
    void loadRecent();
  }, [refreshTree]);

  const reload = async () => {
    await Promise.all([refreshTree(), loadRecent()]);
  };

  const filtered = useMemo(() => filterTree(tree ?? [], query), [tree, query]);

  const act = async (fn: () => Promise<unknown>, okMsg?: string) => {
    try {
      await fn();
      if (okMsg) app.toast('success', okMsg);
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
    await reload();
  };

  const newProject = async () => {
    const name = await app.prompt({ title: 'New project', label: 'Project name', initial: nextName('Project', (tree ?? []).map((p) => p.name)), confirmLabel: 'Create' });
    if (name) await act(() => api.createProject(name, ''));
  };

  return (
    <div className="app-shell">
      <TopBar />
      <div className="home">
        <main className="home-main">
          <div className="home-toolbar">
            <h1>Projects</h1>
            <input
              className="search"
              type="search"
              placeholder="Search projects, spaces, geometries…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="grow" />
            <button className="btn" onClick={() => app.openDirectivities()} title="Source directivities: presets and imported ones (.json, .csv, .sofa)">
              Directivities
            </button>
            <button className="btn" onClick={newProject}>
              New project
            </button>
            <button className="btn btn-primary" onClick={() => setImportPreset({})}>
              Import model
            </button>
          </div>
          {treeError && <ErrorBox message={`Could not load projects: ${treeError}`} onRetry={() => void refreshTree()} />}
          {!tree && !treeError && (
            <div className="loading-row">
              <Spinner /> Loading projects…
            </div>
          )}
          {tree && tree.length === 0 && (
            <EmptyState title="No projects yet">
              Import an <code>.obj</code> model to create your first project, space and geometry.
              <div className="empty-actions">
                <button className="btn btn-primary" onClick={() => setImportPreset({})}>
                  Import model
                </button>
                <button className="btn" onClick={newProject}>
                  New empty project
                </button>
              </div>
            </EmptyState>
          )}
          {tree && tree.length > 0 && filtered.length === 0 && <EmptyState title={`Nothing matches “${query}”`} />}
          {filtered.map((p) => (
            <ProjectCard key={p.id} project={p} act={act} onImport={(preset) => setImportPreset(preset)} forceOpen={!!query} />
          ))}
        </main>
        <aside className="home-side">
          <h2>Recent simulations</h2>
          {recentError && <ErrorBox message={recentError} onRetry={() => void loadRecent()} />}
          {!recent && !recentError && <Spinner />}
          {recent && recent.length === 0 && <div className="muted small">No simulation has been run yet.</div>}
          <ul className="recent-list">
            {recent?.map((r) => (
              <li key={r.id} className="recent-item">
                <a className="recent-name" href={href.editor(r.id)} title="Open in the editor">
                  {r.geometry_name} › {r.name}
                </a>
                <div className="recent-path muted small">
                  {r.project_name} / {r.space_name}
                </div>
                <div className="recent-meta">
                  <StatusBadge status={r.status} stale={r.results_stale} />
                  <span className="muted small">{fmtDate(r.last_run)}</span>
                  {r.duration_s !== null && <span className="muted small">· {fmtDuration(r.duration_s)}</span>}
                </div>
                <div className="recent-links">
                  <a className="icon-link" href={href.editor(r.id)} title="Editor">
                    Edit
                  </a>
                  <a className={`icon-link ${r.last_run ? '' : 'disabled'}`} href={href.results(r.id)} title="Results">
                    Results
                  </a>
                  <a className={`icon-link ${r.last_run ? '' : 'disabled'}`} href={href.auralizer(r.id)} title="Auralizer">
                    Listen
                  </a>
                </div>
              </li>
            ))}
          </ul>
        </aside>
      </div>
      {importPreset && (
        <ImportWizard
          preset={importPreset}
          onClose={() => {
            setImportPreset(null);
            void reload();
          }}
        />
      )}
    </div>
  );
}

function filterTree(tree: Project[], query: string): Project[] {
  const q = query.trim().toLowerCase();
  if (!q) return tree;
  const has = (s: string) => s.toLowerCase().includes(q);
  const out: Project[] = [];
  for (const p of tree) {
    if (has(p.name)) {
      out.push(p);
      continue;
    }
    const spaces = (p.spaces ?? []).filter(
      (s) => has(s.name) || (s.geometries ?? []).some((g) => has(g.name) || (g.simulations ?? []).some((m) => has(m.name))),
    );
    if (spaces.length) out.push({ ...p, spaces });
  }
  return out;
}

type Act = (fn: () => Promise<unknown>, okMsg?: string) => Promise<void>;

function ProjectCard({
  project,
  act,
  onImport,
  forceOpen,
}: {
  project: Project;
  act: Act;
  onImport: (p: ImportPreset) => void;
  forceOpen: boolean;
}) {
  const app = useApp();
  const [open, setOpen] = useState(true);
  const spaces = project.spaces ?? [];
  const isOpen = open || forceOpen;
  return (
    <section className="project-card">
      <header className="project-head">
        <button className="caret" aria-expanded={isOpen} onClick={() => setOpen((o) => !o)}>
          {isOpen ? '▾' : '▸'}
        </button>
        <h2>{project.name}</h2>
        {project.description && <span className="muted small ellipsis">{project.description}</span>}
        <span className="muted small">{plural(spaces.length, 'space')}</span>
        <div className="grow" />
        <button
          className="btn btn-sm"
          onClick={async () => {
            const name = await app.prompt({ title: `New space in ${project.name}`, label: 'Space name', initial: nextName('Space', spaces.map((s) => s.name)), confirmLabel: 'Create' });
            if (name) await act(() => api.createSpace(project.id, name, ''));
          }}
        >
          New space
        </button>
        <button className="btn btn-sm" onClick={() => onImport({ projectId: project.id })}>
          Import here
        </button>
        <Menu
          items={[
            {
              label: 'Rename…',
              onClick: async () => {
                const name = await app.prompt({ title: 'Rename project', label: 'Name', initial: project.name, confirmLabel: 'Rename' });
                if (name && name !== project.name) await act(() => api.updateProject(project.id, { name }));
              },
            },
            {
              label: 'Edit description…',
              onClick: async () => {
                const description = await app.prompt({ title: 'Project description', label: 'Description', initial: project.description, confirmLabel: 'Save', allowEmpty: true });
                if (description !== null) await act(() => api.updateProject(project.id, { description: description.trim() }));
              },
            },
            {
              label: 'Delete…',
              danger: true,
              onClick: async () => {
                const ok = await app.confirm({
                  title: 'Delete project',
                  message: (
                    <>
                      Delete <b>{project.name}</b> with its {plural(spaces.length, 'space')}, all geometries, simulations and results? This cannot be undone.
                    </>
                  ),
                  confirmLabel: 'Delete',
                  danger: true,
                });
                if (ok) await act(() => api.deleteProject(project.id), `Project “${project.name}” deleted`);
              },
            },
          ]}
        />
      </header>
      {isOpen && (
        <div className="project-body">
          {spaces.length === 0 && <div className="muted small pad">No spaces yet. Create one, or import a model here.</div>}
          {spaces.map((s) => (
            <SpaceBlock key={s.id} space={s} project={project} act={act} onImport={onImport} />
          ))}
        </div>
      )}
    </section>
  );
}

function SpaceBlock({ space, project, act, onImport }: { space: Space; project: Project; act: Act; onImport: (p: ImportPreset) => void }) {
  const app = useApp();
  const geoms = space.geometries ?? [];
  return (
    <div className="space-block">
      <div className="space-head">
        <span className="space-icon" aria-hidden="true">
          ▣
        </span>
        <h3>{space.name}</h3>
        <span className="muted small">{plural(geoms.length, 'geometry', 'geometries')}</span>
        <div className="grow" />
        <button className="btn btn-sm btn-ghost" onClick={() => onImport({ projectId: project.id, spaceId: space.id })}>
          Import geometry
        </button>
        <Menu
          items={[
            {
              label: 'Rename…',
              onClick: async () => {
                const name = await app.prompt({ title: 'Rename space', label: 'Name', initial: space.name, confirmLabel: 'Rename' });
                if (name && name !== space.name) await act(() => api.updateSpace(space.id, { name }));
              },
            },
            {
              label: 'Delete…',
              danger: true,
              onClick: async () => {
                const ok = await app.confirm({
                  title: 'Delete space',
                  message: (
                    <>
                      Delete <b>{space.name}</b> with its {plural(geoms.length, 'geometry', 'geometries')} and all their simulations?
                    </>
                  ),
                  confirmLabel: 'Delete',
                  danger: true,
                });
                if (ok) await act(() => api.deleteSpace(space.id), `Space “${space.name}” deleted`);
              },
            },
          ]}
        />
      </div>
      {geoms.length === 0 && <div className="muted small pad">Empty space.</div>}
      {geoms.map((g) => (
        <GeometryRow key={g.id} geometry={g} act={act} />
      ))}
    </div>
  );
}

function GeometryRow({ geometry, act }: { geometry: GeometrySummary; act: Act }) {
  const app = useApp();
  const sims = geometry.simulations ?? [];
  const newSim = async () => {
    const name = await app.prompt({ title: 'New simulation', label: 'Simulation name', initial: nextName('Simulation', sims.map((s) => s.name)), confirmLabel: 'Create' });
    if (!name) return;
    try {
      const sim = await api.createSimulation(geometry.id, name, null);
      await app.refreshTree();
      navigate(href.editor(sim.id));
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };
  return (
    <div className="geometry-row">
      <div className="geometry-head">
        <span className="geo-name">{geometry.name}</span>
        <span className="muted small">
          {geometry.n_triangles.toLocaleString()} triangles · {plural(geometry.n_layers, 'layer')}
        </span>
        {geometry.watertight ? (
          <span className="chip chip-ok" title="Closed volume: wave and geometrical solvers available">
            watertight
          </span>
        ) : (
          <span className="chip chip-warn" title="Not a closed volume: geometrical solver only">
            not watertight
          </span>
        )}
        <div className="grow" />
        <button className="btn btn-sm" onClick={newSim}>
          New simulation
        </button>
        <Menu
          items={[
            {
              label: 'Duplicate…',
              title: 'Same room and objects, no simulation: e.g. to compare empty and furnished',
              onClick: async () => {
                const name = await app.prompt({ title: 'Duplicate geometry', label: 'Name of the copy', initial: `${geometry.name} (copy)`, confirmLabel: 'Duplicate' });
                if (name) await act(() => api.duplicateGeometry(geometry.id, name), `Geometry “${name}” created`);
              },
            },
            {
              label: 'Rename…',
              onClick: async () => {
                const name = await app.prompt({ title: 'Rename geometry', label: 'Name', initial: geometry.name, confirmLabel: 'Rename' });
                if (name && name !== geometry.name) {
                  forgetGeometry(geometry.id);
                  await act(() => api.updateGeometry(geometry.id, { name }));
                }
              },
            },
            {
              label: 'Download source model',
              title: 'The file as it was imported (.obj, .Par, .dxf, .3dm, .ifc or .skp), without the placed objects',
              onClick: async () => {
                try {
                  const { blob, filename } = await api.fetchSourceModel(geometry.id);
                  downloadBlob(blob, filename);
                } catch (e) {
                  app.toast('error', errorMessage(e));
                }
              },
            },
            {
              label: 'Delete…',
              danger: true,
              onClick: async () => {
                const ok = await app.confirm({
                  title: 'Delete geometry',
                  message: (
                    <>
                      Delete <b>{geometry.name}</b> and its {plural(sims.length, 'simulation')}?
                    </>
                  ),
                  confirmLabel: 'Delete',
                  danger: true,
                });
                if (ok) {
                  forgetGeometry(geometry.id);
                  await act(() => api.deleteGeometry(geometry.id), `Geometry “${geometry.name}” deleted`);
                }
              },
            },
          ]}
        />
      </div>
      {sims.length === 0 ? (
        <div className="muted small sim-empty">No simulation yet.</div>
      ) : (
        <ul className="sim-list">
          {sims.map((s) => (
            <SimRow key={s.id} sim={s} act={act} />
          ))}
        </ul>
      )}
    </div>
  );
}

function SimRow({ sim, act }: { sim: SimulationSummary; act: Act }) {
  const app = useApp();
  const hasResults = !!sim.last_run;
  return (
    <li className="sim-row">
      <a className="sim-name" href={href.editor(sim.id)}>
        {sim.name}
      </a>
      <StatusBadge status={sim.status} stale={sim.results_stale} />
      <span className="muted small">{sim.last_run ? `run ${fmtDate(sim.last_run)}` : `created ${fmtDate(sim.created)}`}</span>
      <div className="grow" />
      <a className="icon-link" href={href.editor(sim.id)}>
        Edit
      </a>
      <a className={`icon-link ${hasResults ? '' : 'disabled'}`} href={href.results(sim.id)}>
        Results
      </a>
      <a className={`icon-link ${hasResults ? '' : 'disabled'}`} href={href.auralizer(sim.id)}>
        Listen
      </a>
      <Menu
        items={[
          {
            label: 'Delete…',
            danger: true,
            onClick: async () => {
              const ok = await app.confirm({
                title: 'Delete simulation',
                message: (
                  <>
                    Delete <b>{sim.name}</b> and its results?
                  </>
                ),
                confirmLabel: 'Delete',
                danger: true,
              });
              if (ok) await act(() => api.deleteSimulation(sim.id), `Simulation “${sim.name}” deleted`);
            },
          },
        ]}
      />
    </li>
  );
}
