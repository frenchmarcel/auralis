import type { ReactNode } from 'react';
import { useApp } from '../AppContext';
import { href } from '../lib/router';
import { locateSimulation } from '../lib/tree';

export type View = 'editor' | 'results' | 'auralizer';

export function TopBar({ simId, view, right }: { simId?: string; view?: View; right?: ReactNode }) {
  const { tree } = useApp();
  const loc = simId ? locateSimulation(tree, simId) : null;
  return (
    <header className="topbar">
      <a className="brand" href={href.home()} title="Home">
        <svg viewBox="0 0 32 32" width="20" height="20" aria-hidden="true">
          <path d="M4 20c3-9 6.5-9 8.5 0s5.5 9 8.5 0 3.5-7 7-4" stroke="currentColor" strokeWidth="2.6" fill="none" strokeLinecap="round" />
        </svg>
        <span>Auralis</span>
      </a>
      {loc && (
        <nav className="crumbs" aria-label="Breadcrumb">
          <a href={href.home()}>{loc.project.name}</a>
          <span className="sep">›</span>
          <a href={href.home()}>{loc.space.name}</a>
          <span className="sep">›</span>
          <span>{loc.geometry.name}</span>
          <span className="sep">›</span>
          <span className="crumb-current">{loc.sim.name}</span>
        </nav>
      )}
      {simId && view && (
        <nav className="viewtabs" aria-label="Views">
          <a className={view === 'editor' ? 'active' : ''} href={href.editor(simId)}>
            Editor
          </a>
          <a className={view === 'results' ? 'active' : ''} href={href.results(simId)}>
            Results
          </a>
          <a className={view === 'auralizer' ? 'active' : ''} href={href.auralizer(simId)}>
            Auralizer
          </a>
        </nav>
      )}
      <div className="topbar-right">
        {right}
        <EngineStatus />
      </div>
    </header>
  );
}

function EngineStatus() {
  const { health, healthError } = useApp();
  if (healthError) {
    return (
      <span className="engine-status offline" title={healthError}>
        <span className="dot" /> Server offline
      </span>
    );
  }
  if (!health) return <span className="engine-status">…</span>;
  return (
    <span
      className="engine-status"
      title={`Engine ${health.version}. Native library: ${health.native ? 'available' : 'missing (Python fallback, much slower)'}. GPU: ${health.gpu ? 'available' : 'not available (CPU)'}.`}
    >
      <span className={`dot ${health.native ? 'ok' : 'warn'}`} /> native
      <span className={`dot ${health.gpu ? 'ok' : 'warn'}`} /> GPU
    </span>
  );
}
