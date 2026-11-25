import { useEffect, useState } from 'react';

export type Route =
  | { name: 'home' }
  | { name: 'editor'; simId: string }
  | { name: 'results'; simId: string }
  | { name: 'auralizer'; simId: string }
  | { name: 'notfound'; path: string };

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, '') || '/';
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  if (parts.length === 0) return { name: 'home' };
  if (parts.length === 2 && (parts[0] === 'editor' || parts[0] === 'results' || parts[0] === 'auralizer')) {
    return { name: parts[0], simId: parts[1] };
  }
  return { name: 'notfound', path };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}

export const href = {
  home: () => '#/',
  editor: (simId: string) => `#/editor/${encodeURIComponent(simId)}`,
  results: (simId: string) => `#/results/${encodeURIComponent(simId)}`,
  auralizer: (simId: string) => `#/auralizer/${encodeURIComponent(simId)}`,
};

export function navigate(target: string): void {
  window.location.hash = target.replace(/^#/, '');
}
