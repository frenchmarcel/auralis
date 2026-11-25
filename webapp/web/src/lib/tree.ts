import type { GeometrySummary, Project, SimulationSummary, Space } from '../api';

export interface SimLocation {
  project: Project;
  space: Space;
  geometry: GeometrySummary;
  sim: SimulationSummary;
}

export function locateSimulation(tree: Project[] | null, simId: string): SimLocation | null {
  if (!tree) return null;
  for (const project of tree) {
    for (const space of project.spaces ?? []) {
      for (const geometry of space.geometries ?? []) {
        for (const sim of geometry.simulations ?? []) {
          if (sim.id === simId) return { project, space, geometry, sim };
        }
      }
    }
  }
  return null;
}

export function locateGeometry(
  tree: Project[] | null,
  gid: string,
): { project: Project; space: Space; geometry: GeometrySummary } | null {
  if (!tree) return null;
  for (const project of tree) {
    for (const space of project.spaces ?? []) {
      for (const geometry of space.geometries ?? []) {
        if (geometry.id === gid) return { project, space, geometry };
      }
    }
  }
  return null;
}

/** Next unused name of the form "<base> <n>". */
export function nextName(base: string, existing: string[]): string {
  const used = new Set(existing);
  for (let n = existing.length + 1; ; n++) {
    const cand = `${base} ${n}`;
    if (!used.has(cand)) {
      // Prefer the smallest free number.
      for (let k = 1; k < n; k++) if (!used.has(`${base} ${k}`)) return `${base} ${k}`;
      return cand;
    }
  }
}
