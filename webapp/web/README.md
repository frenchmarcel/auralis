# Auralis web UI

The browser interface of the local Auralis app. It uses Vite, React and
TypeScript, three.js through @react-three/fiber, Plotly and Web Audio. It
talks to the Python server only through the HTTP API in `../API.md`.

## Requirements

Node.js 24 with npm. Everything installs in this folder, with no global
packages.

```bash
npm install
```

## Development

Start the Python server first (it listens on `http://127.0.0.1:8765`), then:

```bash
npm run dev
```

Open http://127.0.0.1:5173. The dev server sends every `/api` request on to
`http://127.0.0.1:8765`.

## Build

```bash
npm run build      # type-checks (tsc --noEmit), then bundles into dist/
npm run typecheck  # type-check only
```

`dist/` is built with base `/`. The Python server serves it at `/`.

Development builds only have `window.__auralisProbe`. It maps world points
to screen pixels for headless smoke tests.

## Layout

- `src/api.ts`: typed client of `../API.md`. All HTTP calls go through it.
- `src/pages/`: Home, import wizard, editor (`editor/`), results, auralizer.
- `src/three/Scene.tsx`: shared 3D parts (Z up, picking, markers, gizmo).
- `src/three/Objects.tsx`: placed furniture. glTF visual models (lazy, per
  instance) over their acoustic proxies.
- `src/pages/editor/useObjectsSync.ts`: debounced
  `PUT /geometries/{gid}/objects`. The simulation autosave is flushed and
  paused around it.
- `src/pages/results/`:
  - comparison cards: `cards.ts`, `ComparisonCards.tsx`
  - chart builders: `charts.ts`
  - chart sections shared by the Model and Report views: `sections.tsx`
  - reflection tracking: `ReflectionsView.tsx`, with paths drawn by
    `src/three/ReflectionPaths.tsx`
- `src/lib/`:
  - `simdata.ts`: loads simulations, results and curves per page (several
    simulations per page)
  - `metrics.ts`: parameter list, receiver averages, STI rating, ISO single
    numbers
  - `export.ts`: CSV, clipboard and download helpers
  - `abCompare.ts`: A/B/C compatibility check of the auralizer
- `src/audio/engine.ts`: Web Audio graph of the auralizer (convolution, and a
  50 ms crossfade when you switch receiver or A/B/C).
- `src/components/`: UI parts and the Plotly wrapper.

Routes use the URL hash: `#/`, `#/editor/<sim>`, `#/results/<sim>`,
`#/auralizer/<sim>`.
