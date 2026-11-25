import { useEffect, useMemo, useState } from 'react';
import { api, type CatalogObject, type FrSmoothing, type MeshData, type PlacedObject, type SimulationSettings } from '../api';
import { useApp } from '../AppContext';
import { TopBar } from '../components/TopBar';
import { EmptyState, ErrorBox, Segmented, Spinner } from '../components/ui';
import { loadCatalog, loadGeometry, loadMesh } from '../lib/cache';
import { layerColor } from '../lib/colors';
import { downloadUrl } from '../lib/export';
import { fmtDate, fmtDuration, fmtHz, fmtNum } from '../lib/format';
import { boundsOf, dirFromAzEl, type Bounds } from '../lib/geom';
import { PARAMS, paramInfo, type ParamKey } from '../lib/metrics';
import { href } from '../lib/router';
import { allSimulations, SimDataStore, useBundles, useCurves, useResultsIndex, type CurveRequest } from '../lib/simdata';
import { locateSimulation } from '../lib/tree';
import { PlacedObjects } from '../three/Objects';
import { LayerMeshes, PointMarkers, SceneCanvas, markerRadius, type PointMarker } from '../three/Scene';
import { firstFreeColor, newCardKey, PART_LABELS, resolveCards, resultReceivers, sanitizeCard, validView, type CompCard, type ResolvedCard, type SavedView } from './results/cards';
import { drawable, type Channel, type PlotType } from './results/charts';
import { ComparisonCards } from './results/ComparisonCards';
import { ReflectionsView } from './results/ReflectionsView';
import { ParamButtons, ParamChart, ResponseChart, ResponseControls, SingleNumbers, type ResponseState } from './results/sections';
import { CombinePanel } from './results/CombinePanel';
import { useCombos } from './results/combos';
import { GridMapsView } from './results/GridMapsView';
import { RunNotices } from './results/RunNotices';
import { ModalMapsView } from './results/ModalMapsView';
import { OfficeView } from './results/OfficeView';
import { WaveFieldView } from './results/WaveFieldView';
import { getSlice, setSlice, useSavedDoc, type DocInfo } from '../lib/savedViews';
import { NamedConfigsButton } from '../components/NamedConfigs';
import type { ComboDef } from './results/cards';

type View = 'model' | 'report' | 'reflections' | 'grids' | 'modal' | 'field' | 'office';
const VIEWS: View[] = ['model', 'report', 'reflections', 'grids', 'modal', 'field', 'office'];
/** Views that need neither the comparison curves nor the results index. */
const OWN_DATA = new Set<View>(['reflections', 'modal', 'field', 'office']);

interface GeomView {
  mesh: MeshData;
  bounds: Bounds;
  placed: PlacedObject[];
}

function useGeometryView(gid: string | null): GeomView | null {
  const [v, setV] = useState<GeomView | null>(null);
  useEffect(() => {
    if (!gid) return;
    let cancelled = false;
    Promise.all([loadGeometry(gid), loadMesh(gid)])
      .then(([g, m]) => !cancelled && setV({ mesh: m, bounds: boundsOf(g.bbox.min, g.bbox.max), placed: g.objects ?? [] }))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [gid]);
  return v;
}

function useCatalog(needed: boolean): Map<string, CatalogObject> {
  const [c, setC] = useState<Map<string, CatalogObject>>(new Map());
  useEffect(() => {
    if (!needed) return;
    let cancelled = false;
    loadCatalog()
      .then((cat) => !cancelled && setC(new Map(cat.objects.map((o) => [o.id, o]))))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [needed]);
  return c;
}

function noiseLabel(n: SimulationSettings['background_noise']): string {
  if (n === undefined || n === null || n === 'none') return 'no background noise';
  if (Array.isArray(n)) return `custom noise (${n.map((v) => fmtNum(v, 0)).join(' / ')} dB)`;
  return `${n} background noise`;
}

/** Restores the page's saved view (server, else this browser's copy) before the page starts. */
export function ResultsPage({ simId }: { simId: string }) {
  const doc = useSavedDoc(simId, 'results');
  // A restored named configuration rebuilds the page from the saved view (the loaded data stay).
  const [epoch, setEpoch] = useState(0);
  const store = useMemo(() => new SimDataStore(), []);
  if (!doc.ready)
    return (
      <div className="app-shell">
        <TopBar simId={simId} view="results" />
        <div className="loading-row pad">
          <Spinner /> Loading results…
        </div>
      </div>
    );
  return <ResultsBody key={epoch} simId={simId} store={store} docInfo={doc.info} saveError={doc.saveError} onRestored={() => setEpoch((e) => e + 1)} />;
}

function ResultsBody({
  simId,
  store,
  docInfo,
  saveError,
  onRestored,
}: {
  simId: string;
  store: SimDataStore;
  docInfo: DocInfo | null;
  saveError: string | null;
  onRestored: () => void;
}) {
  const app = useApp();
  const saved = useMemo(() => validView(getSlice<SavedView>(simId, 'results', 'main')), [simId]);
  const [cards, setCards] = useState<CompCard[]>(
    () => saved?.cards?.length ? saved.cards : [{ key: newCardKey(), simId, sourceId: null, receiverIds: [], color: 0, label: '' }],
  );
  const [activeKey, setActiveKey] = useState<string | null>(saved?.activeKey ?? null);
  const [view, setView] = useState<View>(VIEWS.includes(saved?.view as View) ? (saved!.view as View) : 'model');
  const [param, setParam] = useState<ParamKey>(PARAMS.some((p) => p.key === saved?.param) ? (saved!.param as ParamKey) : 'T30');
  const [showTransition, setShowTransition] = useState(saved?.showTransition ?? true);
  const [allParams, setAllParams] = useState(false);
  // Print layout: the sheet takes the paper width first, so the charts re-layout before printing.
  const [printLayout, setPrintLayout] = useState(false);
  useEffect(() => {
    const before = () => setPrintLayout(true);
    const after = () => setPrintLayout(false);
    window.addEventListener('beforeprint', before);
    window.addEventListener('afterprint', after);
    return () => {
      window.removeEventListener('beforeprint', before);
      window.removeEventListener('afterprint', after);
    };
  }, []);
  const printReport = () => {
    setPrintLayout(true);
    // ResizeObserver -> Plotly resize happens asynchronously: give it time, then print.
    window.setTimeout(() => window.print(), 700);
  };
  const [resp, setResp] = useState<ResponseState>({
    plotType: (saved?.plotType as PlotType) ?? 'edc',
    edcBands: saved?.edcBands ?? [500, 1000],
    smoothing: (saved?.smoothing as FrSmoothing) ?? 3,
    irDb: saved?.irDb ?? false,
    channel: (saved?.channel as Channel) ?? 'LR',
  });
  const setRespPatch = (p: Partial<ResponseState>) => setResp((r) => ({ ...r, ...p }));

  useEffect(() => {
    setSlice(simId, 'results', 'main', { cards, activeKey: activeKey ?? undefined, param, view, showTransition, ...resp });
  }, [simId, cards, activeKey, param, view, showTransition, resp]);

  // Bumped when a view changed the simulation (e.g. added a grid): its bundle is read again.
  const [epoch, setEpoch] = useState(0);
  const simulationChanged = () => {
    store.invalidate(simId);
    setEpoch((e) => e + 1);
  };
  const bundles = useBundles(store, [simId, ...cards.map((c) => c.simId)], epoch);
  const page = bundles.get(simId);
  const sim = page?.data?.sim ?? null;
  const results = page?.data?.results ?? null;

  // Keep every card on a source/receiver that exists in its simulation's results.
  useEffect(() => {
    setCards((prev) => {
      let changed = false;
      const next = prev.map((c) => {
        const r = bundles.get(c.simId)?.data?.results ?? null;
        const s = sanitizeCard(c, r);
        if (s !== c) changed = true;
        return s;
      });
      return changed ? next : prev;
    });
  }, [bundles]);

  const combos = useCombos(cards);
  const [comboEdit, setComboEdit] = useState<{ key: string; def: ComboDef } | null>(null);
  const resolved = useMemo(() => resolveCards(cards, bundles, combos), [cards, bundles, combos]);
  const active = resolved.find((c) => c.card.key === activeKey) ?? resolved[0] ?? null;

  const entries = useMemo(() => allSimulations(app.tree), [app.tree]);
  const index = useResultsIndex(
    store,
    entries.map((e) => e.id),
    !OWN_DATA.has(view),
  );
  const addComboCard = (def: ComboDef) => {
    const c: CompCard = { key: newCardKey(), simId, sourceId: null, receiverIds: [def.receiverId], color: firstFreeColor(cards), label: '', combo: def };
    setCards((prev) => [...prev, c]);
    setActiveKey(c.key);
  };
  const updateComboCard = (key: string, def: ComboDef) => {
    setCards((prev) => prev.map((c) => (c.key === key ? { ...c, receiverIds: [def.receiverId], combo: def } : c)));
    setComboEdit(null);
  };

  const curveReqs = useMemo<CurveRequest[]>(
    () => drawable(resolved).flatMap((c) => c.pairs.map((p) => ({ simId: c.card.simId, sourceId: p.source_id, receiverId: p.receiver_id, part: c.part }))),
    [resolved],
  );
  const { curves, loading: curvesLoading, error: curvesError } = useCurves(store, OWN_DATA.has(view) ? [] : curveReqs, resp.smoothing);

  const updateCard = (key: string, patch: Partial<CompCard>) =>
    setCards((prev) =>
      prev.map((c) => {
        if (c.key !== key) return c;
        const n = { ...c, ...patch };
        if (patch.simId && patch.simId !== c.simId) {
          const r = bundles.get(patch.simId)?.data?.results ?? null;
          return r ? sanitizeCard(n, r) : n;
        }
        if (patch.sourceId && patch.sourceId !== c.sourceId) {
          const r = bundles.get(n.simId)?.data?.results ?? null;
          return sanitizeCard(n, r);
        }
        return n;
      }),
    );
  const addCard = () => {
    const base = active?.card ?? cards[0];
    const r = active?.results ?? null;
    // Next receiver of the active comparison not shown yet, else the same pair.
    const used = new Set(cards.filter((c) => c.simId === base?.simId && c.sourceId === base?.sourceId).flatMap((c) => c.receiverIds));
    const nextRcv = resultReceivers(r, base?.sourceId).find((x) => !used.has(x));
    const c: CompCard = {
      key: newCardKey(),
      simId: base?.simId ?? simId,
      sourceId: base?.sourceId ?? null,
      receiverIds: nextRcv ? [nextRcv] : [...(base?.receiverIds ?? [])],
      color: firstFreeColor(cards),
      label: '',
    };
    setCards((prev) => [...prev, c]);
    setActiveKey(c.key);
  };
  const removeCard = (key: string) => setCards((prev) => (prev.length > 1 ? prev.filter((c) => c.key !== key) : prev));
  const moveCard = (key: string, dir: -1 | 1) =>
    setCards((prev) => {
      const i = prev.findIndex((c) => c.key === key);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= prev.length) return prev;
      const n = [...prev];
      [n[i], n[j]] = [n[j], n[i]];
      return n;
    });

  const activeSim = active?.sim ?? null;
  const geom = useGeometryView(view === 'reflections' ? (sim?.geometry_id ?? null) : (activeSim?.geometry_id ?? null));
  const catalog = useCatalog(!!geom?.placed.length);

  const points = useMemo<PointMarker[]>(() => {
    if (!active?.sim) return [];
    const s = active.sim;
    const rcvs = active.card.receiverIds;
    // A summed card highlights every source of its sum.
    const on = (id: string) => (active.card.combo ? active.card.combo.sources.some((x) => x.source_id === id) : id === active.card.sourceId);
    return [
      ...s.sources.map((x) => ({ kind: 'source' as const, id: x.id, label: x.name, position: x.position, dir: dirFromAzEl(x.azimuth_deg, x.elevation_deg), selected: on(x.id), dim: !on(x.id) })),
      ...s.receivers.map((x) => ({ kind: 'receiver' as const, id: x.id, label: x.name, position: x.position, selected: rcvs.includes(x.id), dim: !rcvs.includes(x.id) })),
    ];
  }, [active]);

  const onMarker = (p: PointMarker, additive: boolean) => {
    if (!active || active.card.combo) return;
    if (p.kind === 'source') {
      if (resultReceivers(active.results, p.id).length) updateCard(active.card.key, { sourceId: p.id });
      return;
    }
    const all = resultReceivers(active.results, active.card.sourceId);
    if (!all.includes(p.id)) return;
    const sel = active.card.receiverIds;
    if (!additive) updateCard(active.card.key, { receiverIds: [p.id] });
    else if (sel.includes(p.id)) {
      if (sel.length > 1) updateCard(active.card.key, { receiverIds: sel.filter((x) => x !== p.id) });
    } else updateCard(active.card.key, { receiverIds: all.filter((x) => x === p.id || sel.includes(x)) });
  };

  const loc = locateSimulation(app.tree, simId);
  const baseName = sim?.name ?? 'results';
  const pInfo = paramInfo(param);
  const noResults = !!page?.data && !results;
  const loadingPage = !page || page.loading;

  const toolbar = (
    <div className="results-toolbar no-print">
      <Segmented<View>
        size="md"
        value={view}
        onChange={setView}
        options={[
          { value: 'model', label: 'Model', title: 'Charts with the 3D model' },
          { value: 'report', label: 'Report', title: 'Charts only, printable' },
          { value: 'reflections', label: 'Reflections', title: 'Reflection tracking: arrivals, reflectogram and paths in 3D' },
          { value: 'grids', label: 'Grid maps', title: 'Maps of the parameters on the receiver grids (surface receivers)' },
          { value: 'modal', label: 'Modal maps', title: 'Forced response of the wave solver on a receiver grid, frequency by frequency' },
          { value: 'field', label: 'Wave field', title: 'Animation of the pressure of the wave solver on a receiver grid after an impulse, slowed down' },
          { value: 'office', label: 'ISO 3382-3', title: 'Open-plan office indicators along a path of receivers (D2,S, Lp,A,S,4m, rC, rD)' },
        ]}
      />
      <NamedConfigsButton
        simId={simId}
        docKey="results"
        what="view"
        contents="the comparison cards, the page shown (Model, Report, Grid maps…), the parameter and response charts, and the settings of the grid-map, modal-map, wave-field and ISO 3382-3 pages"
        restoreEffect="The fixed colour scales of the grid maps belong to this browser, not to a configuration."
        onRestored={onRestored}
      />
      {(docInfo?.error || saveError) && (
        <span
          className="small text-warn"
          title={saveError ? `Saving the view on the server failed: ${saveError}. It is kept in this browser only.` : `The saved view could not be read from the server (${docInfo!.error}); this browser's copy is used.`}
        >
          {saveError ? 'View not saved on the server' : 'Saved view: server unavailable, browser copy used'}
        </span>
      )}
      <div className="grow" />
      {view === 'report' && (
        <>
          <label className="check inline" title="Print every parameter, not only the selected one">
            <input type="checkbox" checked={allParams} onChange={(e) => setAllParams(e.target.checked)} /> All parameters
          </label>
          <button type="button" className="btn btn-sm" onClick={printReport} title="Print or save as PDF (charts only)">
            Print…
          </button>
        </>
      )}
      <button type="button" className="btn btn-sm" disabled={!results} onClick={() => downloadUrl(api.exportXlsxUrl(simId))} title="Workbook: settings, parameters per pair and band, single numbers, points">
        Export .xlsx
      </button>
      <button type="button" className="btn btn-sm" disabled={!results} onClick={() => downloadUrl(api.irsZipUrl(simId))} title="One binaural float32 WAV per pair (shared scale), parameters.json, README">
        IRs .zip
      </button>
    </div>
  );

  const paramCard = (theme: 'dark' | 'light', height: number) => (
    <section className="card">
      <div className="card-head">
        <h3>Parameters</h3>
        <ParamButtons value={param} onChange={setParam} />
        <div className="grow" />
        {!pInfo.scalar && (
          <label className="check inline no-print" title="Dashed line at the wave/geometrical transition frequency">
            <input type="checkbox" checked={showTransition} onChange={(e) => setShowTransition(e.target.checked)} /> Transition
          </label>
        )}
      </div>
      <ParamChart cards={resolved} param={pInfo} showTransition={showTransition} theme={theme} height={height} filename={`${baseName}_${pInfo.label}`} />
    </section>
  );

  const responseCard = (theme: 'dark' | 'light', height: number) => (
    <section className="card">
      <div className="card-head">
        <h3>Response</h3>
        <ResponseControls s={resp} set={setRespPatch} loading={curvesLoading} />
      </div>
      <ResponseChart cards={resolved} s={resp} opts={{ curves, showTransition }} error={curvesError} loading={curvesLoading} theme={theme} height={height} filename={`${baseName}_${resp.plotType}`} />
    </section>
  );

  const cardsBar = (
    <ComparisonCards
      cards={resolved}
      activeKey={active?.card.key ?? null}
      onActivate={setActiveKey}
      onChange={updateCard}
      onRemove={removeCard}
      onMove={moveCard}
      onAdd={addCard}
      entries={entries}
      index={index}
      currentSimId={simId}
      combos={combos}
      onEditCombo={(key) => {
        const c = cards.find((x) => x.key === key);
        if (c?.combo) {
          setActiveKey(key);
          setComboEdit({ key, def: c.combo });
        }
      }}
    />
  );

  return (
    <div className={`app-shell ${view === 'report' ? 'report-mode' : ''}`}>
      <TopBar simId={simId} view="results" />
      {page?.error && (
        <div className="pad">
          <ErrorBox message={page.error} />
        </div>
      )}
      {!page?.error && loadingPage && (
        <div className="loading-row pad">
          <Spinner /> Loading results…
        </div>
      )}
      {sim && !page?.error && (
        <>
          {toolbar}
          {view === 'grids' ? (
            <GridMapsView pageSimId={simId} store={store} entries={entries} index={index} />
          ) : view === 'modal' ? (
            <ModalMapsView key={sim.id} simId={simId} sim={sim} />
          ) : view === 'field' ? (
            <WaveFieldView key={sim.id} simId={simId} sim={sim} onSimulationChanged={simulationChanged} />
          ) : view === 'office' ? (
            results ? (
              <OfficeView key={sim.id} simId={simId} sim={sim} results={results} />
            ) : (
              <EmptyState title="No results yet">
                Run the simulation first: <a href={href.editor(simId)}>open the editor</a>. ISO 3382-3 uses the pair results.
              </EmptyState>
            )
          ) : view === 'reflections' ? (
            <ReflectionsView
              key={sim.id}
              simId={simId}
              sim={sim}
              mesh={geom?.mesh ?? null}
              bounds={geom?.bounds ?? null}
              placed={geom?.placed ?? []}
              catalog={catalog}
              initialSource={active?.card.simId === simId ? active.card.sourceId : null}
              initialReceiver={active?.card.simId === simId ? (active.card.receiverIds[0] ?? null) : null}
            />
          ) : noResults && cards.every((c) => c.simId === simId) ? (
            <EmptyState title="No results yet">
              Run the simulation first: <a href={href.editor(simId)}>open the editor</a>. Reflection tracking works without results (Reflections view).
            </EmptyState>
          ) : view === 'report' ? (
            <div className="results report">
              <div className="results-main">
                {cardsBar}
                <article className={`report-sheet ${printLayout ? 'print-layout' : ''}`}>
                  <header className="report-header">
                    <div>
                      <h1>{sim.name}</h1>
                      <div className="report-sub">{loc ? `${loc.project.name} › ${loc.space.name} › ${loc.geometry.name}` : ''}</div>
                    </div>
                    <div className="report-sub right">
                      Auralis results · {new Date().toLocaleDateString()}
                      {results?.stale ? ' · stale results' : ''}
                    </div>
                  </header>
                  <table className="table compact report-legend">
                    <thead>
                      <tr>
                        <th>Comparison</th>
                        <th>Simulation</th>
                        <th>Source</th>
                        <th>Receiver</th>
                        <th>Run</th>
                      </tr>
                    </thead>
                    <tbody>
                      {resolved.map((c) => (
                        <tr key={c.card.key}>
                          <td>
                            <span className="swatch" style={{ background: c.color }} />
                            {c.label}
                          </td>
                          <td>{c.sim?.name ?? c.card.simId}</td>
                          <td>{c.srcName}</td>
                          <td>{c.rcvLabel}</td>
                          <td className="small">
                            {c.part !== 'hybrid' ? `${PART_LABELS[c.part]} result · ` : ''}
                            {c.results ? `${c.results.settings?.mode ?? ''}${c.results.transition_hz ? `, transition ${fmtHz(c.results.transition_hz)}` : ', geometrical only'}${c.results.stale ? ', stale' : ''}` : (c.error ?? '…')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <SingleNumbers cards={resolved} filename={`${baseName}_single_numbers`} />
                  {allParams ? (
                    <div className="report-grid">
                      {PARAMS.map((p) => (
                        <section className="card" key={p.key}>
                          <h3 title={p.title}>
                            <span dangerouslySetInnerHTML={{ __html: p.axis }} />
                            {p.unit ? ` (${p.unit})` : ''}
                          </h3>
                          <ParamChart cards={resolved} param={p} showTransition={showTransition} theme="light" height={250} filename={`${baseName}_${p.label}`} />
                        </section>
                      ))}
                    </div>
                  ) : (
                    <div className="report-grid">{paramCard('light', 360)}</div>
                  )}
                  <div className="report-grid">{responseCard('light', 360)}</div>
                </article>
              </div>
            </div>
          ) : (
            <div className="results">
              <div className="results-main">
                {cardsBar}
                {results?.stale && (
                  <div className="notice notice-warn">
                    These results are stale: the simulation was edited after this run. <a href={href.editor(simId)}>Re-run it in the editor</a> to update them.
                  </div>
                )}
                <RunNotices cards={resolved} />
                {paramCard('dark', 300)}
                <section className="card">
                  <SingleNumbers cards={resolved} filename={`${baseName}_single_numbers`} />
                </section>
                {responseCard('dark', 300)}
              </div>
              <aside className="results-side">
                <section className="card results-3d">
                  {geom ? (
                    <SceneCanvas bounds={geom.bounds} showGizmo={false}>
                      <LayerMeshes mesh={geom.mesh} shading="ghosted" pickable={false} styleOf={(_, i) => ({ color: layerColor(i) })} />
                      <PlacedObjects objects={geom.placed} catalog={catalog} showVisual pickable={false} />
                      <PointMarkers points={points} radius={markerRadius(geom.bounds)} onPointClick={(p, e) => onMarker(p, e.nativeEvent.ctrlKey || e.nativeEvent.metaKey || e.nativeEvent.shiftKey)} />
                    </SceneCanvas>
                  ) : (
                    <Spinner />
                  )}
                  <div className="muted small pad-x">
                    {active ? (
                      <>
                        <span className="swatch" style={{ background: active.color }} />
                        {active.sim?.name ?? ''}: click a marker to set this comparison (Ctrl-click averages receivers).
                      </>
                    ) : null}
                  </div>
                </section>
                {active?.results && <RunInfo rc={active} />}
                {results && results.pairs.length > 0 && (
                  <CombinePanel
                    key={comboEdit?.key ?? 'new'}
                    simId={simId}
                    sim={sim}
                    results={results}
                    editing={comboEdit}
                    onAdd={addComboCard}
                    onUpdate={updateComboCard}
                    onCancelEdit={() => setComboEdit(null)}
                  />
                )}
                {results && (
                  <section className="card">
                    <h3>Downloads</h3>
                    <div className="row gap wrap">
                      <button type="button" className="btn btn-xs" onClick={() => downloadUrl(api.exportXlsxUrl(simId))}>
                        Parameters .xlsx
                      </button>
                      <button type="button" className="btn btn-xs" onClick={() => downloadUrl(api.irsZipUrl(simId))}>
                        All IRs .zip
                      </button>
                    </div>
                    <table className="table compact">
                      <tbody>
                        {results.pairs.map((p) => (
                          <tr key={`${p.source_id}|${p.receiver_id}`}>
                            <td className="ellipsis">
                              {sim.sources.find((s) => s.id === p.source_id)?.name ?? p.source_id} → {sim.receivers.find((r) => r.id === p.receiver_id)?.name ?? p.receiver_id}
                            </td>
                            <td className="muted small num">{p.timings?.total_s !== undefined ? fmtDuration(p.timings.total_s) : ''}</td>
                            <td>
                              <a className="btn btn-xs" href={api.irWavUrl(simId, p.source_id, p.receiver_id)} download={`${sim.name}_${p.source_id}_${p.receiver_id}.wav`}>
                                ir.wav
                              </a>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="muted small">
                      Binaural, 2 channels, float32; all IRs of a simulation share one scale. Every chart exports its data (CSV, clipboard) and a PNG from its corner buttons.
                    </div>
                  </section>
                )}
              </aside>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function RunInfo({ rc }: { rc: ResolvedCard }) {
  const r = rc.results!;
  const st = r.settings ?? ({} as SimulationSettings);
  return (
    <section className="card">
      <h3>
        <span className="swatch" style={{ background: rc.color }} />
        Run · {rc.sim?.name}
      </h3>
      <dl className="kv small">
        <dt>Engine</dt>
        <dd>{r.engine_used}</dd>
        <dt>Mode</dt>
        <dd>
          {st.mode}
          {st.mode === 'advanced' ? ` (${st.solvers})` : ''}
        </dd>
        <dt>Transition</dt>
        <dd>{r.transition_hz ? fmtHz(r.transition_hz) : 'none (geometrical only)'}</dd>
        <dt>Sample rate</dt>
        <dd>{r.fs} Hz</dd>
        <dt>Parameters from</dt>
        <dd title="IR on which the ISO 3382 parameters (T20 … Ts, STI) are computed">
          {r.metrics_ir === 'omni' ? 'the omnidirectional IR' : 'the mean of the ears (run before v0.5)'}
        </dd>
        {(r.wave_failed?.length ?? 0) > 0 && (
          <>
            <dt>Wave band</dt>
            <dd className="text-warn">failed for {r.wave_failed!.length} source{r.wave_failed!.length === 1 ? '' : 's'} (geometrical only)</dd>
          </>
        )}
        <dt>IR scale</dt>
        <dd title="Common scale applied to all IRs of this simulation (comparisons are put back on one scale)">{fmtNum(r.ir_scale, 4)}</dd>
        <dt>STI inputs</dt>
        <dd>
          {st.speech_effort === 'raised' ? 'raised' : 'normal'} speech effort, {noiseLabel(st.background_noise)}
        </dd>
        {rc.sim?.last_run && (
          <>
            <dt>Last run</dt>
            <dd>{fmtDate(rc.sim.last_run)}</dd>
          </>
        )}
        {r.duration_s !== undefined && (
          <>
            <dt>Run time</dt>
            <dd>{fmtDuration(r.duration_s)}</dd>
          </>
        )}
      </dl>
      {r.stale && <div className="small text-warn">Stale: the simulation was edited after this run.</div>}
    </section>
  );
}
