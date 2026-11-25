import { useCallback, useEffect, useState } from 'react';
import { api, ApiError, errorMessage, type Job, type MotionCheck, type MotionCheckBody, type MotionRender, type RenderStrip, type Simulation } from '../api';
import { useApp } from '../AppContext';
import { NumField, ProgressBar, Segmented, Spinner, StatusBadge, Toggle } from '../components/ui';
import { fmtDate, fmtDuration, fmtNum } from '../lib/format';

/** Render time per second of audio and per source, as a multiple of real time (API.md, measured on CR2 and the Musikverein; tune here). */
export const MOTION_COST_X_REALTIME_PER_SOURCE_GPU = 2.5;
export const MOTION_COST_X_REALTIME_PER_SOURCE_CPU = 7;

const CHECK_DELAY_MS = 400;
const POLL_MS = 700;
const ACTIVE = new Set(['queued', 'running']);

interface MotionPrefs {
  durationMode: 'auto' | 'custom';
  customS: number;
  doppler: boolean;
  reverbDt: number;
  /** Conversation: the sources' recordings take turns, cut at their pauses. */
  turns: boolean;
  gapS: number;
  /** Mix the simulation's background noise (Settings), tracks calibrated as speech. */
  noise: boolean;
}

const DEFAULT_PREFS: MotionPrefs = { durationMode: 'auto', customS: 10, doppler: true, reverbDt: 0.75, turns: false, gapS: 0.45, noise: false };

function loadPrefs(simId: string): MotionPrefs {
  try {
    const raw = window.localStorage.getItem(`auralis.motion.${simId}`);
    return { ...DEFAULT_PREFS, ...(raw ? (JSON.parse(raw) as Partial<MotionPrefs>) : {}) };
  } catch {
    return DEFAULT_PREFS;
  }
}

function storePrefs(simId: string, p: MotionPrefs): void {
  try {
    window.localStorage.setItem(`auralis.motion.${simId}`, JSON.stringify(p));
  } catch {
    /* storage unavailable: not essential */
  }
}

const newestFirst = (l: MotionRender[]) => [...l].sort((a, b) => (a.created < b.created ? 1 : a.created > b.created ? -1 : 0));

/**
 * Animated render (moving sources / receiver) at the selected receiver, with the
 * strips of "Render to WAV"; queued as a job, listed with player, WAV and credits.
 */
export function MotionPanel({
  simId,
  sim,
  receiverId,
  strips,
  masterDb,
  srcName,
  rcvName,
  onListen,
}: {
  simId: string;
  sim: Simulation;
  receiverId: string | null;
  strips: RenderStrip[];
  masterDb: number;
  srcName: (id: string) => string;
  rcvName: (id: string) => string;
  /** A render's player started (element) or stopped (null): lets the map follow it. */
  onListen?: (el: HTMLAudioElement | null, label: string) => void;
}) {
  const app = useApp();
  const [prefs, setPrefs] = useState<MotionPrefs>(() => loadPrefs(simId));
  const [label, setLabel] = useState('');
  const [check, setCheck] = useState<MotionCheck | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [renders, setRenders] = useState<MotionRender[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [lost, setLost] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => storePrefs(simId, prefs), [simId, prefs]);
  const setPref = (p: Partial<MotionPrefs>) => setPrefs((s) => ({ ...s, ...p }));

  const body: MotionCheckBody | null = receiverId ? { receiver_id: receiverId, duration_s: prefs.durationMode === 'auto' ? null : prefs.customS, strips } : null;
  const bodyKey = JSON.stringify(body);

  useEffect(() => {
    if (!body) {
      setCheck(null);
      return;
    }
    let cancelled = false;
    setChecking(true);
    const timer = window.setTimeout(() => {
      api
        .checkMotion(simId, body)
        .then((c) => {
          if (cancelled) return;
          setCheck(c);
          setCheckError(null);
        })
        .catch((e) => !cancelled && setCheckError(errorMessage(e)))
        .finally(() => !cancelled && setChecking(false));
    }, CHECK_DELAY_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // The body is compared by value.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simId, bodyKey]);

  const refreshList = useCallback(async () => {
    try {
      const l = await api.listMotion(simId);
      setRenders(newestFirst(l));
      setListError(null);
    } catch (e) {
      setListError(errorMessage(e));
    }
  }, [simId]);
  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  // Poll the jobs of queued/running renders; refresh the list when one ends.
  const activeKey = (renders ?? [])
    .filter((r) => ACTIVE.has(r.status) && r.job_id && !lost.has(r.job_id))
    .map((r) => r.job_id!)
    .join(',');
  useEffect(() => {
    if (!activeKey) return;
    const ids = activeKey.split(',');
    let cancelled = false;
    let timer: number | undefined;
    const tick = async () => {
      let ended = false;
      await Promise.all(
        ids.map(async (jid) => {
          try {
            const j = await api.getJob(jid);
            if (cancelled) return;
            setJobs((m) => ({ ...m, [jid]: j }));
            if (!ACTIVE.has(j.status)) ended = true;
          } catch (e) {
            // A restarted server forgets its jobs: stop polling that one, say so.
            if (!cancelled && e instanceof ApiError && e.status === 404) setLost((s) => new Set(s).add(jid));
          }
        }),
      );
      if (cancelled) return;
      if (ended) await refreshList();
      if (!cancelled) timer = window.setTimeout(tick, POLL_MS);
    };
    timer = window.setTimeout(tick, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeKey, refreshList]);

  const nSources = strips.filter((s) => !s.muted).length;
  const hasNoise = !!sim.settings.background_noise && sim.settings.background_noise !== 'none';
  const duration = check?.duration_s ?? null;
  // GET /api/health -> gpu (polled by the app); unknown counts as CPU, the slower case.
  const onGpu = app.health?.gpu === true;
  const costFactor = onGpu ? MOTION_COST_X_REALTIME_PER_SOURCE_GPU : MOTION_COST_X_REALTIME_PER_SOURCE_CPU;
  const estimate = duration !== null ? duration * nSources * costFactor : null;
  const rcv = sim.receivers.find((r) => r.id === receiverId);
  const moving = [
    ...(rcv?.path ? [rcv.name] : []),
    ...strips.filter((s) => !s.muted && sim.sources.find((x) => x.id === s.source_id)?.path).map((s) => srcName(s.source_id)),
  ];
  const anyPath = [...sim.sources, ...sim.receivers].some((x) => x.path);
  const canRender = !!body && !!check?.ok && !checking && !submitting;

  const submit = async () => {
    if (!body || !check?.ok) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const r = await api.submitMotion(simId, {
        ...body,
        doppler: prefs.doppler,
        reverb_dt: prefs.reverbDt,
        master_db: masterDb,
        label: label.trim(),
        turn_taking: prefs.turns && nSources >= 2 ? { gap_s: prefs.gapS } : null,
        background_noise: prefs.noise && hasNoise,
      });
      setJobs((m) => ({ ...m, [r.job.id]: r.job }));
      setRenders((l) => newestFirst([{ ...r.render, job_id: r.render.job_id ?? r.job.id }, ...(l ?? []).filter((x) => x.id !== r.render.id)]));
      setLabel('');
    } catch (e) {
      setSubmitError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async (jid: string) => {
    try {
      const j = await api.cancelJob(jid);
      setJobs((m) => ({ ...m, [jid]: j }));
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  const remove = async (r: MotionRender) => {
    const ok = await app.confirm({
      title: 'Delete animated render',
      message: (
        <>
          Delete the render <b>{r.label || fmtDate(r.created)}</b> and its WAV?
        </>
      ),
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteMotion(simId, r.id);
      await refreshList();
    } catch (e) {
      app.toast('error', errorMessage(e));
    }
  };

  return (
    <div className="motion-panel">
      <div className="section-title">Animated render</div>
      {!anyPath ? (
        <div className="small muted">
          Nothing moves in this simulation. Give a source or the receiver a trajectory in the editor (Sources / Receivers tab, “Make it move”).
        </div>
      ) : (
        <>
          <div className="small">
            At <b>{receiverId ? rcvName(receiverId) : '–'}</b> · {nSources} source{nSources === 1 ? '' : 's'} with a sound
            {moving.length > 0 && <span className="muted"> · moving: {moving.join(', ')}</span>}
          </div>
          <div className="motion-fields">
            <span className="small muted">Length</span>
            <Segmented<'auto' | 'custom'>
              value={prefs.durationMode}
              onChange={(v) => setPref({ durationMode: v })}
              options={[
                { value: 'auto', label: 'Auto', title: 'Until the last waypoint of the moving points' },
                { value: 'custom', label: 'Custom', title: 'Fixed length; tracks are cut or padded with silence' },
              ]}
            />
            {prefs.durationMode === 'custom' ? (
              <>
                <NumField value={prefs.customS} min={0.5} max={600} step={1} digits={1} onChange={(v) => setPref({ customS: v ?? 10 })} ariaLabel="Render length (s)" width={60} />
                <span className="unit">s</span>
              </>
            ) : (
              <span className="small mono">{check ? `${fmtNum(check.path_end_s, 2)} s` : '…'}</span>
            )}
          </div>
          <div className="motion-fields">
            <Toggle checked={prefs.doppler} onChange={(v) => setPref({ doppler: v })} label="Doppler" title="Pitch shift of the direct sound (variable delay line)" />
            {nSources >= 2 && (
              <Toggle
                checked={prefs.turns}
                onChange={(v) => setPref({ turns: v })}
                label="Conversation"
                title="The sources' recordings speak in turn, one utterance each (cut at their own pauses), in the order of the mixer"
              />
            )}
            {hasNoise && (
              <Toggle
                checked={prefs.noise}
                onChange={(v) => setPref({ noise: v })}
                label="Background noise"
                title={`Mix the simulation's background noise (${String(sim.settings.background_noise)}), diffuse, at its level; the recordings are taken as speech at the vocal effort of the Settings`}
              />
            )}
            {nSources >= 2 && prefs.turns && (
              <label className="inline-field" title="Silence between two turns">
                <span>gap</span>
                <NumField value={prefs.gapS} min={0} max={5} step={0.05} digits={2} onChange={(v) => setPref({ gapS: v ?? 0.45 })} ariaLabel="Gap between turns (s)" width={52} />
                <span className="unit">s</span>
              </label>
            )}
            <input className="motion-label" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Label (optional)" aria-label="Render label" maxLength={80} />
          </div>
          <details className="motion-advanced">
            <summary className="small muted">Advanced</summary>
            <label className="inline-field" title="The late field is computed at key frames this far apart, then cross-faded">
              <span>Late-field key frames every</span>
              <NumField value={prefs.reverbDt} min={0.1} max={5} step={0.05} digits={2} onChange={(v) => setPref({ reverbDt: v ?? 0.75 })} ariaLabel="Late-field key frame interval (s)" width={52} />
              <span className="unit">s</span>
            </label>
          </details>
          {checkError && <div className="small text-err">Check failed: {checkError}</div>}
          {check && check.blocking.length > 0 && (
            <ul className="reasons">
              {check.blocking.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
          {check && check.warnings.length > 0 && (
            <ul className="motion-warnings small">
              {check.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
          <div className="row gap">
            <button className="btn btn-sm btn-primary" disabled={!canRender} onClick={() => void submit()} title={check && !check.ok ? check.blocking.join('\n') : 'Queue the animated render on the server'}>
              {submitting ? (
                <>
                  <Spinner size={11} /> Queuing…
                </>
              ) : (
                'Render'
              )}
            </button>
            {checking && <Spinner size={11} />}
            {estimate !== null && check?.ok && (
              <span className="small muted" title={`about ${costFactor} × real time per source on the ${onGpu ? 'GPU' : 'CPU'}`}>
                ≈ {fmtDuration(estimate)} on the {onGpu ? 'GPU' : 'CPU'} for {fmtNum(duration, 1)} s × {nSources} source{nSources === 1 ? '' : 's'}
              </span>
            )}
          </div>
          {submitError && <div className="small text-err">{submitError}</div>}
        </>
      )}

      {listError && <div className="small text-err">Renders unavailable: {listError}</div>}
      {renders && renders.length > 0 && (
        <div className="motion-list">
          {renders.map((r) => {
            const job = r.job_id ? jobs[r.job_id] : undefined;
            const active = ACTIVE.has(r.status);
            const title = r.label || fmtDate(r.created);
            return (
              <div key={r.id} className="motion-item">
                <div className="row gap">
                  <StatusBadge status={job && active ? job.status : r.status} />
                  <b className="ellipsis grow" title={title}>
                    {title}
                  </b>
                  {active && r.job_id ? (
                    <button className="btn btn-xs btn-danger-ghost" onClick={() => void cancel(r.job_id!)} disabled={job?.status === 'cancelled'}>
                      Cancel
                    </button>
                  ) : (
                    <button className="btn btn-xs btn-ghost" onClick={() => void remove(r)} title="Delete this render">
                      Delete
                    </button>
                  )}
                </div>
                <div className="small muted">
                  {rcvName(r.receiver_id)} · {r.n_sources} source{r.n_sources === 1 ? '' : 's'}
                  {r.label ? ` · ${fmtDate(r.created)}` : ''}
                </div>
                {active && (
                  <>
                    <ProgressBar value={job?.progress ?? 0} status={job?.status ?? r.status} />
                    <div className="small muted ellipsis" title={job?.stage}>
                      {r.job_id && lost.has(r.job_id)
                        ? 'The server no longer knows this job (restarted?).'
                        : job
                          ? `${job.status === 'queued' ? 'Waiting in the queue' : job.stage || 'running'} · ${Math.round(job.progress * 100)} %${job.eta_s ? ` · about ${fmtDuration(job.eta_s)} left` : ''}`
                          : 'Waiting in the queue'}
                    </div>
                  </>
                )}
                {r.status === 'failed' && <div className="small text-err">{r.error || job?.error || 'The render failed.'}</div>}
                {r.status === 'done' && r.url && (
                  <>
                    <audio
                      controls
                      preload="metadata"
                      src={r.url}
                      onPlay={(e) => onListen?.(e.currentTarget, title)}
                      onPause={() => onListen?.(null, title)}
                      onEnded={() => onListen?.(null, title)}
                    />
                    <div className="row gap small">
                      <a href={r.url} download>
                        Download WAV
                      </a>
                      {r.credits_url && (
                        <a href={r.credits_url} target="_blank" rel="noreferrer">
                          Credits
                        </a>
                      )}
                      {r.result && (
                        <span className="muted">
                          {fmtNum(r.result.duration_s, 1)} s · {r.result.fs} Hz · {r.result.doppler ? 'Doppler' : 'no Doppler'}
                          {Array.isArray(r.result.turns) ? ` · conversation, ${r.result.turns.length} turns` : ''}
                          {r.result.background_noise ? ` · noise ${r.result.background_noise.curve} (${fmtNum(r.result.background_noise.dba, 0)} dB(A))` : ''} · rendered in {fmtDuration(r.result.render_s)}
                        </span>
                      )}
                    </div>
                    {r.result && r.result.credits.length > 0 && (
                      <ul className="motion-credits small muted">
                        {r.result.credits.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    )}
                    {r.result && r.result.warnings.length > 0 && <div className="small text-warn">{r.result.warnings.join(' · ')}</div>}
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
