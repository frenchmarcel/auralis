import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError, errorMessage, type Job, type RunEstimate, type Simulation, type Validation } from '../../api';
import { Modal, ProgressBar, Spinner, StatusBadge } from '../../components/ui';
import { fmtDate, fmtDuration, fmtHz, plural } from '../../lib/format';
import { href } from '../../lib/router';

const ACTIVE = new Set(['queued', 'running']);
const WARN_RE = /warn|fallback|repli|avertissement|deprecat|cpu instead|slower/i;
const ERR_RE = /error|erreur|traceback|exception|failed/i;

export function isActiveJob(job: Job | null): boolean {
  return !!job && ACTIVE.has(job.status);
}

/** Tracks the job of one simulation: finds an active one on mount and polls every 500 ms. */
export function useSimulationJob(simId: string, onFinished: (job: Job) => void) {
  const [job, setJob] = useState<Job | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const finishedRef = useRef(onFinished);
  finishedRef.current = onFinished;

  useEffect(() => {
    let cancelled = false;
    api
      .listJobs()
      .then((jobs) => {
        if (cancelled) return;
        // Animated renders share the queue but never lock the simulation.
        const mine = jobs.filter((j) => j.simulation_id === simId && (j.kind ?? 'simulation') === 'simulation');
        const active = mine.find((j) => ACTIVE.has(j.status));
        if (active) setJob(active);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [simId]);

  const active = isActiveJob(job);
  const jobId = job?.id;
  useEffect(() => {
    if (!active || !jobId) return;
    let cancelled = false;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const j = await api.getJob(jobId);
        if (cancelled) return;
        setJob(j);
        setPollError(null);
        if (!ACTIVE.has(j.status)) {
          finishedRef.current(j);
          return;
        }
      } catch (e) {
        if (!cancelled) setPollError(errorMessage(e));
      }
      if (!cancelled) timer = window.setTimeout(tick, 500);
    };
    timer = window.setTimeout(tick, 500);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [active, jobId]);

  const cancel = useCallback(async () => {
    if (!job) return;
    try {
      setJob(await api.cancelJob(job.id));
    } catch (e) {
      setPollError(errorMessage(e));
    }
  }, [job]);

  return { job, setJob, cancel, pollError };
}

export function RunDialog({
  job,
  simName,
  onClose,
  onCancel,
  pollError,
}: {
  job: Job;
  simName: string;
  onClose: () => void;
  onCancel: () => void;
  pollError: string | null;
}) {
  const [logOpen, setLogOpen] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);
  const warnings = job.log.filter((l) => WARN_RE.test(l)).length;
  const errors = job.log.filter((l) => ERR_RE.test(l)).length;
  const active = isActiveJob(job);

  useEffect(() => {
    if (logOpen && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logOpen, job.log.length]);

  return (
    <Modal title={`Run · ${simName}`} onClose={onClose} width={620}>
      <div className="modal-body run-dialog">
        <div className="row gap">
          <StatusBadge status={job.status} />
          <span className="grow small">{job.stage || (job.status === 'queued' ? 'Waiting for the GPU queue…' : '')}</span>
          <span className="small mono">{Math.round(job.progress * 100)}%</span>
        </div>
        <ProgressBar value={job.progress} status={job.status} />
        <div className="row gap small muted">
          <span>Started {fmtDate(job.started)}</span>
          {active && job.eta_s !== null && <span>· about {fmtDuration(job.eta_s)} remaining</span>}
          {job.finished && <span>· finished {fmtDate(job.finished)}</span>}
        </div>
        {job.sources.length > 0 && (
          <div className="run-sources">
            {job.sources.map((s) => (
              <div key={s.id} className="run-source">
                <span className="point-dot source">{s.id}</span>
                <span className="ellipsis">{s.name}</span>
                <ProgressBar value={s.progress} status={s.progress >= 1 ? 'done' : job.status} />
                <span className="small mono">{Math.round(s.progress * 100)}%</span>
              </div>
            ))}
          </div>
        )}
        {job.status === 'failed' && <div className="error-box">{job.error || 'The run failed (no error message).'}</div>}
        {job.status === 'cancelled' && <div className="warn-box">The run was cancelled.</div>}
        {pollError && <div className="warn-box small">Progress update failed: {pollError}</div>}
        <div className="log-block">
          <button className="log-toggle" onClick={() => setLogOpen((o) => !o)} aria-expanded={logOpen}>
            {logOpen ? '▾' : '▸'} Engine log ({job.log.length} lines
            {warnings > 0 && <span className="text-warn">, {warnings} warning{warnings === 1 ? '' : 's'}</span>}
            {errors > 0 && <span className="text-err">, {errors} error line{errors === 1 ? '' : 's'}</span>})
          </button>
          {logOpen && (
            <pre className="log" ref={logRef}>
              {job.log.map((l, i) => (
                <div key={i} className={ERR_RE.test(l) ? 'log-err' : WARN_RE.test(l) ? 'log-warn' : ''}>
                  {l}
                </div>
              ))}
            </pre>
          )}
          {!logOpen && warnings > 0 && (
            <div className="log-preview">
              {job.log
                .filter((l) => WARN_RE.test(l))
                .slice(-3)
                .map((l, i) => (
                  <div key={i} className="log-warn small mono">
                    {l}
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>
      <div className="modal-footer">
        {active ? (
          <>
            <button className="btn btn-danger" onClick={onCancel}>
              Cancel run
            </button>
            <div className="grow" />
            <button className="btn" onClick={onClose}>
              Hide
            </button>
          </>
        ) : (
          <>
            <div className="grow" />
            <button className="btn" onClick={onClose}>
              Close
            </button>
            {job.status === 'done' && (
              <>
                <a className="btn" href={href.auralizer(job.simulation_id)}>
                  Open auralizer
                </a>
                <a className="btn btn-primary" href={href.results(job.simulation_id)}>
                  Open results
                </a>
              </>
            )}
          </>
        )}
      </div>
    </Modal>
  );
}

const MODE_LABEL: Record<string, string> = { survey: 'Survey (geometrical)', default: 'Default (hybrid)', advanced: 'Advanced' };
const SOLVER_LABEL: Record<string, string> = { both: 'hybrid', wave: 'wave solver only', geometrical: 'geometrical only' };

/** How long the IRs will be, in words (Settings: automatic, fixed length, or decay criterion). */
export function stopCriterionText(st: Simulation['settings']): string {
  if (st.mode !== 'advanced') return 'automatic (from the estimated reverberation time)';
  if (st.ir_length_s) return `${st.ir_length_s} s`;
  if (st.stop_decay_db) return `long enough for a ${st.stop_decay_db} dB decay in every band (+10 dB margin)`;
  return 'automatic (from the estimated reverberation time)';
}

/** Before Run: rough duration (GET /estimate), settings summary, non-blocking warnings. */
export function RunConfirmDialog({
  sim,
  validation,
  onRun,
  onClose,
}: {
  sim: Simulation;
  validation: Validation | null;
  onRun: () => Promise<void> | void;
  onClose: () => void;
}) {
  const [est, setEst] = useState<RunEstimate | null>(null);
  const [estError, setEstError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api
      .getEstimate(sim.id)
      .then((e) => !cancelled && setEst(e))
      .catch((e) => !cancelled && setEstError(e instanceof ApiError && e.status === 404 ? 'this server has no estimate route' : errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [sim.id]);
  const st = sim.settings;
  const nS = sim.sources.length;
  const nR = sim.receivers.length;
  const warnings = validation?.warnings ?? [];
  const usesWave = st.mode === 'default' || (st.mode === 'advanced' && st.solvers !== 'geometrical');
  const go = async () => {
    setStarting(true);
    try {
      await onRun();
    } finally {
      setStarting(false);
    }
  };
  return (
    <Modal title={`Run · ${sim.name}`} onClose={onClose} width={560}>
      <div className="modal-body run-dialog">
        <div className="section-title">Estimated run time</div>
        {est ? (
          <>
            <div className="estimate-total">
              about <b>{fmtDuration(est.total_s)}</b> <span className="muted small">rough</span>
            </div>
            <dl className="kv small estimate-kv">
              <dt>Geometrical solver</dt>
              <dd>
                {fmtDuration(est.pairs_s)} · {plural(nS, 'source')} × {plural(nR, 'receiver')}
              </dd>
              <dt>Wave solver</dt>
              <dd>
                {est.wave_s > 0
                  ? `${fmtDuration(est.wave_s)} · one grid per source, up to ${fmtHz(est.transition_hz)}`
                  : usesWave
                    ? '–'
                    : 'not used'}
              </dd>
              <dt>Receiver grids</dt>
              <dd>{est.grids_s > 0 ? fmtDuration(est.grids_s) : 'none'}</dd>
            </dl>
            {!est.gpu && est.wave_s > 0 && (
              <div className="warn-box small">No GPU available: the wave solver runs on the CPU (about 50× slower than on the GPU).</div>
            )}
            <div className="muted small">
              {est.note}
              {est.wave_s > 0 && !(st.mode === 'advanced' && st.ir_length_s) ? '; the wave-solver part assumes a 2 s IR unless a fixed IR length is set' : ''}.
            </div>
          </>
        ) : estError ? (
          <div className="warn-box small">Estimate unavailable: {estError}</div>
        ) : (
          <div className="loading-row small">
            <Spinner size={12} /> Estimating…
          </div>
        )}
        <dl className="kv small">
          <dt>Mode</dt>
          <dd>
            {MODE_LABEL[st.mode] ?? st.mode}
            {st.mode === 'advanced' ? ` · ${SOLVER_LABEL[st.solvers] ?? st.solvers}` : ''}
          </dd>
          <dt>IR length</dt>
          <dd>{stopCriterionText(st)}</dd>
        </dl>
        {warnings.length > 0 && (
          <div className="warn-box small run-confirm-warnings">
            <b>
              {warnings.length} warning{warnings.length === 1 ? '' : 's'} (the run can still start)
            </b>
            <ul>
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="modal-footer">
        <div className="grow" />
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" disabled={starting} onClick={() => void go()} autoFocus>
          {starting ? (
            <>
              <Spinner size={11} /> Starting…
            </>
          ) : (
            'Run'
          )}
        </button>
      </div>
    </Modal>
  );
}
