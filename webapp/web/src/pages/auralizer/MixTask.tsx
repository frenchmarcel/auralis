import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError, errorMessage, type MixTask, type RenderBody, type RenderResult } from '../../api';
import { ProgressBar } from '../../components/ui';

const POLL_MS = 700;
/** Consecutive failed polls before the task is reported lost. */
const MAX_POLL_ERRORS = 5;
/** A task still queued after this long waits for another mix (the server mixes one at a time). */
const QUEUE_NOTE_MS = 2000;

export type MixOutcome =
  | { status: 'done'; result: RenderResult }
  | { status: 'failed'; error: string }
  | { status: 'cancelled' }
  /** A newer run (or the page closing) took over: ignore. */
  | { status: 'superseded' };

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));
const ended = (t: MixTask) => t.status === 'done' || t.status === 'failed' || t.status === 'cancelled';

/**
 * A server-side mix as a background task (POST .../mixes), polled until it ends.
 * The server mixes one task at a time, so a task this page stops following is cancelled.
 */
export function useMixTask() {
  const [task, setTask] = useState<MixTask | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [since, setSince] = useState(0);
  const token = useRef(0);
  const live = useRef<{ simId: string; id: string } | null>(null);

  const abandon = useCallback((beacon: boolean) => {
    const l = live.current;
    live.current = null;
    if (!l) return;
    const url = `/api/simulations/${encodeURIComponent(l.simId)}/mixes/${encodeURIComponent(l.id)}/cancel`;
    if (beacon && navigator.sendBeacon?.(url)) return;
    void api.cancelMix(l.simId, l.id).catch(() => undefined);
  }, []);

  useEffect(() => {
    const onHide = () => abandon(true);
    window.addEventListener('pagehide', onHide);
    return () => {
      window.removeEventListener('pagehide', onHide);
      token.current++;
      abandon(false);
    };
  }, [abandon]);

  const run = useCallback(async (simId: string, body: RenderBody): Promise<MixOutcome> => {
    const my = ++token.current;
    abandon(false);
    setTask(null);
    setCancelling(false);
    setCancelError(null);
    setStarting(true);
    setSince(Date.now());
    let t: MixTask;
    try {
      t = await api.startMix(simId, body);
    } catch (e) {
      if (my !== token.current) return { status: 'superseded' };
      setStarting(false);
      return { status: 'failed', error: errorMessage(e) };
    }
    if (my !== token.current) {
      void api.cancelMix(simId, t.id).catch(() => undefined);
      return { status: 'superseded' };
    }
    live.current = { simId, id: t.id };
    setStarting(false);
    setTask(t);
    let errors = 0;
    while (!ended(t)) {
      await sleep(POLL_MS);
      if (my !== token.current) return { status: 'superseded' };
      try {
        t = await api.getMix(simId, t.id);
        errors = 0;
      } catch (e) {
        if (my !== token.current) return { status: 'superseded' };
        const gone = e instanceof ApiError && e.status === 404;
        if (gone || ++errors >= MAX_POLL_ERRORS) {
          live.current = null;
          const error = gone ? `the mix task is no longer on the server (${errorMessage(e)})` : `lost contact with the mix task: ${errorMessage(e)}`;
          setTask((old) => (old ? { ...old, status: 'failed', error } : old));
          return { status: 'failed', error };
        }
        continue;
      }
      if (my !== token.current) return { status: 'superseded' };
      setTask(t);
    }
    live.current = null;
    setCancelling(false);
    if (t.status === 'done') {
      if (t.result) return { status: 'done', result: t.result };
      return { status: 'failed', error: 'the mix ended without a result' };
    }
    if (t.status === 'cancelled') return { status: 'cancelled' };
    return { status: 'failed', error: t.error || 'the mix failed (no message from the server)' };
  }, [abandon]);

  const cancel = useCallback(async () => {
    const l = live.current;
    if (!l) return;
    setCancelling(true);
    setCancelError(null);
    try {
      await api.cancelMix(l.simId, l.id);
    } catch (e) {
      setCancelling(false);
      setCancelError(`Cancel failed: ${errorMessage(e)}`);
    }
  }, []);

  const active = starting || (!!task && !ended(task));
  return { task, starting, active, since, cancelling, cancelError, run, cancel };
}

export type MixTaskState = ReturnType<typeof useMixTask>;

/** Progress, stage and Cancel of a running mix task. */
export function MixTaskProgress({ mix, what }: { mix: MixTaskState; what: string }) {
  const t = mix.task;
  if (!mix.active) return null;
  const pct = Math.round((t?.progress ?? 0) * 100);
  const stage = !t ? 'starting…' : t.status === 'queued' ? (Date.now() - mix.since > QUEUE_NOTE_MS ? 'waiting for another mix to finish' : 'queued') : t.stage || 'running';
  return (
    <div className="mix-progress">
      <div className="row gap">
        <ProgressBar value={t?.progress ?? 0} status={t?.status} />
        <span className="small mono">{pct} %</span>
        <button className="btn btn-xs btn-danger-ghost" onClick={() => void mix.cancel()} disabled={!t || mix.cancelling} title="Stop this mix on the server">
          {mix.cancelling ? 'Cancelling…' : 'Cancel'}
        </button>
      </div>
      <div className="small muted ellipsis" title={`${what} · ${stage}`}>
        {what} · {stage}
      </div>
      {mix.cancelError && <div className="small text-err">{mix.cancelError}</div>}
    </div>
  );
}
