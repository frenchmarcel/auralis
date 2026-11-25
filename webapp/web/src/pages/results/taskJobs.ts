import { useCallback, useEffect, useState } from 'react';
import { api, ApiError, errorMessage, type Job, type JobStatus } from '../../api';

const POLL_MS = 1000;
export const ACTIVE_STATUSES = new Set<string>(['queued', 'running']);

/** What the task lists of the server have in common (modal maps, wave-field animations). */
export interface TaskLike {
  id: string;
  status: JobStatus;
  created: string;
  job_id?: string | null;
}

export const newestFirst = <T extends TaskLike>(l: T[]): T[] => [...l].sort((a, b) => (a.created < b.created ? 1 : a.created > b.created ? -1 : 0));

/**
 * A simulation's task list with the jobs of its queued and running tasks polled once a
 * second; the list is read again whenever one of them ends.
 */
export function useTaskJobs<T extends TaskLike>(load: () => Promise<T[]>) {
  const [tasks, setTasks] = useState<T[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [lost, setLost] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      setTasks(newestFirst(await load()));
      setListError(null);
    } catch (e) {
      setListError(errorMessage(e));
    }
  }, [load]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const activeKey = (tasks ?? [])
    .filter((t) => ACTIVE_STATUSES.has(t.status) && t.job_id && !lost.has(t.job_id))
    .map((t) => t.job_id!)
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
            if (!ACTIVE_STATUSES.has(j.status)) ended = true;
          } catch (e) {
            if (!cancelled && e instanceof ApiError && e.status === 404) setLost((s) => new Set(s).add(jid));
          }
        }),
      );
      if (cancelled) return;
      if (ended) await refresh();
      if (!cancelled) timer = window.setTimeout(tick, POLL_MS);
    };
    timer = window.setTimeout(tick, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeKey, refresh]);

  /** A task just queued from this page: listed (and polled) at once. */
  const addQueued = useCallback((task: T, job: Job) => {
    setJobs((m) => ({ ...m, [job.id]: job }));
    setTasks((l) => newestFirst([{ ...task, job_id: task.job_id ?? job.id }, ...(l ?? []).filter((x) => x.id !== task.id)]));
  }, []);

  const cancel = useCallback(async (jid: string) => {
    const j = await api.cancelJob(jid);
    setJobs((m) => ({ ...m, [jid]: j }));
  }, []);

  return { tasks, listError, jobs, lost, refresh, addQueued, cancel };
}
