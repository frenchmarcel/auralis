# -*- coding: utf-8 -*-
"""Job queue: one simulation subprocess at a time (the GPU is exclusive),
progress parsed from the runner's '@@{json}' lines."""
from __future__ import annotations

import collections
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from .store import new_id, now

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COOLDOWN_S = 10.0          # pause between two GPU runs (CUDA context refusals)
RUNNERS = {"simulation": "webapp.server.runner", "motion": "webapp.server.motion_runner",
           "modal": "webapp.server.modal_runner", "field": "webapp.server.field_runner"}
LOG_LINES = 400


class JobManager:
    def __init__(self, store, resume=False):
        self.store = store
        self.jobs = collections.OrderedDict()
        self.procs = {}
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.last_end = 0.0
        self.gpu = None
        self.native = None
        pending = self._recover(resume)
        threading.Thread(target=self._probe, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        self._resubmit(pending)

    def _recover(self, resume):
        """A previous server stopped with jobs in flight: their simulations and renders
        would stay locked as queued/running forever. With `resume` (the real server,
        `python -m webapp.server`), they are queued again, loudly: a simulation keeps the
        sources it had finished (runner checkpoint), a task starts over. Otherwise (an
        import of the app, a test) they are marked failed, loudly."""
        sims, tasks = [], []
        for m in self.store._list("simulations"):
            if m.get("status") in ("queued", "running"):
                if resume:
                    print("[Auralis] WARNING: simulation %s was %s when the server stopped: "
                          "queued again (finished sources are kept)." % (m["id"], m["status"]),
                          flush=True)
                    sims.append(m)
                else:
                    print("[Auralis] WARNING: simulation %s was %s when the server stopped: "
                          "marked failed (run it again)." % (m["id"], m["status"]), flush=True)
                    m["status"] = "failed"
                    self.store.save_simulation(m)
            for kind in ("motion", "modal", "field"):
                for r in self.store.tasks(m["id"], kind):
                    if r["status"] not in ("queued", "running"):
                        continue
                    if resume:
                        print("[Auralis] WARNING: %s task %s of simulation %s was %s when the "
                              "server stopped: queued again (it starts over)."
                              % (kind, r["id"], m["id"], r["status"]), flush=True)
                        tasks.append((m, kind, r))
                    else:
                        self.store.set_task(m["id"], kind, r["id"], status="failed",
                                            error="interrupted: the server stopped")
        return sims, tasks

    def _resubmit(self, pending):
        """Queue the interrupted jobs again: simulations first (a task may need their
        results), each job saying in its log why it runs."""
        sims, tasks = pending
        for m in sims:
            self.submit(m)["log"].append("[Auralis] resumed after a server restart")
        for m, kind, r in tasks:
            self.store.set_task(m["id"], kind, r["id"], status="queued", error=None)
            self.submit_task(m, kind, r)["log"].append(
                "[Auralis] restarted after a server restart (from the beginning)")

    # -- engine probe, in a subprocess: the server never opens a CUDA context
    def _probe(self):
        code = ("import json,sys; sys.path.insert(0, %r); from auralis.engine import "
                "gpu_available, native_available; print(json.dumps([native_available(), "
                "gpu_available()]))" % REPO)
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                 text=True, timeout=60, cwd=REPO)
            self.native, self.gpu = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            self.native, self.gpu = None, None

    def health(self):
        return {"native": self.native, "gpu": self.gpu, "version": "0.1"}

    # -- public API ---------------------------------------------------------
    def submit(self, sim):
        with self.lock:
            for j in self.jobs.values():
                if (j["kind"] == "simulation" and j["simulation_id"] == sim["id"]
                        and j["status"] in ("queued", "running")):
                    return j
            job = {"id": new_id("j"), "kind": "simulation", "simulation_id": sim["id"],
                   "status": "queued", "progress": 0.0, "stage": "queued",
                   "sources": [{"id": s["id"], "name": s["name"], "progress": 0.0}
                               for s in sim["sources"]],
                   "started": None, "finished": None, "eta_s": None, "error": None, "log": []}
            self.jobs[job["id"]] = job
            while len(self.jobs) > 50:
                self.jobs.popitem(last=False)
        sim["status"] = "queued"
        self.store.save_simulation(sim)
        self.q.put(job["id"])
        return job

    def submit_motion(self, sim, info):
        """Animated render `info` (store.create_motion) of simulation `sim`."""
        return self.submit_task(sim, "motion", info)

    def submit_task(self, sim, kind, info):
        """A task `info` (store.create_task) of simulation `sim`, run by RUNNERS[kind]."""
        with self.lock:
            job = {"id": new_id("j"), "kind": kind, "simulation_id": sim["id"],
                   "render_id": info["id"], "status": "queued", "progress": 0.0,
                   "stage": "queued", "sources": [], "started": None, "finished": None,
                   "eta_s": None, "error": None, "log": []}
            self.jobs[job["id"]] = job
            while len(self.jobs) > 50:
                self.jobs.popitem(last=False)
        self.store.set_task(sim["id"], kind, info["id"], job_id=job["id"])
        self.q.put(job["id"])
        return job

    def get(self, jid):
        j = self.jobs.get(jid)
        if j is None:
            raise KeyError(jid)
        return j

    def list(self):
        return list(self.jobs.values())[::-1]

    def cancel(self, jid):
        j = self.get(jid)
        if j["status"] == "queued":
            j["status"], j["finished"] = "cancelled", now()
            self._set_sim(j, "cancelled")
        elif j["status"] == "running":
            j["cancel"] = True
            p = self.procs.get(jid)
            if p is not None:
                p.send_signal(signal.SIGTERM)
                threading.Timer(5.0, lambda: p.poll() is None and p.kill()).start()
        return j

    # -- worker ---------------------------------------------------------------
    def _set_sim(self, job, status, **extra):
        if job.get("kind") in RUNNERS and job["kind"] != "simulation":
            try:
                self.store.set_task(job["simulation_id"], job["kind"], job["render_id"],
                                    status=status, error=job.get("error"),
                                    finished=job.get("finished"))
            except (KeyError, OSError):
                pass
            return
        try:
            m = self.store.get_simulation(job["simulation_id"])
        except KeyError:
            return
        m["status"] = status
        m.update(extra)
        self.store.save_simulation(m)

    def _worker(self):
        while True:
            jid = self.q.get()
            job = self.jobs.get(jid)
            if job is None or job["status"] != "queued":
                continue
            wait = COOLDOWN_S - (time.time() - self.last_end)
            if wait > 0:
                job["stage"] = "cooling down the GPU (%.0f s)" % wait
                time.sleep(wait)
            if job["status"] != "queued":           # cancelled during the cooldown
                continue
            try:
                self._run(job)
            except Exception as exc:          # the queue must survive anything
                job["status"], job["error"] = "failed", "%s: %s" % (type(exc).__name__, exc)
                self._set_sim(job, "failed")
            self.last_end = time.time()
            threading.Thread(target=self._probe, daemon=True).start()

    def _run(self, job):
        job["status"], job["started"], job["stage"] = "running", now(), "starting"
        self._set_sim(job, "running")
        t0 = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        kind = job.get("kind", "simulation")
        cmd = [sys.executable, "-m", RUNNERS[kind], self.store.root, job["simulation_id"]]
        if kind != "simulation":
            cmd.append(job["render_id"])
        p = subprocess.Popen(cmd, cwd=REPO, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        self.procs[job["id"]] = p
        names = {s["id"]: s["name"] for s in job["sources"]}
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@"):
                try:
                    msg = json.loads(line[2:])
                except ValueError:
                    continue
                if "error" in msg:
                    job["error"] = msg["error"]
                if "stage" in msg:
                    job["stage"] = msg["stage"]
                if "progress" in msg:
                    job["progress"] = float(msg["progress"])
                    el = time.time() - t0
                    job["eta_s"] = round(el * (1 - job["progress"]) / job["progress"], 1) \
                        if job["progress"] > 0.03 else None
                if "sources" in msg:
                    job["sources"] = [{"id": k, "name": names.get(k, k), "progress": float(v)}
                                      for k, v in msg["sources"].items()]
            elif line.strip():
                job["log"].append(line)
                del job["log"][:-LOG_LINES]
        rc = p.wait()
        self.procs.pop(job["id"], None)
        job["finished"] = now()
        if job.get("cancel"):
            job["status"], job["stage"] = "cancelled", "cancelled"
            self._set_sim(job, "cancelled")
        elif rc == 0:
            job["status"], job["progress"], job["eta_s"] = "done", 1.0, 0.0
            if job.get("kind", "simulation") != "simulation":
                self._set_sim(job, "done")
            else:
                self._set_sim(job, "done", last_run=now(), results_stale=False,
                              last_duration_s=round(time.time() - t0, 1))
        else:
            job["status"] = "failed"
            # a negative code is a signal: Ctrl+C or a closed terminal kills the runner too
            job["error"] = job["error"] or (
                "interrupted (signal %d): the server was stopped during the run, run it again"
                % -rc if rc < 0 else "runner exited with code %d" % rc)
            self._set_sim(job, "failed")
        job.pop("cancel", None)
