"""
Training job scheduler.

Up to DEER_MAX_JOBS models (default 2) train at the same time, but a second job only starts when
the server can take it:

  - it has at least 2 CPU cores,
  - everything else on the machine (your website traffic, game servers, ...) isn't already using
    more than half the cores (DEER_BUSY_SHARE), and
  - there's enough free memory left (DEER_MIN_FREE_MB).

Jobs that can't start yet wait in a queue and start by themselves as soon as there's room. Running
jobs share the CPU cores between them instead of fighting over them: each is told how many
threads to use, and adjusts while it trains.
"""
import multiprocessing as mp
import os
import threading
import time

from . import db
from .. import training

MAX_JOBS = max(1, int(os.environ.get("DEER_MAX_JOBS", "2")))
BUSY_SHARE = float(os.environ.get("DEER_BUSY_SHARE", "0.5"))
MIN_FREE_MB = int(os.environ.get("DEER_MIN_FREE_MB", "1500"))

jobs = {}     # model_id -> Process/Thread (running)
queue = []    # [{"mid", "steps", "lr_scale", "auto", "since"}] (waiting, in order)
lock = threading.RLock()


# ================================================================== system
def cpu_count():
    try:
        return len(os.sched_getaffinity(0))  # respects container / VPS CPU limits
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def memory_mb():
    """(available, total) in MB, or (None, None) where the OS doesn't say."""
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f if ":" in line}
        return info.get("MemAvailable", 0) // 1024, info.get("MemTotal", 0) // 1024
    except (OSError, ValueError, IndexError):
        return None, None


def load1():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


def threads_each(n_jobs=None):
    n = n_jobs if n_jobs is not None else max(1, len(jobs))
    return max(1, cpu_count() // max(1, n))


def status(probe=None):
    """What the machine is doing, and whether another job may start now (with the reason if not)."""
    cpus = cpu_count()
    load = load1() if probe is None else probe.get("load")
    avail, total = memory_mb() if probe is None else (probe.get("avail"), probe.get("total"))
    running = len(jobs)
    ours = min(cpus, running * threads_each(running)) if running else 0  # load our own jobs cause
    other = None if load is None else max(0.0, load - ours)
    ok, why = True, "Ready to start a job."
    if running >= MAX_JOBS:
        ok, why = False, f"{running} of {MAX_JOBS} training slots are in use."
    elif running >= 1:
        if cpus < 2:
            ok, why = False, "This server has 1 CPU core, so models train one at a time."
        elif other is not None and other > cpus * BUSY_SHARE:
            ok, why = False, f"The server is busy with other work (load {load:.1f} on {cpus} cores)."
        elif other is None and cpus < 4:
            ok, why = False, "Can't read the server's load here, so a second job needs at least 4 cores."
        elif avail is not None and avail < MIN_FREE_MB:
            ok, why = False, f"Only {avail} MB of memory is free; a second job needs {MIN_FREE_MB} MB."
    return {"cpus": cpus, "load": None if load is None else round(load, 2),
            "other_load": None if other is None else round(other, 2),
            "mem_available_mb": avail, "mem_total_mb": total, "running": sorted(jobs), "queued": [q["mid"] for q in queue],
            "max_jobs": MAX_JOBS, "threads_per_job": threads_each(max(1, running)),
            "can_start": ok, "reason": why}


# ==================================================================== jobs
def _threads_file(mid):
    return os.path.join(db.model_dir(mid), "THREADS")


def _tell_threads():
    """Each running job gets an equal share of the cores (it re-reads this while training)."""
    n = threads_each(len(jobs))
    for mid in jobs:
        try:
            os.makedirs(db.model_dir(mid), exist_ok=True)
            with open(_threads_file(mid), "w") as f:
                f.write(str(n))
        except OSError:
            pass


def _spawn(mid, steps, lr_scale, auto):
    """Train in a separate process (keeps the site fast). Falls back to a thread on platforms without
    full multiprocessing support, such as some Android/Termux builds. DEER_TRAIN_MODE=thread forces it."""
    args = (mid, db.DATA_DIR, steps, lr_scale, auto)
    if os.environ.get("DEER_TRAIN_MODE", "process") != "thread":
        try:
            p = mp.get_context("spawn").Process(target=training.run, args=args, daemon=True)
            p.start()
            return p
        except Exception as e:  # pragma: no cover
            print(f"Process training unavailable ({e}); using a thread.")
    t = threading.Thread(target=training.run, args=args, daemon=True)
    t.start()
    return t


def reap():
    for mid, p in list(jobs.items()):
        row = db.q1("SELECT status FROM models WHERE id=?", (mid,))
        if p.is_alive() and row and row["status"] not in ("training", "queued"):
            jobs.pop(mid, None)  # finished and saved; the worker is just exiting, so free the slot now
            continue
        if not p.is_alive():
            p.join(timeout=0)
            jobs.pop(mid, None)
            if row and row["status"] in ("training", "queued"):
                db.ex("UPDATE models SET status='error', error=? WHERE id=?",
                      (f"Training exited unexpectedly (code {getattr(p, 'exitcode', '?')}).", mid))


def busy(mid):
    return mid in jobs or any(q["mid"] == mid for q in queue)


def enqueue(mid, steps, lr_scale=1.0, auto=False):
    with lock:
        if busy(mid):
            return False
        db.ex("UPDATE models SET status='queued', error=NULL, updated_at=? WHERE id=?", (db.now(), mid))
        queue.append({"mid": mid, "steps": steps, "lr_scale": lr_scale, "auto": auto, "since": time.time()})
        dispatch()
        return True


def dispatch(probe=None):
    """Start queued jobs while there's room. Called on every change and every few seconds."""
    with lock:
        reap()
        started = []
        while queue and status(probe)["can_start"]:
            q = queue.pop(0)
            if not db.q1("SELECT id FROM models WHERE id=?", (q["mid"],)):
                continue  # deleted while waiting
            os.makedirs(db.model_dir(q["mid"]), exist_ok=True)
            try:
                os.remove(training.stop_flag(q["mid"]))  # an old Stop must not cancel this new run
            except FileNotFoundError:
                pass
            with open(_threads_file(q["mid"]), "w") as f:
                f.write(str(threads_each(len(jobs) + 1)))
            jobs[q["mid"]] = _spawn(q["mid"], q["steps"], q["lr_scale"], q["auto"])
            started.append(q["mid"])
            probe = None  # after starting one, look at the real machine again
        if started or jobs:
            _tell_threads()
        return started


def unqueue(mid):
    with lock:
        before = len(queue)
        queue[:] = [q for q in queue if q["mid"] != mid]
        return len(queue) != before


def queue_position(mid):
    for i, q in enumerate(queue):
        if q["mid"] == mid:
            return i + 1
    return None


class Dispatcher:
    """Checks every few seconds whether a waiting job can start."""
    def __init__(self, every=3.0):
        self.every, self.stop = every, threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(self.every):
            try:
                dispatch()
            except Exception as e:  # never let the dispatcher die
                print(f"job dispatch failed: {e}")
