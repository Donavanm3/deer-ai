"""Two models can train at once, but only when the server isn't overloaded; others wait in a queue."""
import os, sys, tempfile, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db, jobs as JOBS
from deer.server import app as APP
from make_demo_data import make_chat

H = lambda t: {"Authorization": f"Bearer {t}"}


class Alive:
    def is_alive(self):
        return True


def test_overload_rules(monkeypatch):
    monkeypatch.setattr(JOBS, "jobs", {})
    monkeypatch.setattr(JOBS, "MAX_JOBS", 2)
    probe = lambda load, avail=8000: {"load": load, "avail": avail, "total": 16000}
    monkeypatch.setattr(JOBS, "cpu_count", lambda: 4)
    assert JOBS.status(probe(9.0))["can_start"]                   # the first job always may start
    JOBS.jobs[1] = Alive()
    st = JOBS.status(probe(4.6))                                    # our job uses all 4 cores; 0.6 is other work
    assert st["can_start"] and st["other_load"] == pytest.approx(0.6) and st["threads_per_job"] == 4
    st = JOBS.status(probe(7.0))                                    # 3 cores' worth of other work: too busy
    assert not st["can_start"] and "busy with other work" in st["reason"]
    st = JOBS.status(probe(4.2, avail=900))
    assert not st["can_start"] and "memory" in st["reason"]
    st = JOBS.status({"load": None, "avail": None, "total": None})  # no load info (Windows): 4+ cores is enough
    assert st["can_start"]
    JOBS.jobs[2] = Alive()
    st = JOBS.status(probe(1.0))
    assert not st["can_start"] and "2 of 2" in st["reason"] and st["threads_per_job"] == 2
    monkeypatch.setattr(JOBS, "cpu_count", lambda: 1)
    JOBS.jobs.pop(2)
    st = JOBS.status(probe(1.0))
    assert not st["can_start"] and "1 CPU core" in st["reason"]


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_jobs_")
    with TestClient(APP.app) as client:
        yield client


def test_two_at_a_time_and_queue(c, monkeypatch):
    machine = {"load": 0.3}
    monkeypatch.setattr(JOBS, "MAX_JOBS", 2)
    monkeypatch.setattr(JOBS, "cpu_count", lambda: 4)
    monkeypatch.setattr(JOBS, "load1", lambda: machine["load"] + 4 * len(JOBS.jobs) / max(1, len(JOBS.jobs)) if JOBS.jobs else machine["load"])
    monkeypatch.setattr(JOBS, "memory_mb", lambda: (8000, 16000))
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    ds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                files=[("files", ("c.txt", make_chat(300).encode(), "text/plain"))]).json()["id"]
    ids = [c.post("/api/admin/models", headers=H(A), json={"name": f"m{i}", "kind": "chat", "dataset_id": ds,
           "preset": "tiny", "config": {"eval_interval": 20}}).json()["id"] for i in range(4)]
    status = lambda m: c.get(f"/api/admin/models/{m}", headers=H(A)).json()

    r1 = c.post(f"/api/admin/models/{ids[0]}/train", headers=H(A), json={"steps": 400}).json()
    r2 = c.post(f"/api/admin/models/{ids[1]}/train", headers=H(A), json={"steps": 400}).json()
    assert r1["status"] == "training" and r2["status"] == "training"   # quiet 4-core machine: both run
    assert set(JOBS.jobs) == {ids[0], ids[1]}
    for m in ids[:2]:
        assert open(os.path.join(db.dataset_dir(0).replace("datasets/0", f"models/{m}"), "THREADS")).read() == "2"
    r3 = c.post(f"/api/admin/models/{ids[2]}/train", headers=H(A), json={"steps": 20}).json()
    assert r3["status"] == "queued" and "2 of 2" in r3["message"]
    d = status(ids[2])
    assert d["status"] == "queued" and d["queue_position"] == 1 and "slots" in d["wait_reason"]
    assert c.post(f"/api/admin/models/{ids[2]}/train", headers=H(A), json={"steps": 20}).status_code == 409
    sysst = c.get("/api/admin/system", headers=H(A)).json()
    assert set(sysst["running_names"]) == {"m0", "m1"} and sysst["queued_names"] == ["m2"] and sysst["cpus"] == 4

    # a queued job can be taken out of the queue
    c.post(f"/api/admin/models/{ids[3]}/train", headers=H(A), json={"steps": 20})
    assert c.post(f"/api/admin/models/{ids[3]}/stop", headers=H(A)).json()["ok"]
    assert status(ids[3])["status"] == "stopped" and JOBS.queue_position(ids[3]) is None

    # stop one running job: the queued one starts by itself
    c.post(f"/api/admin/models/{ids[0]}/stop", headers=H(A))
    for _ in range(60):
        if status(ids[2])["status"] in ("training", "ready"):
            break
        time.sleep(1)
    assert status(ids[2])["status"] in ("training", "ready")

    # a busy machine holds the second job back until it calms down
    c.post(f"/api/admin/models/{ids[1]}/stop", headers=H(A))
    for _ in range(180):  # let everything finish (the 1-core test machine is slow with two workers)
        JOBS.reap()
        if not JOBS.jobs and all(status(m)["status"] not in ("training", "queued") for m in ids):
            break
        time.sleep(1)
    assert not JOBS.jobs
    machine["load"] = 9.0
    assert c.post(f"/api/admin/models/{ids[0]}/train", headers=H(A), json={"steps": 400}).json()["status"] == "training"
    r = c.post(f"/api/admin/models/{ids[1]}/train", headers=H(A), json={"steps": 20}).json()
    assert r["status"] == "queued" and "busy with other work" in r["message"]
    machine["load"] = 0.2
    for _ in range(30):
        if ids[1] in JOBS.jobs or status(ids[1])["status"] in ("training", "ready"):
            break
        time.sleep(1)
    assert ids[1] in JOBS.jobs or status(ids[1])["status"] in ("training", "ready")  # started (worker may still be booting)
    c.post(f"/api/admin/models/{ids[0]}/stop", headers=H(A))
