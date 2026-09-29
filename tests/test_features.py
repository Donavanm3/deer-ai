"""Streaming replies, cache headers, saved items, usage display and admin test mode."""
import json, os, sys, tempfile, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from make_demo_data import make_chat

H = lambda t: {"Authorization": f"Bearer {t}"}


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_feat_")
    with TestClient(APP.app) as client:
        yield client


def events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def test_features(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    U = c.post("/api/auth/register", json={"username": "user1", "password": "useruser1"}).json()["token"]

    # static files are always revalidated so updates show without a hard refresh
    for path in ("/", "/admin", "/static/app.js", "/static/admin.js", "/sw.js"):
        assert c.get(path).headers["cache-control"] == "no-cache"

    ds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                files=[("files", ("c.txt", make_chat(600).encode(), "text/plain"))]).json()["id"]
    mid = c.post("/api/admin/models", headers=H(A), json={"name": "chat", "kind": "chat", "dataset_id": ds,
                                                           "preset": "medium", "config": {"eval_interval": 5, "batch_size": 2}}).json()["id"]
    c.post(f"/api/admin/models/{mid}/train", headers=H(A), json={"steps": 5})
    for _ in range(300):
        if c.get(f"/api/admin/models/{mid}", headers=H(A)).json()["status"] == "ready":
            break
        time.sleep(1)

    # unpublished: the admin sees it, and so does the admin's own chat, but test mode hides it
    assert c.post("/api/chat", headers=H(A), json={"text": "hi"}).status_code == 200
    # an almost untrained medium model must still answer quickly (this used to "think" for ages)
    t = time.time()
    r = c.post("/api/chat/stream", headers=H(A), json={"text": "Hello!"})
    ev = events(r)
    assert time.time() - t < 30  # used to take minutes; now well under a second on its own
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "done"
    streamed = "".join(e["text"] for e in ev if e["type"] == "token").strip() or "…"
    assert ev[-1]["message"]["content"] == streamed
    cid = ev[0]["conversation_id"]
    assert len(c.get(f"/api/conversations/{cid}", headers=H(A)).json()["messages"]) == 2

    # --- test mode: admin is treated like a user on a chosen plan
    c.put("/api/me/test-mode", headers=H(A), json={"on": True})
    assert c.get("/api/models", headers=H(A)).json() == []          # only published models now
    assert c.post("/api/chat", headers=H(A), json={"text": "hi"}).status_code == 409
    c.patch(f"/api/admin/models/{mid}", headers=H(A), json={"published": True})
    b = c.get("/api/billing", headers=H(A)).json()
    assert b["plan"]["slug"] == "free" and b["test"]["on"] is True and b["is_admin"] is False
    free = next(p for p in b["plans"] if p["slug"] == "free")
    pro = next(p for p in b["plans"] if p["slug"] == "pro")
    c.patch(f"/api/admin/plans/{free['id']}", headers=H(A), json={k: free[k] for k in (
        "name", "blurb", "price", "currency", "window_hours", "image_limit", "analyze_limit")} | {"chat_limit": 2, "sort": 0, "active": True})
    for _ in range(2):
        r = c.post("/api/chat/stream", headers=H(A), json={"text": "hi"})
        assert r.status_code == 200
    u = events(r)[-1]["usage"]["chat"]
    assert u["used"] == 2 and u["limit"] == 2 and u["resets_at"] > time.time()
    r = c.post("/api/chat/stream", headers=H(A), json={"text": "hi"})
    assert r.status_code == 429 and r.json()["detail"]["limit"]
    # preview Pro without paying
    c.put("/api/me/test-mode", headers=H(A), json={"on": True, "plan_id": pro["id"]})
    assert c.get("/api/usage", headers=H(A)).json()["plan"] == "Pro"
    assert c.post("/api/chat", headers=H(A), json={"text": "hi"}).status_code == 200
    c.post("/api/me/reset-usage", headers=H(A))
    assert c.get("/api/usage", headers=H(A)).json()["usage"]["chat"]["used"] == 0
    c.put("/api/me/test-mode", headers=H(A), json={"on": False})
    assert c.get("/api/usage", headers=H(A)).json()["unlimited_admin"] is True
    assert c.put("/api/me/test-mode", headers=H(U), json={"on": True}).status_code == 403

    # --- usage for regular users
    r = c.post("/api/chat", headers=H(U), json={"text": "Hello!"}).json()
    assert r["usage"]["chat"]["used"] == 1
    assert c.get("/api/usage", headers=H(U)).json()["usage"]["chat"]["limit"] == 2

    # --- saved items
    msg = r["messages"][1]["id"]
    assert c.post("/api/saved", headers=H(U), json={"message_id": msg}).json()["saved"]
    assert c.post("/api/saved", headers=H(A), json={"message_id": msg}).status_code == 404  # not theirs
    saved = c.get("/api/saved", headers=H(U)).json()
    assert saved[0]["id"] == msg and saved[0]["question"] == "Hello!"
    conv = c.get(f"/api/conversations/{r['conversation_id']}", headers=H(U)).json()
    assert [m["saved"] for m in conv["messages"]] == [False, True]
    c.delete(f"/api/saved/{msg}", headers=H(U))
    assert c.get("/api/saved", headers=H(U)).json() == []
