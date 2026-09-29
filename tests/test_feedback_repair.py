"""Regression: feedback left on a deleted conversation must not get mixed up with a new message that
reuses its id (it made a picture look like an empty chat lesson, and approving it crashed)."""
import io, os, sys, tempfile, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from deer import learning
from make_demo_data import make_chat, make_shapes, draw_shape

H = lambda t: {"Authorization": f"Bearer {t}"}


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_fbfix_")
    with TestClient(APP.app) as client:
        yield client


def png():
    b = io.BytesIO(); draw_shape("circle", (220, 50, 50)).save(b, "PNG"); return b.getvalue()


def test_reused_message_ids_and_repair(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    t = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
               files=[("files", ("c.txt", make_chat(200).encode(), "text/plain"))]).json()["id"]
    i = c.post("/api/admin/datasets", headers=H(A), data={"name": "s", "kind": "images"},
               files=[("files", ("s.zip", make_shapes(20), "application/zip"))]).json()["id"]
    for kind, ds in (("chat", t), ("vision", i)):
        m = c.post("/api/admin/models", headers=H(A), json={"name": kind, "kind": kind, "dataset_id": ds, "preset": "tiny"}).json()["id"]
        c.post(f"/api/admin/models/{m}/train", headers=H(A), json={"steps": 10})
        while c.get(f"/api/admin/models/{m}", headers=H(A)).json()["status"] != "ready":
            time.sleep(1)

    # rate a chat reply, delete the conversation, then a picture reply reuses the same message id
    r = c.post("/api/chat", headers=H(A), json={"text": "hi"}).json()
    old_id = r["messages"][1]["id"]
    c.post("/api/feedback", headers=H(A), json={"message_id": old_id, "rating": -1, "correction": "Hello there!"})
    c.delete(f"/api/conversations/{r['conversation_id']}", headers=H(A))
    assert not db.q("SELECT id FROM feedback WHERE message_id=?", (old_id,))  # unreviewed feedback goes with the chat
    r2 = c.post("/api/analyze", headers=H(A), files={"file": ("p.png", png(), "image/png")}).json()
    assert r2["messages"][1]["id"] == old_id  # SQLite reuses the id
    c.post("/api/feedback", headers=H(A), json={"message_id": old_id, "rating": -1, "label": "red circle"})
    items = c.get("/api/admin/learning", headers=H(A)).json()["items"]
    assert [(x["kind"], x["label"], x["prompt"]) for x in items] == [("vision", "red circle", None)]
    rv = c.post("/api/admin/learning/review", headers=H(A), json={"ids": [x["id"] for x in items], "action": "approve"})
    assert rv.status_code == 200 and rv.json()["done"] == 1

    # even if a stale row is still there (e.g. created before this fix), a new rating replaces it
    r3 = c.post("/api/analyze", headers=H(A), files={"file": ("p.png", png(), "image/png")}).json()
    mid = r3["messages"][1]["id"]
    db.ex("INSERT INTO feedback(user_id,message_id,kind,rating,prompt,answer,status,created_at) VALUES(1,?,'chat',-1,'q','a','applied',?)",
          (mid, time.time() - 999))
    assert c.post("/api/feedback", headers=H(A), json={"message_id": mid, "rating": -1, "label": "red circle"}).json()["status"] == "pending"
    assert db.q1("SELECT message_id FROM feedback WHERE prompt='q'")["message_id"] is None  # old approved one kept as history

    # rows already broken by the old bug are repaired on startup and never crash approval
    db.ex("INSERT INTO feedback(user_id,message_id,kind,rating,image_path,status,created_at) VALUES(1,NULL,'chat',-1,'/nope.png','pending',?)", (time.time(),))
    db.ex("INSERT INTO feedback(user_id,message_id,kind,rating,status,created_at) VALUES(1,NULL,'chat',-1,'pending',?)", (time.time(),))
    learning.repair()
    assert db.q1("SELECT kind, status FROM feedback WHERE image_path='/nope.png'") == {"kind": "vision", "status": "info"}
    bad = db.ex("INSERT INTO feedback(user_id,message_id,kind,rating,status,created_at) VALUES(1,NULL,'chat',-1,'pending',?)", (time.time(),))
    rv = c.post("/api/admin/learning/review", headers=H(A), json={"ids": [bad], "action": "approve"}).json()
    assert rv["done"] == 0 and "no question and answer" in rv["skipped"][0]["reason"]
    assert db.q1("SELECT status FROM feedback WHERE id=?", (bad,))["status"] == "info"
