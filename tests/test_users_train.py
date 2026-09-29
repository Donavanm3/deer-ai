"""Users teach the models: lessons are really learned, user messages/pictures feed training,
opt-out is respected, and unpublished models stay hidden."""
import io, os, sys, tempfile, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from make_demo_data import make_chat, make_shapes, draw_shape

H = lambda t: {"Authorization": f"Bearer {t}"}
LESSONS = [("What is your favorite tree?", "My favorite tree is the white oak."),
           ("Where do you live?", "I live in the Kentucky woods."),
           ("What is your favorite food?", "I love acorns and clover."),
           ("What sport do you like?", "I like running through the forest."),
           ("What color are you?", "I am brown with white spots.")]


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_users_")
    with TestClient(APP.app) as client:
        yield client


def wait(c, A, mid):
    for _ in range(900):
        m = c.get(f"/api/admin/models/{mid}", headers=H(A)).json()
        if m["status"] in ("ready", "error", "stopped"):
            return m
        time.sleep(1)


def png(kind="circle", color=(220, 50, 50)):
    b = io.BytesIO(); draw_shape(kind, color).save(b, "PNG"); return b.getvalue()


def test_users_teach(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    U = c.post("/api/auth/register", json={"username": "user1", "password": "useruser1"}).json()["token"]
    X = c.post("/api/auth/register", json={"username": "private", "password": "private12"}).json()["token"]
    tds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                 files=[("files", ("c.txt", make_chat(4000).encode(), "text/plain"))]).json()["id"]
    ids = c.post("/api/admin/datasets", headers=H(A), data={"name": "s", "kind": "images"},
                 files=[("files", ("s.zip", make_shapes(60), "application/zip"))]).json()["id"]
    chat = c.post("/api/admin/models", headers=H(A), json={"name": "chat", "kind": "chat", "dataset_id": tds, "preset": "tiny"}).json()["id"]
    eyes = c.post("/api/admin/models", headers=H(A), json={"name": "eyes", "kind": "vision", "dataset_id": ids, "preset": "tiny"}).json()["id"]
    c.post(f"/api/admin/models/{chat}/train", headers=H(A), json={"steps": 1000}); wait(c, A, chat)
    c.post(f"/api/admin/models/{eyes}/train", headers=H(A), json={"steps": 150}); assert wait(c, A, eyes)["status"] == "ready"

    # ---- unpublished models: hidden from users, and from the admin's chat unless asked
    assert c.get("/api/models", headers=H(U)).json() == []
    assert c.get("/api/models", headers=H(A)).json() == []
    assert {m["name"] for m in c.get("/api/models?all=1", headers=H(A)).json()} == {"chat", "eyes"}
    assert c.get("/api/models?all=1", headers=H(U)).json() == []
    assert c.post("/api/chat", headers=H(A), json={"text": "hi"}).status_code == 200  # admin can still try it
    for m in (chat, eyes):
        c.patch(f"/api/admin/models/{m}", headers=H(A), json={"published": True})
    c.patch(f"/api/admin/models/{eyes}", headers=H(A), json={"published": False})
    assert [m["name"] for m in c.get("/api/models", headers=H(U)).json()] == ["chat"]
    r = c.post("/api/analyze", headers=H(U), data={"model_id": eyes}, files={"file": ("p.png", png(), "image/png")})
    assert r.status_code == 404 and r.json()["detail"]["model_gone"]
    c.patch(f"/api/admin/models/{eyes}", headers=H(A), json={"published": True})

    # ---- privacy: one user opts out
    assert c.put("/api/me/privacy", headers=H(X), json={"train_ok": False}).json()["train_ok"] is False
    assert c.get("/api/usage", headers=H(X)).json()["train_ok"] is False

    # ---- users correct replies; approved lessons are really learned
    for q, a in LESSONS:
        r = c.post("/api/chat/stream", headers=H(U), json={"text": q})
        mid = [l for l in r.text.splitlines() if '"done"' in l][0]
        import json as _j
        msg = _j.loads(mid[6:])["message"]["id"]
        c.post("/api/feedback", headers=H(U), json={"message_id": msg, "rating": -1, "correction": a})
    c.post("/api/chat", headers=H(X), json={"text": "my secret diary entry"})
    items = c.get("/api/admin/learning", headers=H(A)).json()["items"]
    c.post("/api/admin/learning/review", headers=H(A), json={"ids": [i["id"] for i in items if i["kind"] == "chat"], "action": "approve"})
    c.post(f"/api/admin/models/{chat}/learn", headers=H(A))
    m = wait(c, A, chat)
    ll = m["meta"]["last_learn"]
    assert ll["outcome"] == "kept" and ll["lessons_after"] < ll["lessons_before"] / 5
    right = sum(c.post("/api/chat", headers=H(U), json={"text": q, "temperature": 0.2}).json()["messages"][1]["content"] == a
                for q, a in LESSONS)
    assert right >= 4, f"only {right}/5 lessons recalled"

    # ---- what users write feeds the chat model's training text (not opted-out users, not admin)
    ut = open(os.path.join(db.dataset_dir(tds), "user_text.txt")).read()
    assert "User: Where do you live?" in ut and "secret diary" not in ut and "User: hi" not in ut
    c.patch("/api/admin/learning/settings", headers=H(A), json={"from_messages": False})
    c.post("/api/chat", headers=H(U), json={"text": "zebra crossing question"})
    assert "zebra" not in open(os.path.join(db.dataset_dir(tds), "user_text.txt")).read()

    # ---- pictures users send: confident ones arrive in the inbox already labeled
    r = c.post("/api/analyze", headers=H(U), files={"file": ("p.png", png(), "image/png")}).json()
    top = r["messages"][1]["meta"]["predictions"][0]
    auto = [i for i in c.get("/api/admin/learning", headers=H(A)).json()["items"] if i["rating"] == 0]
    assert (top["confidence"] >= 0.9) == (len(auto) == 1)
    if auto:
        assert auto[0]["label"] == top["label"] and auto[0]["image"].startswith("data:image/png")
    c.post("/api/analyze", headers=H(X), files={"file": ("p.png", png(), "image/png")})  # opted out: never collected
    assert len([i for i in c.get("/api/admin/learning", headers=H(A)).json()["items"] if i["rating"] == 0]) == len(auto)

    # ---- Build dataset straight from users
    base = {"kind": "text", "dataset_ids": []}
    p = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"users": {"feedback": True}}}).json()
    assert p["by_source"] == {"user feedback": 5} and "Deer: I live in the Kentucky woods." in "\n".join(p["sample"])
    p = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"users": {"messages": True}, "dedupe": False}}).json()
    assert p["by_source"]["what users wrote"] >= 6 and not any("secret" in x for x in p["sample"])
    p = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"users": {"conversations": True}}}).json()
    assert p["by_source"].get("user conversations", 0) >= 1  # the 5 corrected (disliked) replies are left out
    pi = c.post("/api/admin/curate/preview", headers=H(A), json={"kind": "images", "dataset_ids": [],
                "filters": {"users": {"pictures": True}, "dedupe": False}}).json()
    assert pi["before"] == 1  # only the opted-in user's picture
    built = c.post("/api/admin/curate/build", headers=H(A), json={"kind": "images", "dataset_ids": [ids], "name": "shapes + users",
                   "filters": {"users": {"pictures": True}, "dedupe": False, "label": {"model_id": eyes, "min_confidence": 0.3}}})
    assert built.status_code == 200 and "unlabeled" not in built.json()["stats"]["classes"]
