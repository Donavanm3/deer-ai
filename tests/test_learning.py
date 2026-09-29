"""Learning-over-time tests: feedback, admin review, dataset growth, class growth, auto-learn, rollback."""
import io, json, os, sys, tempfile, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from deer import training
from make_demo_data import make_chat, make_shapes, draw_shape

H = lambda t: {"Authorization": f"Bearer {t}"}


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_learn_")
    with TestClient(APP.app) as client:
        yield client


def wait(c, A, mid):
    for _ in range(300):
        m = c.get(f"/api/admin/models/{mid}", headers=H(A)).json()
        if m["status"] in ("ready", "error", "stopped"):
            return m
        time.sleep(1)
    raise TimeoutError


def png(kind="circle", color=(220, 50, 50)):
    b = io.BytesIO(); draw_shape(kind, color).save(b, "PNG"); return b.getvalue()


def test_worse_guard():
    assert training._worse("vision", 0.9, 0.85) and not training._worse("vision", 0.9, 0.89)
    assert training._worse("chat", 1.0, 1.1) and not training._worse("chat", 1.0, 1.05)
    assert not training._worse("chat", 0.14, 0.16)  # tiny change on an already low loss is fine
    assert training._worse("canvas", 1000, 1100)
    assert not training._worse("chat", None, 2.0)


def test_learning_flow(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    ur = c.post("/api/auth/register", json={"username": "teacher", "password": "teacher123"}).json()
    U = ur["token"]
    tds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                 files=[("files", ("c.txt", make_chat(800).encode(), "text/plain"))]).json()["id"]
    ids = c.post("/api/admin/datasets", headers=H(A), data={"name": "s", "kind": "images"},
                 files=[("files", ("s.zip", make_shapes(30), "application/zip"))]).json()["id"]
    mids = {}
    for kind, ds in (("chat", tds), ("vision", ids), ("canvas", ids)):
        mids[kind] = c.post("/api/admin/models", headers=H(A), json={"name": kind, "kind": kind, "dataset_id": ds,
                            "preset": "tiny", "config": {"eval_interval": 10}}).json()["id"]
        c.post(f"/api/admin/models/{mids[kind]}/train", headers=H(A), json={"steps": 20})
        assert wait(c, A, mids[kind])["status"] == "ready"
        c.patch(f"/api/admin/models/{mids[kind]}", headers=H(A), json={"published": True})
    assert "red circle" in next(m for m in c.get("/api/models", headers=H(U)).json() if m["kind"] == "vision")["classes"]

    # --- users give feedback
    r1 = c.post("/api/chat", headers=H(U), json={"text": "Who made you?"}).json()
    cid = r1["conversation_id"]; m1 = r1["messages"][1]["id"]
    m2 = c.post("/api/chat", headers=H(U), json={"text": "What is your favorite tree?", "conversation_id": cid}).json()["messages"][1]["id"]
    assert c.post("/api/feedback", headers=H(U), json={"message_id": m1, "rating": 1}).json()["status"] == "pending"
    fb = c.post("/api/feedback", headers=H(U), json={"message_id": m2, "rating": -1,
                "correction": "My favorite tree is the white oak."}).json()
    assert fb["status"] == "pending"
    an = c.post("/api/analyze", headers=H(U), data={"conversation_id": cid},
                files={"file": ("p.png", png("square", (130, 40, 200)), "image/png")}).json()["messages"][1]["id"]
    assert c.post("/api/feedback", headers=H(U), json={"message_id": an, "rating": -1, "label": "purple square"}).json()["status"] == "pending"
    im = c.post("/api/imagine", headers=H(U), json={"prompt": "red circle", "conversation_id": cid}).json()["messages"][1]["id"]
    assert c.post("/api/feedback", headers=H(U), json={"message_id": im, "rating": 1}).json()["status"] == "info"
    assert c.post("/api/feedback", headers=H(A), json={"message_id": m1, "rating": 1}).status_code == 400  # not their chat
    conv = c.get(f"/api/conversations/{cid}", headers=H(U)).json()
    assert next(m for m in conv["messages"] if m["id"] == m2)["meta"]["feedback"]["rating"] == -1

    # --- admin reviews
    inbox = c.get("/api/admin/learning", headers=H(A)).json()
    assert len(inbox["items"]) == 3 and any(i.get("image", "").startswith("data:image/png") for i in inbox["items"])
    assert inbox["counts"]["info"] == 1
    r = c.post("/api/admin/learning/review", headers=H(A), json={"ids": [i["id"] for i in inbox["items"]], "action": "approve"}).json()
    assert r["done"] == 3 and r["counts"]["applied"] == 3
    learned = open(os.path.join(db.dataset_dir(tds), "learned.txt")).read()
    assert "Deer: My favorite tree is the white oak." in learned
    ds = next(d for d in c.get("/api/admin/datasets", headers=H(A)).json() if d["id"] == ids)
    assert "purple square" in ds["stats"]["classes"] and ds["stats"]["counts"]["purple square"] == 1
    assert c.post("/api/feedback", headers=H(U), json={"message_id": m2, "rating": 1}).status_code == 400  # already applied

    # --- auto-learn: vision grows a new class through the scheduler
    assert c.get(f"/api/admin/models/{mids['vision']}", headers=H(A)).json()["new_lessons"] == 1
    c.patch(f"/api/admin/models/{mids['vision']}/learning", headers=H(A), json={"auto_learn": True, "learn_steps": 20, "learn_min": 1})
    assert APP._learn_check() == [mids["vision"]]
    m = wait(c, A, mids["vision"])
    assert m["status"] == "ready" and m["meta"]["last_learn"]["outcome"] in ("kept", "reverted")
    if m["meta"]["last_learn"]["outcome"] == "kept":
        assert "purple square" in m["meta"]["classes"] and m["step"] == 40
    assert APP._learn_check() == []  # nothing new since
    r = c.post("/api/analyze", headers=H(U), files={"file": ("p.png", png(), "image/png")})
    assert r.status_code == 200 and r.json()["messages"][1]["meta"]["predictions"]

    # canvas shares the dataset: learn now, then generate
    assert c.post(f"/api/admin/models/{mids['canvas']}/learn", headers=H(A)).status_code == 200
    m = wait(c, A, mids["canvas"])
    assert m["status"] == "ready", m["error"]
    assert c.post("/api/imagine", headers=H(U), json={"prompt": "purple square", "count": 2}).status_code == 200

    # --- manual rollback of the chat model's last run
    c.post(f"/api/admin/models/{mids['chat']}/train", headers=H(A), json={"steps": 10})
    assert wait(c, A, mids["chat"])["step"] == 30
    m = c.post(f"/api/admin/models/{mids['chat']}/rollback", headers=H(A)).json()
    assert m["step"] == 20 and max(l["step"] for l in m["log"]) <= 20
    assert c.post(f"/api/admin/models/{mids['chat']}/rollback", headers=H(A)).status_code == 409
    assert c.post("/api/chat", headers=H(U), json={"text": "hi"}).status_code == 200

    # --- auto-approve
    c.patch("/api/admin/learning/settings", headers=H(A), json={"auto_approve": True})
    m3 = c.post("/api/chat", headers=H(U), json={"text": "Hello!"}).json()["messages"][1]["id"]
    assert c.post("/api/feedback", headers=H(U), json={"message_id": m3, "rating": 1}).json()["status"] == "applied"


def test_class_growth_preserves_knowledge():
    import numpy as np
    from deer.models import build_model, build_config
    from deer.engine import Tensor
    x = Tensor(np.random.default_rng(1).random((2, 32, 32, 3)).astype(np.float32))
    old, new = ["blue", "red"], ["blue", "green", "red"]  # new class lands in the middle
    for kind in ("vision", "canvas"):
        cfg = build_config(kind, "tiny")
        a = build_model(kind, cfg, 2)
        b = build_model(kind, cfg, 3)
        b.load_state_dict(training.remap_classes(kind, a.state_dict(), old, new, cfg))
        if kind == "vision":
            la, lb = a.forward(x).data, b.forward(x).data
            assert np.allclose(la[:, 0], lb[:, 0], atol=1e-5) and np.allclose(la[:, 1], lb[:, 2], atol=1e-5)
        else:
            ia, ib = a.sample([1, 0], seed=3), b.sample([2, 0], seed=3)  # "red" is 1 before, 2 after
            assert np.allclose(ia, ib, atol=1e-5)
