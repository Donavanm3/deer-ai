"""End-to-end API test: setup, datasets, training in a worker process, chat/analyze/imagine, permissions."""
import io, os, sys, time, tempfile
import pytest

DATA = tempfile.mkdtemp(prefix="deer_test_")
os.environ["DEER_DATA"] = DATA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from fastapi.testclient import TestClient
from deer.server.app import app
from make_demo_data import make_chat, make_shapes, draw_shape


@pytest.fixture(scope="module")
def c():
    from deer.server import db
    db.DATA_DIR = DATA
    with TestClient(app) as client:
        yield client


def H(tok):
    return {"Authorization": f"Bearer {tok}"}


def wait(c, tok, mid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = c.get(f"/api/admin/models/{mid}", headers=H(tok)).json()
        if m["status"] in ("ready", "error", "stopped"):
            return m
        time.sleep(1)
    raise TimeoutError


def test_full_flow(c):
    assert c.get("/api/auth/status").json()["needs_setup"] is True
    admin = c.post("/api/auth/register", json={"username": "deer", "password": "antlers123"}).json()
    assert admin["user"]["role"] == "admin"
    A = admin["token"]
    user = c.post("/api/auth/register", json={"username": "fawn", "password": "fawnfawn1"}).json()
    assert user["user"]["role"] == "user"
    U = user["token"]
    assert c.post("/api/auth/login", json={"username": "deer", "password": "wrongpass1"}).status_code == 401
    assert c.get("/api/admin/models", headers=H(U)).status_code == 403
    assert c.get("/api/models").status_code == 401

    # nothing trained yet -> friendly 409
    r = c.post("/api/chat", json={"text": "hi"}, headers=H(U))
    assert r.status_code == 409 and "admin" in r.json()["detail"]

    # datasets
    r = c.post("/api/admin/datasets", headers=H(A), data={"name": "talk", "kind": "text"},
               files=[("files", ("chat.txt", make_chat(800).encode(), "text/plain"))])
    assert r.status_code == 200, r.text
    tds = r.json()["id"]
    r = c.post("/api/admin/datasets", headers=H(A), data={"name": "shapes", "kind": "images"},
               files=[("files", ("s.zip", make_shapes(40), "application/zip"))])
    assert r.status_code == 200, r.text
    ids = r.json()["id"]
    assert r.json()["stats"]["classes"] == ["blue square", "green triangle", "orange ring", "red circle"]
    bad = c.post("/api/admin/datasets", headers=H(A), data={"name": "x", "kind": "images"},
                 files=[("files", ("s.zip", b"notazip", "application/zip"))])
    assert bad.status_code == 400

    # mismatched dataset kind rejected
    assert c.post("/api/admin/models", headers=H(A), json={"name": "x", "kind": "chat", "dataset_id": ids}).status_code == 400
    assert c.post("/api/admin/models", headers=H(A), json={"name": "x", "kind": "chat", "dataset_id": tds,
                                                            "config": {"n_head": 3}}).status_code == 400

    mids = {}
    for kind, ds in (("chat", tds), ("vision", ids), ("canvas", ids)):
        r = c.post("/api/admin/models", headers=H(A), json={"name": kind, "kind": kind, "dataset_id": ds,
                                                             "preset": "tiny", "config": {"eval_interval": 10}})
        assert r.status_code == 200, r.text
        mids[kind] = r.json()["id"]

    for kind in ("chat", "vision", "canvas"):
        r = c.post(f"/api/admin/models/{mids[kind]}/train", headers=H(A), json={"steps": 30})
        assert r.status_code == 200, r.text
        m = wait(c, A, mids[kind])
        assert m["status"] == "ready", m["error"]
        assert m["step"] == 30 and len(m["log"]) >= 3

    # resume training continues from checkpoint
    c.post(f"/api/admin/models/{mids['vision']}/train", headers=H(A), json={"steps": 10})
    assert wait(c, A, mids["vision"])["step"] == 40

    # unpublished models are invisible to users but usable by admin
    assert c.get("/api/models", headers=H(U)).json() == []
    assert c.post("/api/chat", json={"text": "hello"}, headers=H(A)).status_code == 200
    for mid in mids.values():
        assert c.patch(f"/api/admin/models/{mid}", headers=H(A), json={"published": True}).status_code == 200
    assert {m["kind"] for m in c.get("/api/models", headers=H(U)).json()} == {"chat", "vision", "canvas"}

    r = c.post("/api/chat", json={"text": "Hello!"}, headers=H(U)).json()
    cid = r["conversation_id"]
    assert r["messages"][1]["role"] == "assistant" and r["messages"][1]["content"]
    r = c.post("/api/chat", json={"text": "Who are you?", "conversation_id": cid}, headers=H(U)).json()
    assert r["conversation_id"] == cid

    buf = io.BytesIO(); draw_shape("circle", (220, 50, 50)).save(buf, "PNG")
    r = c.post("/api/analyze", headers=H(U), data={"conversation_id": cid},
               files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    meta = r.json()["messages"][1]["meta"]
    assert len(meta["predictions"]) == 4 and meta["stats"]["palette"]
    assert c.post("/api/analyze", headers=H(U), files={"file": ("x.png", b"nope", "image/png")}).status_code == 400

    r = c.post("/api/imagine", json={"prompt": "a green triangle", "count": 3, "conversation_id": cid}, headers=H(U)).json()
    imgs = r["messages"][1]["meta"]["images"]
    assert len(imgs) == 3 and all(i["label"] == "green triangle" for i in imgs)
    assert imgs[0]["data_url"].startswith("data:image/png;base64,")

    conv = c.get(f"/api/conversations/{cid}", headers=H(U)).json()
    assert len(conv["messages"]) == 8
    assert c.get(f"/api/conversations/{cid}", headers=H(A)).status_code == 404  # other users can't read it

    # stop mid-training
    c.post(f"/api/admin/models/{mids['canvas']}/train", headers=H(A), json={"steps": 100000})
    time.sleep(8)
    assert c.post(f"/api/admin/models/{mids['canvas']}/stop", headers=H(A)).status_code == 200
    m = wait(c, A, mids["canvas"])
    assert m["status"] == "stopped" and 30 < m["step"] < 100030

    assert c.get(f"/api/admin/models/{mids['chat']}/export", headers=H(A)).status_code == 200
    assert c.delete(f"/api/admin/datasets/{tds}", headers=H(A)).status_code == 409
    assert c.post(f"/api/admin/models/{mids['chat']}/reset", headers=H(A)).status_code == 200
    assert c.delete(f"/api/admin/models/{mids['chat']}", headers=H(A)).status_code == 200
    assert c.delete(f"/api/admin/datasets/{tds}", headers=H(A)).status_code == 200

    # registration toggle
    c.patch("/api/admin/settings", headers=H(A), json={"allow_registration": False})
    assert c.post("/api/auth/register", json={"username": "late", "password": "latelate1"}).status_code == 403
