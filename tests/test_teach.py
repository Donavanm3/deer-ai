"""Models training models: start from a model, learn from a teacher, and models as data sources."""
import io, os, sys, tempfile, time, zipfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from make_demo_data import make_chat, make_shapes, draw_shape

H = lambda t: {"Authorization": f"Bearer {t}"}


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_teach_")
    with TestClient(APP.app) as client:
        yield client


def make(c, A, name, kind, ds, preset="tiny", steps=20, **extra):
    body = {"name": name, "kind": kind, "dataset_id": ds, "preset": preset, "config": {"eval_interval": 5}} | extra
    if "config" in extra:
        body["config"] = {"eval_interval": 5} | extra["config"]
    r = c.post("/api/admin/models", headers=H(A), json=body)
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    c.post(f"/api/admin/models/{mid}/train", headers=H(A), json={"steps": steps})
    for _ in range(300):
        m = c.get(f"/api/admin/models/{mid}", headers=H(A)).json()
        if m["status"] in ("ready", "error", "stopped"):
            break
        time.sleep(1)
    assert m["status"] == "ready", m["error"]
    return m


def test_models_teach_models(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    tds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                 files=[("files", ("c.txt", make_chat(1500).encode(), "text/plain"))]).json()["id"]
    ids = c.post("/api/admin/datasets", headers=H(A), data={"name": "s", "kind": "images"},
                 files=[("files", ("s.zip", make_shapes(40), "application/zip"))]).json()["id"]
    chat_a = make(c, A, "chat A", "chat", tds, steps=150)
    vis_a = make(c, A, "eyes A", "vision", ids, steps=60)
    can_a = make(c, A, "canvas A", "canvas", ids, steps=20)

    # --- start from a model: a deeper chat model inherits the matching layers and starts far ahead
    fresh = make(c, A, "fresh", "chat", tds, steps=5, config={"n_layer": 3})
    warm = make(c, A, "warm", "chat", tds, steps=5, config={"n_layer": 3}, init_from=chat_a["id"])
    assert 0 < warm["meta"]["started_from"]["copied"] < warm["meta"]["started_from"]["total"]
    assert warm["log"][0]["loss"] < fresh["log"][0]["loss"] - 0.5
    # a vision model on a dataset with fewer classes keeps what overlaps
    sub = c.post("/api/admin/curate/build", headers=H(A), json={"kind": "images", "dataset_ids": [ids], "name": "3 shapes",
                 "filters": {"classes": {"orange ring": ""}}}).json()["id"]
    v3 = make(c, A, "eyes 3", "vision", sub, steps=5, init_from=vis_a["id"])
    assert v3["meta"]["classes"] == ["blue square", "green triangle", "red circle"] and v3["meta"]["started_from"]["copied"] > 0
    assert make(c, A, "canvas warm", "canvas", ids, steps=5, init_from=can_a["id"])["meta"]["started_from"]["copied"] > 0

    # --- teachers
    assert "teacher" in make(c, A, "student chat", "chat", tds, steps=10, teacher=chat_a["id"], teacher_strength=0.7)["meta"]
    stu = make(c, A, "student eyes", "vision", sub, steps=10, teacher=vis_a["id"])
    assert stu["meta"]["teacher"] == "eyes A"
    bad = lambda **kw: c.post("/api/admin/models", headers=H(A), json={"name": "x", "dataset_id": tds, "preset": "tiny"} | kw)
    assert bad(kind="chat", teacher=vis_a["id"]).status_code == 400                       # wrong type
    assert bad(kind="chat", teacher=chat_a["id"], config={"block_size": 128}).status_code == 400  # teacher reads less context
    assert c.post("/api/admin/models", headers=H(A), json={"name": "x", "kind": "canvas", "dataset_id": ids,
                  "preset": "tiny", "teacher": can_a["id"]}).status_code == 400
    untrained = c.post("/api/admin/models", headers=H(A), json={"name": "u", "kind": "chat", "dataset_id": tds, "preset": "tiny"}).json()["id"]
    assert bad(kind="chat", init_from=untrained).status_code == 400

    # --- models as data sources
    p = c.post("/api/admin/curate/preview", headers=H(A), json={"kind": "text", "dataset_ids": [tds],
               "filters": {"generate": {"model_id": chat_a["id"], "count": 30}}}).json()
    assert any(k.startswith("written by chat A") for k in p["by_source"]) and "12 sample" in p["generated_note"]
    r = c.post("/api/admin/curate/build", headers=H(A), json={"kind": "text", "dataset_ids": [], "name": "written",
               "filters": {"generate": {"model_id": chat_a["id"], "count": 40}, "dedupe": False, "remove_junk": False}})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["built_from"]["units"] == 40

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # flat zip = unlabeled pictures
        for i in range(8):
            b = io.BytesIO(); draw_shape("circle" if i % 2 else "square", (220, 50, 50) if i % 2 else (50, 90, 220)).save(b, "PNG")
            z.writestr(f"pic{i}.png", b.getvalue())
    un = c.post("/api/admin/datasets", headers=H(A), data={"name": "unlabeled", "kind": "images"},
                files=[("files", ("u.zip", buf.getvalue(), "application/zip"))]).json()
    assert un["stats"]["classes"] == ["unlabeled"]
    p = c.post("/api/admin/curate/preview", headers=H(A), json={"kind": "images", "dataset_ids": [un["id"]],
               "filters": {"dedupe": False, "label": {"model_id": vis_a["id"], "min_confidence": 0.3}}}).json()
    assert p["labeled_by_model"] > 0 and "unlabeled" not in p["classes"] and set(p["classes"]) <= {"red circle", "blue square", "green triangle", "orange ring"}
    strict = c.post("/api/admin/curate/preview", headers=H(A), json={"kind": "images", "dataset_ids": [un["id"]],
                    "filters": {"label": {"model_id": vis_a["id"], "min_confidence": 0.9999}}}).json()
    assert strict["removed"].get("model unsure", 0) >= 1

    r = c.post("/api/admin/curate/build", headers=H(A), json={"kind": "images", "dataset_ids": [], "name": "drawn",
               "filters": {"dedupe": False, "generate": {"model_id": can_a["id"], "per_class": 6}}}).json()
    assert r["stats"]["counts"] == {c_: 6 for c_ in ["blue square", "green triangle", "orange ring", "red circle"]}
    assert not os.listdir(os.path.join(db.DATA_DIR, "tmp"))  # temp pictures cleaned up
