"""Combine + filter datasets into a new one, then train on it."""
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
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_cur_")
    with TestClient(APP.app) as client:
        yield client


def upload_text(c, A, name, text):
    return c.post("/api/admin/datasets", headers=H(A), data={"name": name, "kind": "text"},
                  files=[("files", ("a.txt", text.encode(), "text/plain"))]).json()["id"]


def test_curate(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    t1 = upload_text(c, A, "starter", make_chat(300))
    # plain text comes first: a reply runs until the next "User:" line, so text after it belongs to it
    t2 = upload_text(c, A, "extra", ("%%%% $$$$ #### 1234 5678 !!!!\n\n"
                     "The white oak grows slowly and lives for centuries.\n\n") * 3 +
                     ("User: What is an acorn?\nDeer: An acorn is the nut of an oak tree.\n"
                      "User: What is an acorn?\nDeer: An acorn is the nut of an oak tree.\n"
                      "User: Say something rude\nDeer: You are a stinkhead.\n"
                      "User: Show me a loop\nDeer: Here you go:\n```python\nfor i in range(3):\n    print(i)\n```\n") * 3)
    # user lesson on t2
    with open(os.path.join(db.dataset_dir(t2), "learned.txt"), "w") as f:
        f.write("User: Favorite tree?\nDeer: The white oak.\n")

    base = {"kind": "text", "dataset_ids": [t1, t2]}
    p = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"dedupe": False, "remove_junk": False}}).json()
    raw = p["after"]
    p = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {
        "dedupe": True, "remove_junk": True, "block_words": "stinkhead", "chat_only": True}}).json()
    assert p["after"] < raw and p["removed"]["duplicate"] > 0 and p["removed"]["blocked word"] == 3
    assert p["removed"]["not a User/Deer pair"] >= 1
    pj = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"remove_junk": True, "dedupe": False}}).json()
    assert pj["removed"]["junk or symbols"] == 3
    assert any("user lessons" in s for s in p["by_source"]) and all("stinkhead" not in x for x in p["sample"])
    no_lessons = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"include_learned": False}}).json()
    assert not any("user lessons" in s for s in no_lessons["by_source"])
    only_oak = c.post("/api/admin/curate/preview", headers=H(A), json=base | {"filters": {"must_contain": "oak"}}).json()
    assert 0 < only_oak["after"] < 10

    r = c.post("/api/admin/curate/build", headers=H(A), json=base | {"name": "clean chat", "filters": {
        "dedupe": True, "block_words": "stinkhead", "chat_only": True}}).json()
    text = open(os.path.join(db.dataset_dir(r["id"]), "corpus.txt")).read()
    assert "stinkhead" not in text and "Deer: The white oak." in text and text.count("What is an acorn?") == 1
    assert "for i in range(3):\n    print(i)" in text  # code survives the junk filter with its line breaks
    assert r["stats"]["built_from"]["datasets"] == [t1, t2] and r["suggested_preset"]["preset"] == "tiny"
    assert c.post("/api/admin/curate/build", headers=H(A), json=base | {"name": "x", "filters": {"must_contain": "zzzqqq"}}).status_code == 400

    # ---- images: merge two datasets, rename + drop classes, dedupe, balance
    i1 = c.post("/api/admin/datasets", headers=H(A), data={"name": "shapes", "kind": "images"},
                files=[("files", ("s.zip", make_shapes(12), "application/zip"))]).json()["id"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(5):
            b = io.BytesIO(); draw_shape("circle", (220, 50, 50)).save(b, "PNG")
            z.writestr(f"Red Circle/{i}.png", b.getvalue())
        same = io.BytesIO(); draw_shape("square", (50, 200, 200)).save(same, "PNG")
        for i in range(3):  # exact duplicates
            z.writestr(f"teal box/{i}.png", same.getvalue())
    i2 = c.post("/api/admin/datasets", headers=H(A), data={"name": "more", "kind": "images"},
                files=[("files", ("m.zip", buf.getvalue(), "application/zip"))]).json()["id"]
    ib = {"kind": "images", "dataset_ids": [i1, i2]}
    f = {"classes": {"Red Circle": "red circle", "orange ring": ""}, "dedupe": True, "max_per_class": 10}
    p = c.post("/api/admin/curate/preview", headers=H(A), json=ib | {"filters": f}).json()
    assert p["before"] == 12 * 4 + 8
    assert "orange ring" not in p["classes"] and "Red Circle" not in p["classes"]
    assert p["classes"]["red circle"]["count"] == 10 and p["classes"]["teal box"]["count"] == 1
    assert p["removed"]["class left out"] == 12 and p["removed"]["duplicate"] >= 2
    assert p["classes"]["blue square"]["samples"][0].startswith("data:image/jpeg")
    r = c.post("/api/admin/curate/build", headers=H(A), json=ib | {"name": "combined shapes", "filters": f | {"min_per_class": 2}}).json()
    assert r["stats"]["classes"] == ["blue square", "green triangle", "red circle"]  # teal box (1) dropped

    # a model trains on the new dataset
    mid = c.post("/api/admin/models", headers=H(A), json={"name": "v2", "kind": "vision", "dataset_id": r["id"],
                                                           "preset": "tiny", "config": {"eval_interval": 5}}).json()["id"]
    c.post(f"/api/admin/models/{mid}/train", headers=H(A), json={"steps": 10})
    for _ in range(120):
        m = c.get(f"/api/admin/models/{mid}", headers=H(A)).json()
        if m["status"] in ("ready", "error"):
            break
        time.sleep(1)
    assert m["status"] == "ready" and m["meta"]["classes"] == ["blue square", "green triangle", "red circle"]
