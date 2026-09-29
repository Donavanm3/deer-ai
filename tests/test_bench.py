"""Benchmarks, the multi-line chat format, code uploads, and the admin overview."""
import io, os, sys, tempfile, time, zipfile
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from deer import bench, chatfmt
from make_demo_data import make_code_chat, make_sprites

H = lambda t: {"Authorization": f"Bearer {t}"}


def test_chat_format():
    t = ("Notes first.\n\nUser: jump?\nDeer: Use:\n```csharp\nUser: not a turn\nrb.AddForce(v);\n```\nThen test it.\n"
         "User: hi\nDeer: Hello!\nUser: no answer yet\n")
    u = chatfmt.parse(t)
    assert u[0] == {"plain": "Notes first."}
    assert u[1]["a"].endswith("Then test it.") and "User: not a turn" in u[1]["a"]
    assert u[2] == {"q": "hi", "a": "Hello!"} and u[3] == {"q": "no answer yet", "a": None}
    # a reply that starts with a code fence on the Deer: line
    u2 = chatfmt.parse("User: loop\nDeer: ```python\nfor i in range(3):\n    print(i)\n```\nUser: next\nDeer: ok\n")
    assert [x["q"] for x in u2] == ["loop", "next"] and u2[0]["a"].endswith("```")
    tr = chatfmt.to_training(t)
    assert tr.count(chatfmt.EOT) == 2 and chatfmt.parse(tr) == u


def test_scoring():
    assert bench.f1("the white oak", "The white oak.") > 0.8 and bench.f1("pizza", "white oak") == 0
    assert bench.code_ok("no code") is None
    assert bench.code_ok("```python\nfor i in range(3):\n    print(i)\n```") is True
    assert bench.code_ok("```python\nfor i in range(3)\n    print(i)\n```") is False
    assert bench.code_ok("```javascript\nfunction f() { return [1, 2]; }\n```") is True
    assert bench.code_ok("```javascript\nfunction f() { return [1, 2; }\n```") is False


def test_stream_stops():
    """Replies end at the end-of-reply byte, and older models stop at the next 'User:' turn."""
    from deer.models import DeerChat, build_config
    m = DeerChat(build_config("chat", "tiny"))
    m.ln.g.data[:] = 0; m.ln.b.data[:] = 1  # final layer outputs all ones, so the head alone picks the byte
    W = m.head.w.data
    W[:] = -1
    W[:, ord("a")] = 1                      # always "a": lets us test stopping deterministically
    assert m.generate(b"User: x\nDeer:", max_new=5, top_k=1, stop=[b"a"]) == ""
    out = m.generate(b"User: x\nDeer:", max_new=5, top_k=1, stop=[b"\nUser:"])
    assert out == "aaaaa"


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_bench_")
    with TestClient(APP.app) as client:
        yield client


def wait(c, A, mid):
    for _ in range(600):
        m = c.get(f"/api/admin/models/{mid}", headers=H(A)).json()
        if m["status"] in ("ready", "error", "stopped"):
            return m
        time.sleep(1)


def test_benchmarks(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    # code: a project zip becomes training text with file names and fenced code
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mygame/main.py", "import pygame\nprint('go')\n")
        z.writestr("mygame/node_modules/junk.js", "x")
        z.writestr("mygame/art/hero.png", b"\x89PNG....")
    r = c.post("/api/admin/datasets", headers=H(A), data={"name": "project", "kind": "text"},
               files=[("files", ("mygame.zip", buf.getvalue(), "application/zip")),
                      ("files", ("chat.txt", make_code_chat(400).encode(), "text/plain"))]).json()
    corpus = open(os.path.join(db.dataset_dir(r["id"]), "corpus.txt")).read()
    assert "File: mygame/main.py\n```python\nimport pygame" in corpus and "junk" not in corpus
    assert r["stats"]["pairs"] >= 400 and r["stats"]["code_blocks"] > 100

    ids = c.post("/api/admin/datasets", headers=H(A), data={"name": "sprites", "kind": "images"},
                 files=[("files", ("s.zip", make_sprites(20), "application/zip"))]).json()["id"]
    mk = lambda name, kind, ds, steps, **cfg: c.post("/api/admin/models", headers=H(A), json={
        "name": name, "kind": kind, "dataset_id": ds, "preset": "tiny", "config": {"eval_interval": 10} | cfg}).json()["id"]
    chat = mk("coder", "chat", r["id"], 0, block_size=64)
    eyes = mk("eyes", "vision", ids, 0)
    eyes2 = mk("eyes2", "vision", ids, 0)
    art = mk("art", "canvas", ids, 0)
    for m, steps in ((chat, 40), (eyes, 80), (eyes2, 5), (art, 20)):
        c.post(f"/api/admin/models/{m}/train", headers=H(A), json={"steps": steps})
        assert wait(c, A, m)["status"] == "ready"

    # a taught multi-line code lesson keeps its line breaks
    from deer import data as D
    D.add_lesson(db.dataset_dir(r["id"]), "loop?", "```python\nfor i in range(2):\n    print(i)\n```")
    assert "for i in range(2):\n    print(i)\n```\x03" in open(os.path.join(db.dataset_dir(r["id"]), "learned.txt")).read()

    # chat benchmark from written questions and from held-out dataset text
    bad = c.post("/api/admin/benchmarks", headers=H(A), json={"name": "x", "kind": "chat", "source": "custom", "text": "no pairs"})
    assert bad.status_code == 400
    b1 = c.post("/api/admin/benchmarks", headers=H(A), json={"name": "mine", "kind": "chat", "source": "custom",
                "text": "User: Hello!\nDeer: Hi! I'm Deer AI.\nUser: loop\nDeer: ```python\nfor i in range(3):\n    print(i)\n```"}).json()
    assert b1["size"] == 2
    b2 = c.post("/api/admin/benchmarks", headers=H(A), json={"name": "held out", "kind": "chat", "source": "dataset",
                "dataset_id": r["id"], "size": 5}).json()
    assert b2["size"] == 5
    runs = c.post(f"/api/admin/benchmarks/{b1['id']}/run", headers=H(A), json={"model_ids": [chat]}).json()["runs"]
    assert 0 <= runs[0]["score"] <= 100 and set(runs[0]["metrics"]) >= {"Word match", "Exact answers", "Characters per second"}
    assert len(runs[0]["rows"]) == 2
    c.post(f"/api/admin/benchmarks/{b2['id']}/run", headers=H(A), json={"model_ids": [chat]})

    # vision: better-trained model should win
    b3 = c.post("/api/admin/benchmarks", headers=H(A), json={"name": "sprites test", "kind": "vision", "dataset_id": ids, "size": 10}).json()
    runs = c.post(f"/api/admin/benchmarks/{b3['id']}/run", headers=H(A), json={"model_ids": [eyes, eyes2]}).json()["runs"]
    assert runs[0]["score"] > runs[1]["score"] and set(runs[0]["per_class"]) == {"coin", "gem", "heart", "key", "potion", "slime", "sword", "tree"}
    assert c.post(f"/api/admin/benchmarks/{b3['id']}/run", headers=H(A), json={"model_ids": [chat]}).status_code == 400  # wrong type

    # canvas judged by an analysis model
    b4 = c.post("/api/admin/benchmarks", headers=H(A), json={"name": "art test", "kind": "canvas", "judge_model_id": eyes, "per_class": 4}).json()
    runs = c.post(f"/api/admin/benchmarks/{b4['id']}/run", headers=H(A), json={"model_ids": [art]}).json()["runs"]
    assert "Recognized as asked" in runs[0]["metrics"] and runs[0]["judge"] == "eyes"

    lst = c.get("/api/admin/benchmarks", headers=H(A)).json()
    top = next(b for b in lst if b["name"] == "sprites test")["leaders"]
    assert top[0]["model_name"] == "eyes"
    detail = c.get(f"/api/admin/benchmarks/{b3['id']}", headers=H(A)).json()
    assert len(detail["runs"]) == 2
    o = c.get("/api/admin/overview", headers=H(A)).json()
    assert o["trained"] == 4 and o["benchmarks"] == 4 and o["bench_runs"] >= 5 and o["datasets"] == {"text": 1, "images": 1}
    c.delete(f"/api/admin/benchmarks/{b1['id']}", headers=H(A))
    assert len(c.get("/api/admin/benchmarks", headers=H(A)).json()) == 3
