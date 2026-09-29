"""Comparing against other AI models: OpenAI-style and Anthropic-style APIs (served by a local fake
server, so the real HTTP code runs), answer sheets, leaderboards, and that nothing trains on them."""
import json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP

H = lambda t: {"Authorization": f"Bearer {t}"}
TEXT = ("User: What is 2 plus 2?\nDeer: 2 plus 2 is 4.\n"
        "User: Write a loop\nDeer: ```python\nfor i in range(3):\n    print(i)\n```\n"
        "User: Who made you?\nDeer: Dashlith made me.\n")
ANSWERS = {"What is 2 plus 2?": "2 plus 2 is 4.", "Write a loop": "```python\nfor i in range(3):\n    print(i)\n```",
           "Who made you?": "I was made by a company."}
SEEN = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, dict(self.headers), body))
        q = body["messages"][-1]["content"]
        if self.path == "/v1/chat/completions":
            if self.headers.get("Authorization") != "Bearer sk-good-key-1234":
                return self._send(401, {"error": {"message": "Incorrect API key provided"}})
            return self._send(200, {"choices": [{"message": {"content": ANSWERS.get(q, "ready")}}]})
        if self.path == "/v1/messages":
            if self.headers.get("x-api-key") != "ak-good-key-5678" or self.headers.get("anthropic-version") != "2023-06-01":
                return self._send(401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}})
            return self._send(200, {"content": [{"type": "text", "text": ANSWERS.get(q, "ready") + " "}]})
        self._send(404, {})

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_ext_")
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with TestClient(APP.app) as client:
        client.fake = f"http://127.0.0.1:{srv.server_address[1]}"
        yield client
    srv.shutdown()


def test_compare_with_other_ai(c):
    A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
    ds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                files=[("files", ("t.txt", (TEXT * 40).encode(), "text/plain"))]).json()["id"]
    mid = c.post("/api/admin/models", headers=H(A), json={"name": "deer", "kind": "chat", "dataset_id": ds, "preset": "tiny"}).json()["id"]
    c.post(f"/api/admin/models/{mid}/train", headers=H(A), json={"steps": 10})
    for _ in range(120):
        if c.get(f"/api/admin/models/{mid}", headers=H(A)).json()["status"] == "ready":
            break
        time.sleep(1)
    b = c.post("/api/admin/benchmarks", headers=H(A), json={"name": "mix", "kind": "chat", "source": "custom", "text": TEXT}).json()

    # set up outside models; keys are never sent back
    presets = c.get("/api/admin/external", headers=H(A)).json()["presets"]
    assert presets["Anthropic (Claude)"]["api"] == "anthropic" and "Ollama (runs on your server)" in presets
    gpt = c.post("/api/admin/external", headers=H(A), json={"name": "GPT", "api": "openai", "base_url": c.fake + "/v1",
                 "model": "gpt-test", "api_key": "sk-good-key-1234"}).json()
    claude = c.post("/api/admin/external", headers=H(A), json={"name": "Claude", "api": "anthropic", "base_url": c.fake,
                    "model": "claude-test", "api_key": "ak-good-key-5678"}).json()
    wrong = c.post("/api/admin/external", headers=H(A), json={"name": "Bad key", "api": "openai", "base_url": c.fake + "/v1",
                   "model": "x", "api_key": "sk-nope-nope"}).json()
    listed = json.dumps(c.get("/api/admin/external", headers=H(A)).json(), ensure_ascii=False)
    assert "sk-good-key-1234" not in listed and "ak-good-key-5678" not in listed and "…1234" in listed and "…5678" in listed
    assert c.post("/api/admin/external", headers=H(A), json={"name": "x", "api": "openai", "base_url": "ftp://example.com", "model": "m"}).status_code == 400
    assert c.post(f"/api/admin/external/{gpt['id']}/test", headers=H(A)).json()["reply"] == "ready"
    r = c.post(f"/api/admin/external/{wrong['id']}/test", headers=H(A))
    assert r.status_code == 400 and "Incorrect API key" in r.json()["detail"]

    # run Deer + both outside models on the same benchmark
    runs = c.post(f"/api/admin/benchmarks/{b['id']}/run", headers=H(A),
                  json={"model_ids": [mid], "external_ids": [gpt["id"], claude["id"]]}).json()["runs"]
    by = {r["model_name"]: r for r in runs}
    assert set(by) == {"deer", "GPT (gpt-test)", "Claude (claude-test)"}
    g = by["GPT (gpt-test)"]
    assert g["source"] == "external" and g["metrics"]["Exact answers"] > 60 and g["metrics"]["Well-formed code"] == 100
    assert by["Claude (claude-test)"]["rows"][0]["got"] == "2 plus 2 is 4."  # trailing space trimmed
    # the anthropic request used the right headers and format; style examples never include the question asked
    path, hdrs, body = next(x for x in SEEN if x[0] == "/v1/messages" and x[2]["messages"][-1]["content"] == "Who made you?")
    assert body["model"] == "claude-test" and "system" in body and body["max_tokens"] == 800
    assert all(m["content"] != "Who made you?" for m in body["messages"][:-1]) and len(body["messages"]) == 5
    # a bad key fails clearly instead of scoring zero
    r = c.post(f"/api/admin/benchmarks/{b['id']}/run", headers=H(A), json={"external_ids": [wrong["id"]]})
    assert r.status_code == 400 and "Incorrect API key" in r.json()["detail"]

    # answer sheets: export questions, paste answers collected anywhere
    qs = c.get(f"/api/admin/benchmarks/{b['id']}/sheet", headers=H(A)).json()["questions"]
    assert qs == ["What is 2 plus 2?", "Write a loop", "Who made you?"]
    sheet = c.post(f"/api/admin/benchmarks/{b['id']}/sheet", headers=H(A),
                   json={"name": "Copilot", "answers": ["4", None, "Dashlith made me."]}).json()
    assert sheet["source"] == "sheet" and sheet["metrics"]["Characters per second"] is None
    text = "User: Who made you?\nDeer: Dashlith made me.\nUser: What is 2 plus 2?\nDeer: 2 plus 2 is 4.\n"
    sheet2 = c.post(f"/api/admin/benchmarks/{b['id']}/sheet", headers=H(A), json={"name": "Manual", "answers": text}).json()
    assert sheet2["rows"][2]["exact"] and sheet2["rows"][1]["got"] == ""
    assert c.post(f"/api/admin/benchmarks/{b['id']}/sheet", headers=H(A), json={"answers": ["only one"]}).status_code == 400

    # leaderboard mixes everyone; vision benchmarks refuse outside models
    leaders = next(x for x in c.get("/api/admin/benchmarks", headers=H(A)).json() if x["id"] == b["id"])["leaders"]
    assert {l["source"] for l in leaders} == {"deer", "external", "sheet"} and leaders[0]["score"] >= leaders[-1]["score"]

    # nothing an outside AI said ends up in training data
    folder = db.dataset_dir(ds)
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            assert "I was made by a company" not in open(os.path.join(folder, fn)).read()
    p = c.post("/api/admin/curate/preview", headers=H(A), json={"kind": "text", "dataset_ids": [],
               "filters": {"users": {"feedback": True, "messages": True, "conversations": True}}}).json()
    assert not any("made by a company" in s for s in p.get("sample", []))
