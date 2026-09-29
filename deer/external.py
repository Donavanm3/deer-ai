"""
Other AI models (ChatGPT, Claude, Gemini, local models via Ollama...) for benchmark comparison.

They only answer benchmark questions so you can compare Deer AI's models against them. Their
answers are stored with the benchmark results and are never used for training: no dataset,
learning or Build dataset feature reads them.

Two API styles cover almost everything:
  openai    - the OpenAI-compatible /chat/completions API (OpenAI, OpenRouter, Google Gemini's
              OpenAI endpoint, Groq, Mistral, Together, Ollama, LM Studio, llama.cpp, vLLM...)
  anthropic - Anthropic's /v1/messages API (Claude)
"""
import json
import time
import urllib.error
import urllib.request

from .server import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS external_models (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, api TEXT NOT NULL, base_url TEXT NOT NULL,
  model TEXT NOT NULL, api_key TEXT NOT NULL DEFAULT '', max_tokens INTEGER NOT NULL DEFAULT 800,
  created_at REAL NOT NULL);
"""

PRESETS = {  # shown in the admin form; base URLs can be edited
    "OpenAI": ("openai", "https://api.openai.com/v1"),
    "Anthropic (Claude)": ("anthropic", "https://api.anthropic.com"),
    "Google Gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "OpenRouter": ("openai", "https://openrouter.ai/api/v1"),
    "Ollama (runs on your server)": ("openai", "http://localhost:11434/v1"),
    "Other OpenAI-compatible": ("openai", ""),
}

SYSTEM = ("You are being evaluated on a benchmark. Answer the user's question directly and briefly, in the "
          "same style and length as the examples. When you write code, put it in a fenced code block with a "
          "language tag, like ```python.")
TIMEOUT = 90


class ExternalError(Exception):
    pass


def init():
    db.conn().executescript(SCHEMA)


def _http(url, headers, body, timeout=TIMEOUT):  # replaced in tests
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def public(row):
    k = row["api_key"]
    return {"id": row["id"], "name": row["name"], "api": row["api"], "base_url": row["base_url"],
            "model": row["model"], "max_tokens": row["max_tokens"],
            "key": f"…{k[-4:]}" if len(k) > 8 else ("set" if k else "none")}


def listing():
    return [public(r) for r in db.q("SELECT * FROM external_models ORDER BY name")]


def add(name, api, base_url, model, api_key="", max_tokens=800):
    if api not in ("openai", "anthropic"):
        raise ExternalError("API style must be openai or anthropic.")
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ExternalError("The base URL must start with http:// or https://.")
    if not (name or "").strip() or not (model or "").strip():
        raise ExternalError("Give it a name and the model's ID (for example gpt-4o-mini).")
    eid = db.ex("INSERT INTO external_models(name,api,base_url,model,api_key,max_tokens,created_at) VALUES(?,?,?,?,?,?,?)",
                (name.strip()[:60], api, base_url, model.strip()[:120], (api_key or "").strip(),
                 max(50, min(int(max_tokens or 800), 8000)), db.now()))
    return public(db.q1("SELECT * FROM external_models WHERE id=?", (eid,)))


def get(eid):
    r = db.q1("SELECT * FROM external_models WHERE id=?", (eid,))
    if not r:
        raise ExternalError("That outside model isn't set up anymore.")
    return r


def ask(ext, question, examples=()):
    """One question to an outside model. `examples` are (question, answer) pairs showing the
    expected style, so replies are scored fairly against short expected answers."""
    shots = []
    for q, a in examples:
        shots += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs = shots + [{"role": "user", "content": question}]
    if ext["api"] == "anthropic":
        url = ext["base_url"] + "/v1/messages"
        headers = {"x-api-key": ext["api_key"], "anthropic-version": "2023-06-01"}
        body = {"model": ext["model"], "max_tokens": ext["max_tokens"], "system": SYSTEM, "messages": msgs,
                "temperature": 0.2}
    else:
        url = ext["base_url"] + "/chat/completions"
        headers = {"Authorization": f"Bearer {ext['api_key']}"} if ext["api_key"] else {}
        body = {"model": ext["model"], "max_tokens": ext["max_tokens"], "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM}] + msgs}
    try:
        status, raw = _http(url, headers, body)
    except OSError as e:
        raise ExternalError(f"Couldn't reach {ext['name']}: {e}")
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        data = {}
    if status >= 300:
        err = data.get("error")
        msg = (err.get("message") if isinstance(err, dict) else err) or raw[:200].decode("utf-8", "replace")
        raise ExternalError(f"{ext['name']} said ({status}): {msg}")
    try:
        if ext["api"] == "anthropic":
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        raise ExternalError(f"{ext['name']} sent back an answer in a format Deer AI doesn't recognize.")


def test(eid):
    ext = get(eid)
    t = time.time()
    reply = ask(ext, "Reply with the single word: ready")
    return {"ok": True, "reply": reply[:200], "seconds": round(time.time() - t, 2)}
