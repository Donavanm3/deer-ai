"""
Benchmarks: fixed tests you run against any of your models to see which is best.

  chat   - questions with expected answers (written by you, or taken from held-out dataset text).
           Scores word overlap with the expected answer, exact matches, whether code in the reply
           is well-formed, and speed.
  vision - a labeled image dataset. Scores accuracy (overall and per class), confidence and speed.
  canvas - draws pictures of every class and asks an analysis model you pick to recognize them.
           Scores how many are recognized as what was asked for, variety, and speed.

A benchmark's test items are frozen when it's created, so every model is judged on the same test.
"""
import ast
import json
import os
import random
import re
import time

import numpy as np
from PIL import Image

from . import chatfmt
from . import data as D
from . import inference as INF
from .server import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, spec TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS bench_runs (
  id INTEGER PRIMARY KEY, benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
  model_id INTEGER, model_name TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0, score REAL NOT NULL,
  results TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_bench_runs ON bench_runs(benchmark_id, score DESC);
"""
MAX_ITEMS = 200
_WORD = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


class BenchError(ValueError):
    pass


def init():
    c = db.conn()
    c.executescript(SCHEMA)
    have = {r["name"] for r in c.execute("PRAGMA table_info(bench_runs)").fetchall()}
    if "source" not in have:  # deer = your model, external = another AI over its API, sheet = pasted answers
        c.execute("ALTER TABLE bench_runs ADD COLUMN source TEXT NOT NULL DEFAULT 'deer'")


# ================================================================== creating
def create(name, kind, source="custom", text="", dataset_id=None, size=50, judge_model_id=None, per_class=8):
    name = (name or "").strip()[:80]
    if not name:
        raise BenchError("Give the benchmark a name.")
    if kind == "chat":
        spec = {"items": _chat_items(source, text, dataset_id, size)}
    elif kind == "vision":
        ds = db.q1("SELECT * FROM datasets WHERE id=? AND kind='images'", (dataset_id,))
        if not ds:
            raise BenchError("Pick an image dataset to test on.")
        spec = {"dataset_id": ds["id"], "dataset_name": ds["name"], "max_per_class": max(1, min(int(size or 50), 500))}
    elif kind == "canvas":
        judge = db.model_row(judge_model_id) if judge_model_id else None
        if not judge or judge["kind"] != "vision":
            raise BenchError("Pick an image analysis model to judge the drawings.")
        spec = {"judge_model_id": judge["id"], "judge_name": judge["name"], "per_class": max(1, min(int(per_class or 8), 64))}
    else:
        raise BenchError("Unknown benchmark type.")
    bid = db.ex("INSERT INTO benchmarks(name,kind,spec,created_at) VALUES(?,?,?,?)",
                (name, kind, json.dumps(spec), db.now()))
    return get(bid)


def _chat_items(source, text, dataset_id, size):
    size = max(1, min(int(size or 50), MAX_ITEMS))
    if source == "dataset":
        ds = db.q1("SELECT * FROM datasets WHERE id=? AND kind='text'", (dataset_id,))
        if not ds:
            raise BenchError("Pick a text dataset.")
        with open(os.path.join(db.dataset_dir(ds["id"]), "corpus.txt"), encoding="utf-8", errors="replace") as f:
            units = [u for u in chatfmt.parse(f.read()) if u.get("a")]
        if not units:
            raise BenchError("That dataset has no User/Deer conversations to test with.")
        held = units[int(len(units) * 0.9):] or units  # the same held-out part training validates on
        seen, items = set(), []
        for u in held:
            if u["q"] not in seen:
                seen.add(u["q"])
                items.append({"prompt": u["q"], "expected": u["a"]})
        random.Random(0).shuffle(items)
        items = items[:size]
    else:
        items = [{"prompt": u["q"], "expected": u["a"]} for u in chatfmt.parse(text or "") if u.get("a")][:MAX_ITEMS]
        if not items:
            raise BenchError('Write at least one test as "User: question" then "Deer: expected answer".')
    return items


def get(bid):
    b = db.q1("SELECT * FROM benchmarks WHERE id=?", (bid,))
    if not b:
        raise BenchError("Benchmark not found.")
    b["spec"] = json.loads(b["spec"])
    b["size"] = len(b["spec"].get("items", [])) if b["kind"] == "chat" else None
    return b


# ==================================================================== scoring
def _words(t):
    return [w.lower() for w in _WORD.findall(t or "")]


def f1(got, expected):
    """Word overlap between a reply and the expected answer (0..1)."""
    g, e = _words(got), _words(expected)
    if not g or not e:
        return float(g == e)
    common = {}
    for w in e:
        common[w] = common.get(w, 0) + 1
    hit = 0
    for w in g:
        if common.get(w, 0) > 0:
            hit += 1
            common[w] -= 1
    if not hit:
        return 0.0
    p, r = hit / len(g), hit / len(e)
    return 2 * p * r / (p + r)


_FENCE = re.compile(r"```([\w+#-]*)\n(.*?)```", re.S)


def code_ok(reply):
    """None if the reply has no code; otherwise whether every code block is well-formed
    (Python is fully parsed; other languages must have balanced brackets and quotes)."""
    blocks = _FENCE.findall(reply or "")
    if not blocks:
        return None
    for lang, code in blocks:
        if lang.lower() in ("python", "py"):
            try:
                ast.parse(code)
            except SyntaxError:
                return False
        elif not _balanced(code):
            return False
    return True


def _balanced(code):
    pairs, stack, quote = {")": "(", "]": "[", "}": "{"}, [], None
    i = 0
    while i < len(code):
        ch = code[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
        i += 1
    return not stack and quote is None


def _norm(t):
    return " ".join((t or "").lower().split())


# ==================================================================== running
def _save(bid, model_id, name, step, res, source):
    rid = db.ex("INSERT INTO bench_runs(benchmark_id,model_id,model_name,step,score,results,created_at,source) "
                "VALUES(?,?,?,?,?,?,?,?)", (bid, model_id, name, step, res["score"], json.dumps(res), db.now(), source))
    return {"id": rid, "model_id": model_id, "model_name": name, "step": step, "source": source, **res}


def run(bid, model_ids, external_ids=()):
    from . import external as EXT
    b = get(bid)
    if not model_ids and not external_ids:
        raise BenchError("Pick at least one model to test.")
    if external_ids and b["kind"] != "chat":
        raise BenchError("Other AI models can only be compared on chat benchmarks.")
    out = []
    for eid in external_ids:
        try:
            ext = EXT.get(eid)
        except EXT.ExternalError as e:
            raise BenchError(str(e))
        res = _run_chat_external(b["spec"], ext)
        out.append(_save(bid, None, f"{ext['name']} ({ext['model']})", 0, res, "external"))
    for mid in model_ids:
        row = db.model_row(mid)
        if not row or row["kind"] != b["kind"]:
            raise BenchError(f"Model #{mid} isn't a {b['kind']} model.")
        try:
            INF.get(mid)
        except INF.NotReady as e:
            raise BenchError(str(e))
        res = {"chat": _run_chat, "vision": _run_vision, "canvas": _run_canvas}[b["kind"]](b["spec"], mid)
        out.append(_save(bid, mid, row["name"], row["step"], res, "deer"))
    return out


def _chat_rows(spec, answer):
    """Ask every question with `answer(i, prompt)` and score the replies."""
    rows, t0, chars = [], time.time(), 0
    for i, it in enumerate(spec["items"]):
        s = time.time()
        err = None
        try:
            got = (answer(i, it["prompt"]) or "").strip()
        except Exception as e:  # one failed question doesn't sink the whole run
            got, err = "", str(e)
        chars += len(got)
        rows.append({"prompt": it["prompt"], "expected": it["expected"], "got": got, "error": err,
                     "f1": round(f1(got, it["expected"]), 3), "exact": _norm(got) == _norm(it["expected"]),
                     "code_expected": "```" in it["expected"], "code_ok": code_ok(got), "seconds": round(time.time() - s, 2)})
    return rows, max(1e-6, time.time() - t0), chars


def _chat_result(rows, secs, chars, speed=True):
    n = len(rows)
    coded = [r for r in rows if r["code_expected"]]
    metrics = {
        "Word match": round(100 * sum(r["f1"] for r in rows) / n, 1),
        "Exact answers": round(100 * sum(r["exact"] for r in rows) / n, 1),
        "Well-formed code": round(100 * sum(1 for r in coded if r["code_ok"]) / len(coded), 1) if coded else None,
        "Characters per second": round(chars / secs, 1) if speed else None,
        "Average reply length": round(sum(len(r["got"]) for r in rows) / n),
    }
    score = metrics["Word match"] if metrics["Well-formed code"] is None else \
        round(0.7 * metrics["Word match"] + 0.3 * metrics["Well-formed code"], 1)
    return {"score": score, "metrics": metrics, "rows": rows, "errors": sum(1 for r in rows if r["error"])}


def _run_chat(spec, mid):
    rows, secs, chars = _chat_rows(spec, lambda i, q: "".join(INF.chat_stream(mid, [], q, temperature=0.2)))
    return _chat_result(rows, secs, chars)


def _examples(items, i, k=2):
    """Style examples for an outside model: other questions from the same benchmark, never the one being asked."""
    others = [items[(i + j) % len(items)] for j in range(1, len(items))][:k]
    return [(o["prompt"], o["expected"]) for o in others]


def _run_chat_external(spec, ext):
    from . import external as EXT
    items = spec["items"]
    rows, secs, chars = _chat_rows(spec, lambda i, q: EXT.ask(ext, q, _examples(items, i)))
    if all(r["error"] for r in rows):
        raise BenchError(rows[0]["error"])
    res = _chat_result(rows, secs, chars)
    res["note"] = "Answered over the internet, so speed includes network time."
    return res


def export_sheet(bid):
    """The questions, for asking any AI by hand (for example in its own chat app)."""
    b = get(bid)
    if b["kind"] != "chat":
        raise BenchError("Answer sheets are for chat benchmarks.")
    return [it["prompt"] for it in b["spec"]["items"]]


def import_sheet(bid, name, answers):
    """Score answers you collected yourself. `answers` is a list in question order, or text in the
    User:/Deer: format (matched to the questions by their wording)."""
    b = get(bid)
    if b["kind"] != "chat":
        raise BenchError("Answer sheets are for chat benchmarks.")
    items = b["spec"]["items"]
    if isinstance(answers, str):
        by_q = {" ".join(u["q"].split()).lower(): u["a"] for u in chatfmt.parse(answers) if u.get("a")}
        answers = [by_q.get(" ".join(it["prompt"].split()).lower()) for it in items]
    if not isinstance(answers, list) or len(answers) != len(items):
        raise BenchError(f"Expected {len(items)} answers, one per question, in the same order.")
    if not any(a for a in answers):
        raise BenchError("None of the answers matched the questions. Keep each question exactly as exported.")
    rows, secs, chars = _chat_rows(b["spec"], lambda i, q: answers[i] or "")
    res = _chat_result(rows, secs, chars, speed=False)
    res["note"] = "Answers were pasted in, so there's no speed score."
    name = (name or "").strip()[:60] or "Pasted answers"
    return _save(bid, None, f"{name} (answer sheet)", 0, res, "sheet")


def _run_vision(spec, mid):
    model, row = INF.get(mid)
    X, y, classes = D.load_images(db.dataset_dir(spec["dataset_id"]), row["config"]["image_size"])
    known = row["meta"].get("classes", [])
    r = np.random.default_rng(0)
    pick = []
    for ci in range(len(classes)):
        idx = np.where(y == ci)[0]
        pick += list(r.permutation(idx)[:spec["max_per_class"]])
    pick = np.array(sorted(pick))
    from .engine import Tensor, xp
    t0 = time.time()
    probs = np.concatenate([model.predict(Tensor(xp.asarray(D.to_float(X[pick[i:i + 64]]))))
                            for i in range(0, len(pick), 64)])
    secs = max(1e-6, time.time() - t0)
    pred = [known[k] for k in probs.argmax(1)]
    truth = [classes[k] for k in y[pick]]
    per = {}
    for p, t in zip(pred, truth):
        d = per.setdefault(t, {"right": 0, "total": 0})
        d["total"] += 1
        d["right"] += int(p == t)
    acc = 100 * sum(p == t for p, t in zip(pred, truth)) / len(pick)
    mistakes = {}
    for p, t in zip(pred, truth):
        if p != t:
            mistakes[f"{t} → {p}"] = mistakes.get(f"{t} → {p}", 0) + 1
    metrics = {"Accuracy": round(acc, 1), "Average confidence": round(100 * float(probs.max(1).mean()), 1),
               "Pictures per second": round(len(pick) / secs, 1)}
    unknown = sorted(set(classes) - set(known))
    return {"score": round(acc, 1), "metrics": metrics, "unknown_classes": unknown,
            "per_class": {c: round(100 * d["right"] / d["total"], 1) for c, d in sorted(per.items())},
            "top_mistakes": sorted(mistakes.items(), key=lambda kv: -kv[1])[:8]}


def _run_canvas(spec, mid):
    model, row = INF.get(mid)
    judge_id = spec["judge_model_id"]
    try:
        judge, jrow = INF.get(judge_id)
    except INF.NotReady:
        raise BenchError("The judge model is gone or untrained. Create the benchmark again with another judge.")
    jcls = jrow["meta"].get("classes", [])
    per, t0, tested, right, variety = {}, time.time(), 0, 0, []
    from .engine import Tensor, xp
    for ci, cls in enumerate(row["meta"]["classes"]):
        imgs = model.sample([ci] * spec["per_class"], seed=100 + ci)  # (N,S,S,3) in [0,1]
        flat = imgs.reshape(len(imgs), -1)
        variety.append(float(np.mean(np.abs(flat - flat.mean(0)))))
        if cls not in jcls:
            per[cls] = None
            continue
        S = jrow["config"]["image_size"]
        batch = np.stack([np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR))
                          for im in imgs])
        pred = judge.predict(Tensor(xp.asarray(D.to_float(batch)))).argmax(1)
        ok = int(sum(jcls[p] == cls for p in pred))
        per[cls] = round(100 * ok / len(imgs), 1)
        tested += len(imgs)
        right += ok
    secs = max(1e-6, time.time() - t0)
    if not tested:
        raise BenchError(f"The judge '{jrow['name']}' doesn't know any of this model's classes.")
    rec = 100 * right / tested
    metrics = {"Recognized as asked": round(rec, 1), "Variety": round(100 * float(np.mean(variety)), 1),
               "Pictures per second": round(tested / secs, 1)}
    return {"score": round(rec, 1), "metrics": metrics, "per_class": per, "judge": jrow["name"]}


# ================================================================== listing
def overview():
    out = []
    for b in db.q("SELECT * FROM benchmarks ORDER BY id DESC"):
        b["spec"] = json.loads(b["spec"])
        b["size"] = len(b["spec"].get("items", [])) if b["kind"] == "chat" else None
        b["leaders"] = db.q("SELECT model_id, model_name, source, MAX(score) score, COUNT(*) runs FROM bench_runs "
                            "WHERE benchmark_id=? GROUP BY COALESCE(model_id, -1), model_name ORDER BY score DESC LIMIT 8",
                            (b["id"],))
        out.append(b)
    return out


def runs(bid):
    rs = db.q("SELECT * FROM bench_runs WHERE benchmark_id=? ORDER BY id DESC LIMIT 100", (bid,))
    for r in rs:
        r["results"] = json.loads(r["results"])
    return rs
