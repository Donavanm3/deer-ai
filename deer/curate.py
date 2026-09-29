"""
Build better datasets from the ones you already have.

Pick any datasets (and the lessons users taught them), filter out duplicates, junk and unwanted
content, rename or merge image classes, balance them, preview the result, and save it as a new
dataset to train a bigger, smarter model on.
"""
import base64
import io
import json
import os
import random
import re
import shutil

import numpy as np
from PIL import Image

from . import data as D
from . import chatfmt
from .server import db

JUNK_RATIO = 0.35  # a line that's more than this share of symbols/digits is treated as junk
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class CurateError(ValueError):
    pass


# ====================================================================== text
def _paragraphs(block):
    """Split non-chat text into paragraphs at blank lines, keeping ``` code blocks whole."""
    out, cur, in_code = [], [], False
    for line in block.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if cur:
                out.append("\n".join(cur))
                cur = []
            continue
        cur.append(line.rstrip())
    if cur:
        out.append("\n".join(cur))
    return out


def _units(text, source):
    """Training units: a User/Deer exchange stays together (replies may span many lines, like code);
    other text is split into paragraphs."""
    out = []
    for u in chatfmt.parse(text):
        if "plain" in u:
            out += [{"text": p, "pair": False, "source": source} for p in _paragraphs(u["plain"])]
        elif u.get("a"):
            out.append({"text": chatfmt.unit_text(u), "pair": True, "answer": u["a"], "source": source})
        elif u.get("q"):
            out.append({"text": chatfmt.unit_text(u), "pair": False, "source": source})
    return out


def _words(csv):
    return [w.strip().lower() for w in (csv or "").split(",") if w.strip()]


def _has_word(text_lower, words):
    return any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text_lower) for w in words)


def text_units(dataset_ids, include_base=True, include_learned=True):
    units = []
    for did in dataset_ids:
        ds = db.q1("SELECT id, name, kind FROM datasets WHERE id=?", (did,))
        if not ds or ds["kind"] != "text":
            raise CurateError(f"Dataset #{did} isn't a text dataset.")
        folder = db.dataset_dir(did)
        if include_base and os.path.exists(os.path.join(folder, "corpus.txt")):
            with open(os.path.join(folder, "corpus.txt"), encoding="utf-8", errors="replace") as f:
                units += _units(f.read(), f"{ds['name']}")
        if include_learned and os.path.exists(os.path.join(folder, "learned.txt")):
            with open(os.path.join(folder, "learned.txt"), encoding="utf-8", errors="replace") as f:
                units += _units(f.read(), f"{ds['name']} (user lessons)")
    return units


DEFAULT_PROMPTS = ["Hello!", "What can you do?", "Who are you?", "Tell me something interesting.",
                   "How are you?", "What should I know?", "Can you help me?", "Thanks!"]
MAX_GENERATED = 2000


def generated_units(units, g, limit=None):
    """Replies written by a trained chat model, asked the questions found in the chosen data."""
    from . import inference as INF
    mid = int(g.get("model_id") or 0)
    row = db.model_row(mid) if mid else None
    if not row or row["kind"] != "chat":
        raise CurateError("Pick a trained chat model to write replies.")
    try:
        INF.get(mid)
    except INF.NotReady as e:
        raise CurateError(str(e))
    prompts = []
    for u in units:
        if u["pair"]:
            q = u["text"].split("\nDeer:", 1)[0][5:].strip()
            if q and q not in prompts:
                prompts.append(q)
    prompts = prompts or DEFAULT_PROMPTS
    n = min(int(g.get("count") or 100), MAX_GENERATED)
    if limit:
        n = min(n, limit)
    rnd = random.Random(1)
    temp = float(g.get("temperature") or 0.7)
    out = []
    for i in range(n):
        q = prompts[i] if i < len(prompts) else rnd.choice(prompts)
        a = INF.chat_reply(mid, [], q, temp)
        out.append({"text": f"User: {q}\nDeer: {a}", "pair": True, "answer": a, "source": f"written by {row['name']}"})
    return out


OK_USERS = "u.train_ok=1 AND u.role!='admin'"  # users who haven't opted out; admin chats are tests


def user_text_units(opts):
    """Training text straight from your users: their feedback, what they wrote, or whole conversations."""
    out = []
    if opts.get("feedback"):  # explicit feedback: liked replies and corrections (anything not rejected)
        for r in db.q("SELECT prompt, answer FROM feedback WHERE kind='chat' AND status IN ('pending','applied') "
                      "AND prompt IS NOT NULL AND answer IS NOT NULL ORDER BY id"):
            out.append({"text": f"User: {r['prompt'].strip()}\nDeer: {r['answer'].strip()}", "pair": True,
                        "answer": r["answer"], "source": "user feedback"})
    if opts.get("messages"):
        for r in db.q(f"SELECT m.content FROM messages m JOIN conversations c ON c.id=m.conversation_id "
                      f"JOIN users u ON u.id=c.user_id WHERE m.role='user' AND m.kind='text' AND {OK_USERS} "
                      "ORDER BY m.id DESC LIMIT 50000"):
            out.append({"text": f"User: {r['content'].strip()}", "pair": False, "source": "what users wrote"})
    if opts.get("conversations"):
        disliked = {r["message_id"] for r in db.q("SELECT message_id FROM feedback WHERE rating<0")}
        prev = None
        for r in db.q(f"SELECT m.id, m.conversation_id, m.role, m.content FROM messages m JOIN conversations c "
                      f"ON c.id=m.conversation_id JOIN users u ON u.id=c.user_id WHERE m.kind='text' AND {OK_USERS} "
                      "ORDER BY m.conversation_id, m.id LIMIT 100000"):
            if r["role"] == "user":
                prev = r
            elif prev and prev["conversation_id"] == r["conversation_id"] and r["id"] not in disliked:
                out.append({"text": f"User: {prev['content'].strip()}\nDeer: {r['content'].strip()}", "pair": True,
                            "answer": r["content"], "source": "user conversations"})
                prev = None
    return out


def _one_line(t):
    return " ".join(str(t).split())[:2000]


def _all_text_units(dataset_ids, f, preview=False):
    units = text_units(dataset_ids, f.get("include_base", True), f.get("include_learned", True)) if dataset_ids else []
    if any((f.get("users") or {}).values()):
        units = units + user_text_units(f["users"])
    g = f.get("generate") or {}
    if g.get("model_id"):
        units = units + generated_units(units, g, limit=12 if preview else None)
    return units


def filter_text(units, f):
    """Returns kept units and a {reason: count} of what was removed (plus a few examples)."""
    dedupe = f.get("dedupe", True)
    chat_only = f.get("chat_only", False)
    min_c, max_c = int(f.get("min_chars") or 0), int(f.get("max_chars") or 0)
    need, block = _words(f.get("must_contain")), _words(f.get("block_words"))
    junk = f.get("remove_junk", True)
    seen, kept = set(), []
    removed, examples = {}, []

    def drop(u, why):
        removed[why] = removed.get(why, 0) + 1
        if len(examples) < 12:
            examples.append({"text": u["text"][:300], "reason": why})

    for u in units:
        t = _CTRL.sub("", u["text"]).strip()
        body = t.replace("User:", "").replace("Deer:", "")
        low = t.lower()
        if chat_only and not u["pair"]:
            drop(u, "not a User/Deer pair")
            continue
        if junk:
            code = "```" in t or sum(t.count(ch) for ch in "{}();=<>[]") >= 4  # code is allowed to be symbol-heavy
            letters = sum(ch.isalpha() or ch.isspace() for ch in body)
            if not body.strip() or (not code and (1 - letters / max(1, len(body))) > JUNK_RATIO):
                drop(u, "junk or symbols")
                continue
            if u["pair"] and len((u.get("answer") or "").strip()) < 2:
                drop(u, "empty answer")
                continue
        if min_c and len(body.strip()) < min_c:
            drop(u, "too short")
            continue
        if max_c and len(body.strip()) > max_c:
            drop(u, "too long")
            continue
        if block and _has_word(low, block):
            drop(u, "blocked word")
            continue
        if need and not _has_word(low, need):
            drop(u, "missing required word")
            continue
        if dedupe:
            key = _WS.sub(" ", low).strip()
            if key in seen:
                drop(u, "duplicate")
                continue
            seen.add(key)
        kept.append(dict(u, text=t))
    if f.get("shuffle"):
        random.Random(0).shuffle(kept)
    cap = int(f.get("max_units") or 0)
    if cap and len(kept) > cap:
        removed["over the limit"] = len(kept) - cap
        kept = kept[:cap]
    return kept, removed, examples


def preview_text(dataset_ids, f):
    units = _all_text_units(dataset_ids, f, preview=True)
    kept, removed, examples = filter_text(units, f)
    size = sum(len(u["text"].encode("utf-8")) + 1 for u in kept)
    by_source = {}
    for u in kept:
        by_source[u["source"]] = by_source.get(u["source"], 0) + 1
    return {"kind": "text", "before": len(units), "after": len(kept), "bytes": size,
            "pairs": sum(u["pair"] for u in kept), "by_source": by_source, "removed": removed,
            "removed_examples": examples, "sample": [u["text"] for u in kept[:15]],
            "suggested_preset": suggest_preset("text", size),
            "generated_note": (f"The preview writes 12 sample replies; building writes {min(int((f.get('generate') or {}).get('count') or 100), MAX_GENERATED)}."
                               if (f.get("generate") or {}).get("model_id") else None)}


def build_text(name, dataset_ids, f):
    units = _all_text_units(dataset_ids, f)
    kept, removed, _ = filter_text(units, f)
    # other text first, conversations after, so a reply never swallows a following paragraph
    kept_out = [u for u in kept if not u["pair"]] + [u for u in kept if u["pair"]]
    text = "\n\n".join(u["text"] for u in kept_out) + "\n"
    if len(text.encode("utf-8")) < 256:
        raise CurateError("Too little text survived the filters (need at least 256 bytes). Loosen them.")
    did = db.ex("INSERT INTO datasets(name,kind,created_at) VALUES(?,?,?)", (name, "text", db.now()))
    try:
        stats = D.save_text_dataset(db.dataset_dir(did), [("combined.txt", text.encode("utf-8"))])
    except D.DatasetError as e:
        db.ex("DELETE FROM datasets WHERE id=?", (did,))
        raise CurateError(str(e))
    stats["built_from"] = {"datasets": list(dataset_ids), "filters": f, "removed": removed,
                           "units": len(kept)}
    db.ex("UPDATE datasets SET stats=? WHERE id=?", (json.dumps(stats), did))
    return did, stats


# ==================================================================== images
def _ahash(path):
    """Picture fingerprint: a 16x16 brightness pattern plus the average color. The same picture
    resized or recompressed gets the same fingerprint; different pictures almost never do."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        g = np.asarray(rgb.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32)
        col = np.asarray(rgb.resize((1, 1), Image.BOX), dtype=np.int32).reshape(3) // 24
    bits = "".join("1" if b else "0" for b in (g > g.mean()).flatten())
    return f"{int(bits, 2):064x}-{col[0]}{col[1]}{col[2]}"


def _hash_cache(folder):
    p = os.path.join(folder, "fingerprints.json")
    try:
        with open(p) as fh:
            return p, json.load(fh)
    except (OSError, ValueError):
        return p, {}


def image_items(dataset_ids):
    items = []
    for did in dataset_ids:
        ds = db.q1("SELECT id, name, kind FROM datasets WHERE id=?", (did,))
        if not ds or ds["kind"] != "images":
            raise CurateError(f"Dataset #{did} isn't an image dataset.")
        folder = db.dataset_dir(did)
        root = os.path.join(folder, "images")
        cpath, cache = _hash_cache(folder)
        changed = False
        for cls in sorted(os.listdir(root)):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                path = os.path.join(cdir, fn)
                key = f"{cls}/{fn}"
                if key not in cache:
                    try:
                        with Image.open(path) as im:
                            w, h = im.size
                        cache[key] = [_ahash(path), w, h]
                    except OSError:
                        continue
                    changed = True
                hv, w, h = cache[key]
                items.append({"path": path, "cls": cls, "hash": hv, "w": w, "h": h,
                              "user": fn.startswith("u"), "dataset": ds["name"]})
        if changed:
            with open(cpath, "w") as fh:
                json.dump(cache, fh)
    return items


def label_with_model(items, lab):
    """Give pictures in the chosen classes (by default the "unlabeled" folder) the label a trained
    analysis model is confident about. Unsure pictures are dropped."""
    from . import inference as INF
    mid = int(lab.get("model_id") or 0)
    row = db.model_row(mid) if mid else None
    if not row or row["kind"] != "vision":
        raise CurateError("Pick a trained image analysis model to label pictures.")
    try:
        INF.get(mid)
    except INF.NotReady as e:
        raise CurateError(str(e))
    targets = set(lab.get("classes") or ["unlabeled"])
    min_conf = float(lab.get("min_confidence") or 0.8)
    for it in items:
        if it["cls"] not in targets:
            continue
        with Image.open(it["path"]) as im:
            best = INF.analyze(mid, im, top=1)[0]
        if best["confidence"] >= min_conf:
            it["cls"], it["labeled_by"] = best["label"], row["name"]
        else:
            it["cls"], it["unsure"] = "", True
    return items


def generated_items(gen, workdir, per_class_cap=None):
    from . import inference as INF
    mid = int(gen.get("model_id") or 0)
    row = db.model_row(mid) if mid else None
    if not row or row["kind"] != "canvas":
        raise CurateError("Pick a trained image creation model to draw pictures.")
    try:
        model, _ = INF.get(mid)
    except INF.NotReady as e:
        raise CurateError(str(e))
    per = min(int(gen.get("per_class") or 20), 500)
    if per_class_cap:
        per = min(per, per_class_cap)
    os.makedirs(workdir, exist_ok=True)
    items = []
    for ci, cls in enumerate(row["meta"]["classes"]):
        imgs = model.sample([ci] * per, temperature=float(gen.get("temperature") or 1.0), seed=ci)
        for j, img in enumerate(imgs):
            path = os.path.join(workdir, f"{ci}_{j}.png")
            Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).save(path)
            items.append({"path": path, "cls": cls, "hash": _ahash(path), "w": img.shape[1], "h": img.shape[0],
                          "user": False, "generated": True, "dataset": f"drawn by {row['name']}"})
    return items


def user_image_items(workdir, limit=20000):
    """Pictures users sent in chat, labeled by their feedback where they gave it
    (Yes = the model's answer, No = the label they picked). The rest go in "unlabeled"."""
    items = []
    rows = db.q(f"SELECT m.id, m.content, m.conversation_id FROM messages m JOIN conversations c ON c.id=m.conversation_id "
                f"JOIN users u ON u.id=c.user_id WHERE m.role='user' AND m.kind='image' AND {OK_USERS} "
                "ORDER BY m.id DESC LIMIT ?", (limit,))
    if rows:
        os.makedirs(workdir, exist_ok=True)
    for r in rows:
        nxt = db.q1("SELECT id, meta FROM messages WHERE conversation_id=? AND id>? AND kind='analysis' ORDER BY id LIMIT 1",
                    (r["conversation_id"], r["id"]))
        label = "unlabeled"
        if nxt:
            fb = db.q1("SELECT rating, label FROM feedback WHERE message_id=? AND status!='rejected'", (nxt["id"],))
            preds = json.loads(nxt["meta"]).get("predictions") or []
            if fb and fb["rating"] > 0 and preds:
                label = preds[0]["label"]
            elif fb and fb["label"]:
                label = fb["label"]
        try:
            head, b64 = r["content"].split(",", 1)
            path = os.path.join(workdir, f"user_{r['id']}.png")
            with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
                im.convert("RGB").save(path)
                w, h = im.size
        except Exception:
            continue
        items.append({"path": path, "cls": label, "hash": _ahash(path), "w": w, "h": h, "user": True,
                      "dataset": "sent by users"})
    return items


def _all_image_items(dataset_ids, f, workdir, preview=False):
    items = image_items(dataset_ids) if dataset_ids else []
    if (f.get("users") or {}).get("pictures"):
        items += user_image_items(workdir)
    lab = f.get("label") or {}
    if lab.get("model_id"):
        items = label_with_model(items, lab)
    gen = f.get("generate") or {}
    if gen.get("model_id"):
        items += generated_items(gen, workdir, per_class_cap=4 if preview else None)
    return items


def filter_images(items, f):
    rename = {k: (v or "").strip() for k, v in (f.get("classes") or {}).items()}
    dedupe = f.get("dedupe", True)
    user_ok = f.get("include_user", True)
    min_side = int(f.get("min_size") or 0)
    removed, seen, kept = {}, set(), []

    def drop(why):
        removed[why] = removed.get(why, 0) + 1

    for it in items:
        if it.get("unsure"):
            drop("model unsure")
            continue
        target = rename.get(it["cls"], it["cls"]) if it["cls"] in rename else it["cls"]
        if not target:
            drop("class left out")
            continue
        target = D._clean_class(target)
        if not target:
            drop("class left out")
            continue
        if not user_ok and it["user"]:
            drop("taught by users")
            continue
        if min_side and min(it["w"], it["h"]) < min_side:
            drop("too small")
            continue
        if dedupe:
            key = (target, it["hash"])
            if key in seen:
                drop("duplicate")
                continue
            seen.add(key)
        kept.append(dict(it, target=target))

    by_cls = {}
    for it in kept:
        by_cls.setdefault(it["target"], []).append(it)
    min_n, max_n = int(f.get("min_per_class") or 0), int(f.get("max_per_class") or 0)
    rnd = random.Random(0)
    final = {}
    for c, lst in sorted(by_cls.items()):
        if min_n and len(lst) < min_n:
            removed["class too small"] = removed.get("class too small", 0) + len(lst)
            continue
        if max_n and len(lst) > max_n:
            removed["balanced (over max per class)"] = removed.get("balanced (over max per class)", 0) + len(lst) - max_n
            lst = rnd.sample(lst, max_n)
        final[c] = lst
    return final, removed


def _thumb(path, size=72):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _workdir():
    import uuid
    return os.path.join(db.DATA_DIR, "tmp", uuid.uuid4().hex)


def preview_images(dataset_ids, f):
    wd = _workdir()
    try:
        return _preview_images(dataset_ids, f, wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _preview_images(dataset_ids, f, wd):
    items = _all_image_items(dataset_ids, f, wd, preview=True)
    final, removed = filter_images(items, f)
    source_classes = {}
    labeled = sum(1 for it in items if it.get("labeled_by"))
    for it in items:
        if it["cls"]:
            source_classes[it["cls"]] = source_classes.get(it["cls"], 0) + 1
    n = sum(len(v) for v in final.values())
    return {"kind": "images", "before": len(items), "after": n, "removed": removed,
            "source_classes": source_classes,
            "classes": {c: {"count": len(v), "samples": [_thumb(x["path"]) for x in v[:4]]} for c, v in final.items()},
            "suggested_preset": suggest_preset("images", n, len(final)), "labeled_by_model": labeled,
            "generated_note": (f"The preview draws up to 4 pictures per class; building draws {min(int((f.get('generate') or {}).get('per_class') or 20), 500)} per class."
                               if (f.get("generate") or {}).get("model_id") else None)}


def build_images(name, dataset_ids, f):
    wd = _workdir()
    try:
        return _build_images(name, dataset_ids, f, wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _build_images(name, dataset_ids, f, wd):
    final, removed = filter_images(_all_image_items(dataset_ids, f, wd), f)
    if not final:
        raise CurateError("No images survived the filters. Loosen them.")
    did = db.ex("INSERT INTO datasets(name,kind,created_at) VALUES(?,?,?)", (name, "images", db.now()))
    folder = db.dataset_dir(did)
    try:
        for c, lst in final.items():
            d = os.path.join(folder, "images", c)
            os.makedirs(d, exist_ok=True)
            for i, it in enumerate(lst):
                shutil.copyfile(it["path"], os.path.join(d, f"{i:06d}.png"))
    except OSError as e:
        shutil.rmtree(folder, ignore_errors=True)
        db.ex("DELETE FROM datasets WHERE id=?", (did,))
        raise CurateError(f"Couldn't copy images: {e}")
    stats = D.image_stats_for(folder)
    stats["built_from"] = {"datasets": list(dataset_ids), "filters": f, "removed": removed}
    db.ex("UPDATE datasets SET stats=? WHERE id=?", (json.dumps(stats), did))
    return did, stats


# ==================================================================== advice
def suggest_preset(kind, size, n_classes=0):
    """Bigger models only get smarter with enough data; otherwise they just memorize."""
    if kind == "text":
        if size >= 5_000_000:
            return {"preset": "medium", "why": "Plenty of text: a medium chat model can use it."}
        if size >= 500_000:
            return {"preset": "small", "why": "A small chat model fits this much text."}
        return {"preset": "tiny", "why": "Under 500 KB of text; a bigger model would just memorize it. Add more text to go bigger."}
    per = size / max(1, n_classes)
    if per >= 800:
        return {"preset": "medium", "why": "Lots of images per class: a medium model at 64 px can use them."}
    if per >= 150:
        return {"preset": "small", "why": "A small model suits this many images per class."}
    return {"preset": "tiny", "why": "Under 150 images per class; add more before going bigger."}


def _has_source(dataset_ids, f):
    return dataset_ids or (f.get("generate") or {}).get("model_id") or any((f.get("users") or {}).values())


def preview(kind, dataset_ids, f):
    if not _has_source(dataset_ids, f):
        raise CurateError("Pick at least one dataset, model or user source.")
    return preview_text(dataset_ids, f) if kind == "text" else preview_images(dataset_ids, f)


def build(kind, name, dataset_ids, f):
    if not _has_source(dataset_ids, f):
        raise CurateError("Pick at least one dataset, model or user source.")
    name = (name or "").strip()[:80]
    if not name:
        raise CurateError("Give the new dataset a name.")
    return build_text(name, dataset_ids, f) if kind == "text" else build_images(name, dataset_ids, f)
