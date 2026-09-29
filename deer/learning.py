"""
Deer AI learns from its users over time.

  1. Users rate replies (thumbs up / down), write a better answer, or fix an image label.
  2. Their feedback lands in the admin's Learning inbox (or is auto-approved if the admin allows it).
  3. Approved lessons are added to the model's dataset: chat lessons to learned.txt, labeled
     images into the image folders (a new label becomes a new class the models grow into).
  4. Models with auto-learn on fine-tune themselves once enough new lessons pile up. Each run
     backs up the old checkpoint and is rolled back automatically if validation gets worse.
"""
import base64
import io
import json
import os
import threading
import uuid

from PIL import Image

from . import data as D
from .server import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  message_id INTEGER, model_id INTEGER, dataset_id INTEGER, kind TEXT NOT NULL,
  rating INTEGER NOT NULL, prompt TEXT, answer TEXT, label TEXT, image_path TEXT,
  status TEXT NOT NULL, created_at REAL NOT NULL, reviewed_at REAL);
CREATE INDEX IF NOT EXISTS ix_fb_status ON feedback(status, id);
CREATE INDEX IF NOT EXISTS ix_fb_ds ON feedback(dataset_id, status, reviewed_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_fb_msg ON feedback(message_id, user_id);
"""
FT_LR_SCALE = 0.3  # auto-learn fine-tunes gently so the model doesn't forget what it knew


class FeedbackError(ValueError):
    pass


def init():
    db.conn().executescript(SCHEMA)
    db.ex("INSERT OR IGNORE INTO settings(key,value) VALUES('learn_auto_approve','0')")
    db.ex("INSERT OR IGNORE INTO settings(key,value) VALUES('learn_from_messages','1')")
    db.ex("INSERT OR IGNORE INTO settings(key,value) VALUES('learn_from_uploads','1')")
    repair()


AUTO_LABEL_CONFIDENCE = 0.9


def _may_learn_from(user):
    """Regular users who haven't opted out (and admins testing as a user)."""
    return bool(user.get("train_ok", 1)) and (user["role"] != "admin" or user.get("test"))


def collect_message(user, model_id, text):
    """What users write becomes extra training text for that chat model's dataset."""
    if db.setting("learn_from_messages") != "1" or not _may_learn_from(user):
        return
    m = db.q1("SELECT dataset_id FROM models WHERE id=?", (model_id,))
    if not m:
        return
    folder = db.dataset_dir(m["dataset_id"])
    D.add_user_text(folder, text)
    ds = db.q1("SELECT stats FROM datasets WHERE id=?", (m["dataset_id"],))
    if ds:
        st = json.loads(ds["stats"])
        st["user_messages"] = st.get("user_messages", 0) + 1
        db.ex("UPDATE datasets SET stats=? WHERE id=?", (json.dumps(st), m["dataset_id"]))


def collect_upload(user, message_id, model_id, predictions, pil):
    """Pictures users send that the analysis model is very sure about go to the Learning inbox
    (or straight in, with auto-approve), labeled with that answer."""
    if db.setting("learn_from_uploads") != "1" or not _may_learn_from(user) or not predictions or not model_id:
        return
    top = predictions[0]
    if top["confidence"] < AUTO_LABEL_CONFIDENCE:
        return
    msg = db.q1("SELECT created_at FROM messages WHERE id=?", (message_id,))
    if msg:
        drop_stale(message_id, msg["created_at"])
    path = os.path.join(learn_dir(), f"{uuid.uuid4().hex}.png")
    pil.convert("RGB").save(path)
    fid = db.ex("INSERT OR IGNORE INTO feedback(user_id,message_id,model_id,kind,rating,answer,label,image_path,status,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)", (user["id"], message_id, model_id, "vision", 0,
                                                 f"auto:{top['confidence']:.3f}", top["label"], path, "pending", db.now()))
    if fid and db.setting("learn_auto_approve") == "1":
        review([fid], "approve")


def drop_stale(message_id, message_created_at):
    """Message ids can be reused after a conversation is deleted. Feedback left over from the old
    message (created before this message existed) must not be mixed up with the new one."""
    for f in db.q("SELECT * FROM feedback WHERE message_id=? AND created_at < ?", (message_id, message_created_at - 0.001)):
        if f["status"] == "applied":
            db.ex("UPDATE feedback SET message_id=NULL WHERE id=?", (f["id"],))  # keep it as history
        else:
            _remove_image(f)
            db.ex("DELETE FROM feedback WHERE id=?", (f["id"],))


def forget_messages(message_ids):
    """Called when a conversation is deleted: unreviewed feedback on it goes too; approved feedback
    stays as training history but no longer points at a message."""
    for mid in message_ids:
        for f in db.q("SELECT * FROM feedback WHERE message_id=?", (mid,)):
            if f["status"] == "applied":
                db.ex("UPDATE feedback SET message_id=NULL WHERE id=?", (f["id"],))
            else:
                _remove_image(f)
                db.ex("DELETE FROM feedback WHERE id=?", (f["id"],))


def _remove_image(f):
    if f.get("image_path") and os.path.exists(f["image_path"]):
        try:
            os.remove(f["image_path"])
        except OSError:
            pass


def _problem(f):
    """Why a feedback item can't be used for training, or None if it's fine."""
    if f["kind"] == "chat" and not (f["prompt"] and f["answer"]):
        return "it has no question and answer to learn from"
    if f["kind"] == "vision" and not (f["label"] and f["image_path"] and os.path.exists(f["image_path"])):
        return "its picture or label is missing"
    if f["kind"] not in ("chat", "vision"):
        return "ratings of drawn pictures are only kept as stats"
    return None


def repair():
    """Fixes items damaged by the old reused-message-id bug, so the inbox never shows broken ones."""
    db.ex("UPDATE feedback SET kind='vision' WHERE kind='chat' AND image_path IS NOT NULL")
    for f in db.q("SELECT * FROM feedback WHERE status='pending'"):
        if _problem(f):
            db.ex("UPDATE feedback SET status='info' WHERE id=?", (f["id"],))


def learn_dir():
    d = os.path.join(db.DATA_DIR, "learning")
    os.makedirs(d, exist_ok=True)
    return d


def _decode_data_url(url):
    try:
        head, b64 = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        raise FeedbackError("The original image for this message is missing.")


# ================================================================== submit
def submit(user, message_id, rating, correction=None, label=None):
    if rating not in (-1, 1):
        raise FeedbackError("Rating must be 1 or -1.")
    m = db.q1("SELECT m.*, c.user_id owner FROM messages m JOIN conversations c ON c.id=m.conversation_id "
              "WHERE m.id=? AND c.user_id=?", (message_id, user["id"]))
    if not m or m["role"] != "assistant":
        raise FeedbackError("You can only rate Deer AI's replies in your own chats.")
    meta = json.loads(m["meta"])
    drop_stale(message_id, m["created_at"])
    old = db.q1("SELECT * FROM feedback WHERE message_id=? AND user_id=?", (message_id, user["id"]))
    if old and old["status"] == "applied":
        raise FeedbackError("Thanks, this one is already part of Deer AI's training.")
    model_id = meta.get("model_id")
    correction = (correction or "").strip()[:2000] or None
    label = (label or "").strip()[:60] or None
    prompt = answer = img_path = None
    status = "info"  # a rating with nothing to learn from: kept for satisfaction stats only

    if m["kind"] == "text":
        kind = "chat"
        prev = db.q1("SELECT content FROM messages WHERE conversation_id=? AND id<? AND role='user' AND kind='text' "
                     "ORDER BY id DESC LIMIT 1", (m["conversation_id"], m["id"]))
        prompt = prev["content"] if prev else None
        answer = m["content"] if rating > 0 else correction
        if prompt and answer:
            status = "pending"
    elif m["kind"] == "analysis":
        kind = "vision"
        preds = meta.get("predictions") or []
        label = label if rating < 0 else (preds[0]["label"] if preds else None)
        if label and model_id:
            prev = db.q1("SELECT content FROM messages WHERE conversation_id=? AND id<? AND role='user' AND kind='image' "
                         "ORDER BY id DESC LIMIT 1", (m["conversation_id"], m["id"]))
            if not prev:
                raise FeedbackError("The original image for this message is missing.")
            im = _decode_data_url(prev["content"]).convert("RGB")
            img_path = os.path.join(learn_dir(), f"{uuid.uuid4().hex}.png")
            im.save(img_path)
            status = "pending"
    elif m["kind"] == "images":
        kind = "canvas"  # never train the generator on its own output; ratings are stats only
    else:
        raise FeedbackError("This message can't be rated.")

    if old:
        if old["image_path"] and old["image_path"] != img_path and os.path.exists(old["image_path"]):
            os.remove(old["image_path"])
        db.ex("UPDATE feedback SET kind=?, rating=?, prompt=?, answer=?, label=?, image_path=?, status=?, model_id=?, "
              "created_at=?, reviewed_at=NULL WHERE id=?",
              (kind, rating, prompt, answer, label, img_path, status, model_id, db.now(), old["id"]))
        fid = old["id"]
    else:
        fid = db.ex("INSERT INTO feedback(user_id,message_id,model_id,kind,rating,prompt,answer,label,image_path,status,"
                    "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (user["id"], message_id, model_id, kind, rating, prompt, answer, label, img_path, status, db.now()))
    if status == "pending" and db.setting("learn_auto_approve") == "1":
        review([fid], "approve")
        status = db.q1("SELECT status FROM feedback WHERE id=?", (fid,))["status"]
    meta["feedback"] = {"rating": rating, "status": status, "label": label, "correction": correction}
    db.ex("UPDATE messages SET meta=? WHERE id=?", (json.dumps(meta), message_id))
    return meta["feedback"]


# ================================================================== review
def review(ids, action):
    """Approve or reject items. Returns how many were done; items that can't be used are
    set aside (with the reason) instead of failing the whole batch."""
    if action not in ("approve", "reject"):
        raise FeedbackError("Action must be approve or reject.")
    done, skipped = 0, []
    for fid in ids:
        f = db.q1("SELECT * FROM feedback WHERE id=? AND status='pending'", (int(fid),))
        if not f:
            continue
        if action == "reject":
            _remove_image(f)
            db.ex("UPDATE feedback SET status='rejected', image_path=NULL, reviewed_at=? WHERE id=?", (db.now(), f["id"]))
            done += 1
            continue
        why = _problem(f)
        if not why:
            try:
                _apply(f)
                done += 1
                continue
            except (OSError, ValueError, D.DatasetError) as e:
                why = f"it couldn't be saved ({e})"
        db.ex("UPDATE feedback SET status='info', reviewed_at=? WHERE id=?", (db.now(), f["id"]))
        skipped.append({"id": f["id"], "reason": why})
    review.last_skipped = skipped
    return done


review.last_skipped = []


def _apply(f):
    model = db.q1("SELECT dataset_id FROM models WHERE id=?", (f["model_id"],))
    if not model or not model["dataset_id"]:
        db.ex("UPDATE feedback SET status='rejected', reviewed_at=? WHERE id=?", (db.now(), f["id"]))
        return
    did = model["dataset_id"]
    folder = db.dataset_dir(did)
    if f["kind"] == "chat":
        D.add_lesson(folder, f["prompt"], f["answer"])
        ds = db.q1("SELECT stats FROM datasets WHERE id=?", (did,))
        stats = json.loads(ds["stats"]) if ds else {}
        stats["lessons"] = stats.get("lessons", 0) + 1
    else:
        with Image.open(f["image_path"]) as im:
            D.add_image(folder, f["label"], im)
        os.remove(f["image_path"])
        stats = D.image_stats_for(folder)
    db.ex("UPDATE datasets SET stats=? WHERE id=?", (json.dumps(stats), did))
    db.ex("UPDATE feedback SET status='applied', dataset_id=?, image_path=NULL, reviewed_at=? WHERE id=?",
          (did, db.now(), f["id"]))


def inbox(status="pending", limit=100):
    rows = db.q("SELECT f.*, u.username, m.name model_name FROM feedback f LEFT JOIN users u ON u.id=f.user_id "
                "LEFT JOIN models m ON m.id=f.model_id WHERE f.status=? ORDER BY f.id DESC LIMIT ?", (status, limit))
    for r in rows:
        p = r.pop("image_path", None)
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                r["image"] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    return rows


def stats():
    counts = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) n FROM feedback GROUP BY status")}
    by_model = db.q("SELECT f.model_id, m.name, m.kind, SUM(f.rating>0) up, SUM(f.rating<0) down FROM feedback f "
                    "JOIN models m ON m.id=f.model_id GROUP BY f.model_id ORDER BY m.name")
    return {"counts": counts, "models": by_model, "auto_approve": db.setting("learn_auto_approve") == "1",
            "from_messages": db.setting("learn_from_messages") == "1",
            "from_uploads": db.setting("learn_from_uploads") == "1",
            "opted_out": db.q1("SELECT COUNT(*) n FROM users WHERE train_ok=0")["n"]}


def new_lessons(model):
    return db.q1("SELECT COUNT(*) n FROM feedback WHERE dataset_id=? AND status='applied' AND reviewed_at>?",
                 (model["dataset_id"], model["last_learn_at"] or 0))["n"]


# =============================================================== scheduler
def tick(jobs, max_jobs, start_job, force_model=None):
    """Start auto-learn runs for models that have collected enough new lessons."""
    started = []
    rows = db.q("SELECT * FROM models WHERE id=?", (force_model,)) if force_model else \
        db.q("SELECT * FROM models WHERE auto_learn=1 ORDER BY last_learn_at")
    for m in rows:
        if m["id"] in jobs or len(jobs) >= max_jobs:
            continue
        if not os.path.exists(os.path.join(db.model_dir(m["id"]), "weights.npz")):
            continue
        if not force_model and new_lessons(m) < max(1, m["learn_min"]):
            continue
        db.ex("UPDATE models SET last_learn_at=?, status='queued', error=NULL WHERE id=?", (db.now(), m["id"]))
        jobs[m["id"]] = start_job(m["id"], m["learn_steps"], FT_LR_SCALE, True)
        started.append(m["id"])
    return started


class Scheduler:
    def __init__(self, interval, fn):
        self.interval, self.fn, self.stop = interval, fn, threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.fn()
            except Exception as e:  # never let the scheduler die
                print(f"auto-learn check failed: {e}")
