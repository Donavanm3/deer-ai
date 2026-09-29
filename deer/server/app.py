"""Deer AI web backend: auth, chat, image analysis, image generation, and the admin training API."""
import base64
import hashlib
import hmac
import io
import json
import multiprocessing as mp
import os
import secrets
import shutil
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import db
from .auth import current_user, admin_user, acts_as_admin
from . import jobs as JOBS
from .billing_api import router as billing_router
from .devapi import router as dev_router
from .. import apikeys
from .. import updater
from .. import billing
from .. import learning
from .. import curate
from .. import bench
from .. import external
from .. import data as D
from .. import inference as INF
from .. import training
from ..models import KIND_DATASET, PRESETS, build_config, build_model

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "static")
SESSION_DAYS = 30
MAX_UPLOAD = 25 * 1024 * 1024

_jobs = JOBS.jobs  # model_id -> running Process (the scheduler in jobs.py owns these)
_fails = defaultdict(list)
_jobs_lock = JOBS.lock
LEARN_INTERVAL = int(os.environ.get("DEER_LEARN_INTERVAL", "300"))  # seconds between auto-learn checks


def _enqueue_learn(mid, steps, lr_scale, auto):
    JOBS.enqueue(mid, steps, lr_scale, auto)
    return True


def _learn_check():
    """Auto-learn: queue a gentle fine-tune for models with enough new lessons."""
    with _jobs_lock:
        JOBS.reap()
        busy = {m: True for m in list(JOBS.jobs) + [q["mid"] for q in JOBS.queue]}
        return learning.tick(busy, 10 ** 6, _enqueue_learn)


@asynccontextmanager
async def lifespan(_app):
    db.init()
    billing.init()
    learning.init()
    bench.init()
    external.init()
    apikeys.init()
    updater.startup_ok()  # we started, so the last update (if any) worked
    sched = learning.Scheduler(LEARN_INTERVAL, _learn_check)
    sched.start()
    JOBS.jobs.clear()   # a freshly started server has no jobs of its own yet
    JOBS.queue.clear()
    dispatcher = JOBS.Dispatcher()
    dispatcher.start()
    yield
    sched.stop.set()
    dispatcher.stop.set()
    JOBS.queue.clear()
    for mid, p in list(_jobs.items()):  # ask running jobs to checkpoint and stop
        open(training.stop_flag(mid), "w").close()
    for p in list(_jobs.values()):
        p.join(timeout=30)
    JOBS.jobs.clear()


app = FastAPI(title="Deer AI", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
app.include_router(billing_router)
app.include_router(dev_router)


class NoStaleAssets:
    """Pages and scripts are always revalidated (cheap 304s), so a normal refresh shows the
    latest version after an update. API answers (like the model list) are never cached."""
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        value = b"no-store" if scope["path"].startswith("/api/") else b"no-cache"

        async def send2(msg):
            if msg["type"] == "http.response.start":
                h = [(k, v) for k, v in msg.get("headers", []) if k.lower() != b"cache-control"]
                msg = dict(msg, headers=h + [(b"cache-control", value)])
            await send(msg)
        await self.inner(scope, receive, send2)


app.add_middleware(NoStaleAssets)

# Pages hosted somewhere else (GitHub Pages) may call this server. List those sites, comma separated:
#   DEER_ALLOWED_ORIGINS=https://yourname.github.io
_origins = [o.strip().rstrip("/") for o in os.environ.get("DEER_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"],
                       allow_headers=["Authorization", "Content-Type", "X-API-Key"], max_age=3600)


def _limit(user, kind, amount=1):
    """Enforce the user's plan limits (Claude-style). Raises 429 with reset time."""
    try:
        billing.check(user, kind, amount)
    except billing.LimitReached as e:
        raise HTTPException(429, {"message": e.message, "resets_at": e.resets_at, "limit": True})


# ======================================================================= auth
def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


class Creds(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.\-]+$")
    password: str = Field(min_length=8, max_length=200)


def _session(uid):
    tok = secrets.token_urlsafe(32)
    db.ex("INSERT INTO sessions(token,user_id,created_at,expires_at) VALUES(?,?,?,?)",
          (tok, uid, db.now(), db.now() + SESSION_DAYS * 86400))
    return tok


@app.get("/api/auth/status")
def auth_status():
    n = db.q1("SELECT COUNT(*) n FROM users")["n"]
    return {"needs_setup": n == 0, "registration_open": n == 0 or db.setting("allow_registration") == "1"}


@app.post("/api/auth/register")
def register(c: Creds):
    first = db.q1("SELECT COUNT(*) n FROM users")["n"] == 0
    if not first and db.setting("allow_registration") != "1":
        raise HTTPException(403, "Registration is closed. Ask the admin for an account.")
    if db.q1("SELECT id FROM users WHERE username=?", (c.username,)):
        raise HTTPException(409, "That username is taken.")
    salt = secrets.token_hex(16)
    uid = db.ex("INSERT INTO users(username,pw_hash,salt,role,created_at) VALUES(?,?,?,?,?)",
                (c.username, _hash(c.password, salt), salt, "admin" if first else "user", db.now()))
    return {"token": _session(uid), "user": {"id": uid, "username": c.username, "role": "admin" if first else "user"}}


@app.post("/api/auth/login")
def login(c: Creds, request: Request):
    ip = request.client.host if request.client else "?"
    _fails[ip] = [t for t in _fails[ip] if t > time.time() - 600]
    if len(_fails[ip]) >= 10:
        raise HTTPException(429, "Too many failed sign-ins. Wait 10 minutes and try again.")
    u = db.q1("SELECT * FROM users WHERE username=?", (c.username,))
    if not u or not hmac.compare_digest(u["pw_hash"], _hash(c.password, u["salt"])):
        _fails[ip].append(time.time())
        raise HTTPException(401, "Wrong username or password.")
    return {"token": _session(u["id"]), "user": {"id": u["id"], "username": u["username"], "role": u["role"]}}


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None), user=Depends(current_user)):
    db.ex("DELETE FROM sessions WHERE token=?", (authorization[7:],))
    return {"ok": True}


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return user


# ===================================================================== models
def _trained(mid):
    return os.path.exists(os.path.join(db.model_dir(mid), "weights.npz"))


def _reap():
    JOBS.reap()


def _public_model(r):
    meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
    return {"id": r["id"], "name": r["name"], "kind": r["kind"], "is_default": bool(r["is_default"]),
            "classes": meta.get("classes", []) if r["kind"] in ("canvas", "vision") else None,
            "trained": _trained(r["id"]), "published": bool(r["published"])}


@app.get("/api/models")
def list_models(user=Depends(current_user), all: int = 0):
    """Published models. Admins get unpublished ones too only when they ask (all=1), so the admin's
    chat matches what users see unless they deliberately switch it on."""
    rows = db.q("SELECT * FROM models ORDER BY kind, published DESC, is_default DESC, id DESC")
    out = []
    for r in rows:
        if not _trained(r["id"]):
            continue
        if not r["published"] and not (all and acts_as_admin(user)):
            continue
        out.append(_public_model(r))
    return out


def _resolve(kind, model_id, user):
    admin = acts_as_admin(user)
    if model_id:
        r = db.model_row(model_id)
        if not r or r["kind"] != kind or (not admin and not r["published"]) or not _trained(r["id"]):
            # the page picked a model that was unpublished or deleted: tell it to refresh and fall back
            raise HTTPException(404, {"message": "That model isn't available anymore.", "model_gone": True, "kind": kind})
        return r["id"]
    cands = [m for m in list_models(user, 0) if m["kind"] == kind]
    if not cands and admin:  # nothing published yet: let the admin try unpublished ones
        cands = [m for m in list_models(user, 1) if m["kind"] == kind]
    if not cands:
        what = {"chat": "chat", "vision": "image analysis", "canvas": "image creation"}[kind]
        raise HTTPException(409, f"No {what} model is available yet. An admin needs to train and publish one.")
    return cands[0]["id"]


# ============================================================== conversations
def _conv(cid, user, seed_title):
    if cid:
        c = db.q1("SELECT * FROM conversations WHERE id=? AND user_id=?", (cid, user["id"]))
        if not c:
            raise HTTPException(404, "Conversation not found.")
        return c["id"]
    title = (seed_title or "New chat").strip().replace("\n", " ")[:48] or "New chat"
    return db.ex("INSERT INTO conversations(user_id,title,created_at,updated_at) VALUES(?,?,?,?)",
                 (user["id"], title, db.now(), db.now()))


def _msg(cid, role, kind, content, meta=None):
    mid = db.ex("INSERT INTO messages(conversation_id,role,kind,content,meta,created_at) VALUES(?,?,?,?,?,?)",
                (cid, role, kind, content, json.dumps(meta or {}), db.now()))
    db.ex("UPDATE conversations SET updated_at=? WHERE id=?", (db.now(), cid))
    return {"id": mid, "role": role, "kind": kind, "content": content, "meta": meta or {}}


@app.get("/api/conversations")
def conversations(user=Depends(current_user)):
    return db.q("SELECT id,title,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 200",
                (user["id"],))


@app.get("/api/conversations/{cid}")
def conversation(cid: int, user=Depends(current_user)):
    c = db.q1("SELECT id,title FROM conversations WHERE id=? AND user_id=?", (cid, user["id"]))
    if not c:
        raise HTTPException(404, "Conversation not found.")
    msgs = db.q("SELECT m.id,m.role,m.kind,m.content,m.meta, s.id IS NOT NULL saved FROM messages m "
                "LEFT JOIN saved s ON s.message_id=m.id AND s.user_id=? WHERE m.conversation_id=? ORDER BY m.id",
                (user["id"], cid))
    for m in msgs:
        m["meta"] = json.loads(m["meta"])
        m["saved"] = bool(m["saved"])
    c["messages"] = msgs
    return c


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: int, user=Depends(current_user)):
    if db.q1("SELECT id FROM conversations WHERE id=? AND user_id=?", (cid, user["id"])):
        learning.forget_messages([m["id"] for m in db.q("SELECT id FROM messages WHERE conversation_id=?", (cid,))])
    db.ex("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, user["id"]))
    return {"ok": True}


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[int] = None
    model_id: Optional[int] = None
    temperature: float = Field(0.8, ge=0.05, le=2.0)


@app.post("/api/chat")
def chat(body: ChatIn, user=Depends(current_user)):
    mid = _resolve("chat", body.model_id, user)
    _limit(user, "chat")
    cid = _conv(body.conversation_id, user, body.text)
    hist = db.q("SELECT role,content FROM messages WHERE conversation_id=? AND kind='text' ORDER BY id DESC LIMIT 8",
                (cid,))
    history = [(h["role"], h["content"]) for h in reversed(hist)]
    u = _msg(cid, "user", "text", body.text)
    try:
        reply = INF.chat_reply(mid, history, body.text, body.temperature)
    except INF.NotReady as e:
        raise HTTPException(409, str(e))
    billing.record(user, "chat")
    learning.collect_message(user, mid, body.text)
    a = _msg(cid, "assistant", "text", reply, {"model_id": mid})
    return {"conversation_id": cid, "messages": [u, a], "usage": billing.usage_summary(user)}


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn, user=Depends(current_user)):
    """Same as /api/chat but sends the reply word by word as it's written (server-sent events)."""
    mid = _resolve("chat", body.model_id, user)
    _limit(user, "chat")
    try:
        INF.get(mid)
    except INF.NotReady as e:
        raise HTTPException(409, str(e))
    cid = _conv(body.conversation_id, user, body.text)
    hist = db.q("SELECT role,content FROM messages WHERE conversation_id=? AND kind='text' ORDER BY id DESC LIMIT 8",
                (cid,))
    history = [(h["role"], h["content"]) for h in reversed(hist)]
    u = _msg(cid, "user", "text", body.text)

    def finish(parts):
        billing.record(user, "chat")
        learning.collect_message(user, mid, body.text)
        return _msg(cid, "assistant", "text", "".join(parts).strip() or "…", {"model_id": mid})

    def events():
        yield _sse({"type": "start", "conversation_id": cid, "user_message": u})
        parts = []
        try:
            for piece in INF.chat_stream(mid, history, body.text, body.temperature):
                parts.append(piece)
                yield _sse({"type": "token", "text": piece})
        except GeneratorExit:  # the user closed the page: keep what was written
            finish(parts)
            raise
        except Exception as e:
            yield _sse({"type": "error", "message": f"Deer AI hit a problem: {e}"})
            return
        a = finish(parts)
        yield _sse({"type": "done", "message": a, "usage": billing.usage_summary(user)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class ImagineIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    conversation_id: Optional[int] = None
    model_id: Optional[int] = None
    count: int = Field(4, ge=1, le=8)
    temperature: float = Field(1.0, ge=0.1, le=2.0)


@app.post("/api/imagine")
def imagine(body: ImagineIn, user=Depends(current_user)):
    mid = _resolve("canvas", body.model_id, user)
    _limit(user, "imagine", body.count)
    cid = _conv(body.conversation_id, user, "Imagine: " + body.prompt)
    u = _msg(cid, "user", "imagine", body.prompt)
    try:
        imgs, matched = INF.imagine(mid, body.prompt, body.count, body.temperature)
    except INF.NotReady as e:
        raise HTTPException(409, str(e))
    billing.record(user, "imagine", body.count)
    note = None if matched else "No trained subject matched your words, so these are random picks from what I know."
    a = _msg(cid, "assistant", "images", "", {"images": imgs, "note": note, "model_id": mid})
    return {"conversation_id": cid, "messages": [u, a], "usage": billing.usage_summary(user)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), conversation_id: Optional[int] = Form(None),
                  model_id: Optional[int] = Form(None), user=Depends(current_user)):
    _limit(user, "analyze")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Images must be under 25 MB.")
    return await run_in_threadpool(_analyze_sync, raw, file.filename, conversation_id, model_id, user)


def _analyze_sync(raw, filename, conversation_id, model_id, user):
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't an image Deer AI can read.")
    stats = D.image_stats(im)
    preds, vid = None, None
    try:
        vid = _resolve("vision", model_id, user)
        preds = INF.analyze(vid, im)
    except HTTPException as e:
        if e.status_code != 409:
            raise
    except INF.NotReady:
        preds = None
    billing.record(user, "analyze")
    thumb = D.ImageOps.exif_transpose(im).convert("RGB")
    thumb.thumbnail((384, 384))
    cid = _conv(conversation_id, user, "Image: " + (filename or "upload"))
    u = _msg(cid, "user", "image", INF.to_data_url(thumb, "JPEG"))
    a = _msg(cid, "assistant", "analysis", "", {"predictions": preds, "stats": stats, "model_id": vid})
    learning.collect_upload(user, a["id"], vid, preds, thumb)
    return {"conversation_id": cid, "messages": [u, a], "usage": billing.usage_summary(user)}


# ================================================================ admin: data
@app.get("/api/admin/presets")
def presets(user=Depends(admin_user)):
    return {"presets": PRESETS, "dataset_kind": KIND_DATASET}


@app.get("/api/admin/datasets")
def list_datasets(user=Depends(admin_user)):
    rows = db.q("SELECT * FROM datasets ORDER BY id DESC")
    for r in rows:
        r["stats"] = json.loads(r["stats"])
        r["used_by"] = db.q1("SELECT COUNT(*) n FROM models WHERE dataset_id=?", (r["id"],))["n"]
    return rows


@app.post("/api/admin/datasets")
async def create_dataset(name: str = Form(...), kind: str = Form(...), files: List[UploadFile] = File(...),
                         user=Depends(admin_user)):
    name = name.strip()[:80]
    if not name:
        raise HTTPException(400, "Give the dataset a name.")
    if kind not in ("text", "images"):
        raise HTTPException(400, "Dataset kind must be text or images.")
    did = db.ex("INSERT INTO datasets(name,kind,created_at) VALUES(?,?,?)", (name, kind, db.now()))
    folder = db.dataset_dir(did)
    try:
        if kind == "text":
            blobs = [(f.filename, await f.read()) for f in files]
            stats = D.save_text_dataset(folder, blobs)
        else:
            if len(files) != 1 or not (files[0].filename or "").lower().endswith(".zip"):
                raise D.DatasetError("Upload one .zip file with a folder per class.")
            stats = D.save_image_dataset(folder, await files[0].read())
    except D.DatasetError as e:
        shutil.rmtree(folder, ignore_errors=True)
        db.ex("DELETE FROM datasets WHERE id=?", (did,))
        raise HTTPException(400, str(e))
    db.ex("UPDATE datasets SET stats=? WHERE id=?", (json.dumps(stats), did))
    return {"id": did, "name": name, "kind": kind, "stats": stats}


@app.delete("/api/admin/datasets/{did}")
def delete_dataset(did: int, user=Depends(admin_user)):
    if db.q1("SELECT id FROM models WHERE dataset_id=?", (did,)):
        raise HTTPException(409, "Delete the models trained on this dataset first.")
    db.ex("DELETE FROM datasets WHERE id=?", (did,))
    shutil.rmtree(db.dataset_dir(did), ignore_errors=True)
    return {"ok": True}


# ============================================================== admin: models
class ModelIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: str
    dataset_id: int
    preset: str = "small"
    config: dict = {}
    init_from: Optional[int] = None      # start from this trained model's knowledge
    teacher: Optional[int] = None        # learn from this model's answers while training
    teacher_strength: float = Field(0.5, gt=0, lt=1)


@app.get("/api/admin/models")
def admin_models(user=Depends(admin_user)):
    _reap()
    rows = db.q("SELECT m.*, d.name dataset_name FROM models m LEFT JOIN datasets d ON d.id=m.dataset_id "
                "ORDER BY m.id DESC")
    for r in rows:
        r["config"], r["meta"] = json.loads(r["config"]), json.loads(r["meta"])
        r["meta"].pop("preview_text", None)
        r["trained"] = _trained(r["id"])
        r["last"] = db.q1("SELECT step,loss,metric FROM training_log WHERE model_id=? ORDER BY id DESC LIMIT 1",
                          (r["id"],))
    return rows


@app.post("/api/admin/models")
def create_model(body: ModelIn, user=Depends(admin_user)):
    if body.kind not in KIND_DATASET:
        raise HTTPException(400, "Model type must be chat, vision or canvas.")
    ds = db.q1("SELECT * FROM datasets WHERE id=?", (body.dataset_id,))
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    if ds["kind"] != KIND_DATASET[body.kind]:
        raise HTTPException(400, f"A {body.kind} model needs a {KIND_DATASET[body.kind]} dataset.")
    cfg_in = dict(body.config)
    if body.init_from:
        cfg_in["init_from"] = body.init_from
    if body.teacher:
        cfg_in["teacher"] = body.teacher
        cfg_in.setdefault("distill_alpha", body.teacher_strength)
    try:
        cfg = build_config(body.kind, body.preset, cfg_in)
    except ValueError as e:
        raise HTTPException(400, str(e))
    for key, what in (("init_from", "start from"), ("teacher", "use as a teacher")):
        if cfg.get(key):
            src = db.model_row(cfg[key])
            if not src or src["kind"] != body.kind:
                raise HTTPException(400, f"The model to {what} must be a {body.kind} model.")
            if not _trained(src["id"]):
                raise HTTPException(400, f"'{src['name']}' hasn't been trained yet, so it can't {what.split()[0]} anything.")
            if key == "teacher" and body.kind == "canvas":
                raise HTTPException(400, "Image creation models can start from another model but can't use a teacher.")
            if key == "teacher" and body.kind == "chat" and src["config"]["block_size"] < cfg["block_size"]:
                raise HTTPException(400, f"Teacher '{src['name']}' reads less context ({src['config']['block_size']}) than this model "
                                         f"({cfg['block_size']}). Pick another teacher or a smaller block_size.")
    stats = json.loads(ds["stats"])
    n_classes = len(stats.get("classes", [])) or 2
    params = build_model(body.kind, cfg, n_classes).num_params()
    mid = db.ex("INSERT INTO models(name,kind,dataset_id,config,params,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (body.name.strip(), body.kind, ds["id"], json.dumps(cfg), params, db.now(), db.now()))
    if not db.q1("SELECT id FROM models WHERE kind=? AND is_default=1", (body.kind,)):
        db.ex("UPDATE models SET is_default=1 WHERE id=?", (mid,))
    return {"id": mid, "params": params, "config": cfg}


@app.get("/api/admin/models/{mid}")
def model_detail(mid: int, user=Depends(admin_user)):
    _reap()
    r = db.model_row(mid)
    if not r:
        raise HTTPException(404, "Model not found.")
    r["trained"] = _trained(mid)
    r["log"] = db.q("SELECT step,loss,metric FROM training_log WHERE model_id=? ORDER BY step", (mid,))
    pp = os.path.join(db.model_dir(mid), "preview.png")
    if r["kind"] == "canvas" and os.path.exists(pp):
        with open(pp, "rb") as f:
            r["preview_url"] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    r["queue_position"] = JOBS.queue_position(mid)
    if r["queue_position"]:
        r["wait_reason"] = JOBS.status()["reason"]
    r["new_lessons"] = learning.new_lessons(r)
    r["has_backup"] = os.path.exists(os.path.join(db.model_dir(mid), "weights.prev.npz"))
    return r


class ModelPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    published: Optional[bool] = None
    is_default: Optional[bool] = None


@app.patch("/api/admin/models/{mid}")
def patch_model(mid: int, body: ModelPatch, user=Depends(admin_user)):
    r = db.model_row(mid)
    if not r:
        raise HTTPException(404, "Model not found.")
    if body.name is not None:
        db.ex("UPDATE models SET name=? WHERE id=?", (body.name.strip(), mid))
    if body.published is not None:
        if body.published and not _trained(mid):
            raise HTTPException(409, "Train the model before publishing it.")
        db.ex("UPDATE models SET published=? WHERE id=?", (int(body.published), mid))
    if body.is_default:
        db.ex("UPDATE models SET is_default=0 WHERE kind=?", (r["kind"],))
        db.ex("UPDATE models SET is_default=1 WHERE id=?", (mid,))
    return model_detail(mid, user)


class TrainIn(BaseModel):
    steps: int = Field(500, ge=1, le=1_000_000)


@app.post("/api/admin/models/{mid}/train")
def train(mid: int, body: TrainIn, user=Depends(admin_user)):
    r = db.model_row(mid)
    if not r:
        raise HTTPException(404, "Model not found.")
    with _jobs_lock:
        JOBS.reap()
        if JOBS.busy(mid):
            raise HTTPException(409, "This model is already training or waiting to train.")
        JOBS.enqueue(mid, body.steps)
        started = mid in JOBS.jobs
    st = JOBS.status()
    return {"ok": True, "status": "training" if started else "queued",
            "message": "Training started." if started else f"Waiting for a free slot: {st['reason']}"}


@app.post("/api/admin/models/{mid}/stop")
def stop(mid: int, user=Depends(admin_user)):
    if JOBS.unqueue(mid):  # still waiting: just take it out of the queue
        db.ex("UPDATE models SET status='stopped' WHERE id=?", (mid,))
        return {"ok": True}
    if mid not in _jobs:
        raise HTTPException(409, "This model isn't training.")
    os.makedirs(db.model_dir(mid), exist_ok=True)
    open(training.stop_flag(mid), "w").close()
    return {"ok": True}


@app.post("/api/admin/models/{mid}/reset")
def reset(mid: int, user=Depends(admin_user)):
    if JOBS.busy(mid):
        raise HTTPException(409, "Stop training before resetting.")
    shutil.rmtree(db.model_dir(mid), ignore_errors=True)
    INF.forget(mid)
    db.ex("DELETE FROM training_log WHERE model_id=?", (mid,))
    db.ex("UPDATE models SET step=0,target_steps=0,status='new',error=NULL,published=0,meta='{}' WHERE id=?", (mid,))
    return {"ok": True}


@app.delete("/api/admin/models/{mid}")
def delete_model(mid: int, user=Depends(admin_user)):
    if JOBS.busy(mid):
        raise HTTPException(409, "Stop training before deleting.")
    db.ex("DELETE FROM models WHERE id=?", (mid,))
    shutil.rmtree(db.model_dir(mid), ignore_errors=True)
    INF.forget(mid)
    return {"ok": True}


@app.get("/api/admin/models/{mid}/export")
def export(mid: int, user=Depends(admin_user)):
    r = db.model_row(mid)
    p = os.path.join(db.model_dir(mid), "weights.npz")
    if not r or not os.path.exists(p):
        raise HTTPException(404, "No trained weights to export.")
    return FileResponse(p, filename=f"deer-{r['kind']}-{mid}-step{r['step']}.npz")


@app.post("/api/admin/models/{mid}/sample")
def sample(mid: int, user=Depends(admin_user)):
    """Quick test output for the admin panel."""
    r = db.model_row(mid)
    try:
        if r["kind"] == "chat":
            return {"text": INF.chat_reply(mid, [], "Hello!")}
        if r["kind"] == "canvas":
            imgs, _ = INF.imagine(mid, "", 4)
            return {"images": imgs}
    except INF.NotReady as e:
        raise HTTPException(409, str(e))
    return {}


# =============================================================== admin: users
@app.get("/api/admin/users")
def users(user=Depends(admin_user)):
    return {"users": db.q("SELECT id,username,role,created_at FROM users ORDER BY id"),
            "allow_registration": db.setting("allow_registration") == "1"}


class UserPatch(BaseModel):
    role: str = Field(pattern="^(admin|user)$")


@app.patch("/api/admin/users/{uid}")
def patch_user(uid: int, body: UserPatch, user=Depends(admin_user)):
    if uid == user["id"] and body.role != "admin":
        raise HTTPException(400, "You can't remove your own admin role.")
    db.ex("UPDATE users SET role=? WHERE id=?", (body.role, uid))
    return {"ok": True}


@app.delete("/api/admin/users/{uid}")
def delete_user(uid: int, user=Depends(admin_user)):
    if uid == user["id"]:
        raise HTTPException(400, "You can't delete your own account here.")
    db.ex("DELETE FROM users WHERE id=?", (uid,))
    return {"ok": True}


class SettingsIn(BaseModel):
    allow_registration: bool


@app.patch("/api/admin/settings")
def patch_settings(body: SettingsIn, user=Depends(admin_user)):
    db.set_setting("allow_registration", "1" if body.allow_registration else "0")
    return {"ok": True}


# ================================================================== curate
class CurateIn(BaseModel):
    kind: str = Field(pattern="^(text|images)$")
    dataset_ids: List[int] = Field(max_length=100)
    filters: dict = {}
    name: Optional[str] = Field(None, max_length=80)


@app.post("/api/admin/curate/preview")
def curate_preview(body: CurateIn, user=Depends(admin_user)):
    try:
        return curate.preview(body.kind, body.dataset_ids, body.filters)
    except (curate.CurateError, D.DatasetError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/admin/curate/build")
def curate_build(body: CurateIn, user=Depends(admin_user)):
    try:
        did, stats = curate.build(body.kind, body.name, body.dataset_ids, body.filters)
    except (curate.CurateError, D.DatasetError) as e:
        raise HTTPException(400, str(e))
    size = stats.get("bytes", 0) if body.kind == "text" else stats.get("images", 0)
    return {"id": did, "stats": stats,
            "suggested_preset": curate.suggest_preset(body.kind, size, len(stats.get("classes", [])))}


# ================================================================== update
def _idle_or_409():
    with _jobs_lock:
        JOBS.reap()
        if JOBS.jobs:
            raise HTTPException(409, "A model is training. Stop it or wait for it to finish, then update.")


def _after_update(res):
    res["restarting"] = updater.restart_soon()
    if not res["restarting"]:
        res["note"] = "Restart Deer AI to start the new version (it isn't running as a service here)."
    return res


def _upd(fn, *a):
    try:
        return fn(*a)
    except updater.UpdateError as e:
        raise HTTPException(400, str(e))


@app.get("/api/admin/update")
def update_status(user=Depends(admin_user)):
    st = updater.public_status()
    st["training"] = bool(JOBS.jobs)
    return st


@app.post("/api/admin/update/zip")
async def update_zip(file: UploadFile = File(...), user=Depends(admin_user)):
    _idle_or_409()
    data = await file.read()
    return _after_update(await run_in_threadpool(_upd, updater.apply_zip, data))


class RollbackIn(BaseModel):
    file: Optional[str] = Field(None, max_length=80)


@app.post("/api/admin/update/rollback")
def update_rollback(body: RollbackIn, user=Depends(admin_user)):
    _idle_or_409()
    pick = _upd(updater.restore, body.file)
    return _after_update({"restored": pick["file"]})


@app.post("/api/admin/update/restart")
def update_restart(user=Depends(admin_user)):
    _idle_or_409()
    return _after_update({})


class GithubIn(BaseModel):
    repo: str = Field(min_length=3, max_length=220)
    branch: str = Field("main", max_length=100)


@app.put("/api/admin/update/github")
def github_connect(body: GithubIn, user=Depends(admin_user)):
    _upd(updater.connect, body.repo, body.branch)
    return update_status(user)


class GithubAuto(BaseModel):
    auto: bool


@app.patch("/api/admin/update/github")
def github_auto(body: GithubAuto, user=Depends(admin_user)):
    updater.set_auto(body.auto)
    return update_status(user)


@app.post("/api/admin/update/github/check")
def github_check(user=Depends(admin_user)):
    return _upd(updater.check)


@app.post("/api/admin/update/github/push")
def github_push(user=Depends(admin_user)):
    return _upd(updater.push)


@app.post("/api/admin/update/github/pull")
def github_pull(user=Depends(admin_user)):
    _idle_or_409()
    res = _upd(updater.pull)
    return _after_update(res) if res["updated"] else res


_webhook_waiting = threading.Event()


def _auto_pull():
    """Update after a push; if a model is training, wait for it to finish first."""
    if _webhook_waiting.is_set():
        return
    _webhook_waiting.set()
    try:
        while True:
            with _jobs_lock:
                JOBS.reap()
                busy = bool(JOBS.jobs or JOBS.queue)
            if not busy:
                break
            time.sleep(30)
        res = updater.pull()
        if res["updated"]:
            updater.restart_soon()
    except Exception as e:
        print(f"auto-update failed: {e}")
    finally:
        _webhook_waiting.clear()


@app.post("/api/github/webhook")
async def github_webhook(request: Request):
    """GitHub calls this after every push (set up under the repo's Settings > Webhooks)."""
    body = await request.body()
    if not updater.verify_webhook(body, request.headers.get("x-hub-signature-256")):
        raise HTTPException(401, "Signature check failed.")
    event = request.headers.get("x-github-event", "")
    if event == "ping":
        return {"ok": True, "message": "Deer AI heard GitHub."}
    cfg = updater.gh_config()
    try:
        ref = json.loads(body).get("ref")
    except ValueError:
        ref = None
    if event != "push" or ref != f"refs/heads/{cfg.get('branch', 'main')}":
        return {"ok": True, "ignored": True}
    if not cfg.get("auto"):
        return {"ok": True, "ignored": True, "reason": "Automatic updates are off."}
    threading.Thread(target=_auto_pull, daemon=True).start()
    return {"ok": True, "updating": True}


# ================================================================== system
@app.get("/api/admin/system")
def system_status(user=Depends(admin_user)):
    with _jobs_lock:
        JOBS.reap()
        st = JOBS.status()
    names = {r["id"]: r["name"] for r in db.q("SELECT id, name FROM models")}
    st["running_names"] = [names.get(m, f"#{m}") for m in st["running"]]
    st["queued_names"] = [names.get(m, f"#{m}") for m in st["queued"]]
    return st


# ================================================================ overview
@app.get("/api/admin/overview")
def overview(user=Depends(admin_user)):
    _reap()
    models = db.q("SELECT id, name, kind, status, published, step, target_steps FROM models ORDER BY updated_at DESC")
    trained = [m for m in models if _trained(m["id"])]
    ds = db.q("SELECT kind, COUNT(*) n FROM datasets GROUP BY kind")
    benches = bench.overview()
    return {
        "datasets": {r["kind"]: r["n"] for r in ds},
        "models": len(models), "trained": len(trained),
        "published": {k: sum(1 for m in trained if m["kind"] == k and m["published"]) for k in ("chat", "vision", "canvas")},
        "training": [m for m in models if m["status"] in ("training", "queued")],
        "users": db.q1("SELECT COUNT(*) n FROM users WHERE role!='admin'")["n"],
        "subscribers": db.q1("SELECT COUNT(*) n FROM subscriptions WHERE status IN ('ACTIVE','APPROVED')")["n"],
        "pending_lessons": db.q1("SELECT COUNT(*) n FROM feedback WHERE status='pending'")["n"],
        "benchmarks": len(benches), "bench_runs": db.q1("SELECT COUNT(*) n FROM bench_runs")["n"],
        "leaders": [{"benchmark": b["name"], "kind": b["kind"], **b["leaders"][0]} for b in benches if b["leaders"]][:6],
        "recent": models[:5],
    }


# =============================================================== benchmark
class BenchIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: str = Field(pattern="^(chat|vision|canvas)$")
    source: str = Field("custom", pattern="^(custom|dataset)$")
    text: str = Field("", max_length=200_000)
    dataset_id: Optional[int] = None
    size: int = Field(50, ge=1, le=500)
    judge_model_id: Optional[int] = None
    per_class: int = Field(8, ge=1, le=64)


@app.get("/api/admin/benchmarks")
def benchmarks(user=Depends(admin_user)):
    return bench.overview()


@app.post("/api/admin/benchmarks")
def create_benchmark(body: BenchIn, user=Depends(admin_user)):
    try:
        return bench.create(body.name, body.kind, body.source, body.text, body.dataset_id, body.size,
                            body.judge_model_id, body.per_class)
    except bench.BenchError as e:
        raise HTTPException(400, str(e))


@app.get("/api/admin/benchmarks/{bid}")
def benchmark_detail(bid: int, user=Depends(admin_user)):
    try:
        b = bench.get(bid)
    except bench.BenchError as e:
        raise HTTPException(404, str(e))
    b["runs"] = bench.runs(bid)
    return b


class BenchRunIn(BaseModel):
    model_ids: List[int] = Field([], max_length=20)
    external_ids: List[int] = Field([], max_length=10)


@app.post("/api/admin/benchmarks/{bid}/run")
def run_benchmark(bid: int, body: BenchRunIn, user=Depends(admin_user)):
    try:
        return {"runs": bench.run(bid, body.model_ids, body.external_ids)}
    except bench.BenchError as e:
        raise HTTPException(400, str(e))


@app.get("/api/admin/benchmarks/{bid}/sheet")
def benchmark_sheet(bid: int, user=Depends(admin_user)):
    try:
        return {"questions": bench.export_sheet(bid)}
    except bench.BenchError as e:
        raise HTTPException(400, str(e))


class SheetIn(BaseModel):
    name: str = Field("", max_length=60)
    answers: Union[List[Optional[str]], str]


@app.post("/api/admin/benchmarks/{bid}/sheet")
def benchmark_import(bid: int, body: SheetIn, user=Depends(admin_user)):
    try:
        return bench.import_sheet(bid, body.name, body.answers)
    except bench.BenchError as e:
        raise HTTPException(400, str(e))


# --------------------------------------------- other AI models (comparison only)
@app.get("/api/admin/external")
def external_list(user=Depends(admin_user)):
    return {"models": external.listing(), "presets": {k: {"api": v[0], "base_url": v[1]} for k, v in external.PRESETS.items()}}


class ExternalIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    api: str = Field(pattern="^(openai|anthropic)$")
    base_url: str = Field(min_length=8, max_length=300)
    model: str = Field(min_length=1, max_length=120)
    api_key: str = Field("", max_length=400)
    max_tokens: int = Field(800, ge=50, le=8000)


@app.post("/api/admin/external")
def external_add(body: ExternalIn, user=Depends(admin_user)):
    try:
        return external.add(body.name, body.api, body.base_url, body.model, body.api_key, body.max_tokens)
    except external.ExternalError as e:
        raise HTTPException(400, str(e))


@app.post("/api/admin/external/{eid}/test")
def external_test(eid: int, user=Depends(admin_user)):
    try:
        return external.test(eid)
    except external.ExternalError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/admin/external/{eid}")
def external_delete(eid: int, user=Depends(admin_user)):
    db.ex("DELETE FROM external_models WHERE id=?", (eid,))
    return {"ok": True}


@app.delete("/api/admin/benchmarks/{bid}")
def delete_benchmark(bid: int, user=Depends(admin_user)):
    db.ex("DELETE FROM benchmarks WHERE id=?", (bid,))
    return {"ok": True}


# =================================================================== saved
class SaveIn(BaseModel):
    message_id: int


def _own_message(user, mid):
    m = db.q1("SELECT m.id FROM messages m JOIN conversations c ON c.id=m.conversation_id "
              "WHERE m.id=? AND c.user_id=?", (mid, user["id"]))
    if not m:
        raise HTTPException(404, "Message not found.")


@app.post("/api/saved")
def save_message(body: SaveIn, user=Depends(current_user)):
    _own_message(user, body.message_id)
    if db.q1("SELECT COUNT(*) n FROM saved WHERE user_id=?", (user["id"],))["n"] >= 2000:
        raise HTTPException(409, "You've saved 2000 items. Remove some to save more.")
    db.ex("INSERT OR IGNORE INTO saved(user_id,message_id,created_at) VALUES(?,?,?)",
          (user["id"], body.message_id, db.now()))
    return {"saved": True}


@app.delete("/api/saved/{message_id}")
def unsave_message(message_id: int, user=Depends(current_user)):
    db.ex("DELETE FROM saved WHERE user_id=? AND message_id=?", (user["id"], message_id))
    return {"saved": False}


@app.get("/api/saved")
def saved_items(user=Depends(current_user)):
    rows = db.q("SELECT m.id, m.role, m.kind, m.content, m.meta, m.conversation_id, c.title, s.created_at saved_at "
                "FROM saved s JOIN messages m ON m.id=s.message_id JOIN conversations c ON c.id=m.conversation_id "
                "WHERE s.user_id=? ORDER BY s.id DESC LIMIT 500", (user["id"],))
    for r in rows:
        r["meta"] = json.loads(r["meta"])
        r["saved"] = True
        if r["kind"] == "text" and r["role"] == "assistant":  # show what it answered
            q = db.q1("SELECT content FROM messages WHERE conversation_id=? AND id<? AND role='user' AND kind='text' "
                      "ORDER BY id DESC LIMIT 1", (r["conversation_id"], r["id"]))
            r["question"] = q["content"] if q else None
    return rows


@app.get("/api/usage")
def usage(user=Depends(current_user)):
    plan, _ = billing.current(user)
    return {"plan": plan["name"], "unlimited_admin": acts_as_admin(user), "usage": billing.usage_summary(user),
            "test": user.get("test"), "role": user["role"], "train_ok": bool(user.get("train_ok", 1)),
            "learning": db.setting("learn_from_messages") == "1" or db.setting("learn_from_uploads") == "1"}


# ================================================================= privacy
class PrivacyIn(BaseModel):
    train_ok: bool


@app.put("/api/me/privacy")
def set_privacy(body: PrivacyIn, user=Depends(current_user)):
    db.ex("UPDATE users SET train_ok=? WHERE id=?", (int(body.train_ok), user["id"]))
    return {"train_ok": body.train_ok}


# =============================================================== test mode
class TestModeIn(BaseModel):
    on: bool
    plan_id: Optional[int] = None  # preview a plan without paying; None = your real subscription or Free


@app.put("/api/me/test-mode")
def set_test_mode(body: TestModeIn, user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admins only.")
    if body.plan_id and not db.q1("SELECT id FROM plans WHERE id=?", (body.plan_id,)):
        raise HTTPException(404, "Plan not found.")
    db.ex("UPDATE users SET test_mode=? WHERE id=?",
          (json.dumps({"on": body.on, "plan_id": body.plan_id}) if body.on else "", user["id"]))
    return {"test": {"on": True, "plan_id": body.plan_id} if body.on else None}


@app.post("/api/me/reset-usage")
def reset_usage(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admins only.")
    db.ex("DELETE FROM usage WHERE user_id=?", (user["id"],))
    return {"ok": True}


# ================================================================ learning
class FeedbackIn(BaseModel):
    message_id: int
    rating: int = Field(ge=-1, le=1)
    correction: Optional[str] = Field(None, max_length=2000)
    label: Optional[str] = Field(None, max_length=60)


@app.post("/api/feedback")
def feedback(body: FeedbackIn, user=Depends(current_user)):
    try:
        return learning.submit(user, body.message_id, body.rating, body.correction, body.label)
    except (learning.FeedbackError, D.DatasetError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/admin/learning")
def learning_inbox(status: str = "pending", user=Depends(admin_user)):
    if status not in ("pending", "applied", "rejected", "info"):
        raise HTTPException(400, "Unknown status.")
    return {"items": learning.inbox(status), **learning.stats()}


class ReviewIn(BaseModel):
    ids: List[int] = Field(max_length=500)
    action: str = Field(pattern="^(approve|reject)$")


@app.post("/api/admin/learning/review")
def learning_review(body: ReviewIn, user=Depends(admin_user)):
    try:
        n = learning.review(body.ids, body.action)
    except (learning.FeedbackError, D.DatasetError) as e:
        raise HTTPException(400, str(e))
    return {"done": n, "skipped": learning.review.last_skipped, **learning.stats()}


class LearnSettingsIn(BaseModel):
    auto_approve: Optional[bool] = None
    from_messages: Optional[bool] = None
    from_uploads: Optional[bool] = None


@app.patch("/api/admin/learning/settings")
def learning_settings(body: LearnSettingsIn, user=Depends(admin_user)):
    for key, val in (("learn_auto_approve", body.auto_approve), ("learn_from_messages", body.from_messages),
                     ("learn_from_uploads", body.from_uploads)):
        if val is not None:
            db.set_setting(key, "1" if val else "0")
    return learning.stats()


class ModelLearnIn(BaseModel):
    auto_learn: bool
    learn_steps: int = Field(300, ge=10, le=100000)
    learn_min: int = Field(10, ge=1, le=100000)


@app.patch("/api/admin/models/{mid}/learning")
def model_learning(mid: int, body: ModelLearnIn, user=Depends(admin_user)):
    if not db.model_row(mid):
        raise HTTPException(404, "Model not found.")
    db.ex("UPDATE models SET auto_learn=?, learn_steps=?, learn_min=? WHERE id=?",
          (int(body.auto_learn), body.learn_steps, body.learn_min, mid))
    return model_detail(mid, user)


@app.post("/api/admin/models/{mid}/learn")
def learn_now(mid: int, user=Depends(admin_user)):
    """Run one guarded fine-tune on the latest lessons right now."""
    with _jobs_lock:
        JOBS.reap()
        if JOBS.busy(mid):
            raise HTTPException(409, "This model is already training or waiting to train.")
        if not _trained(mid):
            raise HTTPException(409, "Train the model first.")
        learning.tick({}, 10 ** 6, _enqueue_learn, force_model=mid)
    return {"ok": True, "status": "training" if mid in JOBS.jobs else "queued"}


@app.post("/api/admin/models/{mid}/rollback")
def rollback(mid: int, user=Depends(admin_user)):
    """Undo the most recent training run (manual or auto-learn)."""
    if JOBS.busy(mid):
        raise HTTPException(409, "Stop training before rolling back.")
    if not db.model_row(mid):
        raise HTTPException(404, "Model not found.")
    info = training.restore_checkpoint(mid)
    if not info:
        raise HTTPException(409, "There's no earlier checkpoint to go back to.")
    INF.forget(mid)
    meta = info["meta"] if info["meta"] is not None else db.model_row(mid)["meta"]
    meta["last_learn"] = {"at": db.now(), "outcome": "rolled back by admin"}
    if info["step"] is not None:
        db.ex("UPDATE models SET step=?, meta=? WHERE id=?", (info["step"], json.dumps(meta), mid))
    else:
        db.ex("UPDATE models SET meta=? WHERE id=?", (json.dumps(meta), mid))
    return model_detail(mid, user)


# ===================================================================== static
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


def _page_route(name):
    path = os.path.join(STATIC, f"{name}.html")

    def page():  # no parameters: nothing from the request can change which file is sent
        return FileResponse(path)
    return page


for _page in ("index", "admin", "plans", "developers"):  # the same pages under their file names (used by GitHub Pages builds)
    app.add_api_route(f"/{_page}.html", _page_route(_page), methods=["GET"], include_in_schema=False)


@app.get("/admin")
def admin_page():
    return FileResponse(os.path.join(STATIC, "admin.html"))


@app.get("/developers")
def developers_page():
    return FileResponse(os.path.join(STATIC, "developers.html"))


@app.get("/plans")
def plans_page():
    return FileResponse(os.path.join(STATIC, "plans.html"))


@app.get("/sw.js")
def service_worker():  # served from the root so it can control the whole app
    return FileResponse(os.path.join(STATIC, "sw.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(os.path.join(STATIC, "manifest.webmanifest"), media_type="application/manifest+json")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
