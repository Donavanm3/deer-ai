"""
The Deer AI developer API (/v1/...) plus the routes behind the API page and Admin > API.

    curl https://your-site/v1/chat \\
      -H "Authorization: Bearer dk_..." -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Write a JS game loop"}]}'
"""
import base64
import io
import json
import secrets
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import db
from .auth import admin_user, current_user
from .. import apikeys as K
from .. import billing as B
from .. import data as D
from .. import inference as INF
from ..billing.paypal import PayPalError

router = APIRouter()


def _err(e: K.ApiError):
    headers = {"Retry-After": str(e.extra["retry_after"])} if "retry_after" in e.extra else None
    body = {"error": {"type": e.extra.get("type", "invalid_request"), "message": e.message,
                      **{k: v for k, v in e.extra.items() if k not in ("type", "retry_after")}}}
    return JSONResponse(body, status_code=e.status, headers=headers)


def _model(kind, key, wanted):
    """Published, trained models of this kind (the owner's own keys may use unpublished ones too)."""
    rows = db.q("SELECT * FROM models WHERE kind=? ORDER BY published DESC, is_default DESC, id DESC", (kind,))
    ok = [r for r in rows if INF.os.path.exists(INF.os.path.join(db.model_dir(r["id"]), "weights.npz"))
          and (r["published"] or K.free(key))]
    if wanted not in (None, ""):
        for r in ok:
            if str(r["id"]) == str(wanted) or r["name"].lower() == str(wanted).lower():
                return r
        raise K.ApiError(404, f"No {kind} model called '{wanted}'. GET /v1/models lists them.", type="not_found")
    if not ok:
        raise K.ApiError(503, f"No {kind} model is available yet.", type="unavailable")
    return ok[0]


def _req_id():
    return "req_" + secrets.token_hex(8)


# ======================================================================= /v1
@router.get("/v1/models")
def v1_models(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    try:
        key = K.authenticate(authorization, x_api_key, None)
    except K.ApiError as e:
        return _err(e)
    out = []
    for kind in ("chat", "analyze", "imagine"):
        mk = {"chat": "chat", "analyze": "vision", "imagine": "canvas"}[kind]
        for r in db.q("SELECT * FROM models WHERE kind=? ORDER BY is_default DESC, id", (mk,)):
            if not INF.os.path.exists(INF.os.path.join(db.model_dir(r["id"]), "weights.npz")):
                continue
            if not r["published"] and not K.free(key):
                continue
            meta = json.loads(r["meta"])
            out.append({"id": r["id"], "name": r["name"], "endpoint": f"/v1/{kind}", "default": bool(r["is_default"]),
                        **({"classes": meta.get("classes", [])} if kind != "chat" else
                           {"context_tokens": json.loads(r["config"])["block_size"]})})
    return {"models": out, "prices": K.prices()}


class ChatMsg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=20000)


class V1Chat(BaseModel):
    messages: List[ChatMsg] = Field(min_length=1, max_length=100)
    model: Optional[str] = None
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    max_tokens: int = Field(1500, ge=1, le=4000)
    stream: bool = False


@router.post("/v1/chat")
def v1_chat(body: V1Chat, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    try:
        key = K.authenticate(authorization, x_api_key, "chat")
        if body.messages[-1].role != "user":
            raise K.ApiError(400, "The last message must be from the user.")
        row = _model("chat", key, body.model)
        history = [(m.role, m.content) for m in body.messages[:-1]]
        text = body.messages[-1].content
        inp = INF.prompt_tokens(row["id"], history, text)
        max_out = min(body.max_tokens, INF.MAX_REPLY)
        K.ensure_funds(key, K.cost_chat(inp, max_out))
    except K.ApiError as e:
        return _err(e)
    rid, t0 = _req_id(), time.time()

    def finish(reply):
        out = len(reply.encode("utf-8"))
        bill = K.record(key, "chat", row["id"], row["name"], inp=inp, out=out, cost=K.cost_chat(inp, out),
                        ms=int((time.time() - t0) * 1000))
        return {"usage": {"input_tokens": inp, "output_tokens": out}, "cost": {"usd": bill["cost"], "charged": bill["charged"]},
                "balance": bill["balance"]}

    stream = INF.chat_stream(row["id"], history, text, body.temperature, max_new=max_out)
    if not body.stream:
        reply = "".join(stream).strip()
        return {"id": rid, "object": "chat.reply", "model": row["name"], "model_id": row["id"], "reply": reply, **finish(reply)}

    def events():
        parts = []
        yield f"data: {json.dumps({'type': 'start', 'id': rid, 'model': row['name']})}\n\n"
        try:
            for piece in stream:
                parts.append(piece)
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
        except GeneratorExit:  # the caller hung up: bill only what was written
            finish("".join(parts))
            raise
        yield f"data: {json.dumps({'type': 'done', 'reply': ''.join(parts).strip(), **finish(''.join(parts))})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class V1Analyze(BaseModel):
    image: str = Field(min_length=10, max_length=36_000_000, description="base64 or a data: URL")
    model: Optional[str] = None
    top: int = Field(5, ge=1, le=50)


@router.post("/v1/analyze")
def v1_analyze(body: V1Analyze, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    try:
        key = K.authenticate(authorization, x_api_key, "analyze")
        row = _model("vision", key, body.model)
        K.ensure_funds(key, K.cost_images(1, "analyze"))
        raw = body.image.split(",", 1)[1] if body.image.startswith("data:") else body.image
        try:
            im = Image.open(io.BytesIO(base64.b64decode(raw, validate=False)))
            im.load()
        except (UnidentifiedImageError, OSError, ValueError):
            raise K.ApiError(400, "'image' must be a base64-encoded PNG, JPEG, GIF or WebP.")
    except K.ApiError as e:
        return _err(e)
    t0 = time.time()
    preds = INF.analyze(row["id"], im, top=body.top)
    stats = D.image_stats(im)
    bill = K.record(key, "analyze", row["id"], row["name"], images=1, cost=K.cost_images(1, "analyze"),
                    ms=int((time.time() - t0) * 1000))
    return {"id": _req_id(), "object": "analysis", "model": row["name"], "model_id": row["id"], "predictions": preds,
            "stats": stats, "usage": {"images": 1}, "cost": {"usd": bill["cost"], "charged": bill["charged"]},
            "balance": bill["balance"]}


class V1Imagine(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    model: Optional[str] = None
    count: int = Field(1, ge=1, le=8)
    temperature: float = Field(1.0, ge=0.1, le=2.0)


@router.post("/v1/imagine")
def v1_imagine(body: V1Imagine, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    try:
        key = K.authenticate(authorization, x_api_key, "imagine")
        row = _model("canvas", key, body.model)
        K.ensure_funds(key, K.cost_images(body.count, "imagine"))
    except K.ApiError as e:
        return _err(e)
    t0 = time.time()
    imgs, matched = INF.imagine(row["id"], body.prompt, body.count, body.temperature)
    bill = K.record(key, "imagine", row["id"], row["name"], images=body.count,
                    cost=K.cost_images(body.count, "imagine"), ms=int((time.time() - t0) * 1000))
    return {"id": _req_id(), "object": "images", "model": row["name"], "model_id": row["id"],
            "images": [{"label": i["label"], "png_base64": i["data_url"].split(",", 1)[1]} for i in imgs],
            "matched_prompt": matched, "usage": {"images": body.count},
            "cost": {"usd": bill["cost"], "charged": bill["charged"]}, "balance": bill["balance"]}


@router.get("/v1/usage")
def v1_usage(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    try:
        key = K.authenticate(authorization, x_api_key, None)
    except K.ApiError as e:
        return _err(e)
    return {"balance": K.balance(key["user_id"]), "key": K.public_key(key),
            "account_spent_this_month": K.dollars(K.month_spend(user_id=key["user_id"]))}


# ============================================================ the API page
def _fail(e):
    if isinstance(e, K.ApiError):
        return HTTPException(e.status, e.message)
    if isinstance(e, PayPalError):
        return HTTPException(502, f"PayPal said: {e}")
    return HTTPException(400, str(e))


@router.get("/api/dev")
def dev_home(user=Depends(current_user)):
    pp = B.client()
    return {"enabled": db.setting("api_enabled", "1") == "1", "balance": K.balance(user["id"]),
            "free": user["role"] == "admin", "keys": K.list_keys(user["id"]), "prices": K.prices(), "packs": K.packs(),
            "min_topup": float(db.setting("api_min_topup", "5")),
            "paypal": {"client_id": B.config()["client_id"]} if pp else None,
            "usage": K.usage_report(user["id"]), "ledger": K.ledger(user["id"]),
            "spent_this_month": K.dollars(K.month_spend(user_id=user["id"]))}


class KeyIn(BaseModel):
    name: str = Field("My key", max_length=60)
    scopes: List[str] = list(K.SCOPES)
    rate_per_min: int = Field(60, ge=1, le=600)
    monthly_cap: float = Field(0, ge=0, le=100000)


@router.post("/api/dev/keys")
def dev_create_key(body: KeyIn, user=Depends(current_user)):
    try:
        return K.create_key(user, body.name, body.scopes, body.rate_per_min, body.monthly_cap)
    except K.ApiError as e:
        raise _fail(e)


class KeyPatch(BaseModel):
    name: Optional[str] = Field(None, max_length=60)
    scopes: Optional[List[str]] = None
    rate_per_min: Optional[int] = Field(None, ge=1, le=600)
    monthly_cap: Optional[float] = Field(None, ge=0, le=100000)
    revoke: bool = False


@router.patch("/api/dev/keys/{kid}")
def dev_update_key(kid: int, body: KeyPatch, user=Depends(current_user)):
    try:
        return K.update_key(user["id"], kid, **body.model_dump())
    except K.ApiError as e:
        raise _fail(e)


class OrderIn(BaseModel):
    amount: float = Field(gt=0, le=1000)


@router.post("/api/dev/credit/order")
def dev_order(body: OrderIn, user=Depends(current_user)):
    if body.amount < float(db.setting("api_min_topup", "5")):
        raise HTTPException(400, f"The smallest top-up is ${db.setting('api_min_topup', '5')}.")
    try:
        pp = B.require_client()
        o = pp.create_order(round(body.amount, 2), "USD", f"api:{user['id']}", "Deer AI API credit")
    except (B.BillingError, PayPalError) as e:
        raise _fail(e)
    return {"id": o["id"]}


class CaptureIn(BaseModel):
    order_id: str = Field(min_length=3, max_length=64)


def credit_from_order(order, expect_user=None):
    """Credits a completed PayPal order once. Used after checkout and by the webhook."""
    unit = (order.get("purchase_units") or [{}])[0]
    custom = unit.get("custom_id") or ""
    caps = ((unit.get("payments") or {}).get("captures") or [])
    cap = next((c for c in caps if c.get("status") == "COMPLETED"), None)
    if not custom.startswith("api:") or not cap:
        raise K.ApiError(400, "That payment isn't a completed Deer AI credit purchase.")
    uid = int(custom[4:])
    if expect_user is not None and uid != expect_user:
        raise K.ApiError(403, "That payment belongs to another account.")
    if cap["amount"]["currency_code"] != "USD":
        raise K.ApiError(400, "Only USD payments can be credited.")
    amount = float(cap["amount"]["value"])
    added = K.add_credit(uid, round(amount * K.MICRO), "topup", f"paypal:{order['id']}", f"PayPal ${amount:.2f}")
    return uid, amount, added


@router.post("/api/dev/credit/capture")
def dev_capture(body: CaptureIn, user=Depends(current_user)):
    try:
        pp = B.require_client()
        try:
            order = pp.capture_order(body.order_id)
        except PayPalError as e:
            if "ORDER_ALREADY_CAPTURED" not in str(e.body):
                raise
            order = pp.get_order(body.order_id)
        _, amount, added = credit_from_order(order, expect_user=user["id"])
    except (K.ApiError, B.BillingError, PayPalError) as e:
        raise _fail(e)
    return {"added": amount if added else 0, "balance": K.balance(user["id"])}


# ================================================================ Admin > API
@router.get("/api/admin/api")
def admin_api(user=Depends(admin_user)):
    since = time.time() - 30 * 86400
    top = db.q("SELECT u.id, u.username, u.api_balance balance, "
               "(SELECT COALESCE(SUM(cost),0) FROM api_usage a WHERE a.user_id=u.id AND a.created_at>=? AND a.charged=1) spent, "
               "(SELECT COUNT(*) FROM api_keys k WHERE k.user_id=u.id AND k.revoked_at IS NULL) keys "
               "FROM users u ORDER BY spent DESC, balance DESC LIMIT 50", (since,))
    for r in top:
        r["balance"], r["spent"] = K.dollars(r["balance"]), K.dollars(r["spent"])
    topups = db.q1("SELECT COALESCE(SUM(amount),0) s, COUNT(*) n FROM credit_ledger WHERE kind='topup' AND created_at>=?", (since,))
    return {"enabled": db.setting("api_enabled", "1") == "1", "prices": K.prices(), "packs": K.packs(),
            "welcome_credit": float(db.setting("api_welcome_credit", "0")), "min_topup": float(db.setting("api_min_topup", "5")),
            "usage": K.usage_report(None), "users": [u for u in top if u["keys"] or u["balance"] or u["spent"]],
            "topups_30d": {"usd": K.dollars(topups["s"]), "count": topups["n"]},
            "outstanding_credit": K.dollars(db.q1("SELECT COALESCE(SUM(api_balance),0) s FROM users")["s"])}


class ApiSettings(BaseModel):
    enabled: Optional[bool] = None
    chat_in_per_1k: Optional[float] = Field(None, ge=0, le=100)
    chat_out_per_1k: Optional[float] = Field(None, ge=0, le=100)
    image: Optional[float] = Field(None, ge=0, le=100)
    analyze: Optional[float] = Field(None, ge=0, le=100)
    welcome_credit: Optional[float] = Field(None, ge=0, le=100)
    min_topup: Optional[float] = Field(None, ge=1, le=1000)
    packs: Optional[str] = Field(None, max_length=100)


@router.patch("/api/admin/api")
def admin_api_settings(body: ApiSettings, user=Depends(admin_user)):
    if body.enabled is not None:
        db.set_setting("api_enabled", "1" if body.enabled else "0")
    K.set_prices(body.model_dump())
    for key, val in (("api_welcome_credit", body.welcome_credit), ("api_min_topup", body.min_topup)):
        if val is not None:
            db.set_setting(key, repr(float(val)))
    if body.packs is not None:
        db.set_setting("api_packs", body.packs)
    return admin_api(user)


class GrantIn(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    amount: float = Field(ge=-10000, le=10000)
    note: str = Field("", max_length=120)


@router.post("/api/admin/api/grant")
def admin_grant(body: GrantIn, user=Depends(admin_user)):
    u = db.q1("SELECT id FROM users WHERE username=?", (body.username,))
    if not u:
        raise HTTPException(404, "No user with that name.")
    if body.amount == 0:
        raise HTTPException(400, "Enter an amount (negative to take credit away).")
    K.add_credit(u["id"], round(body.amount * K.MICRO), "grant", f"grant:{secrets.token_hex(8)}",
                 body.note or f"Added by {user['username']}")
    return admin_api(user)
