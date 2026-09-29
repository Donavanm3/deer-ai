"""Billing routes: plans, usage, PayPal subscribe/cancel, webhooks, and admin controls."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import db
from .auth import admin_user, current_user
from .. import billing as B
from ..billing.paypal import PayPalError

router = APIRouter()


def _fail(e):
    if isinstance(e, PayPalError):
        return HTTPException(502, f"PayPal said: {e}")
    return HTTPException(400, str(e))


@router.get("/api/billing")
def billing(user=Depends(current_user)):
    plan, sub = B.current(user)
    cfg = B.config()
    return {
        "plan": B.public_plan(plan),
        "subscription": {"status": sub["status"], "period_end": sub["period_end"]} if sub else None,
        "usage": B.usage_summary(user),
        "plans": [B.public_plan(p) for p in B.plans()],
        "paypal": {"client_id": cfg["client_id"], "mode": cfg["mode"]} if B.client() else None,
        "user_id": user["id"], "is_admin": user["role"] == "admin" and not user.get("test"),
        "test": user.get("test"), "role": user["role"], "train_ok": bool(user.get("train_ok", 1)),
    }


class ConfirmIn(BaseModel):
    subscription_id: str = Field(min_length=3, max_length=64)


@router.post("/api/billing/confirm")
def confirm(body: ConfirmIn, user=Depends(current_user)):
    try:
        B.confirm(user, body.subscription_id)
    except (B.BillingError, PayPalError) as e:
        raise _fail(e)
    return billing(user)


@router.post("/api/billing/cancel")
def cancel(user=Depends(current_user)):
    try:
        B.cancel(user)
    except (B.BillingError, PayPalError) as e:
        raise _fail(e)
    return billing(user)


@router.post("/api/paypal/webhook")
async def webhook(request: Request):
    raw = await request.body()
    try:
        kind = B.handle_webhook(dict(request.headers), raw)
    except B.BillingError as e:
        raise HTTPException(400, str(e))
    except PayPalError as e:  # let PayPal retry later
        raise HTTPException(503, str(e))
    return {"ok": True, "handled": kind}


# ==================================================================== admin
@router.get("/api/admin/billing")
def admin_billing(user=Depends(admin_user)):
    cfg = B.config()
    subs = db.q("SELECT s.id, s.status, s.period_end, s.paypal_id, s.created_at, u.username, p.name plan "
                "FROM subscriptions s JOIN users u ON u.id=s.user_id JOIN plans p ON p.id=s.plan_id "
                "ORDER BY s.updated_at DESC LIMIT 200")
    rev = db.q("SELECT currency, SUM(amount) total, COUNT(*) n FROM payments WHERE status='COMPLETED' "
               "AND created_at>? GROUP BY currency", (db.now() - 30 * 86400,))
    return {
        "paypal": {"client_id": cfg["client_id"], "mode": cfg["mode"], "webhook_id": cfg["webhook_id"],
                   "secret_set": bool(cfg["secret"]), "configured": B.client() is not None},
        "plans": B.plans(active_only=False), "subscriptions": subs, "revenue_30d": rev,
        "active_paid": sum(1 for s in subs if s["status"] in B.PAID_ACCESS),
    }


class PayPalIn(BaseModel):
    client_id: str = Field(max_length=200)
    secret: Optional[str] = Field(None, max_length=200)
    mode: str = Field(pattern="^(sandbox|live)$")
    webhook_id: str = Field("", max_length=100)


@router.put("/api/admin/billing/paypal")
def save_paypal(body: PayPalIn, user=Depends(admin_user)):
    db.set_setting("paypal_client_id", body.client_id.strip())
    if body.secret:
        db.set_setting("paypal_secret", body.secret.strip())
    db.set_setting("paypal_mode", body.mode)
    db.set_setting("paypal_webhook_id", body.webhook_id.strip())
    pp = B.client()
    if not pp:
        raise HTTPException(400, "Add both the client ID and the secret.")
    try:
        pp.token()
    except (PayPalError, OSError) as e:
        raise HTTPException(400, f"Saved, but PayPal rejected the keys: {e}")
    return admin_billing(user)


class PlanIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    blurb: str = Field("", max_length=120)
    price: float = Field(ge=0, le=10000)
    currency: str = Field("USD", pattern="^[A-Z]{3}$")
    chat_limit: int = Field(ge=-1)
    window_hours: int = Field(5, ge=1, le=720)
    image_limit: int = Field(ge=-1)
    analyze_limit: int = Field(ge=-1)
    sort: int = 0
    active: bool = True


@router.post("/api/admin/plans")
def create_plan(body: PlanIn, user=Depends(admin_user)):
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-") or "plan"
    while db.q1("SELECT id FROM plans WHERE slug=?", (slug,)):
        slug += "-2"
    d = body.model_dump()
    db.ex("INSERT INTO plans(slug,name,blurb,price,currency,chat_limit,window_hours,image_limit,analyze_limit,sort,active,created_at)"
          " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (slug, d["name"], d["blurb"], d["price"], d["currency"], d["chat_limit"],
                                               d["window_hours"], d["image_limit"], d["analyze_limit"], d["sort"],
                                               int(d["active"]), db.now()))
    return admin_billing(user)


@router.patch("/api/admin/plans/{pid}")
def edit_plan(pid: int, body: PlanIn, user=Depends(admin_user)):
    p = db.q1("SELECT * FROM plans WHERE id=?", (pid,))
    if not p:
        raise HTTPException(404, "Plan not found.")
    if not body.active and p["price"] == 0 and not db.q1(
            "SELECT id FROM plans WHERE price=0 AND active=1 AND id!=?", (pid,)):
        raise HTTPException(400, "Keep at least one free plan active; it's what signed-in users get by default.")
    d = body.model_dump()
    db.ex("UPDATE plans SET name=?,blurb=?,price=?,currency=?,chat_limit=?,window_hours=?,image_limit=?,"
          "analyze_limit=?,sort=?,active=? WHERE id=?",
          (d["name"], d["blurb"], d["price"], d["currency"], d["chat_limit"], d["window_hours"], d["image_limit"],
           d["analyze_limit"], d["sort"], int(d["active"]), pid))
    return admin_billing(user)


@router.post("/api/admin/plans/{pid}/sync")
def sync(pid: int, user=Depends(admin_user)):
    try:
        B.sync_plan(pid)
    except (B.BillingError, PayPalError) as e:
        raise _fail(e)
    return admin_billing(user)
