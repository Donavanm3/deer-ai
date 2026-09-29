"""
Deer AI billing: Claude-style plans with usage limits, paid monthly through PayPal Subscriptions.

  Free / Pro / Max plans (editable by the admin)
  chat messages counted in a rolling window (default 5 hours), images and analyses per 24 hours
  PayPal verifies every subscription server-side; webhooks keep status and renewals in sync
"""
import json
import os
import threading
import time
from datetime import datetime

from ..server import db
from .paypal import PayPal, PayPalError

SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, blurb TEXT NOT NULL DEFAULT '',
  price REAL NOT NULL DEFAULT 0, currency TEXT NOT NULL DEFAULT 'USD',
  chat_limit INTEGER NOT NULL, window_hours INTEGER NOT NULL DEFAULT 5,
  image_limit INTEGER NOT NULL, analyze_limit INTEGER NOT NULL,
  sort INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
  paypal_plan_id TEXT, synced TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS subscriptions (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  plan_id INTEGER NOT NULL REFERENCES plans(id), paypal_id TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL, period_end REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_sub_user ON subscriptions(user_id);
CREATE TABLE IF NOT EXISTS payments (
  id INTEGER PRIMARY KEY, user_id INTEGER, subscription_id INTEGER, paypal_txn TEXT UNIQUE NOT NULL,
  amount REAL NOT NULL, currency TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind TEXT NOT NULL, amount INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_usage ON usage(user_id, kind, created_at);
CREATE TABLE IF NOT EXISTS webhook_events (id TEXT PRIMARY KEY, type TEXT, created_at REAL NOT NULL);
"""

# Placeholder prices: change them in Admin > Plans.  -1 means unlimited.
DEFAULT_PLANS = [
    dict(slug="free", name="Free", blurb="Try Deer AI", price=0, chat_limit=20, window_hours=5,
         image_limit=8, analyze_limit=10, sort=0),
    dict(slug="pro", name="Pro", blurb="For everyday use", price=9.99, chat_limit=200, window_hours=5,
         image_limit=100, analyze_limit=200, sort=1),
    dict(slug="max", name="Max", blurb="As much as you want", price=24.99, chat_limit=-1, window_hours=5,
         image_limit=-1, analyze_limit=-1, sort=2),
]
KINDS = {"chat": "chat_limit", "imagine": "image_limit", "analyze": "analyze_limit"}
KIND_WORDS = {"chat": "messages", "imagine": "images", "analyze": "image analyses"}
PAID_ACCESS = {"ACTIVE", "APPROVED"}
GRACE = 3 * 86400  # keep paid access if a renewal webhook is late


class LimitReached(Exception):
    def __init__(self, message, resets_at):
        super().__init__(message)
        self.message, self.resets_at = message, resets_at


class BillingError(Exception):
    pass


def init():
    db.conn().executescript(SCHEMA)
    if not db.q1("SELECT id FROM plans"):
        for p in DEFAULT_PLANS:
            db.ex("INSERT INTO plans(slug,name,blurb,price,chat_limit,window_hours,image_limit,analyze_limit,sort,created_at)"
                  " VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (p["slug"], p["name"], p["blurb"], p["price"], p["chat_limit"], p["window_hours"],
                   p["image_limit"], p["analyze_limit"], p["sort"], db.now()))


# ================================================================ PayPal config
_client = {"key": None, "obj": None}
_client_lock = threading.Lock()


def config():
    return {
        "client_id": os.environ.get("PAYPAL_CLIENT_ID") or db.setting("paypal_client_id", ""),
        "secret": os.environ.get("PAYPAL_CLIENT_SECRET") or db.setting("paypal_secret", ""),
        "mode": os.environ.get("PAYPAL_MODE") or db.setting("paypal_mode", "sandbox"),
        "webhook_id": os.environ.get("PAYPAL_WEBHOOK_ID") or db.setting("paypal_webhook_id", ""),
    }


def client():
    c = config()
    if not (c["client_id"] and c["secret"]):
        return None
    key = (c["client_id"], c["secret"], c["mode"])
    with _client_lock:
        if _client["key"] != key:
            _client["key"], _client["obj"] = key, PayPal(*key)
        return _client["obj"]


def require_client():
    pp = client()
    if not pp:
        raise BillingError("PayPal isn't set up yet. The admin needs to add PayPal keys in Admin > Plans.")
    return pp


# ===================================================================== plans
def plans(active_only=True):
    rows = db.q("SELECT * FROM plans " + ("WHERE active=1 " if active_only else "") + "ORDER BY sort, price")
    return rows


def public_plan(p):
    return {k: p[k] for k in ("id", "slug", "name", "blurb", "price", "currency", "chat_limit", "window_hours",
                              "image_limit", "analyze_limit", "paypal_plan_id")} | {"purchasable": bool(p["paypal_plan_id"]) and p["price"] > 0}


def free_plan():
    return db.q1("SELECT * FROM plans WHERE price=0 AND active=1 ORDER BY sort LIMIT 1") or \
        dict(DEFAULT_PLANS[0], id=0, currency="USD", paypal_plan_id=None)


ADMIN_PLAN = dict(id=-1, slug="admin", name="Admin", blurb="Unlimited", price=0, currency="USD",
                  chat_limit=-1, window_hours=5, image_limit=-1, analyze_limit=-1, paypal_plan_id=None)


def sync_plan(plan_id):
    """Create (or re-create after a price change) the matching PayPal billing plan."""
    p = db.q1("SELECT * FROM plans WHERE id=?", (plan_id,))
    if not p:
        raise BillingError("Plan not found.")
    if p["price"] <= 0:
        raise BillingError("Free plans don't need PayPal.")
    pp = require_client()
    stamp = f"{pp.mode}:{p['price']:.2f}:{p['currency']}"
    if p["paypal_plan_id"] and p["synced"] == stamp:
        return p["paypal_plan_id"]
    pkey = f"paypal_product_id_{pp.mode}"
    product = db.setting(pkey)
    if not product:
        product = pp.create_product("Deer AI", "Deer AI subscription plans")["id"]
        db.set_setting(pkey, product)
    new = pp.create_plan(product, f"Deer AI {p['name']}", p["blurb"], float(p["price"]), p["currency"])["id"]
    old = p["paypal_plan_id"]
    if old and p["synced"] and p["synced"].split(":")[0] == pp.mode:
        try:  # existing subscribers keep their old price; new ones get the new plan
            pp.deactivate_plan(old)
        except PayPalError:
            pass
        _remember_old_plan(old, p["id"])
    db.ex("UPDATE plans SET paypal_plan_id=?, synced=? WHERE id=?", (new, stamp, p["id"]))
    return new


def _remember_old_plan(paypal_plan_id, plan_id):
    m = json.loads(db.setting("paypal_old_plans", "{}"))
    m[paypal_plan_id] = plan_id
    db.set_setting("paypal_old_plans", json.dumps(m))


def plan_for_paypal(paypal_plan_id):
    p = db.q1("SELECT * FROM plans WHERE paypal_plan_id=?", (paypal_plan_id,))
    if p:
        return p
    old = json.loads(db.setting("paypal_old_plans", "{}")).get(paypal_plan_id)
    return db.q1("SELECT * FROM plans WHERE id=?", (old,)) if old else None


# ============================================================ entitlement
def _ts(iso):
    if not iso:
        return None
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def has_access(sub, now=None):
    now = now or time.time()
    if sub["status"] in PAID_ACCESS:
        return sub["period_end"] + GRACE > now
    if sub["status"] in ("CANCELLED", "EXPIRED"):
        return sub["period_end"] > now
    return False  # SUSPENDED, APPROVAL_PENDING


def current(user):
    """Returns (plan, subscription_or_None) that applies to this user right now."""
    if user["role"] == "admin":
        test = user.get("test")
        if not test:
            return ADMIN_PLAN, None
        if test.get("plan_id"):  # admin previewing a plan without paying
            plan = db.q1("SELECT * FROM plans WHERE id=?", (test["plan_id"],))
            if plan:
                return plan, None
    subs = db.q("SELECT * FROM subscriptions WHERE user_id=? ORDER BY updated_at DESC", (user["id"],))
    for s in subs:
        if s["status"] in PAID_ACCESS and s["period_end"] < time.time():
            try:  # renewal webhook may have been missed: ask PayPal
                s = refresh(s) or s
            except (PayPalError, BillingError, OSError):
                pass
        if has_access(s):
            plan = db.q1("SELECT * FROM plans WHERE id=?", (s["plan_id"],))
            if plan:
                return plan, s
    return free_plan(), None


def _window(plan, kind):
    return (plan["window_hours"] if kind == "chat" else 24) * 3600


def usage_summary(user):
    plan, _ = current(user)
    out = {}
    for kind, col in KINDS.items():
        win = _window(plan, kind)
        since = time.time() - win
        used = db.q1("SELECT COALESCE(SUM(amount),0) n, MIN(created_at) first FROM usage "
                     "WHERE user_id=? AND kind=? AND created_at>?", (user["id"], kind, since))
        out[kind] = {"used": used["n"], "limit": plan[col], "window_hours": win // 3600,
                     "resets_at": (used["first"] + win) if used["first"] else None}
    return out


def check(user, kind, amount=1):
    plan, _ = current(user)
    limit = plan[KINDS[kind]]
    if limit < 0:
        return
    win = _window(plan, kind)
    rows = db.q("SELECT amount, created_at FROM usage WHERE user_id=? AND kind=? AND created_at>? ORDER BY created_at",
                (user["id"], kind, time.time() - win))
    used = sum(r["amount"] for r in rows)
    if used + amount <= limit:
        return
    # find when enough usage ages out of the window
    need, freed, resets = used + amount - limit, 0, time.time() + win
    for r in rows:
        freed += r["amount"]
        if freed >= need:
            resets = r["created_at"] + win
            break
    left = max(0, limit - used)
    extra = f" You have {left} left." if left else ""
    raise LimitReached(f"You've reached the {plan['name']} plan limit of {limit} {KIND_WORDS[kind]} "
                       f"per {win // 3600} hours.{extra} Upgrade for more, or wait for it to reset.", resets)


def record(user, kind, amount=1):
    if user["role"] == "admin" and not user.get("test"):
        return
    db.ex("INSERT INTO usage(user_id,kind,amount,created_at) VALUES(?,?,?,?)", (user["id"], kind, amount, db.now()))


# ========================================================== subscriptions
def _upsert(user_id, plan_id, pp_sub):
    period_end = _ts((pp_sub.get("billing_info") or {}).get("next_billing_time"))
    status = pp_sub.get("status", "APPROVAL_PENDING")
    row = db.q1("SELECT * FROM subscriptions WHERE paypal_id=?", (pp_sub["id"],))
    if row:
        pe = period_end or row["period_end"]
        db.ex("UPDATE subscriptions SET status=?, period_end=?, updated_at=? WHERE id=?",
              (status, pe, db.now(), row["id"]))
    else:
        pe = period_end or (time.time() + 31 * 86400)
        db.ex("INSERT INTO subscriptions(user_id,plan_id,paypal_id,status,period_end,created_at,updated_at)"
              " VALUES(?,?,?,?,?,?,?)", (user_id, plan_id, pp_sub["id"], status, pe, db.now(), db.now()))
    return db.q1("SELECT * FROM subscriptions WHERE paypal_id=?", (pp_sub["id"],))


def refresh(sub_row):
    pp = require_client()
    data = pp.get_subscription(sub_row["paypal_id"])
    return _upsert(sub_row["user_id"], sub_row["plan_id"], data)


def confirm(user, paypal_sub_id):
    """Called after the PayPal button approves. Never trusts the browser: re-reads from PayPal."""
    pp = require_client()
    data = pp.get_subscription(paypal_sub_id)
    if str(data.get("custom_id")) != str(user["id"]):
        raise BillingError("This subscription belongs to a different account.")
    plan = plan_for_paypal(data.get("plan_id"))
    if not plan:
        raise BillingError("This subscription isn't for a Deer AI plan.")
    if data.get("status") not in PAID_ACCESS:
        raise BillingError(f"PayPal reports this subscription as {data.get('status', 'unknown').lower()}.")
    row = _upsert(user["id"], plan["id"], data)
    # switching plans: stop billing the old subscription
    for old in db.q("SELECT * FROM subscriptions WHERE user_id=? AND id!=? AND status IN ('ACTIVE','APPROVED')",
                    (user["id"], row["id"])):
        try:
            pp.cancel_subscription(old["paypal_id"], "Switched to another Deer AI plan")
        except PayPalError:
            pass
        db.ex("UPDATE subscriptions SET status='CANCELLED', period_end=?, updated_at=? WHERE id=?",
              (time.time(), db.now(), old["id"]))
    return row


def cancel(user):
    _, sub = current(user)
    if not sub or sub["status"] not in PAID_ACCESS:
        raise BillingError("You don't have an active paid plan to cancel.")
    require_client().cancel_subscription(sub["paypal_id"])
    db.ex("UPDATE subscriptions SET status='CANCELLED', updated_at=? WHERE id=?", (db.now(), sub["id"]))
    return db.q1("SELECT * FROM subscriptions WHERE id=?", (sub["id"],))


# ================================================================= webhooks
def handle_webhook(headers, raw):
    cfg = config()
    if not cfg["webhook_id"]:
        raise BillingError("Webhook ID isn't configured.")
    try:
        event = json.loads(raw)
    except ValueError:
        raise BillingError("Invalid JSON.")
    pp = require_client()
    if not pp.verify_webhook(headers, event, cfg["webhook_id"]):
        raise BillingError("Webhook signature check failed.")
    eid, etype = event.get("id"), event.get("event_type", "")
    if not eid or db.q1("SELECT id FROM webhook_events WHERE id=?", (eid,)):
        return "duplicate"
    res = event.get("resource") or {}

    if etype.startswith("BILLING.SUBSCRIPTION."):
        row = db.q1("SELECT * FROM subscriptions WHERE paypal_id=?", (res.get("id"),))
        if row:
            _upsert(row["user_id"], row["plan_id"], res)
        else:  # buyer closed the tab before the site confirmed; create it from PayPal's data
            plan = plan_for_paypal(res.get("plan_id"))
            uid = res.get("custom_id")
            if plan and uid and str(uid).isdigit() and db.q1("SELECT id FROM users WHERE id=?", (int(uid),)):
                _upsert(int(uid), plan["id"], res)

    elif etype == "PAYMENT.SALE.COMPLETED":
        sub = db.q1("SELECT * FROM subscriptions WHERE paypal_id=?", (res.get("billing_agreement_id"),))
        amt = res.get("amount") or {}
        db.ex("INSERT OR IGNORE INTO payments(user_id,subscription_id,paypal_txn,amount,currency,status,created_at)"
              " VALUES(?,?,?,?,?,?,?)", (sub["user_id"] if sub else None, sub["id"] if sub else None, res.get("id"),
                                          float(amt.get("total", 0)), amt.get("currency", "USD"), "COMPLETED",
                                          db.now()))
        if sub:
            try:
                refresh(sub)
            except (PayPalError, OSError):
                pass

    elif etype == "PAYMENT.CAPTURE.COMPLETED":  # API credit top-up (backup for a checkout the site didn't see finish)
        order_id = ((res.get("supplementary_data") or {}).get("related_ids") or {}).get("order_id")
        if order_id and str(res.get("custom_id", "")).startswith("api:"):
            from ..server.devapi import credit_from_order
            try:
                credit_from_order(pp.get_order(order_id))
            except Exception:
                pass

    elif etype in ("PAYMENT.SALE.REFUNDED", "PAYMENT.SALE.REVERSED"):
        txn = res.get("sale_id") or res.get("id")
        db.ex("UPDATE payments SET status=? WHERE paypal_txn=?", (etype.split(".")[-1], txn))

    db.ex("INSERT INTO webhook_events(id,type,created_at) VALUES(?,?,?)", (eid, etype, db.now()))
    return etype
