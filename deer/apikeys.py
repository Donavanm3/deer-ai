"""
Developer API: API keys, prices, prepaid credit, usage and limits.

Money is kept in micro-dollars (1,000,000 = $1.00) so tiny per-request costs add up exactly.
Deer AI's chat models read and write one byte at a time, so for the API 1 token = 1 byte of text.
"""
import hashlib
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from .server import db

MICRO = 1_000_000
SCOPES = ("chat", "analyze", "imagine")

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL, prefix TEXT NOT NULL, hash TEXT UNIQUE NOT NULL, scopes TEXT NOT NULL,
  rate_per_min INTEGER NOT NULL DEFAULT 60, monthly_cap INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, last_used_at REAL, revoked_at REAL);
CREATE INDEX IF NOT EXISTS ix_keys_user ON api_keys(user_id);
CREATE TABLE IF NOT EXISTS api_usage (
  id INTEGER PRIMARY KEY, key_id INTEGER, user_id INTEGER NOT NULL, endpoint TEXT NOT NULL,
  model_id INTEGER, model_name TEXT, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
  images INTEGER NOT NULL DEFAULT 0, cost INTEGER NOT NULL DEFAULT 0, charged INTEGER NOT NULL DEFAULT 1,
  ms INTEGER NOT NULL DEFAULT 0, status INTEGER NOT NULL DEFAULT 200, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_usage_user ON api_usage(user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_usage_key ON api_usage(key_id, created_at);
CREATE TABLE IF NOT EXISTS credit_ledger (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount INTEGER NOT NULL, kind TEXT NOT NULL, ref TEXT UNIQUE, note TEXT, created_at REAL NOT NULL);
"""

# Starting prices (the admin changes them in Admin > API). Chat is per 1,000 tokens.
DEFAULTS = {
    "api_enabled": "1",
    "api_price_chat_in": "0.0005",     # $ per 1K input tokens
    "api_price_chat_out": "0.0015",    # $ per 1K output tokens
    "api_price_image": "0.002",        # $ per picture created
    "api_price_analyze": "0.001",      # $ per picture analyzed
    "api_welcome_credit": "0",         # $ given with a user's first key
    "api_packs": "5,10,25,50",         # $ amounts offered for top-up
    "api_min_topup": "5",
}


class ApiError(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def init():
    db.conn().executescript(SCHEMA)
    for k, v in DEFAULTS.items():
        db.ex("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))


# ==================================================================== prices
def prices():
    g = lambda k: float(db.setting(k, DEFAULTS[k]))
    return {"chat_in_per_1k": g("api_price_chat_in"), "chat_out_per_1k": g("api_price_chat_out"),
            "image": g("api_price_image"), "analyze": g("api_price_analyze"), "currency": "USD"}


def set_prices(p):
    for key, name in (("chat_in_per_1k", "api_price_chat_in"), ("chat_out_per_1k", "api_price_chat_out"),
                      ("image", "api_price_image"), ("analyze", "api_price_analyze")):
        if key in p and p[key] is not None:
            v = float(p[key])
            if not 0 <= v <= 100:
                raise ApiError(400, "Prices must be between $0 and $100.")
            db.set_setting(name, repr(v))


def cost_chat(inp, out, pr=None):
    pr = pr or prices()
    return round((inp * pr["chat_in_per_1k"] + out * pr["chat_out_per_1k"]) / 1000 * MICRO)


def cost_images(n, kind, pr=None):
    pr = pr or prices()
    return round(n * pr["image" if kind == "imagine" else "analyze"] * MICRO)


def dollars(micros):
    return round(micros / MICRO, 6)


# ====================================================================== keys
def _hash(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


def create_key(user, name, scopes=SCOPES, rate_per_min=60, monthly_cap_dollars=0.0):
    scopes = [s for s in scopes if s in SCOPES]
    if not scopes:
        raise ApiError(400, "Pick at least one thing the key may do.")
    if db.q1("SELECT COUNT(*) n FROM api_keys WHERE user_id=? AND revoked_at IS NULL", (user["id"],))["n"] >= 25:
        raise ApiError(400, "You can have up to 25 active keys. Revoke one first.")
    first = not db.q1("SELECT id FROM api_keys WHERE user_id=?", (user["id"],))
    raw = "dk_" + secrets.token_urlsafe(32)
    kid = db.ex("INSERT INTO api_keys(user_id,name,prefix,hash,scopes,rate_per_min,monthly_cap,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)", (user["id"], (name or "My key").strip()[:60], raw[:10], _hash(raw),
                                            json.dumps(scopes), max(1, min(int(rate_per_min), 600)),
                                            max(0, round(float(monthly_cap_dollars or 0) * MICRO)), db.now()))
    welcome = float(db.setting("api_welcome_credit", "0"))
    if first and welcome > 0 and user["role"] != "admin":
        add_credit(user["id"], round(welcome * MICRO), "welcome", f"welcome:{user['id']}", "Welcome credit")
    return {"key": raw, **public_key(db.q1("SELECT * FROM api_keys WHERE id=?", (kid,)))}


def public_key(k):
    month = month_spend(key_id=k["id"])
    return {"id": k["id"], "name": k["name"], "prefix": k["prefix"] + "…", "scopes": json.loads(k["scopes"]),
            "rate_per_min": k["rate_per_min"], "monthly_cap": dollars(k["monthly_cap"]),
            "spent_this_month": dollars(month), "created_at": k["created_at"], "last_used_at": k["last_used_at"],
            "revoked": k["revoked_at"] is not None}


def list_keys(user_id):
    return [public_key(k) for k in db.q("SELECT * FROM api_keys WHERE user_id=? ORDER BY revoked_at IS NOT NULL, id DESC",
                                        (user_id,))]


def update_key(user_id, kid, **changes):
    k = db.q1("SELECT * FROM api_keys WHERE id=? AND user_id=?", (kid, user_id))
    if not k:
        raise ApiError(404, "Key not found.")
    if changes.get("revoke"):
        db.ex("UPDATE api_keys SET revoked_at=? WHERE id=?", (db.now(), kid))
    if changes.get("name"):
        db.ex("UPDATE api_keys SET name=? WHERE id=?", (changes["name"].strip()[:60], kid))
    if changes.get("rate_per_min") is not None:
        db.ex("UPDATE api_keys SET rate_per_min=? WHERE id=?", (max(1, min(int(changes["rate_per_min"]), 600)), kid))
    if changes.get("monthly_cap") is not None:
        db.ex("UPDATE api_keys SET monthly_cap=? WHERE id=?", (max(0, round(float(changes["monthly_cap"]) * MICRO)), kid))
    if changes.get("scopes") is not None:
        sc = [s for s in changes["scopes"] if s in SCOPES]
        if not sc:
            raise ApiError(400, "Pick at least one thing the key may do.")
        db.ex("UPDATE api_keys SET scopes=? WHERE id=?", (json.dumps(sc), kid))
    return public_key(db.q1("SELECT * FROM api_keys WHERE id=?", (kid,)))


# ============================================================ authentication
_rate = defaultdict(deque)
_rate_lock = threading.Lock()


def authenticate(header_auth, header_key, scope):
    raw = header_key or ""
    if not raw and header_auth and header_auth.startswith("Bearer "):
        raw = header_auth[7:]
    if not raw.startswith("dk_"):
        raise ApiError(401, "Missing API key. Send it as 'Authorization: Bearer dk_...'.", type="authentication_error")
    if db.setting("api_enabled", "1") != "1":
        raise ApiError(503, "The Deer AI API is switched off right now.", type="unavailable")
    k = db.q1("SELECT k.*, u.role, u.username, u.api_balance FROM api_keys k JOIN users u ON u.id=k.user_id WHERE k.hash=?",
              (_hash(raw),))
    if not k or k["revoked_at"] is not None:
        raise ApiError(401, "This API key isn't valid or was revoked.", type="authentication_error")
    if scope and scope not in json.loads(k["scopes"]):
        raise ApiError(403, f"This key isn't allowed to use '{scope}'. Change its permissions on the API page.",
                       type="permission_error")
    now = time.time()
    with _rate_lock:  # requests per minute, per key
        q = _rate[k["id"]]
        while q and q[0] < now - 60:
            q.popleft()
        if len(q) >= k["rate_per_min"]:
            raise ApiError(429, f"Too many requests: this key allows {k['rate_per_min']} per minute.",
                           type="rate_limit_error", retry_after=max(1, int(q[0] + 60 - now) + 1))
        q.append(now)
    db.ex("UPDATE api_keys SET last_used_at=? WHERE id=?", (now, k["id"]))
    return k


def free(key):
    return key["role"] == "admin"  # the owner's own keys are never charged


def ensure_funds(key, max_cost):
    """Refuse a request up front if it could cost more than the account has, or push a key past its cap."""
    if free(key):
        return
    bal = db.q1("SELECT api_balance FROM users WHERE id=?", (key["user_id"],))["api_balance"]
    if bal <= 0 or bal < max_cost:
        raise ApiError(402, f"Not enough API credit (balance ${dollars(bal):.4f}). Add credit on the API page.",
                       type="insufficient_credit", balance=dollars(bal))
    if key["monthly_cap"] and month_spend(key_id=key["id"]) + max_cost > key["monthly_cap"]:
        raise ApiError(402, f"This key reached its monthly spending limit of ${dollars(key['monthly_cap']):.2f}.",
                       type="spending_limit")


def record(key, endpoint, model_id, model_name, inp=0, out=0, images=0, cost=0, ms=0, status=200):
    charged = 0 if free(key) else 1
    db.ex("INSERT INTO api_usage(key_id,user_id,endpoint,model_id,model_name,input_tokens,output_tokens,images,cost,charged,ms,status,created_at) "
          "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (key["id"], key["user_id"], endpoint, model_id, model_name, inp, out, images,
                                                cost, charged, ms, status, db.now()))
    if charged and cost:
        db.ex("UPDATE users SET api_balance = api_balance - ? WHERE id=?", (cost, key["user_id"]))
    return {"cost": dollars(cost), "charged": bool(charged),
            "balance": dollars(db.q1("SELECT api_balance FROM users WHERE id=?", (key["user_id"],))["api_balance"])}


# ==================================================================== credit
def add_credit(user_id, micros, kind, ref, note=""):
    """Idempotent: the same ref (PayPal order, grant id...) is only ever credited once."""
    try:
        db.ex("INSERT INTO credit_ledger(user_id,amount,kind,ref,note,created_at) VALUES(?,?,?,?,?,?)",
              (user_id, micros, kind, ref, note, db.now()))
    except Exception:  # sqlite3.IntegrityError: already credited
        return False
    db.ex("UPDATE users SET api_balance = api_balance + ? WHERE id=?", (micros, user_id))
    return True


def balance(user_id):
    return dollars(db.q1("SELECT api_balance FROM users WHERE id=?", (user_id,))["api_balance"])


def month_start():
    n = datetime.now(timezone.utc)
    return datetime(n.year, n.month, 1, tzinfo=timezone.utc).timestamp()


def month_spend(key_id=None, user_id=None):
    col, val = ("key_id", key_id) if key_id is not None else ("user_id", user_id)
    return db.q1(f"SELECT COALESCE(SUM(cost),0) c FROM api_usage WHERE {col}=? AND charged=1 AND created_at>=?",
                 (val, month_start()))["c"]


# ===================================================================== usage
def usage_report(user_id=None, days=30):
    since = time.time() - days * 86400
    where, args = ("user_id=? AND ", [user_id]) if user_id is not None else ("", [])
    daily = db.q(f"SELECT date(created_at,'unixepoch') day, COUNT(*) requests, SUM(cost) cost, SUM(input_tokens) inp, "
                 f"SUM(output_tokens) outp, SUM(images) images FROM api_usage WHERE {where}created_at>=? GROUP BY day ORDER BY day",
                 (*args, since))
    by_ep = db.q(f"SELECT endpoint, COUNT(*) requests, SUM(cost) cost FROM api_usage WHERE {where}created_at>=? "
                 f"GROUP BY endpoint ORDER BY cost DESC", (*args, since))
    recent = db.q(f"SELECT a.*, k.name key_name FROM api_usage a LEFT JOIN api_keys k ON k.id=a.key_id WHERE "
                  f"{where.replace('user_id', 'a.user_id')}a.created_at>=? ORDER BY a.id DESC LIMIT 50", (*args, since))
    for r in daily + by_ep:
        r["cost"] = dollars(r["cost"] or 0)
    for r in recent:
        r["cost"] = dollars(r["cost"])
    total = db.q1(f"SELECT COUNT(*) requests, COALESCE(SUM(cost),0) cost, COALESCE(SUM(input_tokens),0) inp, "
                  f"COALESCE(SUM(output_tokens),0) outp, COALESCE(SUM(images),0) images FROM api_usage WHERE {where}created_at>=?",
                  (*args, since))
    total["cost"] = dollars(total["cost"])
    return {"days": days, "daily": daily, "by_endpoint": by_ep, "recent": recent, "total": total}


def ledger(user_id, limit=50):
    rows = db.q("SELECT amount, kind, note, created_at FROM credit_ledger WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit))
    for r in rows:
        r["amount"] = dollars(r["amount"])
    return rows


def packs():
    out = []
    for p in db.setting("api_packs", DEFAULTS["api_packs"]).split(","):
        try:
            v = float(p)
            if 1 <= v <= 1000:
                out.append(v)
        except ValueError:
            pass
    return out or [5.0, 10.0, 25.0]
