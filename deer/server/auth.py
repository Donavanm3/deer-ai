"""Request authentication dependencies."""
import json
from typing import Optional
from fastapi import Depends, Header, HTTPException
from . import db


def current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Sign in to continue.")
    tok = authorization[7:]
    r = db.q1("SELECT u.id,u.username,u.role,u.test_mode,u.train_ok FROM sessions s JOIN users u ON u.id=s.user_id "
              "WHERE s.token=? AND s.expires_at>?", (tok, db.now()))
    if not r:
        raise HTTPException(401, "Your session expired. Sign in again.")
    raw, tm = r.pop("test_mode", ""), {}
    if r["role"] == "admin" and raw:
        try:
            tm = json.loads(raw) or {}
        except ValueError:
            tm = {}
    r["test"] = tm if tm.get("on") else None
    return r


def acts_as_admin(user):
    """Admins in test mode see and are limited exactly like regular users."""
    return user["role"] == "admin" and not user.get("test")


def admin_user(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admins only.")
    return user
