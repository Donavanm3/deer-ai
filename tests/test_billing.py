"""Billing tests against a fake PayPal API (no network): plan sync, subscribe, limits, webhooks, cancel."""
import json, os, sys, tempfile, time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deer.server import db

from fastapi.testclient import TestClient
from deer.server.app import app
from deer import billing as B
from deer.billing.paypal import PayPal


class FakePayPal:
    def __init__(self):
        self.plans, self.subs, self.n = {}, {}, 0

    def __call__(self, method, url, headers, body):
        path = url.split(".paypal.com", 1)[1]
        data = json.loads(body) if body and path != "/v1/oauth2/token" else {}
        if path == "/v1/oauth2/token":
            return 200, json.dumps({"access_token": "tok", "expires_in": 3600}).encode()
        if path == "/v1/catalogs/products":
            return 201, b'{"id": "PROD-1"}'
        if path == "/v1/billing/plans":
            self.n += 1
            pid = f"P-{self.n}"
            self.plans[pid] = data
            return 201, json.dumps({"id": pid}).encode()
        if path.endswith("/deactivate"):
            return 204, b""
        if path.startswith("/v1/billing/subscriptions/"):
            sid = path.split("/")[4]
            if sid not in self.subs:
                return 404, b'{"message": "not found"}'
            if path.endswith("/cancel"):
                self.subs[sid]["status"] = "CANCELLED"
                return 204, b""
            return 200, json.dumps(self.subs[sid]).encode()
        if path == "/v1/notifications/verify-webhook-signature":
            ok = data["transmission_sig"] == "good"
            return 200, json.dumps({"verification_status": "SUCCESS" if ok else "FAILURE"}).encode()
        return 404, b"{}"

    def approve(self, plan_id, user_id, days=30):
        self.n += 1
        sid = f"I-{self.n}"
        nxt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + days * 86400))
        self.subs[sid] = {"id": sid, "plan_id": plan_id, "custom_id": str(user_id), "status": "ACTIVE",
                          "billing_info": {"next_billing_time": nxt}}
        return sid


fake = FakePayPal()
PayPal.http = staticmethod(fake)
H = lambda t: {"Authorization": f"Bearer {t}"}
WH = {"paypal-auth-algo": "a", "paypal-cert-url": "c", "paypal-transmission-id": "t",
      "paypal-transmission-sig": "good", "paypal-transmission-time": "now"}


@pytest.fixture(scope="module")
def c():
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_bill_")
    with TestClient(app) as client:
        yield client


def test_billing_flow(c):
    A = c.post("/api/auth/register", json={"username": "boss", "password": "bossboss1"}).json()["token"]
    ur = c.post("/api/auth/register", json={"username": "buyer", "password": "buyerbuyer"}).json()
    U, uid = ur["token"], ur["user"]["id"]

    b = c.get("/api/billing", headers=H(U)).json()
    assert b["plan"]["slug"] == "free" and b["paypal"] is None
    assert [p["slug"] for p in b["plans"]] == ["free", "pro", "max"]
    assert not any(p["purchasable"] for p in b["plans"])

    # can't sync without keys
    pro = next(p for p in b["plans"] if p["slug"] == "pro")
    assert c.post(f"/api/admin/plans/{pro['id']}/sync", headers=H(A)).status_code == 400
    r = c.put("/api/admin/billing/paypal", headers=H(A),
              json={"client_id": "cid", "secret": "zzTOPSECRETzz", "mode": "sandbox", "webhook_id": "WH-1"})
    assert r.status_code == 200 and r.json()["paypal"]["configured"] and "zzTOPSECRETzz" not in json.dumps(r.json())
    r = c.post(f"/api/admin/plans/{pro['id']}/sync", headers=H(A)).json()
    pp_pro = next(p for p in r["plans"] if p["slug"] == "pro")["paypal_plan_id"]
    assert pp_pro == "P-1" and fake.plans["P-1"]["billing_cycles"][0]["pricing_scheme"]["fixed_price"]["value"] == "9.99"
    max_id = next(p["id"] for p in r["plans"] if p["slug"] == "max")
    pp_max = next(p for p in c.post(f"/api/admin/plans/{max_id}/sync", headers=H(A)).json()["plans"]
                  if p["slug"] == "max")["paypal_plan_id"]

    # free limits: tighten to 2 analyses/day and test enforcement at the billing layer
    free = next(p for p in B.plans() if p["slug"] == "free")
    body = {k: free[k] for k in ("name", "blurb", "price", "currency", "chat_limit", "window_hours", "image_limit",
                                  "analyze_limit", "sort")} | {"analyze_limit": 2, "active": True}
    assert c.patch(f"/api/admin/plans/{free['id']}", headers=H(A), json=body).status_code == 200
    user = {"id": uid, "role": "user"}
    B.record(user, "analyze"); B.record(user, "analyze")
    with pytest.raises(B.LimitReached) as e:
        B.check(user, "analyze")
    assert e.value.resets_at > time.time() and "Free plan" in e.value.message
    B.check({"id": 1, "role": "admin"}, "analyze")  # admins unlimited
    with pytest.raises(B.LimitReached):
        B.check(user, "imagine", 9)  # 9 images > 8 free

    # 429 surfaces through the real endpoint with reset info
    r = c.post("/api/analyze", headers=H(U), files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 429 and r.json()["detail"]["limit"] is True

    # subscribe: wrong user rejected, right user accepted
    other = fake.approve(pp_pro, 999)
    assert c.post("/api/billing/confirm", headers=H(U), json={"subscription_id": other}).status_code == 400
    sid = fake.approve(pp_pro, uid)
    b = c.post("/api/billing/confirm", headers=H(U), json={"subscription_id": sid}).json()
    assert b["plan"]["slug"] == "pro" and b["subscription"]["status"] == "ACTIVE"
    B.check(user, "analyze")  # pro allows more

    # upgrade to Max cancels the Pro subscription
    sid2 = fake.approve(pp_max, uid)
    b = c.post("/api/billing/confirm", headers=H(U), json={"subscription_id": sid2}).json()
    assert b["plan"]["slug"] == "max" and fake.subs[sid]["status"] == "CANCELLED"
    assert b["usage"]["chat"]["limit"] == -1

    # webhooks: bad signature rejected, payment recorded, duplicate ignored
    ev = {"id": "WH-EV-1", "event_type": "PAYMENT.SALE.COMPLETED",
          "resource": {"id": "SALE-1", "billing_agreement_id": sid2, "amount": {"total": "24.99", "currency": "USD"}}}
    assert c.post("/api/paypal/webhook", content=json.dumps(ev), headers=WH | {"paypal-transmission-sig": "bad"}).status_code == 400
    assert c.post("/api/paypal/webhook", content=json.dumps(ev), headers=WH).json()["handled"] == "PAYMENT.SALE.COMPLETED"
    assert c.post("/api/paypal/webhook", content=json.dumps(ev), headers=WH).json()["handled"] == "duplicate"
    adm = c.get("/api/admin/billing", headers=H(A)).json()
    assert adm["revenue_30d"][0]["total"] == pytest.approx(24.99) and adm["active_paid"] == 1

    # webhook for a subscription the site never saw (buyer closed the tab)
    u2 = c.post("/api/auth/register", json={"username": "tabcloser", "password": "tabtabtab1"}).json()
    sid3 = fake.approve(pp_pro, u2["user"]["id"])
    ev2 = {"id": "WH-EV-2", "event_type": "BILLING.SUBSCRIPTION.ACTIVATED", "resource": fake.subs[sid3]}
    c.post("/api/paypal/webhook", content=json.dumps(ev2), headers=WH)
    assert c.get("/api/billing", headers=H(u2["token"])).json()["plan"]["slug"] == "pro"

    # suspension (failed payment) removes access immediately
    fake.subs[sid3]["status"] = "SUSPENDED"
    ev3 = {"id": "WH-EV-3", "event_type": "BILLING.SUBSCRIPTION.SUSPENDED", "resource": fake.subs[sid3]}
    c.post("/api/paypal/webhook", content=json.dumps(ev3), headers=WH)
    assert c.get("/api/billing", headers=H(u2["token"])).json()["plan"]["slug"] == "free"

    # cancel keeps access until the paid period ends
    b = c.post("/api/billing/cancel", headers=H(U)).json()
    assert b["plan"]["slug"] == "max" and b["subscription"]["status"] == "CANCELLED"
    db.ex("UPDATE subscriptions SET period_end=? WHERE paypal_id=?", (time.time() - 1, sid2))
    assert c.get("/api/billing", headers=H(U)).json()["plan"]["slug"] == "free"

    # price change creates a new PayPal plan; old subscribers still map to Pro
    pro_row = next(p for p in B.plans() if p["slug"] == "pro")
    body = {k: pro_row[k] for k in ("name", "blurb", "currency", "chat_limit", "window_hours", "image_limit",
                                     "analyze_limit", "sort")} | {"price": 12.0, "active": True}
    c.patch(f"/api/admin/plans/{pro_row['id']}", headers=H(A), json=body)
    new_pp = next(p for p in c.post(f"/api/admin/plans/{pro_row['id']}/sync", headers=H(A)).json()["plans"]
                  if p["slug"] == "pro")["paypal_plan_id"]
    assert new_pp != pp_pro and B.plan_for_paypal(pp_pro)["slug"] == "pro"

    # the last free plan can't be switched off
    free = next(p for p in B.plans() if p["slug"] == "free")
    body = {k: free[k] for k in ("name", "blurb", "price", "currency", "chat_limit", "window_hours", "image_limit",
                                  "analyze_limit", "sort")} | {"active": False}
    assert c.patch(f"/api/admin/plans/{free['id']}", headers=H(A), json=body).status_code == 400
