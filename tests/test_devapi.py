"""Developer API: keys, prices, prepaid credit (PayPal orders), limits, streaming, usage."""
import base64, io, json, os, sys, tempfile, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from fastapi.testclient import TestClient
from deer.server import db
from deer.server import app as APP
from deer import apikeys as K
from deer.billing.paypal import PayPal
from make_demo_data import make_chat, make_shapes, draw_shape

H = lambda t: {"Authorization": f"Bearer {t}"}


class FakePayPal:
    def __init__(self):
        self.orders, self.n = {}, 0

    def __call__(self, method, url, headers, body):
        path = url.split(".paypal.com", 1)[1]
        data = json.loads(body) if body and path != "/v1/oauth2/token" else {}
        if path == "/v1/oauth2/token":
            return 200, b'{"access_token": "t", "expires_in": 3600}'
        if path == "/v2/checkout/orders" and method == "POST":
            self.n += 1
            oid = f"ORDER-{self.n}"
            self.orders[oid] = {"id": oid, "status": "CREATED", "purchase_units": data["purchase_units"]}
            return 201, json.dumps({"id": oid, "status": "CREATED"}).encode()
        if path.startswith("/v2/checkout/orders/"):
            oid = path.split("/")[4]
            o = self.orders.get(oid)
            if not o:
                return 404, b'{"message": "not found"}'
            if path.endswith("/capture"):
                if o["status"] == "COMPLETED":
                    return 422, b'{"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "ORDER_ALREADY_CAPTURED"}]}'
                if o["status"] != "APPROVED":
                    return 422, b'{"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "ORDER_NOT_APPROVED"}]}'
                o["status"] = "COMPLETED"
                u = o["purchase_units"][0]
                u["payments"] = {"captures": [{"id": "CAP-" + oid, "status": "COMPLETED", "amount": u["amount"]}]}
            return 200, json.dumps(o).encode()
        if path == "/v1/notifications/verify-webhook-signature":
            return 200, b'{"verification_status": "SUCCESS"}'
        return 404, b"{}"


fake = FakePayPal()


@pytest.fixture(scope="module")
def c():
    PayPal.http = staticmethod(fake)
    db.DATA_DIR = tempfile.mkdtemp(prefix="deer_devapi_")
    with TestClient(APP.app) as client:
        yield client


def wait(c, A, mid):
    for _ in range(300):
        m = c.get(f"/api/admin/models/{mid}", headers=H(A)).json()
        if m["status"] in ("ready", "error"):
            return m
        time.sleep(1)


def png():
    b = io.BytesIO(); draw_shape("circle", (220, 50, 50)).save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()


def test_developer_api(c):
    A = c.post("/api/auth/register", json={"username": "owner", "password": "ownerowner"}).json()["token"]
    U = c.post("/api/auth/register", json={"username": "dev1", "password": "devdev123"}).json()["token"]
    V = c.post("/api/auth/register", json={"username": "dev2", "password": "devdev456"}).json()["token"]
    tds = c.post("/api/admin/datasets", headers=H(A), data={"name": "t", "kind": "text"},
                 files=[("files", ("c.txt", make_chat(300).encode(), "text/plain"))]).json()["id"]
    ids = c.post("/api/admin/datasets", headers=H(A), data={"name": "s", "kind": "images"},
                 files=[("files", ("s.zip", make_shapes(20), "application/zip"))]).json()["id"]
    mids = {}
    for kind, ds in (("chat", tds), ("vision", ids), ("canvas", ids)):
        mids[kind] = c.post("/api/admin/models", headers=H(A), json={"name": f"deer-{kind}", "kind": kind, "dataset_id": ds,
                            "preset": "tiny", "config": {"eval_interval": 10}}).json()["id"]
        c.post(f"/api/admin/models/{mids[kind]}/train", headers=H(A), json={"steps": 10})
        assert wait(c, A, mids[kind])["status"] == "ready"
    c.patch(f"/api/admin/models/{mids['chat']}", headers=H(A), json={"published": True})
    c.patch(f"/api/admin/models/{mids['canvas']}", headers=H(A), json={"published": True})
    c.patch("/api/admin/api", headers=H(A), json={"welcome_credit": 0.001})

    # ---- keys: shown once, then only a prefix
    k = c.post("/api/dev/keys", headers=H(U), json={"name": "game server"}).json()
    raw = k["key"]
    assert raw.startswith("dk_") and k["prefix"].endswith("…")
    assert raw not in json.dumps(c.get("/api/dev", headers=H(U)).json())
    home = c.get("/api/dev", headers=H(U)).json()
    assert home["balance"] == 0.001 and home["ledger"][0]["kind"] == "welcome"
    KH = lambda key: {"Authorization": f"Bearer {key}"}

    # ---- auth errors
    assert c.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 401
    assert c.post("/v1/chat", headers=KH("dk_nope"), json={"messages": [{"role": "user", "content": "hi"}]}).json()["error"]["type"] == "authentication_error"

    # ---- models: users see published only; the owner's keys see everything
    names = {m["name"] for m in c.get("/v1/models", headers=KH(raw)).json()["models"]}
    assert names == {"deer-chat", "deer-canvas"}
    ak = c.post("/api/dev/keys", headers=H(A), json={"name": "owner key"}).json()["key"]
    assert {m["name"] for m in c.get("/v1/models", headers=KH(ak)).json()["models"]} == {"deer-chat", "deer-vision", "deer-canvas"}

    # ---- not enough credit: refused before any work is done
    r = c.post("/v1/chat", headers=KH(raw), json={"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 1500})
    assert r.status_code == 402 and r.json()["error"]["type"] == "insufficient_credit"
    # the owner never pays
    r = c.post("/v1/analyze", headers=KH(ak), json={"image": png()}).json()
    assert r["cost"]["charged"] is False and r["cost"]["usd"] == 0.001 and len(r["predictions"]) == 4

    # ---- top up with PayPal (not configured yet -> clear error)
    assert c.post("/api/dev/credit/order", headers=H(U), json={"amount": 10}).status_code == 400
    c.put("/api/admin/billing/paypal", headers=H(A), json={"client_id": "cid", "secret": "sec-secret-1", "mode": "sandbox", "webhook_id": "WH"})
    assert c.post("/api/dev/credit/order", headers=H(U), json={"amount": 2}).status_code == 400  # under the minimum
    oid = c.post("/api/dev/credit/order", headers=H(U), json={"amount": 10}).json()["id"]
    assert c.post("/api/dev/credit/capture", headers=H(U), json={"order_id": oid}).status_code == 502  # not approved yet
    fake.orders[oid]["status"] = "APPROVED"  # buyer approves in the PayPal window
    assert c.post("/api/dev/credit/capture", headers=H(V), json={"order_id": oid}).status_code in (403, 502)
    fake.orders[oid]["status"] = "APPROVED"
    r = c.post("/api/dev/credit/capture", headers=H(U), json={"order_id": oid}).json()
    assert r["added"] == 10 and r["balance"] == pytest.approx(10.001)
    r = c.post("/api/dev/credit/capture", headers=H(U), json={"order_id": oid}).json()  # double click: no double credit
    assert r["added"] == 0 and r["balance"] == pytest.approx(10.001)
    ev = {"id": "EV-1", "event_type": "PAYMENT.CAPTURE.COMPLETED",
          "resource": {"id": "CAP-" + oid, "custom_id": f"api:{r and 2}", "supplementary_data": {"related_ids": {"order_id": oid}}}}
    hdr = {"paypal-auth-algo": "a", "paypal-cert-url": "c", "paypal-transmission-id": "t", "paypal-transmission-sig": "s",
           "paypal-transmission-time": "n"}
    c.post("/api/paypal/webhook", content=json.dumps(ev), headers=hdr)
    assert c.get("/api/dev", headers=H(U)).json()["balance"] == pytest.approx(10.001)

    # ---- chat is billed per token, exactly
    before = K.balance(2)
    r = c.post("/v1/chat", headers=KH(raw), json={"messages": [{"role": "user", "content": "Hello!"}]}).json()
    pr = K.prices()
    expect = round((r["usage"]["input_tokens"] * pr["chat_in_per_1k"] + r["usage"]["output_tokens"] * pr["chat_out_per_1k"]) / 1000, 6)
    assert r["cost"]["usd"] == pytest.approx(expect, abs=1e-6) and r["cost"]["charged"]
    assert r["usage"]["input_tokens"] == len(b"User: Hello!\nDeer:") and r["reply"] is not None
    assert K.balance(2) == pytest.approx(before - r["cost"]["usd"], abs=1e-6)
    # conversation history counts as input
    r2 = c.post("/v1/chat", headers=KH(raw), json={"messages": [{"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "Who made you?"}]}).json()
    assert r2["usage"]["input_tokens"] > r["usage"]["input_tokens"]
    assert c.post("/v1/chat", headers=KH(raw), json={"messages": [{"role": "assistant", "content": "x"}]}).status_code == 400
    # streaming
    s = c.post("/v1/chat", headers=KH(raw), json={"messages": [{"role": "user", "content": "Hi"}], "stream": True})
    evs = [json.loads(l[6:]) for l in s.text.splitlines() if l.startswith("data: ")]
    assert evs[0]["type"] == "start" and evs[-1]["type"] == "done" and evs[-1]["cost"]["charged"]
    # images
    r = c.post("/v1/imagine", headers=KH(raw), json={"prompt": "red circle", "count": 3}).json()
    assert len(r["images"]) == 3 and r["cost"]["usd"] == pytest.approx(0.006) and base64.b64decode(r["images"][0]["png_base64"])[:4] == b"\x89PNG"
    assert c.post("/v1/analyze", headers=KH(raw), json={"image": png()}).status_code == 503  # vision not published
    assert c.post("/v1/analyze", headers=KH(raw), json={"image": png(), "model": "deer-vision"}).status_code == 404

    # ---- prices are the admin's to change
    c.patch("/api/admin/api", headers=H(A), json={"image": 0.01})
    assert c.post("/v1/imagine", headers=KH(raw), json={"prompt": "x", "count": 2}).json()["cost"]["usd"] == pytest.approx(0.02)

    # ---- per-key limits: permissions, requests per minute, monthly spending cap, revoke
    kid = k["id"]
    c.patch(f"/api/dev/keys/{kid}", headers=H(U), json={"scopes": ["chat"]})
    assert c.post("/v1/imagine", headers=KH(raw), json={"prompt": "x"}).json()["error"]["type"] == "permission_error"
    k2 = c.post("/api/dev/keys", headers=H(U), json={"name": "slow", "rate_per_min": 2}).json()
    for _ in range(2):
        assert c.get("/v1/usage", headers=KH(k2["key"])).status_code == 200
    r = c.get("/v1/usage", headers=KH(k2["key"]))
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    k3 = c.post("/api/dev/keys", headers=H(U), json={"name": "capped", "monthly_cap": 0.001}).json()
    r = c.post("/v1/imagine", headers=KH(k3["key"]), json={"prompt": "x"})
    assert r.status_code == 402 and r.json()["error"]["type"] == "spending_limit"
    c.patch(f"/api/dev/keys/{kid}", headers=H(U), json={"revoke": True})
    assert c.post("/v1/chat", headers=KH(raw), json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 401
    assert c.patch(f"/api/dev/keys/{kid}", headers=H(V), json={"revoke": True}).status_code == 404  # not their key

    # ---- usage reports
    home = c.get("/api/dev", headers=H(U)).json()
    assert home["usage"]["total"]["requests"] == 5 and {e["endpoint"] for e in home["usage"]["by_endpoint"]} >= {"chat", "imagine"}
    assert home["spent_this_month"] > 0 and home["usage"]["recent"][0]["key_name"]
    adm = c.get("/api/admin/api", headers=H(A)).json()
    assert adm["topups_30d"]["usd"] == 10 and any(u["username"] == "dev1" and u["spent"] > 0 for u in adm["users"])
    # grants and switching the API off
    c.post("/api/admin/api/grant", headers=H(A), json={"username": "dev2", "amount": 2.5, "note": "beta tester"})
    assert c.get("/api/dev", headers=H(V)).json()["balance"] == 2.5
    c.patch("/api/admin/api", headers=H(A), json={"enabled": False})
    assert c.get("/v1/usage", headers=KH(ak)).status_code == 503
    c.patch("/api/admin/api", headers=H(A), json={"enabled": True})
