"""
Minimal PayPal REST client for subscriptions, using only the Python standard library.
Docs: https://developer.paypal.com/docs/api/subscriptions/v1/
"""
import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid

BASES = {"sandbox": "https://api-m.sandbox.paypal.com", "live": "https://api-m.paypal.com"}


class PayPalError(Exception):
    def __init__(self, status, message, body=None):
        super().__init__(f"PayPal {status}: {message}")
        self.status, self.body = status, body


def _urllib_http(method, url, headers, body, timeout=20):
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class PayPal:
    http = staticmethod(_urllib_http)  # replaced in tests

    def __init__(self, client_id, secret, mode="sandbox"):
        if mode not in BASES:
            raise ValueError("PayPal mode must be sandbox or live")
        self.client_id, self.secret, self.mode = client_id, secret, mode
        self.base = BASES[mode]
        self._token, self._exp = None, 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ plumbing
    def token(self):
        with self._lock:
            if self._token and time.time() < self._exp - 60:
                return self._token
            auth = base64.b64encode(f"{self.client_id}:{self.secret}".encode()).decode()
            status, raw = self.http("POST", self.base + "/v1/oauth2/token",
                                    {"Authorization": f"Basic {auth}",
                                     "Content-Type": "application/x-www-form-urlencoded"},
                                    b"grant_type=client_credentials")
            data = _json(raw)
            if status != 200:
                raise PayPalError(status, data.get("error_description") or "could not sign in to PayPal", data)
            self._token, self._exp = data["access_token"], time.time() + int(data.get("expires_in", 3000))
            return self._token

    def call(self, method, path, body=None, idempotent=False):
        headers = {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json",
                   "Prefer": "return=representation"}
        if idempotent:
            headers["PayPal-Request-Id"] = str(uuid.uuid4())
        status, raw = self.http(method, self.base + path, headers,
                                json.dumps(body).encode() if body is not None else None)
        data = _json(raw)
        if status >= 300:
            msg = data.get("message") or data.get("error_description") or "request failed"
            details = data.get("details") or []
            if details:
                msg += " (" + "; ".join(d.get("description", d.get("issue", "")) for d in details) + ")"
            raise PayPalError(status, msg, data)
        return data

    # ----------------------------------------------------------- catalog
    def create_product(self, name, description):
        return self.call("POST", "/v1/catalogs/products",
                         {"name": name, "description": description[:256], "type": "SERVICE",
                          "category": "SOFTWARE"}, idempotent=True)

    def create_plan(self, product_id, name, description, price, currency, interval="MONTH"):
        return self.call("POST", "/v1/billing/plans", {
            "product_id": product_id, "name": name, "description": description[:127] or name,
            "status": "ACTIVE",
            "billing_cycles": [{
                "frequency": {"interval_unit": interval, "interval_count": 1},
                "tenure_type": "REGULAR", "sequence": 1, "total_cycles": 0,
                "pricing_scheme": {"fixed_price": {"value": f"{price:.2f}", "currency_code": currency}},
            }],
            "payment_preferences": {"auto_bill_outstanding": True, "setup_fee_failure_action": "CONTINUE",
                                    "payment_failure_threshold": 2},
        }, idempotent=True)

    def deactivate_plan(self, plan_id):
        return self.call("POST", f"/v1/billing/plans/{plan_id}/deactivate")

    # ------------------------------------------------------ subscriptions
    def get_subscription(self, sub_id):
        return self.call("GET", f"/v1/billing/subscriptions/{_safe_id(sub_id)}")

    def cancel_subscription(self, sub_id, reason="Cancelled by customer"):
        return self.call("POST", f"/v1/billing/subscriptions/{_safe_id(sub_id)}/cancel", {"reason": reason[:127]})

    # ---------------------------------------------------- one-time orders
    def create_order(self, amount, currency, custom_id, description):
        return self.call("POST", "/v2/checkout/orders", {
            "intent": "CAPTURE",
            "purchase_units": [{"amount": {"currency_code": currency, "value": f"{amount:.2f}"},
                                "custom_id": str(custom_id), "description": description[:127]}],
        }, idempotent=True)

    def capture_order(self, order_id):
        return self.call("POST", f"/v2/checkout/orders/{_safe_id(order_id)}/capture", {}, idempotent=True)

    def get_order(self, order_id):
        return self.call("GET", f"/v2/checkout/orders/{_safe_id(order_id)}")

    # ------------------------------------------------------------ webhooks
    def verify_webhook(self, headers, event, webhook_id):
        h = {k.lower(): v for k, v in headers.items()}
        body = {
            "auth_algo": h.get("paypal-auth-algo"), "cert_url": h.get("paypal-cert-url"),
            "transmission_id": h.get("paypal-transmission-id"), "transmission_sig": h.get("paypal-transmission-sig"),
            "transmission_time": h.get("paypal-transmission-time"), "webhook_id": webhook_id,
            "webhook_event": event,
        }
        if not all(body.values()):
            return False
        return self.call("POST", "/v1/notifications/verify-webhook-signature", body).get(
            "verification_status") == "SUCCESS"


def _json(raw):
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {}


def _safe_id(s):
    s = str(s)
    if not s or not all(c.isalnum() or c in "-_" for c in s) or len(s) > 64:
        raise PayPalError(400, "invalid subscription id")
    return s
