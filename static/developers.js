(async () => {
  const { api, esc, toast, gate, MARK, barChart, usd } = Deer;
  const root = document.getElementById("root");
  await gate(root);
  let D, amount = null, newKey = null, lang = "curl";
  const base = Deer.API_BASE || location.origin;
  const SCOPE = { chat: "Chat", analyze: "Analyze pictures", imagine: "Create pictures" };
  const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "never");

  function samples(key) {
    const k = key || "dk_your_key";
    return {
      curl: `curl ${base}/v1/chat \\
  -H "Authorization: Bearer ${k}" \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Write a JS game loop"}]}'`,
      python: `import requests

r = requests.post("${base}/v1/chat",
    headers={"Authorization": "Bearer ${k}"},
    json={"messages": [{"role": "user", "content": "Write a JS game loop"}]})
data = r.json()
print(data["reply"])
print("cost:", data["cost"]["usd"], "balance:", data["balance"])`,
      javascript: `const r = await fetch("${base}/v1/chat", {
  method: "POST",
  headers: { "Authorization": "Bearer ${k}", "Content-Type": "application/json" },
  body: JSON.stringify({ messages: [{ role: "user", content: "Write a JS game loop" }] }),
});
const data = await r.json();
console.log(data.reply, data.cost.usd);`,
      unity: `// Unity C# (UnityWebRequest)
var json = "{\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Name a sword\\"}]}";
var req = new UnityWebRequest("${base}/v1/chat", "POST");
req.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
req.downloadHandler = new DownloadHandlerBuffer();
req.SetRequestHeader("Content-Type", "application/json");
req.SetRequestHeader("Authorization", "Bearer ${k}");
yield return req.SendWebRequest();
Debug.Log(req.downloadHandler.text);`,
    };
  }

  function render() {
    const p = D.prices, u = D.usage;
    const active = D.keys.filter((k) => !k.revoked);
    root.innerHTML = `<div class="dev-page">
      <header><div class="mark">${MARK}<span>Deer AI</span></div><a class="btn" href="./">Back to chat</a></header>
      <h1>API</h1>
      <p class="muted" style="margin:0">Use Deer AI's models from your own games, bots and apps. You pay only for what you use, from prepaid credit.</p>
      ${D.enabled ? "" : `<div class="wait-note" style="margin-top:14px">The API is switched off right now. Keys and credit still work when it's back on.</div>`}
      <div class="stat-cards">
        <div class="stat-card"><b>${D.free ? "Free" : usd(D.balance, 2)}</b><span>${D.free ? "your own keys are never charged" : "credit left"}</span></div>
        <div class="stat-card"><b>${usd(D.spent_this_month)}</b><span>spent this month</span></div>
        <div class="stat-card"><b>${u.total.requests}</b><span>requests, last 30 days</span></div>
        <div class="stat-card"><b>${active.length}</b><span>active key${active.length === 1 ? "" : "s"}</span></div>
      </div>

      ${D.free ? "" : `<div class="panel"><h2>Add credit</h2>
        ${D.paypal ? `<p class="muted" style="margin:4px 0 0;font-size:14px">Credit never expires. Pay once with PayPal; it's added as soon as the payment goes through.</p>
          <div class="packs">${D.packs.map((v) => `<button class="btn" data-pack="${v}" aria-pressed="${amount === v}">$${v}</button>`).join("")}
            <input id="custom-amt" type="number" min="${D.min_topup}" max="1000" step="1" placeholder="Other amount" style="width:150px"></div>
          <div id="pp" style="max-width:420px"></div>`
        : `<p class="muted">Credit top-ups aren't set up yet. Ask the site owner.</p>`}</div>`}

      <div class="panel"><div class="row-head"><h2>API keys</h2></div>
        ${newKey ? `<div class="newkey" role="alert"><b>Copy your new key now.</b> For your safety it won't be shown again.
          <code id="nk">${esc(newKey)}</code><button class="btn primary" id="copy-key">Copy key</button> <button class="btn ghost" id="hide-key">Done</button></div>` : ""}
        <form id="kf" class="grid2" style="align-items:end">
          <div><label for="k-name">Name</label><input id="k-name" type="text" maxlength="60" placeholder="My game server"></div>
          <div><label for="k-rate">Requests per minute</label><input id="k-rate" type="number" min="1" max="600" value="60"></div>
          <div><label for="k-cap">Monthly spending limit, $ (0 = none)</label><input id="k-cap" type="number" min="0" step="0.5" value="0"></div>
          <div><label>Allowed to</label><div style="display:flex;flex-wrap:wrap;gap:4px 14px;padding:8px 0">${Object.entries(SCOPE).map(([s, t]) =>
            `<label class="toggle" style="margin:0"><input type="checkbox" data-scope="${s}" checked> ${t}</label>`).join("")}</div></div>
          <div style="grid-column:1/-1"><button class="btn primary">Create key</button></div>
        </form>
        <div style="margin-top:10px">${D.keys.map((k) => `<div class="keyrow ${k.revoked ? "revoked" : ""}">
          <div><b>${esc(k.name)}</b> <code class="muted">${esc(k.prefix)}</code>${k.revoked ? ` <em class="tag-other">revoked</em>` : ""}</div>
          ${k.revoked ? "<span></span>" : `<button class="btn ghost danger" data-revoke="${k.id}">Revoke</button>`}
          <div class="sub">${k.scopes.map((s) => SCOPE[s]).join(", ")} · ${k.rate_per_min}/min · spent ${usd(k.spent_this_month)} this month${k.monthly_cap ? ` of ${usd(k.monthly_cap, 2)}` : ""} · last used ${when(k.last_used_at)}</div>
        </div>`).join("") || `<p class="muted">No keys yet. Create one above.</p>`}</div></div>

      <div class="panel"><div class="row-head"><h2>Usage</h2><span class="muted">last 30 days · ${usd(u.total.cost)}</span></div>
        <div class="tabs-mini" role="tablist"><button data-uv="cost" aria-selected="true">Cost</button><button data-uv="req" aria-selected="false">Requests</button></div>
        <div id="uchart"></div>
        <div class="grid2" style="margin-top:16px;gap:0 24px">
          <div><h3 class="sub">By endpoint</h3><table><tbody>${u.by_endpoint.map((e) => `<tr><td>/v1/${esc(e.endpoint)}</td><td>${e.requests} request${e.requests === 1 ? "" : "s"}</td><td>${usd(e.cost)}</td></tr>`).join("")
            || `<tr><td class="muted">Nothing yet.</td></tr>`}</tbody></table></div>
          <div><h3 class="sub">Totals</h3><table><tbody>
            <tr><td>Input tokens</td><td>${u.total.inp.toLocaleString()}</td></tr><tr><td>Output tokens</td><td>${u.total.outp.toLocaleString()}</td></tr>
            <tr><td>Pictures</td><td>${u.total.images.toLocaleString()}</td></tr></tbody></table></div>
        </div>
        <details style="margin-top:10px"><summary>Recent requests</summary><div style="overflow-x:auto"><table><thead><tr><th>When</th><th>Endpoint</th><th>Key</th><th>Model</th><th>Tokens in/out</th><th>Pictures</th><th>Cost</th><th>Time</th></tr></thead><tbody>
          ${u.recent.map((r) => `<tr><td>${when(r.created_at)}</td><td>/v1/${esc(r.endpoint)}</td><td>${esc(r.key_name || "")}</td><td>${esc(r.model_name || "")}</td>
            <td>${r.input_tokens}/${r.output_tokens}</td><td>${r.images || ""}</td><td>${r.charged ? usd(r.cost) : "free"}</td><td>${r.ms} ms</td></tr>`).join("")}</tbody></table></div></details>
        ${D.ledger.length ? `<details><summary>Credit history</summary><table><tbody>${D.ledger.map((l) =>
          `<tr><td>${when(l.created_at)}</td><td>${esc(l.note || l.kind)}</td><td>${l.amount > 0 ? "+" : ""}${usd(l.amount, 2)}</td></tr>`).join("")}</tbody></table></details>` : ""}
      </div>

      <div class="panel"><h2>Prices</h2>
        <table><tbody>
          <tr><td>Chat, input</td><td>${usd(p.chat_in_per_1k, 5)} per 1,000 tokens</td></tr>
          <tr><td>Chat, output</td><td>${usd(p.chat_out_per_1k, 5)} per 1,000 tokens</td></tr>
          <tr><td>Create a picture</td><td>${usd(p.image)} each</td></tr>
          <tr><td>Analyze a picture</td><td>${usd(p.analyze)} each</td></tr>
        </tbody></table>
        <p class="muted" style="font-size:13px">Deer AI's chat models read and write one character (byte) at a time, so 1 token = 1 byte of text. Input is what the model reads, including earlier messages you send; it's capped at the model's context length. A request is refused up front if it could cost more than your credit, so you're never charged past zero.</p>
      </div>

      <div class="panel docs"><h2>Quick start</h2>
        <div class="tabs-mini" role="tablist">${["curl", "python", "javascript", "unity"].map((l) => `<button data-lang="${l}" aria-selected="${lang === l}">${{ curl: "curl", python: "Python", javascript: "JavaScript", unity: "Unity C#" }[l]}</button>`).join("")}</div>
        <pre id="sample">${esc(samples(newKey)[lang])}</pre><button class="btn" id="copy-sample">Copy</button>
        <h3>Endpoints</h3>
        <table><tbody>
          <tr><td>POST /v1/chat</td><td><code>messages</code> (list of <code>{role: "user"|"assistant", content}</code>, last from the user), optional <code>model</code> (name or id), <code>temperature</code>, <code>max_tokens</code>, <code>stream</code>. Returns <code>reply</code>, <code>usage</code>, <code>cost</code>, <code>balance</code>.</td></tr>
          <tr><td>POST /v1/analyze</td><td><code>image</code> (base64 or a data: URL), optional <code>model</code>, <code>top</code>. Returns <code>predictions</code> and picture <code>stats</code>.</td></tr>
          <tr><td>POST /v1/imagine</td><td><code>prompt</code>, optional <code>count</code> (1 to 8), <code>model</code>, <code>temperature</code>. Returns <code>images</code> as base64 PNGs.</td></tr>
          <tr><td>GET /v1/models</td><td>Models you can use, with prices.</td></tr>
          <tr><td>GET /v1/usage</td><td>Your balance and this key's spending.</td></tr>
        </tbody></table>
        <h3>Streaming</h3>
        <p class="muted" style="font-size:14px">With <code>"stream": true</code>, <code>/v1/chat</code> sends server-sent events: <code>start</code>, then a <code>token</code> event per piece of text, then <code>done</code> with the full reply, usage and cost. If you disconnect early you only pay for what was written.</p>
        <h3>Errors</h3>
        <table><tbody>
          <tr><td>401</td><td>Missing, wrong or revoked key.</td></tr><tr><td>402</td><td>Not enough credit, or the key's monthly limit is reached.</td></tr>
          <tr><td>403</td><td>The key isn't allowed to use that endpoint.</td></tr><tr><td>404</td><td>No model with that name.</td></tr>
          <tr><td>429</td><td>Too many requests; wait the seconds in the <code>Retry-After</code> header.</td></tr>
          <tr><td>503</td><td>No model of that type is available, or the API is switched off.</td></tr>
        </tbody></table>
        <p class="muted" style="font-size:13px">Errors come back as <code>{"error": {"type", "message"}}</code>. API requests aren't used to train Deer AI.</p>
      </div></div>`;

    let view = "cost";
    const draw = () => barChart(document.getElementById("uchart"), u.daily, (r) => (view === "cost" ? r.cost : r.requests),
      (r) => `<div>Cost <b>${usd(r.cost)}</b></div><div>Requests <b>${r.requests}</b></div><div>Tokens in/out <b>${r.inp}/${r.outp}</b></div>${r.images ? `<div>Pictures <b>${r.images}</b></div>` : ""}`);
    draw();
    root.querySelectorAll("[data-uv]").forEach((b) => (b.onclick = () => {
      view = b.dataset.uv;
      root.querySelectorAll("[data-uv]").forEach((x) => x.setAttribute("aria-selected", x === b));
      draw();
    }));
    root.querySelectorAll("[data-lang]").forEach((b) => (b.onclick = () => { lang = b.dataset.lang; render(); }));
    document.getElementById("copy-sample").onclick = () => navigator.clipboard.writeText(samples(newKey)[lang]).then(() => toast("Copied."));
    if (newKey) {
      document.getElementById("copy-key").onclick = () => navigator.clipboard.writeText(newKey).then(() => toast("Key copied."));
      document.getElementById("hide-key").onclick = () => { newKey = null; render(); };
    }
    document.getElementById("kf").onsubmit = async (e) => {
      e.preventDefault();
      const v = (id) => document.getElementById(id).value;
      try {
        const k = await api("/api/dev/keys", { body: { name: v("k-name").trim() || "My key", rate_per_min: +v("k-rate") || 60,
          monthly_cap: +v("k-cap") || 0, scopes: [...root.querySelectorAll("[data-scope]:checked")].map((x) => x.dataset.scope) } });
        newKey = k.key;
        await reload();
      } catch (ex) { toast(ex.message); }
    };
    root.querySelectorAll("[data-revoke]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Revoke this key? Anything using it stops working right away.")) return;
      await api(`/api/dev/keys/${b.dataset.revoke}`, { method: "PATCH", body: { revoke: true } });
      toast("Key revoked."); reload();
    }));
    root.querySelectorAll("[data-pack]").forEach((b) => (b.onclick = () => { amount = +b.dataset.pack; document.getElementById("custom-amt").value = ""; render(); }));
    const ca = document.getElementById("custom-amt");
    if (ca) ca.onchange = () => { const v = +ca.value; if (v >= D.min_topup) { amount = v; mountPayPal(); } else toast(`The smallest top-up is $${D.min_topup}.`); };
    mountPayPal();
  }

  let sdk;
  async function mountPayPal() {
    const box = document.getElementById("pp");
    if (!box || !D.paypal) return;
    if (!amount) { box.innerHTML = `<p class="muted" style="font-size:14px">Pick an amount to pay with PayPal.</p>`; return; }
    box.innerHTML = `<p style="font-size:14px">Adding <b>$${amount.toFixed(2)}</b></p><div id="pp-btn"></div>`;
    try {
      sdk = sdk || new Promise((ok, fail) => {
        const s = document.createElement("script");
        s.src = `https://www.paypal.com/sdk/js?client-id=${encodeURIComponent(D.paypal.client_id)}&currency=USD&intent=capture`;
        s.onload = ok; s.onerror = () => fail(new Error("PayPal couldn't load. Refresh and try again."));
        document.head.appendChild(s);
      });
      await sdk;
    } catch (e) { return toast(e.message); }
    window.paypal.Buttons({
      style: { shape: "rect", color: "gold", layout: "vertical", label: "pay", height: 44 },
      createOrder: async () => (await api("/api/dev/credit/order", { body: { amount } })).id,
      onApprove: async (data) => {
        try {
          const r = await api("/api/dev/credit/capture", { body: { order_id: data.orderID } });
          toast(`Added $${r.added.toFixed(2)}. Balance: $${r.balance.toFixed(2)}.`);
          amount = null; reload();
        } catch (e) { toast(e.message + " If you were charged, the credit will appear within a few minutes."); }
      },
      onError: () => toast("PayPal hit a problem. You weren't charged. Try again."),
    }).render("#pp-btn");
  }

  async function reload() { D = await api("/api/dev"); render(); }
  await reload();
})();
