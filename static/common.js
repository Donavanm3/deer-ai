// Shared helpers for the Deer AI web client.
const Deer = (() => {
  const TOKEN = "deer_token";
  const MARK = `<svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path d="M16 29c-3.5 0-6-2.7-6-6.5 0-2.3 1.2-4 2.6-5.2L11 13 7.5 11.5 5 6.5l1.8-.6 2 3.6 2.2.7L9.6 5.5l1.9-.4 1.6 5.4 1.3 1.6h3.2l1.3-1.6 1.6-5.4 1.9.4-1.4 4.7 2.2-.7 2-3.6 1.8.6-2.5 5L21 13l-1.6 4.3c1.4 1.2 2.6 2.9 2.6 5.2 0 3.8-2.5 6.5-6 6.5Z"
      fill="currentColor"/><circle cx="13.7" cy="21" r="1.2" fill="var(--pine)"/><circle cx="18.3" cy="21" r="1.2" fill="var(--pine)"/></svg>`;

  // Deer AI's server. Empty = this same site; set in static/config.js when the pages live elsewhere (GitHub Pages).
  const API_BASE = (window.DEER_API || "").replace(/\/+$/, "");
  const token = () => localStorage.getItem(TOKEN);
  const setToken = (t) => (t ? localStorage.setItem(TOKEN, t) : localStorage.removeItem(TOKEN));

  async function api(path, opts = {}) {
    const headers = opts.headers || {};
    if (token()) headers.Authorization = "Bearer " + token();
    let body = opts.body;
    if (body && !(body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(body);
    }
    const res = await fetch(API_BASE + path, { method: opts.method || (body ? "POST" : "GET"), headers, body });
    if (res.status === 401 && !path.startsWith("/api/auth/")) {
      setToken(null);
      location.href = "./";
      throw new Error("Signed out");
    }
    if (opts.raw) return res;
    if (!res.ok) throw await apiError(res);
    return res.json().catch(() => ({}));
  }

  async function apiError(res) {
    const data = await res.json().catch(() => ({}));
    let msg = data.detail, extra = {};
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join(". ");
    else if (msg && typeof msg === "object") { extra = msg; msg = msg.message; }
    const err = new Error(msg || `Request failed (${res.status})`);
    return Object.assign(err, extra, { status: res.status });
  }

  // POST that streams server-sent events. Gives up if nothing arrives for `idleMs`.
  async function stream(path, body, onEvent, idleMs = 45000) {
    const ctrl = new AbortController();
    let timer;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => ctrl.abort(), idleMs); };
    arm();
    try {
      const res = await fetch(API_BASE + path, { method: "POST", signal: ctrl.signal,
        headers: { Authorization: "Bearer " + token(), "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (res.status === 401) { setToken(null); location.href = "./"; return; }
      if (!res.ok) throw await apiError(res);
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        arm();
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i);
          buf = buf.slice(i + 2);
          for (const line of chunk.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
        }
      }
    } catch (e) {
      if (e.name === "AbortError") throw new Error("Deer AI took too long to answer. Please try again.");
      throw e;
    } finally { clearTimeout(timer); }
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  let toastTimer;
  function toast(text) {
    let el = document.querySelector(".toast");
    if (!el) {
      el = document.createElement("div");
      el.className = "toast";
      el.setAttribute("role", "status");
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 3500);
  }

  // Sign-in / register gate. Resolves with the user.
  async function gate(root) {
    if (token()) {
      try { return await api("/api/auth/me"); } catch { setToken(null); }
    }
    const st = await api("/api/auth/status");
    let mode = st.needs_setup ? "register" : "login";
    return new Promise((resolve) => {
      const render = () => {
        const setup = st.needs_setup;
        root.innerHTML = `
          <div class="gate"><form novalidate>
            <div class="mark">${MARK}<span>Deer AI</span></div>
            <h1>${setup ? "Set up Deer AI" : mode === "login" ? "Welcome back" : "Create an account"}</h1>
            <p class="muted">${setup ? "The first account becomes the admin who builds and trains the models." : "Sign in to chat, analyze images and create new ones."}</p>
            <label for="u">Username</label><input id="u" type="text" autocomplete="username" required>
            <label for="p">Password</label><input id="p" type="password" autocomplete="${mode === "login" ? "current-password" : "new-password"}" required minlength="8">
            <div class="err" role="alert"></div>
            <button class="btn primary" style="width:100%;margin-top:8px">${setup ? "Create admin account" : mode === "login" ? "Sign in" : "Create account"}</button>
            ${!setup && st.registration_open ? `<div class="switch">${mode === "login" ? "New here?" : "Have an account?"} <button type="button">${mode === "login" ? "Create an account" : "Sign in"}</button></div>` : ""}
          </form></div>`;
        const f = root.querySelector("form");
        f.querySelector("#u").focus();
        const sw = f.querySelector(".switch button");
        if (sw) sw.onclick = () => { mode = mode === "login" ? "register" : "login"; render(); };
        f.onsubmit = async (e) => {
          e.preventDefault();
          const err = f.querySelector(".err");
          err.textContent = "";
          try {
            const r = await api(`/api/auth/${setup ? "register" : mode}`, {
              body: { username: f.querySelector("#u").value.trim(), password: f.querySelector("#p").value },
            });
            setToken(r.token);
            resolve(r.user);
          } catch (ex) { err.textContent = ex.message; }
        };
      };
      render();
    });
  }

  async function logout() {
    try { await api("/api/auth/logout", { method: "POST" }); } catch {}
    setToken(null);
    location.href = "./";
  }

  function until(ts) {
    if (!ts) return "";
    const m = Math.max(1, Math.round((ts * 1000 - Date.now()) / 60000));
    return m < 60 ? `${m} min` : m < 1440 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${Math.round(m / 1440)} days`;
  }

  // installable app (PWA): works from the phone home screen
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  // Daily bar chart with a hover/touch tooltip. rows: [{day, ...}], value: row => number, tip: row => html
  function barChart(el, rows, value, tip, days = 30) {
    const byDay = Object.fromEntries(rows.map((r) => [r.day, r]));
    const all = [];
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(Date.now() - i * 86400000).toISOString().slice(0, 10);
      all.push(byDay[d] || { day: d, empty: true });
    }
    const max = Math.max(1e-9, ...all.map((r) => (r.empty ? 0 : value(r))));
    el.classList.add("barchart");
    el.innerHTML = `<div class="bc-bars">${all.map((r, i) => `<div class="bc-col" data-i="${i}"><div class="bc-bar" style="height:${r.empty ? 0 : Math.max(2, (value(r) / max) * 100)}%"></div></div>`).join("")}</div>
      <div class="bc-axis"><span>${all[0].day.slice(5)}</span><span>today</span></div><div class="bc-tip" hidden></div>`;
    const tipEl = el.querySelector(".bc-tip");
    const show = (e) => {
      const col = e.target.closest(".bc-col");
      if (!col) return;
      const r = all[+col.dataset.i];
      tipEl.innerHTML = `<b>${new Date(r.day + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" })}</b>${r.empty ? "<div>No requests</div>" : tip(r)}`;
      tipEl.hidden = false;
      const x = col.offsetLeft + col.offsetWidth / 2, w = tipEl.offsetWidth;
      tipEl.style.left = Math.min(Math.max(0, x - w / 2), el.clientWidth - w) + "px";
      el.querySelectorAll(".bc-col").forEach((c) => c.classList.toggle("on", c === col));
    };
    el.onpointermove = show; el.onpointerdown = show;
    el.onpointerleave = () => { tipEl.hidden = true; el.querySelectorAll(".bc-col").forEach((c) => c.classList.remove("on")); };
  }
  const usd = (v, d = 4) => "$" + Number(v || 0).toFixed(v >= 1 ? 2 : d);

  return { API_BASE, api, stream, esc, toast, gate, logout, MARK, token, until, barChart, usd };
})();
