(async () => {
  const { api, stream, esc, toast, gate, logout, MARK, until } = Deer;
  const root = document.getElementById("root");
  const user = await gate(root);

  const S = {
    convs: [], cid: null, msgs: [], models: [], busy: false, imagine: false, view: "chat", usage: null,
    pick: JSON.parse(localStorage.getItem("deer_pick") || "{}"),
  };
  const KIND_NAME = { chat: "Chat", vision: "Image analysis", canvas: "Image creation" };

  root.innerHTML = `
  <div class="app">
    <aside class="side" id="side">
      <div class="top"><div class="mark">${MARK}<span>Deer AI</span></div>
        <button class="btn ghost icon-btn menu-btn" id="close-side" aria-label="Close menu">✕</button></div>
      <button class="btn new" id="new">New chat</button>
      <button class="btn ghost side-link" id="saved-btn">Saved</button>
      <ul class="convs" id="convs" aria-label="Conversations"></ul>
      <div class="foot"><span class="who">${esc(user.username)} <a class="plan-link" id="plan-link" href="plans.html">Plans</a> <a class="plan-link" href="developers.html">API</a></span>
        ${user.role === "admin" ? `<a class="btn" href="admin.html">Admin</a><a class="btn ghost" href="plans.html#test">Test as user</a>` : ""}
        <button class="btn ghost" id="logout">Sign out</button></div>
    </aside>
    <main class="main">
      <div class="bar">
        <button class="btn ghost icon-btn menu-btn" id="open-side" aria-label="Open menu">☰</button>
        <div class="title" id="title">New chat</div>
        <button class="btn ghost icon-btn" id="export" aria-label="Export this chat" title="Export this chat">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v12m0 0-4-4m4 4 4-4M5 21h14"/></svg></button>
        <button class="btn" id="models-btn" aria-expanded="false" aria-controls="models-pop">Models</button>
      </div>
      <div class="models-pop hidden" id="models-pop"></div>
      <div class="test-banner hidden" id="test-banner" role="status"></div>
      <div class="feed" id="feed"><div class="feed-inner" id="feed-inner"></div></div>
      <div class="composer">
        <div class="banner hidden" id="limit" role="alert"><span id="limit-text"></span><a class="btn primary" href="plans.html">See plans</a></div>
        <form id="form">
          <button type="button" class="btn ghost icon-btn" id="attach" aria-label="Analyze an image" title="Analyze an image">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/></svg></button>
          <input type="file" id="file" accept="image/*" hidden>
          <textarea id="text" rows="1" placeholder="Message" aria-label="Message"></textarea>
          <button type="button" class="btn mode" id="mode" aria-pressed="false">Imagine</button>
          <button class="btn primary" id="send">Send</button>
        </form>
        <div class="usage-line" id="usage" aria-live="polite"></div>
        <div class="hint" id="hint">Enter to send, Shift+Enter for a new line. Paste or attach a picture to analyze it.</div>
      </div>
    </main>
  </div>`;

  const $ = (id) => document.getElementById(id);
  const feed = $("feed"), inner = $("feed-inner"), text = $("text"), form = $("form");

  // ------------------------------------------------------------------ models
  async function loadModels() {
    try { S.models = await api(`/api/models${showUnpub() ? "?all=1" : ""}`); } catch { return; }
    S.modelsAt = Date.now();
    // forget picks for models that were unpublished or deleted since this page loaded
    let changed = false;
    ["chat", "vision", "canvas"].forEach((k) => {
      if (S.pick[k] && !S.models.some((m) => m.id == S.pick[k] && m.kind === k)) { delete S.pick[k]; changed = true; }
    });
    if (changed) localStorage.setItem("deer_pick", JSON.stringify(S.pick));
    const pop = $("models-pop");
    const admin = user.role === "admin" && !S.usage?.test;
    pop.innerHTML = ["chat", "vision", "canvas"].map((k) => {
      const list = S.models.filter((m) => m.kind === k);
      if (!list.length) return `<label>${KIND_NAME[k]}</label><div class="muted" style="font-size:14px">None available yet</div>`;
      return `<label for="pick-${k}">${KIND_NAME[k]}</label><select id="pick-${k}" data-kind="${k}">
        ${list.map((m) => `<option value="${m.id}" ${S.pick[k] == m.id ? "selected" : ""}>${esc(m.name)}${m.is_default ? " (default)" : ""}${admin && !m.published ? " (not published, only you see it)" : ""}</option>`).join("")}</select>`;
    }).join("") + (admin ? `<label class="toggle" style="margin-top:14px;font-size:14px"><input type="checkbox" id="show-unpub" ${showUnpub() ? "checked" : ""}>
        Also show my unpublished models (users never see these)</label>` : "");
    const su = $("show-unpub");
    if (su) su.onchange = () => { localStorage.setItem("deer_show_unpub", su.checked ? "1" : ""); loadModels(); };
    pop.querySelectorAll("select").forEach((s) => (s.onchange = () => {
      S.pick[s.dataset.kind] = +s.value;
      localStorage.setItem("deer_pick", JSON.stringify(S.pick));
    }));
  }
  // keep the list current: when the page comes back into view, every minute, and before opening the picker
  document.addEventListener("visibilitychange", () => { if (!document.hidden) loadModels(); });
  setInterval(() => { if (!document.hidden) loadModels(); }, 60000);
  // a request named a model that's gone: drop it, refresh, and let the caller retry with the default
  async function modelGone(e) {
    if (!e.model_gone) return false;
    delete S.pick[e.kind];
    localStorage.setItem("deer_pick", JSON.stringify(S.pick));
    await loadModels();
    return true;
  }
  const modelFor = (k) => S.pick[k] || undefined;
  function showUnpub() { return user.role === "admin" && !S.usage?.test && localStorage.getItem("deer_show_unpub") === "1"; }

  $("models-btn").onclick = () => {
    const pop = $("models-pop");
    if (pop.classList.contains("hidden")) loadModels();
    const open = pop.classList.toggle("hidden") === false;
    $("models-btn").setAttribute("aria-expanded", open);
  };
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#models-pop, #models-btn")) {
      $("models-pop").classList.add("hidden");
      $("models-btn").setAttribute("aria-expanded", "false");
    }
  });

  // ----------------------------------------------------------- conversations
  async function loadConvs() {
    S.convs = await api("/api/conversations");
    $("convs").innerHTML = S.convs.map((c) => `
      <li class="${c.id === S.cid ? "active" : ""}"><button class="open" data-id="${c.id}">${esc(c.title)}</button>
      <button class="del" data-del="${c.id}" aria-label="Delete ${esc(c.title)}">✕</button></li>`).join("")
      || `<li class="muted" style="padding:8px;font-size:14px">Your chats will show up here.</li>`;
  }
  $("convs").onclick = async (e) => {
    const open = e.target.closest("[data-id]"), del = e.target.closest("[data-del]");
    if (open) { await openConv(+open.dataset.id); closeSide(); }
    if (del) {
      await api(`/api/conversations/${del.dataset.del}`, { method: "DELETE" });
      if (+del.dataset.del === S.cid) newChat();
      loadConvs();
    }
  };
  async function openConv(id) {
    const c = await api(`/api/conversations/${id}`);
    S.cid = c.id; S.msgs = c.messages; S.view = "chat";
    $("title").textContent = c.title;
    render(); loadConvs();
  }
  function newChat() {
    S.cid = null; S.msgs = []; S.view = "chat";
    $("title").textContent = "New chat";
    render(); loadConvs(); text.focus();
  }
  $("new").onclick = () => { newChat(); closeSide(); };
  $("logout").onclick = logout;
  const closeSide = () => $("side").classList.remove("open");
  $("open-side").onclick = () => $("side").classList.add("open");
  $("close-side").onclick = closeSide;

  // ------------------------------------------------------------------ render
  function msgHTML(m) {
    const who = m.role === "user" ? "user" : "assistant";
    if (m.kind === "text") {
      if (m.streaming && !m.content) return "";
      const body = who === "assistant" ? rich(m.content) : `<div class="plain">${esc(m.content)}</div>`;
      return `<div class="msg ${who}${m.streaming ? " streaming" : ""}"><div class="bubble">${body}</div></div>`;
    }
    if (m.kind === "imagine") return `<div class="msg user"><div class="tag">Imagine</div><div class="bubble">${esc(m.content)}</div></div>`;
    if (m.kind === "image") return `<div class="msg user"><img class="upload" src="${esc(m.content)}" alt="Image you sent"></div>`;
    if (m.kind === "images") {
      const imgs = m.meta.images || [];
      return `<div class="msg assistant"><div class="gen-grid">${imgs.map((i) =>
        `<figure><img src="${esc(i.data_url)}" alt="Generated ${esc(i.label)}" data-orig="${esc(i.data_url)}"><figcaption><span>${esc(i.label)}</span>
          <span class="art-tools"><button type="button" data-art="pixel" title="Snap to a small pixel-art palette">Pixel art</button>
          <button type="button" data-art="nobg" title="Make the background see-through">No background</button>
          <button type="button" data-art="dl" data-name="deer-${esc(i.label.replace(/\W+/g, "-"))}">Download</button></span></figcaption></figure>`).join("")}</div>
        <div class="art-row"><button type="button" class="btn ghost" data-sheet>Download all as a sprite sheet</button></div>
        ${m.meta.note ? `<div class="note">${esc(m.meta.note)}</div>` : ""}</div>`;
    }
    if (m.kind === "analysis") {
      const { predictions: p, stats: s } = m.meta;
      const preds = p ? p.map((x, i) => `<div class="pred ${i === 0 ? "top" : ""}"><span>${esc(x.label)}</span>
          <div class="track"><div class="fill" style="width:${(x.confidence * 100).toFixed(1)}%"></div></div>
          <span class="pct">${Math.round(x.confidence * 100)}%</span></div>`).join("")
        : `<div class="note">No image analysis model is published yet, so here's what I can measure directly.</div>`;
      const lead = p ? `<div>This looks like <b>${esc(p[0].label)}</b>${p[0].confidence < 0.5 ? ", but I'm not sure" : ""}.</div>` : "";
      const pct = (v) => Math.round(v * 100) + "%";
      return `<div class="msg assistant"><div class="analysis">${lead}${preds}
        <div class="swatches" aria-label="Main colors">${s.palette.map((c) => `<span title="${c.hex}" style="background:${c.hex};flex:${c.share}"></span>`).join("")}</div>
        <div class="facts"><div><b>${s.width}×${s.height}</b>size</div><div><b>${pct(s.brightness)}</b>brightness</div>
        <div><b>${pct(Math.min(1, s.contrast))}</b>contrast</div><div><b>${pct(Math.min(1, s.colorfulness))}</b>colorfulness</div>
        <div><b>${pct(Math.min(1, s.detail))}</b>detail</div></div></div></div>`;
    }
    return "";
  }
  // ------------------------------------------------------------ rich replies
  const RUNNABLE = { html: "html", htm: "html", javascript: "js", js: "js" };
  function inline(t) {  // `code` inside normal text
    return esc(t).replace(/`([^`\n]+)`/g, "<code>$1</code>");
  }
  function rich(text) {
    const parts = [], re = /```([\w+#.-]*)[^\n]*\n([\s\S]*?)(?:```|$)/g;
    let last = 0, mt;
    while ((mt = re.exec(text))) {
      if (mt.index > last) parts.push(`<div class="plain">${inline(text.slice(last, mt.index).replace(/^\n+|\n+$/g, ""))}</div>`);
      const lang = (mt[1] || "").toLowerCase(), code = mt[2].replace(/\n$/, "");
      parts.push(`<div class="codeblock" data-lang="${esc(lang)}"><div class="code-bar"><span>${esc(lang || "code")}</span>
        <button type="button" data-copy>Copy</button>${RUNNABLE[lang] ? `<button type="button" data-run>▶ Run</button>` : ""}</div>
        <pre><code>${esc(code)}</code></pre></div>`);
      last = re.lastIndex;
    }
    if (last < text.length) parts.push(`<div class="plain">${inline(text.slice(last).replace(/^\n+/, ""))}</div>`);
    return parts.join("");
  }
  function runCode(block) {
    const old = block.querySelector("iframe");
    if (old) { old.remove(); block.querySelector("[data-run]").textContent = "▶ Run"; return; }
    const code = block.querySelector("code").textContent, lang = RUNNABLE[block.dataset.lang];
    const page = lang === "html" ? code : `<!doctype html><style>body{margin:0;background:#111;color:#eee;font:14px sans-serif}canvas{display:block;margin:auto;background:#1f2b25}</style>
      <canvas id="game" width="480" height="320"></canvas><script>${code.replace(/<\/script/gi, "<\\/script")}<\/script>`;
    const f = document.createElement("iframe");
    f.sandbox = "allow-scripts";  // runs in isolation: no access to your account, cookies or this page
    f.srcdoc = page;
    f.title = "Code preview";
    block.appendChild(f);
    block.querySelector("[data-run]").textContent = "■ Stop";
    setTimeout(() => f.focus(), 50);
  }

  const ICON = (d) => `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">${d}</svg>`;
  const SUGGEST = [
    { label: "Make a tiny HTML game", text: "Make a tiny HTML game", icon: ICON('<rect x="3" y="6" width="18" height="12" rx="3"/><path d="M8 10v4M6 12h4M15 11h.01M17 13h.01"/>') },
    { label: "Write a JS game loop", text: "Write a JS game loop", icon: ICON('<path d="m8 8-4 4 4 4M16 8l4 4-4 4"/>') },
    { label: "Unity 2D jump script", text: "How do I make a player jump in Unity?", icon: ICON('<path d="M12 19V5M6 11l6-6 6 6"/>') },
    { label: "Draw a slime sprite", text: "slime", imagine: true, icon: ICON('<path d="M4 17c0-5 3.5-9 8-9s8 4 8 9H4Z"/><path d="M9 13h.01M15 13h.01"/>') },
  ];

  // game-art helpers for generated pictures (done in the browser)
  function loadImg(src) { return new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = src; }); }
  async function artOp(img, op) {
    const src = await loadImg(img.src), c = document.createElement("canvas");
    c.width = src.naturalWidth; c.height = src.naturalHeight;
    const g = c.getContext("2d"); g.drawImage(src, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height), px = d.data;
    if (op === "pixel") {  // downscale to 32 px, snap colors to a 4-level-per-channel palette, scale back up sharp
      const s = document.createElement("canvas"); s.width = s.height = 32;
      const sg = s.getContext("2d"); sg.drawImage(src, 0, 0, 32, 32);
      const sd = sg.getImageData(0, 0, 32, 32);
      for (let i = 0; i < sd.data.length; i += 4) for (let k = 0; k < 3; k++) sd.data[i + k] = Math.round(sd.data[i + k] / 85) * 85;
      sg.putImageData(sd, 0, 0);
      g.imageSmoothingEnabled = false; g.clearRect(0, 0, c.width, c.height); g.drawImage(s, 0, 0, c.width, c.height);
      return c.toDataURL("image/png");
    }
    // no background: flood from the edges, clearing pixels close to the corner color
    const w = c.width, h = c.height, ref = [px[0], px[1], px[2]], seen = new Uint8Array(w * h), stack = [];
    const near = (i) => Math.abs(px[i * 4] - ref[0]) + Math.abs(px[i * 4 + 1] - ref[1]) + Math.abs(px[i * 4 + 2] - ref[2]) < 70;
    for (let x = 0; x < w; x++) stack.push(x, (h - 1) * w + x);
    for (let y = 0; y < h; y++) stack.push(y * w, y * w + w - 1);
    while (stack.length) {
      const i = stack.pop();
      if (seen[i] || !near(i)) continue;
      seen[i] = 1; px[i * 4 + 3] = 0;
      const x = i % w, y = (i / w) | 0;
      if (x > 0) stack.push(i - 1); if (x < w - 1) stack.push(i + 1);
      if (y > 0) stack.push(i - w); if (y < h - 1) stack.push(i + w);
    }
    g.putImageData(d, 0, 0);
    return c.toDataURL("image/png");
  }
  async function spriteSheet(imgs) {
    const loaded = await Promise.all(imgs.map((i) => loadImg(i.src)));
    const s = loaded[0].naturalWidth, c = document.createElement("canvas");
    c.width = s * loaded.length; c.height = s;
    const g = c.getContext("2d");
    loaded.forEach((im, k) => g.drawImage(im, k * s, 0, s, s));
    return c.toDataURL("image/png");
  }
  function download(url, name) { const a = document.createElement("a"); a.href = url; a.download = name + ".png"; a.click(); }

  // ---------------------------------------------------------------- feedback
  const UP = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M7 11v9H4v-9h3Zm0 0 4-8c1.7 0 3 1.3 3 3v3h5a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.8 20H7"/></svg>`;
  const DOWN = UP.replace("<path", `<path transform="rotate(180 12 12)"`);
  const NOTE = `<div class="fb-note">Feedback shares this exchange with the admin to help train Deer AI.</div>`;
  S.open = null; // message id whose correction form is open

  const BOOK = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M6 3h12v18l-6-4-6 4V3Z"/></svg>`;
  const saveBtn = (m) => `<button class="fb-btn save ${m.saved ? "on" : ""}" data-save="${m.id}" aria-pressed="${!!m.saved}">${BOOK}<span>${m.saved ? "Saved" : "Save"}</span></button>`;

  function fbHTML(m) {
    if (m.role !== "assistant" || !m.id) return "";
    const f = m.meta.feedback;
    if (f) {
      const learn = f.status === "pending" || f.status === "applied";
      return `<div class="fb"><span class="fb-done">${learn ? "Thanks! Deer AI can learn from this." : "Thanks for the feedback."}</span>
        ${f.status !== "applied" ? `<button class="fb-btn" data-fb-reset="${m.id}">Change</button>` : ""}${saveBtn(m)}</div>`;
    }
    if (S.open === m.id) {
      if (m.kind === "analysis") {
        const vm = S.models.find((x) => x.id === m.meta.model_id);
        const opts = (vm?.classes || []).filter((c) => c !== m.meta.predictions?.[0]?.label);
        return `<form class="fb-form" data-fb-label="${m.id}"><label for="fbl-${m.id}">What is it?</label>
          <select id="fbl-${m.id}">${opts.map((c) => `<option>${esc(c)}</option>`).join("")}<option value="__other">Something else…</option></select>
          <input type="text" class="hidden" maxlength="60" placeholder="Type the right label" aria-label="Right label">
          <div class="fb-actions"><button class="btn primary">Send</button><button type="button" class="btn ghost" data-fb-cancel>Cancel</button></div>${NOTE}</form>`;
      }
      return `<form class="fb-form" data-fb-fix="${m.id}"><label for="fbc-${m.id}">What should Deer AI have said? (optional)</label>
        <textarea id="fbc-${m.id}" rows="2" maxlength="2000"></textarea>
        <div class="fb-actions"><button class="btn primary">Send</button><button type="button" class="btn ghost" data-fb-cancel>Cancel</button></div>${NOTE}</form>`;
    }
    if (m.kind === "analysis" && m.meta.predictions) {
      return `<div class="fb"><span>Is this right?</span><button class="fb-btn" data-fb="${m.id}" data-r="1">Yes</button>
        <button class="fb-btn" data-fb="${m.id}" data-r="-1">No</button>${saveBtn(m)}</div>`;
    }
    if (m.kind === "analysis") return `<div class="fb">${saveBtn(m)}</div>`;
    return `<div class="fb"><button class="fb-btn icon" data-fb="${m.id}" data-r="1" aria-label="Good reply">${UP}</button>
      <button class="fb-btn icon" data-fb="${m.id}" data-r="-1" aria-label="Bad reply">${DOWN}</button>${saveBtn(m)}</div>`;
  }

  async function sendFeedback(id, rating, extra = {}) {
    const m = S.msgs.find((x) => x.id === id);
    try {
      m.meta.feedback = await api("/api/feedback", { body: { message_id: id, rating, ...extra } });
      S.open = null;
      render();
    } catch (e) { toast(e.message); }
  }

  inner.addEventListener("click", async (e) => {
    const sug = e.target.closest("[data-suggest]");
    if (sug) {
      const x = SUGGEST[+sug.dataset.suggest];
      if (!!x.imagine !== S.imagine) $("mode").click();
      text.value = x.text; autosize(); form.requestSubmit(); return;
    }
    const cp = e.target.closest("[data-copy]");
    if (cp) {
      await navigator.clipboard.writeText(cp.closest(".codeblock").querySelector("code").textContent).catch(() => {});
      cp.textContent = "Copied"; setTimeout(() => (cp.textContent = "Copy"), 1500); return;
    }
    const rn = e.target.closest("[data-run]");
    if (rn) return runCode(rn.closest(".codeblock"));
    const art = e.target.closest("[data-art]");
    if (art) {
      const img = art.closest("figure").querySelector("img");
      if (art.dataset.art === "dl") return download(img.src, art.dataset.name);
      img.src = await artOp(img, art.dataset.art);
      img.classList.add("edited");
      return;
    }
    const sheet = e.target.closest("[data-sheet]");
    if (sheet) return download(await spriteSheet([...sheet.closest(".msg").querySelectorAll("figure img")]), "deer-sprite-sheet");
    const sv = e.target.closest("[data-save]");
    if (sv) {
      const id = +sv.dataset.save, on = sv.getAttribute("aria-pressed") !== "true";
      try {
        await api(on ? "/api/saved" : `/api/saved/${id}`, on ? { body: { message_id: id } } : { method: "DELETE" });
        S.msgs.filter((x) => x.id === id).forEach((x) => (x.saved = on));
        if (S.view === "saved" && !on) S.msgs = S.msgs.filter((x) => x.id !== id);
        render();
        toast(on ? "Saved. Find it under Saved in the menu." : "Removed from Saved.");
      } catch (ex) { toast(ex.message); }
      return;
    }
    const open = e.target.closest("[data-open-conv]");
    if (open) return openConv(+open.dataset.openConv);
    const b = e.target.closest("[data-fb]");
    if (b) {
      const id = +b.dataset.fb, r = +b.dataset.r, m = S.msgs.find((x) => x.id === id);
      if (r < 0 && m.kind !== "images") { S.open = id; render(); inner.querySelector(".fb-form textarea, .fb-form select")?.focus(); }
      else sendFeedback(id, r);
    }
    if (e.target.closest("[data-fb-cancel]")) { S.open = null; render(); }
    const reset = e.target.closest("[data-fb-reset]");
    if (reset) { delete S.msgs.find((x) => x.id === +reset.dataset.fbReset).meta.feedback; render(); }
  });
  inner.addEventListener("change", (e) => {
    if (e.target.matches(".fb-form select")) {
      const other = e.target.value === "__other", inp = e.target.parentElement.querySelector("input");
      inp.classList.toggle("hidden", !other);
      if (other) inp.focus();
    }
  });
  inner.addEventListener("submit", (e) => {
    const f = e.target;
    e.preventDefault();
    if (f.dataset.fbFix) sendFeedback(+f.dataset.fbFix, -1, { correction: f.querySelector("textarea").value.trim() || null });
    if (f.dataset.fbLabel) {
      const sel = f.querySelector("select").value;
      const label = sel === "__other" ? f.querySelector("input").value.trim() : sel;
      if (!label) return toast("Type the right label.");
      sendFeedback(+f.dataset.fbLabel, -1, { label });
    }
  });

  let frame = 0;
  const renderSoon = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; render(); }); };

  function savedHTML() {
    if (!S.msgs.length) return `<div class="empty-saved"><h2>Nothing saved yet</h2><p class="muted">Tap Save under any reply, analysis or picture to keep it here.</p></div>`;
    return S.msgs.map((m) => `<div class="saved-item"><div class="saved-head"><span>${esc(m.title)}</span>
      <button class="fb-btn" data-open-conv="${m.conversation_id}">Open chat</button></div>
      ${m.question ? `<div class="msg user"><div class="bubble">${esc(m.question)}</div></div>` : ""}
      ${msgHTML(m).replace(/<\/div>$/, `<div class="fb">${saveBtn(m)}</div></div>`)}</div>`).join("");
  }

  function render() {
    const inSaved = S.view === "saved";
    document.querySelector(".composer").classList.toggle("hidden", inSaved);
    $("export").classList.toggle("hidden", inSaved || !S.msgs.length);
    if (inSaved) { feed.classList.remove("empty"); inner.innerHTML = savedHTML(); return; }
    if (!S.msgs.length && !S.busy) {
      feed.classList.add("empty");
      inner.innerHTML = `<div class="hero"><h1>What are we<br>building today?</h1>
        <p>Deer AI helps with code and game creation, and draws game art. It only knows what its models were trained on.</p>
        <div class="chips">${SUGGEST.map((x, i) => `<button type="button" class="chip-btn" data-suggest="${i}">${x.icon}<span>${esc(x.label)}</span></button>`).join("")}</div></div>`;
      return;
    }
    feed.classList.remove("empty");
    const writing = S.msgs.some((m) => m.streaming && m.content);
    inner.innerHTML = S.msgs.map((m) => msgHTML(m).replace(/<\/div>$/, fbHTML(m) + "</div>")).join("")
      + (S.busy && !writing ? `<div class="msg assistant typing">${S.busy}</div>` : "");
    feed.scrollTop = feed.scrollHeight;
  }

  // ----------------------------------------------------------------- actions
  function setBusy(label) {
    S.busy = label;
    $("send").disabled = $("attach").disabled = !!label;
    render();
  }
  function showLimit(e) {
    $("limit-text").textContent = e.message + (e.resets_at ? ` More in ${until(e.resets_at)}.` : "");
    $("limit").classList.remove("hidden");
  }
  function newConvStarted(id) {
    if (S.cid) return;
    S.cid = id;
    loadConvs().then(() => ($("title").textContent = S.convs[0]?.title || "Chat"));
  }

  async function sendChat(t, retried = false) {
    if (S.busy) return;
    const um = { role: "user", kind: "text", content: t, meta: {} };
    const am = { role: "assistant", kind: "text", content: "", meta: {}, streaming: true };
    S.msgs.push(um, am);
    setBusy("Thinking");
    $("limit").classList.add("hidden");
    try {
      await stream("/api/chat/stream", { text: t, conversation_id: S.cid, model_id: modelFor("chat") }, (ev) => {
        if (ev.type === "start") { Object.assign(um, ev.user_message); newConvStarted(ev.conversation_id); }
        else if (ev.type === "token") { am.content += ev.text; renderSoon(); }
        else if (ev.type === "done") { Object.assign(am, ev.message); delete am.streaming; showUsage(ev.usage); }
        else if (ev.type === "error") throw new Error(ev.message);
      });
      if (am.streaming) throw new Error("The connection dropped before Deer AI finished. Try again.");
    } catch (e) {
      S.msgs = S.msgs.filter((m) => m !== am || m.content);
      if (!um.id) S.msgs = S.msgs.filter((m) => m !== um);
      delete am.streaming;
      if (!retried && await modelGone(e)) {
        setBusy(false);
        toast("That model was taken down, so Deer AI switched to the default one.");
        return sendChat(t, true);
      }
      if (e.limit) showLimit(e); else toast(e.message);
    } finally { setBusy(false); text.focus(); refreshUsage(); }
  }

  async function run(label, fn, optimistic, retried = false) {
    if (S.busy) return;
    if (optimistic) S.msgs.push(optimistic);
    setBusy(label);
    $("limit").classList.add("hidden");
    try {
      const r = await fn();
      if (optimistic) S.msgs.pop();
      S.msgs.push(...r.messages);
      newConvStarted(r.conversation_id);
      if (r.usage) showUsage(r.usage);
    } catch (e) {
      if (optimistic) S.msgs.pop();
      if (!retried && await modelGone(e)) {
        setBusy(false);
        toast("That model was taken down, so Deer AI switched to the default one.");
        return run(label, fn, optimistic, true);
      }
      if (e.limit) showLimit(e); else toast(e.message);
    } finally { setBusy(false); text.focus(); }
  }

  form.onsubmit = (e) => {
    e.preventDefault();
    const t = text.value.trim();
    if (!t) return;
    text.value = ""; autosize();
    if (S.imagine) {
      run("Drawing", () => api("/api/imagine", { body: { prompt: t, conversation_id: S.cid, model_id: modelFor("canvas") } }),
        { role: "user", kind: "imagine", content: t, meta: {} });
    } else {
      sendChat(t);
    }
  };
  function sendImage(file) {
    if (!file || !file.type.startsWith("image/")) return toast("Choose an image file.");
    const fd = new FormData();
    fd.append("file", file);
    if (S.cid) fd.append("conversation_id", S.cid);
    if (modelFor("vision")) fd.append("model_id", modelFor("vision"));
    run("Looking", () => api("/api/analyze", { body: fd }),
      { role: "user", kind: "image", content: URL.createObjectURL(file), meta: {} });
  }
  $("attach").onclick = () => $("file").click();
  $("file").onchange = () => { sendImage($("file").files[0]); $("file").value = ""; };
  text.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) { e.preventDefault(); sendImage(item.getAsFile()); }
  });
  feed.addEventListener("dragover", (e) => e.preventDefault());
  feed.addEventListener("drop", (e) => { e.preventDefault(); sendImage(e.dataTransfer.files[0]); });

  $("mode").onclick = () => {
    S.imagine = !S.imagine;
    $("mode").setAttribute("aria-pressed", S.imagine);
    form.classList.toggle("imagine", S.imagine);
    text.placeholder = S.imagine ? "Describe a picture" : "Message";
    const canvas = S.models.find((m) => m.kind === "canvas" && (m.id == S.pick.canvas || !S.pick.canvas));
    $("hint").textContent = S.imagine
      ? (canvas?.classes?.length ? `I can draw: ${canvas.classes.join(", ")}.` : "No image creation model is published yet.")
      : "Enter to send, Shift+Enter for a new line. Paste or attach a picture to analyze it.";
    text.focus();
    paintUsage();
  };
  function autosize() { text.style.height = "auto"; text.style.height = Math.min(200, text.scrollHeight) + "px"; }
  text.addEventListener("input", autosize);
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });

  // ------------------------------------------------------------------ usage
  function showUsage(u) { if (S.usage) { S.usage.usage = u; paintUsage(); } }
  async function refreshUsage() {
    try { S.usage = await api("/api/usage"); } catch { return; }
    $("plan-link").textContent = S.usage.plan;
    paintUsage();
    const hintBase = "Enter to send, Shift+Enter for a new line. Paste or attach a picture to analyze it.";
    if (!S.imagine) $("hint").innerHTML = esc(hintBase) + (S.usage.learning && S.usage.train_ok && user.role !== "admin"
      ? ` Your chats help train Deer AI (<a href="plans.html#privacy">change</a>).` : "");
    const tb = $("test-banner");
    if (S.usage.test) {
      tb.innerHTML = `<span>Test mode: you're using Deer AI like a user on the <b>${esc(S.usage.plan)}</b> plan.</span>
        <a class="btn" href="plans.html#test">Change</a><button class="btn" id="test-stop">Stop</button>`;
      tb.classList.remove("hidden");
      $("test-stop").onclick = async () => { await api("/api/me/test-mode", { method: "PUT", body: { on: false } }); location.reload(); };
    } else tb.classList.add("hidden");
  }
  function paintUsage() {
    const el = $("usage");
    if (!S.usage || S.usage.unlimited_admin) { el.innerHTML = ""; return; }
    const kind = S.imagine ? "imagine" : "chat";
    const u = S.usage.usage[kind], word = kind === "chat" ? "messages" : "pictures";
    if (u.limit < 0) { el.innerHTML = `<span class="muted">Unlimited ${word} on ${esc(S.usage.plan)}</span>`; return; }
    const left = Math.max(0, u.limit - u.used), pct = u.limit ? (left / u.limit) * 100 : 0;
    const when = u.resets_at ? `more in ${until(u.resets_at)}` : `per ${u.window_hours} h`;
    el.innerHTML = `<div class="u-bar ${pct <= 20 ? "low" : ""}"><div style="width:${pct}%"></div></div>
      <span>${left ? `<b>${left}</b>/${u.limit} ${word} left` : `<b>Out of ${word}</b>`} · ${when}</span>
      <a href="plans.html">${left && pct > 20 ? "Usage" : "Get more"}</a>`;
  }
  setInterval(paintUsage, 60000);

  // ----------------------------------------------------------- saved + export
  $("saved-btn").onclick = async () => {
    closeSide();
    try { S.msgs = await api("/api/saved"); } catch (e) { return toast(e.message); }
    S.view = "saved"; S.cid = null;
    $("title").textContent = "Saved";
    render(); loadConvs();
  };
  $("export").onclick = () => {
    const title = $("title").textContent || "Deer AI chat";
    const lines = [`# ${title}`, ""];
    for (const m of S.msgs) {
      const who = m.role === "user" ? "**You:**" : "**Deer AI:**";
      if (m.kind === "text" || m.kind === "imagine") lines.push(`${who} ${m.kind === "imagine" ? "(imagine) " : ""}${m.content}`);
      else if (m.kind === "image") lines.push(`${who} [sent a picture]`);
      else if (m.kind === "images") lines.push(`${who} [drew ${m.meta.images?.length || 0} pictures: ${(m.meta.images || []).map((i) => i.label).join(", ")}]`);
      else if (m.kind === "analysis") lines.push(`${who} ${m.meta.predictions ? `Looks like ${m.meta.predictions[0].label} (${Math.round(m.meta.predictions[0].confidence * 100)}%)` : "Image measurements"}`);
      lines.push("");
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" }));
    a.download = title.replace(/[^\w\- ]+/g, "").trim().slice(0, 60) + ".md";
    a.click();
  };

  render();
  refreshUsage();
  await Promise.all([loadModels(), loadConvs()]);
  text.focus();
})();
