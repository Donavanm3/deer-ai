(async () => {
  const { api, esc, toast, gate, logout, MARK } = Deer;
  const root = document.getElementById("root");
  const user = await gate(root);
  if (user.role !== "admin") {
    root.innerHTML = `<div class="gate"><div><h1>Admins only</h1><p class="muted">This account can chat but can't build models.</p><a class="btn" href="./">Go to chat</a></div></div>`;
    return;
  }

  const KIND = { chat: "Chat", vision: "Image analysis", canvas: "Image creation" };
  const METRIC = { chat: "Validation loss", vision: "Validation accuracy", canvas: "Validation reconstruction" };
  const S = { tab: "models", models: [], datasets: [], presets: null, sel: null, detail: null, timer: null };

  const I = (d) => `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
  const PAGES = {
    home: ["Home", "Your setup, at a glance", I('<path d="M3 11 12 4l9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1v-9Z"/>')],
    models: ["Models", "Create, train and publish your models", I('<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>')],
    datasets: ["Datasets", "Upload text, code and pictures to train on", I('<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>')],
    build: ["Build dataset", "Combine and filter data into a better dataset", I('<path d="M4 6h16M7 12h10M10 18h4"/>')],
    bench: ["Benchmarks", "Test your models side by side", I('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>')],
    learning: ["Learning", "What users teach Deer AI", I('<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10Z"/>')],
    api: ["API", "Keys, prices and usage for developers", I('<path d="M8 9l-4 3 4 3M16 9l4 3-4 3M13.5 5l-3 14"/>')],
    plans: ["Plans", "Pricing, limits and PayPal", I('<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20"/>')],
    updates: ["Updates", "Update Deer AI from your phone: upload a zip or use GitHub", I('<path d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5"/>')],
    people: ["People", "Accounts and sign-ups", I('<circle cx="9" cy="8" r="4"/><path d="M2 21c0-3.9 3.1-7 7-7s7 3.1 7 7M17 11a3 3 0 1 0 0-6M22 21c0-2.8-1.6-5.2-4-6.3"/>')],
  };
  root.innerHTML = `
  <div class="admin-shell">
    <aside class="anav">
      <div class="mark">${MARK}<span>Deer AI</span></div>
      <nav role="tablist" aria-label="Admin sections">${Object.entries(PAGES).map(([k, [t, , ic]]) =>
        `<button role="tab" data-tab="${k}">${ic}<span>${t}</span></button>`).join("")}</nav>
      <div class="anav-foot"><a class="btn" href="./">Open chat</a><button class="btn ghost" id="logout">Sign out</button></div>
    </aside>
    <main class="admin">
      <div class="page-head"><h1 id="page-title"></h1><p id="page-sub" class="muted"></p></div>
      <section id="view"></section>
    </main>
  </div>`;
  document.getElementById("logout").onclick = logout;
  const view = document.getElementById("view");
  document.querySelector(".anav nav").onclick = (e) => {
    const b = e.target.closest("[data-tab]");
    if (b) show(b.dataset.tab);
  };

  function show(tab) {
    S.tab = tab;
    document.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === tab));
    document.getElementById("page-title").textContent = PAGES[tab][0];
    document.getElementById("page-sub").textContent = PAGES[tab][1];
    history.replaceState(null, "", "#" + tab);
    clearTimeout(S.timer);
    view.innerHTML = `<div class="loading">Loading…</div>`;
    ({ home: showHome, models: showModels, datasets: showDatasets, build: showBuild, bench: showBench, learning: showLearning, api: showApi, updates: showUpdates,
       plans: showPlans, people: showPeople })[tab]();
  }

  const fmt = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
  const fmtParams = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : n);

  // =============================================================== models
  async function showModels() {
    view.innerHTML = `<div class="split">
      <div>
        <div class="panel"><h2>New model</h2>
          <form id="create">
            <label for="m-name">Name</label><input id="m-name" type="text" required maxlength="80" placeholder="Deer Chat v1">
            <div class="grid2">
              <div><label for="m-kind">Type</label><select id="m-kind">
                <option value="chat">Chat</option><option value="vision">Image analysis</option><option value="canvas">Image creation</option></select></div>
              <div><label for="m-preset">Size</label><select id="m-preset"></select></div>
            </div>
            <label for="m-ds">Training data</label><select id="m-ds"></select>
            <div class="muted" id="m-sum" style="font-size:14px;margin-top:8px"></div>
            <h3 class="sub">Learn from your other models (optional)</h3>
            <label for="m-init">Start from</label><select id="m-init"></select>
            <div class="muted" style="font-size:13px;margin-top:4px">Copies what a trained model already knows, so training starts ahead. Parts that don't fit (a bigger model's extra layers) start fresh.</div>
            <div id="m-teach-wrap"><label for="m-teacher">Teacher</label><select id="m-teacher"></select>
              <div class="muted" style="font-size:13px;margin-top:4px">While training, it also learns from the teacher's answers. A teacher can be bigger or a different size.</div>
              <label for="m-str">How much to follow the teacher: <span id="m-str-v">50%</span></label>
              <input id="m-str" type="range" min="10" max="90" step="10" value="50" style="width:100%;accent-color:var(--fawn)"></div>
            <details><summary>Advanced settings</summary>
              <label for="m-cfg">Overrides as JSON, for example {"n_layer": 8, "lr": 0.0005}</label>
              <textarea id="m-cfg" rows="3" placeholder="{}"></textarea></details>
            <div class="err" id="m-err" role="alert"></div>
            <button class="btn primary">Create model</button>
          </form></div>
        <div class="panel"><h2>Your models</h2><ul class="list" id="mlist"></ul></div>
      </div>
      <div id="detail"></div>
    </div>`;
    if (!S.presets) S.presets = await api("/api/admin/presets");
    [S.datasets] = await Promise.all([api("/api/admin/datasets"), refreshModels()]);
    if (S.tab !== "models") return;  // the admin switched tabs while this was loading
    const kindSel = document.getElementById("m-kind"), presetSel = document.getElementById("m-preset");
    const trained = S.models.filter((m) => m.trained);
    const fillTeach = () => {
      const k = kindSel.value;
      const opts = trained.filter((m) => m.kind === k).map((m) => `<option value="${m.id}">${esc(m.name)} (${fmtParams(m.params)}, step ${m.step})</option>`).join("");
      document.getElementById("m-init").innerHTML = `<option value="">Nothing, start fresh</option>${opts}`;
      document.getElementById("m-teacher").innerHTML = `<option value="">No teacher</option>${opts}`;
      document.getElementById("m-teach-wrap").classList.toggle("hidden", k === "canvas");
    };
    document.getElementById("m-str").oninput = (e) => (document.getElementById("m-str-v").textContent = e.target.value + "%");
    const PRESET_NAME = { tiny: "Tiny (fast)", small: "Small", medium: "Medium", "code-small": "Code, small (reads more)", "code-medium": "Code, medium (reads the most)" };
    const fillPresets = () => {
      const keep = presetSel.value;
      presetSel.innerHTML = Object.keys(S.presets.presets[kindSel.value]).map((k) => `<option value="${k}">${PRESET_NAME[k] || k}</option>`).join("");
      presetSel.value = S.presets.presets[kindSel.value][keep] ? keep : "small";
    };
    fillPresets();
    const fillDs = () => {
      const need = S.presets.dataset_kind[kindSel.value];
      const ds = S.datasets.filter((d) => d.kind === need);
      document.getElementById("m-ds").innerHTML = ds.length
        ? ds.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("")
        : `<option value="">Upload a ${need} dataset first</option>`;
      const p = S.presets.presets[kindSel.value][presetSel.value];
      document.getElementById("m-sum").textContent = Object.entries(p).map(([k, v]) => `${k.replace(/_/g, " ")} ${Array.isArray(v) ? v.join("/") : v}`).join(", ");
    };
    kindSel.onchange = () => { fillPresets(); fillTeach(); fillDs(); };
    presetSel.onchange = fillDs;
    fillTeach();
    if (S.prefill) {  // arriving from "Train a model on it"
      kindSel.value = S.prefill.kind; fillPresets(); presetSel.value = S.prefill.preset;
      fillTeach(); fillDs();
      document.getElementById("m-ds").value = S.prefill.dataset_id;
      document.getElementById("m-name").value = S.prefill.name;
      document.getElementById("m-name").focus();
      S.prefill = null;
    } else fillDs();
    document.getElementById("create").onsubmit = async (e) => {
      e.preventDefault();
      const err = document.getElementById("m-err");
      err.textContent = "";
      try {
        const raw = document.getElementById("m-cfg").value.trim();
        let config = {};
        if (raw) { try { config = JSON.parse(raw); } catch { throw new Error("Advanced settings must be valid JSON."); } }
        const ds = document.getElementById("m-ds").value;
        if (!ds) throw new Error("Pick a dataset. Upload one in the Datasets tab.");
        const r = await api("/api/admin/models", { body: {
          name: document.getElementById("m-name").value.trim(), kind: kindSel.value,
          preset: presetSel.value, dataset_id: +ds, config,
          init_from: +document.getElementById("m-init").value || null,
          teacher: kindSel.value !== "canvas" ? (+document.getElementById("m-teacher").value || null) : null,
          teacher_strength: +document.getElementById("m-str").value / 100 } });
        toast(`Model created with ${fmtParams(r.params)} parameters.`);
        e.target.reset(); fillDs();
        S.sel = r.id;
        await refreshModels();
      } catch (ex) { err.textContent = ex.message; }
    };
  }

  async function refreshModels() {
    if (S.tab !== "models") return;
    S.models = await api("/api/admin/models");
    const list = document.getElementById("mlist");
    if (!list) return;
    list.innerHTML = S.models.map((m) => `<li><button class="row ${m.id === S.sel ? "sel" : ""}" data-id="${m.id}">
      <span class="name">${esc(m.name)}</span><span class="chip ${m.status}">${m.status}</span>
      <span class="sub">${KIND[m.kind]}, ${fmtParams(m.params)} params, step ${m.step}${m.published ? ", published" : ""}${m.is_default ? ", default" : ""}</span>
      </button></li>`).join("") || `<li class="muted" style="padding:10px 0">Create your first model above.</li>`;
    list.onclick = (e) => { const b = e.target.closest("[data-id]"); if (b) { S.sel = +b.dataset.id; refreshModels(); } };
    if (!S.sel && S.models.length) S.sel = S.models[0].id;
    await renderDetail();
    const active = S.models.some((m) => m.status === "training" || m.status === "queued");
    clearTimeout(S.timer);
    S.timer = setTimeout(refreshModels, active ? 2000 : 15000);
  }

  async function renderDetail() {
    const box = document.getElementById("detail");
    if (!S.sel || !S.models.some((m) => m.id === S.sel)) { box.innerHTML = ""; S.sel = null; return; }
    const m = await api(`/api/admin/models/${S.sel}`);
    S.detail = m;
    const busy = m.status === "training" || m.status === "queued";
    const last = m.log[m.log.length - 1];
    const pct = m.target_steps ? Math.min(100, (m.step / m.target_steps) * 100) : 0;
    const metricVal = last?.metric == null ? "–" : m.kind === "vision" ? Math.round(last.metric * 100) + "%" : fmt(last.metric);
    const keepSteps = document.getElementById("steps")?.value;
    box.innerHTML = `<div class="panel">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><h2 style="flex:1">${esc(m.name)}</h2><span class="chip ${m.status}">${m.status}</span></div>
      <div class="muted">${KIND[m.kind]} model${m.meta.started_from ? `, started from ${esc(m.meta.started_from.from)} (${m.meta.started_from.copied} of ${m.meta.started_from.total} parts copied)` : ""}${m.meta.teacher ? `, taught by ${esc(m.meta.teacher)}` : ""}</div>
      <div class="stats">
        <div><b>${m.step}${m.status === "training" ? " / " + m.target_steps : ""}</b>steps</div>
        <div><b>${fmt(last?.loss)}</b>training loss</div>
        <div><b>${metricVal}</b>${METRIC[m.kind].toLowerCase()}</div>
        <div><b>${fmtParams(m.params)}</b>parameters</div>
        ${m.meta.steps_per_sec ? `<div><b>${m.meta.steps_per_sec}</b>steps per second</div>` : ""}
      </div>
      ${m.status === "training" ? `<div class="progress" aria-label="Training progress"><div style="width:${pct}%"></div></div>` : ""}
      ${m.status === "queued" ? `<div class="wait-note">${m.queue_position ? `Waiting in line (#${m.queue_position}). ${esc(m.wait_reason || "")} It starts by itself when there's room.` : "Starting…"}</div>` : ""}
      ${m.status === "training" && m.meta.threads ? `<div class="muted" style="font-size:13px;margin-top:6px">Using ${m.meta.threads} CPU thread${m.meta.threads > 1 ? "s" : ""}</div>` : ""}
      ${m.error ? `<div class="err">${esc(m.error)}</div>` : ""}
      <div class="chart-wrap"><canvas id="chart" aria-label="Training chart. Hover or touch to see values."></canvas></div>
      <div class="muted" style="font-size:13px;margin-top:4px"><span style="color:var(--fawn)">━</span> training loss &nbsp; <span style="color:var(--lichen)">━</span> ${METRIC[m.kind].toLowerCase()}</div>
      <div class="actions" style="align-items:end">
        <div style="width:140px"><label for="steps" style="margin-top:0">Steps to train</label><input id="steps" type="number" min="1" max="1000000" value="${keepSteps || (m.kind === "canvas" ? 1000 : 2000)}"></div>
        ${busy ? `<button class="btn" id="stop">${m.status === "queued" ? "Remove from queue" : "Stop and save"}</button>`
               : `<button class="btn primary" id="train">${m.step ? "Continue training" : "Start training"}</button>`}
      </div>
      <div class="actions">
        <label class="toggle"><input type="checkbox" id="pub" ${m.published ? "checked" : ""} ${m.trained ? "" : "disabled"}> Published for users</label>
        ${m.is_default ? `<span class="chip">Default ${KIND[m.kind].toLowerCase()} model</span>` : `<button class="btn" id="def">Make default</button>`}
      </div>
    </div>
    ${m.kind !== "vision" ? `<div class="panel"><h2>Test output</h2>
      <div class="sample" id="sample">${m.kind === "chat"
        ? (m.meta.preview_text ? `User: Hello!\nDeer:${esc(m.meta.preview_text)}` : `<span class="muted">Samples appear here while it trains.</span>`)
        : (m.preview_url ? `<img src="${m.preview_url}" alt="Sample images from the latest checkpoint">` : `<span class="muted">Samples appear here while it trains.</span>`)}</div>
      <div class="actions"><button class="btn" id="try" ${m.trained ? "" : "disabled"}>Generate a new sample</button></div></div>` : ""}
    <div class="panel"><h2>Learn from users</h2>
      <p class="muted" style="font-size:14px;margin:4px 0 0">${m.kind === "chat" ? "Approved liked replies and corrections" : "Approved image labels from users"} are added to this model's dataset.
        With auto-learn on, the model fine-tunes itself when enough new lessons arrive. Each run is checked on held-out data and undone automatically if the model got worse.</p>
      <div class="stats"><div><b>${m.new_lessons}</b>new lessons waiting</div>
        ${m.meta.last_learn ? `<div><b>${esc(m.meta.last_learn.outcome)}</b>last learning run${m.meta.last_learn.at ? ", " + new Date(m.meta.last_learn.at * 1000).toLocaleString() : ""}</div>` : ""}</div>
      <form id="learnf" class="grid2" style="align-items:end">
        <label class="toggle" style="grid-column:1/-1;margin:0 0 4px"><input type="checkbox" id="al" ${m.auto_learn ? "checked" : ""}> Auto-learn</label>
        <div><label for="al-min">Start after this many new lessons</label><input id="al-min" type="number" min="1" value="${m.learn_min}"></div>
        <div><label for="al-steps">Steps per learning run</label><input id="al-steps" type="number" min="10" value="${m.learn_steps}"></div>
        <div class="actions" style="grid-column:1/-1"><button class="btn primary">Save</button>
          <button type="button" class="btn" id="learn-now" ${busy || !m.trained ? "disabled" : ""}>Learn now</button>
          ${m.has_backup ? `<button type="button" class="btn danger" id="undo" ${busy ? "disabled" : ""}>Undo last training run</button>` : ""}</div>
      </form></div>
    <div class="panel"><h2>Manage</h2>
      <div class="muted" style="font-size:14px">Dataset: ${esc(S.models.find((x) => x.id === m.id)?.dataset_name || m.dataset_id)}. Settings: ${esc(JSON.stringify(m.config))}</div>
      <div class="actions">
        <button class="btn" id="export" ${m.trained ? "" : "disabled"}>Download weights</button>
        <button class="btn" id="rename">Rename</button>
        <button class="btn danger" id="reset" ${busy ? "disabled" : ""}>Reset training</button>
        <button class="btn danger" id="delete" ${busy ? "disabled" : ""}>Delete model</button>
      </div></div>`;
    drawChart(document.getElementById("chart"), m.log, m.kind);

    const $ = (id) => document.getElementById(id);
    const act = async (fn, okMsg) => { try { await fn(); if (okMsg) toast(okMsg); } catch (e) { toast(e.message); } await refreshModels(); };
    if ($("train")) $("train").onclick = () => act(async () => {
      const r = await api(`/api/admin/models/${m.id}/train`, { body: { steps: +$("steps").value } });
      toast(r.message || "Training started.");
    });
    if ($("stop")) $("stop").onclick = () => act(() => api(`/api/admin/models/${m.id}/stop`, { method: "POST" }), "Stopping after the current step.");
    $("pub").onchange = (e) => act(() => api(`/api/admin/models/${m.id}`, { method: "PATCH", body: { published: e.target.checked } }),
      e.target.checked ? "Published." : "Unpublished.");
    if ($("def")) $("def").onclick = () => act(() => api(`/api/admin/models/${m.id}`, { method: "PATCH", body: { is_default: true } }), "Default updated.");
    $("learnf").onsubmit = (e) => { e.preventDefault(); act(() => api(`/api/admin/models/${m.id}/learning`, { method: "PATCH",
      body: { auto_learn: $("al").checked, learn_min: +$("al-min").value, learn_steps: +$("al-steps").value } }), "Learning settings saved."); };
    $("learn-now").onclick = () => act(() => api(`/api/admin/models/${m.id}/learn`, { method: "POST" }), "Learning run started.");
    if ($("undo")) $("undo").onclick = () => confirm("Undo the most recent training run? The model goes back to the checkpoint before it.") &&
      act(() => api(`/api/admin/models/${m.id}/rollback`, { method: "POST" }), "Rolled back.");
    $("rename").onclick = () => {
      const n = prompt("New name", m.name);
      if (n && n.trim()) act(() => api(`/api/admin/models/${m.id}`, { method: "PATCH", body: { name: n.trim() } }), "Renamed.");
    };
    $("reset").onclick = () => confirm(`Reset "${m.name}"? Its weights and training history are erased.`) &&
      act(() => api(`/api/admin/models/${m.id}/reset`, { method: "POST" }), "Training reset.");
    $("delete").onclick = () => confirm(`Delete "${m.name}" permanently?`) &&
      act(async () => { await api(`/api/admin/models/${m.id}`, { method: "DELETE" }); S.sel = null; }, "Model deleted.");
    $("export").onclick = async () => {
      const res = await api(`/api/admin/models/${m.id}/export`, { raw: true });
      if (!res.ok) return toast("No weights to download yet.");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(await res.blob());
      a.download = `deer-${m.kind}-${m.id}-step${m.step}.npz`;
      a.click();
    };
    if ($("try")) $("try").onclick = async () => {
      $("try").disabled = true;
      try {
        const r = await api(`/api/admin/models/${m.id}/sample`, { method: "POST" });
        $("sample").innerHTML = r.text != null ? `User: Hello!\nDeer: ${esc(r.text)}`
          : `<div class="gen-grid">${r.images.map((i) => `<figure><img src="${i.data_url}" alt="${esc(i.label)}"><figcaption>${esc(i.label)}</figcaption></figure>`).join("")}</div>`;
      } catch (e) { toast(e.message); }
      $("try").disabled = false;
    };
  }

  function drawChart(cv, log, kind) {
    // Line chart of training loss (left scale) and the validation score (right scale).
    // Hover or touch it to read the exact numbers at any step.
    const wrap = cv.parentElement;
    const css = getComputedStyle(document.documentElement);
    const col = (v) => css.getPropertyValue(v).trim();
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const font = "12px " + (col("--body") || "sans-serif");
    if (log.length < 2) {
      g.font = font; g.fillStyle = col("--mute");
      g.fillText("The chart fills in as training runs.", 8, h / 2);
      return;
    }
    const pad = { l: 48, r: 48, t: 12, b: 24 };
    const xs = log.map((p) => p.step), x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = (st) => pad.l + ((st - x0) / Math.max(1, x1 - x0)) * (w - pad.l - pad.r);
    const fmtV = (key, v) => (v == null ? "–" : kind === "vision" && key === "metric" ? Math.round(v * 100) + "%" : v >= 100 ? v.toFixed(0) : v.toFixed(3));
    const series = ["loss", "metric"].map((key) => {
      const pts = log.filter((p) => p[key] != null);
      if (pts.length < 2) return null;
      const vals = pts.map((p) => p[key]);
      let lo = Math.min(...vals), hi = Math.max(...vals);
      if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
      return { key, pts, lo, hi, color: col(key === "loss" ? "--fawn" : "--lichen"),
               Y: (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b) };
    }).filter(Boolean);

    function paint(hover) {
      g.clearRect(0, 0, w, h);
      g.font = font;
      g.strokeStyle = col("--line"); g.lineWidth = 1;
      for (let i = 0; i <= 3; i++) {  // light grid
        const y = pad.t + (i / 3) * (h - pad.t - pad.b);
        g.globalAlpha = i === 3 ? 1 : 0.35;
        g.beginPath(); g.moveTo(pad.l, y); g.lineTo(w - pad.r, y); g.stroke();
      }
      g.globalAlpha = 1;
      g.fillStyle = col("--mute"); g.textAlign = "left"; g.fillText(`step ${x0}`, pad.l, h - 6);
      g.textAlign = "right"; g.fillText(`step ${x1}`, w - pad.r, h - 6);
      series.forEach((sr, si) => {
        g.strokeStyle = sr.color; g.lineWidth = 2; g.lineJoin = "round";
        g.beginPath();
        sr.pts.forEach((p, i) => (i ? g.lineTo(X(p.step), sr.Y(p[sr.key])) : g.moveTo(X(p.step), sr.Y(p[sr.key]))));
        g.stroke();
        g.fillStyle = sr.color; g.textAlign = si ? "left" : "right";
        const x = si ? w - pad.r + 6 : pad.l - 6;
        g.fillText(fmtV(sr.key, sr.hi), x, pad.t + 10);
        g.fillText(fmtV(sr.key, sr.lo), x, h - pad.b);
      });
      if (hover) {
        const hx = X(hover.step);
        g.strokeStyle = col("--mute"); g.setLineDash([4, 4]); g.lineWidth = 1;
        g.beginPath(); g.moveTo(hx, pad.t); g.lineTo(hx, h - pad.b); g.stroke(); g.setLineDash([]);
        series.forEach((sr) => {
          if (hover[sr.key] == null) return;
          g.fillStyle = col("--bark"); g.strokeStyle = sr.color; g.lineWidth = 2;
          g.beginPath(); g.arc(hx, sr.Y(hover[sr.key]), 4.5, 0, 7); g.fill(); g.stroke();
        });
      }
    }
    paint(null);

    let tip = wrap.querySelector(".chart-tip");
    if (!tip) { tip = document.createElement("div"); tip.className = "chart-tip"; tip.hidden = true; wrap.appendChild(tip); }
    const label = (key) => (key === "loss" ? "Training loss" : METRIC[kind]);
    const nearest = (px) => log.reduce((a, b) => (Math.abs(X(b.step) - px) < Math.abs(X(a.step) - px) ? b : a));
    const move = (e) => {
      const r = cv.getBoundingClientRect();
      const p = nearest(e.clientX - r.left);
      paint(p);
      tip.innerHTML = `<b>Step ${p.step}</b>` + series.map((sr) =>
        `<div><i style="background:${sr.color}"></i>${esc(label(sr.key))}: <b>${fmtV(sr.key, p[sr.key])}</b></div>`).join("");
      tip.hidden = false;
      const left = Math.min(Math.max(X(p.step) + 12, 0), w - tip.offsetWidth - 4);
      tip.style.left = (X(p.step) + 12 + tip.offsetWidth > w ? X(p.step) - tip.offsetWidth - 12 : left) + "px";
      tip.style.top = "8px";
    };
    cv.onpointermove = move;
    cv.onpointerdown = move;
    cv.onpointerleave = () => { tip.hidden = true; paint(null); };
  }

  // ============================================================= datasets
  async function showDatasets() {
    view.innerHTML = `<div class="split">
      <div class="panel"><h2>Upload training data</h2>
        <form id="up">
          <label for="d-name">Name</label><input id="d-name" type="text" required maxlength="80" placeholder="Forest photos">
          <label for="d-kind">Kind</label><select id="d-kind">
            <option value="text">Text, for chat models</option><option value="images">Images, for analysis and creation models</option></select>
          <label for="d-files">Files</label><input id="d-files" type="file" multiple accept=".txt,.md,.csv,.json">
          <p class="muted" id="d-help" style="font-size:14px"></p>
          <div class="err" id="d-err" role="alert"></div>
          <button class="btn primary" id="d-btn">Upload dataset</button>
        </form></div>
      <div class="panel"><h2>Datasets</h2><ul class="list" id="dlist"></ul></div></div>`;
    const kind = document.getElementById("d-kind"), files = document.getElementById("d-files");
    const help = () => {
      const t = kind.value === "text";
      files.multiple = t;
      files.accept = t ? ".txt,.md,.csv,.json,.zip,.py,.js,.ts,.jsx,.tsx,.html,.css,.cs,.lua,.gd,.java,.kt,.cpp,.h,.c,.rs,.go,.sh,.glsl,.hlsl,.shader,.mcfunction,.yml,.yaml,.toml,.xml,.sql,.php,.rb,.swift,.dart" : ".zip";
      document.getElementById("d-help").innerHTML = t
        ? `Text files, code files, or a <b>.zip of a whole project</b> (a game, mod or website; art and build folders are skipped).
           For chat, write conversations like this. A reply can span many lines, code included, until the next <code>User:</code> line:
           <pre class="sample">User: How do I make a player jump in Unity?
Deer: Use an impulse on the Rigidbody2D:
\`\`\`csharp
rb.AddForce(Vector2.up * 7f, ForceMode2D.Impulse);
\`\`\`</pre>
           More text trains a better model; aim for at least 1 MB.`
        : `One .zip with a folder per subject, named the way you want it labeled:<br><code>deer/…jpg</code>, <code>fox/…jpg</code>, <code>oak leaf/…png</code><br>Aim for 100+ images per folder. Analysis models need at least two folders.`;
    };
    kind.onchange = help; help();
    document.getElementById("up").onsubmit = async (e) => {
      e.preventDefault();
      const err = document.getElementById("d-err"), btn = document.getElementById("d-btn");
      err.textContent = "";
      if (!files.files.length) return (err.textContent = "Choose at least one file.");
      const fd = new FormData();
      fd.append("name", document.getElementById("d-name").value.trim());
      fd.append("kind", kind.value);
      [...files.files].forEach((f) => fd.append("files", f));
      btn.disabled = true; btn.textContent = "Uploading…";
      try {
        await api("/api/admin/datasets", { body: fd });
        toast("Dataset uploaded.");
        e.target.reset(); help(); loadDs();
      } catch (ex) { err.textContent = ex.message; }
      btn.disabled = false; btn.textContent = "Upload dataset";
    };
    const loadDs = async () => {
      S.datasets = await api("/api/admin/datasets");
      const el = document.getElementById("dlist");
      el.innerHTML = S.datasets.map((d) => {
        const s = d.stats;
        const sub = d.kind === "text"
          ? `${(s.bytes / 1024).toFixed(0)} KB of text, ${s.lines} lines${s.chat_format ? ", chat format" : ""}`
          : `${s.images} images in ${s.classes.length} classes: ${s.classes.join(", ")}${s.skipped ? `, ${s.skipped} skipped` : ""}`;
        return `<li style="display:flex;gap:12px;align-items:center;padding:12px 0"><div style="flex:1"><div class="name" style="font-weight:700">${esc(d.name)}</div>
          <div class="muted" style="font-size:14px">#${d.id}, ${esc(sub)}${d.used_by ? `, used by ${d.used_by} model(s)` : ""}</div></div>
          <button class="btn danger" data-del="${d.id}" ${d.used_by ? "disabled" : ""}>Delete</button></li>`;
      }).join("") || `<li class="muted" style="padding:10px 0">No datasets yet. Upload one to start training.</li>`;
      el.onclick = async (e) => {
        const b = e.target.closest("[data-del]");
        if (!b || !confirm("Delete this dataset?")) return;
        try { await api(`/api/admin/datasets/${b.dataset.del}`, { method: "DELETE" }); loadDs(); } catch (ex) { toast(ex.message); }
      };
    };
    loadDs();
  }




  // ======================================================== build dataset
  async function showBuild() {
    const [all, models] = await Promise.all([api("/api/admin/datasets"), api("/api/admin/models")]);
    if (S.tab !== "build") return;  // the admin switched tabs while this was loading
    const ready = (k) => models.filter((m) => m.kind === k && m.trained);
    const pick = (id, k, cur, none) => `<select id="${id}"><option value="">${none}</option>${ready(k).map((m) =>
      `<option value="${m.id}" ${+cur === m.id ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select>`;
    const B = S.build || (S.build = { kind: "text", ids: [], f: {}, classes: {}, preview: null });
    const T = B.kind === "text";
    const sources = all.filter((d) => d.kind === B.kind);
    B.ids = B.ids.filter((id) => sources.some((d) => d.id === id));
    const f = Object.assign(T
      ? { include_base: true, include_learned: true, dedupe: true, remove_junk: true, chat_only: false, min_chars: 0, max_chars: 0, must_contain: "", block_words: "", max_units: 0, shuffle: false }
      : { dedupe: true, include_user: true, min_size: 0, min_per_class: 0, max_per_class: 0 }, B.f);
    B.f = f;
    const size = (d) => d.kind === "text" ? `${Math.round(d.stats.bytes / 1024)} KB${d.stats.lessons ? `, ${d.stats.lessons} user lessons` : ""}`
      : `${d.stats.images} images, ${d.stats.classes.length} classes`;
    const chk = (k, label) => `<label class="toggle" style="margin:8px 0"><input type="checkbox" data-f="${k}" ${f[k] ? "checked" : ""}> ${label}</label>`;
    const num = (k, label) => `<div><label for="bf-${k}">${label}</label><input id="bf-${k}" type="number" min="0" data-f="${k}" value="${f[k] || 0}"></div>`;
    const txt = (k, label, ph) => `<label for="bf-${k}">${label}</label><input id="bf-${k}" type="text" data-f="${k}" value="${esc(f[k] || "")}" placeholder="${ph}">`;
    const srcCls = B.preview?.source_classes || {};

    view.innerHTML = `<div class="split">
      <div class="panel"><h2>Build a dataset</h2>
        <p class="muted" style="font-size:14px;margin:4px 0 0">Combine your datasets and what users taught them, filter out the junk, and save it as a new dataset for a bigger, smarter model.</p>
        <div class="seg" role="radiogroup" aria-label="Data type">
          <button type="button" role="radio" aria-checked="${T}" data-kind="text">Text</button>
          <button type="button" role="radio" aria-checked="${!T}" data-kind="images">Images</button></div>
        <label>Combine these datasets</label>
        <div class="checks">${sources.map((d) => `<label class="toggle"><input type="checkbox" data-src="${d.id}" ${B.ids.includes(d.id) ? "checked" : ""}>
          <span>${esc(d.name)} <span class="muted" style="font-size:13px">${esc(size(d))}</span></span></label>`).join("") || `<p class="muted">No ${T ? "text" : "image"} datasets yet.</p>`}</div>
        ${T ? `
          <h3 class="sub">From your users</h3>
          <label class="toggle" style="margin:8px 0"><input type="checkbox" data-u="feedback" ${f.users?.feedback ? "checked" : ""}> Liked replies and corrections (all feedback, reviewed or not)</label>
          <label class="toggle" style="margin:8px 0"><input type="checkbox" data-u="messages" ${f.users?.messages ? "checked" : ""}> Everything users wrote (their messages)</label>
          <label class="toggle" style="margin:8px 0"><input type="checkbox" data-u="conversations" ${f.users?.conversations ? "checked" : ""}> Whole conversations, except replies users disliked</label>
          <p class="muted" style="font-size:13px">Users who opted out are never included. Whole conversations include the model's own replies, so filter hard and prefer feedback.</p>
          <h3 class="sub">Replies written by a model</h3>
          ${pick("g-model", "chat", f.generate?.model_id, "None")}
          <div class="grid2"><div><label for="g-count">How many replies</label><input id="g-count" type="number" min="1" max="2000" value="${f.generate?.count || 200}"></div>
            <div><label for="g-temp">Creativity (0.1 to 1.5)</label><input id="g-temp" type="number" min="0.1" max="1.5" step="0.1" value="${f.generate?.temperature || 0.7}"></div></div>
          <p class="muted" style="font-size:13px">The model answers the questions found in the chosen datasets. Useful for teaching a smaller or fresh model what a good one knows; the filters below still apply, so block anything bad.</p>
          <h3 class="sub">Include</h3>${chk("include_base", "Original text")}${chk("include_learned", "Lessons users taught")}
          <h3 class="sub">Filters</h3>
          ${chk("dedupe", "Remove duplicates")}${chk("remove_junk", "Remove junk, symbols and empty answers")}${chk("chat_only", "Only keep User/Deer conversation pairs")}
          <div class="grid2">${num("min_chars", "Shortest allowed (characters)")}${num("max_chars", "Longest allowed (0 = no limit)")}</div>
          ${txt("block_words", "Remove anything containing", "comma separated, e.g. swear words")}
          ${txt("must_contain", "Only keep what contains", "comma separated, leave empty for all")}
          <div class="grid2">${num("max_units", "Keep at most (0 = all)")}<div>${chk("shuffle", "Shuffle order")}</div></div>`
        : `
          <h3 class="sub">From your users</h3>
          <label class="toggle" style="margin:8px 0"><input type="checkbox" data-u="pictures" ${f.users?.pictures ? "checked" : ""}> Pictures users sent in chat</label>
          <p class="muted" style="font-size:13px">Labeled by their Yes/No feedback where they gave it; the rest go in "unlabeled", which a model below can label.</p>
          <h3 class="sub">Label pictures with a model</h3>
          ${pick("l-model", "vision", f.label?.model_id, "Don't label")}
          <div class="grid2"><div><label for="l-conf">Only when at least this sure (%)</label><input id="l-conf" type="number" min="1" max="100" value="${Math.round((f.label?.min_confidence || 0.8) * 100)}"></div></div>
          <p class="muted" style="font-size:13px">Labels the pictures in an "unlabeled" folder (upload a zip of pictures with no folders to get one). Pictures it's unsure about are left out.</p>
          <h3 class="sub">Pictures drawn by a model</h3>
          ${pick("d-model", "canvas", f.generate?.model_id, "None")}
          <div class="grid2"><div><label for="d-per">Pictures per class</label><input id="d-per" type="number" min="1" max="500" value="${f.generate?.per_class || 20}"></div></div>
          <p class="muted" style="font-size:13px">Gives an analysis model extra examples. Don't use drawn pictures to train a new creation model; it only copies the old one's mistakes.</p>
          <h3 class="sub">Filters</h3>
          ${chk("dedupe", "Remove duplicate pictures")}${chk("include_user", "Include pictures users taught")}
          <div class="grid2">${num("min_size", "Smallest side at least (px)")}${num("min_per_class", "Drop classes with fewer than")}</div>
          <div class="grid2">${num("max_per_class", "Max per class (0 = all; balances classes)")}</div>
          <h3 class="sub">Classes</h3>
          ${Object.keys(srcCls).length ? `<p class="muted" style="font-size:13px;margin:0">Untick to leave a class out. Give two classes the same name to merge them.</p>
          <table class="cls"><tbody>${Object.entries(srcCls).map(([c, n]) => {
            const v = B.classes[c];
            return `<tr><td><input type="checkbox" data-cls-on="${esc(c)}" ${v === "" ? "" : "checked"} aria-label="Include ${esc(c)}"></td>
              <td><input type="text" data-cls="${esc(c)}" value="${esc(v || c)}" aria-label="Name for ${esc(c)}"></td><td class="muted">${n}</td></tr>`;
          }).join("")}</tbody></table>` : `<p class="muted" style="font-size:14px">Pick datasets to see their classes.</p>`}`}
        <div class="actions"><button class="btn primary" id="bp">Preview</button></div>
      </div>
      <div id="bprev">${prevHTML(B)}</div></div>`;

    view.querySelectorAll("[data-kind]").forEach((b) => (b.onclick = () => {
      if (b.dataset.kind !== B.kind) { S.build = { kind: b.dataset.kind, ids: [], f: {}, classes: {}, preview: null }; showBuild(); }
    }));
    const readForm = () => {
      view.querySelectorAll("[data-f]").forEach((el) => {
        f[el.dataset.f] = el.type === "checkbox" ? el.checked : el.type === "number" ? +el.value : el.value;
      });
      B.ids = [...view.querySelectorAll("[data-src]:checked")].map((x) => +x.dataset.src);
      const v = (id) => document.getElementById(id)?.value;
      f.users = {};
      view.querySelectorAll("[data-u]").forEach((el) => { if (el.checked) f.users[el.dataset.u] = true; });
      if (T) f.generate = v("g-model") ? { model_id: +v("g-model"), count: +v("g-count"), temperature: +v("g-temp") } : null;
      else {
        f.label = v("l-model") ? { model_id: +v("l-model"), min_confidence: +v("l-conf") / 100 } : null;
        f.generate = v("d-model") ? { model_id: +v("d-model"), per_class: +v("d-per") } : null;
      }
      view.querySelectorAll("[data-cls]").forEach((el) => {
        const c = el.dataset.cls, on = view.querySelector(`[data-cls-on="${CSS.escape(c)}"]`).checked;
        B.classes[c] = on ? el.value.trim() : "";
      });
    };
    const filters = () => (T ? { ...f } : { ...f, classes: B.classes });
    const doPreview = async () => {
      readForm();
      B.made = null;
      if (!B.ids.length && !f.generate && !Object.keys(f.users).length) { B.preview = null; toast("Pick a dataset, a model or a user source."); return showBuild(); }
      const btn = document.getElementById("bp");
      btn.disabled = true; btn.textContent = "Checking…";
      try { B.preview = await api("/api/admin/curate/preview", { body: { kind: B.kind, dataset_ids: B.ids, filters: filters() } }); }
      catch (e) { toast(e.message); }
      showBuild();
    };
    document.getElementById("bp").onclick = doPreview;
    view.querySelectorAll("[data-src]").forEach((el) => (el.onchange = doPreview));  // refresh classes/counts
    ["l-model", "d-model"].forEach((id) => { const el = document.getElementById(id); if (el) el.onchange = doPreview; });
    const mk = document.getElementById("bmake");
    if (mk) mk.onclick = async () => {
      readForm();
      const name = document.getElementById("bname").value.trim();
      if (!name) return toast("Name the new dataset.");
      mk.disabled = true; mk.textContent = "Building…";
      try {
        const r = await api("/api/admin/curate/build", { body: { kind: B.kind, name, dataset_ids: B.ids, filters: filters() } });
        B.made = { id: r.id, name, preset: r.suggested_preset.preset, why: r.suggested_preset.why };
        toast("Dataset created.");
      } catch (e) { toast(e.message); }
      showBuild();
    };
    view.querySelectorAll("[data-train]").forEach((b) => (b.onclick = () => {
      S.prefill = { kind: b.dataset.train, dataset_id: B.made.id, preset: B.made.preset, name: `${B.made.name} model` };
      S.build = null;
      show("models");
    }));
  }

  function prevHTML(B) {
    const P = B.preview;
    if (B.made) {
      const T = B.kind === "text";
      return `<div class="panel"><h2>"${esc(B.made.name)}" is ready</h2>
        <p>Suggested size: <b>${esc(B.made.preset)}</b>. ${esc(B.made.why)}</p>
        <div class="actions">${T ? `<button class="btn primary" data-train="chat">Train a chat model on it</button>`
          : `<button class="btn primary" data-train="vision">Train an analysis model</button><button class="btn" data-train="canvas">Train a creation model</button>`}</div></div>`;
    }
    if (!P) return `<div class="panel"><h2>Preview</h2><p class="muted">Pick datasets and filters, then press Preview to see exactly what the new dataset will contain.</p></div>`;
    const removed = Object.entries(P.removed);
    return `<div class="panel"><h2>Preview</h2>
      <div class="stats"><div><b>${P.before}</b>${P.kind === "text" ? "lines/pairs found" : "pictures found"}</div>
        <div><b>${P.after}</b>kept</div>${P.kind === "text" ? `<div><b>${Math.round(P.bytes / 1024)} KB</b>of text</div>` : `<div><b>${Object.keys(P.classes).length}</b>classes</div>`}</div>
      ${P.generated_note ? `<p class="muted" style="font-size:14px;margin:0 0 6px">${esc(P.generated_note)}</p>` : ""}
      ${P.labeled_by_model ? `<p class="muted" style="font-size:14px;margin:0 0 6px">${P.labeled_by_model} pictures labeled by the model.</p>` : ""}
      ${removed.length ? `<p class="muted" style="font-size:14px;margin:0">Removed: ${removed.map(([k, n]) => `${n} ${esc(k)}`).join(", ")}.</p>` : ""}
      <p style="font-size:14px">Suggested model size: <b>${esc(P.suggested_preset.preset)}</b>. ${esc(P.suggested_preset.why)}</p>
      ${P.kind === "text"
        ? `<label>Sample of what's kept</label><div class="sample" style="max-height:260px;overflow:auto">${P.sample.map(esc).join("\n")}</div>
           ${P.removed_examples.length ? `<details><summary>Examples of what's removed</summary><div class="sample">${P.removed_examples.map((x) => `[${esc(x.reason)}] ${esc(x.text)}`).join("\n")}</div></details>` : ""}`
        : `<div class="cls-grid">${Object.entries(P.classes).map(([c, v]) => `<div><div class="name"><b>${esc(c)}</b> <span class="muted">${v.count}</span></div>
            <div class="thumbs">${v.samples.map((u) => `<img src="${u}" alt="">`).join("")}</div></div>`).join("")}</div>`}
      <label for="bname">New dataset name</label><input id="bname" type="text" maxlength="80" placeholder="Clean chat v2">
      <div class="actions"><button class="btn primary" id="bmake" ${P.after ? "" : "disabled"}>Create dataset</button></div>
    </div>`;
  }


  // ================================================================= home
  async function showHome() {
    const [o, sys] = await Promise.all([api("/api/admin/overview"), api("/api/admin/system")]);
    if (S.tab !== "home") return;
    const pub = o.published.chat + o.published.vision + o.published.canvas;
    const steps = [
      [(o.datasets.text || 0) + (o.datasets.images || 0) > 0, "Upload training data", "Text or code for chat models, pictures for image models.", "datasets"],
      [o.models > 0, "Create a model", "Pick a type and size. The Code sizes read more at once, which coding needs.", "models"],
      [o.trained > 0, "Train it", "Watch the chart drop. You can stop anytime; progress is saved.", "models"],
      [pub > 0, "Publish it", "Tick Published so users can chat with it.", "models"],
      [o.bench_runs > 0, "Benchmark it", "Score your models on a fixed test and keep the best.", "bench"],
    ];
    const done = steps.filter((x) => x[0]).length;
    const card = (n, label, sub = "") => `<div class="stat-card"><b>${n}</b><span>${label}</span>${sub ? `<small>${sub}</small>` : ""}</div>`;
    view.innerHTML = `
      <div class="stat-cards">
        ${card(`${o.trained}<small>/${o.models}</small>`, "models trained")}
        ${card(pub, "published", `${o.published.chat} chat · ${o.published.vision} analysis · ${o.published.canvas} creation`)}
        ${card(o.users, "users", `${o.subscribers} paying`)}
        ${card(o.pending_lessons, "lessons waiting", o.pending_lessons ? `<a href="#learning" data-go="learning">Review</a>` : "")}
      </div>
      <div class="split">
        <div class="panel"><div class="row-head"><h2>Getting started</h2><span class="muted">${done} of ${steps.length} done</span></div>
          <div class="progress"><div style="width:${(done / steps.length) * 100}%"></div></div>
          <ol class="checklist">${steps.map(([ok, t, d, tab], i) => `<li class="${ok ? "done" : ""}">
            <span class="dot">${ok ? "✓" : i + 1}</span><div><b>${t}</b><p class="muted">${d}</p></div>
            ${ok ? "" : `<button class="btn" data-go="${tab}">Go</button>`}</li>`).join("")}</ol>
          <p class="muted" style="font-size:13px">No starter data yet? On your computer run <code>python tools/make_demo_data.py</code> in the deer-ai folder. It writes coding chats, game sprites and more into <code>examples/</code>. You still train every model yourself.</p>
        </div>
        <div>
          <div class="panel"><div class="row-head"><h2>Training now</h2><span class="muted">${sys.running.length} of ${sys.max_jobs} slots in use</span></div>
            ${o.training.length ? o.training.map((m) => m.status === "queued"
              ? `<div class="train-row"><span>${esc(m.name)}</span><span class="muted" style="font-size:13px">waiting in line</span><small></small></div>`
              : `<div class="train-row"><span>${esc(m.name)}</span><div class="progress"><div style="width:${m.target_steps ? (100 * m.step) / m.target_steps : 0}%"></div></div><small>${m.step}/${m.target_steps}</small></div>`).join("")
            : `<p class="muted">Nothing is training right now.</p>`}
            <div class="sys">
              <div><b>${sys.cpus}</b>CPU cores</div>
              <div><b>${sys.load ?? "–"}</b>load${sys.other_load != null && sys.running.length ? ` (${sys.other_load} other)` : ""}</div>
              <div><b>${sys.mem_available_mb != null ? (sys.mem_available_mb / 1024).toFixed(1) + " GB" : "–"}</b>memory free</div>
            </div>
            <p class="muted" style="font-size:13px;margin:8px 0 0">${sys.can_start ? (sys.running.length ? "Room for another job right now." : "Ready to train.") : esc(sys.reason) + " New jobs wait in line and start by themselves."}</p></div>
          <div class="panel"><h2>Best benchmark scores</h2>${o.leaders.length ? o.leaders.map((l) =>
            `<div class="lead"><span>${esc(l.benchmark)}</span><b>${esc(l.model_name)}</b><span class="score">${l.score}</span></div>`).join("")
            : `<p class="muted">Run a benchmark to compare your models.</p><button class="btn" data-go="bench">Create a benchmark</button>`}</div>
        </div>
      </div>`;
    view.querySelectorAll("[data-go]").forEach((b) => (b.onclick = (e) => { e.preventDefault(); show(b.dataset.go); }));
    S.timer = setTimeout(() => S.tab === "home" && showHome(), o.training.length ? 4000 : 15000);
  }

  // ============================================================ benchmarks
  async function showBench() {
    const [list, models, datasets, ext] = await Promise.all([api("/api/admin/benchmarks"), api("/api/admin/models"),
      api("/api/admin/datasets"), api("/api/admin/external")]);
    if (S.tab !== "bench") return;
    S.ext = ext.models;
    const F = S.benchForm || (S.benchForm = { kind: "chat", source: "dataset" });
    const trained = (k) => models.filter((m) => m.kind === k && m.trained);
    const dsOpts = (k) => datasets.filter((d) => d.kind === k).map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("");
    const kindTxt = { chat: "Chat", vision: "Image analysis", canvas: "Image creation" };
    view.innerHTML = `<div class="split">
      <div class="panel"><h2>New benchmark</h2>
        <p class="muted" style="font-size:14px;margin:4px 0 0">A fixed test. Every model you run on it is scored the same way, so you can see which is best.</p>
        <form id="bf">
          <label for="b-name">Name</label><input id="b-name" type="text" maxlength="80" required placeholder="Unity questions">
          <div class="seg" role="radiogroup" aria-label="Type">${["chat", "vision", "canvas"].map((k) =>
            `<button type="button" role="radio" aria-checked="${F.kind === k}" data-bk="${k}">${kindTxt[k]}</button>`).join("")}</div>
          ${F.kind === "chat" ? `
            <label for="b-src">Questions come from</label><select id="b-src">
              <option value="dataset" ${F.source === "dataset" ? "selected" : ""}>The held-out part of a text dataset</option>
              <option value="custom" ${F.source === "custom" ? "selected" : ""}>Questions I write</option></select>
            ${F.source === "dataset" ? `<label for="b-ds">Dataset</label><select id="b-ds">${dsOpts("text") || "<option value=''>No text datasets yet</option>"}</select>
              <label for="b-size">How many questions</label><input id="b-size" type="number" min="1" max="200" value="30">
              <p class="muted" style="font-size:13px">Uses the last 10% of the dataset, the same part training holds back, so answers aren't simply memorized.</p>`
            : `<label for="b-text">Questions and the answers you expect</label>
              <textarea id="b-text" rows="9" placeholder="User: How do I make a player jump in Unity?&#10;Deer: Add a Rigidbody2D and use AddForce with ForceMode2D.Impulse.&#10;&#10;User: Write a JS game loop&#10;Deer: \`\`\`javascript&#10;function loop(now) { update(); draw(); requestAnimationFrame(loop); }&#10;\`\`\`"></textarea>`}
            <p class="muted" style="font-size:13px">Scored on how closely replies match (word overlap), exact matches, and whether code in replies is well-formed.</p>`
          : F.kind === "vision" ? `
            <label for="b-ds">Labeled pictures to test on</label><select id="b-ds">${dsOpts("images") || "<option value=''>No image datasets yet</option>"}</select>
            <label for="b-size">Pictures per class</label><input id="b-size" type="number" min="1" max="500" value="50">
            <p class="muted" style="font-size:13px">Best with pictures the models didn't train on. Build dataset can split some off.</p>`
          : `
            <label for="b-judge">Judge (an image analysis model)</label><select id="b-judge">${trained("vision").map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("") || "<option value=''>Train an analysis model first</option>"}</select>
            <label for="b-per">Pictures per class</label><input id="b-per" type="number" min="1" max="64" value="8">
            <p class="muted" style="font-size:13px">Each creation model draws every class; the judge checks it drew what was asked.</p>`}
          <div class="err" id="b-err" role="alert"></div>
          <button class="btn primary">Create benchmark</button>
        </form>
        <h2 style="margin-top:26px">Other AI models</h2>
        <p class="muted" style="font-size:14px;margin:4px 0 0">Compare your models with ChatGPT, Claude, Gemini, a local Ollama model and more, on chat benchmarks.
          They only answer the test questions. Their answers are never used for training.</p>
        <ul class="list">${ext.models.map((m) => `<li class="ext-row"><div><b>${esc(m.name)}</b><div class="muted" style="font-size:13px">${esc(m.model)} · key ${esc(m.key)}</div></div>
          <button class="btn" data-xtest="${m.id}">Test</button><button class="btn ghost danger" data-xdel="${m.id}">Remove</button></li>`).join("")
          || `<li class="muted" style="padding:8px 0;font-size:14px">None added yet.</li>`}</ul>
        <details id="xadd"><summary>Add another AI</summary>
          <form id="xf">
            <label for="x-pre">Provider</label><select id="x-pre">${Object.keys(ext.presets).map((k) => `<option>${esc(k)}</option>`).join("")}</select>
            <label for="x-name">Name to show</label><input id="x-name" type="text" maxlength="60" placeholder="ChatGPT">
            <label for="x-model">Model ID</label><input id="x-model" type="text" maxlength="120" placeholder="The model name from the provider's docs">
            <label for="x-key">API key</label><input id="x-key" type="password" autocomplete="off" placeholder="Not needed for Ollama">
            <label for="x-url">Address</label><input id="x-url" type="text">
            <input id="x-api" type="hidden">
            <p class="muted" style="font-size:13px">The questions are sent to that company, and each run uses your API credit with them. Keys stay on your server and are never shown again.</p>
            <div class="err" id="x-err" role="alert"></div>
            <button class="btn primary">Add and test</button>
          </form></details></div>
      <div id="blist">${list.map((b) => benchCard(b, trained(b.kind))).join("") ||
        `<div class="panel empty-state"><h2>No benchmarks yet</h2><p class="muted">Create one on the left, then run your models on it.</p></div>`}</div></div>`;

    view.querySelectorAll("[data-bk]").forEach((b) => (b.onclick = () => { F.kind = b.dataset.bk; showBench(); }));
    const pre = document.getElementById("x-pre");
    const applyPreset = () => {
      const p = ext.presets[pre.value];
      document.getElementById("x-api").value = p.api;
      document.getElementById("x-url").value = p.base_url;
      const nm = document.getElementById("x-name");
      if (!nm.dataset.touched) nm.value = pre.value.replace(/ \(.*\)$/, "");
    };
    pre.onchange = applyPreset; applyPreset();
    document.getElementById("x-name").oninput = (e) => (e.target.dataset.touched = "1");
    if (S.xOpen) document.getElementById("xadd").open = true;
    document.getElementById("xadd").ontoggle = (e) => (S.xOpen = e.target.open);
    document.getElementById("xf").onsubmit = async (e) => {
      e.preventDefault();
      const v = (id) => document.getElementById(id).value.trim();
      const err = document.getElementById("x-err");
      err.textContent = "";
      try {
        const m = await api("/api/admin/external", { body: { name: v("x-name"), api: v("x-api"), base_url: v("x-url"), model: v("x-model"), api_key: v("x-key") } });
        try { const t = await api(`/api/admin/external/${m.id}/test`, { method: "POST" }); toast(`${m.name} answered in ${t.seconds}s.`); }
        catch (ex) { toast(`Added, but the test failed: ${ex.message}`); }
        S.xOpen = false;
        showBench();
      } catch (ex) { err.textContent = ex.message; }
    };
    view.querySelectorAll("[data-xtest]").forEach((b) => (b.onclick = async () => {
      b.disabled = true; b.textContent = "Testing…";
      try { const t = await api(`/api/admin/external/${b.dataset.xtest}/test`, { method: "POST" }); toast(`Working. Replied "${t.reply}" in ${t.seconds}s.`); }
      catch (ex) { toast(ex.message); }
      b.disabled = false; b.textContent = "Test";
    }));
    view.querySelectorAll("[data-xdel]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Remove this AI? Its past benchmark scores stay.")) return;
      await api(`/api/admin/external/${b.dataset.xdel}`, { method: "DELETE" });
      showBench();
    }));
    view.querySelectorAll("[data-sheet-dl]").forEach((b) => (b.onclick = async () => {
      const q = (await api(`/api/admin/benchmarks/${b.dataset.sheetDl}/sheet`)).questions;
      const txt = q.map((x) => `User: ${x}\nDeer: \n`).join("\n");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain" }));
      a.download = "benchmark-questions.txt"; a.click();
    }));
    view.querySelectorAll("[data-sheet-up]").forEach((f) => (f.onsubmit = async (e) => {
      e.preventDefault();
      const answers = f.querySelector("textarea").value;
      try {
        const r = await api(`/api/admin/benchmarks/${f.dataset.sheetUp}/sheet`, { body: { name: f.querySelector("input").value.trim(), answers } });
        toast(`Scored ${r.score}.`); S.openBench = +f.dataset.sheetUp; showBench();
      } catch (ex) { toast(ex.message); }
    }));
    const src = document.getElementById("b-src");
    if (src) src.onchange = () => { F.source = src.value; showBench(); };
    document.getElementById("bf").onsubmit = async (e) => {
      e.preventDefault();
      const v = (id) => document.getElementById(id)?.value;
      const err = document.getElementById("b-err");
      err.textContent = "";
      try {
        await api("/api/admin/benchmarks", { body: { name: v("b-name").trim(), kind: F.kind, source: F.source,
          text: v("b-text") || "", dataset_id: +v("b-ds") || null, size: +v("b-size") || 30,
          judge_model_id: +v("b-judge") || null, per_class: +v("b-per") || 8 } });
        toast("Benchmark created. Pick models and press Run.");
        showBench();
      } catch (ex) { err.textContent = ex.message; }
    };
    view.querySelectorAll("[data-brun]").forEach((btn) => (btn.onclick = async () => {
      const card = btn.closest(".bench"), ids = [...card.querySelectorAll("[data-bm]:checked")].map((x) => +x.dataset.bm);
      const xids = [...card.querySelectorAll("[data-bx]:checked")].map((x) => +x.dataset.bx);
      if (!ids.length && !xids.length) return toast("Tick at least one model.");
      btn.disabled = true; btn.textContent = "Running…";
      card.querySelector(".bench-note").textContent = "Testing. Chat benchmarks take about a second per question for each model.";
      try { await api(`/api/admin/benchmarks/${btn.dataset.brun}/run`, { body: { model_ids: ids, external_ids: xids } }); toast("Benchmark finished."); S.openBench = +btn.dataset.brun; }
      catch (ex) { toast(ex.message); }
      showBench();
    }));
    view.querySelectorAll("[data-bdel]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Delete this benchmark and its results?")) return;
      await api(`/api/admin/benchmarks/${b.dataset.bdel}`, { method: "DELETE" });
      showBench();
    }));
    view.querySelectorAll("[data-bopen]").forEach((b) => (b.onclick = () => {
      S.openBench = S.openBench === +b.dataset.bopen ? null : +b.dataset.bopen;
      showBench();
    }));
    if (S.openBench) {
      const box = document.getElementById(`bd-${S.openBench}`);
      if (box) box.innerHTML = benchDetail(await api(`/api/admin/benchmarks/${S.openBench}`));
    }
  }

  function benchCard(b, models) {
    const kindTxt = { chat: "Chat", vision: "Image analysis", canvas: "Image creation" };
    const what = b.kind === "chat" ? `${b.size} questions` : b.kind === "vision" ? `pictures from ${esc(b.spec.dataset_name)}` : `judged by ${esc(b.spec.judge_name)}`;
    return `<div class="panel bench">
      <div class="row-head"><div><h2>${esc(b.name)}</h2><span class="muted">${kindTxt[b.kind]} · ${what}</span></div>
        <button class="btn ghost danger" data-bdel="${b.id}" aria-label="Delete ${esc(b.name)}">Delete</button></div>
      ${b.leaders.length ? `<div class="bars">${b.leaders.map((l, i) => `<div class="bar-row ${i ? "" : "top"} src-${esc(l.source)}"><span>${esc(l.model_name)}${l.source !== "deer" ? ` <em class="tag-other">${l.source === "sheet" ? "pasted" : "other AI"}</em>` : ""}</span>
        <div class="meter-bar"><div style="width:${Math.max(2, l.score)}%"></div></div><b>${l.score}</b></div>`).join("")}</div>`
        : `<p class="muted">No results yet.</p>`}
      <div class="bench-run">${models.length ? models.map((m, i) => `<label class="toggle"><input type="checkbox" data-bm="${m.id}" ${i < 3 ? "checked" : ""}> ${esc(m.name)}</label>`).join("")
        : `<span class="muted">Train a ${kindTxt[b.kind].toLowerCase()} model to run this.</span>`}
        ${b.kind === "chat" ? (S.ext || []).map((x) => `<label class="toggle other"><input type="checkbox" data-bx="${x.id}"> ${esc(x.name)} <em class="tag-other">other AI</em></label>`).join("") : ""}</div>
      <div class="actions"><button class="btn primary" data-brun="${b.id}" ${models.length || (b.kind === "chat" && (S.ext || []).length) ? "" : "disabled"}>Run</button>
        ${b.leaders.length ? `<button class="btn" data-bopen="${b.id}">${S.openBench === b.id ? "Hide results" : "See results"}</button>` : ""}</div>
      ${b.kind === "chat" ? `<details class="sheet"><summary>Compare with any AI by hand (answer sheet)</summary>
        <p class="muted" style="font-size:13px">Download the questions, ask them in any AI's own app, write each answer after its <code>Deer:</code> line, then paste the file back here.</p>
        <button type="button" class="btn" data-sheet-dl="${b.id}">Download questions</button>
        <form data-sheet-up="${b.id}"><label>Whose answers</label><input type="text" maxlength="60" placeholder="ChatGPT app">
          <label>Filled-in answer sheet</label><textarea rows="5" placeholder="User: ...&#10;Deer: their answer"></textarea>
          <div class="actions"><button class="btn primary">Score these answers</button></div></form></details>` : ""}
      <p class="muted bench-note" style="font-size:13px"></p>
      <div id="bd-${b.id}"></div></div>`;
  }

  function benchDetail(b) {
    if (!b.runs.length) return "";
    const keys = Object.keys(b.runs[0].results.metrics);
    const last = b.runs[0].results;
    let extra = "";
    if (b.kind === "chat") {
      extra = b.runs.slice(0, 6).map((run) => `<details><summary>Answers from ${esc(run.model_name)}</summary>${run.results.rows.map((r) =>
        `<div class="qa"><div><span class="muted">Question</span> ${esc(r.prompt)}</div>
          <div class="grid2"><div><span class="muted">Expected</span><pre>${esc(r.expected)}</pre></div><div><span class="muted">Got</span><pre>${esc(r.got)}</pre></div></div>
          <small class="muted">Match ${Math.round(r.f1 * 100)}%${r.code_ok === null ? "" : r.code_ok ? " · code well-formed" : " · code malformed"}${r.error ? ` · <span style="color:var(--berry)">${esc(r.error)}</span>` : ""}</small></div>`).join("")}</details>`).join("");
    } else {
      const per = Object.entries(last.per_class || {});
      extra = `<h3 class="sub">By class (${esc(b.runs[0].model_name)}, latest run)</h3><div class="bars">${per.map(([c, v]) =>
        `<div class="bar-row"><span>${esc(c)}</span><div class="meter-bar"><div style="width:${v ?? 0}%"></div></div><b>${v ?? "n/a"}</b></div>`).join("")}</div>
        ${(last.top_mistakes || []).length ? `<p class="muted" style="font-size:13px">Most common mix-ups: ${last.top_mistakes.map(([k, n]) => `${esc(k)} (${n})`).join(", ")}</p>` : ""}
        ${(last.unknown_classes || []).length ? `<p class="muted" style="font-size:13px">This model never learned: ${last.unknown_classes.map(esc).join(", ")}</p>` : ""}`;
    }
    const notes = [...new Set(b.runs.map((r) => r.results.note).filter(Boolean))];
    return `${notes.length ? `<p class="muted" style="font-size:13px">${notes.map(esc).join(" ")} Word match compares wording with your expected answers, so long, chatty replies score lower even when they're right. Read the answers below too.</p>` : ""}<div style="overflow-x:auto"><table><thead><tr><th>Model</th><th>Step</th><th>Score</th>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}<th>When</th></tr></thead>
      <tbody>${b.runs.map((r) => `<tr><td>${esc(r.model_name)}${r.source && r.source !== "deer" ? ` <em class="tag-other">${r.source === "sheet" ? "pasted" : "other AI"}</em>` : ""}</td><td>${r.source && r.source !== "deer" ? "–" : r.step}</td><td><b>${r.score}</b></td>${keys.map((k) =>
        `<td>${r.results.metrics[k] ?? "–"}</td>`).join("")}<td>${new Date(r.created_at * 1000).toLocaleString()}</td></tr>`).join("")}</tbody></table></div>${extra}`;
  }


  // ================================================================== API
  async function showApi() {
    const a = await api("/api/admin/api");
    if (S.tab !== "api") return;
    const { usd, barChart } = Deer;
    const u = a.usage, p = a.prices;
    view.innerHTML = `
      <div class="stat-cards">
        <div class="stat-card"><b>${u.total.requests}</b><span>API requests, 30 days</span></div>
        <div class="stat-card"><b>${usd(u.total.cost)}</b><span>usage cost, 30 days</span></div>
        <div class="stat-card"><b>${usd(a.topups_30d.usd, 2)}</b><span>credit bought, 30 days</span><small>${a.topups_30d.count} payments</small></div>
        <div class="stat-card"><b>${usd(a.outstanding_credit, 2)}</b><span>unused credit held</span></div>
      </div>
      <div class="split">
        <div>
          <div class="panel"><h2>Settings</h2>
            <form id="apif">
              <label class="toggle" style="margin:10px 0"><input type="checkbox" id="a-on" ${a.enabled ? "checked" : ""}> API switched on</label>
              <div class="grid2">
                <div><label for="a-in">Chat input, $ per 1K tokens</label><input id="a-in" type="number" step="0.0001" min="0" value="${p.chat_in_per_1k}"></div>
                <div><label for="a-out">Chat output, $ per 1K tokens</label><input id="a-out" type="number" step="0.0001" min="0" value="${p.chat_out_per_1k}"></div>
                <div><label for="a-img">Create a picture, $</label><input id="a-img" type="number" step="0.0001" min="0" value="${p.image}"></div>
                <div><label for="a-an">Analyze a picture, $</label><input id="a-an" type="number" step="0.0001" min="0" value="${p.analyze}"></div>
                <div><label for="a-wel">Welcome credit with first key, $</label><input id="a-wel" type="number" step="0.5" min="0" value="${a.welcome_credit}"></div>
                <div><label for="a-min">Smallest top-up, $</label><input id="a-min" type="number" step="1" min="1" value="${a.min_topup}"></div>
              </div>
              <label for="a-packs">Top-up amounts offered, $ (comma separated)</label><input id="a-packs" type="text" value="${esc(a.packs.join(","))}">
              <p class="muted" style="font-size:13px">1 token = 1 byte of text for Deer models. Price changes apply to the next request. Top-ups use the PayPal keys from Plans; add PAYMENT.CAPTURE.COMPLETED to your PayPal webhook events too.</p>
              <div class="actions"><button class="btn primary">Save</button><a class="btn" href="developers.html">Open the API page</a></div>
            </form></div>
          <div class="panel"><h2>Give or take credit</h2>
            <form id="grantf" class="grid2" style="align-items:end">
              <div><label for="g-user">Username</label><input id="g-user" type="text" required></div>
              <div><label for="g-amt">Amount, $ (negative to take away)</label><input id="g-amt" type="number" step="0.5" required></div>
              <div style="grid-column:1/-1"><label for="g-note">Note</label><input id="g-note" type="text" maxlength="120" placeholder="Beta tester"></div>
              <div><button class="btn primary">Apply</button></div>
            </form></div>
        </div>
        <div>
          <div class="panel"><div class="row-head"><h2>Usage</h2><span class="muted">last 30 days, all users</span></div><div id="achart"></div>
            <table style="margin-top:12px"><tbody>${u.by_endpoint.map((e) => `<tr><td>/v1/${esc(e.endpoint)}</td><td>${e.requests} request${e.requests === 1 ? "" : "s"}</td><td>${usd(e.cost)}</td></tr>`).join("") || `<tr><td class="muted">No API requests yet.</td></tr>`}</tbody></table></div>
          <div class="panel"><h2>Developers</h2><div style="overflow-x:auto"><table><thead><tr><th>User</th><th>Keys</th><th>Spent, 30 days</th><th>Credit left</th></tr></thead><tbody>
            ${a.users.map((x) => `<tr><td>${esc(x.username)}</td><td>${x.keys}</td><td>${usd(x.spent)}</td><td>${usd(x.balance, 2)}</td></tr>`).join("") || `<tr><td colspan="4" class="muted">Nobody has used the API yet.</td></tr>`}
          </tbody></table></div></div>
        </div>
      </div>`;
    barChart(document.getElementById("achart"), u.daily, (r) => r.cost,
      (r) => `<div>Cost <b>${usd(r.cost)}</b></div><div>Requests <b>${r.requests}</b></div>`);
    document.getElementById("apif").onsubmit = async (e) => {
      e.preventDefault();
      const n = (id) => +document.getElementById(id).value;
      try {
        await api("/api/admin/api", { method: "PATCH", body: { enabled: document.getElementById("a-on").checked,
          chat_in_per_1k: n("a-in"), chat_out_per_1k: n("a-out"), image: n("a-img"), analyze: n("a-an"),
          welcome_credit: n("a-wel"), min_topup: n("a-min"), packs: document.getElementById("a-packs").value } });
        toast("API settings saved.");
      } catch (ex) { toast(ex.message); }
    };
    document.getElementById("grantf").onsubmit = async (e) => {
      e.preventDefault();
      const v = (id) => document.getElementById(id).value;
      try {
        await api("/api/admin/api/grant", { body: { username: v("g-user").trim(), amount: +v("g-amt"), note: v("g-note") } });
        toast("Credit updated."); showApi();
      } catch (ex) { toast(ex.message); }
    };
  }


  // =============================================================== updates
  function waitForRestart(msg) {
    view.innerHTML = `<div class="panel empty-state"><h2>${esc(msg)}</h2><p class="muted" id="rs">Deer AI is restarting with the new version. This page reloads by itself.</p></div>`;
    const t0 = Date.now();
    const poll = async () => {
      await new Promise((r) => setTimeout(r, 2500));
      try {
        const r = await fetch((Deer.API_BASE || "") + "/api/auth/status", { cache: "no-store" });
        if (r.ok && Date.now() - t0 > 4000) return location.reload();
      } catch {}
      if (Date.now() - t0 > 120000) { document.getElementById("rs").textContent = "It's taking longer than usual. Refresh in a minute; if the update failed, Deer AI puts the old version back by itself."; return; }
      poll();
    };
    poll();
  }
  function finished(r, what) {
    if (r.restarting) return waitForRestart(what);
    toast(what + (r.note ? " " + r.note : ""));
    showUpdates();
  }

  async function showUpdates() {
    const u = await api("/api/admin/update");
    if (S.tab !== "updates") return;
    const gh = u.github, v = u.version || {};
    const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "");
    const hook = (Deer.API_BASE || location.origin) + "/api/github/webhook";
    const verTxt = v.source === "github" ? `from GitHub, commit ${esc(v.commit || "")}` : v.source === "zip upload" ? "from an uploaded zip"
      : v.source === "backup" ? "restored from a backup" : v.commit ? `commit ${esc(v.commit)}` : "as installed";
    view.innerHTML = `
      ${u.rolled_back ? `<div class="wait-note" style="margin-top:18px"><b>An update was undone automatically</b> on ${when(u.rolled_back.at)}: ${esc(u.rolled_back.reason)} The previous version is running.</div>` : ""}
      ${u.training ? `<div class="wait-note" style="margin-top:18px">A model is training. Updates wait until it finishes or you stop it.</div>` : ""}
      <div class="split">
        <div>
          <div class="panel"><h2>This server</h2>
            <div class="stats"><div><b>${verTxt}</b>running version${v.at ? `, ${when(v.at)}` : ""}</div></div>
            <p class="muted" style="font-size:14px;margin:0">${u.systemd ? "Runs as a service, so it restarts by itself after an update." : "Not running as a service here, so restart it yourself after an update."}</p></div>
          <div class="panel"><h2>Upload an update</h2>
            <p class="muted" style="font-size:14px;margin:4px 0 0">Pick the <b>deer-ai.zip</b> you downloaded (on a phone: from Files or Downloads). Your data, models, users and settings stay as they are. The current version is backed up first, and if the new one can't start it's put back by itself.</p>
            <form id="uz"><input type="file" id="uz-file" accept=".zip,application/zip" style="margin:12px 0;width:100%">
              <button class="btn primary" ${u.training ? "disabled" : ""}>Update Deer AI</button></form></div>
          <div class="panel"><h2>Backups</h2>
            ${u.backups.length ? u.backups.map((b) => `<div class="keyrow"><div><b>${when(b.at)}</b> <span class="muted" style="font-size:13px">${esc(b.note)}</span></div>
              <button class="btn" data-restore="${esc(b.file)}" ${u.training ? "disabled" : ""}>Restore</button></div>`).join("")
              : `<p class="muted">A backup is made before every update. The last ${3} are kept.</p>`}</div>
        </div>
        <div class="panel"><h2>GitHub</h2>
          ${!u.git_installed ? `<p class="err">git isn't installed on this server. Run the installer again (it adds it), or: <code>apt install -y git</code></p>` : ""}
          ${!gh.repo ? `
            <p class="muted" style="font-size:14px;margin:4px 0 0">Connect a GitHub repo to keep Deer AI's code there and update the server with one tap. Everything here works in a phone browser.</p>
            <ol class="steps"><li>On github.com, make a new <b>private</b> repository (no README).</li><li>Enter it below and press Connect.</li></ol>
            <form id="ghf" class="grid2" style="align-items:end">
              <div><label for="gh-repo">Repository</label><input id="gh-repo" type="text" placeholder="yourname/deer-ai" required></div>
              <div><label for="gh-branch">Branch</label><input id="gh-branch" type="text" value="main"></div>
              <div style="grid-column:1/-1"><button class="btn primary">Connect</button></div></form>`
          : `
            <div class="row-head"><div><b>${esc(gh.repo)}</b> <span class="muted">· ${esc(gh.branch)}</span></div><button class="btn ghost" id="gh-change">Change</button></div>
            <h3 class="sub">1. Let the server into the repo</h3>
            <p class="muted" style="font-size:14px;margin:4px 0">Open the link, give it a title like <i>Deer AI server</i>, paste this key, tick <b>Allow write access</b>, then Add key.</p>
            <code class="code" id="gh-key">${esc(gh.public_key || "")}</code>
            <div class="actions" style="margin-top:8px"><button class="btn" id="gh-copy">Copy key</button>
              <a class="btn" href="https://github.com/${esc(gh.repo)}/settings/keys/new" target="_blank" rel="noopener">Open GitHub deploy keys</a>
              <button class="btn" id="gh-check">Check connection</button></div>
            <p id="gh-state" class="muted" style="font-size:14px"></p>
            <h3 class="sub">2. Send code and get updates</h3>
            <div class="actions"><button class="btn" id="gh-push">Push this server's code to GitHub</button>
              <button class="btn primary" id="gh-pull" ${u.training ? "disabled" : ""}>Update from GitHub</button></div>
            <p class="muted" style="font-size:13px">Push once to fill an empty repo. After that, change the code on GitHub (or push from a computer), then press Update.</p>
            <h3 class="sub">3. Automatic updates (optional)</h3>
            <label class="toggle"><input type="checkbox" id="gh-auto" ${gh.auto ? "checked" : ""}> Update by itself when I push to ${esc(gh.branch)}</label>
            <p class="muted" style="font-size:13px;margin:6px 0">Add a webhook on GitHub with these values, content type <b>application/json</b>, and just the push event:</p>
            <label>Payload URL</label><code class="code">${esc(hook)}</code>
            <label>Secret</label><code class="code" id="gh-secret">${esc(gh.webhook_secret || "")}</code>
            <div class="actions" style="margin-top:8px"><button class="btn" id="gh-copy-secret">Copy secret</button>
              <a class="btn" href="https://github.com/${esc(gh.repo)}/settings/hooks/new" target="_blank" rel="noopener">Open GitHub webhooks</a></div>
            <p class="muted" style="font-size:13px">If a model is training when you push, the update waits until it's done.</p>`}
        </div>
      </div>`;

    const $ = (id) => document.getElementById(id);
    $("uz").onsubmit = async (e) => {
      e.preventDefault();
      const f = $("uz-file").files[0];
      if (!f) return toast("Choose the deer-ai.zip file first.");
      if (!confirm("Update Deer AI now? The site restarts for a few seconds.")) return;
      const fd = new FormData(); fd.append("file", f);
      const btn = e.target.querySelector("button"); btn.disabled = true; btn.textContent = "Uploading…";
      try { finished(await api("/api/admin/update/zip", { body: fd }), "Updated."); }
      catch (ex) { toast(ex.message); btn.disabled = false; btn.textContent = "Update Deer AI"; }
    };
    view.querySelectorAll("[data-restore]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Go back to this version? The site restarts for a few seconds.")) return;
      try { finished(await api("/api/admin/update/rollback", { body: { file: b.dataset.restore } }), "Restored."); }
      catch (ex) { toast(ex.message); }
    }));
    if ($("ghf")) $("ghf").onsubmit = async (e) => {
      e.preventDefault();
      try { await api("/api/admin/update/github", { method: "PUT", body: { repo: $("gh-repo").value.trim(), branch: $("gh-branch").value.trim() || "main" } }); showUpdates(); }
      catch (ex) { toast(ex.message); }
    };
    if (!gh.repo) return;
    $("gh-change").onclick = async () => {
      const r = prompt("Repository (owner/name)", gh.repo);
      if (r) { try { await api("/api/admin/update/github", { method: "PUT", body: { repo: r, branch: gh.branch } }); showUpdates(); } catch (ex) { toast(ex.message); } }
    };
    $("gh-copy").onclick = () => navigator.clipboard.writeText(gh.public_key).then(() => toast("Key copied."));
    $("gh-copy-secret").onclick = () => navigator.clipboard.writeText(gh.webhook_secret).then(() => toast("Secret copied."));
    $("gh-auto").onchange = async (e) => { await api("/api/admin/update/github", { method: "PATCH", body: { auto: e.target.checked } }); toast(e.target.checked ? "Automatic updates on." : "Automatic updates off."); };
    const busy = (b, t) => { b.disabled = true; b.dataset.t = b.textContent; b.textContent = t; };
    const idle = (b) => { b.disabled = false; b.textContent = b.dataset.t; };
    $("gh-check").onclick = async (e) => {
      busy(e.target, "Checking…");
      try {
        const r = await api("/api/admin/update/github/check", { method: "POST" });
        $("gh-state").innerHTML = r.empty ? "Connected. The repo is empty: press <b>Push this server's code to GitHub</b>."
          : r.update_available ? `Connected. <b>An update is available</b> (GitHub has ${esc(r.remote_commit.slice(0, 7))}).` : "Connected. The server is up to date.";
      } catch (ex) { $("gh-state").textContent = ex.message; }
      idle(e.target);
    };
    $("gh-push").onclick = async (e) => {
      busy(e.target, "Pushing…");
      try { await api("/api/admin/update/github/push", { method: "POST" }); toast("The server's code is on GitHub."); $("gh-check").click(); }
      catch (ex) { toast(ex.message); }
      idle(e.target);
    };
    $("gh-pull").onclick = async (e) => {
      if (!confirm("Update from GitHub now? If there's a new version, the site restarts for a few seconds.")) return;
      busy(e.target, "Updating…");
      try {
        const r = await api("/api/admin/update/github/pull", { method: "POST" });
        if (!r.updated) { toast("Already up to date."); idle(e.target); } else finished(r, "Updated from GitHub.");
      } catch (ex) { toast(ex.message); idle(e.target); }
    };
  }

  // ============================================================= learning
  async function showLearning(status = "pending") {
    const L = await api(`/api/admin/learning?status=${status}`);
    if (S.tab !== "learning") return;  // the admin switched tabs while this was loading
    const n = (k) => L.counts[k] || 0;
    const pending = status === "pending";
    view.innerHTML = `<div class="split">
      <div>
        <div class="panel"><h2>Learning from users</h2>
          <p class="muted" style="font-size:14px;margin:4px 0 0">When people rate a reply, write a better answer, or fix an image label, it lands here. Approved items join the model's training data.</p>
          <div class="stats"><div><b>${n("pending")}</b>waiting</div><div><b>${n("applied")}</b>approved</div><div><b>${n("rejected")}</b>rejected</div></div>
          <h3 class="sub">Learn from users</h3>
          <label class="toggle"><input type="checkbox" data-ls="from_messages" ${L.from_messages ? "checked" : ""}> Learn from what users write</label>
          <p class="muted" style="font-size:13px;margin:2px 0 8px">Their messages are added to the chat model's training text, so it gets used to how people really talk.</p>
          <label class="toggle"><input type="checkbox" data-ls="from_uploads" ${L.from_uploads ? "checked" : ""}> Learn from pictures users send</label>
          <p class="muted" style="font-size:13px;margin:2px 0 8px">When the analysis model is at least 90% sure, the picture comes here labeled with its answer.</p>
          <label class="toggle"><input type="checkbox" data-ls="auto_approve" ${L.auto_approve ? "checked" : ""}> Approve feedback automatically</label>
          <p class="muted" style="font-size:13px;margin:2px 0 8px">Faster, but anyone could teach the model something wrong or rude. Reviewing is safer.</p>
          <p class="muted" style="font-size:13px">Users can opt out on their Plans page${L.opted_out ? `; ${L.opted_out} have` : ""}. Opted-out users' chats are never used.</p>
        </div>
        <div class="panel"><h2>How people rate each model</h2>
          ${L.models.length ? `<table><thead><tr><th>Model</th><th>Good</th><th>Bad</th><th>Liked</th></tr></thead><tbody>${L.models.map((r) =>
            `<tr><td>${esc(r.name)}</td><td>${r.up}</td><td>${r.down}</td><td>${Math.round((r.up / Math.max(1, r.up + r.down)) * 100)}%</td></tr>`).join("")}</tbody></table>`
            : `<p class="muted">No ratings yet.</p>`}
        </div>
      </div>
      <div class="panel">
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><h2 style="flex:1">Inbox</h2>
          <select id="lf" style="width:auto" aria-label="Show">
            ${[["pending", "Waiting for review"], ["applied", "Approved"], ["rejected", "Rejected"], ["info", "Ratings only"]].map(([v, t]) =>
              `<option value="${v}" ${v === status ? "selected" : ""}>${t}</option>`).join("")}</select></div>
        ${pending && L.items.length ? `<div class="actions"><label class="toggle"><input type="checkbox" id="all"> Select all</label>
          <button class="btn primary" id="ok">Approve selected</button><button class="btn danger" id="no">Reject selected</button></div>` : ""}
        <ul class="list" id="items">${L.items.map((i) => `<li style="display:flex;gap:12px;padding:14px 0;align-items:flex-start">
          ${pending ? `<input type="checkbox" class="pick" value="${i.id}" aria-label="Select" style="width:18px;height:18px;margin-top:3px;accent-color:var(--fawn)">` : ""}
          ${i.image ? `<img src="${i.image}" alt="" style="width:64px;height:64px;object-fit:cover;border-radius:6px;image-rendering:auto">` : ""}
          <div style="flex:1;min-width:0">
            ${i.kind === "chat" && !i.image
              ? `<div><span class="muted">User:</span> ${esc(i.prompt || "")}</div><div><span class="muted">Deer:</span> ${esc(i.answer || "(no better answer given)")}</div>`
              : i.kind === "vision" ? `<div>Labeled as <b>${esc(i.label || "?")}</b></div>` : `<div>Rated generated images</div>`}
            <div class="muted" style="font-size:13px;margin-top:4px">${i.rating === 0 ? `Labeled automatically (${Math.round(parseFloat((i.answer || "auto:0").split(":")[1]) * 100)}% sure), sent` : i.rating > 0 ? "Liked" : i.kind === "chat" && i.answer ? "Corrected" : "Disliked"}
              by ${esc(i.username || "deleted user")}, ${esc(i.model_name || "")}, ${new Date(i.created_at * 1000).toLocaleString()}</div></div></li>`).join("")
          || `<li class="muted" style="padding:12px 0">Nothing here.</li>`}</ul>
      </div></div>`;
    document.getElementById("lf").onchange = (e) => showLearning(e.target.value);
    view.querySelectorAll("[data-ls]").forEach((el) => (el.onchange = async () => {
      await api("/api/admin/learning/settings", { method: "PATCH", body: { [el.dataset.ls]: el.checked } });
      toast("Learning setting saved.");
    }));
    if (!pending || !L.items.length) return;
    const picks = () => [...view.querySelectorAll(".pick:checked")].map((x) => +x.value);
    document.getElementById("all").onchange = (e) => view.querySelectorAll(".pick").forEach((x) => (x.checked = e.target.checked));
    const review = async (action) => {
      const ids = picks();
      if (!ids.length) return toast("Select at least one item.");
      try {
        const r = await api("/api/admin/learning/review", { body: { ids, action } });
        toast(`${r.done} ${action === "approve" ? "approved" : "rejected"}.` + (r.skipped?.length
          ? ` ${r.skipped.length} couldn't be used because ${r.skipped[0].reason}, so ${r.skipped.length === 1 ? "it was" : "they were"} moved to Ratings only.` : ""));
        showLearning("pending");
      } catch (e) { toast(e.message); }
    };
    document.getElementById("ok").onclick = () => review("approve");
    document.getElementById("no").onclick = () => review("reject");
  }

  // ================================================================ plans
  async function showPlans() {
    const b = await api("/api/admin/billing");
    if (S.tab !== "plans") return;  // the admin switched tabs while this was loading
    const pp = b.paypal;
    const hook = (Deer.API_BASE || location.origin) + "/api/paypal/webhook";
    const rev = b.revenue_30d.map((r) => new Intl.NumberFormat(undefined, { style: "currency", currency: r.currency }).format(r.total)).join(" + ") || "$0";
    view.innerHTML = `<div class="split">
      <div>
        <div class="panel"><div style="display:flex;align-items:center;gap:10px"><h2 style="flex:1">PayPal</h2>
          <span class="chip ${pp.configured ? "ready" : ""}">${pp.configured ? (pp.mode === "live" ? "Live" : "Sandbox") : "Not connected"}</span></div>
          <form id="ppf">
            <label for="pp-mode">Mode</label><select id="pp-mode"><option value="sandbox">Sandbox (test money)</option><option value="live" ${pp.mode === "live" ? "selected" : ""}>Live (real money)</option></select>
            <label for="pp-id">Client ID</label><input id="pp-id" type="text" autocomplete="off" value="${esc(pp.client_id)}">
            <label for="pp-sec">Secret</label><input id="pp-sec" type="password" autocomplete="off" placeholder="${pp.secret_set ? "Saved. Leave blank to keep it." : ""}">
            <label for="pp-wh">Webhook ID</label><input id="pp-wh" type="text" autocomplete="off" value="${esc(pp.webhook_id)}">
            <div class="err" id="pp-err" role="alert"></div>
            <button class="btn primary">Save and test</button>
          </form>
          <label>Webhook URL to add in PayPal</label><code class="code">${esc(hook)}</code>
          <p class="muted" style="font-size:14px">Subscribe it to: BILLING.SUBSCRIPTION.ACTIVATED, .UPDATED, .CANCELLED, .SUSPENDED, .EXPIRED, .RE-ACTIVATED, PAYMENT.SALE.COMPLETED, PAYMENT.SALE.REFUNDED, PAYMENT.SALE.REVERSED, and PAYMENT.CAPTURE.COMPLETED (for API credit). The URL must be public HTTPS.</p>
        </div>
        <div class="panel"><h2>Last 30 days</h2>
          <div class="stats"><div><b>${esc(rev)}</b>revenue</div><div><b>${b.active_paid}</b>paying subscribers</div></div></div>
      </div>
      <div>
        <div class="panel"><div style="display:flex;align-items:center;gap:10px"><h2 style="flex:1">Plans</h2><button class="btn" id="add-plan">Add plan</button></div>
          <p class="muted" style="font-size:14px;margin:4px 0 0">Use -1 for unlimited. Messages count over the window; images and analyses count per 24 hours. After changing a paid plan's price, press Link to PayPal again; current subscribers keep their old price.</p>
          <div id="plan-rows"></div></div>
        <div class="panel"><h2>Subscribers</h2><div style="overflow-x:auto"><table><thead><tr><th>User</th><th>Plan</th><th>Status</th><th>Paid through</th></tr></thead><tbody>
          ${b.subscriptions.map((s) => `<tr><td>${esc(s.username)}</td><td>${esc(s.plan)}</td><td>${esc(s.status.toLowerCase())}</td><td>${new Date(s.period_end * 1000).toLocaleDateString()}</td></tr>`).join("")
            || `<tr><td colspan="4" class="muted">No subscribers yet.</td></tr>`}
          </tbody></table></div></div>
      </div></div>`;

    document.getElementById("ppf").onsubmit = async (e) => {
      e.preventDefault();
      const err = document.getElementById("pp-err");
      err.textContent = "";
      try {
        await api("/api/admin/billing/paypal", { method: "PUT", body: {
          mode: document.getElementById("pp-mode").value, client_id: document.getElementById("pp-id").value.trim(),
          secret: document.getElementById("pp-sec").value.trim() || null, webhook_id: document.getElementById("pp-wh").value.trim() } });
        toast("PayPal connected.");
        showPlans();
      } catch (ex) { err.textContent = ex.message; }
    };

    const rows = document.getElementById("plan-rows");
    rows.innerHTML = b.plans.map((p) => `<form class="plan-edit" data-id="${p.id}">
      <div><label>Name</label><input name="name" type="text" value="${esc(p.name)}" required maxlength="40"></div>
      <div><label>Price / month</label><input name="price" type="number" step="0.01" min="0" value="${p.price}"></div>
      <div><label>Messages</label><input name="chat_limit" type="number" min="-1" value="${p.chat_limit}"></div>
      <div><label>Window (hours)</label><input name="window_hours" type="number" min="1" value="${p.window_hours}"></div>
      <div><label>Images / day</label><input name="image_limit" type="number" min="-1" value="${p.image_limit}"></div>
      <div><label>Analyses / day</label><input name="analyze_limit" type="number" min="-1" value="${p.analyze_limit}"></div>
      <div style="display:flex;gap:6px"><button class="btn primary">Save</button>
        ${p.price > 0 ? `<button type="button" class="btn" data-sync="${p.id}">${p.paypal_plan_id ? "Relink" : "Link to PayPal"}</button>` : ""}</div>
      <div style="grid-column:1/-1"><label>Description</label><input name="blurb" type="text" maxlength="120" value="${esc(p.blurb)}"></div>
      <div style="grid-column:1/-1;display:flex;gap:16px;align-items:center;flex-wrap:wrap">
        <label class="toggle"><input name="active" type="checkbox" ${p.active ? "checked" : ""}> Shown to users</label>
        <span class="muted" style="font-size:13px">${p.price > 0 ? (p.paypal_plan_id ? `PayPal plan ${esc(p.paypal_plan_id)} (${esc((p.synced || "").split(":")[0])})` : "Not linked to PayPal yet, so it can't be bought") : "Free plan"}</span></div>
    </form>`).join("");
    rows.querySelectorAll("form").forEach((f) => (f.onsubmit = async (e) => {
      e.preventDefault();
      const v = (n) => f.elements[n].value;
      try {
        await api(`/api/admin/plans/${f.dataset.id}`, { method: "PATCH", body: {
          name: v("name"), blurb: v("blurb"), price: +v("price"), currency: "USD", chat_limit: +v("chat_limit"),
          window_hours: +v("window_hours"), image_limit: +v("image_limit"), analyze_limit: +v("analyze_limit"),
          sort: b.plans.find((p) => p.id == f.dataset.id).sort, active: f.elements.active.checked } });
        toast("Plan saved.");
        showPlans();
      } catch (ex) { toast(ex.message); }
    }));
    rows.querySelectorAll("[data-sync]").forEach((btn) => (btn.onclick = async () => {
      btn.disabled = true;
      try { await api(`/api/admin/plans/${btn.dataset.sync}/sync`, { method: "POST" }); toast("Linked to PayPal."); showPlans(); }
      catch (ex) { toast(ex.message); btn.disabled = false; }
    }));
    document.getElementById("add-plan").onclick = async () => {
      await api("/api/admin/plans", { body: { name: "New plan", blurb: "", price: 5, currency: "USD", chat_limit: 100,
        window_hours: 5, image_limit: 40, analyze_limit: 100, sort: b.plans.length, active: false } });
      showPlans();
    };
  }

  // =============================================================== people
  async function showPeople() {
    const r = await api("/api/admin/users");
    if (S.tab !== "people") return;  // the admin switched tabs while this was loading
    view.innerHTML = `<div class="panel" style="margin-top:20px"><h2>People</h2>
      <label class="toggle" style="margin:12px 0"><input type="checkbox" id="reg" ${r.allow_registration ? "checked" : ""}> Anyone can create an account</label>
      <div style="overflow-x:auto"><table><thead><tr><th>Username</th><th>Role</th><th>Joined</th><th></th></tr></thead><tbody>
      ${r.users.map((u) => `<tr><td>${esc(u.username)}</td>
        <td><select data-role="${u.id}" ${u.id === user.id ? "disabled" : ""} aria-label="Role for ${esc(u.username)}" style="width:auto">
          <option value="user" ${u.role === "user" ? "selected" : ""}>User</option><option value="admin" ${u.role === "admin" ? "selected" : ""}>Admin</option></select></td>
        <td>${new Date(u.created_at * 1000).toLocaleDateString()}</td>
        <td>${u.id === user.id ? "" : `<button class="btn danger" data-del="${u.id}">Remove</button>`}</td></tr>`).join("")}
      </tbody></table></div></div>`;
    document.getElementById("reg").onchange = async (e) => {
      await api("/api/admin/settings", { method: "PATCH", body: { allow_registration: e.target.checked } });
      toast(e.target.checked ? "Registration opened." : "Registration closed.");
    };
    view.querySelectorAll("[data-role]").forEach((s) => (s.onchange = async () => {
      try { await api(`/api/admin/users/${s.dataset.role}`, { method: "PATCH", body: { role: s.value } }); toast("Role updated."); }
      catch (e) { toast(e.message); }
    }));
    view.querySelectorAll("[data-del]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Remove this person and their chats?")) return;
      await api(`/api/admin/users/${b.dataset.del}`, { method: "DELETE" });
      showPeople();
    }));
  }

  show(PAGES[location.hash.slice(1)] ? location.hash.slice(1) : "home");
})();
