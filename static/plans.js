(async () => {
  const { api, esc, toast, gate, MARK, until } = Deer;
  const root = document.getElementById("root");
  await gate(root);
  let B;

  const lim = (n, word, hours) => (n < 0 ? `Unlimited ${word}` : `${n} ${word} every ${hours} hours`);
  const money = (p) => new Intl.NumberFormat(undefined, { style: "currency", currency: p.currency }).format(p.price);
  const date = (ts) => new Date(ts * 1000).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });

  function meter(label, u) {
    if (u.limit < 0) return `<div class="meter"><div class="lbl"><span>${label}</span><b>Unlimited</b></div><div class="track"><div class="fill" style="width:0"></div></div><div class="when">${u.used} used in the last ${u.window_hours} hours</div></div>`;
    const pct = Math.min(100, (u.used / Math.max(1, u.limit)) * 100);
    return `<div class="meter"><div class="lbl"><span>${label}</span><b>${u.used} of ${u.limit}</b></div>
      <div class="track"><div class="fill ${pct >= 100 ? "full" : ""}" style="width:${pct}%"></div></div>
      <div class="when">${u.resets_at ? `Frees up in ${until(u.resets_at)}` : `Resets on a rolling ${u.window_hours}-hour window`}</div></div>`;
  }

  function render() {
    const cur = B.plan, sub = B.subscription;
    let status = "";
    if (B.is_admin) status = "As the admin you have unlimited use. Turn on test mode below to try Deer AI the way your users do.";
    else if (B.test) status = B.test.plan_id ? `Test mode: previewing ${esc(cur.name)} without paying.` : "Test mode: you're on your real plan, like any user. PayPal sandbox checkout works here.";
    else if (sub && sub.status === "CANCELLED") status = `Your ${esc(cur.name)} plan is cancelled and stays active until ${date(sub.period_end)}.`;
    else if (sub) status = `${esc(cur.name)} renews on ${date(sub.period_end)} through PayPal.`;

    root.innerHTML = `<div class="plans-page">
      <header><div class="mark">${MARK}<span>Deer AI</span></div><a class="btn" href="./">Back to chat</a></header>
      <h1>Plans</h1>
      <div class="panel"><h2>You're on ${esc(cur.name)}</h2>
        ${status ? `<p class="muted" style="margin:4px 0 0">${status}</p>` : ""}
        <div class="meters">${meter("Messages", B.usage.chat)}${meter("Images created", B.usage.imagine)}${meter("Images analyzed", B.usage.analyze)}</div>
        ${sub && sub.status === "ACTIVE" && !B.is_admin ? `<div class="actions"><button class="btn danger" id="cancel">Cancel plan</button></div>` : ""}
      </div>
      ${B.role === "admin" ? `<div class="panel" id="test" style="margin-top:16px"><h2>Test as a user</h2>
        <p class="muted" style="margin:4px 0 0">See only published models, hit the same limits, and try checkout, exactly like a user. Your admin panel still works.</p>
        <form id="testf" class="grid2" style="align-items:end">
          <label class="toggle" style="grid-column:1/-1;margin-top:12px"><input type="checkbox" id="t-on" ${B.test ? "checked" : ""}> Test mode</label>
          <div><label for="t-plan">Plan to test</label><select id="t-plan">
            <option value="">My real plan (Free, or what I buy with PayPal)</option>
            ${B.plans.map((p) => `<option value="${p.id}" ${B.test?.plan_id === p.id ? "selected" : ""}>Preview ${esc(p.name)} without paying</option>`).join("")}</select></div>
          <div class="actions" style="margin-top:0"><button class="btn primary">Apply</button><button type="button" class="btn" id="t-reset">Reset my usage</button></div>
        </form></div>` : ""}
      <div class="panel" id="privacy" style="margin-top:16px"><h2>Help improve Deer AI</h2>
        <label class="toggle" style="margin-top:10px"><input type="checkbox" id="train-ok" ${B.train_ok ? "checked" : ""}> Let Deer AI learn from my chats and pictures</label>
        <p class="muted" style="font-size:14px;margin:6px 0 0">When on, what you write and send can be used to train Deer AI's models. Turn it off anytime; it applies from then on. Feedback you give with the thumbs buttons is always shared with the admin.</p></div>
      <div class="plan-grid">${B.plans.map((p) => {
        const isCur = p.id === cur.id;
        const action = isCur ? `<button class="btn" disabled>Current plan</button>`
          : p.price === 0 ? `<div class="muted" style="font-size:14px">${sub ? "You'll return here if you cancel." : ""}</div>`
          : !p.purchasable || !B.paypal ? `<button class="btn" disabled>Coming soon</button>`
          : `<div class="pp" id="pp-${p.id}"></div>`;
        return `<section class="plan ${isCur ? "current" : ""}"><h2>${esc(p.name)}</h2><div class="muted">${esc(p.blurb)}</div>
          <div class="price">${p.price ? money(p) : "Free"}${p.price ? ` <small>per month</small>` : ""}</div>
          <ul><li>${lim(p.chat_limit, "messages", p.window_hours)}</li><li>${lim(p.image_limit, "images created", 24)}</li>
          <li>${lim(p.analyze_limit, "image analyses", 24)}</li></ul>${action}</section>`;
      }).join("")}</div>
      <p class="muted" style="font-size:14px;margin-top:20px">Plans bill monthly through PayPal. Switching plans starts the new one right away and stops billing for the old one. Cancel anytime; you keep access until the end of the period you paid for.</p>
    </div>`;

    document.getElementById("train-ok").onchange = async (e) => {
      await api("/api/me/privacy", { method: "PUT", body: { train_ok: e.target.checked } });
      B.train_ok = e.target.checked;
      toast(e.target.checked ? "Thanks! Your chats can help train Deer AI." : "Your chats won't be used for training.");
    };
    const tf = document.getElementById("testf");
    if (tf) {
      tf.onsubmit = async (e) => {
        e.preventDefault();
        const plan = document.getElementById("t-plan").value;
        await api("/api/me/test-mode", { method: "PUT", body: { on: document.getElementById("t-on").checked, plan_id: plan ? +plan : null } });
        B = await api("/api/billing");
        toast(B.test ? "Test mode on. Open the chat to try it." : "Test mode off.");
        render();
      };
      document.getElementById("t-reset").onclick = async () => {
        await api("/api/me/reset-usage", { method: "POST" });
        B = await api("/api/billing");
        toast("Your usage was reset.");
        render();
      };
    }
    const c = document.getElementById("cancel");
    if (c) c.onclick = async () => {
      if (!confirm(`Cancel ${cur.name}? You keep it until ${date(sub.period_end)}, then move to the free plan.`)) return;
      try { B = await api("/api/billing/cancel", { method: "POST" }); toast("Plan cancelled."); render(); }
      catch (e) { toast(e.message); }
    };
    mountPayPal();
  }

  let sdk;
  function loadSdk() {
    if (sdk) return sdk;
    const currency = (B.plans.find((p) => p.price > 0) || {}).currency || "USD";
    sdk = new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = `https://www.paypal.com/sdk/js?client-id=${encodeURIComponent(B.paypal.client_id)}&vault=true&intent=subscription&currency=${currency}`;
      s.onload = ok;
      s.onerror = () => fail(new Error("PayPal couldn't load. Check your connection and refresh."));
      document.head.appendChild(s);
    });
    return sdk;
  }

  async function mountPayPal() {
    const slots = B.plans.filter((p) => document.getElementById(`pp-${p.id}`));
    if (!slots.length) return;
    try { await loadSdk(); } catch (e) { return toast(e.message); }
    for (const p of slots) {
      window.paypal.Buttons({
        style: { shape: "rect", color: "gold", layout: "vertical", label: "subscribe", height: 45 },
        createSubscription: (data, actions) =>
          actions.subscription.create({ plan_id: p.paypal_plan_id, custom_id: String(B.user_id) }),
        onApprove: async (data) => {
          try {
            B = await api("/api/billing/confirm", { body: { subscription_id: data.subscriptionID } });
            toast(`You're on ${B.plan.name}. Thanks!`);
            render();
          } catch (e) {
            toast(e.message + " If you were charged, it will appear within a few minutes.");
          }
        },
        onError: () => toast("PayPal hit a problem. You weren't charged. Try again."),
      }).render(`#pp-${p.id}`);
    }
  }

  B = await api("/api/billing");
  render();
  if (location.hash === "#privacy") document.getElementById("privacy")?.scrollIntoView();
  if (location.hash === "#test") document.getElementById("test")?.scrollIntoView();
})();
