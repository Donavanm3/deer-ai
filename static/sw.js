// Deer AI service worker: makes the app installable and loads the shell fast.
// API calls always go to the network; pages and assets fall back to cache when offline.
const CACHE = "deer-shell-v5";
const SHELL = ["./", "admin.html", "plans.html", "developers.html", "static/style.css", "static/config.js",
  "static/common.js", "static/app.js", "static/admin.js", "static/plans.js", "static/developers.js", "static/icons/icon-192.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.includes("/api/") || url.pathname.includes("/v1/")) return;
  e.respondWith(fetch(e.request, { cache: "no-cache" }).then((res) => {
    const copy = res.clone();
    if (res.ok) caches.open(CACHE).then((c) => c.put(e.request, copy));
    return res;
  }).catch(() => caches.match(e.request)));
});
