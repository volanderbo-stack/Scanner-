/* Офлайн-кеш. Змінили файли → збільшити VERSION. */
const VERSION = 'scanner-v2';
const FILES = [
  './', './index.html', './app.css', './app.js', './ocr-core.js', './manifest.json',
  './icons/icon.svg', './icons/icon-192.png', './icons/icon-512.png',
  './vendor/tesseract.min.js', './vendor/worker.min.js',
];
// Великі файли (ядро й мовні моделі) кешуються при першому використанні
const LAZY = /\/(vendor\/core|vendor\/opencv|lang)\//;

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  if (LAZY.test(req.url)) {
    // Моделі не змінюються: спершу кеш
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
      return res;
    })));
    return;
  }
  // Сторінка й скрипти: спершу мережа (щоб бачити оновлення), без мережі — кеш
  e.respondWith(fetch(req).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
    return res;
  }).catch(() => caches.match(req, { ignoreSearch: true })));
});
