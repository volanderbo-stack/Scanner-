/* Сканер тексту: інтерфейс і робота з Tesseract.js */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const base = new URL('./', location.href).href;
  const LAST_KEY = 'scanner:last';
  const MODE_KEY = 'scanner:mode';

  // ---------- Tesseract (створюємо один раз) ----------

  let workerPromise = null;
  let onProgress = () => {};

  const STAGES = {
    'loading tesseract core': 'Завантажую розпізнавач…',
    'initializing tesseract': 'Запускаю розпізнавач…',
    'loading language traineddata': 'Завантажую мовні моделі (лише перший раз)…',
    'loading language traineddata (from cache)': 'Завантажую мовні моделі…',
    'initializing api': 'Готую мови…',
    'recognizing text': 'Розпізнаю текст…',
  };

  function getWorker() {
    if (!workerPromise) {
      workerPromise = (async () => {
        const w = await Tesseract.createWorker(['ukr', 'eng'], 1, {
          workerPath: base + 'vendor/worker.min.js',
          corePath: base + 'vendor/core',
          langPath: base + 'lang',
          workerBlobURL: false,
          cacheMethod: 'none',       // кешує service worker
          logger: (m) => onProgress(m),
        });
        await w.setParameters({ tessedit_pageseg_mode: '3', user_defined_dpi: '300' });
        return w;
      })();
      workerPromise.catch(() => { workerPromise = null; });
    }
    return workerPromise;
  }

  // ---------- OpenCV.js (підвантажуємо при першій потребі) ----------

  let cvPromise = null;
  function getCv() {
    if (!cvPromise) {
      cvPromise = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = base + 'vendor/opencv/opencv.js';
        s.async = true;
        s.onerror = () => { cvPromise = null; reject(new Error('opencv')); };
        s.onload = () => {
          // Збірка може віддати Promise або модуль, який дозріває сам
          const wait = () => {
            const cv = window.cv;
            if (cv && cv.Mat) resolve(cv);
            else if (cv && typeof cv.then === 'function' && !cv.__waiting) {
              cv.__waiting = true;
              cv.then((m) => { if (m) { delete m.then; window.cv = m; } wait(); });
            } else setTimeout(wait, 50);
          };
          wait();
        };
        document.head.appendChild(s);
      });
    }
    return cvPromise;
  }

  // ---------- Завантаження картинки ----------

  async function loadBitmap(blob) {
    try {
      return await createImageBitmap(blob, { imageOrientation: 'from-image' });
    } catch (e) {
      // Запасний шлях: <img> сам повертає фото за EXIF
      const url = URL.createObjectURL(blob);
      try {
        const img = new Image();
        img.src = url;
        await img.decode();
        const c = document.createElement('canvas');
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        c.getContext('2d').drawImage(img, 0, 0);
        return c;
      } finally {
        URL.revokeObjectURL(url);
      }
    }
  }

  const tick = () => new Promise((r) => setTimeout(r, 30));

  // Повний цикл: картинка → текст
  async function recognize(blob, progress = () => {}) {
    const t0 = performance.now();
    const log = [];
    const workerReady = getWorker();          // моделі вантажаться паралельно з підготовкою
    progress('Готую фото…', 5);
    const [cv, bitmap] = await Promise.all([getCv(), loadBitmap(blob)]);
    await tick();
    const prepared = OcrCore.prepare(cv, bitmap, log);
    if (bitmap.close) bitmap.close();
    progress('Готую фото…', 15);
    await tick();
    const worker = await workerReady;
    const { data } = await worker.recognize(prepared.canvas, {}, { blocks: true, text: true });
    const text = OcrCore.postprocess(OcrCore.buildText(OcrCore.linesFromResult(data)));
    return { text, log, seconds: (performance.now() - t0) / 1000 };
  }

  // Для автоматичних тестів
  window.ScannerOCR = { recognize };

  // ---------- Інтерфейс ----------

  const views = { start: $('start'), work: $('work'), result: $('result') };
  function show(name) {
    for (const [k, el] of Object.entries(views)) el.hidden = k !== name;
    $('bottom').hidden = name !== 'result';
    window.scrollTo(0, 0);
  }

  let toastTimer;
  function toast(msg) {
    const t = $('toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
  }

  function setProgress(text, pct) {
    $('status').textContent = text;
    $('bar').style.width = Math.max(0, Math.min(100, pct)) + '%';
  }

  let busy = false;
  let previewUrl = null;
  let rawText = '';          // текст з рядками як на фото
  let mode = 'para';
  try { mode = localStorage.getItem(MODE_KEY) || 'para'; } catch (e) { /* ignore */ }

  // «Абзаци»: рядки всередині абзацу склеюються — зручно вставляти в документ
  const asParagraphs = (t) => t.split(/\n{2,}/).map((p) => p.replace(/\s*\n\s*/g, ' ')).join('\n\n');

  function render() {
    $('text').value = mode === 'para' ? asParagraphs(rawText) : rawText;
    $('modePara').classList.toggle('on', mode === 'para');
    $('modeLines').classList.toggle('on', mode === 'lines');
    $('modePara').setAttribute('aria-pressed', mode === 'para');
    $('modeLines').setAttribute('aria-pressed', mode === 'lines');
  }

  for (const [id, m] of [['modePara', 'para'], ['modeLines', 'lines']]) {
    $(id).addEventListener('click', () => {
      mode = m;
      try { localStorage.setItem(MODE_KEY, m); } catch (e) { /* ignore */ }
      render();
    });
  }

  async function handle(blob) {
    if (!blob || busy) return;
    if (blob.type && !blob.type.startsWith('image/')) { toast('Це не картинка'); return; }
    busy = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob);
    $('preview').src = previewUrl;
    $('thumb').src = previewUrl;
    $('thumb').hidden = false;
    show('work');
    setProgress('Готую фото…', 3);
    onProgress = (m) => {
      const label = STAGES[m.status] || 'Працюю…';
      if (m.status === 'recognizing text') setProgress(label, 20 + 80 * (m.progress || 0));
      else setProgress(label, 15);
    };
    try {
      const { text, seconds } = await recognize(blob, setProgress);
      rawText = text || '';
      render();
      $('meta').textContent = text
        ? `Готово за ${seconds.toFixed(1)} с · ${text.split(/\s+/).filter(Boolean).length} слів`
        : 'Текст не знайдено. Спробуй сфотографувати ближче й рівніше.';
      try { localStorage.setItem(LAST_KEY, text); } catch (e) { /* приватний режим */ }
      show('result');
    } catch (e) {
      console.error(e);
      toast(String(e && e.message || e).includes('decode') || e.name === 'InvalidStateError'
        ? 'Не вдалося відкрити фото. Спробуй JPG або PNG.'
        : 'Щось пішло не так. Спробуй ще раз.');
      show('start');
    } finally {
      busy = false;
    }
  }

  for (const id of ['camera', 'gallery']) {
    $(id).addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      e.target.value = '';
      handle(f);
    });
  }

  // Вставка скріншота з буфера (Ctrl+V) і перетягування на комп'ютері
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) { e.preventDefault(); handle(item.getAsFile()); }
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    handle(e.dataTransfer?.files?.[0]);
  });

  $('copy').addEventListener('click', async () => {
    const t = $('text').value;
    try {
      await navigator.clipboard.writeText(t);
    } catch (e) {
      $('text').select();
      document.execCommand('copy');
    }
    toast('Скопійовано');
  });

  $('share').addEventListener('click', async () => {
    const t = $('text').value;
    if (navigator.share) {
      try { await navigator.share({ text: t }); } catch (e) { /* користувач закрив */ }
    } else {
      try { await navigator.clipboard.writeText(t); toast('Скопійовано — встав куди треба'); } catch (e) { /* ignore */ }
    }
  });

  $('again').addEventListener('click', () => show('start'));

  // Останній текст (браузер на телефоні може перезавантажити сторінку у фоні)
  try {
    const last = localStorage.getItem(LAST_KEY);
    if (last) {
      $('last').hidden = false;
      $('last').addEventListener('click', () => {
        rawText = last;
        render();
        $('meta').textContent = 'Останній розпізнаний текст';
        $('thumb').hidden = true;
        show('result');
      });
    }
  } catch (e) { /* ignore */ }

  // Офлайн-режим
  if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }

  // Моделі починаємо вантажити одразу, щоб перше фото оброблялося швидше
  if (!navigator.webdriver) {
    setTimeout(() => { getWorker().catch(() => {}); getCv().catch(() => {}); }, 1500);
  }
})();
