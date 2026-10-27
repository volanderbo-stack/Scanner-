/* Сканер тексту: підготовка картинки (OpenCV.js) і обробка тексту — порт ocr.py. */
(function (root) {
  'use strict';

  // ---------- Canvas ----------

  function canvas(w, h) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w));
    c.height = Math.max(1, Math.round(h));
    return c;
  }

  // Малює джерело (bitmap/canvas) у сірий масив розміром w×h
  function grayFrom(src, w, h, sx = 0, sy = 0, sw = src.width, sh = src.height) {
    const c = canvas(w, h);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, sx, sy, sw, sh, 0, 0, c.width, c.height);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    const g = new Uint8Array(c.width * c.height);
    for (let i = 0, j = 0; i < g.length; i++, j += 4) {
      g[i] = (d[j] * 77 + d[j + 1] * 150 + d[j + 2] * 29) >> 8;
    }
    return { g, w: c.width, h: c.height };
  }

  // ---------- Підготовка (порт ocr.py на OpenCV.js) ----------
  // Усі функції отримують і повертають cv.Mat (CV_8UC1). Вхідні Mat не видаляють.

  function median(arr) {
    if (!arr.length) return 0;
    const s = Array.from(arr).sort((a, b) => a - b);
    return s[s.length >> 1];
  }

  function percentile(data, p) {
    const hist = new Uint32Array(256);
    for (let i = 0; i < data.length; i++) hist[data[i]]++;
    let acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= data.length * p) return v; }
    return 255;
  }

  // Скріншот: багато однакових сусідніх пікселів (рівні заливки, немає шуму сенсора).
  // Рахуємо на шматку в натуральному розмірі, бо масштабування змазує пікселі.
  function isScreenshot(bitmap) {
    const cw = Math.min(600, bitmap.width), ch = Math.min(600, bitmap.height);
    const sx = (bitmap.width - cw) >> 1, sy = (bitmap.height - ch) >> 1;
    const { g, w, h } = grayFrom(bitmap, cw, ch, sx, sy, cw, ch);
    let same = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 1; x < w; x++) if (g[y * w + x] === g[y * w + x - 1]) same++;
    }
    return same / (h * (w - 1)) > 0.75;   // скріншоти на тестах ≥ 0.87, фото ≤ 0.62
  }

  function resized(cv, src, f, interp) {
    const dst = new cv.Mat();
    cv.resize(src, dst, new cv.Size(Math.max(1, Math.round(src.cols * f)), Math.max(1, Math.round(src.rows * f))), 0, 0,
      interp != null ? interp : (f > 1 ? cv.INTER_CUBIC : cv.INTER_AREA));
    return dst;
  }

  // Світлий текст на темному фоні → інвертувати (дивимося локально, тіні не заважають)
  function maybeInvert(cv, g) {
    const local = new cv.Mat();
    cv.GaussianBlur(g, local, new cv.Size(0, 0), 12);
    let dark = 0, bright = 0;
    const a = g.data, b = local.data;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      if (d < -25) dark++; else if (d > 25) bright++;
    }
    local.delete();
    const out = g.clone();
    if (bright > 1.5 * dark) cv.bitwise_not(g, out);
    return { mat: out, inverted: bright > 1.5 * dark };
  }

  function orderQuad(pts) {
    const s = pts.map((p) => p[0] + p[1]), d = pts.map((p) => p[1] - p[0]);
    const idx = (arr, fn) => arr.indexOf(fn(...arr));
    return [pts[idx(s, Math.min)], pts[idx(d, Math.min)], pts[idx(s, Math.max)], pts[idx(d, Math.max)]];
  }

  // Шукає аркуш (найбільший чотирикутник за межами Canny або за яскравістю)
  function findPage(cv, g) {
    const k = 800 / Math.max(g.rows, g.cols);
    const small = resized(cv, g, k, cv.INTER_AREA);
    cv.GaussianBlur(small, small, new cv.Size(5, 5), 0);
    // Темні фото: розтягнути контраст, інакше межі аркуша не видно
    const lo = percentile(small.data, 0.01), hi = percentile(small.data, 0.99);
    small.convertTo(small, cv.CV_8U, 255 / Math.max(1, hi - lo), -lo * 255 / Math.max(1, hi - lo));
    const med = median(small.data.filter((_, i) => i % 7 === 0));
    const masks = [];
    const e = new cv.Mat();
    cv.Canny(small, e, Math.max(10, 0.4 * med), Math.max(30, 0.9 * med));
    const k5 = cv.Mat.ones(5, 5, cv.CV_8U);
    cv.dilate(e, e, k5);
    masks.push(e);
    const t = new cv.Mat();
    cv.threshold(small, t, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    const k15 = cv.Mat.ones(15, 15, cv.CV_8U);
    cv.morphologyEx(t, t, cv.MORPH_CLOSE, k15);
    masks.push(t);
    const areaImg = small.rows * small.cols;
    let best = null, bestArea = 0;
    for (const m of masks) {
      const cnts = new cv.MatVector(), hier = new cv.Mat();
      // RETR_LIST: внутрішній край лінії-межі чистіший за зовнішній
      cv.findContours(m, cnts, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
      const list = [];
      for (let i = 0; i < cnts.size(); i++) { const c = cnts.get(i); list.push([cv.contourArea(c), c]); }
      list.sort((x, y) => y[0] - x[0]);
      list.slice(0, 8).forEach(([, c]) => {
        const hull = new cv.Mat();
        cv.convexHull(c, hull, false, true);
        const peri = cv.arcLength(hull, true);
        let ap = new cv.Mat();
        for (const eps of [0.02, 0.03, 0.05]) {
          ap.delete(); ap = new cv.Mat();
          cv.approxPolyDP(hull, ap, eps * peri, true);
          if (ap.rows === 4) break;
        }
        if (ap.rows === 4) {
          const a = cv.contourArea(ap);
          // чотирикутник має добре описувати контур
          if (cv.contourArea(hull) >= 0.9 * a && a > 0.15 * areaImg && a < 0.97 * areaImg && a > bestArea) {
            const pts = [];
            for (let i = 0; i < 4; i++) pts.push([ap.data32S[2 * i], ap.data32S[2 * i + 1]]);
            const quad = orderQuad(pts);
            // кут на межі кадру — це обрізаний кадр або тінь, а не край аркуша
            const onEdge = quad.some(([x, y]) => x < 3 || y < 3 || x > small.cols - 3 || y > small.rows - 3);
            if (!onEdge) { best = quad; bestArea = a; }
          }
        }
        hull.delete(); ap.delete();
      });
      for (let i = 0; i < cnts.size(); i++) cnts.get(i).delete();
      cnts.delete(); hier.delete();
    }
    [small, e, t, k5, k15].forEach((x) => x.delete());
    return best ? best.map(([x, y]) => [x / k, y / k]) : null;
  }

  function warpPage(cv, g, quad) {
    const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
    const [tl, tr, br, bl] = quad;
    const wd = Math.round(Math.max(dist(tr, tl), dist(br, bl)));
    const ht = Math.round(Math.max(dist(bl, tl), dist(br, tr)));
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, [].concat(...quad));
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, wd, 0, wd, ht, 0, ht]);
    const M = cv.getPerspectiveTransform(src, dst);
    const out = new cv.Mat();
    cv.warpPerspective(g, out, M, new cv.Size(wd, ht), cv.INTER_CUBIC);
    // Обрізати краї аркуша, щоб не лізли тіні/обрамлення
    const p = Math.max(2, Math.floor(0.01 * Math.min(wd, ht)));
    const roi = out.roi(new cv.Rect(p, p, wd - 2 * p, ht - 2 * p));
    const res = roi.clone();
    [src, dst, M, out, roi].forEach((x) => x.delete());
    return res;
  }

  // Медіанна висота символів (px) за зв'язними компонентами
  function textHeight(cv, g) {
    const b = new cv.Mat(), lab = new cv.Mat(), st = new cv.Mat(), cen = new cv.Mat();
    cv.threshold(g, b, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    const n = cv.connectedComponentsWithStats(b, lab, st, cen, 8, cv.CV_32S);
    const hs = [];
    const S = st.data32S;
    for (let i = 1; i < n; i++) {
      const w = S[i * 5 + 2], h = S[i * 5 + 3], area = S[i * 5 + 4];
      if (h >= 4 && h <= 200 && w <= 3 * h && area >= 6) hs.push(h);
    }
    [b, lab, st, cen].forEach((x) => x.delete());
    return hs.length >= 10 ? median(hs) : null;
  }

  // Ділення на фон: прибирає тіні й нерівне світло, текст лишається
  function normalizeIllum(cv, g) {
    const th = textHeight(cv, g) || Math.min(g.rows, g.cols) / 40;
    const k = Math.floor(Math.max(15, 2.5 * th)) | 1;   // ядро більше за літери, але менше за тінь
    const bg = new cv.Mat();
    const ker = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(k, k));
    cv.morphologyEx(g, bg, cv.MORPH_CLOSE, ker);
    cv.medianBlur(bg, bg, k > 21 ? 21 : k);
    const gf = new cv.Mat(), bf = new cv.Mat(), out = new cv.Mat();
    g.convertTo(gf, cv.CV_32F);
    bg.convertTo(bf, cv.CV_32F, 1, 1);
    cv.divide(gf, bf, out, 255);
    const res = new cv.Mat();
    out.convertTo(res, cv.CV_8U);
    [bg, ker, gf, bf, out].forEach((x) => x.delete());
    return res;
  }

  function rescale(cv, g, target) {
    const h = textHeight(cv, g);
    if (!h) return { mat: g.clone(), f: 1 };
    let f = target / h;
    if (f >= 0.8 && f <= 1.25) return { mat: g.clone(), f: 1 };
    f = Math.min(f, 4);
    if (f < 1 && Math.max(g.rows, g.cols) * f < 1000) return { mat: g.clone(), f: 1 };
    // Пам'ять телефона: не більше ~16 Мп
    f = Math.min(f, Math.sqrt(16e6 / (g.rows * g.cols)));
    if (f >= 0.8 && f <= 1.25) return { mat: g.clone(), f: 1 };
    return { mat: resized(cv, g, f), f };
  }

  // Кут нахилу: поворот, де профіль рядків найконтрастніший
  function skewAngle(cv, g) {
    const k = 1000 / Math.max(g.rows, g.cols);
    const small = k < 1 ? resized(cv, g, k, cv.INTER_AREA) : g.clone();
    const b = new cv.Mat();
    cv.threshold(small, b, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    const w = b.cols, h = b.rows, c = new cv.Point(w / 2, h / 2);
    const r = new cv.Mat();
    const score = (a) => {
      const M = cv.getRotationMatrix2D(c, a, 1);
      cv.warpAffine(b, r, M, new cv.Size(w, h), cv.INTER_NEAREST);
      M.delete();
      let prev = null, sc = 0;
      for (let y = 0; y < h; y++) {
        let sum = 0;
        const row = r.data.subarray(y * w, (y + 1) * w);
        for (let x = 0; x < w; x++) sum += row[x];
        if (prev !== null) sc += (sum - prev) ** 2;
        prev = sum;
      }
      return sc;
    };
    let best = 0, bestS = -1;
    for (let a = -10; a <= 10.001; a += 1) { const v = score(a); if (v > bestS) { bestS = v; best = a; } }
    const c0 = best;
    for (let a = c0 - 1; a <= c0 + 1.001; a += 0.2) { const v = score(a); if (v > bestS) { bestS = v; best = a; } }
    [small, b, r].forEach((x) => x.delete());
    return best;
  }

  function rotate(cv, g, angle) {
    const w = g.cols, h = g.rows;
    const M = cv.getRotationMatrix2D(new cv.Point(w / 2, h / 2), angle, 1);
    const cos = Math.abs(M.doubleAt(0, 0)), sin = Math.abs(M.doubleAt(0, 1));
    const nw = Math.round(h * sin + w * cos), nh = Math.round(h * cos + w * sin);
    M.doublePtr(0, 2)[0] += nw / 2 - w / 2;
    M.doublePtr(1, 2)[0] += nh / 2 - h / 2;
    const out = new cv.Mat();
    const bgv = median(g.data.filter((_, i) => i % 97 === 0));
    // Кути заповнюємо кольором фону
    cv.warpAffine(g, out, M, new cv.Size(nw, nh), cv.INTER_CUBIC, cv.BORDER_CONSTANT, new cv.Scalar(bgv));
    M.delete();
    return out;
  }

  // Повна підготовка (як preprocess() в ocr.py). bitmap уже повернутий за EXIF.
  function prepare(cv, bitmap, log = []) {
    const shot = isScreenshot(bitmap);
    log.push(shot ? 'скріншот' : 'фото');
    // Робочий розмір: не більше 3000 px по довшій стороні
    const k = Math.min(1, 3000 / Math.max(bitmap.width, bitmap.height));
    const c = canvas(bitmap.width * k, bitmap.height * k);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, c.width, c.height);
    const rgba = cv.imread(c);
    let g = new cv.Mat();
    cv.cvtColor(rgba, g, cv.COLOR_RGBA2GRAY);
    rgba.delete();
    const step = (next) => { g.delete(); g = next; };

    if (shot) {
      // Скріншот майже не чіпаємо: інверсія темної теми й збільшення
      step(maybeInvert(cv, g).mat);
      const r = rescale(cv, g, 32);
      step(r.mat);
      if (r.f !== 1) log.push(`масштаб ×${r.f.toFixed(2)}`);
    } else {
      const q = findPage(cv, g);
      if (q) { step(warpPage(cv, g, q)); log.push('аркуш знайдено'); }
      const inv = maybeInvert(cv, g);
      step(inv.mat);
      if (inv.inverted) log.push('інверсія');
      step(normalizeIllum(cv, g));
      const r = rescale(cv, g, 32);
      step(r.mat);
      if (r.f !== 1) log.push(`масштаб ×${r.f.toFixed(2)}`);
      const a = skewAngle(cv, g);
      if (Math.abs(a) >= 0.3) { step(rotate(cv, g, a)); log.push(`нахил ${a > 0 ? '+' : ''}${a.toFixed(1)}°`); }
    }
    const out = canvas(g.cols, g.rows);
    cv.imshow(out, g);
    g.delete();
    return { canvas: out, shot };
  }

  // ---------- Обробка тексту ----------

  const CYR = 'а-яґєіїА-ЯҐЄІЇёЁъЪыЫэЭ';
  const RE_CYR = new RegExp(`[${CYR}]`, 'g');
  const RE_CYR1 = new RegExp(`[${CYR}]`);
  const RE_LAT = /[A-Za-z]/g;
  const LAT2CYR = {};
  'aceiopxyABCEHIKMOPTX'.split('').forEach((ch, i) => { LAT2CYR[ch] = 'асеіорхуАВСЕНІКМОРТХ'[i]; });
  LAT2CYR['ï'] = 'ї'; LAT2CYR['Ï'] = 'Ї';
  const CYR2LAT = {};
  for (const [l, c] of Object.entries(LAT2CYR)) if (l !== 'ï' && l !== 'Ï') CYR2LAT[c] = l;
  const ONE_LETTER = new Set(['і', 'й', 'я', 'а', 'в', 'у', 'з', 'о']);
  const EDGE_JUNK = new Set('|\\/_~^`"\'¦°•·;,.:'.split(''));

  const count = (s, re) => (s.match(re) || []).length;
  const mapChars = (s, m) => s.split('').map((ch) => m[ch] || ch).join('');

  function fixWord(w, prevC, nextC) {
    const nc = count(w, RE_CYR), nl = count(w, RE_LAT);
    if (nc && nl) return nc >= nl ? mapChars(w, LAT2CYR) : mapChars(w, CYR2LAT);
    const core = w.replace(/[^\p{L}\p{N}_]/gu, '');
    // окремих «i» та «y» в англійській немає → це «і», «у»
    if ((core === 'i' || core === 'y') && (prevC || nextC)) return w.replace(core, LAT2CYR[core]);
    if (nl && !nc && core.length > 0 && core.length <= 2 && prevC && nextC &&
        core.split('').every((ch) => LAT2CYR[ch])) return mapChars(w, LAT2CYR);
    // «3BIT» → «звіт»
    if (nl && core.includes('3') && prevC && nextC &&
        core.split('').every((ch) => LAT2CYR[ch] || ch === '3')) {
      return w.split('').map((ch) => (ch === '3' ? 'з' : LAT2CYR[ch] || ch)).join('').toLowerCase();
    }
    return w;
  }

  function fixHomoglyphs(text) {
    return text.split('\n').map((line) => {
      const ws = line.split(' ');
      const isc = ws.map((w) => RE_CYR1.test(w) && count(w, RE_CYR) >= count(w, RE_LAT));
      return ws.map((w, i) => {
        const prevC = i > 0 ? isc[i - 1] : (i + 1 < ws.length && isc[i + 1]);
        const nextC = i + 1 < ws.length ? isc[i + 1] : (i > 0 && isc[i - 1]);
        const parts = w.split(/([_=/.:-]+)/);
        if (parts.length > 1) return parts.map((p, k) => (k % 2 === 0 ? fixWord(p, prevC, nextC) : p)).join('');
        return fixWord(w, prevC, nextC);
      }).join(' ');
    }).join('\n');
  }

  function isJunk(L) {
    const t = L.words.join(' ').trim();
    if (!t) return true;
    const compact = t.replace(/ /g, '');
    const alnum = (t.match(/[\p{L}\p{N}]/gu) || []).length;
    if (compact.length <= 2 && !/^\d+$/.test(t) && !ONE_LETTER.has(t.toLowerCase())) return true;
    if (alnum / compact.length < 0.4) return true;
    if (L.confs.length && Math.max(...L.confs) < 50 && alnum <= 6) return true;
    return false;
  }

  function trimEdges(L) {
    const bad = (w, c) => {
      const core = w.replace(/[^\p{L}\p{N}_]/gu, '');
      if (ONE_LETTER.has(core.toLowerCase())) return false;
      return w.split('').every((ch) => EDGE_JUNK.has(ch)) || (core.length <= 2 && c < 50);
    };
    if (L.words.length > 1 && (L.words[0] === '|' || L.words[0] === ']') && /^[а-яґєії]/.test(L.words[1])) L.words[0] = 'І';
    while (L.words.length > 1 && bad(L.words[0], L.confs[0])) { L.words.shift(); L.confs.shift(); }
    while (L.words.length > 1 && bad(L.words[L.words.length - 1], L.confs[L.confs.length - 1])) { L.words.pop(); L.confs.pop(); }
    return L;
  }

  function joinHyphens(lines) {
    const res = lines.slice();
    for (let i = 0; i < res.length - 1; i++) {
      const m = /(\p{L}+)[-‐¬]$/u.exec(res[i]);
      const nxt = res[i + 1].trimStart();
      if (m && nxt && nxt[0] === nxt[0].toLowerCase() && nxt[0] !== nxt[0].toUpperCase()) {
        const sp = nxt.indexOf(' ');
        const first = sp < 0 ? nxt : nxt.slice(0, sp);
        res[i] = res[i].slice(0, -1) + first;
        res[i + 1] = sp < 0 ? '' : nxt.slice(sp + 1);
      }
    }
    return res.filter((r) => r.trim());
  }

  // Рядки Tesseract.js → текст з абзацами
  function buildText(lines) {
    lines = lines.filter((L) => !isJunk(L)).map(trimEdges);
    if (!lines.length) return '';
    const hs = lines.map((L) => L.box[3] - L.box[1]).sort((a, b) => a - b);
    const mh = hs[hs.length >> 1] || 1;
    const gaps = [];
    for (let i = 1; i < lines.length; i++) gaps.push(lines[i].box[1] - lines[i - 1].box[3]);
    gaps.sort((a, b) => a - b);
    const mgap = gaps.length ? gaps[gaps.length >> 2] : 0;     // звичайний проміжок між рядками
    const left0 = Math.min(...lines.map((L) => L.box[0]));
    const paras = [[lines[0]]];
    for (let i = 1; i < lines.length; i++) {
      const prev = lines[i - 1], L = lines[i];
      const gap = L.box[1] - prev.box[3];
      const newPar = gap > mgap + 0.5 * mh || gap > mh ||
        (L.box[0] - prev.box[0] > 1.2 * mh && Math.abs(gap) < 2 * mh && L.box[0] - left0 < 6 * mh) ||
        gap < -0.5 * mh;
      if (newPar) paras.push([L]); else paras[paras.length - 1].push(L);
    }
    return paras.map((p) => {
      let para = joinHyphens(p.map((L) => L.words.join(' '))).join('\n');
      if (p.length > 1 && para.endsWith(',')) para = para.slice(0, -1) + '.';   // абзац не закінчується комою
      return para;
    }).join('\n\n');
  }

  function postprocess(text) {
    text = text.replace(/(?<=[\p{L}\p{N}])['`´ʼ‘’"](?=[\p{L}\p{N}])/gu, '’');
    text = text.replace(/(?<= )--?(?=\s|$)/gm, '—');
    text = text.replace(/^[|\]] (?=[а-яґєії])/gm, 'І ');
    text = text.replace(/(?<![\p{L}\p{N}])(?:Ne|N°|Nº|N2|Мо) ?(?=\d)|(?<![\p{L}\p{N}])(?:її|їй|&) ?(?=\d{3})/gu, '№ ');
    text = fixHomoglyphs(text);
    return text.replace(/[ \t]+/g, ' ').trim();
  }

  // Результат Tesseract.js (blocks) → рядки {words, confs, box}
  function linesFromResult(data) {
    const out = [];
    for (const b of data.blocks || []) {
      for (const p of b.paragraphs || []) {
        for (const l of p.lines || []) {
          const words = [], confs = [];
          for (const w of l.words || []) {
            const t = (w.text || '').trim();
            if (!t) continue;
            words.push(t); confs.push(w.confidence);
          }
          if (words.length) out.push({ words, confs, box: [l.bbox.x0, l.bbox.y0, l.bbox.x1, l.bbox.y1] });
        }
      }
    }
    return out;
  }

  root.OcrCore = { prepare, linesFromResult, buildText, postprocess };
})(typeof window !== 'undefined' ? window : globalThis);
