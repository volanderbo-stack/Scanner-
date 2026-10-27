"""Розпізнавання тексту з фото й скріншотів (українська + англійська).

Запуск:
    python ocr.py photo.jpg
    python ocr.py папка/
Текст друкується в консоль і зберігається в .txt поруч із картинкою.
"""
import argparse
import glob
import os
import re
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # HEIC з айфона
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

import pytesseract

# psm запускаємо паралельно самі, тож внутрішні потоки Tesseract лише заважають
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- Налаштування ----------------

@dataclass
class Settings:
    """Що вмикати. Кожен крок можна вимкнути, щоб evaluate.py міряв його внесок."""
    lang: str = "ukr+eng"
    model: str = "best"          # best | fast — яку теку tessdata брати
    psms: tuple = (3, 4, 6)      # режими сторінки; береться той, де вища впевненість
    # Підготовка
    exif: bool = True            # повернути за EXIF
    auto_type: bool = True       # визначати скріншот/фото (False = все як фото)
    kind: str = "auto"           # auto | photo | screenshot — ручний вибір типу
    invert: bool = True          # світлий текст на темному → інвертувати
    doc: bool = True             # знайти аркуш і виправити перспективу
    illum: bool = True           # вирівняти освітлення (прибрати тіні)
    stretch: bool = False        # розтягнути контраст — не дав стабільного виграшу
    clahe: bool = False          # локальний контраст (CLAHE) — погіршує, вимкнено
    scale: bool = True           # збільшити дрібний текст
    target_h: int = 32           # бажана висота літер, px
    denoise: bool = False        # прибрати шум — не дав стабільного виграшу
    deskew: bool = True          # вирівняти нахил
    binarize: object = False     # False | True (адаптивна) | "otsu" — погіршує, вимкнено
    # Обробка тексту
    hyphen: bool = True          # склеїти переноси
    junk: bool = True            # прибрати сміттєві рядки
    homoglyph: bool = True       # латиниця/кирилиця в словах
    punct: bool = True           # «--» → «—», «| » → «І», знак №, кома в кінці абзацу
    paragraphs: bool = True      # відновити абзаци
    # Рушій
    engine: str = "tesseract"    # tesseract | easyocr
    tessdata: str = ""           # тека з моделями (порожньо = знайти самому)
    debug_dir: str = ""          # куди зберегти підготовлену картинку


DEFAULT = Settings()


# ---------------- Пошук Tesseract і моделей ----------------

def find_tesseract():
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
              os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")]:
        if os.path.exists(p):
            return p
    return None


def has_models(d, lang):
    return d and all(os.path.exists(os.path.join(d, l + ".traineddata"))
                     for l in lang.split("+"))


def find_tessdata(s: Settings):
    """Тека з моделями: --tessdata, змінна TESSDATA_BEST/FAST, ./tessdata_best, тека Tesseract."""
    if s.tessdata:
        return s.tessdata
    cands = [os.environ.get("TESSDATA_" + s.model.upper(), ""),
             os.path.join(HERE, "tessdata_" + s.model),
             os.path.join("/opt", "tessdata_" + s.model)]
    if s.model == "best":
        cands.append(os.path.join(HERE, "tessdata"))
        exe = find_tesseract()
        if exe:
            cands.append(os.path.join(os.path.dirname(exe), "tessdata"))
        cands.append(os.environ.get("TESSDATA_PREFIX", ""))
    for d in cands:
        if has_models(d, s.lang):
            return d
    return ""


# ---------------- Завантаження ----------------

def load_image(path, exif=True):
    """Повертає (BGR-масив, чи є EXIF камери)."""
    im = Image.open(path)
    ex = im.getexif()
    from_camera = bool(ex.get(0x010F) or ex.get(0x0110) or ex.get(0x8769))
    if exif:
        im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), from_camera


def is_screenshot(bgr, from_camera):
    """Скріншот: багато однакових сусідніх пікселів (рівні заливки, нема шуму сенсора)."""
    if from_camera:
        return False
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat = float(np.mean(g[:, 1:] == g[:, :-1]))
    return flat > 0.75   # скріншоти на тестах ≥ 0.87, фото ≤ 0.62


# ---------------- Підготовка картинки ----------------

def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def maybe_invert(g):
    """Світлий текст на темному фоні → інвертувати.

    Дивимося локально: чого більше — пікселів, помітно темніших за своє
    оточення, чи світліших. Тіні й градієнти на це не впливають.
    """
    local = cv2.GaussianBlur(g, (0, 0), 12).astype(np.int16)
    d = g.astype(np.int16) - local
    dark, bright = np.count_nonzero(d < -25), np.count_nonzero(d > 25)
    return 255 - g if bright > 1.5 * dark else g


def order_quad(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_page(bgr):
    """Шукає аркуш (найбільший світлий чотирикутник). Повертає 4 кути або None."""
    h, w = bgr.shape[:2]
    k = 800.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(to_gray(small), (5, 5), 0)
    # Темні фото: розтягнути контраст, інакше межі аркуша не видно
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    g = np.clip((g.astype(np.float32) - lo) * 255 / max(1.0, hi - lo), 0, 255).astype(np.uint8)
    masks = []
    # 1) межі (Canny, пороги від медіани яскравості)
    med = float(np.median(g))
    e = cv2.Canny(g, max(10, 0.4 * med), max(30, 0.9 * med))
    masks.append(cv2.dilate(e, np.ones((5, 5), np.uint8)))
    # 2) аркуш світліший за тло (Otsu)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))
    best, best_area = None, 0
    area_img = small.shape[0] * small.shape[1]
    for m in masks:
        # RETR_LIST: внутрішній край лінії-межі чистіший за зовнішній (до нього липне тло)
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.03, 0.05):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    break
            if len(ap) != 4:
                continue
            a = cv2.contourArea(ap)
            # чотирикутник має добре описувати контур, а не бути «натягнутим» на пляму
            if cv2.contourArea(hull) < 0.9 * a:
                continue
            # Аркуш має займати чималу частину кадру, але не весь кадр
            if 0.15 * area_img < a < 0.97 * area_img and a > best_area:
                quad = order_quad(ap)
                # кут на межі кадру — це не край аркуша, а обрізаний кадр або тінь
                margin = 3
                on_edge = sum((p[0] < margin or p[1] < margin or
                               p[0] > small.shape[1] - margin or
                               p[1] > small.shape[0] - margin) for p in quad)
                if on_edge == 0:
                    best, best_area = quad, a
    return None if best is None else best / k


def warp_page(bgr, quad):
    tl, tr, br, bl = quad
    wd = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    ht = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [wd, 0], [wd, ht], [0, ht]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(bgr, M, (wd, ht), flags=cv2.INTER_CUBIC)
    # Обрізати краї аркуша, щоб не лізли тіні/обрамлення
    p = max(2, int(0.01 * min(wd, ht)))
    return out[p:-p, p:-p]


def normalize_illum(g):
    """Ділення на фон: прибирає тіні й нерівне світло, текст лишається."""
    th = text_height(g) or min(g.shape) / 40
    k = int(max(15, 2.5 * th)) | 1   # ядро більше за літери, але менше за тінь
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.medianBlur(bg, 21 if k > 21 else k)
    out = cv2.divide(g.astype(np.float32), bg.astype(np.float32) + 1, scale=255)
    return np.clip(out, 0, 255).astype(np.uint8)


def stretch_contrast(g):
    """Найтемніше 1% → 0, найсвітліше 0.5% → 255."""
    lo, hi = np.percentile(g, 1), np.percentile(g, 99.5)
    out = (g.astype(np.float32) - lo) * 255 / max(1.0, hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def text_height(g):
    """Медіанна висота символів (px) за зв'язними компонентами."""
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
    hs = [st[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)
          if 4 <= st[i, cv2.CC_STAT_HEIGHT] <= 200
          and st[i, cv2.CC_STAT_WIDTH] <= 3 * st[i, cv2.CC_STAT_HEIGHT]
          and st[i, cv2.CC_STAT_AREA] >= 6]
    return statistics.median(hs) if len(hs) >= 10 else None


def rescale(g, target):
    h = text_height(g)
    if not h:
        return g, 1.0
    f = target / h
    if 0.8 <= f <= 1.25:          # і так добре
        return g, 1.0
    f = min(f, 4.0)
    if f < 1 and max(g.shape) * f < 1000:
        return g, 1.0
    interp = cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA
    return cv2.resize(g, None, fx=f, fy=f, interpolation=interp), f


def skew_angle(g):
    """Кут нахилу: шукаємо поворот, де профіль рядків найконтрастніший."""
    k = 1000.0 / max(g.shape)
    small = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g
    _, b = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = b.shape
    c = (w / 2, h / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    best = max(np.arange(-10, 10.01, 1.0), key=score)
    best = max(np.arange(best - 1, best + 1.01, 0.2), key=score)
    return float(best)


def rotate(g, angle):
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    # Кути заповнюємо кольором фону (BORDER_REPLICATE малює смуги)
    return cv2.warpAffine(g, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=int(np.median(g)))


def preprocess(bgr, screenshot, s: Settings, log):
    """Готує сіру (або ч/б) картинку для OCR."""
    if screenshot:
        # Скріншот майже не чіпаємо: сірий, інверсія темної теми, збільшення
        g = to_gray(bgr)
        if s.invert:
            g = maybe_invert(g)
        if s.scale:
            g, f = rescale(g, s.target_h)
            log.append(f"масштаб ×{f:.2f}")
        return g

    if s.doc:
        q = find_page(bgr)
        if q is not None:
            bgr = warp_page(bgr, q)
            log.append("аркуш знайдено")
    g = to_gray(bgr)
    if s.invert:
        g = maybe_invert(g)
    if s.illum:
        g = normalize_illum(g)
    if s.stretch:
        g = stretch_contrast(g)
    if s.clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    if s.scale:
        g, f = rescale(g, s.target_h)
        log.append(f"масштаб ×{f:.2f}")
    if s.denoise:
        g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if s.deskew:
        a = skew_angle(g)
        if abs(a) >= 0.3:
            g = rotate(g, a)
            log.append(f"нахил {a:+.1f}°")
    if s.binarize == "otsu":
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif s.binarize:
        blk = max(15, int(s.target_h * 1.5) | 1)
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, blk, 15)
    return g


# ---------------- Рушії OCR ----------------

@dataclass
class Line:
    words: list
    confs: list
    box: tuple          # x1, y1, x2, y2

    @property
    def text(self):
        return " ".join(self.words)


def tesseract_lines(g, psm, s: Settings, tessdata):
    cfg = f"--oem 1 --psm {psm}"
    if tessdata:
        cfg += f' --tessdata-dir "{tessdata}"'
    d = pytesseract.image_to_data(g, lang=s.lang, config=cfg,
                                  output_type=pytesseract.Output.DICT)
    lines = {}
    for i, w in enumerate(d["text"]):
        w = w.strip()
        conf = float(d["conf"][i])
        if not w or conf < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        if key not in lines:
            lines[key] = Line([], [], (x, y, x + ww, y + hh))
        L = lines[key]
        L.words.append(w)
        L.confs.append(conf)
        b = L.box
        L.box = (min(b[0], x), min(b[1], y), max(b[2], x + ww), max(b[3], y + hh))
    return list(lines.values())


def mean_conf(lines):
    """Середня впевненість, зважена довжиною слів."""
    tot = n = 0
    for L in lines:
        for w, c in zip(L.words, L.confs):
            tot += c * len(w)
            n += len(w)
    return tot / n if n else 0.0


_easy = None


def easyocr_lines(g):
    global _easy
    import easyocr
    if _easy is None:
        _easy = easyocr.Reader(["uk", "en"], gpu=False, verbose=False)
    res = _easy.readtext(g, detail=1, paragraph=False)
    items = []
    for box, text, conf in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append((min(xs), min(ys), max(xs), max(ys), text, conf * 100))
    # Групуємо фрагменти в рядки за вертикаллю
    items.sort(key=lambda t: (t[1] + t[3]) / 2)
    lines = []
    for it in items:
        cy, h = (it[1] + it[3]) / 2, it[3] - it[1]
        for L in lines:
            ly = (L.box[1] + L.box[3]) / 2
            if abs(ly - cy) < 0.5 * min(h, L.box[3] - L.box[1]):
                L.words.append(it)
                b = L.box
                L.box = (min(b[0], it[0]), min(b[1], it[1]), max(b[2], it[2]), max(b[3], it[3]))
                break
        else:
            lines.append(Line([it], [], it[:4]))
    out = []
    for L in lines:
        L.words.sort(key=lambda t: t[0])
        ws, cs = [], []
        for it in L.words:
            for w in it[4].split():
                ws.append(w)
                cs.append(it[5])
        out.append(Line(ws, cs, tuple(int(v) for v in L.box)))
    return out


def recognize(g, s: Settings, log):
    if s.engine == "easyocr":
        return easyocr_lines(g)
    tessdata = find_tessdata(s)
    if not tessdata:
        log.append(f"УВАГА: моделі tessdata_{s.model} не знайдено, беру стандартні")
    psms = list(s.psms)
    if len(psms) == 1:
        return tesseract_lines(g, psms[0], s, tessdata)
    with ThreadPoolExecutor(len(psms)) as ex:
        results = list(ex.map(lambda p: tesseract_lines(g, p, s, tessdata), psms))
    scores = [mean_conf(r) for r in results]
    k = int(np.argmax(scores))
    log.append("psm " + ", ".join(f"{p}:{c:.0f}" for p, c in zip(psms, scores))
               + f" → {psms[k]}")
    return results[k]


# ---------------- Текст після OCR ----------------

CYR = "а-яґєіїА-ЯҐЄІЇёЁъЪыЫэЭ"
LAT2CYR = dict(zip("aceiopxyABCEHIKMOPTXy", "асеіорхуАВСЕНІКМОРТХу"))
LAT2CYR.update({"ï": "ї", "Ï": "Ї"})
CYR2LAT = {v: k for k, v in LAT2CYR.items() if k not in "ïÏ"}
RE_CYR = re.compile(f"[{CYR}]")
RE_LAT = re.compile("[A-Za-z]")
APOS = re.compile(r"(?<=\w)['`´ʼ‘’\"](?=\w)")


def fix_word(w, prev_cyr, next_cyr):
    nc, nl = len(RE_CYR.findall(w)), len(RE_LAT.findall(w))
    if nc and nl:
        if nc >= nl:
            return "".join(LAT2CYR.get(ch, ch) for ch in w)
        # переважно латинське слово з кириличними «двійниками»
        return "".join(CYR2LAT.get(ch, ch) for ch in w)
    core = re.sub(r"[^\w]", "", w)
    # Окремих «i» та «y» в англійській немає → це українські «і», «у»
    if core in ("i", "y") and (prev_cyr or next_cyr):
        return w.replace(core, LAT2CYR[core])
    # Коротке «латинське» слово серед українських: a → а, o → о
    if nl and not nc and 0 < len(core) <= 2 and prev_cyr and next_cyr \
            and all(ch in LAT2CYR for ch in core):
        return "".join(LAT2CYR.get(ch, ch) for ch in w)
    # «3BIT» → «звіт»: цифра 3 замість «з» разом із латинськими двійниками
    if nl and "3" in core and prev_cyr and next_cyr \
            and all(ch in LAT2CYR or ch == "3" for ch in core):
        return "".join(LAT2CYR.get(ch, "з" if ch == "3" else ch) for ch in w).lower()
    return w


def fix_homoglyphs(text):
    out = []
    for line in text.split("\n"):
        ws = line.split(" ")
        isc = [bool(RE_CYR.search(w)) and len(RE_CYR.findall(w)) >= len(RE_LAT.findall(w))
               for w in ws]
        res = []
        for i, w in enumerate(ws):
            prev_c = isc[i - 1] if i > 0 else (i + 1 < len(ws) and isc[i + 1])
            next_c = isc[i + 1] if i + 1 < len(ws) else (i > 0 and isc[i - 1])
            parts = re.split(r"([_=/.:-]+)", w)
            if len(parts) > 1:  # «BOT_TOKEN=твій_токен» — кожну частину окремо
                res.append("".join(fix_word(p, prev_c, next_c) if k % 2 == 0 else p
                                   for k, p in enumerate(parts)))
            else:
                res.append(fix_word(w, prev_c, next_c))
        out.append(" ".join(res))
    return "\n".join(out)


def is_junk(L: Line):
    t = L.text.strip()
    alnum = sum(ch.isalnum() for ch in t)
    if not t:
        return True
    if len(t.replace(" ", "")) <= 2 and not t.isdigit() and t.lower() not in ONE_LETTER:
        return True
    if alnum / len(t.replace(" ", "")) < 0.4:
        return True
    # Купка коротких «слів» з низькою впевненістю — шум фону
    if L.confs and max(L.confs) < 50 and alnum <= 6:
        return True
    return False


ONE_LETTER = {"і", "й", "я", "а", "в", "у", "з", "о"}   # українські слова з однієї літери
EDGE_JUNK = set("|\\/_~^`\"'¦°•·;,.:")


def trim_edges(L: Line):
    """Прибирає сміття з країв рядка: «" борошна», «5 доки» (5 — шум тла)."""
    def bad(w, c):
        core = re.sub(r"\W", "", w)
        if core.lower() in ONE_LETTER:
            return False
        return all(ch in EDGE_JUNK for ch in w) or (len(core) <= 2 and c < 50)
    # «| наостанок» — це «І наостанок», а не сміття
    if len(L.words) > 1 and L.words[0] in ("|", "]") and re.match(r"[а-яґєії]", L.words[1]):
        L.words[0] = "І"
    while len(L.words) > 1 and bad(L.words[0], L.confs[0]):
        L.words.pop(0)
        L.confs.pop(0)
    while len(L.words) > 1 and bad(L.words[-1], L.confs[-1]):
        L.words.pop()
        L.confs.pop()
    return L


def build_text(lines, s: Settings):
    """Збирає рядки в текст з абзацами."""
    if not lines:
        return ""
    if s.junk:
        lines = [trim_edges(L) for L in lines if not is_junk(L)]
    if not lines:
        return ""
    hs = sorted(L.box[3] - L.box[1] for L in lines)
    mh = hs[len(hs) // 2] or 1
    gaps = sorted(b.box[1] - a.box[3] for a, b in zip(lines, lines[1:]))
    mgap = gaps[len(gaps) // 4] if gaps else 0   # звичайний проміжок між рядками
    lefts = sorted(L.box[0] for L in lines)
    paras, cur = [], [lines[0]]
    for prev, L in zip(lines, lines[1:]):
        gap = L.box[1] - prev.box[3]
        new_par = False
        if s.paragraphs:
            # Абзац: проміжок помітно більший за звичайний між рядками
            if gap > mgap + 0.5 * mh or gap > 1.0 * mh:
                new_par = True
            # абзацний відступ: рядок зсунутий вправо відносно попереднього
            if L.box[0] - prev.box[0] > 1.2 * mh and abs(gap) < 2 * mh \
                    and L.box[0] - lefts[0] < 6 * mh:
                new_par = True
            if gap < -0.5 * mh:   # рядок вище попереднього — інша колонка/блок
                new_par = True
        if new_par:
            paras.append(cur)
            cur = [L]
        else:
            cur.append(L)
    paras.append(cur)
    out = []
    for p in paras:
        txt = [L.text for L in p]
        if s.hyphen:
            txt = join_hyphens(txt)
        para = "\n".join(txt)
        if s.punct and len(p) > 1 and para.endswith(","):
            para = para[:-1] + "."   # абзац не закінчується комою — це крапка
        out.append(para)
    return ("\n\n" if s.paragraphs else "\n").join(out)


def join_hyphens(lines):
    """«сло-» + «во далі» → «слово» + «далі»."""
    res = list(lines)
    for i in range(len(res) - 1):
        m = re.search(r"(\w+)[-‐¬]$", res[i])
        nxt = res[i + 1].lstrip()
        if m and nxt and nxt[0].islower():
            first, _, rest = nxt.partition(" ")
            res[i] = res[i][:-1] + first
            res[i + 1] = rest
    return [r for r in res if r.strip()]


def postprocess(text, s: Settings):
    text = APOS.sub("’", text)
    if s.punct:
        text = re.sub(r"(?<= )--?(?= )", "—", text)          # « -- » → « — »
        text = re.sub(r"(?m)^[|\]] (?=[а-яґєії])", "І ", text)  # «| наостанок» → «І наостанок»
        # знак номера; «її/їй» — лише перед довгим числом (бо «їй 5 років» — звичайні слова)
        text = re.sub(r"(?<!\w)(?:Ne|N°|Nº|N2|Мо) ?(?=\d)|(?<!\w)(?:її|їй) ?(?=\d{3})", "№ ", text)
    if s.homoglyph:
        text = fix_homoglyphs(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip() + "\n"


# ---------------- Головна функція ----------------

def ocr_image(path, s: Settings = DEFAULT):
    """Розпізнає одну картинку. Повертає (текст, журнал кроків)."""
    log = []
    bgr, from_camera = load_image(path, s.exif)
    if s.kind == "auto":
        shot = s.auto_type and is_screenshot(bgr, from_camera)
    else:
        shot = s.kind == "screenshot"
    log.append("скріншот" if shot else "фото")
    g = preprocess(bgr, shot, s, log)
    if s.debug_dir:
        os.makedirs(s.debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(s.debug_dir, os.path.splitext(os.path.basename(path))[0] + ".png"), g)
    lines = recognize(g, s, log)
    return postprocess(build_text(lines, s), s), log


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "*"))):
                if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                    files.append(f)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="OCR фото й скріншотів (укр + англ)")
    ap.add_argument("paths", nargs="+", help="картинка або папка")
    ap.add_argument("--lang", default=DEFAULT.lang)
    ap.add_argument("--engine", default=DEFAULT.engine, choices=["tesseract", "easyocr"])
    ap.add_argument("--tessdata", default="", help="тека з ukr/eng.traineddata (best)")
    ap.add_argument("--psm", default=",".join(map(str, DEFAULT.psms)),
                    help="режими сторінки через кому, напр. 3,4,6")
    ap.add_argument("--type", choices=["auto", "photo", "screenshot"], default="auto")
    ap.add_argument("--no-save", action="store_true", help="не писати .txt")
    ap.add_argument("--debug", default="", help="тека для підготовлених картинок")
    ap.add_argument("-v", "--verbose", action="store_true", help="показати кроки")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    exe = find_tesseract()
    if a.engine == "tesseract":
        if not exe:
            sys.exit("Tesseract не знайдено. Див. README.md → «Встановлення».")
        pytesseract.pytesseract.tesseract_cmd = exe
    s = replace(DEFAULT, lang=a.lang, engine=a.engine, tessdata=a.tessdata, kind=a.type,
                psms=tuple(int(x) for x in a.psm.split(",")), debug_dir=a.debug)
    if a.engine == "tesseract" and not find_tessdata(s):
        print("УВАГА: не знайдено best-моделі ukr/eng, якість буде гіршою. "
              "Див. README.md або вкажіть --tessdata.", file=sys.stderr)

    files = collect(a.paths)
    if not files:
        sys.exit("Картинок не знайдено.")
    for f in files:
        try:
            text, log = ocr_image(f, s)
        except Exception as e:
            print(f"== {f}: помилка: {e}", file=sys.stderr)
            continue
        if len(files) > 1 or a.verbose:
            print(f"== {f}" + (f"  [{'; '.join(log)}]" if a.verbose else ""))
        print(text)
        if not a.no_save:
            out = os.path.splitext(f)[0] + ".txt"
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(text)


if __name__ == "__main__":
    main()
