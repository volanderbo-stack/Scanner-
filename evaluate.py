"""Оцінка якості OCR: CER і WER на картинках з tests/.

Для кожної картинки tests/<назва>.<jpg|png|heic> потрібен tests/<назва>.gt.txt
з правильним текстом. Назви, що починаються зі «screen», — скріншоти, решта — фото.

    python evaluate.py              # лише фінальні налаштування, таблиця по картинках
    python evaluate.py --ablation   # порівняння «до → після» і внесок кожного кроку
    python evaluate.py --engine easyocr
    python evaluate.py --tests tests tests_seed2   # кілька наборів разом
Результат також записується в results.md.
"""
import argparse
import glob
import hashlib
import os
import pickle
import re
import sys
import unicodedata
from dataclasses import replace
from multiprocessing import Pool

import ocr

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "tests")
CACHE = os.path.join(HERE, ".eval_cache")


# ---------------- Метрики ----------------

def lev(a, b):
    """Відстань Левенштейна для рядків або списків слів."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def norm(t):
    """Порівнюємо текст, а не пробіли: усі пробіли/переноси → один пробіл."""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"['`´ʼ‘’]", "’", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(pred, gt):
    p, g = norm(pred), norm(gt)
    return lev(p, g) / max(1, len(g))


def wer(pred, gt):
    p, g = norm(pred).split(), norm(gt).split()
    return lev(p, g) / max(1, len(g))


def paragraphs(t):
    return len([p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()])


# ---------------- Кеш Tesseract ----------------
# Однакова підготовлена картинка + той самий psm → той самий результат,
# тому різні варіанти обробки тексту не запускають Tesseract повторно.

_orig_lines = ocr.tesseract_lines


def cached_lines(g, psm, s, tessdata):
    h = hashlib.sha1(g.tobytes() + str(g.shape).encode()).hexdigest()
    key = f"{h}_{psm}_{s.lang}_{os.path.basename(tessdata or 'default')}"
    path = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = _orig_lines(g, psm, s, tessdata)
    os.makedirs(CACHE, exist_ok=True)
    tmp = path + f".{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    os.replace(tmp, path)
    return res


ocr.tesseract_lines = cached_lines


# ---------------- Набори налаштувань ----------------

F = ocr.DEFAULT
BASE = replace(F, model="fast", psms=(3,), exif=False, auto_type=False, invert=False,
               doc=False, illum=False, stretch=False, clahe=False, scale=False, denoise=False,
               deskew=False, binarize=False, hyphen=False, junk=False,
               homoglyph=False, punct=False, paragraphs=True)

# «До → після»: кроки додаються по одному
CUMULATIVE = [
    ("0. Tesseract як є (fast-моделі, psm 3)", BASE),
]
_steps = [
    ("+ моделі tessdata_best", dict(model="best")),
    ("+ поворот за EXIF", dict(exif=True)),
    ("+ темний фон → інверсія", dict(invert=True)),
    ("+ скріншот/фото окремо", dict(auto_type=True)),
    ("+ пошук аркуша і перспектива", dict(doc=True)),
    ("+ вирівнювання освітлення", dict(illum=True)),
    ("+ масштаб літер до ~32 px", dict(scale=True)),
    ("+ вирівнювання нахилу", dict(deskew=True)),
    ("+ psm 3/4/6 за впевненістю", dict(psms=(3, 4, 6))),
    ("+ обробка тексту", dict(hyphen=True, junk=True, homoglyph=True, punct=True)),
]
_cur = BASE
for i, (name, kw) in enumerate(_steps, 1):
    _cur = replace(_cur, **kw)
    CUMULATIVE.append((f"{i}. {name}", _cur))

# Від фінальних налаштувань: прибрати / додати по одному кроку
ABLATION = [
    ("Фінальні налаштування", F),
    ("− tessdata_best (fast)", replace(F, model="fast")),
    ("− EXIF", replace(F, exif=False)),
    ("− інверсія", replace(F, invert=False)),
    ("− автотип (усе як фото)", replace(F, auto_type=False)),
    ("− освітлення", replace(F, illum=False)),
    ("− масштаб", replace(F, scale=False)),
    ("− нахил", replace(F, deskew=False)),
    ("− склеювання переносів", replace(F, hyphen=False)),
    ("− сміттєві рядки", replace(F, junk=False)),
    ("− латиниця/кирилиця", replace(F, homoglyph=False)),
    ("− виправлення розділових знаків", replace(F, punct=False)),
    # Сміттєві рядки з'являються, коли аркуш не знайдено і в кадр потрапляє тло
    ("− пошук аркуша", replace(F, doc=False)),
    ("− пошук аркуша і − фільтр сміття", replace(F, doc=False, junk=False)),
    ("+ CLAHE", replace(F, clahe=True)),
    ("+ розтягування контрасту", replace(F, stretch=True)),
    ("+ прибирання шуму", replace(F, denoise=True)),
    ("+ адаптивна бінаризація", replace(F, binarize=True)),
    ("+ Otsu-бінаризація", replace(F, binarize="otsu")),
    ("масштаб до 24 px", replace(F, target_h=24)),
    ("масштаб до 28 px", replace(F, target_h=28)),
    ("масштаб до 36 px", replace(F, target_h=36)),
    ("масштаб до 40 px", replace(F, target_h=40)),
    ("лише psm 3", replace(F, psms=(3,))),
    ("лише psm 4", replace(F, psms=(4,))),
    ("лише psm 6", replace(F, psms=(6,))),
    ("лише psm 11", replace(F, psms=(11,))),
    ("psm 3/4 за впевненістю", replace(F, psms=(3, 4))),
    ("psm 3/4/6/11 за впевненістю", replace(F, psms=(3, 4, 6, 11))),
]


# ---------------- Запуск ----------------

def test_images(dirs=None):
    out = []
    gts = []
    for d in dirs or [TESTS]:
        gts += sorted(glob.glob(os.path.join(d, "*.gt.txt")))
    for gt in gts:
        stem = gt[:-7]
        imgs = [p for p in glob.glob(stem + ".*")
                if os.path.splitext(p)[1].lower() in ocr.IMAGE_EXT]
        if imgs:
            out.append((imgs[0], gt))
    return out


def kind(path):
    return "скріншот" if os.path.basename(path).startswith("screen") else "фото"


def run_one(args):
    img, gt_path, s = args
    if s.engine == "tesseract":
        exe = ocr.find_tesseract()
        if exe:
            ocr.pytesseract.pytesseract.tesseract_cmd = exe
    with open(gt_path, encoding="utf-8") as f:
        gt = f.read()
    try:
        pred, _ = ocr.ocr_image(img, s)
    except Exception as e:
        print(f"  ! {os.path.basename(img)}: {e}", file=sys.stderr)
        pred = ""
    return dict(name=os.path.basename(img), kind=kind(img), cer=cer(pred, gt),
                wer=wer(pred, gt), par_ok=paragraphs(pred) == paragraphs(gt), pred=pred)


def evaluate(s, tests, pool):
    return pool.map(run_one, [(i, g, s) for i, g in tests])


def avg(rows, key, k=None):
    xs = [r[key] for r in rows if k is None or r["kind"] == k]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x):
    return f"{100 * x:.1f}%"


def summary_table(title, configs, tests, pool, log):
    lines = [f"### {title}", "",
             "| Налаштування | CER скріншоти | CER фото | CER усі | WER усі | абзаци |",
             "|---|---:|---:|---:|---:|---:|"]
    print("\n" + title)
    for name, s in configs:
        rows = evaluate(s, tests, pool)
        par = sum(r["par_ok"] for r in rows)
        row = (f"| {name} | {pct(avg(rows, 'cer', 'скріншот'))} | {pct(avg(rows, 'cer', 'фото'))} "
               f"| {pct(avg(rows, 'cer'))} | {pct(avg(rows, 'wer'))} | {par}/{len(rows)} |")
        print(row)
        lines.append(row)
        log[name] = rows
    return lines + [""]


def per_image_table(rows):
    lines = ["| Картинка | Тип | CER | WER | Абзаци |", "|---|---|---:|---:|:---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['kind']} | {pct(r['cer'])} | {pct(r['wer'])} "
                     f"| {'✓' if r['par_ok'] else '✗'} |")
    for k in ("скріншот", "фото", None):
        lines.append(f"| **середнє {k or 'усі'}** | | **{pct(avg(rows, 'cer', k))}** "
                     f"| **{pct(avg(rows, 'wer', k))}** | |")
    return lines


def before_after(before, after):
    lines = ["| Картинка | CER до | CER після | WER до | WER після |", "|---|---:|---:|---:|---:|"]
    for b, a in zip(before, after):
        lines.append(f"| {a['name']} | {pct(b['cer'])} | **{pct(a['cer'])}** "
                     f"| {pct(b['wer'])} | **{pct(a['wer'])}** |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="порівняти всі кроки")
    ap.add_argument("--engine", default="tesseract", choices=["tesseract", "easyocr"])
    ap.add_argument("--tests", nargs="+", default=[TESTS],
                    help="теки з картинками й .gt.txt (можна кілька)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--diff", action="store_true", help="показати розпізнаний текст з помилками")
    ap.add_argument("--out", default="", help="куди записати таблиці (results*.md)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not a.out:
        suffix = "" if a.engine == "tesseract" else "_" + a.engine
        a.out = os.path.join(HERE, f"results{suffix}.md")

    tests = test_images(a.tests)
    if not tests:
        sys.exit("У tests/ немає пар картинка + .gt.txt")
    if a.engine == "tesseract" and not ocr.find_tesseract():
        sys.exit("Tesseract не знайдено. Див. README.md.")
    print(f"Картинок: {len(tests)}")

    md = ["# Результати OCR", ""]
    log = {}
    # easyocr і так використовує всі ядра — запускаємо по одній картинці
    jobs = 1 if a.engine == "easyocr" else a.jobs
    with Pool(jobs) as pool:
        final = replace(ocr.DEFAULT, engine=a.engine)
        rows = evaluate(final, tests, pool)
        md += [f"## Фінальні налаштування (рушій: {a.engine})", ""] + per_image_table(rows) + [""]
        print("\n".join(per_image_table(rows)))
        if a.diff:
            for r in rows:
                if r["cer"] > 0:
                    print(f"\n--- {r['name']} (CER {pct(r['cer'])})\n{r['pred']}")
        if a.ablation and a.engine == "tesseract":
            md += ["## Порівняння кроків", ""]
            md += summary_table("До покращень → після (кроки додаються по одному)",
                                CUMULATIVE, tests, pool, log)
            md += summary_table("Внесок кожного кроку (від фінальних налаштувань)",
                                ABLATION, tests, pool, log)
            md += ["### По картинках: до → після", ""] + before_after(
                log[CUMULATIVE[0][0]], rows) + [""]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nЗбережено: {a.out}")


if __name__ == "__main__":
    main()
