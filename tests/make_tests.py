"""Генератор тестових картинок для evaluate.py.

Створює скріншоти й «фото» (з перспективою, тінню, шумом, муаром тощо)
разом із правильним текстом у .gt.txt. Фото тут синтетичні, тому для
чесної оцінки варто докинути в tests/ свої справжні фото з телефону.

Запуск:  python tests/make_tests.py
"""
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SHY = "\u00ad"  # м'який перенос: тут слово можна розірвати дефісом

# Пошук шрифтів з кирилицею (Linux / Windows)
FONT_DIRS = ["/usr/share/fonts/truetype", "C:/Windows/Fonts"]
FONTS = {
    "sans": ["noto/NotoSans-Regular.ttf", "arial.ttf"],
    "sans_bold": ["noto/NotoSans-Bold.ttf", "arialbd.ttf"],
    "serif": ["noto/NotoSerif-Regular.ttf", "times.ttf"],
    "serif_bold": ["noto/NotoSerif-Bold.ttf", "timesbd.ttf"],
    "lserif": ["liberation/LiberationSerif-Regular.ttf", "times.ttf"],
    "lserif_bold": ["liberation/LiberationSerif-Bold.ttf", "timesbd.ttf"],
    "dejavu": ["dejavu/DejaVuSans.ttf", "verdana.ttf"],
    "dejavu_bold": ["dejavu/DejaVuSans-Bold.ttf", "verdanab.ttf"],
    "mono": ["dejavu/DejaVuSansMono.ttf", "consola.ttf"],
    "mono_bold": ["dejavu/DejaVuSansMono-Bold.ttf", "consolab.ttf"],
    "lsans": ["liberation/LiberationSans-Regular.ttf", "arial.ttf"],
    "lsans_bold": ["liberation/LiberationSans-Bold.ttf", "arialbd.ttf"],
}


def font(name, size):
    for d in FONT_DIRS:
        for f in FONTS[name]:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    raise SystemExit(f"Не знайдено шрифт {name}")


# ---------- Тексти (перший рядок кожного — заголовок) ----------

T_BOOK = [
    "Розділ третій",
    "Ранок у Харкові видався холодним і ясним. Над Держпромом ще висів "
    "туман, а трамваї вже дзвеніли на поворотах, розга" + SHY + "няючи сонних голубів. "
    "Марко поспішав на першу пару й подумки повторював фор" + SHY + "мули, які вчора "
    "так і не встиг вивчити.",
    "Біля входу до головного корпусу його чекала Ірина з двома горнят" + SHY + "ками "
    "кави. Вона стояла на ґанку, усміхнулася й сказала, що викладач захворів, "
    "тож лекцію перенесли на четвер. Марко полег" + SHY + "шено зітхнув: нарешті "
    "з’явився час підго" + SHY + "туватися як слід.",
    "Вони сіли на лавку в сквері, розгорнули конспекти і почали розбира" + SHY + "тися "
    "із задачами. Жовте листя повільно падало на сторінки, наче хтось "
    "нагадував: осінь — найкраща пора для навчання, і її не варто гаяти.",
]

T_ARTICLE = [
    "Як навчитися вчитися",
    "Найкращий спосіб запам’ятати нову тему — пояснити її комусь іншому. "
    "Коли ти розповідаєш другові про інтеграли чи закон Ома, мозок змушений "
    "упорядкувати знання й знайти прогалини.",
    "Друга порада: повторюй матеріал через проміжки часу. Перше повторення — "
    "через день, друге — через тиждень, третє — через місяць. Так інформація "
    "переходить у довготривалу пам’ять.",
    "І наостанок: висипайся. Сон перед іспитом корисніший за нічне читання "
    "конспекту.",
]

T_MIXED = [
    "Інструкція з налаштування середовища",
    "1. Завантаж Python 3.12 з офіційного сайту python.org і під час "
    "встановлення познач пункт «Add Python to PATH».",
    "2. Відкрий термінал (PowerShell або Windows Terminal) і виконай команду "
    "pip install opencv-python pytesseract.",
    "3. Перевір, що все працює: команда python --version має показати номер версії.",
    "Порада: якщо бачиш помилку ModuleNotFoundError, найчастіше пакет "
    "встановлено в інше середовище (virtual environment).",
]

T_SMALL = [
    "Правила користування бібліотекою",
    "Читацький квиток видається студентам першого курсу після пред’явлення "
    "студентського посвід" + SHY + "чення. Книжки з абонемента можна брати на строк до "
    "тридцяти днів, підручники — на весь семестр.",
    "У читальній залі заборонено голосно розмовляти, їсти та користуватися "
    "телефоном без беззвучного режиму. Електронні ресурси доступні з будь-якого "
    "комп’ютера універси" + SHY + "тетської мережі.",
    "Якщо книжку загублено, читач повинен замінити її таким самим виданням "
    "або рівноцінним, погодженим із бібліо" + SHY + "текарем.",
]

T_NOTES = [
    "Нотатки",
    "Список покупок на тиждень: хліб, молоко, яйця, гречка, яблука, сир.",
    "Не забути: оплатити гуртожиток до кінця місяця і записатися до лікаря.",
    "Ідея для проєкту: застосунок, який розпізнає текст із фото конспектів.",
]

T_SCHEDULE = [
    "Як скласти розклад на тиждень",
    "Спочатку випиши всі обов’язкові справи: пари, роботу, тренування. Потім "
    "познач час, коли ти найпродуктивніший, і заплануй на нього найскладніші "
    "завдання.",
    "Залишай вільні проміжки між справами. Без них будь-яка затримка руйнує "
    "весь план, і з’являється відчуття, що ти нічого не встигаєш.",
    "У неділю ввечері переглянь, що вдалося, а що ні, і скоригуй розклад на "
    "наступний тиждень.",
]

T_NOTICE = [
    "Оголошення",
    "У суботу, 2 листопада, у спортивному комплексі відбудеться турнір з "
    "настільного тенісу серед першокурсників. Початок о 10:00.",
    "Реєстрація учасників — у профкомі до четверга. Візьміть із собою "
    "студентський квиток і змінне взуття.",
    "Переможці отримають грамоти та призи від спонсорів.",
]

T_LETTER = [
    "Шановні студенти!",
    "Нагадуємо, що реєстрація на вибіркові дисципліни триває до 15 листопада. "
    "Обрати курси можна в особистому кабінеті на сайті університету.",
    "Кожен студент має обрати щонайменше дві дисципліни з переліку. Після "
    "завершення реєстрації змінити вибір буде неможливо.",
    "З питань звертайтеся до деканату, кімната 214.",
]

T_RECEIPT = [
    "Кав’ярня «Зерно»",
    "вул. Студентська, 5",
    "Чек № 004517",
    "Капучино 0,3 л 65,00",
    "Круасан з маслом 48,00",
    "Вода без газу 25,00",
    "Разом: 138,00",
    "Оплата карткою",
    "Дякуємо за покупку!",
]

T_ENG_UKR = [
    "Лекція 5. Machine Learning: основні поняття",
    "Supervised learning — навчання з учителем: модель отримує приклади з "
    "правильними відповідями (labels).",
    "Unsupervised learning — навчання без учителя, наприклад кластеризація "
    "методом k-means.",
    "Overfitting (перенавчання) виникає, коли модель запам’ятовує тренувальні "
    "дані замість того, щоб знаходити загальні закономірності.",
]

T_RECIPE = [
    "Сирники за п’ять хвилин",
    "Змішай пачку кисломолочного сиру, одне яйце, дві ложки борошна й трохи "
    "цукру. Сформуй невеликі кружечки та обваляй їх у борошні.",
    "Смаж на середньому вогні по дві-три хвилини з кожного боку, доки не "
    "з’явиться золотиста скоринка. Подавай зі сметаною або варенням.",
]

T_DORM = [
    "Правила гуртожитку",
    "Тиша в гуртожитку діє з 23:00 до 7:00. Гостей можна запрошувати лише до "
    "22:00 і тільки після запису на вахті.",
    "Кухнею користуються всі мешканці поверху, тому посуд треба мити одразу "
    "після приготування їжі.",
    "За порушення правил студента можуть позбавити місця в гуртожитку.",
]

T_DARK = [
    "README: як запустити бота",
    "Створи файл .env і запиши туди токен: BOT_TOKEN=твій_токен. Нікому "
    "його не показуй і не додавай у git.",
    "Запусти команду python bot.py, а потім напиши боту /start у Telegram.",
    "Якщо бот мовчить, перевір інтернет і чи правильно скопійовано токен.",
]

T_CHAT = [
    ("Оля", "Привіт! Ти вже здав лабораторну з фізики?"),
    ("Я", "Ще ні, сьогодні ввечері доробляю звіт."),
    ("Оля", "Нагадую, що дедлайн у п’ятницю о 18:00."),
    ("Я", "Дякую! Скинеш шаблон титульної сторінки?"),
    ("Оля", "Так, зараз надішлю файл у Telegram."),
]


# ---------- Верстка тексту ----------

def wrap(text, fnt, width, indent=0):
    """Розбиває абзац на рядки. Повертає (рядки для малювання, рядки для GT).

    Слово з м'яким переносом може розірватися дефісом; у GT воно
    записується цілим на першому рядку (так робить і ocr.py).
    """
    words = [(w, True) for w in text.split(" ")]  # (слово, чи йде в GT)
    draw, gt = [], []
    cur, cur_gt = [], []
    avail = width - indent
    i = 0
    while i < len(words):
        w, in_gt = words[i]
        clean = w.replace(SHY, "")
        if fnt.getlength(" ".join(cur + [clean])) <= avail:
            cur.append(clean)
            if in_gt:
                cur_gt.append(clean)
            i += 1
            continue
        if SHY in w:
            head, tail = w.split(SHY, 1)
            if fnt.getlength(" ".join(cur + [head + "-"])) <= avail:
                draw.append(" ".join(cur + [head + "-"]))
                gt.append(" ".join(cur_gt + [clean]))
                words[i] = (tail, False)
                cur, cur_gt, avail = [], [], width
                continue
        draw.append(" ".join(cur))
        gt.append(" ".join(cur_gt))
        cur, cur_gt, avail = [], [], width
    if cur:
        draw.append(" ".join(cur))
        gt.append(" ".join(cur_gt))
    return draw, gt


def render_page(paras, width, fsize, face="serif", bg=(250, 248, 242),
                fg=(25, 25, 25), margin=None, indent=False, gap=0.9,
                title_scale=1.35, line_h=1.45, title_color=None):
    """Малює сторінку. Повертає (PIL.Image, gt_text)."""
    margin = margin or int(fsize * 2.2)
    f = font(face, fsize)
    ft = font(face + "_bold", int(fsize * title_scale))
    tw = width - 2 * margin
    blocks = []  # (font, draw_lines, indent, color)
    gt_paras = []
    for k, p in enumerate(paras):
        ff = ft if k == 0 else f
        ind = int(fsize * 2) if indent and k > 0 else 0
        dl, gl = wrap(p, ff, tw, ind)
        blocks.append((ff, dl, ind, title_color if k == 0 and title_color else fg))
        gt_paras.append("\n".join(gl))
    height = margin * 2
    for ff, dl, _, _ in blocks:
        height += len(dl) * int(ff.size * line_h) + int(fsize * gap)
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    y = margin
    for ff, dl, ind, col in blocks:
        h = int(ff.size * line_h)
        for j, line in enumerate(dl):
            d.text((margin + (ind if j == 0 else 0), y), line, font=ff, fill=col)
            y += h
        y += int(fsize * gap)
    return img, "\n\n".join(gt_paras) + "\n"


# ---------- Ефекти «камери» ----------

def to_np(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def background(h, w, kind, rng):
    if kind == "wood":
        # Стіл: волокна дерева (витягнутий шум + темні прожилки)
        base = np.zeros((h, w, 3), np.float32)
        base[:] = (60, 95, 140)  # BGR, коричневий
        grain = cv2.resize(rng.normal(0, 1, (h // 2, max(2, w // 60))).astype(np.float32),
                           (w, h), interpolation=cv2.INTER_CUBIC)
        base += (grain * 22)[..., None]
        veins = (np.abs(grain) < 0.08).astype(np.float32) * -45
        base += cv2.GaussianBlur(veins, (0, 0), 0.8)[..., None]
        base += rng.normal(0, 4, (h, w, 1)).astype(np.float32)
        return np.clip(cv2.GaussianBlur(base, (0, 0), 0.8), 0, 255)
    if kind == "dark":
        base = np.full((h, w, 3), 45, np.float32)
    else:
        base = np.full((h, w, 3), 170, np.float32)
    base += rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(base, (0, 0), 3), 0, 255)


def place(page, canvas_wh, corners, bg_kind, rng):
    """Кладе сторінку на тло з перспективою (corners — 4 точки TL,TR,BR,BL)."""
    W, H = canvas_wh
    ph, pw = page.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    bg = background(H, W, bg_kind, rng)
    warped = cv2.warpPerspective(page.astype(np.float32), M, (W, H),
                                 flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.ones((ph, pw), np.float32), M, (W, H))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0)[..., None]
    return warped * mask + bg * (1 - mask)


def lighting(img, rng, strength=0.3, vignette=0.25):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    g = 1 - strength * grad
    r = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    g *= 1 - vignette * r ** 2 * 2
    return img * g[..., None]


def shadow(img, poly, darkness=0.5, blur=25):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [np.int32(poly)], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), blur)
    return img * (1 - darkness * m)[..., None]


def finish(img, rng, blur=1.0, noise=4.0, motion=0):
    if motion:
        k = np.zeros((motion, motion), np.float32)
        k[motion // 2, :] = 1.0 / motion
        img = cv2.filter2D(img, -1, k)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def jitter(pts, rng, amount):
    return [[x + rng.uniform(-amount, amount), y + rng.uniform(-amount, amount)]
            for x, y in pts]


def shrink(bgr, long_side):
    """Як телефон/месенджер: зменшити кадр до long_side пікселів."""
    h, w = bgr.shape[:2]
    k = long_side / max(h, w)
    if k >= 1:
        return bgr
    return cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def save_jpg(bgr, path, quality=85, exif=None, long_side=None):
    if long_side:
        bgr = shrink(bgr, long_side)
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    kw = {"quality": quality}
    if exif is not None:
        kw["exif"] = exif
    im.save(path, "JPEG", **kw)


# Частина фото без EXIF — як після пересилання в месенджері
def camera_exif(orientation=1):
    ex = Image.Exif()
    ex[0x010F] = "Synthetic"  # Make
    ex[0x0110] = "TestPhone"  # Model
    ex[0x0112] = orientation
    return ex


def write(name, gt):
    with open(os.path.join(HERE, name + ".gt.txt"), "w", encoding="utf-8") as f:
        f.write(gt)


# ---------- Скріншоти ----------

def screen_chat():
    W = 720
    f = font("lsans", 17)
    fn = font("lsans_bold", 15)
    pad, y = 14, 20
    items = []
    for who, text in T_CHAT:
        lines, _ = wrap(text, f, 430)
        items.append((who, lines))
    H = 40 + sum(32 + len(l) * 25 + 26 for _, l in items)
    img = Image.new("RGB", (W, H), (224, 233, 240))
    d = ImageDraw.Draw(img)
    gt = []
    for who, lines in items:
        mine = who == "Я"
        bw = max(f.getlength(l) for l in lines) + 2 * pad
        bw = max(bw, fn.getlength(who) + 2 * pad)
        bh = len(lines) * 25 + 30 + pad
        x = W - bw - 20 if mine else 20
        col = (220, 248, 198) if mine else (255, 255, 255)
        d.rounded_rectangle([x, y, x + bw, y + bh], 14, fill=col)
        d.text((x + pad, y + 8), who, font=fn, fill=(40, 120, 190))
        for j, l in enumerate(lines):
            d.text((x + pad, y + 32 + j * 25), l, font=f, fill=(20, 20, 20))
        gt.append(who + "\n" + "\n".join(lines))
        y += bh + 18
    img = img.crop((0, 0, W, y + 10))
    img.save(os.path.join(HERE, "screen_chat.png"))
    write("screen_chat", "\n\n".join(gt) + "\n")


def screen_simple(name, paras, width, fsize, face, bg, fg, title_color=None):
    img, gt = render_page(paras, width, fsize, face, bg=bg, fg=fg, gap=0.8,
                          title_scale=1.25, line_h=1.4, title_color=title_color)
    img.save(os.path.join(HERE, name + ".png"))
    write(name, gt)


def screen_web():
    # Сторінка сайту: шапка-меню, заголовок, текст
    body, gt = render_page(T_SCHEDULE, 900, 18, "lsans", bg=(255, 255, 255),
                           fg=(33, 37, 41), title_scale=1.6, line_h=1.5,
                           title_color=(20, 60, 140))
    bar_h = 56
    img = Image.new("RGB", (body.width, body.height + bar_h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, body.width, bar_h], fill=(30, 41, 59))
    fb = font("lsans_bold", 18)
    menu = "Блог      Поради      Контакти"
    d.text((40, 17), menu, font=fb, fill=(240, 240, 240))
    img.paste(body, (0, bar_h))
    img.save(os.path.join(HERE, "screen_web.png"))
    write("screen_web", "Блог Поради Контакти\n\n" + gt)


# ---------- «Фото» ----------

def photo_book(rng):
    page, gt = render_page(T_BOOK, 1300, 34, "serif", bg=(236, 226, 205),
                           fg=(30, 28, 25), indent=True, gap=0.2, line_h=1.4)
    p = to_np(page).astype(np.float32)
    # Вигин сторінки: темніше біля корінця (лівий край) + легка хвиля
    h, w = p.shape[:2]
    xx = np.linspace(0, 1, w)
    shade = 1 - 0.35 * np.exp(-xx / 0.08)
    p *= shade[None, :, None]
    yy, xg = np.mgrid[0:h, 0:w].astype(np.float32)
    mapy = yy + 6 * np.sin(xg / w * np.pi)
    p = cv2.remap(p, xg, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    W, H = 1500, int(1500 * h / w) + 160
    corners = jitter([[60, 70], [W - 50, 50], [W - 70, H - 60], [40, H - 80]], rng, 25)
    img = place(p, (W, H), corners, "wood", rng)
    img = lighting(img, rng, 0.25)
    img = finish(img, rng, blur=1.1, noise=4)
    save_jpg(img, os.path.join(HERE, "photo_book.jpg"), 72, camera_exif(), 1000)
    write("photo_book", gt)


def photo_angle(rng):
    page, gt = render_page(T_ARTICLE, 1200, 34, "lserif", bg=(245, 245, 240))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    W, H = 1700, 1300
    # Сильна перспектива: сфотографовано збоку
    corners = [[260, 160], [1480, 60], [1560, 1240], [180, 1080]]
    img = place(p, (W, H), corners, "dark", rng)
    img = lighting(img, rng, 0.35)
    img = finish(img, rng, blur=1.0, noise=5)
    im = Image.fromarray(cv2.cvtColor(shrink(img, 1250), cv2.COLOR_BGR2RGB))
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        im.save(os.path.join(HERE, "photo_angle.heic"), quality=85,
                exif=camera_exif().tobytes())
    except Exception as e:  # без кодека HEIC — зберегти JPG
        print("HEIC недоступний:", e)
        im.save(os.path.join(HERE, "photo_angle.jpg"), quality=85)
    write("photo_angle", gt)


def photo_shadow(rng):
    page, gt = render_page(T_NOTICE, 1200, 36, "dejavu", bg=(250, 250, 250))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    W, H = 1400, int(1400 * h / w) + 120
    corners = jitter([[50, 50], [W - 50, 60], [W - 40, H - 50], [60, H - 60]], rng, 15)
    img = place(p, (W, H), corners, "light", rng)
    # Тінь від руки/телефона по діагоналі
    img = shadow(img, [[0, H * 0.35], [W * 0.75, 0], [W, 0], [W, H * 0.2],
                       [W * 0.2, H], [0, H]], darkness=0.55, blur=30)
    img = lighting(img, rng, 0.2)
    img = finish(img, rng, blur=1.0, noise=5)
    save_jpg(img, os.path.join(HERE, "photo_shadow.jpg"), 75, None, 900)
    write("photo_shadow", gt)


def photo_monitor(rng):
    page, gt = render_page(T_LETTER, 1100, 26, "lsans", bg=(255, 255, 255),
                           fg=(20, 20, 20))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    # Піксельна сітка екрана (RGB-субпікселі)
    scale = 3
    big = cv2.resize(p, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    grid = np.ones_like(big) * 0.55
    for c in range(3):
        grid[:, (2 - c)::3, c] = 1.0
    grid[::3, :, :] *= 0.6
    big = big * grid * 1.25
    W, H = 1600, int(1600 * h / w) + 200
    corners = jitter([[90, 90], [W - 80, 70], [W - 100, H - 90], [70, H - 70]], rng, 20)
    img = place(big, (W, H), corners, "dark", rng)
    # Муар і відблиск
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    moire = 1 - 0.14 * (np.sin((xx * 0.21 + yy * 0.05)) * np.sin(yy * 0.19))
    img = img * moire[..., None]
    glare = np.exp(-(((xx - W * 0.7) / (W * 0.25)) ** 2 + ((yy - H * 0.3) / (H * 0.3)) ** 2))
    img = img + 70 * glare[..., None]
    img = finish(img, rng, blur=1.2, noise=6)
    save_jpg(img, os.path.join(HERE, "photo_monitor.jpg"), 78, None, 1150)
    write("photo_monitor", gt)


def photo_receipt(rng):
    page, gt = render_page(T_RECEIPT, 560, 22, "mono", bg=(245, 243, 238),
                           fg=(60, 60, 70), gap=0.1, title_scale=1.2, line_h=1.35)
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    W, H = 900, 1000
    cx, cy = W / 2, H / 2
    corners = jitter([[cx - w * 0.45, cy - h * 0.45], [cx + w * 0.45, cy - h * 0.42],
                      [cx + w * 0.47, cy + h * 0.45], [cx - w * 0.43, cy + h * 0.47]], rng, 6)
    img = place(p, (W, H), corners, "wood", rng)
    img = lighting(img, rng, 0.3)
    img = finish(img, rng, blur=0.9, noise=5)
    save_jpg(img, os.path.join(HERE, "photo_receipt.jpg"), 72, camera_exif(), 800)
    write("photo_receipt", "\n".join(T_RECEIPT) + "\n")  # чек — один блок рядків


def photo_exif(rng):
    page, gt = render_page(T_ENG_UKR, 1300, 36, "sans", bg=(252, 252, 250))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    W, H = 1500, int(1500 * h / w) + 140
    corners = jitter([[60, 60], [W - 60, 70], [W - 60, H - 60], [60, H - 70]], rng, 20)
    img = place(p, (W, H), corners, "light", rng)
    img = lighting(img, rng, 0.25)
    img = finish(img, rng, blur=1.0, noise=4)
    # Зберегти «боком»: пікселі повернуті, а EXIF каже, як їх повернути назад
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    save_jpg(img, os.path.join(HERE, "photo_exif_rotated.jpg"), 75, camera_exif(6), 950)
    write("photo_exif_rotated", gt)


def photo_lowlight(rng):
    page, gt = render_page(T_RECIPE, 1200, 36, "lsans", bg=(240, 238, 230))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    W, H = 1400, int(1400 * h / w) + 140
    corners = jitter([[60, 60], [W - 60, 60], [W - 60, H - 60], [60, H - 60]], rng, 20)
    img = place(p, (W, H), corners, "wood", rng)
    img = lighting(img, rng, 0.4, vignette=0.5)
    img = img * 0.3  # темно
    img = finish(img, rng, blur=1.4, noise=9)
    save_jpg(img, os.path.join(HERE, "photo_lowlight.jpg"), 72, None, 1000)
    write("photo_lowlight", gt)


def photo_skew(rng):
    # Аркуш заповнює кадр (країв не видно) і повернутий на ~6°
    page, gt = render_page(T_DORM, 1300, 34, "serif", bg=(248, 248, 244))
    p = to_np(page)
    h, w = p.shape[:2]
    pad = 200
    p = cv2.copyMakeBorder(p, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(244, 248, 248))
    M = cv2.getRotationMatrix2D((p.shape[1] / 2, p.shape[0] / 2), 6.0, 1.0)
    p = cv2.warpAffine(p, M, (p.shape[1], p.shape[0]), borderValue=(244, 248, 248))
    p = p[pad // 2:-pad // 2, pad // 2:-pad // 2].astype(np.float32)
    p = cv2.resize(p, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
    img = lighting(p, rng, 0.2)
    img = finish(img, rng, blur=1.0, noise=4)
    save_jpg(img, os.path.join(HERE, "photo_skew.jpg"), 75, camera_exif(), 950)
    write("photo_skew", gt)


def photo_blur_small(rng):
    # Дрібний текст здалеку + змаз від руки
    page, gt = render_page(T_SMALL, 1300, 30, "lserif", bg=(250, 250, 248))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    W = 1150
    s = 0.62
    H = int(h * s) + 260
    x0, y0 = 120, 110
    corners = jitter([[x0, y0], [x0 + w * s, y0], [x0 + w * s, y0 + h * s], [x0, y0 + h * s]], rng, 12)
    img = place(p, (W, H), corners, "light", rng)
    img = lighting(img, rng, 0.25)
    img = finish(img, rng, blur=1.0, noise=5, motion=4)
    save_jpg(img, os.path.join(HERE, "photo_small_blur.jpg"), 75, None)
    write("photo_small_blur", gt)


def photo_dark_screen(rng):
    # Фото ноутбука з темною темою: світлий текст на темному, сітка пікселів
    page, gt = render_page(T_DARK, 1100, 26, "lsans", bg=(32, 34, 40),
                           fg=(215, 218, 222), title_color=(120, 190, 255))
    p = to_np(page).astype(np.float32)
    h, w = p.shape[:2]
    scale = 3
    big = cv2.resize(p, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    grid = np.ones_like(big) * 0.6
    for c in range(3):
        grid[:, (2 - c)::3, c] = 1.0
    big = big * grid * 1.2
    W, H = 1600, int(1600 * h / w) + 220
    corners = jitter([[110, 120], [W - 70, 80], [W - 120, H - 70], [80, H - 110]], rng, 20)
    img = place(big, (W, H), corners, "light", rng)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glare = np.exp(-(((xx - W * 0.3) / (W * 0.3)) ** 2 + ((yy - H * 0.6) / (H * 0.35)) ** 2))
    img = img + 45 * glare[..., None]
    img = finish(img, rng, blur=1.2, noise=6)
    save_jpg(img, os.path.join(HERE, "photo_dark_screen.jpg"), 78, None, 1150)
    write("photo_dark_screen", gt)


def main():
    global HERE
    # python tests/make_tests.py [seed] [тека] — інший seed дає інший набір для перевірки
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    if len(sys.argv) > 2:
        HERE = sys.argv[2]
        os.makedirs(HERE, exist_ok=True)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    screen_chat()
    screen_simple("screen_small_font", T_SMALL, 640, 12, "dejavu",
                  (255, 255, 255), (30, 30, 30))
    screen_simple("screen_mixed", T_MIXED, 820, 16, "lsans",
                  (250, 250, 250), (25, 25, 25))
    screen_simple("screen_dark", T_NOTES, 700, 17, "sans",
                  (30, 31, 36), (225, 225, 228), title_color=(250, 200, 80))
    screen_web()
    photo_book(rng)
    photo_angle(rng)
    photo_shadow(rng)
    photo_monitor(rng)
    photo_receipt(rng)
    photo_exif(rng)
    photo_lowlight(rng)
    photo_skew(rng)
    photo_blur_small(rng)
    photo_dark_screen(rng)
    print("Готово:", len([f for f in os.listdir(HERE) if f.endswith(".gt.txt")]), "картинок")


if __name__ == "__main__":
    main()
