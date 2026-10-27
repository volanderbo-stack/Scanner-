"""Завантажує моделі Tesseract tessdata_best (ukr, eng, osd) у ./tessdata_best.

    python get_models.py          # best (точні, повільніші)
    python get_models.py fast     # fast — лише для порівняння в evaluate.py
"""
import os
import sys
import urllib.request

KIND = sys.argv[1] if len(sys.argv) > 1 else "best"
URL = f"https://github.com/tesseract-ocr/tessdata_{KIND}/raw/main/{{}}.traineddata"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tessdata_{KIND}")

os.makedirs(OUT, exist_ok=True)
for lang in ("ukr", "eng", "osd"):
    dst = os.path.join(OUT, lang + ".traineddata")
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"{lang}: вже є")
        continue
    print(f"{lang}: завантажую…", end=" ", flush=True)
    urllib.request.urlretrieve(URL.format(lang), dst + ".part")
    os.replace(dst + ".part", dst)
    print(f"{os.path.getsize(dst) / 1e6:.1f} МБ")
print("Готово:", OUT)
