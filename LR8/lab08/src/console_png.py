# Снимок вывода консоли: текстовый лог → PNG (тёмный фон, моноширинный шрифт). python console_png.py console.txt console.png
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
lines = []
for line in open(src, encoding="utf-8").read().replace("‖", "||").splitlines():
    lines += textwrap.wrap(line, 150, subsequent_indent="    ") or [""]
w, h = font.getbbox("M" * 150)[2] + 40, 19 * len(lines) + 40
img = Image.new("RGB", (w, h), (30, 30, 30))
draw = ImageDraw.Draw(img)
for i, line in enumerate(lines):
    draw.text((20, 20 + 19 * i), line, font=font, fill=(220, 220, 220))
img.save(dst)
print(dst, img.size)
