# Lay out experiment results side by side with captions.
# Usage: python3 -I sheet.py out.png height "caption|image.png" ...
import sys

from PIL import Image, ImageDraw, ImageFont

out, height, items = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
except OSError:
    font = ImageFont.load_default()
tiles = []
for item in items:
    caption, path = item.split('|')
    im = Image.open(path).convert('RGB')
    im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    tiles.append((caption, im))
pad, cap = 12, 34
sheet = Image.new('RGB', (sum(t.width for _, t in tiles) + pad * (len(tiles) + 1), height + cap + pad * 2), 'white')
d, x = ImageDraw.Draw(sheet), pad
for caption, im in tiles:
    sheet.paste(im, (x, pad))
    d.text((x + im.width // 2, pad + height + 6), caption, fill=(40, 30, 50), font=font, anchor='mt')
    x += im.width + pad
sheet.save(out)
