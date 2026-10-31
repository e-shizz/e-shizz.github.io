# Eyedropper for master studies: samples a reference image along each hair lock's spine
# and writes the lock's colours (start/middle/end fill, shadow, highlight) to JSON.
# Usage: python3 -I pick.py reference.png spines.json colors.json
import json
import math
import sys

from PIL import Image

ref_path, spines_path, out_path = sys.argv[1:4]
im = Image.open(ref_path).convert('RGB')
W, H = im.size
lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
hexc = lambda c: '#%02x%02x%02x' % tuple(int(v) for v in c)


def pct(cols, p):
    cols = sorted(cols, key=lum)
    return cols[min(len(cols) - 1, int(p * len(cols)))]


out = {}
for lock_id, spec in json.load(open(spines_path)).items():
    pts, w = spec['spine'], spec['w']
    n = len(pts)
    segs = [[], [], []]
    for i in range(int(n * 0.05), int(n * 0.85)):
        (x0, y0), (x1, y1) = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        d = math.hypot(x1 - x0, y1 - y0) or 1
        nx, ny = -(y1 - y0) / d, (x1 - x0) / d
        for f in (-0.25, 0, 0.25):
            x, y = pts[i][0] + nx * f * w, pts[i][1] + ny * f * w
            if 0 <= x < W and 0 <= y < H:
                segs[min(2, 3 * i // n)].append(im.getpixel((int(x), int(y))))
    allc = [c for s in segs for c in s]
    if not allc:
        continue
    # Skin and shirt are lighter than hair; drop very light samples unless the lock is mostly light.
    dark = [c for c in allc if lum(c) <= 205]
    keep = dark if len(dark) > 0.3 * len(allc) else allc
    fill = []
    for s in segs:
        s = [c for c in s if c in keep] or keep
        fill.append(hexc(pct(s, 0.45)))
    out[lock_id] = {'fill': fill, 'shade': hexc(pct(keep, 0.15)), 'light': hexc(pct(keep, 0.85))}

json.dump(out, open(out_path, 'w'), indent=1)
print(f'{len(out)} locks coloured')
