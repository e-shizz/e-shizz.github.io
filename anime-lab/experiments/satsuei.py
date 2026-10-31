# Anime-style compositing ("satsuei") pass over a flat cel render.
# In anime production, flat cels get their polish at compositing: backlight halo, rim light,
# bloom, diffusion glow, light gradients, bokeh, vignette and grain.
# Usage: python3 -I satsuei.py full.png character_rgba.png out.png
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

full_p, char_p, out_p = sys.argv[1:4]
img = np.asarray(Image.open(full_p).convert('RGB'), dtype=np.float32) / 255
A = np.asarray(Image.open(char_p).convert('RGBA').getchannel('A'), dtype=np.float32) / 255
H, W = A.shape
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)


def blur(a, r):
    mode = 'L' if a.ndim == 2 else 'RGB'
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode).filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im, dtype=np.float32) / 255


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


rgb = lambda *c: np.array(c, dtype=np.float32)

# 1. backlight halo: the moon behind her bleeds around the silhouette
halo = blur(A, 30) * (1 - A) * np.clip(1.25 - Y / H, 0, 1)
img = screen(img, halo[..., None] * rgb(1.0, 0.9, 0.96) * 0.42)

# 2. rim light: a bright band just inside the outline, strongest near the top
rim = np.clip((A - blur(A, 4)) * 3, 0, 1) * np.clip(1.15 - Y / H, 0.15, 1)
img = screen(img, rim[..., None] * rgb(1.0, 0.86, 0.95) * 0.55)

# 3. soft airbrush shading on the character: a touch darker toward the lower right
shade = 1 - 0.12 * np.clip(X / W * 0.5 + Y / H * 0.9 - 0.55, 0, 1) * A
img = img * shade[..., None]

# 4. warm light gradient from the top left, cool falloff at the bottom
d = np.hypot(X - 0.12 * W, Y - 0.04 * H) / np.hypot(W, H)
img = screen(img, (np.exp(-(d / 0.55) ** 2) * 0.16)[..., None] * rgb(1.0, 0.85, 0.68))
k = np.clip((Y / H - 0.6) / 0.4, 0, 1)[..., None] * 0.18
img = img * (1 - k * (1 - rgb(0.82, 0.8, 1.0)))

# 5. bokeh: soft discs in the background only
rnd = np.random.default_rng(7)
layer = Image.new('RGB', (W, H), 'black')
dr = ImageDraw.Draw(layer)
for _ in range(38):
    x, y, r = rnd.uniform(0, W), rnd.uniform(0, H), rnd.uniform(5, 24)
    c = tuple(int(v) for v in rnd.choice([(255, 210, 230), (220, 210, 255), (255, 240, 220)]) * rnd.uniform(0.35, 0.8))
    dr.ellipse([x - r, y - r, x + r, y + r], fill=c)
bokeh = np.asarray(layer.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255
img = screen(img, bokeh * (1 - A)[..., None] * 0.7)

# 6. bloom on the brightest areas, then an overall diffusion glow
lum = img @ rgb(0.299, 0.587, 0.114)
bright = img * np.clip((lum - 0.82) / 0.18, 0, 1)[..., None]
img = screen(img, blur(bright, 14) * 0.5)
img = img * 0.85 + screen(img, blur(img, 7)) * 0.15

# 7. win back contrast and saturation lost to the glow
img = (img - 0.5) * 1.08 + 0.5
lum = img @ rgb(0.299, 0.587, 0.114)
img = lum[..., None] + (img - lum[..., None]) * 1.15

# 8. slight chromatic fringing, vignette and film grain
img[..., 0] = np.roll(img[..., 0], 1, axis=1)
img[..., 2] = np.roll(img[..., 2], -1, axis=1)
v = np.hypot((X - W / 2) / (W / 2), (Y - H / 2) / (H / 2))
img = img * (1 - 0.22 * np.clip(v - 0.45, 0, 1) ** 1.5)[..., None]
img = img + rnd.normal(0, 0.012, img.shape).astype(np.float32)

Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out_p)
