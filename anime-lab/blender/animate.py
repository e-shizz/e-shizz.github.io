# Animates the replayed drawing with Grease Pencil's own tools: a frame-by-frame blink,
# drifting petals, and a "line boil" noise modifier on the hair. Renders a PNG sequence.
# Usage: blender -b --factory-startup -P animate.py -- actions.json out_dir
import math
import os

import bpy

ANIMATE = True
exec(compile(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay.py')).read(), 'replay.py', 'exec'))
N = 12
scene.frame_start, scene.frame_end, scene.render.fps = 1, N, 12


def tilt(x, y):  # the same 4° head lean as the drawing
    r = math.radians(-4)
    return 400 + (x - 400) * math.cos(r) - (y - 600) * math.sin(r), 600 + (x - 400) * math.sin(r) + (y - 600) * math.cos(r)


def frame_stroke(frame, pts, widths, color):
    st = frame.strokes.new()
    st.display_mode, st.material_index, st.hardness = '3DSPACE', material('line', color), 0.9
    st.points.add(len(pts))
    wmax = max(widths)
    st.line_width = round(wmax * 1000 / PPU)
    for p, (x, y), w in zip(st.points, pts, widths):
        p.co, p.pressure, p.strength = co(x, y), w / wmax, 1


# Blink: frames 7-8 show closed eyes, frame 9 copies the open eyes back.
for name in ('eyes_white', 'eyes_iris', 'eyes_line'):
    lay = layers[name]
    lay.frames.copy(lay.frames[0]).frame_number = 9
    closed = lay.frames.new(7)
    if name == 'eyes_line':
        for cx, cy, s in ((313, 436, 1), (487, 436, -1)):
            P = lambda u, v: tilt(cx - s * u * 1.08, cy + v * 1.08)
            us = list(range(-46, 55, 4))
            frame_stroke(closed, [P(u, 6 + 10 * (1 - (u / 50) ** 2)) for u in us], [2 + 7 * math.sin(math.pi * i / (len(us) - 1)) for i in range(len(us))], '#2c1830')
            frame_stroke(closed, [P(50, 8), P(58, 4), P(66, 2)], [4, 2.5, 0.5], '#2c1830')

# Petals drift down and sway: each frame is a nudged copy of the first.
fx = layers['fx']
for f in range(2, N + 1):
    fr = fx.frames.copy(fx.frames[0])
    fr.frame_number = f
    for st in fr.strokes:
        for p in st.points:
            p.co.x += 0.04 * math.sin(f * 0.5 + p.co.z * 3)
            p.co.z -= 0.05 * (f - 1)

# Line boil: fresh noise every 2 frames on the hair strokes, like redrawn animation lines.
boil = gpo.grease_pencil_modifiers.new('Boil', 'GP_NOISE')
boil.layer, boil.factor, boil.factor_strength, boil.factor_thickness = 'hair', 0.07, 0.0, 0.1
boil.noise_scale, boil.use_random, boil.step = 0.6, True, 2

scene.render.filepath = os.path.join(out_path, 'frame_')
bpy.ops.render.render(animation=True)
