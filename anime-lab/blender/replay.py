# Replays recorded drawing actions as Blender Grease Pencil strokes (Blender 4.2, GP v2 API).
# Fills become closed fill strokes, brush strokes become pressure-varying line strokes,
# airbrush dots become stacked soft rings, and clipping becomes GP layer masks.
# Usage: blender -b --factory-startup -P replay.py -- actions.json out.png
import json
import math
import sys

import bpy

actions_path, out_path = sys.argv[sys.argv.index('--') + 1:][:2]
actions = json.load(open(actions_path))
W, H, PPU = 800, 1000, 100  # canvas pixels, pixels per Blender unit

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.view_settings.view_transform = 'Standard'
scene.world = bpy.data.worlds.new('World')
scene.world.color = (0, 0, 0)
cam_data = bpy.data.cameras.new('Camera')
cam_data.type, cam_data.ortho_scale = 'ORTHO', H / PPU
cam = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location, cam.rotation_euler = (0, -10, 0), (math.pi / 2, 0, 0)

gpd = bpy.data.grease_pencils.new('Drawing')
gpo = bpy.data.objects.new('Drawing', gpd)
scene.collection.objects.link(gpo)


def co(x, y):
    return (x / PPU - W / PPU / 2, 0, H / PPU / 2 - y / PPU)


def linear(hex_color):
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]


materials = {}


def material(kind, color, alpha=1.0):
    key = (kind, color, round(alpha, 3))
    if key not in materials:
        mat = bpy.data.materials.new(f'{kind}{len(materials)}')
        bpy.data.materials.create_gpencil_data(mat)
        gp = mat.grease_pencil
        rgb = linear(color)
        gp.show_stroke, gp.show_fill = kind == 'line', kind == 'fill'
        gp.color = (*rgb, 1)
        gp.fill_color = (*rgb, alpha)
        gpd.materials.append(mat)
        materials[key] = len(gpd.materials) - 1
    return materials[key]


layers, frames, masks = {}, {}, {}
BLEND = {'source-over': 'REGULAR', 'multiply': 'MULTIPLY', 'screen': 'ADD'}


def new_stroke(layer, pts, mat_index, cyclic=False):
    s = frames[layer].strokes.new()
    s.display_mode, s.material_index, s.use_cyclic = '3DSPACE', mat_index, cyclic
    s.points.add(len(pts))
    for p, (x, y) in zip(s.points, pts):
        p.co = co(x, y)
    return s


for a in actions:
    op = a['op']
    if op == 'layer':
        lay = gpd.layers.new(a['name'], set_active=True)
        lay.blend_mode = BLEND.get(a['mode'], 'REGULAR')
        # Helper layers that only serve as masks: a hidden layer stops masking, 0% opacity doesn't.
        lay.opacity = 0 if a['hidden'] else a['opacity']
        lay.use_lights = False
        layers[a['name']], frames[a['name']] = lay, lay.frames.new(1)
    elif op == 'fill':
        s = new_stroke(a['layer'], a['pts'], material('fill', a['color'], a['opacity']), cyclic=True)
        s.line_width = 1
    elif op == 'stroke':
        wmax = max(a['w']) or 1
        s = new_stroke(a['layer'], a['pts'], material('line', a['color']))
        s.line_width = max(1, round(wmax * 1000 / PPU))
        s.hardness = a['hardness']
        for p, w in zip(s.points, a['w']):
            p.pressure, p.strength = w / wmax, a['opacity']
    elif op == 'dot':
        # A soft airbrush dab as stacked rings: each ring adds a little, densest at the centre.
        # (Single-point GP dots don't grow large enough for airbrush work.)
        n = max(8, min(40, round(a['r'] / 8)))
        q = 1 - (1 - a['alpha']) ** (1 / n)
        for i in range(n):
            r = a['r'] * (1 - i / n)
            ring = [(a['x'] + r * math.cos(t / 48 * 2 * math.pi), a['y'] + r * a['sy'] * math.sin(t / 48 * 2 * math.pi)) for t in range(48)]
            new_stroke(a['layer'], ring, material('fill', a['color'], q), cyclic=True).line_width = 1
    elif op == 'mask':
        lay = layers[a['layer']]
        lay.use_mask_layer = True
        masks[a['layer']] = masks.get(a['layer'], []) + [a]

# GP combines a layer's masks as a union, not an intersection. A layer that is clipped and also has
# a shape subtracted (the cel-shadow crescents) already contains exactly that shape, so it keeps only
# the subtraction.
for name, ms in masks.items():
    lay = layers[name]
    minus = [m for a in ms for m in a.get('minus', [])]
    for a in ([] if minus else ms):
        for name2 in a.get('clip', []):
            lay.mask_layers.add(layers[name2])
    for name2 in minus:
        lay.mask_layers.add(layers[name2])
        lay.mask_layers[-1].invert = True

print(f'{len(actions)} actions -> {sum(len(f.strokes) for f in frames.values())} strokes, {len(materials)} materials')
if not globals().get('ANIMATE'):  # animate.py reuses this script and renders frames itself
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
