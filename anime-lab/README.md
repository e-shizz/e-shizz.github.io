# anime-lab 🎨

Claude learning to draw in anime style by writing SVG code, then rendering it
with headless Chromium (`node render.js in.svg out.png`) to look at and fix.

## The brush engine (`engine/brush.js`)
A small homemade toolkit, since there's no drawing app in the sandbox:
- **Tapered ink strokes:** a smooth curve through points, widened by a "pen pressure" profile (thin, thick, thin).
- **Hair locks:** a clump shape with a sharp tip, a cel-shaded band down one side, and tapered edge lines.
- **Cel shadows:** a shape minus a copy of itself nudged toward the light gives a crisp shadow on the far side.
- **Cast shadows:** the hair shapes are redrawn, offset, in the skin's shadow colour and clipped to the face.
- **Angel-ring shine:** a jagged band of highlights that follows the curve of the head.

Rebuild a drawing: `node attempts/05-original.js && node render.js attempts/05-original.svg attempts/05-original.png`

## Reference
`reference/Wikipe-tan_full_length.png`: **Wikipe-tan** by Kasuga, from Wikimedia Commons,
licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The study drawings in `attempts/` are derivatives, so they're shared under CC BY-SA 3.0 too.

## Log
| # | What changed |
|---|---|
| 01 | First bust: basic shapes. Neck too long, saw-tooth bangs, blocky ornaments. |
| 02 | Bigger, taller eyes with a heavy upper lash; curved bang clumps; tilted puzzle pieces. |
| 03 | Hid the face's top edge under the bangs, connected neck/collar/dress, slimmed side locks. |
| 04 | **New approach:** the brush engine plus an original character (silver-lavender hair, golden eyes, sailor uniform, twilight sky). |
| 05 | Shorter lower face, bigger eyes, thinner neck, 4° head tilt, hair gradient with pink tips, forked strands, back-hair locks. |

## Still far from the reference (attempts 01-03)
- **Pose:** the reference is a 3/4 head turn. Mine is flat front-on.
- **Line weight:** her lines taper like ink. Mine are a uniform 3px stroke.
- **Shading:** she has cel shadows (hair under bangs, neck, skirt folds). I have almost none.
- **Hair:** her twin tails flare out wide with many thin strands. Mine are two smooth blobs.
- **Face:** her chin is smaller and her eyes sit lower and wider apart.

## Master study: cat-eared girl (attempts 06-09)
A one-to-one study of a professional illustration the user shared. **The reference image isn't in this
repo:** its artist and licence are unknown, so it stays local (`reference/private/` is git-ignored).
The method is how artists do master studies, just with code:
1. **Grid measuring:** a labelled grid over the reference (plus 4x zooms of the eyes) to read off
   every landmark: eye corners, iris centres, chin, choker, clips, shoulders, lock tips.
2. **Lightbox overlay:** `compare.py` blends my render 50/50 with the reference to spot misplaced shapes.
3. **Eyedropper:** `pick.py` samples the reference along each hair lock I drew and stores only its colours
   (`attempts/09-colors.json`). Every shape is still drawn by the brush engine.

| # | What changed |
|---|---|
| 06 | First measured pass: face landmarks land almost exactly; hair too dark and blocky. |
| 07 | Lighter hair toward the window, wavy locks, shirt folds as tapered strokes, softer brows. |
| 08 | Eyedropped colours per lock, forehead gap re-measured (narrow V at x≈460-520), S-curve waves and curls. |
| 09 | Hair as one soft mass: fainter shade bands, finer outlines, many more dark/light strand lines. |

Biggest remaining gap: the reference hair has far more strands and subtler lighting than my locks.

Rebuild: `node attempts/09-catgirl.js && node render.js attempts/09-catgirl.svg attempts/09-catgirl.png`

## Experiment: does SVG hold me back? (`experiments/`)
Same original character (attempt 05) through four pipelines. Side by side: `experiments/comparison.png`.

| | Pipeline | Who does what | Result |
|---|---|---|---|
| A | Flat SVG (attempt 05) | all code | Clean but reads as clip art |
| B | SVG + anime compositing (`satsuei.py`) | all code | Glow, halo, rim light, bokeh: looks like an anime frame, but the drawing's flaws stay |
| B2 | B + dense hair strands (`DENSE=1`) | all code | Slightly richer hair texture; small gain |
| C | My drawing → anime diffusion model, img2img (`hybrid.py`) | I design and compose, the model renders | **Best at 40%:** keeps my design (golden eyes, navy collar, bow, tilt, sky) at real anime quality. At 55% the eyes drift orange; at 70% it adds red eyes, a hair bow and cherry trees |
| D | Diffusion model from text only | model only | Polished, but it invents its own girl (purple eyes, pink collar, hair ribbon) |

**Findings**
- SVG vs. canvas vs. raster isn't the real bottleneck; precise spatial layout is. The research says the same
  (SketchAgent, Chat2SVG, StarVector, OmniSVG): pure LLM vector drawings hit limits on layout and shape detail,
  and hybrids (LLM layout + diffusion rendering) do best.
- Compositing is a cheap, all-code upgrade, and it's what real anime production does to flat cels.
- The hybrid is the strongest pipeline: the drawing controls the design, the model supplies the rendering
  skill. The text-only model loses control of the design.

Model: [Lykon/AnyLoRA](https://huggingface.co/Lykon/AnyLoRA) (CreativeML OpenRAIL-M) with
[LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) for 8-step generation on CPU
(~40-80 s per 512×640 image on 4 cores). Weights download to the Hugging Face cache and aren't in this repo.

Reproduce:
```
LAYER=char node attempts/05-original.js && node render.js attempts/05-original-char.svg experiments/05-char.png --transparent
python3 -I experiments/satsuei.py attempts/05-original.png experiments/05-char.png experiments/B-composited.png
python3 -m venv /tmp/sdenv && /tmp/sdenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
  && /tmp/sdenv/bin/pip install diffusers transformers accelerate safetensors peft pillow
/tmp/sdenv/bin/python experiments/hybrid.py attempts/05-original.png experiments/out 0 0.4 0.55 0.7
```

## No more SVG: JavaScript canvas and Blender Grease Pencil (`canvas/`, `blender/`)
The drawing is written once as drawing actions: layers, lasso fills, pressure strokes, airbrush dots and
clipping masks. Two instruments perform the same actions:
- **Canvas** (`canvas/paint.js`): my raster painting engine in headless Chromium. Brush tips are stamped
  along smooth paths with pen pressure, and layers blend with multiply/screen. The hair is ~1,000 painted strands.
- **Blender 4.2 Grease Pencil** (`blender/replay.py`): every action becomes a real GP stroke (pressure +
  hardness per stroke), a fill stroke, or a GP layer mask; rendered headless with EEVEE on software
  OpenGL (Xvfb + Mesa), ~10 s per frame.

Findings: the canvas brushes give the most hand-drawn texture so far. Grease Pencil reproduces the
drawing faithfully, but its real strength is animation (`blender/animate.py`: frame-by-frame blink,
drifting petals, noise "line boil" on the hair). Grease Pencil gotchas found on the way:
- a layer's masks combine as a union, not an intersection
- a *hidden* mask layer stops masking; use 0% opacity instead
- single-point dots don't grow big enough for airbrush work (stacked rings stand in)

Run:
```
node canvas/run.js canvas/girl.js canvas/out/girl-canvas.png canvas/out/girl-actions.json
xvfb-run -a blender -b --factory-startup -P blender/replay.py -- canvas/out/girl-actions.json $PWD/blender/out/girl-gp.png
xvfb-run -a blender -b --factory-startup -P blender/animate.py -- canvas/out/girl-actions.json /tmp/frames/
```
(Blender 4.2 LTS from download.blender.org; on a bare Linux box also `apt-get install libegl1 xvfb`.)

## Next ideas (attempts 04-05)
- A real 3/4 view instead of front-on.
- Hands, or a full upper-body pose.
- Softer painted shading layered over the cel shading.
- A fully original outfit and accessories.
