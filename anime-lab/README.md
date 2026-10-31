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

## Next ideas (attempts 04-05)
- A real 3/4 view instead of front-on.
- Hands, or a full upper-body pose.
- Softer painted shading layered over the cel shading.
- A fully original outfit and accessories.
