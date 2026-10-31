# Hybrid experiment: the brush-engine drawing as the starting image for an anime diffusion model.
# The drawing fixes design and composition; the model repaints the rendering. Runs on CPU via LCM-LoRA.
# Usage: python hybrid.py init.png out_dir [strength ...]   (strength 0 = text only, no init image)
import os
import sys
import time

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, LCMScheduler
from PIL import Image

MODEL, LCM = 'Lykon/AnyLoRA', 'latent-consistency/lcm-lora-sdv1-5'
PROMPT = ('masterpiece, best quality, 1girl, solo, upper body, looking at viewer, long hair, silver hair, '
          'lavender hair, hair between eyes, ahoge, golden eyes, smile, open mouth, blush, serafuku, white sailor shirt, '
          'navy sailor collar, red ribbon bow, twilight sky, stars, moon glow, cherry blossom petals')
NEGATIVE = ('lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, jpeg artifacts, '
            'signature, watermark, blurry, nsfw')

init_path, out_dir = sys.argv[1], sys.argv[2]
strengths = [float(s) for s in sys.argv[3:]] or [0.0, 0.4, 0.55, 0.7]
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(os.cpu_count())

t2i = AutoPipelineForText2Image.from_pretrained(MODEL, variant='fp16', torch_dtype=torch.float32,
                                                safety_checker=None, requires_safety_checker=False)
t2i.scheduler = LCMScheduler.from_config(t2i.scheduler.config)
t2i.load_lora_weights(LCM)
i2i = AutoPipelineForImage2Image.from_pipe(t2i)
init = Image.open(init_path).convert('RGB').resize((512, 640), Image.LANCZOS)

for s in strengths:
    t = time.time()
    g = torch.Generator().manual_seed(1)
    common = dict(prompt=PROMPT, negative_prompt=NEGATIVE, guidance_scale=1.5, generator=g)
    if s == 0:
        im = t2i(width=512, height=640, num_inference_steps=8, **common).images[0]
    else:
        im = i2i(image=init, strength=s, num_inference_steps=max(8, round(4 / s)), **common).images[0]
    name = 'D-text-only.png' if s == 0 else f'C-hybrid-{int(s * 100)}.png'
    im.save(os.path.join(out_dir, name))
    print(f'{name}: {time.time() - t:.0f}s', flush=True)
