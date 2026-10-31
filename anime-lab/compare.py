# Usage: python3 compare.py mine.png reference.png out_prefix
# Writes <prefix>-side.png (reference | mine) and <prefix>-overlay.png (50/50 blend) for checking proportions.
import sys
from PIL import Image
mine, ref, prefix = sys.argv[1:4]
a = Image.open(ref).convert('RGB')
b = Image.open(mine).convert('RGB').resize(a.size)
side = Image.new('RGB', (a.width * 2 + 10, a.height), 'white')
side.paste(a, (0, 0)); side.paste(b, (a.width + 10, 0))
side.resize((side.width // 2, side.height // 2), Image.LANCZOS).save(prefix + '-side.png')
Image.blend(a, b, 0.5).save(prefix + '-overlay.png')
