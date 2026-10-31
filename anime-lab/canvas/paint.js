// paint.js: a small raster painting engine on Canvas 2D, driven by code.
// The tools a digital painter uses: layers with blend modes, pressure brushes stamped along
// smooth paths, soft airbrush dabs, lasso fills, clipping masks and a glow pass.
let W = 800, H = 1000, scratch, sctx;
const layers = [];

function canvas() { const c = document.createElement('canvas'); c.width = W; c.height = H; return c; }
function init(w, h) { W = w; H = h; scratch = canvas(); sctx = scratch.getContext('2d'); }
function layer(name, mode = 'source-over', opacity = 1) {
  const c = canvas(), L = { name, c, ctx: c.getContext('2d'), mode, opacity };
  layers.push(L);
  return L;
}

// ---------- geometry
function spline(pts, segs = 10, closed = false) {
  const n = pts.length, out = [], get = i => closed ? pts[(i + n) % n] : pts[Math.max(0, Math.min(n - 1, i))];
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    const p0 = get(i - 1), p1 = get(i), p2 = get(i + 1), p3 = get(i + 2);
    for (let s = 0; s < segs; s++) {
      const t = s / segs, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map(k => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  if (!closed) out.push(pts[n - 1]);
  return out;
}
const quad = (a, c, b, n = 12) => Array.from({ length: n }, (_, i) => { const t = (i + 1) / n, u = 1 - t; return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]]; });
const cubic = (a, c1, c2, b, n = 16) => Array.from({ length: n }, (_, i) => { const t = (i + 1) / n, u = 1 - t; return [0, 1].map(k => u * u * u * a[k] + 3 * u * u * t * c1[k] + 3 * u * t * t * c2[k] + t * t * t * b[k]); });
function rotate(pts, deg, cx, cy) {
  const r = deg * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
  return pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]);
}
const shift = (pts, dx, dy) => pts.map(([x, y]) => [x + dx, y + dy]);
// Offset a centreline by width(t)/2 on each side: the outline of a tapered shape.
function ribbon(pts, width, segs = 16) {
  const c = spline(pts, segs), len = [0];
  for (let i = 1; i < c.length; i++) len.push(len[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));
  const total = len[len.length - 1] || 1, left = [], right = [];
  c.forEach((p, i) => {
    const a = c[Math.max(i - 1, 0)], b = c[Math.min(i + 1, c.length - 1)], dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) || 1;
    const w = width(len[i] / total) / 2;
    left.push([p[0] - dy / d * w, p[1] + dx / d * w]);
    right.push([p[0] + dy / d * w, p[1] - dx / d * w]);
  });
  return { c, left, right, outline: left.concat(right.slice().reverse()) };
}

// ---------- colour
function rgba(hex, a = 1) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
function lin(L, x0, y0, x1, y1, stops) { const g = L.ctx.createLinearGradient(x0, y0, x1, y1); stops.forEach(([o, c]) => g.addColorStop(o, c)); return g; }
function rad(L, x, y, r, stops) { const g = L.ctx.createRadialGradient(x, y, 0, x, y, r); stops.forEach(([o, c]) => g.addColorStop(o, c)); return g; }

// ---------- brushes
const tips = new Map();
function tip(color, hardness) {
  const key = color + hardness;
  if (!tips.has(key)) {
    const s = 96, c = document.createElement('canvas'); c.width = c.height = s;
    const x = c.getContext('2d'), g = x.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    g.addColorStop(0, rgba(color)); g.addColorStop(Math.min(hardness, 0.97), rgba(color)); g.addColorStop(1, rgba(color, 0));
    x.fillStyle = g; x.fillRect(0, 0, s, s);
    tips.set(key, c);
  }
  return tips.get(key);
}
// Pen pressure along a stroke: ramps up over the first a, down over the last b.
const taper = (a = 0.2, b = 0.3, min = 0.08) => t => Math.max(min, Math.min(1, t / a, (1 - t) / b) ** 0.7);

// Stamp a brush tip along a smooth path. Dabs build up on a scratch canvas first,
// so a semi-transparent stroke stays even instead of darkening where dabs overlap.
function stroke(L, pts, o) {
  const d = o.raw ? pts : spline(pts, o.segs ?? 10), len = [0];
  for (let i = 1; i < d.length; i++) len.push(len[i - 1] + Math.hypot(d[i][0] - d[i - 1][0], d[i][1] - d[i - 1][1]));
  const total = len[len.length - 1];
  if (!(total > 0.5)) return;
  const P = o.pressure || taper(o.a, o.b), tp = tip(o.color, o.hardness ?? 0.8);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of d) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const pad = o.size + 4, bx = Math.max(0, Math.floor(x0 - pad)), by = Math.max(0, Math.floor(y0 - pad));
  const bw = Math.min(W, Math.ceil(x1 + pad)) - bx, bh = Math.min(H, Math.ceil(y1 + pad)) - by;
  if (bw <= 0 || bh <= 0) return;
  sctx.clearRect(bx, by, bw, bh);
  sctx.globalAlpha = o.flow ?? 1;
  for (let s = 0, i = 1; s <= total;) {
    while (i < d.length - 1 && len[i] < s) i++;
    const f = (s - len[i - 1]) / ((len[i] - len[i - 1]) || 1);
    const x = d[i - 1][0] + (d[i][0] - d[i - 1][0]) * f, y = d[i - 1][1] + (d[i][1] - d[i - 1][1]) * f;
    const r = Math.max(0.3, o.size * P(s / total) / 2);
    sctx.drawImage(tp, x - r, y - r, 2 * r, 2 * r);
    s += Math.max(0.4, r * (o.spacing ?? 0.2));
  }
  const c = L.ctx;
  c.save(); c.globalAlpha = o.opacity ?? 1; c.globalCompositeOperation = o.mode || 'source-over';
  c.drawImage(scratch, bx, by, bw, bh, bx, by, bw, bh);
  c.restore();
}

// Lasso fill: a closed shape, smoothed unless sharp, optionally soft-edged.
function fill(L, pts, style, o = {}) {
  const d = o.sharp ? pts : spline(pts, 8, true), c = L.ctx;
  c.save();
  c.globalAlpha = o.opacity ?? 1; c.globalCompositeOperation = o.mode || 'source-over';
  if (o.blur) c.filter = `blur(${o.blur}px)`;
  c.fillStyle = style;
  c.beginPath(); d.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.closePath(); c.fill();
  c.restore();
}

// Soft round airbrush dab (sy squashes it vertically).
function air(L, x, y, r, color, alpha, o = {}) {
  const c = L.ctx;
  c.save(); c.globalCompositeOperation = o.mode || 'source-over';
  c.translate(x, y); if (o.rot) c.rotate(o.rot); c.scale(1, o.sy || 1);
  const g = c.createRadialGradient(0, 0, 0, 0, 0, r);
  g.addColorStop(0, rgba(color, alpha)); g.addColorStop(1, rgba(color, 0));
  c.fillStyle = g; c.beginPath(); c.arc(0, 0, r, 0, Math.PI * 2); c.fill();
  c.restore();
}

// Clipping mask: keep L only where the given layers (or shapes) have paint.
function clip(L, ...bases) {
  const m = canvas(), mc = m.getContext('2d');
  for (const b of bases) {
    if (b.c) mc.drawImage(b.c, 0, 0);
    else { mc.beginPath(); b.forEach(([x, y], i) => i ? mc.lineTo(x, y) : mc.moveTo(x, y)); mc.closePath(); mc.fill(); }
  }
  L.ctx.save(); L.ctx.globalCompositeOperation = 'destination-in'; L.ctx.drawImage(m, 0, 0); L.ctx.restore();
}

// Shadow on the far side of a shape: the shape minus a copy nudged toward the light.
function crescent(L, pts, dx, dy, color, o = {}) {
  const t = canvas(), c = t.getContext('2d'), path = p => { c.beginPath(); p.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.closePath(); };
  if (o.blur) c.filter = `blur(${o.blur}px)`;
  c.fillStyle = color; path(pts); c.fill();
  c.globalCompositeOperation = 'destination-out'; path(shift(pts, dx, dy)); c.fill();
  L.ctx.save(); L.ctx.globalAlpha = o.opacity ?? 1; L.ctx.globalCompositeOperation = o.mode || 'source-over'; L.ctx.drawImage(t, 0, 0); L.ctx.restore();
}

// ---------- finishing
function flatten() {
  const out = canvas(), o = out.getContext('2d');
  for (const L of layers) { o.globalCompositeOperation = L.mode; o.globalAlpha = L.opacity; o.drawImage(L.c, 0, 0); }
  o.globalCompositeOperation = 'source-over'; o.globalAlpha = 1;
  return out;
}
function glow(src, r, strength, filter = '') {
  const g = canvas(), gc = g.getContext('2d');
  gc.filter = `blur(${r}px) ${filter}`; gc.drawImage(src, 0, 0);
  const o = src.getContext('2d');
  o.save(); o.globalCompositeOperation = 'screen'; o.globalAlpha = strength; o.drawImage(g, 0, 0); o.restore();
}
function grain(src, amount, seed = 1) {
  const o = src.getContext('2d'), img = o.getImageData(0, 0, W, H), p = img.data;
  let s = seed;
  for (let i = 0; i < p.length; i += 4) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const n = ((s / 0x7fffffff) - 0.5) * amount;
    p[i] += n; p[i + 1] += n; p[i + 2] += n;
  }
  o.putImageData(img, 0, 0);
}
function rng(seed) {
  return () => { seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

// ---------- drawing actions, recorded so other instruments (Blender) can replay them
const LOG = [], named = {};
const r1 = v => Math.round(v * 10) / 10;
function L(name, mode = 'source-over', opacity = 1, o = {}) {
  const l = layer(name, mode, opacity);
  l.hidden = !!o.hidden; named[name] = l;
  LOG.push({ op: 'layer', name, mode, opacity, hidden: l.hidden });
  return l;
}
function F(l, pts, color, o = {}) {
  const d = o.sharp ? pts : spline(pts, 8, true);
  fill(l, d, color, { ...o, sharp: true });
  LOG.push({ op: 'fill', layer: l.name, pts: d.map(p => p.map(r1)), color, opacity: o.opacity ?? 1 });
}
function S(l, pts, o) {
  const d = o.raw ? pts : spline(pts, o.segs ?? 10), P = o.pressure || taper(o.a, o.b), len = [0];
  for (let i = 1; i < d.length; i++) len.push(len[i - 1] + Math.hypot(d[i][0] - d[i - 1][0], d[i][1] - d[i - 1][1]));
  const total = len[len.length - 1] || 1;
  stroke(l, d, { ...o, raw: true, pressure: P });
  LOG.push({ op: 'stroke', layer: l.name, pts: d.map(p => p.map(r1)), w: len.map(s => r1(o.size * P(s / total))), color: o.color, opacity: (o.opacity ?? 1) * (o.flow ?? 1), hardness: o.hardness ?? 0.8 });
}
function D(l, x, y, r, color, alpha, o = {}) {
  air(l, x, y, r, color, alpha, o);
  LOG.push({ op: 'dot', layer: l.name, x: r1(x), y: r1(y), r: r1(r), color, alpha, sy: o.sy || 1 });
}
function clipL(l, ...names) { clip(l, ...names.map(n => named[n])); LOG.push({ op: 'mask', layer: l.name, clip: names }); }
function minusL(l, name) {
  l.ctx.save(); l.ctx.globalCompositeOperation = 'destination-out'; l.ctx.drawImage(named[name].c, 0, 0); l.ctx.restore();
  LOG.push({ op: 'mask', layer: l.name, minus: [name] });
}
// flatten() skips helper layers that only exist to be masks
const _flatten = flatten;
flatten = function () {
  const keep = layers.splice(0, layers.length);
  layers.push(...keep.filter(l => !l.hidden));
  const out = _flatten();
  layers.splice(0, layers.length, ...keep);
  return out;
};
