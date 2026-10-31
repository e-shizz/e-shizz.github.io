// A tiny "brush engine" for SVG: turns centerlines into tapered, ink-like shapes.
const fmt = n => Math.round(n * 100) / 100;

// Smooth curve through the given points (Catmull-Rom), as a dense polyline.
function spline(pts, segs = 16) {
  const out = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || pts[i + 1];
    for (let s = 0; s < segs; s++) {
      const t = s / segs, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map(k => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * t +
        (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
        (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

// Offset the curve on both sides by width(t)/2, t running 0..1 along its length.
function ribbon(pts, width, segs = 16) {
  const c = spline(pts, segs);
  const len = [0];
  for (let i = 1; i < c.length; i++) len.push(len[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));
  const total = len[len.length - 1] || 1;
  const left = [], right = [];
  c.forEach((p, i) => {
    const a = c[Math.max(i - 1, 0)], b = c[Math.min(i + 1, c.length - 1)];
    const dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) || 1;
    const nx = -dy / d, ny = dx / d, w = width(len[i] / total) / 2;
    left.push([p[0] + nx * w, p[1] + ny * w]);
    right.push([p[0] - nx * w, p[1] - ny * w]);
  });
  return { c, left, right };
}

const path = pts => 'M' + pts.map(p => fmt(p[0]) + ',' + fmt(p[1])).join('L') + 'Z';
const outline = r => path(r.left.concat(r.right.slice().reverse()));
const el = (d, fill, extra = '') => `<path d="${d}" fill="${fill}"${extra ? ' ' + extra : ''}/>`;

// Width profiles. ink: thin-thick-thin pen pressure. lockWidth: full, then a sharp tip.
const ink = (w, a = 0.25, b = 0.35, min = 0) => t => Math.max(min, w * Math.min(1, t / a, (1 - t) / b) ** 0.6);
const lockWidth = (w, tip = 0.55) => t => t < tip ? w : w * Math.cos((t - tip) / (1 - tip) * Math.PI / 2);

function stroke(pts, w, color, opts = {}) {
  return el(outline(ribbon(pts, typeof w === 'function' ? w : ink(w, opts.a, opts.b, opts.min))), color, opts.extra);
}

// A hair clump: body shape, a cel-shaded band along one edge, and tapered edge lines.
function lock(spine, w, o = {}) {
  const r = ribbon(spine, lockWidth(w, o.tip ?? 0.55), 20);
  const [lit, dark] = o.shade === 'left' ? [r.right, r.left] : [r.left, r.right];
  const k = o.shadeK ?? 0.42;
  const inner = dark.map((p, i) => {
    const t = i / (dark.length - 1), f = k * (0.55 + 0.45 * Math.sin(Math.PI * t));
    return [p[0] + (lit[i][0] - p[0]) * f, p[1] + (lit[i][1] - p[1]) * f];
  });
  const edges = [r.left, r.right].map(e => outline(ribbon(e, ink(o.line ?? 2.6, 0.3, 0.2), 1)));
  return { body: outline(r), shade: path(dark.concat(inner.reverse())), edges };
}

// Four-pointed twinkle.
function sparkle(x, y, r, fill, extra = '') {
  const q = r * 0.18, [a, b, c, d] = [x, y, r, q].map(fmt);
  return el(`M${a},${b - c}Q${a + d},${b - d} ${a + c},${b}Q${a + d},${b + d} ${a},${b + c}Q${a - d},${b + d} ${a - c},${b}Q${a - d},${b - d} ${a},${b - c}Z`, fill, extra);
}

// Seeded random so drawings are reproducible.
function rng(seed) {
  return () => {
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

module.exports = { fmt, spline, ribbon, path, outline, el, ink, lockWidth, stroke, lock, sparkle, rng };
