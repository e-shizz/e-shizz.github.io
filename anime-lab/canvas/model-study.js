// Master study of the diffusion model's own image (experiments/out/D-text-only.png), painted with the
// canvas brush engine. Landmarks were measured on a grid; colours are eyedropped from the reference.
// Run: node canvas/run.js canvas/model-study.js canvas/out/model-study.png - canvas/ref/model-full.png
async function draw() {
  init(800, 1000);
  await loadRef(REF_URL);
  const rnd = rng(21);
  const ellipse = (cx, cy, rx, ry, n = 40) => Array.from({ length: n }, (_, i) => [cx + rx * Math.cos(i / n * 2 * Math.PI), cy + ry * Math.sin(i / n * 2 * Math.PI)]);
  const lockW = (w, tip) => t => t < tip ? w : w * Math.cos((t - tip) / (1 - tip) * Math.PI / 2);
  const INK = '#463c5c';

  // Eyedrop a lock's colours along its length: a top-to-tip gradient, a shadow and a highlight tone.
  function lockColors(r) {
    const n = r.c.length, all = [], seg = [[], [], []];
    for (let i = Math.floor(n * 0.05); i < Math.floor(n * 0.85); i++)
      for (const u of [0.25, 0.5, 0.75]) {
        const p = [r.left[i][0] + (r.right[i][0] - r.left[i][0]) * u, r.left[i][1] + (r.right[i][1] - r.left[i][1]) * u];
        all.push(p); seg[Math.min(2, Math.floor(3 * i / n))].push(p);
      }
    return { grad: seg.map(s => pickAlong(s.length ? s : all, 0.5)), shade: pickAlong(all, 0.12), light: pickAlong(all, 0.92) };
  }
  // Lock: gradient fill, soft shadow band down one side, a few strands, thin tapered ink edges.
  function paintLock(l, spine, w, o = {}) {
    const r = ribbon(spine, lockW(w, o.tip ?? 0.6), 20), n = r.c.length, col = o.colors || lockColors(r);
    const a = r.c[0], b = r.c[n - 1];
    F(l, r.outline, lin(l, a[0], a[1], b[0], b[1], [[0.05, col.grad[0]], [0.5, col.grad[1]], [0.95, col.grad[2]]]), { sharp: true });
    const side = o.shadeLeft ? r.left : r.right, other = o.shadeLeft ? r.right : r.left;
    const inner = side.map((p, i) => { const f = 0.3 * (0.5 + 0.5 * Math.sin(Math.PI * i / (n - 1))); return [p[0] + (other[i][0] - p[0]) * f, p[1] + (other[i][1] - p[1]) * f]; });
    F(l, side.concat(inner.reverse()), col.shade, { sharp: true, opacity: o.shadeOpacity ?? 0.3 });
    for (let s = 0, count = Math.round(w * (o.density ?? 0.3)); s < count; s++) {
      const u = 0.08 + rnd() * 0.84, i0 = Math.floor(rnd() * n * 0.3), i1 = Math.min(n - 1, Math.floor(n * (0.5 + rnd() * 0.45)));
      const pts = [];
      for (let i = i0; i <= i1; i++) pts.push([r.left[i][0] + (r.right[i][0] - r.left[i][0]) * u, r.left[i][1] + (r.right[i][1] - r.left[i][1]) * u]);
      if (pts.length > 2) S(l, pts, { raw: true, size: 0.8 + rnd() * 1.2, color: rnd() < 0.5 ? col.light : col.shade, opacity: 0.35 + rnd() * 0.3, hardness: 0.6 });
    }
    for (const e of [r.left, r.right]) S(l, e.slice(Math.floor(n * 0.03)), { raw: true, size: o.line ?? 1.8, color: o.ink || INK, hardness: 0.85, a: 0.3, b: 0.15, opacity: o.lineOpacity ?? 0.9 });
    for (let k = 0; k < (o.split || 0); k++) {
      const i0 = Math.floor(n * (0.5 + rnd() * 0.15)), u = 0.2 + rnd() * 0.6, drift = (o.drift || 0) * (0.5 + rnd()), len = 0.15 + rnd() * 0.2;
      const p0 = [r.left[i0][0] + (r.right[i0][0] - r.left[i0][0]) * u, r.left[i0][1] + (r.right[i0][1] - r.left[i0][1]) * u], end = r.c[n - 1];
      const p2 = [end[0] + drift * 40, end[1] + len * 300], p1 = [(p0[0] + p2[0]) / 2 + drift * 10, (p0[1] + p2[1]) / 2];
      paintLock(l, [p0, p1, p2], 6 + rnd() * 8, { tip: 0.3, colors: col, density: 0.2, line: 1.2, lineOpacity: 0.7 });
    }
    return r;
  }

  // ---------- background: eyedropped sky, stars, moon, cherry blossoms
  const bg = L('bg');
  const sky = [[0, '#142850'], [70, '#2a3d6c'], [190, '#45497f'], [250, '#574e86'], [310, '#6a5890'], [370, '#7d6398'], [430, '#9072a4'], [490, '#a07ead'],
    [550, '#b392b9'], [670, '#d4adcd'], [730, '#dfbad4'], [790, '#e7c4db'], [850, '#f0d0e1'], [910, '#f9dfeb'], [1000, '#fefcfe']];
  const mix = (c1, c2, f) => toHex([0, 1, 2].map(k => parseInt(c1.slice(1 + 2 * k, 3 + 2 * k), 16) * (1 - f) + parseInt(c2.slice(1 + 2 * k, 3 + 2 * k), 16) * f));
  const skyAt = y => { for (let i = 1; i < sky.length; i++) if (y <= sky[i][0]) return mix(sky[i - 1][1], sky[i][1], (y - sky[i - 1][0]) / (sky[i][0] - sky[i - 1][0])); return sky[sky.length - 1][1]; };
  for (let y = 0; y <= 1000; y += 6) S(bg, [[-20, y], [820, y]], { size: 10, color: skyAt(y), hardness: 0.95, pressure: () => 1, segs: 1 });
  D(bg, 688, 108, 150, '#b9b6e0', 0.35); D(bg, 688, 108, 70, '#fff6d8', 0.45);
  for (let i = 0; i < 320; i++) {
    const y = rnd() ** 1.6 * 560, x = rnd() * 800;
    D(bg, x, y, 0.8 + rnd() * 1.6, '#ffffff', 0.35 + rnd() * 0.6);
  }
  F(bg, ellipse(688, 108, 38, 38), '#fff9d4', { sharp: true });
  F(bg, ellipse(700, 118, 22, 18), '#f3eab8', { sharp: true, opacity: 0.35 });
  const blossom = L('blossom');
  for (const p of [[[0, 480], [40, 470], [90, 455], [125, 445]], [[0, 360], [30, 340], [70, 320]], [[0, 250], [25, 230]], [[0, 610], [30, 615], [55, 640]],
    [[800, 300], [760, 290], [720, 282]], [[800, 430], [770, 450], [742, 472]], [[800, 560], [770, 542]], [[800, 880], [760, 900], [740, 930]]])
    S(blossom, p, { size: 5, color: '#4a3346', a: 0.05, b: 0.6 });
  for (const [cx, cy, rad] of [[20, 200, 35], [30, 330, 45], [80, 420, 50], [40, 500, 52], [112, 520, 30], [22, 650, 42],
    [760, 265, 40], [778, 440, 46], [762, 560, 40], [792, 860, 62], [772, 960, 46]]) {
    for (let i = 0; i < rad * 1.4; i++) {
      const a = rnd() * 2 * Math.PI, d = Math.sqrt(rnd()) * rad, x = cx + Math.cos(a) * d, y = cy + Math.sin(a) * d * 0.8;
      const c = ['#f7cfe0', '#fbe6ef', '#eeb0cc', '#f4bcd5'][Math.floor(rnd() * 4)];
      for (let k = 0; k < 5; k++) { const t = k / 5 * 2 * Math.PI + a; D(blossom, x + Math.cos(t) * 3.2, y + Math.sin(t) * 3.2, 3.6, c, 0.9); }
      D(blossom, x, y, 1.6, '#c86a92', 0.7);
    }
  }
  for (const [x, y, a] of [[218, 152, 40], [98, 316, 120], [722, 135, -30], [721, 224, 20], [638, 282, 60], [690, 432, -50], [118, 770, 30], [784, 890, 10]])
    F(blossom, rotate(ellipse(x, y, 5, 10, 14), a, x, y), '#f6c1d8', { sharp: true, opacity: 0.9 });

  // ---------- back hair: light on the lit left, dusky purple on the right, dark behind the neck
  const back = L('backhair');
  F(back, [[300, 400], [530, 400], [565, 640], [270, 640]], '#5a4e6e', { sharp: true });
  for (const [p, w] of [[[[250, 300], [215, 420], [185, 560], [150, 720], [118, 880], [100, 1000]], 60],
    [[[565, 300], [595, 420], [618, 560], [645, 720], [675, 880], [700, 1000]], 60],
    [[[300, 420], [290, 560], [282, 700]], 50], [[[525, 420], [538, 560], [548, 700]], 50]])
    paintLock(back, p, w, { tip: 0.9, density: 0.4 });

  // ---------- neck, shirt, sailor collar, scarf
  const body = L('body');
  F(body, [[365, 460], [458, 460], [455, 560], [478, 605], [412, 650], [345, 605], [368, 560]], pick(410, 560), { sharp: true });
  F(body, [[365, 470], [460, 470], [456, 528], [412, 556], [367, 528]], pick(412, 520), { sharp: true });
  const SHIRT = [[200, 553], [290, 540], [340, 562], [412, 612], [480, 562], [540, 575], [585, 590], [628, 610], [650, 640], [656, 720], [660, 850], [662, 1000], [148, 1000], [146, 850], [142, 720], [140, 630], [158, 585]];
  F(body, SHIRT, lin(body, 140, 0, 660, 0, [[0, pick(190, 760)], [0.45, pick(330, 820)], [1, pick(600, 800)]]), { sharp: true });
  F(body, [[340, 588], [412, 640], [490, 592], [468, 690], [452, 742], [430, 700]], pick(430, 670), { sharp: true });
  const shirtShade = L('shirt_shade');
  for (const [p, w] of [[[[232, 615], [252, 700], [256, 800], [262, 900], [266, 1000]], 26], [[[205, 640], [194, 760], [188, 900], [186, 1000]], 18],
    [[[600, 640], [592, 760], [584, 900], [580, 1000]], 26], [[[330, 745], [348, 820], [360, 900], [366, 1000]], 14], [[[540, 700], [552, 800], [560, 900]], 16]])
    S(shirtShade, p, { size: w, color: pickAlong(p, 0.2), opacity: 0.55, hardness: 0.3 });
  clipL(shirtShade, 'body');
  const collar = L('collar');
  const EDGE_L = [[200, 555], [230, 585], [265, 620], [300, 648], [340, 676], [385, 706], [425, 730], [450, 742]];
  const EDGE_R = [[588, 592], [575, 620], [555, 650], [525, 680], [495, 712], [470, 742]];
  const offset = (pts, d) => ribbon(pts, () => 2 * Math.abs(d), 8)[d < 0 ? 'left' : 'right'];
  const SCARF = pick(420, 860), STRIPE = '#d4709a';
  F(collar, offset(EDGE_L, 1).concat(offset(EDGE_L, 16).reverse()), SCARF, { sharp: true });
  F(collar, offset(EDGE_R, -1).concat(offset(EDGE_R, -16).reverse()), SCARF, { sharp: true });
  for (const d of [-5, -12]) {
    S(collar, offset(EDGE_L, d), { raw: true, size: 3.2, color: STRIPE, pressure: () => 1 });
    S(collar, offset(EDGE_R, -d), { raw: true, size: 3.2, color: STRIPE, pressure: () => 1 });
  }
  for (const dy of [0, 10]) S(collar, [[330, 618 + dy], [412, 632 + dy], [530, 620 + dy]], { size: 3, color: STRIPE, pressure: () => 1 });
  S(collar, EDGE_L, { size: 2.2, color: INK, a: 0.05, b: 0.05 });
  S(collar, EDGE_R, { size: 2.2, color: INK, a: 0.05, b: 0.05 });
  const tails = [[[430, 768], [470, 772], [462, 830], [455, 900], [450, 960], [448, 1004], [392, 1004], [398, 950], [410, 880], [420, 820]],
    [[465, 772], [500, 768], [512, 830], [522, 900], [530, 960], [534, 1004], [480, 1004], [476, 940], [470, 880], [466, 820]]];
  for (const t of tails) { F(collar, t, SCARF); S(collar, t.concat([t[0]]), { size: 1.8, color: '#2e0d1e', pressure: () => 1, segs: 6 }); }
  S(collar, [[450, 800], [436, 880], [410, 990]], { size: 7, color: '#3a1028', opacity: 0.6, hardness: 0.4 });
  S(collar, [[490, 800], [502, 880], [516, 990]], { size: 6, color: '#3a1028', opacity: 0.6, hardness: 0.4 });
  F(collar, [[436, 736], [492, 736], [496, 760], [488, 786], [446, 788], [436, 764]], SCARF);
  S(collar, [[436, 736], [492, 736], [496, 760], [488, 786], [446, 788], [436, 764], [436, 736]], { size: 1.8, color: '#2e0d1e', pressure: () => 1, segs: 4 });
  const shirtLine = L('shirt_line');
  for (const p of [[[158, 585], [142, 640], [140, 760], [146, 900], [148, 1000]], [[628, 610], [652, 660], [656, 780], [660, 1000]], [[350, 600], [395, 625], [430, 640], [470, 625], [490, 600]]])
    S(shirtLine, p, { size: 2, color: INK, opacity: 0.8, a: 0.1, b: 0.1 });

  // ---------- face
  const face = L('face');
  const FACE = [[285, 230], [288, 320], [292, 380], [300, 410], [315, 440], [340, 465], [372, 486], [400, 498], [412, 500], [426, 499], [452, 489], [490, 468], [515, 440], [528, 405], [532, 360], [530, 300], [525, 230]];
  F(face, FACE, lin(face, 300, 0, 530, 0, [[0, pick(330, 450)], [0.5, pick(412, 470)], [1, pick(505, 432)]]), { sharp: true });
  D(face, 350, 425, 62, '#cdb8b1', 0.55, { sy: 0.8 }); D(face, 510, 420, 55, '#98827e', 0.4, { sy: 1.2 }); D(face, 412, 492, 42, '#9e8884', 0.35, { sy: 0.45 });
  F(face, [[300, 385], [287, 388], [282, 405], [287, 425], [300, 433]], pick(292, 410), { sharp: true });
  S(face, [[298, 392], [290, 405], [295, 422]], { size: 1.6, color: '#7a5552', opacity: 0.6 });
  for (const [x, y] of [[333, 392], [488, 392]]) {
    D(face, x, y, 28, pick(x, y + 2), 0.75, { sy: 0.5 });
    for (let k = 0; k < 4; k++) S(face, [[x - 10 + k * 7, y - 4], [x - 14 + k * 7, y + 5]], { size: 1.4, color: '#c6747f', opacity: 0.6, segs: 1 });
  }

  // eyes: whites, purple irises, thick upper lashes with a flick, sparkle highlights
  const white = L('eyes_white'), iris = L('eyes_iris'), lash = L('eyes_line');
  const EYES = [
    { c: [340, 345], open: [[293, 331], [300, 323], [318, 318], [340, 317], [360, 319], [374, 325], [379, 333], [372, 350], [360, 362], [342, 368], [322, 366], [305, 357], [296, 345]],
      lash: [[284, 335], [296, 322], [318, 314], [342, 313], [364, 316], [378, 324], [383, 332]], lower: [[300, 353], [318, 365], [340, 370], [362, 365]], hl: [[330, 331, 6.5], [352, 360, 2.6]] },
    { c: [480, 345], open: [[448, 333], [453, 325], [468, 319], [488, 317], [510, 318], [527, 323], [536, 331], [532, 346], [520, 359], [500, 367], [480, 368], [462, 362], [452, 350]],
      lash: [[443, 332], [450, 323], [470, 315], [495, 313], [518, 316], [534, 324], [542, 335]], lower: [[458, 362], [480, 370], [502, 366], [522, 356]], hl: [[472, 331, 6.5], [492, 360, 2.6]] },
  ];
  for (const e of EYES) {
    const [cx, cy] = e.c;
    F(white, e.open, pick(e.open[1][0] + 4, cy - 2));
    F(iris, ellipse(cx, cy, 26, 27), '#9c5a9f', { sharp: true });
    F(iris, ellipse(cx, cy - 10, 26, 16).filter(([, y]) => y < cy - 2).concat([[cx + 26, cy - 2], [cx - 26, cy - 2]]), '#4f2247', { opacity: 0.85 });
    F(iris, ellipse(cx, cy + 14, 19, 10), '#c584bb', { sharp: true, opacity: 0.85 });
    F(iris, ellipse(cx + 1, cy + 2, 9, 13), '#2a0f2c', { sharp: true });
    S(iris, ellipse(cx, cy, 26, 27, 48).concat([[cx + 26, cy]]), { raw: true, size: 2, color: '#3a1840', pressure: () => 1 });
    S(iris, [e.open[0], e.open[2], e.open[4], e.open[6]], { size: 10, color: '#4a3060', hardness: 0.2, opacity: 0.45 });
    S(lash, e.lash, { size: 10, color: '#1e1424', hardness: 0.9, pressure: t => 0.35 + 0.65 * Math.sin(Math.PI * Math.min(1, t * 1.15)) });
    const o0 = cx < 400 ? e.lash[0] : e.lash[e.lash.length - 1], dir = cx < 400 ? -1 : 1;
    S(lash, [o0, [o0[0] + dir * 6, o0[1] + 7], [o0[0] + dir * 9, o0[1] + 13]], { size: 4, color: '#1e1424', a: 0.1, b: 0.8 });
    F(lash, ellipse(cx + 9, cy + 9, 2.2, 2.2, 10), '#ffe6f6', { sharp: true, opacity: 0.9 });
    S(lash, e.lower, { size: 1.6, color: '#5a3a50', opacity: 0.6, a: 0.3, b: 0.3 });
    for (const [x, y, r] of e.hl) F(lash, ellipse(x, y, r, r * 1.1, 16), '#ffffff', { sharp: true, opacity: 0.95 });
  }
  clipL(iris, 'eyes_white');
  const fl = L('face_line');
  S(fl, [[414, 375], [418, 383], [413, 389]], { size: 2.4, color: '#8a6e70', opacity: 0.7, a: 0.3, b: 0.3 });
  F(fl, ellipse(410, 381, 2.5, 4, 10), '#e9dbd6', { sharp: true, opacity: 0.8 });
  F(fl, [[383, 427], [398, 432], [412, 433], [427, 432], [442, 426], [440, 440], [430, 452], [412, 458], [395, 453], [386, 441]], pick(412, 438));
  F(fl, [[394, 450], [404, 444], [412, 443], [421, 444], [431, 450], [423, 456], [412, 459], [401, 456]], pick(412, 447));
  S(fl, [[392, 452], [402, 458], [412, 460], [423, 457], [432, 451]], { size: 1.6, color: '#7a4a52', opacity: 0.6 });
  S(fl, [[382, 428], [398, 432], [412, 433], [427, 432], [443, 426]], { size: 2.4, color: '#4a2430', a: 0.2, b: 0.2 });
  for (const p of [[[312, 271], [332, 267], [352, 267], [368, 271]], [[452, 271], [470, 267], [490, 267], [508, 272]]]) S(fl, p, { size: 2.5, color: INK, opacity: 0.7, a: 0.2, b: 0.4 });
  S(fl, [[293, 386], [300, 412], [315, 440], [340, 463], [372, 482]], { size: 5, color: pick(330, 440), opacity: 0.9, hardness: 0.5, a: 0.2, b: 0.3 });
  S(fl, [[292, 380], [300, 410], [315, 440], [340, 465], [372, 486], [400, 498], [412, 500], [426, 499], [452, 489], [490, 468], [515, 440], [528, 405]], { size: 2.2, color: '#5a3a44', a: 0.15, b: 0.15 });

  // ---------- front hair: lit left locks, braid and ribbon, dusky right locks, crown, bangs, ahoge
  const hair = L('hair');
  const sideL = [[[[262, 150], [230, 245], [206, 350], [184, 480], [158, 630], [130, 790], [110, 930], [100, 1000]], 46],
    [[[275, 165], [250, 265], [234, 390], [219, 535], [198, 685], [172, 835], [152, 965]], 42],
    [[[292, 205], [276, 305], [267, 420], [259, 540], [247, 680], [231, 820], [215, 960]], 36],
    [[[230, 600], [205, 720], [170, 840], [140, 960]], 18], [[[250, 640], [236, 760], [214, 880], [196, 1000]], 14]];
  const sideR = [[[[540, 150], [575, 255], [595, 385], [614, 535], [637, 685], [666, 835], [694, 965], [702, 1000]], 50],
    [[[532, 175], [556, 285], [570, 415], [583, 565], [602, 715], [626, 855], [646, 985]], 42],
    [[[526, 232], [540, 332], [546, 442], [552, 562], [566, 702], [585, 842], [598, 952]], 32],
    [[[580, 600], [606, 720], [640, 840], [672, 960]], 18], [[[565, 640], [582, 760], [604, 880], [620, 1000]], 14]];
  const locks = [];
  for (const [p, w] of sideL) locks.push(paintLock(hair, p, w, { tip: 0.85, shadeLeft: false, split: w > 30 ? 3 : 0, drift: -1, lineOpacity: 0.7 }));
  for (const [p, w] of sideR) locks.push(paintLock(hair, p, w, { tip: 0.85, shadeLeft: true, split: w > 30 ? 3 : 0, drift: 1, lineOpacity: 0.7 }));
  S(hair, locks[0].left.slice(4), { raw: true, size: 5, color: '#fbf4fb', opacity: 0.8, hardness: 0.4, a: 0.1, b: 0.4 });
  for (let i = 0; i < 7; i++) {
    const t = i / 6, x = 236 - 26 * t, y = 238 + 92 * t, off = i % 2 ? 4 : -4;
    F(hair, rotate(ellipse(x + off, y, 13, 8, 18), -68, x + off, y), pick(x + off, y), { sharp: true });
    S(hair, rotate(ellipse(x + off, y, 13, 8, 18), -68, x + off, y).concat([[x + off + 13 * Math.cos(-68 * Math.PI / 180), y + 13 * Math.sin(-68 * Math.PI / 180)]]), { raw: true, size: 1.4, color: '#9a8fb0', pressure: () => 1 });
  }
  const RIBBON = '#e47e9e';
  paintLock(hair, [[197, 342], [193, 390], [188, 430], [182, 468]], 11, { tip: 0.8, colors: { grad: [RIBBON, RIBBON, '#c95a7e'], shade: '#b84870', light: '#ffc0d4' }, ink: '#7a2848', density: 0 });
  F(hair, [[198, 330], [172, 345], [165, 375], [178, 392], [196, 350]], RIBBON);
  S(hair, [[198, 330], [172, 345], [165, 375], [178, 392], [196, 350]], { size: 1.6, color: '#7a2848', pressure: () => 1, segs: 6 });
  F(hair, ellipse(201, 336, 7, 7, 14), '#c95a7e', { sharp: true });
  const CROWN = [[226, 215], [240, 160], [280, 112], [340, 84], [400, 72], [460, 76], [515, 96], [555, 128], [580, 180], [590, 230], [540, 250], [300, 250]];
  F(hair, CROWN, lin(hair, 240, 0, 580, 0, [[0, pick(270, 140)], [0.5, pick(420, 92)], [1, pick(560, 170)]]));
  S(hair, [[226, 215], [240, 160], [280, 112], [340, 84], [400, 72], [460, 76], [515, 96], [555, 128], [580, 180], [590, 230]], { size: 2.2, color: INK, a: 0.05, b: 0.05 });
  const bangs = [[[[395, 100], [330, 128], [292, 180], [280, 260], [290, 335]], 40], [[[442, 100], [505, 128], [542, 188], [552, 265], [545, 335]], 40],
    [[[402, 100], [352, 140], [322, 198], [309, 270], [308, 340]], 38], [[[436, 100], [481, 140], [506, 200], [517, 270], [515, 340]], 38],
    [[[410, 100], [376, 150], [353, 208], [343, 275], [345, 342]], 34], [[[430, 100], [460, 160], [476, 222], [481, 285], [476, 345]], 34],
    [[[415, 100], [396, 160], [381, 220], [374, 285], [374, 348]], 30], [[[425, 100], [438, 170], [446, 240], [444, 300], [440, 350]], 30],
    [[[420, 100], [413, 170], [406, 240], [403, 300], [406, 355]], 28]];
  const bangR = bangs.map(([p, w]) => paintLock(hair, p, w, { tip: 0.5, shadeLeft: p[p.length - 1][0] > 420 }));
  const shine = L('hair_shine', 'screen');
  for (let x = 305; x <= 545; x += 7 + rnd() * 6) {
    const y = 198 + 0.0006 * (x - 425) ** 2, len = 6 + rnd() * 10;
    S(shine, [[x, y - len], [x + (x - 425) * 0.02, y + len * 0.6]], { size: 3 + rnd() * 3, color: '#d8d0e6', opacity: 0.8, a: 0.5, b: 0.5, segs: 1 });
  }
  clipL(shine, 'hair');
  const ahoge = L('ahoge');
  paintLock(ahoge, [[414, 78], [410, 40], [388, 16], [345, 10], [302, 24], [272, 54], [262, 92]], 20, { tip: 0.35, density: 0.2 });
  // cast shadow of the bangs on the forehead
  const castShadow = L('bang_shadow', 'multiply', 0.35);
  for (const r of bangR) F(castShadow, shift(r.outline, 4, 12), '#b88f99', { sharp: true });
  clipL(castShadow, 'face');
  layers.splice(layers.indexOf(castShadow), 1);
  layers.splice(layers.indexOf(white), 0, castShadow);

  const out = flatten();
  glow(out, 12, 0.14);
  grain(out, 5);
  return out.toDataURL('image/png');
}
