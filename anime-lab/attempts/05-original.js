// Attempt 05: same character; shorter face, bigger eyes, livelier hair, slight head tilt.
// Run: node attempts/05-original.js && node render.js attempts/05-original.svg attempts/05-original.png
const fs = require('fs');
const B = require('../engine/brush');
const { el, stroke, lock, sparkle, ink, rng, fmt } = B;

const W = 800, H = 1000;
const TILT = 'rotate(-4 400 600)';
const CHAR_ONLY = process.env.LAYER === 'char'; // LAYER=char renders just the character, for compositing
const DENSE = process.env.DENSE === '1'; // DENSE=1 fills every hair lock with fine strands
const mirror = pts => pts.map(([x, y]) => [W - x, y]);
const defs = [], out = [];
const C = {
  skin: '#fff1e8', skinShade: '#f5c9bb', skinLine: '#a8645a',
  hair: '#e8e0f4', hairShade: '#b9a9d8', hairBack: '#a593cd', hairBackShade: '#8572b2', hairLine: '#54417c',
  irisTop: '#4a230a', irisMid: '#b8621a', irisLow: '#f3b23c', irisGlow: '#ffe7a0', eyeLine: '#2c1830',
  blouse: '#f8f8fc', blouseShade: '#c9cce3', blouseLine: '#666b96',
  collar: '#2a3266', collarShade: '#1b2149', collarLine: '#10142e',
  ribbon: '#d8394f', ribbonShade: '#a5223a', ribbonLine: '#5a0f1f',
};

// ---------- shapes shared by fills, clips and cast shadows
const FACE = 'M236,262 L238,440 Q292,548 364,598 Q400,620 436,598 Q508,548 562,440 L564,262 Z';
const NECK = 'M358,575 L352,700 L340,730 L400,872 L460,730 L448,700 L442,575 Z';
const TORSO = 'M348,735 L400,860 L452,735 C500,768 595,790 650,824 C696,852 718,960 724,1040 L76,1040 C82,960 104,852 150,824 C205,790 300,768 348,735 Z';
const COLLAR_L = 'M348,735 L400,860 L400,948 L176,820 C225,792 300,768 348,735 Z';
const COLLAR_R = 'M452,735 L400,860 L400,948 L624,820 C575,792 500,768 452,735 Z';
const DICKEY = 'M366,800 L434,800 L400,862 Z';
const BACK = 'M400,128 C292,128 206,196 198,330 C190,460 188,580 168,720 C150,840 132,930 120,1000 L680,1000 C668,930 650,840 632,720 C612,580 610,460 602,330 C594,196 508,128 400,128 Z';
const CROWN = 'M206,345 C200,200 290,128 400,128 C510,128 600,200 594,345 C560,300 470,250 400,250 C330,250 240,300 206,345 Z';

defs.push(`
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#232a63"/><stop offset="0.55" stop-color="#6a5aa6"/><stop offset="1" stop-color="#f4a9bb"/></linearGradient>
<radialGradient id="moon"><stop offset="0" stop-color="#fff8ec" stop-opacity="0.95"/><stop offset="0.4" stop-color="#ffe6dc" stop-opacity="0.5"/><stop offset="1" stop-color="#ffd0e0" stop-opacity="0"/></radialGradient>
<linearGradient id="iris" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${C.irisTop}"/><stop offset="0.45" stop-color="${C.irisMid}"/><stop offset="0.8" stop-color="${C.irisLow}"/><stop offset="1" stop-color="${C.irisGlow}"/></linearGradient>
<linearGradient id="lidShadow" gradientUnits="userSpaceOnUse" x1="0" y1="395" x2="0" y2="430"><stop offset="0" stop-color="#4b3a86" stop-opacity="0.6"/><stop offset="1" stop-color="#4b3a86" stop-opacity="0"/></linearGradient>
<radialGradient id="blush"><stop offset="0" stop-color="#ff8ea4" stop-opacity="0.6"/><stop offset="1" stop-color="#ff8ea4" stop-opacity="0"/></radialGradient>
<linearGradient id="skinGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff6ef"/><stop offset="1" stop-color="${C.skin}"/></linearGradient>
<linearGradient id="crownGrad" x1="0.2" y1="0" x2="0.8" y2="1"><stop offset="0" stop-color="#fbf8ff"/><stop offset="0.6" stop-color="${C.hair}"/></linearGradient>
<radialGradient id="light" cx="0.15" cy="0.05" r="0.95"><stop offset="0" stop-color="#fff4e0" stop-opacity="0.9"/><stop offset="1" stop-color="#3a2a70" stop-opacity="0.6"/></radialGradient>
<linearGradient id="hairGrad" gradientUnits="userSpaceOnUse" x1="0" y1="140" x2="0" y2="900"><stop offset="0" stop-color="#f6f2fc"/><stop offset="0.3" stop-color="${C.hair}"/><stop offset="1" stop-color="#ecc8e6"/></linearGradient>
<linearGradient id="hairShadeGrad" gradientUnits="userSpaceOnUse" x1="0" y1="140" x2="0" y2="900"><stop offset="0" stop-color="#c4b6e2"/><stop offset="0.3" stop-color="${C.hairShade}"/><stop offset="1" stop-color="#c79bd2"/></linearGradient>
<linearGradient id="backShadeGrad" gradientUnits="userSpaceOnUse" x1="0" y1="430" x2="0" y2="760"><stop offset="0" stop-color="#4f3f86"/><stop offset="1" stop-color="${C.hairBackShade}"/></linearGradient>
<filter id="blur3" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3"/></filter>
<clipPath id="faceClip"><path d="${FACE}"/></clipPath>
<clipPath id="neckClip"><path d="${NECK}"/></clipPath>
<clipPath id="torsoClip"><path d="${TORSO}"/></clipPath>
<clipPath id="backClip"><path d="${BACK}"/></clipPath>
<clipPath id="crownClip"><path d="${CROWN}"/></clipPath>
<clipPath id="collarClip"><path d="${COLLAR_L}"/><path d="${COLLAR_R}"/></clipPath>`);

// Cel shadow on the far side of a shape: the shape minus a copy of itself nudged toward the light.
let maskN = 0;
function crescent(d, dx, dy, fill) {
  const id = 'm' + maskN++;
  defs.push(`<mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="${W}" height="${H}"><rect width="${W}" height="${H}" fill="#fff"/><path d="${d}" transform="translate(${dx},${dy})" fill="#000"/></mask>`);
  return `<path d="${d}" fill="${fill}" mask="url(#${id})"/>`;
}

let clipN = 0;
function drawLock(spine, w, o = {}) {
  const L = lock(spine, w, o);
  let s = '<g>' + el(L.body, o.fill || 'url(#hairGrad)') + el(L.shade, o.shadeFill || 'url(#hairShadeGrad)');
  if (o.shine || DENSE) {
    const id = 'c' + clipN++;
    defs.push(`<clipPath id="${id}"><path d="${L.body}"/></clipPath>`);
    s += `<g clip-path="url(#${id})">${DENSE ? fineStrands(spine, w, o.light ?? 0.35) : ''}${o.shine || ''}</g>`;
  }
  s += L.edges.map(d => el(d, o.lineColor || C.hairLine)).join('') + '</g>';
  return { svg: s, body: L.body };
}

// Fine strands following the lock's spine at random offsets, light and dark, clipped to the lock.
let strandSeed = 100;
function fineStrands(spine, w, light) {
  const r = rng(strandSeed++);
  let s = '';
  for (let k = 0, n = Math.round(w / 3); k < n; k++) {
    const f = (r() - 0.5) * 0.95 * w, side = B.ribbon(spine, () => 2 * Math.abs(f), 16)[f < 0 ? 'left' : 'right'];
    const a = Math.floor(r() * side.length * 0.25), b = Math.floor(side.length * (0.55 + r() * 0.45));
    const lit = r() < 0.6, op = lit ? light * (0.6 + r() * 0.7) : 0.18 + r() * 0.2;
    s += stroke(side.slice(a, b), ink(0.6 + r() * 1.1, 0.25, 0.3), lit ? '#ffffff' : C.hairLine, { segs: 1, extra: `opacity="${fmt(op)}"` });
  }
  return s;
}

// The "angel ring": jagged highlight band that follows the curve of the head.
function shineShards() {
  const r = rng(3);
  let s = '';
  for (let i = 0, x = 232; x <= 568; x += 6.5, i++) {
    const y = 262 + 0.0011 * (x - 400) ** 2;
    const up = i % 2 ? 8 + r() * 8 : 14 + r() * 16, down = i % 2 ? 6 + r() * 6 : 10 + r() * 10, hw = 2.5 + r() * 2.5;
    const a = -Math.atan2(x - 400, y - 120) * 180 / Math.PI;
    s += `<path d="M0,${fmt(-up)}L${fmt(hw)},0L0,${fmt(down)}L${fmt(-hw)},0Z" transform="translate(${fmt(x)},${fmt(y)}) rotate(${fmt(a)})" fill="#fff" opacity="0.9"/>`;
  }
  return s;
}

function eye(cx, cy, s, id, k = 1.08) {
  const P = (u, v) => [cx - s * u * k, cy + v * k]; // u points away from the nose, v down
  const open = B.path(B.spline([P(-46, -2), P(-26, -26), P(4, -34), P(30, -28), P(48, -9), P(44, 10), P(28, 34), P(4, 44), P(-22, 38), P(-40, 18), P(-46, -2)], 10));
  defs.push(`<clipPath id="${id}"><path d="${open}"/></clipPath>`);
  const [ix, iy] = P(2, 8);
  let g = `<g clip-path="url(#${id})">` + el(open, '#fff');
  g += `<ellipse cx="${ix}" cy="${iy}" rx="${fmt(28 * k)}" ry="${fmt(39 * k)}" fill="url(#iris)" stroke="${C.irisTop}" stroke-width="2.5"/>`;
  for (let a = 25; a <= 155; a += 18) {
    const r = a * Math.PI / 180, c = Math.cos(r), sn = Math.sin(r);
    g += stroke([[ix + 12 * k * c, iy + 19 * k * sn], [ix + 25 * k * c, iy + 35 * k * sn]], ink(1.8, 0.3, 0.3), C.irisGlow, { extra: 'opacity="0.55"' });
  }
  g += `<ellipse cx="${ix}" cy="${fmt(iy + 2 * k)}" rx="${fmt(11 * k)}" ry="${fmt(19 * k)}" fill="#2a1206"/>`;
  g += el(`M${fmt(ix - 21 * k)},${fmt(iy + 18 * k)}Q${ix},${fmt(iy + 46 * k)} ${fmt(ix + 21 * k)},${fmt(iy + 18 * k)}Q${ix},${fmt(iy + 30 * k)} ${fmt(ix - 21 * k)},${fmt(iy + 18 * k)}Z`, C.irisGlow, 'opacity="0.75"');
  g += el(open, 'url(#lidShadow)') + '</g>';
  // upper lash: thin at the inner corner, heavy at the outer corner, flicked out
  g += stroke([P(-48, -3), P(-28, -27), P(2, -36), P(30, -30), P(48, -12), P(58, -7)],
    t => t < 0.8 ? 3 + 9 * (t / 0.8) ** 1.3 : 12 * ((1 - t) / 0.2) ** 0.7, C.eyeLine);
  g += stroke([P(44, -20), P(54, -29), P(62, -32)], ink(4.5, 0.1, 0.85), C.eyeLine);
  g += stroke([P(52, -11), P(61, -15), P(66, -14)], ink(3.5, 0.1, 0.85), C.eyeLine);
  g += stroke([P(44, 6), P(36, 24), P(18, 38)], ink(2.6, 0.2, 0.7), C.eyeLine, { extra: 'opacity="0.85"' });
  g += stroke([P(-28, -46), P(2, -54), P(32, -48)], ink(2.4, 0.4, 0.4), C.skinLine, { extra: 'opacity="0.7"' });
  // highlights stay on the same side for both eyes (one light source)
  g += `<ellipse cx="${fmt(ix - 10 * k)}" cy="${fmt(iy - 16 * k)}" rx="${fmt(9 * k)}" ry="${fmt(12 * k)}" fill="#fff" transform="rotate(-20 ${fmt(ix - 10 * k)} ${fmt(iy - 16 * k)})"/>`;
  g += `<circle cx="${fmt(ix + 11 * k)}" cy="${fmt(iy + 21 * k)}" r="${fmt(4 * k)}" fill="#fff"/><circle cx="${fmt(ix - 14 * k)}" cy="${fmt(iy + 17 * k)}" r="${fmt(2.2 * k)}" fill="#fff" opacity="0.9"/>`;
  return g;
}

// ---------- hair locks (computed first: their shapes also cast shadows)
const shine = shineShards();
const sides = [];
const lengthen = (pts, dy) => pts.map((p, i) => i === pts.length - 1 ? [p[0], p[1] + dy] : p);
for (const s of [
  { p: [[214, 290], [192, 440], [182, 600], [178, 740], [168, 860]], w: 52, tip: 0.55, back: true },
  { p: [[238, 250], [220, 380], [210, 520], [206, 640], [214, 760], [236, 870]], w: 58, tip: 0.5, dy: 24 },
  { p: [[256, 270], [238, 400], [232, 520], [238, 630], [256, 720], [282, 790]], w: 40, tip: 0.45, dy: -18 },
  { p: [[248, 300], [228, 430], [222, 560], [230, 680], [248, 760]], w: 18, tip: 0.4, dy: 30 },
]) for (const p of [s.p, lengthen(mirror(s.p), s.dy || 0)]) sides.push(drawLock(p, s.w, s.back
  ? { tip: s.tip, fill: C.hairShade, shadeFill: C.hairBackShade }
  : { tip: s.tip, shine }));

const bangRnd = rng(8);
const bangs = [
  [[[262, 205], [244, 280], [234, 345], [236, 380], [244, 398]], 46],
  [[[538, 205], [556, 280], [566, 345], [564, 380], [556, 396]], 46],
  [[[300, 178], [274, 250], [258, 318], [257, 350], [264, 374]], 60],
  [[[503, 180], [528, 250], [544, 318], [545, 346], [538, 368]], 58],
  [[[276, 260], [262, 320], [250, 356], [244, 366]], 18, 0.5],
  [[[526, 260], [540, 320], [552, 352], [558, 360]], 18, 0.5],
  [[[342, 172], [320, 248], [304, 318], [302, 358], [310, 386]], 62],
  [[[462, 172], [482, 248], [496, 318], [498, 354], [490, 380]], 60],
  [[[318, 270], [310, 330], [296, 372], [288, 384]], 20, 0.5],
  [[[486, 270], [494, 330], [506, 366], [512, 376]], 20, 0.5],
  [[[352, 190], [338, 280], [332, 350], [336, 394]], 14, 0.45],
  [[[452, 190], [462, 280], [468, 350], [463, 392]], 14, 0.45],
  [[[422, 176], [431, 262], [436, 340], [431, 398]], 54],
  [[[384, 172], [372, 258], [363, 338], [364, 405]], 58],
  [[[376, 260], [372, 330], [376, 380], [384, 398]], 18, 0.5],
  [[[400, 196], [403, 300], [399, 384], [404, 446]], 28, 0.5],
].map(([p, w, tip]) => drawLock(p, w, { tip: tip ?? 0.55, shine, shadeK: 0.3 + bangRnd() * 0.25 }));
const hanging = [];
for (const [p, w] of [[[[296, 430], [292, 560], [286, 680], [280, 770]], 50], [[[328, 470], [326, 580], [322, 690], [318, 770]], 40]])
  for (const q of [p, mirror(p)]) hanging.push(drawLock(q, w, { tip: 0.6, fill: C.hairBack, shadeFill: C.hairBackShade }).svg);
const ahoge = drawLock([[402, 150], [408, 105], [430, 78], [458, 72], [474, 88]], 15, { tip: 0.35, line: 2.2 });

// ---------- background
if (!CHAR_ONLY) out.push(`<rect width="${W}" height="${H}" fill="url(#sky)"/>`);
const r1 = rng(5);
if (!CHAR_ONLY) for (let i = 0; i < 45; i++) out.push(sparkle(r1() * W, r1() * 620, 1.5 + r1() * 6, '#fff', `opacity="${fmt(0.35 + r1() * 0.65)}"`));
if (!CHAR_ONLY) out.push(`<circle cx="400" cy="340" r="340" fill="url(#moon)"/>`);

// ---------- back hair, with rim light from the moon behind her
out.push(`<g transform="${TILT}">`, el(BACK, C.hairBack));
out.push(el('M300,420 C300,560 292,700 282,1000 L518,1000 C508,700 500,560 500,420 Z', 'url(#backShadeGrad)'), ...hanging);
out.push(`<g clip-path="url(#backClip)"><path d="${BACK}" fill="none" stroke="#ffd9ee" stroke-width="12" filter="url(#blur3)" opacity="0.85"/></g>`);
out.push(`<path d="${BACK}" fill="none" stroke="${C.hairLine}" stroke-width="3"/>`, '</g>');

// ---------- neck and chest
const NECK_EDGE = [[358, 592], [356, 660], [352, 712]];
out.push(el(NECK, C.skin));
out.push(`<g clip-path="url(#neckClip)">${el('M340,560 L460,560 L458,636 Q426,668 400,670 Q370,664 342,620 Z', C.skinShade)}<g transform="translate(6,-12)">${el(COLLAR_L, C.skinShade)}${el(COLLAR_R, C.skinShade)}</g></g>`);
out.push(stroke(NECK_EDGE, ink(3, 0.3, 0.3), C.skinLine), stroke(mirror(NECK_EDGE), ink(3, 0.3, 0.3), C.skinLine));

// ---------- sailor uniform (shifted up 20px)
out.push('<g transform="translate(0,-20)">');
out.push(el(TORSO, C.blouse), crescent(TORSO, -30, -12, C.blouseShade));
out.push(`<g clip-path="url(#torsoClip)"><g transform="translate(7,10)">${el(COLLAR_L, C.blouseShade)}${el(COLLAR_R, C.blouseShade)}</g><g transform="translate(0,20) ${TILT} translate(7,10)">${sides.map(s => el(s.body, C.blouseShade)).join('')}</g></g>`);
for (const p of [[[172, 905], [196, 945], [206, 1000]], [[232, 880], [246, 930]]])
  out.push(stroke(p, ink(2.2), C.blouseLine, { extra: 'opacity="0.5"' }), stroke(mirror(p), ink(2.2), C.blouseLine, { extra: 'opacity="0.5"' }));
out.push(el(DICKEY, C.blouse), stroke([[368, 800], [432, 800]], ink(2.4, 0.1, 0.1), C.blouseLine));
const SHOULDER = [[348, 736], [300, 768], [205, 792], [150, 824], [106, 862], [86, 950], [76, 1040]];
out.push(stroke(SHOULDER, ink(4, 0.06, 0.04), C.blouseLine), stroke(mirror(SHOULDER), ink(4, 0.06, 0.04), C.blouseLine));
out.push(el(COLLAR_L, C.collar), el(COLLAR_R, C.collar), crescent(COLLAR_L + COLLAR_R, -8, -10, C.collarShade));
const STRIPE = [[400, 933], [205, 822]];
out.push(`<g clip-path="url(#collarClip)">${stroke(STRIPE, () => 5, '#fff')}${stroke(mirror(STRIPE), () => 5, '#fff')}</g>`);
for (const p of [[[400, 948], [176, 820]], [[348, 735], [400, 860]]])
  out.push(stroke(p, ink(3, 0.1, 0.1, 1), C.collarLine), stroke(mirror(p), ink(3, 0.1, 0.1, 1), C.collarLine));
// bow
const LOOPS = 'M394,930 C360,898 316,906 320,934 C324,962 366,958 394,944 Z M406,930 C440,898 484,906 480,934 C476,962 434,958 406,944 Z';
for (const p of [[[392, 946], [380, 975], [368, 1010]], [[408, 946], [422, 976], [436, 1010]]])
  out.push(drawLock(p, 28, { tip: 0.75, fill: C.ribbon, shadeFill: C.ribbonShade, line: 2.4, lineColor: C.ribbonLine }).svg);
out.push(el(LOOPS, C.ribbon), crescent(LOOPS, -6, -8, C.ribbonShade));
out.push(`<path d="${LOOPS}" fill="none" stroke="${C.ribbonLine}" stroke-width="2.5" stroke-linejoin="round"/>`);
const FOLD = [[392, 934], [362, 926], [338, 931]];
out.push(stroke(FOLD, ink(2), C.ribbonLine, { extra: 'opacity="0.6"' }), stroke(mirror(FOLD), ink(2), C.ribbonLine, { extra: 'opacity="0.6"' }));
out.push(`<rect x="384" y="922" width="32" height="28" rx="9" fill="${C.ribbon}" stroke="${C.ribbonLine}" stroke-width="2.5"/>`, '</g>');

// ---------- face
out.push(`<g transform="${TILT}">`);
out.push(el(FACE, 'url(#skinGrad)'), crescent(FACE, -14, -12, C.skinShade));
out.push(`<g clip-path="url(#faceClip)"><g transform="translate(5,14)">${el(CROWN, C.skinShade)}${bangs.map(b => el(b.body, C.skinShade)).join('')}</g><g transform="translate(8,6)">${sides.map(s => el(s.body, C.skinShade)).join('')}</g></g>`);
out.push(`<ellipse cx="292" cy="504" rx="48" ry="22" fill="url(#blush)"/><ellipse cx="508" cy="504" rx="48" ry="22" fill="url(#blush)"/>`);
for (const x of [278, 292, 306]) {
  const p = [[x + 5, 496], [x - 3, 510]];
  out.push(stroke(p, ink(2, 0.3, 0.3), '#e46f86', { extra: 'opacity="0.55"' }), stroke(mirror(p), ink(2, 0.3, 0.3), '#e46f86', { extra: 'opacity="0.55"' }));
}
out.push(eye(313, 436, 1, 'eyeL'), eye(487, 436, -1, 'eyeR'));
out.push(stroke([[404, 503], [401, 511], [397, 517]], ink(3, 0.3, 0.4), C.skinLine));
out.push(el('M380,551 Q400,558 420,551 Q414,572 400,573 Q386,572 380,551 Z', '#9c3b4a'));
out.push(el('M386,565 Q400,558 414,565 Q408,573 400,573 Q392,573 386,565 Z', '#ee8a92'));
out.push(stroke([[375, 548], [388, 553], [400, 555], [412, 553], [425, 548]], ink(3, 0.2, 0.2), '#7a2a35'));
const JAW = [[236, 320], [238, 440], [296, 533], [364, 598], [400, 609]];
out.push(stroke(JAW, ink(4.2, 0.3, 0.25), C.skinLine), stroke(mirror(JAW), ink(4.2, 0.3, 0.25), C.skinLine));

// ---------- front hair
out.push(...sides.map(s => s.svg), ahoge.svg);
out.push(el(CROWN, 'url(#crownGrad)'), crescent(CROWN, -12, -16, C.hairShade), `<g clip-path="url(#crownClip)">${shine}</g>`);
out.push(stroke([[206, 345], [203, 262], [238, 184], [318, 136], [400, 128], [482, 136], [562, 184], [597, 262], [594, 345]], ink(4.5, 0.04, 0.04, 1.5), C.hairLine));
for (const p of [[[396, 134], [350, 148], [300, 176]], [[392, 140], [340, 170], [296, 214]]])
  out.push(stroke(p, ink(2, 0.3, 0.3), C.hairShade, { extra: 'opacity="0.8"' }), stroke(mirror(p), ink(2, 0.3, 0.3), C.hairShade, { extra: 'opacity="0.8"' }));
out.push(...bangs.map(b => b.svg));
const BROW = [[354, 372], [326, 361], [296, 360], [270, 369]];
out.push(`<g opacity="0.75">${stroke(BROW, ink(5, 0.08, 0.6), C.hairLine)}${stroke(mirror(BROW), ink(5, 0.08, 0.6), C.hairLine)}</g>`, '</g>');

// ---------- lighting and foreground sparkle
if (!CHAR_ONLY) {
out.push(`<rect width="${W}" height="${H}" fill="url(#light)" style="mix-blend-mode:soft-light" opacity="0.55"/>`);
const r2 = rng(21);
for (let i = 0; i < 14; i++) {
  const x = r2() < 0.5 ? 20 + r2() * 150 : 630 + r2() * 150, y = 60 + r2() * 880;
  out.push(`<path d="M0,-9 C7,-6 7,5 0,9 C-7,5 -7,-6 0,-9Z" transform="translate(${fmt(x)},${fmt(y)}) rotate(${fmt(r2() * 360)})" fill="#ffc9d9" opacity="0.85"/>`);
}
for (let i = 0; i < 8; i++) out.push(sparkle(30 + r2() * 740, 40 + r2() * 300, 4 + r2() * 8, '#fff', 'opacity="0.9"'));

}
fs.writeFileSync(__dirname + '/05-original' + (DENSE ? '-dense' : '') + (CHAR_ONLY ? '-char' : '') + '.svg',
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs>${defs.join('')}</defs>\n${out.join('\n')}\n</svg>\n`);
