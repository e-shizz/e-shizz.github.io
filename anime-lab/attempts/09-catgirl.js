// Attempt 09: master study, round 4: hair as one soft mass with fine strand lines (not ribbons).
// Landmarks were measured off a grid laid over the reference; the drawing itself is the brush engine.
// Run: node attempts/09-catgirl.js  (also writes 09-spines.json; pick.py turns that into 09-colors.json)
//      node render.js attempts/09-catgirl.svg attempts/09-catgirl.png
const fs = require('fs');
const B = require('../engine/brush');
const { el, stroke, lock, ink, pw, rng, fmt, spline } = B;

const W = 857, H = 1200;
const defs = [], out = [], overHair = [];
const C = {
  line: '#24161f', hairLine: '#1f141d', redLid: '#8a3a40',
  skin: '#f4cbd0', skinLight: '#fdeae8', skinShade: '#e4aab3', skinDeep: '#cd8e98', skinLine: '#4a2a32',
  shirt: '#f3bccf', shirtLight: '#fbdbe6', shirtShade: '#dc97b3', shirtDeep: '#b8739a', shirtLine: '#7a3f62',
};

// ---------- shared shapes
const FACE = 'M252,170 C236,250 228,330 236,410 C242,440 252,462 268,478 C284,503 300,525 316,541 C335,556 350,565 366,571 C384,580 400,588 412,593 C422,597 432,600 440,601 C455,598 468,593 480,588 C492,580 502,573 511,566 C518,557 525,549 531,541 C538,532 545,523 551,514 C562,494 574,460 582,425 C592,370 596,300 590,200 C560,160 500,145 430,142 C350,142 290,150 252,170 Z';
const NECK = 'M376,560 L510,548 L506,630 L512,680 L562,668 L440,740 L300,662 L370,676 L384,632 Z';
const SHIRT = 'M300,662 C350,700 400,726 440,728 C480,726 520,700 560,670 C600,650 660,660 705,680 C760,705 800,745 814,800 C822,860 824,950 826,1200 L50,1200 C52,1100 54,1000 58,880 C62,800 72,740 98,694 C120,672 170,662 230,656 C260,656 285,658 300,662 Z';
const CHOKER = 'M372,582 C392,596 418,608 440,611 C462,609 488,597 512,580 L507,623 C488,636 464,646 440,647 C416,646 394,638 380,630 Z';
const BACKHAIR = 'M140,0 L705,0 L692,110 L716,200 L706,262 L732,352 L762,452 L800,560 L838,648 L857,700 L857,1200 L0,1200 L0,660 L32,580 L60,470 L88,350 L118,262 L138,160 Z';
const CAT = 'M296,1200 L296,1040 C296,1000 305,970 318,948 L350,985 C380,976 420,972 450,972 C482,972 512,978 528,986 L548,932 C560,960 566,1000 566,1040 L566,1200 Z';

defs.push(`
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#c3bdc9"/><stop offset="0.55" stop-color="#d8cdd7"/><stop offset="1" stop-color="#f0dde4"/></linearGradient>
<radialGradient id="glow" cx="790" cy="300" r="420" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#fff8fa" stop-opacity="0.95"/><stop offset="1" stop-color="#fff8fa" stop-opacity="0"/></radialGradient>
<linearGradient id="hair" gradientUnits="userSpaceOnUse" x1="120" y1="0" x2="780" y2="0"><stop offset="0" stop-color="#3d2539"/><stop offset="0.4" stop-color="#553750"/><stop offset="0.7" stop-color="#6e4d5f"/><stop offset="0.85" stop-color="#a07c88"/><stop offset="1" stop-color="#cfa5ae"/></linearGradient>
<linearGradient id="hairShade" gradientUnits="userSpaceOnUse" x1="120" y1="0" x2="760" y2="0"><stop offset="0" stop-color="#271a25"/><stop offset="0.3" stop-color="#352432"/><stop offset="0.5" stop-color="#523a49"/><stop offset="0.7" stop-color="#7b5b6a"/><stop offset="1" stop-color="#aa8390"/></linearGradient>
<linearGradient id="shirtGrad" gradientUnits="userSpaceOnUse" x1="50" y1="0" x2="830" y2="0"><stop offset="0" stop-color="#ad7fa9"/><stop offset="0.12" stop-color="#c47a9a"/><stop offset="0.4" stop-color="#e0a2bb"/><stop offset="0.62" stop-color="#f3c0d2"/><stop offset="1" stop-color="#fed2e1"/></linearGradient>
<radialGradient id="blush"><stop offset="0" stop-color="#f17f92" stop-opacity="0.8"/><stop offset="0.6" stop-color="#f17f92" stop-opacity="0.55"/><stop offset="1" stop-color="#f17f92" stop-opacity="0"/></radialGradient>
<radialGradient id="cheekLight"><stop offset="0" stop-color="#fff4f4" stop-opacity="0.9"/><stop offset="1" stop-color="#fff4f4" stop-opacity="0"/></radialGradient>
<linearGradient id="light" x1="0" y1="0" x2="1" y2="0.25"><stop offset="0" stop-color="#2b1f45" stop-opacity="0.55"/><stop offset="0.55" stop-color="#ffffff" stop-opacity="0"/><stop offset="1" stop-color="#fff3f6" stop-opacity="0.9"/></linearGradient>
<filter id="blur2" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2"/></filter>
<filter id="blur8" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="8"/></filter>
<clipPath id="faceClip"><path d="${FACE}"/></clipPath>
<clipPath id="neckClip"><path d="${NECK}"/></clipPath>
<clipPath id="shirtClip"><path d="${SHIRT}"/></clipPath>
<clipPath id="catClip"><path d="${CAT}"/></clipPath>`);

let maskN = 0;
function crescent(d, dx, dy, fill, extra = '') {
  const id = 'm' + maskN++;
  defs.push(`<mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="${W}" height="${H}"><rect width="${W}" height="${H}" fill="#fff"/><path d="${d}" transform="translate(${dx},${dy})" fill="#000"/></mask>`);
  return `<path d="${d}" fill="${fill}" mask="url(#${id})" ${extra}/>`;
}

// Hair lock lit from the right: the shaded band runs down its left edge.
let gradN = 0;
function hairLock(spine, w, o = {}) {
  const L = lock(spine, w, { shade: 'left', line: 2, ...o });
  let fill = o.fill || 'url(#hair)';
  if (Array.isArray(fill)) {
    const id = 'hg' + gradN++, a = spine[0], b = spine[spine.length - 1];
    defs.push(`<linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"><stop offset="0.1" stop-color="${fill[0]}"/><stop offset="0.5" stop-color="${fill[1]}"/><stop offset="0.9" stop-color="${fill[2]}"/></linearGradient>`);
    fill = `url(#${id})`;
  }
  let s = '<g>' + el(L.body, fill) + el(L.shade, o.shadeFill || 'url(#hairShade)', o.shadeOpacity ? `opacity="${o.shadeOpacity}"` : '');
  if (o.strands) s += strandLines(spine, w, o.strands, o.strandColor || '#7a5d72', o.seed || 1);
  s += L.edges.map(d => el(d, o.lineColor || C.hairLine)).join('') + '</g>';
  return { svg: s, body: L.body };
}

// Every eyedropped lock registers its spine; pick.py samples the reference along it.
const spinesOut = {};
let picked = {};
try { picked = JSON.parse(fs.readFileSync(__dirname + '/09-colors.json', 'utf8')); } catch (e) { /* first run: no colours yet */ }
function pickLock(id, spine, w, o = {}) {
  spinesOut[id] = { spine: spline(spine, 8), w };
  const c = picked[id];
  return hairLock(spine, w, c ? { shadeK: 0.22, line: 1.5, ...o, fill: c.fill, shadeFill: c.shade, strandColor: c.light, shadeOpacity: 0.5, strands: (o.strands || 0) * 2 + 3 } : o);
}

// Thin texture lines running along a lock.
function strandLines(spine, w, n, color, seed) {
  const r = rng(seed), c = spline(spine, 12);
  let s = '';
  for (let k = 0; k < n; k++) {
    const col = k % 3 === 2 ? '#241521' : color;
    const f = (r() - 0.5) * 0.55 * w;
    const a = Math.floor(r() * c.length * 0.25), b = Math.floor(c.length * (0.55 + r() * 0.2));
    const pts = c.slice(a, b).map(([x, y]) => [x + f, y]);
    if (pts.length > 3) s += stroke(pts, ink(0.8 + r() * 1.1, 0.3, 0.3), col, { segs: 1, extra: 'opacity="0.75"' });
  }
  return s;
}

function eye(o) {
  const open = B.path(spline(o.open.concat([o.open[0]]), 8));
  defs.push(`<clipPath id="${o.id}"><path d="${open}"/></clipPath>`,
    `<linearGradient id="${o.id}-iris" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#271c4a"/><stop offset="0.42" stop-color="#583b98"/><stop offset="0.72" stop-color="#a77de2"/><stop offset="1" stop-color="#f2c6f6"/></linearGradient>`,
    `<linearGradient id="${o.id}-lid" gradientUnits="userSpaceOnUse" x1="0" y1="${o.lidTop}" x2="0" y2="${o.lidTop + 20}"><stop offset="0" stop-color="#2a1d45" stop-opacity="0.75"/><stop offset="1" stop-color="#2a1d45" stop-opacity="0"/></linearGradient>`);
  const [ix, iy] = o.iris, [rx, ry] = o.r;
  let g = `<g clip-path="url(#${o.id})">` + el(open, '#e4e2ee') + el(o.scleraShade, '#b7b3c8');
  g += `<ellipse cx="${ix}" cy="${iy}" rx="${rx}" ry="${ry}" fill="url(#${o.id}-iris)" stroke="#1b1431" stroke-width="2.2"/>`;
  // crystalline lower half: lavender pool, pink flecks, lighter facets
  g += `<ellipse cx="${ix + 1}" cy="${fmt(iy + ry * 0.48)}" rx="${fmt(rx * 0.72)}" ry="${fmt(ry * 0.38)}" fill="#d9aef6" opacity="0.85"/>`;
  for (const [dx, dy, r, c] of [[-0.35, 0.55, 0.16, '#f7cdf2'], [0.3, 0.6, 0.13, '#f9d6f6'], [0.05, 0.75, 0.12, '#ffffff'], [-0.55, 0.2, 0.1, '#c79cf0'], [0.55, 0.15, 0.1, '#b48ae8']])
    g += `<circle cx="${fmt(ix + dx * rx)}" cy="${fmt(iy + dy * ry)}" r="${fmt(r * rx)}" fill="${c}" opacity="0.9"/>`;
  for (let a = 200; a <= 340; a += 28) {
    const t = a * Math.PI / 180;
    g += stroke([[ix + Math.cos(t) * rx * 0.3, iy + Math.sin(t) * -ry * 0.3], [ix + Math.cos(t) * rx * 0.85, iy + Math.sin(t) * -ry * 0.85]], ink(2, 0.3, 0.3), '#c49af0', { extra: 'opacity="0.6"' });
  }
  g += `<ellipse cx="${ix - 3}" cy="${iy - 4}" rx="${fmt(rx * 0.24)}" ry="${fmt(ry * 0.33)}" fill="#3d2c70" stroke="#120b22" stroke-width="3"/>`;
  g += el(open, `url(#${o.id}-lid)`) + '</g>';
  g += stroke(o.lash, o.lashW, C.line);
  for (const e of o.extra) g += stroke(e[0], e[1], e[2] || C.line);
  for (const [x, y, r] of o.hl) g += `<circle cx="${x}" cy="${y}" r="${r}" fill="#fff"/>`;
  for (const [x, y, rx2, ry2, c] of o.blobs) g += `<ellipse cx="${x}" cy="${y}" rx="${rx2}" ry="${ry2}" fill="${c}" opacity="0.92"/>`;
  return g;
}

// ---------- background: cool wall on the left, glowing window on the right
out.push(`<rect width="${W}" height="${H}" fill="url(#bg)"/>`);
out.push(`<rect x="700" y="30" width="170" height="640" fill="#f3dfe6" filter="url(#blur8)"/>`);
for (const [x, y, w, h] of [[726, 66, 34, 122], [806, 66, 60, 122], [726, 226, 34, 215], [806, 226, 60, 215], [726, 478, 34, 150], [806, 478, 60, 170]])
  out.push(`<rect x="${x - 6}" y="${y - 6}" width="${w + 12}" height="${h + 12}" rx="14" fill="#f8e6ec"/><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#fffbfc"/>`);
out.push(`<rect width="${W}" height="${H}" fill="url(#glow)"/>`);
const rp = rng(4);
for (let i = 0; i < 26; i++) {
  const left = i < 18, x = left ? 15 + rp() * 90 : 640 + rp() * 200, y = 140 + rp() * (left ? 340 : 600);
  out.push(`<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${fmt(1 + rp() * 2.2)}" fill="#fff" opacity="${fmt(0.5 + rp() * 0.5)}"/>`);
}

// ---------- back hair and ears
out.push(el(BACKHAIR, '#3a2537'));
[
  [[[120, 300], [90, 480], [50, 640], [20, 800], [10, 1000]], 60],
  [[[110, 330], [80, 450], [50, 560], [25, 660], [10, 760]], 30],
  [[[132, 360], [106, 470], [80, 580], [50, 680], [30, 780]], 22],
  [[[730, 380], [770, 520], [820, 680], [845, 900], [850, 1150]], 70],
  [[[720, 300], [745, 420], [770, 520], [805, 620], [840, 720]], 50],
  [[[760, 500], [800, 600], [835, 700], [857, 800]], 40],
  [[[700, 250], [715, 330], [735, 420]], 40],
].forEach(([p, w], i) => out.push(pickLock('b' + i, p, w, { tip: 0.85, fill: '#3b2a37', shadeFill: '#2a1c27', strands: 4, strandColor: '#6d5262', seed: 50 + i }).svg));
// left ear
const LEAR = 'M5,25 C40,20 120,12 232,20 L250,140 L162,236 C120,190 60,110 5,25 Z';
out.push(el(LEAR, '#533548'), el('M30,44 C80,40 150,38 212,52 L196,150 L150,212 C110,160 70,100 30,44 Z', '#958399'));
out.push(el('M60,70 L115,92 L92,100 L140,122 L104,128 L150,160 L112,156 L134,196 L98,160 L96,184 L78,140 L58,112 L74,108 Z', '#b3a4b8'));
out.push(el('M150,60 L200,64 L184,78 L206,96 L178,96 L190,122 L162,104 L158,126 L146,96 Z', '#a594aa'));
out.push(el('M40,24 C110,16 180,16 230,22 L222,56 C160,42 100,38 40,30 Z', '#c191a1'));
out.push(stroke([[5, 25], [60, 108], [115, 188], [162, 236]], ink(5, 0.05, 0.2), C.hairLine), stroke([[5, 25], [80, 17], [160, 14], [234, 20]], ink(4, 0.05, 0.1), C.hairLine));
// right ear
const REAR = 'M556,52 C568,30 585,10 598,0 L700,0 C698,40 688,90 672,130 C662,155 650,170 640,180 L600,175 Z';
out.push(el(REAR, '#6f5b74'), el('M588,30 L612,40 L600,58 L620,70 L606,84 L626,96 L610,108 L630,130 L600,140 L585,100 L578,60 Z', '#a99db5'));
out.push(el('M602,8 L690,6 C686,50 676,90 662,118 L640,140 L628,110 L610,120 L612,90 L596,96 L604,64 L588,66 Z', '#fdf9f4'));
out.push(el('M640,20 L668,60 L650,58 L664,92 L642,74', '#d9d0e0'));
out.push(stroke([[700, 0], [690, 60], [672, 128], [640, 180]], ink(5, 0.05, 0.2), C.hairLine), stroke([[556, 52], [576, 22], [598, 0]], ink(4, 0.1, 0.1), C.hairLine));

// ---------- neck, choker, shirt
out.push(el(NECK, C.skin));
out.push(`<g clip-path="url(#neckClip)">${el('M370,560 L515,545 L512,585 L476,602 L440,612 L408,602 L374,585 Z', C.skinDeep)}${el('M380,628 L410,642 L440,648 L476,640 L508,622 L510,664 L476,678 L440,684 L404,678 L382,664 Z', C.skinShade)}${el('M376,560 L394,560 L400,640 L394,700 L372,680 Z', C.skinDeep)}</g>`);
out.push(stroke([[388, 650], [385, 690]], ink(2), C.skinLine, { extra: 'opacity="0.5"' }), stroke([[498, 642], [503, 680]], ink(2), C.skinLine, { extra: 'opacity="0.5"' }));
out.push(stroke([[394, 712], [426, 724]], ink(1.8), C.skinLine, { extra: 'opacity="0.45"' }), stroke([[458, 724], [490, 712]], ink(1.8), C.skinLine, { extra: 'opacity="0.45"' }));
out.push(el(CHOKER, '#221b26'), el('M376,590 C394,600 414,608 434,612 L434,620 C412,616 392,608 378,600 Z', '#7c7682', 'opacity="0.9"'));
out.push(stroke([[374, 584], [404, 600], [440, 611], [476, 602], [510, 582]], ink(1.6, 0.1, 0.1), '#a39dab', { extra: 'opacity="0.8"' }));

out.push(el(SHIRT, 'url(#shirtGrad)'));
const shirtBits = [
  ['M58,880 C62,800 72,740 98,694 C120,672 170,662 230,656 L300,662 L300,1200 L50,1200 C52,1100 54,1000 58,880 Z', '#b57aa2', 0.25],
  ['M240,1020 L300,1020 L300,1200 L230,1200 Z', '#a46a92', 0.5], ['M560,1060 L640,1040 L650,1200 L560,1200 Z', '#c27ea0', 0.4],
  ['M330,760 L520,742 L560,900 L320,920 Z', '#f8d1de', 0.7],
];
const folds = [
  // [points, peak width, colour, opacity]
  [[[800, 780], [750, 830], [690, 890]], 12, '#cf89a9', 0.85], [[[810, 840], [740, 920], [660, 1000]], 10, '#cf89a9', 0.85],
  [[[820, 920], [760, 1000], [700, 1070]], 10, '#cf89a9', 0.8], [[[700, 1000], [660, 1100], [640, 1200]], 16, '#cd86a6', 0.8],
  [[[760, 1060], [720, 1130], [700, 1200]], 14, '#cd86a6', 0.8], [[[690, 700], [660, 740], [630, 770]], 14, '#d08bab', 0.8],
  [[[620, 700], [660, 720], [700, 760]], 14, '#fde4ec', 0.85], [[[720, 840], [740, 900], [745, 960]], 10, '#fde4ec', 0.8],
  [[[600, 980], [625, 1060], [610, 1150]], 12, '#fde4ec', 0.7],
  [[[70, 880], [95, 1000], [130, 1180]], 16, '#e7b3ca', 0.7], [[[110, 940], [150, 1060], [200, 1180]], 14, '#e7b3ca', 0.7],
  [[[140, 1000], [185, 1090], [230, 1170]], 12, '#e7b3ca', 0.65], [[[90, 720], [120, 800], [150, 880]], 14, '#e7b3ca', 0.7],
  [[[100, 770], [175, 850], [250, 920]], 8, '#a66d95', 0.55], [[[240, 1020], [270, 1100], [290, 1200]], 18, '#a66d95', 0.55],
  [[[320, 702], [380, 742], [440, 762], [500, 742], [560, 702]], 18, '#d993b0', 0.55], [[[440, 762], [445, 830], [440, 900]], 10, '#e3a3bd', 0.6],
];
out.push(`<g clip-path="url(#shirtClip)">${shirtBits.map(([d, c, a]) => el(d, c, `opacity="${a}"`)).join('')}${folds.map(([p, w, c, a]) => stroke(p, pw([[0, 1], [0.45, w], [1, 0.5]]), c, { extra: `opacity="${a}"` })).join('')}${stroke([[98, 694], [74, 760], [60, 860], [55, 1010], [52, 1200]], () => 12, '#cdb9e2', { extra: 'opacity="0.8"' })}</g>`);
// cat print
defs.push('<linearGradient id="catShade" x1="0" y1="0" x2="0" y2="1"><stop offset="0.3" stop-color="#c6aecb" stop-opacity="0"/><stop offset="0.55" stop-color="#c6aecb" stop-opacity="0.9"/></linearGradient>');
out.push(el(CAT, '#efdae7'), el(CAT, 'url(#catShade)'), `<g clip-path="url(#catClip)">${stroke([[340, 960], [360, 1040], [350, 1120]], pw([[0, 1], [0.5, 12], [1, 1]]), '#e9dcea')}${stroke([[480, 1000], [470, 1080], [480, 1160]], pw([[0, 1], [0.5, 16], [1, 1]]), '#b89ab9', { extra: 'opacity="0.6"' })}</g>`);
out.push(el('M300,1010 L318,940 L348,1000 Z', '#fbf2f8'), el('M522,1000 L548,932 L560,1012 Z', '#fbf2f8'), stroke([[422, 966], [419, 1020], [414, 1085]], pw([[0, 2], [0.4, 18], [1, 1]]), '#ffffff', { extra: 'opacity="0.7"' }));
out.push(el('M340,1072 L398,1086 L386,1099 L348,1091 Z', '#c06f9e'), el('M452,1086 L506,1068 L501,1089 L462,1099 Z', '#c06f9e'));
// neckline rib
out.push(el('M300,662 C350,700 400,726 440,728 C480,726 520,700 560,670 L566,690 C524,722 482,748 440,752 C398,748 352,724 304,690 Z', '#e7a6bf'));
out.push(stroke([[300, 662], [350, 700], [400, 726], [440, 728], [480, 726], [520, 700], [560, 670]], ink(2.4, 0.05, 0.05), C.shirtLine));
out.push(stroke([[304, 690], [352, 724], [398, 748], [440, 752], [482, 748], [524, 722], [566, 690]], ink(2, 0.05, 0.05), C.shirtLine, { extra: 'opacity="0.7"' }));
out.push(stroke([[98, 694], [74, 760], [60, 860], [55, 1010], [51, 1200]], ink(3, 0.02, 0.02), C.shirtLine), stroke([[705, 680], [770, 712], [808, 770], [820, 860], [824, 1000], [826, 1200]], ink(3, 0.02, 0.02), C.shirtLine));

// ---------- face
const bangDefs = [
  [[[255, 0], [250, 100], [255, 200], [266, 290], [280, 340], [290, 368]], 52, 0.55, 3],
  [[[305, 0], [300, 100], [305, 200], [318, 280], [326, 320], [332, 336]], 50, 0.55, 3],
  [[[338, 40], [340, 140], [348, 230], [356, 300], [354, 340], [348, 358]], 34, 0.5, 2],
  [[[372, 0], [378, 100], [384, 200], [388, 280], [386, 340], [384, 366]], 44, 0.5, 2],
  [[[440, 0], [452, 90], [458, 165], [445, 250], [425, 310], [415, 345]], 44, 0.5, 2],
  [[[490, 0], [500, 80], [508, 165], [530, 250], [548, 310], [556, 345]], 50, 0.5, 2],
  [[[545, 0], [565, 80], [585, 180], [598, 260], [605, 330]], 48, 0.5, 2],
];
const thinBangs = [
  [[[520, 220], [490, 280], [462, 320], [440, 346]], 8],
  [[[478, 200], [466, 260], [452, 310], [446, 345]], 6],
  [[[250, 300], [285, 330], [320, 355], [350, 378]], 7],
  [[[230, 320], [265, 345], [300, 365], [335, 385]], 6],
];
const bangs = bangDefs.map(([p, w, tip, n], i) => pickLock('g' + i, p, w, { tip, strands: n, seed: 10 + i, strandColor: '#8f7ca6' }));
const thin = thinBangs.map(([p, w]) => hairLock(p, w, { tip: 0.4, line: 1.4, fill: '#5a3a4f', shadeFill: '#3a2534' }));

out.push(el(FACE, C.skin), crescent(FACE, 16, -10, C.skinShade));
out.push(`<g clip-path="url(#faceClip)"><g transform="translate(-8,12)">${bangs.concat(thin).map(b => el(b.body, C.skinShade)).join('')}</g>${el('M236,300 C262,340 270,380 262,430 C258,460 262,476 276,494 L262,478 C246,456 236,420 234,380 Z', C.skinDeep)}</g>`);
out.push(`<ellipse cx="505" cy="470" rx="70" ry="80" fill="url(#cheekLight)"/>`);
out.push(`<ellipse cx="305" cy="465" rx="38" ry="23" fill="url(#blush)"/><ellipse cx="501" cy="431" rx="41" ry="23" fill="url(#blush)" transform="rotate(-10 501 431)"/>`);

out.push(eye({
  id: 'eyeL', iris: [318, 415], r: [27, 29], lidTop: 384,
  open: [[258, 391], [276, 387], [294, 385], [312, 386], [328, 390], [340, 396], [349, 405], [343, 424], [333, 438], [318, 446], [300, 445], [283, 440], [269, 429], [260, 412]],
  scleraShade: 'M250,380 L352,380 L352,402 C330,398 300,396 270,404 L250,410 Z',
  lash: [[244, 392], [256, 385], [274, 380], [294, 377.5], [314, 378.5], [330, 383], [342, 391], [351, 402]],
  lashW: pw([[0, 5], [0.35, 13], [0.75, 9], [1, 2.5]]),
  extra: [
    [[[252, 386], [240, 380], [230, 378]], pw([[0, 4], [1, 0]])],
    [[[250, 392], [238, 392], [228, 395]], pw([[0, 3.5], [1, 0]])],
    [[[236, 402], [250, 414], [266, 428], [284, 442]], pw([[0, 1], [0.3, 3.5], [1, 1]]), C.redLid],
    [[[258, 393], [272, 389], [290, 387]], pw([[0, 1], [0.5, 3], [1, 0.5]]), C.redLid],
    [[[284, 444], [300, 448], [318, 448], [334, 441], [344, 428]], pw([[0, 0.5], [0.4, 2.6], [1, 0.5]])],
    [[[349, 405], [352, 412]], pw([[0, 2], [1, 0.5]])],
  ],
  hl: [[312, 392, 3.6], [321, 394.5, 2.3], [303, 404, 1.8], [334, 404, 1.6], [299, 428, 1.5]],
  blobs: [[302, 417, 5, 4, '#ffffff'], [322, 437, 7, 3, '#fbe3ff']],
}));
out.push(eye({
  id: 'eyeR', iris: [513, 378], r: [31, 33], lidTop: 344,
  open: [[484, 377], [488, 360], [496, 349], [510, 345], [530, 343.5], [550, 344.5], [565, 348], [567, 362], [561, 379], [551, 394], [536, 405], [517, 410], [501, 406], [491, 394]],
  scleraShade: 'M480,335 L572,335 L572,352 C550,350 520,350 490,356 L480,360 Z',
  lash: [[481, 357], [488, 348], [500, 342], [520, 338.5], [542, 338], [560, 340], [574, 344]],
  lashW: pw([[0, 3], [0.3, 10], [0.8, 9], [1, 5]]),
  extra: [
    [[[544, 336], [548, 327], [551, 320]], pw([[0, 3], [1, 0]])],
    [[[552, 337], [558, 328], [563, 322]], pw([[0, 3], [1, 0]])],
    [[[485, 366], [474, 376], [463, 386]], pw([[0, 2.5], [1, 0]])],
    [[[486, 372], [477, 384], [468, 394]], pw([[0, 2.5], [1, 0]])],
    [[[489, 380], [482, 391], [476, 399]], pw([[0, 2.2], [1, 0]])],
    [[[497, 404], [512, 411], [528, 409], [541, 403]], pw([[0, 0.5], [0.5, 2.4], [1, 0.5]])],
    [[[503, 411], [520, 414], [537, 410]], pw([[0, 0.5], [0.5, 2], [1, 0.5]]), C.redLid],
    [[[546, 397], [550, 393]], () => 2], [[[552, 389], [556, 385]], () => 2], [[[540, 403], [544, 400]], () => 2],
  ],
  hl: [[512, 354, 4.4], [504.5, 357, 2.2], [495, 361.5, 2], [535, 367, 4], [512, 372, 2], [528, 355, 1.8]],
  blobs: [[518, 399, 7, 5, '#fff3ff'], [497, 375, 3, 4, '#ffffff']],
}));
// the red outer wing of the right eye sits on top of the hair
overHair.push(stroke([[556, 342], [572, 345], [584, 348], [594, 352]], pw([[0, 6], [1, 0.5]]), '#7a2f38'));

out.push(el('M406,447 C405,438 418,436 424,441 C430,436 446,437 447,446 C447,458 436,467 424,468 C412,467 406,457 406,447 Z', '#fffafb'));
out.push(stroke([[412, 474], [412, 481]], pw([[0, 1], [0.5, 2.2], [1, 1]]), '#6a3a44'));
out.push(stroke([[404, 524], [410, 528], [416, 529], [422, 526], [427.5, 522], [433, 526], [438, 529], [444, 526], [450, 521]], ink(2.4, 0.1, 0.1, 0.8), '#4a2a32'));
out.push(stroke([[404, 519], [405, 527]], () => 1.8, '#4a2a32'), stroke([[450, 517], [451, 525]], () => 1.8, '#4a2a32'));
out.push(stroke([[268, 478], [284, 503], [316, 541], [366, 571], [412, 593], [440, 601]], pw([[0, 1], [0.5, 2.5], [1, 3.5]]), C.skinLine));
out.push(stroke([[440, 601], [480, 588], [511, 566], [531, 541], [551, 514]], pw([[0, 3.5], [1, 1]]), C.skinLine));

// ---------- front hair: side locks and the long wavy hair over the shirt
const lit = '#f3d6de', dark = '#8f7ca6';
const front = [
  // [spine, width, tip, strand lines]
  [[[160, 200], [140, 300], [115, 400], [95, 500], [80, 600]], 46, 0.7, 4],
  [[[180, 380], [160, 480], [140, 580], [120, 680], [110, 760]], 40, 0.7, 3],
  [[[212, 0], [203, 100], [193, 220], [188, 330], [190, 430], [196, 520]], 64, 0.6, 5],
  [[[238, 130], [224, 230], [214, 330], [216, 420], [232, 490], [250, 560]], 38, 0.6, 2],
  [[[160, 600], [150, 700], [150, 800], [170, 880], [180, 960], [165, 1040]], 40, 0.7, 3],
  [[[190, 540], [200, 640], [225, 730], [258, 795], [272, 848], [250, 888], [208, 912], [188, 958], [202, 1000], [242, 1024]], 36, 0.7, 3],
  [[[230, 560], [250, 650], [284, 738], [312, 806], [318, 858], [290, 900], [244, 928], [218, 972], [236, 1018], [298, 1058]], 40, 0.7, 3],
  [[[270, 620], [300, 700], [330, 770], [345, 830]], 14, 0.5, 0],
  [[[210, 640], [240, 720], [280, 790]], 12, 0.5, 0],
  [[[662, 180], [690, 300], [712, 410], [735, 520], [762, 620], [790, 720]], 64, 0.8, 5],
  [[[650, 300], [668, 420], [676, 520], [668, 620], [660, 720], [675, 800]], 34, 0.7, 3],
  [[[625, 220], [640, 320], [644, 420], [634, 520], [618, 610], [605, 690]], 46, 0.8, 4],
  [[[590, 200], [598, 300], [592, 400], [580, 480], [568, 560], [562, 640]], 50, 0.85, 4],
  [[[565, 290], [574, 380], [566, 460], [550, 530], [538, 585]], 20, 0.6, 0],
  [[[565, 600], [555, 680], [560, 760], [585, 830], [600, 880], [585, 930], [555, 960], [548, 1000], [570, 1030], [610, 1035]], 40, 0.7, 3],
  [[[600, 620], [600, 700], [620, 780], [650, 850], [670, 910], [668, 960]], 34, 0.7, 2],
  [[[630, 640], [650, 720], [680, 800], [700, 870]], 22, 0.6, 0],
  [[[540, 640], [530, 720], [530, 800], [545, 870]], 18, 0.6, 0],
].map(([p, w, tip, n], i) => pickLock('f' + i, p, w, { tip, strands: n, seed: 30 + i, strandColor: p[0][0] > 450 ? '#f3d6de' : '#8f7ca6' }));
// cast shadows of the front hair on the face and shirt
out.push(`<g clip-path="url(#faceClip)"><g transform="translate(-10,8)">${front.map(f => el(f.body, C.skinDeep)).join('')}</g></g>`);
out.push(`<g clip-path="url(#shirtClip)"><g transform="translate(-12,14)">${front.map(f => el(f.body, C.shirtDeep, 'opacity="0.55"')).join('')}</g></g>`);
out.push(hairLock([[140, 190], [132, 230], [122, 275]], 26, { tip: 0.6, fill: '#9668b8', shadeFill: '#6e4a92' }).svg);
for (const [p, c] of [[[[150, 272], [110, 288], [70, 302], [40, 318]], '#b07ab8'], [[[152, 282], [118, 300], [86, 316], [60, 330]], '#8a5a96'], [[[148, 290], [120, 306], [100, 322]], '#3a2836']])
  out.push(stroke(p, pw([[0, 3], [1, 0]]), c));
out.push(...front.map(f => f.svg));
// pink streak by the right ear
out.push(stroke([[655, 100], [690, 150], [730, 190], [768, 208]], pw([[0, 16], [0.7, 10], [1, 0]]), '#f09cc0'));
out.push(stroke([[662, 112], [700, 160], [740, 195]], pw([[0, 6], [1, 0]]), '#fde0ea'));
out.push(stroke([[650, 95], [700, 140], [745, 180], [772, 206]], pw([[0, 2.5], [1, 0.5]]), C.hairLine));

// ---------- bangs, brows, clips
out.push(...bangs.map(b => b.svg), ...thin.map(b => b.svg));
out.push(stroke([[312, 347], [330, 354], [348, 362], [362, 371]], pw([[0, 0.5], [0.4, 3], [0.85, 3.5], [1, 1]]), '#3b2430', { extra: 'opacity="0.75"' }));
out.push(stroke([[494, 328], [476, 333], [458, 338], [440, 346]], pw([[0, 0.5], [0.4, 3], [0.85, 3.5], [1, 1]]), '#3b2430', { extra: 'opacity="0.75"' }));
out.push(...overHair);
// X clip
const bar = (x1, y1, x2, y2) => stroke([[x1, y1], [x2, y2]], () => 13, '#2a2235') + stroke([[x1, y1], [x2, y2]], () => 9, '#aabbe0') + stroke([[x1 + 2, y1], [x2 - 4, y2 - 3]], () => 2.5, '#eef3ff');
out.push(bar(163, 293, 218, 340), bar(220, 297, 161, 337));
out.push(`<rect x="162" y="357" width="61" height="16" rx="7" fill="#aabbe0" stroke="#2a2235" stroke-width="2.5"/><rect x="168" y="360" width="46" height="3" rx="1.5" fill="#eef3ff"/>`);
// cat clip
out.push(`<g transform="rotate(4 592 260)">
<rect x="567" y="265" width="56" height="15" rx="6" fill="#f2a3c3" stroke="#2a2235" stroke-width="2.5"/><rect x="574" y="268" width="40" height="3" rx="1.5" fill="#ffd9e8"/>
<rect x="571" y="286" width="57" height="17" rx="7" fill="#f2a3c3" stroke="#2a2235" stroke-width="2.5"/><rect x="578" y="289" width="42" height="3.5" rx="1.5" fill="#fff0f6"/>
<path d="M565,262 C562,248 562,236 566,226 L569,213 L581,222 C588,220 594,220 600,222 L607,210 L613,224 C618,236 619,250 617,262 C600,266 582,266 565,262 Z" fill="#2b2230" stroke="#fff" stroke-width="2.5"/>
<path d="M565,262 C562,248 562,236 566,226 L569,213 L581,222 C588,220 594,220 600,222 L607,210 L613,224 C618,236 619,250 617,262" fill="none" stroke="#2a2235" stroke-width="1.2"/>
<circle cx="582" cy="244" r="2.2" fill="#fff"/><circle cx="600" cy="243" r="2.2" fill="#fff"/></g>`);

// ---------- light from the window
out.push(`<rect width="${W}" height="${H}" fill="url(#light)" style="mix-blend-mode:soft-light" opacity="0.6"/>`);

fs.writeFileSync(__dirname + '/09-spines.json', JSON.stringify(spinesOut));
fs.writeFileSync(__dirname + '/09-catgirl.svg',
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs>${defs.join('')}</defs>\n${out.join('\n')}\n</svg>\n`);
