// Original girl #2, painted from scratch with no reference.
// Applies what the master study of the model's image taught: natural proportions, big irises with
// little white, bangs that fall over the eyes, a rim light on the lit side, soft hair gradients and
// split tips. Golden-hour light comes from the right.
// Run: node canvas/run.js canvas/girl2.js canvas/out/girl2.png
function draw() {
  init(800, 1000);
  const rnd = rng(7);
  const C = {
    hair: '#8a5236', hairMid: '#b0703f', hairLight: '#e3a468', hairShine: '#ffdcaa', hairShade: '#57301f', hairDeep: '#3a2028', hairLine: '#2e1a1c',
    skin: '#fde2d4', skinLight: '#fff2e8', skinShade: '#efb7a6', skinLine: '#8a4a44', blush: '#f59a9a', rim: '#fff1d6',
    irisTop: '#0f4a3e', iris: '#1e8a6a', irisLow: '#6fdba0', irisGlow: '#c8ffd8', pupil: '#0a2a24', lash: '#2a1418',
    shirt: '#ffffff', shirtShade: '#d8d2e6', ribbon: '#2f9a6a', ribbonShade: '#1d6a4a', ribbonLight: '#7ad8a8',
    cardi: '#f2e3c6', cardiShade: '#ccb08c', cardiLine: '#6a5038', star: '#ffd23f', starShade: '#e0a020',
  };
  const tilt = pts => rotate(pts, 3, 410, 520);   // head leans a little toward the light
  const ellipse = (cx, cy, rx, ry, n = 40) => Array.from({ length: n }, (_, i) => [cx + rx * Math.cos(i / n * 2 * Math.PI), cy + ry * Math.sin(i / n * 2 * Math.PI)]);
  const lockW = (w, tip) => t => t < tip ? w : w * Math.cos((t - tip) / (1 - tip) * Math.PI / 2);
  const mixHex = (a, b, f) => toHex([0, 1, 2].map(k => parseInt(a.slice(1 + 2 * k, 3 + 2 * k), 16) * (1 - f) + parseInt(b.slice(1 + 2 * k, 3 + 2 * k), 16) * f));

  // A hair lock: gradient fill from root to tip, a soft shadow band on the side away from the light,
  // strands, thin ink edges, and a few thin strands splitting off near the tip.
  function paintLock(l, spine, w, o = {}) {
    const r = ribbon(spine, lockW(w, o.tip ?? 0.6), 20), n = r.c.length, a = r.c[0], b = r.c[n - 1];
    const col = o.colors || { grad: [C.hairShade, C.hair, C.hairMid], shade: C.hairDeep, light: C.hairLight };
    F(l, r.outline, lin(l, a[0], a[1], b[0], b[1], [[0, col.grad[0]], [0.45, col.grad[1]], [1, col.grad[2]]]), { sharp: true });
    const side = o.shadeRight ? r.right : r.left, other = o.shadeRight ? r.left : r.right;
    const inner = side.map((p, i) => { const f = 0.32 * (0.5 + 0.5 * Math.sin(Math.PI * i / (n - 1))); return [p[0] + (other[i][0] - p[0]) * f, p[1] + (other[i][1] - p[1]) * f]; });
    F(l, side.concat(inner.reverse()), col.shade, { sharp: true, opacity: o.shadeOpacity ?? 0.32 });
    for (let s = 0, count = Math.round(w * (o.density ?? 0.35)); s < count; s++) {
      const u = 0.08 + rnd() * 0.84, i0 = Math.floor(rnd() * n * 0.3), i1 = Math.min(n - 1, Math.floor(n * (0.5 + rnd() * 0.45)));
      const pts = [];
      for (let i = i0; i <= i1; i++) pts.push([r.left[i][0] + (r.right[i][0] - r.left[i][0]) * u, r.left[i][1] + (r.right[i][1] - r.left[i][1]) * u]);
      const litSide = o.shadeRight ? u < 0.5 : u > 0.5;
      if (pts.length > 2) S(l, pts, { raw: true, size: 0.8 + rnd() * 1.3, color: litSide && rnd() < 0.7 ? col.light : col.shade, opacity: 0.3 + rnd() * 0.35, hardness: 0.6 });
    }
    for (const e of [r.left, r.right]) S(l, e.slice(Math.floor(n * 0.03)), { raw: true, size: o.line ?? 1.9, color: C.hairLine, hardness: 0.85, a: 0.3, b: 0.15, opacity: o.lineOpacity ?? 0.85 });
    for (let k = 0; k < (o.split || 0); k++) {
      const i0 = Math.floor(n * (0.55 + rnd() * 0.15)), u = 0.2 + rnd() * 0.6, drift = (o.drift || 0) * (0.5 + rnd());
      const p0 = [r.left[i0][0] + (r.right[i0][0] - r.left[i0][0]) * u, r.left[i0][1] + (r.right[i0][1] - r.left[i0][1]) * u], end = r.c[n - 1];
      const p2 = [end[0] + drift * 30, end[1] + 25 + rnd() * 45], p1 = [(p0[0] + p2[0]) / 2 + drift * 8, (p0[1] + p2[1]) / 2];
      paintLock(l, [p0, p1, p2], 5 + rnd() * 7, { tip: 0.3, colors: col, density: 0.2, line: 1.2, lineOpacity: 0.7, shadeRight: o.shadeRight });
    }
    return r;
  }

  // ---------- background: golden-hour sky, soft clouds, the sun's glow on the right
  const bg = L('bg');
  const sky = [[0, '#2c2c6e'], [220, '#5b4795'], [420, '#a8649e'], [600, '#ec8c86'], [760, '#ffb878'], [1000, '#ffe0a0']];
  const skyAt = y => { for (let i = 1; i < sky.length; i++) if (y <= sky[i][0]) return mixHex(sky[i - 1][1], sky[i][1], (y - sky[i - 1][0]) / (sky[i][0] - sky[i - 1][0])); };
  for (let y = 0; y <= 1000; y += 6) S(bg, [[-20, y], [820, y]], { size: 10, color: skyAt(y), hardness: 0.95, pressure: () => 1, segs: 1 });
  D(bg, 700, 640, 420, '#ffd9a0', 0.5); D(bg, 700, 640, 170, '#fff4d6', 0.55);
  for (const [cx, cy, sc] of [[120, 300, 1], [660, 250, 1.2], [70, 620, 0.9], [740, 470, 0.8], [560, 820, 1.1], [180, 860, 1]]) {
    for (let i = 0; i < 14; i++) {
      const x = cx + (rnd() - 0.5) * 220 * sc, y = cy + (rnd() - 0.5) * 50 * sc, r = (30 + rnd() * 40) * sc;
      D(bg, x, y, r, '#e7a6be', 0.35, { sy: 0.45 });
      D(bg, x + r * 0.25, y + r * 0.12, r * 0.7, '#ffd8c0', 0.3, { sy: 0.4 });
    }
  }
  for (const [x, y, s] of [[620, 170, 9], [655, 190, 7], [150, 150, 8], [175, 175, 6]])
    S(bg, [[x - s, y - s * 0.4], [x, y], [x + s, y - s * 0.4]], { size: 2, color: '#4a3050', opacity: 0.7, segs: 4 });

  // ---------- back hair
  const back = L('backhair');
  const NAPE = tilt([[300, 380], [522, 380], [534, 520], [546, 700], [282, 700], [292, 520]]);
  F(back, NAPE, lin(back, 0, 400, 0, 640, [[0, C.hairShade], [1, C.hairDeep]]), { sharp: true });
  for (let i = 0; i < 40; i++) { const x = 318 + rnd() * 196; S(back, tilt([[x, 420 + rnd() * 40], [x + (x - 410) * 0.08, 690]]), { size: 1 + rnd(), color: rnd() < 0.5 ? C.hair : C.hairDeep, opacity: 0.5, segs: 1 }); }
  for (const [p, w, sh] of [[[[250, 280], [222, 420], [204, 560], [186, 680], [176, 770]], 56, false], [[[300, 330], [284, 460], [272, 600], [262, 740]], 46, false],
    [[[572, 280], [600, 420], [618, 560], [634, 680], [646, 770]], 56, true], [[[522, 330], [538, 460], [550, 600], [560, 740]], 46, true]])
    paintLock(back, tilt(p), w, { tip: 0.7, shadeRight: sh, colors: { grad: [C.hairDeep, C.hairShade, C.hair], shade: C.hairDeep, light: C.hairMid }, split: 2, drift: sh ? 1 : -1, density: 0.25 });

  // ---------- neck, shirt, ribbon tie, cardigan
  const body = L('body');
  F(body, [[372, 470], [452, 470], [455, 560], [470, 600], [412, 632], [352, 600], [368, 560]], C.skin, { sharp: true });
  F(body, [[352, 588], [472, 588], [442, 700], [412, 830], [382, 700]], C.shirt, { sharp: true });
  const CARDI_L = [[176, 650], [228, 618], [300, 600], [352, 594], [386, 700], [412, 860], [412, 1004], [138, 1004], [136, 880], [140, 760], [152, 692]];
  const CARDI_R = [[648, 660], [594, 622], [522, 603], [472, 594], [438, 700], [412, 860], [412, 1004], [686, 1004], [684, 880], [678, 760], [666, 694]];
  F(body, CARDI_L, C.cardi, { sharp: true }); F(body, CARDI_R, C.cardi, { sharp: true });
  const bodyShade = L('body_shade', 'multiply');
  F(bodyShade, [[368, 478], [456, 478], [454, 536], [412, 568], [370, 536]], C.skinShade, { sharp: true });
  crescent(bodyShade, CARDI_L, 26, -8, C.cardiShade, { blur: 1 });
  crescent(bodyShade, CARDI_R, 22, -8, C.cardiShade, { blur: 1, opacity: 0.6 });
  F(bodyShade, [[352, 588], [472, 588], [462, 620], [412, 650], [362, 620]], C.shirtShade, { sharp: true, opacity: 0.8 });
  for (const p of [[[230, 700], [252, 800], [262, 920], [266, 1004]], [[190, 760], [182, 880], [180, 1004]], [[590, 720], [576, 840], [570, 1004]], [[630, 780], [640, 900]]])
    S(bodyShade, p, { size: 14, color: C.cardiShade, opacity: 0.6, hardness: 0.3 });
  for (const p of [[[222, 640], [236, 760], [246, 900], [250, 1004]], [[600, 646], [590, 770], [584, 900], [582, 1004]]]) S(bodyShade, p, { size: 6, color: C.cardiShade, opacity: 0.8, hardness: 0.5 });
  clipL(bodyShade, 'body');
  const bodyRim = L('body_rim', 'screen');
  S(bodyRim, [[648, 664], [668, 700], [680, 780], [686, 900], [688, 1004]], { size: 10, color: '#ffcf8a', opacity: 0.7, hardness: 0.3, a: 0.1, b: 0.1 });
  clipL(bodyRim, 'body');
  const bodyLine = L('body_line');
  for (const p of [[[352, 594], [386, 700], [412, 860], [412, 1004]], [[472, 594], [438, 700], [412, 860]]]) {
    S(bodyLine, p, { size: 2.2, color: C.cardiLine, a: 0.05, b: 0.05 });
    S(bodyLine, p.map(([x, y]) => [x + (x < 412 ? -9 : 9), y + 2]), { size: 1.4, color: C.cardiLine, opacity: 0.5, a: 0.05, b: 0.05 });
  }
  for (const y of [880, 950]) { F(bodyLine, ellipse(426, y, 7, 7, 14), '#e8d6b4', { sharp: true }); S(bodyLine, ellipse(426, y, 7, 7, 14).concat([[433, y]]), { raw: true, size: 1.4, color: C.cardiLine, pressure: () => 1 }); }
  for (const p of [[[176, 650], [152, 692], [140, 760], [136, 880], [138, 1004]], [[648, 660], [666, 694], [678, 760], [684, 880], [686, 1004]], [[176, 650], [228, 618], [300, 600], [352, 594]], [[648, 660], [594, 622], [522, 603], [472, 594]]])
    S(bodyLine, p, { size: 2.4, color: C.cardiLine, a: 0.04, b: 0.04 });
  for (const flap of [[[372, 590], [336, 610], [350, 662], [404, 628]], [[452, 590], [488, 610], [474, 662], [420, 628]]]) {
    F(bodyLine, flap, C.shirt, { sharp: true });
    S(bodyLine, flap.concat([flap[0]]), { size: 1.8, color: '#7a7090', pressure: () => 1, segs: 1 });
  }
  for (const loop of [[[408, 636], [374, 618], [364, 644], [404, 654]], [[416, 636], [450, 618], [460, 644], [420, 654]]]) {
    F(bodyLine, loop, C.ribbon);
    F(bodyLine, loop.map(([x, y]) => [x, y + 4]).slice(2).concat([loop[0]]), C.ribbonShade, { opacity: 0.6 });
    S(bodyLine, loop.concat([loop[0]]), { size: 1.6, color: '#0f3a28', pressure: () => 1, segs: 5 });
  }
  for (const tail of [[[404, 650], [390, 702], [402, 708], [412, 662]], [[420, 650], [434, 702], [422, 708], [412, 662]]]) {
    F(bodyLine, tail, C.ribbon, { sharp: true });
    S(bodyLine, tail.concat([tail[0]]), { size: 1.6, color: '#0f3a28', pressure: () => 1, segs: 1 });
  }
  F(bodyLine, ellipse(412, 643, 9, 8, 16), C.ribbonShade, { sharp: true });
  S(bodyLine, [[372, 624], [392, 628]], { size: 2.5, color: C.ribbonLight, opacity: 0.8 });

  // ---------- face
  const FACE = tilt([[288, 250], [290, 340], [294, 400], [302, 432], [318, 462], [345, 488], [376, 507], [400, 516], [412, 518], [424, 516], [450, 506], [482, 486], [508, 458], [524, 425], [532, 385], [534, 320], [530, 250]]);
  const face = L('face');
  F(face, FACE, lin(face, 290, 0, 534, 0, [[0, mixHex(C.skin, C.skinShade, 0.35)], [0.55, C.skin], [1, C.skinLight]]), { sharp: true });
  const ear = tilt([[298, 398], [285, 403], [280, 422], [285, 442], [299, 450]]);
  F(face, ear, C.skin, { sharp: true });
  S(face, tilt([[296, 408], [288, 420], [292, 438]]), { size: 1.6, color: C.skinLine, opacity: 0.5 });
  const faceShade = L('face_shade', 'multiply');
  crescent(faceShade, FACE, 16, -10, C.skinShade, { blur: 2 });
  for (const [x, y] of tilt([[342, 412], [482, 412]])) D(faceShade, x, y, 30, C.blush, 0.45, { sy: 0.5 });
  D(faceShade, 412, 512, 40, C.skinShade, 0.4, { sy: 0.4 });
  const eyes = { white: L('eyes_white'), iris: L('eyes_iris'), line: L('eyes_line') };
  const EYES = [
    { c: [342, 362], open: [[296, 348], [303, 340], [320, 335], [342, 334], [362, 336], [376, 342], [381, 350], [374, 367], [362, 379], [344, 385], [324, 383], [307, 374], [299, 362]],
      lash: [[287, 352], [299, 339], [321, 331], [345, 330], [367, 333], [381, 341], [386, 349]], lower: [[303, 370], [321, 382], [343, 387], [365, 382]],
      crease: [[300, 329], [325, 321], [352, 320], [375, 326]], outer: 0 },
    { c: [478, 362], open: [[446, 350], [451, 342], [466, 336], [486, 334], [508, 335], [525, 340], [534, 348], [530, 363], [518, 376], [498, 384], [478, 385], [460, 379], [450, 367]],
      lash: [[441, 349], [448, 340], [468, 332], [493, 330], [516, 333], [532, 341], [540, 352]], lower: [[456, 379], [478, 387], [500, 383], [520, 373]],
      crease: [[445, 326], [468, 320], [495, 321], [520, 329]], outer: 6 },
  ];
  for (const e of EYES) {
    const [cx, cy] = tilt([e.c])[0], T = tilt;
    F(eyes.white, T(e.open), '#fbfbff');
    F(eyes.iris, ellipse(cx, cy, 26, 28), C.iris, { sharp: true });
    F(eyes.iris, ellipse(cx, cy - 10, 26, 17).filter(([, y]) => y < cy - 3).concat([[cx + 26, cy - 3], [cx - 26, cy - 3]]), C.irisTop, { opacity: 0.88 });
    F(eyes.iris, ellipse(cx, cy + 15, 19, 10), C.irisLow, { sharp: true, opacity: 0.85 });
    for (let a = 30; a <= 150; a += 20) { const t = a * Math.PI / 180; S(eyes.iris, [[cx + 11 * Math.cos(t), cy + 14 * Math.sin(t)], [cx + 23 * Math.cos(t), cy + 26 * Math.sin(t)]], { size: 1.6, color: C.irisGlow, opacity: 0.5, segs: 1 }); }
    F(eyes.iris, ellipse(cx, cy + 2, 9, 13), C.pupil, { sharp: true });
    S(eyes.iris, ellipse(cx, cy, 26, 28, 48).concat([[cx + 26, cy]]), { raw: true, size: 2, color: '#073028', pressure: () => 1 });
    S(eyes.iris, T([e.open[0], e.open[2], e.open[4], e.open[6]]), { size: 11, color: '#3a2a50', hardness: 0.2, opacity: 0.4 });
    S(eyes.line, T(e.lash), { size: 10, color: C.lash, hardness: 0.9, pressure: t => 0.3 + 0.7 * Math.sin(Math.PI * Math.min(1, t * 1.12)) });
    const o0 = T([cx < 410 ? e.lash[0] : e.lash[e.lash.length - 1]])[0], dir = cx < 410 ? -1 : 1;
    S(eyes.line, [o0, [o0[0] + dir * 6, o0[1] + 7], [o0[0] + dir * 8, o0[1] + 13]], { size: 4, color: C.lash, a: 0.1, b: 0.8 });
    S(eyes.line, T(e.lower), { size: 1.6, color: '#6a3a40', opacity: 0.55, a: 0.3, b: 0.3 });
    S(eyes.line, T(e.crease), { size: 1.6, color: C.skinLine, opacity: 0.55, a: 0.3, b: 0.3 });
    F(eyes.line, ellipse(cx + 8, cy - 13, 6.5, 7.5, 16), '#ffffff', { sharp: true });
    F(eyes.line, ellipse(cx - 10, cy + 14, 2.6, 2.6, 12), '#ffffff', { sharp: true, opacity: 0.9 });
    F(eyes.line, ellipse(cx + 12, cy + 10, 1.8, 1.8, 10), '#eaffef', { sharp: true, opacity: 0.85 });
  }
  clipL(eyes.iris, 'eyes_white');
  const fl = L('face_line');
  for (const [x, y] of tilt([[342, 410], [482, 410]])) for (let k = 0; k < 4; k++) S(fl, [[x - 11 + k * 7, y - 4], [x - 15 + k * 7, y + 5]], { size: 1.3, color: '#e0707a', opacity: 0.55, segs: 1 });
  S(fl, tilt([[415, 394], [419, 402], [414, 407]]), { size: 2.3, color: '#c08478', opacity: 0.8, a: 0.3, b: 0.3 });
  F(fl, tilt([[388, 450], [400, 454], [412, 455], [424, 454], [436, 449], [433, 459], [424, 467], [412, 470], [400, 467], [391, 459]]), '#b0525a');
  F(fl, tilt([[397, 465], [405, 461], [412, 460], [419, 461], [427, 465], [420, 469], [412, 471], [404, 469]]), '#f08a8a');
  S(fl, tilt([[386, 449], [400, 454], [412, 455], [424, 454], [438, 448]]), { size: 2.4, color: '#6a2a32', a: 0.2, b: 0.2 });
  S(fl, tilt([[399, 473], [412, 476], [425, 473]]), { size: 1.5, color: '#c06a6a', opacity: 0.6 });
  S(fl, tilt([[294, 400], [302, 432], [318, 462], [345, 488], [376, 507], [400, 516], [412, 518], [424, 516], [450, 506], [482, 486], [508, 458], [524, 425]]), { size: 2.4, color: C.skinLine, pressure: t => 0.4 + 0.6 * Math.sin(Math.PI * t) });
  S(fl, tilt([[452, 505], [484, 485], [508, 458], [524, 425], [532, 385]]), { size: 5, color: C.rim, opacity: 0.85, hardness: 0.5, a: 0.3, b: 0.3 });
  for (const p of [[[318, 300], [338, 294], [360, 295], [376, 300]], [[446, 299], [462, 294], [484, 293], [505, 299]]]) S(fl, tilt(p), { size: 2.6, color: C.hairLine, opacity: 0.7, a: 0.2, b: 0.4 });

  // ---------- front hair: side locks, crown, side-swept bangs, shine, rim light, star clip
  const hairBack = L('hair_back');
  const sideL = [[[[268, 170], [240, 270], [226, 390], [222, 500], [230, 600], [244, 690], [236, 760]], 44], [[[284, 210], [262, 320], [254, 430], [256, 540], [266, 630], [276, 700]], 34], [[[298, 250], [286, 340], [282, 430], [290, 520]], 16]];
  const sideR = [[[[556, 170], [584, 270], [598, 390], [600, 500], [592, 600], [578, 690], [586, 760]], 44], [[[540, 210], [562, 320], [570, 430], [568, 540], [558, 630], [548, 700]], 34], [[[526, 250], [538, 340], [542, 430], [534, 520]], 16]];
  const litCols = { grad: [C.hair, C.hairMid, C.hairLight], shade: C.hairShade, light: C.hairShine };
  for (const [p, w] of sideL) paintLock(hairBack, tilt(p), w, { tip: 0.75, split: w > 30 ? 3 : 0, drift: -1 });
  const litLocks = sideR.map(([p, w]) => paintLock(hairBack, tilt(p), w, { tip: 0.75, shadeRight: false, colors: litCols, split: w > 30 ? 3 : 0, drift: 1 }));
  const CROWN = tilt([[228, 262], [236, 190], [270, 135], [330, 98], [410, 82], [490, 94], [552, 132], [588, 190], [596, 262], [540, 282], [290, 282]]);
  const crown = L('crown');
  F(crown, CROWN, lin(crown, 230, 0, 596, 0, [[0, C.hairShade], [0.55, C.hair], [1, C.hairMid]]));
  for (let i = 0; i < 70; i++) {
    const a = Math.PI * (0.05 + rnd() * 0.9), len = 50 + rnd() * 110, x0 = 472 + Math.cos(a) * 8, y0 = 110 + Math.sin(a) * 6;
    const x1 = 472 + Math.cos(a) * len * 1.5, y1 = 110 + Math.sin(a) * len * 1.0 + 20;
    const lit = x1 > 470;
    S(crown, tilt([[x0, y0], [(x0 + x1) / 2, (y0 + y1) / 2 - 6], [x1, y1]]), { size: 0.9 + rnd() * 1.2, color: lit ? C.hairLight : (rnd() < 0.5 ? C.hairMid : C.hairDeep), opacity: 0.35 + rnd() * 0.3, segs: 6 });
  }
  clip(crown, CROWN);
  S(crown, CROWN.slice(0, 9), { size: 2.2, color: C.hairLine, a: 0.04, b: 0.04 });
  const hair = L('hair');
  const bangDefs = [
    [[[482, 110], [528, 140], [556, 200], [566, 262], [562, 318]], 38], [[[478, 112], [502, 160], [514, 220], [518, 280], [516, 338]], 34],
    [[[462, 110], [400, 130], [340, 170], [300, 230], [284, 300], [288, 355]], 46], [[[466, 112], [420, 140], [372, 190], [344, 250], [332, 310], [336, 368]], 42],
    [[[468, 114], [440, 150], [408, 205], [388, 265], [380, 325], [384, 382]], 38], [[[474, 116], [478, 170], [476, 230], [470, 290], [466, 345]], 30],
    [[[470, 116], [456, 160], [440, 220], [428, 280], [424, 340], [426, 395]], 32],
  ];
  const bangs = bangDefs.map(([p, w]) => {
    const lit = p[p.length - 1][0] > 450;
    return paintLock(hair, tilt(p), w, { tip: 0.5, colors: lit ? litCols : undefined, shadeRight: !lit });
  });
  const shine = L('hair_shine', 'screen');
  for (let x = 300; x <= 570; x += 6 + rnd() * 6) {
    const y = 215 + 0.0007 * (x - 450) ** 2, len = 5 + rnd() * 10, warm = (x - 300) / 270;
    S(shine, tilt([[x, y - len], [x - (450 - x) * 0.02, y + len * 0.6]]), { size: 3 + rnd() * 3, color: C.hairShine, opacity: 0.25 + 0.6 * warm, a: 0.5, b: 0.5, segs: 1 });
  }
  for (const r of litLocks.slice(0, 2)) S(shine, r.right.slice(2), { raw: true, size: 6, color: C.rim, opacity: 0.75, hardness: 0.35, a: 0.1, b: 0.3 });
  S(shine, CROWN.slice(5, 10), { size: 7, color: C.rim, opacity: 0.6, hardness: 0.3, a: 0.2, b: 0.2 });
  clipL(shine, 'hair', 'hair_back', 'crown');
  const starL = L('star_clip');
  const [sx, sy] = tilt([[548, 236]])[0];
  const star = Array.from({ length: 10 }, (_, i) => { const a = -Math.PI / 2 + i * Math.PI / 5 + 0.25, rr = i % 2 ? 7 : 16; return [sx + Math.cos(a) * rr, sy + Math.sin(a) * rr]; });
  F(starL, star, C.star, { sharp: true });
  F(starL, star.filter((_, i) => i >= 3 && i <= 7).concat([[sx, sy]]), C.starShade, { sharp: true, opacity: 0.6 });
  S(starL, star.concat([star[0]]), { size: 1.8, color: '#7a4a10', pressure: () => 1, segs: 1 });
  F(starL, ellipse(sx - 4, sy - 5, 2.5, 2, 10), '#fffbe0', { sharp: true });
  const castShadow = L('bang_shadow', 'multiply', 0.4);
  for (const r of bangs) F(castShadow, shift(r.outline, -6, 12), '#d39a8e', { sharp: true });
  F(castShadow, shift(CROWN, -6, 12), '#d39a8e', { sharp: true });
  clipL(castShadow, 'face');
  layers.splice(layers.indexOf(castShadow), 1);
  layers.splice(layers.indexOf(eyes.white), 0, castShadow);

  // ---------- light: warm wash from the right, dust in the air
  const fx = L('fx', 'screen');
  D(fx, 760, 520, 420, '#ffb070', 0.22);
  for (let i = 0; i < 40; i++) D(fx, rnd() * 800, rnd() * 1000, 1 + rnd() * 2.5, '#fff0c8', 0.4 + rnd() * 0.5);
  const out = flatten();
  glow(out, 10, 0.16);
  grain(out, 5);
  return out.toDataURL('image/png');
}
