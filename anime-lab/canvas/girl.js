// The lavender-haired girl, painted with the canvas brush engine (no SVG).
// Same design as attempt 05, but every mark is a brush action: lasso fills, pressure strokes,
// airbrush dots and clipping masks on blend-mode layers. Hair is hundreds of painted strands.
function draw() {
  init(800, 1000);
  const rnd = rng(12);
  const C = {
    skin: '#fff1e8', skinShade: '#f3c4b6', skinLine: '#a8645a',
    hair: '#e8e0f4', hairShade: '#bcaada', hairBack: '#a593cd', hairBackShade: '#8572b2', hairLine: '#54417c', hairLight: '#ffffff',
    irisTop: '#4a230a', irisMid: '#b8621a', irisLow: '#f3b23c', irisGlow: '#ffe7a0', eyeLine: '#2c1830',
    blouse: '#f8f8fc', blouseShade: '#c9cce3', blouseLine: '#666b96',
    collar: '#2a3266', collarShade: '#1b2149', collarLine: '#10142e',
    ribbon: '#d8394f', ribbonShade: '#a5223a', ribbonLine: '#5a0f1f',
  };
  const tilt = pts => rotate(pts, -4, 400, 600);       // her head leans 4°
  const up = pts => shift(pts, 0, -20);                // the uniform sits 20px higher
  const mirror = pts => pts.map(([x, y]) => [800 - x, y]);
  const ellipse = (cx, cy, rx, ry, n = 40) => Array.from({ length: n }, (_, i) => [cx + rx * Math.cos(i / n * 2 * Math.PI), cy + ry * Math.sin(i / n * 2 * Math.PI)]);
  const cap = (cx, cy, rx, ry, yc, above) => ellipse(cx, cy, rx, ry, 72).filter(([, y]) => above ? y <= yc : y >= yc)
    .sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  const lockW = (w, tip) => t => t < tip ? w : w * Math.cos((t - tip) / (1 - tip) * Math.PI / 2);

  // ---------- shapes (from the attempt-05 design)
  const BACK = tilt([[400, 128], ...cubic([400, 128], [292, 128], [206, 196], [198, 330]), ...cubic([198, 330], [190, 460], [188, 580], [168, 720]),
    ...cubic([168, 720], [150, 840], [132, 930], [120, 1000]), [680, 1000], ...cubic([680, 1000], [668, 930], [650, 840], [632, 720]),
    ...cubic([632, 720], [612, 580], [610, 460], [602, 330]), ...cubic([602, 330], [594, 196], [508, 128], [400, 128])]);
  const CROWN = tilt([[206, 345], ...cubic([206, 345], [200, 200], [290, 128], [400, 128]), ...cubic([400, 128], [510, 128], [600, 200], [594, 345]),
    ...cubic([594, 345], [560, 300], [470, 250], [400, 250]), ...cubic([400, 250], [330, 250], [240, 300], [206, 345])]);
  const FACE = tilt([[236, 262], [238, 440], ...quad([238, 440], [292, 548], [364, 598]), ...quad([364, 598], [400, 620], [436, 598]),
    ...quad([436, 598], [508, 548], [562, 440]), [564, 262]]);
  const NECK = [[358, 575], [352, 700], [340, 730], [400, 872], [460, 730], [448, 700], [442, 575]];
  const TORSO = up([[348, 735], [400, 860], [452, 735], ...cubic([452, 735], [500, 768], [595, 790], [650, 824]), ...cubic([650, 824], [696, 852], [718, 960], [724, 1040]),
    [76, 1040], ...cubic([76, 1040], [82, 960], [104, 852], [150, 824]), ...cubic([150, 824], [205, 790], [300, 768], [348, 735])]);
  const COLLAR_L = up([[348, 735], [400, 860], [400, 948], [176, 820], ...cubic([176, 820], [225, 792], [300, 768], [348, 735])]);
  const COLLAR_R = mirror(COLLAR_L);

  // ---------- hair locks: geometry first, because their shapes also cast shadows
  const sideDefs = [
    { p: [[214, 290], [192, 440], [182, 600], [178, 740], [168, 860]], w: 52, tip: 0.55, back: true },
    { p: [[238, 250], [220, 380], [210, 520], [206, 640], [214, 760], [236, 870]], w: 58, tip: 0.5, dy: 24 },
    { p: [[256, 270], [238, 400], [232, 520], [238, 630], [256, 720], [282, 790]], w: 40, tip: 0.45, dy: -18 },
    { p: [[248, 300], [228, 430], [222, 560], [230, 680], [248, 760]], w: 18, tip: 0.4, dy: 30 },
  ];
  const lengthen = (pts, dy) => pts.map((p, i) => i === pts.length - 1 ? [p[0], p[1] + dy] : p);
  const sides = sideDefs.flatMap(s => [s.p, lengthen(mirror(s.p), s.dy || 0)].map(p => ({ ...s, r: ribbon(tilt(p), lockW(s.w, s.tip), 20) })));
  const bangDefs = [
    [[[262, 205], [244, 280], [234, 345], [236, 380], [244, 398]], 46], [[[538, 205], [556, 280], [566, 345], [564, 380], [556, 396]], 46],
    [[[300, 178], [274, 250], [258, 318], [257, 350], [264, 374]], 60], [[[503, 180], [528, 250], [544, 318], [545, 346], [538, 368]], 58],
    [[[276, 260], [262, 320], [250, 356], [244, 366]], 18, 0.5], [[[526, 260], [540, 320], [552, 352], [558, 360]], 18, 0.5],
    [[[342, 172], [320, 248], [304, 318], [302, 358], [310, 386]], 62], [[[462, 172], [482, 248], [496, 318], [498, 354], [490, 380]], 60],
    [[[318, 270], [310, 330], [296, 372], [288, 384]], 20, 0.5], [[[486, 270], [494, 330], [506, 366], [512, 376]], 20, 0.5],
    [[[352, 190], [338, 280], [332, 350], [336, 394]], 14, 0.45], [[[452, 190], [462, 280], [468, 350], [463, 392]], 14, 0.45],
    [[[422, 176], [431, 262], [436, 340], [431, 398]], 54], [[[384, 172], [372, 258], [363, 338], [364, 405]], 58],
    [[[376, 260], [372, 330], [376, 380], [384, 398]], 18, 0.5], [[[400, 196], [403, 300], [399, 384], [404, 446]], 28, 0.5],
  ];
  const bangs = bangDefs.map(([p, w, tip]) => ({ w, r: ribbon(tilt(p), lockW(w, tip ?? 0.55), 20) }));
  const ahoge = { w: 15, r: ribbon(tilt([[402, 150], [408, 105], [430, 78], [458, 72], [474, 88]]), lockW(15, 0.35), 20) };
  const hanging = [[[[296, 430], [292, 560], [286, 680], [280, 770]], 50], [[[328, 470], [326, 580], [322, 690], [318, 770]], 40]]
    .flatMap(([p, w]) => [p, mirror(p)].map(q => ({ w, r: ribbon(tilt(q), lockW(w, 0.6), 20) })));

  // One lock: lasso fill, a cel-shadow band down its far side, painted strands that
  // converge at the tip (lighter on the lit side), then tapered ink along both edges.
  function paintLock(l, lk, o = {}) {
    const { r, w } = lk, n = r.c.length;
    F(l, r.outline, o.fill || C.hair, { sharp: true });
    const k = o.shadeK ?? 0.38;
    const inner = r.right.map((p, i) => { const f = k * (0.55 + 0.45 * Math.sin(Math.PI * i / (n - 1))); return [p[0] + (r.left[i][0] - p[0]) * f, p[1] + (r.left[i][1] - p[1]) * f]; });
    F(l, r.right.concat(inner.reverse()), o.shadeFill || C.hairShade, { sharp: true, opacity: 0.85 });
    for (let s = 0, count = Math.round(w * (o.density ?? 0.75)); s < count; s++) {
      const u = 0.05 + rnd() * 0.9, a = Math.floor(rnd() * n * 0.3), b = Math.min(n - 1, Math.floor(n * (0.55 + rnd() * 0.45)));
      const pts = [];
      for (let i = a; i <= b; i++) pts.push([r.left[i][0] + (r.right[i][0] - r.left[i][0]) * u, r.left[i][1] + (r.right[i][1] - r.left[i][1]) * u]);
      if (pts.length < 3) continue;
      const lit = rnd() < 0.7 - u * 0.5;
      S(l, pts, { raw: true, size: 0.7 + rnd() * 1.5, color: lit ? (o.light || C.hairLight) : (o.dark || C.hairLine), opacity: lit ? 0.3 + rnd() * 0.4 : 0.12 + rnd() * 0.18, hardness: 0.6, a: 0.25, b: 0.3 });
    }
    for (const e of [r.left, r.right]) S(l, e.slice(Math.floor(n * 0.04)), { raw: true, size: o.line ?? 2.4, color: o.lineColor || C.hairLine, hardness: 0.85, a: 0.3, b: 0.18 });
  }

  // ---------- background: a painted twilight sky
  const bg = L('bg');
  const sky = [[0, [0x23, 0x2a, 0x63]], [0.55, [0x6a, 0x5a, 0xa6]], [1, [0xf4, 0xa9, 0xbb]]];
  const skyAt = t => { for (let i = 1; i < sky.length; i++) if (t <= sky[i][0]) { const f = (t - sky[i - 1][0]) / (sky[i][0] - sky[i - 1][0]); return '#' + sky[i - 1][1].map((c, k) => Math.round(c + (sky[i][1][k] - c) * f).toString(16).padStart(2, '0')).join(''); } };
  for (let y = 0; y <= 1000; y += 8) S(bg, [[-20, y], [820, y]], { size: 14, color: skyAt(y / 1000), hardness: 0.95, pressure: () => 1, segs: 1 });
  D(bg, 400, 340, 360, '#fff8ec', 0.55); D(bg, 400, 330, 170, '#fff8ec', 0.45);
  for (let i = 0; i < 46; i++) {
    const x = rnd() * 800, y = rnd() * 620, s = 2 + rnd() * 7, op = 0.4 + rnd() * 0.6;
    S(bg, [[x - s, y], [x + s, y]], { size: s * 0.35, color: '#ffffff', opacity: op, a: 0.5, b: 0.5, segs: 1 });
    S(bg, [[x, y - s], [x, y + s]], { size: s * 0.35, color: '#ffffff', opacity: op, a: 0.5, b: 0.5, segs: 1 });
  }

  // ---------- back hair
  const back = L('backhair');
  F(back, BACK, C.hairBack, { sharp: true });
  F(back, tilt([[300, 420], [300, 560], [292, 700], [282, 1000], [518, 1000], [508, 700], [500, 560], [500, 420]]), '#6f5d9e');
  for (const lk of hanging) paintLock(back, lk, { fill: C.hairBack, shadeFill: C.hairBackShade, density: 0.5 });
  for (let i = 0; i < 90; i++) {
    const x = 205 + rnd() * 390, y0 = 200 + rnd() * 120, drift = (x - 400) * 0.25;
    S(back, tilt([[x, y0], [x + drift * 0.4, y0 + 250], [x + drift, 1000]]), { size: 0.8 + rnd() * 1.4, color: rnd() < 0.5 ? '#c9bde6' : C.hairLine, opacity: 0.18 + rnd() * 0.2, hardness: 0.6 });
  }
  const rim = L('backhair_rim', 'screen');
  S(rim, BACK.slice(0, -1), { raw: true, size: 14, color: '#ffd9ee', hardness: 0.1, opacity: 0.7, pressure: () => 1 });
  clipL(rim, 'backhair');

  // ---------- neck and uniform
  const body = L('body');
  F(body, NECK, C.skin, { sharp: true });
  F(body, TORSO, C.blouse, { sharp: true });
  F(body, up([[366, 800], [434, 800], [400, 862]]), C.blouse, { sharp: true });
  const shade = L('body_shade', 'multiply');
  F(shade, [[340, 560], [460, 560], [458, 636], ...quad([458, 636], [426, 668], [400, 670]), ...quad([400, 670], [370, 664], [342, 620])], C.skinShade, { sharp: true });
  for (const c of [COLLAR_L, COLLAR_R]) F(shade, shift(c, 7, 10), C.blouseShade, { sharp: true });
  for (const s of sides) F(shade, shift(s.r.outline, 7, 10), C.blouseShade, { sharp: true });
  for (const p of [[[172, 905], [196, 945], [206, 1000]], [[232, 880], [246, 930]]]) for (const q of [p, mirror(p)]) S(shade, up(q), { size: 2.2, color: C.blouseLine, opacity: 0.5 });
  clipL(shade, 'body');
  L('torso_shift', 'source-over', 1, { hidden: true }); F(named.torso_shift, shift(TORSO, -30, -12), '#000000', { sharp: true });
  const torsoShade = L('torso_crescent', 'multiply');
  F(torsoShade, TORSO, C.blouseShade, { sharp: true });
  clipL(torsoShade, 'body'); minusL(torsoShade, 'torso_shift');
  const collar = L('collar');
  F(collar, COLLAR_L, C.collar, { sharp: true }); F(collar, COLLAR_R, C.collar, { sharp: true });
  const collarFx = L('collar_detail');
  S(collarFx, up([[400, 933], [205, 822]]), { size: 5, color: '#ffffff', pressure: () => 1, segs: 1 });
  S(collarFx, up(mirror([[400, 933], [205, 822]])), { size: 5, color: '#ffffff', pressure: () => 1, segs: 1 });
  clipL(collarFx, 'collar');
  const bow = L('bow');
  for (const p of [[[392, 946], [380, 975], [368, 1010]], [[408, 946], [422, 976], [436, 1010]]])
    paintLock(bow, { w: 28, r: ribbon(up(p), lockW(28, 0.75), 20) }, { fill: C.ribbon, shadeFill: C.ribbonShade, light: '#ff9aa8', dark: C.ribbonLine, lineColor: C.ribbonLine, density: 0.3 });
  const loopL = up([[394, 930], ...cubic([394, 930], [360, 898], [316, 906], [320, 934]), ...cubic([320, 934], [324, 962], [366, 958], [394, 944])]);
  for (const lp of [loopL, mirror(loopL)]) {
    F(bow, lp, C.ribbon, { sharp: true });
    F(bow, lp.filter(([, y]) => y > 915 - 20).map(([x, y]) => [x, y + 3]), C.ribbonShade, { sharp: true, opacity: 0.5 });
    S(bow, lp.concat([lp[0]]), { raw: true, size: 2.6, color: C.ribbonLine, pressure: () => 1 });
  }
  F(bow, up([[386, 922], [414, 922], [416, 948], [384, 948]]), C.ribbon);
  S(bow, up([[386, 922], [414, 922], [416, 948], [384, 948], [386, 922]]), { size: 2.4, color: C.ribbonLine, pressure: () => 1, segs: 4 });
  const bodyLine = L('body_line');
  const SHOULDER = up([[348, 736], [300, 768], [205, 792], [150, 824], [106, 862], [86, 950], [76, 1040]]);
  for (const q of [SHOULDER, mirror(SHOULDER)]) S(bodyLine, q, { size: 4, color: C.blouseLine, a: 0.06, b: 0.04 });
  for (const p of [[[400, 948], [176, 820]], [[348, 735], [400, 860]]]) for (const q of [p, mirror(p)]) S(bodyLine, up(q), { size: 3, color: C.collarLine, a: 0.1, b: 0.1 });
  S(bodyLine, up([[368, 800], [432, 800]]), { size: 2.4, color: C.blouseLine, a: 0.1, b: 0.1 });
  for (const q of [[[358, 592], [356, 660], [352, 712]], mirror([[358, 592], [356, 660], [352, 712]])]) S(bodyLine, q, { size: 3, color: C.skinLine });

  // ---------- face
  const face = L('face');
  F(face, FACE, C.skin, { sharp: true });
  const faceShade = L('face_shade', 'multiply');
  F(faceShade, shift(CROWN, 5, 14), C.skinShade, { sharp: true });
  for (const b of bangs) F(faceShade, shift(b.r.outline, 5, 14), C.skinShade, { sharp: true });
  for (const s of sides) F(faceShade, shift(s.r.outline, 8, 6), C.skinShade, { sharp: true });
  clipL(faceShade, 'face');
  L('face_shift', 'source-over', 1, { hidden: true }); F(named.face_shift, shift(FACE, -14, -12), '#000000', { sharp: true });
  const faceCres = L('face_crescent', 'multiply');
  F(faceCres, FACE, C.skinShade, { sharp: true });
  clipL(faceCres, 'face'); minusL(faceCres, 'face_shift');
  const blush = L('blush');
  for (const [x, y] of tilt([[292, 504], [508, 504]])) D(blush, x, y, 48, '#ff8ea4', 0.5, { sy: 0.46 });
  for (const x of [278, 292, 306]) for (const q of [[[x + 5, 496], [x - 3, 510]], mirror([[x + 5, 496], [x - 3, 510]])]) S(blush, tilt(q), { size: 2, color: '#e46f86', opacity: 0.55, segs: 1 });

  // eyes: opening, iris bands clipped inside it, then lashes and highlights on top
  const white = L('eyes_white'), iris = L('eyes_iris'), lash = L('eyes_line');
  for (const [cx, cy, s] of [[313, 436, 1], [487, 436, -1]]) {
    const k = 1.08, P = (u, v) => [cx - s * u * k, cy + v * k], T = pts => tilt(pts);
    F(white, T([P(-46, -2), P(-26, -26), P(4, -34), P(30, -28), P(48, -9), P(44, 10), P(28, 34), P(4, 44), P(-22, 38), P(-40, 18)]), '#ffffff');
    const [ix, iy] = P(2, 8), rx = 28 * k, ry = 39 * k;
    F(iris, T(ellipse(ix, iy, rx, ry)), C.irisMid, { sharp: true });
    F(iris, T(cap(ix, iy, rx, ry, iy - 0.05 * ry, true)), C.irisTop, { sharp: true, opacity: 0.9 });
    F(iris, T(cap(ix, iy, rx, ry, iy + 0.35 * ry, false)), C.irisLow, { sharp: true, opacity: 0.9 });
    for (let a = 25; a <= 155; a += 18) {
      const t = a * Math.PI / 180;
      S(iris, T([[ix + 12 * k * Math.cos(t), iy + 19 * k * Math.sin(t)], [ix + 25 * k * Math.cos(t), iy + 35 * k * Math.sin(t)]]), { size: 1.8, color: C.irisGlow, opacity: 0.55, segs: 1 });
    }
    F(iris, T(ellipse(ix, iy + 2 * k, 11 * k, 19 * k)), '#2a1206', { sharp: true });
    F(iris, T([...quad([ix - 21 * k, iy + 18 * k], [ix, iy + 46 * k], [ix + 21 * k, iy + 18 * k]), ...quad([ix + 21 * k, iy + 18 * k], [ix, iy + 30 * k], [ix - 21 * k, iy + 18 * k])]), C.irisGlow, { sharp: true, opacity: 0.75 });
    S(iris, T(ellipse(ix, iy, rx, ry, 48).concat([[ix + rx, iy]])), { raw: true, size: 2.5, color: C.irisTop, pressure: () => 1 });
    S(iris, T([P(-46, -24), P(4, -31), P(50, -18)]), { size: 16, color: '#4b3a86', hardness: 0.2, opacity: 0.45 });
    S(lash, T([P(-48, -3), P(-28, -27), P(2, -36), P(30, -30), P(48, -12), P(58, -7)]), { size: 12, color: C.eyeLine, hardness: 0.9, pressure: t => t < 0.8 ? (3 + 9 * (t / 0.8) ** 1.3) / 12 : Math.max(0.05, ((1 - t) / 0.2) ** 0.7) });
    S(lash, T([P(44, -20), P(54, -29), P(62, -32)]), { size: 4.5, color: C.eyeLine, a: 0.1, b: 0.85 });
    S(lash, T([P(52, -11), P(61, -15), P(66, -14)]), { size: 3.5, color: C.eyeLine, a: 0.1, b: 0.85 });
    S(lash, T([P(44, 6), P(36, 24), P(18, 38)]), { size: 2.6, color: C.eyeLine, opacity: 0.85, a: 0.2, b: 0.7 });
    S(lash, T([P(-28, -46), P(2, -54), P(32, -48)]), { size: 2.4, color: C.skinLine, opacity: 0.7, a: 0.4, b: 0.4 });
    F(lash, T(rotate(ellipse(ix - 10 * k, iy - 16 * k, 9 * k, 12 * k), -20, ix - 10 * k, iy - 16 * k)), '#ffffff', { sharp: true });
    F(lash, T(ellipse(ix + 11 * k, iy + 21 * k, 4 * k, 4 * k, 16)), '#ffffff', { sharp: true });
    F(lash, T(ellipse(ix - 14 * k, iy + 17 * k, 2.2 * k, 2.2 * k, 12)), '#ffffff', { sharp: true, opacity: 0.9 });
  }
  clipL(iris, 'eyes_white');
  const faceLine = L('face_line');
  S(faceLine, tilt([[404, 503], [401, 511], [397, 517]]), { size: 3, color: C.skinLine, a: 0.3, b: 0.4 });
  F(faceLine, tilt([[380, 551], ...quad([380, 551], [400, 558], [420, 551]), ...quad([420, 551], [414, 572], [400, 573]), ...quad([400, 573], [386, 572], [380, 551])]), '#9c3b4a', { sharp: true });
  F(faceLine, tilt([[386, 565], ...quad([386, 565], [400, 558], [414, 565]), ...quad([414, 565], [408, 573], [400, 573]), ...quad([400, 573], [392, 573], [386, 565])]), '#ee8a92', { sharp: true });
  S(faceLine, tilt([[375, 548], [388, 553], [400, 555], [412, 553], [425, 548]]), { size: 3, color: '#7a2a35', a: 0.2, b: 0.2 });
  const JAW = [[236, 320], [238, 440], [296, 533], [364, 598], [400, 609]];
  for (const q of [JAW, mirror(JAW)]) S(faceLine, tilt(q), { size: 4.2, color: C.skinLine, a: 0.3, b: 0.25 });

  // ---------- front hair, back to front: side locks, ahoge, crown, bangs
  const hair = L('hair');
  for (const s of sides) paintLock(hair, s, s.back ? { fill: C.hairShade, shadeFill: C.hairBackShade, density: 0.5 } : {});
  paintLock(hair, ahoge, { line: 2.2, density: 0.4 });
  F(hair, CROWN, C.hair, { sharp: true });
  S(hair, CROWN.slice(0, 33), { raw: true, size: 30, color: C.hairShade, hardness: 0.2, opacity: 0.5, pressure: t => Math.max(0, (t - 0.45) * 1.8) });
  for (let i = 0; i < 70; i++) {
    const a = Math.PI * (0.15 + rnd() * 0.7), len = 40 + rnd() * 70;
    S(hair, tilt([[400 + Math.cos(Math.PI - a) * 12, 134], [400 + Math.cos(Math.PI - a) * len * 1.2, 134 + Math.sin(a) * len * 0.9]]), { size: 0.8 + rnd(), color: rnd() < 0.6 ? C.hairLight : C.hairLine, opacity: 0.25 + rnd() * 0.25, segs: 1 });
  }
  S(hair, CROWN.slice(0, 33), { raw: true, size: 4.5, color: C.hairLine, a: 0.04, b: 0.04 });
  for (const b of bangs) paintLock(hair, b);
  for (const t of [0.7, 0.8, 0.9]) for (const s of sides.filter(s => !s.back)) { const p = s.r.c[Math.floor(s.r.c.length * t)]; D(hair, p[0], p[1], 34, '#f2bfe0', 0.18); }
  const shine = L('hair_shine', 'screen');
  for (let i = 0, x = 232; x <= 568; x += 6.5, i++) {
    const y = 262 + 0.0011 * (x - 400) ** 2, upL = i % 2 ? 8 + rnd() * 8 : 14 + rnd() * 16, dn = i % 2 ? 6 + rnd() * 6 : 10 + rnd() * 10;
    const a = Math.atan2(x - 400, y - 120), dx = Math.sin(a), dy = Math.cos(a);
    S(shine, tilt([[x - dx * upL, y - dy * upL], [x + dx * dn, y + dy * dn]]), { size: 5 + rnd() * 4, color: '#ffffff', opacity: 0.9, a: 0.5, b: 0.5, segs: 1 });
  }
  clipL(shine, 'hair');
  const brows = L('brows', 'source-over', 0.75);
  const BROW = [[354, 372], [326, 361], [296, 360], [270, 369]];
  for (const q of [BROW, mirror(BROW)]) S(brows, tilt(q), { size: 5, color: C.hairLine, a: 0.08, b: 0.6 });

  // ---------- petals and sparkles in front
  const fx = L('fx');
  for (let i = 0; i < 14; i++) {
    const x = rnd() < 0.5 ? 20 + rnd() * 150 : 630 + rnd() * 150, y = 60 + rnd() * 880;
    F(fx, rotate(ellipse(x, y, 5, 9, 14), rnd() * 360, x, y), '#ffc9d9', { sharp: true, opacity: 0.85 });
  }
  for (let i = 0; i < 8; i++) {
    const x = 30 + rnd() * 740, y = 40 + rnd() * 300, s = 4 + rnd() * 8;
    S(fx, [[x - s, y], [x + s, y]], { size: s * 0.4, color: '#ffffff', a: 0.5, b: 0.5, segs: 1 });
    S(fx, [[x, y - s], [x, y + s]], { size: s * 0.4, color: '#ffffff', a: 0.5, b: 0.5, segs: 1 });
  }

  const out = flatten();
  glow(out, 10, 0.18);
  grain(out, 8);
  return out.toDataURL('image/png');
}
