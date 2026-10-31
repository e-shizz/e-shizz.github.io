// Runs a canvas drawing in headless Chromium; saves the painting and its recorded drawing actions.
// Usage: node canvas/run.js drawing.js out.png [actions.json] [reference.png]
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const [drawing, out, logOut, ref] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage();
  p.on('console', m => console.log('[page]', m.text()));
  p.on('pageerror', e => console.error('[page error]', e.message));
  await p.setContent('<html><body style="margin:0"></body></html>');
  await p.addScriptTag({ content: fs.readFileSync(path.join(__dirname, 'paint.js'), 'utf8') });
  if (ref) await p.evaluate(u => { window.REF_URL = u; }, 'data:image/png;base64,' + fs.readFileSync(ref).toString('base64'));
  await p.addScriptTag({ content: fs.readFileSync(drawing, 'utf8') });
  const t = Date.now();
  const res = await p.evaluate(async () => { const png = await draw(); return { png, log: JSON.stringify(LOG) }; });
  fs.writeFileSync(out, Buffer.from(res.png.split(',')[1], 'base64'));
  if (logOut && logOut !== '-') fs.writeFileSync(logOut, res.log);
  console.log(`painted in ${Date.now() - t} ms, ${JSON.parse(res.log).length} actions`);
  await b.close();
})();
