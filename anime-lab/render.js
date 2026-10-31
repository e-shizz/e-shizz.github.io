// Usage: node render.js in.svg out.png   (size is read from the SVG's width/height)
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const svg = fs.readFileSync(inp, 'utf8');
  const width = +(svg.match(/width="(\d+)"/) || [, 600])[1];
  const height = +(svg.match(/height="(\d+)"/) || [, 700])[1];
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width, height } });
  await p.setContent(`<html><body style="margin:0;background:#fff">${svg}</body></html>`);
  await p.screenshot({ path: out });
  await b.close();
})();
