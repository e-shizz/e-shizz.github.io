// Usage: node render.js in.svg out.png [--transparent]   (size is read from the SVG's width/height)
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [inp, out, flag] = process.argv.slice(2);
  const clear = flag === '--transparent';
  const svg = fs.readFileSync(inp, 'utf8');
  const width = +(svg.match(/width="(\d+)"/) || [, 600])[1];
  const height = +(svg.match(/height="(\d+)"/) || [, 700])[1];
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width, height } });
  await p.setContent(`<html><body style="margin:0;background:${clear ? 'transparent' : '#fff'}">${svg}</body></html>`);
  await p.screenshot({ path: out, omitBackground: clear });
  await b.close();
})();
