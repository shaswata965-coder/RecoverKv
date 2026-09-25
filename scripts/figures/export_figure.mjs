// Export reports/figures/method_overview.html to .svg, .pdf (vector, one page)
// and .png (3x), and report any text that overflows its panel or collides with
// other text.
//
//   node scripts/figures/export_figure.mjs [--check-only] [--shots DIR]
//
// Needs Playwright with a Chromium; on a machine without the bundled browser set
// CHROMIUM_PATH to a local Chromium/Chrome binary.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  const globalRoot = require('child_process').execSync('npm root -g').toString().trim();
  ({ chromium } = require(path.join(globalRoot, 'playwright')));
}

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const src = path.join(root, 'reports', 'figures', 'method_overview.html');
const outBase = path.join(root, 'reports', 'figures', 'method_overview');
const args = process.argv.slice(2);
const checkOnly = args.includes('--check-only');
const shotsIdx = args.indexOf('--shots');
const shotsDir = shotsIdx >= 0 ? args[shotsIdx + 1] : null;

const launch = { headless: true };
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;
const browser = await chromium.launch(launch);
const page = await browser.newPage({ viewport: { width: 2400, height: 1600 }, deviceScaleFactor: 1 });
await page.goto('file://' + src);
// The canvas size lives in the SVG; everything below is sized from it.
const dims = await page.evaluate(() => {
  const s = document.getElementById('figure');
  return { w: +s.getAttribute('width'), h: +s.getAttribute('height') };
});
await page.setViewportSize({ width: dims.w, height: dims.h });
await page.evaluate(() => document.fonts.ready);

const { W, H, problems } = await page.evaluate(() => {
  const svg = document.getElementById('figure');
  const W = +svg.getAttribute('width'), H = +svg.getAttribute('height');
  svg.style.width = W + 'px';
  const out = [];
  const label = (t) => t.textContent.slice(0, 60);
  for (const p of svg.querySelectorAll('g.panel')) {
    const pw = +p.dataset.w, ph = +p.dataset.h;
    const texts = [...p.querySelectorAll(':scope > text')];
    const boxes = texts.map((t) => {
      const b = t.getBBox();
      const tr = t.getAttribute('transform');
      if (tr && tr.startsWith('rotate(-90')) {
        // rotate(-90 cx cy) maps (px, py) to (cx + (py - cy), cy - (px - cx))
        const cx = +t.getAttribute('x'), cy = +t.getAttribute('y');
        return { x: cx + (b.y - cy), y: cy - (b.x + b.width - cx), w: b.height, h: b.width, t, rot: true };
      }
      return { x: b.x, y: b.y, w: b.width, h: b.height, t };
    });
    const name = p.querySelector('text').nextElementSibling ? p.querySelectorAll('text')[1].textContent : '?';
    for (const b of boxes) {
      if (b.x < 4 || b.y < 4 || b.x + b.w > pw - 4 || b.y + b.h > ph - 4)
        out.push(`[${name}] overflow: "${label(b.t)}" x=${b.x.toFixed(0)}..${(b.x + b.w).toFixed(0)} (w ${pw}) y=${b.y.toFixed(0)}..${(b.y + b.h).toFixed(0)} (h ${ph})`);
    }
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (ox > 1.5 && oy > 2.5)
        out.push(`[${name}] overlap: "${label(a.t)}"  <>  "${label(b.t)}"`);
    }
  }
  return { W, H, problems: out };
});
for (const p of problems) console.log(p);
console.log(`${problems.length} layout problem(s)`);

if (shotsDir) {
  fs.mkdirSync(shotsDir, { recursive: true });
  await page.screenshot({ path: path.join(shotsDir, 'full.png'), clip: { x: 0, y: 0, width: W, height: H } });
  const panels = await page.evaluate(() => [...document.querySelectorAll('g.panel')].map((p) => {
    const m = p.getAttribute('transform').match(/translate\(([\d.]+),([\d.]+)\)/);
    return { x: +m[1], y: +m[2], w: +p.dataset.w, h: +p.dataset.h };
  }));
  const hi = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  await hi.goto('file://' + src);
  await hi.evaluate((w) => { document.getElementById('figure').style.width = w + 'px'; }, W);
  for (const [k, p] of panels.entries()) {
    await hi.screenshot({ path: path.join(shotsDir, `panel_${k}.png`),
      clip: { x: p.x - 4, y: p.y - 4, width: p.w + 8, height: p.h + 8 } });
  }
}

if (!checkOnly) {
  const svgText = await page.evaluate(() => {
    const c = document.getElementById('figure').cloneNode(true);
    c.removeAttribute('style');
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(c);
  });
  fs.writeFileSync(outBase + '.svg', svgText);
  await page.emulateMedia({ media: 'print' });
  await page.pdf({ path: outBase + '.pdf', width: `${W}px`, height: `${H}px`, printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }, pageRanges: '1' });
  await page.emulateMedia({ media: 'screen' });
  const png = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 3 });
  await png.goto('file://' + src);
  await png.evaluate((w) => { document.getElementById('figure').style.width = w + 'px'; }, W);
  await png.screenshot({ path: outBase + '.png', clip: { x: 0, y: 0, width: W, height: H } });
  console.log(`wrote ${outBase}.{svg,pdf,png}`);
}
await browser.close();
