// Export the method figures (reports/figures/<name>.html) to .svg, .pdf (vector,
// one page, at print size), .png (600 dpi) and .xml (a diagrams.net / draw.io file
// of native, editable shapes, built by svg_to_drawio.js), and fail on any label
// that is too small to print, overflows its panel, or collides with other text.
//
//   node scripts/figures/export_figure.mjs [name ...] [--check-only] [--shots DIR]
//
// Figures are drawn in points (1 unit = 1 pt, printed at the figure's own width:
// 6.75 in for method_overview, 5.5 in for method_details), so the size check reads
// font sizes straight off the SVG: labels >= 8 pt, subscripts >= 7 pt.
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

const MIN_PT = 8, MIN_SUB_PT = 7, PNG_DPI = 600;

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const args = process.argv.slice(2);
const checkOnly = args.includes('--check-only');
const shotsIdx = args.indexOf('--shots');
const shotsDir = shotsIdx >= 0 ? args[shotsIdx + 1] : null;
const named = args.filter((a, i) => !a.startsWith('--') && !(shotsIdx >= 0 && i === shotsIdx + 1));
const names = named.length ? named : ['method_overview', 'method_details'];

const launch = { headless: true };
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;
const browser = await chromium.launch(launch);

// A page showing the figure at one CSS px per point (so SVG units == CSS px).
async function open(src, scale) {
  const probe = await browser.newPage();
  await probe.goto('file://' + src);
  const { W, H } = await probe.evaluate(() => {
    const vb = document.getElementById('figure').viewBox.baseVal;
    return { W: vb.width, H: vb.height };
  });
  await probe.close();
  const page = await browser.newPage({ viewport: { width: Math.ceil(W), height: Math.ceil(H) },
    deviceScaleFactor: scale });
  await page.goto('file://' + src);
  await page.evaluate((w) => { document.getElementById('figure').style.width = w + 'px'; }, W);
  await page.evaluate(() => document.fonts.ready);
  return { page, W, H };
}

let failed = 0;
for (const name of names) {
  const src = path.join(root, 'reports', 'figures', name + '.html');
  const outBase = path.join(root, 'reports', 'figures', name);
  const { page, W, H } = await open(src, 1);

  const problems = await page.evaluate(({ MIN_PT, MIN_SUB_PT }) => {
    const svg = document.getElementById('figure');
    const out = [];
    const label = (t) => t.textContent.slice(0, 60);
    const sizeOf = (el) => {
      for (let e = el; e && e !== svg.parentNode; e = e.parentNode) {
        const a = e.getAttribute && e.getAttribute('font-size');
        if (a != null) return +a;
      }
      return 16;
    };
    // print size: every text node, and every subscript
    for (const t of svg.querySelectorAll('text')) {
      if (t.closest('defs')) continue;
      const plain = [...t.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())
        || [...t.querySelectorAll('tspan')].some((s) => !s.classList.contains('sub') && s.textContent.trim());
      if (plain && sizeOf(t) < MIN_PT) out.push(`size: "${label(t)}" is ${sizeOf(t)} pt (< ${MIN_PT})`);
      for (const s of t.querySelectorAll('tspan.sub'))
        if (sizeOf(s) < MIN_SUB_PT) out.push(`size: subscript "${s.textContent}" of "${label(t)}" is ${sizeOf(s)} pt (< ${MIN_SUB_PT})`);
    }
    const boxOf = (t) => {
      const b = t.getBBox();
      const tr = t.getAttribute('transform');
      if (tr && tr.startsWith('rotate(-90')) {
        // rotate(-90 cx cy) maps (px, py) to (cx + (py - cy), cy - (px - cx))
        const cx = +t.getAttribute('x'), cy = +t.getAttribute('y');
        return { x: cx + (b.y - cy), y: cy - (b.x + b.width - cx), w: b.height, h: b.width, t };
      }
      return { x: b.x, y: b.y, w: b.width, h: b.height, t };
    };
    for (const p of svg.querySelectorAll('g.panel')) {
      const pw = +p.dataset.w, ph = +p.dataset.h;
      const boxes = [...p.querySelectorAll(':scope > text')].map(boxOf);
      const name = p.querySelector('text').textContent.slice(0, 3);
      for (const b of boxes) {
        if (b.x < 2 || b.y < 1.5 || b.x + b.w > pw - 2 || b.y + b.h > ph - 1.5)
          out.push(`${name} overflow: "${label(b.t)}" x=${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} (w ${pw}) y=${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)} (h ${ph})`);
      }
      for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
        const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
        if (ox > 0.8 && oy > 1.8)
          out.push(`${name} overlap: "${label(a.t)}"  <>  "${label(b.t)}"`);
      }
    }
    return out;
  }, { MIN_PT, MIN_SUB_PT });
  for (const p of problems) console.log(`[${name}] ${p}`);
  console.log(`[${name}] ${W} x ${H} pt (${(W / 72).toFixed(2)} x ${(H / 72).toFixed(2)} in): ` +
    `${problems.length} problem(s)`);
  failed += problems.length;

  if (shotsDir) {
    fs.mkdirSync(shotsDir, { recursive: true });
    const { page: hi } = await open(src, 5);
    await hi.screenshot({ path: path.join(shotsDir, `${name}.png`), clip: { x: 0, y: 0, width: W, height: H } });
    const panels = await hi.evaluate(() => [...document.querySelectorAll('g.panel')].map((p) => {
      const m = p.getAttribute('transform').match(/translate\(([\d.]+),([\d.]+)\)/);
      return { x: +m[1], y: +m[2], w: +p.dataset.w, h: +p.dataset.h };
    }));
    for (const [k, p] of panels.entries())
      await hi.screenshot({ path: path.join(shotsDir, `${name}_panel_${k}.png`),
        clip: { x: Math.max(0, p.x - 2), y: Math.max(0, p.y - 2), width: p.w + 4, height: p.h + 4 } });
    await hi.close();
  }

  if (!checkOnly) {
    const svgText = await page.evaluate(() => {
      const c = document.getElementById('figure').cloneNode(true);
      c.removeAttribute('style');
      return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(c);
    });
    fs.writeFileSync(outBase + '.svg', svgText);
    const conv = fs.readFileSync(path.join(here, 'svg_to_drawio.js'), 'utf8');
    const xml = await page.evaluate(conv + `\n;svgToDrawio(document.getElementById("figure"), ${JSON.stringify(name)});`);
    fs.writeFileSync(outBase + '.xml', xml);
    // PDF at print size: the page's own print CSS sets the figure to W/72 in wide
    await page.evaluate(() => document.getElementById('figure').removeAttribute('style'));
    await page.emulateMedia({ media: 'print' });
    await page.pdf({ path: outBase + '.pdf', width: `${W / 72}in`, height: `${H / 72}in`,
      printBackground: true, margin: { top: 0, right: 0, bottom: 0, left: 0 }, pageRanges: '1' });
    const { page: png } = await open(src, PNG_DPI / 72);
    await png.screenshot({ path: outBase + '.png', clip: { x: 0, y: 0, width: W, height: H } });
    await png.close();
    console.log(`[${name}] wrote ${path.relative(root, outBase)}.{svg,pdf,png,xml}`);
  }
  await page.close();
}
await browser.close();
process.exitCode = failed ? 1 : 0;
