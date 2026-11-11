// Export reports/figures/observation_panels.html to
//   reports/figures/obs_importance_breadth.{pdf,svg,png}
//   reports/figures/obs_tier_dynamics.{pdf,svg,png}
//   reports/figures/obs_gate_recall.{pdf,svg,png}
// through the page's own export code (window.__obsFigures), so the files match
// what the page's buttons save: vector PDF with embedded fonts, standalone SVG,
// 600 dpi PNG. Also reports text that overflows its panel or collides.
//
//   node scripts/figures/export_observation_figures.mjs [--width 2.15] [--size 8]
//        [--family serif|sans] [--no-ci] [--titles] [--lib-dir DIR] [--check-only]
//
// The page loads jsPDF, svg2pdf.js and JSZip from public CDNs. On a machine
// without direct internet, pass --lib-dir with jspdf.umd.min.js,
// svg2pdf.umd.min.js and jszip.min.js and they are served from there.
// Needs Playwright with a Chromium (CHROMIUM_PATH overrides the bundled one).
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
const src = path.join(root, 'reports', 'figures', 'observation_panels.html');
const outDir = path.join(root, 'reports', 'figures');
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };
const settings = {
  widthIn: opt('--width', '2.15'),
  fontPt: Number(opt('--size', '8')),
  family: opt('--family', 'serif'),
  ci: !args.includes('--no-ci'),
  titles: args.includes('--titles'),
};
const libDir = opt('--lib-dir', null);
const checkOnly = args.includes('--check-only');

const launch = { headless: true };
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;
const browser = await chromium.launch(launch);
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
page.on('pageerror', (e) => console.error('page error:', e.message));
if (libDir) {
  const libs = { 'jspdf.umd.min.js': /jspdf/, 'svg2pdf.umd.min.js': /svg2pdf/, 'jszip.min.js': /jszip/ };
  await page.route(/cdnjs\.cloudflare\.com|cdn\.jsdelivr\.net/, (route) => {
    const url = route.request().url();
    const hit = Object.entries(libs).find(([, re]) => re.test(url));
    if (!hit) return route.abort();
    return route.fulfill({ contentType: 'application/javascript', body: fs.readFileSync(path.join(libDir, hit[0])) });
  });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.abort());
}
await page.goto('file://' + src);
await page.evaluate(() => window.__obsFigures.ready);
await page.evaluate((s) => window.__obsFigures.set(s), settings);

// Layout check on the rendered preview, with the page's own test: every text box inside its
// panel, none overlapping (parallel rotated category labels excepted).
const problems = await page.evaluate(() => window.__obsFigures.collisions()
  .flatMap((list, i) => list.map((m) => `(${'abc'[i]}) ${m}`)));
for (const p of problems) console.log(p);
console.log(`${problems.length} layout problem(s) at ${settings.widthIn} in, ${settings.fontPt} pt ${settings.family}`);

if (!checkOnly) {
  const names = { a: 'obs_importance_breadth', b: 'obs_tier_dynamics', c: 'obs_gate_recall' };
  for (const [k, name] of Object.entries(names)) {
    for (const fmt of ['pdf', 'svg', 'png']) {
      const b64 = await page.evaluate(([k, fmt]) => window.__obsFigures.file(k, fmt, 600), [k, fmt]);
      fs.writeFileSync(path.join(outDir, `${name}.${fmt}`), Buffer.from(b64, 'base64'));
    }
    console.log(`wrote ${path.join(outDir, name)}.{pdf,svg,png}`);
  }
}
await browser.close();
process.exit(problems.length && !checkOnly ? 1 : 0);
