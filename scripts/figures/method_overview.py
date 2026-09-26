"""Build the method figures at print size: Figure 1 is 6.75 in (486 pt) wide, the
full width of a two-column page (a figure*); Figure 2 is 5.5 in (396 pt).

Two figures, one unit = one printed point at that width, so a font size here IS
its size on the page. Labels are >= 8 pt, panel titles 9 pt; the only smaller
text is a subscript (>= 7 pt, LaTeX scriptsize at a 10 pt body).
``export_figure.mjs`` fails the build on anything below that, or on text that
collides. Scaled into a narrower text width, Figure 1's labels print smaller
(at 6.3 in: 8 pt -> 7.5 pt).

``method_overview``  -- Figure 1, the runtime loop, nothing else:
  (a) cumulative attention ranks the windows: the attention map (keys across,
      one row per query token), one window's score update for this step
      (alpha_1..alpha_n summed into alpha_window, added to alpha_history), and
      the ranked scores standing on the very cache windows they tier.
  (b) the read gate: the query heads score every card, each head's shares are
      unioned, and the top windows (the gate ratio) are opened.
  (c) one fused pass per token: re-ranked at each window boundary, read in one
      kernel (fp16 and opened int2 in full, the rest through their cards), and
      the next token appended; every window's attention loops back to (a).
  Layout: (c) sits left of (b), so the loop runs clockwise -- down from (a) into
  (b), across into (c), and up the left margin back into (a)'s score update. One
  key under (c) and (b) explains the marks both use.
  Lines: the flow between panels is block arrows, so it never reads as a
  connector. Inside (c) every line is a short straight drop: the re-rank's
  trigger is written in its box, the kernel's row sits under the windows it
  reads (no arrow per window), and the new K, V go straight up to their slot.

``method_details``  -- Figure 2, the mechanisms Figure 1 leaves out:
  (a) demoting a window: its keys as one cluster -> the card; the parts of an
      int2 window, to scale against the same window in fp16.
  (b) crediting unread windows: the card's error measured on the opened ones.
  (c) the byte budget the four cache sections share.

No label names a configuration value (window size, quant ratio, budget, gate
ratio). The drawing is proportioned from one real setting (2 fp16 against 8
int2 windows, 2 of 8 opened), but that is geometry, never text.

Palette: fp16 blue, int2 violet, card green, sink slate, query / new token red,
dropped a dashed outline; every label and mark one solid ink, no greys.

Figure 1's labels are upright (no italics) and start with a capital; a dtype
(fp16, int2) or a symbol (alpha) keeps its own case.

Run:  python scripts/figures/method_overview.py && node scripts/figures/export_figure.mjs
"""

from __future__ import annotations

import html
import math
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

# ---------------------------------------------------------------------------
# palette -- every notation is one solid ink: no greys, no in-between tints
# ---------------------------------------------------------------------------
INK = "#1d2329"
SINK_F, SINK_S = "#e2e6eb", "#5b6878"
FP_F, FP_S, FP_T = "#cfe0f3", "#3c6ea8", "#2c5a8f"
LOC_F = "#eef4fb"
Q_F, Q_S, Q_T, Q_L = "#dccff2", "#6446a4", "#51368a", "#f4f0fb"
CARD_F, CARD_S, CARD_T = "#d4ecdc", "#3b8a5a", "#2a6d45"
QRY_F, QRY_S, QRY_T = "#f6cdc9", "#c0392b", "#a93226"
BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
         "#2171b5", "#08519c", "#08306b"]
PURPLES = ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba",
           "#6a51a3", "#54278f", "#3f007d"]
SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"
HATCH = {"blue": (LOC_F, "#b8cfe8"), "violet": (Q_L, "#b9a8e0")}

FS, FT = 8, 9            # label and panel-title sizes, in points


def ramp(stops, t):
    t = max(0.0, min(1.0, t))
    x = t * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    ca = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    cb = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(p + (q - p) * f):02x}" for p, q in zip(ca, cb))


def hatch_lines(x, y, w, h, ink, gap=2.2, sw=0.5):
    """45-degree hatching clipped to a rectangle, as one vector path (not an SVG
    <pattern>: Chromium's PDF backend rasterises pattern fills)."""
    x1, y1 = x + w, y + h
    step = gap * math.sqrt(2)
    k = x + y + step / 2
    d = []
    while k < x1 + y1:
        xa, xb = max(x, k - y1), min(x1, k - y)
        if xb - xa > 0.1:
            d.append(f"M{xa:.2f},{k - xa:.2f}L{xb:.2f},{k - xb:.2f}")
        k += step
    return f'<path d="{"".join(d)}" stroke="{ink}" stroke-width="{sw}" fill="none"/>'


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
class G:
    def __init__(self):
        self.p = []

    def add(self, s):
        self.p.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=0.6, rx=0, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        if fill.startswith("hatch-"):
            bg, ink = HATCH[fill[6:]]
            self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                     f'rx="{rx}" fill="{bg}"/>')
            self.add(hatch_lines(x, y, w, h, ink))
            fill = "none"
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=0.7, dash=None, arrow=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{arrow})"' if arrow else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{m}{extra}/>')

    def path(self, d, stroke=INK, sw=0.7, fill="none", dash=None, arrow=None, extra=""):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{arrow})"' if arrow else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}{m}{extra}/>')

    def circle(self, cx, cy, r, fill="none", stroke="none", sw=0.6, extra=""):
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{extra}/>')

    def text(self, x, y, s, size=FS, anchor="start", weight=None, fill=INK, rotate=None,
             extra=""):
        w = f' font-weight="{weight}"' if weight else ""
        rot = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{fill}"{w}{rot}{extra}>{html.escape(s)}</text>')

    def text_parts(self, x, y, parts, size=9, anchor="start", weight=None, fill=INK):
        """Text with subscripts, ``parts = [(text, is_sub), ...]``. A subscript is a
        ``<tspan class="sub">`` at 0.8x (>= 7 pt at a 9 pt base); the draw.io export
        turns the same class into ``<sub>``."""
        w = f' font-weight="{weight}"' if weight else ""
        out, low = [], False
        for txt, sub in parts:
            if sub:
                dy = "" if low else f' dy="{size * 0.28:.2f}"'
                out.append(f'<tspan class="sub" font-size="{size * 0.8:.2f}"{dy}>'
                           f"{html.escape(txt)}</tspan>")
                low = True
            else:
                dy = f' dy="{-size * 0.28:.2f}"' if low else ""
                out.append(f"<tspan{dy}>{html.escape(txt)}</tspan>")
                low = False
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{fill}"{w}>{"".join(out)}</text>')

    def use(self, sym, x, y, w, h, extra=""):
        self.add(f'<use href="#{sym}" xlink:href="#{sym}" x="{x:.2f}" y="{y:.2f}" '
                 f'width="{w}" height="{h}"{extra}/>')

    def open(self, x, y, extra=""):
        self.add(f'<g transform="translate({x},{y})"{extra}>')

    def close(self):
        self.add("</g>")


def xmark(g, x, y, r=2.4, stroke=INK, sw=0.9):
    g.path(f"M{x - r},{y - r} L{x + r},{y + r} M{x - r},{y + r} L{x + r},{y - r}",
           stroke=stroke, sw=sw)


def block_arrow(g, x0, y0, x1, y1, shaft=4.5, head=12, hl=9, fill=INK):
    """A filled arrow from (x0, y0) to its tip at (x1, y1): the flow between panels, a
    shape rather than a line, so it never reads as one of a panel's connectors."""
    ln = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / ln, (y1 - y0) / ln
    nx, ny = -dy, dx
    bx, by = x1 - dx * hl, y1 - dy * hl
    pts = [(x0 + nx * shaft / 2, y0 + ny * shaft / 2), (bx + nx * shaft / 2, by + ny * shaft / 2),
           (bx + nx * head / 2, by + ny * head / 2), (x1, y1),
           (bx - nx * head / 2, by - ny * head / 2), (bx - nx * shaft / 2, by - ny * shaft / 2),
           (x0 - nx * shaft / 2, y0 - ny * shaft / 2)]
    g.path("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z", stroke="none", fill=fill)


def bracket(g, x1, x2, y, up=True, stroke=INK, sw=0.6, tick=2.5):
    s = 1 if up else -1
    g.path(f"M{x1},{y + s * tick} L{x1},{y} L{x2},{y} L{x2},{y + s * tick}", stroke=stroke, sw=sw)


def frame(g, w, h, tag, title, sub=None, sub_right=False):
    g.rect(0, 0, w, h, fill="#ffffff", stroke=INK, sw=0.6, rx=4)
    g.text(6, 11, f"{tag}  {title}", size=FT, weight="bold")
    if sub:
        if sub_right:
            g.text(w - 6, 11, sub, size=FS, anchor="end")
        else:
            g.text(6, 21, sub, size=FS)


# ---------------------------------------------------------------------------
# the toy cache: one layout, reused by every panel
# ---------------------------------------------------------------------------
N_FP, N_Q, N_LOC, N_DROP = 2, 8, 2, 3
N_SEL = 2                              # the gate opens a quarter of the int2 windows

# gate: each query head's share of its own int2 mass, per int2 window
_g = random.Random(5)
_PEAKS = [{1: 1.4}, {1: 0.7, 4: 1.0}, {4: 0.8}, {6: 3.8}]       # head 4: a retrieval head
SHARE = []
for pk in _PEAKS:
    row = [_g.gauss(0, 0.3) + pk.get(c, 0.0) for c in range(N_Q)]
    m = max(row)
    z = sum(math.exp(v - m) for v in row)
    SHARE.append([math.exp(v - m) / z for v in row])
UNION = [max(SHARE[h][c] for h in range(4)) for c in range(N_Q)]
SEL = sorted(sorted(range(N_Q), key=lambda c: -UNION[c])[:N_SEL])


class Cache:
    """Geometry of one cache bar: [sink | fp16 | int2 + cards | (dropped) | local].

    An int2 window is drawn ~1/4 the height of an fp16 one: its price with the
    card included.
    """

    def __init__(self, x0, yb, sink_w, fp_w, q_w, loc_w, fp_h, n_drop=0, gap=4.0, wgap=1.6):
        self.yb, self.fp_w, self.q_w, self.loc_w = yb, fp_w, q_w, loc_w
        self.fp_h = fp_h
        self.q_h = fp_h / 4
        self.card = min(q_w - 1, fp_h * 0.45)
        x = x0
        self.sink = (x, x + 5 * sink_w)
        self.sink_w = sink_w
        x += 5 * sink_w + gap
        self.fp = [x + i * (fp_w + wgap) for i in range(N_FP)]
        x = self.fp[-1] + fp_w + gap
        self.q = [x + i * (q_w + wgap) for i in range(N_Q)]
        x = self.q[-1] + q_w + gap
        self.drop = [x + i * (q_w + wgap) for i in range(n_drop)]
        if n_drop:
            x = self.drop[-1] + q_w + gap
        self.loc = [x + i * (loc_w + wgap) for i in range(N_LOC)]
        self.new = self.loc[-1] + loc_w + 1.2
        self.x1 = self.new + 2.4

    # spans and centres
    def span(self, key):
        return {"sink": self.sink, "fp": (self.fp[0], self.fp[-1] + self.fp_w),
                "q": (self.q[0], self.q[-1] + self.q_w),
                "drop": (self.drop[0], self.drop[-1] + self.q_w) if self.drop else None,
                "loc": (self.loc[0], self.x1)}[key]

    def cx(self, key):
        a, b = self.span(key)
        return (a + b) / 2

    def qx(self, c):
        return self.q[c] + self.q_w / 2

    def card_top(self):
        return self.yb - self.q_h - 1 - self.card

    def draw(self, g, mode="plain", fill_newest=False, traced=None):
        yb, fh = self.yb, self.fp_h
        for k in range(5):
            g.rect(self.sink[0] + k * self.sink_w, yb - fh, self.sink_w, fh, fill=SINK_F,
                   stroke=SINK_S, sw=0.4)
        for x in self.fp:
            g.rect(x, yb - fh, self.fp_w, fh, fill=FP_F, stroke=FP_S, sw=0.6)
            for t in range(1, 8):
                g.line(x + t * self.fp_w / 8, yb - fh + 1, x + t * self.fp_w / 8, yb - 1,
                       stroke="#ffffff", sw=0.35)
        for c, x in enumerate(self.q):
            skipped = mode == "gated" and c not in SEL
            if skipped:
                g.add('<g opacity="0.45">')
            g.rect(x, yb - self.q_h, self.q_w, self.q_h, fill=Q_F, stroke=Q_S, sw=0.6,
                   dash="1.5 1" if skipped else None)
            if skipped:
                g.add("</g>")
            g.use("card", x + (self.q_w - self.card) / 2, self.card_top(), self.card, self.card)
            if (mode == "gated" and c in SEL) or traced == c:
                g.rect(x - 0.9, self.card_top() - 0.9, self.q_w + 1.8,
                       yb - self.card_top() + 1.4, stroke=INK, sw=0.9, rx=1)
        for x in self.drop:
            g.rect(x, yb - fh * 0.6, self.q_w, fh * 0.6, stroke=INK, sw=0.5, dash="1.5 1")
            xmark(g, x + self.q_w / 2, yb - fh * 0.3, r=2.2, sw=0.7)
        for i, x in enumerate(self.loc):
            newest = i == len(self.loc) - 1
            if fill_newest and newest:                       # filling, one token per step
                tw = self.loc_w / 8
                g.rect(x, yb - fh, 5 * tw, fh, fill="hatch-blue", stroke="none")
                g.rect(x + 5 * tw, yb - fh, tw, fh, fill=QRY_F, stroke=QRY_S, sw=0.5)
                g.rect(x, yb - fh, self.loc_w, fh, stroke=FP_S, sw=0.6)
            else:
                g.rect(x, yb - fh, self.loc_w, fh, fill="hatch-blue", stroke=FP_S, sw=0.6)
        if not fill_newest:
            g.rect(self.new, yb - fh, 2.4, fh, fill=QRY_F, stroke=QRY_S, sw=0.5)
        g.line(self.sink[0] - 2, yb, self.x1 + 2, yb, stroke=INK, sw=0.5)


def page(svg, w, h, title):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  @page {{ size: {w / 72:.4f}in {h / 72:.4f}in; margin: 0; }}
  html, body {{ margin: 0; padding: 0; background: #ffffff; }}
  body {{ display: flex; justify-content: center; }}
  #figure {{ display: block; width: min(100vw, 1400px); height: auto; }}
  @media print {{ #figure {{ width: {w / 72:.4f}in; height: {h / 72:.4f}in; }} }}
</style>
</head>
<body>
{svg}
<script>
/* Export without any on-page UI:  S = .svg,  P = .png (600 dpi),  Ctrl/Cmd+P = vector PDF. */
(function () {{
  const svg = document.getElementById('figure');
  function save(blob, name) {{
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }}
  function svgText() {{
    const c = svg.cloneNode(true);
    c.removeAttribute('style');
    return '<?xml version="1.0" encoding="UTF-8"?>\\n' + new XMLSerializer().serializeToString(c);
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 's') {{
      save(new Blob([svgText()], {{type: 'image/svg+xml'}}), '{title}.svg');
    }} else if (k === 'p') {{
      const img = new Image(), scale = 600 / 72;
      img.onload = () => {{
        const cv = document.createElement('canvas');
        cv.width = Math.round({w} * scale); cv.height = Math.round({h} * scale);
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, cv.width, cv.height);
        ctx.drawImage(img, 0, 0, cv.width, cv.height);
        cv.toBlob((b) => save(b, '{title}.png'), 'image/png');
      }};
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgText());
    }}
  }});
}})();
</script>
</body>
</html>
"""


def defs():
    def marker(name, col):
        return (f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8.6" refY="5" '
                f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0,0.6 L10,5 L0,9.4 z" fill="{col}"/></marker>')

    # the card: the window's keys as one cluster -- centroid, and the axis to its outlier
    card = (f'<symbol id="card" viewBox="0 0 20 20">'
            f'<rect x="0.8" y="0.8" width="18.4" height="18.4" rx="3.2" fill="{CARD_F}" '
            f'stroke="{CARD_S}" stroke-width="1.4"/>'
            f'<circle cx="5.2" cy="12.6" r="1.6" fill="#4f7fb5"/>'
            f'<circle cx="8.4" cy="15.2" r="1.6" fill="#4f7fb5"/>'
            f'<circle cx="4.6" cy="16" r="1.6" fill="#4f7fb5"/>'
            f'<circle cx="9" cy="11.2" r="1.6" fill="#4f7fb5"/>'
            f'<path d="M6.8,13.8 L14.6,6.2" stroke="{CARD_T}" stroke-width="1.6"/>'
            f'<circle cx="15.4" cy="5.2" r="2.2" fill="{QRY_S}"/>'
            f'<path d="M5.5,12.5 L8.1,15.1 M5.5,15.1 L8.1,12.5" stroke="{INK}" stroke-width="1.2"/>'
            f'</symbol>')
    return ("<defs>" + marker("dark", INK) + marker("slate", SINK_S) + marker("blue", FP_S)
            + marker("violet", Q_S)
            + marker("green", CARD_S) + marker("red", QRY_S) + card + "</defs>")


def svg_doc(g, w, h):
    return (f'<svg id="figure" xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}pt" height="{h}pt" '
            f'viewBox="0 0 {w} {h}" font-family="{SANS}">\n' + defs() + "\n"
            + f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n'
            + "\n".join(g.p) + "\n</svg>")


# ===========================================================================
# Figure 1 -- the runtime loop
# ===========================================================================
W1 = 486                                 # 6.75 in: the full width of a two-column page
LM = 16                                  # left margin: carries the loop back to (a)
XL = 8                                   # the loop's run up the margin, 8 pt clear of the borders
GX = GY = 28                             # gaps between panels, where the flow arrows run
AW, AH = W1 - LM - 1, 135
BW = (AW - GX) / 2
XC, XB = LM, LM + BW + GX                # (c) left, (b) right: the loop runs clockwise
BY = AH + GY
BH = 166
KEY_Y = BY + BH + 15                     # one key under (c) and (b): its marks are in both
H1 = KEY_Y + 5

# (a)'s cache: wide, with the dropped windows still visible as ghosts
CA = Cache(x0=146, yb=93, sink_w=2.2, fp_w=21, q_w=14, loc_w=21, fp_h=20, n_drop=N_DROP,
           gap=7.5, wgap=2.2)
TRACED = 6                               # the int2 window whose score update is shown:
                                         # far right, so the row under it leaves room for
                                         # the loop's arrow and its label
A_ROW = 105                              # (a)'s alpha row
A_SPAN = 224.5                           # its first cell -> the middle of "New score"
# (b) and (c): the same cache, narrower, after the eviction. (c) draws it higher: its
# re-rank box above is short, its kernel box below is level with (b)'s cache, where the
# gate's pick comes in
_CB = dict(x0=9, sink_w=1.8, fp_w=16, q_w=11.5, loc_w=16, fp_h=20, gap=5, wgap=1.8)
CB = Cache(yb=138, **_CB)
CC = Cache(yb=100, **_CB)
KY0 = CC.yb + 18                         # (c)'s kernel box; above it the new K, V go up
KY1 = KY0 + 42


def fig1_a(g):
    frame(g, AW, AH, "(a)", "Cumulative attention ranks the windows",
          "Re-tiered at every window boundary", sub_right=True)

    # ---- the attention map: columns are keys, each row one query token
    N, SINK, WSZ, cell = 18, 2, 8, 3.3
    N_PROMPT = 9
    hx, hy = 50, 31
    rng = random.Random(3)
    P = []
    for i in range(N):
        lg = [rng.gauss(0, 0.45) + (2.1 if j < SINK else 0) + (2.0 if j == 5 else 0)
              + (1.1 if j == 13 else 0) + (1.2 if i - j <= 1 else 0) for j in range(i + 1)]
        m = max(lg)
        ex = [math.exp(v - m) for v in lg]
        s = sum(ex)
        P.append([e / s for e in ex] + [0.0] * (N - i - 1))
    pmax = max(P[i][j] for i in range(N) for j in range(SINK, i + 1))
    for i in range(N):
        for j in range(N):
            col = "#ffffff" if j > i else ramp(BLUES, (P[i][j] / pmax) ** 0.6)
            g.rect(hx + j * cell, hy + i * cell, cell, cell, fill=col, stroke="#ffffff", sw=0.25)
    hw = N * cell
    g.rect(hx, hy, hw, hw, stroke=INK, sw=0.5)
    g.text(hx, hy - 4, "Keys (cached)", size=FS)
    g.line(hx + 53.5, hy - 6.5, hx + hw, hy - 6.5, stroke=INK, sw=0.7, arrow="dark")
    g.text(hx - 23, hy + hw / 2, "Query tokens", size=FS, anchor="middle", rotate=-90)
    yp = hy + N_PROMPT * cell
    for (y0, y1, lab, col) in ((hy, yp, "Prompt", INK), (yp, hy + hw, "Decode", QRY_S)):
        g.path(f"M{hx - 1},{y0 + 0.6} L{hx - 3.5},{y0 + 0.6} L{hx - 3.5},{y1 - 0.6} "
               f"L{hx - 1},{y1 - 0.6}", stroke=col, sw=0.7)
        g.text(hx - 7, (y0 + y1) / 2, lab, size=FS, anchor="middle", fill=col, rotate=-90)
    # this step's row, over one window's keys: the alphas below
    wx0 = hx + SINK * cell
    g.rect(wx0, hy + (N - 1) * cell, WSZ * cell, cell, stroke=INK, sw=0.9)

    # ---- one window's score update, this step
    cw, cg_, ay, ah = 14, 2, A_ROW, 14
    ax0 = CA.qx(TRACED) - A_SPAN              # so "New score" sits under its window
    g.path(f"M{wx0},{hy + hw} L{ax0},{ay} M{wx0 + WSZ * cell},{hy + hw} "
           f"L{ax0 + 5 * cw + 4 * cg_},{ay}", stroke=INK, sw=0.5, dash="1.5 1.2")
    shades = [0.30, 0.85, 0.45, None, 0.70]
    subs = ["1", "2", "3", None, "n"]
    x = ax0
    for v, sb in zip(shades, subs):
        if v is None:
            g.text(x + cw / 2, ay + 10, "…", size=9, anchor="middle")
        else:
            g.rect(x, ay, cw, ah, fill=ramp(BLUES, v), stroke=FP_S, sw=0.5)
            g.text_parts(x + cw / 2, ay + 9.6, [("α", False), (sb, True)], size=9,
                         anchor="middle", fill="#ffffff" if v > 0.6 else INK)
        x += cw + cg_
    x += 0.5
    g.line(x, ay + ah / 2, x + 14, ay + ah / 2, stroke=INK, sw=0.7, arrow="dark")
    g.text(x + 7, ay - 2.5, "Sum", size=FS, anchor="middle")
    x += 16

    def chip(x0, wd, parts, fill, stroke, bold=False, sw=0.6, tcol=INK):
        g.rect(x0, ay, wd, ah, fill=fill, stroke=stroke, sw=sw, rx=2)
        g.text_parts(x0 + wd / 2, ay + 9.8, parts, size=9, anchor="middle", fill=tcol,
                     weight="bold" if bold else None)
        return x0 + wd

    c1 = x
    x = chip(x, 38, [("α", False), ("window", True)], Q_S, Q_S, tcol="#ffffff")
    g.text(x + 6.5, ay + 10.5, "+", size=10, anchor="middle", weight="bold")
    c2 = x + 13
    x = chip(c2, 40, [("α", False), ("history", True)], Q_F, Q_S)
    g.text(x + 6.5, ay + 10.5, "=", size=10, anchor="middle", weight="bold")
    c3 = x + 13
    chip(c3, 48, [("New score", False)], "#ffffff", Q_S, bold=True, sw=1.1)
    ly = ay + ah + 10
    g.text(ax0 + (5 * cw + 4 * cg_) / 2, ly, "This step's row", size=FS, anchor="middle")
    g.text(c1 + 19, ly, "Summed", size=FS, anchor="middle")
    g.text(c2 + 20, ly, "Earlier steps", size=FS, anchor="middle")
    g.text(c3 + 24, ly, "Ranks it", size=FS, anchor="middle")
    g.line(CA.qx(TRACED), ay - 1, CA.qx(TRACED), CA.yb + 1.5, stroke=Q_S, sw=0.9, arrow="violet")
    # the loop from (c) comes in from the left at mid-row, labelled just above it
    g.text(4, ay + ah / 2 - 4, "Fused attention score", size=FS, fill=FP_T)

    # ---- the ranked scores, standing on the windows they tier
    bb, bmax = CA.yb - CA.fp_h - 4, 24
    n = N_FP + N_Q + N_DROP
    scores = [1.0 - 0.85 * (k / (n - 1)) ** 0.85 for k in range(n)]
    cols = ([(x_, CA.fp_w, "fp") for x_ in CA.fp] + [(x_, CA.q_w, "q") for x_ in CA.q]
            + [(x_, CA.q_w, "drop") for x_ in CA.drop])
    q_scores = sorted(scores[N_FP:N_FP + N_Q], reverse=True)
    q_by_col = {c: q_scores[(c * 5 + 2) % N_Q] for c in range(N_Q)}   # windows in time order
    rr = random.Random(7)
    for k, (x_, wd, kd) in enumerate(cols):
        v = q_by_col[k - N_FP] if kd == "q" else scores[k]
        hgt = bmax * v
        if kd == "drop":
            g.rect(x_, bb - hgt, wd, hgt, fill="#ffffff", stroke=INK, sw=0.5, dash="1.5 1")
            continue
        pale, solid = (FP_F, FP_S) if kd == "fp" else (Q_F, Q_S)
        cap = hgt * rr.uniform(0.18, 0.3)                 # this step's share of the score
        g.rect(x_, bb - hgt + cap, wd, hgt - cap, fill=pale, stroke="none")
        g.rect(x_, bb - hgt, wd, cap, fill=solid, stroke="none")
        tr = kd == "q" and k - N_FP == TRACED
        g.rect(x_, bb - hgt, wd, hgt, stroke=INK if tr else solid, sw=0.9 if tr else 0.5)
    g.line(CA.fp[0] - 2, bb, CA.drop[-1] + CA.q_w + 2, bb, stroke=INK, sw=0.5)
    # tier names, straight over their windows
    for key, top, bot, col in (("sink", None, "Sink", SINK_S), ("fp", "Top-k", "fp16", FP_T),
                               ("q", "Next top-k", "int2 + card", Q_T),
                               ("drop", "Rest", "Dropped", INK), ("loc", "Newest", "Local", FP_T)):
        if top:
            g.text(CA.cx(key), 32, top, size=FS, anchor="middle", fill=col, weight="bold")
        g.text(CA.cx(key), 41, bot, size=FS, anchor="middle", fill=col)
    CA.draw(g, traced=TRACED)


def fig1_b(g):
    frame(g, BW, BH, "(b)", "Cards choose what to read", "Per KV head, every decode step")
    top = CB.card_top()

    # the query heads of one GQA group score every card
    qx, qy, qw = CB.cx("q") - 13, 28, 26
    for h in range(4):
        g.rect(qx, qy + h * 3.6, qw, 2.8, fill=QRY_F, stroke=QRY_S, sw=0.5, rx=0.8)
    g.text(qx - 4, qy + 10, "Query heads", size=FS, anchor="end", fill=QRY_T, weight="bold")
    gy, ch, pitch = qy + 25, 7, 9.8
    for c in range(N_Q):
        g.line(CB.cx("q"), qy + 14.5, CB.qx(c), gy - 1.5, stroke=INK, sw=0.4)
    g.text(CB.span("q")[1] + 4, qy + 10, "Score", size=FS)
    g.text(CB.span("q")[1] + 4, qy + 19, "every card", size=FS)

    # each head's share of the cards, then the union across the group
    for h in range(4):
        g.text(CB.span("q")[0] - 3, gy + h * pitch + 6.2, f"Head {h + 1}", size=FS, anchor="end")
        for c, x in enumerate(CB.q):
            g.rect(x, gy + h * pitch, CB.q_w, ch, fill=ramp(PURPLES, SHARE[h][c] ** 0.6 * 1.05),
                   stroke="#ffffff", sw=0.4)
    g.text(CB.span("q")[1] + 4, gy + 3 * pitch + 6.2, "Retrieval", size=FS)
    uy = gy + 4 * pitch + 3
    g.text(CB.span("q")[1] + 4, gy + 4 * pitch + 6.2, "head", size=FS)
    for c in SEL:
        g.rect(CB.q[c] - 1.5, uy - 1.5, CB.q_w + 3, top - 3 - (uy - 1.5), fill="#ebe4f7", rx=1.5)
    g.text(CB.span("q")[0] - 3, uy + 6.2, "Union", size=FS, anchor="end", weight="bold")
    for c, x in enumerate(CB.q):
        g.rect(x, uy, CB.q_w, ch, fill=ramp(PURPLES, UNION[c] ** 0.6 * 1.05), stroke="#ffffff",
               sw=0.4)
        if c in SEL:
            g.rect(x - 0.6, uy - 0.6, CB.q_w + 1.2, ch + 1.2, stroke=INK, sw=0.9)
    for c in SEL:
        g.line(CB.qx(c), uy + ch + 1.5, CB.qx(c), top - 2, stroke=Q_S, sw=0.9, arrow="violet")
    for x1, x2 in ((CB.sink[0], CB.span("fp")[1]), CB.span("loc")):
        bracket(g, x1, x2, CB.yb - CB.fp_h - 3, up=True)
        g.text((x1 + x2) / 2, CB.yb - CB.fp_h - 6, "Always read", size=FS, anchor="middle")
    CB.draw(g, mode="gated")

    # the opened windows (what a step reads of each is in the key under the panels)
    g.text(CB.cx("q"), CB.yb + 14, "Top windows opened, at the gate ratio", size=FS,
           anchor="middle", weight="bold", fill=Q_T)


def fig1_c(g):
    frame(g, BW, BH, "(c)", "One fused pass per token", "Every decode step, every layer")
    top = CC.card_top()
    yfp = CC.yb - CC.fp_h

    # ---- when the newest window fills: re-rank, then promote / demote / drop. The
    # trigger is written in the box, not drawn, so every line here is a short drop
    rx0, ry0, rx1, ry1 = 14, 31, 170, 53
    g.rect(rx0, ry0, rx1 - rx0, ry1 - ry0, fill="#ffffff", stroke=INK, sw=0.7, rx=3)
    g.text((rx0 + rx1) / 2, ry0 + 9.4, "Re-rank by cumulative score", size=FS,
           anchor="middle", weight="bold")
    g.text((rx0 + rx1) / 2, ry0 + 18.4, "when the newest window fills", size=FS,
           anchor="middle")
    y0, yl = ry1 + 0.5, (ry1 + yfp) / 2 + 3
    g.line(CC.cx("fp"), y0, CC.cx("fp"), yfp - 1.5, stroke=FP_S, sw=0.9, arrow="blue")
    g.text(CC.cx("fp") + 3, yl, "Promote", size=FS, fill=FP_T)
    xdm = CC.qx(3)
    g.line(xdm, y0, xdm, top - 1.5, stroke=Q_S, sw=0.9, arrow="violet")
    g.text(xdm + 3, yl - 4.5, "Demote", size=FS, fill=Q_T)
    g.text(xdm + 3, yl + 4.5, "(+ card)", size=FS, fill=Q_T)
    xdr = rx1 - 12
    g.line(xdr, y0, xdr, y0 + 6, stroke=INK, sw=0.7)
    xmark(g, xdr, y0 + 9.5, r=2.4)
    g.text(xdr + 4, y0 + 12.3, "Drop", size=FS)

    CC.draw(g, mode="gated", fill_newest=True)

    # ---- one kernel reads the whole cache: fp16 and opened int2 in full, the rest by
    # card. Its row sits straight under the windows it reads, so no arrow per window
    g.rect(4.5, KY0, BW - 12, KY1 - KY0, fill="#ffffff", stroke=INK, sw=0.6, rx=3)
    g.text(9, KY0 + 9.5, "Fused attention reads:", size=FS, weight="bold")
    mt, mh = KY0 + 13, 12.5
    g.rect(CC.sink[0], mt, CC.sink[1] - CC.sink[0], mh, fill=SINK_F, stroke=SINK_S, sw=0.4)
    for x in CC.fp:
        g.rect(x, mt, CC.fp_w, mh, fill=FP_F, stroke=FP_S, sw=0.5)
    for c, x in enumerate(CC.q):
        if c in SEL:                                       # opened: dequantized on chip
            g.rect(x, mt, CC.q_w, mh, fill=FP_F, stroke=Q_S, sw=0.9)
        else:                                              # unread: its card only
            g.use("card", x + (CC.q_w - CC.card) / 2, mt + mh - CC.card, CC.card, CC.card)
    for x in CC.loc:
        g.rect(x, mt, CC.loc_w, mh, fill="hatch-blue", stroke=FP_S, sw=0.5)
    ly = mt + mh + 8.8
    g.text(CC.qx(SEL[0]), ly, "Decoded", size=FS, anchor="middle", fill=Q_T)
    g.text((CC.qx(4) + CC.qx(5)) / 2, ly, "Card only", size=FS, anchor="middle", fill=CARD_T)

    # ---- the token it produces: its K, V go straight up into the newest window's slot
    xt = CC.loc[-1] + CC.loc_w * 5.5 / 8
    g.line(xt, KY0 - 0.5, xt, CC.yb + 1, stroke=QRY_S, sw=0.9, arrow="red")
    g.text(xt - 3.5, (CC.yb + KY0) / 2 + 3, "Next token's K, V", size=FS, anchor="end",
           fill=QRY_T)


def build_fig1():
    g = G()
    g.open(LM, 0, extra=f' class="panel" data-w="{AW}" data-h="{AH}"')
    fig1_a(g)
    g.close()
    g.open(XB, BY, extra=f' class="panel" data-w="{BW:.1f}" data-h="{BH}"')
    fig1_b(g)
    g.close()
    g.open(XC, BY, extra=f' class="panel" data-w="{BW:.1f}" data-h="{BH}"')
    fig1_c(g)
    g.close()

    # between panels, across the full gaps, as block arrows: the tiered cache feeds the
    # gate (b, right); the gate's pick, level with (b)'s cache, feeds (c)'s kernel
    xq = XB + CB.cx("q")
    block_arrow(g, xq, AH + 1.5, xq, BY - 0.5)
    yb = BY + CB.yb - CB.fp_h / 2
    block_arrow(g, XB - 1.5, yb, XC + BW + 0.5, yb)
    # every window's attention this step, read or credited, goes back into its score (a):
    # out of (c)'s kernel box, up the left margin, into (a)'s first alpha cell. (a)'s own
    # row says what happens to it; the label where it enters says what it carries
    yk = BY + (KY0 + KY1) / 2
    xa = LM + CA.qx(TRACED) - A_SPAN
    ya, r = A_ROW + 7, 4
    g.path(f"M{XC + 4.5},{yk} L{XL + r},{yk} Q{XL},{yk} {XL},{yk - r} L{XL},{ya + r} "
           f"Q{XL},{ya} {XL + r},{ya} L{xa - 0.8},{ya}", stroke=FP_S, sw=1, arrow="blue")

    # the key: what a step reads of each int2 window, for the marks in (b) and (c)
    g.open(LM, KEY_Y - 9, extra=f' class="panel" data-w="{AW}" data-h="13"')
    col = AW / 3
    g.use("card", 4, 1.5, 9, 9)
    g.text(17, 9, "Card: read for every window", size=FS)
    g.rect(col + 4, 2, 9, 8, fill=Q_F, stroke=INK, sw=0.9)
    g.text(col + 17, 9, "Opened: its int2 window is read too", size=FS)
    g.add('<g opacity="0.45">')
    g.rect(2 * col + 4, 2, 9, 8, fill=Q_F, stroke=Q_S, sw=0.6, dash="1.5 1")
    g.add("</g>")
    g.text(2 * col + 17, 9, "Not opened: only its card is read", size=FS)
    g.close()
    return svg_doc(g, W1, H1), W1, H1


# ===========================================================================
# Figure 2 -- the mechanisms
# ===========================================================================
W2 = 396
PW2 = (W2 - 8) / 2
PH2A = 140
BY2 = PH2A + 8
PH2C = 86
H2 = BY2 + PH2C


def swatch(g, x, y, fill, stroke=INK, sw=0.3, dash=None):
    g.rect(x, y - 6.3, 6.5, 6.5, fill=fill, stroke=stroke, sw=sw, dash=dash)


def fig2_a(g):
    frame(g, PW2, PH2A, "(a)", "Demoting a window to int2")
    # the window's keys, taken as one cluster
    cx, cy = 62, 50
    hot = (30, -18)
    n = math.hypot(*hot)
    vd = (hot[0] / n, hot[1] / n)
    g.line(cx - vd[0] * 20, cy - vd[1] * 20, cx + vd[0] * 44, cy + vd[1] * 44, stroke=CARD_S,
           sw=0.5, dash="2 1.5")
    g.circle(cx, cy, 15, stroke=CARD_S, sw=0.5, extra=' stroke-dasharray="1 1.5"')
    for dx, dy in [(-10, 4), (-5, -7), (4, 7), (9, -2), (-12, -2), (1, -9), (6, 3), (-3, 10)]:
        g.circle(cx + dx, cy + dy, 2.2, fill=ramp(BLUES, 0.55), stroke="#ffffff", sw=0.4)
    g.circle(cx + hot[0], cy + hot[1], 2.8, fill=QRY_S, stroke="#ffffff", sw=0.4)
    g.line(cx, cy, cx + hot[0] - 3.3 * vd[0], cy + hot[1] - 3.3 * vd[1], stroke=CARD_T, sw=1,
           arrow="green")
    xmark(g, cx, cy, r=2.4)
    g.text(cx - 20, cy - 5, "its keys", size=FS, anchor="end")
    g.text(cx - 20, cy + 9, "centroid", size=FS, anchor="end")
    g.line(cx - 19, cy + 6, cx - 3, cy + 1, stroke=INK, sw=0.4)
    g.text(cx + hot[0] + 5, cy + hot[1] + 3, "outlier", size=FS, fill=QRY_T)
    ax = 84
    g.line(ax, cy, ax + 11, cy, stroke=INK, sw=0.8, arrow="dark")
    g.use("card", ax + 14, cy - 11, 22, 22)
    for k, s in enumerate(("card: centroid", "+ outlier axis", "+ value centroid")):
        g.text(ax + 41, cy - 5 + 9 * k, s, size=FS, fill=CARD_T, weight="bold")

    # an int2 window's parts, to scale against the same window in fp16
    L0, L1, rh = 8, PW2 - 8, 8
    r1, r2 = 86, 109
    g.text(L0, r1 - 3, "the same window in fp16", size=FS, fill=FP_T)
    g.rect(L0, r1, L1 - L0, rh, fill=FP_F, stroke=FP_S, sw=0.5)
    g.text(L0, r2 - 3, "int2 window + card", size=FS, fill=Q_T, weight="bold")
    parts = [("K codes", 256, "#b7a3e3"), ("V codes", 256, "#d8ccf1"),
             ("scale + zero", 292, "#9a84cf"), ("centroid", 66, "#a9d8bb"),
             ("outlier axis", 140, "#5fae82"), ("value centroid", 66, "#cfe9d9")]
    tot = sum(v for _, v, _ in parts)
    wq = (L1 - L0) * tot / 4096
    x = L0
    for _, v, col in parts:
        g.rect(x, r2, wq * v / tot, rh, fill=col, stroke="#ffffff", sw=0.4)
        x += wq * v / tot
    g.rect(L0, r2, wq, rh, stroke=Q_S, sw=0.6)
    # key: int2 parts in one column, card parts in the next
    for k, (lab, _, col) in enumerate(parts):
        kx = L0 + wq + 8 if k < 3 else 122
        ky = r2 + 6.5 + 10 * (k % 3)
        swatch(g, kx, ky, col)
        g.text(kx + 9, ky, lab, size=FS)


def fig2_b(g):
    frame(g, PW2, PH2A, "(b)", "Crediting the unread windows")
    x0, gap_, bw = 24, 3, 17.5
    base = 90
    rr = random.Random(4)
    est = [rr.uniform(12, 22) + (20 if c == SEL[-1] else 0) for c in range(N_Q)]
    gaps = {SEL[0]: 10, SEL[1]: 16}
    mean_gap = sum(gaps.values()) / len(gaps)
    for c in range(N_Q):
        x = x0 + c * (bw + gap_)
        cxm = x + bw / 2
        ye = base - est[c]
        if c in SEL:                                      # opened: its exact mass
            v = est[c] + gaps[c]
            g.rect(x, base - v, bw, v, fill=Q_F, stroke=INK, sw=0.9)
            g.line(cxm, ye - 2.4, cxm, base - v + 0.8, stroke=INK, sw=0.7)
        else:                                             # unread: estimate + average gap
            v = est[c] + mean_gap
            g.rect(x, base - v, bw, v, fill="hatch-violet", stroke=Q_S, sw=0.5, dash="1.5 1")
            g.line(cxm, ye - 2.4, cxm, base - v + 1.2, stroke=CARD_S, sw=0.7, arrow="green")
        g.circle(cxm, ye, 2.2, fill="#ffffff", stroke=CARD_S, sw=0.9)
    g.line(x0 - 3, base, x0 + N_Q * (bw + gap_), base, stroke=INK, sw=0.5)
    g.text(x0 - 8, base - 28, "log mass", size=FS, anchor="middle", rotate=-90)
    ky = base + 13
    g.circle(11.25, ky - 2.9, 2.2, fill="#ffffff", stroke=CARD_S, sw=0.9)
    g.text(17, ky, "the card's estimate, every window", size=FS)
    swatch(g, 8, ky + 10, Q_F, sw=0.9)
    g.text(17, ky + 10, "opened: exact, which measures the gap", size=FS)
    swatch(g, 8, ky + 20, "hatch-violet", stroke=Q_S, sw=0.5, dash="1.5 1")
    g.text(17, ky + 20, "unread: estimate + the average gap", size=FS)
    g.text(8, ky + 32, "credited mass enters the output and the score", size=FS)


def fig2_c(g):
    frame(g, W2, PH2C, "(c)", "The four cache sections share one byte budget")
    cc = Cache(x0=50, yb=40, sink_w=2.2, fp_w=28, q_w=18, loc_w=28, fp_h=18, gap=6, wgap=2)
    cc.draw(g)
    fixed_sink, fixed_loc = 10 / 870, 128 / 870
    ev = 1 - fixed_sink - fixed_loc
    parts = [("sink", fixed_sink), ("fp", 0.3 * ev), ("q", 0.7 * ev * 804 / 1076),
             ("card", 0.7 * ev * 272 / 1076), ("loc", fixed_loc)]
    bx0, bx1, byt, bht = cc.sink[0], cc.x1, cc.yb + 16, 11
    xs_ = [bx0]
    for _, f in parts:
        xs_.append(xs_[-1] + f * (bx1 - bx0))
    seg = dict(zip([k for k, _ in parts], zip(xs_[:-1], xs_[1:])))
    for key, col in (("sink", SINK_F), ("fp", FP_F), ("q", Q_F), ("loc", LOC_F)):
        a0, a1 = cc.span(key)
        b0 = seg[key][0]
        b1 = seg["card"][1] if key == "q" else seg[key][1]
        g.path(f"M{a0},{cc.yb + 1} L{a1},{cc.yb + 1} L{b1},{byt - 0.5} L{b0},{byt - 0.5} Z",
               fill=col, stroke="none", extra=' opacity="0.6"')
    fills = {"sink": (SINK_F, SINK_S), "fp": (FP_F, FP_S), "q": (Q_F, Q_S),
             "card": (CARD_F, CARD_S), "loc": ("hatch-blue", FP_S)}
    for key, (x0, x1) in seg.items():
        f, st = fills[key]
        g.rect(x0, byt, x1 - x0, bht, fill=f, stroke=st, sw=0.5)
    for key, lab, col in (("fp", "fp16", FP_T), ("q", "int2 windows", Q_T),
                          ("card", "cards", CARD_T), ("loc", "local", FP_T)):
        x0, x1 = seg[key]
        g.text((x0 + x1) / 2, byt + 8.2, lab, size=FS, anchor="middle", fill=col, weight="bold")
    g.text(bx0 - 3, byt + 8.2, "sink: fixed", size=FS, anchor="end", fill=SINK_S)
    g.text(bx1 + 3, byt + 8.2, "fixed", size=FS, fill=FP_T)
    bracket(g, seg["fp"][0], seg["card"][1], byt + bht + 2, up=False, tick=2)
    g.text((seg["fp"][0] + seg["card"][1]) / 2, byt + bht + 11,
           "evictable bytes, split between fp16 and int2 + cards by the quant ratio", size=FS,
           anchor="middle")


def build_fig2():
    g = G()
    g.open(0, 0, extra=f' class="panel" data-w="{PW2:.1f}" data-h="{PH2A}"')
    fig2_a(g)
    g.close()
    g.open(PW2 + 8, 0, extra=f' class="panel" data-w="{PW2:.1f}" data-h="{PH2A}"')
    fig2_b(g)
    g.close()
    g.open(0, BY2, extra=f' class="panel" data-w="{W2}" data-h="{PH2C}"')
    fig2_c(g)
    g.close()
    return svg_doc(g, W2, H2), W2, H2


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("method_overview", build_fig1), ("method_details", build_fig2)):
        svg, w, h = fn()
        out = OUT_DIR / f"{name}.html"
        out.write_text(page(svg, w, h, name), encoding="utf-8")
        print(f"wrote {out} ({w} x {h} pt = {w / 72:.2f} x {h / 72:.2f} in)")
    print(f"SEL={SEL}")


if __name__ == "__main__":
    build()
