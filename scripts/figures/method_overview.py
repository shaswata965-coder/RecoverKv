"""Build ``reports/figures/method_overview.html`` -- the method figure.

Three panels, no equations: the equations live in the methodology section. The
figure is built around ONE component, the cache bar

    [ sink | fp16 windows | int2 windows + cards | local ]

drawn identically in every panel and annotated differently in each, so the
reader learns the cache once and then watches it being scored, gated and read:

  (a) the attention map -- columns are keys (cached tokens), each row one query
      token's attention, the prompt's rows then one row per decode step -- is
      summed down each column, each
      window's tokens stack into its score, the ranked scores split into
      top-k fp16 / next top-k int2 / dropped, and the cache sections are mapped
      onto the byte budget they share. "Scoring one window" shows one step's
      update: the window's token attentions alpha_1..alpha_n (light and dark)
      summed into alpha_window, added to alpha_history, giving the new score.
      A zoom shows the card as a cluster summary of the window, and an int2
      window's parts in one row, to scale against the same window in fp16.
  (b) the query fans briefly onto a per-head score grid whose columns sit over
      the cards; the union of the heads' shares opens the top windows at the
      gate ratio, straight down into the cache bar -- and every window's
      cumulative score still grows, exactly if read, from its card if not
  (c) a generated token fills the newest window; the last closes it and the
      cache re-ranks (promote / demote / drop). One fused pass reads fp16 and
      the opened windows decoded on chip, the unread through their cards, and
      the gap measured on the opened windows lifts every unread card's estimate

No label names a configuration value -- window size, quant ratio, budget or
gate ratio -- so the figure describes the method at any setting. The drawing
itself is proportioned from one real setting (2 fp16 against 16 int2 windows
keeps the int2 share of the evictable bytes near the shipped split; the budget
bar and the int2 row are to scale for it), but that is geometry, never text.

Three more components are reused rather than redrawn: the card glyph (an SVG
``<symbol>``), the window bar (score in (a) and (b), mass in (c)) and the query
glyph. An int2 window is drawn shorter than an fp16 one, in proportion to its
price with the card included, so the bar itself carries the compression.

The figure is one self-contained SVG in a bare HTML page, so it exports without
a converter: ``scripts/figures/export_figure.mjs`` writes .svg, a one-page vector
.pdf, a 3x .png and a diagrams.net / draw.io .xml of native, editable shapes
(``svg_to_drawio.js``), and fails loudly on text that overflows or collides.

Run:  python scripts/figures/method_overview.py
"""

from __future__ import annotations

import html
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "reports" / "figures" / "method_overview.html"

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
# every notation is one solid ink: no greys, no in-between tints for text or marks
INK = INK2 = INK3 = "#1d2329"
RULE = INK
SINK_F, SINK_S = "#e2e6eb", "#5b6878"
FP_F, FP_S, FP_T = "#cfe0f3", "#3c6ea8", "#2c5a8f"
LOC_F = "#eef4fb"
Q_F, Q_S, Q_T, Q_L = "#dccff2", "#6446a4", "#51368a", "#f4f0fb"
CARD_F, CARD_S, CARD_T = "#d4ecdc", "#3b8a5a", "#2a6d45"
DROP_F, DROP_S = "#ffffff", INK
QRY_F, QRY_S, QRY_T = "#f6cdc9", "#c0392b", "#a93226"
BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
         "#2171b5", "#08519c", "#08306b"]
PURPLES = ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba",
           "#6a51a3", "#54278f", "#3f007d"]
SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"

HATCH = {"blue": (LOC_F, "#b8cfe8"), "gray": ("#ffffff", INK),
         "violet": (Q_L, "#c9bce8")}


def ramp(stops, t):
    t = max(0.0, min(1.0, t))
    x = t * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    ca = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    cb = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(p + (q - p) * f):02x}" for p, q in zip(ca, cb))


def hatch_lines(x, y, w, h, ink, gap=4.5, sw=1.0):
    """45-degree hatching clipped to a rectangle, as one vector path.

    Not an SVG <pattern>: Chromium's PDF backend rasterises pattern fills, and
    the figure has to stay vector end to end.
    """
    x1, y1 = x + w, y + h
    step = gap * math.sqrt(2)
    k = x + y + step / 2
    d = []
    while k < x1 + y1:
        xa, xb = max(x, k - y1), min(x1, k - y)
        if xb - xa > 0.2:
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

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, dash=None,
             extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        if fill.startswith("hatch-"):
            bg, ink = HATCH[fill[6:]]
            self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                     f'rx="{rx}" fill="{bg}"/>')
            self.add(hatch_lines(x, y, w, h, ink))
            fill = "none"
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=INK2, sw=1.0, dash=None, arrow=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{arrow})"' if arrow else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{m}{extra}/>')

    def path(self, d, stroke=INK2, sw=1.0, fill="none", dash=None, arrow=None, extra=""):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{arrow})"' if arrow else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}{m}{extra}/>')

    def circle(self, cx, cy, r, fill="none", stroke="none", sw=1.0, extra=""):
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{extra}/>')

    def text(self, x, y, s, size=13, anchor="start", weight=None, fill=INK, italic=False,
             rotate=None, extra=""):
        w = f' font-weight="{weight}"' if weight else ""
        it = ' font-style="italic"' if italic else ""
        rot = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{fill}"{w}{it}{rot}{extra}>{html.escape(s)}</text>')

    def text_parts(self, x, y, parts, size=13, anchor="start", weight=None, fill=INK):
        """Text with subscripts: ``parts`` is ``[(text, is_sub), ...]``.

        Subscripts are ``<tspan class="sub">`` shifted down and shrunk, so they do
        not depend on a font carrying Unicode subscript glyphs; the draw.io export
        turns the same class into ``<sub>``.
        """
        w = f' font-weight="{weight}"' if weight else ""
        out, low = [], False
        for txt, sub in parts:
            if sub:
                dy = "" if low else f' dy="{size * 0.3:.1f}"'
                out.append(f'<tspan class="sub" font-size="{size * 0.72:.1f}"{dy}>'
                           f"{html.escape(txt)}</tspan>")
                low = True
            else:
                dy = f' dy="{-size * 0.3:.1f}"' if low else ""
                out.append(f"<tspan{dy}>{html.escape(txt)}</tspan>")
                low = False
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{fill}"{w}>{"".join(out)}</text>')

    def use(self, sym, x, y, w, h, extra=""):
        self.add(f'<use href="#{sym}" xlink:href="#{sym}" x="{x:.2f}" y="{y:.2f}" width="{w}" height="{h}"{extra}/>')

    def open(self, x, y, extra=""):
        self.add(f'<g transform="translate({x},{y})"{extra}>')

    def close(self):
        self.add("</g>")


def xmark(g, x, y, r=4, stroke=DROP_S, sw=1.4):
    g.path(f"M{x - r},{y - r} L{x + r},{y + r} M{x - r},{y + r} L{x + r},{y - r}", stroke=stroke, sw=sw)


def bracket(g, x1, x2, y, up=True, stroke=INK3, sw=1.0, tick=5):
    s = 1 if up else -1
    g.path(f"M{x1},{y + s * tick} L{x1},{y} L{x2},{y} L{x2},{y + s * tick}", stroke=stroke, sw=sw)


# ---------------------------------------------------------------------------
# the shared toy cache
# ---------------------------------------------------------------------------
# 2 fp16 + 16 int2 windows puts ~68% of the evictable bytes in int2
# (16 x 1076 B vs 2 x 4096 B per head) -- the shipped 70 : 30 split, drawn to count.
N_FP, N_Q, N_LOC, N_DROP = 2, 16, 3, 4
N_SEL = 4                                   # ceil(0.25 * N_Q)

# gate: each query head's share of its own int2 mass, per int2 window
_g = random.Random(5)
_PEAKS = [{2: 1.7}, {2: 0.8, 9: 1.4}, {6: 1.3}, {12: 4.2}]       # head 4 = retrieval head
SHARE = []
for pk in _PEAKS:
    row = [_g.gauss(0, 0.3) + pk.get(c, 0.0) for c in range(N_Q)]
    m = max(row)
    z = sum(math.exp(v - m) for v in row)
    SHARE.append([math.exp(v - m) / z for v in row])
UNION = [max(SHARE[h][c] for h in range(4)) for c in range(N_Q)]
SEL = sorted(sorted(range(N_Q), key=lambda c: -UNION[c])[:N_SEL])

# layout of the cache bar, in panel coordinates
BX0 = 26
SINK_W = 3.6
FP_W, FP_G = 20, 2
Q_W, Q_G = 13, 2
FP_H, Q_H, CARD = 46, 12, 13                 # int2 ~4x shorter than fp16, as its bytes are


def cache_layout():
    x = BX0
    lay = {"sink": (x, x + 5 * SINK_W)}
    x += 5 * SINK_W + 8
    lay["fp"] = [x + i * (FP_W + FP_G) for i in range(N_FP)]
    x = lay["fp"][-1] + FP_W + 9
    lay["q"] = [x + i * (Q_W + Q_G) for i in range(N_Q)]
    x = lay["q"][-1] + Q_W + 9
    lay["loc"] = [x + i * (FP_W + FP_G) for i in range(N_LOC)]
    lay["new"] = lay["loc"][-1] + FP_W + 2
    return lay


LAY = cache_layout()
SPAN = {
    "sink": LAY["sink"],
    "fp": (LAY["fp"][0], LAY["fp"][-1] + FP_W),
    "q": (LAY["q"][0], LAY["q"][-1] + Q_W),
    "loc": (LAY["loc"][0], LAY["new"] + 4),
}
C = {k: (a + b) / 2 for k, (a, b) in SPAN.items()}
QX = [x + Q_W / 2 for x in LAY["q"]]          # int2 column centres


def cache_bar(g, yb, mode="plain"):
    """The one cache component. ``yb`` is the baseline all blocks stand on.

    mode "plain": every window as stored.  mode "gated": opened int2 windows
    solid, the rest faded to what the step actually touches -- their card.
    """
    x0, _ = LAY["sink"]
    for k in range(5):
        g.rect(x0 + k * SINK_W, yb - FP_H, SINK_W, FP_H, fill=SINK_F, stroke=SINK_S, sw=0.6)
    for x in LAY["fp"]:
        g.rect(x, yb - FP_H, FP_W, FP_H, fill=FP_F, stroke=FP_S, sw=0.9)
        for t in range(1, 8):                      # the window's 8 tokens
            g.line(x + t * FP_W / 8, yb - FP_H + 2, x + t * FP_W / 8, yb - 2, stroke="#ffffff",
                   sw=0.5)
    for c, x in enumerate(LAY["q"]):
        skipped = mode == "gated" and c not in SEL
        op = ' opacity="0.42"' if skipped else ""
        g.add(f"<g{op}>")
        g.rect(x, yb - Q_H, Q_W, Q_H, fill=Q_F, stroke=Q_S, sw=0.9, dash="2 1.5" if skipped else None)
        g.add("</g>")
        g.use("card", x, yb - Q_H - 3 - CARD, CARD, CARD)
        if mode == "gated" and c in SEL:
            g.rect(x - 1.5, yb - Q_H - 4.5 - CARD, Q_W + 3, Q_H + CARD + 6, stroke=INK, sw=1.5, rx=2)
    for x in LAY["loc"]:
        g.rect(x, yb - FP_H, FP_W, FP_H, fill="hatch-blue", stroke=FP_S, sw=0.9)
    g.rect(LAY["new"], yb - FP_H, 4, FP_H, fill=QRY_F, stroke=QRY_S, sw=0.8)
    g.line(BX0 - 4, yb, LAY["new"] + 8, yb, stroke=INK3, sw=0.8)


def card_top(yb):
    return yb - Q_H - 3 - CARD


# ---------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------
PW, PH = 440, 710
GAP = 30
LM = 34                                     # left margin carries the feedback arrow
PX = [LM, LM + PW + GAP, LM + 2 * (PW + GAP)]
PY = 46
W = PX[2] + PW + 14
H = PY + PH + 40
YB = 352                                    # the cache bar's baseline, same in every panel


def frame(g, tag, title, sub):
    g.rect(0, 0, PW, PH, fill="#ffffff", stroke=RULE, sw=1, rx=9)
    g.text(16, 26, f"{tag}  {title}", size=16.5, weight="bold")
    g.text(16, 45, sub, size=13, fill=INK2, italic=True)


def arrow_down(g, x, y, lab=None):
    """A labelled axis arrow pointing down (drawn, so it survives any font)."""
    g.line(x, y, x, y + 22, stroke=INK2, sw=1.1, arrow="dark")


# ---------------------------------------------------------------------------
# (a) cumulative attention ranks windows
# ---------------------------------------------------------------------------
TIER_TINT = {"fp": (FP_F, "#b7d0ec", FP_S), "q": (Q_F, "#c6b6e8", Q_S),
             "drop": ("#ececec", "#dadada", DROP_S)}


def panel_a(g):
    frame(g, "(a)", "Cumulative attention ranks windows", "re-tiered at every window boundary")

    # ---- attention map, with its axes named
    N, SINK, WSZ, cell = 26, 2, 8, 4.2
    N_PROMPT = 18                               # rows 0..17 prompt, the rest one per decode step
    hx, hy = 52, 84
    rng = random.Random(3)
    P = []
    for i in range(N):
        lg = [rng.gauss(0, 0.45) + (2.1 if j < SINK else 0) + (2.0 if j == 13 else 0)
              + (1.2 if j == 21 else 0) + (1.2 if i - j <= 1 else 0) for j in range(i + 1)]
        m = max(lg)
        ex = [math.exp(v - m) for v in lg]
        s = sum(ex)
        P.append([e / s for e in ex] + [0.0] * (N - i - 1))
    pmax = max(P[i][j] for i in range(N) for j in range(SINK, i + 1))
    for i in range(N):
        for j in range(N):
            col = "#ffffff" if j > i else ramp(BLUES, (P[i][j] / pmax) ** 0.6)
            g.rect(hx + j * cell, hy + i * cell, cell, cell, fill=col, stroke="#ffffff", sw=0.4)
    hw = N * cell
    g.rect(hx, hy, hw, hw, stroke=INK3, sw=0.7)
    # axes: columns are keys (cached tokens); each row is one query token's attention,
    # the prompt's rows first, then one new row per decode step
    g.text(hx, hy - 8, "keys (cached tokens)", size=11.5, fill=INK)
    g.line(hx + 114, hy - 12, hx + 138, hy - 12, stroke=INK, sw=1.1, arrow="dark")
    g.text(hx - 25, hy + hw / 2, "query tokens, in order", size=11, anchor="middle", fill=INK,
           rotate=-90)
    yp = hy + N_PROMPT * cell
    for (y0, y1, lab, col) in ((hy, yp, "prompt", INK), (yp, hy + hw, "decode", QRY_S)):
        g.path(f"M{hx - 1.5},{y0 + 0.8} L{hx - 5},{y0 + 0.8} L{hx - 5},{y1 - 0.8} L{hx - 1.5},{y1 - 0.8}",
               stroke=col, sw=1.2)
        g.text(hx - 9, (y0 + y1) / 2, lab, size=10, anchor="middle", fill=col, rotate=-90)
    g.rect(hx, yp, hw, hy + hw - yp, stroke=QRY_S, sw=1, dash="3 2")
    # summing down each column gives every key its cumulative attention
    xs_ = hx + hw + 7
    g.line(xs_, hy + 2, xs_, hy + hw + 1, stroke=INK, sw=1.1, arrow="dark")
    g.text(xs_ + 9, hy + hw / 2, "summed per column", size=11, anchor="middle", fill=INK,
           rotate=-90)

    # ---- each token's attention, summed down the query axis
    colsum = [sum(P[i][j] for i in range(N)) for j in range(N)]
    cmax = max(colsum[SINK:])
    by, bh = hy + hw + 24, 20
    HL = 1                   # the window holding the hot token: it ranks first, stays fp16
    for j in range(N):
        hh = bh * min(1.0, colsum[j] / cmax)
        w_ = (j - SINK) // WSZ
        col = SINK_S if j < SINK else (ramp(BLUES, 0.55) if w_ % 2 == 0 else ramp(BLUES, 0.35))
        g.rect(hx + j * cell + 0.6, by - hh, cell - 1.2, hh, fill=col)
    g.line(hx, by, hx + hw, by, stroke=INK, sw=0.8)
    x1 = hx + (SINK + HL * WSZ) * cell
    xw = x1 + WSZ * cell / 2
    bracket(g, x1 + 0.5, x1 + WSZ * cell - 0.5, by + 4, up=False, stroke=INK, sw=1.4, tick=4)
    g.text(xw, by + 32, "one window", size=11.5, anchor="middle", fill=INK)
    g.text(hx - 3, by - 4, "sink", size=10.5, anchor="end", fill=SINK_S, weight="bold")

    # ---- window scores, ranked: each bar is one window's tokens stacked, plus what decode added
    sx, sbase, smax, bw, bgp = 208, by, 104, 8, 2      # tallest bar stays under the tier labels
    n = N_FP + N_Q + N_DROP
    scores = [1.0 - 0.9 * (k / (n - 1)) ** 0.8 for k in range(n)]
    kinds = ["fp"] * N_FP + ["q"] * N_Q + ["drop"] * N_DROP
    hl_bar = 0
    rr = random.Random(12)
    for k, (v, kd) in enumerate(zip(scores, kinds)):
        a, b_, st = TIER_TINT[kd]
        x = sx + k * (bw + bgp)
        total = smax * v
        if kd == "drop":                       # dropped: outline only, it is not kept
            g.rect(x, sbase - total, bw, total, fill="#ffffff", stroke=INK, sw=0.8, dash="2 1.5")
            continue
        pre = total * 0.8                      # prefill attention, token by token
        parts = [rr.uniform(0.4, 1.6) for _ in range(8)]
        if k == hl_bar:
            parts[3] = 6.0                     # the hot token of the traced window
        z = sum(parts)
        y = sbase
        for t, pz in enumerate(parts):
            hseg = pre * pz / z
            g.rect(x, y - hseg, bw, hseg, fill=a if t % 2 == 0 else b_)
            y -= hseg
        g.rect(x, sbase - total, bw, total - pre, fill="hatch-" + {"fp": "blue", "q": "violet"}[kd])
        g.rect(x, sbase - total, bw, total, stroke=INK if k == hl_bar else st,
               sw=1.6 if k == hl_bar else 0.7)
    g.line(sx - 2, sbase, sx + n * (bw + bgp), sbase, stroke=INK, sw=0.8)
    g.text(sx - 8, sbase - 62, "cumulative window score", size=11, anchor="middle", fill=INK,
           rotate=-90)
    # which bars are which
    ex_ = lambda k: sx + k * (bw + bgp)
    lab_y = 92
    bracket(g, ex_(0), ex_(N_FP) - bgp, lab_y + 5, up=True, stroke=FP_S, sw=1.2, tick=3)
    g.text(ex_(0) - 2, lab_y - 14, "top-k", size=12, weight="bold", fill=FP_T)
    g.text(ex_(0) - 2, lab_y, "fp16", size=11.5, fill=FP_T)
    bracket(g, ex_(N_FP), ex_(N_FP + N_Q) - bgp, lab_y + 5, up=True, stroke=Q_S, sw=1.2, tick=3)
    g.text((ex_(N_FP) + ex_(N_FP + N_Q)) / 2 + 8, lab_y - 14, "next top-k, quantized", size=12,
           anchor="middle", weight="bold", fill=Q_T)
    g.text((ex_(N_FP) + ex_(N_FP + N_Q)) / 2 + 8, lab_y, "int2 + card", size=11.5,
           anchor="middle", fill=Q_T)
    bracket(g, ex_(N_FP + N_Q), ex_(n) - bgp, lab_y + 5, up=True, stroke=INK, sw=1.2, tick=3)
    g.text((ex_(N_FP + N_Q) + ex_(n)) / 2, lab_y, "dropped", size=11.5, anchor="middle", fill=INK)
    # stack key
    kx, ky = 296, 126
    g.rect(kx, ky - 9, 9, 10, fill=Q_F)
    g.rect(kx, ky - 19, 9, 10, fill="#c6b6e8")
    g.text(kx + 14, ky - 5, "one token's attention", size=11, fill=INK)
    g.rect(kx, ky + 6, 9, 10, fill="hatch-violet", stroke=Q_S, sw=0.6)
    g.text(kx + 14, ky + 15, "added per decode step", size=11, fill=INK)

    # ---- the traced window: its tokens become the top bar (straight connector)
    xb0 = ex_(hl_bar) + bw / 2
    yt = by + 14
    g.path(f"M{xw},{by + 5} L{xw},{yt} L{xb0},{yt} L{xb0},{by + 2}", stroke=INK, sw=1.1,
           dash="3 2", arrow="dark")

    # ---- into the tiers, straight
    xfp = ex_(1) + bw / 2 + 1                  # leaves from the second fp16 bar
    xq = (ex_(N_FP) + ex_(N_FP + N_Q)) / 2     # the int2 group sits over the int2 section
    xd = (ex_(N_FP + N_Q) + ex_(n)) / 2
    yturn = YB - FP_H - 18
    g.path(f"M{xfp},{by + 2} L{xfp},{yturn} L{C['fp']},{yturn} L{C['fp']},{YB - FP_H - 3}",
           stroke=FP_S, sw=1.6, arrow="blue")
    g.line(xq, by + 2, xq, card_top(YB) - 3, stroke=Q_S, sw=1.6, arrow="violet")
    xmark(g, xd, by + 16, r=4.5, stroke=INK, sw=1.8)

    cache_bar(g, YB)

    # ---- how the four parts make up the budget (bytes, to scale)
    # 4096-token prompt + 256 generated, budget 0.20: 870 tokens' worth of fp16 KV.
    # sink 5 + local 128 are fixed; the rest splits 30 : 70 by bytes.
    fixed_sink, fixed_loc = 5 / 870, 128 / 870
    ev = 1 - fixed_sink - fixed_loc
    parts = [("sink", fixed_sink), ("fp", 0.3 * ev), ("q", 0.7 * ev * 804 / 1076),
             ("card", 0.7 * ev * 272 / 1076), ("loc", fixed_loc)]
    bx0, bx1, byt, bht = BX0 - 4, LAY["new"] + 8, YB + 46, 18
    xs_ = [bx0]
    for _, f in parts:
        xs_.append(xs_[-1] + f * (bx1 - bx0))
    seg = dict(zip([k for k, _ in parts], zip(xs_[:-1], xs_[1:])))
    # each cache section maps onto its share of the bytes
    for key, (a0, a1), col in (("sink", SPAN["sink"], SINK_F), ("fp", SPAN["fp"], FP_F),
                               ("q", SPAN["q"], Q_F), ("loc", SPAN["loc"], LOC_F)):
        b0 = seg[key][0]
        b1 = seg["card"][1] if key == "q" else seg[key][1]
        g.path(f"M{a0},{YB + 2} L{a1},{YB + 2} L{b1},{byt - 1} L{b0},{byt - 1} Z", fill=col,
               stroke="none", extra=' opacity="0.55"')
    fills = {"sink": (SINK_F, SINK_S), "fp": (FP_F, FP_S), "q": (Q_F, Q_S),
             "card": (CARD_F, CARD_S), "loc": ("hatch-blue", FP_S)}
    for key, (x0, x1) in seg.items():
        f, st = fills[key]
        g.rect(x0, byt, x1 - x0, bht, fill=f, stroke=st, sw=0.9)
    for key, lab, col in (("fp", "fp16", FP_T), ("q", "int2", Q_T), ("card", "cards", CARD_T),
                          ("loc", "local", FP_T)):
        x0, x1 = seg[key]
        g.text((x0 + x1) / 2, byt + 13, lab, size=11.5, anchor="middle", fill=col, weight="bold")
    # what the budget is, and how it divides
    ly = byt + bht + 16
    bracket(g, seg["fp"][0], seg["card"][1], byt + bht + 4, up=False, stroke=INK3, tick=4)
    g.text((seg["fp"][0] + seg["card"][1]) / 2, ly + 2, "evictable bytes, split by the quant ratio",
           size=11.5, anchor="middle", fill=INK2)
    g.text(seg["loc"][1], ly + 2, "fixed", size=11.5, anchor="end", fill=INK3)
    g.text((bx0 + bx1) / 2, ly + 19, "all four together make up the cache budget",
           size=11.5, anchor="middle", fill=INK, weight="bold")

    # ---- scoring one window: its tokens' attention summed, added to its history
    sx0, sy0, sw0, sh0 = 16, YB + 110, PW - 32, 74
    g.rect(sx0, sy0, sw0, sh0, fill="#ffffff", stroke=RULE, sw=1, rx=6)
    g.text(sx0 + 12, sy0 + 18, "scoring one window, every decode step", size=12.5, weight="bold",
           fill=INK)
    cw_, cg_, cy0, chh = 22, 2, sy0 + 30, 24
    shades = [0.30, 0.85, 0.45, None, 0.18, 0.70]            # light and dark tokens
    labels = [("1", 0), ("2", 0), ("3", 0), None, ("n−1", 0), ("n", 0)]
    x = sx0 + 12
    for v, lab in zip(shades, labels):
        if v is None:                                         # the tokens not drawn
            g.text(x + cw_ / 2, cy0 + 16, "…", size=13, anchor="middle", fill=INK)
        else:
            g.rect(x, cy0, cw_, chh, fill=ramp(BLUES, v), stroke=FP_S, sw=0.8)
            g.text_parts(x + cw_ / 2, cy0 + 15, [("α", False), (lab[0], True)], size=11.5,
                         anchor="middle", fill="#ffffff" if v > 0.6 else INK)
        x += cw_ + cg_
    xa = x + 2
    g.line(xa, cy0 + chh / 2, xa + 16, cy0 + chh / 2, stroke=INK, sw=1.3, arrow="dark")
    g.text(xa + 8, cy0 + chh + 12, "sum", size=11, anchor="middle", fill=INK)

    def chip_(x0, wd, parts, fill, stroke, bold=False, sw=0.9):
        g.rect(x0, cy0, wd, chh, fill=fill, stroke=stroke, sw=sw, rx=4)
        g.text_parts(x0 + wd / 2, cy0 + 16, parts, size=12, anchor="middle", fill=INK,
                     weight="bold" if bold else None)
        return x0 + wd

    x = chip_(xa + 22, 54, [("α", False), ("window", True)], ramp(BLUES, 0.35), FP_S)
    g.text(x + 10, cy0 + 17, "+", size=15, anchor="middle", fill=INK, weight="bold")
    x = chip_(x + 20, 54, [("α", False), ("history", True)], "#ffffff", FP_S)
    g.text(x + 10, cy0 + 17, "=", size=15, anchor="middle", fill=INK, weight="bold")
    chip_(x + 20, sx0 + sw0 - 12 - (x + 20), [("new score", False)], FP_F, FP_S, bold=True,
          sw=1.4)

    # ---- demotion, magnified: the card clusters the window, and what an int2 window holds
    zh = 152
    zx, zy, zw = 16, sy0 + sh0 + 10, PW - 32
    g.rect(zx, zy, zw, zh, fill="#ffffff", stroke=RULE, sw=1, rx=6)
    g.text(zx + 12, zy + 19, "demoting a window to int2", size=12.5, weight="bold", fill=INK)
    cx, cy = zx + 98, zy + 58
    vd = (0.83, -0.56)
    pts = [(-15, 6), (-8, -10), (6, 11), (13, -3), (-19, -3), (2, -13), (9, 4)]
    hot = (52, -30)
    g.line(cx - vd[0] * 34, cy - vd[1] * 34, cx + vd[0] * 72, cy + vd[1] * 72, stroke=CARD_S,
           sw=0.9, dash="4 3")
    g.circle(cx, cy, 26, stroke=CARD_S, sw=0.9, extra=' stroke-dasharray="2 3"')
    for dx, dy in pts:
        g.circle(cx + dx, cy + dy, 3.6, fill=ramp(BLUES, 0.55), stroke="#ffffff", sw=0.7)
    g.circle(cx + hot[0], cy + hot[1], 4.6, fill=QRY_S, stroke="#ffffff", sw=0.8)
    g.line(cx, cy, cx + hot[0] - 5, cy + hot[1] + 3.5, stroke=CARD_T, sw=1.8, arrow="green")
    xmark(g, cx, cy, r=4, stroke=INK, sw=1.8)
    g.text(cx - 34, cy + 4, "centroid", size=11.5, anchor="end", fill=INK)
    g.line(cx - 32, cy, cx - 6, cy, stroke=INK3, sw=0.7)
    g.text(cx + hot[0] + 8, cy + hot[1] + 4, "outlier", size=11.5, fill=QRY_T)
    g.text(cx - 28, cy - 24, "its keys", size=11.5, anchor="end", fill=INK)
    rx0 = zx + 206
    g.line(rx0 - 30, cy, rx0 - 6, cy, stroke=INK2, sw=1.3, arrow="dark")
    g.use("card", rx0, cy - 20, 40, 40)
    g.text(rx0 + 48, cy - 7, "card: the keys", size=11.5, fill=CARD_T, weight="bold")
    g.text(rx0 + 48, cy + 8, "as one cluster, kept as", size=11.5, fill=INK)
    g.text(rx0 + 48, cy + 22, "centroid + outlier axis", size=11.5, fill=INK)

    # one int2 window, to scale against the same window in fp16, in one row
    L0, L1 = zx + 12, zx + zw - 12
    rw, rh = L1 - L0, 12
    r1, r2 = zy + 98, zy + 116
    g.rect(L0, r1, rw, rh, fill=FP_F, stroke=FP_S, sw=0.8)
    g.text(L1 - 6, r1 + 9.5, "the same window in fp16", size=10.5, anchor="end", fill=FP_T)
    # relative sizes of the parts (their ratio, not a byte count, is what is drawn)
    parts = [("K codes", 256, "#b7a3e3"), ("V codes", 256, "#d8ccf1"), ("scale + zero", 292, "#9a84cf"),
             ("centroid", 66, "#a9d8bb"), ("outlier axis", 140, "#5fae82"),
             ("value centroid", 66, "#cfe9d9")]
    tot = sum(v for _, v, _ in parts)
    wq = rw * tot / 4096
    x = L0
    for _, v, col in parts:
        g.rect(x, r2, wq * v / tot, rh, fill=col, stroke="#ffffff", sw=0.6)
        x += wq * v / tot
    g.rect(L0, r2, wq, rh, stroke=Q_S, sw=0.9)
    g.text(L0, r2 + rh + 14, "int2 window + card", size=11, fill=Q_T, weight="bold")
    # the key to the parts, beside the bar
    kx = L0 + wq + 14
    for line, items in enumerate((parts[:3], parts[3:])):
        xx = kx
        yy = r2 + 1 + line * 16
        for lab, _, col in items:
            g.rect(xx, yy, 9, 9, fill=col, stroke=INK3, sw=0.4)
            g.text(xx + 13, yy + 8.5, lab, size=10.5, fill=INK2)
            xx += 24 + len(lab) * 5.6


# ---------------------------------------------------------------------------
# (b) cards choose which int2 windows are read
# ---------------------------------------------------------------------------
def panel_b(g):
    frame(g, "(b)", "Cards choose what to read", "every decode step, per KV head")
    top = card_top(YB)

    # the query: the heads of one GQA group, fanning briefly onto every card's column
    qx, qy = C["q"] - 22, 62
    for h in range(4):
        g.rect(qx, qy + h * 8, 44, 6, fill=QRY_F, stroke=QRY_S, sw=0.8, rx=1.5)
    g.text(qx - 8, qy + 12, "query", size=12.5, anchor="end", weight="bold", fill=QRY_T)
    g.text(qx - 8, qy + 27, "heads of one group", size=12, anchor="end", fill=QRY_T)
    gy, ch = qy + 76, 12                        # per-head score grid, one column per card
    umax = max(UNION)
    for c in range(N_Q):
        v = UNION[c] / umax
        g.line(C["q"], qy + 33, QX[c], gy - 3, stroke=Q_S if c in SEL else INK,
               sw=0.8 + 2.2 * v if c in SEL else 0.5)
    g.text(SPAN["q"][1] - 30, qy + 22, "score every card", size=12, fill=INK)

    # each head's share of the cards, then the union across the group
    for h in range(4):
        g.text(SPAN["q"][0] - 8, gy + h * (ch + 2) + 10, f"head {h + 1}", size=11, anchor="end",
               fill=INK)
        for c, x in enumerate(LAY["q"]):
            g.rect(x, gy + h * (ch + 2), Q_W, ch, fill=ramp(PURPLES, SHARE[h][c] ** 0.6 * 1.05),
                   stroke="#ffffff", sw=0.6)
    g.text(SPAN["q"][1] + 8, gy + 3 * (ch + 2) + 10, "retrieval head", size=11, fill=INK)
    uy = gy + 4 * (ch + 2) + 10
    # the opened columns, carried straight down from the vote to the cards
    for c in SEL:
        g.rect(LAY["q"][c] - 2.5, uy - 3, Q_W + 5, top - 6 - (uy - 3), fill="#ebe4f7", rx=3)
    g.text(SPAN["q"][0] - 8, uy + 10, "union", size=11, anchor="end", weight="bold", fill=INK)
    for c, x in enumerate(LAY["q"]):
        g.rect(x, uy, Q_W, ch, fill=ramp(PURPLES, UNION[c] ** 0.6 * 1.05), stroke="#ffffff", sw=0.6)
        if c in SEL:
            g.rect(x - 1, uy - 1, Q_W + 2, ch + 2, stroke=INK, sw=1.5)
    g.text(C["q"], uy + 30, "top windows opened, at the gate ratio", size=12.5, anchor="middle",
           weight="bold", fill=Q_T)
    for c in SEL:
        g.line(QX[c], uy + 38, QX[c], top - 5, stroke=Q_S, sw=1.6, arrow="violet")
    for key in ("fp", "loc"):
        x1, x2 = SPAN[key]
        bracket(g, x1, x2, YB - FP_H - 8, up=True)
        g.text((x1 + x2) / 2, YB - FP_H - 14, "always read", size=11.5, anchor="middle", fill=INK)
    cache_bar(g, YB, mode="gated")

    # the cumulative score keeps rolling, read or not
    ry = YB + 44
    g.line(16, ry - 18, PW - 16, ry - 18, stroke=RULE, sw=1, dash="3 3")
    g.text(16, ry, "cumulative score keeps rolling, read or not", size=12.5, weight="bold",
           fill=INK)
    base = ry + 170
    rr = random.Random(21)
    cols = ([(x, FP_W, "fp") for x in LAY["fp"]]
            + [(x, Q_W, "sel" if c in SEL else "skip") for c, x in enumerate(LAY["q"])]
            + [(x, FP_W, "fp") for x in LAY["loc"]])
    k = 2.0
    for x, wd, kd in cols:
        before = k * {"fp": rr.uniform(38, 52), "sel": rr.uniform(26, 36),
                      "skip": rr.uniform(12, 26)}[kd]
        add = k * {"fp": rr.uniform(10, 16), "sel": rr.uniform(10, 18), "skip": rr.uniform(3, 7)}[kd]
        pale = {"fp": "#e3edf8", "sel": "#ece6f7", "skip": "#ece6f7"}[kd]
        g.rect(x, base - before, wd, before, fill=pale, stroke="none")
        if kd == "skip":
            g.rect(x, base - before - add, wd, add, fill="hatch-violet", stroke=Q_S, sw=0.8)
        else:
            g.rect(x, base - before - add, wd, add, fill=FP_S if kd == "fp" else Q_S)
        # mark each int2 window exactly as the key below does: opened = black border,
        # card only = dashed outline
        if kd == "sel":
            g.rect(x - 0.7, base - before - add - 0.7, wd + 1.4, before + add + 0.7, stroke=INK,
                   sw=1.5)
        elif kd == "skip":
            g.rect(x, base - before - add, wd, before + add, stroke=Q_S, sw=0.8, dash="2 1.5")
    g.line(BX0 - 4, base, LAY["new"] + 8, base, stroke=INK, sw=0.8)
    ky = base + 22
    g.rect(24, ky - 10, 12, 11, fill="#ece6f7")
    g.text(41, ky, "score so far", size=11.5, fill=INK)
    g.rect(122, ky - 10, 12, 11, fill=Q_S)
    g.text(139, ky, "+ exact mass (read)", size=11.5, fill=INK)
    g.rect(262, ky - 10, 12, 11, fill="hatch-violet", stroke=Q_S, sw=0.8)
    g.text(279, ky, "+ its card's credit", size=11.5, fill=INK)

    # key
    ky2 = ky + 30
    g.rect(24, ky2 - 11, 14, 13, fill=Q_F, stroke=INK, sw=1.4)
    g.text(44, ky2, "opened", size=12, fill=INK)
    g.add('<g opacity="0.42">')
    g.rect(112, ky2 - 11, 14, 13, fill=Q_F, stroke=Q_S, sw=0.9, dash="2 1.5")
    g.add("</g>")
    g.text(132, ky2, "only its card is read", size=12, fill=INK)


# ---------------------------------------------------------------------------
# (c) one fused pass: generate a token, read the cache, credit the rest
# ---------------------------------------------------------------------------
def panel_c(g):
    frame(g, "(c)", "One fused pass per generated token", "every decode step, every layer")
    top = card_top(YB)

    # ---- the newest window fills one token per step
    nx0, ny0, sw_ = LAY["new"] + 6 - 8 * 14, 74, 14
    g.text(nx0 + 8 * sw_ - 2, ny0 - 8, "newest window: one token per step", size=12, anchor="end",
           weight="bold", fill=FP_T)
    for t in range(8):
        x = nx0 + t * sw_
        if t < 5:
            g.rect(x, ny0, sw_ - 2, 22, fill="hatch-blue", stroke=FP_S, sw=0.8)
        elif t == 5:
            g.rect(x, ny0, sw_ - 2, 22, fill=QRY_F, stroke=QRY_S, sw=1.2)
        else:
            g.rect(x, ny0, sw_ - 2, 22, fill="#ffffff", stroke=INK3, sw=0.8, dash="2 2")
    g.text(nx0 + 5.5 * sw_ - 1, ny0 + 36, "token t", size=11, anchor="middle", fill=QRY_T)
    xb = nx0 + 8 * sw_ + 2
    g.line(xb, ny0 - 4, xb, ny0 + 26, stroke=QRY_S, sw=1.4, dash="3 2")
    nw = LAY["loc"][-1]
    g.path(f"M{nw},{YB - FP_H - 2} L{nx0},{ny0 + 24} M{nw + FP_W},{YB - FP_H - 2} "
           f"L{nx0 + 8 * sw_ - 2},{ny0 + 24}", stroke=RULE, sw=0.9, dash="3 3")

    # ---- at the window boundary: re-rank, promote / demote / drop
    chx, chy, chw, chh = 40, 136, 196, 24
    ym = chy + chh / 2
    g.path(f"M{xb},{ny0 + 26} L{xb},{ym} L{chx + chw + 3},{ym}", stroke=QRY_S, sw=1.4,
           arrow="dark")
    g.text(nx0 - 8, ny0 + 9, "last token closes it:", size=11.5, anchor="end", fill=QRY_T,
           weight="bold")
    g.text(nx0 - 8, ny0 + 24, "window boundary", size=11.5, anchor="end", fill=QRY_T)
    g.rect(chx, chy, chw, chh, fill="#ffffff", stroke=INK2, sw=1, rx=12)
    g.text(chx + chw / 2, chy + 16, "re-rank by cumulative score", size=12, anchor="middle",
           weight="bold")
    # promote into fp16
    g.line(C["fp"], chy + chh + 1, C["fp"], YB - FP_H - 3, stroke=FP_S, sw=1.5, arrow="blue")
    g.text(C["fp"] + 8, chy + chh + 92, "promote:", size=11.5, fill=FP_T, weight="bold")
    g.text(C["fp"] + 8, chy + chh + 106, "rose → fp16", size=11.5, fill=FP_T)
    # demote into int2
    xdm = QX[5]
    g.line(xdm, chy + chh + 1, xdm, top - 3, stroke=Q_S, sw=1.5, arrow="violet")
    g.text(xdm + 8, chy + chh + 60, "demote: fell → 2-bit", size=11.5, fill=Q_T, weight="bold")
    g.text(xdm + 8, chy + chh + 74, "+ card, written once", size=11.5, fill=Q_T)
    # drop
    xdr = chx + chw - 12
    g.path(f"M{xdr},{chy + chh + 2} L{xdr},{chy + chh + 18}", stroke=DROP_S, sw=1.5)
    xmark(g, xdr, chy + chh + 25, r=4.5, sw=1.8)
    g.text(xdr + 10, chy + chh + 29, "drop the rest", size=11.5, fill=INK3)

    cache_bar(g, YB, mode="gated")

    # ---- what the one fused pass reads
    ky0, ky1 = YB + 34, YB + 116
    kb = ky1 - 8
    g.rect(16, ky0, PW - 32, ky1 - ky0, fill="#ffffff", stroke=INK3, sw=1, rx=6)
    g.text(26, ky0 + 16, "fused two-tier attention: what it reads", size=12, weight="bold")
    for x in LAY["fp"]:
        g.rect(x, kb - FP_H * 0.9, FP_W, FP_H * 0.9, fill=FP_F, stroke=FP_S, sw=0.8)
    g.rect(LAY["sink"][0], kb - FP_H * 0.9, LAY["sink"][1] - LAY["sink"][0], FP_H * 0.9,
           fill=SINK_F, stroke=SINK_S, sw=0.6)
    for x in LAY["loc"]:
        g.rect(x, kb - FP_H * 0.9, FP_W, FP_H * 0.9, fill="hatch-blue", stroke=FP_S, sw=0.8)
    g.rect(LAY["new"], kb - FP_H * 0.9, 4, FP_H * 0.9, fill=QRY_F, stroke=QRY_S, sw=0.8)
    for c, x in enumerate(LAY["q"]):
        if c in SEL:                                  # decoded to full height, on chip
            g.rect(x, kb - FP_H * 0.9, Q_W, FP_H * 0.9, fill=FP_F, stroke=Q_S, sw=1.3)
        else:
            g.use("card", x, kb - CARD, CARD, CARD)
    g.line(BX0 - 4, kb, LAY["new"] + 8, kb, stroke=INK3, sw=0.7)
    for x, col, arrow, dash, sw in [(C["fp"], FP_S, "blue", None, 1.5), (C["loc"], FP_S, "blue", None, 1.5)] + \
            [(QX[c], Q_S if c in SEL else CARD_S, "violet" if c in SEL else "green",
              None if c in SEL else "2 2", 1.4 if c in SEL else 0.9) for c in range(N_Q)]:
        g.line(x, YB + 3, x, ky0 - 2, stroke=col, sw=sw, dash=dash, arrow=arrow)

    # ---- the token it generates
    oy = ky1 + 14
    g.rect(PW - 16 - 148, oy, 148, 24, fill="#ffffff", stroke=INK2, sw=1, rx=5)
    g.text(PW - 16 - 74, oy + 16, "output → next token", size=12, anchor="middle", weight="bold")
    g.line(PW - 16 - 74, ky1 + 1, PW - 16 - 74, oy - 1, stroke=INK, sw=1.3, arrow="dark")
    g.path(f"M{PW - 16},{oy + 12} L{PW - 7},{oy + 12} L{PW - 7},{YB - FP_H / 2} "
           f"L{LAY['new'] + 7},{YB - FP_H / 2}", stroke=QRY_S, sw=1.4, arrow="red")
    g.text(PW - 16 - 156, oy + 16, "its K, V appended", size=11.5, anchor="end", fill=QRY_T)

    # ---- realigning the unread windows from the read ones
    ry = oy + 52
    g.line(16, ry - 16, PW - 16, ry - 16, stroke=RULE, sw=1, dash="3 3")
    g.text(16, ry, "credit for unread windows", size=12.5, weight="bold")
    base = PH - 30
    Y = lambda v: base - 12 - v                   # v in px above a floor (log scale)
    rr = random.Random(4)
    est = [rr.uniform(14, 26) + (26 if c == SEL[-1] else 0) for c in range(N_Q)]
    gaps = {c: g_ for c, g_ in zip(SEL, (13, 7, 10, 16))}
    gap_mean = sum(gaps.values()) / len(gaps)
    for x, v in zip(LAY["fp"] + LAY["loc"], (70, 52, 60, 66, 72)):
        g.rect(x, Y(v), FP_W, base - Y(v), fill=FP_F, stroke=FP_S, sw=0.8)
    for c, x in enumerate(LAY["q"]):
        cxm = x + Q_W / 2
        if c in SEL:
            v = est[c] + gaps[c]
            g.rect(x, Y(v), Q_W, base - Y(v), fill=Q_F, stroke=Q_S, sw=0.9)
            g.circle(cxm, Y(est[c]), 3.3, fill="#ffffff", stroke=CARD_S, sw=1.4)
            g.line(cxm, Y(est[c]) - 4, cxm, Y(v) + 1.5, stroke=INK, sw=1.1)
        else:
            v = est[c] + gap_mean
            g.rect(x, Y(v), Q_W, base - Y(v), fill="hatch-violet", stroke=Q_S, sw=0.8,
                   dash="2 1.5")
            g.circle(cxm, Y(est[c]), 3.3, fill="#ffffff", stroke=CARD_S, sw=1.4)
            g.line(cxm, Y(est[c]) - 4, cxm, Y(v) + 2.5, stroke=CARD_S, sw=1, arrow="green")
    g.line(BX0 - 4, base, LAY["new"] + 8, base, stroke=INK3, sw=0.8)
    # the two rules, by example
    lx = 16
    g.circle(lx + 6, ry + 18, 3.3, fill="#ffffff", stroke=CARD_S, sw=1.4)
    g.text(lx + 16, ry + 22, "card's estimate", size=11.5, fill=INK2)
    g.rect(lx + 112, ry + 12, 10, 11, fill=Q_F, stroke=Q_S, sw=0.8)
    g.text(lx + 128, ry + 22, "opened: exact → measures the card's gap", size=11.5, fill=INK2)
    g.rect(lx + 112, ry + 30, 10, 11, fill="hatch-violet", stroke=Q_S, sw=0.8, dash="2 1.5")
    g.text(lx + 128, ry + 40, "unread: lifted by the average gap", size=11.5, fill=INK2)


# ---------------------------------------------------------------------------
# legend and cross-panel arrows
# ---------------------------------------------------------------------------
def legend(g):
    x, y = LM, 24
    items = [("sink", SINK_F, SINK_S, None), ("fp16 window", FP_F, FP_S, None),
             ("local window", "hatch-blue", FP_S, None), ("int2 window", Q_F, Q_S, None),
             ("card", "card", None, None), ("credited from its card", "hatch-violet", Q_S, "2 1.5"),
             ("query / new token", QRY_F, QRY_S, None), ("dropped", "x", None, None)]
    for lab, f, s, d in items:
        if f == "card":
            g.use("card", x, y - 13, 16, 16)
        elif f == "x":
            xmark(g, x + 8, y - 5, r=4.5, sw=1.8)
        else:
            g.rect(x, y - 11, 16, 13, fill=f, stroke=s, sw=0.9, dash=d)
        g.text(x + 22, y, lab, size=12.5, fill=INK2)
        x += 40 + len(lab) * 6.6


def arrows(g):
    y = PY + YB - 26
    for k in range(2):
        g.line(PX[k] + PW + 3, y, PX[k + 1] - 4, y, stroke=INK, sw=2, arrow="dark")
    # every step's window mass is added to the cumulative score: (c) -> (a)
    x_from = PX[2] + C["q"]
    yb = PY + PH + 22
    ya = PY + 150
    g.path(f"M{x_from},{PY + PH + 2} L{x_from},{yb - 6} Q{x_from},{yb} {x_from - 6},{yb} "
           f"L{16},{yb} Q{10},{yb} {10},{yb - 6} L{10},{ya + 6} Q{10},{ya} {16},{ya} "
           f"L{PX[0] - 3},{ya}", stroke=FP_S, sw=2, arrow="blue")
    g.add(f'<rect x="{PX[1] + 20}" y="{yb - 9}" width="{PW - 40}" height="18" fill="#ffffff"/>')
    g.text(PX[1] + PW / 2, yb + 4.5, "every window's mass is added to its cumulative score",
           size=12.5, anchor="middle", fill=FP_T, weight="bold")


def defs():
    def marker(name, col):
        return (f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8.6" refY="5" '
                f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                f'<path d="M0,0.6 L10,5 L0,9.4 z" fill="{col}"/></marker>')

    # the card: the window's keys as one cluster -- centroid, and the axis to its outlier
    card = (f'<symbol id="card" viewBox="0 0 20 20">'
            f'<rect x="0.8" y="0.8" width="18.4" height="18.4" rx="3.2" fill="{CARD_F}" '
            f'stroke="{CARD_S}" stroke-width="1"/>'
            f'<circle cx="5.2" cy="12.6" r="1.35" fill="#4f7fb5"/>'
            f'<circle cx="8.4" cy="15.2" r="1.35" fill="#4f7fb5"/>'
            f'<circle cx="4.6" cy="16" r="1.35" fill="#4f7fb5"/>'
            f'<circle cx="9" cy="11.2" r="1.35" fill="#4f7fb5"/>'
            f'<path d="M6.8,13.8 L14.6,6.2" stroke="{CARD_T}" stroke-width="1.2"/>'
            f'<circle cx="15.4" cy="5.2" r="1.9" fill="{QRY_S}"/>'
            f'<path d="M5.7,12.7 L7.9,14.9 M5.7,14.9 L7.9,12.7" stroke="{INK}" stroke-width="1"/>'
            f'</symbol>')
    return ("<defs>" + marker("dark", INK) + marker("gray", INK3) + marker("blue", FP_S)
            + marker("violet", Q_S) + marker("green", CARD_S) + marker("red", QRY_S) + card
            + "</defs>")


def build():
    g = G()
    g.add(defs())
    g.rect(0, 0, W, H, fill="#ffffff")
    legend(g)
    for k, fn in enumerate((panel_a, panel_b, panel_c)):
        g.open(PX[k], PY, extra=f' class="panel" data-w="{PW}" data-h="{PH}"')
        fn(g)
        g.close()
    arrows(g)
    svg = (f'<svg id="figure" xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{SANS}">\n' + "\n".join(g.p) + "\n</svg>")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RecoverKV method</title>
<style>
  @page {{ size: {W}px {H}px; margin: 0; }}
  html, body {{ margin: 0; padding: 0; background: #ffffff; }}
  body {{ display: flex; justify-content: center; }}
  #figure {{ display: block; width: min(100vw, {W}px); height: auto; }}
  @media print {{ #figure {{ width: {W}px; height: {H}px; }} }}
</style>
</head>
<body>
{svg}
<script>
/* Export without any on-page UI:  S = .svg,  P = .png (3x),  Ctrl/Cmd+P = vector PDF. */
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
      save(new Blob([svgText()], {{type: 'image/svg+xml'}}), 'recoverkv_method.svg');
    }} else if (k === 'p') {{
      const img = new Image(), scale = 3;
      img.onload = () => {{
        const cv = document.createElement('canvas');
        cv.width = {W} * scale; cv.height = {H} * scale;
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, cv.width, cv.height);
        ctx.drawImage(img, 0, 0, cv.width, cv.height);
        cv.toBlob((b) => save(b, 'recoverkv_method.png'), 'image/png');
      }};
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgText());
    }}
  }});
}})();
</script>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB), {W}x{H}, SEL={SEL}")


if __name__ == "__main__":
    build()
