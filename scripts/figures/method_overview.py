"""Build ``reports/figures/method_overview.html`` -- the method figure.

Three panels, no equations: the equations live in the methodology section. The
figure is built around ONE component, the cache bar

    [ sink | fp16 windows | int2 windows + cards | local ]

drawn identically in every panel and annotated differently in each, so the
reader learns the cache once and then watches it being scored, gated and read:

  (a) cumulative attention ranks windows into the tiers, and demotion writes a
      window as 2-bit codes plus a card (a cluster summary of its keys)
  (b) the query scores every card; each head's preferences are unioned and the
      top quarter of the int2 windows is opened
  (c) one fused pass reads fp16 and the opened int2 windows, credits the rest
      through their cards, and the window masses feed back into the scores

Three more components are reused rather than redrawn: the card glyph (an SVG
``<symbol>``), the window bar (score in (a), mass in (c)) and the query glyph.
An int2 window is drawn about 4x shorter than an fp16 one, which is its price
with the card included (1076 B vs 4096 B per head), so the bar itself carries
the compression.

The figure is one self-contained SVG in a bare HTML page, so it exports without
a converter: ``scripts/figures/export_figure.mjs`` writes .svg, a one-page vector
.pdf and a 3x .png, and fails loudly on text that overflows or collides.

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
INK, INK2, INK3 = "#1d2329", "#4b5563", "#8a939c"
RULE = "#c9d0d7"
SINK_F, SINK_S = "#e4ddf1", "#7b68a6"
FP_F, FP_S, FP_T = "#cfe0f3", "#3c6ea8", "#2c5a8f"
LOC_F = "#eef4fb"
Q_F, Q_S, Q_T, Q_L = "#f8cfa6", "#c8691f", "#a0500f", "#fdf1e4"
CARD_F, CARD_S, CARD_T = "#d4ecdc", "#3b8a5a", "#2a6d45"
DROP_F, DROP_S = "#ececec", "#a3a3a3"
QRY_F, QRY_S, QRY_T = "#f6cdc9", "#c0392b", "#a93226"
BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
         "#2171b5", "#08519c", "#08306b"]
ORANGES = ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913",
           "#d94801", "#a63603", "#7f2704"]
SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"

HATCH = {"blue": (LOC_F, "#b8cfe8"), "gray": ("#f4f4f4", "#b9b9b9"),
         "orange": (Q_L, "#efc59c")}


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
N_FP, N_Q, N_LOC = 4, 8, 3
N_SEL = 2                                   # ceil(0.25 * N_Q)

# gate: each query head's share of its own int2 mass, per int2 window
_g = random.Random(5)
_PEAKS = [{1: 1.5}, {1: 0.6, 3: 0.9}, {3: 0.7}, {5: 3.6}]       # head 4 = retrieval head
SHARE = []
for pk in _PEAKS:
    row = [_g.gauss(0, 0.3) + pk.get(c, 0.0) for c in range(N_Q)]
    m = max(row)
    z = sum(math.exp(v - m) for v in row)
    SHARE.append([math.exp(v - m) / z for v in row])
UNION = [max(SHARE[h][c] for h in range(4)) for c in range(N_Q)]
SEL = sorted(sorted(range(N_Q), key=lambda c: -UNION[c])[:N_SEL])

# layout of the cache bar, in panel coordinates
BX0 = 24
SINK_W = 3.6
FP_W, FP_G = 18, 2
Q_W, Q_G = 16, 3
FP_H, Q_H, CARD = 46, 12, 16                 # int2 ~4x shorter than fp16, as its bytes are


def cache_layout():
    x = BX0
    lay = {"sink": (x, x + 5 * SINK_W)}
    x += 5 * SINK_W + 7
    lay["fp"] = [x + i * (FP_W + FP_G) for i in range(N_FP)]
    x = lay["fp"][-1] + FP_W + 9
    lay["q"] = [x + i * (Q_W + Q_G) for i in range(N_Q)]
    x = lay["q"][-1] + Q_W + 9
    lay["loc"] = [x + i * (FP_W + FP_G) for i in range(N_LOC)]
    lay["new"] = lay["loc"][-1] + FP_W + 2
    return lay


LAY = cache_layout()


def cache_bar(g, yb, mode="plain"):
    """The one cache component. ``yb`` is the baseline all blocks stand on.

    mode "plain": every window as stored.  mode "gated": opened int2 windows
    solid, the rest faded to what the step actually touches -- their card.
    """
    x0, x1 = LAY["sink"]
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
            g.rect(x - 1.5, yb - Q_H - 4.5 - CARD, Q_W + 3, Q_H + CARD + 6, stroke=INK, sw=1.6, rx=2)
    for x in LAY["loc"]:
        g.rect(x, yb - FP_H, FP_W, FP_H, fill="hatch-blue", stroke=FP_S, sw=0.9)
    g.rect(LAY["new"], yb - FP_H, 4, FP_H, fill=QRY_F, stroke=QRY_S, sw=0.8)
    g.line(BX0 - 4, yb, LAY["new"] + 8, yb, stroke=INK3, sw=0.8)


def centers():
    return {
        "sink": sum(LAY["sink"]) / 2,
        "fp": (LAY["fp"][0] + LAY["fp"][-1] + FP_W) / 2,
        "q": (LAY["q"][0] + LAY["q"][-1] + Q_W) / 2,
        "loc": (LAY["loc"][0] + LAY["new"] + 4) / 2,
    }


C = centers()

# ---------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------
PW, PH = 390, 464
GAP = 30
LM = 34                                     # left margin carries the feedback arrow
PX = [LM, LM + PW + GAP, LM + 2 * (PW + GAP)]
PY = 46
W = PX[2] + PW + 14
H = PY + PH + 40
YB = 262                                    # the cache bar's baseline, same in every panel


def frame(g, tag, title, sub):
    g.rect(0, 0, PW, PH, fill="#ffffff", stroke=RULE, sw=1.2, rx=9)
    g.text(16, 26, f"{tag}  {title}", size=16.5, weight="bold")
    g.text(16, 45, sub, size=13, fill=INK2, italic=True)


# ---------------------------------------------------------------------------
# (a) cumulative attention ranks windows
# ---------------------------------------------------------------------------
def panel_a(g):
    frame(g, "(a)", "Cumulative attention ranks windows", "re-tiered every 8 decode steps")

    # attention, summed down the query axis
    N, cell, hx, hy = 16, 6.4, 24, 72
    rng = random.Random(3)
    P = []
    for i in range(N):
        lg = [rng.gauss(0, 0.45) + (2.2 if j < 2 else 0) + (1.9 if j == 7 else 0)
              + (1.1 if j == 11 else 0) + (1.2 if i - j <= 1 else 0) for j in range(i + 1)]
        m = max(lg)
        ex = [math.exp(v - m) for v in lg]
        s = sum(ex)
        P.append([e / s for e in ex] + [0.0] * (N - i - 1))
    pmax = max(P[i][j] for i in range(N) for j in range(2, i + 1))
    for i in range(N):
        for j in range(N):
            col = "#f3f4f6" if j > i else ramp(BLUES, (P[i][j] / pmax) ** 0.6)
            g.rect(hx + j * cell, hy + i * cell, cell, cell, fill=col, stroke="#ffffff", sw=0.5)
    g.rect(hx, hy, N * cell, N * cell, stroke=INK3, sw=0.7)
    g.text(hx + N * cell / 2, hy - 7, "attention", size=12, anchor="middle", fill=INK2)
    colsum = [sum(P[i][j] for i in range(N)) for j in range(N)]
    cmax = max(colsum[2:])
    by = hy + N * cell + 22
    for j in range(N):
        hh = 16 * min(1.0, colsum[j] / cmax)
        g.rect(hx + j * cell + 0.8, by - hh, cell - 1.6, hh,
               fill=INK3 if j < 2 else ramp(BLUES, 0.6))
    g.line(hx, by, hx + N * cell, by, stroke=INK3, sw=0.7)

    # window scores, ranked
    sx, sbase, smax = 168, 176, 98
    scores = [0.97, 0.9, 0.83, 0.76] + [0.62, 0.56, 0.5, 0.45, 0.4, 0.35, 0.3, 0.26] + [0.16, 0.11, 0.07]
    kinds = ["fp"] * 4 + ["q"] * 8 + ["drop"] * 3
    bw, bg_ = 11, 2.2
    g.text(sx, 64, "window scores, ranked", size=12, fill=INK2)
    for k, (v, kd) in enumerate(zip(scores, kinds)):
        f, st = {"fp": (FP_F, FP_S), "q": (Q_F, Q_S), "drop": (DROP_F, DROP_S)}[kd]
        g.rect(sx + k * (bw + bg_), sbase - smax * v, bw, smax * v, fill=f, stroke=st, sw=0.8)
    g.line(sx - 2, sbase, sx + 15 * (bw + bg_), sbase, stroke=INK3, sw=0.7)
    g.path(f"M{hx + N * cell + 8},{hy + 52} L{sx - 10},{hy + 52}", stroke=INK2, sw=1.3, arrow="dark")

    # into the tiers
    gx = lambda a, b: sx + (a + b) / 2 * (bw + bg_) - bg_ / 2
    top_card = YB - Q_H - 3 - CARD
    g.path(f"M{gx(0, 4)},{sbase + 4} C{gx(0, 4)},{sbase + 30} {C['fp']},{sbase + 16} "
           f"{C['fp']},{YB - FP_H - 5}", stroke=FP_S, sw=1.6, arrow="blue")
    g.path(f"M{gx(4, 12)},{sbase + 4} C{gx(4, 12)},{sbase + 26} {C['q']},{sbase + 20} "
           f"{C['q']},{top_card - 5}", stroke=Q_S, sw=1.6, arrow="orange")
    xd = gx(12, 15)
    g.line(xd, sbase + 4, xd, sbase + 22, stroke=DROP_S, sw=1.6)
    xmark(g, xd, sbase + 29, r=4.5, sw=1.8)
    g.text(xd - 9, sbase + 33, "dropped", size=12, anchor="end", fill=INK3)

    cache_bar(g, YB)
    ly = YB + 17
    for key, lab in (("sink", "sink"), ("fp", "fp16"), ("q", "int2 + card"), ("loc", "local")):
        g.text(C[key], ly, lab, size=12.5, anchor="middle",
               fill={"sink": SINK_S, "fp": FP_T, "q": Q_T, "loc": FP_T}[key], weight="bold")

    # demotion, magnified: one window becomes 2-bit codes + a card
    zx, zy, zw, zh = 16, 296, PW - 32, 152
    wx = LAY["q"][2]
    g.path(f"M{wx},{YB + 22} L{zx + 8},{zy} M{wx + Q_W},{YB + 22} L{zx + zw - 8},{zy}",
           stroke=RULE, sw=1, dash="3 3")
    g.rect(zx, zy, zw, zh, fill="#fbfcfd", stroke=RULE, sw=1, rx=6)
    g.text(zx + 12, zy + 20, "demoting one window", size=12.5, weight="bold", fill=INK2)
    # the window's keys as a cluster
    cx, cy = zx + 86, zy + 88
    vd = (0.83, -0.56)
    pts = [(-17, 7), (-9, -11), (7, 13), (15, -3), (-22, -3), (2, -15), (10, 5)]
    hot = (62, -42)
    g.line(cx - vd[0] * 40, cy - vd[1] * 40, cx + vd[0] * 92, cy + vd[1] * 92, stroke=CARD_S,
           sw=0.9, dash="4 3")
    g.circle(cx, cy, 30, fill="none", stroke=CARD_S, sw=0.9, extra=' stroke-dasharray="2 3"')
    for dx, dy in pts:
        g.circle(cx + dx, cy + dy, 4, fill=ramp(BLUES, 0.55), stroke="#ffffff", sw=0.7)
    g.circle(cx + hot[0], cy + hot[1], 5, fill=QRY_S, stroke="#ffffff", sw=0.8)
    g.line(cx, cy, cx + hot[0] - 6, cy + hot[1] + 4, stroke=CARD_T, sw=2, arrow="green")
    xmark(g, cx, cy, r=4.5, stroke=INK, sw=2)
    g.text(cx, cy + 48, "8 keys of the window", size=12, anchor="middle", fill=INK2)
    g.text(cx + hot[0] + 8, cy + hot[1] + 4, "outlier", size=12, fill=QRY_T)
    g.text(cx - 34, cy - 34, "centroid", size=12, anchor="end", fill=INK)
    g.line(cx - 32, cy - 31, cx - 5, cy - 5, stroke=INK3, sw=0.7)
    # result
    rx0 = zx + 208
    g.line(rx0 - 34, cy - 8, rx0 - 8, cy - 8, stroke=INK2, sw=1.3, arrow="dark")
    g.use("card", rx0, zy + 34, 44, 44)
    g.text(rx0 + 54, zy + 50, "card", size=12.5, weight="bold", fill=CARD_T)
    g.text(rx0 + 54, zy + 66, "centroid +", size=12, fill=INK2)
    g.text(rx0 + 54, zy + 81, "outlier axis", size=12, fill=INK2)
    g.rect(rx0, zy + 104, 44, 11, fill=Q_F, stroke=Q_S, sw=0.9)
    g.text(rx0 + 54, zy + 114, "2-bit K, V", size=12.5, weight="bold", fill=Q_T)
    g.text(rx0 + 54, zy + 130, "¼ of fp16", size=12, fill=INK2)


# ---------------------------------------------------------------------------
# (b) cards choose which int2 windows are read
# ---------------------------------------------------------------------------
def panel_b(g):
    frame(g, "(b)", "Cards choose what to read", "every decode step, per KV head")

    # the query: 4 heads sharing one KV head
    qx, qy = C["q"] - 22, 66
    for h in range(4):
        g.rect(qx, qy + h * 8, 44, 6, fill=QRY_F, stroke=QRY_S, sw=0.8, rx=1.5)
    g.text(qx - 8, qy + 18, "query", size=12.5, anchor="end", weight="bold", fill=QRY_T)
    g.text(qx - 8, qy + 33, "4 heads", size=12, anchor="end", fill=QRY_T)

    # every card is scored against it
    top_card = YB - Q_H - 3 - CARD
    umax = max(UNION)
    for c, x in enumerate(LAY["q"]):
        v = UNION[c] / umax
        sel = c in SEL
        g.line(C["q"], qy + 34, x + Q_W / 2, top_card - 2,
               stroke=Q_S if sel else "#c3c8ce", sw=0.8 + 2.6 * v)
    g.text(C["q"] + 30, qy + 64, "score every card", size=12, fill=INK2)

    for key in ("fp", "loc"):
        x1 = LAY["fp"][0] if key == "fp" else LAY["loc"][0]
        x2 = (LAY["fp"][-1] + FP_W) if key == "fp" else LAY["new"] + 4
        bracket(g, x1, x2, YB - FP_H - 8, up=True)
        g.text((x1 + x2) / 2, YB - FP_H - 14, "always read", size=12, anchor="middle", fill=INK3)

    # selected columns, shaded from card to vote
    gy, ch = 296, 12
    uy = gy + 4 * (ch + 2) + 12
    for c in SEL:
        x = LAY["q"][c]
        g.rect(x - 2.5, top_card - 6, Q_W + 5, uy + ch + 4 - (top_card - 6), fill="#fdebd9", rx=3)
    cache_bar(g, YB, mode="gated")

    # each head's own preference, then the union across the group
    for h in range(4):
        g.text(LAY["q"][0] - 8, gy + h * (ch + 2) + 10, f"head {h + 1}", size=11.5, anchor="end",
               fill=INK2)
        for c, x in enumerate(LAY["q"]):
            g.rect(x, gy + h * (ch + 2), Q_W, ch, fill=ramp(ORANGES, SHARE[h][c] ** 0.6 * 1.05),
                   stroke="#ffffff", sw=0.6)
    g.line(C["q"], gy + 4 * (ch + 2) + 1, C["q"], uy - 2, stroke=INK3, sw=1, arrow="gray")
    g.text(LAY["q"][0] - 8, uy + 10, "union", size=11.5, anchor="end", weight="bold", fill=INK)
    for c, x in enumerate(LAY["q"]):
        g.rect(x, uy, Q_W, ch, fill=ramp(ORANGES, UNION[c] ** 0.6 * 1.05), stroke="#ffffff", sw=0.6)
        if c in SEL:
            g.rect(x - 1, uy - 1, Q_W + 2, ch + 2, stroke=INK, sw=1.6)
    g.text(C["q"], uy + 32, "top 25% opened", size=12.5, anchor="middle", weight="bold", fill=Q_T)
    g.text(LAY["q"][-1] + Q_W + 10, gy + 3 * (ch + 2) + 10, "retrieval", size=11.5, fill=INK3)
    g.text(LAY["q"][-1] + Q_W + 10, gy + 3 * (ch + 2) + 24, "head", size=11.5, fill=INK3)

    # key
    ky = PH - 26
    g.rect(24, ky - 11, 14, 13, fill=Q_F, stroke=INK, sw=1.4)
    g.text(44, ky, "opened", size=12, fill=INK2)
    g.add('<g opacity="0.42">')
    g.rect(112, ky - 11, 14, 13, fill=Q_F, stroke=Q_S, sw=0.9, dash="2 1.5")
    g.add("</g>")
    g.text(132, ky, "only its card is read", size=12, fill=INK2)


# ---------------------------------------------------------------------------
# (c) one fused pass attends and credits
# ---------------------------------------------------------------------------
def panel_c(g):
    frame(g, "(c)", "One fused pass reads the cache", "every decode step, every layer")

    # tiles of whole windows, and the gathered int2 tile
    ty = YB - FP_H - 12
    bracket(g, LAY["sink"][0], LAY["fp"][-1] + FP_W, ty, up=True, stroke=INK2)
    g.text(C["fp"] - 10, ty - 7, "tile", size=12, anchor="middle", fill=INK2)
    bracket(g, LAY["loc"][0], LAY["new"] + 4, ty, up=True, stroke=INK2)
    g.text(C["loc"], ty - 7, "tile", size=12, anchor="middle", fill=INK2)
    top_card = YB - Q_H - 3 - CARD
    xs = [LAY["q"][c] + Q_W / 2 for c in SEL]
    # the opened windows, gathered into one tile and decoded back to full height
    tcx = C["q"]
    bw_, bh_, bgap = 20, FP_H, 6
    tw = N_SEL * bw_ + (N_SEL - 1) * bgap + 24
    tx0, ty0, th_ = tcx - tw / 2, 88, FP_H + 22
    g.rect(tx0, ty0, tw, th_, fill="#fffaf5", stroke=Q_S, sw=1.3, rx=5)
    for k in range(N_SEL):
        x = tx0 + 12 + k * (bw_ + bgap)
        g.rect(x, ty0 + 11, bw_, bh_, fill=FP_F, stroke=Q_S, sw=1.2)
        for t in range(1, 8):
            g.line(x + t * bw_ / 8, ty0 + 13, x + t * bw_ / 8, ty0 + 9 + bh_, stroke="#ffffff", sw=0.5)
    g.text(tx0 - 10, ty0 + 30, "gathered tile", size=12.5, anchor="end", weight="bold", fill=Q_T)
    g.text(tx0 - 10, ty0 + 46, "2-bit → fp16", size=12, anchor="end", fill=Q_T)
    g.text(tx0 - 10, ty0 + 61, "on chip", size=12, anchor="end", fill=Q_T)
    for k, x in enumerate(xs):
        xt = tx0 + 12 + k * (bw_ + bgap) + bw_ / 2
        g.path(f"M{x},{top_card - 7} C{x},{top_card - 40} {xt},{ty0 + th_ + 30} {xt},{ty0 + th_ + 1}",
               stroke=Q_S, sw=1, dash="3 3")

    cache_bar(g, YB, mode="gated")

    # into the kernel
    ky, kh = 316, 26
    kx1 = 294
    for x, col, arrow in [(LAY["fp"][1] + FP_W / 2, FP_S, "blue"), (C["loc"], FP_S, "blue")] + \
            [(LAY["q"][c] + Q_W / 2, Q_S, "orange") for c in SEL]:
        g.line(x, YB + 3, x, ky - 3, stroke=col, sw=1.6, arrow=arrow)
    for c, x in enumerate(LAY["q"]):
        if c not in SEL:
            g.line(x + Q_W / 2, YB + 3, x + Q_W / 2, ky - 3, stroke=CARD_S, sw=1, dash="2.5 2",
                   arrow="green")
    g.text(LAY["q"][0] - 6, YB + 36, "via card", size=12, anchor="end", fill=CARD_T)
    g.rect(16, ky, kx1 - 16, kh, fill="#f3f4f6", stroke=INK3, sw=1, rx=6)
    g.text((16 + kx1) / 2, ky + 17.5, "fused two-tier attention", size=13, anchor="middle",
           weight="bold")
    g.line(kx1 + 2, ky + kh / 2, kx1 + 16, ky + kh / 2, stroke=INK, sw=1.4, arrow="dark")
    g.rect(kx1 + 20, ky - 7, PW - 16 - kx1 - 20, kh + 14, fill="#ffffff", stroke=INK2, sw=1, rx=5)
    g.text((kx1 + 20 + PW - 16) / 2, ky + 10, "attention", size=12, anchor="middle")
    g.text((kx1 + 20 + PW - 16) / 2, ky + 25, "output", size=12, anchor="middle")

    # every window's mass this step, under its own column
    mbase, mh = 432, 48
    g.line(C["q"], ky + kh + 2, C["q"], mbase - mh - 8, stroke=INK, sw=1.4, arrow="dark")
    g.text(24, ky + kh + 26, "window mass", size=12, fill=INK2)
    rng = random.Random(8)
    fp_v, loc_v = [0.95, 0.55, 0.75, 0.5], [0.6, 0.8, 0.9]
    for x, v in zip(LAY["fp"] + LAY["loc"], fp_v + loc_v):
        g.rect(x, mbase - mh * v, FP_W, mh * v, fill=FP_F, stroke=FP_S, sw=0.8)
    for c, x in enumerate(LAY["q"]):
        if c in SEL:
            v = 0.85 if c == SEL[-1] else 0.45
            g.rect(x, mbase - mh * v, Q_W, mh * v, fill=Q_F, stroke=Q_S, sw=0.8)
        else:
            v = 0.1 + 0.1 * rng.random()
            g.rect(x, mbase - mh * v, Q_W, mh * v, fill="hatch-orange", stroke="#d9a57a", sw=0.8,
                   dash="2 1.5")
    g.line(BX0 - 4, mbase, LAY["new"] + 8, mbase, stroke=INK3, sw=0.7)
    g.rect(LAY["q"][0], mbase + 9, 12, 11, fill="hatch-orange", stroke="#d9a57a", sw=0.8,
           dash="2 1.5")
    g.text(LAY["q"][0] + 18, mbase + 19, "credited from its card", size=12, fill=INK2)


# ---------------------------------------------------------------------------
# legend and cross-panel arrows
# ---------------------------------------------------------------------------
def legend(g):
    x, y = LM, 24
    items = [("sink", SINK_F, SINK_S, None), ("fp16 window", FP_F, FP_S, None),
             ("local window", "hatch-blue", FP_S, None), ("int2 window", Q_F, Q_S, None),
             ("card", "card", None, None), ("query / new token", QRY_F, QRY_S, None),
             ("dropped", "x", None, None)]
    for lab, f, s, d in items:
        if f == "card":
            g.use("card", x, y - 13, 16, 16)
        elif f == "x":
            xmark(g, x + 8, y - 5, r=4.5, sw=1.8)
        else:
            g.rect(x, y - 11, 16, 13, fill=f, stroke=s, sw=0.9, dash=d)
        g.text(x + 22, y, lab, size=12.5, fill=INK2)
        x += 40 + len(lab) * 6.6
    g.text(W - 14, y, "Llama-3.1-8B · window = 8 tokens · int2 window + card ≈ ¼ of an fp16 window",
           size=12.5, anchor="end", fill=INK3)


def arrows(g):
    y = PY + YB - 26
    for k in range(2):
        g.line(PX[k] + PW + 3, y, PX[k + 1] - 4, y, stroke=INK, sw=2, arrow="dark")
    # every step's window mass is added to the cumulative score: (c) -> (a)
    x_from = PX[2] + C["q"]
    yb = PY + PH + 22
    g.path(f"M{x_from},{PY + PH + 2} L{x_from},{yb - 6} Q{x_from},{yb} {x_from - 6},{yb} "
           f"L{16},{yb} Q{10},{yb} {10},{yb - 6} L{10},{PY + 170} Q{10},{PY + 164} {16},{PY + 164} "
           f"L{PX[0] - 3},{PY + 164}", stroke=FP_S, sw=2, arrow="blue")
    g.add(f'<rect x="{PX[1] + 60}" y="{yb - 9}" width="316" height="18" fill="#ffffff"/>')
    g.text(PX[1] + 218, yb + 4.5, "window mass is added to the cumulative score",
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
            + marker("orange", Q_S) + marker("green", CARD_S) + card + "</defs>")


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
