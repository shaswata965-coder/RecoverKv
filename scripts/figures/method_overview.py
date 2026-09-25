"""Build ``reports/figures/method_overview.html`` -- the end-to-end method figure.

The figure is one self-contained SVG (its CSS lives inside it), wrapped in a
bare HTML page. Keeping it pure SVG is what makes it exportable without a
converter: the ``<svg>`` element is itself a valid ``.svg`` file, Chromium
prints it to a single vector PDF page, and a screenshot gives the PNG.
``scripts/figures/export_figure.mjs`` does all three.

Everything drawn here is taken from the code, not from memory: the tiers and
their order (``policy.compute_two_tier_retain``), the int2 layout
(``quant/quantizer.py``), the card (``quant/sketch.py``), the gate's per-head
share union (``gate_kernel``), the kernel's tiling and epilogue
(``decode_kernel._two_tier_decode_kernel``) and the skipped-window credit
(``scorer.fill_skipped_window_scores``). Toy sizes are used where the real ones
would not fit (4 tokens per window in panel (a), 24 evictable windows in (b));
the byte counts and the operating point are the shipped ones.

Run:  python scripts/figures/method_overview.py
"""

from __future__ import annotations

import html
import math
import random
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "reports" / "figures" / "method_overview.html"

W, H = 1800, 1224

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
INK, INK2, INK3 = "#1d2329", "#4b5563", "#8a939c"
RULE = "#c3cad2"
SINK_F, SINK_S = "#e4ddf1", "#7b68a6"
FP_F, FP_S, FP_T = "#cfe0f3", "#3c6ea8", "#2c5a8f"
LOC_F = "#eef4fb"
Q_F, Q_S, Q_T, Q_L = "#f8cfa6", "#c8691f", "#a0500f", "#fdf1e4"
CARD_F, CARD_S, CARD_T = "#d4ecdc", "#3b8a5a", "#2a6d45"
DROP_F, DROP_S = "#ededed", "#a3a3a3"
QRY_F, QRY_S, QRY_T = "#f6cdc9", "#c0392b", "#a93226"

BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
         "#2171b5", "#08519c", "#08306b"]
ORANGES = ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913",
           "#d94801", "#a63603", "#7f2704"]

SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"
SERIF = "'Times New Roman', 'Liberation Serif', Times, serif"


def ramp(stops, t):
    t = max(0.0, min(1.0, t))
    x = t * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    a, b = stops[i], stops[i + 1]
    ca = [int(a[k:k + 2], 16) for k in (1, 3, 5)]
    cb = [int(b[k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(p + (q - p) * f):02x}" for p, q in zip(ca, cb))


# ---------------------------------------------------------------------------
# text with inline math:  "plain $math$ plain"
#   math: letters italic, digits/operators upright, _{..} / ^{..} scripts,
#   \rm{..} upright, \bar{X} overbar, \, thin space
# ---------------------------------------------------------------------------
UPRIGHT = {"exp", "max", "mean", "LSE", "ln", "log", "round", "argmax", "lse",
           "WSUM", "WMAX", "SEL", "min", "RoPE", "sel", "read", "skip", "fp16",
           "int2", "int4", "int8", "B"}
LETTER = re.compile(r"[A-Za-z\u03b1-\u03c9\u2113]+")


def _math_runs(src, shift=0.0, scale=1.0, upright=False, bar=False, kind=0):
    """-> [(text, shift_em, scale, upright, bar, kind)]

    ``shift_em`` is the baseline offset in em of the base math size (down is
    positive), ``scale`` the font-size factor, ``kind`` the top-level script
    this run belongs to (-1 sub, +1 sup, 0 base) -- what stacking keys on.
    """
    out, buf, i = [], "", 0

    def flush():
        nonlocal buf
        if buf:
            out.append((buf, shift, scale, upright, bar, kind))
            buf = ""

    def group(j):
        if j < len(src) and src[j] == "{":
            depth, k = 1, j + 1
            while depth:
                depth += {"{": 1, "}": -1}.get(src[k], 0)
                k += 1
            return src[j + 1:k - 1], k
        return src[j], j + 1

    while i < len(src):
        c = src[i]
        if c in "_^":
            flush()
            g, i = group(i + 1)
            sub = c == "_"
            out += _math_runs(g, shift + (0.28 if sub else -0.42) * scale,
                              max(0.55, scale * 0.72), upright, bar,
                              kind or (-1 if sub else 1))
        elif src.startswith(r"\rm", i):
            flush()
            g, i = group(i + 3)
            out += _math_runs(g, shift, scale, True, bar, kind)
        elif src.startswith(r"\bar", i):
            flush()
            g, i = group(i + 4)
            out += _math_runs(g, shift, scale, upright, True, kind)
        elif src.startswith(r"\,", i):
            buf += " "
            i += 2
        elif src.startswith("\\{", i) or src.startswith("\\}", i):
            buf += src[i + 1]
            i += 2
        elif c == "\\" and i + 1 < len(src) and src[i + 1].isalpha():
            flush()                          # \ln, \log, \exp ... -> upright word
            m = re.match(r"\\([A-Za-z]+)", src[i:])
            out.append((m.group(1), shift, scale, True, bar, kind))
            i += len(m.group(0))
        else:
            buf += c
            i += 1
    flush()
    return out


def _split_letters(run):
    text, shift, scale, up, bar, kind = run
    parts, pos = [], 0
    for m in LETTER.finditer(text):
        if m.start() > pos:
            parts.append((text[pos:m.start()], shift, scale, True, bar, kind))
        word = m.group(0)
        parts.append((word, shift, scale, up or word in UPRIGHT, bar, kind))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], shift, scale, True, bar, kind))
    return parts


#: advance widths (em) of the letters that carry an overbar, Times italic
_BAR_W = {"v": 0.444, "S": 0.5, "a": 0.5, "x": 0.444}


def _em(text):
    """Rough advance width of ``text`` in em, for stacking scripts (Times)."""
    w = 0.0
    for ch in text:
        if ch in "ijltfr()[],.;:'′|!":
            w += 0.30
        elif ch.isdigit() or ch.islower():
            w += 0.50
        elif ch.isupper():
            w += 0.66
        elif ch in "  ":
            w += 0.20
        else:
            w += 0.56
    return w


def rich(s, size):
    """SVG tspans for a string with $math$ segments."""
    segs = re.split(r"(\$[^$]*\$)", s)
    out, cur, pending = [], 0.0, 0.0
    msize = size * 1.14
    for seg in segs:
        if not seg:
            continue
        if seg.startswith("$"):
            runs = []
            for r in _math_runs(seg[1:-1]):
                runs += _split_letters(r)
            # stack a subscript group directly after a superscript group (and
            # vice versa) under it, the way TeX sets a^{(t)}_{h,w}
            prev_kind, grp_w, under = 0, 0.0, None
            for text, shift, scale, up, bar, kind in runs:
                fs = msize * scale
                dx, pending = pending, 0.0
                if kind != prev_kind:
                    if kind and prev_kind and kind == -prev_kind:
                        dx, under = -grp_w, grp_w          # stack under/over it
                    elif under is not None:
                        dx, under = max(0.0, under - grp_w), None
                    grp_w = 0.0
                grp_w += _em(text) * fs
                prev_kind = kind
                off = shift * msize
                dy = off - cur
                cur = off
                style = "" if up else ' font-style="italic"'
                dy_s = f' dy="{dy:.2f}"' if abs(dy) > 1e-6 else ""
                dx_s = f' dx="{dx:.2f}"' if abs(dx) > 1e-6 else ""
                out.append(f'<tspan font-family="{SERIF}" font-size="{fs:.1f}"'
                           f'{style}{dx_s}{dy_s}>{html.escape(text)}</tspan>')
                if bar:
                    # an overbar: a macron glyph centred over the letter, lowered
                    # to x-height for lowercase, then the pen is put back
                    wl = _BAR_W.get(text, 0.5) * fs
                    wm = 0.333 * fs
                    ital = 0.0 if up else 0.07 * fs
                    lift = 0.17 * fs if text.islower() else -0.02 * fs
                    out.append(f'<tspan font-family="{SERIF}" font-size="{fs:.1f}" '
                               f'dx="{-(wl + wm) / 2 + ital:.2f}" dy="{lift:.2f}">¯</tspan>')
                    cur += lift
                    pending = (wl - wm) / 2 - ital
            if under is not None:                  # segment ended on a stack
                pending = max(0.0, under - grp_w)
        else:
            dy = -cur
            cur = 0.0
            dy_s = f' dy="{dy:.2f}"' if abs(dy) > 1e-6 else ""
            dx_s = f' dx="{pending:.2f}"' if pending > 1e-6 else ""
            pending = 0.0
            out.append(f"<tspan{dx_s}{dy_s}>{html.escape(seg)}</tspan>")
    return "".join(out)


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------
#: hatch name -> (background, line colour)
HATCH = {"blue": (LOC_F, "#b8cfe8"), "gray": ("#f4f4f4", "#b9b9b9"),
         "orange": (Q_L, "#efc59c")}


def hatch_lines(x, y, w, h, ink, gap=5.0, sw=1.1):
    """45-degree hatching clipped to a rectangle, as one vector path."""
    x1, y1 = x + w, y + h
    step = gap * math.sqrt(2)
    k = x + y + step / 2
    d = []
    while k < x1 + y1:
        xa, xb = max(x, k - y1), min(x1, k - y)
        if xb - xa > 0.2:
            d.append(f"M{xa:.2f},{k - xa:.2f}L{xb:.2f},{k - xb:.2f}")
        k += step
    return (f'<path d="{"".join(d)}" stroke="{ink}" stroke-width="{sw}" fill="none" '
            f'stroke-linecap="butt"/>')


class G:
    def __init__(self):
        self.p = []

    def add(self, s):
        self.p.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0,
             dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        if fill.startswith("url(#hatch-"):
            # Hatching is drawn as real line segments, not an SVG <pattern>:
            # Chromium's PDF backend rasterises pattern fills, and the figure
            # has to stay vector end to end.
            bg, ink = HATCH[fill[len("url(#hatch-"):-1]]
            self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                     f'rx="{rx}" fill="{bg}" stroke="none"/>')
            self.add(hatch_lines(x, y, w, h, ink))
            fill = "none"
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                 f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=INK2, sw=1.0, dash=None, arrow=None,
             start=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{arrow})"' if arrow else ""
        ms = f' marker-start="url(#ah-{start})"' if start else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{m}{ms}{extra}/>')

    def path(self, d, stroke=INK2, sw=1.0, fill="none", dash=None, arrow=None,
             start=None, extra=""):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{arrow})"' if arrow else ""
        ms = f' marker-start="url(#ah-{start})"' if start else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
                 f'{da}{m}{ms}{extra}/>')

    def text(self, x, y, s, size=13, anchor="start", weight=None, fill=INK,
             italic=False, cls="t", rotate=None, extra=""):
        w = f' font-weight="{weight}"' if weight else ""
        it = ' font-style="italic"' if italic else ""
        rot = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(f'<text class="{cls}" x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
                 f'text-anchor="{anchor}" fill="{fill}"{w}{it}{rot}{extra}>'
                 f"{rich(s, size)}</text>")

    def circle(self, cx, cy, r, fill="none", stroke="none", sw=1.0, extra=""):
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def open(self, x, y, extra=""):
        self.add(f'<g transform="translate({x},{y})"{extra}>')

    def close(self):
        self.add("</g>")


def chip(g, x, y, label, fill, stroke, tcol, size=12, pad=6, h=20, w=None, dash=None):
    """A rounded label chip; returns its width."""
    if w is None:
        w = len(re.sub(r"[${}_^\\]|\\rm", "", label)) * size * 0.56 + 2 * pad
    g.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1, rx=4, dash=dash)
    g.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, anchor="middle", fill=tcol)
    return w


def brace(g, x1, x2, y, down=True, stroke=INK3, sw=1.0, depth=6):
    """Horizontal curly-ish bracket from x1 to x2 opening upward (down=True)."""
    s = 1 if down else -1
    xm = (x1 + x2) / 2
    g.path(f"M{x1},{y} q0,{s*depth/2} {depth/2},{s*depth/2} L{xm - depth/2},{y + s*depth/2} "
           f"q{depth/2},0 {depth/2},{s*depth/2} q0,{-s*depth/2} {depth/2},{-s*depth/2} "
           f"L{x2 - depth/2},{y + s*depth/2} q{depth/2},0 {depth/2},{-s*depth/2}",
           stroke=stroke, sw=sw)


def frame(g, w, h, tag, title, sub):
    g.rect(0, 0, w, h, fill="#ffffff", stroke=RULE, sw=1.2, rx=9)
    g.text(16, 27, f"{tag}", size=17, weight="bold")
    g.text(44, 27, title, size=17, weight="bold")
    g.text(16, 47, sub, size=12.5, fill=INK2, italic=True)


def num(g, x, y, n, fill=INK, size=11):
    """Circled step number."""
    g.circle(x, y, 8.2, fill=fill, stroke="none")
    g.text(x, y + 3.9, str(n), size=size, anchor="middle", fill="#ffffff", weight="bold")


# ---------------------------------------------------------------------------
# shared toy data -- one cache, seen by every panel
# ---------------------------------------------------------------------------
E = 24                      # evictable windows in the toy cache
K_FP, N_Q = 4, 12           # tier sizes in the toy
N_SEL = 3                   # ceil(0.25 * N_Q)
_rng = random.Random(11)
SCORES = [0.30, 0.62, 0.21, 0.40, 0.12, 0.83, 0.33, 0.47, 0.09, 0.27, 0.36, 0.71,
          0.15, 0.51, 0.24, 0.94, 0.43, 0.18, 0.31, 0.56, 0.07, 0.38, 0.66, 0.26]
_rank = sorted(range(E), key=lambda i: -SCORES[i])
FP_IDS = sorted(_rank[:K_FP])
Q_IDS = sorted(_rank[K_FP:K_FP + N_Q])
DROP_IDS = sorted(_rank[K_FP + N_Q:])
TIER = {i: "fp" for i in FP_IDS} | {i: "q" for i in Q_IDS} | {i: "drop" for i in DROP_IDS}
THR_FP = (SCORES[_rank[K_FP - 1]] + SCORES[_rank[K_FP]]) / 2
THR_Q = (SCORES[_rank[K_FP + N_Q - 1]] + SCORES[_rank[K_FP + N_Q]]) / 2

# gate: per-head log-mass over the N_Q int2 windows (4 query heads, one KV head)
_g = random.Random(5)
LOGM = []
for h, (peaks, off, sd) in enumerate([
        ({1: 1.3, 9: 1.0}, 0.0, 0.35),          # diffuse head
        ({4: 1.5, 9: 0.9}, 3.0, 0.35),          # louder head: large q.anchor baseline
        ({4: 0.7, 10: 0.6}, 1.2, 0.40),
        ({6: 4.2}, -1.0, 0.25)]):               # retrieval head: one needle window
    LOGM.append([off + _g.gauss(0, sd) + peaks.get(c, 0.0) for c in range(N_Q)])


def _lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


SHARE = [[x - _lse(row) for x in row] for row in LOGM]
UNION = [max(SHARE[h][c] for h in range(4)) for c in range(N_Q)]
SEL = sorted(sorted(range(N_Q), key=lambda c: -UNION[c])[:N_SEL])
RAWMAX_SEL = sorted(sorted(range(N_Q), key=lambda c: -max(LOGM[h][c] for h in range(4)))[:N_SEL])


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
X0 = 58
GAP = 30
COLW = [488, 640, 554]
COLX = [X0, X0 + COLW[0] + GAP, X0 + COLW[0] + GAP + COLW[1] + GAP]
TOP_Y, TOP_H = 54, 548
MID = 56
BOT_Y, BOT_H = TOP_Y + TOP_H + MID, 560


# ---------------------------------------------------------------------------
# (a) cumulative window attention
# ---------------------------------------------------------------------------
def panel_a(g):
    w, h = COLW[0], TOP_H
    frame(g, w, h, "(a)", "Cumulative window attention",
          "prefill sums attention down the query axis; each decode step adds its mass")

    N, SINK, WS, NW, cell = 22, 2, 4, 5, 7.6
    hx, hy = 46, 86
    rng = random.Random(3)
    P = [[0.0] * N for _ in range(N)]
    for i in range(N):
        lg = []
        for j in range(i + 1):
            v = rng.gauss(0, 0.45)
            v += 2.3 if j < SINK else 0
            v += 2.0 if j == 8 else 0
            v += 1.2 if j == 13 else 0
            v += 1.3 if i - j <= 1 else 0
            lg.append(v)
        m = max(lg)
        ex = [math.exp(v - m) for v in lg]
        s = sum(ex)
        for j in range(i + 1):
            P[i][j] = ex[j] / s
    pmax = max(P[i][j] for i in range(N) for j in range(SINK, i + 1))
    g.text(hx + N * cell / 2, hy - 7, "keys $j$ →", size=12, anchor="middle", fill=INK2)
    g.text(hx - 9, hy + N * cell / 2, "queries $i$ →", size=12, anchor="middle",
           fill=INK2, rotate=-90)
    for i in range(N):
        for j in range(N):
            if j > i:
                col = "#f3f4f6"
            else:
                col = ramp(BLUES, (P[i][j] / pmax) ** 0.6)
            g.rect(hx + j * cell, hy + i * cell, cell, cell, fill=col,
                   stroke="#ffffff", sw=0.6)
    g.rect(hx, hy, N * cell, N * cell, stroke=INK3, sw=0.8)

    # column sums
    s = [sum(P[i][j] for i in range(N)) for j in range(N)]
    smax = max(s[SINK:])
    by, bh = hy + N * cell + 44, 34
    g.text(hx - 8, by - 12, "$s_{j}$", size=13, anchor="end")
    for j in range(N):
        x = hx + j * cell + 1
        if j < SINK:
            g.rect(x, by - bh, cell - 2, bh, fill="url(#hatch-gray)", stroke=INK3, sw=0.6)
        else:
            hh = bh * s[j] / smax
            g.rect(x, by - hh, cell - 2, hh, fill=ramp(BLUES, 0.55), stroke="none")
    g.line(hx, by, hx + N * cell, by, stroke=INK3, sw=0.8)
    g.text(hx + SINK * cell, by + 14, "sink", size=11, anchor="end", fill=INK3)
    g.text(hx + SINK * cell, by + 26, "(unscored)", size=10.5, anchor="end", fill=INK3)

    # window grouping
    wy, wbh = by + 74, 30
    g.text(hx - 8, wy - 10, "$S_{h,w}$", size=13, anchor="end")
    Sw = []
    for k in range(NW):
        x1 = hx + (SINK + k * WS) * cell
        x2 = x1 + WS * cell
        brace(g, x1 + 1, x2 - 1, by + 6, down=True, depth=6)
        Sw.append(sum(s[SINK + k * WS: SINK + (k + 1) * WS]))
    wmax = max(Sw)
    for k in range(NW):
        x1 = hx + (SINK + k * WS) * cell
        xm = x1 + WS * cell / 2
        g.line(xm, by + 13, xm, wy - wbh - 4, stroke=INK3, sw=0.8, arrow="gray")
        hh = wbh * Sw[k] / wmax
        g.rect(x1 + 5, wy - hh, WS * cell - 10, hh, fill=ramp(BLUES, 0.75))
        g.text(xm, wy + 13, f"$w_{{{k + 1}}}$", size=12, anchor="middle", fill=INK2)
    g.line(hx + SINK * cell, wy, hx + N * cell, wy, stroke=INK3, sw=0.8)
    g.text(hx + N * cell + 8, wy - 8, "Σ over the", size=10.5, fill=INK3)
    g.text(hx + N * cell + 8, wy + 5, "window", size=10.5, fill=INK3)

    # equations
    ex, ey = 256, 84
    lines = [
        ("prefill, per key (FA2-backward form)", None),
        (None, "$s_{h,j} = Σ_{i} \\rm{exp}(q_{i}·k_{j}/√d − L_{i})$"),
        ("$L_{i}$ reused from the forward pass", None),
        ("per window, sink stripped", None),
        (None, "$S_{h,w} = Σ_{j ∈ w} s_{h,j}$"),
        ("every decode step $t$  (mass from (f))", None),
        (None, "$S_{h,w} ← S_{h,w} + a^{(t)}_{h,w}$"),
        ("rank key at eviction", None),
        (None, "$\\bar{S}_{w} = \\rm{mean}_{h} S_{h,w}$"),
    ]
    y = ey
    for note, eq in lines:
        if note:
            y += 17
            g.text(ex, y, note, size=11.5, fill=INK2)
        else:
            y += 24
            g.text(ex + 6, y, eq, size=14)
            y += 6
    g.line(ex - 12, ey + 4, ex - 12, y + 6, stroke=RULE, sw=1)

    # timeline: cumulative score of three windows, evictions every ws steps
    tx0, tx1, ty0, ty1 = 84, 424, 444, 514
    g.line(16, ty0 - 40, w - 16, ty0 - 40, stroke=RULE, sw=1, dash="3 3")
    g.text(16, ty0 - 20, "$\\bar{S}_{w}$ of three windows across decode steps", size=12, fill=INK2)
    g.rect(tx0 - 58, ty1 - 18, 50, 18, fill="#f3f4f6", stroke=INK3, sw=0.8, rx=3)
    g.text(tx0 - 33, ty1 - 5, "prefill", size=11, anchor="middle", fill=INK2)
    step = (tx1 - tx0) / 24
    g.line(tx0, ty1, tx1 + 6, ty1, stroke=INK2, sw=1, arrow="dark")
    for t in range(25):
        x = tx0 + t * step
        big = t % 8 == 0
        g.line(x, ty1, x, ty1 + (5 if big else 2.5), stroke=INK2, sw=0.8)
        if big:
            g.text(x, ty1 + 17, str(t), size=11, anchor="middle", fill=INK2)
            g.line(x, ty0 + 2, x, ty1, stroke=INK3, sw=0.7, dash="2 3")
            g.path(f"M{x - 4.5},{ty0 - 6} L{x + 4.5},{ty0 - 6} L{x},{ty0 + 1} z",
                   fill=INK2, stroke="none")
    g.text(tx1 + 12, ty1 + 5, "$t$", size=13)
    g.path(f"M{w - 180},{ty0 - 29} L{w - 171},{ty0 - 29} L{w - 175.5},{ty0 - 22} z", fill=INK2,
           stroke="none")
    g.text(w - 16, ty0 - 20, "eviction, every $ws$ = 8 steps", size=11.5, anchor="end", fill=INK2)
    curves = [(0.34, 0.021, FP_S, "fp16", None), (0.22, 0.0075, Q_S, "int2", None),
              (0.12, 0.0012, DROP_S, "dropped", 16)]
    ys = lambda v: ty1 - 3 - (ty1 - ty0 - 12) * v / 0.9
    for base, inc, col, lab, end in curves:
        pts = [(tx0 - 8, ys(base))]
        v = base
        stop = end if end is not None else 24
        for t in range(stop + 1):
            if t:
                v += inc * (1 + 0.8 * math.sin(t * 1.7 + base * 10))
            pts.append((tx0 + t * step, ys(v)))
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        g.path(d, stroke=col, sw=1.8)
        xe, ye = pts[-1]
        if end is not None:
            g.path(f"M{xe - 4},{ye - 4} L{xe + 4},{ye + 4} M{xe - 4},{ye + 4} L{xe + 4},{ye - 4}",
                   stroke=col, sw=1.6)
            g.text(xe + 8, ye + 4, lab, size=11, fill=INK3)
        else:
            g.text(xe + 5, ye + 4, lab, size=11, fill=col, weight="bold")


# ---------------------------------------------------------------------------
# (b) eviction: rank -> tier -> compact
# ---------------------------------------------------------------------------
def tier_fill(t):
    return {"fp": (FP_F, FP_S), "q": (Q_F, Q_S), "drop": (DROP_F, DROP_S)}[t]


def panel_b(g):
    w, h = COLW[1], TOP_H
    frame(g, w, h, "(b)", "Eviction: rank → tier → compact",
          "every $ws$ = 8 steps; all $L$ layers fold into $R = L·B$ rows, one compiled pass")

    cw, cg = 15, 1.6
    ax = 20
    ay = 178                      # axis cells top
    sink_w = 5 * 5
    for k in range(5):
        g.rect(ax + k * 5, ay, 5, 22, fill=SINK_F, stroke=SINK_S, sw=0.6)
    ex0 = ax + sink_w + 6
    # evictable windows with their score bars
    bb, bmax = ay - 8, 86
    for i in range(E):
        x = ex0 + i * (cw + cg)
        f, st = tier_fill(TIER[i])
        hh = bmax * SCORES[i]
        g.rect(x, bb - hh, cw, hh, fill=f, stroke=st, sw=0.8)
        g.rect(x, ay, cw, 22, fill=f, stroke=st, sw=0.8)
        if TIER[i] == "drop":
            g.path(f"M{x + 3},{ay + 5} L{x + cw - 3},{ay + 17} M{x + 3},{ay + 17} L{x + cw - 3},{ay + 5}",
                   stroke=DROP_S, sw=1)
        g.text(x + cw / 2, ay + 34, str(i), size=9.5, anchor="middle", fill=INK3)
    ex1 = ex0 + E * (cw + cg) - cg
    g.line(ex0 - 2, bb, ex1 + 2, bb, stroke=INK3, sw=0.8)
    g.text(ex0 - 5, bb - bmax + 10, "$\\bar{S}_{w}$", size=13, anchor="end")
    yfp, yq = bb - bmax * THR_FP, bb - bmax * THR_Q
    g.line(ex0 - 2, yfp, ex1 + 8, yfp, stroke=FP_T, sw=1, dash="4 3")
    g.line(ex0 - 2, yq, ex1 + 8, yq, stroke=Q_T, sw=1, dash="4 3")
    lx = ex1 + 14
    g.text(lx, yfp - 6, "top $k_{fp}$ → fp16", size=12, fill=FP_T, weight="bold")
    g.text(lx, (yfp + yq) / 2 + 4, "next $N_{q}$ → int2 + card", size=12, fill=Q_T, weight="bold")
    g.text(lx, yq + 16, "rest → dropped", size=12, fill=INK3, weight="bold")
    # local + new token
    lx0 = ex1 + 6
    g.rect(lx0, ay, 86, 22, fill="url(#hatch-blue)", stroke=FP_S, sw=0.8)
    g.text(lx0 + 43, ay + 15.5, "local", size=11.5, anchor="middle", fill=FP_T, weight="bold")
    g.rect(lx0 + 90, ay, 9, 22, fill=QRY_F, stroke=QRY_S, sw=1)
    by = ay + 41
    brace(g, ax, ax + sink_w, by, depth=5)
    g.text(ax + sink_w / 2, by + 18, "sink", size=11, anchor="middle", fill=SINK_S)
    brace(g, ex0, ex1, by, depth=6)
    g.text((ex0 + ex1) / 2, by + 18,
           "evictable windows (chronological id), ranked by $\\bar{S}_{w}$",
           size=11.5, anchor="middle", fill=INK2)
    brace(g, lx0, lx0 + 86, by, depth=5)
    g.text(lx0 + 43, by + 18, "128 tok, kept", size=11, anchor="middle", fill=FP_T)
    g.line(lx0 + 94.5, ay + 24, lx0 + 94.5, by + 22, stroke=QRY_S, sw=0.9)
    g.text(lx0 + 99, by + 34, "$k_{t}, v_{t}$ appended", size=11, anchor="end", fill=QRY_T)

    # --- the two stores, after compaction ---
    sy = 276
    fx, fw = 16, 262
    qx, qw = 360, 264
    fbh, qbh = 128, 206
    g.rect(fx, sy, fw, fbh, fill="#fafcfe", stroke=FP_S, sw=1.1, rx=6)
    g.text(fx + 10, sy + 20, "fp16 KV store", size=13, weight="bold", fill=FP_T)
    g.text(fx + 110, sy + 20, "post-RoPE · contiguous", size=11.5, fill=INK2)
    sx, sy2 = fx + 12, sy + 34
    for k in range(5):
        g.rect(sx + k * 4.4, sy2, 4.4, 24, fill=SINK_F, stroke=SINK_S, sw=0.5)
    x = sx + 26
    for i in FP_IDS:
        g.rect(x, sy2, 22, 24, fill=FP_F, stroke=FP_S, sw=0.8)
        g.text(x + 11, sy2 + 16, str(i), size=10.5, anchor="middle", fill=FP_T)
        x += 24
    g.rect(x + 2, sy2, 116, 24, fill="url(#hatch-blue)", stroke=FP_S, sw=0.8)
    g.text(x + 60, sy2 + 16, "local (16 windows)", size=11, anchor="middle", fill=FP_T)
    g.rect(x + 121, sy2, 8, 24, fill=QRY_F, stroke=QRY_S, sw=0.9)
    g.text(fx + 12, sy2 + 44, "[ sink ‖ fp windows by id ‖ local ‖ $k_{t}$ ]", size=12, fill=INK2)
    g.text(fx + 12, sy2 + 63, "compact first, then append", size=11.5, fill=INK3)
    g.text(fx + 12, sy2 + 80, "positions are never renumbered", size=11.5, fill=INK3)

    g.rect(qx, sy, qw, qbh, fill="#fffcf8", stroke=Q_S, sw=1.1, rx=6)
    g.text(qx + 10, sy + 20, "int2 slot table", size=13, weight="bold", fill=Q_T)
    g.text(qx + 112, sy + 20, "pre-RoPE · frozen positions", size=11.5, fill=INK2)
    slots = ([("a", i) for i in Q_IDS] + [("d", FP_IDS[0]), ("d", FP_IDS[2])]
             + [("f", None), ("f", None)])
    s0x, s0y, sw_, sh_ = qx + 14, sy + 34, 26, 44
    for k, (state, wid) in enumerate(slots):
        cx = s0x + (k % 8) * (sw_ + 4)
        cy = s0y + (k // 8) * (sh_ + 20)
        if state == "f":
            g.rect(cx, cy, sw_, sh_, fill="#ffffff", stroke=INK3, sw=0.8, dash="2 2")
            continue
        op = ' opacity="0.45"' if state == "d" else ""
        g.add(f"<g{op}>")
        yy = cy
        for col, lh in ((Q_F, 16), ("#fbe3c8", 8), (CARD_F, 12), ("#e6e6e6", 8)):
            g.rect(cx, yy, sw_, lh, fill=col, stroke="none")
            yy += lh
        g.rect(cx, cy, sw_, sh_, fill="none", stroke=Q_S if state == "a" else INK3, sw=0.8)
        g.add("</g>")
        if state == "d":
            g.rect(cx, cy, sw_, sh_, fill="url(#hatch-gray)", stroke="none")
        g.text(cx + sw_ / 2, cy + sh_ + 13, str(wid), size=10, anchor="middle",
               fill=Q_T if state == "a" else INK3)
    lgx, lgy = qx + 14, sy + qbh - 30
    for col, lab in ((Q_F, "K,V codes"), ("#fbe3c8", "grid"), (CARD_F, "card"),
                     ("#e6e6e6", "positions")):
        g.rect(lgx, lgy - 9, 10, 10, fill=col, stroke=INK3, sw=0.5)
        g.text(lgx + 14, lgy, lab, size=11, fill=INK2)
        lgx += 26 + len(lab) * 6.0
    g.text(qx + 14, lgy + 18, "faded: dormant (promoted, kept) · dashed: free", size=11,
           fill=INK3)

    mx0, mx1 = fx + fw, qx
    g.path(f"M{mx0 + 2},{sy + 40} C{mx0 + 30},{sy + 22} {mx1 - 30},{sy + 22} {mx1 - 2},{sy + 40}",
           stroke=Q_S, sw=1.6, arrow="orange")
    g.text((mx0 + mx1) / 2, sy + 16, "demote", size=12, anchor="middle", fill=Q_T, weight="bold")
    g.path(f"M{mx1 - 2},{sy + 92} C{mx1 - 30},{sy + 110} {mx0 + 30},{sy + 110} {mx0 + 2},{sy + 92}",
           stroke=FP_S, sw=1.6, arrow="blue")
    g.text((mx0 + mx1) / 2, sy + 124, "promote", size=12, anchor="middle", fill=FP_T, weight="bold")

    # what each move does
    ry = sy + fbh + 30
    rules = [
        (Q_T, "demote", "un-RoPE · quantize · build card → (c)"),
        (FP_T, "promote", "dequantize · re-RoPE at own positions"),
        (Q_T, "re-demote", "reactivate the dormant slot,"),
        (None, "", "no re-quantization"),
        (INK3, "drop", "free the slot and its card"),
    ]
    yy = ry
    for col, a, b in rules:
        if col:
            g.text(22, yy, a, size=11.5, fill=col, weight="bold")
        g.text(92, yy, b, size=11.5, fill=INK2)
        yy += 17
    g.text(16, h - 14, "bytes mode: $q$ = 0.70 of the evictable budget buys int2 windows at "
           "1076 B vs 4096 B per head", size=11.5, fill=INK2)


# ---------------------------------------------------------------------------
# (c) demotion: 2-bit window + rank-1 card
# ---------------------------------------------------------------------------
def panel_c(g):
    w, h = COLW[2], TOP_H
    frame(g, w, h, "(c)", "Demotion: 2-bit window + rank-1 card",
          "one window ($ws$ = 8 tokens) of one KV head, $d$ = 128 · written once, frozen for life")
    rng = random.Random(9)

    def grid(x, y, rows, cols, cs, hl_col=None, hl_row=None, hot_row=None):
        for r in range(rows):
            for c in range(cols):
                v = 0.35 + rng.uniform(-0.45, 0.45) * 0.6
                if c in (3, 11):
                    v += 0.35                   # a massive-activation channel
                if hot_row is not None and r == hot_row:
                    v += 0.25
                g.rect(x + c * cs, y + r * cs, cs, cs, fill=ramp(BLUES, v), stroke="#ffffff",
                       sw=0.4)
        g.rect(x, y, cols * cs, rows * cs, stroke=INK3, sw=0.7)
        if hl_col is not None:
            g.rect(x + hl_col * cs - 0.5, y - 0.5, cs + 1, rows * cs + 1, stroke=Q_S, sw=1.6)
        if hl_row is not None:
            g.rect(x - 0.5, y + hl_row * cs - 0.5, cols * cs + 1, cs + 1, stroke=Q_S, sw=1.6)

    # --- the 2-bit window --------------------------------------------------
    ix, iy, cs = 18, 88, 6.5
    gw, gh = 16 * cs, 8 * cs
    vy = iy + 82
    g.text(ix, iy - 8, "$K$ (post-RoPE)", size=12, fill=INK2)
    grid(ix, iy, 8, 16, cs, hot_row=5)
    g.text(ix, vy - 8, "$V$", size=12, fill=INK2)
    grid(ix, vy, 8, 16, cs)
    bx = ix + gw + 22
    g.line(ix + gw + 4, iy + gh / 2, bx - 3, iy + gh / 2, stroke=INK2, sw=1.2, arrow="dark")
    g.rect(bx, iy + gh / 2 - 11, 62, 22, fill="#f3f4f6", stroke=INK3, sw=0.8, rx=4)
    g.text(bx + 31, iy + gh / 2 + 4, "un-RoPE", size=11.5, anchor="middle", fill=INK2)
    kx = bx + 80
    g.line(bx + 62, iy + gh / 2, kx - 3, iy + gh / 2, stroke=INK2, sw=1.2, arrow="dark")
    grid(kx, iy, 8, 16, cs, hl_col=6)
    g.text(kx + 6.5 * cs, iy - 7, "channel $c$", size=11, anchor="middle", fill=Q_T)
    g.line(ix + gw + 4, vy + gh / 2, kx - 3, vy + gh / 2, stroke=INK2, sw=1.2, arrow="dark")
    g.text((ix + gw + kx) / 2, vy + gh / 2 - 6, "no RoPE on values", size=10.5, anchor="middle",
           fill=INK3)
    grid(kx, vy, 8, 16, cs, hl_row=2)
    g.text(kx + 6.5 * cs, vy - 7, "token $i$", size=11, anchor="middle", fill=Q_T)
    tx = kx + gw + 14
    g.text(tx, iy + 10, "keys: per channel", size=12, weight="bold", fill=Q_T)
    g.text(tx, iy + 27, "$(s_{c}, z_{c})$ over the 8 tokens", size=12, fill=INK2)
    g.text(tx, iy + 43, "channel-major, 4 tokens/byte", size=11.5, fill=INK3)
    g.text(tx, iy + 64, "$q = \\rm{round}((x − z)/s) ∈ \\{0,1,2,3\\}$", size=12.5)
    g.text(tx, iy + 81, "$x ≈ q·s + z$", size=12.5)
    g.text(tx, vy + 16, "values: per token", size=12, weight="bold", fill=Q_T)
    g.text(tx, vy + 33, "$(s_{i}, z_{i})$ over 128 channels", size=12, fill=INK2)
    g.text(tx, vy + 49, "token-major, 4 channels/byte", size=11.5, fill=INK3)
    g.text(ix, vy + gh + 22, "grid: scale and zero at 1 B each, plus one fp16 scale per group of "
           "≤ 32 entries", size=11.5, fill=INK2)

    # --- the card --------------------------------------------------------------
    cy0 = 266
    g.line(16, cy0 - 12, w - 16, cy0 - 12, stroke=RULE, sw=1, dash="3 3")
    g.text(18, cy0 + 6, "card, built from the post-RoPE keys (no extra rotation)", size=12,
           weight="bold", fill=CARD_T)
    px, py, pw, ph = 18, cy0 + 16, 176, 128
    g.rect(px, py, pw, ph, fill="#fbfdfb", stroke=RULE, sw=0.8, rx=4)
    mu = (px + 58, py + 86)
    vdir = (0.83, -0.56)
    pts = [(-16, 6), (-8, -10), (6, 12), (14, -4), (-20, -4), (2, -14), (10, 4)]
    hot = (84, -57)
    L0 = (mu[0] - vdir[0] * 46, mu[1] - vdir[1] * 46)
    L1 = (mu[0] + vdir[0] * 116, mu[1] + vdir[1] * 116)
    g.line(*L0, *L1, stroke=CARD_S, sw=0.9, dash="4 3")
    for dx, dy in pts + [hot]:
        x, y = mu[0] + dx, mu[1] + dy
        t = dx * vdir[0] + dy * vdir[1]
        fx, fy = mu[0] + t * vdir[0], mu[1] + t * vdir[1]
        g.line(x, y, fx, fy, stroke=INK3, sw=0.6)
        g.line(fx - 2.5 * vdir[1], fy + 2.5 * vdir[0], fx + 2.5 * vdir[1], fy - 2.5 * vdir[0],
               stroke=CARD_S, sw=1.2)
    for dx, dy in pts:
        g.circle(mu[0] + dx, mu[1] + dy, 3.4, fill=ramp(BLUES, 0.5), stroke="#ffffff", sw=0.6)
    hx_, hy_ = mu[0] + hot[0], mu[1] + hot[1]
    g.circle(hx_, hy_, 4.6, fill=QRY_S, stroke="#ffffff", sw=0.8)
    g.line(mu[0], mu[1], mu[0] + vdir[0] * 42, mu[1] + vdir[1] * 42, stroke=CARD_T, sw=2,
           arrow="green")
    g.path(f"M{mu[0] - 4},{mu[1] - 4} L{mu[0] + 4},{mu[1] + 4} M{mu[0] - 4},{mu[1] + 4} "
           f"L{mu[0] + 4},{mu[1] - 4}", stroke=INK, sw=1.8)
    g.text(mu[0] - 34, mu[1] + 10, "$μ$", size=14)
    g.text(mu[0] + 26, mu[1] - 30, "$v$", size=14, fill=CARD_T)
    g.text(hx_ - 8, hy_ - 7, "hot token $k_{i*}$", size=11, anchor="end", fill=QRY_T)
    g.text(px + pw - 6, py + ph - 8, "ticks: $t_{i}$", size=11, anchor="end", fill=CARD_T)

    ex = px + pw + 20
    eqs = ["$μ = \\rm{mean}_{i}\\, k_{i}$",
           "$i* = \\rm{argmax}_{i} ‖k_{i} − μ‖$",
           "$v = (k_{i*} − μ)/‖k_{i*} − μ‖$",
           "$t_{i} = (k_{i} − μ)·v$",
           "$k_{i} ≈ μ + t_{i}\\,v$",
           "$\\bar{v} = \\rm{mean}_{i}\\, v_{i}$"]
    notes = ["$a$: frozen per-head anchor", "farthest point = the hot token", "unit direction", None,
             "rank-1 key model", "value centroid"]
    for k, (e, n) in enumerate(zip(eqs, notes)):
        y = cy0 + 34 + k * 21
        g.text(ex, y, e, size=13.5)
        if n:
            g.text(ex + 176, y - 1, n, size=11, fill=CARD_T if "model" in n else INK3)

    # card fields
    ty = cy0 + 172
    cols = [ex, ex + 62, ex + 196, ex + 244]
    for c, t in zip(cols, ["field", "holds", "bits", "B / head"]):
        g.text(c, ty, t, size=11, fill=INK3, weight="bold")
    g.line(ex, ty + 5, w - 18, ty + 5, stroke=RULE, sw=0.8)
    rows = [("$μ − a$", "key mean − anchor", "4", "64 + 2"),
            ("$v$", "deviation direction", "8", "128 + 2"),
            ("$t$", "per-token projection", "8", "8 + 2"),
            ("$\\bar{v} − a_{v}$", "value centroid", "4", "64 + 2")]
    for k, r in enumerate(rows):
        yy = ty + 21 + k * 17
        for c, t in zip(cols, r):
            g.text(c, yy, t, size=13 if c == cols[0] else 12, fill=INK)
    yy = ty + 21 + 4 * 17
    g.line(ex, yy - 11, w - 18, yy - 11, stroke=RULE, sw=0.8)
    g.text(cols[1], yy + 3, "card (+2 = fp16 scale per field)", size=11.5, fill=INK2)
    g.text(cols[3], yy + 3, "272", size=12.5, weight="bold", fill=CARD_T)

    # bytes per head per window
    by = cy0 + 172
    g.text(18, by, "bytes / head / window", size=11.5, fill=INK2, weight="bold")
    sc = 0.042
    g.rect(18, by + 8, 4096 * sc, 11, fill=FP_F, stroke=FP_S, sw=0.8)
    g.text(18, by + 33, "fp16 K + V: 4096", size=11.5, fill=FP_T)
    xx = 18
    for val, col, st in ((512, Q_F, Q_S), (292, "#fbe3c8", Q_S), (272, CARD_F, CARD_S)):
        g.rect(xx, by + 42, val * sc, 11, fill=col, stroke=st, sw=0.8)
        xx += val * sc
    g.text(xx + 6, by + 52, "1076", size=12, weight="bold", fill=Q_T)
    g.text(18, by + 68, "codes 512 + grid 292 + card 272", size=11, fill=Q_T)
    g.text(18, by + 84, "3.81× cheaper than fp16", size=11.5, weight="bold", fill=Q_T)


# ---------------------------------------------------------------------------
# (d) the read gate
# ---------------------------------------------------------------------------
def panel_d(g):
    w, h = COLW[2], BOT_H
    frame(g, w, h, "(d)", "Read gate: score every card, open the top 25%",
          "per KV head: its 4 query heads choose together · 2 launches per layer")

    # query
    qx, qy = 18, 70
    g.text(qx, qy + 6, "query $q_{t}$", size=12.5, weight="bold", fill=QRY_T)
    for k in range(4):
        g.rect(qx, qy + 14 + k * 12, 64, 9, fill=QRY_F, stroke=QRY_S, sw=0.8, rx=2)
        g.text(qx + 70, qy + 22 + k * 12, f"$h={k + 1}$", size=10.5, fill=QRY_T)
    g.text(qx, qy + 76, "post-RoPE, 4 query", size=10.5, fill=INK3)
    g.text(qx, qy + 88, "heads share the KV head", size=10.5, fill=INK3)

    # step 1: card score
    sx = 150
    num(g, sx, qy + 2, 1, fill=CARD_S)
    g.text(sx + 14, qy + 7, "card score: two $d$-length dots per head", size=12.5, weight="bold",
           fill=CARD_T)
    # card chip
    cx, cy = sx - 4, qy + 22
    parts = [("$μ$", 38), ("$v$", 38), ("$t$", 18), ("$\\bar{v}$", 38)]
    xx = cx
    for lab, ww in parts:
        op = 0.4 if lab == "$\\bar{v}$" else 1
        g.add(f'<g opacity="{op}">')
        g.rect(xx, cy, ww, 18, fill=CARD_F, stroke=CARD_S, sw=0.8)
        g.text(xx + ww / 2, cy + 13.5, lab, size=12, anchor="middle", fill=CARD_T)
        g.add("</g>")
        xx += ww
    g.text(cx, cy + 32, "$\\bar{v}$ not read here", size=10.5, fill=INK3)
    ex = cx + 148
    eqs = ["$m_{h} = q_{h}·(a + μ_{w})$,  $g_{h} = q_{h}·v_{w}$",
           "$x_{h,i} = (m_{h} + t_{i}\\,g_{h})/√d$",
           "$ℓ_{h}(w) = \\rm{LSE}_{i}\\, x_{h,i}$"]
    for k, e in enumerate(eqs):
        g.text(ex, cy + 12 + k * 21, e, size=13)

    # step 2: per-head share matrix
    my = 198
    num(g, 26, my, 2, fill=Q_S)
    g.text(40, my + 5, "per-head share", size=12.5, weight="bold", fill=Q_T)
    g.text(160, my + 5, "$σ_{h}(w) = ℓ_{h}(w) − \\rm{LSE}_{w′}\\, ℓ_{h}(w′)$", size=13)
    mx, cw_, ch_ = 70, 30, 20
    gy = my + 22
    for c in range(N_Q):
        g.text(mx + c * cw_ + cw_ / 2, gy - 4, str(Q_IDS[c]), size=10, anchor="middle", fill=INK3)
    for hh in range(4):
        g.text(mx - 8, gy + hh * ch_ + 14, f"$h={hh + 1}$", size=11, anchor="end", fill=INK2)
        for c in range(N_Q):
            p = math.exp(SHARE[hh][c])
            g.rect(mx + c * cw_, gy + hh * ch_, cw_, ch_, fill=ramp(ORANGES, p ** 0.55 * 1.05),
                   stroke="#ffffff", sw=1)
    g.rect(mx, gy, N_Q * cw_, 4 * ch_, stroke=INK3, sw=0.7)
    g.text(mx + N_Q * cw_ + 8, gy + 3 * ch_ + 14, "retrieval head", size=10.5, fill=INK3)

    # step 3: union
    uy = gy + 4 * ch_ + 36
    num(g, 26, uy, 3, fill=Q_S)
    g.text(40, uy + 5, "GQA union", size=12.5, weight="bold", fill=Q_T)
    g.text(128, uy + 5, "$r(w) = \\rm{max}_{h}\\, σ_{h}(w)$", size=13)
    g.text(290, uy + 5, "every head keeps its own top windows", size=11, fill=INK3)
    ux, ury = mx, uy + 16
    g.text(mx - 8, ury + 14, "$r$", size=12, anchor="end", fill=INK2)
    for c in range(N_Q):
        p = math.exp(UNION[c])
        g.rect(ux + c * cw_, ury, cw_, ch_, fill=ramp(ORANGES, p ** 0.55 * 1.05),
               stroke="#ffffff", sw=1)
    g.rect(ux, ury, N_Q * cw_, ch_, stroke=INK3, sw=0.7)
    for c in SEL:
        g.rect(ux + c * cw_ - 1, ury - 1, cw_ + 2, ch_ + 2, stroke=INK, sw=2)

    # step 4: top-n_sel
    ty = uy + 70
    num(g, 26, ty, 4, fill=INK)
    g.text(40, ty + 5, "top $n_{sel} = ⌈0.25\\,N_{q}⌉$ per KV head →", size=12.5, weight="bold")
    xx = 300
    g.text(xx, ty + 5, "SEL =", size=12.5, weight="bold")
    xx += 44
    for c in SEL:
        ww = chip(g, xx, ty - 9, str(Q_IDS[c]), Q_F, Q_S, Q_T, size=12, w=28)
        xx += ww + 4
    g.text(xx + 4, ty + 5, "(ascending)", size=11, fill=INK3)

    # callouts
    g.text(40, ty + 26, "to (e): SEL, and $ℓ_{h}(w)$ of every window for the skipped ones' credit",
           size=11.5, fill=INK2)
    cy = ty + 50
    g.rect(16, cy, w - 32, 94, fill="#f8f9fb", stroke=RULE, sw=0.9, rx=5)
    g.text(28, cy + 19, "Shares, never raw logits.", size=12, weight="bold")
    g.text(28, cy + 36, "A raw $ℓ_{h}$ carries its head's baseline $q_{h}·a$ and scale $|q_{h}|$, so the "
           "loudest", size=11.5, fill=INK2)
    raw = ", ".join(str(Q_IDS[c]) for c in RAWMAX_SEL)
    needle = Q_IDS[max(range(N_Q), key=lambda c: SHARE[3][c])]
    g.text(28, cy + 52, f"head picks for all four: a raw max reads {{{raw}}} and misses "
           f"the needle, {needle}.", size=11.5, fill=INK2)
    g.text(28, cy + 74, "Cost: card + ratio × window; break-even ratio $= 1 − 272/804 = 0.66$.",
           size=11.5, fill=INK2)


# ---------------------------------------------------------------------------
# (e) the fused kernel
# ---------------------------------------------------------------------------
def panel_e(g):
    w, h = COLW[1], BOT_H
    frame(g, w, h, "(e)", "Fused two-tier decode kernel",
          "one program per (batch, KV head) serves all 4 query heads: each K/V tile is loaded once")

    ay = 150                         # program track
    th = 30
    x = 18
    wpx = 15.2                       # one window
    # sink prologue
    sink_x = x
    for k in range(5):
        g.rect(x + k * 4.2, ay, 4.2, th, fill=SINK_F, stroke=SINK_S, sw=0.5)
    x += 21 + 8
    tiles = []
    body = [("fp", i) for i in FP_IDS] + [("loc", None)] * 16 + [("new", None)]
    fp_tiles = [body[0:8], body[8:16], body[16:21]]
    for tl in fp_tiles:
        tx0 = x
        for k in range(8):
            if k < len(tl):
                kind, wid = tl[k]
                if kind == "fp":
                    g.rect(x, ay, wpx, th, fill=FP_F, stroke=FP_S, sw=0.6)
                    g.text(x + wpx / 2, ay + 19, str(wid), size=9.5, anchor="middle", fill=FP_T)
                elif kind == "loc":
                    g.rect(x, ay, wpx, th, fill="url(#hatch-blue)", stroke=FP_S, sw=0.6)
                else:
                    g.rect(x, ay, wpx / 8, th, fill=QRY_F, stroke=QRY_S, sw=0.6)
                    g.rect(x + wpx / 8, ay, wpx * 7 / 8, th, fill="#ffffff", stroke="#d0d5da",
                           sw=0.6, dash="2 2")
            else:
                g.rect(x, ay, wpx, th, fill="#ffffff", stroke="#d0d5da", sw=0.6, dash="2 2")
            x += wpx
        g.rect(tx0, ay, 8 * wpx, th, stroke=INK, sw=1.5)
        tiles.append((tx0, x))
        x += 7
    # Q tile
    qt0 = x + 6
    x = qt0
    for k in range(8):
        if k < N_SEL:
            g.rect(x, ay, wpx, th, fill=Q_F, stroke=Q_S, sw=0.7)
            g.text(x + wpx / 2, ay + 19, str(Q_IDS[SEL[k]]), size=9.5, anchor="middle", fill=Q_T)
        else:
            g.rect(x, ay, wpx, th, fill="#ffffff", stroke="#d0d5da", sw=0.6, dash="2 2")
        x += wpx
    g.rect(qt0, ay, 8 * wpx, th, stroke=INK, sw=1.5)
    qt1 = x

    # labels above the track
    ly = ay - 12
    num(g, sink_x + 10, ly - 8, 1, fill=SINK_S)
    g.text(sink_x + 22, ly - 4, "sink", size=12, weight="bold", fill=SINK_S)
    num(g, (tiles[0][0] + tiles[-1][1]) / 2 - 90, ly - 8, 2, fill=FP_S)
    g.text((tiles[0][0] + tiles[-1][1]) / 2 - 78, ly - 4, "fp16 body: whole-window tiles", size=12,
           weight="bold", fill=FP_T)
    brace(g, tiles[0][0], tiles[-1][1], ay - 4, down=False, depth=6)
    # int2 tier above the Q tile, with the SEL indirection
    iy = 72
    ix0 = qt1 - N_Q * 19.5 + 1
    num(g, ix0 - 14, iy - 6, 3, fill=Q_S)
    g.text(ix0, iy - 2, "int2 tier: visit SEL only", size=12, weight="bold", fill=Q_T)
    for c in range(N_Q):
        xx = ix0 + c * 19.5
        if c in SEL:
            g.rect(xx, iy + 6, 17, 20, fill=Q_F, stroke=Q_S, sw=0.9)
        else:
            g.rect(xx, iy + 6, 17, 20, fill=Q_L, stroke="#e2b48c", sw=0.7, dash="2 2")
        g.text(xx + 8.5, iy + 20, str(Q_IDS[c]), size=9.5, anchor="middle",
               fill=Q_T if c in SEL else "#c79b76")
    for k, c in enumerate(SEL):
        xs = ix0 + c * 19.5 + 8.5
        xd = qt0 + k * wpx + wpx / 2
        g.path(f"M{xs},{iy + 27} C{xs},{iy + 50} {xd},{ay - 34} {xd},{ay - 3}", stroke=Q_S, sw=1.1,
               arrow="orange")
    g.text(ix0 - 10, iy + 20, "skipped windows: card only", size=10.5, anchor="end", fill="#c79b76")
    g.text(ix0 - 10, iy + 42, "slot $j$ reads window SEL[$j$]", size=11.5, anchor="end", fill=Q_T)

    # under-track annotations
    uy = ay + th + 16
    g.text(sink_x, uy + 12, "softmax", size=10.5, fill=INK3)
    g.text(sink_x, uy + 25, "only", size=10.5, fill=INK3)
    brace(g, tiles[0][0], tiles[0][1], ay + th + 4, depth=5)
    g.text(tiles[0][0] + 20, uy + 12,
           "8 windows × 8 = 64 keys", size=10.5, fill=INK2)
    g.text((tiles[1][0] + tiles[1][1]) / 2 + 22, uy + 12,
           "no window straddles a tile", size=10.5, anchor="middle", fill=INK2)
    g.text((tiles[2][0] + tiles[2][1]) / 2 + 4, uy + 12, "masked lanes", size=10.5,
           anchor="middle", fill=INK3)
    brace(g, qt0, qt1, ay + th + 4, depth=5)
    g.text((qt0 + qt1) / 2, uy + 12, "one tile of selected windows", size=10.5, anchor="middle",
           fill=Q_T)

    # register pipeline for an int2 lane
    py = 262
    g.text(18, py - 8, "per int2 lane, entirely in registers (no fp16 copy of the tier in HBM)",
           size=12, fill=INK2, weight="bold")
    stages = [("u8 codes", "4 per byte", Q_L, Q_S),
              ("(b ≫ 2i) & 3", "2-bit code", Q_L, Q_S),
              ("· s + z", "1-byte grid", Q_L, Q_S),
              ("RoPE", "cos, sin @ pos", "#eef3f9", FP_S),
              ("tl.dot(q, k)", "logits $x_{j}$", "#f3f4f6", INK3)]
    bx = 18
    bw, bh = 108, 38
    for k, (a, b, f, s) in enumerate(stages):
        g.rect(bx, py, bw, bh, fill=f, stroke=s, sw=0.9, rx=4)
        g.text(bx + bw / 2, py + 16, a, size=12, anchor="middle", weight="bold", fill=INK)
        g.text(bx + bw / 2, py + 31, b, size=10.5, anchor="middle", fill=INK2)
        if k < len(stages) - 1:
            g.line(bx + bw + 1, py + bh / 2, bx + bw + 14, py + bh / 2, stroke=INK2, sw=1.1,
                   arrow="dark")
        bx += bw + 15

    # online softmax
    oy = 342
    g.rect(16, oy, 300, 150, fill="#f8f9fb", stroke=RULE, sw=0.9, rx=5)
    g.text(28, oy + 20, "online softmax, per query head", size=12, weight="bold")
    g.text(28, oy + 36, "running $m$, $ℓ$, $\\rm{acc} ∈ ℝ^{d}$; base 2, $\\log_{2}e$ in the scale",
           size=11, fill=INK3)
    eqs = ["$m′ = \\rm{max}(m, \\rm{max}_{j} x_{j})$",
           "$ℓ ← ℓ·2^{m − m′} + Σ_{j} p_{j}$",
           "$\\rm{acc} ← \\rm{acc}·2^{m − m′} + p\\,V$"]
    for k, e in enumerate(eqs):
        g.text(34, oy + 62 + k * 22, e, size=13)
    g.text(28, oy + 136, "same $q·k^{⊤}$ gives the attention and the scores", size=11, fill=INK3)

    # per-window emission + epilogue
    ex = 332
    g.rect(ex, oy, w - ex - 16, 150, fill="#fffaf5", stroke=RULE, sw=0.9, rx=5)
    num(g, ex + 16, oy + 16, 4, fill=INK)
    g.text(ex + 30, oy + 21, "per-window mass, then epilogue", size=12, weight="bold")
    eqs = ["$\\rm{WSUM}_{w} = Σ_{j ∈ w} p_{j}$,  $\\rm{WMAX}_{w} = m′$",
           "$\\rm{lse} = m + \\log_{2} ℓ$",
           "$a_{w} = \\rm{WSUM}_{w} · 2^{\\rm{WMAX}_{w} − \\rm{lse}}$"]
    for k, e in enumerate(eqs):
        g.text(ex + 14, oy + 48 + k * 24, e, size=13)
    g.text(ex + 14, oy + 116, "exact mass of every window read", size=11, fill=INK3)
    g.text(ex + 14, oy + 132, "over $W$ values, not over $S$ keys", size=11, fill=INK3)

    # launch budget
    g.text(18, oy + 176, "per layer per step: the gate's 2 launches, this kernel, the score gather, "
           "the two K/V writes", size=11.5, fill=INK2)
    g.text(18, oy + 193, "⑤ the skipped windows are credited in the same epilogue → (f)",
           size=11.5, fill=INK2)


# ---------------------------------------------------------------------------
# (f) credit the skipped windows
# ---------------------------------------------------------------------------
def panel_f(g):
    w, h = COLW[0], BOT_H
    frame(g, w, h, "(f)", "Credit skipped windows, renormalise",
          "kernel epilogue ⑤ · every term is an exact no-op at gate ratio 1.0")

    # bars in log domain
    bx0, by0, bw, gap = 56, 262, 28, 4
    ylo, yhi = -7.0, 0.0
    ph = 150
    Y = lambda v: by0 - ph * (v - ylo) / (yhi - ylo)
    # use head 4-like numbers: estimate for all, exact for selected
    est = [LOGM[3][c] - _lse(LOGM[3]) - 0.4 for c in range(N_Q)]
    rng = random.Random(4)
    dev = [rng.gauss(0.55, 0.35) for _ in range(N_Q)]
    exact = {c: min(est[c] + dev[c], -0.2) for c in SEL}
    delta = sum(exact[c] - est[c] for c in SEL) / len(SEL)
    g.text(bx0 - 8, Y(yhi) - 14, "$\\ln a_{w}$", size=13)
    for v in (-6, -4, -2, 0):
        g.line(bx0 - 4, Y(v), bx0 + N_Q * (bw + gap), Y(v), stroke="#eceff2", sw=0.8)
        g.text(bx0 - 8, Y(v) + 4, str(v), size=10, anchor="end", fill=INK3)
    for c in range(N_Q):
        x = bx0 + c * (bw + gap)
        if c in SEL:
            g.rect(x, Y(exact[c]), bw, by0 - Y(exact[c]), fill=Q_F, stroke=Q_S, sw=0.9)
            g.circle(x + bw / 2, Y(est[c]), 3.6, fill="#ffffff", stroke=CARD_S, sw=1.4)
        else:
            top = est[c] + delta
            g.rect(x, Y(top), bw, by0 - Y(top), fill="url(#hatch-orange)", stroke="#d9a57a",
                   sw=0.9, dash="3 2")
            g.circle(x + bw / 2, Y(est[c]), 3.6, fill="#ffffff", stroke=CARD_S, sw=1.4)
            g.line(x + bw / 2, Y(est[c]) - 4, x + bw / 2, Y(top) + 3, stroke=CARD_T, sw=1,
                   arrow="green")
        g.text(x + bw / 2, by0 + 14, str(Q_IDS[c]), size=10, anchor="middle",
               fill=Q_T if c in SEL else "#c79b76")
    g.line(bx0 - 4, by0, bx0 + N_Q * (bw + gap), by0, stroke=INK3, sw=0.8)

    # legend
    ly = 76
    g.rect(bx0, ly - 10, 14, 11, fill=Q_F, stroke=Q_S, sw=0.8)
    g.text(bx0 + 20, ly, "read: exact $a_{w}$", size=11.5, fill=INK2)
    g.circle(bx0 + 150, ly - 4, 3.6, fill="#ffffff", stroke=CARD_S, sw=1.4)
    g.text(bx0 + 158, ly, "card estimate $e^{ℓ_{w}}$", size=11.5, fill=INK2)
    g.rect(bx0 + 296, ly - 10, 14, 11, fill="url(#hatch-orange)", stroke="#d9a57a", sw=0.8, dash="3 2")
    g.text(bx0 + 316, ly, "skipped: credited", size=11.5, fill=INK2)

    # equations
    ey = by0 + 42
    g.text(18, ey, "deviation, measured on the windows that were read", size=12, weight="bold")
    g.text(30, ey + 24, "$δ = \\rm{mean}_{w ∈ \\rm{SEL}} (\\ln a_{w} − ℓ_{w})$", size=13.5)
    g.text(270, ey + 24, "mean log ratio", size=11, fill=INK3)
    g.text(270, ey + 37, "(one needle cannot inflate it)", size=11, fill=INK3)
    g.text(18, ey + 66, "every skipped window is credited, to the scores and the output", size=12,
           weight="bold")
    g.text(30, ey + 90, "$\\hat{a}_{w} = \\rm{exp}(ℓ_{w} + δ)$, $w ∉ \\rm{SEL}$".replace("\\hat{a}", "â"),
           size=13.5)
    g.text(30, ey + 118, "$o = (o_{\\rm{read}} + Σ_{w ∉ \\rm{SEL}} â_{w} \\bar{v}_{w}) / (1 + Σ_{w ∉ \\rm{SEL}} â_{w})$",
           size=13.5)
    g.text(30, ey + 144, "$a_{w} ← a_{w} / (1 + Σ â)$ for every window", size=13.5)
    g.text(18, ey + 170, "a skipped window scored 0 would rank last and be evicted:", size=11.5,
           fill=INK2)
    g.text(18, ey + 186, "the gate would destroy the tier it reads less often.", size=11.5,
           fill=INK2)
    # outputs
    oy = h - 24
    g.rect(18, oy - 14, 118, 20, fill="#f3f4f6", stroke=INK3, sw=0.8, rx=4)
    g.text(77, oy + 0.5, "$o_{t}$ → o_proj", size=12, anchor="middle")
    g.text(146, oy, "attention output, on to the next layer", size=11.5, fill=INK3)


# ---------------------------------------------------------------------------
# legend, lanes, cross-panel arrows
# ---------------------------------------------------------------------------
def legend(g):
    x, y = X0, 22
    items = [
        ("sink", SINK_F, SINK_S, None),
        ("fp16 window (top $k_{fp}$)", FP_F, FP_S, None),
        ("local window (fp16)", "url(#hatch-blue)", FP_S, None),
        ("int2 window, read", Q_F, Q_S, None),
        ("int2 window, skipped", "url(#hatch-orange)", "#d9a57a", "3 2"),
        ("card", CARD_F, CARD_S, None),
        ("dropped", DROP_F, DROP_S, None),
        ("query / new token", QRY_F, QRY_S, None),
    ]
    for lab, f, s, d in items:
        g.rect(x, y - 11, 16, 13, fill=f, stroke=s, sw=0.9, dash=d)
        g.text(x + 22, y, lab, size=12.5, fill=INK2)
        x += 34 + len(re.sub(r"\$|[{}_]|\\rm", "", lab)) * 6.9
    g.text(W - 14, y, "Llama-3.1-8B: $H_{q}$ = 32, $H_{kv}$ = 8, $d$ = 128 · $ws$ = 8, sink 5, local 128 · "
           "budget 0.20, $q$ = 0.70, gate ratio 0.25",
           size=12.5, anchor="end", fill=INK2)


def lanes(g):
    for y, hgt, lab in ((TOP_Y, TOP_H, "CACHE · scored every step, re-tiered every ws steps"),
                        (BOT_Y, BOT_H, "DECODE STEP · every token, every layer")):
        g.rect(14, y, 30, hgt, fill="#f1f3f6", stroke="none", rx=6)
        g.text(33, y + hgt / 2, lab, size=12.5, anchor="middle", fill=INK2, weight="bold",
               rotate=-90, extra=' letter-spacing="1.2"')


def flow_label(g, x, y, s, anchor="middle", size=12, fill=INK):
    g.text(x, y, s, size=size, anchor=anchor, fill=fill)


def arrows(g):
    mid0, mid1 = TOP_Y + TOP_H, BOT_Y
    # (a) -> (b)
    y = TOP_Y + 150
    g.line(COLX[0] + COLW[0] + 2, y, COLX[1] - 3, y, stroke=INK, sw=2, arrow="dark")
    # (b) -> (c)
    y = TOP_Y + 358
    g.line(COLX[1] + COLW[1] + 2, y, COLX[2] - 3, y, stroke=Q_S, sw=2, arrow="orange")
    # (c) -> (d)
    x = COLX[2] + 300
    g.line(x, mid0 + 2, x, mid1 - 3, stroke=CARD_S, sw=2, arrow="green")
    flow_label(g, x + 10, mid0 + 32, "cards: $μ, v, t, \\bar{v}$ (272 B / head)", anchor="start",
               fill=CARD_T)
    # (b) -> (e)
    x = COLX[1] + 320
    g.line(x, mid0 + 2, x, mid1 - 3, stroke=INK, sw=2, arrow="dark")
    flow_label(g, x + 10, mid0 + 32, "fp16 store + int2 codes, read in place", anchor="start",
               fill=INK2)
    # (d) -> (e): SEL leaves step 4 and enters the int2 tier row
    xg = COLX[2] - GAP / 2
    y0, y1 = BOT_Y + 406, BOT_Y + 88
    g.path(f"M{COLX[2] - 2},{y0} H{xg + 4} Q{xg},{y0} {xg},{y0 - 4} V{y1 + 4} "
           f"Q{xg},{y1} {xg - 4},{y1} H{COLX[1] + COLW[1] + 3}",
           stroke=Q_S, sw=2, arrow="orange")
    # (e) -> (f)
    y = BOT_Y + 400
    g.line(COLX[1] - 2, y, COLX[0] + COLW[0] + 3, y, stroke=Q_S, sw=2, arrow="orange")
    # (f) -> (a)
    x = COLX[0] + 250
    g.line(x, mid1 - 2, x, mid0 + 3, stroke=FP_S, sw=2, arrow="blue")
    flow_label(g, x - 10, mid0 + 32, "$a^{(t)}_{h,w}$ for every window", anchor="end", fill=FP_T)


def defs():
    def marker(name, col):
        return (f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8.6" refY="5" '
                f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                f'<path d="M0,0.6 L10,5 L0,9.4 z" fill="{col}"/></marker>')

    return ("<defs>"
            + marker("dark", INK) + marker("gray", INK3) + marker("blue", FP_S)
            + marker("orange", Q_S) + marker("green", CARD_T) + marker("red", QRY_S)
            + "</defs>")


def build():
    g = G()
    g.add(defs())
    g.rect(0, 0, W, H, fill="#ffffff")
    legend(g)
    lanes(g)
    for fn, x, y in ((panel_a, COLX[0], TOP_Y), (panel_b, COLX[1], TOP_Y),
                     (panel_c, COLX[2], TOP_Y), (panel_f, COLX[0], BOT_Y),
                     (panel_e, COLX[1], BOT_Y), (panel_d, COLX[2], BOT_Y)):
        g.open(x, y, extra=f' class="panel" data-w="{COLW[COLX.index(x)]}" '
                           f'data-h="{TOP_H if y == TOP_Y else BOT_H}"')
        fn(g)
        g.close()
    arrows(g)
    body = "\n".join(g.p)
    svg = (f'<svg id="figure" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{SANS}">\n'
           f"<style>text{{font-kerning:normal;}} .t{{dominant-baseline:auto;}}</style>\n"
           f"{body}\n</svg>")
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
    c.setAttribute('width', '{W}'); c.setAttribute('height', '{H}');
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
    print(f"wrote {OUT}  ({len(page) / 1024:.0f} KB)  SEL={[Q_IDS[c] for c in SEL]} "
          f"rawmax={[Q_IDS[c] for c in RAWMAX_SEL]} fp={FP_IDS} q={Q_IDS}")


if __name__ == "__main__":
    build()
