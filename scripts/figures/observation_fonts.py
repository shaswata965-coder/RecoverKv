"""Embed the figure fonts in ``reports/figures/observation_panels.html``.

The observation figures set their text in Liberation Serif / Liberation Sans
(metric-compatible with Times and Helvetica, SIL OFL). Both the on-screen
preview and the PDF export read the same base64 TTFs from the page's last
``<script>`` line, so text widths measured on screen are the widths jsPDF
embeds, and every exported PDF carries its fonts.

Each face is subset to Latin-1 plus the punctuation, arrows, Greek and maths
symbols a figure label is likely to need, with hinting and layout features
dropped (jsPDF applies no kerning, so the browser must not either).

Run after editing the page if the fonts line was lost:

    python scripts/figures/observation_fonts.py [--font-dir /usr/share/fonts/truetype/liberation]

Needs fontTools (``pip install fonttools``).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

PAGE = Path(__file__).resolve().parents[2] / "reports" / "figures" / "observation_panels.html"
UNICODES = (
    "U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+0391-03A9,U+03B1-03C9,U+2009,U+2013-2014,"
    "U+2018-201D,U+2022,U+2026,U+2030,U+2032-2033,U+2190-2195,U+2211,U+2212,U+221A,U+221E,"
    "U+2248,U+2260,U+2264-2265"
)
FACES = {
    "serif": {"normal": "LiberationSerif-Regular", "italic": "LiberationSerif-Italic", "bold": "LiberationSerif-Bold"},
    "sans": {"normal": "LiberationSans-Regular", "italic": "LiberationSans-Italic", "bold": "LiberationSans-Bold"},
}
LINE = re.compile(r"<script>window\.FIG_FONTS=.*?;(?:/\*FIG_FONTS\*/)?</script>", re.S)


def subset_b64(path: Path) -> str:
    opts = subset.Options()
    opts.hinting = False
    opts.layout_features = []
    opts.name_IDs = ["*"]
    font = TTFont(str(path))
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=subset.parse_unicodes(UNICODES))
    sub.subset(font)
    buf = io.BytesIO()
    font.save(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--font-dir", default="/usr/share/fonts/truetype/liberation")
    args = ap.parse_args()
    fonts = {fam: {st: subset_b64(Path(args.font_dir) / f"{name}.ttf") for st, name in styles.items()}
             for fam, styles in FACES.items()}
    html = PAGE.read_text(encoding="utf-8")
    line = f"<script>window.FIG_FONTS={json.dumps(fonts, separators=(',', ':'))};</script>"
    new, n = LINE.subn(lambda _m: line, html)
    if n != 1:
        raise SystemExit(f"expected one FIG_FONTS script line in {PAGE}, found {n}")
    PAGE.write_text(new, encoding="utf-8")
    kb = sum(len(v) for fam in fonts.values() for v in fam.values()) / 1024
    print(f"embedded 6 faces ({kb:.0f} KB base64) in {PAGE}")


if __name__ == "__main__":
    main()
