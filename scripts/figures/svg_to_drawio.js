// Convert the rendered method figure (one inline SVG) into a diagrams.net / draw.io
// file: an <mxfile> whose cells are native, editable shapes.
//
// Runs in the browser, on the live SVG, so every position is the one Chromium
// actually laid out -- text boxes come from getBBox(), not from a font-width guess.
//
//   rect             -> rectangle vertex (rounded where rx > 0)
//   circle           -> ellipse vertex
//   text             -> text vertex (rotation kept)
//   line, open path  -> edge; curves are sampled into waypoints, markers -> arrows
//   hatching, filled polygons, the card <symbol>
//                    -> image vertex holding that one piece as SVG (draw.io has no
//                       native hatch fill or free polygon; the piece still moves,
//                       resizes and deletes as one object)
//
// Each panel (<g class="panel">) becomes a draw.io group, so a panel moves as one.
// The figure is drawn in points (1 unit = 1 pt); draw.io works in 96-dpi pixels,
// so every length and font size is scaled by 96/72 and an 8 pt label stays 8 pt.
// Loaded by export_figure.mjs; defines svgToDrawio(svgElement, name) -> XML string.

function svgToDrawio(svg, name = 'figure') {
  const vb = svg.viewBox.baseVal;
  const W = vb.width, H = vb.height;
  const S = 96 / 72;
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const b64 = (s) => btoa(unescape(encodeURIComponent(s)));
  const r2 = (v) => Math.round(v * 100) / 100;
  const cells = [];
  let nextId = 2;
  const newId = () => 'c' + (nextId++);

  const attr = (el, name, dflt) => {
    // Presentation attributes are inherited: walk up to the SVG root.
    for (let e = el; e && e !== svg.parentNode; e = e.parentNode) {
      if (e.getAttribute && e.getAttribute(name) != null) return e.getAttribute(name);
    }
    return dflt;
  };
  const opacityOf = (el) => {
    let o = 1;
    for (let e = el; e && e !== svg; e = e.parentNode) {
      const a = e.getAttribute && e.getAttribute('opacity');
      if (a != null) o *= +a;
    }
    return o;
  };
  // element user space -> the figure's own units (viewBox), whatever CSS size it has
  const root = svg.getScreenCTM().inverse();
  const ctm = (el) => root.multiply(el.getScreenCTM());
  const apply = (m, x, y) => ({ x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f });

  // ---- groups: one per panel ------------------------------------------------
  const groups = new Map();
  for (const p of svg.querySelectorAll('g.panel')) {
    const m = ctm(p);
    const g = { id: newId(), x: m.e, y: m.f, w: +p.dataset.w, h: +p.dataset.h };
    groups.set(p, g);
    cells.push(`<mxCell id="${g.id}" value="" style="group;fillColor=none;strokeColor=none;" vertex="1" connectable="0" parent="1">` +
      `<mxGeometry x="${r2(S * g.x)}" y="${r2(S * g.y)}" width="${r2(S * g.w)}" height="${r2(S * g.h)}" as="geometry"/></mxCell>`);
  }
  const parentOf = (el) => {
    const p = el.closest('g.panel');
    return p ? groups.get(p) : { id: '1', x: 0, y: 0 };
  };

  // ---- style helpers ---------------------------------------------------------
  const strokeStyle = (el) => {
    const stroke = attr(el, 'stroke', 'none');
    const sw = +attr(el, 'stroke-width', 1);
    const dash = el.getAttribute('stroke-dasharray');
    let s = `strokeColor=${stroke === 'none' ? 'none' : stroke};strokeWidth=${r2(S * sw)};`;
    if (dash) s += `dashed=1;fixDash=1;dashPattern=${dash.split(/[\s,]+/).map((v) => r2(S * v)).join(' ')};`;
    return s;
  };
  const opacityStyle = (el) => {
    const o = opacityOf(el);
    return o < 0.999 ? `opacity=${Math.round(o * 100)};` : '';
  };
  const vertex = (el, x, y, w, h, style, value = '') => {
    const par = parentOf(el);
    cells.push(`<mxCell id="${newId()}" value="${esc(value)}" style="${style}" vertex="1" ` +
      `parent="${par.id}"><mxGeometry x="${r2(S * (x - par.x))}" y="${r2(S * (y - par.y))}" ` +
      `width="${r2(S * w)}" height="${r2(S * h)}" as="geometry"/></mxCell>`);
  };
  const edge = (el, pts, style) => {
    const par = parentOf(el);
    const P = pts.map((p) => ({ x: r2(S * (p.x - par.x)), y: r2(S * (p.y - par.y)) }));
    const mid = P.slice(1, -1).map((p) => `<mxPoint x="${p.x}" y="${p.y}"/>`).join('');
    cells.push(`<mxCell id="${newId()}" value="" style="${style}" edge="1" parent="${par.id}">` +
      `<mxGeometry relative="1" as="geometry">` +
      `<mxPoint x="${P[0].x}" y="${P[0].y}" as="sourcePoint"/>` +
      `<mxPoint x="${P[P.length - 1].x}" y="${P[P.length - 1].y}" as="targetPoint"/>` +
      (mid ? `<Array as="points">${mid}</Array>` : '') + `</mxGeometry></mxCell>`);
  };
  const arrowStyle = (el) => {
    const m = el.getAttribute('marker-end');
    const sw = +attr(el, 'stroke-width', 1);
    // the SVG marker is 6 stroke widths long
    return m ? `endArrow=block;endFill=1;endSize=${r2(Math.max(3, 5 * sw * S))};` : 'endArrow=none;';
  };
  const edgeBase = 'html=1;rounded=0;edgeStyle=none;curved=0;startArrow=none;jumpStyle=none;';

  // An element of the figure, alone, as an SVG image vertex (hatching, polygons).
  const imageOf = (el, extraSvg = '') => {
    const b = el.getBBox();
    const pad = +attr(el, 'stroke-width', 1) + 1;
    const x = b.x - pad, y = b.y - pad, w = b.width + 2 * pad, h = b.height + 2 * pad;
    const clone = el.cloneNode(true);
    clone.removeAttribute('opacity');                 // carried by the cell's own style
    for (const a of ['fill', 'stroke', 'stroke-width']) {
      if (!clone.getAttribute(a)) clone.setAttribute(a, attr(el, a, a === 'fill' ? 'black' : 'none'));
    }
    const src = `<svg xmlns="http://www.w3.org/2000/svg" width="${r2(w)}" height="${r2(h)}" ` +
      `viewBox="${r2(x)} ${r2(y)} ${r2(w)} ${r2(h)}">${extraSvg}${new XMLSerializer()
        .serializeToString(clone)}</svg>`;
    const m = ctm(el);
    const o = apply(m, x, y);
    vertex(el, o.x, o.y, w, h, `shape=image;html=1;imageAspect=0;image=data:image/svg+xml,${b64(src)};` +
      opacityStyle(el));
  };

  // ---- path parsing: absolute M L H V C Q Z (all this figure uses) -----------
  const subpaths = (d) => {
    const out = [];
    let cur = null, x = 0, y = 0, sx = 0, sy = 0, curved = false;
    const re = /([MLHVCQZ])([^MLHVCQZ]*)/gi;
    let m;
    while ((m = re.exec(d))) {
      const cmd = m[1].toUpperCase();
      const n = (m[2].match(/-?\d*\.?\d+(?:e-?\d+)?/gi) || []).map(Number);
      if (cmd === 'M') {
        if (cur) out.push({ pts: cur, curved });
        x = n[0]; y = n[1]; sx = x; sy = y; cur = [{ x, y }]; curved = false;
        for (let i = 2; i + 1 < n.length; i += 2) { x = n[i]; y = n[i + 1]; cur.push({ x, y }); }
      } else if (cmd === 'L') {
        for (let i = 0; i + 1 < n.length; i += 2) { x = n[i]; y = n[i + 1]; cur.push({ x, y }); }
      } else if (cmd === 'H') { x = n[0]; cur.push({ x, y }); }
      else if (cmd === 'V') { y = n[0]; cur.push({ x, y }); }
      else if (cmd === 'C' || cmd === 'Q') {
        const k = cmd === 'C' ? 6 : 4;
        for (let i = 0; i + k - 1 < n.length; i += k) {
          const x0 = x, y0 = y, seg = 14;
          for (let s = 1; s <= seg; s++) {
            const t = s / seg, u = 1 - t;
            let px, py;
            if (cmd === 'C') {
              px = u * u * u * x0 + 3 * u * u * t * n[i] + 3 * u * t * t * n[i + 2] + t * t * t * n[i + 4];
              py = u * u * u * y0 + 3 * u * u * t * n[i + 1] + 3 * u * t * t * n[i + 3] + t * t * t * n[i + 5];
            } else {
              px = u * u * x0 + 2 * u * t * n[i] + t * t * n[i + 2];
              py = u * u * y0 + 2 * u * t * n[i + 1] + t * t * n[i + 3];
            }
            cur.push({ x: px, y: py });
          }
          x = n[i + k - 2]; y = n[i + k - 1];
        }
        curved = true;
      } else if (cmd === 'Z') { cur.push({ x: sx, y: sy }); x = sx; y = sy; }
    }
    if (cur) out.push({ pts: cur, curved });
    return out;
  };

  // A text's label: plain text, or HTML with <sub> where the figure set a subscript
  // (<tspan class="sub">). draw.io renders html=1 labels as HTML.
  const htmlEsc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const labelOf = (el) => {
    if (!el.querySelector('tspan.sub')) return el.textContent;
    return [...el.childNodes].map((n) => {
      if (n.nodeType === 3) return htmlEsc(n.textContent);
      const t = htmlEsc(n.textContent);
      return n.classList && n.classList.contains('sub') ? `<sub>${t}</sub>` : t;
    }).join('');
  };

  // ---- walk the figure in paint order ----------------------------------------
  const els = svg.querySelectorAll('rect, circle, line, path, text, use');
  for (const el of els) {
    if (el.closest('defs')) continue;
    const tag = el.tagName.toLowerCase();
    const m = ctm(el);

    if (tag === 'rect') {
      const w = +el.getAttribute('width'), h = +el.getAttribute('height');
      if (w >= W - 1 && h >= H - 1) continue;         // the page background
      if (w <= 0 || h <= 0) continue;
      const o = apply(m, +el.getAttribute('x'), +el.getAttribute('y'));
      const rx = +(el.getAttribute('rx') || 0);
      const fill = attr(el, 'fill', 'black');
      vertex(el, o.x, o.y, w, h,
        `rounded=${rx > 0 ? 1 : 0};absoluteArcSize=1;arcSize=${r2(S * rx * 2)};whiteSpace=wrap;html=1;` +
        `fillColor=${fill === 'none' ? 'none' : fill};` + strokeStyle(el) + opacityStyle(el));
    } else if (tag === 'circle') {
      const r = +el.getAttribute('r');
      const o = apply(m, +el.getAttribute('cx'), +el.getAttribute('cy'));
      const fill = attr(el, 'fill', 'black');
      vertex(el, o.x - r, o.y - r, 2 * r, 2 * r,
        `ellipse;shape=ellipse;perimeter=ellipsePerimeter;whiteSpace=wrap;html=1;fillColor=${fill === 'none' ? 'none' : fill};` +
        strokeStyle(el) + opacityStyle(el));
    } else if (tag === 'line') {
      const a = apply(m, +el.getAttribute('x1'), +el.getAttribute('y1'));
      const b = apply(m, +el.getAttribute('x2'), +el.getAttribute('y2'));
      edge(el, [a, b], edgeBase + arrowStyle(el) + strokeStyle(el) + opacityStyle(el));
    } else if (tag === 'path') {
      const fill = attr(el, 'fill', 'black');
      const subs = subpaths(el.getAttribute('d'));
      const hatch = subs.length > 2 && subs.every((s) => s.pts.length === 2 && !s.curved);
      if ((fill && fill !== 'none') || hatch) { imageOf(el); continue; }
      subs.forEach((s, k) => {
        const pts = s.pts.map((p) => apply(m, p.x, p.y));
        // arrowhead only on the last subpath, where SVG draws it
        const arrow = k === subs.length - 1 ? arrowStyle(el) : 'endArrow=none;';
        edge(el, pts, edgeBase + arrow + strokeStyle(el) + opacityStyle(el));
      });
    } else if (tag === 'text') {
      const b = el.getBBox();
      const size = +attr(el, 'font-size', 12);
      const anchor = el.getAttribute('text-anchor') || 'start';
      const bold = (el.getAttribute('font-weight') || '') === 'bold';
      const italic = (el.getAttribute('font-style') || '') === 'italic';
      const fontStyle = (bold ? 1 : 0) + (italic ? 2 : 0);
      const pad = 2;
      const w = b.width + pad, h = b.height;
      const align = anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left';
      const c = apply(m, b.x + b.width / 2, b.y + b.height / 2);   // centre, rotation included
      const rot = Math.round(Math.atan2(m.b, m.a) * 180 / Math.PI);
      let x = c.x - w / 2;
      if (!rot) x = align === 'left' ? c.x - b.width / 2 : align === 'right' ? c.x + b.width / 2 - w : x;
      vertex(el, x, c.y - h / 2, w, h,
        `text;html=1;fillColor=none;strokeColor=none;whiteSpace=nowrap;align=${align};verticalAlign=middle;spacing=0;` +
        `spacingLeft=0;spacingRight=0;spacingTop=0;spacingBottom=0;fontFamily=Arial;` +
        `fontSize=${r2(S * size)};fontColor=${attr(el, 'fill', '#000000')};fontStyle=${fontStyle};` +
        (rot ? `rotation=${rot};` : '') + opacityStyle(el), labelOf(el));
    } else if (tag === 'use') {
      const ref = (el.getAttribute('href') || el.getAttribute('xlink:href') || '').slice(1);
      const sym = svg.querySelector(`symbol[id="${ref}"]`);
      if (!sym) continue;
      const w = +el.getAttribute('width'), h = +el.getAttribute('height');
      const o = apply(m, +el.getAttribute('x'), +el.getAttribute('y'));
      const src = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" ` +
        `viewBox="${sym.getAttribute('viewBox')}">${sym.innerHTML}</svg>`;
      vertex(el, o.x, o.y, w, h,
        `shape=image;html=1;imageAspect=0;aspect=fixed;image=data:image/svg+xml,${b64(src)};` +
        opacityStyle(el));
    }
  }

  return '<?xml version="1.0" encoding="UTF-8"?>\n' +
    `<mxfile host="app.diagrams.net" type="device">` +
    `<diagram id="recoverkv-${name}" name="${name}">` +
    `<mxGraphModel dx="${r2(S * W)}" dy="${r2(S * H)}" grid="0" gridSize="10" guides="1" tooltips="1" connect="0" ` +
    `arrows="0" fold="1" page="1" pageScale="1" pageWidth="${Math.ceil(S * W)}" pageHeight="${Math.ceil(S * H)}" ` +
    `background="#ffffff" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>` +
    cells.join('') + `</root></mxGraphModel></diagram></mxfile>\n`;
}
