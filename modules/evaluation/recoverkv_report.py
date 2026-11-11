"""RecoverKV aggregate report: every observation and experiment on one page.

Reads the tree ``scripts/run_recoverkv_all.sh`` writes::

    <root>/obs/<dataset>/<run>/obs_<arm>/all_results.json   Observations I-V
    <root>/lb/<arm>_b<budget>/<dataset>.{jsonl,meta.json}   LongBench
    <root>/ruler/<arm>_b<budget>/<task>.{jsonl,meta.json}   RULER

and writes ``<out>/report.md``, ``<out>/report.html`` (self-contained, figures
embedded), ``<out>/tables/*.csv`` and ``<out>/figures/*.png``.

What it checks before it reports anything
-----------------------------------------
* **Every accuracy sidecar's read-gate verdict.** A run whose ``read_gate`` is
  missing or not ``gated`` did not run the method; it is listed and excluded
  from every comparison (CLAUDE.md: a missing or non-``gated`` verdict means the
  number beside it is not a gated number).
* **Every arm's knobs.** A directory called ``oneway_b0.10`` whose sidecar says
  ``quant_promotion=bidir`` is a mislabelled run; it is listed and excluded.
* **Same examples.** Paired comparisons use only example ids present in both
  arms, and the count of ids missing from either side is reported.
* **Matched bytes (E2).** LongBench sidecars record the tier sizes the budget
  resolved to on the first example; Bidir and OneWay must agree.

The experiments
---------------
* **E1 — reading a quarter.** Per-head gate recall against the bytes the gate
  reads (every card plus ``ratio`` of the window data), next to a same-size
  hindsight pick; accuracy across gate ratios.
* **E2 — recovery at matched bytes.** Bidir (shipped) vs OneWay
  (``quant_promotion=oneway``), paired per example, with a dose-response of the
  accuracy gap against the attention mass promotions recover (Observation V).
* **E3 — what a recovered window is made of.** A = dequantized promotion
  (shipped), B = original fp oracle (outside the budget), C = no promotion; the
  headline is gap closure, ``(A - C) / (B - C)``: the share of the oracle's
  promotion benefit the deployable payload keeps.

CLI::

    python -m modules.evaluation.recoverkv_report --root outputs/recoverkv_<sha> \\
        --out outputs/recoverkv_<sha>/report
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import qevict_metrics as QM
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:                                        # pragma: no cover
    HAS_MPL = False

#: Arms and what each must record in its sidecar.
ARM_KNOBS: Dict[str, Tuple[str, str]] = {
    "bidir": ("bidir", "dequant"),
    "oneway": ("oneway", "dequant"),
    "original": ("bidir", "original"),
}
SHIPPED_GATE = 0.25
BENCH_NAME = {"lb": "LongBench", "ruler": "RULER"}
#: E3's three versions of a promoted window.
E3_ARMS = (("A", "bidir", "dequantized promotion (shipped)"),
           ("B", "original", "original fp promotion (oracle)"),
           ("C", "oneway", "no promotion"))


# ---------------------------------------------------------------------------
# per-example scoring (the rules longbench_scoring / ruler_scoring apply)
# ---------------------------------------------------------------------------


def longbench_example_scores(path: Path, dataset: str) -> Dict[str, float]:
    """``{_id: score in [0, 100]}``; a null prediction scores 0 (THUDM)."""
    from modules.evaluation.longbench_scoring import (
        METRIC_FN_REGISTRY, _FIRST_LINE_DATASETS, _load_dataset2metric)
    metric = METRIC_FN_REGISTRY[_load_dataset2metric()[dataset]]
    out: Dict[str, float] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        ex = json.loads(line)
        key = str(ex.get("_id", i))
        pred = ex.get("pred")
        if pred is None:
            out[key] = 0.0
            continue
        if dataset in _FIRST_LINE_DATASETS:
            pred = pred.lstrip("\n").split("\n")[0]
        all_classes = ex.get("all_classes") or []
        out[key] = 100.0 * max(
            (metric(pred, gt, all_classes=all_classes)
             for gt in ex.get("answers", [])), default=0.0)
    return out


_CONTROL = re.compile(r"[\x00-\x1f]")


def ruler_example_scores(path: Path, task: str) -> Dict[str, float]:
    """``{id: score in [0, 100]}``; a null prediction is NaN (dropped, as the
    RULER scorer drops it)."""
    part = task.split("_")[0] == "qa"
    out: Dict[str, float] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        ex = json.loads(line)
        key = str(ex.get("id", i))
        pred = ex.get("pred")
        refs = ex.get("answer", []) or []
        if pred is None or not refs:
            out[key] = float("nan")
            continue
        p = _CONTROL.sub("", pred.strip()).strip().lower()
        hits = [1.0 if r.lower() in p else 0.0 for r in refs]
        out[key] = 100.0 * (max(hits) if part else sum(hits) / len(hits))
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


@dataclass
class AccRun:
    bench: str               # "lb" | "ruler"
    arm: str                 # bidir | oneway | original | gate<r>
    budget: float
    task: str
    scores: Dict[str, float]
    meta: Dict[str, Any]
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def mean(self) -> float:
        v = np.array([x for x in self.scores.values() if np.isfinite(x)])
        return float(v.mean()) if v.size else float("nan")


def _arm_gate(arm: str) -> float:
    return float(arm[4:]) if arm.startswith("gate") else SHIPPED_GATE


def check_acc_run(run: AccRun) -> List[str]:
    """Why this run cannot be quoted as its arm (empty = it can)."""
    m = run.meta
    problems = []
    verdict = (m.get("read_gate") or {}).get("verdict")
    if verdict != "gated":
        problems.append(f"read_gate verdict {verdict!r}, not 'gated'")
    want = ARM_KNOBS.get(run.arm, ("bidir", "dequant"))
    got = (m.get("quant_promotion", "bidir"), m.get("quant_promote_source", "dequant"))
    if "quant_promotion" not in m:
        problems.append("sidecar predates quant_promotion / quant_promote_source")
    elif got != want:
        problems.append(f"sidecar knobs {got} do not match arm {run.arm} {want}")
    rf = (m.get("read_gate") or {}).get("read_fraction")
    g = _arm_gate(run.arm)
    # The gate opens ceil(ratio * n) windows, so on a small int2 tier the
    # realised fraction sits ABOVE the ratio -- legitimately. Reading LESS than
    # the arm's ratio is the one direction a correct run cannot produce.
    if rf is not None and float(rf) < g - 0.02:
        problems.append(f"realised read fraction {float(rf):.3f} is below the "
                        f"arm's gate ratio {g}")
    if abs(float(m.get("cache_budget", run.budget) or run.budget) - run.budget) > 1e-9:
        problems.append(f"cache_budget {m.get('cache_budget')} vs dir {run.budget}")
    return problems


def load_accuracy(root: Path) -> List[AccRun]:
    runs: List[AccRun] = []
    for bench in ("lb", "ruler"):
        for d in sorted((root / bench).glob("*_b*")):
            mt = re.fullmatch(r"(.+)_b([0-9.]+)", d.name)
            if not d.is_dir() or not mt:
                continue
            arm, budget = mt.group(1), float(mt.group(2))
            for j in sorted(d.glob("*.jsonl")):
                task = j.stem
                if task.endswith(".memory"):
                    continue
                meta_p = j.with_suffix(".meta.json")
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                try:
                    scores = (longbench_example_scores(j, task) if bench == "lb"
                              else ruler_example_scores(j, task))
                    err = None
                except Exception as e:                         # noqa: BLE001
                    # Listed, not dropped: a file the report silently skipped
                    # would read as a job that never ran.
                    log.warning("cannot score %s: %s", j, e)
                    scores, err = {}, f"cannot score: {type(e).__name__}: {e}"
                run = AccRun(bench, arm, budget, task, scores, meta)
                if err:
                    run.problems.append(err)
                elif not meta:
                    run.problems.append("no .meta.json (the run did not finish)")
                else:
                    run.problems += check_acc_run(run)
                runs.append(run)
    return runs


def _run_env(run_dir: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    p = run_dir / "run.env"
    if p.exists():
        for tok in p.read_text().split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                env.setdefault(k, v)
    return env


def load_observations(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``{(dataset, arm): all_results.json}``."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for f in sorted(root.glob("obs/*/*/obs_*/all_results.json")):
        arm = f.parent.name[len("obs_"):]
        dataset = _run_env(f.parent.parent).get("dataset", f.parent.parent.name)
        try:
            out[(dataset, arm)] = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            log.warning("unreadable %s: %s", f, e)
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def paired(a: Dict[str, float], b: Dict[str, float], n_boot: int = 2000,
           seed: int = 0) -> Dict[str, Any]:
    """``b - a`` per example over the ids both arms scored."""
    common = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    d = np.array([b[k] - a[k] for k in common], dtype=float)
    mu, lo, hi = QM.bootstrap_mean_ci(d, 0.95, n_boot, seed)
    return {"delta": mu, "ci_lower": lo, "ci_upper": hi, "n": int(d.size),
            "wins": int(np.sum(d > 1e-9)), "losses": int(np.sum(d < -1e-9)),
            "ties": int(np.sum(np.abs(d) <= 1e-9)),
            "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
            "diffs": d}


def macro_paired(pairs: Sequence[Dict[str, Any]], n_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Macro (mean over datasets) paired delta, with a stratified bootstrap:
    examples are resampled WITHIN each dataset, so a big dataset cannot
    dominate and a small one keeps its own noise."""
    diffs = [p["diffs"] for p in pairs if p["n"] > 0]
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean([d.mean() for d in diffs]))
    if n_boot <= 0:
        return mu, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.mean([d[rng.integers(0, d.size, d.size)].mean() for d in diffs])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mu, float(lo), float(hi)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# document model: one list of blocks, rendered to Markdown and to HTML
# ---------------------------------------------------------------------------


def _f(x: Any, d: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{v:.{d}f}"


def _p(x: Any, d: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{100 * v:.{d}f}%"


def _ci(mu: Any, lo: Any, hi: Any, d: int = 2, signed: bool = True) -> str:
    try:
        m = float(mu)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(m):
        return "—"
    s = f"{m:+.{d}f}" if signed else f"{m:.{d}f}"
    try:
        if math.isfinite(float(lo)) and math.isfinite(float(hi)):
            s += f" [{float(lo):+.{d}f}, {float(hi):+.{d}f}]" if signed else \
                 f" [{float(lo):.{d}f}, {float(hi):.{d}f}]"
    except (TypeError, ValueError):
        pass
    return s


class Doc:
    def __init__(self) -> None:
        self.blocks: List[Tuple[str, Any]] = []
        self.tables: Dict[str, Tuple[List[str], List[List[Any]]]] = {}

    def h1(self, t): self.blocks.append(("h1", t))
    def h2(self, t): self.blocks.append(("h2", t))
    def h3(self, t): self.blocks.append(("h3", t))
    def p(self, t): self.blocks.append(("p", t))
    def note(self, t): self.blocks.append(("note", t))
    def ul(self, items): self.blocks.append(("ul", list(items)))

    def table(self, name: str, headers: List[str], rows: List[List[Any]],
              caption: str = "") -> None:
        self.tables[name] = (headers, rows)
        self.blocks.append(("table", (headers, rows, caption)))

    def fig(self, path: Optional[Path], caption: str) -> None:
        if path is not None:
            self.blocks.append(("fig", (path, caption)))

    # -- rendering -----------------------------------------------------------

    def markdown(self, rel: Path) -> str:
        out: List[str] = []
        for kind, b in self.blocks:
            if kind in ("h1", "h2", "h3"):
                out += ["#" * int(kind[1]) + " " + b, ""]
            elif kind == "p":
                out += [b, ""]
            elif kind == "note":
                out += ["> " + b, ""]
            elif kind == "ul":
                out += [f"- {x}" for x in b] + [""]
            elif kind == "table":
                h, rows, cap = b
                if cap:
                    out += [f"*{cap}*", ""]
                out.append("| " + " | ".join(h) + " |")
                out.append("| " + " | ".join("---" for _ in h) + " |")
                out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
                out.append("")
            elif kind == "fig":
                path, cap = b
                out += [f"![{cap}]({path.relative_to(rel)})", "", f"*{cap}*", ""]
        return "\n".join(out)

    def html(self, title: str) -> str:
        def inline(t: str) -> str:
            t = html.escape(t)
            t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
            t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
            return t

        body: List[str] = []
        toc: List[str] = []
        for kind, b in self.blocks:
            if kind == "h1":
                body.append(f"<h1>{inline(b)}</h1>")
            elif kind == "h2":
                anchor = re.sub(r"[^a-z0-9]+", "-", b.lower()).strip("-")
                toc.append(f'<a href="#{anchor}">{inline(b)}</a>')
                body.append(f'<h2 id="{anchor}">{inline(b)}</h2>')
            elif kind == "h3":
                body.append(f"<h3>{inline(b)}</h3>")
            elif kind == "p":
                body.append(f"<p>{inline(b)}</p>")
            elif kind == "note":
                body.append(f'<p class="note">{inline(b)}</p>')
            elif kind == "ul":
                body.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in b)
                            + "</ul>")
            elif kind == "table":
                h, rows, cap = b
                t = ['<div class="tw"><table>']
                if cap:
                    t.append(f"<caption>{inline(cap)}</caption>")
                t.append("<thead><tr>" + "".join(f"<th>{inline(str(c))}</th>"
                                                 for c in h) + "</tr></thead><tbody>")
                for r in rows:
                    t.append("<tr>" + "".join(f"<td>{inline(str(c))}</td>"
                                              for c in r) + "</tr>")
                t.append("</tbody></table></div>")
                body.append("".join(t))
            elif kind == "fig":
                path, cap = b
                data = base64.b64encode(Path(path).read_bytes()).decode()
                body.append(f'<figure><img alt="{html.escape(cap)}" '
                            f'src="data:image/png;base64,{data}"/>'
                            f"<figcaption>{inline(cap)}</figcaption></figure>")
        nav = "<nav>" + " · ".join(toc) + "</nav>" if toc else ""
        return _HTML.format(title=html.escape(title), nav=nav, body="\n".join(body))


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#ffffff; --fg:#1d1f23; --muted:#5d6470; --line:#dfe3e8;
        --head:#f4f6f8; --accent:#2f5bd3; --note:#f6f3e6; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#15171b; --fg:#e7e9ed; --muted:#a3aab5; --line:#2e333b;
          --head:#1d2026; --accent:#8fb0ff; --note:#262417; }} }}
body {{ background:var(--bg); color:var(--fg); margin:0;
       font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.7rem; margin:.2em 0 .4em; }}
h2 {{ font-size:1.3rem; margin:2em 0 .5em; padding-top:.6em;
     border-top:1px solid var(--line); }}
h3 {{ font-size:1.05rem; margin:1.4em 0 .4em; }}
nav {{ font-size:.9rem; color:var(--muted); margin-bottom:1em; }}
nav a {{ color:var(--accent); text-decoration:none; }}
p.note {{ background:var(--note); padding:8px 12px; border-radius:6px; }}
.tw {{ overflow-x:auto; margin:.6em 0 1.2em; }}
table {{ border-collapse:collapse; font-size:.86rem; min-width:50%; }}
caption {{ text-align:left; color:var(--muted); padding-bottom:4px; }}
th, td {{ border:1px solid var(--line); padding:4px 8px; text-align:right;
         white-space:nowrap; }}
th {{ background:var(--head); }}
td:first-child, th:first-child {{ text-align:left; }}
code {{ font-size:.9em; }}
figure {{ margin:1em 0 1.6em; }}
figure img {{ max-width:100%; background:#fff; border-radius:4px; }}
figcaption {{ color:var(--muted); font-size:.88rem; }}
</style></head>
<body><main>{nav}
{body}
</main></body></html>
"""


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ledger(obs: Dict[Tuple[str, str], Dict], datasets: List[str],
               out: Path) -> Optional[Path]:
    if not HAS_MPL or not datasets:
        return None
    parts = ["fp", "local", "fresh", "q_read", "q_skipped", "evicted", "sink"]
    colors = ["#2f5bd3", "#6f8fe6", "#a9bdf2", "#2e9c6a", "#e0a526",
              "#c8423a", "#9aa1ab"]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(datasets) + 1.4))
    for i, d in enumerate(datasets):
        led = {r["part"]: r["share_of_head_mass"] for r in
               obs[(d, "gate0.25")]["observation4"]["ledger_table"]}
        left = 0.0
        for p, c in zip(parts, colors):
            v = float(led.get(p) or 0.0)
            ax.barh(i, v, left=left, color=c, label=p if i == 0 else None)
            left += v
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("share of each query head's attention (per head, then averaged)")
    ax.legend(ncol=7, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False)
    return _save(fig, out)


def fig_recall_vs_bytes(pts: Dict[str, List[Tuple[float, float, float]]],
                        card_frac: float, out: Path) -> Optional[Path]:
    """pts[dataset] = [(read_fraction, recall, oracle_recall)]."""
    if not HAS_MPL or not pts:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for d, rows in sorted(pts.items()):
        rows = sorted(rows)
        x = [card_frac + r for r, _, _ in rows]
        ax.plot(x, [y for _, y, _ in rows], "o-", label=d)
        ax.plot(x, [o for _, _, o in rows], ":", color=ax.lines[-1].get_color(),
                alpha=0.6)
    grid = np.linspace(0, 1, 11)
    ax.plot(card_frac + grid, grid, "k--", lw=1, label="uniform pick")
    ax.axvline(1.0, color="#999", lw=0.8)
    ax.text(1.0, 0.02, " reading every window, no cards", fontsize=8,
            color="#666")
    ax.set(xlabel="bytes read per int2 window per step, relative to its data "
                  "(cards + opened windows)",
           ylabel="per-head int2 recall", ylim=(0, 1.02))
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out)


def fig_dose_response(x: List[float], y: List[float], labels: List[str],
                      out: Path, xlabel: str) -> Optional[Path]:
    if not HAS_MPL or len(x) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.scatter(x, y, s=36)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set(xlabel=xlabel, ylabel="accuracy: Bidir − OneWay (points)")
    return _save(fig, out)


def fig_bars(groups: List[str], series: Dict[str, List[float]], out: Path,
             ylabel: str) -> Optional[Path]:
    if not HAS_MPL or not groups or not series:
        return None
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(groups) + 2), 4))
    n = len(series)
    w = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(np.arange(len(groups)) + (i - (n - 1) / 2) * w,
               [v if np.isfinite(v) else 0 for v in vals], w, label=name)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _obs_get(res: Dict, *path, default=None):
    cur: Any = res
    for k in path:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
            cur = cur[k]
        else:
            return default
        if cur is None:
            return default
    return cur


def _row(rows: List[Dict], key: str, value: Any) -> Dict:
    return next((r for r in rows or [] if r.get(key) == value), {})


def _ledger(res: Dict) -> Dict[str, float]:
    return {r["part"]: r["share_of_head_mass"]
            for r in _obs_get(res, "observation4", "ledger_table", default=[])}


def _rate(res: Dict, move: str, delta: int = 1) -> Any:
    for r in _obs_get(res, "observation5", "rate_table", default=[]):
        if r.get("move") == move and int(r.get("delta", -1)) == delta:
            return r.get("rate_mean")
    return None


def _outcome(res: Dict, move: str, key: str) -> Any:
    return _row(_obs_get(res, "observation5", "outcome_table", default=[]),
                "move", move).get(key)


def section_observations(doc: Doc, obs, figdir: Path) -> List[str]:
    datasets = sorted({d for d, a in obs if a == "gate0.25"})
    doc.h2("Observations I–V (shipped arm: gate 0.25)")
    if not datasets:
        doc.p("No observation run with the gate0.25 arm was found.")
        return []
    doc.p("Each row is one dataset: its documents' first PREFILL tokens plus the "
          "full-KV model's greedy continuation, teacher-forced through the "
          "cache. Masses are the full-KV model's attention (the ground truth); "
          "decisions and reads are the cache's.")
    rows = []
    for d in datasets:
        r = obs[(d, "gate0.25")]
        m10 = _row(_obs_get(r, "observation1", "mass_table", default=[]),
                   "top_fraction", 0.1).get("mean_cumulative_mass")
        o2 = {x["policy"]: x for x in _obs_get(r, "observation2", "summary_table",
                                               default=[])}
        lir = _obs_get(r, "observation3", "oracle", "quantifiable_result",
                       "global_lir_mean")
        sm = _obs_get(r, "observation5", "summary", default={})
        rows.append([
            d, _p(m10),
            _p(o2.get("measured_fp_only", {}).get("future_missed_mass_mean")),
            _p(o2.get("measured_fp_plus_q", {}).get("future_missed_mass_mean")),
            _p(lir), _p(_rate(r, "promote")), _f(_outcome(r, "promote", "lift")),
            _f(sm.get("swap_gain_mean")), _p(sm.get("hit_promote_mean")),
            _p(sm.get("eviction_regret_mean")),
            _p(sm.get("decision_fidelity_mean"))])
    doc.table("obs_why_tiers",
              ["dataset", "top-10% mass", "FMM fp only", "FMM fp+int2",
               "revival (oracle)", "P(promote)", "promote lift", "swap gain",
               "promote hit", "evict regret", "decision fidelity"], rows,
              "Why three tiers and why promotion. FMM = share of the next H steps' "
              "attention on windows the event dropped; lift = moved windows' "
              "future attention over the candidates' average.")
    rows = []
    for d in datasets:
        r = obs[(d, "gate0.25")]
        led = _ledger(r)
        rs = _obs_get(r, "observation4", "recall_summary", default={})
        exact = sum(float(led.get(k) or 0) for k in ("fp", "local", "fresh"))
        rows.append([d, _p(exact), _p(led.get("q_read")), _p(led.get("q_skipped")),
                     _p(led.get("evicted")), _p(led.get("sink")),
                     _p(rs.get("head_recall_mean")), _p(rs.get("oracle_recall_mean")),
                     _p(rs.get("worst_head_overall")),
                     _p(rs.get("realised_read_fraction"))])
    doc.table("obs_read_ledger",
              ["dataset", "read exactly", "int2 opened", "int2 skipped (centroid)",
               "evicted", "sink", "per-head recall", "hindsight pick",
               "worst head", "read fraction"], rows,
              "Observation IV: where each query head's attention landed at each "
              "decode step, per head then averaged.")
    doc.fig(fig_ledger(obs, datasets, figdir / "obs_ledger.png"),
            "Decode read ledger per dataset (gate 0.25).")
    return datasets


def section_e1(doc: Doc, obs, acc: List[AccRun], figdir: Path,
               n_boot: int, findings: List[str]) -> None:
    doc.h2("E1 — Reading a quarter, seeing nearly all of it")
    doc.p("The read gate opens a fraction of the int2 windows per KV head each "
          "step, chosen from a card per window. Recall is per query head: the "
          "share of that head's int2-tier attention sitting in the windows its "
          "KV group opened. The hindsight pick opens the same number of windows "
          "chosen with the true attention (best summed per-head share). The byte "
          "axis counts what the step reads: every card plus the opened windows.")
    gates = sorted({(a, float(a[4:])) for (_, a) in obs if a.startswith("gate")},
                   key=lambda t: t[1])
    datasets = sorted({d for d, _ in obs})
    pts: Dict[str, List[Tuple[float, float, float]]] = {}
    card_frac = float("nan")
    rows = []
    for d in datasets:
        for a, g in gates:
            r = obs.get((d, a))
            if r is None:
                continue
            rs = _obs_get(r, "observation4", "recall_summary", default={})
            led = _ledger(r)
            meta = r.get("metadata", {})
            bq, bc = meta.get("bytes_per_q_window"), meta.get("bytes_per_gate_card")
            if bq and bc and not math.isfinite(card_frac):
                card_frac = float(bc) / max(float(bq) - float(bc), 1.0)
            rf = rs.get("realised_read_fraction")
            if rs.get("gated") and rf is not None:
                pts.setdefault(d, []).append(
                    (float(rf), float(rs.get("head_recall_mean") or np.nan),
                     float(rs.get("oracle_recall_mean") or np.nan)))
            rows.append([d, g, _p(rf), _p(rs.get("head_recall_mean")),
                         _p(rs.get("oracle_recall_mean")),
                         _p(rs.get("worst_head_overall")),
                         _p(rs.get("heads_below_0.90")),
                         _p(led.get("q_skipped")),
                         _f(rs.get("mass_lift_per_opened_window"))])
    if rows:
        doc.table("e1_recall", ["dataset", "gate", "read fraction",
                                "per-head recall", "hindsight", "worst head",
                                "heads < 90%", "int2 skipped", "mass per opened window"],
                  rows, "Per-head recall across gate ratios (observation arms).")
        cf = card_frac if math.isfinite(card_frac) else 0.338
        doc.fig(fig_recall_vs_bytes(pts, cf, figdir / "e1_recall_vs_bytes.png"),
                "Per-head int2 recall against bytes read (solid: gate; dotted: "
                "hindsight pick of the same size; dashed: a uniform pick). "
                f"Cards cost {cf:.2f}x an int2 window's data, so reading every "
                "window through the gate costs more than reading it without one.")
        shipped = [(_obs_get(obs[(d, "gate0.25")], "observation4", "recall_summary",
                             default={})) for d in datasets if (d, "gate0.25") in obs]
        shipped = [rs for rs in shipped if rs.get("gated")]
        if shipped:
            rec = [float(rs.get("head_recall_mean") or np.nan) for rs in shipped]
            orc = [float(rs.get("oracle_recall_mean") or np.nan) for rs in shipped]
            rf = [float(rs.get("realised_read_fraction") or np.nan) for rs in shipped]
            findings.append(
                f"E1: opening {_p(np.nanmean(rf), 0)} of the int2 windows, the "
                f"gate captured a mean {_p(np.nanmean(rec))} of each query head's "
                f"int2-tier attention (hindsight pick of the same size: "
                f"{_p(np.nanmean(orc))}) across {len(shipped)} dataset(s).")
    else:
        doc.p("No gate-ratio observation arms were found.")

    # card widths
    cards = sorted({a for (_, a) in obs if a.startswith("card")})
    if cards:
        rows = []
        for d in datasets:
            for a in ["gate0.25"] + cards:
                r = obs.get((d, a))
                if r is None:
                    continue
                led = _ledger(r)
                rs = _obs_get(r, "observation4", "recall_summary", default={})
                g = r.get("geometry", {})
                rows.append([d, a, g.get("N_q", "—"), g.get("top_k_fp", "—"),
                             _p(led.get("evicted")), _p(led.get("q_skipped")),
                             _p(rs.get("head_recall_mean"))])
        doc.h3("Card width (bytes mode: a cheaper card buys more int2 windows)")
        doc.table("e1_card_width", ["dataset", "arm", "int2 windows (N_q)",
                                    "fp windows", "evicted", "int2 skipped",
                                    "per-head recall"], rows)

    # accuracy across gate ratios
    arms = sorted({"bidir"} | {r.arm for r in acc if r.arm.startswith("gate")},
                  key=_arm_gate)
    by = {(r.bench, r.arm, r.task): r for r in acc
          if r.ok and abs(r.budget - 0.20) < 1e-9 and r.arm in arms}
    tasks = sorted({(b, t) for (b, a, t) in by if a != "bidir"})
    if tasks:
        doc.h3("Accuracy across gate ratios (budget 0.20)")
        head = ["benchmark", "task"] + [
            f"gate {SHIPPED_GATE if a == 'bidir' else a[4:]}" for a in arms] + \
            ["0.25 − 1.0 (paired)"]
        rows = []
        pairs = []
        for b, t in tasks:
            vals = [_f(by[(b, a, t)].mean) if (b, a, t) in by else "—" for a in arms]
            cmp = "—"
            if (b, "bidir", t) in by and (b, "gate1.0", t) in by:
                pr = paired(by[(b, "gate1.0", t)].scores, by[(b, "bidir", t)].scores,
                            n_boot)
                pairs.append(pr)
                cmp = _ci(pr["delta"], pr["ci_lower"], pr["ci_upper"])
            rows.append([BENCH_NAME[b], t] + vals + [cmp])
        doc.table("e1_accuracy", head, rows,
                  "Mean score (0-100). The last column is the shipped gate minus "
                  "reading every window, paired per example.")
        if pairs:
            mu, lo, hi = macro_paired(pairs, n_boot)
            doc.p(f"Macro over {len(pairs)} task(s): gate 0.25 − gate 1.0 = "
                  f"{_ci(mu, lo, hi)} points.")
            findings.append(
                f"E1: reading 25% of the int2 tier instead of all of it changed "
                f"accuracy by {_ci(mu, lo, hi)} points (macro, {len(pairs)} tasks).")


def _acc_index(acc: List[AccRun]) -> Dict[Tuple[str, str, float, str], AccRun]:
    return {(r.bench, r.arm, round(r.budget, 4), r.task): r for r in acc if r.ok}


def section_e2(doc: Doc, obs, acc: List[AccRun], figdir: Path, n_boot: int,
               findings: List[str]) -> None:
    doc.h2("E2 — Recovery at matched bytes: Bidir vs OneWay")
    doc.p("Two policies identical in every respect but one: OneWay "
          "(`quant_promotion=oneway`) never promotes an int2 window back to fp "
          "(F → Q → E), Bidir (shipped) does (F ⇄ Q → E). Same examples, greedy "
          "decoding, eviction scores, quantizer, window size, q and byte budget — "
          "the tier sizes come from the budget resolver, which does not know "
          "about promotion. Deltas are Bidir − OneWay, paired per example.")
    idx = _acc_index(acc)
    budgets = sorted({b for (_, a, b, _) in idx if a == "oneway"})
    if not budgets:
        doc.p("No OneWay accuracy runs were found.")
        return
    macro_by_budget: Dict[Tuple[str, float], Tuple[float, float, float, int]] = {}
    delta_020: Dict[str, float] = {}
    for bench, bname in (("lb", "LongBench"), ("ruler", "RULER")):
        for b in budgets:
            tasks = sorted({t for (be, a, bb, t) in idx
                            if be == bench and bb == b and a == "oneway"})
            if not tasks:
                continue
            rows, pairs = [], []
            mismatched = 0
            for t in tasks:
                A, C = idx.get((bench, "bidir", b, t)), idx.get((bench, "oneway", b, t))
                if A is None or C is None:
                    continue
                pr = paired(C.scores, A.scores, n_boot)
                pairs.append(pr)
                ga = A.meta.get("resolved_geometry_first_example") or {}
                gc = C.meta.get("resolved_geometry_first_example") or {}
                same = (ga.get("top_k_fp"), ga.get("N_q")) == (gc.get("top_k_fp"),
                                                             gc.get("N_q"))
                mismatched += int(bool(ga) and bool(gc) and not same)
                rows.append([t, _f(A.mean), _f(C.mean),
                             _ci(pr["delta"], pr["ci_lower"], pr["ci_upper"]),
                             f"{pr['wins']}/{pr['ties']}/{pr['losses']}", pr["n"],
                             "yes" if same or not (ga and gc) else "NO"])
                if abs(b - 0.20) < 1e-9 and bench == "lb":
                    delta_020[t] = pr["delta"]
            if not pairs:
                continue
            mu, lo, hi = macro_paired(pairs, n_boot)
            macro_by_budget[(bench, b)] = (mu, lo, hi, len(pairs))
            rows.append(["**macro**", _f(np.mean([idx[(bench, 'bidir', b, t)].mean
                                                   for t in tasks
                                                   if (bench, 'bidir', b, t) in idx])),
                         _f(np.mean([idx[(bench, 'oneway', b, t)].mean for t in tasks
                                     if (bench, 'oneway', b, t) in idx])),
                         _ci(mu, lo, hi), "", "", ""])
            doc.table(f"e2_{bench}_b{b}",
                      ["task", "Bidir", "OneWay", "Δ [95% CI]", "W/T/L", "n",
                       "tier sizes match"], rows,
                      f"{bname}, cache budget {b:.2f}.")
            if mismatched:
                doc.note(f"{mismatched} task(s) at budget {b:.2f} resolved "
                         "different tier sizes for the two arms — that is not a "
                         "matched-bytes comparison; check the sidecars.")
            wins = sum(1 for p in pairs if p["delta"] > 0)
            findings.append(
                f"E2 ({bname}, {int(round(b * 100))}%): Bidir − OneWay = "
                f"{_ci(mu, lo, hi)} points macro; Bidir ahead on {wins} of "
                f"{len(pairs)} task(s).")
    if macro_by_budget:
        groups = [f"{BENCH_NAME[bn]} {int(round(b * 100))}%"
                  for (bn, b) in macro_by_budget]
        doc.fig(fig_bars(groups, {"Bidir − OneWay": [v[0] for v in
                                                     macro_by_budget.values()]},
                         figdir / "e2_macro.png", "points (macro, paired)"),
                "Macro paired accuracy gain of promotion, by benchmark and budget.")

    # dose-response against Observation V
    xs, ys, labs = [], [], []
    for (d, a), r in obs.items():
        if a != "gate0.25" or d not in delta_020:
            continue
        rec = _outcome(r, "promote", "future_mass_share")
        if rec is not None:
            xs.append(float(rec))
            ys.append(delta_020[d])
            labs.append(d)
    if xs:
        rho = spearman(xs, ys)
        doc.h3("Dose-response: does promotion pay where it recovers attention?")
        doc.fig(fig_dose_response(xs, ys, labs, figdir / "e2_dose_response.png",
                                  "attention recovered by promotions (Observation "
                                  "V: share of candidates' next-H mass on promoted "
                                  "windows)"),
                f"Each point is a LongBench dataset at budget 0.20. Spearman ρ = "
                f"{_f(rho)} over {len(xs)} dataset(s).")
        if math.isfinite(rho):
            findings.append(
                f"E2: the Bidir − OneWay gap tracks the attention promotions "
                f"recover (Spearman ρ = {_f(rho)}, {len(xs)} datasets).")

    # mechanism from the observation arms
    rows = []
    for d in sorted({d for d, a in obs if a == "oneway"}):
        b_, o_ = obs.get((d, "gate0.25")), obs.get((d, "oneway"))
        if b_ is None or o_ is None:
            continue
        lb, lo_ = _ledger(b_), _ledger(o_)
        pr_o = _rate(o_, "promote")
        rows.append([d, _p(_rate(b_, "promote")), _p(pr_o),
                     _p(lb.get("q_skipped")), _p(lo_.get("q_skipped")),
                     _p(lb.get("evicted")), _p(lo_.get("evicted")),
                     _f(_outcome(o_, "stay_q", "lift"))])
        if pr_o is not None and float(pr_o) > 0:
            doc.note(f"{d}: the OneWay observation arm shows promotions "
                     f"({_p(pr_o)}) — the knob did not take effect; do not use it.")
    if rows:
        doc.h3("Mechanism (observation arms)")
        doc.table("e2_mechanism", ["dataset", "P(promote) Bidir", "P(promote) OneWay",
                                   "int2 skipped Bidir", "int2 skipped OneWay",
                                   "evicted Bidir", "evicted OneWay",
                                   "OneWay stay-in-int2 lift"], rows,
                  "Without promotion, windows whose attention revives stay in the "
                  "gated tier: more of their attention is read only through "
                  "centroids (int2 skipped), and the windows stuck there carry "
                  "above-average future attention (lift > 1).")


def section_e3(doc: Doc, obs, acc: List[AccRun], figdir: Path, n_boot: int,
               findings: List[str]) -> None:
    doc.h2("E3 — What a recovered window is made of")
    doc.p("For windows that were quantized and later promoted: A = the shipped "
          "dequantized fp residency, B = the original pre-quantization fp K/V "
          "(an oracle held OUTSIDE the byte budget, evaluation only), C = no "
          "promotion (OneWay). Gap closure (A − C) / (B − C) is the share of the "
          "oracle's promotion benefit the deployable payload keeps; it is shown "
          "only where B − C is at least one point.")
    idx = _acc_index(acc)
    budgets = sorted({b for (_, a, b, _) in idx if a == "original"})
    if not budgets:
        doc.p("No original-payload (oracle) accuracy runs were found.")
        return
    groups, series = [], {"A dequant": [], "B original": [], "C none": []}
    for bench, bname in (("lb", "LongBench"), ("ruler", "RULER")):
        for b in budgets:
            tasks = sorted({t for (be, a, bb, t) in idx
                            if be == bench and bb == b and a == "original"})
            rows, pairs_ba = [], []
            means = {k: [] for k in "ABC"}
            closures = []
            for t in tasks:
                runs = {k: idx.get((bench, arm, b, t)) for k, arm, _ in E3_ARMS}
                if any(v is None for v in runs.values()):
                    continue
                pr = paired(runs["A"].scores, runs["B"].scores, n_boot)
                pairs_ba.append(pr)
                a_, b2, c_ = runs["A"].mean, runs["B"].mean, runs["C"].mean
                for k, v in zip("ABC", (a_, b2, c_)):
                    means[k].append(v)
                gc = (a_ - c_) / (b2 - c_) if (b2 - c_) >= 1.0 else float("nan")
                if math.isfinite(gc):
                    closures.append(gc)
                rows.append([t, _f(a_), _f(b2), _f(c_),
                             _ci(pr["delta"], pr["ci_lower"], pr["ci_upper"]),
                             _f(a_ - c_), _p(gc, 0)])
            if not pairs_ba:
                continue
            mu, lo, hi = macro_paired(pairs_ba, n_boot)
            mA, mB, mC = (float(np.mean(means[k])) for k in "ABC")
            gc_macro = (mA - mC) / (mB - mC) if (mB - mC) >= 1.0 else float("nan")
            rows.append(["**macro**", _f(mA), _f(mB), _f(mC), _ci(mu, lo, hi),
                         _f(mA - mC), _p(gc_macro, 0)])
            doc.table(f"e3_{bench}_b{b}",
                      ["task", "A dequant", "B original (oracle)", "C no promotion",
                       "B − A [95% CI]", "A − C", "gap closure"], rows,
                      f"{bname}, cache budget {b:.2f}.")
            groups.append(f"{bname} {int(round(b * 100))}%")
            series["A dequant"].append(mA)
            series["B original"].append(mB)
            series["C none"].append(mC)
            findings.append(
                f"E3 ({bname}, {int(round(b * 100))}%): the oracle fp payload "
                f"adds {_ci(mu, lo, hi)} points over the dequantized one; "
                f"promotion itself is worth {_f(mA - mC)} points (A − C)"
                + (f", so the deployable payload keeps {_p(gc_macro, 0)} of the "
                   "oracle's benefit." if math.isfinite(gc_macro) else "."))
    doc.fig(fig_bars(groups, series, figdir / "e3_abc.png", "mean score (macro)"),
            "A / B / C by benchmark and budget.")


def section_integrity(doc: Doc, root: Path, acc: List[AccRun], obs) -> None:
    doc.h2("Run integrity")
    env = (root / "run.env").read_text().strip().splitlines() \
        if (root / "run.env").exists() else []
    if env:
        doc.ul([f"`{line}`" for line in env])
    jobs = (root / "jobs.txt").read_text().split("\n") \
        if (root / "jobs.txt").exists() else []
    jobs = [j for j in jobs if j.strip()]
    failed = (root / "failed_jobs.txt").read_text().strip().splitlines() \
        if (root / "failed_jobs.txt").exists() else []
    bad = [r for r in acc if not r.ok]
    obs_paths = {}
    for (d, a), r in obs.items():
        obs_paths.setdefault(r.get("metadata", {}).get("read_path", "?"), []).append(
            f"{d}/{a}")
    doc.table("integrity", ["check", "result"], [
        ["jobs queued", len(jobs) or "—"],
        ["jobs failed", len(failed)],
        ["accuracy files scored", len(acc)],
        ["accuracy files usable (gated, knobs match the arm)", len(acc) - len(bad)],
        ["observation arms", len(obs)],
        ["observation read paths", ", ".join(f"{k}: {len(v)}"
                                             for k, v in sorted(obs_paths.items()))],
    ])
    if bad:
        doc.h3("Excluded accuracy files")
        doc.table("excluded", ["bench", "arm", "budget", "task", "why"],
                  [[r.bench, r.arm, r.budget, r.task, "; ".join(r.problems)]
                   for r in bad])
    bad_obs = [f"{d}/{a}: {r['metadata'].get('read_path')}"
               for (d, a), r in obs.items()
               if r.get("metadata", {}).get("read_path") not in ("gated", None)]
    if bad_obs:
        doc.note("Observation arms not on the recorded gated path (Observation IV "
                 "counts their whole int2 tier as read): " + "; ".join(bad_obs))
    if failed:
        doc.h3("Failed jobs")
        doc.ul([f"`{f}`" for f in failed])


def section_definitions(doc: Doc) -> None:
    doc.h2("How to read the numbers")
    doc.ul([
        "**Per-head recall** — for one query head, the share of its int2-tier "
        "attention (full-KV model's) that sat in windows its KV group's gate "
        "opened; averaged over heads after, never summed across heads first "
        "(a group sum is dominated by its loudest head).",
        "**Hindsight pick** — the same number of windows per KV head, chosen with "
        "the true attention to maximise the group's summed per-head share. A "
        "ceiling for any card-based gate of that size.",
        "**Read exactly / int2 opened / int2 skipped / evicted** — where a head's "
        "attention landed at a decode step: fp and local tokens, int2 windows "
        "read at 2 bits, int2 windows reached only through their value centroid, "
        "and windows the cache no longer holds.",
        "**P(promote)** — P(int2 → fp at the next eviction | int2 now). The "
        "three-state chain keeps evicted windows out of the denominator.",
        "**Lift** — a moved window's attention over the next H steps divided by "
        "the average over every window that eviction decided. Promote lift > 1 "
        "and demote lift < 1 mean the moves went the right way; swap gain is "
        "their difference.",
        "**Promote hit** — share of promotions into the hindsight-best fp set "
        "(the top-|F| windows by future attention).",
        "**Eviction regret** — share of the decided windows' future attention "
        "that sat on windows evicted at that event.",
        "**Decision fidelity** — Jaccard of the cache's fp set with the full-KV "
        "ranking of the same survivors: how well the cache's own running scores "
        "(gate-filled for skipped int2 windows) rank what it holds.",
        "**Paired Δ** — per-example difference over ids both arms scored; the CI "
        "is a percentile bootstrap. Macro CIs resample within each task.",
    ])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_report(root: Path, out: Path, n_boot: int = 2000) -> Dict[str, Path]:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    obs = load_observations(root)
    acc = load_accuracy(root)
    log.info("report: %d observation arm(s), %d accuracy file(s)", len(obs), len(acc))

    doc = Doc()
    doc.h1("RecoverKV — observations and experiments")
    doc.p(f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from "
          f"`{root}`.")
    findings: List[str] = []
    head_at = len(doc.blocks)
    section_integrity(doc, root, acc, obs)
    section_observations(doc, obs, figdir)
    section_e1(doc, obs, acc, figdir, n_boot, findings)
    section_e2(doc, obs, acc, figdir, n_boot, findings)
    section_e3(doc, obs, acc, figdir, n_boot, findings)
    section_definitions(doc)
    # Findings go first, but can only be written once every section has run.
    key = [("h2", "Key findings"),
           ("ul", findings or ["No experiment produced a comparable result yet."]),
           ("note", "Every sentence above is computed from the tables below; a "
                    "result that did not resolve is left out rather than "
                    "reported as zero. Excluded files are listed under Run "
                    "integrity.")]
    doc.blocks[head_at:head_at] = key

    (out / "tables").mkdir(exist_ok=True)
    for name, (h, rows) in doc.tables.items():
        with open(out / "tables" / f"{name}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(h)
            w.writerows(rows)
    md = out / "report.md"
    md.write_text(doc.markdown(out), encoding="utf-8")
    hp = out / "report.html"
    hp.write_text(doc.html("RecoverKV report"), encoding="utf-8")
    (out / "findings.txt").write_text("\n".join(findings) + "\n", encoding="utf-8")
    for s in findings:
        log.info("finding: %s", s)
    return {"markdown": md, "html": hp}


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    paths = build_report(a.root, a.out or a.root / "report", a.bootstrap)
    print((a.out or a.root / "report").joinpath("findings.txt").read_text())
    print(f"report: {paths['html']}")


if __name__ == "__main__":
    _cli()
