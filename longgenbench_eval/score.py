"""Score judged LongGenBench generations: the paper's Table 2 columns.

    python -m longgenbench_eval.score --setting 16K --data data/longgenbench \\
        --run ours runs/ours.jsonl --run fullkv runs/fullkv.jsonl --json scores.json

Each ``--run NAME GENERATIONS [VERDICTS]`` is scored with CR, STIC-1 and STIC-2
(the paper's three metrics) and Table 2's derived Len and wAvg, overall and per
task type. VERDICTS defaults to ``GENERATIONS`` with ``.verdicts.jsonl``; it is
refused unless its meta records the hash of exactly that generations file.

With several runs, every run is scored on the documents they ALL have (a
baseline that generated a subset is compared on that subset), and each run after
the first gets ``run - first`` with a 95% interval from a paired bootstrap over
documents. The paper's Llama-3.1-8B-Instruct row is printed for context: it was
generated with vLLM in bf16 on all 400 documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .data import load_all
from .judge import sha16
from .metrics import aggregate, doc_stats, load_generations, load_verdicts, paired_bootstrap, rounded
from .reference import PAPER_TABLE2, TABLE2_COLUMNS, TYPES


def verdicts_path(gen: str | Path) -> Path:
    g = Path(gen)
    return g.parent / (g.name[:-len(g.suffix)] + ".verdicts.jsonl")


def load_run(docs: Sequence[Dict[str, Any]], gen: str | Path,
             ver: Optional[str | Path] = None) -> Tuple[Dict[int, Optional[str]], Dict[int, List[Dict]]]:
    """``(outputs, verdicts)`` for one run, after checking they belong together."""
    gen = Path(gen)
    ver = Path(ver) if ver else verdicts_path(gen)
    meta_p = ver.with_suffix(".meta.json")
    if not meta_p.is_file():
        raise FileNotFoundError(f"{meta_p} missing: judge {gen} first (python -m longgenbench_eval.judge)")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("generations_sha") != sha16(gen):
        raise ValueError(f"{ver} judged a different version of {gen}: judge it again")
    return load_generations(gen, docs), load_verdicts(ver)


def run_stats(docs_by_id: Dict[int, Dict], outputs: Dict[int, Optional[str]],
              verdicts: Dict[int, List[Dict]], ids: Sequence[int]) -> List[Dict[str, Any]]:
    return [doc_stats(docs_by_id[i], outputs[i], [v["verdict"] for v in verdicts.get(i, [])]) for i in ids]


def score(setting: str, data_dir: str | Path, runs: Sequence[Tuple[str, str, Optional[str]]],
          iters: int = 2000) -> Dict[str, Any]:
    docs = load_all(data_dir, setting)
    by_id = {d["id"]: d for d in docs}
    loaded = {name: load_run(docs, g, v) for name, g, v in runs}
    common = sorted(set.intersection(*(set(o) for o, _ in loaded.values())))
    if not common:
        raise ValueError("the runs have no document in common")
    result: Dict[str, Any] = {"setting": setting, "ids": common, "runs": {}, "deltas": {}}
    scopes = [("ALL", common)] + [(t, [i for i in common if by_id[i]["type"] == t]) for t in TYPES[setting]]
    stats = {name: run_stats(by_id, o, v, common) for name, (o, v) in loaded.items()}
    pos = {i: k for k, i in enumerate(common)}
    first = runs[0][0]
    for name in loaded:
        result["runs"][name] = {}
        for scope, ids in scopes:
            if not ids:
                continue
            s = rounded(aggregate([stats[name][pos[i]] for i in ids]))
            result["runs"][name][scope] = s
        if name != first:
            result["deltas"][name] = {}
            for scope, ids in scopes:
                if ids:
                    d = paired_bootstrap([stats[first][pos[i]] for i in ids],
                                         [stats[name][pos[i]] for i in ids], iters=iters)
                    result["deltas"][name][scope] = {k: {kk: round(vv, 1) for kk, vv in v.items()}
                                                     for k, v in d.items()}
    result["paper_llama31_8b"] = dict(zip(TABLE2_COLUMNS, PAPER_TABLE2[setting]["Llama-3.1-8B-Instruct"]))
    return result


def format_table(result: Dict[str, Any]) -> str:
    def f(x):
        return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.1f}" if isinstance(x, float) else str(x))

    hdr = f"{'run':18s} {'scope':10s} {'docs':>5s} {'CR':>6s} {'STIC-1':>7s} {'STIC-2':>7s} {'Len':>6s} {'wAvg':>6s}"
    lines = [hdr, "-" * len(hdr)]
    for name, scopes in result["runs"].items():
        for scope, s in scopes.items():
            lines.append(f"{name:18s} {scope:10s} {s['n_docs']:>5d} {f(s['CR']):>6s} {f(s['STIC-1']):>7s} "
                         f"{f(s['STIC-2']):>7s} {f(s['Len']):>6s} {f(s['wAvg']):>6s}")
        lines.append("")
    p = result["paper_llama31_8b"]
    lines.append(f"{'paper Llama3.1-8B':18s} {'ALL':10s} {400:>5d} {p['CR']:>6.1f} {p['STIC-1']:>7.1f} "
                 f"{p['STIC-2']:>7.1f} {p['Len']:>6d} {p['wAvg']:>6.1f}   (Table 2; vLLM, bf16, all 400; context)")
    if result["deltas"]:
        first = next(iter(result["runs"]))
        lines.append(f"\nrun - {first} on the same documents, [95% CI] from a paired bootstrap over documents:")
        for name, scopes in result["deltas"].items():
            for scope, d in scopes.items():
                lines.append(f"  {name:16s} {scope:10s} " + "   ".join(
                    f"{k} {v['delta']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]" for k, v in d.items()))
    lines.append("\nCR, STIC-1, STIC-2, wAvg in %; Len in words. STIC-1 = yes / instructions judged, STIC-2 = yes / "
                 "all instructions,\nwAvg = CR x STIC-2, Len = mean words over outputs with > 40 entries and > 2000 "
                 "words (paper, Table 2).")
    return "\n".join(lines)


def _cli_main() -> None:
    p = argparse.ArgumentParser(description="Score judged LongGenBench generations (paper Table 2 columns).")
    p.add_argument("--setting", choices=("16K", "32K"), default="16K")
    p.add_argument("--data", default="data/longgenbench")
    p.add_argument("--run", nargs="+", action="append", required=True, metavar="NAME GEN [VERDICTS]",
                   help="repeatable: a run name, its generations file, optionally its verdicts file")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--json", default=None, help="also write the result here")
    a = p.parse_args()
    runs = []
    for r in a.run:
        if len(r) not in (2, 3):
            p.error("--run takes NAME GENERATIONS [VERDICTS]")
        runs.append((r[0], r[1], r[2] if len(r) == 3 else None))
    result = score(a.setting, a.data, runs, a.iters)
    print(format_table(result))
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _cli_main()
