"""The aggregate report over a whole run tree (scripts/run_recoverkv_all.sh).

Builds a synthetic tree with every arm the driver produces -- observation arms
from the simulated three-tier cache, LongBench and RULER prediction files with
sidecars -- plus two runs that must be EXCLUDED (an ungated one and one whose
sidecar knobs contradict its arm), and checks the report's numbers and its
refusals. CPU-only, no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modules.evaluation import qevict_observations as QO
from modules.evaluation import recoverkv_report as RR
from tests.test_observation_gate_and_tiers import _sim_pair

LB_TASKS = ("hotpotqa", "passage_retrieval_en")
RULER_TASKS = ("cwe", "qa_1")
#: Share of examples each arm gets right (the rest answer wrong).
ACC = {"bidir": 0.70, "oneway": 0.55, "original": 0.74, "gate1.0": 0.71,
       "gate0.10": 0.62}


def _write_obs(root: Path, dataset: str, arm: str, **sim) -> None:
    run = root / "obs" / dataset.replace(":", "_") / f"{dataset}_run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.env").write_text(f"dataset={dataset} corpus=x\n")
    base, ours = _sim_pair(T=40, **sim)
    ours["metadata"].update(quant_promote_source="original" if arm == "original"
                            else "dequant")
    paths = []
    for name, d in (("base", base), (f"ours_{arm}", ours)):
        p = run / f"parity_{name}.npz"
        np.savez_compressed(str(p), **d["arrays"], metadata_json=np.array(
            [json.dumps(d["metadata"])], dtype=object))
        paths.append(p)
    QO.run_observations(paths[0], paths[1], run / f"obs_{arm}", fmm_horizon=4,
                        n_boot=20, figures=False, primary_inactivity=2,
                        primary_lir_horizon=2)


def _meta(arm: str, budget: float, verdict: str = "gated",
          promotion: str = None) -> dict:
    knobs = RR.ARM_KNOBS.get(arm, ("bidir", "dequant"))
    return {"read_gate": {"verdict": verdict,
                          "read_fraction": RR._arm_gate(arm)},
            "quant_promotion": promotion or knobs[0],
            "quant_promote_source": knobs[1], "cache_budget": budget,
            "resolved_geometry_first_example": {"top_k_fp": 10, "N_q": 40}}


def _write_acc(root: Path, bench: str, arm: str, budget: float, task: str,
               n: int = 60, **meta_kw) -> None:
    d = root / bench / f"{arm}_b{budget}"
    d.mkdir(parents=True, exist_ok=True)
    # The same questions in every arm, ordered by difficulty: an arm gets the
    # easiest ACC[arm] share right, so arms are nested and the pairing is real.
    right = (np.arange(n) + 0.5) / n < ACC[arm]
    lines = []
    for i in range(n):
        if bench == "lb":
            gold = "Paragraph 3" if task == "passage_retrieval_en" else "paris"
            pred = gold if right[i] else "london"
            lines.append({"pred": pred, "answers": [gold], "all_classes": None,
                          "length": 100, "_id": f"q{i}"})
        else:
            lines.append({"pred": "the answer is alpha" if right[i] else "none",
                          "answer": ["alpha"], "task": task,
                          "max_new_tokens": 32, "id": i})
    (d / f"{task}.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (d / f"{task}.meta.json").write_text(json.dumps(_meta(arm, budget, **meta_kw)))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("recoverkv")
    (root / "run.env").write_text("commit=abc smoke=1\n")
    (root / "jobs.txt").write_text("obs hotpotqa\nlb bidir 0.2 hotpotqa\n")
    (root / "failed_jobs.txt").write_text("")
    for ds in ("hotpotqa", "passage_retrieval_en"):
        _write_obs(root, ds, "gate0.25", gate_ratio=0.25)
        _write_obs(root, ds, "gate1.0", gate_ratio=1.0)
        _write_obs(root, ds, "oneway", gate_ratio=0.25, oneway=True)
        _write_obs(root, ds, "original", gate_ratio=0.25)
    for budget in (0.1, 0.2):
        for arm in ("bidir", "oneway", "original"):
            for t in LB_TASKS:
                _write_acc(root, "lb", arm, budget, t)
            for t in RULER_TASKS:
                _write_acc(root, "ruler", arm, budget, t)
    for arm in ("gate1.0", "gate0.10"):
        for t in LB_TASKS:
            _write_acc(root, "lb", arm, 0.2, t)
    # Two runs that must be refused.
    _write_acc(root, "lb", "gate0.10", 0.2, "hotpotqa", verdict="not-expected")
    _write_acc(root, "ruler", "oneway", 0.1, "qa_1", promotion="bidir")
    out = root / "report"
    RR.build_report(root, out, n_boot=200)
    return root, out


class TestScoring:
    def test_per_example_scores_match_the_official_scorers(self, tree):
        from modules.evaluation.longbench_scoring import score_predictions as lb_score
        from modules.evaluation.ruler_scoring import score_predictions as ru_score
        root, _ = tree
        d = root / "lb" / "bidir_b0.2"
        ref = lb_score(d)
        for t in LB_TASKS:
            ex = RR.longbench_example_scores(d / f"{t}.jsonl", t)
            assert np.mean(list(ex.values())) == pytest.approx(ref[t], abs=0.01)
        d = root / "ruler" / "bidir_b0.2"
        ref = ru_score(d)
        for t in RULER_TASKS:
            ex = RR.ruler_example_scores(d / f"{t}.jsonl", t)
            assert np.nanmean(list(ex.values())) == pytest.approx(ref[t], abs=0.01)

    def test_paired_uses_only_common_ids(self):
        a = {"x": 0.0, "y": 100.0, "z": 50.0}
        b = {"x": 100.0, "y": 100.0, "w": 0.0}
        p = RR.paired(a, b, n_boot=50)
        assert p["n"] == 2 and p["delta"] == pytest.approx(50.0)
        assert (p["wins"], p["ties"], p["losses"]) == (1, 1, 0)
        assert p["only_a"] == 1 and p["only_b"] == 1

    def test_spearman(self):
        assert RR.spearman([1, 2, 3, 4], [2, 4, 6, 9]) == pytest.approx(1.0)
        assert np.isnan(RR.spearman([1, 2], [1, 2]))


class TestIntegrity:
    def test_ungated_and_mislabelled_runs_are_excluded(self, tree):
        root, _ = tree
        runs = RR.load_accuracy(root)
        bad = {(r.bench, r.arm, r.budget, r.task): r.problems
               for r in runs if not r.ok}
        assert any("not 'gated'" in p for p in bad[("lb", "gate0.10", 0.2, "hotpotqa")])
        assert any("do not match arm oneway" in p
                   for p in bad[("ruler", "oneway", 0.1, "qa_1")])
        assert len(bad) == 2

    def test_read_fraction_may_exceed_the_ratio_but_not_fall_below_it(self):
        """ceil(ratio * n) over a small tier reads MORE than the ratio; a run
        reading less than its arm's ratio was configured as another arm."""
        above = RR.AccRun("lb", "gate0.10", 0.2, "t", {}, _meta("gate0.10", 0.2))
        above.meta["read_gate"]["read_fraction"] = 0.19
        assert RR.check_acc_run(above) == []
        below = RR.AccRun("lb", "bidir", 0.2, "t", {}, _meta("bidir", 0.2))
        below.meta["read_gate"]["read_fraction"] = 0.10
        assert any("below the arm's gate ratio" in p for p in RR.check_acc_run(below))

    def test_the_excluded_runs_are_listed_in_the_report(self, tree):
        _, out = tree
        md = (out / "report.md").read_text()
        assert "Excluded accuracy files" in md
        assert "not 'gated'" in md and "do not match arm oneway" in md


class TestReport:
    def test_artifacts(self, tree):
        _, out = tree
        for f in ("report.md", "report.html", "findings.txt",
                  "tables/e2_lb_b0.2.csv", "tables/e3_lb_b0.2.csv",
                  "tables/obs_read_ledger.csv", "tables/e1_recall.csv"):
            assert (out / f).exists(), f
        html = (out / "report.html").read_text()
        assert "data:image/png;base64," in html
        assert "prefers-color-scheme: dark" in html

    def test_findings_cover_every_experiment(self, tree):
        _, out = tree
        f = (out / "findings.txt").read_text()
        assert "E1:" in f and "E2 (LongBench, 20%)" in f and "E3 (RULER, 10%)" in f

    def test_e2_is_paired_and_positive_here(self, tree):
        _, out = tree
        rows = list(csvrows(out / "tables" / "e2_lb_b0.2.csv"))
        macro = next(r for r in rows if r[0] == "**macro**")
        assert macro[3].startswith("+")            # Bidir ahead, by construction
        assert all(r[-1] == "yes" for r in rows if r[0] != "**macro**")

    def test_e3_gap_closure(self, tree):
        _, out = tree
        rows = list(csvrows(out / "tables" / "e3_lb_b0.2.csv"))
        macro = next(r for r in rows if r[0] == "**macro**")
        a, b, c = (float(x) for x in macro[1:4])
        assert b > a > c
        assert macro[-1].endswith("%")
        assert float(macro[-1][:-1]) == pytest.approx(100 * (a - c) / (b - c), abs=1)

    def test_oneway_observation_arm_shows_no_promotions(self, tree):
        _, out = tree
        rows = list(csvrows(out / "tables" / "e2_mechanism.csv"))
        assert rows and all(r[2] == "0.0%" for r in rows)
        md = (out / "report.md").read_text()
        assert "did not take effect" not in md


def csvrows(path: Path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[1:]
