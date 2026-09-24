"""The observation path, as the decode step now runs it: gate, tiers, moves.

Pins the schema-1.3 additions to the parity runners and the two observations
built on them (``modules/evaluation/qevict_observations.py``):

* **Observation IV — the decode read ledger.** Each query head's ground-truth
  attention at each step, split by how the decode read it: exactly (fp, local,
  fresh), at int2 (a window the read gate opened for that head's KV group),
  through the centroid only (an int2 window it skipped), or not at all
  (evicted). Gate recall is per query head and never pooled across the group
  first — the blind spot behind the §12 GQA-union defect.
* **Observation V — tier dynamics.** The F / Q / E chain: promotion, demotion
  and eviction as separate moves (Observation III's binary flips pool Q with E),
  scored against the attention that actually arrives next.

CPU-only. One test drives the real cache through ``flash_decode._run_fused``
(the CPU gate reference, a stubbed decode kernel) and pays the CPU Inductor
compile of the eviction.
"""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

from modules.evaluation import qevict_observations as QO
from utils import qevict_metrics as QM


# ---------------------------------------------------------------------------
# F / Q / E states and transitions
# ---------------------------------------------------------------------------


def _states(rows):
    """``rows``: per event, a string over windows of F/Q/E/-."""
    code = {"F": 0, "Q": 1, "E": 2, "-": -1}
    return np.array([[[code[c] for c in row] for row in rows]], dtype=np.int8)


class TestTierStates:
    def test_assignment(self):
        fp = np.array([[[1, 0, 0, 0]]], bool)
        q = np.array([[[0, 1, 0, 0]]], bool)
        band = np.array([[1, 1, 1, 0]], bool)
        st = QM.tier_states(fp, q, band)
        assert st.tolist() == [[[0, 1, 2, -1]]]

    def test_a_window_in_both_tiers_is_an_error(self):
        both = np.ones((1, 1, 2), bool)
        with pytest.raises(ValueError, match="both fp and Q"):
            QM.tier_states(both, both, np.ones((1, 2), bool))


class TestTierTransitions:
    def test_counts_rates_and_entry(self):
        st = _states(["FQQ-",
                      "QFEF",
                      "QFEE"])
        tr = QM.tier_transitions(st, 1)
        c = tr["pooled_counts"]
        F, Q, E = QM.STATE_F, QM.STATE_Q, QM.STATE_E
        # event 0 -> 1: F->Q, Q->F, Q->E ; event 1 -> 2: Q->Q, F->F, E->E, F->E
        assert c[F, Q] == 1 and c[Q, F] == 1 and c[Q, E] == 1
        assert c[Q, Q] == 1 and c[F, F] == 1 and c[E, E] == 1 and c[F, E] == 1
        assert tr["entry_counts"].sum() == 1          # window 3 enters as F
        assert tr["entry_counts"][0, F] == 1
        assert tr["resurrections"] == 0
        # P(promote | Q) = Q->F / (Q->F + Q->E + Q->Q) = 1/3
        assert tr["rate_promote"][0] == pytest.approx(1 / 3)
        assert tr["rate_demote"][0] == pytest.approx(1 / 3)   # F->Q / (F->Q, F->F, F->E)

    def test_evicted_windows_do_not_dilute_the_promotion_rate(self):
        """The defect this replaces: the binary fp-membership matrix counts an
        evicted window as a promotable 0 at every later event."""
        rows = ["FQE"] + ["QFE"] * 6
        st = _states(rows)
        tr = QM.tier_transitions(st, 1)
        # Q->F happens once (event 0->1 for window 1); after that window 0 is Q
        # and stays Q, window 1 stays F. The E window never enters the Q row.
        assert tr["pooled_counts"][QM.STATE_E, QM.STATE_E] == 6
        sel = np.where(st == QM.STATE_F, 1, np.where(st >= 0, 0, -1))
        binary = QM.binary_transition(sel, 1)["pooled_probabilities"][0, 1]
        assert tr["rate_promote"][0] > binary

    def test_resurrection_is_counted(self):
        st = _states(["FE", "FQ"])
        assert QM.tier_transitions(st, 1)["resurrections"] == 1


class TestTransitionOutcomes:
    def _case(self, promote_hot: bool):
        # 3 events at steps 0, 2, 4; windows: 0,1 start F; 2,3 start Q.
        st = _states(["FFQQ",
                      "FQFQ" if promote_hot else "FQQF",
                      "FQFQ" if promote_hot else "FQQF"])
        T, W = 12, 4
        mass = np.zeros((1, T, W))
        mass[0, :, 0] = 0.2
        mass[0, :, 2] = 0.5          # the hot window
        # Window 1 is demoted in both cases: cold when the promotion is good,
        # hot when it is bad (demoting a window the future attends).
        mass[0, :, 1] = 0.01 if promote_hot else 0.4
        mass[0, :, 3] = 0.01
        return QM.transition_outcomes(st, mass, [0, 2, 4], horizon=4)

    def test_a_good_promotion_has_lift_and_hits(self):
        out = self._case(promote_hot=True)
        assert out["events_scored"] == 2
        assert out["lift_promote"][0] > 1.5
        assert out["lift_demote"][0] < 0.5
        assert out["swap_gain"][0] > 1.0
        assert out["hit_promote"][0] == 1.0
        assert out["hit_demote"][0] == 1.0

    def test_a_bad_promotion_is_scored_as_one(self):
        out = self._case(promote_hot=False)
        assert out["lift_promote"][0] < 1.0
        assert out["hit_promote"][0] == 0.0
        assert out["swap_gain"][0] < 0.0

    def test_events_without_a_full_future_are_censored(self):
        st = _states(["FQ", "QF"])
        out = QM.transition_outcomes(st, np.ones((1, 3, 2)), [0, 2], horizon=4)
        assert out["events_scored"] == 0
        assert np.isnan(out["lift_promote"][0])

    def test_eviction_regret_is_the_future_mass_of_what_was_evicted(self):
        st = _states(["FQQ", "FQE"])
        mass = np.zeros((1, 8, 3))
        mass[0, :, 2] = 1.0          # the future attends the evicted window only
        out = QM.transition_outcomes(st, mass, [0, 1], horizon=3)
        assert out["eviction_regret"][0] == pytest.approx(1.0)
        assert out["hit_evict"][0] == 0.0


class TestDecisionFidelity:
    def test_matching_and_swapped_rankings(self):
        st = _states(["FQQE"])
        good = np.array([[[3.0, 2.0, 1.0, 9.0]]])   # E window ignored
        bad = np.array([[[1.0, 3.0, 2.0, 9.0]]])
        assert QM.tier_decision_fidelity(st, good)[0, 0] == 1.0
        assert QM.tier_decision_fidelity(st, bad)[0, 0] == 0.0


# ---------------------------------------------------------------------------
# The read ledger
# ---------------------------------------------------------------------------


class TestSurvivorToBase:
    def test_evicted_vs_fresh_and_the_gate_scatter(self):
        ids = np.array([[0, 2, 3, 5, -1]])
        tier = np.array([[0, 1, 1, 2, -1]])
        gate = np.zeros((1, 2, 5), bool)
        gate[0, 0, 1] = True           # kv0 opened window 2
        gate[0, 1, 2] = True           # kv1 opened window 3
        cls, read, oob = QM.survivor_to_base(ids, tier, 8, gate)
        assert cls[0].tolist() == [QM.CLS_FP, QM.CLS_EVICTED, QM.CLS_Q, QM.CLS_Q,
                                   QM.CLS_EVICTED, QM.CLS_LOCAL, QM.CLS_FRESH,
                                   QM.CLS_FRESH]
        assert read[0, 0].nonzero()[0].tolist() == [2]
        assert read[0, 1].nonzero()[0].tolist() == [3]
        assert oob == 0

    def test_out_of_range_ids_are_counted_not_scattered(self):
        _, _, oob = QM.survivor_to_base(np.array([[0, 9]]), np.array([[0, 0]]), 4)
        assert oob == 1


def _ledger_case():
    """2 query heads per KV head. Q windows 1 and 2; kv0 opened 1, kv1 opened 2."""
    T, H, W = 2, 4, 4
    cls = np.array([[QM.CLS_FRESH] * 4,
                    [QM.CLS_FP, QM.CLS_Q, QM.CLS_Q, QM.CLS_EVICTED]])
    read = np.zeros((T, 2, W), bool)
    read[1, 0, 1] = True
    read[1, 1, 2] = True
    m = np.zeros((T, H, W))
    m[1, 0] = [0.1, 0.6, 0.1, 0.1]    # kv0, mostly on its opened window
    m[1, 1] = [0.0, 0.0, 0.01, 0.0]   # kv0, all int2 mass on the SKIPPED one
    m[1, 2] = [0.2, 0.0, 0.5, 0.0]    # kv1, on its opened window
    m[1, 3] = [0.0, 0.3, 0.3, 0.3]    # kv1, half on a skipped window
    return m, cls, read, np.array([False, True])


class TestDecodeReadLedger:
    def test_parts_sum_to_one_per_head(self):
        m, cls, read, fired = _ledger_case()
        led = QM.decode_read_ledger(m, cls, read, fired, rep=2)
        assert led["parts"].shape == (1, 4, len(QM.LEDGER_PARTS))
        np.testing.assert_allclose(led["parts"].sum(-1), 1.0)
        p = dict(zip(QM.LEDGER_PARTS, led["parts"][0, 0]))
        assert p["q_read"] == pytest.approx(0.6)
        assert p["q_skipped"] == pytest.approx(0.1)
        assert p["evicted"] == pytest.approx(0.1)
        assert p["sink"] == pytest.approx(0.1)

    def test_query_heads_read_their_own_kv_group(self):
        m, cls, read, fired = _ledger_case()
        led = QM.decode_read_ledger(m, cls, read, fired, rep=2)
        r = QM.ledger_head_recall(led["parts"], led["gated"])["recall"]
        np.testing.assert_allclose(r, [0.6 / 0.7, 0.0, 1.0, 0.5])

    def test_recall_is_per_head_not_pooled_over_the_group(self):
        """Head 1 puts only 1% of its mass on int2 and the gate skips all of it.
        Pooling the group's raw mass reads ~0.86 for kv0; per head, head 1 got
        none of its int2 mass read. That is the §12 blind spot."""
        m, cls, read, fired = _ledger_case()
        led = QM.decode_read_ledger(m, cls, read, fired, rep=2)
        parts = led["parts"][0]
        qr = parts[:, QM.LEDGER_PARTS.index("q_read")]
        qs = parts[:, QM.LEDGER_PARTS.index("q_skipped")]
        pooled_kv0 = qr[:2].sum() / (qr[:2] + qs[:2]).sum()
        assert pooled_kv0 > 0.8
        assert QM.ledger_head_recall(led["parts"], led["gated"])["recall"][1] == 0.0

    def test_ungated_steps_read_the_whole_tier(self):
        m, cls, read, _ = _ledger_case()
        led = QM.decode_read_ledger(m, cls, read, np.array([False, False]), rep=2)
        assert np.all(led["parts"][..., QM.LEDGER_PARTS.index("q_skipped")] == 0)
        assert np.all(np.isnan(led["read_frac"]))

    def test_read_fraction_and_the_hindsight_ceiling(self):
        m, cls, read, fired = _ledger_case()
        led = QM.decode_read_ledger(m, cls, read, fired, rep=2)
        np.testing.assert_allclose(led["read_frac"][0], [0.5, 0.5])
        hr = QM.ledger_head_recall(led["parts"], led["gated"], led["oracle_q_read"])
        # The oracle picks by the group's summed per-head share: kv0 heads are
        # (6/7, 1/7) and (0, 1) on windows (1, 2) -> window 2 wins (8/7 vs 6/7).
        np.testing.assert_allclose(hr["oracle_recall"][:2], [0.1 / 0.7, 1.0])

    def test_the_oracle_never_loses_to_the_gate_on_the_group_objective(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            T, H, W = 4, 4, 10
            m = rng.dirichlet(np.ones(W + 1), size=(T, H))[..., :W]
            cls = rng.choice([QM.CLS_FP, QM.CLS_Q, QM.CLS_EVICTED], size=(T, W))
            read = rng.random((T, 2, W)) < 0.4
            led = QM.decode_read_ledger(m, cls, read, np.ones(T, bool), rep=2,
                                        first_step=0)
            share = lambda x: x / np.maximum(led["q_mass"], 1e-30)
            qr = share(led["parts"][..., QM.LEDGER_PARTS.index("q_read")])
            orc = share(np.nan_to_num(led["oracle_q_read"]))
            for kv in range(2):
                hs = slice(2 * kv, 2 * kv + 2)
                assert np.all(orc[:, hs].sum(1) >= qr[:, hs].sum(1) - 1e-9)

    def test_min_q_share_drops_heads_with_no_int2_mass(self):
        m, cls, read, fired = _ledger_case()
        led = QM.decode_read_ledger(m, cls, read, fired, rep=2)
        r = QM.ledger_head_recall(led["parts"], led["gated"], min_q_share=0.05)
        assert np.isnan(r["recall"][1])


# ---------------------------------------------------------------------------
# A simulated three-tier cache with permanent eviction and a per-KV-head gate
# ---------------------------------------------------------------------------


def _sim_pair(*, T=48, L=2, H=4, H_kv=2, prefill_len=8, local=2, k_fp=3, n_q=4,
              evict_every=4, gate=True, bad_kv=1, record_heads=True):
    """``window_size = 1`` so a window is a token; no sink tokens.

    Heads of KV group 0 follow a drifting hot window; group 1 attends a second
    one. The gate opens half the Q tier per KV head, by true mass for group 0
    and by INVERTED mass for group ``bad_kv`` — so that group's recall is low
    and the other's is high, which the per-head numbers must show.
    """
    rng = np.random.default_rng(1)
    S, W = 1, prefill_len + T
    w_act = np.minimum(prefill_len + np.arange(T), W)
    ew = np.maximum(w_act - local, 0)
    step = np.zeros((S, T, L, H, W))
    for t in range(T):
        n = w_act[t]
        step[:, t, :, :, :n] = 0.02 * rng.random((L, H, n))
        a = (t // 6) % max(int(ew[t]), 1)
        b = (t // 9 + 3) % max(int(ew[t]), 1)
        step[:, t, :, :2, a] += 0.5
        step[:, t, :, 2:, b] += 0.5
        step[:, t, :, :, n - 1] += 0.2
    step /= step.sum(-1, keepdims=True) / 0.8          # sink keeps 0.2
    cum = np.cumsum(step, axis=1)
    ev_mask = np.zeros((S, T), bool)
    ev_mask[0, 1::evict_every] = True
    rep = H // H_kv
    Wo = W
    ids = np.full((S, T, L, Wo), -1, np.int64)
    tier = np.full((S, T, L, Wo), -1, np.int64)
    gread = np.zeros((S, T, L, H_kv, Wo), bool)
    gfired = np.zeros((S, T, L), bool)
    for li in range(L):
        evicted, q_set = set(), set()
        for t in range(T):
            if ev_mask[0, t]:
                band = [w for w in range(ew[t]) if w not in evicted]
                score = cum[0, t, li].mean(0)
                order = sorted(band, key=lambda w: -score[w])
                q_set = set(order[k_fp:k_fp + n_q])
                evicted |= set(order[k_fp + n_q:])
            surv = [w for w in range(w_act[t]) if w not in evicted]
            ids[0, t, li, :len(surv)] = surv
            tags = [1 if w in q_set else (2 if w >= ew[t] else 0) for w in surv]
            tier[0, t, li, :len(surv)] = tags
            if gate and q_set and t >= 1:
                gfired[0, t, li] = True
                qs = sorted(q_set)
                n_sel = math.ceil(0.5 * len(qs))
                for kv in range(H_kv):
                    heads = slice(kv * rep, (kv + 1) * rep)
                    mass = step[0, t, li, heads][:, qs].sum(0)
                    pick = np.argsort(mass if kv == bad_kv else -mass)[:n_sel]
                    chosen = {qs[i] for i in pick}
                    gread[0, t, li, kv, :len(surv)] = [w in chosen for w in surv]
    base_arrays = {
        "window_scores": cum.astype(np.float16),
        "step_window_scores": step.mean(3).astype(np.float32),
        "top_window_indices": np.zeros((S, T, L, k_fp), np.int64),
        "eviction_step_mask": np.zeros((S, T), bool),
        "generated_tokens": np.zeros((S, T), np.int64),
    }
    if record_heads:
        base_arrays["step_window_scores_heads"] = step.astype(np.float16)
    meta = {"mode": "parity_base", "schema_version": "1.3", "seed": 42,
            "window_size": 1, "num_sink_tokens": 0, "prefill_len": prefill_len,
            "gen_len": T, "local_window_size_resolved": local,
            "top_k_windows": k_fp + n_q, "model_name": "synthetic",
            "num_samples": S, "score_accum_dtype": "float32",
            "num_attention_heads": H, "num_key_value_heads": H_kv}
    ours_arrays = {
        "window_scores": cum.astype(np.float16),
        "top_window_indices": np.zeros((S, T, L, k_fp), np.int64),
        "eviction_step_mask": ev_mask,
        "all_window_ids": ids, "all_window_tier": tier,
        "retained_window_ids": ids,
        "retained_window_scores": np.zeros((S, T, L, H, Wo), np.float16),
    }
    ours_meta = dict(meta, mode="parity_ours", top_k_fp=k_fp, N_q=n_q,
                     quant_ratio=0.7, cache_budget=0.2)
    if gate:
        ours_arrays["gate_read"] = gread
        ours_arrays["gate_fired"] = gfired
        ours_meta.update(read_gate={"verdict": "gated", "read_fraction": 0.5,
                                    "gated": 10, "fired": 10},
                         gate_recorded=True, quant_gate_ratio=0.5)
    else:
        ours_meta.update(read_gate={"verdict": "not-expected"},
                         gate_recorded=False)
    return ({"arrays": base_arrays, "metadata": meta, "path": "b"},
            {"arrays": ours_arrays, "metadata": ours_meta, "path": "o"})


class TestObservationIV:
    def test_per_head_recall_separates_the_two_kv_groups(self):
        base, ours = _sim_pair()
        inp = QO.build_observation_inputs(base, ours)
        assert inp.geometry["read_path"] == "gated"
        obs4 = QO.analyse_decode_reads(base, ours, inp, n_boot=50)
        hr = obs4["head_recall"]
        good, bad = QM.nanmean(hr[:, :2]), QM.nanmean(hr[:, 2:])
        # Group 0's int2 mass is diffuse, so half the windows hold ~70% of it;
        # group 1's gate picks the LEAST massive half.
        assert good > bad + 0.4 and bad < 0.3
        rs = obs4["recall_summary"]
        assert rs["gated"] and rs["head_mass_source"] == "recorded_head_step_mass"
        assert rs["realised_read_fraction"] == pytest.approx(0.5, abs=0.15)
        assert rs["oracle_recall_mean"] >= rs["head_recall_mean"]
        assert rs["heads_below_0.50"] >= 0.4
        parts = {r["part"]: r["share_of_head_mass"] for r in obs4["ledger_table"]}
        total = sum(parts[p] for p in QM.LEDGER_PARTS)
        assert total == pytest.approx(1.0, abs=1e-6)
        assert parts["q_skipped"] > 0 and parts["evicted"] > 0
        assert parts["decode_missed_mass"] == pytest.approx(
            parts["q_skipped"] + parts["evicted"])

    def test_an_ungated_run_reads_the_whole_tier(self):
        base, ours = _sim_pair(gate=False)
        inp = QO.build_observation_inputs(base, ours)
        assert inp.geometry["read_path"] == "ungated"
        obs4 = QO.analyse_decode_reads(base, ours, inp, n_boot=50)
        parts = {r["part"]: r["share_of_head_mass"] for r in obs4["ledger_table"]}
        assert parts["q_skipped"] == 0
        assert not obs4["recall_summary"]["gated"]

    def test_without_per_head_mass_it_differences_and_says_so(self):
        base, ours = _sim_pair(record_heads=False)
        inp = QO.build_observation_inputs(base, ours)
        obs4 = QO.analyse_decode_reads(base, ours, inp, n_boot=50)
        assert obs4["recall_summary"]["head_mass_source"] == "differenced_cumulative"


class TestObservationV:
    def test_the_simulated_cache_never_resurrects(self):
        base, ours = _sim_pair()
        inp = QO.build_observation_inputs(base, ours)
        assert inp.diagnostics["tier_resurrections"] == 0
        assert set(np.unique(inp.states)) <= {-1, 0, 1, 2}

    def test_rates_outcomes_and_fidelity(self):
        base, ours = _sim_pair()
        inp = QO.build_observation_inputs(base, ours)
        obs5 = QO.analyse_tier_dynamics(inp, horizon=4, deltas=(1, 2), n_boot=50)
        moves = {r["move"] for r in obs5["rate_table"]}
        assert moves == set(QM.MOVES)
        for r in obs5["rate_table"]:
            assert np.isnan(r["rate_mean"]) or 0.0 <= r["rate_mean"] <= 1.0
        # Rows of the pooled lag-1 matrix are distributions (E is absorbing).
        pooled = obs5["pooled_transition"]
        np.testing.assert_allclose(np.nansum(pooled, axis=1)[~np.isnan(pooled).all(1)], 1)
        assert pooled[QM.STATE_E, QM.STATE_E] == 1.0
        sm = obs5["summary"]
        assert sm["events_scored"] > 0
        assert 0.0 <= sm["decision_fidelity_mean"] <= 1.0
        assert {r["move"] for r in obs5["outcome_table"]} == set(QM.MOVES)


class TestReadPathInfo:
    @pytest.mark.parametrize("meta,path", [
        ({"read_gate": {"verdict": "gated"}, "gate_recorded": True}, "gated"),
        ({"read_gate": {"verdict": "gated"}}, "gated-unrecorded"),
        ({"read_gate": {"verdict": "not-expected"}}, "ungated"),
        ({"read_gate": {"verdict": "NOT-GATED-partial"}}, "NOT-GATED"),
        ({"read_gate": {"verdict": "NOT-GATED-kernel-never-fired"}}, "NOT-GATED"),
        ({}, "unknown"),
    ])
    def test_verdicts(self, meta, path):
        assert QO.read_path_info(meta)["read_path"] == path


class TestEndToEnd:
    def test_writes_observations_four_and_five(self, tmp_path):
        base, ours = _sim_pair()
        paths = []
        for name, d in (("base", base), ("ours", ours)):
            p = tmp_path / f"parity_{name}.npz"
            np.savez_compressed(str(p), **d["arrays"], metadata_json=np.array(
                [json.dumps(d["metadata"])], dtype=object))
            paths.append(p)
        res = QO.run_observations(paths[0], paths[1], tmp_path / "out",
                                  fmm_horizon=4, n_boot=50, figures=False,
                                  primary_inactivity=2, primary_lir_horizon=2)
        out = tmp_path / "out"
        for f in ("observation4_ledger_table.csv", "observation4_recall_summary.csv",
                  "observation4_layer_table.csv", "observation5_rate_table.csv",
                  "observation5_outcome_table.csv", "observation5_summary.csv"):
            assert (out / f).exists(), f
        md = (out / "paper_results.md").read_text()
        assert "Observation IV" in md and "Observation V" in md
        assert "Read path: **gated**" in md
        assert "pools Q with evicted" in md          # the policy flips, labelled
        js = json.loads((out / "all_results.json").read_text())
        assert js["metadata"]["read_path"] == "gated"
        assert "observation4" in js and "observation5" in js
        z = np.load(res["npz_path"], allow_pickle=True)
        assert {"ledger_by_trace", "head_recall", "tier_states",
                "transition_counts"} <= set(z.files)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


class TestRunnerHelpers:
    def test_gate_read_row(self):
        from modules.evaluation.ours_parity_runner import _gate_read_row
        ids = np.array([0, 3, 5, 7, -1])
        picked = np.array([[3, 7], [5, 5]])
        row = _gate_read_row(ids, picked, 2)
        assert row.tolist() == [[False, True, False, True, False],
                                [False, False, True, False, False]]
        assert not _gate_read_row(ids, None, 2).any()

    def test_select_parity_articles_skips_short_ones(self):
        from modules.evaluation.base_parity_runner import select_parity_articles
        tok = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=True: text.split())
        arts = ["a b", "a b c d", "x", "a b c d e", "a b c d"]
        assert select_parity_articles(arts, tok, 0, 2, 4) == [1, 3]
        assert select_parity_articles(arts, tok, 2, 5, 4) == [3, 4]
        from utils.config import ParityValidationError
        with pytest.raises(ParityValidationError):
            select_parity_articles(arts, tok, 0, 1, 10)


class TestCorpusLoaderKnobs:
    def _jsonl(self, tmp_path):
        p = tmp_path / "lb.jsonl"
        rows = [{"input": "q1", "context": "long doc one", "task": "a"},
                {"input": "q2", "context": "long doc two", "task": "b"}]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def test_longbench_needs_the_context_field(self, tmp_path):
        from data.corpus_loader import CorpusLoader
        p = self._jsonl(tmp_path)
        assert CorpusLoader(str(p)).load() == ["q1", "q2"]    # the trap
        assert CorpusLoader(str(p), text_field="context").load() == [
            "long doc one", "long doc two"]

    def test_record_filter_and_slug(self, tmp_path):
        from data.corpus_loader import CorpusLoader
        p = self._jsonl(tmp_path)
        ld = CorpusLoader(str(p), text_field="context", record_filter="task=b")
        assert ld.load() == ["long doc two"]
        assert ld.slug == "lb-b"
        with pytest.raises(ValueError, match="key=value"):
            CorpusLoader(str(p), record_filter="task")


# ---------------------------------------------------------------------------
# The observer, on the real cache
# ---------------------------------------------------------------------------


class TestGateObserver:
    def test_set_clear_and_refuse_a_second(self):
        from modules.windowed_cache import flash_decode
        try:
            flash_decode.set_gate_observer(lambda *a: None)
            with pytest.raises(RuntimeError, match="already installed"):
                flash_decode.set_gate_observer(lambda *a: None)
        finally:
            flash_decode.clear_gate_observer()
        assert flash_decode._OBSERVER["fn"] is None

    def test_the_recorder_names_active_int2_windows_on_the_real_cache(self, monkeypatch):
        """Drive the production cache through ``_run_fused`` (CPU gate
        reference, stubbed decode kernel) on both the per-layer and the
        layer-major path, and check every recorded pick is ``n_sel`` distinct
        ACTIVE int2 window ids per KV head — i.e. that ``sel`` was mapped
        through the same ``active_order`` the gate scored."""
        import torch
        from modules.evaluation.ours_parity_runner import GateRecorder
        from modules.windowed_cache import flash_decode
        from modules.windowed_cache.cache import WindowedCache
        from modules.windowed_cache.config import WindowedCacheConfig

        class _Rope(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = d

            def forward(self, x, position_ids):
                f = position_ids.to(torch.float32).unsqueeze(-1) * torch.arange(
                    1, self.d // 2 + 1, dtype=torch.float32) * 0.01
                e = torch.cat([f, f], dim=-1)
                return e.cos(), e.sin()

        L, Bt, h_kv, h_q, d, ws, P = 2, 2, 2, 4, 16, 4, 32
        cfg = types.SimpleNamespace(num_key_value_heads=h_kv,
                                    num_attention_heads=h_q,
                                    hidden_size=h_q * d, head_dim=d)
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=ws, num_sink_tokens=0, local_window_size=ws,
                cache_budget=0.35, quant_ratio=0.5, first_eviction_step=0,
                quant_gate_ratio=0.5),
            prefill_len=P, model_config=cfg, kv_dtype=torch.float32,
            rope_module=_Rope(d), num_layers=L, max_tokens=P + 64)
        cache._fused_decode_active = True
        armed = []

        def stub_decode(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win,
                        *, sel, logmass, centroids, vm_bits: int = 4):
            w = n_body_win + qtier["k_codes"].shape[1]
            return (torch.zeros_like(q_hd),
                    torch.rand(q_hd.shape[0], q_hd.shape[1], w))

        monkeypatch.setattr(flash_decode, "set_pending", armed.append)
        monkeypatch.setattr(flash_decode, "fused_two_tier_decode", stub_decode)
        rec = GateRecorder()
        rec.bind(cache)
        flash_decode.set_gate_observer(rec)

        def merged(i):
            st, store = cache._states[i], cache._stores[i]
            return -(-st.seq_length // ws) + store.num_active_windows

        g = torch.Generator().manual_seed(0)
        checked = joint_checked = 0
        try:
            for i in range(L):
                k = torch.randn(Bt, h_kv, P, d, generator=g)
                cache.update(k, torch.randn_like(k), i,
                             cache_kwargs={"cache_position": torch.arange(P)})
            for i in range(L):
                cache.cache_kwargs[i]["window_scores"] = torch.rand(
                    Bt, h_q, merged(i), generator=g)
            for t in range(3):
                rec.clear()
                for i in range(L):
                    k = torch.randn(Bt, h_kv, 1, d, generator=g)
                    k_fp, v_fp = cache.update(
                        k, torch.randn_like(k), i,
                        cache_kwargs={"cache_position": torch.tensor([P + t])})
                    if not armed:
                        cache.cache_kwargs[i]["window_scores"] = torch.rand(
                            Bt, h_q, merged(i), generator=g)
                        continue
                    ctx = armed.pop()
                    q = torch.randn(Bt, 1, h_q, d, generator=g)
                    flash_decode._run_fused(ctx, q, k_fp.transpose(1, 2),
                                            v_fp.transpose(1, 2))
                    active = cache._stores[i].active_ids()
                    n = active.shape[1]
                    pick = rec.host()[i]
                    assert pick.shape == (Bt, h_kv, max(1, math.ceil(0.5 * n)))
                    for b in range(Bt):
                        act = set(active[b].tolist())
                        for kv in range(h_kv):
                            ids = pick[b, kv].tolist()
                            assert len(set(ids)) == len(ids)
                            assert set(ids) <= act
                    checked += 1
                    joint_checked += cache._joint is not None
        finally:
            flash_decode.clear_gate_observer()
        assert checked > 0 and joint_checked > 0, "no gated step was observed"
