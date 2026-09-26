"""The promotion ablation (modules/evaluation/promotion_ablation.py).

The no-promotion column of its Table 18 is a REPLAY of the one-way policy on a
bidirectional run's recorded ranking signal, and a re-run of the read gate on
the run's recorded card estimates. Both are only as good as their fidelity to
the real rules, so both are pinned against the real code, not a copy of it:

1. The tier replay against ``EvictionPolicy.compute_two_tier_retain`` -- the
   function the cache's eviction calls -- driven over a growing window axis at
   the collector's own geometry. Replaying ``bidir`` reproduces a ``bidir``
   run; replaying ``oneway`` on that same run reproduces a ``oneway`` run of the
   policy (given scores for every window it keeps), and never promotes.
2. The gate rule against ``sketch.group_share`` + ``select_windows``.
3. The whole thing end to end on a synthetic collector zip: the checks come
   out clean, the one-way arm never promotes and has no fp revivals, and the
   table and report are written.
"""

from __future__ import annotations

import json
import math
import types

import numpy as np
import pytest
import torch

from modules.evaluation import promotion_ablation as PA
from modules.evaluation.observation_collector import ZipWriter
from modules.quant.sketch import group_share, select_windows
from modules.windowed_cache.policy import EvictionPolicy
from utils.sticky_metrics import flush_geometry

P, NS, WS, GEN = 16, 0, 2, 40
T = GEN + 1
W = math.ceil((P + GEN - NS) / WS)
LOCAL_W, K_FP, N_Q = 2, 2, 5
H, HK = 4, 2


def _events():
    return np.arange(1, T, WS)


def _simulate(scores: np.ndarray, oneway: bool, full_scores: bool, L: int = 1):
    """A run of the real policy: ``(tier [T,L,W], evict_step [T],
    rank_signal [R,L,H,W], rank_steps [R])`` with collector tier codes.

    ``scores`` ``[L, R, H, W]`` is what each eviction ranks. The recorded rank
    signal holds it on the merged axis (``full_scores=False``, as the collector
    records it) or on every window created so far (``True``).
    """
    pol = EvictionPolicy(types.SimpleNamespace(
        window_size=WS, num_sink_tokens=NS, local_tokens=LOCAL_W * WS,
        top_k_windows=K_FP + N_Q, quant_ratio=0.5, top_k_fp=K_FP, N_q=N_Q,
        first_eviction_step=0))
    w_act, _, _ = flush_geometry(T, W, P, NS, WS, LOCAL_W)
    ev = _events()
    tier = np.full((T, L, W), -1, np.int8)
    rank = np.full((ev.size, L, H, W), np.nan, np.float32)
    for li in range(L):
        state = np.full(W, -1)            # -1 none, 0 fp, 1 int2, 2 undecided, 3 evicted
        r = 0
        for t in range(T):
            made = state[:w_act[t]]
            made[made == -1] = 2
            if r < ev.size and t == ev[r]:
                merged = np.flatnonzero(np.isin(state, (0, 1, 2)))
                sc = torch.from_numpy(scores[li, r][:, merged].copy())[None]
                block = (torch.from_numpy(state[merged] == 1)[None] if oneway else None)
                idx, tr = pol.compute_two_tier_retain(sc, block)
                idx, tr = idx[0].numpy(), tr[0].numpy()
                lw = min(LOCAL_W, merged.size)
                band = merged[:merged.size - lw]
                new = state.copy()
                new[band] = 3
                for j, code in zip(idx, tr):
                    if j < merged.size - lw:
                        new[merged[j]] = code
                seen = np.arange(w_act[t]) if full_scores else merged
                rank[r, li][:, seen] = scores[li, r][:, seen]
                state = new
                r += 1
            merged = np.flatnonzero(np.isin(state, (0, 1, 2)))
            row = np.full(W, -1, np.int8)
            row[(state == 0) | (state == 2)] = 0
            row[state == 1] = 1
            row[state == 3] = 3
            row[merged[-LOCAL_W:]] = 2
            tier[t, li] = row
    evict = np.zeros(T, bool)
    evict[ev] = True
    return tier, evict, rank, ev.astype(np.int32)


def _scores(seed: int, L: int = 1) -> np.ndarray:
    """Fresh ranking at every eviction, so a bidir run promotes often."""
    return np.random.default_rng(seed).random((L, _events().size, H, W)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. the tier replay is the policy
# ---------------------------------------------------------------------------


class TestReplayIsThePolicy:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_replaying_bidir_reproduces_a_bidir_run(self, seed):
        tier, evict, rank, steps = _simulate(_scores(seed), oneway=False,
                                             full_scores=False)
        out, diag = PA.replay_tiers(tier, evict, rank, steps, K_FP, N_Q, LOCAL_W,
                                    oneway=False)
        np.testing.assert_array_equal(out, tier)
        assert diag["unscored"] == 0 and diag["kept_differs"] == 0
        assert diag["promotions"] > 0, "no promotion happened -- nothing checked"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_replaying_oneway_on_a_bidir_run_is_a_oneway_run(self, seed):
        s = _scores(seed)
        tier_b, evict, rank_b, steps = _simulate(s, oneway=False, full_scores=True)
        tier_o, _, _, _ = _simulate(s, oneway=True, full_scores=True)
        out, diag = PA.replay_tiers(tier_b, evict, rank_b, steps, K_FP, N_Q, LOCAL_W,
                                    oneway=True)
        np.testing.assert_array_equal(out, tier_o)
        assert diag["promotions"] == 0 and diag["unscored"] == 0
        assert diag["fp_differs"] > 0, "one-way never diverged -- nothing checked"

    def test_oneway_never_promotes_and_keeps_the_tier_sizes(self):
        tier, evict, rank, steps = _simulate(_scores(3), oneway=False, full_scores=False)
        out, diag = PA.replay_tiers(tier, evict, rank, steps, K_FP, N_Q, LOCAL_W,
                                    oneway=True)
        ev = np.flatnonzero(evict)
        for a, b in zip(ev[:-1], ev[1:]):
            prev, cur = out[a, 0], out[b, 0]
            assert not ((prev == PA.TIER_Q) & (cur == PA.TIER_FP)).any()
        for e in ev[1:]:
            assert ((out[e, 0] == PA.TIER_FP).sum(), (out[e, 0] == PA.TIER_Q).sum()) == \
                ((tier[e, 0] == PA.TIER_FP).sum(), (tier[e, 0] == PA.TIER_Q).sum())

    def test_a_window_without_a_score_is_counted_and_ranked_last(self):
        """Dropping it keeps a window the run had evicted, which has no score
        at the next eviction either -- so the replay diverges from there on,
        and every such ranking is counted rather than guessed."""
        tier, evict, rank, steps = _simulate(_scores(4), oneway=False, full_scores=False)
        ev = np.flatnonzero(evict)
        kept = np.flatnonzero(tier[ev[3], 0] == PA.TIER_Q)
        rank = rank.copy()
        rank[4, 0][:, kept[0]] = np.nan
        out, diag = PA.replay_tiers(tier, evict, rank, steps, K_FP, N_Q, LOCAL_W,
                                    oneway=False)
        assert diag["unscored"] >= 1 and diag["kept_differs"] >= 1
        assert out[ev[4], 0, kept[0]] == PA.TIER_EVICTED
        np.testing.assert_array_equal(out[:ev[4]], tier[:ev[4]])

    def test_mismatched_rank_steps_raise(self):
        tier, evict, rank, steps = _simulate(_scores(0), oneway=False, full_scores=False)
        with pytest.raises(ValueError, match="rank_signal_steps"):
            PA.replay_tiers(tier, evict, rank, steps + 1, K_FP, N_Q, LOCAL_W,
                            oneway=False)


# ---------------------------------------------------------------------------
# 2. the gate rule is sketch's
# ---------------------------------------------------------------------------


class TestGateRule:
    def test_gate_pick_is_group_share_then_select_windows(self):
        g = np.random.default_rng(0)
        N, Wn, n_sel = 50, 20, 4
        card = (g.normal(size=(N, H, Wn)) * 3).astype(np.float32)
        known = g.random((N, Wn)) < 0.6
        known[:, :n_sel + 1] = True
        pick = PA.gate_pick(card, known, np.zeros_like(known), np.full(N, n_sel), HK)
        for i in range(N):
            k = np.flatnonzero(known[i])
            est = group_share(torch.from_numpy(card[i][:, k])[None], HK)
            ref = select_windows(est, n_sel)[0].numpy()               # [HK, nk]
            np.testing.assert_array_equal(pick[i][:, k], ref)
            assert not pick[i][:, ~known[i]].any()

    def test_forced_windows_open_for_every_kv_head_inside_the_budget(self):
        g = np.random.default_rng(1)
        N, Wn, n_sel = 30, 12, 3
        card = g.normal(size=(N, H, Wn)).astype(np.float32)
        known = np.ones((N, Wn), bool)
        known[:, 0] = False
        forced = np.zeros((N, Wn), bool)
        forced[:, 0] = True
        pick = PA.gate_pick(card, known, forced, np.full(N, n_sel), HK)
        assert pick[:, :, 0].all()
        assert (pick.sum(-1) == n_sel).all()

    def test_replay_gate_keeps_the_record_where_the_tiers_agree(self):
        g = np.random.default_rng(2)
        Tn, L, Wn = 6, 1, 10
        tier = np.full((Tn, L, Wn), PA.TIER_Q, np.int8)
        tier[:, :, :2] = PA.TIER_FP
        arm = tier.copy()
        arm[3:, 0, 0], arm[3:, 0, 5] = PA.TIER_Q, PA.TIER_FP      # a promotion undone
        card = g.normal(size=(Tn, L, H, Wn)).astype(np.float32)
        card[tier[:, :, None, :].repeat(H, 2) != PA.TIER_Q] = np.nan
        fired = np.ones((Tn, L), bool)
        rec = np.zeros((Tn, L, HK, Wn), bool)
        for t in range(Tn):
            rec[t, 0] = PA.gate_pick(card[t, 0][None], (tier[t, 0] == PA.TIER_Q)[None],
                                     np.zeros((1, Wn), bool), np.array([3]), HK)[0]
        assert PA.gate_check(tier, rec, fired, card) == (Tn * HK, Tn * HK)
        for mode, opened in (("open", True), ("closed", False)):
            out, diag = PA.replay_gate(tier, arm, rec, fired, card, mode)
            np.testing.assert_array_equal(out[:3], rec[:3])
            assert (out[3:, 0, :, 0] == opened).all()
            assert not out[3:, 0, :, 5].any()                       # now fp
            assert (out[3:, 0].sum(-1) == 3).all()
            assert diag == {"rerun_rows": 3, "blocked_windows": 3, "over_budget_rows": 0}
        with pytest.raises(ValueError):
            PA.replay_gate(tier, arm, rec, fired, card, "half")

    def test_open_fills_an_overflowing_gate_with_the_heaviest_blocked_windows(self):
        """Two windows without a card, one slot per KV head: ``open`` gives the
        slot to whichever holds more of that group's FullKV int2 mass."""
        Tn, L, Wn = 2, 1, 8
        tier = np.full((Tn, L, Wn), PA.TIER_Q, np.int8)
        tier[:, :, :2] = PA.TIER_FP
        arm = tier.copy()
        arm[1, 0, :2], arm[1, 0, 2:4] = PA.TIER_Q, PA.TIER_FP
        card = np.zeros((Tn, L, H, Wn), np.float32)
        fired = np.ones((Tn, L), bool)
        rec = np.zeros((Tn, L, HK, Wn), bool)
        rec[:, 0, :, 4] = True
        fm = np.full((Tn, L, H, Wn), 0.01, np.float32)
        fm[1, 0, :2, 0] = 0.5                      # KV head 0's group wants window 0
        fm[1, 0, 2:, 1] = 0.5                      # KV head 1's group wants window 1
        with pytest.raises(ValueError, match="full_mass"):
            PA.replay_gate(tier, arm, rec, fired, card, "open")
        out, diag = PA.replay_gate(tier, arm, rec, fired, card, "open", fm)
        assert diag["over_budget_rows"] == 1
        assert out[1, 0, 0].tolist() == [True] + [False] * 7
        assert out[1, 0, 1].tolist() == [False, True] + [False] * 6
        closed, _ = PA.replay_gate(tier, arm, rec, fired, card, "closed", fm)
        assert not closed[1, 0, :, :2].any() and (closed[1, 0].sum(-1) == 1).all()


# ---------------------------------------------------------------------------
# small metric pieces
# ---------------------------------------------------------------------------


def test_swap_mass_counts_each_promotion_against_the_window_it_displaced():
    Tn, Wn = 12, 4
    tier = np.full((Tn, 1, Wn), PA.TIER_Q, np.int8)
    tier[:, 0, 0] = PA.TIER_FP
    tier[5:, 0, 0], tier[5:, 0, 1] = PA.TIER_Q, PA.TIER_FP          # 1 promoted, 0 displaced
    evict = np.zeros(Tn, bool)
    evict[[1, 5]] = True
    fm = np.zeros((Tn, 1, 2, Wn), np.float16)
    fm[:, 0, :, 1], fm[:, 0, :, 0] = 0.3, 0.1
    out = PA.swap_mass(tier, evict, fm, horizon=4)
    np.testing.assert_allclose([out["promoted_sum"][0], out["demoted_sum"][0]],
                               [0.3, 0.1], rtol=1e-3)
    assert (out["promoted_n"][0], out["demoted_n"][0]) == (1, 1)
    assert PA.swap_mass(tier, evict, fm, horizon=7)["promoted_n"][0] == 0   # censored


def test_score_agreement_is_one_when_the_cache_ranks_on_the_truth():
    tier, evict, _, _ = _simulate(_scores(5), oneway=False, full_scores=False)
    g = np.random.default_rng(5)
    fm = g.random((T, 1, H, W)).astype(np.float16)
    cum = np.cumsum(fm.astype(np.float32).mean(2), 0)                # [T, 1, W]
    ev = np.flatnonzero(evict)
    rank = np.stack([np.repeat(cum[e - 1][:, None, :], H, 1) for e in ev])
    assert PA.score_agreement(tier, evict, rank, fm)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. end to end, on a synthetic collector zip
# ---------------------------------------------------------------------------


def _synthetic_zip(tmp_path, n_samples=2, L=2, n_sel=2):
    g = np.random.default_rng(7)
    w = ZipWriter(tmp_path / "run.zip")
    w_act, _, _ = flush_geometry(T, W, P, NS, WS, LOCAL_W)
    samples = []
    for s in range(n_samples):
        tier, evict, rank, steps = _simulate(_scores(10 + s, L), oneway=False,
                                             full_scores=False, L=L)
        fm = np.zeros((T, L, H, W), np.float32)
        for t in range(T):
            x = g.gamma(0.3, size=(L, H, w_act[t]))
            fm[t, :, :, :w_act[t]] = 0.6 * x / x.sum(-1, keepdims=True)
        card = np.full((T, L, H, W), np.nan, np.float32)
        gate = np.zeros((T, L, HK, W), bool)
        fired = np.zeros((T, L), bool)
        for t in range(1, T):
            for li in range(L):
                q = tier[t, li] == PA.TIER_Q
                if not q.any():
                    continue
                card[t, li][:, q] = np.log(fm[t, li][:, q] + 1e-4) + g.normal(0, .3, (H, q.sum()))
                gate[t, li] = PA.gate_pick(card[t, li][None], q[None], np.zeros((1, W), bool),
                                           np.array([n_sel]), HK)[0]
                fired[t, li] = True
        w.add_sample(f"sample_{s:03d}.npz", {
            "full_mass": fm.astype(np.float16), "tier": tier, "gate_open": gate,
            "gate_fired": fired, "evict_step": evict, "tokens": np.arange(T),
            "rank_signal": rank, "rank_signal_steps": steps,
            "card_logmass": card.astype(np.float16)})
        samples.append({"file": f"sample_{s:03d}.npz", "article_index": s,
                        "article_sha": f"s{s}"})
    meta = {
        "schema_version": "1.0",
        "config": {"model_path": "synthetic", "seed": 0, "article_index": 0,
                   "quant_ratio": 0.7, "gate_ratio": 0.25, "cache_budget": 0.2,
                   "quant_budget_mode": "bytes", "quant_promotion": "bidir",
                   "quant_promote_source": "dequant"},
        "dataset_resolved": {"corpus": "synthetic"},
        "model": {"num_layers": L, "num_heads": H, "num_kv_heads": HK, "head_dim": 8},
        "axes": {"T": T, "S_max": P + GEN, "W": W, "window_size": WS,
                 "num_sink_tokens": NS, "prefill": P, "gen": GEN},
        "resolved_geometry": {"top_k_fp": K_FP, "N_q": N_Q, "top_k_windows": 3,
                              "local_windows": LOCAL_W},
        "read_gate": {"verdict": "gated"},
        "samples": samples,
    }
    return w.close(meta)


def test_end_to_end_on_a_synthetic_run(tmp_path):
    path = _synthetic_zip(tmp_path)
    res = PA.run(path, tmp_path / "out", horizon=4, n_boot=50)
    c = res["checks"]
    assert c["bidir_replay_mismatch"] == 0 and c["bidir_replay_events"] > 0
    assert c["gate_rows_reproduced"] == c["gate_rows"] > 0
    assert c["oneway_promotions"] == 0 and c["oneway_fp_differs"] > 0
    t = res["table"]
    assert t["promotions"]["without"] == 0 and t["promotions"]["with"] > 0
    assert t["lir"]["without"] == 0.0
    for key in ("fmm_kept", "fmm_exact", "fp", "read", "q_skipped", "held_read",
                "q_recall", "qsa"):
        assert np.isfinite(t[key]["with"]) and np.isfinite(t[key]["without"]), key
        assert t[key]["traces"] == 4
    # read + missed is all of each head's attention, in both arms
    for arm in ("with", "without"):
        assert 0.0 < t["read"][arm] <= 1.0
    # the closed bracket only differs from the open one in the gate
    for key in ("fp", "fp_window_share"):
        assert res["table_closed"][key]["without"] == pytest.approx(t[key]["without"])
    assert res["swap"]["promoted_windows"] > 0
    md = (tmp_path / "out" / "promotion_ablation.md").read_text()
    assert "Table 18" in md and "one-way replay" in md
    js = json.loads((tmp_path / "out" / "promotion_ablation.json").read_text())
    assert js["no_promotion_source"] == "replay"


def test_the_replay_refuses_a_oneway_run(tmp_path):
    path = _synthetic_zip(tmp_path, n_samples=1, L=1)
    import zipfile
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        members = {n: zf.read(n) for n in zf.namelist()}
    meta["config"]["quant_promotion"] = "oneway"
    bad = tmp_path / "oneway.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        for n, b in members.items():
            zf.writestr(n, json.dumps(meta) if n == "meta.json" else b)
    with pytest.raises(ValueError, match="bidirectional"):
        PA.replay_run(bad)
