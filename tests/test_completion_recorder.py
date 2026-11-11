"""The batched completion recorder (modules/evaluation/completion_recorder.py).

1. ``head_step_stats`` against a numpy transcription of the reduction the
   completion-ablation notebook applied to the B = 1 collector's arrays.
2. **Batched == per row.** One B = 2 cache and two B = 1 caches receive the
   same per-row keys, values, queries and scores through the real gate (the
   fused kernel stubbed, as ``test_observation_collector`` does). The B = 2
   recorder's records for row b must equal the B = 1 recorder's on cache b,
   and at B = 1 the recorder must agree with the collector's
   ``record_completion_rungs`` path, the one the GPU runs validated.
3. The int2 tier read for a whole batch equals the collector's B = 1 read.
4. ``RowTracker`` drops a row from the step after its EOS.

CPU only; no GPU, no Triton, no model.
"""

from __future__ import annotations

import math
import types
import warnings

import numpy as np
import pytest
import torch

from modules.evaluation import completion_recorder as CR
from modules.evaluation import observation_collector as OC
from modules.windowed_cache.completion import COMPLETION_RUNGS

L, HK, HQ, D, WS, NS, P, STEPS = 2, 2, 4, 16, 4, 2, 48, 40
RATIO = 0.5


class _Rope(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x, position_ids):
        f = position_ids.to(torch.float32).unsqueeze(-1) * torch.arange(
            1, self.d // 2 + 1, dtype=torch.float32) * 0.01
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()


# ---------------------------------------------------------------------------
# 1. the per-head reduction
# ---------------------------------------------------------------------------


def _head_step_numpy(read, lms, err, shift, cva, floor=CR.FLOOR):
    """The notebook's _head_step, every window in the int2 tier, shift kept in log space."""
    R, U = read, ~read
    lms, e = lms.astype(np.float64), err.astype(np.float64)
    fin = np.isfinite(lms) & np.isfinite(e)
    m = np.where(fin, np.exp(np.where(fin, lms, -np.inf)), 0.0)
    Uw, Uf = U & fin, U & fin & (m >= floor)
    nU = Uf.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wU = np.where(Uw, m, 0.0)
        sw = wU.sum(1)
        e_unc = e + shift[:, None]
        out = [sw / (1.0 + sw),
               (wU * np.abs(np.where(Uw, e, 0))).sum(1) / sw,
               np.where(nU > 0, np.abs(np.where(Uf, e, 0)).sum(1) / nU, np.nan),
               (wU * np.where(Uw, e, 0)).sum(1) / sw,
               np.where(nU > 0, np.where(Uf, e, 0).sum(1) / nU, np.nan),
               np.log((wU * np.exp(np.where(Uw, e, 0))).sum(1)) - np.log(sw),
               np.where(nU >= 4, np.nanstd(np.where(Uf, e, np.nan), axis=1, ddof=1), np.nan),
               (wU * np.abs(np.where(Uw, e_unc, 0))).sum(1) / sw,
               shift.astype(np.float64), cva.astype(np.float64)]
    out = np.stack(out, -1)
    valid = (R.sum(1) > 0) & (Uw.sum(1) > 0)
    return np.where(valid[:, None] & np.isfinite(out), out, np.nan), valid


@pytest.mark.parametrize("seed", range(6))
def test_head_step_stats_matches_the_notebooks_reduction(seed):
    g = np.random.default_rng(seed)
    H, n = 6, 20
    read = g.random((H, n)) < 0.3
    read[0] = False                         # a head that read nothing: invalid
    read[1] = True                          # a head that left nothing unread: invalid
    lms = np.log(g.uniform(1e-8, 0.3, (H, n)))
    lms[2, :5] = -np.inf                    # empty windows
    err = g.normal(0, 0.5, (H, n))
    err[3, 4] = np.nan
    shift, cva = g.normal(0.2, 0.1, H), g.uniform(0, 1e-3, H)
    want, wvalid = _head_step_numpy(read, lms, err, shift, cva)
    got, valid = CR.head_step_stats(torch.tensor(read), torch.tensor(lms), torch.tensor(err),
                                    torch.tensor(shift), torch.tensor(cva))
    assert (valid.numpy() == wvalid).all() and not wvalid[0] and not wvalid[1]
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# 2-3. a B = 2 cache against two B = 1 caches
# ---------------------------------------------------------------------------


def _drive(rows, data, monkeypatch, collector_check=False):
    """Run the cache over ``data`` for ``rows``; record every decode step with the recorder.

    ``data`` holds per-row tensors, so a B = 1 run on row b sees exactly what row b
    of a B = 2 run sees. Returns the recorder's result (and, with
    ``collector_check``, the collector buffers recorded on the same states).
    """
    from modules.evaluation.ours_parity_runner import GateRecorder
    from modules.windowed_cache import flash_decode
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig

    B = len(rows)
    mcfg = types.SimpleNamespace(num_key_value_heads=HK, num_attention_heads=HQ, hidden_size=HQ * D,
                                 head_dim=D, num_hidden_layers=L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cache = WindowedCache(
            config=WindowedCacheConfig(
                window_size=WS, num_sink_tokens=NS, local_window_size=WS, cache_budget=0.35,
                quant_ratio=0.5, first_eviction_step=0, quant_gate_ratio=RATIO),
            prefill_len=P, model_config=mcfg, kv_dtype=torch.float32, rope_module=_Rope(D),
            num_layers=L, max_tokens=STEPS)
    cache._fused_decode_active = True
    model = types.SimpleNamespace(config=mcfg)
    rec = CR.CompletionRecorder(model, cache, stride=1, eos_ids=[0])
    rec._gate = GateRecorder()
    rec._gate.bind(cache)
    armed = []
    step = {"t": 0, "layer": 0}

    def stub_decode(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win,
                    *, sel, logmass, centroids, vm_bits: int = 4):
        w = n_body_win + qtier["k_codes"].shape[1]
        ws_ = torch.stack([data["wsum"][step["t"]][step["layer"]][r][:, :w] for r in rows])
        return torch.zeros_like(q_hd), ws_

    monkeypatch.setattr(flash_decode, "set_pending", armed.append)
    monkeypatch.setattr(flash_decode, "fused_two_tier_decode", stub_decode)
    flash_decode.set_gate_observer(rec._gate)
    scale = D ** -0.5
    ref = None
    if collector_check:
        S_max, W = P + STEPS, -(-(P + STEPS - NS) // WS)
        ref = OC.SampleBuffers(STEPS + 1, L, HQ, HK, W, S_max, False, torch.device("cpu"),
                               completion_rungs=True)
        shadow = OC.ShadowKV(L, HK, S_max, D, torch.float32, torch.device("cpu"))
        OC.attach_shadow(cache, shadow)

    def merged(i):
        st, store = cache._states[i], cache._stores[i]
        return -(-st.seq_length // WS) + store.num_active_windows

    def stack(key, *idx):
        return torch.stack([data[key][idx[0]][idx[1]][r] for r in rows]) if len(idx) == 2 else \
            torch.stack([data[key][idx[0]][r] for r in rows])

    try:
        for i in range(L):
            k = stack("k0", i)
            cache.update(k, stack("v0", i), i, cache_kwargs={"cache_position": torch.arange(P)})
            cache.cache_kwargs[i]["window_scores"] = stack("sc0", i)[..., :merged(i)]
        for t in range(1, STEPS + 1):
            step["t"] = t
            rec.t = t
            rec._begin(list(range(B)), torch.device("cpu"))
            for i in range(L):
                step["layer"] = i
                k_fp, v_fp = cache.update(stack("k", t, i), stack("v", t, i), i,
                                          cache_kwargs={"cache_position": torch.tensor([P + t - 1])})
                q = stack("q", t, i)                                          # [B, HQ, D]
                if armed:
                    flash_decode._run_fused(armed.pop(), q.view(B, 1, HQ, D), k_fp.transpose(1, 2),
                                            v_fp.transpose(1, 2))
                else:
                    cache.cache_kwargs[i]["window_scores"] = stack("sc", t, i)[..., :merged(i)]
                # the "kernel output": the held full read, row by row
                st = cache._states[i]
                tier = CR.held_int2_tier_batched(cache._stores[i], cache.rope_module, WS, D, torch.float32)
                if collector_check:                 # the batched read IS the collector's read at B = 1
                    one = OC.held_int2_tier(cache._stores[i], cache.rope_module, WS, D, torch.float32)
                    assert (one is None) == (tier is None)
                    if one is not None:
                        assert all(torch.equal(x, y[0]) for x, y in zip(one, tier))
                        data["tier_checks"] = data.get("tier_checks", 0) + 1
                o = []
                for b in range(B):
                    K, V = st.key_states[b].float(), st.value_states[b].float()
                    if tier is not None:
                        K, V = torch.cat([K, tier[0][b]], 1), torch.cat([V, tier[1][b]], 1)
                    o.append(OC.attend(q[b], K, V, scale, HQ // HK)[0])
                o = torch.stack(o)
                if collector_check and i in rec._gate.picks:
                    OC.record_decode_layer(ref, t, i, q[0], o[0], cache, shadow, rec._gate, NS, WS, scale)
                rec.record_layer(i, q, o)
                for d in (rec._gate.picks, rec._gate.logmass, rec._gate.active_ids):
                    d.pop(i, None)
            rec._flush()
            rec.recording = False
    finally:
        flash_decode.clear_gate_observer()
    out = rec.result()
    return (out, ref.to_numpy()) if collector_check else out


@pytest.fixture(scope="module")
def runs():
    g = torch.Generator().manual_seed(11)
    W_max = 64

    def r(*shape):
        return torch.randn(*shape, generator=g)

    data = {
        "k0": [[r(HK, P, D) for _ in range(2)] for _ in range(L)],
        "v0": [[r(HK, P, D) for _ in range(2)] for _ in range(L)],
        "sc0": [[torch.rand(HQ, W_max, generator=g) for _ in range(2)] for _ in range(L)],
        "k": [None] + [[[r(HK, 1, D) for _ in range(2)] for _ in range(L)] for _ in range(STEPS)],
        "v": [None] + [[[r(HK, 1, D) for _ in range(2)] for _ in range(L)] for _ in range(STEPS)],
        "q": [None] + [[[r(HQ, D) for _ in range(2)] for _ in range(L)] for _ in range(STEPS)],
        "sc": [None] + [[[torch.rand(HQ, W_max, generator=g) * 2.0 ** t for _ in range(2)]
                         for _ in range(L)] for t in range(1, STEPS + 1)],
        "wsum": [None] + [[[torch.rand(HQ, W_max, generator=g) * 2.0 ** t for _ in range(2)]
                           for _ in range(L)] for t in range(1, STEPS + 1)],
    }
    mp = pytest.MonkeyPatch()
    try:
        out = {"b2": _drive([0, 1], data, mp), "b1_0": _drive([0], data, mp, collector_check=True),
               "b1_1": _drive([1], data, mp)}
        out["tier_checks"] = data.get("tier_checks", 0)
        return out
    finally:
        mp.undo()


def test_the_gate_fired_and_something_was_recorded(runs):
    out = runs["b2"]
    assert out["fired"].any(), "the gate never fired -- nothing checked"
    assert np.isfinite(out["err"]).any() and np.isfinite(out["stats"]).any()
    assert set(out["rows"].tolist()) == {0, 1}


def test_each_batched_row_equals_its_own_b1_run(runs):
    b2 = runs["b2"]
    for row, key in ((0, "b1_0"), (1, "b1_1")):
        b1 = runs[key][0] if isinstance(runs[key], tuple) else runs[key]
        sel = b2["rows"] == row
        assert (b2["steps"][sel] == b1["steps"]).all()
        assert (b2["fired"][sel] == b1["fired"]).all()
        np.testing.assert_allclose(b2["err"][sel], b1["err"], rtol=1e-4, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(b2["stats"][sel], b1["stats"], rtol=1e-4, atol=1e-6, equal_nan=True)


def test_batched_int2_read_equals_the_collectors(runs):
    assert runs["tier_checks"] > 0


def test_b1_agrees_with_the_collector_path(runs):
    """Same states, two code paths: the recorder's per-head reduction of its own rungs
    must equal the same reduction of the collector's per-window arrays."""
    out, ref = runs["b1_0"]
    K = len(COMPLETION_RUNGS)
    checked = 0
    for r, t in enumerate(out["steps"]):
        for i in np.flatnonzero(out["fired"][r]):
            if not ref["gate_fired"][t, i]:
                continue
            act = np.flatnonzero(np.isfinite(ref["comp_log_mstar"][t, i, 0]))
            go = ref["gate_open"][t, i][:, act]
            read = np.repeat(go, HQ // HK, axis=0)
            lms = ref["comp_log_mstar"][t, i][:, act].astype(np.float64)
            le = ref["comp_logerr_cal"][t, i][:, act].astype(np.float64)
            want, valid = _head_step_numpy(read, lms, le, ref["comp_uncal_shift"][t, i],
                                           np.full(HQ, np.nan))
            got = out["stats"][r, i]
            # cal_vs_actual (last) differs by construction: the collector normalises by the
            # full-KV output, the recorder by o_held. The collector stores masses in fp16.
            np.testing.assert_allclose(got[valid, :-1], want[valid, :-1], rtol=2e-2, atol=2e-3)
            assert np.isnan(got[~valid]).all()
            # The rungs agree up to that normalisation: full_oracle is 0 in both, and the
            # ratio of the two errors is one number per head (|o_full| / |o_held|).
            fo = COMPLETION_RUNGS.index("full_oracle")
            assert np.nanmax(np.abs(out["err"][r, i][:, fo])) < 1e-5
            assert np.nanmax(np.abs(ref["comp_err"][t, i, :, fo, 0])) < 1e-5
            a, b = out["err"][r, i][valid], ref["comp_err"][t, i, :, :, 0][valid]
            big = b > 1e-4
            ratio = np.where(big, a / np.where(big, b, 1.0), np.nan)
            for row_ in ratio:
                v = row_[np.isfinite(row_)]
                assert v.size and (v.max() - v.min()) < 1e-3 * v.max()
            checked += 1
    assert checked > 0
    assert K == out["err"].shape[-1]


def test_calibrated_rung_reproduces_the_kernel_output(runs):
    """o_actual here is the held full read, so cal_vs_actual measures cal against
    full_oracle: finite, and the 'off' rung is never better than 'cal' on average."""
    out = runs["b2"]
    cva = out["stats"][..., CR.STAT_NAMES.index("cal_vs_actual")]
    assert np.isfinite(cva).any()
    off, cal = COMPLETION_RUNGS.index("off"), COMPLETION_RUNGS.index("cal")
    assert np.nanmedian(out["err"][..., cal]) <= np.nanmedian(out["err"][..., off])


# ---------------------------------------------------------------------------
# 4. rows that have finished
# ---------------------------------------------------------------------------


def test_row_tracker_drops_a_row_after_its_eos():
    tr = CR.RowTracker([7, 9])
    ids = torch.tensor([[1, 2, 3], [1, 2, 9], [1, 2, 4]])
    stop = tr(ids, None)
    assert not stop.any()
    assert tr.alive.tolist() == [True, False, True]
    tr(torch.cat([ids, torch.tensor([[7], [5], [4]])], 1), None)   # a dead row stays dead
    assert tr.alive.tolist() == [False, False, True]


def test_recorder_refuses_a_zero_stride():
    model = types.SimpleNamespace(config=types.SimpleNamespace(
        num_attention_heads=HQ, num_key_value_heads=HK, head_dim=D, hidden_size=HQ * D, num_hidden_layers=L))
    cache = types.SimpleNamespace(resolved=types.SimpleNamespace(window_size=WS))
    with pytest.raises(ValueError):
        CR.CompletionRecorder(model, cache, stride=0, eos_ids=[0])


# ---------------------------------------------------------------------------
# 5. the hooks, inside a real generate() at B = 2
# ---------------------------------------------------------------------------


def test_hooks_hand_record_layer_the_models_own_query_and_output(monkeypatch):
    """Every stride-th decode step, every layer, record_layer gets the batched post-RoPE
    query and pre-o_proj output. Row b's equals what the collector's (validated) B = 1
    taps rebuild in a B = 1 generate of row b alone; a row that emitted EOS is dropped
    from the next recorded step on."""
    pytest.importorskip("transformers")
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, LogitsProcessorList, \
        StoppingCriteriaList

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).eval()
    H, Dh, B, P, G, stride, EOS = 4, 16, 2, 12, 9, 2, 63
    ids = torch.randint(0, 60, (B, P))

    class ForceEos:                                    # `row` emits EOS as its 3rd new token
        def __init__(self, row):
            self.n, self.row = 0, row

        def __call__(self, input_ids, scores):
            self.n += 1
            if self.n == 3 and self.row is not None:
                scores[self.row] = -1e9
                scores[self.row, EOS] = 1e9
            return scores

    def gen(x, eos_row, stopping=()):
        with torch.no_grad():
            model.generate(x, attention_mask=torch.ones_like(x), past_key_values=DynamicCache(),
                           max_new_tokens=G, do_sample=False, pad_token_id=0, eos_token_id=None,
                           logits_processor=LogitsProcessorList([ForceEos(eos_row)]),
                           stopping_criteria=StoppingCriteriaList(list(stopping)))

    cache = DynamicCache()
    cache.resolved = types.SimpleNamespace(window_size=4)
    rec = CR.CompletionRecorder(model, cache, stride=stride, eos_ids=[EOS])
    monkeypatch.setattr(rec, "_batch", lambda: B)
    got = []
    monkeypatch.setattr(rec, "record_layer",
                        lambda li, q, o: got.append((rec.t, li, list(rec._rows), q.clone(), o.clone())))
    rec.attach()
    try:
        gen(ids, 1, [rec.tracker])
    finally:
        rec.detach()
    steps = sorted({t for t, *_ in got})
    assert steps == [t for t in range(1, G) if t % stride == 0]
    for t, li, rows, q, o in got:
        assert q.shape == (B, H, Dh) and o.shape == (B, H, Dh)
        assert rows == ([0, 1] if t < 3 else [0])     # row 1's EOS was token 3: dead from step 3

    for b in range(B):
        ref = []
        taps = OC.ModelTaps(model, H, Dh, lambda li, q, o: ref.append((li, q.clone(), o.clone())))
        try:
            gen(ids[b:b + 1], 0 if b == 1 else None)
        finally:
            taps.remove()
        dec = [(li, q, o) for li, q, o in ref if q.shape[1] == 1]
        for t, li, rows, q, o in got:
            if b not in rows:
                continue
            rli, rq, ro = dec[(t - 1) * cfg.num_hidden_layers + li]
            assert rli == li
            torch.testing.assert_close(q[b], rq[:, 0], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(o[b], ro[:, 0], rtol=1e-4, atol=1e-5)
