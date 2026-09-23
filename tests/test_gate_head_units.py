"""The read gate's GQA union, taken in each query head's own units.

The gate keeps ``n_sel`` windows per KV head, chosen by a max over that head's
``rep`` query heads. A max is only a union if the heads are measured in the same
units, and raw logits are not: every query head carries its own constant
``q . k_common``, which its softmax discards and a cross-head max does not. So
the head with the largest offset picks the whole group's windows and the others
read their hot windows only through value centroids.

That was invisible to the test that set the operating point
(``tests/test_sketch.py::test_recall_at_the_shipped_25_percent_ratio``): it pools
raw ``exp(logit)`` across a KV head's query heads before measuring recall, which
is dominated by the same head the raw union serves. Measured per QUERY head on
that test's own fixture, the raw union leaves the worst head nothing.

``quant_gate_head_norm`` takes each head's estimate relative to its own Q-tier
log-partition first (:func:`modules.quant.sketch.in_head_units`). Auto-on where
the model's projections carry a bias (Qwen2), off on Llama/Mistral so those
columns are byte-identical until a run says otherwise.

CPU-only. The kernel parity test runs the real Triton kernels in Triton's
interpreter, in a subprocess, because the interpreter is chosen at decoration.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from modules.quant.sketch import (
    Sketch,
    build_sketch,
    gate_and_score,
    group_max,
    head_log_partition,
    in_head_units,
    select_windows,
    value_centroid,
)
from modules.windowed_cache.gate_kernel import gate_reference

D, WS = 128, 8
SCALING = 1.0 / math.sqrt(D)
REPO = Path(__file__).resolve().parents[1]


def _recall_fixture(hkv, hq, nw, seed):
    """``test_recall_at_the_shipped_25_percent_ratio``'s fixture, verbatim:
    keys with a massive-activation common mode and one hot token per window."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(hkv, D, generator=g) * 0.4
    a[:, 3] += 40.0
    a[:, 17] -= 25.0
    k = a[None, :, None, :] + torch.randn(nw, hkv, WS, D, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 6.0
    v = torch.randn(nw, hkv, WS, D, generator=g)
    q = torch.randn(1, hq, D, generator=g) * 1.5
    return k, a, v, q


def _card(k, a, v):
    s = build_sketch(k, a, v, v.mean(dim=(0, 2)))
    return Sketch(*[f.unsqueeze(0) for f in s])


def _per_query_head_recall(k, q, keep, hkv):
    """Share of each query head's OWN Q-tier softmax mass in the kept windows."""
    hq, nw = q.shape[1], k.shape[0]
    rep = hq // hkv
    logits = SCALING * torch.einsum(
        "nhwd,bhrd->bhrnw", k, q.reshape(1, hkv, rep, D)).reshape(1, hq, nw, WS)
    p = torch.softmax(logits.logsumexp(-1), dim=-1)                  # [1,hq,nw]
    return (p * keep.repeat_interleave(rep, dim=1)).sum(-1).flatten()


# ---------------------------------------------------------------------------
# the failure, and the fix, on the fixture the operating point was set on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hkv,hq", [(8, 32), (4, 28)], ids=["llama-rep4", "qwen-rep7"])
def test_the_raw_union_starves_query_heads_and_head_units_do_not(hkv, hq):
    """Per QUERY head, at the shipped 0.25 ratio.

    Raw: the worst head keeps none of its Q-tier mass (measured 0.0000 at both
    geometries) -- it is served nothing the group reads. In head units the worst
    head keeps >= 0.99 and the mean is >= 0.999. Both geometries fail raw; the
    fix is auto-on only for biased models (see the config test below).
    """
    nw = 271
    n_sel = math.ceil(0.25 * nw)
    raw, units = [], []
    for seed in range(31, 35):
        k, a, v, q = _recall_fixture(hkv, hq, nw, seed)
        logmass, est = gate_and_score(q, _card(k, a, v), a, SCALING)
        keep_raw = select_windows(group_max(est, hkv), n_sel)
        keep_units = select_windows(group_max(in_head_units(est, logmass), hkv), n_sel)
        raw.append(_per_query_head_recall(k, q, keep_raw, hkv))
        units.append(_per_query_head_recall(k, q, keep_units, hkv))
    raw, units = torch.cat(raw), torch.cat(units)
    assert raw.min() < 0.5, f"raw union no longer starves a head ({raw.min():.4f})"
    assert units.min() >= 0.99, f"worst head in head units: {units.min():.4f}"
    assert units.mean() >= 0.999, f"mean in head units: {units.mean():.4f}"


def test_the_pooled_metric_the_operating_point_was_set_on_could_not_see_it():
    """Why the shipped recall test passed: pooling raw exp(logit) over the group
    before measuring gives the raw union ~1.0 on the same fixture where a query
    head gets nothing. Kept so nobody re-derives the operating point from it."""
    hkv, hq, nw = 8, 32, 271
    k, a, v, q = _recall_fixture(hkv, hq, nw, seed=31)
    _, est = gate_and_score(q, _card(k, a, v), a, SCALING)
    keep = select_windows(group_max(est, hkv), math.ceil(0.25 * nw))
    true = SCALING * torch.einsum(
        "nhwd,bhrd->bhrnw", k, q.reshape(1, hkv, hq // hkv, D)).reshape(1, hq, nw, WS)
    pooled = true.logsumexp(-1).reshape(1, hkv, hq // hkv, nw).logsumexp(2).exp()
    pooled_recall = (pooled * keep).sum(-1) / pooled.sum(-1)
    assert pooled_recall.min() > 0.995
    assert _per_query_head_recall(k, q, keep, hkv).min() < 0.5


# ---------------------------------------------------------------------------
# the invariance that makes it the right quantity
# ---------------------------------------------------------------------------


def _constant_channel_case(hkv=2, rep=4, nw=64, seed=0):
    """Keys whose channel ``C`` is EXACTLY constant across every token, so a query
    component along it adds the same logit to every key: a pure per-head offset,
    which each head's softmax cannot see."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(nw, hkv, WS, D, generator=g)
    k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 4.0
    k[..., 7] = 50.0
    v = torch.randn(nw, hkv, WS, D, generator=g)
    q = torch.randn(1, hkv * rep, D, generator=g)
    q[..., 7] = 0.0
    a = k.mean(dim=(0, 2))
    return k, a, v, q


def test_head_units_are_invariant_to_a_per_head_logit_offset_and_raw_is_not():
    k, a, v, q = _constant_channel_case()
    hkv, n_sel = 2, 16
    card = _card(k, a, v)
    shifted = q.clone()
    # A different offset per head, +-12 * 50 / sqrt(128) = +-53 nats -- and
    # invisible to every head's own attention, since channel 7 is constant.
    shifted[0, :, 7] = torch.linspace(-12.0, 12.0, q.shape[1])

    def pick(qq, units):
        logmass, est = gate_and_score(qq, card, a, SCALING)
        rank = in_head_units(est, logmass) if units else est
        return group_max(rank, hkv).topk(n_sel, dim=-1).indices.sort(-1).values

    assert torch.equal(pick(q, True), pick(shifted, True))
    assert not torch.equal(pick(q, False), pick(shifted, False)), (
        "the offsets did not move the raw union; this test has no teeth")


def test_head_units_keep_every_heads_own_ranking():
    """A constant shift per head: the head's ranking of its own windows, and so
    every rep == 1 selection, cannot move."""
    k, a, v, q = _recall_fixture(4, 28, 50, seed=3)
    logmass, est = gate_and_score(q, _card(k, a, v), a, SCALING)
    shift = in_head_units(est, logmass) - est
    assert torch.allclose(shift, shift[..., :1].expand_as(shift), atol=1e-5)
    assert torch.allclose(-shift[..., :1], head_log_partition(logmass), atol=1e-5)


def test_head_log_partition_is_the_logsumexp():
    x = torch.randn(3, 5, 17) * 20
    assert torch.allclose(head_log_partition(x), x.logsumexp(-1, keepdim=True), atol=1e-5)


def test_rep_1_is_untouched_by_the_switch():
    """MHA: nothing to put in common units. Both statements of the gate skip it,
    so the selection is identical, not merely equivalent."""
    k, a, v, q = _recall_fixture(4, 4, 60, seed=5)
    card = _card(k, a, v)
    off = gate_reference(q, *card, a, SCALING, 15, head_norm=False)
    on = gate_reference(q, *card, a, SCALING, 15, head_norm=True)
    assert torch.equal(off[0], on[0]) and torch.equal(off[1], on[1])


def test_ratio_1_is_still_the_provable_no_op():
    """The control arm selects every window in either unit system."""
    k, a, v, q = _recall_fixture(4, 28, 40, seed=6)
    card = _card(k, a, v)
    for hn in (False, True):
        sel, _ = gate_reference(q, *card, a, SCALING, 40, head_norm=hn)
        assert torch.equal(sel, torch.arange(40).expand(1, 4, 40))


def test_head_norm_only_moves_the_selection_not_logmass():
    """``logmass`` is what skipped windows score and attend with; the switch is
    a selection rule and must leave it bit-identical."""
    k, a, v, q = _recall_fixture(4, 28, 60, seed=7)
    card = _card(k, a, v)
    _, lm_off = gate_reference(q, *card, a, SCALING, 15, head_norm=False)
    _, lm_on = gate_reference(q, *card, a, SCALING, 15, head_norm=True)
    assert torch.equal(lm_off, lm_on)


# ---------------------------------------------------------------------------
# the attention output, through the decode kernel's own CPU oracle
# ---------------------------------------------------------------------------


def _qwen_like_step(seed, offset_sd=40.0):
    """A GQA group of 7 with retrieval heads (3 hot tokens each, +12 nats) and
    per-head constant offsets, over [sink | fp body | Q tier]."""
    g = torch.Generator().manual_seed(seed)
    hkv, rep, n_q, n_fp, sink = 2, 7, 160, 24, 5
    hq = hkv * rep
    s_fp = sink + n_fp * WS
    S = s_fp + n_q * WS
    k = torch.randn(1, hkv, S, D, generator=g)
    k[..., 96:104] = (torch.randn(hkv, 8, generator=g) * 30.0)[None, :, None, :]
    v = torch.randn(1, hkv, S, D, generator=g)
    q = torch.randn(1, hq, D, generator=g)
    q[..., 96:104] = 0.0
    q = q / q.norm(dim=-1, keepdim=True) * math.sqrt(D)
    for h in range(0, hq, 2):                                  # retrieval heads
        for _ in range(3):
            w = int(torch.randint(0, n_q, (1,), generator=g))
            t = int(torch.randint(0, WS, (1,), generator=g))
            qh = q[0, h]
            k[0, h // rep, s_fp + w * WS + t] += qh / qh.norm() * (12.0 / (SCALING * qh.norm()))
    off = torch.randn(hq, generator=g) * offset_sd
    for h in range(hq):
        c = k[0, h // rep, 0, 96:104]
        q[0, h, 96:104] = c / (c @ c) * (off[h] / SCALING)
    return q, k, v, s_fp, sink, n_q, hkv, rep


def test_every_query_head_is_served_in_the_attention_output():
    """End to end through ``two_tier_window_reference`` -- the decode kernel's
    oracle, skipped windows attending through their centroids at the calibrated
    weight -- against the full read. Retrieval heads are the ones that break:
    their hot token is exactly what a centroid cannot represent."""
    from modules.windowed_cache.decode_kernel import two_tier_window_reference

    errs = {False: [], True: []}
    for seed in range(3):
        q, k, v, s_fp, sink, n_q, hkv, rep = _qwen_like_step(seed)
        kq = k[:, :, s_fp:].reshape(1, hkv, n_q, WS, D).permute(0, 2, 1, 3, 4)[0]
        vq = v[:, :, s_fp:].reshape(1, hkv, n_q, WS, D).permute(0, 2, 1, 3, 4)[0]
        a, va = kq.mean(dim=(0, 2)), vq.mean(dim=(0, 2))
        card = Sketch(*[f.unsqueeze(0) for f in build_sketch(kq, a, vq, va)])
        cent = value_centroid(card, va)[0].permute(1, 0, 2)
        cent = cent.repeat_interleave(rep, dim=0).unsqueeze(0)
        nbw = -(-(s_fp - sink) // WS)
        exact, _ = two_tier_window_reference(q, k, v, SCALING, sink, WS, nbw, Sfp=s_fp)
        for hn in (False, True):
            sel, lm = gate_reference(q, *card, a, SCALING, math.ceil(0.25 * n_q),
                                     head_norm=hn)
            out, _ = two_tier_window_reference(
                q, k, v, SCALING, sink, WS, nbw, Sfp=s_fp,
                sel=sel, logmass=lm, centroids=cent)
            errs[hn].append(((out - exact).norm(dim=-1) / math.sqrt(D)).flatten())
    raw, units = torch.cat(errs[False]), torch.cat(errs[True])
    assert units.max() < 0.2 < raw.max(), (units.max(), raw.max())
    assert units.mean() < 0.25 * raw.mean(), (units.mean(), raw.mean())


# ---------------------------------------------------------------------------
# configuration: auto from the model, overridable, threaded everywhere
# ---------------------------------------------------------------------------


def _hf(name, **kw):
    transformers = pytest.importorskip("transformers")
    base = dict(hidden_size=512, num_attention_heads=4, num_key_value_heads=4,
                num_hidden_layers=2, intermediate_size=64, vocab_size=64)
    base.update(kw)
    return getattr(transformers, name)(**base)


def _resolve(model_cfg, **kw):
    from modules.windowed_cache.config import WindowedCacheConfig

    cfg = WindowedCacheConfig(window_size=8, num_sink_tokens=5,
                              local_window_size=128, cache_budget=0.2,
                              quant_ratio=kw.pop("quant_ratio", 0.7), **kw)
    return cfg.resolve(4096, model_cfg, torch.float16, 32)


def test_auto_is_on_for_qwen2_and_off_for_llama_and_mistral():
    assert _resolve(_hf("Qwen2Config")).quant_gate_head_norm is True
    assert _resolve(_hf("LlamaConfig")).quant_gate_head_norm is False
    assert _resolve(_hf("MistralConfig")).quant_gate_head_norm is False


def test_the_switch_overrides_and_needs_a_q_tier():
    assert _resolve(_hf("LlamaConfig"), quant_gate_head_norm=True).quant_gate_head_norm
    assert not _resolve(_hf("Qwen2Config"),
                        quant_gate_head_norm=False).quant_gate_head_norm
    assert not _resolve(_hf("Qwen2Config"), quant_ratio=0.0).quant_gate_head_norm
    with pytest.raises(ValueError, match="quant_gate_head_norm"):
        _resolve(_hf("Qwen2Config"), quant_gate_head_norm=1)


def test_the_yaml_schema_takes_it_and_rejects_non_bools():
    from utils.config import CacheConfig, ConfigValidationError

    assert CacheConfig().quant_gate_head_norm is None
    assert CacheConfig(quant_gate_head_norm=True).quant_gate_head_norm is True
    with pytest.raises(ConfigValidationError, match="quant_gate_head_norm"):
        CacheConfig(quant_gate_head_norm="yes")


def test_the_factory_passes_auto_through_rather_than_dropping_it():
    from modules.windowed_cache.config import WindowedCacheConfig
    from utils.cache_factory import quant_gate_head_norm_kwargs

    assert quant_gate_head_norm_kwargs(WindowedCacheConfig, None) == {
        "quant_gate_head_norm": None}
    assert quant_gate_head_norm_kwargs(WindowedCacheConfig, False) == {
        "quant_gate_head_norm": False}


@pytest.mark.parametrize("path", [
    "modules/evaluation/longbench_runner.py",
    "modules/evaluation/gsm8k_runner.py",
    "modules/evaluation/ruler_runner.py",
    "modules/evaluation/ours_parity_runner.py",
    "modules/evaluation/perf_runner.py",
    "scripts/audit_e2e.py",
])
def test_every_runner_threads_the_yaml_knob(path):
    """The repo's recurring failure: a knob in the YAML that a runner never
    passes is silently inert (quant_gate_ratio, quant_budget_mode, 6b8a188)."""
    src = (REPO / path).read_text()
    assert 'getattr(cfg.cache, "quant_gate_head_norm", None)' in src


def test_the_longbench_sidecar_records_the_resolved_value():
    src = (REPO / "modules/evaluation/longbench_runner.py").read_text()
    assert '"quant_gate_head_norm": bool(getattr(r, "quant_gate_head_norm", False))' in src


# ---------------------------------------------------------------------------
# the cache hands it to the gate on BOTH context builders
# ---------------------------------------------------------------------------


def _cache(model, head_norm=None):
    transformers = pytest.importorskip("transformers")
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig

    cls = getattr(transformers, f"{model}Config")
    mcfg = cls(hidden_size=1024, num_attention_heads=8, num_key_value_heads=2,
               num_hidden_layers=1, intermediate_size=64, vocab_size=64,
               max_position_embeddings=4096, rope_theta=1.0e6)
    rope_cls = {
        "Qwen2": "transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding",
        "Llama": "transformers.models.llama.modeling_llama.LlamaRotaryEmbedding",
    }[model]
    mod, name = rope_cls.rsplit(".", 1)
    rope = getattr(__import__(mod, fromlist=[name]), name)(config=mcfg)
    ccfg = WindowedCacheConfig(
        window_size=8, num_sink_tokens=4, local_window_size=16,
        cache_budget=0.5, quant_ratio=0.5, quant_gate_head_norm=head_norm)
    cache = WindowedCache(
        config=ccfg, prefill_len=260, model_config=mcfg, kv_dtype=torch.float32,
        rope_module=rope, num_layers=1, max_tokens=8)
    return cache, rope


def _evict(cache, rope):
    from modules.quant.effective import rotate_key_window

    hk, sink, n_win = 2, 4, 32
    T = sink + n_win * WS
    g = torch.Generator().manual_seed(0)
    pos = torch.arange(T)
    k_post = rotate_key_window(torch.randn(hk, T, D, generator=g), pos, rope)
    st = cache._states[0]
    st.replace(k_post.unsqueeze(0).clone(), torch.randn(1, hk, T, D, generator=g),
               pos.unsqueeze(0).clone())
    st.window_scores = torch.linspace(100.0, 1.0, n_win).expand(1, 8, n_win).clone()
    st.original_window_ids = torch.arange(n_win).unsqueeze(0)
    cache._policies[0].initialize_after_prefill(T)
    cache._prefill_done[0] = True
    cache._evict_two_tier(0, step=0)
    store = cache._stores[0]
    store.commit_active_count(store.num_active_windows)
    return store


@pytest.mark.parametrize("model,forced,expect", [
    ("Qwen2", None, True), ("Llama", None, False),
    ("Qwen2", False, False), ("Llama", True, True),
])
def test_both_fused_context_builders_carry_the_resolved_switch(model, forced, expect):
    """Per-layer ``_fused_qtier`` and the production layer-major
    ``_build_joint_fused_ctx`` each assemble the gate dict ``_run_fused`` reads.
    The joint one rebuilds it key by key, so a key added to ``_gate_ctx`` alone
    would be dropped on the path every real run takes."""
    cache, rope = _cache(model, head_norm=forced)
    store = _evict(cache, rope)
    n = store.num_active_windows
    assert n > 0
    cache._fused_qtier(0, store, 1, n, WS)
    per_layer = cache._fused_ctx[0]["gate"]
    cache._fused_ctx[0] = None
    cache._build_joint_fused_ctx(store, 1, n, WS)
    joint = cache._fused_ctx[0]["gate"]
    for gate in (per_layer, joint):
        assert gate["head_norm"] is expect
    assert set(joint) == set(per_layer), "the joint builder dropped a gate key"


def test_run_fused_hands_the_switch_to_the_gate_and_has_no_default(monkeypatch):
    """Indexed, not ``.get``: a context without it would gate in raw units and
    look exactly like the configured run."""
    from modules.windowed_cache import flash_decode

    seen = {}

    class _Stop(Exception):
        pass

    def fake_gate(q, card, anchor, scaling, n_sel, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(flash_decode, "fused_gate", fake_gate)
    ctx = {"layer_idx": 0, "n_kv_heads": 2, "score_meta": (None, 0),
           "num_sink": 4, "window_size": 8, "scaling": SCALING,
           "gate": {"card": None, "anchor": None, "n_sel": 1, "centroid": None,
                    "head_norm": True}}
    q = torch.randn(1, 1, 8, D)
    kv = torch.randn(1, 12, 2, D)
    with pytest.raises(_Stop):
        flash_decode._run_fused(ctx, q, kv, kv)
    assert seen == {"head_norm": True}
    del ctx["gate"]["head_norm"]
    with pytest.raises(KeyError):
        flash_decode._run_fused(ctx, q, kv, kv)


def test_the_store_level_gate_takes_the_same_switch():
    """``QuantizedStore.gate_and_select`` is the store's statement of the gate;
    it must agree with the reference in both unit systems."""
    cache, rope = _cache("Qwen2")
    store = _evict(cache, rope)
    q = torch.randn(1, 8, D) * 3.0
    q[..., :4] += torch.linspace(-30, 30, 8)[None, :, None]
    slots = store.table.active_order(store.num_active_windows)
    card = Sketch(*store.table.gather_sketch(slots))
    for hn in (False, True):
        keep, _, _ = store.gate_and_select(q, SCALING, 0.25, head_norm=hn)
        sel, _ = gate_reference(q, *card, store._anchor, SCALING,
                                store.cap_for_ratio(0.25), head_norm=hn)
        picked = keep.nonzero()[:, -1].reshape(sel.shape)
        assert torch.equal(picked, sel.long())


# ---------------------------------------------------------------------------
# the Triton kernels themselves, executed (interpreter) against the reference
# ---------------------------------------------------------------------------


_INTERPRETED = textwrap.dedent("""
    import math, os, sys, torch
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from modules.quant.sketch import build_sketch
    from modules.windowed_cache import gate_kernel as gk
    D, WS = 128, 8
    bad = []
    for hkv, rep, nw, seed in ((4, 7, 37, 1), (2, 4, 100, 2), (2, 3, 21, 3)):
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(hkv, D, generator=g) * 0.4
        a[:, 3] += 40.0; a[:, 17] -= 25.0
        k = a[None, :, None, :] + torch.randn(nw, hkv, WS, D, generator=g)
        k[:, :, 0, :] += torch.randn(nw, hkv, D, generator=g) * 6.0
        v = torch.randn(nw, hkv, WS, D, generator=g)
        q = (torch.randn(2, hkv * rep, D, generator=g) * 1.5).to(torch.float16)
        s = build_sketch(k, a, v, v.mean(dim=(0, 2)))
        card = tuple(f.unsqueeze(0).expand(2, *f.shape).contiguous() for f in s)
        n_sel = math.ceil(0.25 * nw)
        for hn in (False, True):
            ref_sel, ref_lm = gk.gate_reference(
                q.float(), *card, a, 1 / math.sqrt(D), n_sel, head_norm=hn)
            sel, lm = gk._gate_triton(q, card, a, 1 / math.sqrt(D), n_sel,
                                      head_norm=hn)
            if not torch.equal(sel.long(), ref_sel.long()):
                bad.append(("sel", hkv, rep, nw, hn))
            if (lm - ref_lm).abs().max() > 1e-3:
                bad.append(("logmass", hkv, rep, nw, hn,
                            float((lm - ref_lm).abs().max())))
    print("BAD", bad)
    sys.exit(1 if bad else 0)
""")


def test_the_triton_gate_kernels_match_the_reference_when_executed():
    """Not a compile check: ``_gate_triton`` -- the scan kernel in both
    ``HEAD_NORM`` specializations and the union kernel -- run by Triton's
    interpreter on CPU tensors, selection compared exactly with
    :func:`gate_reference`. Shapes are chosen so no window count is a multiple
    of any ``BLOCK_W`` rung and ``rep`` is not a power of two (padded head
    lanes in the union kernel). ``-W error::RuntimeWarning`` makes any
    ``log(0)`` or ``inf - inf`` in a lane that is later selected away fail too."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", _INTERPRETED],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]


# ---------------------------------------------------------------------------
# the run records which union it used
# ---------------------------------------------------------------------------


def test_the_gate_report_records_which_union_the_run_used():
    """Every quality sidecar stores ``read_gate`` (the gate report). It must say
    which units the union was taken in, from the layers that actually ran, not
    from a config default -- a Qwen row gated in raw units is the regression."""
    from modules.windowed_cache import flash_decode

    flash_decode.reset_stats()
    assert flash_decode.stats()["union"] is None
    for n_norm, want in ((28, "per-head"), (0, "raw"), (5, "mixed")):
        flash_decode.reset_stats()
        flash_decode._STATS.update(fired=28, gated=28, head_norm=n_norm)
        assert flash_decode.gate_report(True)["union"] == want
    flash_decode.reset_stats()


def test_run_fused_counts_head_norm_layers(monkeypatch):
    from modules.windowed_cache import flash_decode

    class _Stop(Exception):
        pass

    def fake_gate(*a, **kw):
        return torch.zeros(1, 2, 1, dtype=torch.long), torch.zeros(1, 8, 3)

    def fake_decode(*a, **kw):
        raise _Stop

    monkeypatch.setattr(flash_decode, "fused_gate", fake_gate)
    monkeypatch.setattr(flash_decode, "fused_two_tier_decode", fake_decode)
    flash_decode.reset_stats()
    for hn in (True, False, True):
        ctx = {"layer_idx": 0, "n_kv_heads": 2, "score_meta": (None, 0),
               "num_sink": 4, "window_size": 8, "scaling": SCALING, "qtier": {},
               "gate": {"card": None, "anchor": None, "n_sel": 1,
                        "centroid": None, "head_norm": hn}}
        kv = torch.randn(1, 12, 2, D)
        with pytest.raises(_Stop):
            flash_decode._run_fused(ctx, torch.randn(1, 1, 8, D), kv, kv)
    s = flash_decode.stats()
    assert (s["gated"], s["head_norm"], s["union"]) == (3, 2, "mixed")
    flash_decode.reset_stats()
