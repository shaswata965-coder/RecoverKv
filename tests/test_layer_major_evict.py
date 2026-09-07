"""Layer-major decode (DECODE_SPEED_PLAN.md §4.1) is byte-identical to per-layer.

**What is being pinned.** §4.1 batches the eviction across every layer, and that
forces one ordering change: with per-layer state, layer ``i`` appended step
``t``'s token and *then* evicted; with one layer-major store the eviction has to
run before **any** layer appends, because when layer 0's ``update`` is called at
step ``t`` layers ``1..L-1`` have not produced token ``t`` yet. So
"append then compact" becomes "compact then append".

The claim that makes that safe is that the retained set does not depend on this
step's token:

* ``compute_two_tier_retain`` / ``compute_retain_window_indices`` read **only**
  ``window_scores`` — its width and its values. Neither is touched by the append
  (``policy.total_tokens`` feeds no decision; it only backs two properties the
  eviction never reads).
* this step's token has no score column yet, and it lands in the local region,
  which is force-fp and never eligible for the Q tier or for dropping.

So the cache at the **end** of each step must be identical either way, and that
is what these tests assert — tensor for tensor, over enough steps to cross
several evictions, at ``q > 0`` and ``q == 0``, at ``B > 1``, and with the score
convention the real model uses.

**The score convention matters and is the reason this file drives the cache the
way it does.** In a real forward, layer ``i``'s score hook writes
``cache.cache_kwargs[i]["window_scores"]`` *after* layer ``i``'s attention, so
the scores waiting at step ``t`` describe the key set as of step ``t-1``. These
tests reproduce that exactly (``_write_pending_scores`` runs at the end of a
step, sized to the store as it then stands) rather than handing ``update()`` a
width that anticipates the token about to be appended.
"""

from __future__ import annotations

import os

import pytest
import torch

from modules.windowed_cache.cache import WindowedCache
from modules.windowed_cache.config import WindowedCacheConfig


H_KV, H_Q, D = 2, 2, 4


class _Cfg:
    """Minimal stand-in for a HF PretrainedConfig."""

    num_key_value_heads = H_KV
    num_attention_heads = H_Q
    hidden_size = H_Q * D
    head_dim = D


class _Rope(torch.nn.Module):
    """Deterministic rotary stub: cos/sin from the position, shaped like HF's."""

    def forward(self, x, position_ids):
        pos = position_ids.to(torch.float32)
        freqs = pos.unsqueeze(-1) * torch.arange(
            1, D // 2 + 1, dtype=torch.float32, device=pos.device
        ) * 0.01
        emb = torch.cat([freqs, freqs], dim=-1)          # [B, T, D]
        return emb.cos(), emb.sin()


def _make_cache(*, layer_major, quant_ratio, num_layers, prefill_len,
                ws=4, num_sink=0, budget=0.5):
    os.environ["STICKYKV_LAYER_MAJOR_DECODE"] = "1" if layer_major else "0"
    try:
        cfg = WindowedCacheConfig(
            window_size=ws,
            num_sink_tokens=num_sink,
            local_window_size=ws,
            cache_budget=budget,
            quant_ratio=quant_ratio,
            first_eviction_step=0,
        )
        cache = WindowedCache(
            config=cfg,
            prefill_len=prefill_len,
            model_config=_Cfg(),
            kv_dtype=torch.float32,
            rope_module=_Rope(),
            num_layers=num_layers,
            max_tokens=prefill_len + 64,
        )
    finally:
        del os.environ["STICKYKV_LAYER_MAJOR_DECODE"]
    assert cache._layer_major is layer_major
    return cache


def _merged_windows(cache, layer_idx, ws, num_sink):
    """Merged (fp body + Q) window count of the store AS IT STANDS."""
    st = cache._states[layer_idx]
    body = max(st.seq_length - num_sink, 0)
    store = cache._stores[layer_idx]
    n_q = store.num_active_windows if store is not None else 0
    return (body + ws - 1) // ws + n_q


def _write_pending_scores(cache, num_layers, ws, num_sink, gen):
    """Emulate the score hooks: after a step, leave each layer's scores waiting.

    Sized to the store as it stands at the end of the step — which is what a
    score pass over the keys attention just saw produces — and identical across
    the two cache instances because ``gen`` is a seeded generator replayed the
    same way for both.
    """
    for i in range(num_layers):
        W = _merged_windows(cache, i, ws, num_sink)
        B = cache._states[i].key_states.shape[0]
        cache.cache_kwargs[i]["window_scores"] = torch.rand(
            B, H_Q, W, generator=gen, dtype=torch.float32
        )


def _drive(cache, *, num_layers, prefill_len, steps, batch, ws, num_sink, seed):
    """Prefill + ``steps`` decode steps; snapshot the whole cache after each."""
    gen = torch.Generator().manual_seed(seed)
    kv = torch.Generator().manual_seed(seed + 1)

    for i in range(num_layers):
        k = torch.randn(batch, H_KV, prefill_len, D, generator=kv)
        v = torch.randn(batch, H_KV, prefill_len, D, generator=kv)
        cache.update(k, v, i, cache_kwargs={
            "cache_position": torch.arange(prefill_len)
        })
    _write_pending_scores(cache, num_layers, ws, num_sink, gen)

    snapshots = []
    for t in range(steps):
        pos = prefill_len + t
        for i in range(num_layers):
            k = torch.randn(batch, H_KV, 1, D, generator=kv)
            v = torch.randn(batch, H_KV, 1, D, generator=kv)
            cache.update(k, v, i, cache_kwargs={
                "cache_position": torch.tensor([pos])
            })
        snapshots.append(_snapshot(cache, num_layers))
        _write_pending_scores(cache, num_layers, ws, num_sink, gen)
    return snapshots


def _snapshot(cache, num_layers):
    """Everything a step can change, per layer, cloned."""
    out = []
    for i in range(num_layers):
        st = cache._states[i]
        store = cache._stores[i]
        rec = {
            "k": st.key_states.clone(),
            "v": st.value_states.clone(),
            "pos": st.position_ids.clone(),
            "scores": None if st.window_scores is None else st.window_scores.clone(),
            "wids": (None if st.original_window_ids is None
                     else st.original_window_ids.clone()),
            "seq_len": st.seq_length,
            "eff_len": cache.get_seq_length(i),
            "n_active": 0 if store is None else store.num_active_windows,
        }
        if store is not None and store.table is not None:
            t = store.table
            rec.update({
                "slot_wid": t.slot_wid.clone(),
                "slot_active": t.slot_active.clone(),
                "slot_pos": t.slot_pos.clone(),
                "key_codes": t.key_codes.clone(),
                "key_scale": t.key_scale.clone(),
                "key_zero": t.key_zero.clone(),
                "val_codes": t.val_codes.clone(),
                "val_scale": t.val_scale.clone(),
                "val_zero": t.val_zero.clone(),
            })
        out.append(rec)
    return out


def _assert_same(a, b, label):
    assert len(a) == len(b), f"{label}: step count differs"
    for step, (sa, sb) in enumerate(zip(a, b)):
        for layer, (la, lb) in enumerate(zip(sa, sb)):
            assert la.keys() == lb.keys(), (
                f"{label}: step {step} layer {layer} field set differs "
                f"({sorted(set(la) ^ set(lb))})"
            )
            for field in la:
                x, y = la[field], lb[field]
                where = f"{label}: step {step}, layer {layer}, {field}"
                if x is None or not torch.is_tensor(x):
                    assert x == y, f"{where}: {x!r} != {y!r}"
                    continue
                assert x.shape == y.shape, (
                    f"{where}: shape {tuple(x.shape)} != {tuple(y.shape)}"
                )
                assert torch.equal(x, y), f"{where}: values differ"


# ---------------------------------------------------------------------------
# The equivalence gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant_ratio", [0.0, 0.5])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_layer_major_matches_per_layer(quant_ratio, num_layers, batch):
    """The whole cache, every step, tensor for tensor.

    Enough decode steps (14) at ``ws=4`` to cross the first eviction, the join
    that follows it, and three more batched evictions.
    """
    kw = dict(num_layers=num_layers, prefill_len=16, steps=14, batch=batch,
              ws=4, num_sink=0, seed=20260907)
    ref = _drive(_make_cache(layer_major=False, quant_ratio=quant_ratio,
                             num_layers=num_layers, prefill_len=16), **kw)
    got = _drive(_make_cache(layer_major=True, quant_ratio=quant_ratio,
                             num_layers=num_layers, prefill_len=16), **kw)
    _assert_same(ref, got, f"q={quant_ratio} L={num_layers} B={batch}")


def test_layer_major_matches_per_layer_across_promotions():
    """The same equivalence at a budget that actually moves windows between tiers.

    The default fixture fills the Q tier but never empties a slot back out, so it
    exercises demotion only. This one (tighter budget, 30 steps, ``B=2``) drives
    the store to hold **dormant** entries — windows that were promoted Q -> fp and
    kept their frozen codes for a later re-demotion (design §10) — which is the
    branch where the eviction's ``promote_many`` / ``reactivate_many`` /
    ``demote_many`` all fire in one pass. That is the most intricate part of the
    body and therefore the part most worth pinning across the row-axis change.
    """
    kw = dict(num_layers=3, prefill_len=32, steps=30, batch=2,
              ws=4, num_sink=0, seed=1)
    mk = dict(quant_ratio=0.5, num_layers=3, prefill_len=32, budget=0.35)
    ref = _drive(_make_cache(layer_major=False, **mk), **kw)
    got = _drive(_make_cache(layer_major=True, **mk), **kw)
    _assert_same(ref, got, "promotions")

    # Non-vacuity: the run really did leave dormant entries behind.
    cache = _make_cache(layer_major=True, **mk)
    _drive(cache, **kw)
    table = cache._joint_store.table
    live = (table.slot_wid != -1).sum(1)
    active = table.slot_active.sum(1)
    assert int((live - active).max()) > 0, (
        "no dormant slots: this fixture never exercised a promotion, so the "
        "equivalence it asserts is weaker than intended"
    )


def test_sink_tokens_survive_the_in_place_body_rewrite():
    """``replace_body`` writes only ``[num_sink:]`` — the sinks must be untouched.

    The layer-major eviction stops ``cat``-ing the sink prefix back on and writes
    the compacted body straight into the buffer behind it. If the sinks were ever
    NOT byte-identical at the same offsets across an eviction, that would silently
    corrupt them, so it is checked against the path that does re-``cat`` them.
    """
    kw = dict(num_layers=2, prefill_len=20, steps=14, batch=1,
              ws=4, num_sink=4, seed=7)
    ref = _drive(_make_cache(layer_major=False, quant_ratio=0.5, num_layers=2,
                             prefill_len=20, num_sink=4), **kw)
    got = _drive(_make_cache(layer_major=True, quant_ratio=0.5, num_layers=2,
                             prefill_len=20, num_sink=4), **kw)
    _assert_same(ref, got, "num_sink=4")
    # And the sinks really are constant, not merely equal between the two paths.
    for step in got[1:]:
        for layer, rec in enumerate(step):
            assert torch.equal(rec["pos"][:, :4], got[0][layer]["pos"][:, :4]), (
                f"layer {layer}: sink positions moved across an eviction"
            )


# ---------------------------------------------------------------------------
# The mechanism itself
# ---------------------------------------------------------------------------


def test_the_join_actually_happens_and_only_once():
    """Otherwise the equivalence tests above would be comparing two eager paths."""
    cache = _make_cache(layer_major=True, quant_ratio=0.5, num_layers=3,
                        prefill_len=16)
    assert cache._joint is None
    _drive(cache, num_layers=3, prefill_len=16, steps=6, batch=1, ws=4,
           num_sink=0, seed=3)
    assert cache._joint is not None, "the layer-major store was never built"
    # Row axis is L*B, laid out layer-major.
    assert cache._joint.key_states.shape[0] == 3
    joint = cache._joint
    assert cache._joint_store.table.slot_wid.shape[0] == 3
    # Driving further must reuse it, never rebuild.
    for i in range(3):
        W = _merged_windows(cache, i, 4, 0)
        cache.cache_kwargs[i]["window_scores"] = torch.rand(1, H_Q, W)
    for i in range(3):
        k = torch.randn(1, H_KV, 1, D)
        cache.update(k, k.clone(), i, cache_kwargs={
            "cache_position": torch.tensor([99])})
    assert cache._joint is joint


def test_per_layer_views_report_one_layer_not_all():
    """``_states[i]`` / ``_stores[i]`` stay per-layer shaped after the join.

    The score hooks and the parity runner index them, so a view that leaked the
    ``L*B`` row axis would feed a 3x-too-tall tensor into scoring.
    """
    cache = _make_cache(layer_major=True, quant_ratio=0.5, num_layers=3,
                        prefill_len=16)
    _drive(cache, num_layers=3, prefill_len=16, steps=6, batch=2, ws=4,
           num_sink=0, seed=11)
    assert cache._joint.key_states.shape[0] == 6          # L*B
    for i in range(3):
        st = cache._states[i]
        assert st.key_states.shape[0] == 2, "state view leaked the layer axis"
        assert st.position_ids.shape[0] == 2
        assert st.window_scores.shape[0] == 2
        store = cache._stores[i]
        assert store.table.slot_wid.shape[0] == 2, "table view leaked the axis"
        assert store.active_ids().shape[0] == 2
    # Layer i's view must be layer i's rows of the joint store.
    for i in range(3):
        assert torch.equal(
            cache._states[i].key_states, cache._joint.key_states[2 * i:2 * i + 2]
        )


def test_slot_table_view_refuses_to_be_mutated():
    """A rebinding mutator on a row slice would vanish; it must raise instead."""
    cache = _make_cache(layer_major=True, quant_ratio=0.5, num_layers=3,
                        prefill_len=16)
    _drive(cache, num_layers=3, prefill_len=16, steps=6, batch=1, ws=4,
           num_sink=0, seed=5)
    view = cache._stores[1].table
    with pytest.raises(RuntimeError, match="read-only"):
        view.retain_only(torch.zeros(1, 1, view.n_slots, dtype=torch.bool))


def test_inline_scores_after_the_step_opened_raise():
    """A silent one-step shift in the eviction's evidence must not be possible."""
    cache = _make_cache(layer_major=True, quant_ratio=0.5, num_layers=2,
                        prefill_len=16)
    _drive(cache, num_layers=2, prefill_len=16, steps=6, batch=1, ws=4,
           num_sink=0, seed=9)
    k = torch.randn(1, H_KV, 1, D)
    W = _merged_windows(cache, 1, 4, 0)
    cache.update(k, k.clone(), 0, cache_kwargs={
        "cache_position": torch.tensor([100])})
    with pytest.raises(RuntimeError, match="already opened"):
        cache.update(k, k.clone(), 1, cache_kwargs={
            "cache_position": torch.tensor([100]),
            "window_scores": torch.rand(1, H_Q, W),
        })


def test_partially_scored_step_raises():
    """One layer's hook returning silently must not be averaged into the rest."""
    cache = _make_cache(layer_major=True, quant_ratio=0.5, num_layers=3,
                        prefill_len=16)
    _drive(cache, num_layers=3, prefill_len=16, steps=6, batch=1, ws=4,
           num_sink=0, seed=13)
    W = _merged_windows(cache, 0, 4, 0)
    cache.cache_kwargs[0]["window_scores"] = torch.rand(1, H_Q, W)
    cache.cache_kwargs[1]["window_scores"] = torch.rand(1, H_Q, W)
    cache.cache_kwargs[2].pop("window_scores", None)
    k = torch.randn(1, H_KV, 1, D)
    with pytest.raises(RuntimeError, match="produced no window scores"):
        cache.update(k, k.clone(), 0, cache_kwargs={
            "cache_position": torch.tensor([200])})


def test_inside_a_real_model_forward():
    """The join survives HF's own ``Cache`` protocol, not just direct ``update``.

    Everything above drives ``cache.update()`` by hand. This drives a real
    ``LlamaForCausalLM`` and lets transformers call the cache: ``get_seq_length``
    for the causal mask, ``update`` with HF's own ``cache_position``, and the
    returned effective K/V feeding real attention. Scores are written the way the
    score hook writes them — into ``cache.cache_kwargs[i]`` after the forward, so
    they describe the key set attention just saw — but without installing the
    hook, because the flash backend's prefill scorer is Triton-or-error and this
    box has no CUDA.

    The assertion is the same one as everywhere else in this file: per-layer and
    layer-major must produce the same tokens.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    ws, num_sink, prefill, steps = 4, 0, 32, 16
    ids = torch.randint(0, 256, (1, prefill))

    def run(layer_major):
        os.environ["STICKYKV_LAYER_MAJOR_DECODE"] = "1" if layer_major else "0"
        try:
            cache = WindowedCache(
                config=WindowedCacheConfig(
                    window_size=ws, num_sink_tokens=num_sink,
                    local_window_size=ws, cache_budget=0.5, quant_ratio=0.5,
                    first_eviction_step=0,
                ),
                prefill_len=prefill, model_config=cfg, kv_dtype=torch.float32,
                rope_module=model.model.rotary_emb,
                num_layers=cfg.num_hidden_layers, max_tokens=steps,
            )
        finally:
            del os.environ["STICKYKV_LAYER_MAJOR_DECODE"]

        gen = torch.Generator().manual_seed(4242)

        def score(inp):
            with torch.no_grad():
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
            for i in range(cfg.num_hidden_layers):
                W = _merged_windows(cache, i, ws, num_sink)
                cache.cache_kwargs[i]["window_scores"] = torch.rand(
                    1, cfg.num_attention_heads, W, generator=gen
                )
            return out.logits[:, -1, :].argmax(-1, keepdim=True)

        tok = score(ids)
        toks = []
        for _ in range(steps):
            tok = score(tok)
            toks.append(int(tok))
        return toks, cache

    ref, ref_cache = run(False)
    got, got_cache = run(True)
    assert ref_cache._joint is None
    assert got_cache._joint is not None, "the join never happened in a real forward"
    assert ref == got, (
        f"generated tokens diverged:\n  per-layer   {ref}\n  layer-major {got}"
    )


def test_multi_token_update_after_the_join_raises():
    """A second prompt is undefined against a store sized for one."""
    cache = _make_cache(layer_major=True, quant_ratio=0.5, num_layers=2,
                        prefill_len=16)
    _drive(cache, num_layers=2, prefill_len=16, steps=6, batch=1, ws=4,
           num_sink=0, seed=17)
    k = torch.randn(1, H_KV, 3, D)
    with pytest.raises(RuntimeError, match="only single-token decode"):
        cache.update(k, k.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(300, 303)})


# ---------------------------------------------------------------------------
# window_size range — what the decode path supports today
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128])
def test_window_size_range_q0(ws):
    """At ``q == 0`` every ``ws`` from 1 to 128 works, and both paths agree.

    Pins the supported range rather than assuming it. The fp-only path has no
    int2 crumb packing, so nothing constrains ``ws`` to a multiple of 4 here —
    and nothing in the layer-major eviction cares either, since its row axis is
    orthogonal to the window axis.
    """
    prefill = max(64, ws * 6)
    kw = dict(num_layers=2, prefill_len=prefill, steps=max(10, min(ws * 2, 24)),
              batch=1, ws=ws, num_sink=0, seed=31)
    mk = dict(quant_ratio=0.0, num_layers=2, prefill_len=prefill)
    ref = _drive(_make_cache(layer_major=False, ws=ws, **mk), **kw)
    got = _drive(_make_cache(layer_major=True, ws=ws, **mk), **kw)
    _assert_same(ref, got, f"q=0 ws={ws}")


@pytest.mark.parametrize("ws", [4, 8, 12, 16, 32, 64, 128])
def test_window_size_range_q_positive(ws):
    """At ``q > 0`` the int2 crumb packing needs ``ws % 4 == 0``; within that,
    4 through 128 all work and both paths agree."""
    prefill = max(64, ws * 6)
    kw = dict(num_layers=2, prefill_len=prefill, steps=max(10, min(ws * 2, 24)),
              batch=1, ws=ws, num_sink=0, seed=37)
    mk = dict(quant_ratio=0.5, num_layers=2, prefill_len=prefill)
    ref = _drive(_make_cache(layer_major=False, ws=ws, **mk), **kw)
    got = _drive(_make_cache(layer_major=True, ws=ws, **mk), **kw)
    _assert_same(ref, got, f"q=0.5 ws={ws}")


@pytest.mark.parametrize("ws", [1, 2, 3, 6])
def test_quant_rejects_window_size_not_multiple_of_four(ws):
    """The int2 constraint is a hard config rejection, not a silent degrade —
    which is what bounds §5.1/§5.3's supported range at ``q > 0``."""
    with pytest.raises(ValueError, match="divisible by 4"):
        WindowedCacheConfig(
            window_size=ws, num_sink_tokens=0, local_window_size=ws,
            cache_budget=0.5, quant_ratio=0.5, first_eviction_step=0,
        )
