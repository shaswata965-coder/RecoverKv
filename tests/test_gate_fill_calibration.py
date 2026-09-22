"""The windows the gate skips must not inherit the mass of the ones it read.

Two defects behind the RULER 32k regression, both on the flash path's decode:

1. **The skipped-window fill was calibrated by a ratio of sums.** A skipped int2
   window is credited ``c * exp(logmass)``, and ``c`` was
   ``sum(exact) / sum(estimate)`` over the windows the gate read. On a retrieval
   step -- a head copying one token out of a read int2 window -- ``sum(exact)`` IS
   that token, which a rank-1 card does not model, so ``c`` became that token's
   estimation error and every skipped window was credited with a share of the
   needle's mass. The same weight is what the skipped windows ATTEND with
   (their value centroids), so the needle's value was diluted by ``1 + sum(fill)``
   and replaced by a blend of centroids -- the "right first few characters, then
   invented" failure. ``c`` is now the mean LOG ratio.

2. **The window scores were accumulated in fp16.** Nearly every decode step's
   evidence was below half an ULP of the prefill total and rounded away.

Everything here runs the repo's own CPU oracle of the fused kernel
(``two_tier_window_reference``) with real cards (``build_sketch`` /
``gate_and_score``), so it pins the algorithm the Triton kernel mirrors.
"""

from __future__ import annotations

import math

import pytest
import torch

from modules.quant.sketch import (
    Sketch, build_sketch, gate_and_score, group_max, value_centroid,
)
from modules.windowed_cache.decode_kernel import two_tier_window_reference
from modules.windowed_cache.scorer import fill_skipped_window_scores

D, H_KV, REP, SINK = 64, 2, 4, 5


def _scenario(seed, *, sharp, n_q=160, n_body=12, ws=8, where="q"):
    """A two-tier store whose query puts ``sharp`` on one hot token.

    Keys carry a common mode (massive channels) plus per-window and per-token
    variation; the hot token is a RANDOM token of a random window -- not the
    farthest point the card's direction is built on, which is the case a
    retrieval step actually presents.
    """
    g = torch.Generator().manual_seed(seed)
    anc = torch.zeros(H_KV, D)
    anc[:, :4] = 8.0 * torch.randn(H_KV, 4, generator=g)

    def keys(nw):
        return (anc[None, :, None, :]
                + 0.6 * torch.randn(nw, H_KV, 1, D, generator=g)
                + torch.randn(nw, H_KV, ws, D, generator=g))

    kq, kb = keys(n_q), keys(n_body)
    ks = anc[None, :, None, :] + torch.randn(1, H_KV, SINK, D, generator=g)
    vq = torch.randn(n_q, H_KV, ws, D, generator=g)
    vb = torch.randn(n_body, H_KV, ws, D, generator=g)
    vs = torch.randn(1, H_KV, SINK, D, generator=g)

    src, nw = (kq, n_q) if where == "q" else (kb, n_body)
    hw = int(torch.randint(0, nw, (1,), generator=g))
    ht = int(torch.randint(0, ws, (1,), generator=g))
    q = 0.3 * torch.randn(1, H_KV * REP, D, generator=g)
    for h in range(H_KV):
        kt = src[hw, h, ht] - anc[h]
        q[0, h * REP:(h + 1) * REP] += sharp * kt / kt.norm()

    def flat(x):
        return x.permute(1, 0, 2, 3).reshape(1, H_KV, -1, D)

    v_anc = vq.mean(dim=(0, 2))
    return dict(
        q=q, ws=ws, n_q=n_q, n_body=n_body, hot_window=hw, where=where,
        k=torch.cat([ks, flat(kb), flat(kq)], dim=2),
        v=torch.cat([vs, flat(vb), flat(vq)], dim=2),
        card=Sketch(*(t.unsqueeze(0) for t in build_sketch(kq, anc, vq, v_anc))),
        anc=anc, v_anc=v_anc,
    )


def _step(d, ratio=0.25):
    """Run one gated decode step on the kernel oracle.

    Returns a dict with the gated and exact outputs, the output with no
    centroids, the mass the step CREDITED to the windows it skipped (before the
    output renormalisation), the mass those windows TRULY hold on the read
    set's softmax footing, and whether the hot window was read.
    """
    q, ws, n_q = d["q"], d["ws"], d["n_q"]
    scaling = D ** -0.5
    logmass, est = gate_and_score(q, d["card"], d["anc"], scaling)
    n_sel = max(1, math.ceil(ratio * n_q))
    sel = group_max(est, H_KV).topk(n_sel, dim=-1).indices.sort(-1).values
    cent = (value_centroid(d["card"], d["v_anc"]).permute(0, 2, 1, 3)
            .repeat_interleave(REP, dim=1))
    sfp = SINK + d["n_body"] * ws
    common = (q, d["k"], d["v"], scaling, SINK, ws, d["n_body"])
    out_exact, _ = two_tier_window_reference(*common, Sfp=sfp)
    out, _ = two_tier_window_reference(
        *common, Sfp=sfp, sel=sel, logmass=logmass, centroids=cent)
    # Without centroids the fill still runs but nothing is renormalised, so
    # the skipped columns hold exactly the mass the step credited them.
    out_none, wsum = two_tier_window_reference(
        *common, Sfp=sfp, sel=sel, logmass=logmass)

    read = torch.zeros(1, H_KV, n_q, dtype=torch.bool)
    read.scatter_(-1, sel, True)
    read = read.repeat_interleave(REP, dim=1)
    credited = torch.where(read, 0.0, wsum[..., d["n_body"]:]).sum(-1)

    logits = torch.einsum("bhd,bhsd->bhs", q,
                          d["k"].repeat_interleave(REP, dim=1)) * scaling
    per_win = logits[..., sfp:].reshape(1, -1, n_q, ws).logsumexp(-1)
    lse_read = torch.cat([logits[..., :sfp],
                          torch.where(read, per_win, float("-inf"))],
                         dim=-1).logsumexp(-1, keepdim=True)
    true_skip = torch.where(read, 0.0, (per_win - lse_read).exp()).sum(-1)
    return dict(out=out, exact=out_exact, out_none=out_none, credited=credited,
                true_skip=true_skip, read=read, wsum=wsum, logmass=logmass,
                hot_read=bool(read[0, 0, d["hot_window"]]))


def _rel(a, b):
    return ((a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1)).mean().item()


def _old_fill_credit(r, n_body):
    """The ratio-of-sums fill this file replaces, recomputed from the same step."""
    read, logmass = r["read"], r["logmass"]
    num = torch.where(read, r["wsum"][..., n_body:], 0.0).sum(-1, keepdim=True)
    rel = (logmass - torch.where(read, logmass, float("-inf"))
           .amax(-1, keepdim=True)).exp()
    fill = num * rel / (rel * read).sum(-1, keepdim=True)
    return torch.where(read, 0.0, fill).sum(-1)


# ---------------------------------------------------------------------------
# The regression: a peaked head reading a needle out of the int2 tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("sharp", [0.0, 8.0, 12.0, 20.0])
def test_skipped_windows_are_credited_the_mass_they_hold(seed, sharp):
    """The credited mass must track what the skipped windows TRULY hold, from a
    diffuse head (they hold most of it) to a retrieval head (they hold ~none).

    Under the old ratio-of-sums fill a retrieval head -- one token of a READ
    int2 window holding most of the mass -- credited the skipped windows a share
    of that token: 0.2-1.3 of extra mass on steps where they truly held < 0.05.
    """
    r = _step(_scenario(seed, sharp=sharp))
    got, want = r["credited"], r["true_skip"]
    assert torch.all((got - want).abs() <= 0.25 * want + 0.02), (got, want)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("sharp", [8.0, 12.0, 20.0])
def test_a_needle_in_a_read_window_is_not_diluted(seed, sharp):
    """The case RULER is made of. The skipped windows ATTEND at the weight they
    are credited, so over-crediting them diluted the needle's value by
    ``1 + credit`` and blended in centroids -- output error 0.25-0.53 at ratio
    0.25 under the old fill. It must be the needle's value."""
    r = _step(_scenario(seed, sharp=sharp))
    assert r["hot_read"], "scenario must exercise a READ needle window"
    assert _rel(r["out"], r["exact"]) < 0.05


def test_the_old_fill_is_what_this_file_catches():
    """Guard the guard: the ratio-of-sums fill, on the same step, breaks the
    credited-mass contract above. If this ever stops failing, the scenario no
    longer discriminates and the regression tests prove nothing."""
    d = _scenario(0, sharp=12.0)
    r = _step(d)
    old = _old_fill_credit(r, d["n_body"])
    assert r["true_skip"].max().item() < 0.05
    assert old.max().item() > 0.2


@pytest.mark.parametrize("seed", range(4))
def test_a_needle_in_the_fp_tier_is_unchanged(seed):
    """When the needle is exact (fp tier), the read Q windows hold little and
    the fill has nothing to transfer; the result is accurate either way."""
    r = _step(_scenario(seed, sharp=12.0, where="fp"))
    assert _rel(r["out"], r["exact"]) < 0.05


# ---------------------------------------------------------------------------
# What the fill is FOR must still work: diffuse heads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_diffuse_heads_still_attend_through_skipped_windows(seed):
    """With no peak, the skipped 75% hold most of the mass, and the centroids are
    what keep the output close. The fix must not turn that off."""
    r = _step(_scenario(seed, sharp=0.0))
    assert r["credited"].min().item() > 0.5
    assert _rel(r["out"], r["exact"]) < 0.5 * _rel(r["out_none"], r["exact"])


# ---------------------------------------------------------------------------
# The spec: scorer.fill_skipped_window_scores
# ---------------------------------------------------------------------------


def _spec_inputs(n=40, n_keep=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    logmass = torch.randn(1, 2, n, generator=g)
    keep = torch.zeros(1, 2, n, dtype=torch.bool)
    keep[..., :n_keep] = True
    return logmass, keep


def test_spec_recovers_a_common_factor_exactly():
    """A card off by one constant factor is corrected exactly -- the property the
    self-calibration exists for, kept by the log form."""
    logmass, keep = _spec_inputs()
    exact = 3.7 * logmass.exp()
    out = fill_skipped_window_scores(torch.where(keep, exact, 0.0), keep, logmass)
    torch.testing.assert_close(out, exact, rtol=1e-5, atol=0)


def test_spec_one_peaked_read_window_moves_c_by_its_share_only():
    """One read window 1e6x its estimate -- a needle -- moves the correction by
    ln(1e6)/n_keep in the log, not by a factor ~1e6/n_keep."""
    logmass, keep = _spec_inputs(n_keep=10)
    exact = logmass.exp().clone()
    exact[..., 3] *= 1e6
    out = fill_skipped_window_scores(torch.where(keep, exact, 0.0), keep, logmass)
    want_c = math.exp(math.log(1e6) / 10)
    got_c = out[..., 10:] / logmass[..., 10:].exp()
    torch.testing.assert_close(got_c, torch.full_like(got_c, want_c), rtol=1e-4, atol=0)


def test_spec_a_zero_read_window_is_left_out_not_fatal():
    """exact == 0 has no finite log. It must be skipped, not drag c to zero."""
    logmass, keep = _spec_inputs(n_keep=10)
    exact = 2.0 * logmass.exp()
    exact[..., 0] = 0.0
    out = fill_skipped_window_scores(torch.where(keep, exact, 0.0), keep, logmass)
    torch.testing.assert_close(out[..., 10:], 2.0 * logmass[..., 10:].exp(),
                               rtol=1e-5, atol=0)


def test_spec_and_kernel_oracle_agree():
    """The kernel oracle's skipped-column scores are the spec's fill."""
    d = _scenario(1, sharp=10.0)
    r = _step(d)
    qcols = r["wsum"][..., d["n_body"]:]
    spec = fill_skipped_window_scores(torch.where(r["read"], qcols, 0.0),
                                      r["read"], r["logmass"])
    torch.testing.assert_close(qcols, spec, rtol=1e-4, atol=1e-12)


# ---------------------------------------------------------------------------
# Defect 2: the running window score is fp32, whatever the KV dtype
# ---------------------------------------------------------------------------


def test_hook_hands_the_cache_fp32_scores(monkeypatch):
    """The prefill scores seed ``state.window_scores`` and fix its dtype for the
    whole decode. At fp16 a decode step's mass (< 1e-2 per window) is below half
    an ULP of a 32k-row prefill total and rounds to nothing, so every
    re-eviction ranked on the prompt alone. Drive the real hook over an fp16
    model and check what it hands the cache."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from modules.windowed_cache import hooks
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig
    from modules.windowed_cache.score_kernel import token_scores_torch

    seen = []

    def fake_scores(q, k, scaling, *, lse=None, chunk=1024,
                    softmax_dtype=torch.bfloat16, out_dtype=None):
        # The Triton prefill kernel cannot run here; its oracle can. What is
        # under test is the dtype the hook ASKS for.
        seen.append(out_dtype)
        return token_scores_torch(q, k, scaling).to(out_dtype or q.dtype)

    monkeypatch.setattr(hooks, "compute_token_scores", fake_scores)
    monkeypatch.setattr(hooks, "_install_lse_source", lambda h: (False, "off"))

    cfg = LlamaConfig(vocab_size=64, hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, max_position_embeddings=512,
                      attn_implementation="eager")
    model = LlamaForCausalLM(cfg).to(torch.float16).eval()
    prompt = 64
    cache_cfg = WindowedCacheConfig(window_size=8, num_sink_tokens=4,
                                    local_window_size=16, cache_budget=0.5,
                                    quant_ratio=0.0)
    cache = WindowedCache(config=cache_cfg, prefill_len=prompt, model_config=cfg,
                          kv_dtype=torch.float16,
                          rope_module=model.model.rotary_emb,
                          num_layers=cfg.num_hidden_layers, max_tokens=8)
    handles = hooks.install_score_hooks(model, cache, cache_cfg)
    try:
        with torch.no_grad():
            model(torch.randint(0, 64, (1, prompt)), past_key_values=cache,
                  use_cache=True)
    finally:
        handles.remove()

    assert seen and all(dt is torch.float32 for dt in seen)
    for i in range(cfg.num_hidden_layers):
        assert cache.cache_kwargs[i]["window_scores"].dtype is torch.float32

    # And the point of it: a decode step's mass survives the running sum.
    total = torch.full((1,), 1024.0)
    step = torch.full((1,), 1e-3)
    assert (total + step).item() != 1024.0
    assert (total.half() + step.half()).item() == 1024.0
