"""The completion-term ablation (``quant_completion``), pinned on CPU.

Three layers, each checked against something it does not share code with:

* the **spec** -- :func:`two_tier_window_reference` with ``completion=``, which
  computes every variant directly from the oracle's own read-set output and
  ``lse``;
* the **production epilogue** -- :func:`completion.apply_completion`, which
  recovers ``o_bar``, the fill and ``log Z_A`` by INVERTING the kernel's
  calibrated output. Its result must match the spec on real cards
  (``build_sketch``) and the shipped gate (``gate_reference``);
* the **plumbing** -- config validation, the factory that every runner builds its
  cache through, the sidecar record, ``_run_fused`` (CPU gate reference,
  stubbed decode kernel, real cache), and the cache refusing to run the knob on
  a path that has no epilogue.

Plus brute-force checks of the two variants that have a closed form: ``off`` is
the read-set attention, and ``uncalibrated`` is ``exp(logmass - log Z_A)``.
"""
import math
import types

import pytest
import torch

from modules.quant.sketch import value_centroid
from modules.windowed_cache.completion import (apply_completion, variant_mass,
                                               window_noise)
from modules.windowed_cache.decode_kernel import two_tier_window_reference
from modules.windowed_cache.gate_kernel import gate_reference
from tests.test_gate_fill_calibration import D, H_KV, REP, SINK, _scenario
from utils.completion_spec import (CompletionSpec, completion_record,
                                   parse_completion)

H_Q = H_KV * REP
LAYER = 3


def _setup(seed=0, sharp=0.0, where="q", ratio=0.25, exp2=False):
    d = _scenario(seed, sharp=sharp, where=where)
    q, ws, n_q = d["q"], d["ws"], d["n_q"]
    scaling = D ** -0.5
    n_sel = max(1, math.ceil(ratio * n_q))
    sel, logmass = gate_reference(q, *d["card"], d["anc"], scaling, n_sel)
    cent = (value_centroid(d["card"], d["v_anc"]).permute(0, 2, 1, 3)
            .repeat_interleave(REP, dim=1))
    sfp = SINK + d["n_body"] * ws
    common = (q, d["k"], d["v"], scaling, SINK, ws, d["n_body"])
    kw = dict(Sfp=sfp, exp2=exp2, sel=sel, logmass=logmass, centroids=cent)
    wids = (torch.arange(n_q) * 7 + 3).unsqueeze(0)            # stable, non-trivial ids
    return dict(d=d, q=q, sel=sel, logmass=logmass, cent=cent, sfp=sfp,
                common=common, kw=kw, scaling=scaling, wids=wids)


def _ref(s, spec):
    comp = None
    if spec is not None:
        comp = dict(spec._asdict())
        if spec.sigma > 0:
            comp["eps"] = window_noise(spec.seed, LAYER, s["wids"], H_Q)
    return two_tier_window_reference(*s["common"], **s["kw"], completion=comp)


def _epilogue(s, spec, out_cal, w_cal):
    d = s["d"]
    return apply_completion(
        spec, out=out_cal, wsum=w_cal, n_body_win=d["n_body"], sel=s["sel"],
        logmass=s["logmass"], centroids=(d["card"].vm_q, d["card"].vm_s, d["v_anc"]),
        vm_bits=4, q=s["q"], k_fp=d["k"][:, :, :s["sfp"]], scaling=s["scaling"],
        wids=s["wids"], layer_idx=LAYER)


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


SPECS = [
    CompletionSpec("calibrated"),
    CompletionSpec("off"),
    CompletionSpec("uncalibrated"),
    CompletionSpec("calibrated", scale=0.5),
    CompletionSpec("calibrated", scale=2.0),
    CompletionSpec("calibrated", scale=4.0),
    CompletionSpec("calibrated", sigma=0.7, seed=1),
    CompletionSpec("calibrated", scale=2.0, scores="coupled"),
    CompletionSpec("calibrated", sigma=1.5, seed=2, scores="coupled"),
    CompletionSpec("uncalibrated", scores="coupled"),
]


# ---------------------------------------------------------------------------
# the production epilogue against the spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.mode}-c{s.scale}-s{s.sigma}-{s.scores}")
@pytest.mark.parametrize("sharp,where", [(0.0, "q"), (8.0, "q"), (12.0, "fp")])
@pytest.mark.parametrize("exp2", [False, True])
def test_epilogue_matches_the_spec(spec, sharp, where, exp2):
    for seed in range(3):
        s = _setup(seed, sharp=sharp, where=where, exp2=exp2)
        out_cal, w_cal = _ref(s, None)
        out_ref, w_ref = _ref(s, spec)
        out_ep, w_ep = _epilogue(s, spec, out_cal, w_cal)
        assert _rel(out_ep, out_ref) < 2e-5, (seed, _rel(out_ep, out_ref))
        assert torch.allclose(w_ep, w_ref, rtol=2e-5, atol=1e-7), seed  # mass units: fp32 noise on O(1) normalisers


@pytest.mark.parametrize("ratio", [0.10, 0.50, 1.0])
def test_epilogue_matches_the_spec_at_other_read_ratios(ratio):
    s = _setup(4, sharp=4.0, ratio=ratio)
    out_cal, w_cal = _ref(s, None)
    for spec in SPECS:
        out_ref, w_ref = _ref(s, spec)
        out_ep, w_ep = _epilogue(s, spec, out_cal, w_cal)
        assert _rel(out_ep, out_ref) < 2e-5, spec
        assert torch.allclose(w_ep, w_ref, rtol=2e-5, atol=1e-7), spec


def test_calibrated_at_scale_one_is_the_identity():
    for seed in range(4):
        s = _setup(seed, sharp=6.0)
        out_cal, w_cal = _ref(s, None)
        out_ep, w_ep = _epilogue(s, CompletionSpec("calibrated"), out_cal, w_cal)
        assert _rel(out_ep, out_cal) < 1e-6
        assert w_ep is w_cal                     # scores untouched, not recomputed


def test_off_is_attention_over_the_read_set_alone():
    """Independent of the spec: the oracle with no centroids IS o_bar."""
    for seed in range(4):
        s = _setup(seed, sharp=3.0)
        out_cal, w_cal = _ref(s, None)
        kw = {k: v for k, v in s["kw"].items() if k != "centroids"}
        o_bar, _ = two_tier_window_reference(*s["common"], **kw)
        out_ep, _ = _epilogue(s, CompletionSpec("off"), out_cal, w_cal)
        assert _rel(out_ep, o_bar) < 2e-6


def test_uncalibrated_is_the_card_over_the_exact_read_set_normaliser():
    """Brute force: logits over A, then exp(logmass - lse_A) per skipped window."""
    for seed in range(4):
        s = _setup(seed, sharp=5.0)
        d, q, sfp = s["d"], s["q"], s["sfp"]
        out_cal, w_cal = _ref(s, None)
        read = torch.zeros(1, H_KV, d["n_q"], dtype=torch.bool)
        read.scatter_(-1, s["sel"], True)
        read = read.repeat_interleave(REP, dim=1)
        logits = torch.einsum("bhd,bhsd->bhs", q,
                              d["k"].repeat_interleave(REP, dim=1)) * s["scaling"]
        per_win = logits[..., sfp:].reshape(1, H_Q, d["n_q"], d["ws"]).logsumexp(-1)
        lse_a = torch.cat([logits[..., :sfp],
                           torch.where(read, per_win, float("-inf"))], -1).logsumexp(-1, keepdim=True)
        m = torch.where(read, 0.0, (s["logmass"] - lse_a).exp())
        kw = {k: v for k, v in s["kw"].items() if k != "centroids"}
        o_bar, _ = two_tier_window_reference(*s["common"], **kw)
        want = (o_bar + torch.einsum("bhn,bhnd->bhd", m, s["cent"])) / (1 + m.sum(-1, keepdim=True))
        out_ep, _ = _epilogue(s, CompletionSpec("uncalibrated"), out_cal, w_cal)
        assert _rel(out_ep, want) < 2e-5


def test_the_variants_actually_differ():
    """A guard against an epilogue that silently returns its input."""
    s = _setup(1, sharp=0.0)
    out_cal, w_cal = _ref(s, None)
    outs = {spec: _epilogue(s, spec, out_cal, w_cal)[0] for spec in SPECS[:7]}
    base = outs[CompletionSpec("calibrated")]
    for spec, o in outs.items():
        if spec != CompletionSpec("calibrated"):
            assert _rel(o, base) > 1e-4, spec


def test_output_scores_keep_the_shipped_fill_and_coupled_moves_it():
    s = _setup(2, sharp=0.0)
    out_cal, w_cal = _ref(s, None)
    _, w_out = _epilogue(s, CompletionSpec("calibrated", scale=2.0), out_cal, w_cal)
    _, w_cpl = _epilogue(s, CompletionSpec("calibrated", scale=2.0, scores="coupled"), out_cal, w_cal)
    assert w_out is w_cal
    nb = s["d"]["n_body"]
    skipped_cal = w_cal[..., nb:].sum(-1)
    assert bool((w_cpl[..., nb:].sum(-1) > skipped_cal).all())
    # every coupled step is still a distribution over [body | Q] plus the sinks
    assert bool((w_cpl.sum(-1) <= 1 + 1e-6).all())


# ---------------------------------------------------------------------------
# the noise
# ---------------------------------------------------------------------------


def test_noise_is_a_pure_function_of_seed_layer_head_and_window():
    wids = torch.tensor([[5, 900, 17, 3]])
    a = window_noise(7, 2, wids, 8)
    assert torch.equal(a, window_noise(7, 2, wids, 8))
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.equal(window_noise(7, 2, wids[:, perm], 8), a[..., perm])   # follows the window
    assert not torch.equal(a, window_noise(8, 2, wids, 8))
    assert not torch.equal(a, window_noise(7, 3, wids, 8))
    assert not torch.equal(a[:, 0], a[:, 1])


def test_noise_is_standard_normal_and_independent_across_heads():
    z = window_noise(0, 0, torch.arange(40000).unsqueeze(0), 4)[0]   # [H, n]
    assert z.mean(-1).abs().max() < 0.03
    assert (z.std(-1) - 1).abs().max() < 0.03
    assert abs(float(((z.abs() < 1.959964).float().mean()) - 0.95)) < 0.01
    c = torch.corrcoef(z)
    assert (c - torch.eye(4)).abs().max() < 0.03


def test_noise_preserves_the_skipped_total_exactly():
    g = torch.Generator().manual_seed(0)
    fill = torch.rand(2, 4, 50, generator=g) * 0.01
    skip = torch.rand(2, 4, 50, generator=g) > 0.25
    fill = torch.where(skip, fill, 0.0)
    eps = torch.randn(2, 4, 50, generator=g)
    for scale in (1.0, 2.0):
        spec = CompletionSpec("calibrated", scale=scale, sigma=1.0)
        m = variant_mass(spec, fill, skip, torch.zeros_like(fill), None, eps)
        assert torch.allclose(m.sum(-1), scale * fill.sum(-1), rtol=1e-5)
        assert bool((m[~skip] == 0).all())
        assert not torch.allclose(m, fill * scale)


# ---------------------------------------------------------------------------
# config, factory, sidecar
# ---------------------------------------------------------------------------


def test_the_spec_refuses_anything_inert_or_ambiguous():
    assert parse_completion() is None
    assert parse_completion("off") == CompletionSpec("off")
    assert parse_completion("calibrated", 2, 0.5, 3, "coupled") == CompletionSpec(
        "calibrated", 2.0, 0.5, 3, "coupled")
    bad = [
        dict(mode=None, scale=2.0),                       # set, but null runs the kernel
        dict(mode=None, scores="coupled"),
        dict(mode="none"),                                # the override parser's trap
        dict(mode="sideways"),
        dict(mode="off", scale=2.0),
        dict(mode="uncalibrated", sigma=0.5),
        dict(mode="off", scores="coupled"),               # would evict the int2 tier
        dict(mode="calibrated", scale=0.0),
        dict(mode="calibrated", scale=float("inf")),
        dict(mode="calibrated", sigma=-1.0),
        dict(mode="calibrated", seed=True),
        dict(mode="calibrated", scale=True),
        dict(mode="calibrated", scores="both"),
    ]
    for b in bad:
        with pytest.raises(ValueError):
            parse_completion(**b)


def test_experiment_config_validates():
    from utils.config import CacheConfig, ConfigValidationError
    CacheConfig(quant_ratio=0.7, quant_completion="off")
    with pytest.raises(ConfigValidationError):
        CacheConfig(quant_ratio=0.7, quant_completion="none")
    with pytest.raises(ConfigValidationError):
        CacheConfig(quant_ratio=0.0, quant_completion="off")      # no int2 tier
    with pytest.raises(ConfigValidationError):
        CacheConfig(quant_ratio=0.7, quant_completion_scale=2.0)   # mode unset


def test_cache_config_resolves_it_and_warns():
    from modules.windowed_cache.config import WindowedCacheConfig
    base = dict(window_size=4, num_sink_tokens=0, local_window_size=4,
                cache_budget=0.35, quant_ratio=0.5)
    mc = types.SimpleNamespace(num_key_value_heads=2, num_attention_heads=4,
                               hidden_size=64, head_dim=16)
    assert WindowedCacheConfig(**base).resolve(64, mc, torch.float32, 16).quant_completion is None
    with pytest.warns(UserWarning, match="ablation of the method"):
        c = WindowedCacheConfig(**base, quant_completion="calibrated",
                                quant_completion_scale=2.0)
    assert c.resolve(64, mc, torch.float32, 16).quant_completion == CompletionSpec("calibrated", 2.0)
    with pytest.raises(ValueError, match="quant_ratio is 0"):
        WindowedCacheConfig(**{**base, "quant_ratio": 0.0}, quant_completion="off")


def test_the_factory_threads_it_and_the_sidecar_records_it():
    from utils.cache_factory import quant_tier_policy_kwargs, quant_tier_policy_record
    from utils.config import CacheConfig
    assert quant_tier_policy_kwargs(CacheConfig(quant_ratio=0.7)) == {}
    c = CacheConfig(quant_ratio=0.7, quant_completion="calibrated",
                    quant_completion_sigma=0.4, quant_completion_seed=2)
    assert quant_tier_policy_kwargs(c) == dict(
        quant_completion="calibrated", quant_completion_scale=1.0,
        quant_completion_sigma=0.4, quant_completion_seed=2,
        quant_completion_scores="output")
    assert quant_tier_policy_record(c)["quant_completion"] == dict(
        mode="calibrated", scale=1.0, sigma=0.4, seed=2, scores="output")
    assert quant_tier_policy_record(CacheConfig())["quant_completion"]["mode"] == "native"
    assert completion_record(None)["mode"] == "native"


def test_every_runner_override_reaches_the_cache_config():
    """The exact override strings the notebook passes, through main.py's own
    parser and load_config, land on CacheConfig and in the factory kwargs."""
    import main as M
    from utils.cache_factory import quant_tier_policy_kwargs
    from utils.config import load_config
    items = ["cache.quant_completion=off", "cache.quant_completion_scale=1.0",
             "cache.quant_completion_sigma=0.0", "cache.quant_completion_seed=0",
             "cache.quant_completion_scores=output"]
    ov = {"cache": {i.split("=")[0].split(".")[1]: M._parse_value(i.split("=")[1]) for i in items}}
    cfg = load_config("configs/longbench_ours_flash_attn.yaml", overrides=ov)
    assert cfg.cache.quant_completion == "off"
    assert quant_tier_policy_kwargs(cfg.cache)["quant_completion"] == "off"


# ---------------------------------------------------------------------------
# _run_fused and the cache
# ---------------------------------------------------------------------------


class _Rope(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x, position_ids):
        f = position_ids.to(torch.float32).unsqueeze(-1) * torch.arange(
            1, self.d // 2 + 1, dtype=torch.float32) * 0.01
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()


def _cache(**knob):
    from modules.windowed_cache.cache import WindowedCache
    from modules.windowed_cache.config import WindowedCacheConfig
    L, h_kv, h_q, d, ws, P = 2, 2, 4, 16, 4, 32
    cfg = types.SimpleNamespace(num_key_value_heads=h_kv, num_attention_heads=h_q,
                                hidden_size=h_q * d, head_dim=d)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = WindowedCache(
            config=WindowedCacheConfig(
                window_size=ws, num_sink_tokens=0, local_window_size=ws,
                cache_budget=0.35, quant_ratio=0.5, first_eviction_step=0,
                quant_gate_ratio=0.5, **knob),
            prefill_len=P, model_config=cfg, kv_dtype=torch.float32,
            rope_module=_Rope(d), num_layers=L, max_tokens=P + 64)
    return c, dict(L=L, h_kv=h_kv, h_q=h_q, d=d, ws=ws, P=P)


def _drive(cache, g, dims, on_decode):
    L, h_kv, h_q, d, ws, P = (dims[k] for k in ("L", "h_kv", "h_q", "d", "ws", "P"))
    Bt = 1

    def merged(i):
        st, store = cache._states[i], cache._stores[i]
        return -(-st.seq_length // ws) + store.num_active_windows

    for i in range(L):
        k = torch.randn(Bt, h_kv, P, d, generator=g)
        cache.update(k, torch.randn_like(k), i, cache_kwargs={"cache_position": torch.arange(P)})
    for i in range(L):
        cache.cache_kwargs[i]["window_scores"] = torch.rand(Bt, h_q, merged(i), generator=g)
    for t in range(3):
        for i in range(L):
            k = torch.randn(Bt, h_kv, 1, d, generator=g)
            k_fp, v_fp = cache.update(k, torch.randn_like(k), i,
                                      cache_kwargs={"cache_position": torch.tensor([P + t])})
            if not on_decode(i, k_fp, v_fp):
                cache.cache_kwargs[i]["window_scores"] = torch.rand(Bt, h_q, merged(i), generator=g)


@pytest.mark.parametrize("knob", [{}, {"quant_completion": "off"},
                                  {"quant_completion": "calibrated", "quant_completion_sigma": 0.5}])
def test_run_fused_applies_it_on_every_gated_layer_and_only_when_set(monkeypatch, knob):
    from modules.windowed_cache import flash_decode
    cache, dims = _cache(**knob)
    cache._fused_decode_active = True
    armed, calls = [], []

    def stub_decode(q_hd, k_fp, v_fp, qtier, scaling, num_sink, n_body_win, *,
                    sel, logmass, centroids, vm_bits=4, **extra):
        calls.append(extra)
        w = n_body_win + qtier["k_codes"].shape[1]
        dt = extra.get("out_dtype") or q_hd.dtype
        return (torch.randn(q_hd.shape, dtype=dt),
                torch.rand(q_hd.shape[0], q_hd.shape[1], w) / (2 * w))

    monkeypatch.setattr(flash_decode, "set_pending", armed.append)
    monkeypatch.setattr(flash_decode, "fused_two_tier_decode", stub_decode)
    flash_decode.reset_stats()
    g = torch.Generator().manual_seed(0)
    outs = []

    def on_decode(i, k_fp, v_fp):
        if not armed:
            return False
        ctx = armed.pop()
        q = torch.randn(1, 1, dims["h_q"], dims["d"], generator=g)
        o = flash_decode._run_fused(ctx, q, k_fp.transpose(1, 2), v_fp.transpose(1, 2))
        outs.append(o)
        return True

    _drive(cache, g, dims, on_decode)
    st = flash_decode.stats()
    assert st["gated"] > 0
    if knob:
        assert st["completion_applied"] == st["gated"]
        assert all(c == {"out_dtype": torch.float32} for c in calls)
    else:
        assert st["completion_applied"] == 0
        assert all(c == {} for c in calls)                 # the pre-knob call, exactly
    assert all(o.dtype == torch.float32 and torch.isfinite(o).all() for o in outs)


def test_the_cache_refuses_the_knob_on_a_path_with_no_epilogue():
    cache, dims = _cache(quant_completion="off")
    cache._fused_decode_active = False                    # eager / fused off
    g = torch.Generator().manual_seed(0)
    with pytest.raises(RuntimeError, match="quant_completion is set"):
        _drive(cache, g, dims, lambda *a: False)


def test_the_default_cache_still_takes_the_materialize_path():
    cache, dims = _cache()
    cache._fused_decode_active = False
    _drive(cache, torch.Generator().manual_seed(0), dims, lambda *a: False)


# ---------------------------------------------------------------------------
# Experiment 1b: every rung on one state, against the same oracle
# ---------------------------------------------------------------------------


def _rungs(s):
    from modules.windowed_cache.completion import completion_rungs
    d, sfp = s["d"], s["sfp"]
    read_kv = torch.zeros(H_KV, d["n_q"], dtype=torch.bool)
    read_kv.scatter_(-1, s["sel"][0], True)
    nu = value_centroid(d["card"], d["v_anc"])[0].permute(1, 0, 2)          # [Hk, n, D]
    return completion_rungs(
        s["q"][0], d["k"][0, :, :sfp], d["v"][0, :, :sfp], d["k"][0, :, sfp:],
        d["v"][0, :, sfp:], d["ws"], read_kv, s["logmass"][0], nu, s["scaling"],
        s["wids"][0], LAYER, noise_seed=0)


@pytest.mark.parametrize("sharp,where", [(0.0, "q"), (8.0, "q"), (12.0, "fp")])
def test_every_rung_matches_the_spec_on_the_same_state(sharp, where):
    from modules.windowed_cache.completion import COMPLETION_RUNGS
    for seed in range(3):
        s = _setup(seed, sharp=sharp, where=where)
        r = _rungs(s)
        got = dict(zip(COMPLETION_RUNGS, r["outs"]))
        want = {"cal": _ref(s, None)[0][0],
                "uncal": _ref(s, CompletionSpec("uncalibrated"))[0][0],
                "off": two_tier_window_reference(
                    *s["common"], **{k: v for k, v in s["kw"].items() if k != "centroids"})[0][0],
                "full_oracle": two_tier_window_reference(*s["common"], Sfp=s["sfp"])[0][0]}
        for k, c in (("scale0.5", 0.5), ("scale0.8", 0.8), ("scale1.25", 1.25),
                     ("scale2", 2.0), ("scale4", 4.0)):
            want[k] = _ref(s, CompletionSpec("calibrated", scale=c))[0][0]
        for k, sg in (("noise0.25", 0.25), ("noise0.5", 0.5), ("noise1", 1.0), ("noise2", 2.0)):
            want[k] = _ref(s, CompletionSpec("calibrated", sigma=sg, seed=0))[0][0]
        for k, w in want.items():
            assert _rel(got[k], w) < 2e-5, (seed, k, _rel(got[k], w))


def test_the_oracles_and_the_per_window_masses():
    from modules.windowed_cache.completion import COMPLETION_RUNGS
    s = _setup(5, sharp=6.0)
    r = _rungs(s)
    got = dict(zip(COMPLETION_RUNGS, r["outs"]))
    read, skip = r["read"], ~r["read"]
    # delta is fitted on the read windows, so their residual averages to zero
    assert float(torch.where(read, r["logerr_cal"], 0.0).sum(-1).abs().max()) < 1e-4
    # the exact masses, on Z_A: read + fp tier sum to 1
    kw = {k: v for k, v in s["kw"].items() if k != "centroids"}
    _, w_none = two_tier_window_reference(*s["common"], **kw)
    nb = s["d"]["n_body"]
    assert torch.allclose(torch.where(read, r["log_mstar"].exp(), 0.0),
                          torch.where(read, w_none[0, :, nb:], 0.0), rtol=1e-4, atol=1e-7)
    # the uncalibrated shift is log m~_uncal - log m~_cal, per head
    lm = s["logmass"][0]
    assert torch.allclose((lm - r["log_za"][:, None]) - (lm + r["delta"][:, None]),
                          r["uncal_shift"][:, None].expand_as(lm))
    # the mass oracle is the exact masses through the stored centroids
    m = torch.where(skip, r["log_mstar"].exp(), 0.0)
    o_bar = got["off"]
    want = (o_bar + torch.einsum("hn,hnd->hd", m, s["cent"][0])) / (1 + m.sum(-1, keepdim=True))
    assert _rel(got["mass_oracle"], want) < 2e-5
