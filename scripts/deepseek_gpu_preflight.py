#!/usr/bin/env python3
"""Check DeepSeek-V2's GPU-only paths before a long run.

Everything the DeepSeek-V2-Lite port runs only on a GPU is checked here, on a
small random model with V2-Lite's REAL attention geometry (``kv_lora_rank``
512, RoPE 64, nope 128, v 128, 16 heads, YaRN, read from the checkpoint's own
``config.json``; only width, depth and the MoE are shrunk). Each check compares
a GPU path with the CPU path the repo's tests already pin:

1. **Structure.** The owned MLA forward -- FA2 prefill over the decompressed
   heads, absorbed decode over the latent -- against the UNMODIFIED pinned
   upstream eager forward on the same weights (teacher-forced logits).
2. **Prefill scores.** The Triton score kernel at the zero-padded 256-wide head,
   fed FA2's ``softmax_lse``, against the eager path's own probabilities.
3. **Fused decode.** At ``quant_gate_ratio = 1.0`` the fused route (gate kernel +
   MLA decode kernel, through ``flash_decode.run_pending``) against the
   materialize path, through evictions, with ``armed == fired == gated``.
4. **The gate at the run's ratio.** Verdict ``gated``, realised read fraction at
   most ``ceil(ratio * n) / n``, no NaN.
5. **The compiled eviction ran** (``torch.compile``, never eager).

Exit status 0 only if every check passes; a JSON summary is printed last.

    python scripts/deepseek_gpu_preflight.py --model-dir /content/models/DeepSeek-V2-Lite-Chat --budget 0.10

``--interpret`` runs checks 3-5 on a CPU under ``TRITON_INTERPRET=1`` (the same
shims as ``tests/interp_mla_kernels.py``). That checks this script and the
kernels' arithmetic, not the GPU lowering, which is the point of running it on
the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

PINNED_MODELING_SHA = "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"


def _interpreter_shims() -> None:
    """Route the production dispatchers to the Triton kernels on CPU tensors."""
    import modules.windowed_cache.decode_kernel as dk
    import modules.windowed_cache.flash_decode as fd
    import modules.windowed_cache.gate_kernel as gk
    import modules.windowed_cache.mla_decode_kernel as mk
    from modules.quant import sketch as S

    mk._require_cuda = lambda q: None
    guard = ('    if not (_HAS_TRITON and q.is_cuda):\n'
             '        reason = "triton not installed" if not _HAS_TRITON else "not on CUDA"\n')
    src = open(dk.__file__).read()
    assert src.count(guard) == 1, "fused_two_tier_decode's guard moved; update this shim"
    exec(compile(src.replace(guard, '    if not _HAS_TRITON:\n        reason = ""\n'),
                 dk.__file__, "exec"), dk.__dict__)
    fd.fused_two_tier_decode = dk.fused_two_tier_decode
    dk._TUNE_AFTER = 10 ** 9
    fd.fused_gate = lambda q, card, anchor, scaling, n_sel, bits=None: \
        gk._gate_triton(q, card, anchor, scaling, n_sel, S._as_bits(bits))


def _tiny_config(model_dir: Path, attn: str):
    from models.deepseek_v2.upstream.configuration_deepseek import DeepseekV2Config

    d = json.loads((model_dir / "config.json").read_text())
    for k in ("architectures", "auto_map", "model_type", "transformers_version",
              "torch_dtype", "quantization_config"):
        d.pop(k, None)
    # Every attention / RoPE dimension stays the checkpoint's own; only the
    # width, depth and the MoE shrink.
    d.update(hidden_size=512, intermediate_size=1024, moe_intermediate_size=256,
             num_hidden_layers=3, n_routed_experts=8, num_experts_per_tok=2,
             n_shared_experts=1, vocab_size=4096, attn_implementation=attn)
    return DeepseekV2Config(**d)


def _models(model_dir: Path, dtype, device, own_attn: str):
    """The upstream eager reference and an identical model for the owned forward.

    The owned model is built with the attention implementation the run loads
    (``flash_attention_2`` on the GPU): the upstream model forward then nulls the
    all-ones mask exactly as in the run, and the flash class is what
    ``install_recoverkv_mla`` replaces.
    """
    from models.deepseek_v2.upstream import modeling_deepseek as up

    torch.manual_seed(0)
    ref = up.DeepseekV2ForCausalLM(_tiny_config(model_dir, "eager")).to(dtype).to(device).eval()
    own = up.DeepseekV2ForCausalLM(_tiny_config(model_dir, own_attn)).to(dtype).to(device).eval()
    own.load_state_dict(ref.state_dict())
    return ref, own


def _forced(model, cache, ids, steps):
    """Prefill, then teacher-force ``steps - 1`` decode steps. Per decode step:

    * ``attn``: every layer's attention output (after ``o_proj``) -- compared
      directly, because in a random-weight model the attention is a small term
      in the residual stream and a kernel error can vanish from the logits;
    * ``scores``: every layer's window scores for the step, as the decode path
      wrote them for the cache (the fused kernel's own output on that path);
    * ``kept``: the cache's retained window ids, so two runs are compared only
      while they hold the same windows.
    """
    B, P = ids.shape
    dev = ids.device
    layers = model.model.layers
    L = len(layers)
    grab = {}
    hooks = [layers[i].self_attn.register_forward_hook(
        lambda m, args, out, i=i: grab.__setitem__(i, out[0][:, -1].float().clone()))
        for i in range(L)]
    windowed = hasattr(cache, "_states")
    run = {"logits": [], "attn": [], "scores": [], "kept": []}
    try:
        with torch.no_grad():
            o = model(input_ids=ids[:, :P - steps], past_key_values=cache, use_cache=True,
                      attention_mask=torch.ones(B, P - steps, dtype=torch.long, device=dev),
                      position_ids=torch.arange(P - steps, device=dev).unsqueeze(0).expand(B, -1))
            run["logits"].append(o.logits[:, -1].float())
            for t in range(P - steps, P - 1):
                o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True,
                          attention_mask=torch.ones(B, t + 1, dtype=torch.long, device=dev),
                          position_ids=torch.full((B, 1), t, device=dev))
                run["logits"].append(o.logits[:, -1].float())
                run["attn"].append(torch.stack([grab[i] for i in range(L)]))
                if windowed:
                    sc = [cache.cache_kwargs[i].get("window_scores") for i in range(L)]
                    run["scores"].append(None if any(s is None for s in sc) else
                                         [s.float().clone() for s in sc])
                    run["kept"].append(torch.cat([cache._states[i].original_window_ids
                                                  for i in range(L)]).clone())
    finally:
        for h in hooks:
            h.remove()
    run["logits"] = torch.stack(run["logits"], 1)
    return run


def _compare(got, want, rel):
    """Max relative errors of two runs, over the steps both held the same windows."""
    n = len(want["attn"])
    same = n
    for k, (a, b) in enumerate(zip(got["kept"], want["kept"])):
        if a.shape != b.shape or not torch.equal(a, b):
            same = k
            break
    attn = max(((x - y).abs().max() / y.abs().max().clamp_min(1e-30)).item()
               for x, y in zip(got["attn"][:same], want["attn"][:same])) if same else float("nan")
    score = 0.0
    for x, y in zip(got["scores"][:same], want["scores"][:same]):
        if x is None or y is None:
            continue
        for xs, ys in zip(x, y):
            score = max(score, ((xs - ys).abs().sum() / ys.abs().sum().clamp_min(1e-30)).item())
    logits = (got["logits"][:, :same + 1] - want["logits"][:, :same + 1]).abs().max().item()
    agree = (got["logits"].argmax(-1) == want["logits"].argmax(-1)).float().mean().item()
    out = {"steps_compared": same, "of_steps": n, "attn_rel_err": attn,
           "score_rel_l1": score, "max_abs_dlogit": logits, "top1_agree": agree}
    return (same > 0 and attn <= rel and score <= rel), out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-dir", required=True, type=Path,
                   help="the downloaded checkpoint directory (its config.json is read)")
    p.add_argument("--budget", type=float, default=0.10)
    p.add_argument("--gate", type=float, default=0.25)
    p.add_argument("--prompt", type=int, default=3000)
    p.add_argument("--steps", type=int, default=33)
    p.add_argument("--interpret", action="store_true",
                   help="CPU, TRITON_INTERPRET=1: checks 3-5 only")
    a = p.parse_args()

    from models.deepseek_v2 import attention as A
    from models.deepseek_v2.loader import install_recoverkv_mla, make_cache
    from modules.windowed_cache import cache as cache_mod
    from modules.windowed_cache import flash_decode as fd
    from modules.windowed_cache.config import WindowedCacheConfig

    results, ok = {}, True

    def record(name, passed, **info):
        nonlocal ok
        ok = ok and bool(passed)
        results[name] = {"pass": bool(passed), **info}
        print(f"{'PASS' if passed else 'FAIL'} {name}: {info}", flush=True)

    mf = a.model_dir / "modeling_deepseek.py"
    if mf.exists():
        sha = hashlib.sha256(mf.read_bytes()).hexdigest()
        record("checkpoint remote code is the pinned oracle", sha == PINNED_MODELING_SHA,
               sha=sha[:16])

    if a.interpret:
        if os.environ.get("TRITON_INTERPRET") != "1":
            raise SystemExit("--interpret needs TRITON_INTERPRET=1 in the environment")
        _interpreter_shims()
        device, dtype = torch.device("cpu"), torch.float32
    else:
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device; this script checks the GPU paths (or use --interpret)")
        device, dtype = torch.device("cuda"), torch.bfloat16
    gpu = not a.interpret
    ref, own = _models(a.model_dir, dtype, device,
                       "flash_attention_2" if gpu else "eager")
    handle = install_recoverkv_mla(own, flash=gpu)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(10, 4000, (1, a.prompt + a.steps), generator=g).to(device)
    # bf16 on the GPU: two correct implementations of the same attention differ by
    # bf16 rounding (~1e-2 of the output's scale). fp32 under the interpreter: 1e-4.
    rel = 5e-2 if dtype == torch.bfloat16 else 1e-4

    def wcfg(budget, q, gate):
        return WindowedCacheConfig(window_size=8, num_sink_tokens=5, local_window_size=128,
                                   cache_budget=budget, quant_ratio=q, quant_gate_ratio=gate,
                                   quant_budget_mode="bytes", first_eviction_step=0)

    # 1. structure: owned (FA2 prefill + absorbed decode, latent cache) vs upstream
    if gpu:
        from transformers import DynamicCache
        A.reset_path_stats()
        want = _forced(ref, DynamicCache(), ids, a.steps)
        got = _forced(own, DynamicCache(), ids, a.steps)
        good, info = _compare(got, want, rel)
        record("1 owned forward vs upstream (latent FullKV)",
               good and info["steps_compared"] == a.steps - 1
               and A.path_stats()["prefill_flash"] > 0,
               **info, paths=A.path_stats())

    # 2. prefill scores: score kernel on FA2's lse vs the eager probabilities
    if gpu:
        def prefill_scores(flash):
            for m in A_mods:
                m._recoverkv_flash = flash
            c = make_cache(own, wcfg(1.0, 0.0, 1.0), prefill_len=a.prompt, max_tokens=8,
                           fused_decode=False)
            with torch.no_grad():
                own(input_ids=ids[:, :a.prompt], past_key_values=c, use_cache=True,
                    position_ids=torch.arange(a.prompt, device=device).unsqueeze(0))
            return [c.cache_kwargs[i]["window_scores"].float()
                    for i in range(own.config.num_hidden_layers)]
        from models.deepseek_v2.loader import attention_modules
        A_mods = attention_modules(own)
        saved = [m._recoverkv_flash for m in A_mods]
        try:
            sf, se = prefill_scores(True), prefill_scores(False)
        finally:
            for m, f in zip(A_mods, saved):
                m._recoverkv_flash = f
        errs = [((x - y).abs().sum() / y.abs().sum()).item() for x, y in zip(sf, se)]
        record("2 prefill window scores: Triton score kernel vs eager", max(errs) <= 3e-2,
               rel_l1_per_layer=[round(e, 5) for e in errs])

    # 3. fused decode at gate 1.0 vs materialize, through evictions
    cache_mod.reset_evict_path_stats()
    fd.reset_stats()
    A.reset_path_stats()
    c_ref = make_cache(own, wcfg(a.budget, 0.70, 1.0), prefill_len=a.prompt,
                       max_tokens=a.steps, fused_decode=False)
    want = _forced(own, c_ref, ids, a.steps)
    n_q = c_ref._joint_store.num_active_windows if c_ref._joint_store is not None else 0
    assert A.path_stats()["decode_fused"] == 0
    fd.reset_stats()
    A.reset_path_stats()
    c_fused = make_cache(own, wcfg(a.budget, 0.70, 1.0), prefill_len=a.prompt,
                         max_tokens=a.steps, fused_decode=True)
    got = _forced(own, c_fused, ids, a.steps)
    s1, p1 = fd.stats(), A.path_stats()
    good, info = _compare(got, want, rel)
    # Both runs prefill identically, so they evict identically until a decode
    # step's scores (fused kernel vs materialize) break a near-tie differently.
    # Comparing at least up to the first joint eviction (step 8) is required.
    record("3 fused decode (gate 1.0) vs materialize",
           good and info["steps_compared"] >= 8 and n_q > 0 and p1["decode_fused"] > 0
           and s1["armed"] == s1["fired"] == s1["gated"] == p1["decode_fused"]
           and s1["read_fraction"] == 1.0,
           **info, int2_windows=n_q,
           counters={k: s1[k] for k in ("armed", "fired", "gated", "read_fraction")})

    # 3b. the same comparison in fp32, on the GPU only. bf16's 5e-2 tolerance is
    #     what two correct implementations need at the run's dtype, and it would
    #     not see a 1% kernel error; fp32 (the kernels' dots run at TF32, ~1e-3)
    #     with eager prefill isolates the decode kernels at 5e-3, which does.
    if gpu:
        _, own32 = _models(a.model_dir, torch.float32, device, "eager")
        h32 = install_recoverkv_mla(own32, flash=False)
        try:
            fd.reset_stats()
            want32 = _forced(own32, make_cache(own32, wcfg(a.budget, 0.70, 1.0),
                                               prefill_len=a.prompt, max_tokens=a.steps,
                                               fused_decode=False), ids, a.steps)
            fd.reset_stats()
            got32 = _forced(own32, make_cache(own32, wcfg(a.budget, 0.70, 1.0),
                                              prefill_len=a.prompt, max_tokens=a.steps,
                                              fused_decode=True), ids, a.steps)
            s32 = fd.stats()
            good, info = _compare(got32, want32, 5e-3)
            record("3b fused decode vs materialize, fp32 (decode kernels only)",
                   good and info["steps_compared"] >= 8 and s32["fired"] > 0
                   and s32["gated"] == s32["fired"], **info)
        finally:
            h32.remove()
            del own32

    # 4. the gate at the run's ratio
    fd.reset_stats()
    c_gate = make_cache(own, wcfg(a.budget, 0.70, a.gate), prefill_len=a.prompt,
                        max_tokens=a.steps, fused_decode=True)
    got = _forced(own, c_gate, ids, a.steps)["logits"]
    rep = fd.gate_report(fd.expect_gated("flash_attn", 0.70, True))
    n = c_gate._joint_store.num_active_windows
    cap = math.ceil(a.gate * n) / n if n else 0.0
    record(f"4 gate at ratio {a.gate}",
           rep["verdict"] == fd.GATE_OK and rep["read_fraction"] is not None
           and rep["read_fraction"] <= cap + 1e-9 and not torch.isnan(got).any()
           and not torch.isinf(got).any(),
           verdict=rep["verdict"], read_fraction=rep["read_fraction"], cap=round(cap, 4),
           fired=rep["fired"], gated=rep["gated"])

    # 5. the eviction was compiled, every time
    ev = cache_mod.evict_path_stats()
    record("5 compiled eviction", ev["compiled"] > 0 and ev["eager"] == 0, **ev)

    handle.remove()
    print(json.dumps({"ok": ok, "device": str(device), "dtype": str(dtype),
                      "budget": a.budget, "gate": a.gate,
                      "torch": torch.__version__, "checks": results}, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
