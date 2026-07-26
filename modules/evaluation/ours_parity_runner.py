"""OursParityRunner — windowed cache, either backend (Suite A).

Uses the full windowed cache system from modules.windowed_cache (flash) or
modules.windowed_eager_cache (eager), selected via config.cache.backend_package.
Teacher-forces from the base npz's generated_tokens — never samples.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
from data.corpus_loader import CorpusLoader
from utils.cache_factory import get_cache_classes, validate_backend_attn_pairing
from utils.config import (
    FIRST_EVICTION_STEP_DEFAULT, ConfigValidationError, ExperimentConfig,
    ParityValidationError,
)
from utils.env_capture import capture_environment
from utils.hashing import sha256_string, sha256_tokenizer
from utils.logger import get_logger

log = get_logger(__name__)

def _load_base_npz(path: str) -> dict:
    """Load base npz and extract arrays + metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base run npz not found: {p}")
    data = np.load(str(p), allow_pickle=True)
    meta_str = str(data["metadata_json"][0])
    meta = json.loads(meta_str)
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    return {"arrays": arrays, "metadata": meta}


def _extract_row_retained(ws_row: Tensor, orig_row: Optional[Tensor],
                          tk: int, ws_sz: int, lws,
                          q_ids_row: Optional[Tensor] = None):
    """Extract one (sample, layer)'s parity arrays from a single row's scores.

    This is the per-row generalisation of the original single-sample extraction
    (which hard-coded ``ws_v[0]``).  For a B=1 batch with ``orig_row`` taken
    from row 0 it is byte-identical to the legacy path.

    Parameters
    ----------
    ws_row : Tensor
        Shape ``[H_q, W]`` — cumulative per-window scores for this row/layer.
    orig_row : Tensor or None
        Shape ``[W]`` — original window ids for this row (``None`` ⇒ identity,
        used for the attention-fallback path where ids aren't tracked).
    tk, ws_sz, lws
        top-K windows, window size, and ``local_window_size`` (int or float).
    q_ids_row : Tensor or None
        Shape ``[n_active]`` — original ids of the windows currently held in the
        int2 Q tier for this row/layer (``store.active_ids()[bi]``).  ``None`` at
        ``quant_ratio == 0`` (no Q tier).  Used only to tag ``all_tier_arr``; it
        does not affect any legacy array, so the ``q == 0`` path is unchanged.

    Returns
    -------
    (tk_arr, ws_arr, ret_ids_arr, ret_sc_arr, all_ids_arr, all_tier_arr)
        The first four mirror the legacy
        ``step_tk``/``step_ws``/``step_ret_ids``/``step_ret_scores`` byte-for-byte.
        The last two are new, aligned to ``ws_arr``'s ``W`` axis:
        ``all_ids_arr`` ``[W]`` original id per score column, ``all_tier_arr``
        ``[W]`` tier per column — 0 = fp-evictable, 1 = Q, 2 = local.
    """
    dev = ws_row.device
    W = ws_row.shape[-1]

    # local windows (always retained, compact eW..W-1)
    if isinstance(lws, float):
        Sp = max(W * ws_sz, 1)
        lt = math.ceil(lws * Sp)
        r2 = lt % ws_sz
        if r2:
            lt += ws_sz - r2
        lnw = lt // ws_sz
    else:
        lnw = lws // ws_sz
    lnw = min(lnw, W)
    eW = W - lnw

    # ── evictable top-K ──────────────────────────────────────────────
    if eW > 0 and tk > 0:
        ev = ws_row[:, :eW].mean(dim=0)              # [eW] (mean across heads)
        k = min(tk, eW)
        compact_ev = ev.topk(k).indices.cpu()        # [k] (topk score order)
        orig_ev = (orig_row[compact_ev.to(dev)].cpu()
                   if orig_row is not None else compact_ev)
        tk_arr = orig_ev.numpy()
    else:
        compact_ev = torch.zeros(0, dtype=torch.long)
        orig_ev = torch.zeros(0, dtype=torch.long)
        tk_arr = np.zeros(min(tk, max(W, 1)), dtype=np.int64)

    ws_arr = ws_row.cpu().to(torch.float16).numpy()  # [H_q, W]

    # ── local windows ────────────────────────────────────────────────
    compact_loc = torch.arange(eW, W, dtype=torch.long, device=dev)
    orig_loc = (orig_row[compact_loc].cpu()
                if orig_row is not None else compact_loc.cpu())

    # ── all retained: evictable ∪ local, sorted by original position ──
    all_compact = torch.cat([compact_ev, compact_loc.cpu()])
    all_orig = torch.cat([orig_ev.long(), orig_loc.long()])
    order = torch.argsort(all_orig)
    ret_ids_arr = all_orig[order].numpy()            # [M]
    ret_compact = all_compact[order].to(dev)         # [M]
    ret_sc_arr = ws_row[:, ret_compact].cpu().to(torch.float16).numpy()  # [H_q, M]

    # ── full survivor axis: id + tier per score column (new, additive) ──
    # The score columns [0, W) are the merged survivor windows in chronological
    # (compact) order.  original_window_ids maps each to its absolute id; the Q
    # store's active_ids() names which of the *evictable* survivors are int2.
    all_ids_full = (orig_row.cpu().long().numpy()
                    if orig_row is not None
                    else np.arange(W, dtype=np.int64))          # [W]
    all_tier_arr = np.full(W, 2, dtype=np.int64)               # default: local
    all_tier_arr[:eW] = 0                                       # evictable → fp
    if q_ids_row is not None and eW > 0:
        q_np = q_ids_row.detach().cpu().long().numpy()
        if q_np.size:
            is_q = np.isin(all_ids_full[:eW], q_np)
            all_tier_arr[:eW][is_q] = 1                        # int2 Q tier
    return tk_arr, ws_arr, ret_ids_arr, ret_sc_arr, all_ids_full, all_tier_arr

class OursParityRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        # Fail-fast: validate backend-attn pairing before any I/O or model load.
        validate_backend_attn_pairing(
            config.cache.backend_package, config.model.attn_implementation
        )
        # Resolve the cache class trio once; reuse throughout run().
        (
            self._WindowedCache,
            self._WindowedCacheConfig,
            self._install_score_hooks,
        ) = get_cache_classes(config.cache.backend_package)

    def run(self) -> Path:
        cfg = self.config
        p = cfg.parity
        w = cfg.window
        log.info("=== Ours Parity Runner (backend=%s) ===", cfg.cache.backend_package)

        # Fail fast on an unsupported transformers version: the windowed cache's
        # RoPE handling assumes monotonic cache_position (transformers <= 4.47).
        from utils.cache_factory import assert_transformers_version_supported

        assert_transformers_version_supported()

        # Global knobs from data config (must match base run for parity).
        num_samples_cfg = max(1, int(getattr(cfg.data, "num_samples", 1)))
        max_tokens = getattr(cfg.data, "max_tokens", None)
        ratio = float(getattr(cfg.data, "ratio", 1.0))
        prefill_len, gen_len = cfg.data.resolved_lengths(p.prefill_len, p.gen_len)

        # 2. Load base npz
        base_path = cfg.base_run_npz
        if not base_path:
            raise ValueError("config.base_run_npz is required for parity_ours mode")
        base = _load_base_npz(base_path)
        base_meta = base["metadata"]
        base_gen_tokens = base["arrays"]["generated_tokens"]

        # Detect base NPZ schema: legacy 1D [num_steps] or new 2D [num_samples, num_steps]
        if base_gen_tokens.ndim == 1:
            base_gen_tokens = base_gen_tokens[np.newaxis, :]   # [1, num_steps]
        num_samples_base = base_gen_tokens.shape[0]

        # Read the per-sample shas list (new schema) or fall back to single sha (legacy)
        base_shas = base_meta.get("article_shas")
        if not base_shas:
            base_shas = [base_meta.get("article_sha")]

        # Determine effective num_samples: must match base; warn if config disagrees
        if num_samples_cfg != num_samples_base:
            log.warning(
                "num_samples=%d in config but base npz has %d samples; "
                "using base value (%d) for parity.",
                num_samples_cfg, num_samples_base, num_samples_base,
            )
        num_samples = num_samples_base
        base_num_steps = base_gen_tokens.shape[1]
        log.info("Base npz loaded: %d sample(s), %d generated tokens each",
                 num_samples, base_num_steps)

        # gen_len must not exceed what the base run actually generated.
        if gen_len > base_num_steps:
            raise ParityValidationError(
                f"gen_len={gen_len} (from data.max_tokens*ratio split or parity.gen_len) "
                f"exceeds base npz's recorded {base_num_steps} steps. "
                f"Re-run parity_base with matching tokens/ratio."
            )

        # 3. Validate identicality
        from utils.config import validate_parity_pair
        validate_parity_pair(base_meta, cfg)

        # 4. Load corpus + validate per-sample article shas
        loader = CorpusLoader(p.dataset)
        articles = loader.load()
        samples_shas: List[str] = []
        for sample_idx in range(num_samples):
            article_idx = p.article_index + sample_idx
            if article_idx >= len(articles):
                raise ParityValidationError(
                    f"article_index+sample {article_idx} out of range "
                    f"({len(articles)} articles)"
                )
            sha = sha256_string(articles[article_idx])
            samples_shas.append(sha)
            base_sha_i = base_shas[sample_idx] if sample_idx < len(base_shas) else None
            if base_sha_i and sha != base_sha_i:
                raise ParityValidationError(
                    f"Sample {sample_idx} article SHA mismatch: "
                    f"base={base_sha_i}, ours={sha}"
                )

        # 5. Cache classes (resolved in __post_init__)
        WC, WCC, install_hooks = (
            self._WindowedCache,
            self._WindowedCacheConfig,
            self._install_score_hooks,
        )

        # 6. Load model + tokenizer (once across all samples)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tok_sha = sha256_tokenizer(tokenizer)
        if base_meta.get("tokenizer_sha") and tok_sha != base_meta["tokenizer_sha"]:
            raise ParityValidationError(
                f"Tokenizer SHA mismatch: base={base_meta['tokenizer_sha']}, ours={tok_sha}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            attn_implementation=cfg.model.attn_implementation, device_map="auto")
        model.eval()
        n_layers = model.config.num_hidden_layers

        # Get rope module once
        rope = None
        for name, mod in model.named_modules():
            if "rotary" in name.lower() or "rope" in name.lower():
                rope = mod; break
        if rope is None:
            for name, mod in model.named_modules():
                if hasattr(mod, "rotary_emb"):
                    rope = mod.rotary_emb; break
        if rope is None:
            raise ConfigValidationError(
                "Could not locate a RoPE module on the model. WindowedCache "
                "requires a rotary embedding module for key rerotation; "
                "expected a submodule named '*rotary*'/'*rope*' or any "
                "module exposing a `.rotary_emb` attribute."
            )

        # H2O-style cumulative scoring: no observation window — every query row contributes.
        # top_k_windows is derived from cache.cache_budget so the Jaccard signal slices
        # at exactly the K the production eviction policy actually keeps.
        tk = w.resolved_top_k(cfg.cache.cache_budget, prefill_len, gen_len)
        ns, ws_sz = w.num_sink_tokens, w.window_size
        budget = cfg.cache.cache_budget if cfg.cache.cache_budget is not None else 0.5

        # Per-sample storage
        samples_topk: List[np.ndarray] = []
        samples_ws: List[np.ndarray] = []
        samples_evict: List[np.ndarray] = []
        samples_ret_ids: List[np.ndarray] = []     # [num_steps, n_layers, M] int64, -1 pad
        samples_ret_scores: List[np.ndarray] = []  # [num_steps, n_layers, H_q, M] float16
        samples_win_ids: List[np.ndarray] = []      # [num_steps, n_layers, W] int64, -1 pad
        samples_win_tier: List[np.ndarray] = []     # [num_steps, n_layers, W] int64, -1 pad

        resolved_cfg = None   # captured from the cache for tier metadata (below)
        t0 = time.time()

        batch_size = max(1, int(getattr(cfg.data, "batch_size", 1)))

        for chunk_start in range(0, num_samples, batch_size):
            chunk = list(range(chunk_start, min(chunk_start + batch_size, num_samples)))
            Bc = len(chunk)

            # Build the equal-length batched prefill. Each article is truncated
            # to prefill_len; batching requires they all reach that length.
            tok_list = []
            for sample_idx in chunk:
                article_idx = p.article_index + sample_idx
                t = tokenizer.encode(articles[article_idx], return_tensors="pt",
                                     add_special_tokens=True)[:, :prefill_len]
                tok_list.append(t)
            lengths = {t.shape[1] for t in tok_list}
            if len(lengths) > 1:
                raise ParityValidationError(
                    f"Batched parity (data.batch_size={batch_size}) requires "
                    f"equal-length prefills, but samples {chunk} tokenized to "
                    f"lengths {sorted(lengths)}. Use data.batch_size=1 for "
                    f"corpora with articles shorter than prefill_len."
                )
            tokens = torch.cat(tok_list, dim=0).to(model.device)   # [Bc, L]

            log.info(
                "── Chunk samples %d–%d/%d (articles %d–%d) ──",
                chunk[0] + 1, chunk[-1] + 1, num_samples,
                p.article_index + chunk[0], p.article_index + chunk[-1],
            )

            # Fresh cache + hooks per chunk (cache state must reset).
            cache_config = WCC(
                window_size=w.window_size, num_sink_tokens=w.num_sink_tokens,
                local_window_size=w.local_window_size, cache_budget=budget,
                rerotate_on_evict=getattr(cfg.cache, "rerotate_on_evict", False),
                quant_ratio=getattr(cfg.cache, "quant_ratio", 0.0),
                first_eviction_step=getattr(cfg.cache, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT))
            cache = WC(config=cache_config, prefill_len=prefill_len,
                       model_config=model.config,
                       kv_dtype=dtypes.get(cfg.model.dtype, torch.float16),
                       rope_module=rope,
                       num_layers=n_layers,
                       max_tokens=gen_len)
            # Resolved tier counts (top_k_fp / N_q / quant_ratio) for the npz
            # metadata; identical across chunks, so capturing the last is fine.
            resolved_cfg = cache.resolved
            hooks = install_hooks(model, cache, cache_config)

            # Per-row forced-token streams: [Bc, num_steps].
            chunk_forced = base_gen_tokens[chunk]
            acc_scores: List[Optional[Tensor]] = [None] * n_layers
            # Per-row, per-step accumulators (one list per sample in the chunk).
            all_topk       = [[] for _ in range(Bc)]
            all_ws         = [[] for _ in range(Bc)]
            all_evict      = [[] for _ in range(Bc)]
            all_ret_ids    = [[] for _ in range(Bc)]
            all_ret_scores = [[] for _ in range(Bc)]
            all_win_ids    = [[] for _ in range(Bc)]   # [num_steps, n_layers, W] orig ids
            all_win_tier   = [[] for _ in range(Bc)]   # [num_steps, n_layers, W] 0=fp,1=Q,2=local
            gen_kwargs: Dict[str, Any] = {}
            if cfg.cache.backend_package == "eager":
                gen_kwargs["output_attentions"] = True

            try:
                with torch.no_grad():
                    for step in range(gen_len):
                        if step == 0:
                            inp = tokens.clone()
                        else:
                            inp = torch.tensor(
                                [[int(chunk_forced[bi, step - 1])] for bi in range(Bc)],
                                device=model.device,
                            )   # [Bc, 1]
                        out = model(input_ids=inp, past_key_values=cache, use_cache=True,
                                    return_dict=True, **gen_kwargs)
                        # cache.update() increments _generation_step AFTER the
                        # eviction check, so the step that was checked is
                        # (_generation_step - 1). Ask the policy's should_evict
                        # directly rather than re-deriving the cadence here — the
                        # first eviction is at a fixed decode step, not a window
                        # boundary (EvictionPolicy.should_evict), and this keeps the
                        # mask from drifting. The generation step is shared across
                        # the batch; -1 during prefill is guarded by should_evict's
                        # `step < first_eviction_step` branch.
                        evicted = any(
                            cache._policies[li].should_evict(
                                cache._generation_step[li] - 1
                            )
                            for li in range(n_layers)
                        )
                        if getattr(out, "attentions", None):
                            for li in range(n_layers):
                                a = out.attentions[li]
                                # Sum over ALL query rows (cumulative across steps via acc_scores).
                                ts = a.sum(dim=-2)
                                if acc_scores[li] is None:
                                    acc_scores[li] = ts.clone()
                                else:
                                    oS = acc_scores[li].shape[-1]
                                    nS = ts.shape[-1]
                                    if nS > oS:
                                        acc_scores[li] = torch.cat([acc_scores[li],
                                            torch.zeros(ts.shape[0], ts.shape[1], nS-oS,
                                                        device=ts.device, dtype=ts.dtype)], -1)
                                    elif nS < oS:
                                        ts = torch.nn.functional.pad(ts, (0, oS-nS))
                                    acc_scores[li][..., :max(nS,oS)] += ts[..., :max(nS,oS)]

                        # Precompute per-layer effective scores once (shared across
                        # the batch); None marks "no scores yet" for this layer.
                        layer_wsv: List[Optional[Tensor]] = []
                        layer_orig: List[Optional[Tensor]] = []
                        layer_qids: List[Optional[Tensor]] = []
                        for li in range(n_layers):
                            cs = cache._states[li]
                            # Q-tier window ids for this layer (None at q==0 or
                            # before the first eviction).  Same set for the whole
                            # step; index per row below.
                            _store = cache._stores[li]
                            layer_qids.append(
                                _store.active_ids() if _store is not None else None
                            )
                            if cs.window_scores is not None:
                                layer_wsv.append(cs.window_scores)            # [B, H_q, W]
                                layer_orig.append(cs.original_window_ids)     # [B, W] or None
                            elif acc_scores[li] is not None:
                                ac  = acc_scores[li]
                                ps  = ac[..., ns:]
                                rem = ps.shape[-1] % ws_sz
                                if rem: ps = torch.nn.functional.pad(ps, (0, ws_sz - rem))
                                W_tmp = ps.shape[-1] // ws_sz
                                layer_wsv.append(
                                    ps.reshape(ps.shape[0], ps.shape[1], W_tmp, ws_sz).sum(-1))
                                layer_orig.append(None)
                            else:
                                layer_wsv.append(None)
                                layer_orig.append(None)

                        # Extract per sample-in-chunk, per layer (runner loops are
                        # fine — only cache/state/policy/scorer are loop-free).
                        for bi in range(Bc):
                            all_evict[bi].append(evicted)
                            step_tk, step_ws, step_ret_ids, step_ret_scores = [], [], [], []
                            step_win_ids, step_win_tier = [], []
                            for li in range(n_layers):
                                ws_v = layer_wsv[li]
                                if ws_v is None:
                                    step_tk.append(np.zeros(min(tk, 1), dtype=np.int64))
                                    step_ws.append(np.zeros((1, 1), dtype=np.float16))
                                    step_ret_ids.append(np.zeros(0, dtype=np.int64))
                                    step_ret_scores.append(np.zeros((1, 0), dtype=np.float16))
                                    step_win_ids.append(np.full(1, -1, dtype=np.int64))
                                    step_win_tier.append(np.full(1, -1, dtype=np.int64))
                                    continue
                                orig_full = layer_orig[li]
                                orig_row = orig_full[bi] if orig_full is not None else None
                                q_full = layer_qids[li]
                                q_row = q_full[bi] if q_full is not None else None
                                a_tk, a_ws, a_rid, a_rsc, a_wid, a_wtier = (
                                    _extract_row_retained(
                                        ws_v[bi], orig_row, tk, ws_sz,
                                        w.local_window_size, q_row))
                                step_tk.append(a_tk)
                                step_ws.append(a_ws)
                                step_ret_ids.append(a_rid)
                                step_ret_scores.append(a_rsc)
                                step_win_ids.append(a_wid)
                                step_win_tier.append(a_wtier)

                            # ── stack per-layer results for this step/sample ──
                            all_topk[bi].append(np.stack(step_tk, 0))
                            all_ws[bi].append(np.stack(step_ws, 0))
                            # Full survivor axis shares window_scores' W per step
                            # (uniform across layers), so it stacks directly.
                            all_win_ids[bi].append(np.stack(step_win_ids, 0))   # [n_layers, W]
                            all_win_tier[bi].append(np.stack(step_win_tier, 0)) # [n_layers, W]
                            mM_s = max(len(x) for x in step_ret_ids)
                            mH_s = max(x.shape[0] for x in step_ret_scores)
                            p_rid = [np.pad(x, [(0, mM_s - len(x))], constant_values=-1)
                                     for x in step_ret_ids]
                            p_rsc = [np.pad(x, [(0, mH_s - x.shape[0]),
                                                (0, mM_s - x.shape[1])])
                                     for x in step_ret_scores]
                            all_ret_ids[bi].append(np.stack(p_rid, 0))    # [n_layers, mM_s]
                            all_ret_scores[bi].append(np.stack(p_rsc, 0)) # [n_layers, mH_s, mM_s]

                        if (step+1) % 100 == 0:
                            log.info("  Step %d/%d", step+1, gen_len)
            finally:
                hooks.remove()

            # Finalize each sample in the chunk (pad per-step, append to samples_*
            # in global sample order so the leading sample axis stays in order).
            for bi in range(Bc):
                s_topk, s_ws, s_evict = all_topk[bi], all_ws[bi], all_evict[bi]
                s_rid, s_rsc = all_ret_ids[bi], all_ret_scores[bi]
                s_wid, s_wtier = all_win_ids[bi], all_win_tier[bi]

                mW = max(x.shape[-1] for x in s_ws)
                mK = max(x.shape[-1] for x in s_topk)
                pws = [np.pad(x, [(0,0),(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in s_ws]
                ptk = [np.pad(x, [(0,0),(0,mK-x.shape[-1])], constant_values=-1) if x.shape[-1]<mK else x for x in s_topk]
                mH = max(x.shape[-2] for x in pws)
                pws = [np.pad(x, [(0,0),(0,mH-x.shape[-2]),(0,0)]) if x.shape[-2]<mH else x for x in pws]
                # Full survivor axis shares window_scores' W → pad to the same mW.
                pwid = [np.pad(x, [(0,0),(0,mW-x.shape[-1])], constant_values=-1)
                        if x.shape[-1] < mW else x for x in s_wid]
                pwtier = [np.pad(x, [(0,0),(0,mW-x.shape[-1])], constant_values=-1)
                          if x.shape[-1] < mW else x for x in s_wtier]

                samples_topk.append(np.stack(ptk, 0))
                samples_ws.append(np.stack(pws, 0))
                samples_evict.append(np.array(s_evict, dtype=bool))
                samples_win_ids.append(np.stack(pwid, 0))    # [num_steps, n_layers, mW]
                samples_win_tier.append(np.stack(pwtier, 0)) # [num_steps, n_layers, mW]

                # Pad retained arrays across steps (M and H_q may grow over time)
                mM2  = max(x.shape[-1]  for x in s_rid)
                mH2  = max(x.shape[-2]  for x in s_rsc)
                prid = [np.pad(x, [(0,0),(0, mM2 - x.shape[-1])],   constant_values=-1)
                        if x.shape[-1] < mM2 else x for x in s_rid]
                prsc = [np.pad(x, [(0,0),(0, mH2 - x.shape[-2]),(0, mM2 - x.shape[-1])])
                        if (x.shape[-2] < mH2 or x.shape[-1] < mM2) else x
                        for x in s_rsc]
                samples_ret_ids.append(np.stack(prid, 0))    # [num_steps, n_layers, mM2]
                samples_ret_scores.append(np.stack(prsc, 0)) # [num_steps, n_layers, mH2, mM2]

            # Memory hygiene: free per-chunk cache, hooks, and tensors.
            del cache, cache_config, acc_scores, all_topk, all_ws, all_evict, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()

        # Align K, W, H_q, M across samples (could differ if eviction trajectories diverged)
        max_K  = max(x.shape[-1] for x in samples_topk)
        max_W  = max(x.shape[-1] for x in samples_ws)
        max_H  = max(x.shape[-2] for x in samples_ws)
        max_M  = max(x.shape[-1] for x in samples_ret_ids)
        max_Hr = max(x.shape[-2] for x in samples_ret_scores)
        aligned_topk, aligned_ws = [], []
        aligned_ret_ids, aligned_ret_scores = [], []
        aligned_win_ids, aligned_win_tier = [], []
        for tkarr, wsarr, ridarr, rscarr, widarr, wtierarr in zip(
                samples_topk, samples_ws, samples_ret_ids, samples_ret_scores,
                samples_win_ids, samples_win_tier):
            if tkarr.shape[-1] < max_K:
                tkarr = np.pad(tkarr, [(0,0),(0,0),(0, max_K - tkarr.shape[-1])],
                               constant_values=-1)
            if wsarr.shape[-2] < max_H:
                wsarr = np.pad(wsarr, [(0,0),(0,0),(0, max_H - wsarr.shape[-2]),(0,0)])
            if wsarr.shape[-1] < max_W:
                wsarr = np.pad(wsarr, [(0,0),(0,0),(0,0),(0, max_W - wsarr.shape[-1])])
            if ridarr.shape[-1] < max_M:
                ridarr = np.pad(ridarr, [(0,0),(0,0),(0, max_M - ridarr.shape[-1])],
                                constant_values=-1)
            if rscarr.shape[-2] < max_Hr:
                rscarr = np.pad(rscarr, [(0,0),(0,0),(0, max_Hr - rscarr.shape[-2]),(0,0)])
            if rscarr.shape[-1] < max_M:
                rscarr = np.pad(rscarr, [(0,0),(0,0),(0,0),(0, max_M - rscarr.shape[-1])])
            # Full survivor axis: same W axis as window_scores → pad to max_W.
            if widarr.shape[-1] < max_W:
                widarr = np.pad(widarr, [(0,0),(0,0),(0, max_W - widarr.shape[-1])],
                                constant_values=-1)
            if wtierarr.shape[-1] < max_W:
                wtierarr = np.pad(wtierarr, [(0,0),(0,0),(0, max_W - wtierarr.shape[-1])],
                                  constant_values=-1)
            aligned_topk.append(tkarr)
            aligned_ws.append(wsarr)
            aligned_ret_ids.append(ridarr)
            aligned_ret_scores.append(rscarr)
            aligned_win_ids.append(widarr)
            aligned_win_tier.append(wtierarr)

        top_window_indices     = np.stack(aligned_topk, 0)
        window_scores          = np.stack(aligned_ws, 0)
        eviction_step_mask     = np.stack(samples_evict, 0)
        retained_window_ids    = np.stack(aligned_ret_ids, 0)    # [S, T, L, M]
        retained_window_scores = np.stack(aligned_ret_scores, 0) # [S, T, L, H, M]
        all_window_ids         = np.stack(aligned_win_ids, 0)    # [S, T, L, W]
        all_window_tier        = np.stack(aligned_win_tier, 0)   # [S, T, L, W]

        elapsed = time.time() - t0
        log.info("Done: %d samples, %.1fs", num_samples, elapsed)

        # Must match WindowedCacheConfig.resolve(), which uses prefill_len - num_sink_tokens.
        St = prefill_len - ns
        if isinstance(w.local_window_size, float):
            lr = math.ceil(w.local_window_size * St)
            r2 = lr % ws_sz
            if r2: lr += ws_sz - r2
        else:
            lr = w.local_window_size

        # Two-tier split for the faithfulness suite (design.md §7). At q==0 the
        # resolver guarantees top_k_fp == top_k_windows and N_q == 0, so the
        # tier-aware metrics collapse to the legacy single-tier numbers.
        q_ratio  = float(getattr(resolved_cfg, "quant_ratio", 0.0)) if resolved_cfg else 0.0
        top_k_fp = int(getattr(resolved_cfg, "top_k_fp", tk)) if resolved_cfg else tk
        n_q      = int(getattr(resolved_cfg, "N_q", 0)) if resolved_cfg else 0

        env = capture_environment()
        meta = {
            "schema_version": "1.2",                  # bumped: full-survivor tier arrays
            "mode": "parity_ours",
            "seed": cfg.run.seed,
            "dataset": p.dataset,
            "article_id": p.article_index,
            "article_index_start": p.article_index,
            "num_samples": num_samples,
            "article_shas": samples_shas,
            "article_sha": samples_shas[0],
            "tokenizer_sha": tok_sha,
            "prefill_len": prefill_len,
            "max_tokens": max_tokens,
            "gen_len": gen_len,
            "ratio": ratio,
            "window_size": w.window_size,
            "num_sink_tokens": ns,
            "local_window_size_resolved": lr,
            "top_k_windows": tk,
            "quant_ratio": q_ratio,
            "top_k_fp": top_k_fp,
            "N_q": n_q,
            # The eviction SCHEDULE, alongside the eviction geometry above.
            # Without it the eviction_step_mask in this npz cannot be attributed
            # to an operating point after the fact, and Suites B and E read this
            # npz as their only record of how the run was configured.
            "first_eviction_step": int(
                getattr(resolved_cfg, "first_eviction_step",
                        getattr(cfg.cache, "first_eviction_step",
                                FIRST_EVICTION_STEP_DEFAULT))
            ),
            "rerotate_on_evict": bool(
                getattr(cfg.cache, "rerotate_on_evict", False)
            ),
            "model_name": cfg.model.name,
            "model_revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "attn_implementation": cfg.model.attn_implementation,
            "cache_backend": "windowed",
            "cache_backend_package": cfg.cache.backend_package,
            "cache_budget": budget,
            **env,
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        be = cfg.cache.backend_package or "unknown"
        # loader.slug, not p.dataset — a local corpus path is not a filename.
        npz = Path(cfg.output_path) if cfg.output_path else od / f"parity_ours_{be}_{loader.slug}_{samples_shas[0][:8]}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz),
            top_window_indices=top_window_indices,
            window_scores=window_scores,
            eviction_step_mask=eviction_step_mask,
            generated_tokens=base_gen_tokens,           # carry-through from base [num_samples, num_steps]
            retained_window_ids=retained_window_ids,    # [S, T, L, M] original IDs, -1 pad
            retained_window_scores=retained_window_scores,  # [S, T, L, H, M] ours' scores
            all_window_ids=all_window_ids,              # [S, T, L, W] id per score col, -1 pad
            all_window_tier=all_window_tier,            # [S, T, L, W] 0=fp,1=Q,2=local,-1=pad
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        with open(npz.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved: %s", npz)
        return npz
