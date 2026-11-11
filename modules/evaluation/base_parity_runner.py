"""BaseParityRunner — vanilla model, no monkey-patch (Suite A).

Runs the stock HF model with DynamicCache and output_attentions=True.
Computes Top-K window rankings from attention matrices at each step.
No hooks, no eviction. Saves generated tokens for ours runner replay.

Two window-score arrays are recorded:

  ``window_scores``       ``[S, T, L, H, W]`` fp16 — *cumulative* H2O scores,
                          the signal the eviction policy ranks on.
  ``step_window_scores``  ``[S, T, L, W]`` fp32 — the head-mean mass deposited
                          by **this step's** query rows alone (step 0 = the
                          whole prefill).  Recorded because it cannot be
                          recovered from the cumulative array: by ``T = 1024``
                          one fp16 ulp of the accumulated score exceeds half a
                          per-step delta, so differencing returns noise.  The
                          QEvict observation suite scores routing decisions
                          against future attention and needs this signal
                          (``modules/evaluation/qevict_observations.py``).
  ``step_window_scores_heads``  ``[S, T, L, H, W]`` fp16 — the same per-step
                          mass PER QUERY HEAD, recorded only with
                          ``parity.record_head_step_mass``.  The read gate picks
                          per KV head, and a cross-head sum of attention must be
                          normalised per head first (CLAUDE.md, §12), so gate
                          recall and the decode read ledger are per head.

Both cumulative arrays are **accumulated in fp32** (schema >= 1.3), matching the
cache's own ``hooks.SCORE_ACCUM_DTYPE``.  They used to accumulate in the
attention dtype (fp16): one decode step adds well under 1e-2 to a window, below
half an fp16 ULP of any total above ~8, so the oracle ranking that Observations
I–III and the tier study score against was the prompt's ranking — the exact
defect ``DISTANCE_TO_GOAL.md`` §11 fixed in the cache, left standing in the
ground truth it is measured against.  Storage stays fp16; only the running sum
changed.

Articles shorter than ``prefill_len`` tokens are **skipped** (schema >= 1.3),
and the indices actually used are recorded as ``article_indices``.  Every
observation assumes the metadata ``prefill_len`` for every sample — a short
article silently shifts its whole window geometry.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
from data.corpus_loader import CorpusLoader
from utils.config import ExperimentConfig, ParityValidationError
from utils.env_capture import capture_environment
from utils.hashing import sha256_string, sha256_tokenizer
from utils.logger import get_logger

log = get_logger(__name__)

#: Running-sum dtype of the cumulative window scores. Mirrors
#: ``modules.windowed_cache.hooks.SCORE_ACCUM_DTYPE`` (not imported: this runner
#: must not touch the windowed cache -- ``test_base_no_hooks_installed``).
SCORE_ACCUM_DTYPE = torch.float32


def select_parity_articles(articles: List[str], tokenizer, start: int, n: int,
                           prefill_len: int) -> List[int]:
    """Indices of the first ``n`` articles from ``start`` with >= ``prefill_len`` tokens.

    Shared by both parity runners (the ours run replays the base run's recorded
    ``article_indices`` rather than re-deriving them, and checks the shas).
    Returns fewer than ``n`` when the corpus runs out, and warns; raises if none
    qualify.
    """
    picked: List[int] = []
    skipped = 0
    for idx in range(max(0, int(start)), len(articles)):
        n_tok = len(tokenizer.encode(articles[idx], add_special_tokens=True))
        if n_tok >= prefill_len:
            picked.append(idx)
            if len(picked) >= n:
                break
        else:
            skipped += 1
    if not picked:
        raise ParityValidationError(
            f"no article from index {start} has >= {prefill_len} tokens "
            f"({len(articles)} articles in the corpus). Lower prefill_len or use "
            "a corpus of longer documents.")
    if skipped:
        log.info("skipped %d article(s) shorter than prefill_len=%d tokens",
                 skipped, prefill_len)
    if len(picked) < n:
        log.warning("only %d article(s) from index %d reach prefill_len=%d "
                    "(asked for %d)", len(picked), start, prefill_len, n)
    return picked


def _base_row_topk(ws_row: Tensor, eW: int, tk: int):
    """Per-row top-K extraction for the base (full-cache) parity run.

    The per-row generalisation of the original ``ws_v[0]`` extraction; for a
    B=1 batch it is byte-identical to the legacy path. Window indices are in
    original space (the base run uses a DynamicCache and never evicts, so
    compact index == original index).

    Parameters
    ----------
    ws_row : Tensor
        Shape ``[H_q, W]`` — cumulative per-window scores for this row/layer.
    eW : int
        Number of evictable windows (``W - local_windows``), computed once per
        layer because it depends only on the shared geometry.
    tk : int
        top-K windows to record.

    Returns
    -------
    (tk_arr, ws_arr) : numpy arrays
        ``tk_arr`` mirrors ``step_tk`` (top-K indices, score order);
        ``ws_arr`` mirrors ``step_ws`` (full per-window scores, ``[H_q, W]``).
    """
    W = ws_row.shape[-1]
    if eW > 0 and tk > 0:
        ev = ws_row[:, :eW].mean(dim=0)          # [eW] (mean across heads)
        k = min(tk, eW)
        tk_arr = ev.topk(k).indices.cpu().numpy()
    else:
        tk_arr = np.zeros(min(tk, W), dtype=np.int64)
    ws_arr = ws_row.cpu().to(torch.float16).numpy()
    return tk_arr, ws_arr


class BaseParityRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        p = cfg.parity
        w = cfg.window
        log.info("=== Base Parity Runner ===")

        # Global knobs from data config (override parity defaults).
        # num_samples == 1 + max_tokens is None preserves the legacy schema (single sample, sample-axis size 1).
        num_samples = max(1, int(getattr(cfg.data, "num_samples", 1)))
        max_tokens = getattr(cfg.data, "max_tokens", None)
        ratio = float(getattr(cfg.data, "ratio", 1.0))
        # Resolve effective lengths: when max_tokens is set, split by ratio;
        # otherwise fall back to parity.prefill_len / parity.gen_len.
        prefill_len, gen_len = cfg.data.resolved_lengths(p.prefill_len, p.gen_len)

        # 1. Load corpus once.
        text_field = getattr(p, "text_field", None)
        record_filter = getattr(p, "record_filter", None)
        loader = CorpusLoader(
            p.dataset,
            text_field=text_field if isinstance(text_field, str) else None,
            record_filter=record_filter if isinstance(record_filter, str) else None)
        articles = loader.load()
        record_heads = bool(getattr(p, "record_head_step_mass", False) is True)

        log.info(
            "Sampling %d article(s) from index %d, prefill_len=%d, gen_len=%d "
            "(max_tokens=%s, ratio=%.3f)",
            num_samples, p.article_index, prefill_len, gen_len,
            "None" if max_tokens is None else int(max_tokens), ratio,
        )

        # 2. Load model + tokenizer (once across all samples).
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tok_sha = sha256_tokenizer(tokenizer)

        # Only articles that fill the whole prefill: the observation geometry
        # assumes the metadata prefill_len for every sample.
        article_indices = select_parity_articles(
            articles, tokenizer, p.article_index, num_samples, prefill_len)
        num_samples = len(article_indices)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            attn_implementation="eager", device_map="auto")
        model.eval()
        n_layers = model.config.num_hidden_layers
        # H2O-style cumulative scoring: no observation window — every query row contributes.
        # top_k_windows is derived from cache.cache_budget so the Jaccard signal slices
        # at exactly the K the production eviction policy would have kept.
        tk = w.resolved_top_k(cfg.cache.cache_budget, prefill_len, gen_len)
        ns, ws_sz = w.num_sink_tokens, w.window_size

        # Per-sample storage
        samples_topk: List[np.ndarray] = []     # each: [num_steps, num_layers, K]
        samples_ws: List[np.ndarray] = []        # each: [num_steps, num_layers, H_q, W]
        samples_step_ws: List[np.ndarray] = []   # each: [num_steps, num_layers, W]
        samples_step_ws_h: List[np.ndarray] = [] # each: [num_steps, num_layers, H, W]
        samples_gen_toks: List[np.ndarray] = []  # each: [num_steps]
        samples_shas: List[str] = []

        t0 = time.time()

        batch_size = max(1, int(getattr(cfg.data, "batch_size", 1)))

        for chunk_start in range(0, num_samples, batch_size):
            chunk = list(range(chunk_start, min(chunk_start + batch_size, num_samples)))
            Bc = len(chunk)

            # Build the equal-length batched prefill (each article truncated to
            # prefill_len; batching requires they all reach that length).
            tok_list = []
            for sample_idx in chunk:
                article_idx = article_indices[sample_idx]
                article_text = articles[article_idx]
                samples_shas.append(sha256_string(article_text))
                t = tokenizer.encode(article_text, return_tensors="pt",
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
                article_indices[chunk[0]], article_indices[chunk[-1]],
            )

            acc_scores: List[Optional[Tensor]] = [None] * n_layers
            # Per-row, per-step accumulators (one list per sample in the chunk).
            all_topk = [[] for _ in range(Bc)]
            all_ws   = [[] for _ in range(Bc)]
            all_step_ws = [[] for _ in range(Bc)]   # per-step (non-cumulative)
            all_step_ws_h = [[] for _ in range(Bc)] # per-step, per query head
            gen_toks = [[] for _ in range(Bc)]
            input_ids = tokens.clone()
            next_tok = None          # loop-carried; step 0 uses input_ids
            pkv = DynamicCache()
            with torch.no_grad():
                for step in range(gen_len):
                    # Each row generates its own greedy continuation: [Bc, 1].
                    inp = input_ids if step == 0 else next_tok.unsqueeze(1)
                    out = model(input_ids=inp, past_key_values=pkv, use_cache=True,
                                output_attentions=True, return_dict=True)
                    pkv = out.past_key_values
                    next_tok = out.logits[:, -1, :].argmax(dim=-1)   # [Bc]
                    for bi in range(Bc):
                        gen_toks[bi].append(int(next_tok[bi].item()))
                    step_ws_layers: List[np.ndarray] = []
                    step_ws_h_layers: List[np.ndarray] = []
                    for li in range(n_layers):
                        a = out.attentions[li]
                        # Sum over ALL query rows of this step (cumulative across
                        # steps via acc_scores), in SCORE_ACCUM_DTYPE: the running
                        # sum below must not round a decode step away (module doc).
                        ts = a.sum(dim=-2, dtype=SCORE_ACCUM_DTYPE)
                        # Per-step (non-cumulative) head-mean window mass: the
                        # same post-sink windowing as the cumulative path, but
                        # on this step's contribution alone.  Kept in fp32 —
                        # it is unrecoverable from the fp16 cumulative array.
                        sp = ts[..., ns:]
                        rem_s = sp.shape[-1] % ws_sz
                        if rem_s:
                            sp = torch.nn.functional.pad(sp, (0, ws_sz - rem_s))
                        W_s = sp.shape[-1] // ws_sz
                        sp_h = sp.reshape(sp.shape[0], sp.shape[1], W_s, ws_sz).sum(-1)
                        step_ws_layers.append(
                            sp_h.mean(dim=1)                # [B, W] head-mean
                            .float().cpu().numpy())
                        if record_heads:                    # [B, H, W] per head
                            step_ws_h_layers.append(
                                sp_h.to(torch.float16).cpu().numpy())
                        if acc_scores[li] is None:
                            acc_scores[li] = ts.clone()
                        else:
                            oS = acc_scores[li].shape[-1]
                            nS = ts.shape[-1]
                            if nS > oS:
                                acc_scores[li] = torch.cat([acc_scores[li],
                                    torch.zeros(ts.shape[0], ts.shape[1], nS-oS,
                                                device=ts.device, dtype=ts.dtype)], -1)
                            acc_scores[li][..., :nS] += ts

                    # Precompute per-layer effective scores + geometry once
                    # (shared across the batch); extract per row afterwards.
                    layer_wsv: List[Tensor] = []
                    layer_eW: List[int] = []
                    for li in range(n_layers):
                        ac = acc_scores[li]
                        ps = ac[..., ns:]
                        Sp = ps.shape[-1]
                        rem = Sp % ws_sz
                        if rem: ps = torch.nn.functional.pad(ps, (0, ws_sz-rem))
                        W = ps.shape[-1] // ws_sz
                        ws_v = ps.reshape(ps.shape[0], ps.shape[1], W, ws_sz).sum(-1)
                        lws = w.local_window_size
                        if isinstance(lws, float):
                            lt = math.ceil(lws * Sp)
                            r2 = lt % ws_sz
                            if r2: lt += ws_sz - r2
                            lnw = lt // ws_sz
                        else:
                            lnw = lws // ws_sz
                        lnw = min(lnw, W)
                        layer_wsv.append(ws_v)
                        layer_eW.append(W - lnw)

                    for bi in range(Bc):
                        step_tk, step_ws = [], []
                        for li in range(n_layers):
                            a_tk, a_ws = _base_row_topk(
                                layer_wsv[li][bi], layer_eW[li], tk)
                            step_tk.append(a_tk)
                            step_ws.append(a_ws)
                        all_topk[bi].append(np.stack(step_tk, 0))
                        all_ws[bi].append(np.stack(step_ws, 0))
                        all_step_ws[bi].append(
                            np.stack([x[bi] for x in step_ws_layers], 0))  # [L, W]
                        if record_heads:
                            all_step_ws_h[bi].append(
                                np.stack([x[bi] for x in step_ws_h_layers], 0))  # [L, H, W]

                    if (step+1) % 100 == 0:
                        log.info("  Step %d/%d", step+1, gen_len)

            # Finalize each sample in the chunk (pad per-step, append in global
            # sample order so the leading sample axis stays ordered).
            for bi in range(Bc):
                s_topk, s_ws, s_gen = all_topk[bi], all_ws[bi], gen_toks[bi]
                s_sws = all_step_ws[bi]
                mW = max(x.shape[-1] for x in s_ws)
                mK = max(x.shape[-1] for x in s_topk)
                pws = [np.pad(x, [(0,0),(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in s_ws]
                ptk = [np.pad(x, [(0,0),(0,mK-x.shape[-1])], constant_values=-1) if x.shape[-1]<mK else x for x in s_topk]
                psws = [np.pad(x, [(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in s_sws]
                samples_topk.append(np.stack(ptk, 0))
                samples_ws.append(np.stack(pws, 0))
                samples_step_ws.append(np.stack(psws, 0))
                if record_heads:
                    pswh = [np.pad(x, [(0, 0), (0, 0), (0, mW - x.shape[-1])])
                            if x.shape[-1] < mW else x for x in all_step_ws_h[bi]]
                    samples_step_ws_h.append(np.stack(pswh, 0))
                samples_gen_toks.append(np.array(s_gen, dtype=np.int64))

            # Memory hygiene: free per-chunk tensors before the next chunk.
            del acc_scores, all_topk, all_ws, all_step_ws, all_step_ws_h, gen_toks, pkv, input_ids, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()

        # Align K and W across samples (could differ from sample to sample)
        max_K = max(x.shape[-1] for x in samples_topk)
        max_W = max(x.shape[-1] for x in samples_ws)
        aligned_topk, aligned_ws, aligned_step_ws = [], [], []
        aligned_step_ws_h = [
            np.pad(x, [(0, 0), (0, 0), (0, 0), (0, max(0, max_W - x.shape[-1]))])
            for x in samples_step_ws_h] if record_heads else []
        for tkarr, wsarr, swsarr in zip(samples_topk, samples_ws, samples_step_ws):
            if tkarr.shape[-1] < max_K:
                tkarr = np.pad(tkarr, [(0, 0), (0, 0), (0, max_K - tkarr.shape[-1])],
                               constant_values=-1)
            if wsarr.shape[-1] < max_W:
                wsarr = np.pad(wsarr, [(0, 0), (0, 0), (0, 0), (0, max_W - wsarr.shape[-1])])
            if swsarr.shape[-1] < max_W:
                swsarr = np.pad(swsarr, [(0, 0), (0, 0), (0, max_W - swsarr.shape[-1])])
            aligned_topk.append(tkarr)
            aligned_ws.append(wsarr)
            aligned_step_ws.append(swsarr)

        # Stack along leading sample axis: [num_samples, num_steps, num_layers, ...]
        top_window_indices = np.stack(aligned_topk, 0)
        window_scores = np.stack(aligned_ws, 0)
        step_window_scores = np.stack(aligned_step_ws, 0).astype(np.float32)
        extra_arrays: Dict[str, np.ndarray] = {}
        if record_heads:
            extra_arrays["step_window_scores_heads"] = np.stack(
                aligned_step_ws_h, 0).astype(np.float16)     # [S, T, L, H, W]
        generated_tokens = np.stack(samples_gen_toks, 0)
        eviction_step_mask = np.zeros((num_samples, gen_len), dtype=bool)

        elapsed = time.time() - t0
        total_tokens = num_samples * gen_len
        log.info("Done: %d samples, %.1fs (%.1f tok/s overall)",
                 num_samples, elapsed, total_tokens / max(elapsed, 1e-6))

        # Resolve local_window_size — must match WindowedCacheConfig.resolve(),
        # which uses prefill_len - num_sink_tokens (NOT including gen_len).
        St = prefill_len - ns
        if isinstance(w.local_window_size, float):
            lr = math.ceil(w.local_window_size * St)
            r2 = lr % ws_sz
            if r2: lr += ws_sz - r2
        else:
            lr = w.local_window_size

        env = capture_environment()
        meta = {
            # 1.3: fp32 score accumulation, full-prefill article selection
            # (article_indices), optional step_window_scores_heads.
            "schema_version": "1.3",
            "mode": "parity_base",
            "seed": cfg.run.seed,
            "dataset": p.dataset,
            "article_id": p.article_index,                # first article (back-compat)
            "article_index_start": p.article_index,
            "article_indices": [int(i) for i in article_indices],
            "text_field": text_field if isinstance(text_field, str) else None,
            "record_filter": record_filter if isinstance(record_filter, str) else None,
            "score_accum_dtype": "float32",
            "record_head_step_mass": record_heads,
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(getattr(
                model.config, "num_key_value_heads",
                model.config.num_attention_heads)),
            "num_samples": num_samples,
            "article_shas": samples_shas,
            "article_sha": samples_shas[0],               # back-compat: first sample
            "tokenizer_sha": tok_sha,
            "prefill_len": prefill_len,
            "max_tokens": max_tokens,
            "gen_len": gen_len,
            "ratio": ratio,
            "window_size": w.window_size,
            "num_sink_tokens": ns,
            "local_window_size_resolved": lr,
            "top_k_windows": tk,
            "model_name": cfg.model.name,
            "model_revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "attn_implementation": "eager",
            "cache_backend": "dynamic",
            "cache_backend_package": None,
            "cache_budget": None,
            **env,
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        # loader.slug, not p.dataset — a local corpus path is not a filename.
        npz = Path(cfg.output_path) if cfg.output_path else od / f"parity_base_{loader.slug}_{samples_shas[0][:8]}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz),
            top_window_indices=top_window_indices,
            window_scores=window_scores,
            step_window_scores=step_window_scores,   # [S, T, L, W] fp32, per-step
            **extra_arrays,                          # step_window_scores_heads
            eviction_step_mask=eviction_step_mask,
            generated_tokens=generated_tokens,
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        with open(npz.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved: %s", npz)
        return npz
