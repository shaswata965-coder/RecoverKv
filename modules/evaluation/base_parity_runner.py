"""BaseParityRunner — vanilla model, no monkey-patch (Suite A).

Runs the stock HF model with DynamicCache and output_attentions=True.
Computes Top-K window rankings from attention matrices at each step.
No hooks, no eviction. Saves generated tokens for ours runner replay.
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
        loader = CorpusLoader(p.dataset)
        articles = loader.load()

        # Clamp num_samples to available articles starting at article_index.
        available = len(articles) - p.article_index
        if num_samples > available:
            log.warning(
                "num_samples=%d exceeds %d available articles from index %d; "
                "clamping to %d", num_samples, available, p.article_index, available,
            )
            num_samples = max(1, available)

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
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            attn_implementation="eager", device_map="auto")
        model.eval()
        n_layers = model.config.num_hidden_layers
        # H2O-style cumulative scoring: no observation window — every query row contributes.
        # top_k_windows is derived from cache.cache_budget so the Jaccard signal slices
        # at exactly the K the production eviction policy would have kept.
        tk = w.resolved_top_k(cfg.cache, prefill_len, gen_len)
        ns, ws_sz = w.num_sink_tokens, w.window_size

        # Per-sample storage
        samples_topk: List[np.ndarray] = []     # each: [num_steps, num_layers, K]
        samples_ws: List[np.ndarray] = []        # each: [num_steps, num_layers, H_q, W]
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
                article_idx = p.article_index + sample_idx
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
                p.article_index + chunk[0], p.article_index + chunk[-1],
            )

            acc_scores: List[Optional[Tensor]] = [None] * n_layers
            # Per-row, per-step accumulators (one list per sample in the chunk).
            all_topk = [[] for _ in range(Bc)]
            all_ws   = [[] for _ in range(Bc)]
            gen_toks = [[] for _ in range(Bc)]
            input_ids = tokens.clone()
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
                    for li in range(n_layers):
                        a = out.attentions[li]
                        # Sum over ALL query rows of this step (cumulative across steps via acc_scores).
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

                    if (step+1) % 100 == 0:
                        log.info("  Step %d/%d", step+1, gen_len)

            # Finalize each sample in the chunk (pad per-step, append in global
            # sample order so the leading sample axis stays ordered).
            for bi in range(Bc):
                s_topk, s_ws, s_gen = all_topk[bi], all_ws[bi], gen_toks[bi]
                mW = max(x.shape[-1] for x in s_ws)
                mK = max(x.shape[-1] for x in s_topk)
                pws = [np.pad(x, [(0,0),(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in s_ws]
                ptk = [np.pad(x, [(0,0),(0,mK-x.shape[-1])], constant_values=-1) if x.shape[-1]<mK else x for x in s_topk]
                samples_topk.append(np.stack(ptk, 0))
                samples_ws.append(np.stack(pws, 0))
                samples_gen_toks.append(np.array(s_gen, dtype=np.int64))

            # Memory hygiene: free per-chunk tensors before the next chunk.
            del acc_scores, all_topk, all_ws, gen_toks, pkv, input_ids, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()

        # Align K and W across samples (could differ from sample to sample)
        max_K = max(x.shape[-1] for x in samples_topk)
        max_W = max(x.shape[-1] for x in samples_ws)
        aligned_topk, aligned_ws = [], []
        for tkarr, wsarr in zip(samples_topk, samples_ws):
            if tkarr.shape[-1] < max_K:
                tkarr = np.pad(tkarr, [(0, 0), (0, 0), (0, max_K - tkarr.shape[-1])],
                               constant_values=-1)
            if wsarr.shape[-1] < max_W:
                wsarr = np.pad(wsarr, [(0, 0), (0, 0), (0, 0), (0, max_W - wsarr.shape[-1])])
            aligned_topk.append(tkarr)
            aligned_ws.append(wsarr)

        # Stack along leading sample axis: [num_samples, num_steps, num_layers, ...]
        top_window_indices = np.stack(aligned_topk, 0)
        window_scores = np.stack(aligned_ws, 0)
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
            "schema_version": "1.1",                     # bumped: leading sample axis
            "mode": "parity_base",
            "seed": cfg.run.seed,
            "dataset": p.dataset,
            "article_id": p.article_index,                # first article (back-compat)
            "article_index_start": p.article_index,
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
        npz = Path(cfg.output_path) if cfg.output_path else od / f"parity_base_{p.dataset}_{samples_shas[0][:8]}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz),
            top_window_indices=top_window_indices,
            window_scores=window_scores,
            eviction_step_mask=eviction_step_mask,
            generated_tokens=generated_tokens,
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        with open(npz.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved: %s", npz)
        return npz
