"""GSM8K runner — one press, one compression ratio, one auditable run directory.

    python -m gsm8k.run_gsm8k --model <path> --press_name snapkv --compression_ratio 0.5

Writes ``<output_dir>/predictions.jsonl`` and ``<output_dir>/meta.json``. Score with::

    python -m gsm8k.calculate_metrics --runs results/gsm8k/*/

Why this exists next to ``evaluate.py``
---------------------------------------
``evaluate.py`` groups by context, writes a CSV, and has no stopping condition beyond
EOS. For GSM8K that is not enough to produce a number anyone should act on, so this
runner adds the three things a compression comparison needs:

*   **Stop strings** (via ``gsm8k.pipeline``), so a 512-token cap costs ~150 tokens of
    wall clock and run-on continuations never reach the scorer.
*   **A pre-flight budget gate** (``gsm8k.press_budget``). GSM8K prompts are ~150 tokens
    and every press here has a 32-token force-kept observation window, so above a
    certain compression ratio the budget *is* the window and SnapKV, AdaKV and
    DefensiveKV all collapse to the same "keep the last k tokens" policy. That is not a
    result about the methods. The gate computes it before any GPU work and refuses to
    run unless you pass ``--allow_degenerate``.
*   **Measured, not requested, compression.** The diagnostic reads the retained KV
    straight off the cache object (per head, for the AdaKV family) instead of trusting
    ``compression_ratio``, and reports the end-to-end ``effective_retention`` that
    accounts for the uncompressed question and every generated token.

Read the diagnostic block at the end of the run before you read the accuracy.
"""

from __future__ import annotations

import gc
import os
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm

from gsm8k.create_huggingface_dataset import (
    STOP_STRINGS,
    load_gsm8k_dataset,
    read_manifest,
    split_context_question,
)
from gsm8k.press_budget import (
    effective_retention,
    measure_cache,
    preflight,
    resolve_budget,
)

# Importing the module registers the "gsm8k-kv-press-text-generation" pipeline task.
from gsm8k import pipeline as _gsm8k_pipeline  # noqa: F401
from gsm8k.pipeline import TASK_NAME

from kvpress import (
    AdaKVPress,
    CriticalKVPress,
    EfficientAdaSnapKVPress,
    EfficientDefensiveKVPress,
    EfficientLayerDefensiveKVPress,
    ObservedAttentionPress,
    SnapKVPress,
    StreamingLLMPress,
)
from kvpress.presses.efficient_ada_global_scorer_press import (
    EfficientAdaGlobalScorerPress,
)
from kvpress.presses.efficient_ada_scorer_press import EfficientAdaScorerPress


# ---------------------------------------------------------------------------
# Press registry
# ---------------------------------------------------------------------------

#: ``press_name -> factory``. Deliberately a small set: the three families the GSM8K
#: comparison is about, plus a full-cache control and a StreamingLLM reference that
#: costs nothing to run and bounds "how well does position alone do".
PRESS_BUILDERS = {
    "none": lambda: None,
    "snapkv": lambda: SnapKVPress(),
    # Mask-based AdaKV simulation. Head-wise *selection*, but no memory saving — it
    # prints a warning to that effect on every layer. Kept because it is the reference
    # implementation of the AdaKV allocation rule.
    "adakv": lambda: AdaKVPress(SnapKVPress()),
    # The real head-wise implementation (flattened per-head cache + varlen flash attn).
    "efficient_adasnapkv": lambda: EfficientAdaSnapKVPress(),
    "defensivekv": lambda: EfficientDefensiveKVPress(),
    "layer_defensivekv": lambda: EfficientLayerDefensiveKVPress(),
    "streaming_llm": lambda: StreamingLLMPress(),
    # CriticalKV: rescales a ScorerPress's scores by the L1 norm of Wo @ values, then
    # force-keeps a first-stage top-k. Uniform per-head budget (plain ScorerPress.compress),
    # fully vectorised over the batch, real pruning -> safe at B > 1.
    "criticalkv": lambda: CriticalKVPress(SnapKVPress()),
}

#: Presses that are NOT safe to run batched, with the reason. Checked in the runner so a
#: sweep fails fast instead of producing numbers whose meaning depends on batch
#: composition.
BATCH_UNSAFE_PRESSES = {
    "critical_adakv": (
        "CriticalAdaKVPress pools its head-budget allocation across the batch: "
        "`head_budgets` is a [num_key_value_heads] tensor scatter_add-ed from "
        "`top_indices` flattened over ALL rows (criticalkv_press.py:131-134), so at "
        "B > 1 each head's budget is the batch total rather than the row's own. "
        "Sequences would compete for each other's budget. Run it with --batch_size 1, "
        "or add a per-row variant like BatchedLayerDefensiveKVPress."
    ),
}

#: Aliases so the shell scripts can use either vocabulary.
PRESS_ALIASES = {
    "full_cache": "none",
    "baseline": "none",
    "adasnapkv": "adakv",
    "efficient_defensivekv": "defensivekv",
    "efficient_layer_defensivekv": "layer_defensivekv",
}


def build_press(
    press_name: str,
    compression_ratio: float,
    window_size: Optional[int] = None,
    kernel_size: Optional[int] = None,
    batched: bool = False,
):
    """Instantiate a press by name and apply the compression ratio.

    ``batched=True`` swaps in the per-row variant for presses whose upstream budget
    allocation folds the batch axis into a single top-k. Only ``layer_defensivekv``
    needs it today: ``EfficientAdaGlobalScorerPress.compress`` ranks a flattened
    ``[L*B*H*T]`` tensor, so at ``B > 1`` sequences would compete for each other's
    budget and a row's retention would depend on its batch-mates. See
    ``gsm8k.batched_presses``.
    """
    name = PRESS_ALIASES.get(press_name, press_name)
    if name not in PRESS_BUILDERS:
        raise ValueError(
            f"Unknown press_name {press_name!r}. Available: "
            f"{', '.join(sorted(set(PRESS_BUILDERS) | set(PRESS_ALIASES)))}"
        )
    if batched and name == "layer_defensivekv":
        from gsm8k.batched_presses import BatchedLayerDefensiveKVPress

        press = BatchedLayerDefensiveKVPress()
    else:
        press = PRESS_BUILDERS[name]()
    if press is None:
        return None

    press.compression_ratio = compression_ratio
    if window_size is not None and hasattr(press, "window_size"):
        press.window_size = window_size
    if window_size is not None and hasattr(press, "press") and hasattr(press.press, "window_size"):
        press.press.window_size = window_size
    if kernel_size is not None and hasattr(press, "kernel_size"):
        press.kernel_size = kernel_size
    if kernel_size is not None and hasattr(press, "press") and hasattr(press.press, "kernel_size"):
        press.press.kernel_size = kernel_size
    return press


def press_window_size_of(press) -> int:
    """The press's force-kept observation window, or 0 when it has none.

    ``StreamingLLMPress`` has no observation window (it keeps sinks + a recent slice by
    construction), so its budget never collapses the way the score-based presses do.
    """
    if press is None:
        return 0
    if hasattr(press, "window_size"):
        return int(press.window_size)
    inner = getattr(press, "press", None)
    if inner is not None and hasattr(inner, "window_size"):
        return int(inner.window_size)
    return 0


def needs_var_flash_attn(press) -> bool:
    """Whether the press requires the head-wise varlen flash-attention patch."""
    return isinstance(press, (EfficientAdaScorerPress, EfficientAdaGlobalScorerPress))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class GSM8KRunner:
    """End-to-end GSM8K prediction runner for one (press, compression_ratio) cell."""

    def __init__(
        self,
        model: str,
        press_name: str = "none",
        compression_ratio: float = 0.0,
        data_dir: str = "data/gsm8k_cot",
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        num_samples: Any = "max",
        shard: int = 0,
        num_shards: int = 1,
        compress_questions: bool = True,
        max_new_tokens: Optional[int] = None,
        press_window_size: Optional[int] = None,
        press_kernel_size: Optional[int] = None,
        allow_degenerate: bool = False,
        skip_oom: bool = False,
        seed: int = 42,
        batch_size: int = 1,
        pad_to_multiple: Optional[int] = 16,
        max_budget_drift_pct: float = 10.0,
    ) -> None:
        self.model_name = model
        self.press_name = PRESS_ALIASES.get(press_name, press_name)
        self.compression_ratio = float(compression_ratio)
        self.data_dir = Path(data_dir)
        self.compress_questions = bool(compress_questions)
        # batch_size > 1 routes through gsm8k.batched_pipeline. pad_to_multiple=None
        # gives exact-length groups (no padding at all), which is the cleanest
        # configuration — but note it does NOT reproduce B=1 token-for-token: cuBLAS
        # picks different GEMM kernels per batch size, and bf16 rounding then flips
        # near-tied greedy argmaxes (measured 3/6 rows with no press, 4/6 under SnapKV).
        # What is exact at a fixed batch size is determinism and permutation
        # invariance — both verified. Compare batched vs B=1 on ACCURACY within its CI,
        # never on string equality. See batched_pipeline.py "Validation contract".
        self.batch_size = max(1, int(batch_size))
        self.pad_to_multiple = pad_to_multiple
        self.max_budget_drift_pct = float(max_budget_drift_pct)
        self.max_new_tokens_override = max_new_tokens
        self.press_window_size = press_window_size
        self.press_kernel_size = press_kernel_size
        self.allow_degenerate = bool(allow_degenerate)
        self.skip_oom = bool(skip_oom)
        self.seed = int(seed)
        self.num_samples = num_samples
        self.shard = int(shard)
        self.num_shards = int(num_shards)

        if self.press_name == "none" and self.compression_ratio:
            raise ValueError(
                "press_name='none' is the full-cache control; it cannot take a "
                "compression_ratio. Pass a press, or drop --compression_ratio."
            )
        if self.press_name != "none" and not (0.0 < self.compression_ratio < 1.0):
            raise ValueError(
                f"compression_ratio must be in (0, 1) for press {self.press_name!r}, "
                f"got {self.compression_ratio}. Note the DefensiveKV convention: this is "
                f"the fraction REMOVED, so 0.8 keeps 20%."
            )
        if self.batch_size > 1 and self.press_name in BATCH_UNSAFE_PRESSES:
            raise ValueError(
                f"press {self.press_name!r} cannot be run at batch_size "
                f"{self.batch_size}.\n\n{BATCH_UNSAFE_PRESSES[self.press_name]}"
            )
        if self.batch_size > 1 and not self.compress_questions:
            raise ValueError(
                "batch_size > 1 requires compress_questions=True. Batched rows share "
                "one question turn after the prefill; with compress_questions=False "
                "each row carries its own problem statement there, so the question "
                "lengths differ and the batch would need a second ragged axis."
            )

        if output_dir is None:
            tag = "full_cache" if self.press_name == "none" else (
                f"{self.press_name}_cr{self.compression_ratio:.2f}"
            )
            output_dir = f"results/gsm8k/{tag}"
        self.output_dir = Path(output_dir)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.pipe = None
        self.press = None

        # accumulators
        self._context_tokens: List[int] = []
        self._question_tokens: List[int] = []
        self._gen_tokens: List[int] = []
        self._effective_retentions: List[float] = []
        self._measured_kv: List[int] = []
        self._full_kv: List[int] = []
        self._n_degenerate = 0
        self._n_press_skipped = 0
        self._stop_reasons: Dict[str, int] = {}
        self._head_len_min: Optional[int] = None
        self._head_len_max: Optional[int] = None
        self._head_wise = False
        self._batch_plan_meta: Dict[str, Any] = {}
        #: One entry per OOM backoff: {"group_size", "from", "to"}. Non-empty means
        #: some rows ran at a smaller batch than requested, so they are not exactly
        #: comparable to the rest (batched output is batch-size dependent).
        self._batch_size_backoffs: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _load_pipeline(self):
        from transformers import pipeline as hf_pipeline

        press = self.press
        model_kwargs: Dict[str, Any] = {}

        # Mirrors evaluate.py's attention selection exactly, so a run here is the same
        # computation the paper's harness would do.
        if isinstance(press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
        elif needs_var_flash_attn(press):
            from kvpress.ada_attn import replace_var_flash_attn

            replace_var_flash_attn(model_name=self.model_name)
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                print(
                    "flash_attn not installed; falling back to the default attention "
                    "implementation. Throughput will differ from the reference setup, "
                    "accuracy will not."
                )

        model_kwargs["torch_dtype"] = "auto"

        if self.batch_size > 1:
            from gsm8k.batched_pipeline import BATCHED_TASK_NAME

            task = BATCHED_TASK_NAME
            # The head-wise varlen attention carries a stale `assert bsz == 1`; the
            # reshape beneath it is already batch-correct (see batched_presses.py).
            if needs_var_flash_attn(press):
                from gsm8k.batched_presses import enable_batched_ada_attention

                n = enable_batched_ada_attention()
                print(f"Relaxed `assert bsz == 1` on {n} varlen attention class(es).")
        else:
            task = TASK_NAME

        # Load the tokenizer EXPLICITLY rather than letting `pipeline()` do it.
        # `pipeline()` treats the tokenizer as optional: it swallows load failures and
        # leaves `pipe.tokenizer = None`, which then surfaces deep inside
        # `_sanitize_parameters` as `NoneType has no attribute model_max_length` — a
        # trace that points at the pipeline and says nothing about the real cause.
        # Loading it here means the actual error (404, gated repo, missing cache)
        # reaches the user, and it is raised BEFORE the 16 GB of weights are moved.
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise SystemExit(
                f"Could not load a tokenizer for {self.model_name!r}: "
                f"{type(e).__name__}: {e}\n\n"
                f"If the model is already in your HF cache, this is usually the hub "
                f"client requesting a path the repo does not publish (e.g. "
                f"'additional_chat_templates'). Re-run offline:\n"
                f"    export HF_HUB_OFFLINE=1"
            ) from e

        print(f"Loading model {self.model_name} on {self.device} ...")
        if self.device == "auto":
            pipe = hf_pipeline(
                task, model=self.model_name, tokenizer=tokenizer,
                device_map="auto", model_kwargs=model_kwargs,
            )
        else:
            pipe = hf_pipeline(
                task, model=self.model_name, tokenizer=tokenizer,
                device=self.device, model_kwargs=model_kwargs,
            )
        # `pipeline()` treats the tokenizer as optional and swallows load failures, so a
        # tokenizer that 404s leaves `pipe.tokenizer = None` and surfaces 40 frames later
        # as `NoneType has no attribute model_max_length`. Say what actually happened.
        if pipe.tokenizer is None:
            raise SystemExit(
                f"No tokenizer could be loaded for {self.model_name!r}. transformers "
                f"treats it as optional and swallowed the error, but this pipeline "
                f"needs it. Load it yourself to see the real failure:\n"
                f"    AutoTokenizer.from_pretrained({self.model_name!r})"
            )

        print(f"model dtype: {pipe.model.dtype}", flush=True)
        return pipe

    def _context_length(self, example: Dict[str, Any]) -> int:
        """Tokenized length of the part the press will compress.

        Uses the pipeline's own ``preprocess`` so the count includes the chat template
        exactly as the run will see it — a budget computed on the raw string would be
        off by the template's ~25 tokens, which matters when the whole context is ~150.
        """
        context, question = split_context_question(example, self.compress_questions)
        tensors = self.pipe.preprocess(
            context,
            questions=[question],
            answer_prefix=example.get("answer_prefix", ""),
            max_context_length=int(1e10),
        )
        return int(tensors["context_ids"].shape[1])

    # -----------------------------------------------------------------
    # pre-flight
    # -----------------------------------------------------------------

    def _preflight(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refuse to burn GPU hours on a configuration that cannot measure the press."""
        window = press_window_size_of(self.press)
        if self.press is None or window == 0:
            return {}

        print("\nPre-flight budget check (tokenizing contexts) ...")
        lengths = [self._context_length(ex) for ex in tqdm(examples, desc="preflight")]
        report = preflight(lengths, self.compression_ratio, window)
        print(report.format())

        reasons = report.blocking_reasons()
        if reasons:
            print("\n  BUDGET GATE:")
            for r in reasons:
                print(f"    * {r}")
            if not self.allow_degenerate:
                raise SystemExit(
                    "\n  Refusing to run: this configuration would not measure the press.\n"
                    "  Re-run with a workable compression_ratio / --press_window_size, or\n"
                    "  pass --allow_degenerate True to record it anyway (it will be flagged\n"
                    "  in meta.json and in the comparison table)."
                )
            print(
                "\n  --allow_degenerate set: proceeding. The run will be marked "
                "degenerate and is NOT a measurement of the scoring method."
            )
        else:
            print("  budget OK: the press has free slots to rank with on every example.")
        return report.to_dict()

    # -----------------------------------------------------------------
    # run
    # -----------------------------------------------------------------

    def run(self) -> None:
        self._seed_everything()

        manifest = read_manifest(self.data_dir)
        examples = load_gsm8k_dataset(
            self.data_dir,
            num_samples=self.num_samples,
            shard=self.shard,
            num_shards=self.num_shards,
        )
        if not examples:
            raise SystemExit(f"No GSM8K examples loaded from {self.data_dir}")

        self.press = build_press(
            self.press_name,
            self.compression_ratio,
            self.press_window_size,
            self.press_kernel_size,
            batched=self.batch_size > 1,
        )
        self.pipe = self._load_pipeline()

        preflight_report = self._preflight(examples)

        if self.batch_size > 1:
            self._run_batched(examples, manifest, preflight_report)
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "predictions.jsonl"

        print(
            f"\nGSM8K run: {len(examples)} examples, press={self.press_name}, "
            f"compression_ratio={self.compression_ratio}, "
            f"compress_questions={self.compress_questions}, out={self.output_dir}"
        )

        run_start = time.time()
        n_examples = n_failed = n_oom = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(tqdm(examples, desc="gsm8k")):
                try:
                    pred, stats = self._predict(ex)
                except torch.cuda.OutOfMemoryError as e:
                    if not self.skip_oom:
                        raise
                    print(f"example {i}: OOM -- recording as a failure")
                    pred, stats = f"Failure: OOM {e}", {}
                    n_oom += 1
                    n_failed += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:  # noqa: BLE001 - a crash must not be scored as wrong
                    print(f"example {i}: generation failed -- {e}")
                    pred, stats = f"Failure: {e}", {}
                    n_failed += 1

                record = {
                    "id": i,
                    "pred": pred,
                    "answer": ex["answer"],
                    "task": ex["task"],
                    "max_new_tokens": ex["max_new_tokens"],
                    **stats,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_examples += 1

        run_end = time.time()
        elapsed = run_end - run_start
        eps = n_examples / elapsed if elapsed > 0 else 0.0

        print(
            f"\nGSM8K: {n_examples} examples, {n_failed} failed ({n_oom} OOM), "
            f"{elapsed:.1f}s ({eps:.2f} ex/s)"
        )
        self._report_compression_diagnostic()
        self._write_meta(
            manifest, preflight_report, n_examples, n_failed, n_oom,
            run_start, run_end, eps,
        )
        print(f"Predictions written to {out_path}")

    # -----------------------------------------------------------------
    # batched run
    # -----------------------------------------------------------------

    def _predict_batch_with_backoff(
        self, group: List[int], examples: List[Dict[str, Any]]
    ) -> "Tuple[List[Tuple[str, Dict[str, Any]]], int, List[Dict[str, Any]]]":
        """Run one group, halving the batch on OOM instead of losing it.

        Returns ``(results, batch_size_used, backoff_events)``.

        **Backing off is not free and is never silent.** Batched output is batch-size
        dependent — cuBLAS picks different GEMM kernels per batch size, and in bf16 the
        resulting rounding flips near-tied greedy argmaxes (measured: 3/6 rows differ
        between B=6 and B=1 with no press, 4/6 under SnapKV, because a press's top-k
        turns a rounding difference into a different *retention* set). So rows generated
        under a backoff are not on the same footing as the rest of the run.

        Every backoff is therefore recorded per row (``batch_size_used``), summarised in
        ``meta.json``, and surfaced by the scorer as a comparability warning. Recovering
        the work is worth it; pretending the recovery was free is not.
        """
        rows = [examples[i] for i in group]
        size = len(rows)
        events: List[Dict[str, Any]] = []

        while True:
            try:
                if size >= len(rows):
                    return self._predict_batch(rows), len(rows), events
                # Sub-batch: run the group in chunks of `size`, concatenating results.
                out: List[Tuple[str, Dict[str, Any]]] = []
                for start in range(0, len(rows), size):
                    out.extend(self._predict_batch(rows[start : start + size]))
                return out, size, events

            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                gc.collect()
                if size <= 1:
                    print(f"  batch OOM even at size 1 -- recording {len(rows)} failures")
                    return [(f"Failure: OOM {e}", {}) for _ in rows], 1, events
                new_size = max(1, size // 2)
                events.append({
                    "group_size": len(rows), "from": size, "to": new_size,
                })
                print(
                    f"  OOM at batch size {size} -> retrying this group at {new_size}. "
                    f"NOTE: these rows are generated at a different batch size than the "
                    f"rest of the run; recorded in meta.json and flagged by the scorer."
                )
                size = new_size

            except Exception as e:  # noqa: BLE001 - a crash must not be scored as wrong
                print(f"  batch of {len(rows)}: failed -- {e}")
                return [(f"Failure: {e}", {}) for _ in rows], size, events

    def _run_batched(
        self,
        examples: List[Dict[str, Any]],
        manifest: Dict[str, Any],
        preflight_report: Dict[str, Any],
    ) -> None:
        """Group examples by prompt length and generate a group at a time."""
        from gsm8k.batching import assert_drift_within, plan_batches

        lengths = [self._context_length(ex) for ex in tqdm(examples, desc="lengths")]
        plan = plan_batches(lengths, self.batch_size, self.pad_to_multiple)
        print(f"\nbatch plan: {plan.summary()}")

        # Padding sizes the budget off the group max, so a short row is compressed
        # less than requested. Bounded and reported -- but fail if the plan makes it
        # large, because the alternative is a plausible number that quietly
        # corresponds to a weaker ratio than the run's name claims.
        drift = (
            assert_drift_within(
                plan, lengths, self.compression_ratio, self.max_budget_drift_pct
            )
            if self.press is not None
            else {"mean_pct": 0.0, "worst_pct": 0.0, "worst_tokens": 0.0}
        )
        print(
            f"  padding budget drift: mean {drift['mean_pct']:.2f}%  "
            f"worst {drift['worst_pct']:.2f}%  ({drift['worst_tokens']:.0f} extra KV entries)"
        )
        self._batch_plan_meta = {
            "batch_size": self.batch_size,
            "pad_to_multiple": self.pad_to_multiple,
            "n_batches": plan.n_batches,
            "mean_batch_size": round(plan.mean_batch_size, 2),
            "max_intra_group_spread": plan.max_spread,
            "budget_drift": drift,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "predictions.jsonl"
        print(
            f"\nGSM8K run: {len(examples)} examples, press={self.press_name}, "
            f"compression_ratio={self.compression_ratio}, batch_size={self.batch_size}, "
            f"out={self.output_dir}"
        )

        run_start = time.time()
        records: Dict[int, Dict[str, Any]] = {}
        n_failed = n_oom = 0

        # Crash-durable progress. The previous version buffered every result and wrote
        # once at the end, so a kill at batch 50/52 destroyed an hour of finished work
        # (observed: a cr=0.90 cell OOM-killed at batch 3 left an empty directory).
        # `.partial` is appended and flushed per batch; the ordered predictions.jsonl is
        # written at the end as before. On a crash the partial file is what survives,
        # and it is enough to resume or to score what completed.
        partial_path = self.output_dir / "predictions.partial.jsonl"
        partial = open(partial_path, "w", encoding="utf-8")

        try:
            for group in tqdm(plan.groups, desc="batches"):
                results, used_bs, backoffs = self._predict_batch_with_backoff(group, examples)
                if backoffs:
                    self._batch_size_backoffs.extend(backoffs)

                for idx, (pred, stats) in zip(group, results):
                    ex = examples[idx]
                    rec = {
                        "id": idx,
                        "pred": pred,
                        "answer": ex["answer"],
                        "task": ex["task"],
                        "max_new_tokens": ex["max_new_tokens"],
                        # Which batch size THIS row was actually generated at. Batched
                        # output is batch-size dependent, so a row produced under a
                        # backoff is not on the same footing as the rest.
                        "batch_size_used": used_bs,
                        **stats,
                    }
                    records[idx] = rec
                    partial.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    if isinstance(pred, str) and pred.startswith("Failure:"):
                        n_failed += 1
                        if "OOM" in pred:
                            n_oom += 1

                partial.flush()
                os.fsync(partial.fileno())
        finally:
            partial.close()

        # Written in dataset order, not batch order, so predictions.jsonl is
        # comparable line-for-line with a B=1 run.
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(len(examples)):
                f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
        partial_path.unlink(missing_ok=True)  # superseded by the ordered file

        run_end = time.time()
        elapsed = run_end - run_start
        eps = len(examples) / elapsed if elapsed > 0 else 0.0
        print(
            f"\nGSM8K: {len(examples)} examples, {n_failed} failed ({n_oom} OOM), "
            f"{elapsed:.1f}s ({eps:.2f} ex/s)"
        )
        self._report_compression_diagnostic()
        self._write_meta(
            manifest, preflight_report, len(examples), n_failed, n_oom,
            run_start, run_end, eps,
        )
        print(f"Predictions written to {out_path}")

    def _predict_batch(
        self, rows: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate one group. Returns ``(pred, stats)`` per row, in group order."""
        contexts, questions = zip(
            *(split_context_question(ex, self.compress_questions) for ex in rows)
        )
        question = questions[0]
        max_gen = int(self.max_new_tokens_override or rows[0]["max_new_tokens"])

        gen = self.pipe.generate_batch(
            contexts=list(contexts),
            question=question,
            answer_prefix=rows[0].get("answer_prefix", ""),
            press=self.press,
            max_new_tokens=max_gen,
            stop_strings=STOP_STRINGS,
        )

        m = gen.cache_measurement
        n_layers = max(len(m.per_layer_entries), 1)
        bsz = len(rows)
        # Per-row retained tokens: the measurement is a whole-batch total, and every
        # row shares the padded width, so dividing by (rows * heads * layers) gives
        # the mean retained tokens per row.
        heads_per_layer = (
            m.full_kv_entries / (gen.padded_length * n_layers)
            if gen.padded_length and m.full_kv_entries
            else 0
        )
        kept_tokens = (
            m.total_kv_entries / (heads_per_layer * n_layers) / bsz
            if heads_per_layer
            else gen.padded_length
        )

        out: List[Tuple[str, Dict[str, Any]]] = []
        for i, ex in enumerate(rows):
            ctx_len = gen.context_lengths[i]
            budget = resolve_budget(
                gen.padded_length,
                0.0 if self.press is None else self.compression_ratio,
                press_window_size_of(self.press) or 1,
            )
            eff = effective_retention(
                kept_tokens, gen.question_tokens, gen.n_generated[i], ctx_len
            )

            self._context_tokens.append(ctx_len)
            self._question_tokens.append(gen.question_tokens)
            self._gen_tokens.append(gen.n_generated[i])
            self._effective_retentions.append(eff)
            self._measured_kv.append(m.total_kv_entries // bsz)
            self._full_kv.append(m.full_kv_entries // bsz)
            self._stop_reasons[gen.stop_reasons[i]] = (
                self._stop_reasons.get(gen.stop_reasons[i], 0) + 1
            )
            if budget.degenerate:
                self._n_degenerate += 1
            if m.head_wise:
                self._head_wise = True
            if m.head_len_min is not None:
                self._head_len_min = (
                    m.head_len_min if self._head_len_min is None
                    else min(self._head_len_min, m.head_len_min)
                )
            if m.head_len_max is not None:
                self._head_len_max = (
                    m.head_len_max if self._head_len_max is None
                    else max(self._head_len_max, m.head_len_max)
                )

            out.append((
                gen.texts[i],
                {
                    "context_tokens": ctx_len,
                    "padded_context_tokens": gen.padded_length,
                    "question_tokens": gen.question_tokens,
                    "gen_tokens": gen.n_generated[i],
                    "stop_reason": gen.stop_reasons[i],
                    "hit_token_cap": gen.stop_reasons[i] == "max_new_tokens",
                    "n_kept": budget.n_kept,
                    "free_slots": budget.free_slots,
                    "degenerate": bool(budget.degenerate),
                    "press_skipped": False,
                    "batch_size": bsz,
                    "measured_kv_entries": m.total_kv_entries // bsz,
                    "full_kv_entries": m.full_kv_entries // bsz,
                    "measured_context_compression": round(m.measured_compression, 4),
                    "effective_retention": round(eff, 4),
                },
            ))
        return out

    # -----------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------

    def _predict(self, ex: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate one GSM8K answer. Returns ``(pred, per_example_stats)``."""
        context, question = split_context_question(ex, self.compress_questions)
        max_gen = int(self.max_new_tokens_override or ex["max_new_tokens"])

        # A context at or below the press's observation window would trip the press's
        # own `assert q_len > window_size` mid-prefill and kill the run. Fall back to
        # full cache for that example and count it, rather than losing everything.
        press = self.press
        window = press_window_size_of(press)
        press_skipped = False
        if press is not None and window > 0:
            if self._context_length(ex) <= window:
                press, press_skipped = None, True

        output = self.pipe(
            context,
            questions=[question],
            answer_prefix=ex.get("answer_prefix", ""),
            press=press,
            max_new_tokens=max_gen,
            stop_strings=STOP_STRINGS,
        )

        gen = output["generations"][0]
        pred = gen["text"]
        context_tokens = int(output["context_length"])
        question_tokens = int(gen["question_tokens"])
        gen_tokens = int(gen["n_generated"])
        measurement = output["cache_measurement"]

        budget = resolve_budget(
            context_tokens,
            0.0 if press is None else self.compression_ratio,
            window or 1,
        )
        # The retained context size measured off the cache, expressed back in tokens so
        # it is comparable with n_kept. For the head-wise caches this is the mean over
        # heads: individual heads keep different amounts, which is the point of AdaKV.
        n_layers = max(len(measurement.per_layer_entries), 1)
        heads_per_layer = (
            measurement.full_kv_entries / (context_tokens * n_layers)
            if context_tokens > 0 and measurement.full_kv_entries > 0
            else 0
        )
        kept_tokens = (
            measurement.total_kv_entries / (heads_per_layer * n_layers)
            if heads_per_layer
            else context_tokens
        )
        eff_ret = effective_retention(
            kept_tokens, question_tokens, gen_tokens, context_tokens
        )

        # accumulate
        self._context_tokens.append(context_tokens)
        self._question_tokens.append(question_tokens)
        self._gen_tokens.append(gen_tokens)
        self._effective_retentions.append(eff_ret)
        self._measured_kv.append(measurement.total_kv_entries)
        self._full_kv.append(measurement.full_kv_entries)
        self._stop_reasons[gen["stop_reason"]] = (
            self._stop_reasons.get(gen["stop_reason"], 0) + 1
        )
        if press_skipped:
            self._n_press_skipped += 1
        elif budget.degenerate:
            self._n_degenerate += 1
        if measurement.head_wise:
            self._head_wise = True
        if measurement.head_len_min is not None:
            self._head_len_min = (
                measurement.head_len_min if self._head_len_min is None
                else min(self._head_len_min, measurement.head_len_min)
            )
        if measurement.head_len_max is not None:
            self._head_len_max = (
                measurement.head_len_max if self._head_len_max is None
                else max(self._head_len_max, measurement.head_len_max)
            )

        stats: Dict[str, Any] = {
            "context_tokens": context_tokens,
            "question_tokens": question_tokens,
            "gen_tokens": gen_tokens,
            "stop_reason": gen["stop_reason"],
            "hit_token_cap": gen["stop_reason"] == "max_new_tokens",
            "n_kept": budget.n_kept,
            "free_slots": budget.free_slots,
            "degenerate": bool(budget.degenerate and not press_skipped),
            "press_skipped": press_skipped,
            "measured_kv_entries": measurement.total_kv_entries,
            "full_kv_entries": measurement.full_kv_entries,
            "measured_context_compression": round(measurement.measured_compression, 4),
            "effective_retention": round(eff_ret, 4),
        }
        return pred, stats

    # -----------------------------------------------------------------
    # diagnostics + meta
    # -----------------------------------------------------------------

    def _compression_summary(self) -> Dict[str, Any]:
        n = len(self._context_tokens)
        if n == 0:
            return {"compression_measured": False}

        full_kv = sum(self._full_kv)
        measured_kv = sum(self._measured_kv)
        return {
            "compression_measured": True,
            "n_measured": n,
            "mean_context_tokens": round(sum(self._context_tokens) / n, 1),
            "mean_question_tokens": round(sum(self._question_tokens) / n, 1),
            "mean_gen_tokens": round(sum(self._gen_tokens) / n, 1),
            # Ratio of totals, not a mean of per-example ratios: the examples have
            # different lengths, so the mean of ratios would silently reweight them.
            "measured_context_compression": (
                round(1.0 - measured_kv / full_kv, 4) if full_kv else None
            ),
            "requested_compression_ratio": self.compression_ratio,
            "mean_effective_retention": round(
                sum(self._effective_retentions) / n, 4
            ),
            "n_examples_degenerate": self._n_degenerate,
            "pct_examples_degenerate": round(100.0 * self._n_degenerate / n, 2),
            "n_examples_press_skipped": self._n_press_skipped,
            "pct_examples_press_skipped": round(100.0 * self._n_press_skipped / n, 2),
            "stop_reasons": dict(self._stop_reasons),
            "head_wise_cache": self._head_wise,
            "head_len_min": self._head_len_min,
            "head_len_max": self._head_len_max,
        }

    def _report_compression_diagnostic(self) -> None:
        s = self._compression_summary()
        if not s.get("compression_measured"):
            return

        print("\n--- compression diagnostic (read this before the accuracy) ---")
        print(
            f"  context {s['mean_context_tokens']:.1f} tok  + question "
            f"{s['mean_question_tokens']:.1f} tok  + generated "
            f"{s['mean_gen_tokens']:.1f} tok"
        )
        print(
            f"  measured context compression  {s['measured_context_compression']}  "
            f"(requested {self.compression_ratio})"
        )
        print(f"  effective retention (end to end)  {s['mean_effective_retention']}")
        print(
            f"  degenerate {s['n_examples_degenerate']}/{s['n_measured']} "
            f"({s['pct_examples_degenerate']:.1f}%)   "
            f"press skipped {s['n_examples_press_skipped']}/{s['n_measured']} "
            f"({s['pct_examples_press_skipped']:.1f}%)"
        )
        print(f"  stop reasons  {s['stop_reasons']}")
        if s["head_wise_cache"]:
            print(
                f"  head-wise budgets  min {s['head_len_min']}  max {s['head_len_max']} "
                f"tokens/head"
            )
            if s["head_len_min"] == s["head_len_max"]:
                print(
                    "  !! every head got the SAME budget: the adaptive allocation did "
                    "not differentiate, so this run does not distinguish AdaKV-family "
                    "presses from uniform SnapKV."
                )

        if self.press is None:
            return

        requested = self.compression_ratio
        measured = s["measured_context_compression"]
        if measured is not None and abs(measured - requested) > 0.02:
            print(
                f"  !! measured compression {measured:.3f} differs from the requested "
                f"{requested:.3f} by more than 2 points. Integer flooring on a ~"
                f"{s['mean_context_tokens']:.0f}-token context is coarse; report the "
                f"measured number."
            )
        if s["pct_examples_degenerate"] > 10.0:
            print(
                f"  !! the budget collapsed into the press's observation window on "
                f"{s['pct_examples_degenerate']:.1f}% of examples. On those, every press "
                f"here reduces to 'keep the last n_kept tokens' -- this run does NOT "
                f"measure the scoring method. Report it as a control."
            )
        if s["pct_examples_press_skipped"] > 0.0:
            print(
                f"  !! the press was SKIPPED on {s['pct_examples_press_skipped']:.1f}% of "
                f"examples (context <= window_size); those ran at full cache."
            )
        eff = s["mean_effective_retention"]
        if requested >= 0.5 and eff > 0.6:
            print(
                f"  !! compression_ratio={requested} but end-to-end retention is "
                f"{eff:.3f}. kvpress compresses the context ONCE at prefill; the question "
                f"and all {s['mean_gen_tokens']:.0f} generated tokens are never pruned. On "
                f"a short prompt with a long CoT answer the nominal ratio does not "
                f"translate -- quote effective_retention, not compression_ratio."
            )

    def _write_meta(
        self,
        manifest: Dict[str, Any],
        preflight_report: Dict[str, Any],
        n_examples: int,
        n_failed: int,
        n_oom: int,
        run_start: float,
        run_end: float,
        eps: float,
    ) -> None:
        meta = {
            "task": "gsm8k",
            "run_name": self.output_dir.name,
            "num_examples": n_examples,
            "num_failed": n_failed,
            "num_oom": n_oom,
            # Dataset identity — the scorer refuses to compare runs whose dataset_sha
            # or compress_questions setting disagree.
            "dataset_dir": str(self.data_dir),
            "dataset_sha": manifest["dataset_sha"],
            "prompt_style": manifest["prompt_style"],
            "dataset_max_new_tokens": manifest["max_new_tokens"],
            "max_new_tokens_override": self.max_new_tokens_override,
            "compress_questions": self.compress_questions,
            "shard": self.shard,
            "num_shards": self.num_shards,
            "num_samples_requested": self.num_samples,
            # Model / press
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(getattr(self.pipe.model, "dtype", "unknown")),
            "attn_implementation": getattr(
                self.pipe.model.config, "_attn_implementation", "unknown"
            ),
            "press_name": self.press_name,
            "press_repr": str(self.press) if self.press is not None else None,
            "press_window_size": press_window_size_of(self.press),
            "press_kernel_size": getattr(self.press, "kernel_size", None),
            "compression_ratio": self.compression_ratio if self.press else None,
            "head_wise_press": needs_var_flash_attn(self.press),
            "allow_degenerate": self.allow_degenerate,
            "batching": self._batch_plan_meta or {"batch_size": 1},
            # Non-empty => some rows ran at a reduced batch size after an OOM. Their
            # generations are not on the same numerical footing as the rest of the run
            # (see _predict_batch_with_backoff), so the scorer flags it.
            "batch_size_backoffs": self._batch_size_backoffs,
            "n_batch_size_backoffs": len(self._batch_size_backoffs),
            "stop_strings": STOP_STRINGS,
            "decoding": "greedy",
            "seed": self.seed,
            "preflight": preflight_report,
            **self._compression_summary(),
            "transformers_version": _version("transformers"),
            "torch_version": torch.__version__,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "run_started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start)
            ),
            "run_finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_end)
            ),
            "examples_per_second": round(eps, 4),
        }
        (self.output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )


def _version(pkg: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(pkg)
    except Exception:
        return "unknown"


def main(**kwargs) -> None:
    """CLI entry point. See :class:`GSM8KRunner` for the parameters."""
    GSM8KRunner(**kwargs).run()


if __name__ == "__main__":
    Fire(main)
