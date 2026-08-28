"""PerfRunner — wall-clock benchmarks (Suite C).

TTFT, throughput (tok/s), TPOT, peak memory across the backend x budget
factorial. Supports eager (Kaggle-safe) and flash_attn (Ampere+) backends.
Gracefully skips configs that OOM or require unavailable flash-attn.

Method-agnostic by construction
-------------------------------
The measurement core builds a cache, runs one prefill and N decode steps, and
records timings and phase peaks. It never inspects what the cache does, so a
pressed, unpressed, quantized or evicted method is measured identically. Our own
windowed cache has a built-in backend because it is ours; anything else plugs in
through ``cache_backend: external`` + ``method_factory`` and is handed the budget
we resolved, so it is held to the same operating point rather than its own
defaults.

The one method property the core must know is ``compaction`` -- where a method
does its one-off compaction (``prefill`` / ``decode_step0`` / ``none``) -- because
the same physical work belongs in a different reported column depending on the
design. See :func:`resolve_compaction_phase`.

Comparability with the published prompt-compression line of work
-----------------------------------------------------------------
This runner measures THIS PROJECT'S cache and nothing else — no third-party press
is imported or run, and ``protocol`` selects no foreign code path. What can be
borrowed is the *measurement conditions*, taken from one reference
implementation we have actually read: ``evaluation/efficiency_evaluate.py`` in
the DefensiveKV release (ICLR 2026), a fork of NVIDIA's kvpress, whose PRESS_DICT
can drive SnapKV / AdaKV / CriticalKV / DefensiveKV. Note what that does and does
not buy: the published papers do not document a protocol at this resolution (see
``configs/eval_efficiency.yaml``), so matching it makes our numbers reproducible
and comparable to a RE-RUN of a baseline, not automatically to a printed table —
which is why ``cache_backend: external`` exists. Reproducing that reference
implementation's *numbers* needs four things this runner would otherwise do
differently, each now a knob on
:class:`~utils.config.PerfConfig` (all defaulting to the historical behaviour,
all flipped together by ``configs/eval_efficiency.yaml``):

* ``prompt_mode="repeat_sentence"`` — they prefill a tiled English sentence, not
  uniform random token ids. Random ids are out-of-distribution, and an eviction
  policy that ranks by attention score is measured on the score distribution.
* ``budget_basis="context"`` — their budget is ``ratio * context_length``; ours
  is ``ratio * (prefill_len + gen_len)`` (design.md §7).
* ``prefill_logits="last_only"`` — see the field docs; the full [B, L, V] prefill
  logits tensor is method-independent and dwarfs the KV cache it is being
  compared against.
* ``decode_step0_ms`` / ``tpot_steady_ms``, recorded below — their press
  compresses *inside* the prefill forward, so compression is charged to prefill
  latency. Ours cannot (the score hook is a ``forward_hook``, so prefill has no
  scores yet); it fires on decode step 0 instead. Charging that to TPOT would
  flatter our TTFT and penalise our TPOT against their published columns, so the
  two are split and ``prefill_plus_compress_ms`` is the directly comparable one.

None of this changes what a ``native``-protocol run measures.
"""
from __future__ import annotations
import json, math, os, pathlib, time, gc
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from utils.cache_memory import MemoryProbe, format_peak_report
from utils.config import FIRST_EVICTION_STEP_DEFAULT, ExperimentConfig
from utils.env_capture import capture_environment
from utils.logger import get_logger

log = get_logger(__name__)


def _phase_mb(phase) -> float:
    """A phase's peak in MB — device-level if we have it, else torch allocated.

    ``PhasePeak`` carries both; the device figure is what an OOM is decided on,
    so prefer it. On CPU neither exists and this is ``nan``, not ``0``: the phase
    did have a footprint (host RSS, in the probe's own report), and writing a 0
    into a GPU-peak column would read as "used nothing".
    """
    if phase is None:
        return float("nan")
    peak = phase.device_used_peak or phase.torch_alloc_peak
    return peak / (1024 * 1024) if peak else float("nan")

def resolve_method_factory(spec: str):
    """Import a ``"package.module:callable"`` spec and return the callable.

    This is the seam that lets Suite C benchmark a method it knows nothing
    about. It matters more than it looks: the published efficiency protocols in
    this space are UNDER-SPECIFIED (papers report "peak memory and decoding
    latency" without stating context length, batch size, warmup rounds, dtype or
    prompt), so quoting a number out of a paper and putting it beside ours is not
    a controlled comparison. The only sound way to get a baseline is to run the
    baseline yourself, in-process, under the identical protocol -- which requires
    being able to plug one in.

    The factory is called as::

        factory(model=..., tokenizer=..., prefill_len=..., gen_len=...,
                batch_size=..., budget_tokens=..., dtype=..., **method_kwargs)

    and must return a *method handle* implementing:

        new_cache()                  -> a fresh ``past_key_values`` per run (required)
        install_hooks(model, cache)  -> object with .remove(), or None (optional)
        requires_output_attentions   -> bool attribute (optional, default False)
        describe()                   -> str for the npz metadata (optional)

    Nothing here touches this project's cache packages, so an external method
    needs no knowledge of them.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"method_factory must be 'package.module:callable', got {spec!r}")
    mod_name, _, attr = spec.partition(":")
    import importlib
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ValueError(f"method_factory module {mod_name!r} is not importable: {e}") from e
    try:
        factory = getattr(mod, attr)
    except AttributeError as e:
        raise ValueError(f"method_factory {spec!r}: {mod_name} has no {attr!r}") from e
    if not callable(factory):
        raise ValueError(f"method_factory {spec!r} resolved to a non-callable")
    return factory


def resolve_budget_tokens(cache_budget, prefill_len: int, budget_horizon: int):
    """Absolute token budget an external method must honour.

    Computed here, from the same ratio and the same basis the windowed cache
    resolves against, so a plugged-in baseline is held to the identical budget
    rather than to whatever its own defaults would have picked. Returns None for
    an unbudgeted (full-cache) config.
    """
    if cache_budget is None:
        return None
    return int(float(cache_budget) * (prefill_len + budget_horizon))


#: Where a method does its one-off cache compaction. This is the ONLY thing the
#: measurement core needs to know about a method, and it exists because the same
#: physical work lands in a different column depending on the design:
#:
#:   "prefill"      compaction happens inside the prefill forward (press-style:
#:                  SnapKV / AdaKV / StreamingLLM-in-update). Already inside TTFT.
#:   "decode_step0" compaction happens on the first decode step, because the
#:                  scores it needs do not exist until the prefill has returned.
#:                  This project's cache. Charged to step 0, not to TTFT.
#:   "none"         nothing is compacted (full cache, or a quantize-only method
#:                  that never evicts).
#:
#: Everything else -- timing, phases, memory, warmup, cooldown -- is identical for
#: every method, which is what makes the core method-agnostic.
COMPACTION_PHASES = ("prefill", "decode_step0", "none")


def resolve_compaction_phase(config: dict, method=None) -> str:
    """Where this row compacts: explicit ``compaction:`` key, else inferred.

    Getting this wrong is not cosmetic. The published ``prefill_latency`` column
    means "input to first token, including any compression paid on the way", so a
    ``decode_step0`` method must add step 0 to its TTFT and a ``prefill`` method
    must NOT -- its compaction is already inside TTFT. Adding step 0 to a
    press-style or full-cache row overstates its prefill by one whole decode step.
    """
    explicit = config.get("compaction")
    if explicit is not None:
        if explicit not in COMPACTION_PHASES:
            raise ValueError(
                f"config {config.get('name')!r}: compaction must be one of "
                f"{list(COMPACTION_PHASES)}, got {explicit!r}")
        return explicit
    backend = config.get("cache_backend", "dynamic")
    if backend == "windowed":
        return "decode_step0"
    if backend == "external":
        # An external method declares its own; presses are the common case, and
        # "prefill" and "none" map identically anyway (neither adds step 0).
        return getattr(method, "compaction_phase", "prefill")
    return "none"


def uses_eager_scoring(cache_backend: str, cache_package: "Optional[str]") -> bool:
    """True when scores come from the eager hook reading ``attn_weights``."""
    return cache_backend == "windowed" and cache_package == "eager"


def needs_output_attentions(cache_backend: str, cache_package: "Optional[str]",
                            attn_implementation: str, config: dict) -> bool:
    """Whether this cell's forwards must pass ``output_attentions=True``.

    The eager score hook reads ``attn_weights`` off the attention module's
    output, and transformers only populates it under ``output_attentions=True``.
    Without it the hook silently no-ops -- scoring is disabled and eviction falls
    back to sink+local -- so the cell times a *different method* than its name
    says, and times it optimistically, because the scoring work never ran.

    Suite C used to gate this on ``install_hooks_for_measurement``, which is an
    unrelated baseline-overhead knob, so every windowed+eager cell measured the
    degraded path. The quality runners gate on the backend package
    (longbench_runner.py:434, ours_parity_runner.py:345); this matches them.
    """
    if attn_implementation != "eager":
        return False
    if config.get("_external_needs_attn"):
        # An external method may score off attn_weights exactly as our eager
        # backend does; it must not be silently starved of them either.
        return True
    return (uses_eager_scoring(cache_backend, cache_package)
            or bool(config.get("install_hooks_for_measurement")))


def _describe_score_backend() -> str:
    """Which scoring path is live, or why we could not tell.

    Imported lazily and defensively: the cache packages pull in the flash-attn
    hooks, which are not importable on every box this suite runs on, and failing
    to *describe* the backend must never fail the benchmark.
    """
    try:
        from modules.windowed_cache.score_kernel import describe_prefill_backend
        return describe_prefill_backend(torch.cuda.is_available())
    except Exception as e:  # pragma: no cover - environment dependent
        return f"unknown ({type(e).__name__}: {e})"


#: Section names torch has used for "dynamo could not trace this" across
#: versions. 2.x up to ~2.5 used ``graph_break``; newer builds (2.12 here) file
#: the same events under ``unimplemented``. Reading only one name silently
#: reports zero breaks on the other, which is the exact false negative this
#: whole diagnostic exists to prevent — so read all of them.
_DYNAMO_BREAK_SECTIONS = ("graph_break", "unimplemented", "unimplemented_with_reason")


def _dynamo_counters() -> Dict[str, int]:
    """Normalized ``torch._dynamo`` counters, or ``{}`` if unavailable.

    Used to verify that ``STICKYKV_COMPILE_EVICT`` actually *fused* the eviction
    rather than merely tracing it. ``torch.compile`` never fails loudly on a
    graph break — it silently splits the region and runs the pieces eagerly — so
    a compiled eviction that broke on every store mutation issues the same ~273
    launches per layer the eager body does, and reads in the timings as "the
    compiled eviction bought nothing". The counters are the only way to tell that
    apart from "it fused and the eviction was never the bottleneck".

    Normalized keys (raw per-section sums are kept alongside, prefixed
    ``raw_``, so a future rename is visible rather than silently zero):

    ``graph_breaks``
        Places dynamo bailed out of the graph. Nonzero on a compiled eviction
        means part of the body is running eagerly.
    ``frames_ok`` / ``frames_total``
        Frames dynamo compiled in this window. Counters are zeroed *after*
        warmup, so anything here is compilation happening inside the measured
        runs — i.e. a recompile, and its latency is inside the timings. This is
        the version-independent recompile signal: torch has no stable
        ``recompiles`` counter section.
    ``unique_graphs``
        Distinct compiled graphs, for reading how badly it fragmented.
    """
    try:
        from torch._dynamo.utils import counters
    except Exception:  # pragma: no cover - torch build dependent
        return {}
    raw: Dict[str, Dict[str, Any]] = {}
    for section, entries in counters.items():
        try:
            raw[str(section)] = dict(entries)
        except Exception:
            continue

    def _sum(section: str) -> int:
        vals = raw.get(section, {}).values()
        return int(sum(v for v in vals if isinstance(v, (int, float))))

    def _get(section: str, key: str) -> int:
        v = raw.get(section, {}).get(key, 0)
        return int(v) if isinstance(v, (int, float)) else 0

    out: Dict[str, int] = {
        "graph_breaks": sum(_sum(s) for s in _DYNAMO_BREAK_SECTIONS),
        "frames_total": _get("frames", "total"),
        "frames_ok": _get("frames", "ok"),
        "unique_graphs": _get("stats", "unique_graphs"),
    }
    for section in raw:
        out[f"raw_{section}"] = _sum(section)
    return out


def _reset_dynamo_counters() -> None:
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:  # pragma: no cover - torch build dependent
        pass


def _lse_recompute_count() -> Optional[int]:
    """How many times ``compute_lse`` ran, or ``None`` if unreadable."""
    try:
        from modules.windowed_cache.score_kernel import lse_recompute_count
        return int(lse_recompute_count())
    except Exception:  # pragma: no cover - environment dependent
        return None


def _reset_lse_recompute_count() -> None:
    try:
        from modules.windowed_cache.score_kernel import reset_lse_recompute_count
        reset_lse_recompute_count()
    except Exception:  # pragma: no cover - environment dependent
        pass


def _lse_reuse_requested() -> bool:
    """Whether the run asked for L to come from the forward (the default)."""
    return os.environ.get(
        "STICKYKV_SCORE_LSE_FROM_FORWARD", "1"
    ).strip().lower() in ("1", "true", "yes", "on")


def _lse_transient_gb(batch_size: int, model_config: Any, prefill_len: int) -> float:
    """GB of the ``[B, H_kv, rep, chunk, S]`` fp32 block ``compute_lse`` builds.

    Sized so the L-reuse failure below can say what it costs at THIS shape rather
    than quoting one example. ``chunk`` is the query-row block
    (``STICKYKV_PREFILL_SCORE_CHUNK``, default 1024) and ``S`` is the prefill
    length, so the term is quadratic in context and linear in batch.
    """
    try:
        chunk = int(os.environ.get("STICKYKV_PREFILL_SCORE_CHUNK", "1024") or 1024)
    except (TypeError, ValueError):
        chunk = 1024
    h_q = int(getattr(model_config, "num_attention_heads", 0) or 0)
    n = max(1, int(batch_size)) * h_q * min(max(chunk, 1), prefill_len) * prefill_len
    return n * 4 / (1024 ** 3)


def _evict_path_stats() -> Dict[str, int]:
    try:
        from modules.windowed_cache.cache import evict_path_stats
        return evict_path_stats()
    except Exception:  # pragma: no cover - environment dependent
        return {}


def _reset_evict_path_stats() -> None:
    try:
        from modules.windowed_cache.cache import reset_evict_path_stats
        reset_evict_path_stats()
    except Exception:  # pragma: no cover - environment dependent
        pass


def describe_tier_geometry(resolved, prefill_len: int) -> Dict[str, Any]:
    """What the resolved budget actually buys, in TOKENS the decode must walk.

    ``cache_budget`` is a **byte** budget (design.md §7). int2 is ~8x denser than
    fp16, so spending a fraction ``q`` of those bytes on the Q tier buys ~8q of
    the token count, and the *number of keys attention reads* is not the budget
    at all — it is ``fp_tokens + N_q * window_size``. At ``quant_ratio=0.5``,
    ``cache_budget=0.5`` that comes out ABOVE the prompt length: 4885 effective
    keys for a 4096-token prefill on Llama-3-8B geometry (1.19x), against 0.50x
    for the same budget at ``quant_ratio=0.0``.

    That is not a bug in the resolver — those really are half the bytes — but it
    is decisive for a *latency* table, and nothing in the npz previously recorded
    it. A q=0.5 row and a q=0.0 row at the same nominal budget do 2.4x different
    amounts of attention work, and a reader comparing either against a full-cache
    baseline needs ``expansion`` to know which. Recorded per config so the two
    rows can be read side by side.
    """
    ws = int(resolved.window_size)
    fp_tokens = (
        int(resolved.num_sink_tokens)
        + int(resolved.top_k_fp) * ws
        + int(resolved.local_tokens)
    )
    q_tokens = int(resolved.N_q) * ws
    s_eff = fp_tokens + q_tokens
    # The same byte budget spent entirely on fp16 — i.e. what the quant_ratio=0.0
    # row at this budget retains. `top_k_windows` is exactly that count (it is
    # what top_k_fp collapses to at q=0), so this needs no second resolve.
    s_eff_q0 = (
        int(resolved.num_sink_tokens)
        + int(resolved.top_k_windows) * ws
        + int(resolved.local_tokens)
    )
    return {
        "window_size": ws,
        "num_sink_tokens": int(resolved.num_sink_tokens),
        "local_tokens": int(resolved.local_tokens),
        "top_k_windows": int(resolved.top_k_windows),
        "top_k_fp": int(resolved.top_k_fp),
        "N_q": int(resolved.N_q),
        "quant_ratio": float(resolved.quant_ratio),
        "fp_tokens": fp_tokens,
        "q_tokens": q_tokens,
        # The number of keys one decode step attends over — the quantity that
        # actually sets decode cost, as opposed to the byte budget.
        "s_eff": s_eff,
        # What the SAME byte budget retains at quant_ratio=0.0 — the reference
        # point for reading the row above.
        "s_eff_at_q0": s_eff_q0,
        # s_eff / prefill_len. >= 1.0 means the "compressed" cache is longer than
        # the prompt it compressed.
        "expansion": (s_eff / prefill_len) if prefill_len else float("nan"),
        "expansion_at_q0": (s_eff_q0 / prefill_len) if prefill_len else float("nan"),
        # Serial iterations of the fused decode kernel's Q-tier loop, per layer
        # per token (decode_kernel.py: `for w in range(0, n_active)`).
        "decode_q_loop_iters": int(resolved.N_q),
    }


def _nanmax2(a: float, b: float) -> float:
    """max(a, b) treating nan as missing; nan only if BOTH are missing."""
    if a != a:
        return b
    if b != b:
        return a
    return max(a, b)


#: LongBench's dataset names, read from the config the loader itself uses so the
#: two cannot drift. Used only to route a bare name to the right loader.
def _longbench_names() -> "set[str]":
    import json
    from pathlib import Path as _P
    cfg = _P(__file__).resolve().parents[2] / "data" / "longbench_configs" / "dataset2maxlen.json"
    try:
        with open(cfg, encoding="utf-8") as fh:
            return set(json.load(fh))
    except Exception:          # pragma: no cover - missing config file
        return set()


#: Suffixes CorpusLoader parses structurally (one record per line / JSON array).
#: Anything else on disk is read as a single plain-text document.
_STRUCTURED_SUFFIXES = (".jsonl", ".ndjson", ".json")


def iter_path_texts(path: "pathlib.Path"):
    """Yield documents from a file or directory, whatever the extension.

    CorpusLoader only recognises .txt/.jsonl/.ndjson/.json, so a corpus that
    happens to be ``notes.md``, ``corpus.text`` or an extensionless dump was
    rejected outright. For a wall-clock benchmark the extension is irrelevant --
    what is needed is bytes of text -- so structured formats are delegated to
    CorpusLoader for proper record parsing and everything else is read as one
    plain-text document.

    Directories are walked recursively in sorted path order, so the corpus is
    always presented in the same sequence and a run is reproducible.
    """
    from data.corpus_loader import CorpusLoader
    if path.is_file():
        if path.suffix.lower() in _STRUCTURED_SUFFIXES:
            for article in CorpusLoader(str(path)).load():
                yield article
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                yield text
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield from iter_path_texts(child)


def iter_corpus_texts(data_source: str):
    """Yield document strings from *data_source*, whatever kind of corpus it is.

    Routing, in order:

    1. an explicit ``longbench:`` / ``corpus:`` prefix wins;
    2. a path that EXISTS on disk is read from disk -- any extension, file or
       directory (this is the common case: point the benchmark at your own text);
    3. a bare name matching a LongBench dataset goes to the LongBench loader;
    4. anything else goes to ``CorpusLoader``, which resolves hub corpora
       (``wikitext-103``, ``pg19``) and the ``local:`` prefix.

    A path is checked before the name lookup so a local directory can never be
    shadowed by a dataset that happens to share its name.
    """
    import pathlib as _pl

    if data_source.startswith("longbench:"):
        from data.longbench_loader import load_longbench_dataset
        for rec in load_longbench_dataset(data_source[len("longbench:"):]):
            yield rec["context"] + "\n\n" + rec["input"]
        return

    name = data_source[len("corpus:"):] if data_source.startswith("corpus:") else data_source
    forced_corpus = data_source.startswith("corpus:")

    candidate = _pl.Path(name[len("local:"):] if name.startswith("local:") else name)
    if candidate.exists():
        yield from iter_path_texts(candidate)
        return

    if not forced_corpus and name in _longbench_names():
        from data.longbench_loader import load_longbench_dataset
        for rec in load_longbench_dataset(name):
            yield rec["context"] + "\n\n" + rec["input"]
        return

    # Not a path, not a LongBench name. It may still be a hub corpus; if it is
    # not, say so in terms of what was actually tried, because the bare
    # "Unsupported dataset" this used to surface reads as a bad NAME when the
    # usual cause is a mistyped or non-existent PATH.
    from data.corpus_loader import CorpusLoader
    try:
        articles = CorpusLoader(name).load()
    except Exception as e:
        raise ValueError(
            f"perf.data_source={data_source!r} is not a readable corpus. It is "
            f"not an existing file or directory, not one of the "
            f"{len(_longbench_names())} LongBench dataset names, and "
            f"CorpusLoader rejected it ({type(e).__name__}: {e}). Pass a path to "
            f"a text file or directory, a LongBench name, or a built-in corpus "
            f"(wikitext-103, pg19)."
        ) from e
    for article in articles:
        yield article


def chunk_texts_to_length(texts, tokenizer, length: int, pool: int) -> "List[torch.Tensor]":
    """Up to *pool* chunks of exactly *length* tokens, taken from *texts* in order.

    Tokenizes INCREMENTALLY and stops as soon as the pool is full. The previous
    implementation joined the whole corpus into one string and tokenized it in a
    single call, which is fine for a LongBench split and hopeless for
    wikitext-103: ~100M tokens materialized to select a handful of rows.

    Documents are concatenated with a blank line between them and cut on token
    boundaries, so a chunk may span a document boundary. For a wall-clock
    benchmark that is irrelevant -- what is being measured is shaped by the token
    count, not by where an article ends.
    """
    chunks: "List[torch.Tensor]" = []
    buf: "List[int]" = []
    sep = tokenizer.encode("\n\n", add_special_tokens=False)
    for text in texts:
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.extend(sep)
        while len(buf) >= length:
            chunks.append(torch.tensor(buf[:length], dtype=torch.long))
            del buf[:length]
            if len(chunks) >= pool:
                return chunks
    return chunks


def build_repeat_sentence_ids(tokenizer, sentence: str, prefill_len: int,
                              batch_size: int) -> torch.Tensor:
    """``sentence`` tiled and truncated to *exactly* ``prefill_len`` tokens.

    The kvpress-family harness builds its prompt as ``sentence * context_length``
    then slices to ``context_length`` tokens. Tiling by the token count rather
    than by the character count avoids encoding ~1.4 MB of text for a 32k prompt,
    but it cannot be computed in one shot: tokenizers merge across the seam
    between copies, so ``len(encode(s * n)) != n * len(encode(s))``. Hence encode,
    check, and grow — then slice, so the length is exact whatever the tokenizer did.

    Every row is identical, which is what a pure wall-clock benchmark wants: no
    ragged padding, and the eviction policy sees the same score distribution on
    every row.
    """
    once = tokenizer.encode(sentence, add_special_tokens=False)
    if len(once) == 0:
        raise ValueError(f"repeat_sentence tokenizes to nothing: {sentence!r}")
    reps = max(1, math.ceil(prefill_len / len(once)) + 1)
    ids = tokenizer.encode(sentence * reps, add_special_tokens=False)
    while len(ids) < prefill_len:
        reps *= 2
        ids = tokenizer.encode(sentence * reps, add_special_tokens=False)
    row = torch.tensor(ids[:prefill_len], dtype=torch.long).unsqueeze(0)
    return row.expand(max(1, int(batch_size)), -1).contiguous()


def _logits_kwarg_name(model) -> "Optional[str]":
    """Name of the "only compute the last position's logits" kwarg, or None.

    transformers called it ``num_logits_to_keep`` through 4.4x (this project pins
    4.47.1) and renamed it ``logits_to_keep`` in 4.49. Probing the signature keeps
    ``prefill_logits="last_only"`` working across both instead of pinning the
    runner to one spelling.
    """
    import inspect
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return None
    for nm in ("logits_to_keep", "num_logits_to_keep"):
        if nm in params:
            return nm
    return None


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # type: ignore
        return True
    except ImportError:
        return False

def build_cells(pc) -> List[tuple]:
    """The ``(prefill_len, gen_len, batch_size)`` cells this run will measure.

    An explicit ``grid:`` wins and is used verbatim. Otherwise the cells are the
    CARTESIAN PRODUCT of ``prefill_lengths x gen_lengths x batch_sizes``, each
    axis falling back to its scalar (``gen_len`` / ``batch_size``) when the list
    is empty -- so every pre-existing config produces exactly the cells it did
    before, while a sweep can be driven entirely from the CLI::

        --override perf.grid=[] perf.prefill_lengths=[4096,8192] \
                   perf.gen_lengths=[129,1025] perf.batch_sizes=[1,4]

    Note ``gen_len`` is decode steps + 1 (the prefill emits the first token);
    ``scripts/run_efficiency.sh --decode`` does that conversion so the off-by-one
    is not left to the caller.
    """
    if pc.grid:
        return [
            (int(g["prefill_len"]), int(g["gen_len"]),
             int(g.get("batch_size", pc.batch_size)))
            for g in pc.grid
        ]
    gens = list(pc.gen_lengths) or [pc.gen_len]
    batches = list(pc.batch_sizes) or [pc.batch_size]
    return [(int(p), int(g), int(b))
            for p in pc.prefill_lengths for g in gens for b in batches]


class PerfRunner:
    """Suite C — TTFT, throughput, TPOT benchmarks."""
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        # Per-config diagnostics for the CURRENT grid cell, filled by
        # _measure_config and folded into the npz by _save. Reset per cell:
        # tier geometry and the L-transient are both shape-dependent, so a
        # carried-over entry would attribute one cell's evidence to another.
        self._config_diag: Dict[str, Dict[str, Any]] = {}

    def run(self) -> List[Path]:
        cfg = self.config
        pc = cfg.perf
        log.info("=== Performance Runner ===")
        flash_ok = _flash_attn_available()
        if not flash_ok:
            log.info("flash-attn not available — flash configs will be skipped")
        env = capture_environment()
        # Try to lock GPU clocks
        clocks_locked = False
        if pc.enable_clock_locking:
            clocks_locked = self._try_lock_clocks()
            settle = float(getattr(pc, "clock_settle_s", 0.0) or 0.0)
            if clocks_locked and settle > 0:
                # nvidia-smi returns before the part is actually at the requested
                # clock; measuring through the ramp puts the transition in run 0.
                log.info("waiting %.1fs for clocks to settle", settle)
                time.sleep(settle)
        if not clocks_locked and not float(getattr(pc, "cooldown_s", 0.0) or 0.0):
            log.warning(
                "clocks are NOT locked and cooldown_s is 0 — back-to-back runs "
                "will heat the GPU and clock it down, so later runs are measured "
                "on a slower part than earlier ones. Set perf.cooldown_s (and "
                "perf.enable_clock_locking where permissions allow) before "
                "quoting these numbers."
            )
        # Build (prefill_len, gen_len) grid. Explicit `grid:` wins; otherwise
        # fall back to legacy `prefill_lengths` x scalar `gen_len`.
        cells = build_cells(pc)
        log.info("%d cell(s): %s", len(cells),
                 ", ".join(f"(p={p}, gen={g}, bs={b})" for p, g, b in cells))
        output_paths = []
        for prefill_len, gen_len, batch_size in cells:
            log.info("--- Cell: prefill=%d gen=%d batch=%d ---", prefill_len, gen_len, batch_size)
            result = self._run_prefill(prefill_len, gen_len, batch_size, pc, cfg, flash_ok, env)
            result["clocks_locked"] = clocks_locked
            result["gen_len"] = gen_len
            result["batch_size"] = batch_size
            path = self._save(result, prefill_len, gen_len, batch_size, cfg, env, clocks_locked)
            output_paths.append(path)
        return output_paths

    def _run_prefill(self, prefill_len: int, gen_len: int, batch_size: int, pc, cfg, flash_ok: bool, env: dict) -> dict:
        configs = pc.configs
        n_configs = len(configs)
        n_runs = pc.num_measurement_runs
        self._config_diag = {}
        ttft = np.full((n_configs, n_runs), np.nan)
        throughput = np.full((n_configs, n_runs), np.nan)
        tpot = np.full((n_configs, n_runs), np.nan)
        e2e_latency = np.full((n_configs, n_runs), np.nan)
        # Compression accounting (see _measure_config): step 0 is where this
        # design compacts the prompt, so it is neither prefill nor steady state.
        decode_step0 = np.full((n_configs, n_runs), np.nan)
        tpot_steady = np.full((n_configs, n_runs), np.nan)
        prefill_plus_compress = np.full((n_configs, n_runs), np.nan)
        peak_mem = np.full((n_configs, n_runs), np.nan)
        # Peak detail, per (config, run). `peak_mem` above stays exactly what it
        # was (torch max_memory_allocated) so old npz readers keep working; these
        # add what a max-B decision actually needs — see utils/cache_memory.py's
        # snapshot_device_memory for why allocated / reserved / device-used are
        # three different numbers and only the last one decides an OOM.
        peak_reserved = np.full((n_configs, n_runs), np.nan)
        peak_device = np.full((n_configs, n_runs), np.nan)
        device_total = np.full((n_configs, n_runs), np.nan)
        peak_prefill = np.full((n_configs, n_runs), np.nan)
        peak_decode = np.full((n_configs, n_runs), np.nan)
        # Step 0 still holds the whole prompt; only `steady` is the compressed
        # cache. Reporting one "decode peak" hides the compression entirely.
        peak_decode_step0 = np.full((n_configs, n_runs), np.nan)
        peak_decode_steady = np.full((n_configs, n_runs), np.nan)
        peak_host_rss = np.full((n_configs, n_runs), np.nan)
        alloc_retries = np.full((n_configs, n_runs), np.nan)
        # A config is "skipped" for three very different reasons and max-B is
        # DEFINED by the mask (eval_perf_batched.yaml: "a config's max-B is the
        # largest batch_size it did NOT skip"). Conflating an OOM with a config
        # error silently reports a smaller max-B than the method achieves, so
        # record which happened. `skipped` stays the union, for compatibility.
        skipped = np.zeros(n_configs, dtype=bool)
        oom = np.zeros(n_configs, dtype=bool)
        errored = np.zeros(n_configs, dtype=bool)
        skip_reason = ["" for _ in range(n_configs)]
        names = []
        attn_impls = []
        for ci, c in enumerate(configs):
            name = c.get("name", f"config_{ci}")
            names.append(name)
            attn_impls.append(c.get("attn_implementation", "eager"))
            # Check flash requirement
            if c.get("requires_flash_attn", False) and not flash_ok:
                if pc.skip_if_flash_attn_unavailable:
                    log.info("Skipping %s (flash-attn unavailable)", name)
                    skipped[ci] = True
                    skip_reason[ci] = "flash_attn_unavailable"
                    continue
            log.info("Running config: %s", name)
            try:
                measurements = self._measure_config(c, prefill_len, gen_len, batch_size, pc, cfg)
                for ri, m in enumerate(measurements):
                    ttft[ci, ri] = m["ttft_ms"]
                    throughput[ci, ri] = m["throughput_tokps"]
                    tpot[ci, ri] = m["tpot_ms"]
                    e2e_latency[ci, ri] = m["e2e_latency_ms"]
                    decode_step0[ci, ri] = m["decode_step0_ms"]
                    tpot_steady[ci, ri] = m["tpot_steady_ms"]
                    prefill_plus_compress[ci, ri] = m["prefill_plus_compress_ms"]
                    peak_mem[ci, ri] = m["peak_memory_mb"]
                    peak_reserved[ci, ri] = m["peak_reserved_mb"]
                    peak_device[ci, ri] = m["peak_device_used_mb"]
                    device_total[ci, ri] = m["device_total_mb"]
                    peak_prefill[ci, ri] = m["peak_prefill_mb"]
                    peak_decode[ci, ri] = m["peak_decode_mb"]
                    peak_decode_step0[ci, ri] = m["peak_decode_step0_mb"]
                    peak_decode_steady[ci, ri] = m["peak_decode_steady_mb"]
                    peak_host_rss[ci, ri] = m["peak_host_rss_mb"]
                    alloc_retries[ci, ri] = m["alloc_retries"]
            except torch.cuda.OutOfMemoryError:
                if pc.skip_if_oom:
                    log.warning("OOM on %s at prefill %d batch %d — skipping",
                                name, prefill_len, batch_size)
                    skipped[ci] = True
                    oom[ci] = True
                    skip_reason[ci] = "oom"
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                # NOT an OOM: a config / env / code error. It says nothing about
                # whether this batch size fits, so it must not be read as one.
                log.warning("Error on %s: %s: %s — skipping (NOT an OOM)",
                            name, type(e).__name__, e)
                skipped[ci] = True
                errored[ci] = True
                skip_reason[ci] = f"error: {type(e).__name__}: {e}"[:200]
        if errored.any():
            log.warning(
                "%d config(s) failed for non-OOM reasons at prefill=%d batch=%d: "
                "%s. These are NOT max-B evidence — fix them before reading the "
                "ladder.",
                int(errored.sum()), prefill_len, batch_size,
                ", ".join(n for n, e in zip(names, errored) if e),
            )
        return {"names": names, "attn_impls": attn_impls, "ttft": ttft,
                "throughput": throughput, "tpot": tpot, "e2e_latency": e2e_latency,
                "decode_step0": decode_step0, "tpot_steady": tpot_steady,
                "prefill_plus_compress": prefill_plus_compress,
                "peak_mem": peak_mem, "skipped": skipped,
                "oom": oom, "errored": errored, "skip_reason": skip_reason,
                "peak_reserved": peak_reserved, "peak_device": peak_device,
                "device_total": device_total, "peak_prefill": peak_prefill,
                "peak_decode": peak_decode,
                "peak_decode_step0": peak_decode_step0,
                "peak_decode_steady": peak_decode_steady,
                "peak_host_rss": peak_host_rss,
                "alloc_retries": alloc_retries,
                "diagnostics": dict(self._config_diag)}

    @staticmethod
    def _warmup_decode_steps(pc, c: dict, cfg, gen_len: int) -> int:
        """How many decode steps each warmup round runs.

        Auto sizes it to cross the first compaction, because that is the step
        carrying the one-time costs: the Triton eviction kernel JIT-compiles and
        autotunes on its first call, which happens at ``first_eviction_step``. A
        prefill-only warmup never reaches it, so the compile lands in measurement
        run 0 and reads as a slow compaction.
        """
        explicit = getattr(pc, "warmup_decode_steps", None)
        if explicit is not None:
            return max(0, min(int(explicit), max(gen_len - 1, 0)))
        if c.get("cache_backend") == "windowed":
            fes = c.get("first_eviction_step",
                        getattr(cfg.cache, "first_eviction_step",
                                FIRST_EVICTION_STEP_DEFAULT))
            want = int(fes) + 2
        else:
            want = 2      # baseline: just warm the decode kernels
        return max(0, min(want, max(gen_len - 1, 0)))

    @staticmethod
    def _cooldown(pc, label: str) -> None:
        """Quiesce the device, then idle so the next run starts from rest.

        Back-to-back timed loops run the GPU hot and it clocks down, so later
        runs are measured on a slower part than earlier ones. That drift is
        monotonic across the run index, which makes it easy to read as method
        variance rather than as the thermal artefact it is.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        secs = float(getattr(pc, "cooldown_s", 0.0) or 0.0)
        if secs > 0:
            log.debug("cooldown %.1fs after %s", secs, label)
            time.sleep(secs)

    def _measure_config(self, c: dict, prefill_len: int, gen_len: int, batch_size: int, pc, cfg) -> List[dict]:
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtypes.get(cfg.model.dtype, torch.float16)
        attn_impl = c.get("attn_implementation", "eager")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=torch_dtype, attn_implementation=attn_impl,
            device_map="auto")
        model.eval()
        # Create input. batch_size>1 measures batched throughput; the windowed
        # cache evicts each row independently (equal-length inputs, no padding).
        batch_size = max(1, int(batch_size))
        data_source = c.get("data_source") or getattr(pc, "data_source", None)
        # `auto` preserves the historical rule (dataset iff data_source is set);
        # the explicit modes are what a standardized protocol pins.
        prompt_mode = c.get("prompt_mode") or getattr(pc, "prompt_mode", "auto")
        if prompt_mode == "auto":
            prompt_mode = "dataset" if data_source else "random_tokens"
        if prompt_mode == "dataset":
            input_ids = self._build_input_ids(
                data_source, tokenizer, prefill_len, batch_size, cfg.run.seed
            ).to(model.device)
        elif prompt_mode == "repeat_sentence":
            sentence = c.get("repeat_sentence") or getattr(
                pc, "repeat_sentence", "The quick brown fox jumps over the lazy dog."
            )
            input_ids = build_repeat_sentence_ids(
                tokenizer, sentence, prefill_len, batch_size
            ).to(model.device)
        else:
            input_ids = torch.randint(
                100, 30000, (batch_size, prefill_len), device=model.device
            )
        log.info("prompt_mode=%s -> input_ids %s", prompt_mode, tuple(input_ids.shape))
        # Setup cache
        cache_backend = c.get("cache_backend", "dynamic")
        cache_pkg = c.get("cache_package")
        budget = c.get("cache_budget")
        # Windowed-cache setup: resolve classes + RoPE once, but DO NOT install
        # hooks here. Each warmup/measurement iteration creates a fresh cache,
        # so hooks must be (re)installed on the cache the model actually
        # receives, then removed before the next iteration.
        WC = WCC = install_hooks = None
        cc = None
        rope = None
        if cache_backend == "windowed" and cache_pkg:
            from utils.cache_factory import (
                assert_transformers_version_supported,
                get_cache_classes,
                validate_backend_attn_pairing,
            )
            # Fail fast: the windowed cache's RoPE handling assumes monotonic
            # cache_position (transformers <= 4.47).
            assert_transformers_version_supported()
            # Realize the compiled decode paths on GPU. Both are opt-in (
            # torch.compile buys nothing on the CPU dev box), and both are enabled
            # here with setdefault semantics so an explicit env — including a
            # deliberate "0" — always wins.
            #
            # STICKYKV_COMPILE_EVICT is the one that matters. Decode at these
            # shapes is bound by DISPATCH COUNT, not by the kernels the dispatches
            # feed: TPOT is flat to within 4% across a 32x change in batch size and
            # a 4.2x change in effective sequence length, which only happens when
            # the GPU is idle waiting on the host. Counting ATen dispatches through
            # WindowedCache.update on Llama-3-8B geometry (ws=8, q=0.5):
            #
            #   eviction step   273 launching ops/layer  ->  8,736 launches, x32 layers
            #   normal step       9 launching ops/layer  ->    288 launches
            #   amortized (ws=8) 42 ops/layer/token      ->  1,344 launches/token
            #                    ... of which the eviction step is 81%.
            #
            # So the eviction step IS the budget, and the compiled body is the
            # thing built to collapse it. It had never been switched on anywhere
            # outside its unit test, which means every perf number recorded to date
            # measured the eager eviction. Verify it with the dynamo counters and
            # evict_path_stats() below rather than trusting the flag.
            #
            # STICKYKV_COMPILE_READ, by contrast, is INERT on the fused decode
            # path: _read_fn is only reachable through QuantizedStore.
            # effective_q_tier, and the fused path (default on CUDA since the
            # decode kernel landed) hands raw int2 straight to Triton and never
            # materializes the Q tier. It is left enabled because it is still the
            # fast path whenever the fused kernel is off (STICKYKV_FUSED_DECODE=0,
            # or an empty Q tier), not because it does anything for the rows below.
            for _flag in ("STICKYKV_COMPILE_READ", "STICKYKV_COMPILE_EVICT"):
                if torch.cuda.is_available() and os.environ.get(_flag) is None:
                    os.environ[_flag] = "1"
                    log.info("enabled %s=1 for CUDA windowed decode", _flag)
            # Fair measurement: reject a backend/attn mismatch up front, exactly
            # as the quality runners do (longbench/gsm8k/ruler/ours_parity). Without
            # it, cache_package='eager' paired with flash attention (or vice
            # versa) would install a score hook that never gets its inputs and
            # silently time the degraded sink+local method under this config's
            # name — the one measurement-integrity hole Suite C had. Raises
            # ConfigValidationError, caught per-config as `errored` (not `oom`).
            validate_backend_attn_pairing(cache_pkg, attn_impl)
            WC, WCC, install_hooks = get_cache_classes(cache_pkg)
            # Window geometry is per-config-overridable, like cache_budget and
            # quant_ratio, so one run can sweep window_size across rows instead
            # of needing a separate invocation per value.
            w = cfg.window
            window_size = int(c.get("window_size", w.window_size))
            num_sink_tokens = int(c.get("num_sink_tokens", w.num_sink_tokens))
            local_window_size = c.get("local_window_size", w.local_window_size)
            # quant_ratio is per-config (like cache_budget), falling back to the
            # shared cache.quant_ratio; 0.0 keeps the pure-fp16 path (design.md §7).
            quant_ratio = c.get("quant_ratio", getattr(cfg.cache, "quant_ratio", 0.0))
            # None (default) = auto: memoize the dequantized Q tier at B=1, not
            # above. The default is set by the memory bound (the memo costs
            # ~149 MB/row against ~131 MB/row of actual KV, halving max-B); what
            # it trades that for — ~8x fewer Q-tier dequants per step — is a
            # wall-clock question only this suite can answer. Set it explicitly
            # to measure both sides. See BATCHING_PLAN.md §5.
            memoize = c.get("quant_memoize_read",
                            getattr(cfg.cache, "quant_memoize_read", None))
            # Decode step of the first eviction (independent of window_size);
            # per-config override falling back to the shared cache setting.
            first_eviction_step = c.get(
                "first_eviction_step", getattr(cfg.cache, "first_eviction_step", FIRST_EVICTION_STEP_DEFAULT)
            )
            cc = WCC(window_size=window_size, num_sink_tokens=num_sink_tokens,
                      local_window_size=local_window_size,
                      cache_budget=budget if budget is not None else 0.5,
                      rerotate_on_evict=getattr(cfg.cache, "rerotate_on_evict", False),
                      quant_ratio=quant_ratio,
                      quant_memoize_read=memoize,
                      first_eviction_step=first_eviction_step)
            # Two-pass RoPE discovery (mirrors ours_parity_runner.py).
            for nm, mod in model.named_modules():
                if "rotary" in nm.lower() or "rope" in nm.lower():
                    rope = mod; break
            if rope is None:
                for nm, mod in model.named_modules():
                    if hasattr(mod, "rotary_emb"):
                        rope = mod.rotary_emb; break
            if rope is None:
                from utils.config import ConfigValidationError
                raise ConfigValidationError(
                    "Could not locate a RoPE module on the model. WindowedCache "
                    "requires a rotary embedding module for key rerotation."
                )

        # What the fractional budget is a fraction OF. `prefill_plus_gen`
        # (default) sizes it against the FULL expected sequence per design.md §7,
        # which is what ours_parity_runner and longbench_runner both pass, so the
        # perf numbers correspond to the quality suites' operating point.
        # `context` sizes it against the prompt alone — the kvpress family's
        # `compression_ratio = 1 - budget / context_length`. The two differ by a
        # factor (prefill+gen)/prefill: negligible at their gen_len=100 over a
        # 32k prompt, a different cache entirely at gen_len >> prefill.
        budget_basis = c.get("budget_basis") or getattr(
            pc, "budget_basis", "prefill_plus_gen")
        budget_horizon = 0 if budget_basis == "context" else gen_len
        # Per-config diagnostics, keyed by config name and folded into the npz
        # metadata by _save. Everything here answers "did this row measure the
        # method it says it did", which timings alone cannot. Registered NOW and
        # mutated in place below, so a row that OOMs or errors still carries the
        # geometry that explains why — which is exactly when a reader wants it,
        # and it is known before the first forward.
        diag: Dict[str, Any] = {}
        self._config_diag[c.get("name", "config")] = diag
        if cache_backend == "windowed":
            # Only meaningful where there IS a budget; a full-KV baseline has none.
            log.info("budget_basis=%s -> budget = %s x (%d + %d) tokens",
                     budget_basis, budget, prefill_len, budget_horizon)
        if cache_backend == "windowed" and cc is not None:
            # What that budget buys in KEYS, not bytes. See describe_tier_geometry.
            geom = describe_tier_geometry(
                cc.resolve(prefill_len, model.config, torch_dtype, budget_horizon),
                prefill_len,
            )
            diag["tier_geometry"] = geom
            log.info(
                "tier geometry: fp %d tok (sink %d + %d x %d + local %d) + Q %d tok "
                "(%d int2 windows) = %d effective keys = %.2fx prefill",
                geom["fp_tokens"], geom["num_sink_tokens"], geom["top_k_fp"],
                geom["window_size"], geom["local_tokens"], geom["q_tokens"],
                geom["N_q"], geom["s_eff"], geom["expansion"],
            )
            if geom["expansion"] >= 1.0:
                log.warning(
                    "config %s: the compressed cache is LONGER than the prompt — "
                    "%d effective keys for a %d-token prefill (%.2fx) at "
                    "cache_budget=%s, quant_ratio=%.2f. cache_budget is a BYTE "
                    "budget and int2 is ~8x denser, so half the bytes buys ~4x the "
                    "tokens. These bytes are real, but this row does MORE attention "
                    "work per decode step than the full-cache baseline, so read its "
                    "latency as a memory result, not a speed one — and compare it "
                    "against the quant_ratio=0.0 row at the same budget, which "
                    "retains %d keys (%.2fx).",
                    c.get("name"), geom["s_eff"], prefill_len, geom["expansion"],
                    budget, geom["quant_ratio"],
                    geom["s_eff_at_q0"], geom["expansion_at_q0"],
                )

        def _new_cache_and_hooks():
            """``(past_key_values, hooks_or_None)`` for one fresh run."""
            if cache_backend == "external":
                cache = method.new_cache()
                inst = getattr(method, "install_hooks", None)
                return cache, (inst(model, cache) if inst is not None else None)
            if cache_backend == "windowed" and cache_pkg:
                cache = _make_windowed_cache()
                return cache, install_hooks(model, cache, cc)
            return (DynamicCache() if cache_backend == "dynamic" else None), None

        def _make_windowed_cache():
            return WC(config=cc, prefill_len=prefill_len,
                      model_config=model.config, kv_dtype=torch_dtype,
                      rope_module=rope,
                      num_layers=model.config.num_hidden_layers,
                      max_tokens=budget_horizon)

        gen_kwargs = {}
        # The eager score hook reads attn_weights off the attention module's
        # output, which transformers only populates when output_attentions=True.
        # WITHOUT IT THE HOOK SILENTLY NO-OPS: scoring is disabled and eviction
        # degrades to sink+local, so the suite times a DIFFERENT METHOD than the
        # one the config names -- and it looks fast, because the scoring work that
        # the fused kernel exists to accelerate never ran.
        #
        # The quality runners gate this on the backend package for exactly this
        # reason (longbench_runner.py:434, ours_parity_runner.py:345). Suite C
        # gated it on `install_hooks_for_measurement`, which is an unrelated
        # baseline-overhead knob, so every windowed+eager perf cell measured the
        # degraded path. Gate on the package, like the runners we compare against.
        # External method (cache_backend: external). Built once per config, after
        # the budget basis is known so it is held to the SAME absolute budget as
        # our own cache. See resolve_method_factory for the handle protocol.
        method = None
        if cache_backend == "external":
            spec = c.get("method_factory")
            if not spec:
                raise ValueError(
                    f"config {c.get('name')!r}: cache_backend='external' requires "
                    f"method_factory: 'package.module:callable'")
            factory = resolve_method_factory(spec)
            budget_tokens = resolve_budget_tokens(budget, prefill_len, budget_horizon)
            method = factory(
                model=model, tokenizer=tokenizer, prefill_len=prefill_len,
                gen_len=gen_len, batch_size=batch_size,
                budget_tokens=budget_tokens, dtype=torch_dtype,
                **(c.get("method_kwargs") or {}),
            )
            if not hasattr(method, "new_cache"):
                raise TypeError(
                    f"method_factory {spec!r} returned {type(method).__name__}, "
                    f"which has no new_cache() -- see resolve_method_factory")
            if getattr(method, "requires_output_attentions", False):
                c = {**c, "_external_needs_attn": True}
            log.info("external method %s: %s (budget_tokens=%s)", spec,
                     method.describe() if hasattr(method, "describe") else
                     type(method).__name__, budget_tokens)

        eager_scoring = uses_eager_scoring(cache_backend, cache_pkg)
        if needs_output_attentions(cache_backend, cache_pkg, attn_impl, c):
            gen_kwargs["output_attentions"] = True
        if eager_scoring and attn_impl != "eager":
            log.warning(
                "config %s pairs cache_package='eager' with attn_implementation=%r. "
                "The eager score hook needs eager attention to read attn_weights; "
                "scoring will be disabled and eviction will degrade to sink+local.",
                c.get("name"), attn_impl)
        if eager_scoring:
            # Quadratic and unavoidable on this backend: output_attentions
            # materializes [B, H, L, L]. At L=8192 that is ~4 GB per layer in
            # fp16, and the standard protocol's 24k-40k contexts are out of reach
            # entirely. The flash package reconstructs scores from the LSE and has
            # no such term -- which is also why eager and flash timings must never
            # be put in the same column.
            log.info("eager scoring active for %s: output_attentions=True "
                     "(materializes [B, H, %d, %d] during prefill)",
                     c.get("name"), prefill_len, prefill_len)
        # Prefill-only kwargs. Decode steps feed one token, so their logits are
        # [B, 1, V] regardless and the knob is a no-op there.
        prefill_kwargs = dict(gen_kwargs)
        prefill_logits = c.get("prefill_logits") or getattr(pc, "prefill_logits", "full")
        if prefill_logits == "last_only":
            kw = _logits_kwarg_name(model)
            if kw is None:
                log.warning(
                    "prefill_logits='last_only' requested but %s.forward exposes "
                    "neither logits_to_keep nor num_logits_to_keep — falling back "
                    "to full [B, L, V] prefill logits. Peak memory will include a "
                    "method-independent vocab tensor.", type(model).__name__)
            else:
                prefill_kwargs[kw] = 1
                log.info("prefill_logits=last_only via %s=1", kw)
        # Warmup: prefill AND decode. See PerfConfig.warmup_decode_steps for why
        # stopping at prefill puts the Triton eviction kernel's compile time into
        # measurement run 0.
        warm_decode = self._warmup_decode_steps(pc, c, cfg, gen_len)
        log.info("warmup: %d round(s) of prefill + %d decode step(s)",
                 pc.num_warmup_runs, warm_decode)
        for _ in range(pc.num_warmup_runs):
            hooks = None
            try:
                with torch.no_grad():
                    pkv, hooks = _new_cache_and_hooks()
                    out = model(input_ids=input_ids, past_key_values=pkv,
                                use_cache=True, return_dict=True, **prefill_kwargs)
                    if warm_decode:
                        w_pkv = out.past_key_values
                        w_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        for _ in range(warm_decode):
                            out = model(input_ids=w_tok, past_key_values=w_pkv,
                                        use_cache=True, return_dict=True, **gen_kwargs)
                            w_pkv = out.past_key_values
                            w_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        del w_pkv, w_tok
                    del out
            finally:
                if hooks is not None:
                    hooks.remove()
            self._cooldown(pc, "warmup round")
        # Zeroed AFTER warmup so the counters describe the measured runs only:
        # warmup deliberately compiles and autotunes, so its dynamo activity is
        # expected and would otherwise swamp the signal we want (recompiles and
        # graph breaks that keep happening once the shapes have settled).
        _reset_evict_path_stats()
        _reset_dynamo_counters()
        # Measurement runs
        measurements = []
        for ri in range(pc.num_measurement_runs):
            hooks = None
            # One probe per measurement run. It records the peak the run reaches
            # — which is NOT the footprint at the end, and is what caps the batch
            # size (BATCHING_PLAN.md §5). Phases split prefill from decode
            # because which one holds the peak decides whether compression can
            # raise max-B at this shape at all. The poller is off: a background
            # thread sampling mem_get_info during a wall-clock benchmark is exactly
            # the kind of interference this suite must not have. The forward hook
            # is likewise off; the per-phase boundaries below are enough, and the
            # allocator's own peak counters are exact regardless of sampling.
            probe = MemoryProbe(
                label=f"{c.get('name', 'config')} bs={batch_size} "
                      f"prefill={prefill_len} gen={gen_len}",
                device=model.device,
                poll_interval_s=None,
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                probe.start()
                # Zeroed per run, immediately before the prefill it describes:
                # this counts compute_lse calls during THIS prefill.
                _reset_lse_recompute_count()
                # TTFT — synchronize BEFORE t0 so prior async work doesn't taint
                # the measurement and the sync's own latency isn't timed.
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(), probe.phase("prefill"):
                    pkv, hooks = _new_cache_and_hooks()
                    out = model(input_ids=input_ids, past_key_values=pkv,
                                use_cache=True, return_dict=True, **prefill_kwargs)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                ttft_ms = (t1 - t0) * 1000
                # L-reuse: kernel-or-error, like every other kernel contract here.
                #
                # _install_lse_source defaults to backend="auto", which falls back
                # flashinfer -> flash -> recompute WITHOUT raising, and the flash
                # capture latches off on its own if the installed build rejects
                # return_attn_probs. The cost of landing on the last rung is not
                # cosmetic: compute_token_scores then calls compute_lse, which is a
                # second O(N^2) pass per prefill layer AND materializes a
                # [B, H_kv, rep, chunk, S] fp32 transient — 17.6 GB at prefill=4096,
                # batch=32, which is exactly the cell that OOMs while batch=32 at
                # prefill=2048 (8.8 GB) survives. So a silent degrade shows up as a
                # doubled TTFT and a max-B that belongs to the fallback rather than
                # to the method, and neither is distinguishable from a real result
                # in the recorded numbers.
                #
                # Escape hatch is the documented one: STICKYKV_SCORE_LSE_FROM_FORWARD=0
                # says "measure the recompute path on purpose", and then this is
                # silent. Only the flash package reaches compute_lse — the eager
                # backend reads attn_weights off the module and never scores this way.
                lse_misses = _lse_recompute_count()
                if (cache_backend == "windowed" and cache_pkg == "flash_attn"
                        and _lse_reuse_requested() and lse_misses):
                    from utils.config import ConfigValidationError
                    raise ConfigValidationError(
                        f"config {c.get('name')!r}: L-reuse was requested (the "
                        f"default) but compute_lse ran {lse_misses} time(s) during "
                        f"this prefill — L did NOT come from the forward, so every "
                        f"prefill layer paid a second O(N^2) pass plus a "
                        f"[B, H, chunk, S] fp32 transient (~{_lse_transient_gb(batch_size, model.config, prefill_len):.1f} GB "
                        f"at this shape). The installed L-source reported "
                        f"{getattr(hooks, 'lse_label', 'unknown')!r}. Install "
                        f"flashinfer on this box (or a flash-attn build that honours "
                        f"return_attn_probs), or set "
                        f"STICKYKV_SCORE_LSE_FROM_FORWARD=0 to measure the recompute "
                        f"path deliberately. Refusing to report a TTFT and a max-B "
                        f"that belong to the fallback."
                    )
                if ri == 0 and cache_backend == "windowed":
                    diag["lse"] = {
                        "label": getattr(hooks, "lse_label", None),
                        "active": bool(getattr(hooks, "lse_active", False)),
                        "recompute_count": lse_misses,
                        "reuse_requested": _lse_reuse_requested(),
                    }
                    diag["fused_decode"] = bool(getattr(hooks, "fused_decode", False))
                # Generate tokens for TPOT
                pkv = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = time.perf_counter()
                # Decode step 0 is timed on its own because it is not a steady
                # state step: with first_eviction_step=0 it is where the whole
                # prompt is scored and compacted to budget. Averaging it into
                # TPOT spreads a one-off O(prefill) cost over gen_len steps, so
                # the same method reports a different "per-token latency" purely
                # as a function of how many tokens the config asked for — and
                # against the kvpress family, whose press runs inside prefill, it
                # lands in the wrong column entirely.
                n_decode = max(gen_len - 1, 0)
                t_step0 = t2
                # Two phases, not one. Step 0 still holds the WHOLE prompt -- it
                # is the step that compacts it -- so a single "decode" phase
                # reports the uncompressed peak and hides the very footprint this
                # method exists to shrink. `decode_steady` is the compressed cache.
                with torch.no_grad():
                    if n_decode >= 1:
                        with probe.phase("decode_step0"):
                            out = model(input_ids=next_tok, past_key_values=pkv,
                                        use_cache=True, return_dict=True, **gen_kwargs)
                            pkv = out.past_key_values
                            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        t_step0 = time.perf_counter()
                    if n_decode >= 2:
                        with probe.phase("decode_steady"):
                            for _ in range(n_decode - 1):
                                out = model(input_ids=next_tok, past_key_values=pkv,
                                            use_cache=True, return_dict=True, **gen_kwargs)
                                pkv = out.past_key_values
                                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t3 = time.perf_counter()
                gen_time = t3 - t2
                tpot_ms = (gen_time / max(gen_len - 1, 1)) * 1000
                # Step 0 alone, and the steady state after it. Steady-state TPOT
                # needs >= 2 decode steps to exist; with fewer it is nan, never 0.
                decode_step0_ms = (t_step0 - t2) * 1000 if n_decode >= 1 else float("nan")
                tpot_steady_ms = (
                    ((t3 - t_step0) / (n_decode - 1)) * 1000
                    if n_decode >= 2 else float("nan")
                )
                # The column that lines up with the kvpress family's
                # `prefill_latency`: prompt forward + the compaction of that
                # prompt, wherever the implementation happens to run it.
                prefill_plus_compress_ms = ttft_ms + (
                    decode_step0_ms if n_decode >= 1 else 0.0)
                e2e_latency_ms = (t3 - t0) * 1000
                # End-to-end throughput includes prefill (TTFT) + decode time; this
                # mirrors the legacy field name but is NOT decode-only. Counts all
                # batch_size rows (B=1 ⇒ identical to the legacy value).
                throughput_tokps = (batch_size * gen_len) / max(gen_time + (t1-t0), 1e-9)
                peak = probe.stop().report()
                phase_peak = {p.name: p for p in peak.phases}
                measurements.append({
                    "ttft_ms": ttft_ms, "throughput_tokps": throughput_tokps,
                    "tpot_ms": tpot_ms, "e2e_latency_ms": e2e_latency_ms,
                    "decode_step0_ms": decode_step0_ms,
                    "tpot_steady_ms": tpot_steady_ms,
                    "prefill_plus_compress_ms": prefill_plus_compress_ms,
                    # Unchanged field, unchanged meaning: torch allocated peak.
                    "peak_memory_mb": peak.torch_alloc_peak / (1024 * 1024),
                    "peak_reserved_mb": peak.torch_reserved_peak / (1024 * 1024),
                    "peak_device_used_mb": peak.device_used_peak / (1024 * 1024),
                    "device_total_mb": peak.device_total / (1024 * 1024),
                    "peak_prefill_mb": _phase_mb(phase_peak.get("prefill")),
                    # Unchanged meaning for old readers -- the worst of the two
                    # decode phases is what a single "decode" phase reported.
                    "peak_decode_mb": _nanmax2(
                        _phase_mb(phase_peak.get("decode_step0")),
                        _phase_mb(phase_peak.get("decode_steady")),
                    ),
                    "peak_decode_step0_mb": _phase_mb(phase_peak.get("decode_step0")),
                    "peak_decode_steady_mb": _phase_mb(phase_peak.get("decode_steady")),
                    "peak_host_rss_mb": peak.host_rss_peak / (1024 * 1024),
                    "alloc_retries": float(peak.num_alloc_retries),
                })
                if ri == 0:
                    log.info("\n%s", format_peak_report(peak))
            finally:
                probe.stop()
                if hooks is not None:
                    hooks.remove()
            self._cooldown(pc, f"measurement run {ri}")
        # -- did the compiled eviction actually fuse? ------------------------
        # Two separate questions, and the flag answers neither. (1) Did the
        # compiled body RUN — evict_path_stats splits evictions by path, and any
        # `eager` count on a run that set STICKYKV_COMPILE_EVICT means the
        # dispatch we were trying to remove is still being paid. (2) Did it fuse
        # or just trace — torch.compile responds to a graph break by splitting
        # the region and running the pieces eagerly, silently, so a "compiled"
        # eviction that broke on every store mutation issues the same ~273
        # launches per layer and is indistinguishable in the timings from "the
        # eviction was never the bottleneck". The eviction body mutates
        # QuantizedStore in place and calls free_slots, so breaks are the
        # expected failure mode, not a remote one.
        if cache_backend == "windowed":
            ev = _evict_path_stats()
            dyn = _dynamo_counters()
            diag["eviction"] = {"path_stats": ev, "dynamo_counters": dyn}
            log.info("eviction path: %s | dynamo: %s", ev or "n/a", dyn or "n/a")
            if os.environ.get("STICKYKV_COMPILE_EVICT", "0").strip().lower() in (
                    "1", "true", "yes", "on") and ev.get("eager"):
                log.warning(
                    "config %s: STICKYKV_COMPILE_EVICT is on but %d eviction(s) "
                    "still ran the EAGER body (compiled: %d). Those steps paid the "
                    "full ~273-launch-per-layer eviction this flag exists to "
                    "remove.", c.get("name"), ev["eager"], ev.get("compiled", 0))
            if dyn.get("graph_breaks"):
                log.warning(
                    "config %s: %d dynamo graph break(s) during the MEASURED runs "
                    "(warmup is excluded). torch.compile runs the broken-out "
                    "regions eagerly, so the compiled eviction may be issuing the "
                    "eager launch count under a compiled name. Inspect with "
                    "TORCH_LOGS=graph_breaks before trusting this row's TPOT.",
                    c.get("name"), dyn["graph_breaks"])
            if dyn.get("frames_ok"):
                log.warning(
                    "config %s: dynamo compiled %d frame(s) during the MEASURED "
                    "runs (counters are zeroed after warmup, so this is "
                    "recompilation). The eviction is compiled dynamic=True "
                    "precisely so the varying window count does not retrigger "
                    "compilation; this is compile latency inside the timings.",
                    c.get("name"), dyn["frames_ok"])
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return measurements

    def _save(self, result: dict, prefill_len: int, gen_len: int, batch_size: int, cfg, env: dict, clocks_locked: bool) -> Path:
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        npz_path = od / f"perf_prefill{prefill_len}_gen{gen_len}_bs{batch_size}.npz"
        pc = cfg.perf
        meta = {
            "prefill_len": prefill_len,
            "gen_len": gen_len,
            "batch_size": batch_size,
            # Protocol provenance. Without these an npz cannot say whether its
            # numbers are comparable to a published table or only to itself.
            # The model and dtype belong here for the same reason: an efficiency
            # number is only comparable against the SAME model, and
            # capture_environment() records the machine, not the run.
            "model_name": cfg.model.name,
            "model_revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "protocol": getattr(pc, "protocol", "native"),
            "prompt_mode": getattr(pc, "prompt_mode", "auto"),
            "repeat_sentence": getattr(pc, "repeat_sentence", None),
            # WHICH corpus, when prompt_mode is dataset. Without it a run says it
            # used real text but not what text, which is not reproducible -- and
            # for a path it is the only record of where the tokens came from.
            "data_source": getattr(pc, "data_source", None),
            "budget_basis": getattr(pc, "budget_basis", "prefill_plus_gen"),
            "prefill_logits": getattr(pc, "prefill_logits", "full"),
            "num_warmup_runs": getattr(pc, "num_warmup_runs", None),
            "num_measurement_runs": getattr(pc, "num_measurement_runs", None),
            "warmup_decode_steps": getattr(pc, "warmup_decode_steps", None),
            "cooldown_s": getattr(pc, "cooldown_s", 0.0),
            # Which scoring path actually ran. A silent Triton fallback (kernel
            # not installed on the GPU box) looks exactly like "the fused kernel
            # bought nothing", so an efficiency number is unattributable without
            # this. See modules/windowed_cache/score_kernel.py.
            "score_backend": _describe_score_backend(),
            # Per-config evidence about WHICH method each row measured, keyed by
            # config name: `tier_geometry` (how many keys the byte budget buys —
            # see describe_tier_geometry), `lse` (which L-source installed and
            # whether compute_lse ran anyway), `fused_decode`, and `eviction`
            # (eager-vs-compiled eviction counts plus dynamo graph-break /
            # recompile counters). A timing column cannot say any of this, and
            # every one of them has silently changed what a row measured before.
            "diagnostics": result.get("diagnostics", {}),
            # Where each row compacts, so the printer can charge compression to
            # the right column per method instead of assuming this project's shape.
            "compaction": {
                c.get("name", f"config_{i}"): resolve_compaction_phase(c)
                for i, c in enumerate(getattr(pc, "configs", []))
            },
            # Each row's method identity, so a baseline table says what it ran.
            "methods": {
                c.get("name", f"config_{i}"): (
                    c.get("method_factory") if c.get("cache_backend") == "external"
                    else f"{c.get('cache_backend', 'dynamic')}"
                        + (f"/{c['cache_package']}" if c.get("cache_package") else "")
                )
                for i, c in enumerate(getattr(pc, "configs", []))
            },
            # The full per-config list, so the npz says what operating point each
            # row was: cache_budget, quant_ratio, quant_memoize_read,
            # first_eviction_step are all per-config overrides.
            "perf_configs": getattr(pc, "configs", []),
            # Config-level window geometry. Per-config overrides (if any) are in
            # `perf_configs`, which is the authoritative per-row record.
            "window": {
                "window_size": getattr(cfg.window, "window_size", None),
                "num_sink_tokens": getattr(cfg.window, "num_sink_tokens", None),
                "local_window_size": getattr(cfg.window, "local_window_size", None),
            },
            "cache_defaults": {
                "quant_ratio": getattr(cfg.cache, "quant_ratio", None),
                "quant_memoize_read": getattr(cfg.cache, "quant_memoize_read", None),
                "first_eviction_step": getattr(cfg.cache, "first_eviction_step", None),
                "rerotate_on_evict": getattr(cfg.cache, "rerotate_on_evict", None),
            },
            **env,
            "clocks_locked": clocks_locked,
        }
        np.savez_compressed(
            str(npz_path),
            config_names=np.array(result["names"], dtype=object),
            attn_implementations=np.array(result["attn_impls"], dtype=object),
            ttft_ms=result["ttft"],
            throughput_tokps=result["throughput"],
            tpot_ms=result["tpot"],
            e2e_latency_ms=result["e2e_latency"],
            # Compression accounting. tpot_ms keeps its original meaning (mean
            # over ALL decode steps) so existing readers are unaffected.
            decode_step0_ms=result["decode_step0"],
            tpot_steady_ms=result["tpot_steady"],
            prefill_plus_compress_ms=result["prefill_plus_compress"],
            peak_memory_mb=result["peak_mem"],
            skipped_mask=result["skipped"],
            # Why a config was skipped. `skipped_mask` is the union of all three
            # and stays first-class for old readers; max-B must be read off
            # `oom_mask` alone (see _run_prefill).
            oom_mask=result["oom"],
            error_mask=result["errored"],
            skip_reason=np.array(result["skip_reason"], dtype=object),
            # Peak detail per (config, run), all MB.
            peak_reserved_mb=result["peak_reserved"],
            peak_device_used_mb=result["peak_device"],
            device_total_mb=result["device_total"],
            peak_prefill_mb=result["peak_prefill"],
            peak_decode_mb=result["peak_decode"],
            peak_decode_step0_mb=result["peak_decode_step0"],
            peak_decode_steady_mb=result["peak_decode_steady"],
            peak_host_rss_mb=result["peak_host_rss"],
            alloc_retries=result["alloc_retries"],
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        log.info("Saved perf: %s", npz_path)
        return npz_path

    def _build_input_ids(self, data_source: str, tokenizer, prefill_len: int,
                         batch_size: int, seed: int) -> torch.Tensor:
        """``[batch_size, prefill_len]`` of real text from *data_source*.

        Accepts any corpus the project can already load, not just LongBench:

            2wikimqa, qasper, ...      a LongBench dataset (21 names)
            wikitext-103, pg19         a hub corpus, via data.corpus_loader
            ./data/mine.jsonl          a local file or directory of .txt/.jsonl
            longbench:NAME             explicit, when a name is ambiguous
            corpus:NAME_OR_PATH        explicit

        Rows are equal-length by construction (fixed-size chunks, no padding),
        which is what the cache supports and what keeps a batched timing honest.
        """
        import random
        texts = iter_corpus_texts(data_source)
        log.info("Loading '%s' for perf inputs (prefill_len=%d, batch_size=%d)",
                 data_source, prefill_len, batch_size)
        chunks = chunk_texts_to_length(
            texts, tokenizer, prefill_len,
            # Sample from a pool rather than taking the first `batch_size`
            # chunks, but cap the pool: wikitext-103 is ~100M tokens and
            # tokenizing all of it to benchmark a 4-row batch would cost far more
            # than the benchmark. The pool is the first `pool` chunks in corpus
            # order, which is deterministic and enough to de-correlate rows.
            pool=max(batch_size * 4, 16),
        )
        if len(chunks) < batch_size:
            raise ValueError(
                f"Corpus '{data_source}' yields only {len(chunks)} chunk(s) of "
                f"{prefill_len} tokens but batch_size={batch_size}. "
                f"Reduce batch_size or prefill_len, or use a larger corpus."
            )
        rng = random.Random(seed)
        selected = rng.sample(chunks, batch_size)
        return torch.stack(selected, dim=0)  # [batch_size, prefill_len]

    def _try_lock_clocks(self) -> bool:
        import subprocess
        try:
            r = subprocess.run(["nvidia-smi", "-lgc", "1400,1400"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                log.info("GPU clocks locked"); return True
            log.warning("nvidia-smi -lgc failed: %s", r.stderr)
        except Exception as e:
            log.warning("Clock locking failed: %s", e)
        return False
