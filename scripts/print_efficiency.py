"""Print Suite C results in the kvpress-family efficiency format.

The prompt-compression papers (SnapKV, AdaKV, CriticalKV, DefensiveKV) all report
efficiency through kvpress' ``evaluation/efficiency_evaluate.py``, whose stdout is
one line per (method, context_length):

    budget B context_length C prefill_latency(s) P decoding_latency(s) D
        max_memory_allocated(GB) A max_memory_reserved(GB) R

This script emits that line from our npz files so a column can be placed next to
a published table, and then prints what that line hides.

THE ACCOUNTING, which is the whole reason this is a separate printer.
kvpress' press compresses inside the prefill forward, so its ``prefill_latency``
covers prompt forward + compaction and its ``decoding_latency`` is a clean steady
state. This design cannot compress during prefill -- the score hook is a
forward_hook, so the prefill scores do not exist until the forward has returned --
so it compacts on decode step 0 instead. The same physical work therefore lands
in a different column. Mapping our fields onto theirs naively would report a
TTFT that excludes compression (flattering) and a TPOT that includes it amortized
over gen_len (penalising, and gen_len-dependent). So:

    their prefill_latency   <- our prefill_plus_compress_ms   (ttft + step 0)
    their decoding_latency  <- our tpot_steady_ms             (steps 1..n)

Both are printed alongside the raw ttft_ms/tpot_ms so the split stays visible.

Usage:
    python scripts/print_efficiency.py --npz-dir outputs/perf_kvpress
    python scripts/print_efficiency.py --npz-dir outputs/perf_kvpress --out eff.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MB_PER_GB = 1024.0


def _key(path: Path) -> tuple[int, int, int]:
    """``(prefill, gen, batch)`` parsed from the filename, for numeric ordering."""
    stem = path.stem
    try:
        return (int(stem.split("prefill")[1].split("_")[0]),
                int(stem.split("_gen")[1].split("_")[0]),
                int(stem.split("_bs")[1]))
    except (IndexError, ValueError):
        return (0, 0, 0)


def _col(data: dict, name: str, n_configs: int, n_runs: int) -> np.ndarray:
    """A (config, run) array, or all-NaN if this npz predates the field.

    Old npz files stay readable; the columns they cannot fill print as "-"
    rather than as a plausible-looking zero.
    """
    if name not in data:
        return np.full((n_configs, n_runs), np.nan)
    return np.asarray(data[name], dtype=float)


def _stat(row: np.ndarray, fn) -> float:
    row = row[~np.isnan(row)]
    return float(fn(row)) if row.size else float("nan")


def _fmt(v: float, prec: int) -> str:
    return "-" if np.isnan(v) else f"{v:.{prec}f}"


def build_report(files: list[Path]) -> str:
    lines: list[str] = []
    emit = lines.append

    emit("=" * 100)
    emit("Efficiency report (comparable protocol)")
    emit("=" * 100)

    warned_protocol = False
    for path in files:
        data = dict(np.load(str(path), allow_pickle=True))
        meta = json.loads(str(data["metadata_json"][0]))
        names = [str(n) for n in data["config_names"]]
        n_cfg = len(names)
        n_run = int(np.asarray(data["ttft_ms"]).shape[1])

        prefill_len = int(meta.get("prefill_len", 0))
        gen_len = int(meta.get("gen_len", 0))
        protocol = meta.get("protocol", "native")
        if protocol != "standard" and not warned_protocol:
            emit("")
            emit(f"[warn] {path.name} was recorded under protocol={protocol!r}, not")
            emit("       'standard'. Its prompt, budget basis and prefill-logits handling")
            emit("       may differ from the published harness -- see")
            emit("       configs/eval_efficiency.yaml. These numbers are NOT comparable")
            emit("       to a published table until re-run under that config.")
            warned_protocol = True

        ttft = _col(data, "ttft_ms", n_cfg, n_run)
        tpot = _col(data, "tpot_ms", n_cfg, n_run)
        step0 = _col(data, "decode_step0_ms", n_cfg, n_run)
        tpot_ss = _col(data, "tpot_steady_ms", n_cfg, n_run)
        pplusc = _col(data, "prefill_plus_compress_ms", n_cfg, n_run)
        alloc = _col(data, "peak_memory_mb", n_cfg, n_run)
        reserved = _col(data, "peak_reserved_mb", n_cfg, n_run)
        steady = _col(data, "peak_decode_steady_mb", n_cfg, n_run)
        p_step0 = _col(data, "peak_decode_step0_mb", n_cfg, n_run)

        # Where each row compacts. Only a decode_step0 method adds step 0 to its
        # prefill column; a press-style row already paid it inside TTFT, and a
        # full-cache row never paid it at all. Assuming one shape for every row
        # overstates the others' prefill by a whole decode step.
        compaction = meta.get("compaction", {})
        prefill_col = np.where(
            np.array([compaction.get(n, "decode_step0") == "decode_step0"
                      for n in names])[:, None],
            pplusc, ttft)

        oom = np.asarray(data["oom_mask"]) if "oom_mask" in data else np.zeros(n_cfg, bool)
        err = np.asarray(data["error_mask"]) if "error_mask" in data else np.zeros(n_cfg, bool)

        emit("")
        emit(f"--- {path.name} | context_length={prefill_len} "
             f"decode_steps={max(gen_len - 1, 0)} "
             f"batch={meta.get('batch_size')} "
             f"prompt={meta.get('prompt_mode')} "
             f"budget_basis={meta.get('budget_basis')} "
             f"prefill_logits={meta.get('prefill_logits')} ---")
        emit(f"    model={meta.get('model_name', '?')} dtype={meta.get('dtype', '?')} "
             f"gpu={meta.get('gpu_name', '?')} "
             f"rounds={meta.get('num_measurement_runs', n_run)} "
             f"(warmup {meta.get('num_warmup_runs', '?')})")
        emit("")

        # -- the published line, using THEIR statistic (mean over the rounds) --
        emit("  [published-table format -- mean over rounds, directly comparable]")
        for ci, name in enumerate(names):
            if err[ci]:
                emit(f"    {name:<22} ERROR (not an OOM -- this row is not evidence)")
                continue
            if oom[ci]:
                emit(f"    {name:<22} OOM")
                continue
            emit(
                f"    {name:<22} "
                f"prefill_latency(s) {_fmt(_stat(prefill_col[ci], np.mean) / 1000, 3):>8}  "
                f"decoding_latency(s) {_fmt(_stat(tpot_ss[ci], np.mean) / 1000, 5):>9}  "
                f"max_memory_allocated(GB) {_fmt(_stat(alloc[ci], np.mean) / MB_PER_GB, 2):>7}  "
                f"max_memory_reserved(GB) {_fmt(_stat(reserved[ci], np.mean) / MB_PER_GB, 2):>7}"
            )

        # -- what that line folds together --
        emit("")
        emit("  [this suite's split -- median over rounds]")
        emit(f"    {'config':<22} {'compaction':>13} {'ttft_ms':>10} {'step0_ms':>10} {'=prefill+c':>11} "
             f"{'tpot_ms':>10} {'tpot_steady':>12} {'peak_GB':>9} {'steadyKV_GB':>12}")
        for ci, name in enumerate(names):
            if oom[ci] or err[ci]:
                continue
            emit(
                f"    {name:<22} "
                f"{compaction.get(name, 'decode_step0'):>13} "
                f"{_fmt(_stat(ttft[ci], np.median), 1):>10} "
                f"{_fmt(_stat(step0[ci], np.median), 1):>10} "
                f"{_fmt(_stat(pplusc[ci], np.median), 1):>11} "
                f"{_fmt(_stat(tpot[ci], np.median), 3):>10} "
                f"{_fmt(_stat(tpot_ss[ci], np.median), 3):>12} "
                f"{_fmt(_stat(alloc[ci], np.median) / MB_PER_GB, 2):>9} "
                f"{_fmt(_stat(steady[ci], np.median) / MB_PER_GB, 2):>12}"
            )

        # How much of tpot_ms is the one-off compaction -- i.e. how much a naive
        # tpot-vs-decoding_latency comparison would have overcharged this method.
        emit("")
        for ci, name in enumerate(names):
            if oom[ci] or err[ci]:
                continue
            if compaction.get(name, "decode_step0") != "decode_step0":
                continue        # nothing one-off landed in step 0 for this row
            t_all = _stat(tpot[ci], np.median)
            t_ss = _stat(tpot_ss[ci], np.median)
            if np.isnan(t_all) or np.isnan(t_ss) or t_ss <= 0:
                continue
            infl = (t_all / t_ss - 1.0) * 100
            if infl >= 1.0:
                emit(f"    note: {name} raw tpot_ms is {infl:.0f}% above steady state "
                     f"-- that is the step-0 compaction amortized over "
                     f"{max(gen_len - 1, 1)} steps, not per-token cost.")

        # peak vs steady: the whole-run peak cannot show compression at these
        # shapes, so say so rather than letting a flat peak column read as
        # "compression does nothing".
        emit("")
        emit("    peak_GB is the whole-run peak and is dominated by the UNCOMPRESSED")
        emit("    prompt: this design cannot compact during prefill, so peak =")
        emit(f"    max(prefill_len + 1, budget) and at prefill={prefill_len} with "
             f"{max(gen_len - 1, 0)} generated")
        emit("    tokens the prompt always wins. steadyKV_GB is the phase AFTER step 0")
        emit("    compacted -- that is the compressed cache, and the memory claim.")

    emit("")
    emit("=" * 100)
    emit("At batch size 1 KV compression cannot produce a decode speedup: decode")
    emit("reads every weight once per step and that floor is independent of cache")
    emit("size. Expect a tie on decoding_latency here. The throughput claim is")
    emit("tokens/s at max-B (configs/eval_perf_batched.yaml), not B=1 latency.")
    emit("")
    emit("Comparing against a published table: match the MODEL and the attention")
    emit("backend first -- a flash_attention_2 number and an eager number are not")
    emit("the same measurement. And note prefill_logits='last_only' excludes the")
    emit("method-independent [B, L, V] prefill logits tensor (~10 GB fp16 at 40k")
    emit("for a 128k vocab) that the upstream harness leaves in its GB columns.")
    emit("=" * 100)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", default="outputs/perf_kvpress")
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    files = sorted(npz_dir.glob("perf_prefill*_gen*_bs*.npz"), key=_key)
    if not files:
        print(f"[error] no perf npz files in {npz_dir}", file=sys.stderr)
        return 1

    report = build_report(files)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\n[saved] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
