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

        # -- what the budget bought, in KEYS rather than bytes ---------------
        # cache_budget is a BYTE budget. int2 is ~8x denser than fp16, so a
        # quant_ratio=0.5 row and a quant_ratio=0.0 row at the same nominal
        # budget retain wildly different numbers of tokens -- and it is the token
        # count, not the byte count, that sets decode latency. Without this the
        # two rows look like the same operating point with different numbers, and
        # the obvious reading ("quantization made it 2x slower") attributes to the
        # int2 path what is really 2.4x more attention work.
        diags = meta.get("diagnostics", {}) or {}
        geoms = [(n, (diags.get(n) or {}).get("tier_geometry")) for n in names]
        if any(g for _, g in geoms):
            emit("")
            emit("  [what the budget bought -- KEYS one decode step attends over]")
            emit(f"    {'config':<22} {'mode':>7} {'q':>5} {'fp_tok':>8} {'int2_tok':>9} "
                 f"{'S_eff':>8} {'x prefill':>10} {'bytes/fp16':>11} {'Qloop':>7}")
            for name, g in geoms:
                if not g:
                    continue
                bvf = g.get("bytes_vs_fp16", float("nan"))
                emit(
                    f"    {name:<22} {g.get('quant_budget_mode', '?'):>7} "
                    f"{g['quant_ratio']:>5.2f} {g['fp_tokens']:>8d} "
                    f"{g['q_tokens']:>9d} {g['s_eff']:>8d} {g['expansion']:>9.2f}x "
                    f"{_fmt(bvf * 100, 1):>10}% {g['decode_q_loop_iters']:>7d}"
                )

            # An eviction that drops nothing is not an eviction. tier_counts
            # clamps n_q to the windows that exist, so past the saturation point
            # the pass only moves windows between tiers and the row is measuring
            # a full cache under a compression name.
            nodrop = [(n, g) for n, g in geoms
                      if g and g.get("first_eviction_windows_dropped") == 0]
            for name, g in nodrop:
                emit("")
                emit(f"    [warn] {name}: the FIRST eviction drops 0 of the "
                     f"{g.get('first_eviction_windows_offered')} windows it is offered.")
                emit(f"           tier_counts clamped n_q to {g.get('first_eviction_n_q')} "
                     f"against a resolved N_q of {g['N_q']}, so this row retains")
                emit("           everything and only moves windows between tiers.")

            # s_eff is the asymptote; a short generation never gets there, so two
            # cells of the same config at different gen_len are different points.
            midfill = [(n, g) for n, g in geoms
                       if g and g.get("steps_to_steady_state", 0) > max(gen_len - 1, 0)]
            for name, g in midfill:
                emit("")
                emit(f"    [warn] {name}: the Q tier needs ~{g['steps_to_steady_state']} "
                     f"decode steps to reach its {g['s_eff']}-key steady state;")
                emit(f"           this cell runs {max(gen_len - 1, 0)}. Measured mid-fill -- "
                     f"not comparable to a longer-generation")
                emit("           cell of the same config.")

            expanded = [(n, g) for n, g in geoms
                        if g and g["expansion"] >= 1.0 and not g.get("q_invariant")]
            if expanded:
                emit("")
                for name, g in expanded:
                    emit(
                        f"    [warn] {name}: the compressed cache is LONGER than the "
                        f"prompt -- {g['s_eff']} keys for a {prefill_len}-token "
                        f"prefill ({g['expansion']:.2f}x)."
                    )
                ref = expanded[0][1]
                emit("")
                emit("           Under quant_budget_mode='bytes', quant_ratio divides the")
                emit("           BYTES. An int2 window costs ~3.9x less than an fp16 one, so")
                emit("           the retained KEY COUNT grows with q and the operating point")
                emit("           moves under a knob nominally chosen for quality.")
                emit(f"           quant_budget_mode='tokens' (the default) holds it at "
                     f"{ref['s_eff_at_q0']} keys ({ref['expansion_at_q0']:.2f}x) for every q,")
                emit("           and spends the saved bytes instead.")

        # -- which path each row actually ran --------------------------------
        # Every entry here has silently changed what a row measured at least
        # once: the L-source degrades flashinfer -> flash -> recompute without
        # raising, and torch.compile answers a graph break by running the region
        # eagerly under a compiled name. Timings cannot distinguish any of that
        # from "the optimization did not help".
        if any(diags.get(n) for n in names):
            emit("")
            emit("  [provenance -- which path each row actually ran]")
            for name in names:
                d = diags.get(name) or {}
                if not d:
                    continue
                lse = d.get("lse") or {}
                evd = d.get("eviction") or {}
                ev = evd.get("path_stats") or {}
                dyn = evd.get("dynamo_counters") or {}
                bits = []
                if lse:
                    misses = lse.get("recompute_count")
                    bits.append(
                        f"L={lse.get('label')}"
                        + ("" if not misses else f" (RECOMPUTED x{misses})")
                    )
                if "fused_decode" in d:
                    bits.append(f"fused_decode={'on' if d['fused_decode'] else 'off'}")
                if ev:
                    bits.append(
                        f"evict compiled/eager={ev.get('compiled', 0)}/{ev.get('eager', 0)}"
                    )
                if evd.get("compile_failed"):
                    bits.append(f"COMPILE_FAILED[{evd['compile_failed']}]")
                if dyn.get("graph_breaks"):
                    bits.append(f"GRAPH_BREAKS={dyn['graph_breaks']}")
                if dyn.get("frames_ok"):
                    bits.append(f"RECOMPILED_FRAMES={dyn['frames_ok']}")
                if bits:
                    emit(f"    {name:<22} " + "  ".join(bits))
            emit("")
            emit("    COMPILE_FAILED means torch.compile could not lower the eviction on")
            emit("    this build; every eviction fell back to eager, so TPOT is eager-path.")
            emit("    evict compiled/eager: any nonzero `eager` on a run that set")
            emit("    STICKYKV_COMPILE_EVICT means those eviction steps still paid the")
            emit("    full ~273-launch-per-layer dispatch cost the flag exists to remove.")
            emit("    GRAPH_BREAKS means torch.compile split the eviction and ran the")
            emit("    pieces eagerly -- compiled in name, eager in launch count.")
            emit("    RECOMPILED_FRAMES counts frames dynamo compiled AFTER warmup, so")
            emit("    that compile latency landed inside the measured timings.")

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
