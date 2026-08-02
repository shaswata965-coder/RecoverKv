#!/usr/bin/env python
"""TODO: one-line module summary."""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGES = ("corpus", "base", "ours", "study", "zip")

BASE_WINDOW_SIZES: Sequence[int] = (1, 8, 32)
OURS_CONDITIONS: Sequence[tuple] = (
    ("R1", 1, 0.0),
    ("R2", 8, 0.0),
    ("R3", 8, 0.7),
    ("R4", 32, 0.7),
)
R0_WINDOW_SIZE = 1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_transformers_version() -> None:
    import transformers

    from utils.cache_factory import is_transformers_version_supported

    v = transformers.__version__
    if not is_transformers_version_supported(v):
        raise SystemExit(
            f"transformers {v} is newer than the supported 4.47.1 — the "
            "windowed cache relies on cache_position advancing monotonically, "
            "which newer transformers versions break. pip install "
            '"transformers==4.47.1" and retry.'
        )
    log(f"transformers {v} OK (<= 4.47.1)")


def build_corpus(corpus_file: Path, model_dir: Path, out_jsonl: Path,
                 prefill: int, samples: int) -> Path:
    if out_jsonl.exists():
        log(f"corpus already built: {out_jsonl} (delete it to rebuild)")
        return out_jsonl

    from transformers import AutoTokenizer

    log(f"reading {corpus_file} ...")
    text = corpus_file.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"\n(?= = [^=])", text)
    articles = [p.strip() for p in parts if p.strip()]
    log(f"split into {len(articles)} articles")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    min_len = prefill + 8
    long_enough = [
        a for a in articles
        if len(tok(a, add_special_tokens=True).input_ids) >= min_len
    ]
    log(f"{len(long_enough)}/{len(articles)} articles have >= {min_len} tokens")
    if len(long_enough) < samples:
        raise SystemExit(
            f"need >= {samples} articles of >= {min_len} tokens, found "
            f"{len(long_enough)}. Lower --prefill/--samples or point at a "
            "larger corpus file (wiki.train.tokens, not wiki.valid/test)."
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for a in long_enough[:max(samples, 50)]:
            f.write(json.dumps({"text": a}) + "\n")
    log(f"corpus -> {out_jsonl}")
    return out_jsonl


def print_geometry(model_dir: Path, prefill: int, gen: int, budget: float,
                   sink: int, local: int) -> None:
    import torch
    from transformers import AutoConfig

    from modules.windowed_cache.config import WindowedCacheConfig

    mc = AutoConfig.from_pretrained(str(model_dir))
    log(f"{mc.num_hidden_layers} layers | {mc.num_attention_heads} q-heads | "
        f"{getattr(mc, 'num_key_value_heads', mc.num_attention_heads)} kv-heads")
    header = f"{'run':<4}{'ws':>4}{'q':>5}{'top_k':>7}{'fp':>5}{'N_q':>5}{'alive_win':>11}{'alive_tok':>11}"
    print(header)
    for rid, ws, q in OURS_CONDITIONS:
        r = WindowedCacheConfig(
            window_size=ws, num_sink_tokens=sink, local_window_size=local,
            cache_budget=budget, quant_ratio=q,
        ).resolve(prefill, mc, torch.float16, gen)
        alive = r.top_k_fp + r.N_q
        print(f"{rid:<4}{ws:>4}{q:>5}{r.top_k_windows:>7}{r.top_k_fp:>5}"
              f"{r.N_q:>5}{alive:>11}{alive * ws:>11}")
    print()


def run_main(config: str, overrides: Dict[str, object], tag: str) -> None:
    cmd = [sys.executable, "main.py", "--config", config, "--override",
          *[f"{k}={v}" for k, v in overrides.items()]]
    log(f"=== {tag} ===")
    log("  " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode:
        raise SystemExit(f"FAILED ({proc.returncode}): {tag}")
    log(f"{tag} done in {time.time() - t0:.0f}s")


def geom_overrides(model_dir: Path, corpus: Path, ws: int, *, prefill: int,
                   gen: int, samples: int, budget: float, sink: int,
                   local: int, output_dir: Path, **extra) -> Dict[str, object]:
    d: Dict[str, object] = {
        "model.name": str(model_dir), "model.revision": "none",
        "model.dtype": "float16",
        "parity.dataset": str(corpus), "parity.article_index": 0,
        "parity.prefill_len": prefill, "parity.gen_len": gen,
        "data.prefill_len": prefill, "data.gen_len": gen,
        "data.num_samples": samples, "data.batch_size": 1,
        "cache.cache_budget": budget, "cache.first_eviction_step": 0,
        "window.num_sink_tokens": sink, "cache.num_sink_tokens": sink,
        "window.window_size": ws, "cache.window_size": ws,
        "window.local_window_size": local, "cache.local_window_size": local,
        "telemetry.output_dir": str(output_dir),
    }
    d.update(extra)
    return d


def run_base_stage(model_dir: Path, corpus: Path, work_dir: Path,
                   **geom_kwargs) -> Dict[int, Path]:
    base_paths = {ws: work_dir / f"parity_base_ws{ws}.npz"
                 for ws in BASE_WINDOW_SIZES}
    for ws, path in base_paths.items():
        if path.exists():
            log(f"BASE ws={ws} already exists: {path} (skipping)")
            continue
        run_main(
            "configs/eval_parity_base.yaml",
            geom_overrides(model_dir, corpus, ws, output_dir=work_dir,
                          output_path=str(path), **geom_kwargs),
            f"BASE ws={ws}" + ("  <- this is R0" if ws == R0_WINDOW_SIZE else
                              "  (teacher-forcing source)"),
        )
    return base_paths


def run_ours_stage(model_dir: Path, corpus: Path, work_dir: Path,
                   base_paths: Dict[int, Path], quant_promotion: bool = True,
                   **geom_kwargs) -> Dict[str, Path]:
    ours_paths = {rid: work_dir / f"parity_ours_{rid}.npz"
                 for rid, _, _ in OURS_CONDITIONS}
    for rid, ws, q in OURS_CONDITIONS:
        path = ours_paths[rid]
        if path.exists():
            log(f"OURS {rid} already exists: {path} (skipping)")
            continue
        run_main(
            "configs/eval_parity_ours_eager.yaml",
            geom_overrides(
                model_dir, corpus, ws, output_dir=work_dir,
                **{"cache.quant_ratio": q, "cache.backend_package": "eager",
                   "cache.quant_promotion": quant_promotion,
                   "base_run_npz": str(base_paths[ws]),
                   "output_path": str(path)},
                **geom_kwargs,
            ),
            f"OURS {rid}: ws={ws} quant_ratio={q} "
            f"promotion={'on' if quant_promotion else 'STICKY-OFF'}",
        )
    return ours_paths


def run_study_stage(base_paths: Dict[int, Path], ours_paths: Dict[str, Path],
                    work_dir: Path, fmm_horizon: int, seed: int,
                    trace_axis: str, bootstrap: int) -> Path:
    out_dir = work_dir / "tier_study"
    cmd = [
        sys.executable, "-m", "modules.evaluation.tier_study.combined",
        "--r0-npz", str(base_paths[R0_WINDOW_SIZE]),
        "--r1-npz", str(ours_paths["R1"]), "--r2-npz", str(ours_paths["R2"]),
        "--r3-npz", str(ours_paths["R3"]), "--r4-npz", str(ours_paths["R4"]),
        "--output-dir", str(out_dir), "--fmm-horizon", str(fmm_horizon),
        "--trace-axis", trace_axis, "--bootstrap-samples", str(bootstrap),
        "--seed", str(seed),
    ]
    log("=== TIER STUDY (combined M1-M7) ===")
    log("  " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode:
        raise SystemExit(f"FAILED ({proc.returncode}): tier study")
    log(f"tier study done in {time.time() - t0:.0f}s")
    return out_dir


def zip_outputs(work_dir: Path, base_paths: Dict[int, Path],
                ours_paths: Dict[str, Path], study_dir: Path) -> Path:
    candidates = [*base_paths.values(), *ours_paths.values(),
                 *(study_dir.rglob("*") if study_dir.exists() else [])]
    generated = sorted({f for f in candidates if f.is_file()})
    if not generated:
        raise SystemExit(f"nothing to zip under {work_dir}")

    zip_path = work_dir / "tier_study_run.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in generated:
            zf.write(f, arcname=f.relative_to(work_dir))

    log(f"{len(generated)} file(s) zipped:")
    for f in generated:
        print(f"  {f.relative_to(work_dir)}  ({f.stat().st_size / 1e6:.1f} MB)")
    log(f"-> {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB total)")
    return zip_path


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Local HF model directory (Llama 3.1 8B or similar).")
    ap.add_argument("--corpus-file", type=Path, required=True,
                    help="Path to a raw WikiText-103 file, e.g. wiki.train.tokens.")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "outputs" / "tier_study_run",
                    help="Where every npz/csv/json/zip lands.")
    ap.add_argument("--stage", choices=STAGES, default="corpus",
                    help="Start at this stage, reusing files from earlier "
                         "stages already on disk (default: run everything).")

    ap.add_argument("--prefill", type=int, default=512,
                    help="Prompt length in tokens. NOTE: R0 is recorded at "
                         "window_size=1, so its npz grows as "
                         "S*T*L*H*(prefill+gen) — see the sizing table in the "
                         "module docstring BEFORE raising this. On an 80 GB "
                         "A100 the GPU-side ceiling (output_attentions, "
                         "[B,H,P,P] x L) is prefill ~5120; host RAM binds "
                         "first. Default is a CPU-feasible smoke value, not "
                         "a reportable one — see the module docstring for "
                         "the 80 GB A100 override.")
    ap.add_argument("--gen", type=int, default=256,
                    help="Decode length in tokens. Sets T = 1 + gen, so it "
                         "multiplies R0's size TWICE (via T and via W) — the "
                         "most expensive axis. It also bounds per-head "
                         "fidelity: the per-head step mass is differenced from "
                         "the fp16 CUMULATIVE scores, and by gen ~1000 one "
                         "fp16 ulp of the accumulation is comparable to a "
                         "whole per-step delta. Per-LAYER metrics use the "
                         "recorded fp32 step_window_scores and stay exact at "
                         "any gen; per-HEAD M1/M2 degrade past ~512 and are "
                         "noise by 2048.")
    ap.add_argument("--samples", type=int, default=2,
                    help="num_samples (documents) — also the bootstrap axis. "
                         "docs/evaluation.md: >= 8 for anything reported; the "
                         "default of 2 is a smoke value below that floor. "
                         "Pair with --trace-axis sample once you raise it.")
    ap.add_argument("--budget", type=float, default=0.20,
                    help="cache.cache_budget (fraction of prefill+gen).")
    ap.add_argument("--sink", type=int, default=5)
    ap.add_argument("--local", type=int, default=32,
                    help="local_window_size in tokens — must be a multiple "
                         "of every window_size used (1, 8, 32 -> 32 works).")
    ap.add_argument("--no-quant-promotion", action="store_true",
                    help="sticky Q: a window demoted to int2 never buys back "
                         "an fp16 seat — it stays int2 until evicted outright. "
                         "Applies uniformly to every OURS_CONDITIONS run with "
                         "quant_ratio > 0 (R3, R4); R1/R2 are unaffected "
                         "since they never populate the Q tier. Default is "
                         "the promoting policy (this flag off).")
    ap.add_argument("--fmm-horizon", type=int, default=32,
                    help="H for M1/M6(b) — the repo-wide standard horizon "
                         "(qevict_observations and m1's own default both use "
                         "32). At the default gen=256, ws=32's last flush is "
                         "at decode step 224, leaving 31 future steps: H=32 "
                         "right-censors that ONE trailing flush (out of 8) "
                         "rather than scoring it over a shorter, easier "
                         "horizon. (At gen=1024 the same holds with 32 "
                         "flushes, last at step 992.)")
    ap.add_argument("--trace-axis", choices=("sample_layer", "sample", "layer"),
                    default="sample",
                    help="Bootstrap axis. 'sample' (default) bootstraps over "
                         "documents, which is what --samples >= 8 buys you — "
                         "population-level CIs suitable for reporting. "
                         "'sample_layer' is only defensible for a single-"
                         "document sanity run.")
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv: List[str] = None) -> None:
    args = parse_args(argv)
    if not args.model_dir.exists():
        raise SystemExit(f"--model-dir not found: {args.model_dir}")
    if not args.corpus_file.exists():
        raise SystemExit(f"--corpus-file not found: {args.corpus_file}")

    check_transformers_version()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work_dir}")

    geom_kwargs = dict(prefill=args.prefill, gen=args.gen, samples=args.samples,
                       budget=args.budget, sink=args.sink, local=args.local)

    do = lambda s: STAGES.index(args.stage) <= STAGES.index(s)  # noqa: E731

    corpus_path = work_dir / f"{args.corpus_file.stem}_filtered.jsonl"
    if do("corpus"):
        corpus_path = build_corpus(args.corpus_file, args.model_dir, corpus_path,
                                   args.prefill, args.samples)
    elif not corpus_path.exists():
        raise SystemExit(f"--stage {args.stage} but corpus not built yet: "
                         f"{corpus_path} missing. Run --stage corpus first.")

    print_geometry(args.model_dir, args.prefill, args.gen, args.budget,
                   args.sink, args.local)

    base_paths = {ws: work_dir / f"parity_base_ws{ws}.npz" for ws in BASE_WINDOW_SIZES}
    if do("base"):
        base_paths = run_base_stage(args.model_dir, corpus_path, work_dir, **geom_kwargs)

    ours_paths = {rid: work_dir / f"parity_ours_{rid}.npz" for rid, _, _ in OURS_CONDITIONS}
    if do("ours"):
        for ws, p in base_paths.items():
            if not p.exists():
                raise SystemExit(f"--stage {args.stage} needs {p} (base run) "
                                 "first — run an earlier stage.")
        ours_paths = run_ours_stage(args.model_dir, corpus_path, work_dir,
                                    base_paths,
                                    quant_promotion=not args.no_quant_promotion,
                                    **geom_kwargs)

    study_dir = work_dir / "tier_study"
    if do("study"):
        for p in ours_paths.values():
            if not p.exists():
                raise SystemExit(f"--stage {args.stage} needs {p} (ours run) "
                                 "first — run an earlier stage.")
        study_dir = run_study_stage(base_paths, ours_paths, work_dir,
                                    args.fmm_horizon, args.seed,
                                    args.trace_axis, args.bootstrap_samples)
        report = study_dir / "all_metrics.md"
        if report.exists():
            print("\n" + "=" * 78)
            print(report.read_text(encoding="utf-8"))

    if do("zip"):
        zip_outputs(work_dir, base_paths, ours_paths, study_dir)

    log("all done.")


if __name__ == "__main__":
    main()
