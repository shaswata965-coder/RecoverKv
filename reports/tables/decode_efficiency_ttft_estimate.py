"""Estimate the full-KV TTFT cells of decode_efficiency.tex (prefill 256).

The full-KV printout (outputs/efficiency_fullkv_2f06bc4) has no TTFT column.
These are ESTIMATES built from prefills that were measured in the same harness;
the table marks them with a dagger. A measured full-KV prefill replaces them.

Model
-----
perf_runner times TTFT as one prefill forward: sync, build cache + hooks,
model(prompt), sync. Decode length never enters it, so every 256/x cell of a
method measures the same TTFT(method, B), and

    TTFT(method, B) = forward(B) + extra(method, B)

forward(B) is the plain HF forward over B x 256 tokens with FlashAttention: a
fixed host/launch cost plus GEMM work linear in B (attention is 0.5% of the
FLOPs at 256 tokens). Ours measures it plus a small extra; eager adds the
materialised attention path on top of it.

1. forward(B): fit a + b*B to ours' measured TTFTs at prefill 256 (every decode
   length), dropping cells >30% off the median at the same B -- the prefill work
   is identical there, so such a cell is not measuring it. The slope implies
   ~180 TFLOPS of GEMM, a normal A100 prefill rate, so the fit is physical.
2. Subtract ours' own prefill work: one fused Triton score pass per layer (an
   extra q.k^T over the prompt) and a few hook launches.
3. Eager: add the eager-over-flash gap. B = 1: this run's own eager - flash
   decode-step gap (extra launches), plus score-tensor traffic. B = 32: the
   measured prefill gap at exactly 256 x 32 in
   reports/decode_throughput_benchmarks.html (0.672 vs 0.605 s). That report is
   another environment, so only the GPU-bound gap is taken from it, not its
   absolute TTFT; the traffic model below predicts ~45 ms of it.
4. Cross-check against MiKV (eager-based, measured separately).

Uncertainty: TTFT reruns move +/-6% (DISTANCE_TO_GOAL.md s10), and the B = 1
cells of every method sit within 0.069-0.084 s. Quote +/-0.01 s at B = 1 and
+/-0.05 s at B = 32.

Run: python reports/tables/decode_efficiency_ttft_estimate.py  (numpy only)
"""
import numpy as np

# Llama-3-8B
L, d, H, HKV, HD, FFN = 32, 4096, 32, 8, 128, 14336
N = 256                                            # prompt tokens
LIN = L * (2 * d * d + 2 * d * HKV * HD + 3 * d * FFN)
FLOP_SEQ = 2 * LIN * N                             # GEMM FLOPs per sequence
LAUNCH = 15e-6                                     # s per extra kernel launch
HBM = 1.6e12                                       # achieved A100-80GB B/s

# Measured TTFT (s), transcribed from the printouts.
OURS = {1: [0.077, 0.082, 0.078, 0.079],           # 256/1024, 4096, 8192, 16384
        8: [0.201, 0.239, 0.217, 0.609],
        16: [0.372, 0.374, 0.374, 0.360],
        32: [0.677, 0.705, 1.106, 0.712]}
MIKV = {1: [0.069, 0.084], 32: [1.020]}            # 256/8192, 256/16384
# This run's full-KV TPOT (eager, flash) at B = 1: the per-forward launch gap.
DECODE_B1 = [(0.0490, 0.0467), (0.0520, 0.0489), (0.0513, 0.0457),
             (0.0489, 0.0451), (0.0483, 0.0449)]
REPORT_GAP_B32 = 0.672 - 0.605                     # eager - flash, 256 x 32


def fit_forward():
    xs, ys, dropped = [], [], []
    for b, vals in OURS.items():
        med = np.median(vals)
        for y in vals:
            if abs(y - med) / med > 0.30:
                dropped.append((b, y))
            else:
                xs.append(b)
                ys.append(y)
    slope, icpt = np.polyfit(xs, ys, 1)
    rms = np.sqrt(np.mean((np.array(ys) - icpt - slope * np.array(xs)) ** 2))
    return icpt, slope, rms, dropped


def ours_extra():
    """Fixed and per-sequence cost of ours' prefill scoring over plain flash."""
    fixed = L * 3 * LAUNCH                          # hook + score-kernel launches
    flop = L * 2 * H * N * N * HD                   # q.k^T, not causal-skipped
    read = L * 2 * H * N * HD * 2                   # q and k, fp16
    return fixed, flop / 100e12 + read / HBM        # small-tile rate ~100 TFLOPS


def eager_traffic(b):
    """Bytes the HF eager attention path moves over flash, whole model."""
    s = b * H * N * N * 2                           # one fp16 score tensor
    kv = b * H * N * HD * 2                         # one repeated K or V
    qo = b * H * N * HD * 2
    per_layer = (2 * (kv + kv // 4)                 # repeat_kv(k), repeat_kv(v)
                 + 2 * qo + s                       # q.k^T
                 + 4 * s                            # * scale, + mask
                 + 3 * s + 3 * s                    # softmax to fp32, back to fp16
                 + s + kv + qo + 2 * qo)            # p.v, transpose().contiguous()
    return L * per_layer


def main():
    a, b, rms, dropped = fit_forward()
    fixed, per_seq = ours_extra()
    launch_gap = float(np.median([e - f for e, f in DECODE_B1]))
    gap = {1: launch_gap + eager_traffic(1) / HBM, 32: REPORT_GAP_B32}

    print(f"ours: TTFT = {a:.4f} + {b * 1e3:.2f} ms * B  "
          f"(rms {rms * 1e3:.1f} ms; dropped {dropped}); "
          f"GEMM rate {FLOP_SEQ / b / 1e12:.0f} TFLOPS")
    print(f"ours-only prefill work: {fixed * 1e3:.1f} ms + {per_seq * 1e3:.2f} ms/seq")
    print(f"eager - flash: B=1 {gap[1] * 1e3:.1f} ms (launch gap {launch_gap * 1e3:.1f}); "
          f"B=32 {gap[32] * 1e3:.0f} ms measured, "
          f"{(eager_traffic(32) / HBM + L * 6 * LAUNCH) * 1e3:.0f} ms by traffic model")
    for bs in (1, 32):
        flash = (a - fixed) + (b - per_seq) * bs
        eager = flash + gap[bs]
        mikv = float(np.mean(MIKV[bs]))
        print(f"B={bs:>2}: full KV flash {flash:.3f} s, eager {eager:.3f} s | "
              f"MiKV {mikv:.3f} s -> MiKV's own prefill work {mikv - eager:+.3f} s")


if __name__ == "__main__":
    main()
