"""Run ONE branch's int2 store on real Qwen2.5-7B activations; report what it does
to attention. Run it once per branch on the same dump and compare the outputs:

    python scripts/measure_gate_real_cpu.py --model qwen --lb-dir <LB> --dump-dir D
    git worktree add /tmp/qevict origin/int2_qwen
    python scripts/compare_int2_store.py .            D ours.pt
    python scripts/compare_int2_store.py /tmp/qevict  D qevict.pt

(The script imports ``modules`` from ``<branch_root>``, so each branch's own
store, quantizer and RoPE inverse run -- not this branch's.)

Per layer and prompt: the kept set is the policy's (top k_fp fp, next N_q int2 by
prefill score -- identical code on both branches), geometry from the branch's
own resolve(). The int2 windows go through the branch's OWN path exactly as its
cache calls it: unrotate_key_window on the bf16 post-RoPE keys -> QuantizedStore
.demote_many -> effective_q_tier (dequantize + re-rotate) at bf16. Then the
last prompt tokens' queries attend over [sink | fp | int2 tier] with the
dequantized tier vs the same kept set held exactly, and vs the whole prompt.
"""
import glob, os, sys
import torch

root, dump_dir, out_path = sys.argv[1:4]
sys.path.insert(0, root)
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding  # noqa: E402
from modules.quant.store import QuantizedStore  # noqa: E402
from modules.quant.effective import unrotate_key_window  # noqa: E402
from modules.windowed_cache.config import WindowedCacheConfig  # noqa: E402
import inspect  # noqa: E402

try:
    from modules.quant.quantizer import grid_dtype_for
except ImportError:
    grid_dtype_for = None

WS, SINK, LOCAL = 8, 5, 128
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
cfg.max_position_embeddings = 131072
cfg.rope_scaling = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
rope = Qwen2RotaryEmbedding(config=cfg)
kw = dict(cache_budget=0.20, window_size=WS, num_sink_tokens=SINK,
          local_window_size=LOCAL, quant_ratio=0.70)
if "quant_budget_mode" in inspect.signature(WindowedCacheConfig).parameters:
    kw["quant_budget_mode"] = "bytes"
wcfg = WindowedCacheConfig(**kw)
store_params = inspect.signature(QuantizedStore.__init__).parameters
ensure_params = inspect.signature(QuantizedStore.ensure).parameters
DT = torch.bfloat16
gdt = grid_dtype_for(DT) if grid_dtype_for else torch.float16


def attn(q, K, V, rep):
    """q [nq,Hq,D], K/V [Hkv,S,D] fp32 -> out [nq,Hq,D], logits [nq,Hq,S]."""
    D = q.shape[-1]
    Kx = K.repeat_interleave(rep, 0)                       # [Hq,S,D]
    Vx = V.repeat_interleave(rep, 0)
    lg = torch.einsum("nhd,hsd->nhs", q, Kx) * D ** -0.5
    p = torch.softmax(lg, -1)
    return torch.einsum("nhs,hsd->nhd", p, Vx), lg


res = {}
for f in sorted(glob.glob(os.path.join(dump_dir, "L*_p*.pt"))):
    d = torch.load(f)
    L = int(os.path.basename(f)[1:3])
    k, v, q, colsum = d["k"], d["v"], d["q_last"].float(), d["colsum"]
    Hkv, T, D = k.shape
    Hq = q.shape[1]
    rep = Hq // Hkv
    r = wcfg.resolve(T, cfg, DT, d["gen"])
    nb = T - SINK
    W = -(-nb // WS)
    local_w = LOCAL // WS
    ev = W - local_w
    cs = torch.nn.functional.pad(colsum[:, SINK:], (0, W * WS - nb)).reshape(Hq, W, WS).sum(-1)
    order = torch.argsort(cs.mean(0)[:ev], descending=True)
    k_fp, n_q = min(r.top_k_fp, ev), min(r.N_q, ev - min(r.top_k_fp, ev))
    fp_w = torch.sort(order[:k_fp]).values
    q_w = torch.sort(order[k_fp:k_fp + n_q]).values

    def toks(w):
        return (SINK + w[:, None] * WS + torch.arange(WS)).reshape(-1)

    fp_tok = torch.cat([toks(fp_w), torch.arange(SINK + ev * WS, T)])
    q_tok = toks(q_w)

    # ---- the branch's own int2 path, as its cache calls it ----
    k_post_q = k[:, q_tok][None]                            # [1,H,n*ws,D] bf16
    prange = q_tok[None]                                    # [1, n*ws]
    k_pre = unrotate_key_window(k_post_q, prange, rope)
    k_pre = k_pre.reshape(1, Hkv, n_q, WS, D).permute(0, 2, 1, 3, 4)
    v_q = v[:, q_tok].reshape(Hkv, n_q, WS, D).permute(1, 0, 2, 3)[None]
    skw = {}
    if "key_anchor" in store_params:
        skw["key_anchor"] = True                            # auto-on for Qwen2
    if "grid_dtype" in store_params:
        skw["grid_dtype"] = gdt
    st = QuantizedStore(WS, D, Hkv, n_slots=n_q, memoize_read=False, **skw)
    if "grid_dtype" in ensure_params:
        st.ensure(1, torch.device("cpu"), grid_dtype=gdt)
    else:
        st.ensure(1, torch.device("cpu"))
    slots = st.table.free_slots(n_q)
    valid = torch.ones(1, n_q, dtype=torch.bool)
    st.demote_many(slots, valid, q_w[None].long(), k_pre, v_q,
                   q_tok.reshape(1, n_q, WS).long())
    st.commit_active_count(n_q)
    kh, vh, ph = st.effective_q_tier(rope, DT)
    assert torch.equal(ph.reshape(-1), q_tok), "tier order mismatch"
    kh, vh = kh[0].float(), vh[0].float()

    kf, vf = k.float(), v.float()
    sink = torch.arange(SINK)
    K_true = torch.cat([kf[:, sink], kf[:, fp_tok], kf[:, q_tok]], 1)
    V_true = torch.cat([vf[:, sink], vf[:, fp_tok], vf[:, q_tok]], 1)
    K_hat = torch.cat([kf[:, sink], kf[:, fp_tok], kh], 1)
    V_hat = torch.cat([vf[:, sink], vf[:, fp_tok], vh], 1)
    out_full, _ = attn(q, kf, vf, rep)
    out_kept, lg_true = attn(q, K_true, V_true, rep)
    out_int2, lg_hat = attn(q, K_hat, V_hat, rep)
    out_konly, _ = attn(q, K_hat, V_true, rep)             # key error alone

    def rel(a, b):
        return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).flatten()

    nq0 = SINK + fp_tok.numel()
    dl = (lg_hat - lg_true)[..., nq0:]                      # int2 tokens' logit error
    lse = lambda x: x.logsumexp(-1)                         # noqa: E731
    qmass_shift = (lse(lg_hat[..., nq0:]) - lse(lg_true[..., nq0:])).flatten()
    qshare = (torch.softmax(lg_true, -1)[..., nq0:].sum(-1)).flatten()
    row = dict(
        evict_err=rel(out_kept, out_full), int2_err=rel(out_int2, out_kept),
        int2_key_err=rel(out_konly, out_kept), total_err=rel(out_int2, out_full),
        logit_rms=dl.pow(2).mean(-1).sqrt().flatten(), logit_bias=dl.mean(-1).flatten(),
        qmass_shift=qmass_shift, qshare=qshare,
        val_rel=((vh - vf[:, q_tok]).norm(dim=-1) / vf[:, q_tok].norm(dim=-1).clamp_min(1e-6)).flatten(),
        key_rel=((kh - kf[:, q_tok]).norm(dim=-1) / kf[:, q_tok].norm(dim=-1).clamp_min(1e-6)).flatten(),
    )
    for key, val in row.items():
        res.setdefault(L, {}).setdefault(key, []).extend(val.tolist())
    print(f"L{L:02d} {d['name']:15} n_q={n_q} int2_err med {row['int2_err'].median():.4f} "
          f"p90 {row['int2_err'].quantile(.9):.4f} | logit bias med {row['logit_bias'].median():+.3f} "
          f"rms {row['logit_rms'].median():.3f} | qmass shift med {row['qmass_shift'].median():+.3f}",
          flush=True)
torch.save(res, out_path)
