"""The read gate on REAL activations: Qwen2.5-7B vs Llama-3.1-8B, CPU.

Real LongBench prompts go through the real model (weights streamed per layer by
HTTP range reads; linears in bf16, attention in fp32). At every layer the cache
is laid out the way the windowed cache lays it out after the first eviction:

  sink (5) | fp body: top-k_fp windows by prefill score + local 128 | Q tier: next N_q

with k_fp / N_q from the repo's own WindowedCacheConfig.resolve at the shipped
operating point (budget 0.20, q 0.70, bytes, gate 0.25) and the ranking the
policy uses (prefill attention mass per window, mean over query heads). The
Q tier's cards are built with the repo's build_sketch from the post-RoPE keys
(anchor = post-RoPE mean of the demoted windows, the store's rule), selected
with gate_reference (head-unit union on Qwen, raw on Llama: the shipped auto
setting), and read through two_tier_window_reference with centroids -- against
the same oracle reading every window. Keys are NOT int2-quantized in either arm,
so the difference is the gate alone.

Queries: the last NQ prompt positions (the first decode step's query is the
last prompt token's).

Recall is split three ways, so the two Qwen-specific suspects get a number each:
  recall_solo    true window weights, each query head picks its own n_sel
  recall_shared  true weights, one n_sel per KV head (head-unit union), so
                 solo -> shared is what GQA sharing costs (rep 7 vs rep 4)
  recall         the shipped card's selection, so shared -> card is the card

Usage (needs huggingface.co; ~15 min per model on a 4-core AMX box):
    python scripts/measure_gate_real_cpu.py --model qwen  --lb-dir <LongBench/data>
    python scripts/measure_gate_real_cpu.py --model llama --lb-dir <LongBench/data>
    python scripts/measure_gate_real_cpu.py --summarize qwen llama
"""
import argparse, json, math, os, sys, time, threading, queue
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from hf_range_tensors import Repo  # noqa: E402

from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from modules.quant.sketch import Sketch, build_sketch, value_centroid  # noqa: E402
from modules.windowed_cache.config import WindowedCacheConfig  # noqa: E402
from modules.windowed_cache.decode_kernel import two_tier_window_reference  # noqa: E402
from modules.windowed_cache.gate_kernel import gate_reference  # noqa: E402
from modules.windowed_cache.gate_diagnosis import gate_error_metrics  # noqa: E402
from utils.prompting import encode_prompt  # noqa: E402

MODELS = {
    "qwen": dict(repo="Qwen/Qwen2.5-7B-Instruct", head_norm=True,
                 overrides=dict(max_position_embeddings=131072,
                                rope_scaling={"rope_type": "yarn", "factor": 4.0,
                                              "original_max_position_embeddings": 32768}),
                 kv_dtype=torch.bfloat16),
    "llama": dict(repo="unsloth/Meta-Llama-3.1-8B-Instruct", head_norm=False,
                  overrides={}, kv_dtype=torch.float16),
}
WS, SINK, LOCAL = 8, 5, 128
CHAT = {"narrativeqa", "musique"}          # triviaqa is raw (few-shot)


def build_prompts(tok, T, spec, root):
    tmpl = json.load(open(os.path.join(os.path.dirname(HERE), "data/longbench_configs/dataset2prompt.json")))
    out = []
    for ds, idx in spec:
        ex = [json.loads(l) for l in open(os.path.join(root, f"{ds}.jsonl"))][idx]
        prompt = tmpl[ds].format(**ex)
        ids = tok(prompt, truncation=False, add_special_tokens=False).input_ids
        if len(ids) > T:                                      # THUDM middle truncation
            half = T // 2
            prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                      + tok.decode(ids[-half:], skip_special_tokens=True))
        if ds in CHAT:
            prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        ids = encode_prompt(tok, prompt, return_tensors="pt").input_ids[0]
        out.append((f"{ds}[{idx}]", ids))
    return out


def prefetch(repo, names_per_layer):
    """Background thread fetching layer L+1 while L computes."""
    q = queue.Queue(maxsize=2)

    def work():
        for L, names in enumerate(names_per_layer):
            ts = {}
            threads = []
            def get(n):
                ts[n] = repo.tensor(n)
            for n in names:
                th = threading.Thread(target=get, args=(n,)); th.start(); threads.append(th)
            for th in threads:
                th.join()
            q.put((L, ts))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    return q


def rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(torch.bfloat16)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def measure_layer(kpost, v, qlast, colsum, res, cfg, spec, ratios, head_norm):
    """kpost/v [Hkv,T,D] fp32 (bf16-rounded), qlast [NQ,Hq,D], colsum [Hq,T]."""
    Hkv, T, D = kpost.shape
    Hq = qlast.shape[1]
    rep = Hq // Hkv
    scaling = D ** -0.5
    nb = T - SINK
    W = -(-nb // WS)
    local_w = LOCAL // WS
    ev = W - local_w
    # window scores per query head over the body, sink stripped
    cs = F.pad(colsum[:, SINK:], (0, W * WS - nb)).reshape(Hq, W, WS).sum(-1)
    order = torch.argsort(cs.mean(0)[:ev], descending=True)
    k_fp, n_q = res.top_k_fp, res.N_q
    fp_w = torch.sort(order[:k_fp]).values
    q_w = torch.sort(order[k_fp:k_fp + n_q]).values
    if n_q < 8:
        return None

    def toks(wins):
        return (SINK + wins[:, None] * WS + torch.arange(WS)).reshape(-1)

    local_tok = torch.arange(SINK + ev * WS, T)
    fp_tok = torch.cat([toks(fp_w), local_tok])
    q_tok = toks(q_w)
    idx = torch.cat([torch.arange(SINK), fp_tok, q_tok])
    k_eff, v_eff = kpost[:, idx][None], v[:, idx][None]
    Sfp = SINK + fp_tok.numel()
    n_body_win = -(-fp_tok.numel() // WS)

    kq = kpost[:, q_tok].reshape(Hkv, n_q, WS, D).permute(1, 0, 2, 3)   # [N,Hkv,ws,D]
    vq = v[:, q_tok].reshape(Hkv, n_q, WS, D).permute(1, 0, 2, 3)
    anc, vanc = kq.mean(dim=(0, 2)), vq.mean(dim=(0, 2))                 # store's rule
    card = Sketch(*[f.unsqueeze(0) for f in build_sketch(kq, anc, vq, vanc)])
    cent = value_centroid(card, vanc)[0].permute(1, 0, 2).repeat_interleave(rep, 0)[None]

    out = {}
    for qi in range(qlast.shape[0]):
        q = qlast[qi][None]
        out_f, wsum_f = two_tier_window_reference(q, k_eff, v_eff, scaling, SINK, WS,
                                                  n_body_win, Sfp=Sfp)
        for r in ratios:
            n_sel = max(1, math.ceil(r * n_q))
            sel, lm = gate_reference(q, *card, anc[None], scaling, n_sel,
                                     head_norm=head_norm)
            out_g, _ = two_tier_window_reference(q, k_eff, v_eff, scaling, SINK, WS,
                                                 n_body_win, Sfp=Sfp, sel=sel,
                                                 logmass=lm, centroids=cent)
            m = gate_error_metrics(out_g, out_f, wsum_f, n_body_win, sel, lm)
            # Recall decomposition with a PERFECT card (true window weights):
            #   solo   -- each query head picks its own n_sel (no GQA sharing)
            #   shared -- one n_sel per KV head, union in head units (as shipped
            #             on Qwen), so solo -> shared is the cost of sharing
            #   (card) -- m["recall"]: the shipped selection
            wq = wsum_f[..., n_body_win:]
            share = wq / wq.sum(-1, keepdim=True).clamp_min(1e-30)       # [1,Hq,n]
            solo = share.topk(n_sel, dim=-1).values.sum(-1)
            crit = share.clamp_min(1e-30).log().reshape(1, Hkv, rep, -1).amax(2)
            keep = torch.zeros_like(crit, dtype=torch.bool)
            keep.scatter_(-1, crit.topk(n_sel, dim=-1).indices, True)
            shared = (share * keep.repeat_interleave(rep, dim=1)).sum(-1)
            m["recall_solo"], m["recall_shared"] = solo, shared
            d = out.setdefault(r, defaultdict(list))
            for k_, t_ in m.items():
                d[k_].extend(t_.flatten().tolist())
    return out, dict(n_q=n_q, k_fp=k_fp)


KEYS = ("out_err", "card_tv", "recall_solo", "recall_shared", "recall",
        "q_share", "skip_share", "skip_mass_err")


def summarize(models, out_dir):
    def q(xs, p):
        t = torch.tensor(xs, dtype=torch.float64)
        return float(t.quantile(p)) if t.numel() else float("nan")
    for model in models:
        res = torch.load(os.path.join(out_dir, f"gate_real_{model}.pt"))
        for r in (0.25, 0.5):
            print(f"{model} ratio {r}:")
            for k in KEYS:
                xs = [x for L in sorted(res) for x in res[L].get(r, {}).get(k, [])]
                print(f"   {k:14} p10 {q(xs, .1):+8.3f}  median {q(xs, .5):+8.3f}  "
                      f"p90 {q(xs, .9):+8.3f}")
        print(f"{model} per layer at 0.25: out_err p90, recall p10 card / shared-oracle")
        for L in sorted(res):
            d = res[L].get(0.25)
            if d:
                print(f"   L{L:>2} {q(d['out_err'], .9):.3f}  {q(d['recall'], .1):.2f} / "
                      f"{q(d['recall_shared'], .1):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", nargs="+", choices=list(MODELS))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--T", type=int, default=6144)
    ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--lb-dir",
                    help="directory holding LongBench's triviaqa/narrativeqa/musique .jsonl "
                         "(unzip data.zip from the THUDM/LongBench dataset repo)")
    ap.add_argument("--out-dir", default="outputs/gate_real")
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.summarize, args.out_dir)
    if not args.model or not args.lb_dir:
        ap.error("--model and --lb-dir are required unless --summarize")
    M = MODELS[args.model]
    repo = Repo(M["repo"])
    cfg = AutoConfig.from_pretrained(M["repo"])
    for k, v in M["overrides"].items():
        setattr(cfg, k, v)
    tok = AutoTokenizer.from_pretrained(M["repo"])
    spec = [("triviaqa", 0), ("triviaqa", 1), ("narrativeqa", 0), ("musique", 0)]
    prompts = build_prompts(tok, args.T, spec, args.lb_dir)
    print("prompts:", [(n, int(ids.numel())) for n, ids in prompts], flush=True)

    L_all = cfg.num_hidden_layers if args.layers is None else args.layers
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // Hq
    rep = Hq // Hkv
    eps = cfg.rms_norm_eps
    gen = {"triviaqa": 32, "narrativeqa": 128, "musique": 32}
    wcfg = WindowedCacheConfig(cache_budget=0.20, window_size=WS, num_sink_tokens=SINK,
                               local_window_size=LOCAL, quant_ratio=0.70,
                               quant_budget_mode="bytes", quant_gate_ratio=0.25)
    resolved = [wcfg.resolve(int(ids.numel()), cfg, M["kv_dtype"], gen[n.split("[")[0]])
                for n, ids in prompts]

    # RoPE from the model's own rotary module (YaRN / llama3 as configured)
    if args.model == "qwen":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as RE
    else:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as RE
    rope = RE(config=cfg)
    cossin = []
    for _, ids in prompts:
        pos = torch.arange(ids.numel())[None]
        c, s = rope(torch.zeros(1, dtype=torch.float32), pos)
        cossin.append((c[0], s[0]))                              # [T, D]

    emb = repo.tensor("model.embed_tokens.weight").to(torch.bfloat16)
    hs = [emb[ids] for _, ids in prompts]
    del emb
    pre = "model.layers.{}."
    names = []
    for L in range(L_all):
        p = pre.format(L)
        n = [p + s for s in ("input_layernorm.weight", "post_attention_layernorm.weight",
                             "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                             "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                             "mlp.gate_proj.weight", "mlp.up_proj.weight",
                             "mlp.down_proj.weight")]
        if args.model == "qwen":
            n += [p + s for s in ("self_attn.q_proj.bias", "self_attn.k_proj.bias",
                                  "self_attn.v_proj.bias")]
        names.append(n)
    fetch = prefetch(repo, names)

    ratios = (0.25, 0.5)
    results = {}
    t_start = time.time()
    while True:
        item = fetch.get()
        if item is None:
            break
        L, w = item
        p = pre.format(L)
        W_ = {k[len(p):]: (t if "norm" in k or "bias" in k else t.to(torch.bfloat16))
              for k, t in w.items()}
        layer_res = defaultdict(lambda: defaultdict(list))
        geo = []
        for pi, (name, ids) in enumerate(prompts):
            h = hs[pi]
            T = h.shape[0]
            x = rmsnorm(h, W_["input_layernorm.weight"], eps)
            q = (x @ W_["self_attn.q_proj.weight"].T).float()
            k = (x @ W_["self_attn.k_proj.weight"].T).float()
            v = (x @ W_["self_attn.v_proj.weight"].T).float()
            if args.model == "qwen":
                q = q + W_["self_attn.q_proj.bias"]
                k = k + W_["self_attn.k_proj.bias"]
                v = v + W_["self_attn.v_proj.bias"]
            # the cache holds bf16/fp16 tensors: round like the model does
            q = q.to(torch.bfloat16).float().reshape(T, Hq, D).transpose(0, 1)
            k = k.to(torch.bfloat16).float().reshape(T, Hkv, D).transpose(0, 1)
            v = v.to(torch.bfloat16).float().reshape(T, Hkv, D).transpose(0, 1)
            c, s = cossin[pi]
            q = (q * c + rotate_half(q) * s).to(torch.bfloat16).float()
            k = (k * c + rotate_half(k) * s).to(torch.bfloat16).float()
            # causal attention, fp32, head by head; column sums for the scores
            attn = torch.empty(Hq, T, D)
            colsum = torch.empty(Hq, T)
            mask = torch.ones(T, T, dtype=torch.bool).triu(1)
            for hq in range(Hq):
                kv = hq // rep
                lg = (q[hq] @ k[kv].T) * D ** -0.5
                lg.masked_fill_(mask, float("-inf"))
                pr = torch.softmax(lg, -1)
                attn[hq] = pr @ v[kv]
                colsum[hq] = pr.sum(0)
                del lg, pr
            r = measure_layer(k, v, q[:, -args.nq:].transpose(0, 1).contiguous(),
                              colsum, resolved[pi], cfg, None, ratios, M["head_norm"])
            if r is not None:
                m, g = r
                geo.append(g)
                for rr, d in m.items():
                    for kk, vv in d.items():
                        layer_res[rr][kk].extend(vv)
            o = attn.transpose(0, 1).reshape(T, Hq * D).to(torch.bfloat16)
            h = h + (o @ W_["self_attn.o_proj.weight"].T)
            x = rmsnorm(h, W_["post_attention_layernorm.weight"], eps)
            g_ = x @ W_["mlp.gate_proj.weight"].T
            u_ = x @ W_["mlp.up_proj.weight"].T
            h = h + ((F.silu(g_.float()) * u_.float()).to(torch.bfloat16)
                     @ W_["mlp.down_proj.weight"].T)
            hs[pi] = h
        results[L] = {rr: {kk: vv for kk, vv in d.items()} for rr, d in layer_res.items()}
        med = lambda xs: float(torch.tensor(xs).median()) if xs else float("nan")
        d25 = results[L].get(0.25, {})
        print(f"L{L:>2} [{time.time()-t_start:6.0f}s] geo {geo[0] if geo else None} | 0.25: "
              f"out_err {med(d25.get('out_err', [])):.4f} card_tv {med(d25.get('card_tv', [])):.3f} "
              f"recall solo/shared/card {med(d25.get('recall_solo', [])):.3f}/{med(d25.get('recall_shared', [])):.3f}/{med(d25.get('recall', [])):.3f} skip_share {med(d25.get('skip_share', [])):.3f} "
              f"skip_mass_err {med(d25.get('skip_mass_err', [])):+.2f}", flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(results, os.path.join(args.out_dir, f"gate_real_{args.model}.pt"))
    print("done", time.time() - t_start)


if __name__ == "__main__":
    main()
