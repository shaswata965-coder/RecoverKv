"""Left-padded batches over per-row caches (``models/deepseek_v2/row_batch.py``).

The batched LongBench harness (``modules/evaluation/longbench_qevict_batched.py``)
copies DefensiveKV's batched SnapKV script: contexts left-padded, per-row
positions, the question block appended left-padded, lockstep greedy decode. These
tests pin what that batching must not change: every row of a padded batch
decodes exactly what that row decodes alone, whatever sits in its pad columns.
Tiny DeepSeek-V2 (MLA, YaRN, MoE) at fp32 on CPU, so the method runs its
materialize read (the fused kernel needs CUDA); the row split itself is the same
code on both.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from models.deepseek_v2.loader import install_recoverkv_mla, make_cache
from models.deepseek_v2.row_batch import RowBatchCache
from modules.evaluation.longbench_qevict_batched import make_batches, run_batch
from modules.windowed_cache.config import WindowedCacheConfig
from tests.test_deepseek_mla import tiny_model

CTX = [61, 37, 50]      # context lengths of the three rows (row 1 is padded most)
QL = [5, 3, 7]          # question lengths (row 1 has the most question padding)
NEW = 9


def _rows(seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = [torch.randint(1, 120, (n,), generator=g).tolist() for n in CTX]
    q = [torch.randint(1, 120, (n,), generator=g).tolist() for n in QL]
    return ctx, q


def _qevict_config():
    return WindowedCacheConfig(window_size=4, num_sink_tokens=2, local_window_size=8,
                               cache_budget=0.3, quant_ratio=0.7, quant_gate_ratio=0.25,
                               first_eviction_step=0)


def _run(m, make, ctx, q, pad_id=0):
    trace = []
    gen, caches = run_batch(m, pad_id, make, ctx, q, NEW, torch.tensor([127]), trace=trace)
    return torch.tensor(gen), torch.stack(trace, 1), caches


def _upstream_greedy(m, ids, steps):
    """Upstream's decompressed attention, one row, no padding: the reference."""
    with torch.no_grad():
        cache = DynamicCache()
        out = m(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True,
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                position_ids=torch.arange(len(ids))[None])
        logits = [out.logits[:, -1].float()]
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = [tok]
        for t in range(steps - 1):
            out = m(input_ids=tok, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(1, len(ids) + t + 1, dtype=torch.long),
                    position_ids=torch.tensor([[len(ids) + t]]))
            logits.append(out.logits[:, -1].float())
            tok = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(tok)
    return torch.cat(toks, 1)[0], torch.stack(logits, 1)[0]


def test_padded_fullkv_batch_matches_upstream_per_row():
    """Latent FullKV rows: each padded row == upstream on that row's own prompt."""
    m = tiny_model()
    ctx, q = _rows()
    refs = [_upstream_greedy(m, c + qq, NEW) for c, qq in zip(ctx, q)]
    h = install_recoverkv_mla(m)
    try:
        gen, logits, _ = _run(m, lambda n_ctx, n_q: DynamicCache(), ctx, q)
    finally:
        h.remove()
    for i, (ref_tok, ref_logits) in enumerate(refs):
        steps = gen.shape[1]
        assert torch.equal(gen[i], ref_tok[:steps]), i
        assert (logits[i] - ref_logits[:steps]).abs().max().item() < 1e-4, i


def test_padded_qevict_batch_matches_each_row_alone():
    """QEvict rows: a row's tokens and logits do not depend on its batch-mates."""
    m = tiny_model()
    ctx, q = _rows(1)
    cfg = _qevict_config()
    h = install_recoverkv_mla(m)
    try:
        make = lambda n_ctx, n_q: make_cache(m, cfg, prefill_len=n_ctx, max_tokens=n_q + NEW)
        gen, logits, caches = _run(m, make, ctx, q)
        alone = [_run(m, make, [c], [qq]) for c, qq in zip(ctx, q)]
    finally:
        h.remove()
    steps = gen.shape[1]
    for i, (g1, l1, c1) in enumerate(alone):
        assert torch.equal(gen[i, :g1.shape[1]], g1[0]), i
        assert (logits[i, :l1.shape[1]] - l1[0]).abs().max().item() < 1e-5, i
    for i, c in enumerate(caches):
        # Each row holds its own tokens only, at its own budget, and compressed.
        assert c._tokens_seen == CTX[i] + QL[i] + steps - 1
        assert c.resolved.total_budget_bytes == int(
            0.3 * (CTX[i] + QL[i] + NEW) * c.resolved.bytes_per_token)
        assert c.get_seq_length(0) < c._tokens_seen
    assert len({c.resolved.total_budget_bytes for c in caches}) == len(caches)
    assert any(c._joint_store is not None and c._joint_store.num_active_windows > 0
               for c in caches), "no row built an int2 tier; the test is not exercising QEvict"


def test_pad_columns_never_reach_a_row():
    """Changing what sits in the pad columns changes nothing, bit for bit."""
    m = tiny_model()
    ctx, q = _rows(2)
    cfg = _qevict_config()
    h = install_recoverkv_mla(m)
    try:
        make = lambda n_ctx, n_q: make_cache(m, cfg, prefill_len=n_ctx, max_tokens=n_q + NEW)
        a = _run(m, make, ctx, q, pad_id=0)
        b = _run(m, make, ctx, q, pad_id=99)
    finally:
        h.remove()
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_row_batch_cache_refuses_a_call_without_starts():
    m = tiny_model()
    h = install_recoverkv_mla(m)
    try:
        rbc = RowBatchCache([DynamicCache(), DynamicCache()])
        with pytest.raises(RuntimeError, match="begin_call"):
            m.model(input_ids=torch.ones(2, 4, dtype=torch.long),
                    position_ids=torch.arange(4)[None].expand(2, -1),
                    past_key_values=rbc, use_cache=True)
    finally:
        h.remove()


def test_batches_are_the_baselines():
    """Longest first, at most batch_size rows and max_batch_tokens padded tokens."""
    lens = [10, 50, 30, 30, 20, 45, 5]
    assert make_batches(lens, 3, 10_000, False) == [[1, 5, 2], [3, 4, 0], [6]]
    # 2 x 50 = 100 fits a 120 cap and a third row (150) does not; 4 x 30 = 120 fits.
    assert make_batches(lens, 8, 120, False) == [[1, 5], [2, 3, 4, 0], [6]]
    assert make_batches(lens, 3, 10_000, True) == [[6, 0, 4], [2, 3, 5], [1]]


def test_context_is_compressed_before_the_first_question_token_attends():
    """DefensiveKV's protocol: compress after the context, then append the question.

    At ``first_eviction_step = 0`` the cache evicts on its first single-token step,
    before that step's query attends -- here, the first question token.
    """
    m = tiny_model()
    ctx, q = _rows(3)
    n = len(ctx[0])
    h = install_recoverkv_mla(m)
    try:
        c = make_cache(m, _qevict_config(), prefill_len=n, max_tokens=len(q[0]) + NEW)
        rbc = RowBatchCache([c])
        with torch.no_grad():
            rbc.begin_call([0])
            m.model(input_ids=torch.tensor([ctx[0]]), position_ids=torch.arange(n)[None],
                    past_key_values=rbc, use_cache=True)
            assert c.get_seq_length(0) == n
            rbc.begin_call([0])
            m.model(input_ids=torch.tensor([[q[0][0]]]), position_ids=torch.tensor([[n]]),
                    past_key_values=rbc, use_cache=True)
    finally:
        h.remove()
    # The full-precision tier now holds the budgeted context plus the question token.
    assert c.get_seq_length(0) < n
