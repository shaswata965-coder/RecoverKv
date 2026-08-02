
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from transformers import Cache as _HFCacheBase
except ImportError:
    _HFCacheBase = object  # type: ignore[assignment,misc]

from .config import ResolvedConfig, WindowedCacheConfig
from .policy import EvictionPolicy
from .scorer import accumulate
from .state import CacheState
from .telemetry import NullTelemetry, Telemetry

from modules.quant import (
    QuantizedStore,
    materialize_effective_kv,
    unrotate_key_window,
)
from modules.quant.effective import rotate_key_window
from modules.quant.slots import n_slots_for


class WindowedCache(_HFCacheBase):

    def __init__(
        self,
        config: WindowedCacheConfig,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
        rope_module: torch.nn.Module,
        num_layers: int,
        max_tokens: int,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        if isinstance(_HFCacheBase, type) and _HFCacheBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):
                pass

        self.config = config
        self.resolved = config.resolve(prefill_len, model_config, kv_dtype, max_tokens)
        self.rope_module = rope_module
        self.num_layers = num_layers
        self.telemetry = telemetry if telemetry is not None else NullTelemetry()

        self._q: float = self.resolved.quant_ratio
        self._promote: bool = self.resolved.quant_promotion
        self._stores: List[Optional[QuantizedStore]] = [None] * num_layers
        if self._q > 0.0:
            num_kv_heads = getattr(
                model_config, "num_key_value_heads",
                getattr(model_config, "num_attention_heads", None),
            )
            head_dim = getattr(model_config, "head_dim", None)
            if head_dim is None:
                nh = getattr(model_config, "num_attention_heads", None)
                hidden = getattr(model_config, "hidden_size", None)
                head_dim = hidden // nh
            n_slots = n_slots_for(self.resolved.top_k_fp, self.resolved.N_q)
            self._stores = [
                QuantizedStore(
                    self.resolved.window_size, head_dim, num_kv_heads,
                    n_slots=n_slots,
                    memoize_read=self.resolved.quant_memoize_read is not False,
                )
                for _ in range(num_layers)
            ]
        self._memoization_resolved = False

        r = self.resolved
        steady = (
            r.num_sink_tokens
            + r.top_k_fp * r.window_size
            + r.local_tokens
            + 2 * r.window_size
        )
        prefill_cap = prefill_len + max(
            r.window_size, r.first_eviction_step + 1
        )
        self._states: List[CacheState] = [
            CacheState(capacity=steady, prefill_capacity=prefill_cap)
            for _ in range(num_layers)
        ]
        self._policies: List[EvictionPolicy] = [
            EvictionPolicy(self.resolved) for _ in range(num_layers)
        ]
        self._generation_step: List[int] = [0] * num_layers
        self._prefill_done: List[bool] = [False] * num_layers
        self._next_original_window_id: List[int] = [0] * num_layers

        self.cache_kwargs: Dict[int, Dict[str, Any]] = {
            i: {} for i in range(num_layers)
        }

        self._last_effective_k: List[Optional[Tensor]] = [None] * num_layers

        self._last_score_meta: List[Optional[Any]] = [None] * num_layers


    def get_seq_length(self, layer_idx: int = 0) -> int:
        t_fp = self._states[layer_idx].seq_length
        store = self._stores[layer_idx]
        if store is None:
            return t_fp
        return t_fp + store.num_active_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def _resolve_memoization(self, batch_size: int) -> None:
        if self._memoization_resolved:
            return
        self._memoization_resolved = True
        pref = self.resolved.quant_memoize_read
        memo = (batch_size == 1) if pref is None else pref
        for store in self._stores:
            if store is not None:
                store.memoize_read = memo

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]

        self._resolve_memoization(key_states.shape[0])

        pos = None
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            pos = cache_kwargs["cache_position"]

        state.append(key_states, value_states, pos)
        n_new = key_states.shape[2]
        policy.extend_total_after_append(n_new)

        is_prefill = not self._prefill_done[layer_idx]
        if is_prefill:
            policy.initialize_after_prefill(state.seq_length)
            self._prefill_done[layer_idx] = True

        merged_kwargs = {}
        if cache_kwargs is not None:
            merged_kwargs.update(cache_kwargs)
        merged_kwargs.update(self.cache_kwargs.get(layer_idx, {}))
        new_window_scores = merged_kwargs.get("window_scores")

        layer_kwargs = self.cache_kwargs.get(layer_idx)
        if layer_kwargs is not None and "window_scores" in layer_kwargs:
            del layer_kwargs["window_scores"]

        if new_window_scores is not None:
            if state.window_scores is None:
                state.window_scores = new_window_scores.clone()
                W = new_window_scores.shape[-1]
                B_w = new_window_scores.shape[0]
                state.original_window_ids = (
                    torch.arange(W, device=new_window_scores.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(B_w, -1)
                    .contiguous()
                )
                self._next_original_window_id[layer_idx] = W
            else:
                W_old = state.window_scores.shape[-1]
                W_new = new_window_scores.shape[-1]
                if W_new > W_old:
                    pad = torch.zeros(
                        state.window_scores.shape[0],
                        state.window_scores.shape[1],
                        W_new - W_old,
                        device=state.window_scores.device,
                        dtype=state.window_scores.dtype,
                    )
                    state.window_scores = torch.cat(
                        [state.window_scores, pad], dim=-1
                    )
                    if state.original_window_ids is not None:
                        n_extra = W_new - W_old
                        start_id = self._next_original_window_id[layer_idx]
                        B_w = state.original_window_ids.shape[0]
                        extra = (
                            torch.arange(
                                start_id, start_id + n_extra,
                                device=state.original_window_ids.device,
                                dtype=torch.long,
                            )
                            .unsqueeze(0)
                            .expand(B_w, -1)
                        )
                        state.original_window_ids = torch.cat(
                            [state.original_window_ids, extra], dim=1
                        )
                        self._next_original_window_id[layer_idx] = start_id + n_extra
                elif W_new < W_old:
                    pad = torch.zeros(
                        new_window_scores.shape[0],
                        new_window_scores.shape[1],
                        W_old - W_new,
                        device=new_window_scores.device,
                        dtype=new_window_scores.dtype,
                    )
                    new_window_scores = torch.cat(
                        [new_window_scores, pad], dim=-1
                    )
                accumulate(state.window_scores, new_window_scores)

        step = self._generation_step[layer_idx]
        should_evict = not is_prefill and policy.should_evict(step)

        if should_evict and state.window_scores is not None and self._q > 0.0:
            self._evict_two_tier(layer_idx, step)
        elif should_evict and state.window_scores is not None:
            B = state.key_states.shape[0]
            H_q = state.window_scores.shape[1]

            retained_window_idx = policy.compute_retain_window_indices(
                state.window_scores
            )
            retain_token_idx = policy.expand_to_token_indices(
                retained_window_idx, state.window_scores.shape[2]
            )

            self.telemetry.record_scores(
                layer_idx, step, state.window_scores, retain_token_idx
            )

            old_positions = (
                torch.gather(
                    state.position_ids, 1,
                    retain_token_idx.to(state.position_ids.device),
                ).clone()
                if self.resolved.rerotate_on_evict
                else None
            )

            state.slice_and_keep(retain_token_idx)

            if self.resolved.rerotate_on_evict:
                state.rerotate_keys(self.rope_module, old_positions)

            idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
            state.window_scores = torch.gather(
                state.window_scores, dim=-1, index=idx_w
            ).contiguous()

            if state.original_window_ids is not None:
                state.original_window_ids = torch.gather(
                    state.original_window_ids, 1,
                    retained_window_idx.to(state.original_window_ids.device),
                ).contiguous()

            policy.set_total_after_compaction(state.seq_length)

        if not is_prefill:
            self._generation_step[layer_idx] = step + 1

        if self._q > 0.0:
            eff_k, eff_v, score_meta = self._materialize(layer_idx)
            self._last_effective_k[layer_idx] = eff_k
            self._last_score_meta[layer_idx] = score_meta
            return eff_k, eff_v
        return state.key_states, state.value_states


    def _materialize(self, layer_idx: int) -> Tuple[Tensor, Tensor, Any]:
        state = self._states[layer_idx]
        store = self._stores[layer_idx]
        if store is None or store.num_active_windows == 0:
            return state.key_states, state.value_states, None

        return materialize_effective_kv(
            state.key_states,
            state.value_states,
            state.position_ids,
            store,
            num_sink=self.resolved.num_sink_tokens,
            window_size=self.resolved.window_size,
            rope_module=self.rope_module,
            out_dtype=state.key_states.dtype,
        )

    @staticmethod
    def _compact(src: Tensor, mask: Tensor, width: int) -> Tuple[Tensor, Tensor]:
        B = mask.shape[0]
        rank = mask.cumsum(1) - 1
        dump = torch.full_like(rank, width)
        idx = torch.where(mask & (rank < width), rank.clamp_min(0), dump)
        out = torch.full((B, width + 1), -1, dtype=src.dtype, device=src.device)
        out.scatter_(1, idx, src)
        valid = torch.zeros((B, width + 1), dtype=torch.bool, device=src.device)
        valid.scatter_(1, idx, mask)
        return out[:, :width], valid[:, :width]

    def _evict_two_tier(self, layer_idx: int, step: int) -> None:
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]
        store = self._stores[layer_idx]
        rope = self.rope_module

        ws = self.resolved.window_size
        num_sink = self.resolved.num_sink_tokens
        dtype = state.key_states.dtype
        device = state.key_states.device

        B, H_kv, T_fp, D = state.key_states.shape
        T_body = T_fp - num_sink
        W = state.window_scores.shape[2]
        n_q_prev = store.num_active_windows

        store.ensure(B, device)
        all_has, all_is_q, all_slot, all_match = store.lookup(
            state.original_window_ids
        )

        n_q_res = 0 if self._promote else n_q_prev
        retained_idx, new_tier = policy.compute_two_tier_retain(
            state.window_scores,
            None if self._promote else all_is_q,
            n_q_res,
        )
        k_fp, n_q, local_w = policy.tier_counts(W, n_q_res)
        n_fp = k_fp + local_w

        self.telemetry.record_scores(layer_idx, step, state.window_scores, retained_idx)

        wids = torch.gather(state.original_window_ids, 1, retained_idx)
        is_q_new = new_tier == 1

        n_slots = all_match.shape[-1]
        has_entry = torch.gather(all_has, 1, retained_idx)
        is_q_cur = torch.gather(all_is_q, 1, retained_idx)
        slot_of = torch.gather(all_slot, 1, retained_idx)
        match = torch.gather(
            all_match, 1, retained_idx.unsqueeze(-1).expand(B, -1, n_slots)
        )

        demote = is_q_new & ~is_q_cur
        fresh = demote & ~has_entry
        react = demote & has_entry
        promote = (
            ~is_q_new & is_q_cur if self._promote
            else torch.zeros_like(is_q_new)
        )

        store.retain_only(match)

        sentinel = torch.iinfo(torch.long).max
        fp_key = torch.where(is_q_new, torch.full_like(wids, sentinel), wids)
        take = torch.argsort(fp_key, dim=1)[:, :n_fp]
        fp_wids = torch.gather(wids, 1, take)
        fp_prom = torch.gather(promote, 1, take)
        fp_slot = torch.gather(slot_of, 1, take)
        prom_rank = fp_prom.cumsum(1) - 1

        body_k = state.key_states[:, :, num_sink:, :]
        body_v = state.value_states[:, :, num_sink:, :]
        body_pos = state.position_ids[:, num_sink:]
        body_wid = ((body_pos - num_sink) // ws).contiguous()

        p_max = min(self.resolved.N_q, n_fp) if self._promote else 0
        prom_slot, prom_valid = self._compact(fp_slot, fp_prom, p_max)
        prom_k = prom_v = prom_pos = None
        if p_max > 0:
            k_pre, v_pre, pos_pre = store.promote_many(prom_slot, prom_valid, dtype)
            prom_k = rotate_key_window(
                k_pre.permute(0, 2, 1, 3, 4).reshape(B, H_kv, p_max * ws, D),
                pos_pre.reshape(B, p_max * ws),
                rope,
            ).to(dtype)
            prom_v = v_pre.to(dtype).permute(0, 2, 1, 3, 4).reshape(
                B, H_kv, p_max * ws, D)
            prom_pos = pos_pre.reshape(B, p_max * ws).to(body_pos.dtype)

        react_slot, react_valid = self._compact(slot_of, react, n_q)
        store.reactivate_many(react_slot, react_valid)

        fresh_wid, fresh_valid = self._compact(wids, fresh, n_q)
        if n_q > 0 and T_body > 0:
            start_f = torch.searchsorted(body_wid, fresh_wid.clamp_min(0))
            tok_f = (
                start_f.unsqueeze(-1) + torch.arange(ws, device=device)
            ).reshape(B, n_q * ws).clamp_(0, T_body - 1)
            idx_d = tok_f.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, n_q * ws, D)
            k_post = torch.gather(body_k, 2, idx_d)
            v_tok = torch.gather(body_v, 2, idx_d)
            prange = torch.gather(body_pos, 1, tok_f)
            k_pre_d = unrotate_key_window(k_post, prange, rope)
            store.demote_many(
                store.table.free_slots(n_q), fresh_valid, fresh_wid,
                k_pre_d.reshape(B, H_kv, n_q, ws, D).permute(0, 2, 1, 3, 4),
                v_tok.reshape(B, H_kv, n_q, ws, D).permute(0, 2, 1, 3, 4),
                prange.reshape(B, n_q, ws),
            )

        n_ev_fp_cur = W - local_w - n_q_prev
        new_body_len = k_fp * ws + (T_body - n_ev_fp_cur * ws)
        i = torch.arange(new_body_len, device=device)
        rank = (i // ws).clamp(max=n_fp - 1)
        jj = rank.unsqueeze(0).expand(B, -1)
        off = (i - rank * ws).unsqueeze(0)

        start_of_rank = torch.searchsorted(body_wid, fp_wids)
        src_fp = (torch.gather(start_of_rank, 1, jj) + off).clamp_(0, max(T_body - 1, 0))
        idx_fp = src_fp.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, new_body_len, D)
        new_k = torch.gather(body_k, 2, idx_fp)
        new_v = torch.gather(body_v, 2, idx_fp)
        new_pos = torch.gather(body_pos, 1, src_fp)

        if p_max > 0:
            tok_is_prom = torch.gather(fp_prom, 1, jj)
            src_pr = (
                torch.gather(prom_rank, 1, jj) * ws + off
            ).clamp_(0, p_max * ws - 1)
            idx_pr = src_pr.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, new_body_len, D)
            sel = tok_is_prom.unsqueeze(1).unsqueeze(-1)
            new_k = torch.where(sel, torch.gather(prom_k, 2, idx_pr), new_k)
            new_v = torch.where(sel, torch.gather(prom_v, 2, idx_pr), new_v)
            new_pos = torch.where(
                tok_is_prom, torch.gather(prom_pos, 1, src_pr), new_pos
            )

        state.replace(
            torch.cat([state.key_states[:, :, :num_sink, :], new_k], dim=2),
            torch.cat([state.value_states[:, :, :num_sink, :], new_v], dim=2),
            torch.cat([state.position_ids[:, :num_sink], new_pos], dim=1),
        )

        H_q = state.window_scores.shape[1]
        idx_w = retained_idx.unsqueeze(1).expand(B, H_q, -1)
        state.window_scores = torch.gather(
            state.window_scores, dim=-1, index=idx_w
        ).contiguous()
        state.original_window_ids = torch.gather(
            state.original_window_ids, 1, retained_idx
        ).contiguous()

        store.commit_active_count(n_q)
        policy.set_total_after_compaction(
            state.seq_length + store.num_active_tokens
        )

    def reorder_cache(self, beam_idx: Tensor) -> None:
        raise NotImplementedError(
            "WindowedCache does not support beam search (reorder_cache). "
            "Use greedy or sampling decoding."
        )
