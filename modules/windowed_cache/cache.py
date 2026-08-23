"""WindowedCache — HuggingFace Cache integration for windowed KV cache.

Orchestration only.  No scoring math, no Top-K math, no attention computation,
no RoPE math — only calls into :mod:`state` and :mod:`policy`.

NOTE: This module is byte-identical to ``modules/windowed_eager_cache/cache.py``
(backends only differ in their ``hooks.py``). Any change here MUST be mirrored
to the eager twin until the duplication is refactored away.
"""

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
    """Windowed KV cache with H2O-style cumulative eviction.

    Supports ``B > 1`` at any ``quant_ratio``, for **equal-length** prompts.
    Rows evict divergently — each ranks its own windows — but the tier split
    keeps the same *count* per row, so both tiers stay dense and rectangular
    (BATCHING_PLAN.md §3). Ragged / left-padded batches are not implemented, at
    either tier (BATCHING_PLAN.md §4 Phase 3).

    Parameters
    ----------
    config : WindowedCacheConfig
        User-facing configuration.
    prefill_len : int
        Number of tokens in the prompt (used for budget resolution).
    model_config
        HuggingFace ``PretrainedConfig`` or compatible.
    kv_dtype : torch.dtype
        Data type of the KV cache tensors.
    rope_module : nn.Module
        The model's rotary embedding module, used for key re-rotation only
        when ``config.rerotate_on_evict`` is enabled.
    num_layers : int
        Number of transformer layers.
    telemetry : Telemetry, optional
        Telemetry recorder.  Defaults to :class:`NullTelemetry`.
    """

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
                # transformers >= 4.50 changed Cache.__init__ to require
                # `layers` or `layer_class_to_replicate`. We manage our own
                # per-layer state (self._states) and override the full Cache
                # interface, so skipping the base init is safe.
                pass

        self.config = config
        self.resolved = config.resolve(prefill_len, model_config, kv_dtype, max_tokens)
        self.rope_module = rope_module
        self.num_layers = num_layers
        self.telemetry = telemetry if telemetry is not None else NullTelemetry()

        # Two-tier quantization (design.md §2–§8). q == 0 disables the Q tier and
        # every path below is byte-identical to the single-tier fp16 cache.
        self._q: float = self.resolved.quant_ratio
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
            # Slots are bounded by config — retain_only drops every non-retained
            # window, so live entries never exceed top_k_fp + N_q and no growth
            # policy is needed (design §6; see n_slots_for).
            n_slots = n_slots_for(self.resolved.top_k_fp, self.resolved.N_q)
            self._stores = [
                QuantizedStore(
                    self.resolved.window_size, head_dim, num_kv_heads,
                    n_slots=n_slots,
                    # Provisional: `None` means auto, resolved from the real batch
                    # size at the first update() (see _resolve_memoization).
                    memoize_read=self.resolved.quant_memoize_read is not False,
                )
                for _ in range(num_layers)
            ]
            # Softmax scale for the fused decode kernel (Llama/Qwen: head_dim**-0.5).
            self._attn_scaling: float = head_dim ** -0.5
        else:
            self._attn_scaling = 1.0
        self._memoization_resolved = False

        # Fused two-tier decode (decode_kernel + flash_decode). Off until the flash
        # score hook activates it (CUDA + triton + STICKYKV_FUSED_DECODE on); the
        # eager backend and CPU tests never activate it, so they keep the
        # materialize read path. See update()'s q>0 return.
        self._fused_decode_active: bool = False

        # Per-layer state and policy. The fp store is preallocated so append()
        # never re-cats the whole store (BATCHING_PLAN.md §5: at max batch that
        # copy is ~4x the weight traffic and the largest single term in the step).
        #
        # Sized to the EVICTION BUDGET, not to prefill + max_tokens. The latter is
        # StaticCache's rule and is right for a cache that only grows; this one
        # compacts back to the budget every window_size steps, so prefill-sizing
        # would pin the fp store at its un-evicted size for the whole decode —
        # ~708 MB/row vs ~74 MB/row on Llama-3.1-8B at the qasper steady state,
        # capping B at ~73 against the ~458 the method is supposed to reach. The
        # prompt still needs a full-size buffer for exactly one pass, before the
        # first eviction can score it; `replace` releases it at that eviction.
        r = self.resolved
        steady = (
            r.num_sink_tokens
            + r.top_k_fp * r.window_size     # == top_k_windows at q = 0
            + r.local_tokens
            + 2 * r.window_size              # a window of growth + slack
        )
        # The prompt is resident until the FIRST eviction (EvictionPolicy.
        # should_evict fires it at r.first_eviction_step, not at a window
        # boundary), so the prefill buffer must cover the prompt plus every decode
        # token appended up to and including that step: prefill_len +
        # first_eviction_step + 1. The `window_size` term keeps the historical
        # slack for large windows; the max() guarantees no safety-net realloc when
        # window_size <= that step.
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
        # Running counter of the next original-sequence window ID to assign
        # when new windows appear (post-eviction or as generation extends the cache).
        # Without this, the W_new > W_old branch would emit compact-space indices
        # that collide with surviving original IDs.
        self._next_original_window_id: List[int] = [0] * num_layers

        # Shared scratch for cache_kwargs communication with hooks
        self.cache_kwargs: Dict[int, Dict[str, Any]] = {
            i: {} for i in range(num_layers)
        }

        # Fix #2: the flash score hook needs the SAME effective K that update()
        # built for this layer this pass. Stash it here so the hook reuses it
        # instead of calling _materialize a second time (which would
        # re-dequantize the entire Q tier again, per layer, per decode step).
        # Stays None at q == 0 (the hook reads state.key_states directly there).
        self._last_effective_k: List[Optional[Tensor]] = [None] * num_layers

        # Paired with _last_effective_k: the score-scatter map (order, q_token_len)
        # that undoes the unsorted [sink ‖ body ‖ Q] effective-K layout on the
        # score axis. None when the Q tier is empty (no reorder needed). Both
        # backends' score hooks consume it; see modules.quant.effective.
        self._last_score_meta: List[Optional[Any]] = [None] * num_layers

    # -----------------------------------------------------------------
    # HF Cache interface
    # -----------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the **effective** sequence length for *layer_idx*.

        At ``q > 0`` this is ``T_fp + T_q`` (design §5): HF uses it to size the
        causal mask over the returned effective K/V, and the flash hook uses it
        as the key count. It is a key-count report only — query positioning
        still follows HF's monotonic absolute positions (no override). At
        ``q == 0`` the Q tier is empty, so this is exactly ``state.seq_length``.
        """
        t_fp = self._states[layer_idx].seq_length
        store = self._stores[layer_idx]
        if store is None:
            return t_fp
        return t_fp + store.num_active_tokens

    def get_max_length(self) -> Optional[int]:
        """Return ``None`` — windowed cache doesn't have a static max."""
        return None

    def _resolve_memoization(self, batch_size: int) -> None:
        """Settle the Q-tier read memo once the real batch size is known.

        ``quant_memoize_read=None`` (the default) means: **on at B=1, off above.**
        The memo holds the whole Q tier dequantized to fp16 per layer for the
        whole decode — ~149 MB/row at the qasper steady state against ~125 MB/row
        of actual two-tier KV. At B=1 that is invisible (decode is weight-bound —
        BATCHING_PLAN.md §5) and it saves 7 of every 8 steps' dequant. At B>1 it
        is charged per row and halves the batch that fits, which is the one thing
        the whole method exists to raise. An explicit True/False always wins.
        """
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
        """Append new KV states and optionally evict.

        Steps:
        1. ``state.append(k, v, pos)``
        2. Pull pre-computed window scores from *cache_kwargs*.
        3. Accumulate into ``state.window_scores``.
        4. If ``policy.should_evict(step)``:
           a. Two-step retain: window indices → token indices.
           b. ``state.slice_and_keep`` (surviving keys keep their original RoPE).
           c. Optionally ``state.rerotate_keys`` when ``rerotate_on_evict``.
           d. Gather ``state.window_scores`` by retained window indices.
        5. Return ``(state.key_states, state.value_states)``.
        """
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]

        self._resolve_memoization(key_states.shape[0])

        # Extract position_ids from cache_kwargs if provided
        pos = None
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            pos = cache_kwargs["cache_position"]

        # 1. Append
        state.append(key_states, value_states, pos)
        n_new = key_states.shape[2]
        policy.extend_total_after_append(n_new)

        # Detect prefill vs generation
        is_prefill = not self._prefill_done[layer_idx]
        if is_prefill:
            policy.initialize_after_prefill(state.seq_length)
            self._prefill_done[layer_idx] = True

        # 2. Pull pre-computed window scores
        merged_kwargs = {}
        if cache_kwargs is not None:
            merged_kwargs.update(cache_kwargs)
        merged_kwargs.update(self.cache_kwargs.get(layer_idx, {}))
        new_window_scores = merged_kwargs.get("window_scores")

        # Clear consumed scores so a hook that silently returns on the next step
        # does not cause stale values to be re-accumulated.
        layer_kwargs = self.cache_kwargs.get(layer_idx)
        if layer_kwargs is not None and "window_scores" in layer_kwargs:
            del layer_kwargs["window_scores"]

        # 3. Initialize or accumulate window_scores
        if new_window_scores is not None:
            if state.window_scores is None:
                state.window_scores = new_window_scores.clone()
                # Initialize identity mapping: compact index i == original window i.
                # Stored per row ([B, W]) so divergent per-row eviction keeps each
                # row's surviving window identities independently.
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
                # Handle size mismatch: new scores may cover more windows
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
                    # Extend original_window_ids for new windows using the
                    # running original-sequence counter, not compact-space
                    # indices (which would collide with surviving IDs).
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
                    # Symmetric pad on the incoming scores so in-place += works
                    # without changing accumulate's contract. No
                    # original_window_ids change: no new windows appeared.
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

        # 4. Eviction
        step = self._generation_step[layer_idx]
        should_evict = not is_prefill and policy.should_evict(step)

        if should_evict and state.window_scores is not None and self._q > 0.0:
            # Two-tier eviction (design §5). B == 1 in v1 (guarded at append).
            self._evict_two_tier(layer_idx, step)
        elif should_evict and state.window_scores is not None:
            B = state.key_states.shape[0]
            H_q = state.window_scores.shape[1]

            # a. Two-step retain
            retained_window_idx = policy.compute_retain_window_indices(
                state.window_scores
            )
            retain_token_idx = policy.expand_to_token_indices(
                retained_window_idx, state.window_scores.shape[2]
            )

            # Telemetry
            self.telemetry.record_scores(
                layer_idx, step, state.window_scores, retain_token_idx
            )

            # b. Snapshot old positions before compaction (only needed when
            #    re-rotating; scoped to the flag to avoid confusion).
            #    position_ids is [B, T]; gather per row so each row's snapshot
            #    matches the tokens it actually keeps.
            old_positions = (
                torch.gather(
                    state.position_ids, 1,
                    retain_token_idx.to(state.position_ids.device),
                ).clone()
                if self.resolved.rerotate_on_evict
                else None
            )

            # c. Compact K/V. Surviving keys keep their original RoPE rotation
            #    and position_ids are gathered to their original values.
            state.slice_and_keep(retain_token_idx)

            # d. Optionally re-rotate keys to contiguous positions
            #    (StreamingLLM-style). OFF by default: HF generate advances the
            #    query's cache_position monotonically (it does not re-derive it
            #    from get_seq_length each step on transformers <= 4.47), so
            #    re-rotating keys to contiguous positions while the query stays
            #    at its original absolute position corrupts the RoPE relative
            #    phase after the first eviction. Keeping original positions
            #    matches KVPress / H2O and is correct on any version.
            if self.resolved.rerotate_on_evict:
                state.rerotate_keys(self.rope_module, old_positions)

            # e. Gather window_scores by retained_window_idx
            idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
            state.window_scores = torch.gather(
                state.window_scores, dim=-1, index=idx_w
            ).contiguous()

            # f. Keep original_window_ids in sync with the surviving windows.
            #    Gather per row ([B, W]) because rows may retain different windows.
            if state.original_window_ids is not None:
                state.original_window_ids = torch.gather(
                    state.original_window_ids, 1,
                    retained_window_idx.to(state.original_window_ids.device),
                ).contiguous()

            # Update policy
            policy.set_total_after_compaction(state.seq_length)

        # Advance generation step (only after prefill is done)
        if not is_prefill:
            self._generation_step[layer_idx] = step + 1

        # 5. Return. At q > 0 the return is the interleaved effective K/V so
        #    attention (and the eager scorer) see both tiers; at q == 0 this is
        #    the live fp store, byte-identical to the single-tier cache.
        if self._q > 0.0:
            store = self._stores[layer_idx]
            # Fused decode path: on a real decode step with a non-empty Q tier and
            # the flash kernel active, hand the Q tier + scatter map to the
            # flash_attn_func patch and return the FP TIER for it to attend over —
            # no fp16 concat, and the kernel emits the eviction score. Everything
            # else (prefill, empty Q tier, eager backend, fused disabled) falls
            # through to the materialize path below, unchanged.
            if (self._fused_decode_active and not is_prefill
                    and store is not None and store.num_active_windows > 0):
                from modules.quant.effective import compute_score_meta
                from .decode_kernel import rope_cos_sin_halves
                from . import flash_decode
                # Gather the active Q windows' RAW int2 fields (no dequant — the
                # kernel unpacks + dequants + RoPEs in registers), plus the RoPE
                # halves (position-only, cheap) and the score-scatter map.
                n = store.num_active_windows
                B = state.key_states.shape[0]
                ws = self.resolved.window_size
                idx = store.table.active_order(n)                     # [B, n] slots
                kc, ks, kz, vc, vs, vz, qpos = store.table.gather(idx)
                kc = kc.reshape(B, n, *kc.shape[1:])
                ks = ks.reshape(B, n, *ks.shape[1:])
                kz = kz.reshape(B, n, *kz.shape[1:])
                vc = vc.reshape(B, n, *vc.shape[1:])
                vs = vs.reshape(B, n, *vs.shape[1:])
                vz = vz.reshape(B, n, *vz.shape[1:])
                qpos_flat = qpos.reshape(B, n * ws)
                cos_h, sin_h = rope_cos_sin_halves(self.rope_module, qpos_flat)
                score_meta = compute_score_meta(
                    state.position_ids, qpos_flat,
                    self.resolved.num_sink_tokens, ws,
                )
                flash_decode.set_pending({
                    "layer_idx": layer_idx,
                    "qtier": {
                        "k_codes": kc, "k_scale": ks, "k_zero": kz,
                        "v_codes": vc, "v_scale": vs, "v_zero": vz,
                        "cos": cos_h, "sin": sin_h, "window_size": ws,
                    },
                    "score_meta": score_meta,
                    "num_sink": self.resolved.num_sink_tokens,
                    "window_size": ws,
                    "scaling": self._attn_scaling,
                    "cache": self,
                })
                # The patch writes window_scores; the score forward-hook skips this
                # step. Clear the materialize stashes so no stale effective-K leaks.
                self._last_effective_k[layer_idx] = None
                self._last_score_meta[layer_idx] = None
                return state.key_states, state.value_states

            eff_k, eff_v, score_meta = self._materialize(layer_idx)
            # Hand the effective K + its score-scatter map to the score hook
            # (fix #2) so it need not rebuild them. Overwritten each step; the
            # hook consumes (clears) them.
            self._last_effective_k[layer_idx] = eff_k
            self._last_score_meta[layer_idx] = score_meta
            return eff_k, eff_v
        return state.key_states, state.value_states

    # -----------------------------------------------------------------
    # Two-tier read path + eviction (design §5, §8) — q > 0, B = 1
    # -----------------------------------------------------------------

    def _materialize(self, layer_idx: int) -> Tuple[Tensor, Tensor, Any]:
        """Build the ``[sink ‖ body ‖ Q]`` effective K/V for one layer.

        Returns ``(eff_k, eff_v, score_meta)`` — freshly-built tensors (callers
        must not assume they alias the stored fp cache, design §9) plus the
        score-scatter map the hook needs, or ``score_meta=None`` at an empty Q
        tier (the fp store, byte-identical).
        """
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
        """Gather ``src``'s masked lanes to the front of a ``[B, width]`` tensor.

        The counterpart to ``BATCHING_PLAN.md §3``'s rectangularity result. The
        *retained* counts are equal across rows, but the **transition** counts
        are not: ``demote = is_q_new & ~is_q_cur`` depends on where each row's Q
        tier already is, so row 0 may demote 4 windows while row 1 demotes none.
        Allocating by *count* would therefore need a host sync to learn the max;
        allocating by **rank** into a worst-case-width tensor does not.

        ``width`` must be a proven upper bound on ``mask.sum(1)`` — the caller
        takes it from the budget resolver. Lanes beyond a row's count are
        ``valid=False`` and carry ``-1``; overflow lanes (if the bound were ever
        wrong) are routed to a dump column rather than scattering out of bounds,
        so a bad bound drops work loudly in tests instead of corrupting memory.

        Returns ``(compacted [B, width], valid [B, width])``.
        """
        B = mask.shape[0]
        rank = mask.cumsum(1) - 1                                   # [B, W]
        dump = torch.full_like(rank, width)
        idx = torch.where(mask & (rank < width), rank.clamp_min(0), dump)
        out = torch.full((B, width + 1), -1, dtype=src.dtype, device=src.device)
        out.scatter_(1, idx, src)
        valid = torch.zeros((B, width + 1), dtype=torch.bool, device=src.device)
        valid.scatter_(1, idx, mask)
        return out[:, :width], valid[:, :width]

    def _evict_two_tier(self, layer_idx: int, step: int) -> None:
        """One two-tier eviction (design §5), per row, entirely on device.

        Ranks the merged window axis, assigns tiers, moves boundary-crossers
        (demote K→Q / promote Q→K), rebuilds the fp store as
        ``[sink ‖ fp windows in chronological id order]`` keeping every survivor's
        original absolute positions, and updates the Q store. Positions are never
        renumbered.

        **No Python loops and no host syncs.** The tier bookkeeping used to be
        host-side Python sets over ``.tolist()``ed window ids — which is both a
        pipeline stall per layer per eviction and, more fundamentally, unable to
        carry a batch axis at all: rows evict divergently, so "the set of Q
        windows" is per row. Every decision below is ``[B, W]`` mask algebra
        against the slot table instead, and every tensor width comes from
        :meth:`EvictionPolicy.tier_counts` — host ints that are identical across
        rows (BATCHING_PLAN.md §3), so nothing has to be measured off a tensor.
        """
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

        # --- 1–2. Rank + tier assignment on the merged axis -----------------
        retained_idx, new_tier = policy.compute_two_tier_retain(state.window_scores)
        k_fp, n_q, local_w = policy.tier_counts(W)
        n_fp = k_fp + local_w

        # Telemetry: snapshot the merged-axis scores. For the two-tier path the
        # retained indices are merged-WINDOW indices (not token indices — the fp
        # and Q survivors live in different stores).
        self.telemetry.record_scores(layer_idx, step, state.window_scores, retained_idx)

        wids = torch.gather(state.original_window_ids, 1, retained_idx)   # [B, W_ret]
        is_q_new = new_tier == 1

        # --- Resolve window ids → slots ONCE, then decide with masks --------
        store.ensure(B, device)
        has_entry, is_q_cur, slot_of, match = store.lookup(wids)

        demote = is_q_new & ~is_q_cur      # currently fp, wants Q
        fresh = demote & ~has_entry        # never quantized → quantize now
        react = demote & has_entry         # dormant → reactivate, NO requant (§10)
        promote = ~is_q_new & is_q_cur     # currently Q, wants fp

        # --- 3a. Free dropped entries FIRST (§6) ----------------------------
        # Before allocating, not after: it is what makes n_slots_for's
        # top_k_fp + N_q bound exact rather than merely likely. Safe in either
        # order — retain_only only touches windows that are NOT retained, and
        # every demote/promote/reactivate target is retained by construction.
        store.retain_only(match)

        # --- fp window order: retained-fp windows, ascending id -------------
        # `wids` is already ascending (the merged axis is chronological), so
        # pushing the Q-tier picks to a sentinel and sorting compacts the fp
        # picks to the front, still ascending. Everything the fp rebuild needs
        # is then gathered onto that same axis.
        sentinel = torch.iinfo(torch.long).max
        fp_key = torch.where(is_q_new, torch.full_like(wids, sentinel), wids)
        take = torch.argsort(fp_key, dim=1)[:, :n_fp]                  # [B, n_fp]
        fp_wids = torch.gather(wids, 1, take)
        fp_prom = torch.gather(promote, 1, take)
        fp_slot = torch.gather(slot_of, 1, take)
        prom_rank = fp_prom.cumsum(1) - 1                              # [B, n_fp]

        body_k = state.key_states[:, :, num_sink:, :]
        body_v = state.value_states[:, :, num_sink:, :]
        body_pos = state.position_ids[:, num_sink:]
        # Token → window id. Non-decreasing per row (the fp store is
        # [sink ‖ windows by ascending id] and generation appends the newest
        # tokens at the end), which is what lets searchsorted below resolve a
        # window to its run in one shot — no host span map, no sync.
        body_wid = ((body_pos - num_sink) // ws).contiguous()          # [B, T_body]

        # --- 3b. Promote (Q→fp): ONE batched dequant + ONE RoPE -------------
        # Width is the BOUND, not the count: promotions are ragged per row. The
        # invalid lanes dequantize garbage from free slots and are dropped by the
        # mask below — bit-identical for the valid ones, since every op here is
        # per-window or per-token pointwise.
        p_max = min(self.resolved.N_q, n_fp)
        prom_slot, prom_valid = self._compact(fp_slot, fp_prom, p_max)
        prom_k = prom_v = prom_pos = None
        if p_max > 0:
            k_pre, v_pre, pos_pre = store.promote_many(prom_slot, prom_valid, dtype)
            prom_k = rotate_key_window(
                k_pre.permute(0, 2, 1, 3, 4).reshape(B, H_kv, p_max * ws, D),
                pos_pre.reshape(B, p_max * ws),
                rope,
            ).to(dtype)                                                # [B,H,p*ws,D]
            prom_v = v_pre.to(dtype).permute(0, 2, 1, 3, 4).reshape(
                B, H_kv, p_max * ws, D)
            prom_pos = pos_pre.reshape(B, p_max * ws).to(body_pos.dtype)

        # --- 3c. Demote (fp→Q): un-rotate + quantize the FRESH windows ------
        # Must read the OLD fp store, so it runs before the rebuild below.
        # Reactivations are free — codes/grid/positions are frozen (§10) — so
        # they are a bit flip and never touch this path.
        react_slot, react_valid = self._compact(slot_of, react, n_q)
        store.reactivate_many(react_slot, react_valid)

        fresh_wid, fresh_valid = self._compact(wids, fresh, n_q)
        if n_q > 0 and T_body > 0:
            start_f = torch.searchsorted(body_wid, fresh_wid.clamp_min(0))  # [B, n_q]
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

        # --- 4. Rebuild the fp store: [sink ‖ fp windows by id] -------------
        # Every retained fp window contributes ws tokens except the newest, which
        # may be partial — and it is always last, because a window's id fixes its
        # position range and only the newest can straddle the end of the
        # sequence. So new-body token i belongs to fp window rank i // ws at
        # offset i % ws, and the length is pure config arithmetic: the retained
        # evictable windows are full, and the local tokens are whatever the body
        # holds beyond the evictable-fp windows currently in it.
        n_ev_fp_cur = W - local_w - n_q_prev            # evictable windows in fp now
        new_body_len = k_fp * ws + (T_body - n_ev_fp_cur * ws)
        i = torch.arange(new_body_len, device=device)
        # The newest kept window can run LONGER than ws. When T_body is not a whole
        # number of windows, the trailing partial window has no score column of its
        # own yet, so the retain step (which works on the W scored windows) never
        # counts it — physically it sits right after the newest retained window in
        # the chronological body. Addressing it as a phantom rank n_fp would gather
        # out of bounds (start_of_rank has only n_fp entries), so clamp the rank to
        # the last kept window and let its offset run past ws: those <= ws-1 tail
        # tokens fold into the newest local window, which we keep to the end anyway
        # (negligible vs budget), and get re-split into their own window on the next
        # scoring pass. When the body IS a whole number of kept windows this never
        # triggers — i // ws already tops out at n_fp - 1 — so it is a no-op on every
        # config that worked before.
        rank = (i // ws).clamp(max=n_fp - 1)
        jj = rank.unsqueeze(0).expand(B, -1)                          # [B, T_new]
        off = (i - rank * ws).unsqueeze(0)                            # [1, T_new]

        start_of_rank = torch.searchsorted(body_wid, fp_wids)          # [B, n_fp]
        src_fp = (torch.gather(start_of_rank, 1, jj) + off).clamp_(0, max(T_body - 1, 0))
        idx_fp = src_fp.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, new_body_len, D)
        new_k = torch.gather(body_k, 2, idx_fp)
        new_v = torch.gather(body_v, 2, idx_fp)
        new_pos = torch.gather(body_pos, 1, src_fp)

        if p_max > 0:
            # A promoted window has no tokens in the fp store — splice its
            # freshly-rotated copy in at its chronological slot (design §5 step
            # 3): same layout, different source.
            tok_is_prom = torch.gather(fp_prom, 1, jj)                 # [B, T_new]
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

        # --- 5. Gather scores + ids to the retained merged axis -------------
        H_q = state.window_scores.shape[1]
        idx_w = retained_idx.unsqueeze(1).expand(B, H_q, -1)
        state.window_scores = torch.gather(
            state.window_scores, dim=-1, index=idx_w
        ).contiguous()
        state.original_window_ids = torch.gather(
            state.original_window_ids, 1, retained_idx
        ).contiguous()

        # --- 6. Commit counts; policy total tracks T_fp + T_q ---------------
        store.commit_active_count(n_q)
        policy.set_total_after_compaction(
            state.seq_length + store.num_active_tokens
        )

    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Beam search is out of scope (v1)."""
        raise NotImplementedError(
            "WindowedCache does not support beam search (reorder_cache). "
            "Use greedy or sampling decoding."
        )
