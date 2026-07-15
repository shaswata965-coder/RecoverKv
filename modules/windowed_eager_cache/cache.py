"""WindowedCache — HuggingFace Cache integration for windowed KV cache.

Orchestration only.  No scoring math, no Top-K math, no attention computation,
no RoPE math — only calls into :mod:`state` and :mod:`policy`.

NOTE: This module is byte-identical to ``modules/windowed_cache/cache.py``
(backends only differ in their ``hooks.py``). Any change here MUST be mirrored
to the flash twin until the duplication is refactored away.
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


class WindowedCache(_HFCacheBase):
    """Windowed KV cache with H2O-style cumulative eviction.

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
            self._stores = [
                QuantizedStore(self.resolved.window_size, head_dim, num_kv_heads)
                for _ in range(num_layers)
            ]

        # Per-layer state and policy
        self._states: List[CacheState] = [CacheState() for _ in range(num_layers)]
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

        # v1 quantization is batch-size 1 only (design §10): the ledger/store and
        # materialize_effective_kv run per row. B > 1 stays fully supported at
        # q == 0 (byte-identical); only the Q tier is gated.
        if self._q > 0.0 and key_states.shape[0] != 1:
            raise NotImplementedError(
                "Two-tier quantization (quant_ratio > 0) is batch-size 1 only in "
                f"v1 (design.md §10); got batch size {key_states.shape[0]}. Use "
                "quant_ratio=0 for B > 1."
            )

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
            retain_token_idx = policy.expand_to_token_indices(retained_window_idx)

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
            eff_k, eff_v = self._materialize(layer_idx)
            # Hand the effective K to the score hook (fix #2) so it need not
            # rebuild it. Overwritten each step; the hook consumes (clears) it.
            self._last_effective_k[layer_idx] = eff_k
            return eff_k, eff_v
        return state.key_states, state.value_states

    # -----------------------------------------------------------------
    # Two-tier read path + eviction (design §5, §8) — q > 0, B = 1
    # -----------------------------------------------------------------

    def _materialize(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """Interleave fp + Q tiers into the effective K/V for one layer (row 0).

        Returns a freshly-built tensor; callers must not assume it aliases the
        stored fp cache (design §9). Empty Q tier ⇒ the fp store, byte-identical.
        """
        state = self._states[layer_idx]
        store = self._stores[layer_idx]
        if store is None or store.num_active_windows == 0:
            return state.key_states, state.value_states

        eff_k, eff_v = materialize_effective_kv(
            state.key_states[0],
            state.value_states[0],
            state.position_ids[0],
            store,
            num_sink=self.resolved.num_sink_tokens,
            window_size=self.resolved.window_size,
            rope_module=self.rope_module,
            out_dtype=state.key_states.dtype,
        )
        return eff_k.unsqueeze(0), eff_v.unsqueeze(0)

    def _evict_two_tier(self, layer_idx: int, step: int) -> None:
        """One two-tier eviction (design §5), operating on row 0 (B = 1).

        Ranks the merged window axis, assigns tiers, moves boundary-crossers
        (demote K→Q / promote Q→K), rebuilds the fp store as
        ``[sink ‖ fp windows in chronological id order]`` keeping every survivor's
        original absolute positions, and updates the Q store + ledger. Positions
        are never renumbered.
        """
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]
        store = self._stores[layer_idx]
        rope = self.rope_module

        ws = self.resolved.window_size
        num_sink = self.resolved.num_sink_tokens
        dtype = state.key_states.dtype

        # --- 1–2. Rank + tier assignment on the merged axis -----------------
        retained_idx, new_tier = policy.compute_two_tier_retain(state.window_scores)
        ri = retained_idx[0]                    # [W_ret] merged indices, ascending
        nt = new_tier[0].tolist()               # 0 = fp, 1 = Q
        owids = state.original_window_ids[0]     # window id per merged index

        # Telemetry: snapshot the merged-axis scores. For the two-tier path the
        # retained indices are merged-WINDOW indices (not token indices — the fp
        # and Q survivors live in different stores).
        self.telemetry.record_scores(layer_idx, step, state.window_scores, retained_idx)

        retained = [(int(owids[m]), nt[k]) for k, m in enumerate(ri.tolist())]
        retained_wids = set(w for w, _ in retained)
        new_fp = sorted(w for w, t in retained if t == 0)
        new_q = set(w for w, t in retained if t == 1)

        def cur_is_q(wid: int) -> bool:
            return store.ledger.is_active(wid)

        # fp-store token → window id (sink tokens map to < 0 and are excluded).
        pos_row = state.position_ids[0]                       # [T_fp]
        tok_wid = (pos_row - num_sink) // ws                  # [T_fp]

        # --- 3a. Build the new fp windows (fp→fp keep + Q→fp promote) -------
        new_fp_windows = []  # (wid, k[H,ntok,D] post-RoPE, v, pos[ntok])
        for wid in new_fp:
            if cur_is_q(wid):
                # Promote: dequant → RoPE at original positions → splice.
                k_pre, v, prange = store.promote(wid, out_dtype=dtype)
                k_rot = rotate_key_window(k_pre, prange, rope).to(dtype)
                new_fp_windows.append((wid, k_rot, v.to(dtype), prange.to(pos_row.dtype)))
            else:
                sel = (tok_wid == wid).nonzero(as_tuple=True)[0]
                k_tok = state.key_states[0][:, sel, :]
                v_tok = state.value_states[0][:, sel, :]
                new_fp_windows.append((wid, k_tok, v_tok, pos_row[sel]))

        # --- 3b. Demote (fp→Q): first-time un-rotate+quantize or reactivate --
        for wid in new_q:
            if cur_is_q(wid):
                continue  # Q→Q: stays active, codes untouched
            if store.has_entry(wid):
                store.reactivate(wid)  # dormant → active; codes/grid frozen (§10)
            else:
                sel = (tok_wid == wid).nonzero(as_tuple=True)[0]
                k_post = state.key_states[0][:, sel, :]
                v_tok = state.value_states[0][:, sel, :]
                prange = pos_row[sel]
                k_pre = unrotate_key_window(k_post, prange, rope)
                store.demote(wid, k_pre, v_tok, prange)

        # --- 3c. Drop entries for windows dropped outright (§6) -------------
        store.retain_only(retained_wids)

        # --- 4. Reassemble the fp store: [sink ‖ fp windows by id] ----------
        sink_k = state.key_states[0][:, :num_sink, :]
        sink_v = state.value_states[0][:, :num_sink, :]
        sink_pos = pos_row[:num_sink]
        new_fp_windows.sort(key=lambda t: t[0])
        parts_k = [sink_k] + [w[1] for w in new_fp_windows]
        parts_v = [sink_v] + [w[2] for w in new_fp_windows]
        parts_pos = [sink_pos] + [w[3] for w in new_fp_windows]

        state.key_states = torch.cat(parts_k, dim=1).unsqueeze(0).contiguous()
        state.value_states = torch.cat(parts_v, dim=1).unsqueeze(0).contiguous()
        state.position_ids = torch.cat(parts_pos, dim=0).unsqueeze(0).contiguous()

        # --- 5. Gather scores + ids to the retained merged axis -------------
        H_q = state.window_scores.shape[1]
        idx_w = ri.view(1, 1, -1).expand(1, H_q, -1)
        state.window_scores = torch.gather(state.window_scores, dim=-1, index=idx_w).contiguous()
        state.original_window_ids = torch.gather(
            state.original_window_ids, 1, ri.view(1, -1)
        ).contiguous()

        # --- 6. Policy total tracks the MERGED length (T_fp + T_q) ----------
        policy.set_total_after_compaction(
            state.seq_length + store.num_active_tokens
        )

    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Beam search is out of scope (v1)."""
        raise NotImplementedError(
            "WindowedCache does not support beam search (reorder_cache). "
            "Use greedy or sampling decoding."
        )
