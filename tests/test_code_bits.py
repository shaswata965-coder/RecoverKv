"""The K/V code width is ONE constant, and everything that depends on it follows.

``modules.quant.quantizer.CODE_BITS`` (2, 4 or 8) is the only place the Q tier's
code width is set. These tests pin that every consumer derives from it rather
than restating it: the levels and the packing, the slot table's shapes, the
budget's price for a window (against the bytes the table actually allocates),
the config's divisibility check, the sidecar record, and -- under the Triton
interpreter -- the decode kernel's in-register unpack against a dequantized
reference. They hold at every width, so the same file runs on every build.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from modules.quant.quantizer import (CODE_BITS, CODE_BITS_ALLOWED,
                                     CODES_PER_BYTE, _LEVELS,
                                     bytes_per_q_window, grid_bytes_per_head,
                                     pack_crumbs_last, quantize_key_windows,
                                     quantize_value_windows,
                                     unpack_crumbs_last)
from modules.quant.slots import GRID_FIELDS, QuantSlotTable

REPO = Path(__file__).resolve().parents[1]


def test_the_width_is_a_supported_one_and_the_rest_derives_from_it():
    assert CODE_BITS in CODE_BITS_ALLOWED == (2, 4, 8)
    assert CODES_PER_BYTE == 8 // CODE_BITS
    assert _LEVELS == float(2 ** CODE_BITS - 1)


def test_every_code_value_survives_the_packing_in_its_own_lane():
    """All 2**CODE_BITS values, in every lane of a byte, low lanes first."""
    L = 2 ** CODE_BITS
    codes = torch.arange(L * CODES_PER_BYTE, dtype=torch.int64) % L
    codes = codes.reshape(-1, CODES_PER_BYTE).to(torch.uint8)       # [L, P]
    packed = pack_crumbs_last(codes)
    assert packed.shape == (L, 1) and packed.dtype == torch.uint8
    expect = sum(codes[:, j].to(torch.int64) << (CODE_BITS * j)
                 for j in range(CODES_PER_BYTE))
    assert torch.equal(packed[:, 0].to(torch.int64), expect)
    assert torch.equal(unpack_crumbs_last(packed, CODES_PER_BYTE), codes)


def test_the_quantizer_uses_every_level_and_no_more():
    k = torch.randn(16, 4, 8, 64, generator=torch.Generator().manual_seed(0))
    for packed, n in ((quantize_key_windows(k)[0], 8),
                      (quantize_value_windows(k)[0], 64)):
        codes = unpack_crumbs_last(packed, n)
        assert int(codes.min()) == 0 and int(codes.max()) == 2 ** CODE_BITS - 1


@pytest.mark.parametrize("H, D, ws", [(8, 128, 8), (2, 64, 16), (4, 128, 32)])
def test_the_price_of_a_window_is_the_bytes_the_table_allocates_for_it(H, D, ws):
    """The budget prices a window with ``bytes_per_q_window``; the slot table
    allocates it. If the two ever disagree the memory claim is wrong, so they
    are compared here against the tensors, not against a formula."""
    t = QuantSlotTable(1, 1, ws, D, H, torch.device("cpu"))
    held = sum(getattr(t, f).numel() * getattr(t, f).element_size()
               for f in ("key_codes", "val_codes", *GRID_FIELDS))
    assert held == bytes_per_q_window(H, D, ws)
    codes = 2 * D * ws * CODE_BITS // 8
    assert bytes_per_q_window(H, D, ws) == H * (
        codes + grid_bytes_per_head(D) + grid_bytes_per_head(ws))
    assert t.key_codes.shape[-1] == ws // CODES_PER_BYTE
    assert t.val_codes.shape[-1] == D // CODES_PER_BYTE


def test_the_config_refuses_a_window_the_packing_cannot_hold():
    from modules.windowed_cache.config import WindowedCacheConfig

    def make(ws):
        return WindowedCacheConfig(window_size=ws, num_sink_tokens=0,
                                   local_window_size=ws * 4, cache_budget=0.5,
                                   quant_ratio=0.5)

    make(CODES_PER_BYTE * 2)
    if CODES_PER_BYTE > 1:
        with pytest.raises(ValueError, match=f"divisible by {CODES_PER_BYTE}"):
            make(CODES_PER_BYTE * 2 + 1)


def test_the_sidecar_records_the_width():
    """A config cannot say which width a run used -- the width is a build
    constant -- so every sidecar records it next to ``quant_card_bits``."""
    from utils.cache_factory import quant_code_bits_record
    assert quant_code_bits_record() == CODE_BITS

    from modules.evaluation.perf_runner import describe_tier_geometry
    from modules.windowed_cache.config import WindowedCacheConfig

    class _MC:
        num_attention_heads, num_key_value_heads = 32, 8
        hidden_size, head_dim = 4096, 128

    r = WindowedCacheConfig(window_size=8, num_sink_tokens=5, local_window_size=128,
                            cache_budget=0.2, quant_ratio=0.7).resolve(
        3600, _MC(), torch.float16, 128)
    assert describe_tier_geometry(r, 3600)["quant_code_bits"] == CODE_BITS


def test_the_decode_kernel_unpacks_the_width_under_the_triton_interpreter():
    """``tests/interp_code_bits.py``: the kernel against a dequantized reference.

    Needs a Triton whose interpreter runs this repo's kernels; 3.9.0 does not
    (``mask has unsupported dtype``, also hit by ``interp_card_kernels.py``),
    3.8.0 does.
    """
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, str(REPO / "tests" / "interp_code_bits.py")],
        env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
