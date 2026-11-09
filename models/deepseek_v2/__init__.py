"""DeepSeek-V2 (MLA) support for RecoverKV.

``geometry`` maps the latent onto the cache (one KV head of width ``Dc + Dr``
whose value is its latent dims), ``rope`` restates DeepSeek's RoPE at absolute
positions, ``attention`` is the owned MLA forward, ``loader`` installs it and
builds the cache. ``upstream/`` is the vendored reference oracle.
See ``docs/deepseek_v2_lite/PLAN.md``.
"""

from .geometry import MLACacheConfig, MLAGeometry, softmax_scale_from_config
from .rope import PackedRope

__all__ = ["MLACacheConfig", "MLAGeometry", "PackedRope", "softmax_scale_from_config"]
