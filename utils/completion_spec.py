"""The completion-term ablation knob: what it may be set to, and how it is recorded.

**Evaluation only. The shipped method never sets it.**

A gated decode step reads ``quant_gate_ratio`` of the int2 tier and credits every
window it skipped with a *completion term*: the card's mass estimate corrected by
the read windows' mean log error, ``m~(u) = exp(logmass(u) + delta)``, attending
through the window's value centroid. This knob replaces that term with a
controlled variant, to answer how sensitive quality is to its accuracy:

``mode``
    ``None`` (default)  -- the shipped kernel path, untouched. Every other value
    routes the step through the evaluation epilogue
    (:mod:`modules.windowed_cache.completion`), including ``"calibrated"``.
    ``"calibrated"``    -- the shipped term, recomputed by the epilogue. At
    ``scale = 1``, ``sigma = 0`` it is the identity: the control arm for the
    epilogue itself.
    ``"off"``           -- no completion: the output is attention over the read
    set alone, ``o = o_bar``.
    ``"uncalibrated"``  -- ``m~(u) = exp(logmass(u) - log Z_A)``: the card's raw
    estimate, normalised by the step's exact read-set normaliser but without
    the read windows' correction. NOT ``delta = 0``, which would also drop the
    normalisation.

``scale`` (``c``), calibrated only
    ``m~(u) * c`` for every skipped window: a common-scale mass error.

``sigma`` (nats, with ``seed``), calibrated only
    ``m~(u) * exp(eps_u)``, ``eps ~ N(0, sigma^2)`` per (layer, query head,
    window id) and fixed for the window's life, then rescaled so the skipped
    windows' total is unchanged: a pure redistribution error, orthogonal to
    ``scale``.

``scores``
    ``"output"`` (default) -- the variant changes the attention output only; the
    eviction scores keep the shipped calibrated fill, so the cache's tier
    decisions are driven by the same rule as the control. ``"coupled"`` -- the
    variant's masses also replace the fill in the eviction scores. Refused with
    ``"off"``: a skipped window would score zero, rank last and be evicted, which
    measures the tier collapsing, not the completion term.

Every perturbation acts AFTER the gate's selection and AFTER ``delta``, so which
windows are read is identical to the control at the same cache state.

This module is torch-free so a config can be validated without importing the
cache.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional

MODES = ("calibrated", "off", "uncalibrated")
SCORES = ("output", "coupled")


class CompletionSpec(NamedTuple):
    mode: str
    scale: float = 1.0
    sigma: float = 0.0
    seed: int = 0
    scores: str = "output"


def _num(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    return float(v)


def parse_completion(mode: Any = None, scale: Any = 1.0, sigma: Any = 0.0,
                     seed: Any = 0, scores: Any = "output"
                     ) -> Optional[CompletionSpec]:
    """Validate the five knob fields. ``None`` = the shipped path.

    Raises ``ValueError`` on anything that would be inert or ambiguous, so a
    mistyped arm fails at load instead of running the control under its name.
    """
    scale, sigma = _num("quant_completion_scale", scale), _num("quant_completion_sigma", sigma)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"quant_completion_seed must be an int, got {seed!r}")
    if mode is None:
        if (scale, sigma, seed, scores) != (1.0, 0.0, 0, "output"):
            raise ValueError(
                "quant_completion_scale / _sigma / _seed / _scores are set but "
                "quant_completion is null, which runs the shipped kernel path and "
                "would ignore them. Set quant_completion: calibrated (or another "
                "mode) to run a variant.")
        return None
    if mode not in MODES:
        raise ValueError(f"quant_completion must be null or one of {MODES}, got "
                         f"{mode!r}. ('none' is not accepted: the override parser "
                         "turns the string 'none' into null; use 'off'.)")
    if scores not in SCORES:
        raise ValueError(f"quant_completion_scores must be one of {SCORES}, got {scores!r}")
    if not (scale > 0.0) or scale != scale or scale == float("inf"):
        raise ValueError(f"quant_completion_scale must be a finite number > 0, got "
                         f"{scale} (scale 0 is quant_completion: off)")
    if not (sigma >= 0.0) or sigma == float("inf"):
        raise ValueError(f"quant_completion_sigma must be finite and >= 0, got {sigma}")
    if mode != "calibrated" and (scale != 1.0 or sigma != 0.0):
        raise ValueError(
            f"quant_completion_scale / _sigma perturb the calibrated term; they "
            f"have no meaning with mode {mode!r}. Use mode 'calibrated'.")
    if mode == "off" and scores == "coupled":
        raise ValueError(
            "quant_completion: off with scores: coupled would credit every skipped "
            "window with zero eviction score, so they rank last and are evicted. "
            "That measures the int2 tier collapsing, not the completion term.")
    return CompletionSpec(mode, scale, sigma, seed, scores)


def completion_record(spec: Optional[CompletionSpec]) -> dict:
    """What a sidecar stores. ``mode: "native"`` is the shipped kernel path."""
    if spec is None:
        return {"mode": "native", "scale": 1.0, "sigma": 0.0, "seed": 0, "scores": "output"}
    return dict(spec._asdict())


def from_config(cache_cfg: Any) -> Optional[CompletionSpec]:
    """Parse the knob off any config object carrying the ``quant_completion*`` fields."""
    return parse_completion(
        getattr(cache_cfg, "quant_completion", None),
        getattr(cache_cfg, "quant_completion_scale", 1.0),
        getattr(cache_cfg, "quant_completion_sigma", 0.0),
        getattr(cache_cfg, "quant_completion_seed", 0),
        getattr(cache_cfg, "quant_completion_scores", "output"),
    )
