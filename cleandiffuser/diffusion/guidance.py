"""Reusable guidance helpers for diffusion sampling."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

SUPPORTED_GUIDANCE_MODES = ("standard", "optimization")


def validate_guidance_config(
    guidance_mode: str,
    w_cg: float,
    optimization_guidance_scale: float = 0.0,
) -> None:
    if guidance_mode not in SUPPORTED_GUIDANCE_MODES:
        raise ValueError(
            f"Unknown guidance_mode={guidance_mode!r}. "
            f"Supported modes: {SUPPORTED_GUIDANCE_MODES}."
        )
    if guidance_mode == "optimization" and w_cg != 0.0:
        raise ValueError(
            "guidance_mode='optimization' is incompatible with w_cg != 0. "
            "Use optimization_guidance_scale for reward-gradient input shifting "
            "and set w_cg=0.0."
        )


def compute_guided_model_input(
    xt: torch.Tensor,
    t: torch.Tensor,
    classifier=None,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
    prior: Optional[torch.Tensor] = None,
    guidance_mode: str = "standard",
    optimization_guidance_scale: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build the denoiser query point for optimization-style guidance.

    In optimization mode the chain state ``xt`` is not modified; only the
    denoiser input is shifted along the reward gradient.
    """
    if (
        guidance_mode == "standard"
        or optimization_guidance_scale == 0.0
        or classifier is None
    ):
        return xt, None

    log_p, grad = classifier.gradients(xt.clone(), t, condition_cg)
    grad = grad.detach()

    if fix_mask is not None:
        grad = grad * (1.0 - fix_mask)

    x_model_input = xt + optimization_guidance_scale * grad

    if fix_mask is not None and prior is not None:
        x_model_input = x_model_input * (1.0 - fix_mask) + prior * fix_mask

    return x_model_input, log_p
