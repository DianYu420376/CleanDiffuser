"""GGDM G_loss and Landing DRGD baselines for frozen VP diffusion models."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from cleandiffuser.diffusion.guidance import compute_reward_gradient
from cleandiffuser.diffusion.tweedie_adapter import TweedieAdapter
from cleandiffuser.utils import at_least_ndim


def should_apply_ggdm_guidance(
    guidance_mode: str,
    ggdm_interval: float,
    loop_i: int | None = None,
    last_steps: int | None = None,
) -> bool:
    """GGDM G_loss is active when interval is set (optionally only on last steps)."""
    if guidance_mode not in ("ggdm_g_loss", "ggdm_g_loss_mean") or ggdm_interval == 0.0:
        return False
    if loop_i is None or last_steps is None:
        return True
    return loop_i <= last_steps


def _model_pred_noise(
    xt: torch.Tensor,
    t: torch.Tensor,
    model,
    *,
    predict_noise: bool,
    alpha: Union[torch.Tensor, float],
    sigma: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Convert a CleanDiffuser denoiser output to an epsilon prediction."""
    pred_raw = model["diffusion"](xt, t, None)
    if predict_noise:
        return pred_raw
    alpha_b = at_least_ndim(alpha, xt.dim())
    sigma_b = at_least_ndim(sigma, xt.dim())
    return (xt - alpha_b * pred_raw) / (sigma_b + 1e-12)


def apply_ggdm_g_loss_prediction(
    xt: torch.Tensor,
    t: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    sigma: Union[torch.Tensor, float],
    model,
    classifier,
    *,
    predict_noise: bool,
    ggdm_interval: float,
    obs_noise_sigma: float = 0.1,
    beta_coef: float = 1.0,
    beta_constant: Optional[float] = None,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
    ggdm_target_y: Optional[torch.Tensor] = None,
    ggdm_g_anchor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """G_loss guidance step matching GGDMOptim ``cond_score_v2``.

    Reference: https://github.com/yukang123/GGDMOptim/blob/main/simulations/diffusion.py

    Fixed-anchor mode (Algorithm 1): pass both ``ggdm_g_anchor`` and ``ggdm_target_y``
    from the query point z. Per-step live mode: leave both None to recompute g and
    y = delta + g^T x_t from the current x_t each reverse step.

    Always returns a DDPM noise prediction ``pred_noise`` regardless of the
    underlying model parameterization.
    """
    alpha_b = at_least_ndim(alpha, xt.dim())
    sigma_b = at_least_ndim(sigma, xt.dim())
    h_t = sigma_b ** 2

    if (ggdm_target_y is None) ^ (ggdm_g_anchor is None):
        raise ValueError(
            "GGDM G_loss requires both ggdm_g_anchor and ggdm_target_y from the "
            "query point, or neither for per-step live targets."
        )

    if ggdm_target_y is None:
        _, ggdm_g_anchor = compute_reward_gradient(xt, t, classifier, condition_cg, fix_mask)
        flat_g = ggdm_g_anchor.reshape(ggdm_g_anchor.shape[0], -1)
        flat_xt = xt.reshape(xt.shape[0], -1)
        target_y = ggdm_interval + (flat_g * flat_xt).sum(dim=1, keepdim=True)
    else:
        flat_g = ggdm_g_anchor.reshape(ggdm_g_anchor.shape[0], -1)
        target_y = ggdm_target_y.reshape(ggdm_g_anchor.shape[0], -1)

    use_unscaled_g = getattr(classifier, "uses_reward_gradient", False)
    g_spatial = ggdm_g_anchor.reshape(xt.shape)
    if fix_mask is not None:
        g_spatial = g_spatial * (1.0 - fix_mask)

    if use_unscaled_g:
        with torch.no_grad():
            pred_noise = _model_pred_noise(
                xt, t, model, predict_noise=predict_noise, alpha=alpha, sigma=sigma
            )
        uncond_score = -pred_noise / (sigma_b + 1e-12)
        x0_hat = (xt + h_t * uncond_score) / (alpha_b + 1e-12)
        guidance_dir = g_spatial
    else:
        x_in = xt.detach().requires_grad_(True)
        pred_noise = _model_pred_noise(
            x_in, t, model, predict_noise=predict_noise, alpha=alpha, sigma=sigma
        )
        uncond_score = -pred_noise / (sigma_b + 1e-12)
        x0_hat = (x_in + h_t * uncond_score) / (alpha_b + 1e-12)
        guidance_dir = torch.autograd.grad(
            (x0_hat.reshape(x0_hat.shape[0], -1) * flat_g).sum(), x_in, retain_graph=False
        )[0]
        if fix_mask is not None:
            guidance_dir = guidance_dir * (1.0 - fix_mask)

    flat_x0 = x0_hat.reshape(x0_hat.shape[0], -1)
    value = (flat_x0 * flat_g).sum(dim=1, keepdim=True)

    grad_norm = guidance_dir.reshape(guidance_dir.shape[0], -1).norm(dim=1, keepdim=True)
    while grad_norm.dim() < xt.dim():
        grad_norm = grad_norm.unsqueeze(-1)

    if beta_constant is not None:
        beta_t = torch.full_like(alpha_b, float(beta_constant))
    else:
        beta_t = (alpha_b ** 2 / 2.0) / (
            (alpha_b ** 2) * (obs_noise_sigma ** 2) + h_t * grad_norm ** 2 + 1e-12
        )
    beta_t = beta_t * beta_coef

    while target_y.dim() < xt.dim():
        target_y = target_y.unsqueeze(-1)
    while value.dim() < xt.dim():
        value = value.unsqueeze(-1)

    gradient_2 = 2.0 * (value - target_y) * guidance_dir
    cond_score = uncond_score - beta_t * gradient_2
    pred_noise_out = -cond_score * sigma_b

    log_p, _ = compute_reward_gradient(xt, t, classifier, condition_cg, fix_mask)
    return pred_noise_out, log_p


def compute_ggdm_anchor(
    xt: torch.Tensor,
    classifier,
    *,
    ggdm_interval: float,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed G_loss target y = delta + g^T z and reward gradient g at landed z."""
    t = torch.zeros((xt.shape[0],), device=xt.device, dtype=torch.float32)
    _, grad_reward = compute_reward_gradient(xt, t, classifier, condition_cg, fix_mask)
    flat_g = grad_reward.reshape(xt.shape[0], -1)
    flat_x = xt.reshape(xt.shape[0], -1)
    dot_gx = (flat_g * flat_x).sum(dim=1, keepdim=True)
    target_y = ggdm_interval + dot_gx
    return target_y, grad_reward.detach()


def compute_ggdm_anchor_mean(
    xt: torch.Tensor,
    classifier,
    *,
    ggdm_interval: float,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GGDM-mean anchor: shared g = ∇f(z̄), per-particle y_i = δ + g^T z_i."""
    batch_size = xt.shape[0]
    z_bar = xt.mean(dim=0, keepdim=True)
    t = torch.zeros((1,), device=xt.device, dtype=torch.float32)
    _, grad_at_mean = compute_reward_gradient(
        z_bar, t, classifier, condition_cg, fix_mask
    )
    grad_reward = grad_at_mean.expand(batch_size, *grad_at_mean.shape[1:]).contiguous()
    flat_g = grad_reward.reshape(batch_size, -1)
    flat_x = xt.reshape(batch_size, -1)
    dot_gx = (flat_g * flat_x).sum(dim=1, keepdim=True)
    target_y = ggdm_interval + dot_gx
    return target_y, grad_reward.detach()


def compute_ggdm_target_y(
    xt: torch.Tensor,
    classifier,
    *,
    ggdm_interval: float,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target_y, _ = compute_ggdm_anchor(
        xt,
        classifier,
        ggdm_interval=ggdm_interval,
        condition_cg=condition_cg,
        fix_mask=fix_mask,
    )
    return target_y


def drgd_step(
    x: torch.Tensor,
    adapter: TweedieAdapter,
    classifier,
    *,
    t: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    sigma: Union[torch.Tensor, float],
    step_size: float,
    condition_cg=None,
    prior: Optional[torch.Tensor] = None,
    scale_step_by_vp: bool = False,
) -> torch.Tensor:
    """One DRGD update for reward ascent on the learned manifold."""
    _, grad_f = compute_reward_gradient(x, t, classifier, condition_cg, adapter.fix_mask)
    _, jvp = adapter.jvp_pi_dot_grad(x, grad_f, t, alpha, sigma)
    if scale_step_by_vp:
        alpha_b = at_least_ndim(alpha, x.dim())
        sigma_b = at_least_ndim(sigma, x.dim())
        step_size = step_size * (sigma_b ** 2) / (alpha_b + 1e-12)
    x_tangent = x + step_size * jvp
    x_next = adapter.denoise(x_tangent, t, alpha, sigma)
    if adapter.fix_mask is not None and prior is not None:
        x_next = x_next * (1.0 - adapter.fix_mask) + prior * adapter.fix_mask
    return x_next


def run_drgd_refinement(
    x_init: torch.Tensor,
    adapter: TweedieAdapter,
    classifier,
    *,
    n_steps: int,
    step_size: float,
    t: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    sigma: Union[torch.Tensor, float],
    prior: Optional[torch.Tensor] = None,
    condition_cg=None,
    preserve_history: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Refine a landed sample with discrete DRGD at fixed noise scale."""
    x = x_init.detach().clone()
    history = [x.detach().cpu().numpy()] if preserve_history else None
    adapter.reset_call_count()

    for _ in range(n_steps):
        x = drgd_step(
            x,
            adapter,
            classifier,
            t=t,
            alpha=alpha,
            sigma=sigma,
            step_size=step_size,
            condition_cg=condition_cg,
            prior=prior,
        )
        if preserve_history:
            history.append(x.detach().cpu().numpy())

    log = {"drgd_model_calls": adapter.call_count}
    if preserve_history:
        log["drgd_history"] = np.stack(history, axis=1)
    return x, log
