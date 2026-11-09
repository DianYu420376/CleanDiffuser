"""Frozen denoiser adapter returning v_t(x) = E[x_0 | x_t = x].

Used by GGDM (Guo et al., arXiv:2404.14743) and Landing-with-the-Score
(Kharitenko et al., arXiv:2509.23357) baselines. See compute_pi_t for the
VP forward convention x_t = alpha_t x_0 + sigma_t eps.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from cleandiffuser.diffusion.guidance import compute_pi_t
from cleandiffuser.utils import at_least_ndim


class TweedieAdapter:
    """Evaluate pi_t(x) = E[x_0 | x_t = x] from a frozen CleanDiffuser model.

    Parameters stay detached; only input derivatives are requested by callers
    via ``torch.autograd.grad`` or ``torch.func.jvp`` on ``pi_t``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        predict_noise: bool,
        fix_mask: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
        model_timestep_fn: Optional[Callable[[Union[int, float]], float]] = None,
    ):
        self._model = model
        self.predict_noise = predict_noise
        self.fix_mask = fix_mask
        self.prior = prior
        self._model_timestep_fn = model_timestep_fn
        self.call_count = 0

        for param in self._model.parameters():
            param.requires_grad_(False)
        self._model.eval()

    def reset_call_count(self) -> None:
        self.call_count = 0

    def _apply_fix_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.fix_mask is None or self.prior is None:
            return x
        return x * (1.0 - self.fix_mask) + self.prior * self.fix_mask

    def _to_model_t(
        self,
        t: torch.Tensor,
        schedule_t: Optional[Union[int, float]] = None,
    ) -> torch.Tensor:
        if schedule_t is not None and self._model_timestep_fn is not None:
            mapped = self._model_timestep_fn(schedule_t)
            return torch.full_like(t, float(mapped))
        return t

    def denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        alpha: Union[torch.Tensor, float],
        sigma: Union[torch.Tensor, float],
        condition: Any = None,
        *,
        schedule_t: Optional[Union[int, float]] = None,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return v_t(x) = E[x_0 | x_t = x] (Tweedie / clean estimate).

        Differentiable w.r.t. ``x`` when ``x.requires_grad`` is True.
        Model parameters are never updated and do not receive gradients.
        """
        x_in = self._apply_fix_mask(x)
        t_model = self._to_model_t(t, schedule_t)

        with torch.set_grad_enabled(x_in.requires_grad or create_graph):
            pred = self._model["diffusion"](x_in, t_model, condition)

        self.call_count += 1
        return compute_pi_t(x_in, pred, self.predict_noise, alpha, sigma)

    def pi_t(self, *args, **kwargs) -> torch.Tensor:
        """Alias for :meth:`denoise`."""
        return self.denoise(*args, **kwargs)

    def score_from_pi(
        self,
        x: torch.Tensor,
        pi: torch.Tensor,
        sigma: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """VP score s(x, sigma) = (pi(x) - x) / sigma^2 (eq. 6, arXiv:2509.23357)."""
        sigma_b = at_least_ndim(sigma, x.dim())
        return (pi - x) / (sigma_b ** 2 + 1e-12)

    def link_gradient(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        alpha: Union[torch.Tensor, float],
        sigma: Union[torch.Tensor, float],
        condition: Any = None,
        *,
        schedule_t: Optional[Union[int, float]] = None,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Approximate nabla ell_sigma(x) = x + sigma^2 s(x, sigma) (eq. 6).

        Source: Kharitenko et al., arXiv:2509.23357, Section 3.
        """
        pi = self.denoise(
            x,
            t,
            alpha,
            sigma,
            condition,
            schedule_t=schedule_t,
            create_graph=create_graph,
        )
        sigma_b = at_least_ndim(sigma, x.dim())
        return x + (sigma_b ** 2) * self.score_from_pi(x, pi, sigma)

    def jvp_pi_dot_grad(
        self,
        x: torch.Tensor,
        grad_f: torch.Tensor,
        t: torch.Tensor,
        alpha: Union[torch.Tensor, float],
        sigma: Union[torch.Tensor, float],
        condition: Any = None,
        *,
        schedule_t: Optional[Union[int, float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pi(x) and v'(x) grad_f via a single inner product backward pass.

        Implements Remark 4 (arXiv:2509.23357): v'(x) g = nabla_x <v(x), g> with g detached.

        Returns:
            pi: v(x) = E[x_0 | x_t = x]
            jvp: v'(x) grad_f
        """
        x_req = x.detach().requires_grad_(True)
        pi = self.denoise(
            x_req,
            t,
            alpha,
            sigma,
            condition,
            schedule_t=schedule_t,
            create_graph=True,
        )
        grad_f_det = grad_f.detach()
        inner = (pi * grad_f_det).sum()
        jvp = torch.autograd.grad(inner, x_req, retain_graph=False, create_graph=False)[0]
        if self.fix_mask is not None:
            jvp = jvp * (1.0 - self.fix_mask)
        return pi.detach() if not x.requires_grad else pi, jvp
