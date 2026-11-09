"""Fixed-noise plug-and-play proximal-gradient descent (PnP-PGD).

    x^{k+1} = D_{sigma_bar}( x^k - eta * grad f(x^k) )

``D_{sigma_bar}`` is an existing pretrained CleanDiffuser denoiser evaluated at a
single, fixed discrete noise index ``k`` of the model's schedule. The noise level
never changes across iterations, no DDIM relaxation is applied, and the denoiser
is never differentiated (it runs under ``torch.no_grad``).

This is the plain PnP-PGD algorithmic structure with our existing denoiser; it is
not the specially constructed proximal denoiser of Hurault et al. (2022).

Conventions (match ``DiscreteDiffusionSDE``):

* VP forward process ``x_t = alpha_k x_0 + sigma_k eps``; the network takes the
  model timestep (float index, see ``model_timestep``) and predicts either
  ``x_0`` (``predict_noise=False``) or ``eps``.
* Fixed (inpainted) dimensions always hold the clean, unscaled prior, exactly as
  in training (``add_noise``) and sampling (re-pin after every step).
* ``input_scaling="vp"`` feeds ``alpha_k * x`` to the network, i.e. treats the
  clean-space iterate ``x`` as ``x_0 + (sigma_k/alpha_k) eps`` so that
  ``D(x) = E[x_0 | x_t = alpha_k x]`` is a Gaussian denoiser with VE noise level
  ``sigma_bar = sigma_k / alpha_k``. ``input_scaling="none"`` feeds ``x``
  unscaled, which is the convention used by the existing DRGD baseline
  (``TweedieAdapter.denoise``); then ``sigma_bar`` is reported as ``sigma_k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

import torch

GradFn = Callable[[torch.Tensor], torch.Tensor]
ConditionFn = Callable[[torch.Tensor], torch.Tensor]
MetricsFn = Callable[[int, torch.Tensor], Any]

SUPPORTED_INPUT_SCALINGS = ("vp", "none")


def model_timestep(schedule_index: Union[int, float], diffusion_steps: int, training_diffusion_steps: int) -> float:
    """Network time input for a schedule index (mirrors ``DiscreteDiffusionSDE.sample``)."""
    if training_diffusion_steps == diffusion_steps:
        return float(schedule_index)
    return float(schedule_index) * (training_diffusion_steps - 1) / max(diffusion_steps - 1, 1)


class FixedNoiseDenoiser:
    """``D_{sigma_bar}(x)``: clean-sample prediction of a frozen network at one noise index."""

    def __init__(
        self,
        network: torch.nn.Module,
        *,
        alpha: float,
        sigma: float,
        model_t: float,
        predict_noise: bool,
        fix_mask: Optional[Union[torch.Tensor, float]] = None,
        prior: Optional[torch.Tensor] = None,
        input_scaling: str = "vp",
        condition: Any = None,
        noise_index: Optional[int] = None,
    ):
        if input_scaling not in SUPPORTED_INPUT_SCALINGS:
            raise ValueError(f"input_scaling must be one of {SUPPORTED_INPUT_SCALINGS}, got {input_scaling!r}")
        if not alpha > 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.network = network
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.model_t = float(model_t)
        self.predict_noise = bool(predict_noise)
        self.fix_mask = fix_mask
        self.prior = prior
        self.input_scaling = input_scaling
        self.condition = condition
        self.noise_index = noise_index
        self.call_count = 0
        self.model_t_seen: List[float] = []

    @property
    def sigma_bar(self) -> float:
        """Effective Gaussian noise level of ``D`` in the iterate's (clean) coordinates."""
        return self.sigma / self.alpha if self.input_scaling == "vp" else self.sigma

    @classmethod
    def from_agent(
        cls,
        agent,
        noise_index: int,
        *,
        prior: Optional[torch.Tensor] = None,
        use_ema: bool = True,
        input_scaling: str = "vp",
        condition: Any = None,
    ) -> "FixedNoiseDenoiser":
        """Build from a loaded ``DiscreteDiffusionSDE``; ``noise_index`` indexes ``agent.alpha``."""
        n_levels = int(agent.alpha.shape[0])
        if not 0 <= int(noise_index) < n_levels:
            raise ValueError(f"noise_index {noise_index} outside [0, {n_levels - 1}]")
        model = agent.model_ema if use_ema else agent.model
        return cls(
            model["diffusion"],
            alpha=float(agent.alpha[noise_index]),
            sigma=float(agent.sigma[noise_index]),
            model_t=model_timestep(noise_index, agent.diffusion_steps, agent.training_diffusion_steps),
            predict_noise=agent.predict_noise,
            fix_mask=agent.fix_mask,
            prior=prior,
            input_scaling=input_scaling,
            condition=condition,
            noise_index=int(noise_index),
        )

    def _pin(self, x: torch.Tensor) -> torch.Tensor:
        if self.prior is None or self.fix_mask is None:
            return x
        return x * (1.0 - self.fix_mask) + self.prior * self.fix_mask

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        x_in = self.alpha * x if self.input_scaling == "vp" else x
        x_in = self._pin(x_in)
        t = torch.full((x.shape[0],), self.model_t, dtype=torch.float32, device=x.device)
        pred = self.network(x_in, t, self.condition)
        x0 = (x_in - self.sigma * pred) / self.alpha if self.predict_noise else pred
        self.call_count += 1
        self.model_t_seen.append(float(t[0]))
        return self._pin(x0)


@dataclass
class PnPPGDResult:
    x: torch.Tensor
    history: Optional[List[torch.Tensor]]
    metrics: List[Any]
    n_denoiser_evals: int
    n_grad_evals: int
    eta: float
    n_iters: int
    sigma_bar: Optional[float]
    model_t_seen: List[float] = field(default_factory=list)


def pnp_pgd(
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    grad_f: GradFn,
    x_init: torch.Tensor,
    *,
    eta: float,
    n_iters: int,
    condition_fn: Optional[ConditionFn] = None,
    record_history: bool = False,
    metrics_fn: Optional[MetricsFn] = None,
) -> PnPPGDResult:
    """Run ``x <- D(x - eta * grad_f(x))`` for ``n_iters`` iterations.

    ``grad_f`` returns the objective gradient (descent direction is ``-grad_f``).
    ``condition_fn`` (e.g. re-pinning the inpainted initial observation) is applied
    to the initial iterate and after every denoiser call. ``metrics_fn(k, x_k)`` is
    called for k = 0..n_iters (k = 0 is the initial iterate).
    """
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")
    x = x_init.detach().clone()
    if condition_fn is not None:
        x = condition_fn(x)
    history = [x.clone()] if record_history else None
    metrics = [metrics_fn(0, x)] if metrics_fn is not None else []
    n_grad = 0
    calls_before = getattr(denoiser, "call_count", None)
    for k in range(1, n_iters + 1):
        g = grad_f(x).detach()
        n_grad += 1
        x = denoiser(x - eta * g)
        if condition_fn is not None:
            x = condition_fn(x)
        if record_history:
            history.append(x.clone())
        if metrics_fn is not None:
            metrics.append(metrics_fn(k, x))
    n_den = n_iters if calls_before is None else denoiser.call_count - calls_before
    seen = list(getattr(denoiser, "model_t_seen", []))
    seen = seen[len(seen) - n_den:] if n_den else []
    return PnPPGDResult(
        x=x,
        history=history,
        metrics=metrics,
        n_denoiser_evals=int(n_den),
        n_grad_evals=n_grad,
        eta=float(eta),
        n_iters=int(n_iters),
        sigma_bar=getattr(denoiser, "sigma_bar", None),
        model_t_seen=seen,
    )


def reward_classifier_grad_f(classifier, fix_mask: Optional[Union[torch.Tensor, float]] = None) -> GradFn:
    """``grad f = -grad R`` for a ``RuntimeRewardClassifier`` (f = -R), masked like guidance.

    Matches ``guidance.compute_reward_gradient`` (detached, multiplied by ``1 - fix_mask``).
    """

    def grad_f(x: torch.Tensor) -> torch.Tensor:
        t = torch.zeros((x.shape[0],), device=x.device)
        _, grad_r = classifier.gradients(x.clone(), t, None)
        grad_r = grad_r.detach()
        if fix_mask is not None:
            grad_r = grad_r * (1.0 - fix_mask)
        return -grad_r

    return grad_f


def pin_fn(prior: torch.Tensor, fix_mask: Union[torch.Tensor, float]) -> ConditionFn:
    """Hard inpainting overwrite used by the existing sampler after every step."""

    def condition(x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - fix_mask) + prior * fix_mask

    return condition
