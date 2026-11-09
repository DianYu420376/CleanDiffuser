import pytest
import torch

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from cleandiffuser.diffusion.baseline_guidance import (
    compute_ggdm_target_y,
    drgd_step,
)
from cleandiffuser.diffusion.tweedie_adapter import TweedieAdapter


class _IdentityDiffusion(torch.nn.Module):
    def forward(self, x, t, c=None):
        return x


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._modules = {"diffusion": _IdentityDiffusion()}

    def __getitem__(self, key):
        return self._modules[key]


def test_drgd_step_increases_reward():
    target = torch.tensor([[[1.0, 0.0]]])
    mean = torch.zeros_like(target)
    std = torch.ones_like(target)

    def reward_fn(x, c=None):
        x_phys = x * std + mean
        return -((x_phys - target) ** 2).sum(dim=(1, 2))

    classifier = RuntimeRewardClassifier(reward_fn, device="cpu")
    adapter = TweedieAdapter(_MockModel(), predict_noise=False)
    x = torch.zeros((1, 1, 2))
    t = torch.zeros((1,))
    alpha = torch.tensor(1.0)
    sigma = torch.tensor(0.01)

    r0 = reward_fn(x, None).item()
    x1 = drgd_step(
        x,
        adapter,
        classifier,
        t=t,
        alpha=alpha,
        sigma=sigma,
        step_size=0.5,
    )
    r1 = reward_fn(x1, None).item()
    assert r1 > r0


def test_ggdm_fixed_anchor_requires_both():
    from cleandiffuser.diffusion.baseline_guidance import apply_ggdm_g_loss_prediction

    def reward_fn(x, c=None):
        return -(x ** 2).sum(dim=(1, 2))

    classifier = RuntimeRewardClassifier(reward_fn, device="cpu")
    model = _MockModel()
    x = torch.randn(1, 1, 2)
    t = torch.zeros((1,))
    g = torch.ones_like(x)

    with pytest.raises(ValueError, match="both ggdm_g_anchor and ggdm_target_y"):
        apply_ggdm_g_loss_prediction(
            x,
            t,
            alpha=torch.tensor(0.9),
            sigma=torch.tensor(0.1),
            model=model,
            classifier=classifier,
            predict_noise=True,
            ggdm_interval=0.1,
            ggdm_target_y=None,
            ggdm_g_anchor=g,
        )


def test_ggdm_target_y_is_anchor_fixed():
    def reward_fn(x, c=None):
        return -(x ** 2).sum(dim=(1, 2))

    classifier = RuntimeRewardClassifier(reward_fn, device="cpu")
    x_anchor = torch.tensor([[[2.0, 1.0]]])
    y_anchor = compute_ggdm_target_y(
        x_anchor,
        classifier,
        ggdm_interval=0.1,
    )
    x_other = torch.tensor([[[0.0, 0.0]]])
    y_other = compute_ggdm_target_y(
        x_other,
        classifier,
        ggdm_interval=0.1,
    )
    assert not torch.allclose(y_anchor, y_other)


def test_ggdm_mean_anchor_uses_shared_gradient():
    from cleandiffuser.diffusion.baseline_guidance import compute_ggdm_anchor_mean
    from cleandiffuser.diffusion.guidance import compute_reward_gradient

    def reward_fn(x, c=None):
        return -(x ** 2).sum(dim=(1, 2))

    classifier = RuntimeRewardClassifier(reward_fn, device="cpu")
    batch = torch.tensor(
        [
            [[2.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    y, g = compute_ggdm_anchor_mean(batch, classifier, ggdm_interval=0.1)
    z_bar = batch.mean(dim=0, keepdim=True)
    _, g_bar = compute_reward_gradient(
        z_bar, torch.zeros((1,)), classifier, None, None
    )
    assert torch.allclose(g[0], g[1])
    assert torch.allclose(g[0], g_bar[0])
    assert y.shape == (2, 1)
    assert not torch.allclose(y[0], y[1])
