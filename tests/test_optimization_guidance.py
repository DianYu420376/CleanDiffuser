import pytest
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import (
    compute_guided_model_input,
    validate_guidance_config,
)
from cleandiffuser.nn_diffusion import DiT1d


DEVICE = "cpu"
HORIZON = 5
OBS_DIM = 4
ACT_DIM = 2
DIM = OBS_DIM + ACT_DIM
TOL = 1e-5


class _MockReturnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, t, c=None):
        return self.scale * x.reshape(x.shape[0], -1).sum(dim=1, keepdim=True)


class MockReturnClassifier(BaseClassifier):
    def __init__(self, device=DEVICE):
        super().__init__(_MockReturnNet(), device=device)

    def logp(self, x, noise, c=None):
        return self.model_ema(x, noise, c)

    def loss(self, x, noise, y):
        pred = self.model(x, noise, c=None)
        return ((pred - y) ** 2).mean()


@pytest.fixture
def planning_setup():
    nn_diffusion = DiT1d(
        DIM,
        emb_dim=32,
        d_model=64,
        n_heads=4,
        depth=2,
        timestep_emb_type="fourier",
    )
    fix_mask = torch.zeros((HORIZON, DIM))
    fix_mask[0, :OBS_DIM] = 1.0
    loss_weight = torch.ones((HORIZON, DIM))

    agent = DiscreteDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=MockReturnClassifier(device=DEVICE),
        diffusion_steps=5,
        device=DEVICE,
    )
    agent.eval()
    return agent, fix_mask


def _make_prior(batch_size=2):
    prior = torch.randn(batch_size, HORIZON, DIM)
    prior[:, 0, :OBS_DIM] = torch.linspace(1.0, 4.0, OBS_DIM)
    return prior


def test_validate_guidance_config_conflict():
    with pytest.raises(ValueError, match="incompatible with w_cg"):
        validate_guidance_config("optimization", w_cg=0.3)


def test_validate_guidance_config_unknown_mode():
    with pytest.raises(ValueError, match="Unknown guidance_mode"):
        validate_guidance_config("invalid", w_cg=0.0)


def test_compute_guided_model_input_passthrough():
    xt = torch.randn(2, HORIZON, DIM)
    t = torch.zeros(2, dtype=torch.long)
    out, log_p = compute_guided_model_input(xt, t, classifier=None)
    assert torch.allclose(out, xt)
    assert log_p is None


def test_compute_guided_model_input_masks_fixed_dims():
    xt = torch.randn(1, HORIZON, DIM)
    t = torch.zeros(1, dtype=torch.long)
    prior = _make_prior(1)
    fix_mask = torch.zeros((HORIZON, DIM))
    fix_mask[0, :OBS_DIM] = 1.0
    classifier = MockReturnClassifier(device=DEVICE)

    x_model_input, log_p = compute_guided_model_input(
        xt,
        t,
        classifier=classifier,
        fix_mask=fix_mask,
        prior=prior,
        guidance_mode="optimization",
        optimization_guidance_scale=0.1,
    )

    assert log_p is not None
    assert torch.allclose(x_model_input[:, 0, :OBS_DIM], prior[:, 0, :OBS_DIM], atol=TOL)
    assert not torch.allclose(x_model_input, xt)


def test_compute_guided_model_input_chain_invariant():
    xt = torch.randn(1, HORIZON, DIM)
    xt_copy = xt.clone()
    t = torch.zeros(1, dtype=torch.long)
    prior = _make_prior(1)
    fix_mask = torch.zeros((HORIZON, DIM))
    fix_mask[0, :OBS_DIM] = 1.0
    classifier = MockReturnClassifier(device=DEVICE)

    compute_guided_model_input(
        xt,
        t,
        classifier=classifier,
        fix_mask=fix_mask,
        prior=prior,
        guidance_mode="optimization",
        optimization_guidance_scale=0.1,
    )

    assert torch.allclose(xt, xt_copy)


def test_standard_sampling_unchanged_with_defaults(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)
    torch.manual_seed(0)

    traj_a, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="standard",
        optimization_guidance_scale=0.0,
    )

    torch.manual_seed(0)
    traj_b, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
    )

    assert traj_a.shape == prior.shape
    assert torch.allclose(traj_a, traj_b, atol=TOL)


def test_standard_w_cg_changes_output(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)

    torch.manual_seed(1)
    traj_none, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="standard",
    )

    torch.manual_seed(1)
    traj_cg, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.3,
        guidance_mode="standard",
    )

    assert not torch.allclose(traj_none, traj_cg, atol=TOL)


def test_optimization_sampling_runs(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)

    traj, log = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="optimization",
        optimization_guidance_scale=0.05,
    )

    assert traj.shape == prior.shape
    assert log["log_p"] is not None


def test_fixed_observation_hard_enforced(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(3)

    traj, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=3,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="optimization",
        optimization_guidance_scale=0.05,
    )

    assert torch.allclose(traj[:, 0, :OBS_DIM], prior[:, 0, :OBS_DIM], atol=TOL)
