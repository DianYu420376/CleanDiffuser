import pytest
import torch

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.diffusion.pnp_pgd import (
    FixedNoiseDenoiser,
    model_timestep,
    pin_fn,
    pnp_pgd,
    reward_classifier_grad_f,
)
from cleandiffuser.nn_diffusion import JannerUNet1d


class _LinearX0Net(torch.nn.Module):
    """x0-prediction stand-in: pred = P x (a linear 'projection'); records t and grad mode."""

    def __init__(self, P):
        super().__init__()
        self.register_buffer("P", P)
        self.t_seen = []
        self.grad_enabled_seen = []

    def forward(self, x, t, c=None):
        self.t_seen.append(t.detach().clone())
        self.grad_enabled_seen.append(torch.is_grad_enabled())
        return x @ self.P.T


def _projector(dim=4, rank=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=g))
    return q @ q.T


def _quadratic(c):
    def f(x):
        return 0.5 * ((x - c) ** 2).sum(dim=(1, 2))

    def grad_f(x):
        return x - c

    return f, grad_f


def test_update_equation_matches_manual_recursion():
    P = _projector()
    net = _LinearX0Net(P)
    D = FixedNoiseDenoiser(net, alpha=1.0, sigma=0.1, model_t=3.0, predict_noise=False, input_scaling="none")
    c = torch.randn(1, 1, 4, generator=torch.Generator().manual_seed(1))
    _, grad_f = _quadratic(c)
    x0 = torch.randn(3, 1, 4, generator=torch.Generator().manual_seed(2))
    eta, K = 0.3, 7

    res = pnp_pgd(D, grad_f, x0, eta=eta, n_iters=K, record_history=True)

    x = x0.clone()
    for k in range(K):
        x = (x - eta * (x - c)) @ P.T
        torch.testing.assert_close(res.history[k + 1], x)
    torch.testing.assert_close(res.x, x)
    assert res.n_denoiser_evals == K and res.n_grad_evals == K
    assert len(res.history) == K + 1


def test_noise_level_is_fixed_across_iterations():
    net = _LinearX0Net(_projector())
    D = FixedNoiseDenoiser(net, alpha=0.9, sigma=0.4, model_t=5.0, predict_noise=False)
    _, grad_f = _quadratic(torch.zeros(1, 1, 4))
    res = pnp_pgd(D, grad_f, torch.randn(2, 1, 4), eta=0.5, n_iters=10)
    assert len(net.t_seen) == 10
    for t in net.t_seen:
        assert torch.all(t == 5.0)
    assert res.model_t_seen == [5.0] * 10
    assert D.alpha == pytest.approx(0.9) and D.sigma == pytest.approx(0.4)
    assert res.sigma_bar == pytest.approx(0.4 / 0.9)


def test_denoiser_runs_without_grad_and_params_untouched():
    net = _LinearX0Net(_projector())
    D = FixedNoiseDenoiser(net, alpha=1.0, sigma=0.1, model_t=0.0, predict_noise=False)
    _, grad_f = _quadratic(torch.ones(1, 1, 4))
    x0 = torch.randn(2, 1, 4, requires_grad=True)
    res = pnp_pgd(D, grad_f, x0, eta=0.1, n_iters=3)
    assert net.grad_enabled_seen == [False, False, False]
    assert not res.x.requires_grad


def test_vp_input_scaling_and_eps_conversion():
    class _ConstEps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x, t, c=None):
            self.inputs.append(x.clone())
            return torch.full_like(x, 0.5)

    net = _ConstEps()
    alpha, sigma = 0.8, 0.6
    D = FixedNoiseDenoiser(net, alpha=alpha, sigma=sigma, model_t=1.0, predict_noise=True, input_scaling="vp")
    x = torch.randn(2, 3, 4)
    out = D(x)
    torch.testing.assert_close(net.inputs[0], alpha * x)
    torch.testing.assert_close(out, (alpha * x - sigma * 0.5) / alpha)
    assert D.sigma_bar == pytest.approx(sigma / alpha)

    D_none = FixedNoiseDenoiser(net, alpha=alpha, sigma=sigma, model_t=1.0, predict_noise=False, input_scaling="none")
    D_none(x)
    torch.testing.assert_close(net.inputs[1], x)
    assert D_none.sigma_bar == pytest.approx(sigma)


def test_pinned_dimensions_preserved():
    net = _LinearX0Net(0.5 * torch.eye(4))
    fix_mask = torch.zeros(1, 3, 4)
    fix_mask[:, 0, :2] = 1.0
    prior = torch.zeros(1, 3, 4)
    prior[:, 0, :2] = torch.tensor([1.5, -2.0])
    D = FixedNoiseDenoiser(net, alpha=1.0, sigma=0.1, model_t=0.0, predict_noise=False,
                           fix_mask=fix_mask, prior=prior)
    _, grad_f = _quadratic(torch.zeros(1, 3, 4))
    res = pnp_pgd(D, grad_f, torch.randn(2, 3, 4), eta=0.2, n_iters=4,
                  condition_fn=pin_fn(prior, fix_mask), record_history=True)
    for h in res.history:
        torch.testing.assert_close(h[:, 0, :2], prior[:, 0, :2].expand(2, 2))


def test_fixed_point_is_stationary():
    P = _projector()
    net = _LinearX0Net(P)
    D = FixedNoiseDenoiser(net, alpha=1.0, sigma=0.1, model_t=0.0, predict_noise=False, input_scaling="none")
    c = torch.randn(1, 1, 4, generator=torch.Generator().manual_seed(3))
    _, grad_f = _quadratic(c)
    # Fixed point of x = P(x - eta (x - c)) for x in range(P) is x* = P c.
    x_star = c @ P.T
    res = pnp_pgd(D, grad_f, x_star, eta=0.4, n_iters=5)
    torch.testing.assert_close(res.x, x_star)


def test_reward_classifier_grad_f_sign_and_mask():
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    def reward_fn(x, c=None):
        return -((x - target) ** 2).sum(dim=(1, 2))

    clf = RuntimeRewardClassifier(reward_fn)
    mask = torch.zeros(1, 2, 2)
    mask[:, 0, :] = 1.0
    grad_f = reward_classifier_grad_f(clf, mask)
    x = torch.zeros(1, 2, 2)
    expected = 2 * (x - target) * (1 - mask)
    torch.testing.assert_close(grad_f(x), expected)


def _tiny_agent(diffusion_steps, training_diffusion_steps):
    nn = JannerUNet1d(3, model_dim=8, emb_dim=8, dim_mult=[1], timestep_emb_type="positional",
                      attention=False, kernel_size=3)
    fix_mask = torch.zeros(4, 3)
    fix_mask[0, :2] = 1.0
    return DiscreteDiffusionSDE(nn, None, fix_mask, loss_weight=torch.ones(4, 3), device="cpu",
                                diffusion_steps=diffusion_steps,
                                training_diffusion_steps=training_diffusion_steps,
                                predict_noise=False)


@pytest.mark.parametrize("T,Ttr", [(20, 20), (100, 20)])
def test_from_agent_uses_schedule_and_sampler_timestep(T, Ttr):
    agent = _tiny_agent(T, Ttr)
    k = 9 if T == 20 else 47
    D = FixedNoiseDenoiser.from_agent(agent, k)
    assert D.alpha == pytest.approx(float(agent.alpha[k]))
    assert D.sigma == pytest.approx(float(agent.sigma[k]))
    assert D.model_t == pytest.approx(model_timestep(k, T, Ttr))
    if T == Ttr:
        assert D.model_t == float(k)
    else:
        assert D.model_t == pytest.approx(k * (Ttr - 1) / (T - 1))

    # Denoiser output equals a direct network call at the same timestep on alpha*x (pinned).
    prior = torch.zeros(1, 4, 3)
    prior[:, 0, :2] = torch.tensor([0.3, -0.2])
    D = FixedNoiseDenoiser.from_agent(agent, k, prior=prior)
    x = torch.randn(2, 4, 3)
    x_in = D.alpha * x * (1 - agent.fix_mask) + prior * agent.fix_mask
    with torch.no_grad():
        ref = agent.model_ema["diffusion"](x_in, torch.full((2,), D.model_t), None)
    ref = ref * (1 - agent.fix_mask) + prior * agent.fix_mask
    torch.testing.assert_close(D(x), ref)


def test_from_agent_rejects_out_of_range_index():
    agent = _tiny_agent(20, 20)
    with pytest.raises(ValueError):
        FixedNoiseDenoiser.from_agent(agent, 20)


def test_zero_iterations_returns_conditioned_init():
    net = _LinearX0Net(_projector())
    D = FixedNoiseDenoiser(net, alpha=1.0, sigma=0.1, model_t=0.0, predict_noise=False)
    x0 = torch.randn(2, 1, 4)
    res = pnp_pgd(D, lambda x: x, x0, eta=0.1, n_iters=0)
    torch.testing.assert_close(res.x, x0)
    assert res.n_denoiser_evals == 0 and res.n_grad_evals == 0
