import pytest
import torch
import torch.nn as nn

from cleandiffuser.diffusion.guidance import compute_pi_t
from cleandiffuser.diffusion.tweedie_adapter import TweedieAdapter


DEVICE = "cpu"
HORIZON = 3
DIM = 4
TOL = 1e-5


class _MockDiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.5))

    def forward(self, x, t, c=None):
        return self.weight * x


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._modules = {"diffusion": _MockDiffusionNet()}

    def __getitem__(self, key):
        return self._modules[key]


@pytest.fixture
def noise_predict_setup():
    model = _MockModel()
    adapter = TweedieAdapter(model, predict_noise=True)
    return adapter, model


@pytest.fixture
def x0_predict_setup():
    model = _MockModel()
    adapter = TweedieAdapter(model, predict_noise=False)
    return adapter, model


def test_denoise_noise_prediction_matches_compute_pi_t(noise_predict_setup):
    adapter, _ = noise_predict_setup
    x = torch.randn(2, HORIZON, DIM, requires_grad=True)
    t = torch.zeros(2)
    alpha = torch.tensor(0.8)
    sigma = torch.tensor(0.6)

    pi = adapter.denoise(x, t, alpha, sigma)
    pred = adapter._model["diffusion"](x, t, None)
    expected = compute_pi_t(x, pred, True, alpha, sigma)

    assert torch.allclose(pi, expected, atol=TOL)
    assert adapter.call_count == 1


def test_denoise_x0_prediction(x0_predict_setup):
    adapter, _ = x0_predict_setup
    x = torch.randn(1, HORIZON, DIM)
    t = torch.zeros(1)
    pi = adapter.denoise(x, t, 0.8, 0.6)
    pred = adapter._model["diffusion"](x, t, None)
    assert torch.allclose(pi, pred, atol=TOL)


def test_model_params_do_not_receive_gradients(noise_predict_setup):
    adapter, model = noise_predict_setup
    x = torch.randn(1, HORIZON, DIM, requires_grad=True)
    t = torch.zeros(1)
    pi = adapter.denoise(x, t, 0.8, 0.6)
    loss = pi.sum()
    loss.backward()

    assert x.grad is not None
    assert model["diffusion"].weight.grad is None


def test_jvp_matches_autograd(noise_predict_setup):
    adapter, _ = noise_predict_setup
    x = torch.randn(1, HORIZON, DIM)
    grad_f = torch.randn_like(x)
    t = torch.zeros(1)
    alpha = torch.tensor(0.8)
    sigma = torch.tensor(0.6)

    x_req = x.detach().requires_grad_(True)
    pi_ref = adapter.denoise(x_req, t, alpha, sigma, create_graph=True)
    jvp_ref = torch.autograd.grad((pi_ref * grad_f).sum(), x_req)[0]

    _, jvp = adapter.jvp_pi_dot_grad(x, grad_f, t, alpha, sigma)
    assert torch.allclose(jvp, jvp_ref, atol=1e-4)


def test_fix_mask_restores_pinned_dims():
    model = _MockModel()
    fix_mask = torch.zeros(HORIZON, DIM)
    fix_mask[0, :2] = 1.0
    prior = torch.randn(1, HORIZON, DIM)
    adapter = TweedieAdapter(
        model,
        predict_noise=False,
        fix_mask=fix_mask,
        prior=prior,
    )
    x = torch.randn(1, HORIZON, DIM)
    x_pinned = x.clone()
    x_pinned[:, 0, :2] = 999.0

    pi = adapter.denoise(x_pinned, torch.zeros(1), 0.8, 0.6)
    pred = model["diffusion"](
        x_pinned * (1 - fix_mask) + prior * fix_mask,
        torch.zeros(1),
        None,
    )
    assert torch.allclose(pi[:, 0, :2], pred[:, 0, :2], atol=TOL)


def test_reset_call_count(noise_predict_setup):
    adapter, _ = noise_predict_setup
    x = torch.randn(1, HORIZON, DIM)
    adapter.denoise(x, torch.zeros(1), 0.8, 0.6)
    assert adapter.call_count == 1
    adapter.reset_call_count()
    assert adapter.call_count == 0
