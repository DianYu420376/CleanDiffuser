"""Shared code for the sparse trajectory reconstruction (linear inverse problem) experiment.

All methods share: the pretrained unicycle diffusion checkpoint (read-only), the
training normalizer of full64_6k, the o_0 inpainting convention (fix_mask[0,:4]),
per-trajectory x_T, per-trajectory observation masks, and the objective

    f(x) = 0.5 * ||A x - y||^2   (normalized coordinates),  grad f = A^T (A x - y).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REBUTTAL_DIR = Path(__file__).resolve().parents[1]
WORKTREE = REBUTTAL_DIR.parents[1]
for p in (str(WORKTREE / "pipelines"), str(WORKTREE), str(REBUTTAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "disable")

import gym  # noqa: E402

import cleandiffuser.env.unicycle  # noqa: E402,F401
import unicycle_eval as ue  # noqa: E402
from cleandiffuser.classifier import RuntimeRewardClassifier  # noqa: E402
from cleandiffuser.dataset.unicycle_dataset import UnicycleDataset, load_unicycle_hdf5  # noqa: E402
from cleandiffuser.diffusion.pnp_pgd import (  # noqa: E402
    FixedNoiseDenoiser,
    pin_fn,
    pnp_pgd,
    reward_classifier_grad_f,
)
from rebuttal_pnp import safe_io  # noqa: E402

PROTOCOL_PATH = REBUTTAL_DIR / "configs" / "inverse_protocol.json"
HORIZON = 64
OBS_DIM = 4
ACT_DIM = 2
X_DIM = OBS_DIM + ACT_DIM

METHOD_LABELS = {
    "ground_truth": "Ground truth",
    "unguided": "Unguided",
    "pdg": "PDG",
    "dcg": "DCG",
    "pnp_pgd": "PnP-PGD",
    "pnp_pgd_ddim10_K10": "PnP-PGD (DDIM-10 init, K=10)",
}
METHOD_COLORS = {
    "ground_truth": "black",
    "unguided": "#7F7F7F",
    "pdg": "#2166AC",
    "dcg": "#E66101",
    "pnp_pgd": "#E7298A",
    "pnp_pgd_ddim10_K10": "#F29CC8",
}


def load_protocol() -> dict:
    with open(PROTOCOL_PATH) as f:
        return json.load(f)


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- data


def training_normalizer(training_hdf5: str):
    raw = load_unicycle_hdf5(training_hdf5)
    ds = UnicycleDataset(raw, horizon=HORIZON)
    return ds.get_normalizer(), raw


def episodes_from_flat(data: dict) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(data["episode_ends"], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    obs = np.stack([data["observations"][s:e] for s, e in zip(starts, ends)]).astype(np.float32)
    act = np.stack([data["actions"][s:e] for s, e in zip(starts, ends)]).astype(np.float32)
    return obs, act


def make_masks(n: int, seed: int, n_obs: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Observed timesteps per trajectory (sorted, from {1..63}) and the (n,64,6) entry mask."""
    rng = np.random.default_rng(seed)
    idx = np.stack([np.sort(rng.choice(np.arange(1, HORIZON), n_obs, replace=False)) for _ in range(n)])
    M = np.zeros((n, HORIZON, X_DIM), dtype=np.float32)
    for j in range(n):
        M[j, idx[j], 0] = 1.0
        M[j, idx[j], 1] = 1.0
    return idx.astype(np.int64), M


def unobserved_time_mask(obs_idx: np.ndarray) -> np.ndarray:
    """(n,64) bool: t in {1..63} not observed (t=0 is pinned, excluded)."""
    n = obs_idx.shape[0]
    U = np.ones((n, HORIZON), dtype=bool)
    U[:, 0] = False
    for j in range(n):
        U[j, obs_idx[j]] = False
    return U


def x_norm_from_phys(normalizer, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    x = np.zeros(obs.shape[:-1] + (X_DIM,), dtype=np.float32)
    x[..., :OBS_DIM] = normalizer.normalize(obs.astype(np.float32))
    x[..., OBS_DIM:] = act.astype(np.float32)
    return x


def make_xT(n: int, base_seed: int, device: str = "cuda:0") -> torch.Tensor:
    xs = []
    for j in range(n):
        torch.manual_seed(base_seed + j)
        xs.append(torch.randn((1, HORIZON, X_DIM), device=device))
    return torch.cat(xs, 0)


# --------------------------------------------------------------------------- objective


class MaskedLSQReward:
    """R(x) = -0.5 * ||M * (x - Y)||^2 per sample (M binary entry mask, Y = M * x*_norm)."""

    def __init__(self, M: torch.Tensor, Y: torch.Tensor):
        self.set(M, Y)

    def set(self, M: torch.Tensor, Y: torch.Tensor):
        self.M = M
        self.Y = Y

    def __call__(self, x: torch.Tensor, c=None) -> torch.Tensor:
        if x.shape[0] != self.M.shape[0]:
            raise ValueError(f"batch mismatch: x {tuple(x.shape)} vs mask {tuple(self.M.shape)}")
        r = self.M * (x - self.Y)
        return -0.5 * (r**2).sum(dim=(1, 2))

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return -self(x)

    def grad_f_analytic(self, x: torch.Tensor) -> torch.Tensor:
        return self.M * (x - self.Y)


# --------------------------------------------------------------------------- model


def load_agents(protocol: dict, device: str = "cuda:0"):
    """agent20: diffusion_steps=20 (unguided / PDG / PnP-PGD); agent100: diffusion_steps=100, training 20 (DCG)."""
    m = protocol["model"]
    ckpt = protocol["checkpoint"]

    def _args(diffusion_steps, training_steps):
        return SimpleNamespace(
            model_dim=m["model_dim"], dim_mult=m["dim_mult"], ema_rate=m["ema_rate"], device=device,
            diffusion_steps=diffusion_steps, training_diffusion_steps=training_steps,
            predict_noise=m["predict_noise"], ckpt="latest", ckpt_path=ckpt,
        )

    save_path = Path(ckpt).parent
    agent20 = ue._load_agent(_args(20, 20), save_path, OBS_DIM, ACT_DIM, HORIZON, classifier=None)
    agent100 = ue._load_agent(_args(100, 20), save_path, OBS_DIM, ACT_DIM, HORIZON, classifier=None)
    return agent20, agent100


def _common_sample_kwargs(n: int, sample_steps: int, xT: torch.Tensor) -> dict:
    return dict(
        solver="ddim", n_samples=n, sample_steps=sample_steps, sample_step_schedule="uniform",
        use_ema=True, temperature=1.0, initial_xt=xT,
    )


def sample_unguided(agent20, prior, xT, sample_steps: int = 20) -> torch.Tensor:
    agent20.classifier = None
    x, _ = agent20.sample(
        prior, w_cg=0.0, guidance_mode="standard", optimization_guidance_scale=0.0,
        optimization_guidance_last_steps=10, **_common_sample_kwargs(prior.shape[0], sample_steps, xT),
    )
    return x.detach()


def sample_pdg(agent20, clf, prior, xT, w_cg: float) -> torch.Tensor:
    agent20.classifier = clf
    try:
        x, _ = agent20.sample(
            prior, w_cg=float(w_cg), guidance_mode="standard", optimization_guidance_scale=0.0,
            optimization_guidance_last_steps=10, **_common_sample_kwargs(prior.shape[0], 20, xT),
        )
    finally:
        agent20.classifier = None
    return x.detach()


def sample_dcg(agent100, clf, prior, xT, eta: float, last_steps: int = 10) -> torch.Tensor:
    agent100.classifier = clf
    try:
        x, _ = agent100.sample(
            prior, w_cg=0.0, guidance_mode="optimization", optimization_guidance_scale=float(eta),
            optimization_guidance_last_steps=int(last_steps), **_common_sample_kwargs(prior.shape[0], 20, xT),
        )
    finally:
        agent100.classifier = None
    return x.detach()


def run_pnp(agent20, clf, prior, x_init, noise_index: int, eta: float, K: int, metrics_fn=None,
            input_scaling: str = "vp"):
    D = FixedNoiseDenoiser.from_agent(agent20, int(noise_index), prior=prior, use_ema=True, input_scaling=input_scaling)
    grad_f = reward_classifier_grad_f(clf, agent20.fix_mask)
    res = pnp_pgd(D, grad_f, x_init, eta=float(eta), n_iters=int(K),
                  condition_fn=pin_fn(prior, agent20.fix_mask), metrics_fn=metrics_fn)
    return res, D


def build_prior(normalizer, o0_phys: np.ndarray, device: str) -> torch.Tensor:
    n = o0_phys.shape[0]
    prior = torch.zeros((n, HORIZON, X_DIM), device=device)
    prior[:, 0, :OBS_DIM] = torch.tensor(normalizer.normalize(o0_phys.astype(np.float32)), device=device)
    return prior


# --------------------------------------------------------------------------- metrics


def per_traj_metrics(x_hat_norm: np.ndarray, gt_obs: np.ndarray, gt_act: np.ndarray, obs_idx: np.ndarray,
                     M: np.ndarray, Y: np.ndarray, normalizer, env=None, with_dyn: bool = True) -> dict:
    """Per-trajectory metrics (arrays of shape (n,)). Physical units unless stated."""
    n = x_hat_norm.shape[0]
    obs_hat = normalizer.unnormalize(x_hat_norm[..., :OBS_DIM]).astype(np.float64)
    act_hat = ue._clip_actions(x_hat_norm[..., OBS_DIM:]).astype(np.float64)
    gt_obs = gt_obs.astype(np.float64)
    U = unobserved_time_mask(obs_idx)
    out = {k: np.zeros(n) for k in ("unobs_xy_mse", "unobs_obs4_mse", "action_mse", "obs_xy_rmse", "f_norm",
                                     "e_dyn", "e_dyn_norm", "pin_err_o0")}
    sq = (obs_hat - gt_obs) ** 2
    obs_std = ue._obs_std_vector(normalizer, OBS_DIM)
    for j in range(n):
        u = U[j]
        out["unobs_xy_mse"][j] = sq[j, u, :2].mean()
        out["unobs_obs4_mse"][j] = sq[j, u, :].mean()
        out["action_mse"][j] = ((act_hat[j] - gt_act[j]) ** 2).mean()
        out["obs_xy_rmse"][j] = np.sqrt(sq[j, obs_idx[j], :2].mean())
        r = M[j] * (x_hat_norm[j].astype(np.float64) - Y[j])
        out["f_norm"][j] = 0.5 * (r**2).sum()
        out["pin_err_o0"][j] = np.abs(obs_hat[j, 0] - gt_obs[j, 0]).max()
        if with_dyn:
            roll = ue._open_loop_rollout_obs(env, gt_obs[j, 0].astype(np.float32), act_hat[j].astype(np.float32), HORIZON)
            gaps = ue._feasibility_gaps(obs_hat[j].astype(np.float32), roll, obs_std)
            out["e_dyn"][j] = gaps["mean_l2_future"]
            out["e_dyn_norm"][j] = gaps["mean_l2_norm_future"]
        else:
            out["e_dyn"][j] = np.nan
            out["e_dyn_norm"][j] = np.nan
    return out


def make_env():
    return gym.make("Unicycle-v0", max_episode_steps=HORIZON, terminate_on_oob=False)


class TorchMetric:
    """Fast GPU per-iteration metrics (unobserved-xy MSE in physical units and f) for PnP histories."""

    def __init__(self, normalizer, gt_obs: np.ndarray, obs_idx: np.ndarray, rew: MaskedLSQReward, device):
        self.mean = torch.tensor(normalizer.mean[:2], dtype=torch.float64, device=device)
        self.std = torch.tensor(normalizer.std[:2], dtype=torch.float64, device=device)
        self.gt_xy = torch.tensor(gt_obs[..., :2], dtype=torch.float64, device=device)
        U = unobserved_time_mask(obs_idx)
        self.U = torch.tensor(U, dtype=torch.float64, device=device)
        self.nU = self.U.sum(1) * 2.0
        self.rew = rew

    def __call__(self, k, x):
        xy = x[..., :2].double() * self.std + self.mean
        mse = (((xy - self.gt_xy) ** 2).sum(-1) * self.U).sum(1) / self.nU
        f = self.rew.f(x).double()
        return {"unobs_xy_mse": mse.detach().cpu().numpy(), "f_norm": f.detach().cpu().numpy()}


def mean_std(a) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
