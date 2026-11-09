"""Shared machinery for the unicycle PnP-PGD rebuttal experiment.

Re-implements the per-candidate loop of ``unicycle_eval.run_plan_eval`` (ue:1395-1612)
without any of its file outputs, by calling the *same* functions from
``pipelines/unicycle_eval.py`` (``_load_agent``, ``_sample_plan``,
``_make_xy_tracking_reward_fn``, ``_tracking_reward_score``,
``_trajectory_norm_from_obs_act``, ``_open_loop_rollout_obs``,
``_feasibility_gaps``, ``_clip_actions``) and adds cost instrumentation and timing.

Protocol (identical to the paper runs): candidates sampled sequentially with
``n_samples=1``; candidate with seed ``s`` is preceded by ``set_seed(s)``; reward
temperature 5.0; DDIM / uniform schedule / temperature 1.0 / EMA weights.
Normalization uses the OLD dataset ``CleanDiffuser/results/unicycle_offline/
unicycle_offline.hdf5`` exactly as the paper (``ue`` default through repo_dir).
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

WORKTREE = Path("/u/rzhang26/CleanDiffuser_rebuttal_pnp")
for p in (WORKTREE / "pipelines", WORKTREE, WORKTREE / "experiments" / "rebuttal_pnp_2027"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "disable")

import gym  # noqa: E402

import unicycle_eval as ue  # noqa: E402
from cleandiffuser.classifier import RuntimeRewardClassifier  # noqa: E402
from cleandiffuser.dataset.unicycle_dataset import UnicycleDataset, load_unicycle_hdf5  # noqa: E402
from cleandiffuser.diffusion.guidance import validate_guidance_config  # noqa: E402
from cleandiffuser.diffusion.pnp_pgd import (  # noqa: E402
    FixedNoiseDenoiser,
    pin_fn,
    pnp_pgd,
    reward_classifier_grad_f,
)
from guidance_comparison_eval import build_configs  # noqa: E402
from reference_trajectories import get_reference_trajectory  # noqa: E402
from utils import set_seed  # noqa: E402

from rebuttal_pnp.instrument import GradCounter, NetworkCounter, sync  # noqa: E402

import time  # noqa: E402

ORIG = Path("/u/rzhang26/CleanDiffuser")
CKPT_PATH = ORIG / "results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt"
DATASET_PATH = ORIG / "results/unicycle_offline/unicycle_offline.hdf5"
PAPER_ROOT = ORIG / "results/unicycle_eval/reference_plan_paper_v1"
HORIZON = 64
REWARD_TEMPERATURE = 5.0
# grad f = (2*5/63) (x~ - y~) on xy dims t>=1  ->  Lipschitz constant L = 10/63.
LIPSCHITZ = 2.0 * REWARD_TEMPERATURE / (HORIZON - 1)

TEST_REFERENCES = ["circle", "random_smooth_1", "random_smooth_7"]
TEST_LABELS = {"circle": "Circle", "random_smooth_1": "Random Trajectory 1", "random_smooth_7": "Random Trajectory 2"}
VAL_REFERENCES = ["half_heart", "random_smooth_2", "random_smooth_4"]

# Existing methods: paper config names and agent settings (see run_reference_plan_paper_*.sbatch).
BASELINES = {
    "monte_carlo": dict(cfg_name="standard_w_cg0", agent="T20", paper_dir="standard_w_cg0", label="Monte Carlo"),
    "pdg": dict(cfg_name="standard_w_cg30", agent="T20", paper_dir="standard_w_cg30", label="PDG"),
    "dcg": dict(cfg_name="optimization_scale_30", agent="T100", paper_dir="optimization_scale_30", label="DCG"),
    "drgd": dict(cfg_name="drgd", agent="T20", paper_dir="drgd_gamma15", label="DRGD ($\\gamma=15$)"),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_args(diffusion_steps: int, training_diffusion_steps: int = 20, sampling_steps: int = 20, device: str = "cuda:0"):
    """Namespace with exactly the fields ue._load_agent / ue._sample_plan read (paper values)."""
    return SimpleNamespace(
        model_dim=32,
        dim_mult=[1, 2, 2],
        ema_rate=0.9999,
        device=device,
        diffusion_steps=diffusion_steps,
        training_diffusion_steps=training_diffusion_steps,
        predict_noise=False,
        ckpt="latest",
        ckpt_path=str(CKPT_PATH),
        solver="ddim",
        sampling_steps=sampling_steps,
        sample_step_schedule="uniform",
        use_ema=True,
        optimization_guidance_last_steps=10,
        temperature=1.0,
        seed=0,
        drgd_steps=20,
        drgd_gamma=15.0,
        drgd_noise_index=-1,
        reward_temperature=REWARD_TEMPERATURE,
    )


@dataclass
class Setup:
    device: str
    normalizer: object
    obs_dim: int
    act_dim: int
    obs_std: np.ndarray
    agents: Dict[str, object]
    args: Dict[str, SimpleNamespace]


def load_setup(device: str = "cuda:0") -> Setup:
    raw = load_unicycle_hdf5(str(DATASET_PATH))
    dataset = UnicycleDataset(raw, horizon=HORIZON)
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    args = {
        "T20": make_args(20, 20, 20, device),
        "T100": make_args(100, 20, 20, device),
        "T20_S10": make_args(20, 20, 10, device),  # budget-matched init (DDIM 10 steps on T=20)
    }
    agents = {
        "T20": ue._load_agent(args["T20"], Path("."), obs_dim, act_dim, HORIZON, classifier=None),
        "T100": ue._load_agent(args["T100"], Path("."), obs_dim, act_dim, HORIZON, classifier=None),
    }
    return Setup(device, normalizer, obs_dim, act_dim, ue._obs_std_vector(normalizer, obs_dim), agents, args)


@dataclass
class RefContext:
    name: str
    target_xy: np.ndarray
    init_state: tuple
    init_obs: np.ndarray
    env: object
    reward_fn: Callable
    classifier: RuntimeRewardClassifier
    prior: torch.Tensor


def make_ref(setup: Setup, name: str) -> RefContext:
    ref = get_reference_trajectory(name, num_steps=64)
    target_xy = ref.xy[:HORIZON].copy()
    env = gym.make("Unicycle-v0", max_episode_steps=HORIZON, terminate_on_oob=False)
    env.reset(options={"initial_state": ref.init_pose})
    init_obs = ue._current_obs(env)
    reward_fn = ue._make_xy_tracking_reward_fn(setup.normalizer, target_xy, temperature=REWARD_TEMPERATURE)
    prior = torch.zeros((1, HORIZON, setup.obs_dim + setup.act_dim), device=setup.device)
    # Fill the prior once (ue._sample_plan refills it identically before every sample).
    obs_norm = torch.tensor(setup.normalizer.normalize(init_obs[None, :]), device=setup.device, dtype=torch.float32)
    prior[:, 0, : setup.obs_dim] = obs_norm
    return RefContext(name, target_xy, tuple(ref.init_pose), init_obs, env,
                      reward_fn, RuntimeRewardClassifier(reward_fn, device=setup.device), prior)


def evaluate_plan(setup: Setup, ctx: RefContext, plan_norm: np.ndarray) -> dict:
    """Exactly the post-processing / metrics of ue.run_plan_eval for one candidate."""
    plan_norm = np.asarray(plan_norm, dtype=np.float32)
    planned_obs = setup.normalizer.unnormalize(plan_norm[:, : setup.obs_dim]).astype(np.float32)
    planned_act = ue._clip_actions(plan_norm[:, setup.obs_dim:])
    rollout_obs = ue._open_loop_rollout_obs(ctx.env, ctx.init_obs, planned_act, HORIZON)
    plan_reward = ue._tracking_reward_score(plan_norm, ctx.reward_fn, setup.device)
    rollout_norm = ue._trajectory_norm_from_obs_act(setup.normalizer, rollout_obs, planned_act, setup.obs_dim)
    rollout_reward = ue._tracking_reward_score(rollout_norm, ctx.reward_fn, setup.device)
    feas = ue._feasibility_gaps(planned_obs, rollout_obs, setup.obs_std)
    return {
        "plan_reward": plan_reward,
        "rollout_reward": rollout_reward,
        "feasibility_mean_l2_future": feas["mean_l2_future"],
        "feasibility_mean_l2_norm_future": feas["mean_l2_norm_future"],
        "planned_obs": planned_obs,
        "planned_act": planned_act,
        "planned_xy": planned_obs[:, :2],
        "rollout_xy": rollout_obs[:, :2],
        "plan_start_error": float(np.linalg.norm(planned_obs[0, :2] - ctx.init_obs[:2])),
    }


def baseline_cfg(cfg_name: str) -> dict:
    cfgs = build_configs(ue.DEFAULT_OPT_SCALES, include_monte_carlo=False, config_names=[cfg_name])
    assert len(cfgs) == 1
    cfg = cfgs[0]
    validate_guidance_config(cfg["guidance_mode"], cfg["w_cg"], cfg["optimization_guidance_scale"])
    return cfg


def _uses_classifier(cfg: dict) -> bool:
    # Same predicate as ue.run_plan_eval.
    return cfg["guidance_mode"] in ("standard", "optimization", "ggdm_g_loss", "drgd") and (
        cfg["w_cg"] != 0.0 or cfg["optimization_guidance_scale"] != 0.0 or cfg["guidance_mode"] in ("ggdm_g_loss", "drgd")
    )


class Counters:
    """Attach NFE / backward / grad counters for a timed region."""

    def __init__(self, setup: Setup, ctx: RefContext):
        self.nets = [NetworkCounter(a.model_ema["diffusion"]) for a in setup.agents.values()]
        self.grad = GradCounter(ctx.classifier)

    def __enter__(self):
        for n in self.nets:
            n.__enter__()
        self.grad.__enter__()
        return self

    def __exit__(self, *exc):
        self.grad.__exit__(*exc)
        for n in self.nets:
            n.__exit__(*exc)
        return False

    def as_dict(self) -> dict:
        return {
            "nfe": int(sum(n.forward_samples for n in self.nets)),
            "net_forward_calls": int(sum(n.forward_calls for n in self.nets)),
            "net_backward": int(sum(n.backward_samples for n in self.nets)),
            "grad_evals": int(self.grad.samples),
        }


def _timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def run_baseline_candidate(setup: Setup, ctx: RefContext, method: str, seed: int) -> dict:
    """One candidate of an existing method; returns plan_norm + costs + seconds (sampling only)."""
    spec = BASELINES[method]
    cfg = baseline_cfg(spec["cfg_name"])
    agent = setup.agents[spec["agent"]]
    args = setup.args[spec["agent"]]
    agent.classifier = ctx.classifier if _uses_classifier(cfg) else None
    set_seed(seed)
    with Counters(setup, ctx) as cnt:
        (planned_obs, planned_act, plan_norm), secs = _timed(
            lambda: ue._sample_plan(agent, setup.normalizer, ctx.init_obs, ctx.prior, args, cfg,
                                    setup.obs_dim, setup.act_dim, seed=seed)
        )
    return {"plan_norm": plan_norm, "costs": cnt.as_dict(), "seconds": secs, "cfg": cfg}


def unguided_init(setup: Setup, ctx: RefContext, seed: int, sampling_steps: int = 20) -> dict:
    """Monte Carlo plan for ``seed`` (exactly the monte_carlo baseline when sampling_steps=20)."""
    cfg = baseline_cfg("standard_w_cg0")
    agent = setup.agents["T20"]
    args = setup.args["T20"] if sampling_steps == 20 else setup.args[f"T20_S{sampling_steps}"]
    assert args.sampling_steps == sampling_steps
    agent.classifier = None
    set_seed(seed)
    with Counters(setup, ctx) as cnt:
        (_, _, plan_norm), secs = _timed(
            lambda: ue._sample_plan(agent, setup.normalizer, ctx.init_obs, ctx.prior, args, cfg,
                                    setup.obs_dim, setup.act_dim, seed=seed)
        )
    return {"plan_norm": plan_norm, "costs": cnt.as_dict(), "seconds": secs}


def gaussian_init(setup: Setup, ctx: RefContext, seed: int) -> dict:
    """The x_T that DDIM sampling of seed ``seed`` starts from (same draw as DiscreteDiffusionSDE.sample)."""
    agent = setup.agents["T20"]
    set_seed(seed)
    with Counters(setup, ctx) as cnt:
        def _draw():
            xt = torch.randn((1, *ctx.prior.shape[1:]), device=setup.device) * 1.0
            return xt * (1.0 - agent.fix_mask) + ctx.prior * agent.fix_mask
        xt, secs = _timed(_draw)
    return {"x": xt, "costs": cnt.as_dict(), "seconds": secs}


def make_denoiser(setup: Setup, ctx: RefContext, k: int, input_scaling: str = "vp") -> FixedNoiseDenoiser:
    return FixedNoiseDenoiser.from_agent(setup.agents["T20"], k, prior=ctx.prior, use_ema=True, input_scaling=input_scaling)


def run_pnp(setup: Setup, ctx: RefContext, x_init: torch.Tensor, *, k: int, eta: float, n_iters: int,
            input_scaling: str = "vp", record_history: bool = False) -> dict:
    agent = setup.agents["T20"]
    den = make_denoiser(setup, ctx, k, input_scaling)
    grad_f = reward_classifier_grad_f(ctx.classifier, agent.fix_mask)
    cond = pin_fn(ctx.prior, agent.fix_mask)
    with Counters(setup, ctx) as cnt:
        res, secs = _timed(lambda: pnp_pgd(den, grad_f, x_init, eta=eta, n_iters=n_iters,
                                           condition_fn=cond, record_history=record_history))
        final = res.x[0].detach().cpu().numpy().astype(np.float32)
    hist = None
    if record_history:
        hist = np.stack([h[0].detach().cpu().numpy() for h in res.history], 0).astype(np.float32)
    return {
        "plan_norm": final,
        "history": hist,
        "costs": cnt.as_dict(),
        "seconds": secs,
        "n_denoiser_evals": res.n_denoiser_evals,
        "n_grad_evals": res.n_grad_evals,
        "sigma_bar": res.sigma_bar,
        "model_t_seen": sorted(set(res.model_t_seen)),
    }


def to_device_plan(setup: Setup, plan_norm: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(plan_norm, dtype=np.float32)[None], device=setup.device)


def pack_candidates(ctx: RefContext, cands: List[dict]) -> dict:
    """Same keys / dtypes as ue._save_plan_trajectories."""
    return dict(
        reference_xy=ctx.target_xy.astype(np.float32),
        init_xy=ctx.init_obs[:2].astype(np.float32),
        init_obs=ctx.init_obs.astype(np.float32),
        planned_xy=np.stack([c["planned_xy"] for c in cands]).astype(np.float32),
        rollout_xy=np.stack([c["rollout_xy"] for c in cands]).astype(np.float32),
        planned_obs=np.stack([c["planned_obs"] for c in cands]).astype(np.float32),
        planned_act=np.stack([c["planned_act"] for c in cands]).astype(np.float32),
        candidate_idx=np.asarray([c["candidate_idx"] for c in cands], dtype=np.int32),
        seed=np.asarray([c["seed"] for c in cands], dtype=np.int32),
        plan_reward=np.asarray([c["plan_reward"] for c in cands], dtype=np.float64),
        rollout_reward=np.asarray([c["rollout_reward"] for c in cands], dtype=np.float64),
        feasibility_mean_l2_future=np.asarray([c["feasibility_mean_l2_future"] for c in cands], dtype=np.float64),
    )


def summarize(cands: List[dict]) -> dict:
    out = {}
    for key in ("plan_reward", "rollout_reward", "feasibility_mean_l2_future"):
        m, s = ue._mean_std([c[key] for c in cands])
        out[key] = {"mean": m, "std": s}
    m, s = ue._mean_std([-c["plan_reward"] for c in cands])
    out["c_plan"] = {"mean": m, "std": s}
    m, s = ue._mean_std([-c["rollout_reward"] for c in cands])
    out["c_rollout"] = {"mean": m, "std": s}
    out["e_dyn"] = dict(out["feasibility_mean_l2_future"])
    out["n"] = len(cands)
    return out


def static_config() -> dict:
    return {
        "checkpoint": str(CKPT_PATH),
        "checkpoint_sha256": sha256_file(CKPT_PATH),
        "normalizer_dataset": str(DATASET_PATH),
        "normalizer_dataset_sha256": sha256_file(DATASET_PATH),
        "horizon": HORIZON,
        "reward_temperature": REWARD_TEMPERATURE,
        "lipschitz_L": LIPSCHITZ,
        "solver": "ddim",
        "sample_step_schedule": "uniform",
        "temperature": 1.0,
        "use_ema": True,
        "n_samples_per_call": 1,
        "seeding": "set_seed(seed) (torch/numpy/random) immediately before each candidate",
        "agents": {
            "T20": vars(make_args(20, 20, 20)),
            "T100": vars(make_args(100, 20, 20)),
            "T20_S10": vars(make_args(20, 20, 10)),
        },
        "baselines": BASELINES,
        "env": {"id": "Unicycle-v0", "max_episode_steps": HORIZON, "terminate_on_oob": False},
        "metrics": {
            "c_plan": "-plan_reward (reward on unclipped normalized plan)",
            "c_rollout": "-rollout_reward (rollout of clipped planned actions from o_0)",
            "e_dyn": "feasibility_mean_l2_future: mean over t=1..63 of physical L2 over 4 obs dims",
            "std": "ddof=1",
        },
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
