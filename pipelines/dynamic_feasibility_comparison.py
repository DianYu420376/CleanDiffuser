"""Compare dynamic feasibility of guided diffusion plans vs open-loop sim rollouts.

For each of ``min_trajectories`` initial conditions:
  1. Sample one guided diffusion plan (horizon 32: obs + act).
  2. Open-loop rollout the planned actions in Hopper-v4 from that state.
  3. Compare the 32-step rollout observation sequence to the diffusion plan.

All guidance configs are evaluated on the *same* initial conditions for a fair,
paired comparison. This collects exactly ``min_trajectories`` plans per config
(no full-length MPC evaluation rollouts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
from omegaconf import OmegaConf

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import validate_guidance_config
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from d4rl_render_utils import env_reset, env_step, make_sim_eval_env, resolve_ckpt_stem
from guidance_comparison_eval import build_configs
from utils import set_seed

os.environ.setdefault("MUJOCO_GL", "disable")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _episode_starts(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    done = np.asarray(terminals, dtype=bool) | np.asarray(timeouts, dtype=bool)
    ends = np.where(done)[0]
    return np.concatenate([[0], ends[:-1] + 1]).astype(np.int64)


def _load_task_settings(repo_dir: Path, task: str) -> dict:
    task_yaml = repo_dir / "configs" / "diffuser" / "mujoco" / "task" / f"{task}.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"Missing task config: {task_yaml}")
    return OmegaConf.to_container(OmegaConf.load(task_yaml), resolve=True)


def _get_mujoco_state(env) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = env.unwrapped
    qpos = np.array(unwrapped.data.qpos, dtype=np.float64).copy()
    qvel = np.array(unwrapped.data.qvel, dtype=np.float64).copy()
    return qpos, qvel


def _set_mujoco_state(env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    env.unwrapped.set_state(
        np.asarray(qpos, dtype=np.float64),
        np.asarray(qvel, dtype=np.float64),
    )


def _current_obs(env) -> np.ndarray:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "_get_obs"):
        return np.asarray(unwrapped._get_obs(), dtype=np.float32)
    return env_reset(env)


def _load_agent(
    args,
    save_path: str,
    obs_dim: int,
    act_dim: int,
    horizon: int,
) -> DiscreteDiffusionSDE:
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )
    nn_classifier = HalfJannerUNet1d(
        horizon,
        obs_dim + act_dim,
        out_dim=1,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.dim_mult,
        timestep_emb_type="positional",
        kernel_size=3,
    )
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    fix_mask = torch.zeros((horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion,
        None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        training_diffusion_steps=args.training_diffusion_steps,
        predict_noise=args.predict_noise,
    )

    ckpt_stem = resolve_ckpt_stem(str(args.ckpt))
    agent.load(save_path + f"diffusion_ckpt_{ckpt_stem}.pt")
    agent.classifier.load(save_path + f"classifier_ckpt_{ckpt_stem}.pt")
    agent.eval()
    return agent


def _sample_initial_conditions(
    env,
    raw_dataset: dict,
    count: int,
    seed: int,
    init_mode: str,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    inits: list[dict] = []

    if init_mode == "dataset_episode_starts":
        qpos = raw_dataset["infos/qpos"]
        qvel = raw_dataset["infos/qvel"]
        starts = _episode_starts(raw_dataset["terminals"], raw_dataset["timeouts"])
        pick = rng.choice(starts, size=count, replace=count > len(starts))
        for traj_idx, start in enumerate(pick):
            env_reset(env)
            _set_mujoco_state(env, qpos[start], qvel[start])
            inits.append(
                {
                    "traj_idx": traj_idx,
                    "init_mode": init_mode,
                    "dataset_index": int(start),
                    "qpos": qpos[start].astype(np.float64),
                    "qvel": qvel[start].astype(np.float64),
                    "obs": _current_obs(env),
                }
            )
        return inits

    if init_mode == "env_reset":
        for traj_idx in range(count):
            set_seed(seed + traj_idx)
            env_reset(env)
            qpos, qvel = _get_mujoco_state(env)
            inits.append(
                {
                    "traj_idx": traj_idx,
                    "init_mode": init_mode,
                    "dataset_index": None,
                    "qpos": qpos,
                    "qvel": qvel,
                    "obs": _current_obs(env),
                }
            )
        return inits

    raise ValueError(f"Unknown init_mode: {init_mode}")


def _sample_best_plan(
    agent: DiscreteDiffusionSDE,
    normalizer,
    obs: np.ndarray,
    prior: torch.Tensor,
    args,
    config: dict,
    obs_dim: int,
    act_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
    prior.zero_()
    prior[:, 0, :obs_dim] = obs_norm

    traj, log = agent.sample(
        prior,
        solver=args.solver,
        n_samples=args.num_candidates,
        sample_steps=args.sampling_steps,
        sample_step_schedule=args.sample_step_schedule,
        use_ema=args.use_ema,
        w_cg=config["w_cg"],
        guidance_mode=config["guidance_mode"],
        optimization_guidance_scale=config["optimization_guidance_scale"],
        optimization_guidance_last_steps=args.optimization_guidance_last_steps,
        temperature=args.temperature,
    )

    logp = log["log_p"].view(args.num_candidates, 1, -1).sum(-1)
    idx = int(logp.argmax(0).item())
    best = traj.view(args.num_candidates, 1, args.horizon, -1)[idx, 0].detach().cpu().numpy()

    planned_obs = normalizer.unnormalize(best[:, :obs_dim])
    planned_act = np.clip(best[:, obs_dim:], -1.0, 1.0)
    return planned_obs.astype(np.float32), planned_act.astype(np.float32)


def _open_loop_rollout_obs(
    env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    obs_dim: int,
) -> np.ndarray:
    env_reset(env)
    _set_mujoco_state(env, qpos, qvel)
    rollout_obs = np.zeros((horizon, obs_dim), dtype=np.float32)

    for t in range(horizon):
        rollout_obs[t] = _current_obs(env)[:obs_dim]
        if t >= horizon - 1:
            break
        _, _, done, _ = env_step(env, actions[t])
        if done:
            rollout_obs[t + 1 :] = rollout_obs[t]
            break

    return rollout_obs


def _obs_std_vector(normalizer, obs_dim: int) -> np.ndarray:
    """Per-dimension Gaussian normalizer std used for z-score feasibility gaps."""
    std = np.asarray(normalizer.std, dtype=np.float64).reshape(-1)
    if std.size < obs_dim:
        raise ValueError(f"Normalizer std has {std.size} dims, expected {obs_dim}")
    std = std[:obs_dim].copy()
    std[std == 0] = 1.0
    return std


def _feasibility_gaps(
    planned_obs: np.ndarray,
    rollout_obs: np.ndarray,
    obs_std: np.ndarray,
    *,
    store_per_step: bool,
) -> dict[str, float | list[float]]:
    delta = planned_obs - rollout_obs
    per_step = np.linalg.norm(delta, axis=1)
    per_step_norm = np.linalg.norm(delta / obs_std, axis=1)
    future = per_step[1:] if per_step.size > 1 else per_step[:0]
    future_norm = per_step_norm[1:] if per_step_norm.size > 1 else per_step_norm[:0]
    out: dict[str, float | list[float]] = {
        "mean_l2_all": float(per_step.mean()),
        "std_l2_all": float(per_step.std(ddof=0)),
        "mean_l2_future": float(future.mean()) if future.size else 0.0,
        "std_l2_future": float(future.std(ddof=0)) if future.size else 0.0,
        "max_l2": float(per_step.max()),
        "final_l2": float(per_step[-1]),
        "mean_l2_norm_all": float(per_step_norm.mean()),
        "std_l2_norm_all": float(per_step_norm.std(ddof=0)),
        "mean_l2_norm_future": float(future_norm.mean()) if future_norm.size else 0.0,
        "std_l2_norm_future": float(future_norm.std(ddof=0)) if future_norm.size else 0.0,
        "max_l2_norm": float(per_step_norm.max()),
        "final_l2_norm": float(per_step_norm[-1]),
    }
    if store_per_step:
        out["per_step_l2"] = per_step.astype(float).tolist()
        out["per_step_l2_norm"] = per_step_norm.astype(float).tolist()
    return out


def _collect_paired_trajectories(
    env,
    agent: DiscreteDiffusionSDE,
    normalizer,
    obs_std: np.ndarray,
    raw_dataset: dict,
    args,
    configs: list[dict],
    obs_dim: int,
    act_dim: int,
) -> dict[str, list[dict]]:
    inits = _sample_initial_conditions(
        env,
        raw_dataset,
        count=args.min_trajectories,
        seed=args.seed,
        init_mode=args.init_mode,
    )

    prior = torch.zeros(
        (args.num_candidates, args.horizon, obs_dim + act_dim),
        device=args.device,
    )
    samples: dict[str, list[dict]] = {cfg["name"]: [] for cfg in configs}

    for init in inits:
        qpos = init["qpos"]
        qvel = init["qvel"]
        obs = init["obs"]

        for config in configs:
            validate_guidance_config(
                str(config["guidance_mode"]),
                float(config["w_cg"]),
                float(config["optimization_guidance_scale"]),
            )
            planned_obs, planned_act = _sample_best_plan(
                agent,
                normalizer,
                obs,
                prior,
                args,
                config,
                obs_dim,
                act_dim,
            )
            rollout_obs = _open_loop_rollout_obs(
                env, qpos, qvel, planned_act, args.horizon, obs_dim
            )
            gaps = _feasibility_gaps(
                planned_obs,
                rollout_obs,
                obs_std,
                store_per_step=args.store_per_step,
            )
            samples[config["name"]].append(
                {
                    "traj_idx": init["traj_idx"],
                    "init_mode": init["init_mode"],
                    "dataset_index": init["dataset_index"],
                    "config": config["name"],
                    **gaps,
                }
            )

        if (init["traj_idx"] + 1) % 10 == 0 or init["traj_idx"] + 1 == args.min_trajectories:
            _log(f"[progress] {init['traj_idx'] + 1}/{args.min_trajectories} initial conditions done")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="/u/rzhang26/CleanDiffuser")
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--run-suffix", default="medium_expert_v4_reroll")
    parser.add_argument(
        "--dataset-h5path",
        default="/u/rzhang26/.d4rl/datasets/hopper_medium_expert_v4_reroll.hdf5",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-trajectories", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--init-mode",
        choices=("dataset_episode_starts", "env_reset"),
        default="dataset_episode_starts",
        help="How to pick the 100 initial conditions.",
    )
    parser.add_argument(
        "--configs",
        default="monte_carlo_w_cg0,standard_w_cg0p3,optimization_scale_0p1,optimization_scale_0p25",
        help="Comma-separated guidance config names.",
    )
    parser.add_argument(
        "--opt-scales",
        default="0.05,0.1,0.25,0.5",
        help="Optimization scales used to resolve optimization_scale_* config names.",
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--dim-mult", default=None)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument(
        "--training-diffusion-steps",
        type=int,
        default=20,
        help="Timestep range the checkpoint was trained on (maps to model/classifier t).",
    )
    parser.add_argument("--sample-step-schedule", default="uniform")
    parser.add_argument(
        "--optimization-guidance-last-steps",
        type=int,
        default=None,
        help="Apply optimization on the last N reverse steps (default: sampling_steps // 2).",
    )
    parser.add_argument("--solver", default="ddpm")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-rate", type=float, default=0.9999)
    parser.add_argument("--action-loss-weight", type=float, default=10.0)
    parser.add_argument("--terminal-penalty", type=float, default=-100.0)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument(
        "--store-per-step",
        action="store_true",
        help="Include per-step L2 gaps in summary.json (off by default to keep output small).",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    task_settings = _load_task_settings(repo_dir, args.task)
    if args.horizon is None:
        args.horizon = int(task_settings["horizon"])
    if args.dim_mult is None:
        args.dim_mult = task_settings["dim_mult"]
    elif isinstance(args.dim_mult, str):
        args.dim_mult = [int(x) for x in args.dim_mult.split(",") if x.strip()]
    else:
        args.dim_mult = list(args.dim_mult)

    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.optimization_guidance_last_steps is None:
        args.optimization_guidance_last_steps = args.sampling_steps // 2
    set_seed(args.seed)

    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    include_monte_carlo = "monte_carlo_w_cg0" in config_names
    opt_scales = [float(x.strip()) for x in args.opt_scales.split(",") if x.strip()]
    configs = build_configs(
        opt_scales=opt_scales,
        include_monte_carlo=include_monte_carlo,
        config_names=config_names,
    )

    save_path = f"results/diffuser_d4rl_mujoco/{args.task}/"
    if args.run_suffix:
        save_path = f"{save_path}{args.run_suffix}/"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.repo_dir) / save_path / "dynamic_feasibility" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("============================================================")
    _log("Dynamic feasibility comparison")
    _log("============================================================")
    _log(f"task={args.task} ckpt={args.ckpt} save_path={save_path}")
    _log(f"horizon={args.horizon} dim_mult={args.dim_mult}")
    _log(
        f"diffusion_steps={args.diffusion_steps} sampling_steps={args.sampling_steps} "
        f"training_diffusion_steps={args.training_diffusion_steps} "
        f"opt_guidance_last_steps={args.optimization_guidance_last_steps}"
    )
    _log(f"configs={[c['name'] for c in configs]}")
    _log(
        f"min_trajectories={args.min_trajectories} init_mode={args.init_mode} "
        f"seed={args.seed}"
    )
    _log(f"output_dir={output_dir}")

    env_data = gym.make(args.task)
    raw_dataset = env_data.get_dataset(h5path=str(args.dataset_h5path))
    env_data.close()

    if args.init_mode == "dataset_episode_starts":
        for key in ("infos/qpos", "infos/qvel"):
            if key not in raw_dataset:
                raise KeyError(
                    f"{key} missing from dataset; use --init-mode env_reset instead."
                )

    dataset = D4RLMuJoCoDataset(
        raw_dataset,
        horizon=args.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()
    obs_std = _obs_std_vector(normalizer, obs_dim)
    _log(
        "obs normalizer std (dataset z-score): "
        + ", ".join(f"{v:.4f}" for v in obs_std.tolist())
    )

    agent = _load_agent(args, save_path, obs_dim, act_dim, args.horizon)
    env_eval, sim_name = make_sim_eval_env(
        args.task,
        sim_env_name=None,
        render=False,
        ignore_termination=False,
    )
    _log(f"rollout sim env: {sim_name}")

    all_samples = _collect_paired_trajectories(
        env_eval,
        agent,
        normalizer,
        obs_std,
        raw_dataset,
        args,
        configs,
        obs_dim,
        act_dim,
    )
    env_eval.close()

    final_summary = {"configs": {}}
    _log("\n========== Final dynamic feasibility summary ==========")
    for config in configs:
        name = config["name"]
        samples = all_samples[name]
        mean_all = [s["mean_l2_all"] for s in samples]
        mean_future = [s["mean_l2_future"] for s in samples]
        mean_norm_all = [s["mean_l2_norm_all"] for s in samples]
        mean_norm_future = [s["mean_l2_norm_future"] for s in samples]
        max_l2 = [s["max_l2"] for s in samples]
        final_l2 = [s["final_l2"] for s in samples]
        max_l2_norm = [s["max_l2_norm"] for s in samples]
        final_l2_norm = [s["final_l2_norm"] for s in samples]

        m_all, s_all = _mean_std(mean_all)
        m_future, s_future = _mean_std(mean_future)
        m_norm_all, s_norm_all = _mean_std(mean_norm_all)
        m_norm_future, s_norm_future = _mean_std(mean_norm_future)
        m_max, s_max = _mean_std(max_l2)
        m_final, s_final = _mean_std(final_l2)
        m_max_norm, s_max_norm = _mean_std(max_l2_norm)
        m_final_norm, s_final_norm = _mean_std(final_l2_norm)

        final_summary["configs"][name] = {
            "n_trajectories": len(samples),
            "mean_gap_mean_l2_all": m_all,
            "std_gap_mean_l2_all": s_all,
            "mean_gap_mean_l2_future": m_future,
            "std_gap_mean_l2_future": s_future,
            "mean_gap_mean_l2_norm_all": m_norm_all,
            "std_gap_mean_l2_norm_all": s_norm_all,
            "mean_gap_mean_l2_norm_future": m_norm_future,
            "std_gap_mean_l2_norm_future": s_norm_future,
            "mean_gap_max_l2": m_max,
            "std_gap_max_l2": s_max,
            "mean_gap_final_l2": m_final,
            "std_gap_final_l2": s_final,
            "mean_gap_max_l2_norm": m_max_norm,
            "std_gap_max_l2_norm": s_max_norm,
            "mean_gap_final_l2_norm": m_final_norm,
            "std_gap_final_l2_norm": s_final_norm,
            "guidance_mode": config["guidance_mode"],
            "w_cg": config["w_cg"],
            "optimization_guidance_scale": config["optimization_guidance_scale"],
        }

        _log(
            f"{name:28s} n={len(samples):4d}  "
            f"mean_l2_future={m_future:7.4f} ± {s_future:6.4f}  "
            f"mean_l2_norm_future={m_norm_future:7.4f} ± {s_norm_future:6.4f}"
        )

    payload = {
        "task": args.task,
        "ckpt": args.ckpt,
        "run_suffix": args.run_suffix,
        "dataset_h5path": args.dataset_h5path,
        "sim_env": sim_name,
        "horizon": args.horizon,
        "min_trajectories": args.min_trajectories,
        "init_mode": args.init_mode,
        "seed": args.seed,
        "solver": args.solver,
        "diffusion_steps": args.diffusion_steps,
        "sampling_steps": args.sampling_steps,
        "training_diffusion_steps": args.training_diffusion_steps,
        "optimization_guidance_last_steps": args.optimization_guidance_last_steps,
        "sample_step_schedule": args.sample_step_schedule,
        "num_candidates": args.num_candidates,
        "configs": configs,
        "samples": all_samples,
        "final_summary": final_summary,
        "output_dir": str(output_dir),
        "obs_normalizer_std": obs_std.astype(float).tolist(),
        "method": (
            "For each initial condition: sample one horizon-length guided plan, "
            "open-loop rollout planned actions for 32 steps, compare obs sequences. "
            "Physical L2 uses raw simulator observations; normalized L2 divides "
            "per-dim errors by the training dataset Gaussian std (z-score units)."
        ),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"Saved {summary_path}")

    if any(len(all_samples[c["name"]]) < args.min_trajectories for c in configs):
        missing = [c["name"] for c in configs if len(all_samples[c["name"]]) < args.min_trajectories]
        _log(f"[warn] Configs below min_trajectories: {missing}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
