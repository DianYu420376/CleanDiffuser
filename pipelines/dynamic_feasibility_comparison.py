"""Compare dynamic feasibility of guided diffusion plans vs open-loop sim rollouts.

For each MPC replanning step during evaluation:
  1. Sample a guided diffusion trajectory (obs + act over the planning horizon).
  2. Open-loop rollout the same action sequence in Hopper-v4 from the current sim state.
  3. Compare the rollout observation sequence to the diffusion plan.

Collects at least ``min_trajectories`` samples per guidance config and reports
mean/std of the dynamic feasibility gap (L2 in observation space).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import d4rl  # noqa: F401
import gym
import numpy as np
import torch

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import validate_guidance_config
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from d4rl_render_utils import env_reset, env_step, make_sim_eval_env, resolve_ckpt_stem
from guidance_comparison_eval import STANDARD_CONFIG, build_configs
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
        predict_noise=args.predict_noise,
    )

    ckpt_stem = resolve_ckpt_stem(str(args.ckpt))
    agent.load(save_path + f"diffusion_ckpt_{ckpt_stem}.pt")
    agent.classifier.load(save_path + f"classifier_ckpt_{ckpt_stem}.pt")
    agent.eval()
    return agent


def _sample_best_plan(
    agent: DiscreteDiffusionSDE,
    normalizer,
    obs: np.ndarray,
    args,
    config: dict,
    obs_dim: int,
    act_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    prior = torch.zeros((1, args.horizon, obs_dim + act_dim), device=args.device)
    obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
    prior[:, 0, :obs_dim] = obs_norm

    traj, log = agent.sample(
        prior.repeat(args.num_candidates, 1, 1),
        solver=args.solver,
        n_samples=args.num_candidates,
        sample_steps=args.sampling_steps,
        use_ema=args.use_ema,
        w_cg=config["w_cg"],
        guidance_mode=config["guidance_mode"],
        optimization_guidance_scale=config["optimization_guidance_scale"],
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
    _set_mujoco_state(env, qpos, qvel)
    rollout_obs = np.zeros((horizon, obs_dim), dtype=np.float32)

    for t in range(horizon):
        rollout_obs[t] = _current_obs(env)[:obs_dim]
        if t >= horizon - 1:
            break
        _, _, done, _ = env_step(env, actions[t])
        if done:
            if t < horizon - 1:
                rollout_obs[t + 1 :] = rollout_obs[t]
            break

    return rollout_obs


def _feasibility_gaps(planned_obs: np.ndarray, rollout_obs: np.ndarray) -> dict[str, float]:
    per_step = np.linalg.norm(planned_obs - rollout_obs, axis=1)
    future = per_step[1:] if per_step.size > 1 else per_step[:0]
    return {
        "mean_l2_all": float(per_step.mean()),
        "std_l2_all": float(per_step.std(ddof=0)),
        "mean_l2_future": float(future.mean()) if future.size else 0.0,
        "std_l2_future": float(future.std(ddof=0)) if future.size else 0.0,
        "max_l2": float(per_step.max()),
        "final_l2": float(per_step[-1]),
        "per_step_l2": per_step.astype(float).tolist(),
    }


def _collect_config_trajectories(
    env,
    agent: DiscreteDiffusionSDE,
    normalizer,
    args,
    config: dict,
    obs_dim: int,
    act_dim: int,
    seed: int,
    target_count: int,
) -> list[dict]:
    set_seed(seed)
    samples: list[dict] = []
    episode_idx = 0

    while len(samples) < target_count:
        obs = env_reset(env)
        ep_steps = 0
        ep_sample_idx = 0

        while ep_steps < args.max_steps and len(samples) < target_count:
            qpos, qvel = _get_mujoco_state(env)
            planned_obs, planned_act = _sample_best_plan(
                agent, normalizer, obs, args, config, obs_dim, act_dim
            )
            rollout_obs = _open_loop_rollout_obs(
                env, qpos, qvel, planned_act, args.horizon, obs_dim
            )
            gaps = _feasibility_gaps(planned_obs, rollout_obs)

            samples.append(
                {
                    "seed": seed,
                    "episode": episode_idx,
                    "replan_step": ep_steps,
                    "sample_idx": len(samples),
                    "config": config["name"],
                    **gaps,
                }
            )
            ep_sample_idx += 1

            act0 = planned_act[0]
            obs, _, done, _ = env_step(env, act0)
            ep_steps += 1
            if done:
                break

        episode_idx += 1
        if episode_idx > args.max_episodes_per_seed:
            _log(
                f"[warn] seed={seed} config={config['name']}: "
                f"stopped after {episode_idx} episodes with only {len(samples)} samples"
            )
            break

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
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--min-trajectories", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-episodes-per-seed", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--configs",
        default="standard_w_cg0p3,optimization_scale_0p1,optimization_scale_0p25",
        help="Comma-separated guidance config names.",
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--dim-mult", default="1,2,4")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--solver", default="ddpm")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-rate", type=float, default=0.9999)
    parser.add_argument("--action-loss-weight", type=float, default=10.0)
    parser.add_argument("--terminal-penalty", type=float, default=-100.0)
    parser.add_argument("--discount", type=float, default=0.997)
    args = parser.parse_args()

    args.device = args.device if torch.cuda.is_available() else "cpu"
    args.dim_mult = [int(x) for x in str(args.dim_mult).split(",") if x.strip()]

    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    configs = build_configs(
        opt_scales=[0.1, 0.25],
        config_names=config_names,
    )

    save_path = f"results/diffuser_d4rl_mujoco/{args.task}/"
    if args.run_suffix:
        save_path = f"{save_path}{args.run_suffix}/"

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.repo_dir) / save_path / "dynamic_feasibility" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("============================================================")
    _log("Dynamic feasibility comparison")
    _log("============================================================")
    _log(f"task={args.task} ckpt={args.ckpt} save_path={save_path}")
    _log(f"configs={[c['name'] for c in configs]}")
    _log(f"min_trajectories={args.min_trajectories} seeds={args.seed_start}..{args.seed_end}")
    _log(f"output_dir={output_dir}")

    env_data = gym.make(args.task)
    raw_dataset = env_data.get_dataset(h5path=str(args.dataset_h5path))
    env_data.close()

    dataset = D4RLMuJoCoDataset(
        raw_dataset,
        horizon=args.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()

    agent = _load_agent(args, save_path, obs_dim, act_dim, args.horizon)
    env_eval, sim_name = make_sim_eval_env(
        args.task,
        sim_env_name=None,
        render=False,
        ignore_termination=False,
    )
    _log(f"rollout sim env: {sim_name}")

    all_samples: dict[str, list[dict]] = {cfg["name"]: [] for cfg in configs}
    per_seed_summary: list[dict] = []

    for seed in range(args.seed_start, args.seed_end + 1):
        seed_entry = {"seed": seed, "configs": {}}
        _log(f"\n========== Seed {seed} ==========")

        for config in configs:
            name = config["name"]
            remaining = max(0, args.min_trajectories - len(all_samples[name]))
            if remaining == 0:
                _log(f"[skip] config={name} already has {len(all_samples[name])} samples")
                seed_entry["configs"][name] = {"n_new": 0, "n_total": len(all_samples[name])}
                continue

            validate_guidance_config(
                str(config["guidance_mode"]),
                float(config["w_cg"]),
                float(config["optimization_guidance_scale"]),
            )
            _log(f"[collect] seed={seed} config={name} need={remaining}")
            new_samples = _collect_config_trajectories(
                env_eval,
                agent,
                normalizer,
                args,
                config,
                obs_dim,
                act_dim,
                seed=seed,
                target_count=remaining,
            )
            all_samples[name].extend(new_samples)
            seed_entry["configs"][name] = {
                "n_new": len(new_samples),
                "n_total": len(all_samples[name]),
            }
            _log(f"  collected {len(new_samples)} -> total {len(all_samples[name])}")

        per_seed_summary.append(seed_entry)

        if all(len(all_samples[c["name"]]) >= args.min_trajectories for c in configs):
            _log("Reached min_trajectories for all configs; stopping early.")
            break

    env_eval.close()

    final_summary = {"configs": {}}
    _log("\n========== Final dynamic feasibility summary ==========")
    for config in configs:
        name = config["name"]
        samples = all_samples[name]
        mean_all = [s["mean_l2_all"] for s in samples]
        std_all = [s["std_l2_all"] for s in samples]
        mean_future = [s["mean_l2_future"] for s in samples]
        max_l2 = [s["max_l2"] for s in samples]
        final_l2 = [s["final_l2"] for s in samples]

        m_all, s_all = _mean_std(mean_all)
        m_future, s_future = _mean_std(mean_future)
        m_max, s_max = _mean_std(max_l2)
        m_final, s_final = _mean_std(final_l2)

        final_summary["configs"][name] = {
            "n_trajectories": len(samples),
            "mean_gap_mean_l2_all": m_all,
            "std_gap_mean_l2_all": s_all,
            "mean_gap_mean_l2_future": m_future,
            "std_gap_mean_l2_future": s_future,
            "mean_gap_max_l2": m_max,
            "std_gap_max_l2": s_max,
            "mean_gap_final_l2": m_final,
            "std_gap_final_l2": s_final,
            "guidance_mode": config["guidance_mode"],
            "w_cg": config["w_cg"],
            "optimization_guidance_scale": config["optimization_guidance_scale"],
        }

        _log(
            f"{name:28s} n={len(samples):4d}  "
            f"mean_l2_all={m_all:7.4f} ± {s_all:6.4f}  "
            f"mean_l2_future={m_future:7.4f} ± {s_future:6.4f}"
        )

    payload = {
        "task": args.task,
        "ckpt": args.ckpt,
        "run_suffix": args.run_suffix,
        "dataset_h5path": args.dataset_h5path,
        "sim_env": sim_name,
        "horizon": args.horizon,
        "min_trajectories": args.min_trajectories,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "num_candidates": args.num_candidates,
        "configs": configs,
        "per_seed_summary": per_seed_summary,
        "samples": all_samples,
        "final_summary": final_summary,
        "output_dir": str(output_dir),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"Saved {summary_path}")

    missing = [
        c["name"]
        for c in configs
        if len(all_samples[c["name"]]) < args.min_trajectories
    ]
    if missing:
        _log(
            f"[warn] Configs below min_trajectories ({args.min_trajectories}): {missing}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
