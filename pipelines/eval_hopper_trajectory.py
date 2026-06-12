"""Non-rendering rollout eval with hop survival / fall trajectory analysis."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import d4rl  # noqa: F401
import gym
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d4rl_render_utils import env_reset, env_step, is_offline_d4rl_env, make_sim_eval_env
from diffuser_d4rl_mujoco import _load_checkpoints  # noqa: E402
from utils import set_seed

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d


def _hopper_height(obs: np.ndarray) -> float:
    return float(obs[0])


def _hopper_angle(obs: np.ndarray) -> float:
    return float(obs[1])


def _is_healthy(height: float, angle: float, z_min: float = 0.7, angle_max: float = 0.2) -> bool:
    return height > z_min and abs(angle) < angle_max


def _log(msg: str) -> None:
    print(msg, flush=True)


def _classify_episode(survival_steps: int, fell: bool, long_hop_threshold: int = 200) -> str:
    if not fell and survival_steps >= long_hop_threshold:
        return "sustained_hop"
    if fell and survival_steps < 30:
        return "immediate_fall"
    if fell and survival_steps < long_hop_threshold:
        return "short_hop_then_fall"
    if not fell:
        return "no_termination_within_limit"
    return "other"


def _select_action(agent, prior, obs_norm, args, obs_dim, act_dim):
    prior[:, 0, :obs_dim] = obs_norm
    t0 = time.perf_counter()
    traj, log = agent.sample(
        prior.repeat(args.num_candidates, 1, 1),
        solver=args.solver,
        n_samples=args.num_candidates,
        sample_steps=args.sampling_steps,
        use_ema=args.use_ema,
        w_cg=args.task.w_cg,
        guidance_mode=args.guidance_mode,
        optimization_guidance_scale=args.optimization_guidance_scale,
        temperature=args.temperature,
    )
    sample_s = time.perf_counter() - t0

    logp = log["log_p"].view(args.num_candidates, 1, -1).sum(-1)
    idx = logp.argmax(0)
    act = traj.view(args.num_candidates, 1, args.task.horizon, -1)[idx, 0, 0, obs_dim:]
    act = act.clip(-1.0, 1.0).cpu().numpy()
    return act, sample_s, float(logp[idx, 0])


def benchmark_sample(agent, normalizer, args, obs_dim, act_dim, env_eval):
    """Time one diffusion sample to estimate rollout wall-clock."""
    prior = torch.zeros((1, args.task.horizon, obs_dim + act_dim), device=args.device)
    obs = env_reset(env_eval)
    obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    _, sample_s, _ = _select_action(agent, prior, obs_norm, args, obs_dim, act_dim)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    return sample_s


def rollout_with_trajectory(
    env_eval,
    agent,
    normalizer,
    args,
    obs_dim,
    act_dim,
    max_steps: int = 1000,
    episode_idx: int = 0,
    log_interval: int = 5,
    expected_sample_s: float | None = None,
):
    prior = torch.zeros((1, args.task.horizon, obs_dim + act_dim), device=args.device)
    obs = env_reset(env_eval)
    ep_reward = 0.0
    heights = []
    angles = []
    rewards = []
    done_steps = []
    fell = False
    survival_steps = 0
    ep_start = time.perf_counter()
    total_sample_s = 0.0
    total_env_s = 0.0

    _log(
        f"[episode {episode_idx}] start "
        f"(max_steps={max_steps}, num_candidates={args.num_candidates}, "
        f"sampling_steps={args.sampling_steps})"
    )

    for step in range(1, max_steps + 1):
        height = _hopper_height(obs)
        angle = _hopper_angle(obs)
        heights.append(height)
        angles.append(angle)

        obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
        act, sample_s, logp = _select_action(agent, prior, obs_norm, args, obs_dim, act_dim)
        total_sample_s += sample_s

        t0 = time.perf_counter()
        obs, rew, done, _ = env_step(env_eval, act)
        total_env_s += time.perf_counter() - t0

        ep_reward += rew
        rewards.append(float(rew))
        survival_steps = step

        if step == 1 or step % log_interval == 0 or done:
            elapsed = time.perf_counter() - ep_start
            est_total = None
            if expected_sample_s is not None:
                est_total = expected_sample_s * max_steps + total_env_s * (max_steps / step)
            _log(
                f"[episode {episode_idx} step {step}/{max_steps}] "
                f"z={height:.3f} angle={angle:.3f} rew={rew:.2f} done={done} "
                f"sample={sample_s:.2f}s env={total_env_s:.4f}s "
                f"elapsed={elapsed:.1f}s"
                + (f" est_full_ep~{est_total/60:.1f}min" if est_total else "")
            )

        if done:
            fell = True
            done_steps.append(step)
            _log(f"[episode {episode_idx}] terminated at step {step}")
            break

    ep_elapsed = time.perf_counter() - ep_start
    _log(
        f"[episode {episode_idx}] finished in {ep_elapsed:.1f}s "
        f"(steps={survival_steps}, sample_total={total_sample_s:.1f}s, env_total={total_env_s:.3f}s)"
    )

    alive_heights = heights[:survival_steps]
    alive_angles = angles[:survival_steps]

    return {
        "total_reward": float(ep_reward),
        "survival_steps": survival_steps,
        "fell": fell,
        "first_fall_step": done_steps[0] if done_steps else None,
        "min_height": float(min(alive_heights)) if alive_heights else None,
        "max_height": float(max(alive_heights)) if alive_heights else None,
        "mean_height": float(np.mean(alive_heights)) if alive_heights else None,
        "final_height": float(heights[-1]),
        "max_abs_angle": float(max(abs(a) for a in alive_angles)) if alive_angles else None,
        "healthy_fraction": float(np.mean([_is_healthy(h, a) for h, a in zip(alive_heights, alive_angles)])),
        "classification": _classify_episode(survival_steps, fell),
        "height_trace": [float(h) for h in heights[:: max(1, len(heights) // 50)]],
    }


def build_agent(args, obs_dim, act_dim):
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon,
        obs_dim + act_dim,
        out_dim=1,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="positional",
        kernel_size=3,
    )
    classifier = CumRewClassifier(nn_classifier, device=args.device)
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
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
    return agent


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/diffuser/mujoco/mujoco.yaml")
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--guidance_mode", default=None, choices=["standard", "optimization"])
    parser.add_argument("--optimization_guidance_scale", type=float, default=None)
    parser.add_argument("--w_cg", type=float, default=None)
    args_cli = parser.parse_args()

    base = OmegaConf.load(Path(__file__).resolve().parent / args_cli.config)
    task_cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / f"configs/diffuser/mujoco/task/{args_cli.task}.yaml"
    )
    args = OmegaConf.merge(base, {"task": task_cfg, "mode": "inference", "ckpt": args_cli.ckpt})
    if args_cli.guidance_mode is not None:
        args.guidance_mode = args_cli.guidance_mode
    if args_cli.optimization_guidance_scale is not None:
        args.optimization_guidance_scale = args_cli.optimization_guidance_scale
    if args_cli.w_cg is not None:
        args.task.w_cg = args_cli.w_cg
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args_cli.seed)

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}/"
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(),
        horizon=args.task.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    agent = build_agent(args, obs_dim, act_dim)
    _load_checkpoints(agent, save_path, args.ckpt)
    agent.eval()
    normalizer = dataset.get_normalizer()

    use_sim_fallback = is_offline_d4rl_env(env)
    env.close()

    env_eval, sim_name = make_sim_eval_env(
        args.task.env_name,
        sim_env_name=args.sim_env_name,
        render=False,
        ignore_termination=False,
    )
    _log(f"[eval] Sim env: {sim_name} (offline_fallback={use_sim_fallback})")
    _log(
        f"[eval] Checkpoint: {args_cli.ckpt}, episodes: {args_cli.num_episodes}, "
        f"max_steps: {args_cli.max_steps}, log_interval: {args_cli.log_interval}"
    )
    _log(
        f"[eval] Sampling config: num_candidates={args.num_candidates}, "
        f"sampling_steps={args.sampling_steps}, w_cg={args.task.w_cg}, "
        f"guidance_mode={args.guidance_mode}, "
        f"optimization_guidance_scale={args.optimization_guidance_scale}"
    )

    _log("[eval] Benchmarking one agent.sample() call...")
    sample_s = benchmark_sample(agent, normalizer, args, obs_dim, act_dim, env_eval)
    worst_case_s = sample_s * args_cli.max_steps * args_cli.num_episodes
    _log(
        f"[eval] One sample call took {sample_s:.2f}s. "
        f"Each env step runs one sample with {args.num_candidates} candidates. "
        f"Worst-case wall time (all episodes hit max_steps): ~{worst_case_s/60:.1f} min."
    )

    episodes = []
    run_start = time.perf_counter()
    for ep in range(args_cli.num_episodes):
        set_seed(args_cli.seed + ep)
        ep_result = rollout_with_trajectory(
            env_eval,
            agent,
            normalizer,
            args,
            obs_dim,
            act_dim,
            max_steps=args_cli.max_steps,
            episode_idx=ep,
            log_interval=args_cli.log_interval,
            expected_sample_s=sample_s,
        )
        ep_result["episode"] = ep
        episodes.append(ep_result)
        _log(
            f"[episode {ep} summary] class={ep_result['classification']}, "
            f"survival={ep_result['survival_steps']}, reward={ep_result['total_reward']:.1f}, "
            f"fall_step={ep_result['first_fall_step']}, min_z={ep_result['min_height']:.3f}"
        )

    env_eval.close()

    summary = {
        "task": args.task.env_name,
        "sim_env": sim_name,
        "ckpt": args_cli.ckpt,
        "seed": args_cli.seed,
        "guidance_mode": args.guidance_mode,
        "optimization_guidance_scale": float(args.optimization_guidance_scale),
        "w_cg": float(args.task.w_cg),
        "num_episodes": args_cli.num_episodes,
        "max_steps": args_cli.max_steps,
        "class_counts": {},
        "mean_survival_steps": float(np.mean([e["survival_steps"] for e in episodes])),
        "mean_reward": float(np.mean([e["total_reward"] for e in episodes])),
        "episodes": episodes,
    }
    for ep in episodes:
        summary["class_counts"][ep["classification"]] = summary["class_counts"].get(ep["classification"], 0) + 1

    _log(f"[eval] Total wall time: {(time.perf_counter() - run_start)/60:.1f} min")
    _log("\n=== Trajectory Eval Summary ===")
    _log(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    _log(f"class_counts: {summary['class_counts']}")

    out = args_cli.output or f"results/{args.pipeline_name}/{args.task.env_name}/trajectory_eval_{args_cli.ckpt}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"[eval] Saved {out}")


if __name__ == "__main__":
    main()
