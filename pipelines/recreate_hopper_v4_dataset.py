"""Re-roll a D4RL Hopper v2 dataset through Gymnasium Hopper-v4 physics.

Loads ``hopper-medium-v2`` (or any compatible Hopper offline task), keeps the
original per-episode initial MuJoCo states and action sequences, and rewrites
observations, next observations, rewards, and ``infos/qpos`` / ``infos/qvel`` from
open-loop Hopper-v4 rollouts.

MuJoCo stepping is CPU-bound. This script parallelizes rollouts across worker
processes and optionally uses a CUDA device for fast buffer assembly / validation.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gym
import h5py
import numpy as np

import d4rl  # noqa: F401

# Physics-only MuJoCo (no GL context) must be set before gymnasium import in workers.
os.environ.setdefault("MUJOCO_GL", "disable")


@dataclass(frozen=True)
class EpisodeSpan:
    start: int
    end: int  # inclusive index in the flat dataset


def _episode_spans(terminals: np.ndarray, timeouts: np.ndarray) -> list[EpisodeSpan]:
    done = np.asarray(terminals, dtype=bool) | np.asarray(timeouts, dtype=bool)
    ends = np.where(done)[0]
    if ends.size == 0:
        raise ValueError("Dataset has no terminal/timeout markers.")
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [EpisodeSpan(int(s), int(e)) for s, e in zip(starts, ends)]


def _load_source_dataset(task: str, h5path: str | None) -> tuple[dict, str]:
    env = gym.make(task)
    try:
        ds = env.get_dataset(h5path=h5path) if h5path else env.get_dataset()
        source_path = h5path or env.dataset_filepath
    finally:
        env.close()
    return ds, source_path


class _IgnoreTerminationWrapper:
    """Keep rollout length fixed even if Hopper-v4 signals termination."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, False, info

    def close(self):
        return self.env.close()


def _make_hopper_v4_env(ignore_termination: bool):
    import gymnasium as gymnasium

    env = gymnasium.make("Hopper-v4")
    if ignore_termination:
        env = _IgnoreTerminationWrapper(env)
    return env


def _rollout_episode(
    start: int,
    end: int,
    actions: np.ndarray,
    init_qpos: np.ndarray,
    init_qvel: np.ndarray,
    ignore_termination: bool,
    source_timeout_at_end: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    bool,
]:
    """Roll one source episode; optionally truncate when Hopper-v4 terminates."""
    env = _make_hopper_v4_env(ignore_termination)
    env.reset()
    unwrapped = env.unwrapped
    unwrapped.set_state(np.asarray(init_qpos, dtype=np.float64), np.asarray(init_qvel, dtype=np.float64))

    max_len = end - start + 1
    observations = []
    next_observations = []
    rewards = []
    qpos_out = []
    qvel_out = []
    act_out = []
    terminated_early = False

    for global_idx in range(start, end + 1):
        qpos_out.append(np.array(unwrapped.data.qpos, dtype=np.float64))
        qvel_out.append(np.array(unwrapped.data.qvel, dtype=np.float64))
        observations.append(np.asarray(unwrapped._get_obs(), dtype=np.float32))
        act_out.append(np.asarray(actions[global_idx], dtype=np.float32))
        next_obs, reward, terminated, truncated, _ = env.step(actions[global_idx])
        next_observations.append(np.asarray(next_obs, dtype=np.float32))
        rewards.append(np.float32(reward))
        if not ignore_termination and (terminated or truncated):
            terminated_early = True
            break

    env.close()

    n = len(observations)
    terminals = np.zeros(n, dtype=bool)
    timeouts = np.zeros(n, dtype=bool)
    if n > 0:
        if terminated_early:
            terminals[-1] = True
        elif source_timeout_at_end:
            timeouts[-1] = True

    return (
        np.stack(observations, axis=0),
        np.stack(next_observations, axis=0),
        np.asarray(rewards, dtype=np.float32),
        np.stack(qpos_out, axis=0),
        np.stack(qvel_out, axis=0),
        np.stack(act_out, axis=0),
        terminals,
        timeouts,
        terminated_early,
    )


def _rollout_episode_packed(args: tuple) -> tuple:
    """Top-level pickleable worker entry point."""
    (
        start,
        end,
        actions,
        init_qpos,
        init_qvel,
        ignore_termination,
        source_timeout_at_end,
    ) = args
    return _rollout_episode(
        start,
        end,
        actions,
        init_qpos,
        init_qvel,
        ignore_termination,
        source_timeout_at_end,
    )


def _copy_hdf5_metadata(src_path: str, dst_file: h5py.File, skip_prefixes: Iterable[str]) -> None:
    skip = set(skip_prefixes)
    with h5py.File(src_path, "r") as src:
        def _copy_node(name: str, obj):
            if name in skip:
                return
            if isinstance(obj, h5py.Dataset):
                src.copy(obj, dst_file, name=name)

        src.visititems(_copy_node)


def _assemble_on_device(
    device: str,
    observations: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> str:
    """Optional GPU pass to validate / touch buffers (cheap sanity check)."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[reroll] CUDA unavailable; falling back to CPU assembly.")
        device = "cpu"

    dev = torch.device(device)
    for arr in (observations, next_observations, rewards, qpos, qvel):
        tensor = torch.as_tensor(arr, device=dev)
        if torch.isnan(tensor).any():
            raise RuntimeError("NaN detected in rerolled dataset buffers.")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return device


def recreate_dataset(
    source_task: str = "hopper-medium-v2",
    source_h5path: str | None = None,
    output_path: str | Path | None = None,
    num_workers: int | None = None,
    max_episodes: int | None = None,
    ignore_termination: bool = True,
    assemble_device: str = "cpu",
) -> Path:
    t0 = time.time()
    ds, source_path = _load_source_dataset(source_task, source_h5path)

    actions = np.asarray(ds["actions"], dtype=np.float32)
    terminals = np.asarray(ds["terminals"], dtype=bool)
    timeouts = np.asarray(ds["timeouts"], dtype=bool)
    source_qpos = np.asarray(ds["infos/qpos"], dtype=np.float64)
    source_qvel = np.asarray(ds["infos/qvel"], dtype=np.float64)

    spans = _episode_spans(terminals, timeouts)
    if max_episodes is not None:
        spans = spans[:max_episodes]

    if output_path is None:
        stem = source_task.replace("-", "_").replace("v2", "v4_reroll")
        output_path = Path.home() / ".d4rl" / "datasets" / f"{stem}.hdf5"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (
            span.start,
            span.end,
            actions,
            source_qpos[span.start],
            source_qvel[span.start],
            ignore_termination,
            bool(timeouts[span.end]),
        )
        for span in spans
    ]

    num_workers = num_workers or max(1, (os.cpu_count() or 1) - 1)
    print(
        f"[reroll] source={source_path}\n"
        f"[reroll] source_episodes={len(spans)} source_transitions={spans[-1].end + 1}\n"
        f"[reroll] sim=Hopper-v4 workers={num_workers} truncate_on_fall={not ignore_termination}\n"
        f"[reroll] output={output_path}"
    )

    episode_chunks: list[tuple] = [None] * len(spans)
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_map = {
            pool.submit(_rollout_episode_packed, args): idx
            for idx, args in enumerate(worker_args)
        }
        for fut in as_completed(future_map):
            episode_chunks[future_map[fut]] = fut.result()
            completed += 1
            if completed % 100 == 0 or completed == len(spans):
                print(f"[reroll] finished {completed}/{len(spans)} episodes")

    obs_chunks = []
    next_obs_chunks = []
    rew_chunks = []
    act_chunks = []
    qpos_chunks = []
    qvel_chunks = []
    term_chunks = []
    timeout_chunks = []
    n_truncated = 0
    for chunk in episode_chunks:
        obs, next_obs, rew, qpos_ep, qvel_ep, act_ep, term_ep, to_ep, fell = chunk
        obs_chunks.append(obs)
        next_obs_chunks.append(next_obs)
        rew_chunks.append(rew)
        act_chunks.append(act_ep)
        qpos_chunks.append(qpos_ep)
        qvel_chunks.append(qvel_ep)
        term_chunks.append(term_ep)
        timeout_chunks.append(to_ep)
        if fell:
            n_truncated += 1

    observations = np.concatenate(obs_chunks, axis=0)
    next_observations = np.concatenate(next_obs_chunks, axis=0)
    rewards = np.concatenate(rew_chunks, axis=0)
    actions_out = np.concatenate(act_chunks, axis=0)
    qpos_out = np.concatenate(qpos_chunks, axis=0)
    qvel_out = np.concatenate(qvel_chunks, axis=0)
    terminals_out = np.concatenate(term_chunks, axis=0)
    timeouts_out = np.concatenate(timeout_chunks, axis=0)
    n_transitions = observations.shape[0]

    print(
        f"[reroll] output transitions={n_transitions} "
        f"episodes_truncated_by_fall={n_truncated}/{len(spans)}"
    )

    if assemble_device != "cpu":
        assemble_device = _assemble_on_device(
            assemble_device,
            observations,
            next_observations,
            rewards,
            qpos_out,
            qvel_out,
        )

    reroll_keys = {
        "observations",
        "next_observations",
        "rewards",
        "actions",
        "terminals",
        "timeouts",
        "infos/qpos",
        "infos/qvel",
        "infos/action_log_probs",
    }
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as out:
        out.create_dataset("observations", data=observations, compression="gzip")
        out.create_dataset("next_observations", data=next_observations, compression="gzip")
        out.create_dataset("rewards", data=rewards, compression="gzip")
        out.create_dataset("actions", data=actions_out, compression="gzip")
        out.create_dataset("terminals", data=terminals_out, compression="gzip")
        out.create_dataset("timeouts", data=timeouts_out, compression="gzip")
        out.create_dataset("infos/qpos", data=qpos_out, compression="gzip")
        out.create_dataset("infos/qvel", data=qvel_out, compression="gzip")
        if "infos/action_log_probs" in ds and not ignore_termination:
            print("[reroll] skipping infos/action_log_probs (episode lengths changed after truncation)")
        elif "infos/action_log_probs" in ds:
            out.create_dataset(
                "infos/action_log_probs",
                data=np.asarray(ds["infos/action_log_probs"][:n_transitions]),
                compression="gzip",
            )

        _copy_hdf5_metadata(
            source_path,
            out,
            skip_prefixes=reroll_keys,
        )
        out.attrs["reroll_source_task"] = source_task
        out.attrs["reroll_source_h5"] = str(source_path)
        out.attrs["reroll_sim_env"] = "Hopper-v4"
        out.attrs["reroll_truncate_on_fall"] = bool(not ignore_termination)
        out.attrs["reroll_ignore_termination"] = bool(ignore_termination)
        out.attrs["reroll_episodes_truncated_by_fall"] = int(n_truncated)

    elapsed = time.time() - t0
    stats = analyze_dataset_quality(output_path)
    print(f"[reroll] saved {output_path} ({elapsed:.1f}s)")
    print(f"[reroll] quality: {stats}")
    return output_path


def analyze_dataset_quality(h5path: str | Path) -> dict:
    with h5py.File(h5path, "r") as f:
        qpos = f["infos/qpos"][:]
        rewards = f["rewards"][:]
        terminals = f["terminals"][:]
        timeouts = f["timeouts"][:]

    height = qpos[:, 1]
    ang = qpos[:, 2]
    fallen = (height <= 0.7) | (np.abs(ang) >= 0.2)

    spans = _episode_spans(terminals, timeouts)
    returns = [float(rewards[s : e + 1].sum()) for s, e in ((sp.start, sp.end) for sp in spans)]

    return {
        "transitions": int(rewards.shape[0]),
        "episodes": len(spans),
        "mean_step_reward": float(rewards.mean()),
        "median_step_reward": float(np.median(rewards)),
        "mean_episode_return": float(np.mean(returns)),
        "median_episode_return": float(np.median(returns)),
        "frac_transitions_fallen": float(fallen.mean()),
        "frac_episodes_terminal_fall": float(terminals.sum() / max(len(spans), 1)),
        "median_episode_len": float(np.median([e - s + 1 for s, e in ((sp.start, sp.end) for sp in spans)])),
    }


CHECK_HORIZONS = (1, 25, 50, 100)


def validate_trajectory_replay(
    h5path: str | Path,
    num_trajectories: int = 100,
    ref_percentile: float = 50.0,
) -> dict:
    """Open-loop replay on sampled episodes; print 1/25/50/100-step errors per trajectory."""
    import gymnasium as gymnasium

    with h5py.File(h5path, "r") as f:
        qpos = f["infos/qpos"][:]
        qvel = f["infos/qvel"][:]
        actions = f["actions"][:]
        next_obs = f["next_observations"][:]
        terminals = f["terminals"][:]
        timeouts = f["timeouts"][:]

    spans = _episode_spans(terminals, timeouts)
    if num_trajectories < len(spans):
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(spans), size=num_trajectories, replace=False))
        spans = [spans[i] for i in pick]
    else:
        num_trajectories = len(spans)

    ref_scale = float(np.percentile(np.linalg.norm(next_obs, axis=1), ref_percentile))
    if ref_scale <= 0:
        raise ValueError("Observation reference scale must be positive.")

    env = gymnasium.make("Hopper-v4")
    env.reset()
    unwrapped = env.unwrapped

    per_traj: list[dict] = []
    print(
        f"[validate] dataset={h5path}\n"
        f"[validate] trajectories={num_trajectories} horizons={CHECK_HORIZONS}\n"
        f"[validate] ref_scale (p{ref_percentile:g} next_obs L2)={ref_scale:.6f}"
    )

    for traj_idx, span in enumerate(spans):
        start, end = span.start, span.end
        seg_len = end - start + 1
        unwrapped.set_state(qpos[start], qvel[start])

        horizon_errors: dict[int, float] = {}
        for horizon in CHECK_HORIZONS:
            if seg_len < horizon:
                horizon_errors[horizon] = float("nan")
                continue

            unwrapped.set_state(qpos[start], qvel[start])
            max_err = 0.0
            for k in range(horizon):
                sim_next, _, terminated, _, _ = env.step(actions[start + k])
                err = float(
                    np.linalg.norm(np.asarray(sim_next, dtype=np.float32) - next_obs[start + k])
                )
                max_err = max(max_err, err)
                if terminated:
                    break
            horizon_errors[horizon] = 100.0 * max_err / ref_scale

        entry = {
            "trajectory": traj_idx,
            "start": start,
            "length": seg_len,
            "horizon_pct": horizon_errors,
        }
        per_traj.append(entry)

        parts = [
            f"h{h}={horizon_errors[h]:.4f}%"
            if np.isfinite(horizon_errors[h])
            else f"h{h}=n/a"
            for h in CHECK_HORIZONS
        ]
        print(
            f"[validate] traj {traj_idx:3d} start={start:6d} len={seg_len:4d}  "
            + "  ".join(parts)
        )

    env.close()

    summary = {}
    for horizon in CHECK_HORIZONS:
        vals = np.array(
            [t["horizon_pct"][horizon] for t in per_traj if np.isfinite(t["horizon_pct"][horizon])],
            dtype=np.float64,
        )
        summary[f"h{horizon}"] = {
            "n": int(vals.size),
            "median_pct": float(np.median(vals)) if vals.size else float("nan"),
            "max_pct": float(vals.max()) if vals.size else float("nan"),
        }

    worst = max(
        (summary[f"h{h}"]["max_pct"] for h in CHECK_HORIZONS if summary[f"h{h}"]["n"] > 0),
        default=float("nan"),
    )
    passed = bool(np.isfinite(worst) and worst < 0.1)
    print(
        f"[validate] summary median_pct="
        + ", ".join(f"h{h}={summary[f'h{h}']['median_pct']:.6f}" for h in CHECK_HORIZONS)
    )
    print(
        f"[validate] summary max_pct="
        + ", ".join(f"h{h}={summary[f'h{h}']['max_pct']:.6f}" for h in CHECK_HORIZONS)
    )
    print(f"[validate] worst_case_max_pct={worst:.6f}%  pass(<0.1%)={passed}")

    return {
        "h5path": str(h5path),
        "num_trajectories": num_trajectories,
        "ref_scale": ref_scale,
        "horizons": list(CHECK_HORIZONS),
        "per_trajectory": per_traj,
        "summary": summary,
        "worst_case_max_pct": worst,
        "passed": passed,
    }


def _self_check(output_path: Path, num_samples: int = 2000) -> dict:
    """Verify rerolled dataset is self-consistent under Hopper-v4 replay."""
    import gymnasium as gymnasium

    with h5py.File(output_path, "r") as f:
        qpos = f["infos/qpos"][:]
        qvel = f["infos/qvel"][:]
        obs = f["observations"][:]
        actions = f["actions"][:]
        next_obs = f["next_observations"][:]

    n = min(num_samples, len(obs) - 1)
    env = gymnasium.make("Hopper-v4")
    env.reset()
    unwrapped = env.unwrapped

    static = np.zeros(n, dtype=np.float64)
    onestep = np.zeros(n, dtype=np.float64)
    for t in range(n):
        unwrapped.set_state(qpos[t], qvel[t])
        sim_obs = np.asarray(unwrapped._get_obs(), dtype=np.float32)
        static[t] = np.linalg.norm(sim_obs - obs[t])

        unwrapped.set_state(qpos[t], qvel[t])
        sim_next, _, _, _, _ = env.step(actions[t])
        onestep[t] = np.linalg.norm(np.asarray(sim_next, dtype=np.float32) - next_obs[t])

    env.close()
    return {
        "static_l2_median": float(np.median(static)),
        "static_l2_max": float(static.max()),
        "onestep_l2_median": float(np.median(onestep)),
        "onestep_l2_max": float(onestep.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task", default="hopper-medium-v2")
    parser.add_argument("--source-h5path", default=None)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Default: ~/.d4rl/datasets/hopper_medium_v4_reroll.hdf5",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None, help="Smoke-test subset.")
    parser.add_argument(
        "--ignore-termination",
        action="store_true",
        help="Keep rolling after fall to preserve source episode lengths (old behavior).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print mean reward / fall stats for an existing HDF5 and exit.",
    )
    parser.add_argument(
        "--assemble-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Optional CUDA buffer validation pass after CPU rollouts.",
    )
    parser.add_argument("--self-check", action="store_true", help="Run Hopper-v4 consistency check.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip reroll; validate an existing HDF5 with per-trajectory open-loop replay.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help="Number of episode trajectories for --validate-only.",
    )
    args = parser.parse_args()

    default_h5 = Path.home() / ".d4rl" / "datasets" / "hopper_medium_v4_reroll.hdf5"
    h5path = args.output_path or default_h5

    if args.analyze_only:
        print(analyze_dataset_quality(h5path))
        return

    if args.validate_only:
        validate_trajectory_replay(h5path, num_trajectories=args.num_trajectories)
        return

    output_path = recreate_dataset(
        source_task=args.source_task,
        source_h5path=args.source_h5path,
        output_path=args.output_path,
        num_workers=args.num_workers,
        max_episodes=args.max_episodes,
        ignore_termination=args.ignore_termination,
        assemble_device=args.assemble_device,
    )

    if args.self_check:
        stats = _self_check(output_path)
        print("[reroll] self-check:", stats)
        validate_trajectory_replay(output_path, num_trajectories=args.num_trajectories)


if __name__ == "__main__":
    main()
