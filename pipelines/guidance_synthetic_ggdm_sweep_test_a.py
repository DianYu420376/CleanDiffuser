#!/usr/bin/env python3
"""GGDM-only hyperparameter sweep on synthetic Test A with convergence overlay."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import torch

from guidance_synthetic_subspace import (
    AMBIENT_DIM,
    HORIZON,
    _build_agent,
    _load_geometry,
    _make_reward_fn,
    _resolve_paths,
    constraint_violation,
    reward_values,
    solve_optimal_feasible,
)
from guidance_synthetic_visualize_subspace import (
    _apply_paper_style,
    _configure_violation_axis,
    _load_normalizer,
    _save_figure,
    _style_axis_grid,
    sample_paths_with_history,
)
from utils import set_seed

DEFAULT_INTERVALS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
DEFAULT_BETAS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
DEFAULT_ROUNDS = [6, 8, 10]
DCG_REWARD = {"test_a": -5.24, "test_b": -8.09}


def _config_label(interval: float, beta: float, rounds: int) -> str:
    return f"δ={interval:g}, β={beta:g}, R={rounds}"


def _score_run(metrics: dict, feasible_optimum: dict, dcg_reward: float = -5.24) -> float:
    """Higher is better: close to feasible optimum, low violation, stay below DCG."""
    reward = metrics["reward_mean"]
    viol = metrics["violation_mean"]
    opt_reward = feasible_optimum["reward"]
    gap = abs(reward - opt_reward)
    score = -gap - 10.0 * viol
    if reward > dcg_reward + 0.005:
        score -= 500.0 * (reward - dcg_reward)
    return score


def _default_json(test_name: str) -> str:
    return f"results/guidance_synthetic/{test_name}/ggdm_sweep_results.json"


def _run_sweep(args, test_name: str) -> dict:
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    normalizer = _load_normalizer(root / "normalizer.npz")
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)

    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(_resolve_paths(test_name, Path(args.output_root))["ckpt"]))
    agent.eval()

    set_seed(args.seed)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    initial_xts = (
        torch.randn((args.n_paths, HORIZON, AMBIENT_DIM), device=args.device, generator=gen)
        * args.temperature
    )

    x_opt, u_opt = solve_optimal_feasible(geometry)
    opt_reward = float(reward_values(x_opt[None], geometry)[0])
    feasible_optimum = {
        "ambient": x_opt.tolist(),
        "subspace": u_opt.tolist(),
        "reward": opt_reward,
        "violation": float(constraint_violation(x_opt[None], geometry)[0]),
    }
    print(
        f"[{test_name}] Feasible optimum: u={u_opt[:2]}, reward={opt_reward:.4f}"
    )

    rounds_options = (
        list(args.ggdm_rounds)
        if isinstance(args.ggdm_rounds, (list, tuple))
        else [int(args.ggdm_rounds)]
    )
    runs: list[dict] = []
    grid = list(product(args.intervals, args.beta_coefs, rounds_options))
    json_path = Path(args.output_json or _default_json(test_name))
    json_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_partial() -> None:
        serializable = {
            **{k: v for k, v in sweep_data.items() if k != "runs"},
            "runs": [
                {
                    **{k: v for k, v in run.items() if k not in ("reward_curves", "violation_curves", "paths")},
                    "score": run["score"],
                    "reward_curves": np.asarray(run["reward_curves"]).tolist(),
                    "violation_curves": np.asarray(run["violation_curves"]).tolist(),
                }
                for run in runs
            ],
        }
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)

    sweep_data = {
        "test": test_name,
        "seed": args.seed,
        "n_paths": args.n_paths,
        "feasible_optimum": feasible_optimum,
        "runs": runs,
    }

    for idx, (interval, beta, rounds) in enumerate(grid):
        args.ggdm_interval = interval
        args.ggdm_beta_coef = beta
        args.ggdm_rounds = rounds
        label = _config_label(interval, beta, rounds)
        print(f"[{idx + 1}/{len(grid)}] GGDM {label}")
        paths, reward_curves, violation_curves, metrics = sample_paths_with_history(
            args,
            geometry,
            normalizer,
            agent,
            reward_fn,
            guidance_mode="ggdm_g_loss",
            w_cg=0.0,
            opt_scale=0.0,
            initial_xts=initial_xts,
        )
        runs.append(
            {
                "label": label,
                "ggdm_interval": interval,
                "ggdm_beta_coef": beta,
                "ggdm_rounds": rounds,
                "metrics": metrics,
                "score": _score_run(
                    metrics,
                    feasible_optimum,
                    dcg_reward=args.dcg_reward if args.dcg_reward is not None else DCG_REWARD[test_name],
                ),
                "reward_curves": reward_curves,
                "violation_curves": violation_curves,
                "paths": paths,
            }
        )
        sweep_data["runs"] = runs
        runs.sort(
            key=lambda r: (
                r["score"],
                r["metrics"]["reward_mean"],
                -r["metrics"]["violation_mean"],
            ),
            reverse=True,
        )
        _write_partial()
        print(
            f"  -> reward={metrics['reward_mean']:.4f}, viol={metrics['violation_mean']:.4f}",
            flush=True,
        )

    args.ggdm_rounds = rounds_options
    return sweep_data


def _plot_sweep(args, sweep_data: dict, test_name: str) -> None:
    _apply_paper_style()
    opt_objective = -float(sweep_data["feasible_optimum"]["reward"])
    viol_floor = 1e-4
    runs = sweep_data["runs"]

    cmap = colormaps["turbo"]
    n = max(len(runs), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    panels = [
        ("reward_curves", "Objective", axes[0]),
        ("violation_curves", "Constraint violation", axes[1]),
    ]

    for run_idx, run in enumerate(runs):
        color = cmap(run_idx / max(n - 1, 1))
        for key, _, ax in panels:
            curves = np.asarray(run[key])
            steps = np.arange(curves.shape[1])
            if key == "reward_curves":
                y = -curves.mean(axis=0)
            else:
                y = np.maximum(curves.mean(axis=0), viol_floor)
            ax.plot(steps, y, color=color, linewidth=1.4, alpha=0.88, label=run["label"])

    for key, subtitle, ax in panels:
        if key == "reward_curves":
            ax.axhline(
                opt_objective,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Optimal (feasible)",
                zorder=3,
            )
        if key == "violation_curves":
            _configure_violation_axis(ax)
        ax.set_xlabel("Step (GGDM reverse diffusion)")
        ax.set_title(subtitle)
        _style_axis_grid(ax, minor=True)

    axes[0].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )
    fig.tight_layout()
    out_dir = Path(args.output_root) / test_name / "figures"
    output_name = args.output_name or f"{test_name}_ggdm_sweep"
    _save_figure(fig, out_dir, output_name)
    plt.close(fig)

    # Compact final-objective ranking plot.
    fig2, ax2 = plt.subplots(figsize=(8.5, max(4.0, 0.22 * len(runs))))
    rewards = [-r["metrics"]["reward_mean"] for r in runs]
    labels = [r["label"] for r in runs]
    y_pos = np.arange(len(runs))
    ax2.barh(y_pos, rewards, color=[cmap(i / max(n - 1, 1)) for i in range(n)], alpha=0.9)
    ax2.axvline(opt_objective, color="black", linestyle="--", linewidth=1.8, label="Optimal")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Final objective (−reward)")
    ax2.set_title(f"GGDM sweep ranking ({test_name})")
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    _save_figure(fig2, out_dir, f"{output_name}_ranking")
    plt.close(fig2)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["test_a", "test_b", "both"], default="both")
    p.add_argument("--output-root", default="results/guidance_synthetic")
    p.add_argument("--output-name", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--dcg-reward", type=float, default=None, help="Override DCG ceiling for scoring.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--training-diffusion-steps", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--solver", default="ddim")
    p.add_argument("--ggdm-solver", default="ddpm")
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--n-paths", type=int, default=3)
    p.add_argument("--optimization-guidance-last-steps", type=int, default=50)
    p.add_argument("--ggdm-interval", type=float, default=0.05)
    p.add_argument("--ggdm-beta-coef", type=float, default=0.5)
    p.add_argument("--ggdm-rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--drgd-steps", type=int, default=50)
    p.add_argument("--drgd-gamma", type=float, default=10.0)
    p.add_argument("--drgd-noise-index", type=int, default=49)
    p.add_argument("--ggdm-obs-noise-sigma", type=float, default=0.1)
    p.add_argument("--intervals", nargs="+", type=float, default=DEFAULT_INTERVALS)
    p.add_argument("--beta-coefs", nargs="+", type=float, default=DEFAULT_BETAS)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def _serialize_sweep(sweep_data: dict) -> dict:
    return {
        **{k: v for k, v in sweep_data.items() if k != "runs"},
        "runs": [
            {
                **{k: v for k, v in run.items() if k not in ("reward_curves", "violation_curves", "paths")},
                "score": run["score"],
                "reward_curves": np.asarray(run["reward_curves"]).tolist(),
                "violation_curves": np.asarray(run["violation_curves"]).tolist(),
            }
            for run in sweep_data["runs"]
        ],
    }


def _print_summary(test_name: str, sweep_data: dict) -> dict:
    best = sweep_data["runs"][0]
    opt_r = sweep_data["feasible_optimum"]["reward"]
    print(
        f"[{test_name}] Best GGDM config: {best['label']} -> "
        f"reward={best['metrics']['reward_mean']:.4f}±{best['metrics']['reward_std']:.4f} "
        f"(opt={opt_r:.4f}, gap={abs(best['metrics']['reward_mean'] - opt_r):.4f}), "
        f"violation={best['metrics']['violation_mean']:.4f}, score={best['score']:.4f}"
    )
    print(f"[{test_name}] Top 5:")
    for r in sweep_data["runs"][:5]:
        print(
            f"  {r['label']}: reward={r['metrics']['reward_mean']:.4f}, "
            f"viol={r['metrics']['violation_mean']:.4f}, score={r['score']:.4f}"
        )
    return {
        "test": test_name,
        "ggdm_interval": best["ggdm_interval"],
        "ggdm_beta_coef": best["ggdm_beta_coef"],
        "ggdm_rounds": best["ggdm_rounds"],
        "ggdm_metrics": best["metrics"],
        "score": best["score"],
    }


def main() -> None:
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]
    best_by_test: dict[str, dict] = {}

    for test_name in tests:
        json_path = Path(args.output_json or _default_json(test_name))
        if args.plot_only:
            if not json_path.is_file():
                raise FileNotFoundError(f"Missing sweep JSON: {json_path}")
            with open(json_path) as f:
                sweep_data = json.load(f)
            _plot_sweep(args, sweep_data, test_name)
            continue

        sweep_data = _run_sweep(args, test_name)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(_serialize_sweep(sweep_data), f, indent=2)
        print(f"Wrote {json_path}")

        _plot_sweep(args, sweep_data, test_name)
        best_by_test[test_name] = _print_summary(test_name, sweep_data)

    if args.plot_only or not best_by_test:
        return

    tune_path = Path(args.output_root) / "baseline_tune_best.json"
    if tune_path.is_file():
        with open(tune_path) as f:
            tune_data = json.load(f)
    else:
        tune_data = {"seed": args.seed, "tests": {}}
    for test_name, best in best_by_test.items():
        entry = tune_data.setdefault("tests", {}).setdefault(test_name, {"test": test_name})
        entry.update(
            {
                "ggdm_interval": best["ggdm_interval"],
                "ggdm_beta_coef": best["ggdm_beta_coef"],
                "ggdm_rounds": best["ggdm_rounds"],
                "ggdm_metrics": best["ggdm_metrics"],
            }
        )
    with open(tune_path, "w") as f:
        json.dump(tune_data, f, indent=2)
    print(f"Updated GGDM entries in {tune_path}")


if __name__ == "__main__":
    main()
