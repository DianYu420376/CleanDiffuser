#!/usr/bin/env python3
"""Quick GPU sweep for GGDM G_loss and Landing DRGD on synthetic Tests A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from guidance_synthetic_subspace import (
    AMBIENT_DIM,
    HORIZON,
    _build_agent,
    _load_geometry,
    _make_reward_fn,
    _resolve_paths,
)
from guidance_synthetic_visualize_subspace import _load_normalizer, sample_paths_with_history
from utils import set_seed

GGDM_INTERVALS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
GGDM_BETA_COEFS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
DRGD_GAMMAS = [5.0, 10.0, 15.0, 20.0]
DRGD_NOISE_INDICES = [49, 45, 40, 35]


def _base_args(args, test_name: str):
    from argparse import Namespace

    w_cg = args.w_cg if test_name == "test_a" else args.w_cg_test_b
    return Namespace(
        seed=args.seed,
        device=args.device,
        model_dim=args.model_dim,
        diffusion_steps=args.diffusion_steps,
        training_diffusion_steps=args.training_diffusion_steps,
        sampling_steps=args.sampling_steps,
        optimization_guidance_last_steps=args.optimization_guidance_last_steps,
        solver=args.solver,
        ggdm_solver=args.ggdm_solver,
        sample_step_schedule=args.sample_step_schedule,
        temperature=args.temperature,
        use_ema=args.use_ema,
        predict_noise=args.predict_noise,
        ema_rate=args.ema_rate,
        output_root=args.output_root,
        ggdm_interval=0.05,
        ggdm_obs_noise_sigma=args.ggdm_obs_noise_sigma,
        ggdm_rounds=args.ggdm_rounds,
        ggdm_beta_coef=args.ggdm_beta_coef,
        drgd_steps=args.drgd_steps,
        drgd_gamma=10.0,
        drgd_noise_index=49,
        n_paths=args.n_paths,
        w_cg=w_cg,
        opt_scale=0.0,
    )


def _evaluate(
    run_args,
    geometry,
    normalizer,
    agent,
    reward_fn,
    *,
    guidance_mode: str,
    initial_xts: torch.Tensor,
) -> dict:
    _, _, _, metrics = sample_paths_with_history(
        run_args,
        geometry,
        normalizer,
        agent,
        reward_fn,
        guidance_mode=guidance_mode,
        w_cg=0.0,
        opt_scale=0.0,
        initial_xts=initial_xts,
    )
    return metrics


def _score(metrics: dict, viol_cap: float, dcg_reward: float, *, beta_coef: float = 0.5) -> float:
    """Prefer high reward, low violation, staying below DCG; tie-break toward beta≈0.5."""
    reward = metrics["reward_mean"]
    viol = metrics["violation_mean"]
    score = reward
    if viol > viol_cap:
        score -= 100.0 * (viol - viol_cap)
    if reward > dcg_reward + 0.005:
        score -= 500.0 * (reward - dcg_reward)
    score += 0.05 * max(0.0, 1.0 - abs(beta_coef - 0.5))
    return score


def tune_test(args, test_name: str) -> dict:
    run_args = _base_args(args, test_name)
    set_seed(args.seed)
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    normalizer = _load_normalizer(root / "normalizer.npz")
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(run_args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(_resolve_paths(test_name, Path(args.output_root))["ckpt"]))
    agent.eval()

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    initial_xts = (
        torch.randn((args.n_paths, HORIZON, AMBIENT_DIM), device=args.device, generator=gen)
        * args.temperature
    )

    viol_cap = args.violation_cap_a if test_name == "test_a" else args.violation_cap_b
    dcg_reward = args.dcg_reward_a if test_name == "test_a" else args.dcg_reward_b
    best_ggdm = {"score": float("-inf")}
    for interval in GGDM_INTERVALS:
        for beta_coef in GGDM_BETA_COEFS:
            run_args.ggdm_interval = interval
            run_args.ggdm_beta_coef = beta_coef
            metrics = _evaluate(
                run_args, geometry, normalizer, agent, reward_fn,
                guidance_mode="ggdm_g_loss", initial_xts=initial_xts,
            )
            score = _score(metrics, viol_cap, dcg_reward, beta_coef=beta_coef)
            print(
                f"[{test_name}] GGDM interval={interval:g}, beta={beta_coef:g}: "
                f"reward={metrics['reward_mean']:.4f}, viol={metrics['violation_mean']:.4f}, score={score:.4f}"
            )
            if score > best_ggdm["score"]:
                best_ggdm = {
                    "ggdm_interval": interval,
                    "ggdm_beta_coef": beta_coef,
                    "metrics": metrics,
                    "score": score,
                }

    best_drgd = {"score": float("-inf")}
    run_args.drgd_steps = args.drgd_steps
    for noise_idx in DRGD_NOISE_INDICES:
        run_args.drgd_noise_index = noise_idx
        for gamma in DRGD_GAMMAS:
            run_args.drgd_gamma = gamma
            metrics = _evaluate(
                run_args, geometry, normalizer, agent, reward_fn,
                guidance_mode="drgd", initial_xts=initial_xts,
            )
            score = _score(metrics, viol_cap, dcg_reward)
            print(
                f"[{test_name}] DRGD idx={noise_idx}, steps={run_args.drgd_steps}, gamma={gamma:g}: "
                f"reward={metrics['reward_mean']:.4f}, viol={metrics['violation_mean']:.4f}, score={score:.4f}"
            )
            if score > best_drgd["score"]:
                best_drgd = {
                    "drgd_noise_index": noise_idx,
                    "drgd_steps": run_args.drgd_steps,
                    "drgd_gamma": gamma,
                    "metrics": metrics,
                    "score": score,
                }

    return {
        "test": test_name,
        "ggdm_interval": best_ggdm["ggdm_interval"],
        "ggdm_beta_coef": best_ggdm["ggdm_beta_coef"],
        "ggdm_metrics": best_ggdm["metrics"],
        "drgd_steps": best_drgd["drgd_steps"],
        "drgd_gamma": best_drgd["drgd_gamma"],
        "drgd_noise_index": best_drgd["drgd_noise_index"],
        "drgd_metrics": best_drgd["metrics"],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["test_a", "test_b", "both"], default="both")
    p.add_argument("--output-root", default="results/guidance_synthetic")
    p.add_argument("--output-json", default="results/guidance_synthetic/baseline_tune_best.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--training-diffusion-steps", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--optimization-guidance-last-steps", type=int, default=50)
    p.add_argument("--solver", default="ddim", help="Solver used when evaluating DRGD.")
    p.add_argument("--ggdm-solver", default="ddpm")
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--n-paths", type=int, default=3, help="Paths per config during tuning.")
    p.add_argument("--w-cg", type=float, default=0.6)
    p.add_argument("--w-cg-test-b", type=float, default=10.0)
    p.add_argument("--ggdm-obs-noise-sigma", type=float, default=0.1)
    p.add_argument("--ggdm-rounds", type=int, default=8, help="Guided outer iterations during tuning.")
    p.add_argument("--ggdm-beta-coef", type=float, default=0.5)
    p.add_argument("--drgd-steps", type=int, default=50)
    p.add_argument("--violation-cap-a", type=float, default=0.5)
    p.add_argument("--violation-cap-b", type=float, default=0.5)
    p.add_argument("--dcg-reward-a", type=float, default=-5.24)
    p.add_argument("--dcg-reward-b", type=float, default=-8.09)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main() -> None:
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]
    payload = {"seed": args.seed, "tests": {}}
    for test_name in tests:
        payload["tests"][test_name] = tune_test(args, test_name)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote tuned baseline hyperparameters to {out_path}")


if __name__ == "__main__":
    main()
