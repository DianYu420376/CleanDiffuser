#!/usr/bin/env python3
"""Paper-ready subspace figures: PCG baseline vs DCG guidance on Tests A and B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
import numpy as np
import torch

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from guidance_synthetic_subspace import (
    AMBIENT_DIM,
    HORIZON,
    GeometryMeta,
    _build_agent,
    _load_geometry,
    _make_reward_fn,
    _resolve_paths,
    _sample_batch_metrics,
    constraint_violation,
    reward_values,
    solve_optimal_feasible,
)
from cleandiffuser.utils import GaussianNormalizer
from utils import set_seed

PDG_COLOR = "#2166AC"
DCG_COLOR = "#E66101"
GGDM_COLOR = "#4DAF4A"
DRGD_COLOR = "#984EA3"
FEASIBLE_FACE = "#EAF2FA"
FEASIBLE_FACE_ALPHA = 0.55
FEASIBLE_EDGE = "#2F4F6F"
FEASIBLE_EDGE_WIDTH = 1.8
INIT_COLOR = "#404040"
OPTIMAL_COLOR = "#C41E3A"

METHOD_STYLES = {
    "standard": {"color": PDG_COLOR, "label": "PCG"},
    "optimization": {"color": DCG_COLOR, "label": "DCG"},
    "ggdm_g_loss": {"color": GGDM_COLOR, "label": "GGDM"},
    "ggdm_g_loss_mean": {"color": GGDM_COLOR, "label": "GGDM"},
    "drgd": {"color": DRGD_COLOR, "label": "DRGD"},
}
SAMPLE_METHODS = ("standard", "optimization", "ggdm_g_loss_mean", "drgd")
CONVERGENCE_METHODS = SAMPLE_METHODS
PATH_PLOT_METHODS = ("standard", "optimization", "drgd")
PLOT_METHODS = SAMPLE_METHODS  # backward-compatible alias for cache I/O
GGDM_COMPARISON_METHODS = ("ggdm_g_loss", "ggdm_g_loss_mean")
GGDM_GUIDANCE_MODES = ("ggdm_g_loss", "ggdm_g_loss_mean")
LEGACY_PCG_DCG_SUFFIX = "pcg_dcg_paper_s50_last50"
LEGACY_METADATA_KEYS = (
    "test",
    "seed",
    "n_paths",
    "w_cg",
    "opt_scale",
    "diffusion_steps",
    "training_diffusion_steps",
    "sampling_steps",
    "optimization_guidance_last_steps",
    "solver",
    "sample_step_schedule",
    "temperature",
    "use_ema",
    "predict_noise",
)

# Per-test defaults from matched 50/50/50 sweeps.
TEST_DEFAULTS = {
    "test_a": {"w_cg": 0.6, "opt_scale": 15.0, "plot_limit": 2.0},
    "test_b": {"w_cg": 10.0, "opt_scale": 10.0, "plot_limit": 1.8},
}

PAPER_FONT_SCALE = 1.35
PATH_LEGEND_FONTSIZE = 9.0
LEGEND_FRAME_ALPHA = 0.55


def _apply_paper_style() -> None:
    s = PAPER_FONT_SCALE
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11 * s,
            "axes.labelsize": 12 * s,
            "axes.titlesize": 13 * s,
            "legend.fontsize": 10 * s,
            "xtick.labelsize": 10 * s,
            "ytick.labelsize": 10 * s,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "lines.linewidth": 2.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _style_axis_grid(ax, *, minor: bool = True) -> None:
    ax.grid(True, which="major", color="#D0D0D0", linewidth=0.9, alpha=0.95)
    if minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", color="#E8E8E8", linewidth=0.55, alpha=0.95)


def _load_normalizer(path: Path) -> GaussianNormalizer:
    normalizer = GaussianNormalizer(np.zeros((1, 1, AMBIENT_DIM)))
    npz = np.load(path)
    normalizer.mean = npz["mean"]
    normalizer.std = npz["std"]
    return normalizer


def ambient_to_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return x @ basis


def subspace_to_ambient(u: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return u @ basis.T


def _generate_shared_initial_xts(
    args,
    n_paths: int,
) -> tuple[torch.Tensor, list[int]]:
    """One initial noise tensor per path, each from a distinct reproducible seed."""
    path_seeds = [args.seed + i for i in range(n_paths)]
    chunks: list[torch.Tensor] = []
    for path_seed in path_seeds:
        gen = torch.Generator(device=args.device)
        gen.manual_seed(path_seed)
        chunks.append(
            torch.randn((1, HORIZON, AMBIENT_DIM), device=args.device, generator=gen)
            * args.temperature
        )
    return torch.cat(chunks, dim=0), path_seeds


def _test_params(args, test_name: str) -> tuple[float, float, float]:
    if test_name == "test_a":
        return args.w_cg, args.opt_scale, args.plot_limit_a
    return args.w_cg_test_b, args.opt_scale_test_b, args.plot_limit_b


def _slice_grid(geometry: GeometryMeta, lim: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = np.linspace(-lim, lim, n)
    u2 = np.linspace(-lim, lim, n)
    u1g, u2g = np.meshgrid(u1, u2)
    sub_dim = geometry.subspace_dim
    u = np.zeros((n * n, sub_dim), dtype=np.float64)
    u[:, 0] = u1g.ravel()
    u[:, 1] = u2g.ravel()
    x = subspace_to_ambient(u, geometry.basis_t)
    rewards = reward_values(x, geometry).reshape(n, n)
    return u1g, u2g, rewards


def _extract_ambient_history(log: dict, normalizer: GaussianNormalizer) -> np.ndarray:
    """Return physical ambient trajectory with shape (steps+1, ambient_dim)."""
    hist = log["sample_history"]
    if hist.ndim == 5:
        x_norm = hist[0, :, 0, 0, :]
    elif hist.ndim == 4:
        x_norm = hist[0, :, 0, :]
    else:
        x_norm = hist[0]
    if x_norm.ndim == 3:
        x_norm = x_norm[:, 0, :]
    return normalizer.unnormalize(x_norm)


def _extract_ambient_histories_batch(log: dict, normalizer: GaussianNormalizer) -> np.ndarray:
    """Return physical ambient trajectories with shape (n_paths, steps+1, ambient_dim)."""
    hist = log["sample_history"]
    if hist.ndim == 5:
        x_norm = hist[:, :, 0, 0, :]
    elif hist.ndim == 4:
        x_norm = hist[:, :, 0, :]
    else:
        x_norm = hist
    if x_norm.ndim == 4:
        x_norm = x_norm[:, :, 0, :]
    return normalizer.unnormalize(x_norm)


def _extract_subspace_history(
    log: dict,
    normalizer: GaussianNormalizer,
    basis: np.ndarray,
) -> np.ndarray:
    return ambient_to_subspace(_extract_ambient_history(log, normalizer), basis)


def _metrics_along_history(
    ambient_hist: np.ndarray,
    geometry: GeometryMeta,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = reward_values(ambient_hist, geometry)
    violations = constraint_violation(ambient_hist, geometry)
    return rewards, violations


def _append_drgd_history(
    ambient_hist: np.ndarray,
    log: dict,
    normalizer: GaussianNormalizer,
) -> np.ndarray:
    drgd_hist = log.get("drgd_history")
    if drgd_hist is None:
        return ambient_hist
    x_norm = np.asarray(drgd_hist)
    if x_norm.ndim == 5:
        x_norm = x_norm[0, :, 0, 0, :]
    elif x_norm.ndim == 4:
        x_norm = x_norm[0, :, 0, :]
    elif x_norm.ndim == 3:
        x_norm = x_norm[0]
    drgd_phys = normalizer.unnormalize(x_norm)
    if drgd_phys.ndim == 3:
        drgd_phys = drgd_phys[:, 0, :]
    return np.concatenate([ambient_hist, drgd_phys[1:]], axis=0)


def sample_paths_with_history(
    args,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
    agent,
    reward_fn,
    *,
    guidance_mode: str,
    w_cg: float,
    opt_scale: float,
    initial_xts: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return subspace paths, reward curves, violation curves, and final metrics."""
    uses_classifier = guidance_mode in ("standard", "optimization", *GGDM_GUIDANCE_MODES, "drgd") and (
        w_cg != 0.0
        or opt_scale != 0.0
        or guidance_mode in (*GGDM_GUIDANCE_MODES, "drgd")
    )
    if uses_classifier:
        agent.classifier = RuntimeRewardClassifier(reward_fn, device=args.device)
    else:
        agent.classifier = None

    n_paths = initial_xts.shape[0]
    prior = torch.zeros((1, HORIZON, AMBIENT_DIM), device=args.device)
    subspace_paths: list[np.ndarray] = []
    reward_curves: list[np.ndarray] = []
    violation_curves: list[np.ndarray] = []
    final_trajs: list[torch.Tensor] = []

    if guidance_mode == "ggdm_g_loss_mean":
        ggdm_solver = getattr(args, "ggdm_solver", "ddpm")
        sample_kwargs = dict(
            solver=ggdm_solver,
            n_samples=n_paths,
            sample_steps=args.sampling_steps,
            sample_step_schedule=args.sample_step_schedule,
            use_ema=args.use_ema,
            w_cg=w_cg,
            guidance_mode=guidance_mode,
            optimization_guidance_scale=opt_scale,
            optimization_guidance_last_steps=args.optimization_guidance_last_steps,
            ggdm_interval=args.ggdm_interval,
            ggdm_obs_noise_sigma=args.ggdm_obs_noise_sigma,
            ggdm_beta_coef=args.ggdm_beta_coef,
            ggdm_rounds=args.ggdm_rounds,
            ggdm_round_seed=args.seed,
            drgd_steps=0,
            drgd_gamma=args.drgd_gamma,
            drgd_noise_index=args.drgd_noise_index,
            temperature=args.temperature,
            preserve_history=True,
            requires_grad=False,
            initial_xt=initial_xts,
        )
        traj, log = agent.sample(prior, **sample_kwargs)
        ambient_batch = _extract_ambient_histories_batch(log, normalizer)
        for i in range(n_paths):
            ambient_hist = ambient_batch[i]
            rewards, violations = _metrics_along_history(ambient_hist, geometry)
            subspace_paths.append(ambient_to_subspace(ambient_hist, geometry.basis_t))
            reward_curves.append(rewards)
            violation_curves.append(violations)
            final_trajs.append(traj[i : i + 1])
    else:
        for i in range(n_paths):
            ggdm_solver = getattr(args, "ggdm_solver", "ddpm")
            traj_solver = ggdm_solver if guidance_mode in GGDM_GUIDANCE_MODES else args.solver
            sample_kwargs = dict(
                solver=traj_solver,
                n_samples=1,
                sample_steps=args.sampling_steps,
                sample_step_schedule=args.sample_step_schedule,
                use_ema=args.use_ema,
                w_cg=w_cg,
                guidance_mode=guidance_mode,
                optimization_guidance_scale=opt_scale,
                optimization_guidance_last_steps=args.optimization_guidance_last_steps,
                ggdm_interval=args.ggdm_interval,
                ggdm_obs_noise_sigma=args.ggdm_obs_noise_sigma,
                ggdm_beta_coef=args.ggdm_beta_coef,
                drgd_steps=args.drgd_steps if guidance_mode == "drgd" else 0,
                drgd_gamma=args.drgd_gamma,
                drgd_noise_index=args.drgd_noise_index,
                ggdm_rounds=args.ggdm_rounds if guidance_mode in GGDM_GUIDANCE_MODES else 0,
                ggdm_round_seed=(args.seed + i) if guidance_mode in GGDM_GUIDANCE_MODES else None,
                temperature=args.temperature,
                preserve_history=True,
                requires_grad=False,
            )
            sample_kwargs["initial_xt"] = initial_xts[i : i + 1]

            traj, log = agent.sample(prior, **sample_kwargs)
            ambient_hist = _extract_ambient_history(log, normalizer)
            ambient_hist = _append_drgd_history(ambient_hist, log, normalizer)
            rewards, violations = _metrics_along_history(ambient_hist, geometry)
            subspace_paths.append(ambient_to_subspace(ambient_hist, geometry.basis_t))
            reward_curves.append(rewards)
            violation_curves.append(violations)
            final_trajs.append(traj)

    paths = np.stack(subspace_paths, axis=0)
    reward_curves = np.stack(reward_curves, axis=0)
    violation_curves = np.stack(violation_curves, axis=0)
    metrics = _sample_batch_metrics(torch.cat(final_trajs, dim=0), geometry, normalizer)
    return paths, reward_curves, violation_curves, metrics


def _plot_feasible_region(ax, geometry: GeometryMeta) -> None:
    if geometry.test == "test_a":
        half_w = float(geometry.square_half_width)
        patch = Rectangle(
            (-half_w, -half_w),
            2 * half_w,
            2 * half_w,
            facecolor=FEASIBLE_FACE,
            alpha=FEASIBLE_FACE_ALPHA,
            edgecolor=FEASIBLE_EDGE,
            linewidth=FEASIBLE_EDGE_WIDTH,
            linestyle="-",
            zorder=2,
            label="Feasible region",
        )
    else:
        axes = np.asarray(geometry.ellipsoid_axes, dtype=np.float64)
        patch = Ellipse(
            (0.0, 0.0),
            width=2.0 * axes[0],
            height=2.0 * axes[1],
            facecolor=FEASIBLE_FACE,
            alpha=FEASIBLE_FACE_ALPHA,
            edgecolor=FEASIBLE_EDGE,
            linewidth=FEASIBLE_EDGE_WIDTH,
            linestyle="-",
            zorder=2,
            label="Feasible region",
        )
    ax.add_patch(patch)


def _plot_backward_paths(
    ax,
    paths: np.ndarray,
    *,
    color: str,
    label: str,
    arrow_stride: int,
) -> None:
    n_paths, n_steps, _ = paths.shape
    for i in range(n_paths):
        p = paths[i]
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.4, alpha=0.78, zorder=5)
        ax.scatter(
            p[0, 0],
            p[0, 1],
            color=INIT_COLOR,
            s=16,
            edgecolors="white",
            linewidths=0.4,
            zorder=6,
        )
        ax.scatter(
            p[-1, 0],
            p[-1, 1],
            color=color,
            s=28,
            marker="x",
            linewidths=1.1,
            zorder=6,
        )
        for t in range(0, n_steps - 1, arrow_stride):
            ax.annotate(
                "",
                xy=(p[t + 1, 0], p[t + 1, 1]),
                xytext=(p[t, 0], p[t, 1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, alpha=0.55, shrinkA=0, shrinkB=0),
            )
    ax.plot([], [], color=color, linewidth=2.0, label=label)


def _save_figure(fig, out_dir: Path, fig_name: str) -> None:
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{fig_name}.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None)
        print(f"Wrote {out_path}")


def _trajectory_cache_paths(root: Path, fig_name: str) -> tuple[Path, Path]:
    cache_dir = root / "trajectories"
    return cache_dir / f"{fig_name}.npz", cache_dir / f"{fig_name}.meta.json"


def _sampling_metadata(args, test_name: str, w_cg: float, opt_scale: float) -> dict:
    return {
        "test": test_name,
        "seed": args.seed,
        "n_paths": args.n_paths,
        "n_paths_plot": args.n_paths_plot,
        "shared_initialization": True,
        "w_cg": w_cg,
        "opt_scale": opt_scale,
        "diffusion_steps": args.diffusion_steps,
        "training_diffusion_steps": args.training_diffusion_steps,
        "sampling_steps": args.sampling_steps,
        "optimization_guidance_last_steps": args.optimization_guidance_last_steps,
        "solver": args.solver,
        "sample_step_schedule": args.sample_step_schedule,
        "temperature": args.temperature,
        "use_ema": args.use_ema,
        "predict_noise": args.predict_noise,
        "ggdm_interval": args.ggdm_interval,
        "ggdm_obs_noise_sigma": args.ggdm_obs_noise_sigma,
        "ggdm_rounds": args.ggdm_rounds,
        "ggdm_beta_coef": args.ggdm_beta_coef,
        "drgd_steps": args.drgd_steps,
        "drgd_gamma": args.drgd_gamma,
        "drgd_noise_index": args.drgd_noise_index,
    }


def _save_trajectory_cache(
    cache_npz: Path,
    cache_meta: Path,
    run_data: dict,
    metadata: dict,
) -> None:
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for mode in PLOT_METHODS:
        payload = run_data["methods"][mode]
        for key in ("paths", "reward_curves", "violation_curves"):
            arrays[f"{mode}_{key}"] = payload[key]
    arrays["feasible_optimum_subspace"] = np.asarray(
        run_data["feasible_optimum"]["subspace"], dtype=np.float64
    )
    arrays["feasible_optimum_ambient"] = np.asarray(
        run_data["feasible_optimum"]["ambient"], dtype=np.float64
    )
    if "initial_xts" in run_data:
        arrays["initial_xts"] = np.asarray(run_data["initial_xts"], dtype=np.float32)
    np.savez_compressed(cache_npz, **arrays)

    meta_payload = {
        **metadata,
        "path_seeds": run_data.get("path_seeds"),
        "fig_name": run_data["fig_name"],
        "feasible_optimum": run_data["feasible_optimum"],
        "methods": {
            mode: {
                "label": run_data["methods"][mode]["label"],
                "metrics": run_data["methods"][mode]["metrics"],
            }
            for mode in PLOT_METHODS
        },
    }
    with open(cache_meta, "w") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"Wrote trajectory cache {cache_npz}")


def _legacy_fig_name(test_name: str) -> str:
    return f"{test_name}_{LEGACY_PCG_DCG_SUFFIX}"


def _load_legacy_pcg_dcg(
    args,
    test_name: str,
    w_cg: float,
    opt_scale: float,
) -> dict | None:
    """Load frozen PCG/DCG trajectories from the original two-method paper cache."""
    root = Path(args.output_root) / test_name
    fig_name = _legacy_fig_name(test_name)
    cache_npz, cache_meta = _trajectory_cache_paths(root, fig_name)
    if not cache_npz.is_file() or not cache_meta.is_file():
        return None

    metadata = _sampling_metadata(args, test_name, w_cg, opt_scale)
    with open(cache_meta) as f:
        cached = json.load(f)
    for key in LEGACY_METADATA_KEYS:
        if cached.get(key) != metadata[key]:
            print(
                f"[{test_name}] Legacy PCG/DCG cache mismatch on {key!r}: "
                f"{cached.get(key)!r} != {metadata[key]!r}"
            )
            return None

    data = np.load(cache_npz)
    methods: dict = {}
    for mode in ("standard", "optimization"):
        methods[mode] = {
            "label": cached["methods"][mode]["label"],
            "paths": data[f"{mode}_paths"],
            "reward_curves": data[f"{mode}_reward_curves"],
            "violation_curves": data[f"{mode}_violation_curves"],
            "metrics": cached["methods"][mode]["metrics"],
        }
    print(f"[{test_name}] Loaded legacy PCG/DCG trajectories from {cache_npz}")
    return methods


def _load_trajectory_cache(cache_npz: Path, cache_meta: Path, metadata: dict) -> dict | None:
    if not cache_npz.is_file() or not cache_meta.is_file():
        return None
    with open(cache_meta) as f:
        cached = json.load(f)
    for key, value in metadata.items():
        if cached.get(key) != value:
            return None

    data = np.load(cache_npz)
    run_data = {
        "test": cached["test"],
        "w_cg": cached["w_cg"],
        "opt_scale": cached["opt_scale"],
        "fig_name": cached["fig_name"],
        "feasible_optimum": cached["feasible_optimum"],
        "methods": {},
    }
    for mode in PLOT_METHODS:
        run_data["methods"][mode] = {
            "label": METHOD_STYLES[mode]["label"],
            "paths": data[f"{mode}_paths"],
            "reward_curves": data[f"{mode}_reward_curves"],
            "violation_curves": data[f"{mode}_violation_curves"],
            "metrics": cached["methods"][mode]["metrics"],
        }
    print(f"Loaded trajectory cache {cache_npz}")
    return run_data


def _load_trajectory_cache_relaxed(cache_npz: Path, cache_meta: Path) -> dict | None:
    """Load a trajectory cache without requiring baseline hyperparameters to match."""
    if not cache_npz.is_file() or not cache_meta.is_file():
        return None
    with open(cache_meta) as f:
        cached = json.load(f)
    data = np.load(cache_npz)
    run_data = {
        "test": cached["test"],
        "w_cg": cached["w_cg"],
        "opt_scale": cached["opt_scale"],
        "fig_name": cached["fig_name"],
        "feasible_optimum": cached["feasible_optimum"],
        "methods": {},
    }
    for mode in PLOT_METHODS:
        run_data["methods"][mode] = {
            "label": cached["methods"][mode]["label"],
            "paths": data[f"{mode}_paths"],
            "reward_curves": data[f"{mode}_reward_curves"],
            "violation_curves": data[f"{mode}_violation_curves"],
            "metrics": cached["methods"][mode]["metrics"],
        }
    print(f"Loaded trajectory cache {cache_npz} (relaxed metadata)")
    return run_data


def _partial_resample_run_data(
    args,
    test_name: str,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
    *,
    w_cg: float,
    opt_scale: float,
    fig_name: str,
    resample_methods: tuple[str, ...],
    cache_npz: Path,
    cache_meta: Path,
) -> dict:
    run_data = _load_trajectory_cache_relaxed(cache_npz, cache_meta)
    if run_data is None:
        raise FileNotFoundError(f"Missing trajectory cache for partial resample: {cache_npz}")

    if args.use_legacy_pcg_dcg:
        legacy_methods = _load_legacy_pcg_dcg(args, test_name, w_cg, opt_scale)
        if legacy_methods is not None:
            run_data["methods"].update(legacy_methods)

    ckpt_paths = _resolve_paths(test_name, Path(args.output_root))
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(ckpt_paths["ckpt"]))
    agent.eval()

    with open(cache_meta) as f:
        cached = json.load(f)
    cached_npz = np.load(cache_npz)
    if "initial_xts" in cached_npz:
        initial_xts = torch.from_numpy(cached_npz["initial_xts"]).to(args.device)
        path_seeds = cached.get("path_seeds")
    else:
        initial_xts, path_seeds = _generate_shared_initial_xts(args, args.n_paths)
    run_data["path_seeds"] = path_seeds
    run_data["initial_xts"] = initial_xts.detach().cpu().numpy()

    traj_specs = {
        "standard": (w_cg, 0.0),
        "optimization": (0.0, opt_scale),
        "ggdm_g_loss_mean": (0.0, 0.0),
        "drgd": (0.0, 0.0),
    }
    for mode in resample_methods:
        if mode not in traj_specs:
            raise ValueError(f"Unknown resample method {mode!r}")
        w_val, opt_val = traj_specs[mode]
        paths, reward_curves, violation_curves, metrics = sample_paths_with_history(
            args,
            geometry,
            normalizer,
            agent,
            reward_fn,
            guidance_mode=mode,
            w_cg=w_val,
            opt_scale=opt_val,
            initial_xts=initial_xts,
        )
        run_data["methods"][mode] = {
            "label": METHOD_STYLES[mode]["label"],
            "paths": paths,
            "reward_curves": reward_curves,
            "violation_curves": violation_curves,
            "metrics": metrics,
        }
        extra = ""
        if mode in GGDM_GUIDANCE_MODES:
            extra = (
                f", ggdm_interval={args.ggdm_interval:g}, "
                f"ggdm_beta={args.ggdm_beta_coef:g}, ggdm_rounds={args.ggdm_rounds}"
            )
        elif mode == "drgd":
            extra = f", drgd_steps={args.drgd_steps}, drgd_gamma={args.drgd_gamma:g}"
        print(
            f"[{test_name}] Resampled {METHOD_STYLES[mode]['label']} "
            f"(w_cg={w_val:g}, opt={opt_val:g}{extra}): "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}  "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )
    return run_data


def _sample_run_data(
    args,
    test_name: str,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
    *,
    w_cg: float,
    opt_scale: float,
    fig_name: str,
) -> dict:
    traj_specs = [
        ("standard", w_cg, 0.0),
        ("optimization", 0.0, opt_scale),
        ("ggdm_g_loss_mean", 0.0, 0.0),
        ("drgd", 0.0, 0.0),
    ]

    ckpt_paths = _resolve_paths(test_name, Path(args.output_root))
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(ckpt_paths["ckpt"]))
    agent.eval()

    initial_xts, path_seeds = _generate_shared_initial_xts(args, args.n_paths)

    x_opt, u_opt = solve_optimal_feasible(geometry)
    opt_reward = float(reward_values(x_opt[None], geometry)[0])
    opt_violation = float(constraint_violation(x_opt[None], geometry)[0])
    run_data: dict = {
        "test": test_name,
        "w_cg": w_cg,
        "opt_scale": opt_scale,
        "fig_name": fig_name,
        "path_seeds": path_seeds,
        "initial_xts": initial_xts.detach().cpu().numpy(),
        "feasible_optimum": {
            "ambient": x_opt.tolist(),
            "subspace": u_opt.tolist(),
            "reward": opt_reward,
            "violation": opt_violation,
        },
        "methods": {},
    }
    print(
        f"[{test_name}] Shared init: {args.n_paths} paths "
        f"(seeds {path_seeds[0]}..{path_seeds[-1]}), plot first {args.n_paths_plot}"
    )
    print(
        f"[{test_name}] Feasible optimum: u={u_opt[:2]}, "
        f"reward={opt_reward:.4f}, violation={opt_violation:.6f}"
    )

    legacy_methods = None
    if args.use_legacy_pcg_dcg and not args.resample_legacy:
        legacy_methods = _load_legacy_pcg_dcg(args, test_name, w_cg, opt_scale)

    for mode, w_val, opt_val in traj_specs:
        if legacy_methods is not None and mode in legacy_methods:
            run_data["methods"][mode] = legacy_methods[mode]
            metrics = legacy_methods[mode]["metrics"]
            label = legacy_methods[mode]["label"]
            extra = ""
        else:
            paths, reward_curves, violation_curves, metrics = sample_paths_with_history(
                args,
                geometry,
                normalizer,
                agent,
                reward_fn,
                guidance_mode=mode,
                w_cg=w_val,
                opt_scale=opt_val,
                initial_xts=initial_xts,
            )
            style = METHOD_STYLES[mode]
            run_data["methods"][mode] = {
                "label": style["label"],
                "paths": paths,
                "reward_curves": reward_curves,
                "violation_curves": violation_curves,
                "metrics": metrics,
            }
            label = style["label"]
            extra = ""
            if mode in GGDM_GUIDANCE_MODES:
                extra = (
                    f", ggdm_interval={args.ggdm_interval:g}, "
                    f"ggdm_beta={args.ggdm_beta_coef:g}, ggdm_rounds={args.ggdm_rounds}"
                )
            elif mode == "drgd":
                extra = f", drgd_steps={args.drgd_steps}, drgd_gamma={args.drgd_gamma:g}"
        if legacy_methods is not None and mode in legacy_methods:
            if mode == "standard":
                extra = f", w_cg={w_val:g} [legacy cache]"
            elif mode == "optimization":
                extra = f", opt={opt_val:g} [legacy cache]"
        print(
            f"[{test_name}] {label} (w_cg={w_val:g}, opt={opt_val:g}{extra}): "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}  "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )
    return run_data


def _render_paths_figure(
    args,
    test_name: str,
    geometry: GeometryMeta,
    plot_limit: float,
    run_data: dict,
) -> None:
    u1g, u2g, rewards = _slice_grid(geometry, plot_limit, args.grid_resolution)
    arrow_stride = max(1, args.sampling_steps // 10)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    levels = np.linspace(rewards.min(), min(0.0, rewards.max()), 20)
    cf = ax.contourf(u1g, u2g, rewards, levels=levels, cmap="bone", alpha=0.55, zorder=1)
    ax.contour(u1g, u2g, rewards, levels=levels[::2], colors="#BDBDBD", linewidths=0.5, alpha=0.8, zorder=1)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=PATH_LEGEND_FONTSIZE)

    _plot_feasible_region(ax, geometry)
    u_opt = np.asarray(run_data["feasible_optimum"]["subspace"], dtype=np.float64)
    ax.scatter(
        u_opt[0],
        u_opt[1],
        marker="*",
        s=160,
        color=OPTIMAL_COLOR,
        edgecolors="white",
        linewidths=0.7,
        zorder=7,
        label="Optimal (feasible)",
    )

    n_plot = min(args.n_paths_plot, run_data["methods"][PATH_PLOT_METHODS[0]]["paths"].shape[0])
    for mode in PATH_PLOT_METHODS:
        style = METHOD_STYLES[mode]
        _plot_backward_paths(
            ax,
            run_data["methods"][mode]["paths"][:n_plot],
            color=style["color"],
            label=style["label"],
            arrow_stride=arrow_stride,
        )

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="upper left",
        fontsize=PATH_LEGEND_FONTSIZE,
        markerscale=0.8,
        handlelength=1.5,
        borderpad=0.35,
        labelspacing=0.3,
        frameon=True,
        framealpha=LEGEND_FRAME_ALPHA,
        edgecolor="#CCCCCC",
    )
    _style_axis_grid(ax, minor=True)

    out_dir = Path(args.output_root) / test_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_dir, run_data["fig_name"])
    plt.close(fig)


def load_run_data(args, test_name: str) -> dict:
    """Load cached trajectories or sample if needed."""
    w_cg, opt_scale, _plot_limit = _test_params(args, test_name)
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    normalizer = _load_normalizer(root / "normalizer.npz")

    fig_name = args.output_name or (
        f"{test_name}_guidance_paper_s{args.sampling_steps}_last{args.optimization_guidance_last_steps}"
    )
    metadata = _sampling_metadata(args, test_name, w_cg, opt_scale)
    cache_npz, cache_meta = _trajectory_cache_paths(root, fig_name)

    run_data = None
    if args.resample_methods:
        if args.plot_only or args.convergence_only or args.paths_only:
            raise ValueError("--resample-methods requires GPU sampling; omit plot-only flags.")
        run_data = _partial_resample_run_data(
            args,
            test_name,
            geometry,
            normalizer,
            w_cg=w_cg,
            opt_scale=opt_scale,
            fig_name=fig_name,
            resample_methods=tuple(args.resample_methods),
            cache_npz=cache_npz,
            cache_meta=cache_meta,
        )
        _save_trajectory_cache(cache_npz, cache_meta, run_data, metadata)
    elif not args.resample:
        run_data = _load_trajectory_cache(cache_npz, cache_meta, metadata)

    if run_data is None:
        if args.plot_only or args.convergence_only or args.paths_only:
            raise FileNotFoundError(
                f"No trajectory cache for {test_name} at {cache_npz}. "
                "Run without --plot-only/--convergence-only first (or pass --resample)."
            )
        run_data = _sample_run_data(
            args,
            test_name,
            geometry,
            normalizer,
            w_cg=w_cg,
            opt_scale=opt_scale,
            fig_name=fig_name,
        )
        _save_trajectory_cache(cache_npz, cache_meta, run_data, metadata)
    return run_data


def plot_paths(args, test_name: str) -> dict:
    _apply_paper_style()
    set_seed(args.seed)

    w_cg, opt_scale, plot_limit = _test_params(args, test_name)
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")

    run_data = load_run_data(args, test_name)
    if not args.convergence_only:
        _render_paths_figure(args, test_name, geometry, plot_limit, run_data)
    return run_data


def _violation_tick_values(ymin: float, ymax: float) -> list[float]:
    ticks: list[float] = []
    p_lo = int(np.floor(np.log10(ymin))) - 1
    p_hi = int(np.ceil(np.log10(ymax))) + 1
    for p in range(p_lo, p_hi + 1):
        for mult in (1.0, 2.0, 5.0):
            val = mult * (10.0**p)
            if ymin <= val <= ymax:
                ticks.append(val)
    return ticks


def _violation_tick_label(value: float, _pos: int) -> str:
    exp = int(np.floor(np.log10(value)))
    coeff = value / (10.0**exp)
    if abs(coeff - 1.0) < 1e-6:
        return rf"$10^{{{exp}}}$"
    return rf"${coeff:g}\times10^{{{exp}}}$"


def _configure_violation_axis(ax) -> None:
    """Log-scale violation axis: keep autoscale y-limits, 1-2-5 ticks inside view."""
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ticks = _violation_tick_values(ymin, ymax)
    if ticks:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_violation_tick_label))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.set_ylim(ymin, ymax)


def plot_convergence(args, test_name: str, run_data: dict) -> None:
    _apply_paper_style()
    opt_objective = -float(run_data["feasible_optimum"]["reward"])
    viol_floor = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    panels = [
        ("reward_curves", "Objective", axes[0]),
        ("violation_curves", "Constraint violation", axes[1]),
    ]

    for key, subtitle, ax in panels:
        for mode in CONVERGENCE_METHODS:
            curves = run_data["methods"][mode][key]
            style = METHOD_STYLES[mode]
            curve_steps = np.arange(curves.shape[1])
            if key == "reward_curves":
                curves = -curves
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            if key == "violation_curves":
                mean = np.maximum(mean, viol_floor)
                mean_lo = np.maximum(mean - std, viol_floor)
                mean_hi = mean + std
            else:
                mean_lo = mean - std
                mean_hi = mean + std
            ax.plot(curve_steps, mean, color=style["color"], linewidth=2.6, label=style["label"])
            ax.fill_between(curve_steps, mean_lo, mean_hi, color=style["color"], alpha=0.18)
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
        ax.set_xlabel("Step (reverse diffusion / DRGD refine)")
        ax.set_title(subtitle)
        _style_axis_grid(ax, minor=True)

    axes[0].legend(loc="best", frameon=True, framealpha=LEGEND_FRAME_ALPHA, edgecolor="#CCCCCC")
    fig.tight_layout()

    out_dir = Path(args.output_root) / test_name / "figures"
    conv_name = f"{run_data['fig_name']}_convergence"
    _save_figure(fig, out_dir, conv_name)
    plt.close(fig)


def _ggdm_comparison_fig_name(args, test_name: str) -> str:
    if args.output_name:
        return args.output_name
    return (
        f"{test_name}_ggdm_particle_vs_mean_s{args.sampling_steps}"
        f"_last{args.optimization_guidance_last_steps}"
    )


def _paper_fig_name(args, test_name: str) -> str:
    return (
        f"{test_name}_guidance_paper_s{args.sampling_steps}"
        f"_last{args.optimization_guidance_last_steps}"
    )


def _metrics_from_final_curves(
    reward_curves: np.ndarray,
    violation_curves: np.ndarray,
) -> dict:
    final_rewards = reward_curves[:, -1]
    final_violations = violation_curves[:, -1]
    return {
        "reward_mean": float(final_rewards.mean()),
        "reward_std": float(final_rewards.std()),
        "violation_mean": float(final_violations.mean()),
        "violation_std": float(final_violations.std()),
        "violation_max": float(final_violations.max()),
    }


def _sample_ggdm_comparison_run_data(
    args,
    test_name: str,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
    *,
    w_cg: float,
    opt_scale: float,
    fig_name: str,
    paper_cache_npz: Path,
    paper_cache_meta: Path,
) -> dict:
    base_data = _load_trajectory_cache_relaxed(paper_cache_npz, paper_cache_meta)
    if base_data is None:
        raise FileNotFoundError(
            f"Missing paper trajectory cache for GGDM comparison: {paper_cache_npz}"
        )

    ckpt_paths = _resolve_paths(test_name, Path(args.output_root))
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(ckpt_paths["ckpt"]))
    agent.eval()

    initial_xts, path_seeds = _generate_shared_initial_xts(args, args.n_paths)

    run_data = {
        "test": test_name,
        "w_cg": w_cg,
        "opt_scale": opt_scale,
        "fig_name": fig_name,
        "path_seeds": path_seeds,
        "initial_xts": initial_xts.detach().cpu().numpy(),
        "feasible_optimum": base_data["feasible_optimum"],
        "methods": {},
    }
    dcg_payload = base_data["methods"]["optimization"]
    n_dcg = dcg_payload["paths"].shape[0]
    if n_dcg != args.n_paths:
        paths, reward_curves, violation_curves, metrics = sample_paths_with_history(
            args,
            geometry,
            normalizer,
            agent,
            reward_fn,
            guidance_mode="optimization",
            w_cg=0.0,
            opt_scale=opt_scale,
            initial_xts=initial_xts,
        )
        dcg_payload = {
            "label": METHOD_STYLES["optimization"]["label"],
            "paths": paths,
            "reward_curves": reward_curves,
            "violation_curves": violation_curves,
            "metrics": metrics,
        }
        print(
            f"[{test_name}] GGDM comparison resampled {dcg_payload['label']} "
            f"(opt={opt_scale:g}, n_paths={args.n_paths}): "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}  "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )
    elif n_dcg > args.n_paths:
        dcg_payload = {
            **dcg_payload,
            "paths": dcg_payload["paths"][: args.n_paths],
            "reward_curves": dcg_payload["reward_curves"][: args.n_paths],
            "violation_curves": dcg_payload["violation_curves"][: args.n_paths],
            "metrics": _metrics_from_final_curves(
                dcg_payload["reward_curves"][: args.n_paths],
                dcg_payload["violation_curves"][: args.n_paths],
            ),
        }
    run_data["methods"]["optimization"] = dcg_payload

    for mode in GGDM_COMPARISON_METHODS:
        paths, reward_curves, violation_curves, metrics = sample_paths_with_history(
            args,
            geometry,
            normalizer,
            agent,
            reward_fn,
            guidance_mode=mode,
            w_cg=0.0,
            opt_scale=0.0,
            initial_xts=initial_xts,
        )
        run_data["methods"][mode] = {
            "label": METHOD_STYLES[mode]["label"],
            "paths": paths,
            "reward_curves": reward_curves,
            "violation_curves": violation_curves,
            "metrics": metrics,
        }
        print(
            f"[{test_name}] GGDM comparison resampled {METHOD_STYLES[mode]['label']} "
            f"(delta={args.ggdm_interval:g}, beta={args.ggdm_beta_coef:g}, "
            f"rounds={args.ggdm_rounds}, n_paths={args.n_paths}): "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}  "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )
    return run_data


def _render_ggdm_comparison_paths_figure(
    args,
    test_name: str,
    geometry: GeometryMeta,
    plot_limit: float,
    run_data: dict,
) -> None:
    u1g, u2g, rewards = _slice_grid(geometry, plot_limit, args.grid_resolution)
    arrow_stride = max(1, args.sampling_steps // 10)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    levels = np.linspace(rewards.min(), min(0.0, rewards.max()), 20)
    cf = ax.contourf(u1g, u2g, rewards, levels=levels, cmap="bone", alpha=0.55, zorder=1)
    ax.contour(u1g, u2g, rewards, levels=levels[::2], colors="#BDBDBD", linewidths=0.5, alpha=0.8, zorder=1)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=PATH_LEGEND_FONTSIZE)

    _plot_feasible_region(ax, geometry)
    u_opt = np.asarray(run_data["feasible_optimum"]["subspace"], dtype=np.float64)
    ax.scatter(
        u_opt[0],
        u_opt[1],
        marker="*",
        s=160,
        color=OPTIMAL_COLOR,
        edgecolors="white",
        linewidths=0.7,
        zorder=7,
        label="Optimal (feasible)",
    )

    dcg_style = METHOD_STYLES["optimization"]
    _plot_backward_paths(
        ax,
        run_data["methods"]["optimization"]["paths"],
        color=dcg_style["color"],
        label=dcg_style["label"],
        arrow_stride=arrow_stride,
    )

    for mode in GGDM_COMPARISON_METHODS:
        style = METHOD_STYLES[mode]
        _plot_backward_paths(
            ax,
            run_data["methods"][mode]["paths"],
            color=style["color"],
            label=style["label"],
            arrow_stride=arrow_stride,
        )

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="upper left",
        fontsize=PATH_LEGEND_FONTSIZE,
        markerscale=0.8,
        handlelength=1.5,
        borderpad=0.35,
        labelspacing=0.3,
        frameon=True,
        framealpha=LEGEND_FRAME_ALPHA,
        edgecolor="#CCCCCC",
    )
    _style_axis_grid(ax, minor=True)

    out_dir = Path(args.output_root) / test_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_dir, run_data["fig_name"])
    plt.close(fig)


def _plot_ggdm_comparison_convergence(args, test_name: str, run_data: dict) -> None:
    _apply_paper_style()
    opt_objective = -float(run_data["feasible_optimum"]["reward"])
    viol_floor = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    panels = [
        ("reward_curves", "Objective", axes[0]),
        ("violation_curves", "Constraint violation", axes[1]),
    ]
    plot_modes = ("optimization", *GGDM_COMPARISON_METHODS)

    for key, subtitle, ax in panels:
        for mode in plot_modes:
            curves = run_data["methods"][mode][key]
            style = METHOD_STYLES[mode]
            curve_steps = np.arange(curves.shape[1])
            if key == "reward_curves":
                curves = -curves
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            if key == "violation_curves":
                mean = np.maximum(mean, viol_floor)
                mean_lo = np.maximum(mean - std, viol_floor)
                mean_hi = mean + std
            else:
                mean_lo = mean - std
                mean_hi = mean + std
            ax.plot(curve_steps, mean, color=style["color"], linewidth=2.6, label=style["label"])
            ax.fill_between(curve_steps, mean_lo, mean_hi, color=style["color"], alpha=0.18)
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
        ax.set_xlabel("Step (reverse diffusion / DRGD refine)")
        ax.set_title(subtitle)
        _style_axis_grid(ax, minor=True)

    axes[0].legend(loc="best", frameon=True, framealpha=LEGEND_FRAME_ALPHA, edgecolor="#CCCCCC")
    fig.tight_layout()

    out_dir = Path(args.output_root) / test_name / "figures"
    conv_name = f"{run_data['fig_name']}_convergence"
    _save_figure(fig, out_dir, conv_name)
    plt.close(fig)


def _write_ggdm_comparison_metrics(args, test_name: str, run_data: dict) -> None:
    metrics_report = []
    for mode in ("optimization", *GGDM_COMPARISON_METHODS):
        payload = run_data["methods"][mode]
        metrics_report.append(
            {
                "test": test_name,
                "label": payload["label"],
                "guidance_mode": mode,
                "w_cg": run_data["w_cg"] if mode == "standard" else 0.0,
                "optimization_guidance_scale": run_data["opt_scale"] if mode == "optimization" else 0.0,
                "ggdm_interval": args.ggdm_interval if mode in GGDM_GUIDANCE_MODES else 0.0,
                "ggdm_beta_coef": args.ggdm_beta_coef if mode in GGDM_GUIDANCE_MODES else 0.0,
                "ggdm_rounds": args.ggdm_rounds if mode in GGDM_GUIDANCE_MODES else 0,
                "drgd_steps": args.drgd_steps if mode == "drgd" else 0,
                "drgd_gamma": args.drgd_gamma if mode == "drgd" else 0.0,
                "n_paths": args.n_paths,
                "shared_initial_seed": args.seed,
                **payload["metrics"],
            }
        )
    out_path = Path(args.output_root) / test_name / "figures" / f"{run_data['fig_name']}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "feasible_optimum": run_data.get("feasible_optimum"),
                "methods": metrics_report,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


def plot_ggdm_mean_comparison(args, test_name: str) -> dict:
    _apply_paper_style()
    set_seed(args.seed)

    w_cg, opt_scale, plot_limit = _test_params(args, test_name)
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    normalizer = _load_normalizer(root / "normalizer.npz")

    fig_name = _ggdm_comparison_fig_name(args, test_name)
    paper_cache_npz, paper_cache_meta = _trajectory_cache_paths(root, _paper_fig_name(args, test_name))
    comparison_cache_npz, comparison_cache_meta = _trajectory_cache_paths(root, fig_name)

    run_data = None
    if not args.plot_only and not args.convergence_only and not args.paths_only:
        run_data = _sample_ggdm_comparison_run_data(
            args,
            test_name,
            geometry,
            normalizer,
            w_cg=w_cg,
            opt_scale=opt_scale,
            fig_name=fig_name,
            paper_cache_npz=paper_cache_npz,
            paper_cache_meta=paper_cache_meta,
        )
        metadata = _sampling_metadata(args, test_name, w_cg, opt_scale)
        metadata["comparison"] = "ggdm_particle_vs_mean"
        _save_ggdm_comparison_cache(comparison_cache_npz, comparison_cache_meta, run_data, metadata)
    else:
        run_data = _load_ggdm_comparison_cache(comparison_cache_npz, comparison_cache_meta)
        if run_data is None:
            raise FileNotFoundError(
                f"No GGDM comparison cache for {test_name} at {comparison_cache_npz}. "
                "Run with --compare-ggdm-mean first."
            )

    if not args.convergence_only:
        _render_ggdm_comparison_paths_figure(args, test_name, geometry, plot_limit, run_data)
    if not args.skip_convergence and not args.paths_only:
        _plot_ggdm_comparison_convergence(args, test_name, run_data)
    if not args.convergence_only and not args.paths_only:
        _write_ggdm_comparison_metrics(args, test_name, run_data)
    return run_data


def _save_ggdm_comparison_cache(
    cache_npz: Path,
    cache_meta: Path,
    run_data: dict,
    metadata: dict,
) -> None:
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for mode in ("optimization", *GGDM_COMPARISON_METHODS):
        payload = run_data["methods"][mode]
        for key in ("paths", "reward_curves", "violation_curves"):
            arrays[f"{mode}_{key}"] = payload[key]
    arrays["feasible_optimum_subspace"] = np.asarray(
        run_data["feasible_optimum"]["subspace"], dtype=np.float64
    )
    arrays["feasible_optimum_ambient"] = np.asarray(
        run_data["feasible_optimum"]["ambient"], dtype=np.float64
    )
    np.savez_compressed(cache_npz, **arrays)

    meta_payload = {
        **metadata,
        "fig_name": run_data["fig_name"],
        "feasible_optimum": run_data["feasible_optimum"],
        "methods": {
            mode: {
                "label": run_data["methods"][mode]["label"],
                "metrics": run_data["methods"][mode]["metrics"],
            }
            for mode in ("optimization", *GGDM_COMPARISON_METHODS)
        },
    }
    with open(cache_meta, "w") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"Saved GGDM comparison cache {cache_npz}")


def _load_ggdm_comparison_cache(cache_npz: Path, cache_meta: Path) -> dict | None:
    if not cache_npz.is_file() or not cache_meta.is_file():
        return None
    with open(cache_meta) as f:
        cached = json.load(f)
    data = np.load(cache_npz)
    run_data = {
        "test": cached["test"],
        "w_cg": cached["w_cg"],
        "opt_scale": cached["opt_scale"],
        "fig_name": cached["fig_name"],
        "feasible_optimum": cached["feasible_optimum"],
        "methods": {},
    }
    for mode in ("optimization", *GGDM_COMPARISON_METHODS):
        run_data["methods"][mode] = {
            "label": cached["methods"][mode]["label"],
            "paths": data[f"{mode}_paths"],
            "reward_curves": data[f"{mode}_reward_curves"],
            "violation_curves": data[f"{mode}_violation_curves"],
            "metrics": cached["methods"][mode]["metrics"],
        }
    print(f"Loaded GGDM comparison cache {cache_npz}")
    return run_data


def plot_training_data_sanity(args, test_name: str) -> None:
    """Scatter training data in (u1, u2) subspace coords with feasible region overlay."""
    _apply_paper_style()
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    dataset_path = root / "dataset.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing training dataset: {dataset_path}")

    data = np.load(dataset_path)["data"]
    basis = geometry.basis_t
    u = ambient_to_subspace(data.astype(np.float64), basis)

    if test_name == "test_a":
        plot_limit = args.plot_limit_a
    else:
        plot_limit = args.plot_limit_b

    max_pts = args.sanity_max_points
    if len(u) > max_pts:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(u), size=max_pts, replace=False)
        u_plot = u[idx]
    else:
        u_plot = u

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    _plot_feasible_region(ax, geometry)
    ax.scatter(
        u_plot[:, 0],
        u_plot[:, 1],
        s=4,
        alpha=0.25,
        color="#4D4D4D",
        edgecolors="none",
        rasterized=True,
        label=f"Training data ({len(u_plot):,} shown)",
        zorder=3,
    )

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Subspace coordinate $u_1$")
    ax.set_ylabel(r"Subspace coordinate $u_2$")
    title = "Test A" if test_name == "test_a" else "Test B"
    ax.set_title(f"{title}: training distribution vs feasible region")
    ax.legend(loc="upper right", frameon=True, framealpha=LEGEND_FRAME_ALPHA, edgecolor="#CCCCCC")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, alpha=0.8)

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_name = f"{test_name}_training_data_sanity"
    _save_figure(fig, out_dir, fig_name)
    plt.close(fig)
    print(f"[{test_name}] training samples={len(data):,}, plotted={len(u_plot):,}")


def _write_metrics(args, test_name: str, run_data: dict) -> None:
    metrics_report = []
    for mode, payload in run_data["methods"].items():
        metrics_report.append(
            {
                "test": test_name,
                "label": payload["label"],
                "guidance_mode": mode,
                "w_cg": run_data["w_cg"] if mode == "standard" else 0.0,
                "optimization_guidance_scale": run_data["opt_scale"] if mode == "optimization" else 0.0,
                "ggdm_interval": args.ggdm_interval if mode in GGDM_GUIDANCE_MODES else 0.0,
                "drgd_steps": args.drgd_steps if mode == "drgd" else 0,
                "drgd_gamma": args.drgd_gamma if mode == "drgd" else 0.0,
                "n_paths": args.n_paths,
                "shared_initial_seed": args.seed,
                **payload["metrics"],
            }
        )
    out_path = Path(args.output_root) / test_name / "figures" / f"{run_data['fig_name']}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "feasible_optimum": run_data.get("feasible_optimum"),
                "methods": metrics_report,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["test_a", "test_b", "both"], default="both")
    p.add_argument("--output-root", default="results/guidance_synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--training-diffusion-steps", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--optimization-guidance-last-steps", type=int, default=50)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--solver", default="ddim", help="Solver for PCG/DCG/DRGD.")
    p.add_argument("--ggdm-solver", default="ddpm", help="Solver for GGDM (GGDMOptim uses DDPM).")
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plot-limit-a", type=float, default=TEST_DEFAULTS["test_a"]["plot_limit"])
    p.add_argument("--plot-limit-b", type=float, default=TEST_DEFAULTS["test_b"]["plot_limit"])
    p.add_argument("--grid-resolution", type=int, default=200)
    p.add_argument(
        "--n-paths",
        type=int,
        default=100,
        help="Number of trajectories to sample (used for convergence statistics).",
    )
    p.add_argument(
        "--n-paths-plot",
        type=int,
        default=5,
        help="Number of trajectories to draw on the landscape/path figure.",
    )
    p.add_argument("--w-cg", type=float, default=TEST_DEFAULTS["test_a"]["w_cg"])
    p.add_argument("--opt-scale", type=float, default=TEST_DEFAULTS["test_a"]["opt_scale"])
    p.add_argument("--w-cg-test-b", type=float, default=TEST_DEFAULTS["test_b"]["w_cg"])
    p.add_argument("--opt-scale-test-b", type=float, default=TEST_DEFAULTS["test_b"]["opt_scale"])
    p.add_argument("--ggdm-interval", type=float, default=0.05, help="GGDM G_loss target increment delta.")
    p.add_argument("--ggdm-interval-test-b", type=float, default=0.05)
    p.add_argument("--ggdm-obs-noise-sigma", type=float, default=0.1)
    p.add_argument("--ggdm-rounds", type=int, default=10, help="Guided outer iterations after unguided landing (GGDMOptim opt_rounds).")
    p.add_argument("--ggdm-beta-coef", type=float, default=0.5)
    p.add_argument(
        "--no-tuned-baselines",
        action="store_true",
        help="Do not override GGDM/DRGD hyperparameters from baseline_tune_best.json.",
    )
    p.add_argument(
        "--tuned-drgd-only",
        action="store_true",
        help="Load only DRGD hyperparameters from baseline_tune_best.json (keep CLI GGDM settings).",
    )
    p.add_argument("--drgd-steps", type=int, default=50)
    p.add_argument("--drgd-gamma", type=float, default=10.0)
    p.add_argument("--drgd-noise-index", type=int, default=49)
    p.add_argument("--output-name", default=None)
    p.add_argument("--skip-convergence", action="store_true")
    p.add_argument(
        "--compare-ggdm-mean",
        action="store_true",
        help="Plot GGDM particle vs mean comparison (resample GGDM only; DCG from paper cache).",
    )
    p.add_argument(
        "--resample-methods",
        nargs="+",
        choices=SAMPLE_METHODS,
        default=None,
        help="Resample only these methods; load the rest from the saved trajectory cache.",
    )
    p.add_argument(
        "--resample",
        action="store_true",
        help="Force re-sampling external baselines and overwrite saved trajectory cache.",
    )
    p.add_argument(
        "--use-legacy-pcg-dcg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse PCG/DCG trajectories from the original pcg_dcg paper cache.",
    )
    p.add_argument(
        "--resample-legacy",
        action="store_true",
        help="Also resample PCG/DCG instead of loading the legacy cache.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from saved trajectories only (skip GPU sampling).",
    )
    p.add_argument(
        "--convergence-only",
        action="store_true",
        help="Replot convergence panels from cached trajectories only (fastest).",
    )
    p.add_argument(
        "--paths-only",
        action="store_true",
        help="Replot path figures from cached trajectories only.",
    )
    p.add_argument(
        "--sanity-data",
        action="store_true",
        help="Only plot training data vs feasible region (no GPU sampling).",
    )
    p.add_argument("--sanity-max-points", type=int, default=20_000)
    args = p.parse_args()
    if args.n_paths_plot > args.n_paths:
        raise ValueError(f"--n-paths-plot ({args.n_paths_plot}) cannot exceed --n-paths ({args.n_paths})")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def _apply_test_baseline_defaults(args, test_name: str) -> None:
    tune_path = Path(args.output_root) / "baseline_tune_best.json"
    if not tune_path.is_file():
        if test_name == "test_b" and args.ggdm_interval == 0.05:
            args.ggdm_interval = args.ggdm_interval_test_b
        return

    with open(tune_path) as f:
        tuned = json.load(f)
    test_cfg = tuned.get("tests", {}).get(test_name)
    if not test_cfg:
        if test_name == "test_b" and args.ggdm_interval == 0.05:
            args.ggdm_interval = args.ggdm_interval_test_b
        return

    if getattr(args, "tuned_drgd_only", False):
        args.drgd_steps = test_cfg["drgd_steps"]
        args.drgd_gamma = test_cfg["drgd_gamma"]
        args.drgd_noise_index = test_cfg.get("drgd_noise_index", 5)
        print(
            f"[{test_name}] Using tuned DRGD from {tune_path}: "
            f"drgd_steps={args.drgd_steps}, drgd_gamma={args.drgd_gamma:g}, "
            f"drgd_noise_index={args.drgd_noise_index}"
        )
        return

    if getattr(args, "no_tuned_baselines", False):
        return

    args.ggdm_interval = test_cfg["ggdm_interval"]
    args.ggdm_beta_coef = test_cfg.get("ggdm_beta_coef", args.ggdm_beta_coef)
    args.ggdm_rounds = test_cfg.get("ggdm_rounds", args.ggdm_rounds)
    args.drgd_steps = test_cfg["drgd_steps"]
    args.drgd_gamma = test_cfg["drgd_gamma"]
    args.drgd_noise_index = test_cfg.get("drgd_noise_index", 5)
    print(
        f"[{test_name}] Using tuned baselines from {tune_path}: "
        f"ggdm_interval={args.ggdm_interval:g}, ggdm_beta={args.ggdm_beta_coef:g}, "
        f"ggdm_rounds={args.ggdm_rounds}, "
        f"drgd_steps={args.drgd_steps}, drgd_gamma={args.drgd_gamma:g}, "
        f"drgd_noise_index={args.drgd_noise_index}"
    )
    if test_name == "test_b" and args.ggdm_interval == 0.05:
        args.ggdm_interval = args.ggdm_interval_test_b


def main():
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]

    if args.sanity_data:
        for test_name in tests:
            plot_training_data_sanity(args, test_name)
        return

    for test_name in tests:
        _apply_test_baseline_defaults(args, test_name)
        if args.compare_ggdm_mean:
            plot_ggdm_mean_comparison(args, test_name)
            continue
        run_data = plot_paths(args, test_name)
        if not args.skip_convergence and not args.paths_only:
            plot_convergence(args, test_name, run_data)
        if not args.convergence_only and not args.paths_only:
            _write_metrics(args, test_name, run_data)


if __name__ == "__main__":
    main()
