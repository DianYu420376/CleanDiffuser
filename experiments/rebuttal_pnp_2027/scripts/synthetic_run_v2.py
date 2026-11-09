#!/usr/bin/env python3
"""ICLR-2027 rebuttal, synthetic subspace tests A/B: re-run DCG / PDG / GGDM / DRGD with the
paper hyperparameters from the cached paper initializations (instrumented, full ambient
history), and add the fixed-noise PnP-PGD baseline x <- D_sigma_bar(x - eta grad f(x)) with
validation-only hyperparameter selection, ablations, cost counts and timing.

GPU compute only; all outputs go through ``rebuttal_pnp.safe_io`` into a fresh run dir under
``results/synthetic/``. Figures/tables are produced by ``synthetic_report.py`` from that dir.
Nothing is written into /u/rzhang26/CleanDiffuser (read-only inputs only).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REB))

import guidance_synthetic_subspace as gss  # noqa: E402
import guidance_synthetic_visualize_subspace as vis  # noqa: E402
from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier  # noqa: E402
from cleandiffuser.diffusion.pnp_pgd import (  # noqa: E402
    FixedNoiseDenoiser,
    pnp_pgd,
    reward_classifier_grad_f,
)
from cleandiffuser.utils import SUPPORTED_SAMPLING_STEP_SCHEDULE  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402
from rebuttal_pnp.instrument import GradCounter, NetworkCounter, cuda_time, sync, warmup  # noqa: E402
from rebuttal_pnp.synthetic_geometry import METRIC_KEYS, SyntheticGeometry  # noqa: E402
from utils import set_seed  # noqa: E402

D = gss.AMBIENT_DIM
H = gss.HORIZON


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CODE_FILES = (
    REB / "scripts" / "synthetic_run_v2.py",
    REB / "rebuttal_pnp" / "synthetic_geometry.py",
    REB / "rebuttal_pnp" / "safe_io.py",
    REB / "rebuttal_pnp" / "instrument.py",
    REB.parents[1] / "cleandiffuser" / "diffusion" / "pnp_pgd.py",
    REB.parents[1] / "cleandiffuser" / "classifier" / "runtime_reward.py",
    REB.parents[1] / "pipelines" / "guidance_synthetic_subspace.py",
    REB.parents[1] / "pipelines" / "guidance_synthetic_visualize_subspace.py",
)


def code_hashes() -> dict:
    """sha256 of the code that produced the run (scripts/ is gitignored, so the git commit alone is not enough)."""
    return {str(p): (sha256(p) if p.is_file() else None) for p in CODE_FILES}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Context per test
# --------------------------------------------------------------------------------------
class TestCtx:
    def __init__(self, cfg: dict, test: str, device: str, drgd_cfg: dict):
        self.cfg, self.test, self.device = cfg, test, device
        root = Path(cfg["orig_root"]) / test
        self.paths = {
            "geometry": root / "geometry.json",
            "normalizer": root / "normalizer.npz",
            "ckpt": root / "diffusion_ckpt_latest.pt",
            "cache_npz": root / "trajectories" / (cfg["paper_cache_name"].format(test=test) + ".npz"),
            "cache_meta": root / "trajectories" / (cfg["paper_cache_name"].format(test=test) + ".meta.json"),
        }
        self.geometry = gss._load_geometry(self.paths["geometry"])
        self.normalizer = vis._load_normalizer(self.paths["normalizer"])
        self.sg = SyntheticGeometry(self.geometry)
        m, s = cfg["model"], cfg["sampling"]
        self.args = SimpleNamespace(
            model_dim=m["model_dim"],
            diffusion_steps=m["diffusion_steps"],
            training_diffusion_steps=m["training_diffusion_steps"],
            ema_rate=m["ema_rate"],
            predict_noise=m["predict_noise"],
            device=device,
            seed=cfg["seed"],
            temperature=s["temperature"],
        )
        self.agent = gss._build_agent(self.args, classifier=None, diffusion_steps=m["diffusion_steps"])
        self.agent.load(str(self.paths["ckpt"]))
        self.agent.eval()
        self.net = self.agent.model_ema["diffusion"]
        self.reward_fn = gss._make_reward_fn(self.geometry, self.normalizer, device)
        self.prior = torch.zeros((1, H, D), device=device)
        self.std = np.asarray(self.normalizer.std, dtype=np.float64).reshape(-1)
        self.mean = np.asarray(self.normalizer.mean, dtype=np.float64).reshape(-1)
        self.L_norm = float(2.0 * np.max(self.std ** 2))
        hp = cfg["paper_hparams"][test]
        self.w_cg, self.opt_scale = float(hp["w_cg"]), float(hp["opt_scale"])
        self.drgd = drgd_cfg
        cache = np.load(self.paths["cache_npz"])
        self.cache = {k: cache[k] for k in cache.files}
        with open(self.paths["cache_meta"]) as f:
            self.cache_meta = json.load(f)
        self.x_init_test = torch.from_numpy(self.cache["initial_xts"]).to(device)

    def new_classifier(self):
        clf = RuntimeRewardClassifier(self.reward_fn, device=self.device)
        self.agent.classifier = clf
        return clf

    def to_phys(self, x_norm: np.ndarray) -> np.ndarray:
        x = np.asarray(x_norm, dtype=np.float64)
        assert x.shape[-1] == D
        return x * self.std + self.mean

    def gen_inits(self, seed0: int, n: int) -> torch.Tensor:
        a = SimpleNamespace(seed=seed0, device=self.device, temperature=self.args.temperature)
        x, _ = vis._generate_shared_initial_xts(a, n)
        return x

    def sample_kwargs(self, mode: str, n: int, *, history: bool, path_index: int | None = None) -> dict:
        """Exactly the kwargs vis.sample_paths_with_history passes (paper run)."""
        s, g = self.cfg["sampling"], self.cfg["ggdm"]
        is_ggdm = mode == "ggdm_g_loss_mean"
        w = self.w_cg if mode == "standard" else 0.0
        opt = self.opt_scale if mode == "optimization" else 0.0
        kw = dict(
            solver=s["ggdm_solver"] if is_ggdm else s["solver"],
            n_samples=n,
            sample_steps=s["sampling_steps"],
            sample_step_schedule=s["sample_step_schedule"],
            use_ema=s["use_ema"],
            w_cg=w,
            guidance_mode=mode,
            optimization_guidance_scale=opt,
            optimization_guidance_last_steps=s["optimization_guidance_last_steps"],
            ggdm_interval=g["interval"],
            ggdm_obs_noise_sigma=g["obs_noise_sigma"],
            ggdm_beta_coef=g["beta_coef"],
            drgd_steps=self.drgd["drgd_steps"] if mode == "drgd" else 0,
            drgd_gamma=self.drgd["drgd_gamma"],
            drgd_noise_index=self.drgd["drgd_noise_index"],
            ggdm_rounds=g["rounds"] if is_ggdm else 0,
            ggdm_round_seed=(g["round_seed"] if is_ggdm else None),
            temperature=s["temperature"],
            preserve_history=history,
            requires_grad=False,
        )
        return kw


def hist_from_log(log_: dict) -> np.ndarray:
    h = np.asarray(log_["sample_history"])  # (n, T, 1, D)
    return h.reshape(h.shape[0], h.shape[1], D)


# --------------------------------------------------------------------------------------
# Existing methods
# --------------------------------------------------------------------------------------
def run_existing(ctx: TestCtx, mode: str, x_init: torch.Tensor, *, per_path: bool) -> dict:
    """PDG ('standard'), DCG ('optimization'), DRGD ('drgd'), unguided DDIM ('unguided')."""
    if mode == "unguided":
        ctx.agent.classifier = None
        smode = "standard"
    else:
        ctx.new_classifier()
        smode = mode
    n = x_init.shape[0]
    hists, finals = [], []
    with NetworkCounter(ctx.net) as nc, (GradCounter(ctx.agent.classifier) if ctx.agent.classifier is not None else _Null()) as gc:
        if per_path:
            for i in range(n):
                kw = ctx.sample_kwargs(smode, 1, history=True)
                if mode == "unguided":
                    kw["w_cg"] = 0.0
                kw["initial_xt"] = x_init[i : i + 1]
                traj, lg = ctx.agent.sample(ctx.prior, **kw)
                hists.append(hist_from_log(lg))
                finals.append(traj.detach().cpu().numpy().reshape(1, D))
        else:
            kw = ctx.sample_kwargs(smode, n, history=True)
            if mode == "unguided":
                kw["w_cg"] = 0.0
            kw["initial_xt"] = x_init
            traj, lg = ctx.agent.sample(ctx.prior, **kw)
            hists.append(hist_from_log(lg))
            finals.append(traj.detach().cpu().numpy().reshape(n, D))
    hist = np.concatenate(hists, 0)
    final = np.concatenate(finals, 0)
    counts = {**nc.as_dict(), **gc.as_dict()}
    return {"hist_norm": hist, "final_norm": final, "counts": counts, "n": n}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def as_dict(self):
        return {"grad_calls": 0, "grad_samples": 0}


def ggdm_replica(ctx: TestCtx, x_init: torch.Tensor, *, seed_before: int | None) -> dict:
    """Round-by-round replica of DiscreteDiffusionSDE._sample_ggdm_algorithm1 through the public
    ``sample`` kwargs (ggdm_rounds=0, ggdm_query_xt=...), recording every round's final iterate.
    Mirrors vis.sample_paths_with_history (batched, ggdm_round_seed=args.seed)."""
    g, s = ctx.cfg["ggdm"], ctx.cfg["sampling"]
    clf = ctx.new_classifier()
    n = x_init.shape[0]
    inner = dict(
        solver="ddpm",
        n_samples=n,
        sample_steps=s["sampling_steps"],
        sample_step_schedule=s["sample_step_schedule"],
        use_ema=s["use_ema"],
        temperature=s["temperature"],
        condition_cfg=None,
        mask_cfg=None,
        w_cfg=0.0,
        condition_cg=None,
        optimization_guidance_last_steps=s["optimization_guidance_last_steps"],
        optimization_guidance_alpha_sigma_scale=False,
        ddim_eta=0.0,
        drgd_steps=0,
        drgd_gamma=ctx.drgd["drgd_gamma"],
        drgd_noise_index=ctx.drgd["drgd_noise_index"],
        diffusion_x_sampling_steps=0,
        warm_start_reference=None,
        warm_start_forward_level=0.3,
    )
    fields = dict(
        ggdm_interval=g["interval"],
        ggdm_obs_noise_sigma=g["obs_noise_sigma"],
        ggdm_beta_coef=g["beta_coef"],
        ggdm_beta_constant=None,
    )
    if seed_before is not None:
        set_seed(seed_before)
    round_finals, round_counts = [], []
    with NetworkCounter(ctx.net) as nc, GradCounter(clf) as gc:
        xt, _ = ctx.agent.sample(
            ctx.prior, guidance_mode="standard", w_cg=0.0, optimization_guidance_scale=0.0,
            initial_xt=x_init, preserve_history=False, requires_grad=False,
            ggdm_interval=0.0, ggdm_rounds=0, **inner,
        )
        query = xt.detach()
        round_finals.append(query.cpu().numpy().reshape(n, D))
        round_counts.append({**nc.as_dict(), **gc.as_dict()})
        last_hist = None
        for r in range(g["rounds"]):
            gen = torch.Generator(device=ctx.device)
            gen.manual_seed(int(g["round_seed"]) + 1000 * (r + 1))
            fresh = torch.randn((n, *ctx.prior.shape[1:]), device=ctx.device, generator=gen) * s["temperature"]
            fresh = fresh * (1.0 - ctx.agent.fix_mask) + ctx.prior * ctx.agent.fix_mask
            is_last = r == g["rounds"] - 1
            xt, lg = ctx.agent.sample(
                ctx.prior, guidance_mode=g["guidance_mode"], w_cg=0.0, optimization_guidance_scale=0.0,
                initial_xt=fresh, ggdm_query_xt=query, preserve_history=is_last, requires_grad=True,
                ggdm_rounds=0, **inner, **fields,
            )
            query = xt.detach()
            round_finals.append(query.cpu().numpy().reshape(n, D))
            round_counts.append({**nc.as_dict(), **gc.as_dict()})
            if is_last:
                last_hist = hist_from_log(lg)
    return {
        "hist_norm": last_hist,
        "final_norm": round_finals[-1],
        "round_finals_norm": np.stack(round_finals, 1),  # (n, R+1, D); index 0 = unguided landing
        "round_counts": round_counts,
        "counts": {**nc.as_dict(), **gc.as_dict()},
        "n": n,
    }


def ggdm_original(ctx: TestCtx, x_init: torch.Tensor, *, seed_before: int | None, history: bool = False):
    ctx.new_classifier()
    kw = ctx.sample_kwargs("ggdm_g_loss_mean", x_init.shape[0], history=history)
    kw["initial_xt"] = x_init
    if seed_before is not None:
        set_seed(seed_before)
    traj, lg = ctx.agent.sample(ctx.prior, **kw)
    return traj, lg


# --------------------------------------------------------------------------------------
# PnP-PGD
# --------------------------------------------------------------------------------------
def run_pnp(ctx: TestCtx, x_init: torch.Tensor, k: int, eta: float, n_iters: int, scaling: str,
            *, record: bool = True) -> dict:
    clf = RuntimeRewardClassifier(ctx.reward_fn, device=ctx.device)
    den = FixedNoiseDenoiser.from_agent(ctx.agent, k, prior=None, use_ema=True, input_scaling=scaling)
    grad_f = reward_classifier_grad_f(clf, fix_mask=ctx.agent.fix_mask)
    with NetworkCounter(ctx.net) as nc, GradCounter(clf) as gc:
        res = pnp_pgd(den, grad_f, x_init, eta=eta, n_iters=n_iters, record_history=record)
    out = {
        "counts": {**nc.as_dict(), **gc.as_dict()},
        "n_denoiser_evals": res.n_denoiser_evals,
        "n_grad_evals": res.n_grad_evals,
        "sigma_bar": den.sigma_bar,
        "alpha": den.alpha,
        "sigma": den.sigma,
        "model_t_unique": sorted(set(res.model_t_seen)),
        "n": int(x_init.shape[0]),
    }
    if record:
        h = torch.stack(res.history, 1).detach().cpu().numpy()  # (n, T, 1, D)
        out["hist_norm"] = h.reshape(h.shape[0], h.shape[1], D)
    out["final_norm"] = res.x.detach().cpu().numpy().reshape(-1, D)
    return out


def unguided_ddim(ctx: TestCtx, x_init: torch.Tensor) -> torch.Tensor:
    ctx.agent.classifier = None
    kw = ctx.sample_kwargs("standard", x_init.shape[0], history=False)
    kw["w_cg"] = 0.0
    kw["initial_xt"] = x_init
    traj, _ = ctx.agent.sample(ctx.prior, **kw)
    return traj.detach()


def pnp_grid(ctx: TestCtx, x_val: torch.Tensor, variant: dict, ks, etas_over_L, n_iters: int, existing=None):
    """Run every (k, eta) for n_iters iterations; per-iteration validation metrics."""
    scaling = variant["input_scaling"]
    out = {}
    for k in ks:
        for e in etas_over_L:
            key = (int(k), float(e))
            if key in out or (existing is not None and key in existing):
                continue
            r = run_pnp(ctx, x_val, int(k), float(e) / ctx.L_norm, n_iters, scaling)
            phys = ctx.to_phys(r["hist_norm"])
            met = ctx.sg.metrics(phys)
            out[key] = {"metrics": met, "sigma_bar": r["sigma_bar"], "counts": r["counts"],
                        "model_t_unique": r["model_t_unique"], "final_phys": phys[:, -1]}
    return out


def select(grid: dict, K_grid) -> tuple:
    best, best_val = None, np.inf
    for (k, e), g in sorted(grid.items()):
        for K in K_grid:
            v = float(np.mean(g["metrics"]["dist_xstar"][:, K]))
            if not np.isfinite(v):
                continue
            if v < best_val:
                best, best_val = (k, e, int(K)), v
    return best, best_val


# --------------------------------------------------------------------------------------
# helpers for saving
# --------------------------------------------------------------------------------------
def cum_costs(method: str, T: int, ctx: TestCtx) -> dict:
    """Analytic cumulative per-path costs at history index 0..T-1 (cross-checked vs counters)."""
    s = np.arange(T, dtype=np.float64)
    z = np.zeros(T)
    if method in ("standard", "optimization"):
        return {"cum_nfe": s, "cum_grad": s, "cum_bwd": z}
    if method == "unguided":
        return {"cum_nfe": s, "cum_grad": z, "cum_bwd": z}
    if method == "drgd":
        return {"cum_nfe": 2 * s, "cum_grad": s, "cum_bwd": s}
    if method.startswith("pnp"):
        off = 50.0 if method == "pnp_ddim_init" else 0.0
        return {"cum_nfe": s + off, "cum_grad": s, "cum_bwd": z}
    raise ValueError(method)


def save_method(run_t: Path, name: str, ctx: TestCtx, hist_norm: np.ndarray, extra: dict | None = None,
                costs: dict | None = None):
    phys = ctx.to_phys(hist_norm)
    met = ctx.sg.metrics(phys)
    arrays = {"hist_phys": phys, "hist_norm": hist_norm.astype(np.float64)}
    arrays.update({f"m_{k}": v for k, v in met.items()})
    if costs is not None:
        arrays.update(costs)
    if extra:
        arrays.update(extra)
    safe_io.write_npz(run_t / f"{name}.npz", **arrays)
    return phys, met


def summarize(met: dict, idx: int = -1) -> dict:
    return {k: {"mean": float(np.mean(met[k][:, idx])), "std": float(np.std(met[k][:, idx])),
                "max": float(np.max(met[k][:, idx]))} for k in METRIC_KEYS}


def time_fn(fn, warm: int, reps: int) -> dict:
    warmup(fn, warm)
    ts = []
    for _ in range(reps):
        d = {}
        with cuda_time(d, "s"):
            fn()
        ts.append(d["s"])
    return {"median_s": statistics.median(ts), "all_s": ts}


# --------------------------------------------------------------------------------------
def run_test(cfg: dict, test: str, device: str, run_dir: Path, quick: bool, drgd_all: dict) -> dict:
    drgd_cfg = {k: drgd_all["tests"][test][k] for k in ("drgd_steps", "drgd_gamma", "drgd_noise_index")}
    ctx = TestCtx(cfg, test, device, drgd_cfg)
    run_t = safe_io.ensure_dir(run_dir / test)
    n_test = 4 if quick else ctx.x_init_test.shape[0]
    x_test = ctx.x_init_test[:n_test]
    summary: dict = {"test": test, "n_test": n_test, "L_norm": ctx.L_norm,
                     "std": ctx.std.tolist(), "mean": ctx.mean.tolist(),
                     "x_star_exact": ctx.sg.x_star.tolist(), "f_star_exact": ctx.sg.f_star,
                     "x_star_cache": ctx.cache["feasible_optimum_ambient"].tolist(),
                     "f_star_cache": -float(ctx.cache_meta["feasible_optimum"]["reward"]),
                     "drgd": drgd_cfg}
    sched = SUPPORTED_SAMPLING_STEP_SCHEDULE[cfg["sampling"]["sample_step_schedule"]](
        cfg["model"]["diffusion_steps"], cfg["sampling"]["sampling_steps"]).tolist()
    di = int(drgd_cfg["drgd_noise_index"])
    summary["sampling_schedule"] = sched
    summary["drgd_effective_agent_index"] = int(sched[di])
    summary["drgd_alpha_sigma"] = [float(ctx.agent.alpha[sched[di]]), float(ctx.agent.sigma[sched[di]])]
    log(f"{test}: L_norm={ctx.L_norm:.6f}, f*={ctx.sg.f_star:.6f}, DRGD schedule idx {di} -> agent idx {sched[di]}")

    # sanity: cached init == regenerated init (CUDA RNG, seeds 0..99)
    regen = ctx.gen_inits(cfg["seed"], ctx.x_init_test.shape[0]).cpu().numpy()
    summary["init_regen_max_abs_diff"] = float(np.max(np.abs(regen - ctx.cache["initial_xts"])))
    x_val = ctx.gen_inits(cfg["pnp"]["val_seed_start"], 4 if quick else cfg["pnp"]["n_val"])
    summary["val_seeds"] = list(range(cfg["pnp"]["val_seed_start"], cfg["pnp"]["val_seed_start"] + x_val.shape[0]))
    safe_io.write_npz(run_t / "inits.npz", test_init_norm=x_test.cpu().numpy(), val_init_norm=x_val.cpu().numpy())

    # ---------------- 1. existing methods, paper protocol (per-path loop, GGDM batched) -------------
    set_seed(cfg["seed"])  # paper: set_seed(args.seed) at the start of plot_paths for each test
    results = {}
    for mode in ("standard", "optimization"):
        log(f"{test}: {mode} per-path")
        results[mode] = run_existing(ctx, mode, x_test, per_path=True)
    log(f"{test}: GGDM replica (round-by-round)")
    results["ggdm"] = ggdm_replica(ctx, x_test, seed_before=cfg["seed"])
    log(f"{test}: drgd per-path")
    results["drgd"] = run_existing(ctx, "drgd", x_test, per_path=True)

    # GGDM: also run the original Algorithm-1 code path with the same seed -> replica check
    traj_orig, lg_orig = ggdm_original(ctx, x_test, seed_before=cfg["seed"], history=True)
    orig_final = traj_orig.detach().cpu().numpy().reshape(-1, D)
    summary["ggdm_replica_vs_original_max_abs_final"] = float(np.max(np.abs(orig_final - results["ggdm"]["final_norm"])))
    summary["ggdm_replica_vs_original_max_abs_hist"] = float(np.max(np.abs(hist_from_log(lg_orig) - results["ggdm"]["hist_norm"])))

    # ---------------- reproduction check vs cached paper curves ----------------
    repro = {}
    for mode, key in (("standard", "standard"), ("optimization", "optimization"), ("ggdm", "ggdm_g_loss_mean"), ("drgd", "drgd")):
        phys = ctx.to_phys(results[mode]["hist_norm"])
        rew = gss.reward_values(phys.reshape(-1, D), ctx.geometry).reshape(phys.shape[:2])
        vio = gss.constraint_violation(phys.reshape(-1, D), ctx.geometry).reshape(phys.shape[:2])
        sub = vis.ambient_to_subspace(phys, ctx.geometry.basis_t)
        c_r = ctx.cache[f"{key}_reward_curves"][:n_test]
        c_v = ctx.cache[f"{key}_violation_curves"][:n_test]
        c_p = ctx.cache[f"{key}_paths"][:n_test]
        cm = ctx.cache_meta["methods"][key]["metrics"]
        repro[mode] = {
            "max_abs_reward_curve": float(np.max(np.abs(rew - c_r))),
            "max_abs_violation_curve": float(np.max(np.abs(vio - c_v))),
            "max_abs_subspace_path": float(np.max(np.abs(sub - c_p))),
            "final_reward_mean_rerun": float(rew[:, -1].mean()), "final_reward_mean_cache": float(c_r[:, -1].mean()),
            "final_reward_std_rerun": float(rew[:, -1].std()), "final_reward_std_cache": float(c_r[:, -1].std()),
            "final_violation_mean_rerun": float(vio[:, -1].mean()), "final_violation_mean_cache": float(c_v[:, -1].mean()),
            "final_violation_std_rerun": float(vio[:, -1].std()), "final_violation_std_cache": float(c_v[:, -1].std()),
            "cache_meta_metrics": cm,
        }
        log(f"{test}: repro {mode}: max|dR|={repro[mode]['max_abs_reward_curve']:.3e} max|dV|={repro[mode]['max_abs_violation_curve']:.3e}")
    summary["reproduction"] = repro

    # ---------------- batched re-runs (timing convention) + batched-vs-per-path check -------------
    batched = {}
    for mode in ("standard", "optimization", "drgd"):
        b = run_existing(ctx, mode, x_test, per_path=False)
        batched[mode] = float(np.max(np.abs(b["hist_norm"] - results[mode]["hist_norm"])))
        results[mode]["counts_batched"] = b["counts"]
    summary["batched_vs_perpath_max_abs_hist_norm"] = batched
    results["unguided"] = run_existing(ctx, "unguided", x_test, per_path=False)

    # ---------------- save existing methods ----------------
    method_metrics = {}
    for name in ("standard", "optimization", "drgd", "unguided"):
        r = results[name]
        T = r["hist_norm"].shape[1]
        costs = cum_costs(name, T, ctx)
        _, met = save_method(run_t, name, ctx, r["hist_norm"], costs=costs)
        method_metrics[name] = met
        n = r["n"]
        exp = {"nfe": costs["cum_nfe"][-1], "grad": costs["cum_grad"][-1], "bwd": costs["cum_bwd"][-1]}
        c = r["counts"]
        meas = {"nfe": c["net_forward_samples"] / n, "grad": c["grad_samples"] / n, "bwd": c["net_backward_samples"] / n}
        summary.setdefault("costs", {})[name] = {"expected_per_path": exp, "measured_per_path": meas,
                                                 "raw_counts_perpath_run": c,
                                                 "raw_counts_batched_run": r.get("counts_batched"),
                                                 "match": all(abs(exp[k] - meas[k]) < 1e-9 for k in exp)}
        summary.setdefault("final", {})[name] = summarize(met)
    # GGDM: last-round history + per-round finals
    gg = results["ggdm"]
    rf_phys = ctx.to_phys(gg["round_finals_norm"])
    rf_met = ctx.sg.metrics(rf_phys)
    R = cfg["ggdm"]["rounds"]
    n = gg["n"]
    rc = gg["round_counts"]
    round_nfe = np.array([c["net_forward_samples"] / n for c in rc])
    round_grad = np.array([c["grad_samples"] / n for c in rc])
    round_bwd = np.array([c["net_backward_samples"] / n for c in rc])
    S = cfg["sampling"]["sampling_steps"]
    T = gg["hist_norm"].shape[1]
    last_costs = {"cum_nfe": R * S + np.arange(T, dtype=np.float64),
                  "cum_grad": (R - 1) * S + (R - 1) / n + np.concatenate([[0.0], np.arange(1, T) + 1.0 / n]),
                  "cum_bwd": np.zeros(T)}
    extra = {"round_finals_phys": rf_phys, "round_cum_nfe": round_nfe, "round_cum_grad": round_grad,
             "round_cum_bwd": round_bwd}
    extra.update({f"round_m_{k}": v for k, v in rf_met.items()})
    _, met = save_method(run_t, "ggdm", ctx, gg["hist_norm"], extra=extra, costs=last_costs)
    method_metrics["ggdm"] = met
    exp = {"nfe": (R + 1) * S, "grad": R * S + R / n, "bwd": 0.0}
    meas = {"nfe": round_nfe[-1], "grad": round_grad[-1], "bwd": round_bwd[-1]}
    summary["costs"]["ggdm"] = {"expected_per_path": exp, "measured_per_path": meas, "raw_counts": gg["counts"],
                                "match": all(abs(exp[k] - meas[k]) < 1e-9 for k in exp),
                                "note": "grad = R*S per-step classifier.gradients calls made by apply_ggdm_g_loss_prediction "
                                        "(only used for log_p, not for the update) + R anchor gradients at the particle mean "
                                        "(batch of 1, shared by all n particles -> R/n per path). Algorithmically only the R "
                                        "anchor gradients are needed."}
    summary["final"]["ggdm"] = summarize(met)
    summary["ggdm_round_expected_nfe"] = [(r + 1) * S for r in range(R + 1)]

    # ---------------- 2. PnP-PGD validation grids ----------------
    pc = cfg["pnp"]
    n_it = pc["n_iters_curve"]
    ks0, es0, Kg = list(pc["noise_indices"]), list(pc["eta_over_L"]), list(pc["K_grid"])
    ext = pc["boundary_extension"]
    x_val_ddim = unguided_ddim(ctx, x_val)
    x_test_ddim = unguided_ddim(ctx, x_test)
    pnp_sel = {}
    grid_rows = []
    for vname, var in pc["variants"].items():
        xv = x_val_ddim if var["init"] == "ddim_unguided" else x_val
        log(f"{test}: PnP grid {vname}")
        grid = pnp_grid(ctx, xv, var, ks0, es0, n_it)
        sel0, val0 = select(grid, Kg)
        # v2 boundary rule: each of the four directions (k lower/upper, eta lower/upper) may be extended
        # AT MOST ONCE by one point, on validation only. After an extension we re-select; if the new
        # selection lies on a boundary in a direction that has not been extended yet (e.g. v1's Test A
        # no-VP-scaling case: eta extended, then k=5 selected on the never-extended lower k boundary),
        # that direction is extended once too. Directions already extended are never extended again.
        ks, es = list(ks0), list(es0)
        extended, ext_rounds, done_dirs = [], [], set()
        sel, val = sel0, val0
        while True:
            new_dirs = []
            if sel[0] == min(ks) and "k_lower" not in done_dirs:
                new_dirs.append("k_lower")
            if sel[0] == max(ks) and "k_upper" not in done_dirs:
                new_dirs.append("k_upper")
            if sel[1] == min(es) and "eta_lower" not in done_dirs:
                new_dirs.append("eta_lower")
            if sel[1] == max(es) and "eta_upper" not in done_dirs:
                new_dirs.append("eta_upper")
            if not new_dirs:
                break
            msgs = []
            for d_ in new_dirs:
                done_dirs.add(d_)
                if d_ == "k_lower":
                    ks = [ext["noise_index_lower"]] + ks; msgs.append(f"noise_index lower -> {ext['noise_index_lower']}")
                elif d_ == "k_upper":
                    ks = ks + [ext["noise_index_upper"]]; msgs.append(f"noise_index upper -> {ext['noise_index_upper']}")
                elif d_ == "eta_lower":
                    es = [min(es) / ext["eta_factor"]] + es; msgs.append(f"eta lower -> {min(es)}/L")
                else:
                    es = es + [max(es) * ext["eta_factor"]]; msgs.append(f"eta upper -> {max(es)}/L")
            extended += msgs
            grid.update(pnp_grid(ctx, xv, var, ks, es, n_it, existing=grid))
            prev = (sel, val)
            sel, val = select(grid, Kg)
            ext_rounds.append({"trigger_selection": list(prev[0]), "trigger_value": prev[1], "directions": new_dirs,
                               "added": msgs, "new_selection": list(sel), "new_value": val})
        on_new_boundary = []
        if extended:
            if sel[0] in (min(ks), max(ks)) and sel[0] not in ks0:
                on_new_boundary.append("noise_index")
            if sel[1] in (min(es), max(es)) and sel[1] not in es0:
                on_new_boundary.append("eta")
        on_final_boundary = [nm for nm, cond in (("k_lower", sel[0] == min(ks)), ("k_upper", sel[0] == max(ks)),
                                                  ("eta_lower", sel[1] == min(es)), ("eta_upper", sel[1] == max(es))) if cond]
        log(f"{test}: {vname} initial sel {sel0} ({val0:.4f}); extended={extended}; final sel {sel} ({val:.4f}); "
            f"on final-grid boundary (already extended once, not extended again): {on_final_boundary}")
        # save full validation grid
        keys = sorted(grid.keys())
        arr = {"grid_k": np.array([k for k, _ in keys]), "grid_eta_over_L": np.array([e for _, e in keys]),
               "grid_sigma_bar": np.array([grid[q]["sigma_bar"] for q in keys])}
        for m in METRIC_KEYS:
            arr[f"m_{m}"] = np.stack([grid[q]["metrics"][m] for q in keys], 0)  # (G, n_val, T)
        safe_io.write_npz(run_t / f"val_grid_{vname}.npz", **arr)
        for q in keys:
            for K in sorted(set(Kg + [n_it])):
                row = {"variant": vname, "k": q[0], "sigma_bar": grid[q]["sigma_bar"], "eta_over_L": q[1],
                       "eta": q[1] / ctx.L_norm, "K": K, "in_selection_grid": K in Kg,
                       "is_extension": (q[0] not in ks0) or (q[1] not in es0)}
                for m in METRIC_KEYS:
                    v = grid[q]["metrics"][m][:, K]
                    row[f"{m}_mean"] = float(np.mean(v)); row[f"{m}_std"] = float(np.std(v))
                row["selected"] = (q[0], q[1], K) == sel
                grid_rows.append(row)
        pnp_sel[vname] = {"initial_selection": list(sel0), "initial_value": val0, "extended": extended,
                          "selection": list(sel), "value": val, "selected_on_extended_boundary": on_new_boundary,
                          "extension_rounds": ext_rounds, "directions_extended": sorted(done_dirs),
                          "selected_on_final_grid_boundary": on_final_boundary,
                          "final_grid_k": list(ks), "final_grid_eta_over_L": list(es),
                          "sigma_bar": grid[(sel[0], sel[1])]["sigma_bar"], "eta": sel[1] / ctx.L_norm,
                          "model_t_unique": grid[(sel[0], sel[1])]["model_t_unique"], **var}
        # ---- frozen -> test ----
        xt0 = x_test_ddim if var["init"] == "ddim_unguided" else x_test
        r = run_pnp(ctx, xt0, sel[0], sel[1] / ctx.L_norm, n_it, var["input_scaling"])
        T = r["hist_norm"].shape[1]
        costs = cum_costs(vname, T, ctx)
        _, met = save_method(run_t, vname, ctx, r["hist_norm"], costs=costs,
                             extra={"selected_K": np.array(sel[2])})
        method_metrics[vname] = met
        c = r["counts"]; n = r["n"]
        init_nfe = 50.0 if var["init"] == "ddim_unguided" else 0.0
        exp = {"nfe": init_nfe + n_it, "grad": float(n_it), "bwd": 0.0}
        meas = {"nfe": init_nfe + c["net_forward_samples"] / n, "grad": c["grad_samples"] / n, "bwd": c["net_backward_samples"] / n}
        summary["costs"][vname] = {"expected_per_path_K50": exp, "measured_per_path_K50": meas, "raw_counts": c,
                                   "init_nfe_per_path": init_nfe,
                                   "match": all(abs(exp[k] - meas[k]) < 1e-9 for k in exp),
                                   "at_selected_K": {"nfe": init_nfe + sel[2], "grad": sel[2], "bwd": 0}}
        summary["final"][vname] = {"at_selected_K": summarize(met, sel[2]), "at_K50": summarize(met, -1),
                                   "selected_K": sel[2]}
    safe_io.write_json(run_t / "val_grid_rows.json", grid_rows)
    summary["pnp_selection"] = pnp_sel

    # ---------------- 4. timing (batched 100 test paths and batch-of-1), same job ----------------
    tc = cfg["timing"]
    warm, reps = (1, 1) if quick else (tc["warmup"], tc["repeats"])

    def mk_existing(mode, xi):
        def fn():
            if mode == "unguided":
                ctx.agent.classifier = None
            else:
                ctx.new_classifier()
            kw = ctx.sample_kwargs("standard" if mode == "unguided" else mode, xi.shape[0], history=False)
            if mode == "unguided":
                kw["w_cg"] = 0.0
            kw["initial_xt"] = xi
            ctx.agent.sample(ctx.prior, **kw)
        return fn

    def mk_pnp(vname, xi, K):
        var = pc["variants"][vname]
        s = pnp_sel[vname]["selection"]

        def fn():
            x0 = unguided_ddim(ctx, xi) if var["init"] == "ddim_unguided" else xi
            clf = RuntimeRewardClassifier(ctx.reward_fn, device=ctx.device)
            den = FixedNoiseDenoiser.from_agent(ctx.agent, s[0], use_ema=True, input_scaling=var["input_scaling"])
            pnp_pgd(den, reward_classifier_grad_f(clf, ctx.agent.fix_mask), x0, eta=s[1] / ctx.L_norm, n_iters=K)
        return fn

    timing = {"convention": "wall-clock of one call processing the whole batch (batch=100 test paths, or batch=1 = path 0) "
                            "on one GPU, torch.cuda.synchronize before/after, after warmup; model loading excluded; "
                            "no history recording, no counters/hooks attached; median of repeats",
              "warmup": warm, "repeats": reps, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for bname, xi in (("batch100", x_test), ("batch1", x_test[:1])):
        tb = {}
        for mode in ("standard", "optimization", "drgd", "unguided"):
            tb[mode] = time_fn(mk_existing(mode, xi), warm, reps)
        tb["ggdm"] = time_fn(lambda: ggdm_original(ctx, xi, seed_before=None), warm, reps)
        for vname in pc["variants"]:
            K = pnp_sel[vname]["selection"][2]
            tb[f"{vname}@K{K}"] = time_fn(mk_pnp(vname, xi, K), warm, reps)
            tb[f"{vname}@K{n_it}"] = time_fn(mk_pnp(vname, xi, n_it), warm, reps)
        timing[bname] = tb
        log(f"{test}: timing {bname}: " + ", ".join(f"{k}={v['median_s']*1e3:.1f}ms" for k, v in tb.items()))
    summary["timing"] = timing
    safe_io.write_json(run_t / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REB / "configs" / "synthetic_v2.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tests", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 4 test paths, 4 val paths, 1 timing repeat")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    tests = a.tests or cfg["tests"]
    tag = ("quick_" if a.quick else "") + (a.tag or "synthetic")
    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "synthetic", tag=tag)
    log(f"run dir {run_dir}")
    with open(cfg["drgd_tuned_json"]) as f:
        drgd_all = json.load(f)
    inputs = {}
    for t in tests:
        root = Path(cfg["orig_root"]) / t
        for name in ("geometry.json", "normalizer.npz", "diffusion_ckpt_latest.pt",
                     f"trajectories/{cfg['paper_cache_name'].format(test=t)}.npz",
                     f"trajectories/{cfg['paper_cache_name'].format(test=t)}.meta.json"):
            p = root / name
            inputs[str(p)] = sha256(p)
    inputs[cfg["drgd_tuned_json"]] = sha256(Path(cfg["drgd_tuned_json"]))
    safe_io.write_json(run_dir / "config.json", {
        "config": cfg, "config_path": a.config, "argv": sys.argv, "quick": a.quick, "device": a.device,
        "tests": tests, "inputs_sha256": inputs, "drgd_tuned": drgd_all,
        "provenance": safe_io.provenance(), "code_sha256": code_hashes(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    t0 = time.time()
    for t in tests:
        run_test(cfg, t, a.device, run_dir, a.quick, drgd_all)
    safe_io.write_json(run_dir / "DONE.json", {"elapsed_s": time.time() - t0})
    log(f"done in {time.time() - t0:.1f}s -> {run_dir}")


if __name__ == "__main__":
    main()
