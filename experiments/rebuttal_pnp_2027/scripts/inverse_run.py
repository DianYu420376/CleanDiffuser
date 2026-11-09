#!/usr/bin/env python3
"""Sparse trajectory reconstruction: smoke / validation tuning / test evaluation (GPU).

  --phase smoke : test-phase code path on validation trajectories 0..4 with fixed (untuned)
                  hyper-parameters + correctness checks (gradient, pinning, broadcasting, counts).
  --phase val   : full validation grids for PDG, DCG, PnP-PGD (+ DDIM-10/K=10 ablation),
                  pre-registered selection + one-time boundary extension; writes selected.json.
  --phase test  : all methods on the 100 test trajectories with the frozen selected
                  hyper-parameters, in ONE job: batch-1 timed runs (warm-up, synchronize),
                  batched instrumented runs (NFE / grads / backward counts), metrics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inverse_common as ic  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402
from rebuttal_pnp.instrument import GradCounter, NetworkCounter, cuda_time, sync  # noqa: E402

DEVICE = "cuda:0"
METRIC_KEYS = ("unobs_xy_mse", "unobs_obs4_mse", "action_mse", "obs_xy_rmse", "f_norm", "e_dyn", "e_dyn_norm", "pin_err_o0")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- context


class Ctx:
    def __init__(self, data_dir: Path, split: str, idx=None):
        self.proto = ic.load_protocol()
        self.data_dir = Path(data_dir)
        self.data_path = self.data_dir / f"{split}_data.npz"
        d = np.load(self.data_path)
        sel = slice(None) if idx is None else np.asarray(idx)
        self.split = split
        self.gt_obs = d["gt_obs"][sel]
        self.gt_act = d["gt_act"][sel]
        self.x_star = d["x_star_norm"][sel]
        self.obs_idx = d["obs_idx"][sel]
        self.M = d["M"][sel]
        self.Y = d["Y"][sel]
        self.traj_ids = np.arange(d["gt_obs"].shape[0])[sel]
        self.n = self.gt_obs.shape[0]
        self.normalizer, _ = ic.training_normalizer(self.proto["training_hdf5"])
        base = self.proto["xT_seeds"]["base_test" if split == "test" else "base_val"]
        self.seeds = base + self.traj_ids
        # x_T exactly as specified: per trajectory torch.manual_seed(seed_j); torch.randn((1,64,6), cuda:0)
        self.xT = torch.cat([ic.make_xT(1, int(s), DEVICE) for s in self.seeds], 0)
        self.prior = ic.build_prior(self.normalizer, self.gt_obs[:, 0], DEVICE)
        self.Mt = torch.tensor(self.M, device=DEVICE)
        self.Yt = torch.tensor(self.Y, device=DEVICE)
        self.rew = ic.MaskedLSQReward(self.Mt, self.Yt)
        self.clf = ic.RuntimeRewardClassifier(self.rew, device=DEVICE)
        self.agent20, self.agent100 = ic.load_agents(self.proto, DEVICE)
        self.env = ic.make_env()

    def sub(self, j):
        """Per-trajectory (batch-1) views, switching the reward to trajectory j."""
        sl = slice(j, j + 1)
        self.rew.set(self.Mt[sl], self.Yt[sl])
        return self.prior[sl], self.xT[sl]

    def full(self):
        self.rew.set(self.Mt, self.Yt)
        return self.prior, self.xT

    def metrics(self, x_norm: np.ndarray, with_dyn=True) -> dict:
        return ic.per_traj_metrics(x_norm, self.gt_obs, self.gt_act, self.obs_idx, self.M, self.Y,
                                   self.normalizer, self.env, with_dyn=with_dyn)


# --------------------------------------------------------------------------- methods


def run_method(ctx: Ctx, method: str, params: dict, prior, xT, *, timer=None, metrics_fn=None):
    """Returns (x_hat tensor, extras). ``timer`` (dict) receives sub-timings for PnP."""
    timer = {} if timer is None else timer
    extras = {}
    if method == "unguided":
        x = ic.sample_unguided(ctx.agent20, prior, xT, sample_steps=20)
    elif method == "pdg":
        x = ic.sample_pdg(ctx.agent20, ctx.clf, prior, xT, params["w_cg"])
    elif method == "dcg":
        x = ic.sample_dcg(ctx.agent100, ctx.clf, prior, xT, params["eta"],
                          ctx.proto["methods"]["dcg"]["optimization_guidance_last_steps"])
    elif method in ("pnp_pgd", "pnp_pgd_ddim10_K10"):
        init_steps = 20 if method == "pnp_pgd" else 10
        with cuda_time(timer, "init_s"):
            x0 = ic.sample_unguided(ctx.agent20, prior, xT, sample_steps=init_steps)
        with cuda_time(timer, "iters_s"):
            res, D = ic.run_pnp(ctx.agent20, ctx.clf, prior, x0, params["noise_index"], params["eta"], params["K"],
                                metrics_fn=metrics_fn)
        x = res.x
        extras = {"x_init": x0, "pnp_metrics": res.metrics, "n_denoiser_evals": res.n_denoiser_evals,
                  "n_grad_evals": res.n_grad_evals, "sigma_bar": res.sigma_bar, "alpha": D.alpha, "sigma": D.sigma,
                  "model_t": D.model_t, "model_t_seen_unique": sorted(set(res.model_t_seen))}
    else:
        raise ValueError(method)
    return x, extras


def analytic_costs(method: str, params: dict) -> dict:
    if method == "unguided":
        return {"nfe": 20, "grads": 0, "net_backward": 0, "init_nfe": 0}
    if method == "pdg":
        return {"nfe": 20, "grads": 20, "net_backward": 0, "init_nfe": 0}
    if method == "dcg":
        return {"nfe": 20, "grads": 10, "net_backward": 0, "init_nfe": 0}
    if method == "pnp_pgd":
        return {"nfe": 20 + params["K"], "grads": params["K"], "net_backward": 0, "init_nfe": 20}
    if method == "pnp_pgd_ddim10_K10":
        return {"nfe": 10 + params["K"], "grads": params["K"], "net_backward": 0, "init_nfe": 10}
    raise ValueError(method)


def counted_run(ctx: Ctx, method: str, params: dict, prior, xT, metrics_fn=None):
    agent = ctx.agent100 if method == "dcg" else ctx.agent20
    n = prior.shape[0]
    with NetworkCounter(agent.model_ema["diffusion"]) as nc, GradCounter(ctx.clf) as gc:
        x, extras = run_method(ctx, method, params, prior, xT, metrics_fn=metrics_fn)
        sync()
    meas = {**nc.as_dict(), **gc.as_dict()}
    per_sample = {"nfe": meas["net_forward_samples"] / n, "grads": meas["grad_samples"] / n,
                  "net_backward": meas["net_backward_samples"] / n}
    exp = analytic_costs(method, params)
    ok = all(abs(per_sample[k] - exp[k]) < 1e-9 for k in ("nfe", "grads", "net_backward"))
    return x, extras, {"measured_raw": meas, "measured_per_sample": per_sample, "analytic": exp, "match": ok}


# --------------------------------------------------------------------------- checks


def correctness_checks(ctx: Ctx) -> dict:
    prior, xT = ctx.full()
    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randn(prior.shape, device=DEVICE, generator=g) * 2.0
    t = torch.zeros((x.shape[0],), device=DEVICE)
    R, grad_R = ctx.clf.gradients(x.clone(), t, None)
    analytic = ctx.rew.grad_f_analytic(x)
    grad_f = ic.reward_classifier_grad_f(ctx.clf, ctx.agent20.fix_mask)(x)
    f_direct = 0.5 * ((ctx.Mt * x - ctx.Yt) ** 2).sum((1, 2))
    out = {
        "grad_autograd_vs_AT(Ax-y)_maxabs": float((-grad_R - analytic).abs().max()),
        "grad_f_helper_vs_AT(Ax-y)_maxabs": float((grad_f - analytic).abs().max()),
        "reward_vs_minus_f_maxabs": float((R.reshape(-1) + f_direct).abs().max()),
        "A_selects_only_xy_observed": bool(((ctx.Mt[..., 2:] == 0).all() and (ctx.Mt[:, 0] == 0).all()).item()),
        "observed_entries_per_traj": ctx.Mt.sum((1, 2)).cpu().numpy().tolist(),
        "Y_equals_A_xstar_maxabs": float(np.abs(ctx.Y - ctx.M * ctx.x_star).max()),
        "f_at_ground_truth_max": float(ctx.rew.f(torch.tensor(ctx.x_star, device=DEVICE)).abs().max()),
        "fix_mask_shape": list(ctx.agent20.fix_mask.shape),
        "fix_mask_rows": {"row0": ctx.agent20.fix_mask[0, 0].cpu().numpy().tolist(),
                          "sum": float(ctx.agent20.fix_mask.sum())},
        "prior_shape": list(prior.shape), "xT_shape": list(xT.shape),
        "prior_row0_vs_normalized_o0_maxabs": float(np.abs(prior[:, 0, :4].cpu().numpy()
                                                           - ctx.normalizer.normalize(ctx.gt_obs[:, 0])).max()),
        "agent20": {"diffusion_steps": ctx.agent20.diffusion_steps, "training_diffusion_steps": ctx.agent20.training_diffusion_steps,
                    "predict_noise": ctx.agent20.predict_noise, "n_alpha": int(ctx.agent20.alpha.shape[0])},
        "agent100": {"diffusion_steps": ctx.agent100.diffusion_steps, "training_diffusion_steps": ctx.agent100.training_diffusion_steps},
        "clip_pred": bool(ctx.agent20.clip_pred),
        "schedule_T20": {"alpha": ctx.agent20.alpha.cpu().numpy().tolist(), "sigma": ctx.agent20.sigma.cpu().numpy().tolist()},
    }
    # x_T determinism: regenerating trajectory 0's x_T gives the same tensor
    out["xT_regen_maxabs"] = float((ic.make_xT(1, int(ctx.seeds[0]), DEVICE) - xT[:1]).abs().max())
    # batched sampling with per-sample prior == per-trajectory sampling
    xb = ic.sample_unguided(ctx.agent20, prior, xT)
    diffs = []
    for j in range(min(ctx.n, 5)):
        pj, xj = prior[j:j + 1], xT[j:j + 1]
        diffs.append(float((ic.sample_unguided(ctx.agent20, pj, xj) - xb[j:j + 1]).abs().max()))
    out["unguided_batched_vs_single_maxabs"] = max(diffs)
    out["unguided_pin_maxabs"] = float((xb[:, 0, :4] - prior[:, 0, :4]).abs().max())
    return out


# --------------------------------------------------------------------------- validation


def _score(m):
    return float(np.mean(m["unobs_xy_mse"]))


def _select(entries, keys):
    """argmin of val score; tie-break lower nfe, lower grads, grid order."""
    order = sorted(range(len(entries)), key=lambda i: (entries[i]["val_mean_unobs_xy_mse"]
                                                        if np.isfinite(entries[i]["val_mean_unobs_xy_mse"]) else np.inf,
                                                        entries[i]["cost"]["nfe"], entries[i]["cost"]["grads"], i))
    return entries[order[0]]


def _summ(m):
    return {k: {"mean": float(np.nanmean(m[k])), "std_ddof1": float(np.nanstd(m[k], ddof=1)),
                "median": float(np.nanmedian(m[k])), "n_nonfinite": int((~np.isfinite(m[k])).sum())} for k in METRIC_KEYS}


def phase_val(args):
    ctx = Ctx(args.data_dir, "val")
    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "inverse", run_id=args.run_id, tag="val")
    log(f"val run dir {run_dir}; n={ctx.n}")
    checks = correctness_checks(ctx)
    log(f"checks: grad diff {checks['grad_autograd_vs_AT(Ax-y)_maxabs']}, batched-vs-single {checks['unguided_batched_vs_single_maxabs']}")
    prior, xT = ctx.full()
    P = ctx.proto["methods"]
    grids = {}
    npz = {}

    def eval_point(method, params, store_key):
        metric = ic.TorchMetric(ctx.normalizer, ctx.gt_obs, ctx.obs_idx, ctx.rew, DEVICE) if method.startswith("pnp") else None
        t0 = time.perf_counter()
        x, extras, cost = counted_run(ctx, method, params, prior, xT, metrics_fn=metric)
        xn = x.cpu().numpy()
        m = ctx.metrics(xn)
        npz[f"{store_key}__x"] = xn
        for k in METRIC_KEYS:
            npz[f"{store_key}__{k}"] = m[k]
        e = {"method": method, "params": params, "val_mean_unobs_xy_mse": _score(m), "summary": _summ(m),
             "per_traj_unobs_xy_mse": m["unobs_xy_mse"], "cost": cost["analytic"], "cost_check": cost,
             "wall_s_batched_incl_counters": time.perf_counter() - t0}
        if extras.get("pnp_metrics"):
            curve = np.stack([d["unobs_xy_mse"] for d in extras["pnp_metrics"]])  # (K+1, n)
            fcurve = np.stack([d["f_norm"] for d in extras["pnp_metrics"]])
            npz[f"{store_key}__iter_unobs_xy_mse"] = curve
            npz[f"{store_key}__iter_f_norm"] = fcurve
            e["iter_mean_unobs_xy_mse"] = curve.mean(1)
            e["iter_mean_f_norm"] = fcurve.mean(1)
            e["sigma_bar"] = extras["sigma_bar"]
            e["model_t_seen_unique"] = extras["model_t_seen_unique"]
        if not cost["match"]:
            log(f"WARNING cost mismatch {method} {params}: {cost}")
        log(f"  {method} {params}: val unobs-xy MSE {e['val_mean_unobs_xy_mse']:.5f}")
        return e

    # Unguided reference (no hyper-parameters)
    grids["unguided"] = [eval_point("unguided", {}, "unguided")]

    def tune_1d(method, key):
        vals = list(P[method]["grid"][key])
        entries = [eval_point(method, {key: v}, f"{method}__{key}{v}") for v in vals]
        sel = _select(entries, [key])
        ext = None
        v = sel["params"][key]
        if v == min(vals) or v == max(vals):
            direction = "low" if v == min(vals) else "high"
            nv = P[method]["extension"][key][direction]
            log(f"  {method}: selected {key}={v} on {direction} boundary -> extending once with {nv}")
            entries.append(eval_point(method, {key: nv}, f"{method}__{key}{nv}"))
            ext = {"dimension": key, "direction": direction, "added": nv, "selected_before": v}
            sel = _select(entries, [key])
        vals_all = sorted({e["params"][key] for e in entries})
        on_bd = sel["params"][key] in (min(vals_all), max(vals_all))
        return {"entries": entries, "selected": sel["params"], "selected_val_mean_unobs_xy_mse": sel["val_mean_unobs_xy_mse"],
                "extension": ext, "selected_on_boundary_after_extension": bool(on_bd and ext is not None),
                "selected_on_boundary_no_extension_needed": False}

    grids["pdg"] = tune_1d("pdg", "w_cg")
    grids["dcg"] = tune_1d("dcg", "eta")

    def tune_pnp(method):
        g = P[method]["grid"]
        ks, etas = list(g["noise_index"]), list(g["eta"])
        Ks = list(g["K"]) if "K" in g else [P[method]["K"]]
        cache = {}

        def run_ke(k, eta):
            # nested K: run K_max once, evaluate all K in the grid from the same iterate sequence
            if (k, eta) in cache:
                return
            Kmax = max(Ks)
            metric = ic.TorchMetric(ctx.normalizer, ctx.gt_obs, ctx.obs_idx, ctx.rew, DEVICE)
            snaps = {}

            def mfn(it, x):
                if it in Ks:
                    snaps[it] = x.detach().clone()
                return metric(it, x)

            params = {"noise_index": k, "eta": eta, "K": Kmax}
            x, extras, cost = counted_run(ctx, method, params, prior, xT, metrics_fn=mfn)
            curve = np.stack([d["unobs_xy_mse"] for d in extras["pnp_metrics"]])
            fcurve = np.stack([d["f_norm"] for d in extras["pnp_metrics"]])
            key0 = f"{method}__k{k}_eta{eta}"
            npz[f"{key0}__iter_unobs_xy_mse"] = curve
            npz[f"{key0}__iter_f_norm"] = fcurve
            npz[f"{key0}__x_init"] = extras["x_init"].cpu().numpy()
            for K in Ks:
                xn = snaps[K].cpu().numpy()
                m = ctx.metrics(xn)
                sk = f"{key0}_K{K}"
                npz[f"{sk}__x"] = xn
                for mk in METRIC_KEYS:
                    npz[f"{sk}__{mk}"] = m[mk]
                pK = {"noise_index": k, "eta": eta, "K": K}
                cache.setdefault((k, eta), []).append({
                    "method": method, "params": pK, "val_mean_unobs_xy_mse": _score(m), "summary": _summ(m),
                    "per_traj_unobs_xy_mse": m["unobs_xy_mse"], "cost": analytic_costs(method, pK),
                    "cost_check_at_Kmax": cost, "sigma_bar": extras["sigma_bar"],
                    "model_t_seen_unique": extras["model_t_seen_unique"],
                    "iter_mean_unobs_xy_mse": curve.mean(1)[: K + 1], "iter_mean_f_norm": fcurve.mean(1)[: K + 1],
                })
                log(f"  {method} k={k} eta={eta} K={K}: val unobs-xy MSE {_score(m):.5f} (sigma_bar={extras['sigma_bar']:.4f})")

        for k, eta in itertools.product(ks, etas):
            run_ke(k, eta)
        entries = [e for v in cache.values() for e in v]
        sel = _select(entries, None)
        ext = []
        for dim, vals in (("noise_index", ks), ("eta", etas)):
            v = sel["params"][dim]
            if v == min(vals) or v == max(vals):
                direction = "low" if v == min(vals) else "high"
                ext.append({"dimension": dim, "direction": direction, "added": P[method]["extension"][dim][direction],
                            "selected_before": v})
        if ext:
            for e_ in ext:
                (ks if e_["dimension"] == "noise_index" else etas).append(e_["added"])
            log(f"  {method}: boundary selection {sel['params']} -> extending once: {ext}")
            for k, eta in itertools.product(ks, etas):
                run_ke(k, eta)
            entries = [e for v in cache.values() for e in v]
            sel = _select(entries, None)
        on_bd = {dim: sel["params"][dim] in (min(vals), max(vals)) for dim, vals in (("noise_index", ks), ("eta", etas))}
        return {"entries": entries, "selected": sel["params"], "selected_val_mean_unobs_xy_mse": sel["val_mean_unobs_xy_mse"],
                "extension": ext or None, "selected_on_boundary_final": on_bd,
                "K_selected_on_grid_max": bool(sel["params"]["K"] == max(Ks) and len(Ks) > 1),
                "grid_final": {"noise_index": sorted(ks), "eta": sorted(etas), "K": Ks}}

    grids["pnp_pgd"] = tune_pnp("pnp_pgd")
    grids["pnp_pgd_ddim10_K10"] = tune_pnp("pnp_pgd_ddim10_K10")

    gt_m = ctx.metrics(ctx.x_star)
    selected = {m: grids[m]["selected"] for m in ("pdg", "dcg", "pnp_pgd", "pnp_pgd_ddim10_K10")}
    selected["unguided"] = {}
    config = {
        "phase": "val", "args": vars(args), "protocol": ctx.proto, "protocol_sha256": ic.sha256_file(ic.PROTOCOL_PATH),
        "data_npz": str(ctx.data_path), "data_npz_sha256": ic.sha256_file(ctx.data_path),
        "checkpoint": ctx.proto["checkpoint"], "checkpoint_sha256": ic.sha256_file(ctx.proto["checkpoint"]),
        "training_hdf5_for_normalizer": ctx.proto["training_hdf5"], "training_hdf5_sha256": ic.sha256_file(ctx.proto["training_hdf5"]),
        "normalizer_mean": ctx.normalizer.mean, "normalizer_std": ctx.normalizer.std,
        "xT_seeds": ctx.seeds, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "provenance": safe_io.provenance(),
    }
    safe_io.write_json(run_dir / "config.json", config)
    safe_io.write_json(run_dir / "checks.json", checks)
    safe_io.write_json(run_dir / "val_grid.json", {"grids": grids, "ground_truth_summary": _summ(gt_m)})
    safe_io.write_json(run_dir / "selected.json", selected)
    safe_io.write_npz(run_dir / "val_raw.npz", traj_ids=ctx.traj_ids, **npz,
                      **{f"ground_truth__{k}": gt_m[k] for k in METRIC_KEYS})
    log(f"selected: {json.dumps(selected)}")
    log(f"VAL_RUN_DIR {run_dir}")


# --------------------------------------------------------------------------- test / smoke


def phase_test(args, smoke=False):
    if smoke:
        ctx = Ctx(args.data_dir, "val", idx=np.arange(5))
        selected = {"unguided": {}, "pdg": {"w_cg": 1.0}, "dcg": {"eta": 1.0},
                    "pnp_pgd": {"noise_index": 5, "eta": 0.5, "K": 10},
                    "pnp_pgd_ddim10_K10": {"noise_index": 5, "eta": 0.5, "K": 10}}
        sel_src = "smoke defaults (untuned)"
    else:
        ctx = Ctx(args.data_dir, "test")
        sel_path = Path(args.val_run) / "selected.json"
        selected = json.load(open(sel_path))
        sel_src = str(sel_path)
    tag = "smoke" if smoke else "test"
    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "inverse", run_id=args.run_id, tag=tag)
    log(f"{tag} run dir {run_dir}; n={ctx.n}; selected={selected}")
    checks = correctness_checks(ctx)
    log(f"checks: {json.dumps({k: v for k, v in checks.items() if k != 'schedule_T20'})}")
    methods = ["unguided", "pdg", "dcg", "pnp_pgd", "pnp_pgd_ddim10_K10"]
    results, npz = {}, {}
    gpu = torch.cuda.get_device_name(0)

    for method in methods:
        params = selected[method]
        log(f"== {method} {params}")
        # ---- warm-up (batch 1, trajectory 0) and batch-1 timed runs
        for _ in range(args.warmup):
            p, x = ctx.sub(0)
            run_method(ctx, method, params, p, x)
        sync()
        times, t_init, t_iter, xs = [], [], [], []
        for j in range(ctx.n):
            p, x = ctx.sub(j)
            tm = {}
            with cuda_time(tm, "total_s"):
                xh, extras = run_method(ctx, method, params, p, x, timer=tm)
            times.append(tm["total_s"])
            t_init.append(tm.get("init_s", 0.0))
            t_iter.append(tm.get("iters_s", 0.0))
            xs.append(xh)
        x_b1 = torch.cat(xs, 0)
        # ---- batched timed run (throughput), then batched instrumented run
        p, x = ctx.full()
        run_method(ctx, method, params, p, x)
        tb = {}
        with cuda_time(tb, "total_s"):
            xb_timed, _ = run_method(ctx, method, params, p, x, timer=tb)
        metric = ic.TorchMetric(ctx.normalizer, ctx.gt_obs, ctx.obs_idx, ctx.rew, DEVICE) if method.startswith("pnp") else None
        xb, extras, cost = counted_run(ctx, method, params, p, x, metrics_fn=metric)
        xn = x_b1.cpu().numpy()
        m = ctx.metrics(xn)
        m_batched = ctx.metrics(xb.cpu().numpy(), with_dyn=False)
        res = {
            "params": params, "summary": _summ(m), "cost": cost,
            "time_batch1_ms": {"mean": 1e3 * float(np.mean(times)), "std_ddof1": 1e3 * float(np.std(times, ddof=1)) if ctx.n > 1 else 0.0,
                               "median": 1e3 * float(np.median(times)),
                               "init_mean": 1e3 * float(np.mean(t_init)), "iters_mean": 1e3 * float(np.mean(t_iter))},
            "time_batched_per_traj_ms": 1e3 * tb["total_s"] / ctx.n,
            "time_batched_init_per_traj_ms": 1e3 * tb.get("init_s", 0.0) / ctx.n,
            "batch1_vs_batched_x_maxabs": float((x_b1 - xb).abs().max()),
            "batched_timed_vs_counted_x_maxabs": float((xb_timed - xb).abs().max()),
            "batched_mean_unobs_xy_mse": float(np.mean(m_batched["unobs_xy_mse"])),
            "pin_maxabs_norm": float((x_b1[:, 0, :4] - ctx.prior[:, 0, :4]).abs().max()),
        }
        if method.startswith("pnp"):
            res["sigma_bar"] = extras["sigma_bar"]
            res["model_t_seen_unique"] = extras["model_t_seen_unique"]
            res["alpha_sigma"] = [extras["alpha"], extras["sigma"]]
            curve = np.stack([d["unobs_xy_mse"] for d in extras["pnp_metrics"]])
            fcurve = np.stack([d["f_norm"] for d in extras["pnp_metrics"]])
            npz[f"{method}__iter_unobs_xy_mse_batched"] = curve
            npz[f"{method}__iter_f_norm_batched"] = fcurve
            npz[f"{method}__x_init_batched"] = extras["x_init"].cpu().numpy()
            res["iter_mean_unobs_xy_mse_batched"] = curve.mean(1)
        npz[f"{method}__x"] = xn
        npz[f"{method}__x_batched"] = xb.cpu().numpy()
        npz[f"{method}__time_batch1_s"] = np.asarray(times)
        npz[f"{method}__time_init_s"] = np.asarray(t_init)
        npz[f"{method}__time_iters_s"] = np.asarray(t_iter)
        for k in METRIC_KEYS:
            npz[f"{method}__{k}"] = m[k]
        results[method] = res
        log(f"   unobs-xy MSE {res['summary']['unobs_xy_mse']['mean']:.5f} ± {res['summary']['unobs_xy_mse']['std_ddof1']:.5f}; "
            f"obs RMSE {res['summary']['obs_xy_rmse']['mean']:.4f}; e_dyn {res['summary']['e_dyn']['mean']:.4f}; "
            f"time {res['time_batch1_ms']['mean']:.2f} ms; cost {cost['measured_per_sample']} match={cost['match']}; "
            f"b1-vs-batched {res['batch1_vs_batched_x_maxabs']:.2e}")

    # PnP init must equal the unguided sample of the same seed
    if "pnp_pgd" in results:
        d = float(np.abs(npz["pnp_pgd__x_init_batched"] - npz["unguided__x_batched"]).max())
        checks["pnp_init_vs_unguided_batched_maxabs"] = d
    gt_m = ctx.metrics(ctx.x_star)
    for k in METRIC_KEYS:
        npz[f"ground_truth__{k}"] = gt_m[k]
    results["ground_truth"] = {"summary": _summ(gt_m)}

    config = {
        "phase": tag, "args": vars(args), "selected_from": sel_src, "selected": selected,
        "protocol": ctx.proto, "protocol_sha256": ic.sha256_file(ic.PROTOCOL_PATH),
        "data_npz": str(ctx.data_path), "data_npz_sha256": ic.sha256_file(ctx.data_path), "traj_ids": ctx.traj_ids,
        "checkpoint": ctx.proto["checkpoint"], "checkpoint_sha256": ic.sha256_file(ctx.proto["checkpoint"]),
        "training_hdf5_for_normalizer": ctx.proto["training_hdf5"], "training_hdf5_sha256": ic.sha256_file(ctx.proto["training_hdf5"]),
        "normalizer_mean": ctx.normalizer.mean, "normalizer_std": ctx.normalizer.std, "xT_seeds": ctx.seeds,
        "timing_convention": ctx.proto["timing"], "warmup_calls": args.warmup, "gpu": gpu, "torch": torch.__version__,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "provenance": safe_io.provenance(),
    }
    safe_io.write_json(run_dir / "config.json", config)
    safe_io.write_json(run_dir / "checks.json", checks)
    safe_io.write_json(run_dir / "results.json", results)
    safe_io.write_npz(run_dir / "raw.npz", traj_ids=ctx.traj_ids, gt_obs=ctx.gt_obs, gt_act=ctx.gt_act,
                      x_star_norm=ctx.x_star, obs_idx=ctx.obs_idx, M=ctx.M, Y=ctx.Y, xT=ctx.xT.cpu().numpy(),
                      seeds=ctx.seeds, **npz)
    log(f"{tag.upper()}_RUN_DIR {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "val", "test"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--val-run", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cudnn.benchmark = False
    if args.phase == "val":
        phase_val(args)
    else:
        if args.phase == "test" and not args.val_run:
            raise SystemExit("--val-run required for test")
        phase_test(args, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()
