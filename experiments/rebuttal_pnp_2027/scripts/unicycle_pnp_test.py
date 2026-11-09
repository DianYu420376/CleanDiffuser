#!/usr/bin/env python
"""Unicycle TEST evaluation: re-run Monte Carlo / PDG / DCG / DRGD and the frozen PnP-PGD
variants on circle, random_smooth_1, random_smooth_7 with candidate seeds 0..99.

All methods run in this single GPU job. Per candidate: set_seed(seed); the sampling call
(and for PnP-PGD the init + iterations) is timed with torch.cuda.synchronize() before and
after; env rollout and metric computation are timed separately (not part of runtime).
Warm-up: 3 throwaway candidates (seeds 90000..90002) per (reference, method) before the
timed loop. Batching: n_samples=1 sequential (the paper protocol). Counters (forward
hooks on both agents' EMA networks + GradCounter on the reward classifier) are attached
during the timed region for every method.

PnP-PGD variants (hyperparameters frozen from the validation selection.json):
  pnp_pgd           : init = Monte Carlo DDIM-20 plan of the seed; selected (k, eta, K); vp scaling
  pnp_pgd_budget20  : ablation (a) init = DDIM-10 (T=20) unguided plan + K=10, selected (k, eta)
  pnp_pgd_from_xT   : ablation (b) init = Gaussian x_T of the seed; (k, eta, K) selected separately on validation
  pnp_pgd_noscale   : ablation (c) input_scaling="none" with the selected (k, eta, K) of pnp_pgd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unicycle_common as uc  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402

WARMUP_SEEDS = [90000, 90001, 90002]
ALL_METHODS = ["monte_carlo", "pdg", "dcg", "drgd", "pnp_pgd", "pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale"]


def pnp_specs(selection: dict) -> dict:
    mc, xt = selection["mc"], selection["xT"]
    return {
        "pnp_pgd": dict(init="mc20", k=mc["k"], eta=mc["eta"], eta_mult=mc["eta_mult"], K=mc["K"], input_scaling="vp"),
        "pnp_pgd_budget20": dict(init="mc10", k=mc["k"], eta=mc["eta"], eta_mult=mc["eta_mult"], K=10, input_scaling="vp"),
        "pnp_pgd_from_xT": dict(init="xT", k=xt["k"], eta=xt["eta"], eta_mult=xt["eta_mult"], K=xt["K"], input_scaling="vp"),
        "pnp_pgd_noscale": dict(init="mc20", k=mc["k"], eta=mc["eta"], eta_mult=mc["eta_mult"], K=mc["K"], input_scaling="none"),
    }


def run_pnp_candidate(setup, ctx, spec, seed, *, with_history: bool):
    if spec["init"] == "mc20":
        init = uc.unguided_init(setup, ctx, seed, 20)
        x0 = uc.to_device_plan(setup, init["plan_norm"])
    elif spec["init"] == "mc10":
        init = uc.unguided_init(setup, ctx, seed, 10)
        x0 = uc.to_device_plan(setup, init["plan_norm"])
    else:
        init = uc.gaussian_init(setup, ctx, seed)
        x0 = init["x"]
        init["plan_norm"] = x0[0].detach().cpu().numpy()
    res = uc.run_pnp(setup, ctx, x0, k=spec["k"], eta=spec["eta"], n_iters=spec["K"],
                     input_scaling=spec["input_scaling"], record_history=False)
    hist = None
    hist_maxdiff = None
    if with_history:  # untimed second pass for per-iteration metrics (deterministic)
        res_h = uc.run_pnp(setup, ctx, x0, k=spec["k"], eta=spec["eta"], n_iters=spec["K"],
                           input_scaling=spec["input_scaling"], record_history=True)
        hist = res_h["history"]
        hist_maxdiff = float(np.abs(hist[-1] - res["plan_norm"]).max())
    return init, res, hist, hist_maxdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--selection-json", required=True)
    ap.add_argument("--references", nargs="+", default=uc.TEST_REFERENCES)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--n-candidates", type=int, default=100)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    selection = json.load(open(args.selection_json))["selected"]
    specs = pnp_specs(selection)
    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "unicycle", run_id=args.run_id)
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_candidates))
    config = {
        "phase": "test",
        "references": args.references,
        "methods": args.methods,
        "seeds": seeds,
        "warmup_seeds": WARMUP_SEEDS,
        "selection_json": args.selection_json,
        "selection": selection,
        "pnp_specs": specs,
        "timing": "per-candidate wall clock of sampling only (cuda synchronize before/after); excludes model loading, env rollout and metrics; n_samples=1 sequential; counters attached during timed region",
        "static": uc.static_config(),
        "provenance": safe_io.provenance(),
        "argv": sys.argv,
    }
    safe_io.write_json(run_dir / "config.json", config)

    setup = uc.load_setup(args.device)
    mc_plans = {}
    for rname in args.references:
        ctx = uc.make_ref(setup, rname)
        for method in args.methods:
            is_pnp = method.startswith("pnp")
            spec = specs.get(method)
            # Warm-up (throwaway)
            for ws in WARMUP_SEEDS:
                if is_pnp:
                    run_pnp_candidate(setup, ctx, spec, ws, with_history=False)
                else:
                    uc.run_baseline_candidate(setup, ctx, method, ws)
            cands, plans, costs, per_iter = [], [], [], []
            t_method = time.time()
            for ci, s in enumerate(seeds):
                rec = {"candidate_idx": ci, "seed": s}
                if is_pnp:
                    init, res, hist, hd = run_pnp_candidate(setup, ctx, spec, s, with_history=True)
                    plan = res["plan_norm"]
                    c = {
                        "init_nfe": init["costs"]["nfe"], "init_net_backward": init["costs"]["net_backward"],
                        "init_grad_evals": init["costs"]["grad_evals"], "init_seconds": init["seconds"],
                        "iter_nfe": res["costs"]["nfe"], "iter_net_backward": res["costs"]["net_backward"],
                        "iter_grad_evals": res["costs"]["grad_evals"], "iter_seconds": res["seconds"],
                        "pnp_reported_denoiser_evals": res["n_denoiser_evals"], "pnp_reported_grad_evals": res["n_grad_evals"],
                        "history_final_maxabsdiff": hd,
                    }
                    c["nfe"] = c["init_nfe"] + c["iter_nfe"]
                    c["net_backward"] = c["init_net_backward"] + c["iter_net_backward"]
                    c["grad_evals"] = c["init_grad_evals"] + c["iter_grad_evals"]
                    c["seconds"] = c["init_seconds"] + c["iter_seconds"]
                    if spec["init"] == "mc20" and (rname, s) in mc_plans:
                        c["init_vs_monte_carlo_maxabsdiff"] = float(np.abs(init["plan_norm"] - mc_plans[(rname, s)]).max())
                    t0 = time.perf_counter()
                    it_metrics = [uc.evaluate_plan(setup, ctx, h) for h in hist]
                    per_iter.append({
                        "c_plan": [-m["plan_reward"] for m in it_metrics],
                        "c_rollout": [-m["rollout_reward"] for m in it_metrics],
                        "e_dyn": [m["feasibility_mean_l2_future"] for m in it_metrics],
                        "history": hist,
                    })
                    c["per_iter_eval_seconds"] = time.perf_counter() - t0
                    c["sigma_bar"] = res["sigma_bar"]
                    c["model_t_seen"] = res["model_t_seen"]
                else:
                    out = uc.run_baseline_candidate(setup, ctx, method, s)
                    plan = out["plan_norm"]
                    c = {**out["costs"], "seconds": out["seconds"]}
                    if method == "monte_carlo":
                        mc_plans[(rname, s)] = plan
                t0 = time.perf_counter()
                ev = uc.evaluate_plan(setup, ctx, plan)
                c["rollout_and_metric_seconds"] = time.perf_counter() - t0
                rec.update(ev)
                cands.append(rec)
                plans.append(plan)
                costs.append(c)
            summ = uc.summarize(cands)
            out_dir = safe_io.ensure_dir(run_dir / rname / method / "reference_plan" / "trajectories")
            safe_io.write_npz(out_dir / "candidates.npz", **uc.pack_candidates(ctx, cands))
            safe_io.write_npz(out_dir / "plans_norm.npz", plan_norm=np.stack(plans).astype(np.float32),
                              seed=np.asarray(seeds, dtype=np.int32))
            ranked = sorted(cands, key=lambda c: c["plan_reward"], reverse=True)
            safe_io.write_json(out_dir / "candidates.meta.json", {
                "reference_name": rname, "config_name": method, "horizon": uc.HORIZON, "n_candidates": len(cands),
                "method_spec": spec if is_pnp else uc.BASELINES[method],
                "ranked_candidates": [
                    {"rank": r, **{k: cand[k] for k in ("candidate_idx", "seed", "plan_reward", "rollout_reward",
                                                        "feasibility_mean_l2_future", "feasibility_mean_l2_norm_future",
                                                        "plan_start_error")}}
                    for r, cand in enumerate(ranked, start=1)],
            })
            keys = [k for k in costs[0] if isinstance(costs[0][k], (int, float)) and costs[0][k] is not None]
            cost_arrays = {k: [cc.get(k) for cc in costs] for k in costs[0]}
            cost_summary = {k: {"mean": float(np.mean([cc[k] for cc in costs])), "std": float(np.std([cc[k] for cc in costs], ddof=1)),
                                "min": float(np.min([cc[k] for cc in costs])), "max": float(np.max([cc[k] for cc in costs]))}
                            for k in keys if all(isinstance(cc.get(k), (int, float)) for cc in costs)}
            safe_io.write_json(out_dir / "costs_timing.json", {
                "per_candidate": cost_arrays, "summary": cost_summary,
                "timing_convention": config["timing"], "warmup_seeds": WARMUP_SEEDS,
                "wallclock_method_total_s": time.time() - t_method,
            })
            safe_io.write_json(out_dir / "summary.json", {
                "reference_name": rname, "method": method, "metrics": summ,
                "costs": cost_summary, "spec": spec if is_pnp else uc.BASELINES[method],
                "init_state": list(ctx.init_state),
            })
            if is_pnp:
                safe_io.write_npz(out_dir / "per_iteration.npz",
                                  c_plan=np.array([p["c_plan"] for p in per_iter]),
                                  c_rollout=np.array([p["c_rollout"] for p in per_iter]),
                                  e_dyn=np.array([p["e_dyn"] for p in per_iter]),
                                  history_plan_norm=np.stack([p["history"] for p in per_iter]).astype(np.float32),
                                  seed=np.asarray(seeds, dtype=np.int32),
                                  init_nfe=np.array([cc["init_nfe"] for cc in costs]))
            print(f"[test] {rname} {method}: c_plan={summ['c_plan']['mean']:.4f}±{summ['c_plan']['std']:.4f} "
                  f"c_rollout={summ['c_rollout']['mean']:.4f}±{summ['c_rollout']['std']:.4f} "
                  f"e_dyn={summ['e_dyn']['mean']:.4f}±{summ['e_dyn']['std']:.4f} | "
                  f"nfe={cost_summary['nfe']['mean']:.1f} bwd={cost_summary['net_backward']['mean']:.1f} "
                  f"grads={cost_summary['grad_evals']['mean']:.1f} ms={1e3 * cost_summary['seconds']['mean']:.2f}", flush=True)
        ctx.env.close()
    print(f"Wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
