#!/usr/bin/env python
"""PnP-PGD hyperparameter selection on the unicycle VALIDATION set (disjoint from test).

Validation references: half_heart, random_smooth_2, random_smooth_4; seeds 1000..1019.
Grid: k in {2,5,9} (T=20 schedule), eta in {0.25,0.5,1.0}/L, K in {5,10,20}.
K is obtained from the iterates of a single K=20 run (PnP-PGD is deterministic given
x_init, so the K=5/10 iterates are exactly the first 5/10 iterates of the K=20 run).

Criterion (fixed in advance): minimise the validation mean c_rollout, averaged over the
3 validation references (each reference: mean over 20 seeds); ties (equal after
rounding to 1e-9) -> smaller mean e_dyn. If the selected k or eta lies on the grid
boundary, the grid is extended ONCE by one point in that direction
(k: 1 below / 13 above; eta: 0.125/L below / 2.0/L above), the full cross product of
the extended grid is evaluated, and the selection is redone on the extended grid.

Two initialisation variants are selected independently:
  * ``mc``  : unguided DDIM-20 (Monte Carlo) plan of the same seed  (primary PnP-PGD)
  * ``xT``  : the Gaussian x_T of the same seed                      (ablation b)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unicycle_common as uc  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402

K_GRID = [2, 5, 9]
ETA_MULT_GRID = [0.25, 0.5, 1.0]
K_ITERS_GRID = [5, 10, 20]
K_EXT = {"low": 1, "high": 13}
ETA_EXT = {"low": 0.125, "high": 2.0}
VAL_SEEDS = list(range(1000, 1020))


def select(grid: dict, ks, etas, iters) -> dict:
    rows = []
    for k, em, K in itertools.product(ks, etas, iters):
        g = grid[(k, em)]
        c_roll = float(np.mean(g["c_rollout"][:, :, K].mean(axis=1)))
        e_dyn = float(np.mean(g["e_dyn"][:, :, K].mean(axis=1)))
        rows.append({"k": k, "eta_mult": em, "K": K, "val_c_rollout": c_roll, "val_e_dyn": e_dyn})
    best = min(rows, key=lambda r: (round(r["val_c_rollout"], 9), r["val_e_dyn"]))
    return {"best": best, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variants", nargs="+", default=["mc", "xT"])
    ap.add_argument("--n-seeds", type=int, default=len(VAL_SEEDS), help="smoke tests only")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "unicycle", run_id=args.run_id)
    seeds = VAL_SEEDS[: args.n_seeds]
    config = {
        "phase": "validation",
        "references": uc.VAL_REFERENCES,
        "seeds": seeds,
        "k_grid": K_GRID,
        "eta_mult_grid": ETA_MULT_GRID,
        "eta_grid": [m / uc.LIPSCHITZ for m in ETA_MULT_GRID],
        "K_grid": K_ITERS_GRID,
        "max_iters_run": max(K_ITERS_GRID),
        "boundary_extension": {"k": K_EXT, "eta_mult": ETA_EXT},
        "criterion": "min mean_ref(mean_seed c_rollout at K); ties (round 1e-9) -> smaller e_dyn",
        "input_scaling": "vp",
        "variants": args.variants,
        "static": uc.static_config(),
        "provenance": safe_io.provenance(),
        "argv": sys.argv,
    }
    safe_io.write_json(run_dir / "config.json", config)

    setup = uc.load_setup(args.device)
    sbars = {k: float(uc.make_denoiser(setup, uc.make_ref(setup, uc.VAL_REFERENCES[0]), k).sigma_bar)
             for k in sorted(set(K_GRID) | set(K_EXT.values()))}
    ctxs = {r: uc.make_ref(setup, r) for r in uc.VAL_REFERENCES}
    n_iter = max(K_ITERS_GRID)
    out = {}

    for variant in args.variants:
        t0 = time.time()
        inits = {}
        init_metrics = {m: np.zeros((len(ctxs), len(seeds))) for m in ("c_plan", "c_rollout", "e_dyn")}
        for ri, (rname, ctx) in enumerate(ctxs.items()):
            for si, s in enumerate(seeds):
                if variant == "mc":
                    init = uc.unguided_init(setup, ctx, s, 20)
                    x0 = uc.to_device_plan(setup, init["plan_norm"])
                else:
                    x0 = uc.gaussian_init(setup, ctx, s)["x"]
                inits[(rname, s)] = x0
                ev = uc.evaluate_plan(setup, ctx, x0[0].cpu().numpy())
                init_metrics["c_plan"][ri, si] = -ev["plan_reward"]
                init_metrics["c_rollout"][ri, si] = -ev["rollout_reward"]
                init_metrics["e_dyn"][ri, si] = ev["feasibility_mean_l2_future"]

        grid = {}

        def eval_point(k, em):
            if (k, em) in grid:
                return
            eta = em / uc.LIPSCHITZ
            arr = {m: np.zeros((len(ctxs), len(seeds), n_iter + 1)) for m in ("c_plan", "c_rollout", "e_dyn")}
            costs = []
            for ri, (rname, ctx) in enumerate(ctxs.items()):
                for si, s in enumerate(seeds):
                    res = uc.run_pnp(setup, ctx, inits[(rname, s)], k=k, eta=eta, n_iters=n_iter, record_history=True)
                    costs.append(res["costs"])
                    for it in range(n_iter + 1):
                        ev = uc.evaluate_plan(setup, ctx, res["history"][it])
                        arr["c_plan"][ri, si, it] = -ev["plan_reward"]
                        arr["c_rollout"][ri, si, it] = -ev["rollout_reward"]
                        arr["e_dyn"][ri, si, it] = ev["feasibility_mean_l2_future"]
            arr["costs_example"] = costs[0]
            arr["costs_all_equal"] = all(c == costs[0] for c in costs)
            grid[(k, em)] = arr
            print(f"[val {variant}] k={k} eta={em}/L: c_rollout@K=5,10,20 = "
                  + ", ".join(f"{arr['c_rollout'][:, :, K].mean():.4f}" for K in K_ITERS_GRID), flush=True)

        ks, ems = list(K_GRID), list(ETA_MULT_GRID)
        for k, em in itertools.product(ks, ems):
            eval_point(k, em)
        sel0 = select(grid, ks, ems, K_ITERS_GRID)
        extension = {"k": None, "eta_mult": None}
        b = sel0["best"]
        if b["k"] == min(K_GRID):
            extension["k"] = K_EXT["low"]
        elif b["k"] == max(K_GRID):
            extension["k"] = K_EXT["high"]
        if b["eta_mult"] == min(ETA_MULT_GRID):
            extension["eta_mult"] = ETA_EXT["low"]
        elif b["eta_mult"] == max(ETA_MULT_GRID):
            extension["eta_mult"] = ETA_EXT["high"]
        if extension["k"] is not None:
            ks = sorted(ks + [extension["k"]])
        if extension["eta_mult"] is not None:
            ems = sorted(ems + [extension["eta_mult"]])
        for k, em in itertools.product(ks, ems):
            eval_point(k, em)
        sel1 = select(grid, ks, ems, K_ITERS_GRID)
        b1 = sel1["best"]
        still_boundary = {
            "k": b1["k"] in (min(ks), max(ks)),
            "eta_mult": b1["eta_mult"] in (min(ems), max(ems)),
            "K": b1["K"] in (min(K_ITERS_GRID), max(K_ITERS_GRID)),
        }

        # Save full grid
        keys = sorted(grid.keys())
        safe_io.write_npz(
            run_dir / f"validation_grid_{variant}.npz",
            k=np.array([kk for kk, _ in keys]),
            eta_mult=np.array([e for _, e in keys]),
            sigma_bar=np.array([sbars[kk] for kk, _ in keys]),
            c_plan=np.stack([grid[p]["c_plan"] for p in keys]),
            c_rollout=np.stack([grid[p]["c_rollout"] for p in keys]),
            e_dyn=np.stack([grid[p]["e_dyn"] for p in keys]),
            init_c_plan=init_metrics["c_plan"], init_c_rollout=init_metrics["c_rollout"], init_e_dyn=init_metrics["e_dyn"],
            references=np.array(list(ctxs.keys())), seeds=np.array(seeds),
        )
        per_ref_rows = []
        for (k, em) in keys:
            for K in range(n_iter + 1):
                row = {"k": k, "sigma_bar": sbars[k], "eta_mult": em, "eta": em / uc.LIPSCHITZ, "K": K,
                       "in_prespecified_grid": (k in K_GRID and em in ETA_MULT_GRID and K in K_ITERS_GRID),
                       "in_selection_grid": (k in ks and em in ems and K in K_ITERS_GRID)}
                for ri, r in enumerate(ctxs):
                    for m in ("c_plan", "c_rollout", "e_dyn"):
                        row[f"{r}/{m}_mean"] = float(grid[(k, em)][m][ri, :, K].mean())
                        row[f"{r}/{m}_std"] = float(grid[(k, em)][m][ri, :, K].std(ddof=1)) if len(seeds) > 1 else 0.0
                for m in ("c_plan", "c_rollout", "e_dyn"):
                    row[f"avg/{m}"] = float(np.mean(grid[(k, em)][m][:, :, K].mean(axis=1)))
                per_ref_rows.append(row)
        result = {
            "variant": variant,
            "selection_initial_grid": sel0,
            "boundary_extension_applied": extension,
            "selection_final": sel1,
            "selected": {**b1, "eta": b1["eta_mult"] / uc.LIPSCHITZ, "sigma_bar": sbars[b1["k"]], "input_scaling": "vp"},
            "still_on_boundary_after_extension": still_boundary,
            "grid_points_evaluated": [list(p) for p in keys],
            "sigma_bar_by_k": sbars,
            "init_metrics_avg": {m: float(np.mean(v.mean(axis=1))) for m, v in init_metrics.items()},
            "costs_example_per_candidate": {f"{p[0]}_{p[1]}": grid[p]["costs_example"] for p in keys},
            "costs_all_equal": {f"{p[0]}_{p[1]}": grid[p]["costs_all_equal"] for p in keys},
            "rows_all_iterations": per_ref_rows,
            "wallclock_s": time.time() - t0,
        }
        safe_io.write_json(run_dir / f"validation_{variant}.json", result)
        out[variant] = result["selected"]
        print(f"[val {variant}] initial best={sel0['best']} extension={extension} final={b1}", flush=True)

    safe_io.write_json(run_dir / "selection.json", {"selected": out, "validation_run_dir": str(run_dir)})
    print(f"Wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
