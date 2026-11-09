#!/usr/bin/env python
"""Unicycle PnP-PGD: post-selection supplementary GPU job (reviewer fixes, v2).

NOTHING here is used for hyperparameter selection: the frozen selection (val_3565048/selection.json)
is read and kept; the test-set numbers of Table 2 stay those of test_3565048.

Part A  (reporting only, untimed): on the 3 test references and seeds 0..99, run every PnP-PGD variant
        with its frozen (k, eta) up to K_MAX=20 iterations with record_history=True and evaluate EVERY
        iterate with the same ue post-processing (unnormalize, clip, open-loop rollout, metrics).
        Checks: iterates 0..K* are bit-identical to the test_3565048 per_iteration.npz history (determinism),
        and the final K* iterate metrics reproduce the test_3565048 candidates.npz.
Part B  (timing diagnostic): explains why the PnP-PGD init (the same ue._sample_plan call as Monte Carlo)
        was timed ~6-9 ms slower than the Monte Carlo baseline in test_3565048. For each test reference,
        100 candidates (seeds 0..99), 3 warm-up candidates first, synchronize around timed regions, under
        several loop protocols:
          B1 mc_only            : run_baseline_candidate("monte_carlo") then evaluate_plan (= test protocol for MC)
          B2 init_only          : unguided_init(20) then evaluate_plan
          B3 init_plus_iter     : unguided_init(20) + run_pnp(K*) (no history pass), then evaluate_plan
          B4 test_protocol      : exactly the test_3565048 PnP loop body (init + timed iterations + untimed
                                  history pass + per-iteration evaluation + final evaluation)
          B5 test_protocol_nogc : B4 with Python's cyclic GC disabled for the loop and gc.collect() run
                                  between candidates (outside the timed regions)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unicycle_common as uc  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402
from unicycle_pnp_test import pnp_specs, run_pnp_candidate, WARMUP_SEEDS  # noqa: E402

K_MAX = 20
VARIANTS = ["pnp_pgd", "pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale"]


def eval_hist(setup, ctx, hist):
    ms_ = [uc.evaluate_plan(setup, ctx, h) for h in hist]
    return (np.array([-m["plan_reward"] for m in ms_]), np.array([-m["rollout_reward"] for m in ms_]),
            np.array([m["feasibility_mean_l2_future"] for m in ms_]))


def part_a(setup, specs, run_dir, ref_test_dir, refs, seeds):
    out = {}
    for rname in refs:
        ctx = uc.make_ref(setup, rname)
        for v in VARIANTS:
            spec = dict(specs[v])
            Kstar = spec["K"]
            spec_long = dict(spec, K=K_MAX)
            old_it = dict(np.load(ref_test_dir / rname / v / "reference_plan" / "trajectories" / "per_iteration.npz"))
            old_c = dict(np.load(ref_test_dir / rname / v / "reference_plan" / "trajectories" / "candidates.npz"))
            cp, cr, ed, hists, maxdiff_hist = [], [], [], [], []
            for ci, s in enumerate(seeds):
                init, res, hist, _ = run_pnp_candidate(setup, ctx, spec_long, s, with_history=True)
                a, b, c = eval_hist(setup, ctx, hist)
                cp.append(a); cr.append(b); ed.append(c); hists.append(hist)
                maxdiff_hist.append(float(np.abs(hist[: Kstar + 1] - old_it["history_plan_norm"][ci]).max()))
            cp, cr, ed = np.stack(cp), np.stack(cr), np.stack(ed)
            n = len(seeds)
            assert np.array_equal(old_c["seed"][:n], np.asarray(seeds))
            old_c = {k: old_c[k][:n] for k in ("rollout_reward", "plan_reward", "feasibility_mean_l2_future")}
            chk = {
                "K_star": Kstar,
                "maxabs_history_vs_test_3565048_iters_0..Kstar": float(np.max(maxdiff_hist)),
                "maxabs_c_rollout_at_Kstar_vs_test_candidates": float(np.abs(cr[:, Kstar] + old_c["rollout_reward"]).max()),
                "maxabs_c_plan_at_Kstar_vs_test_candidates": float(np.abs(cp[:, Kstar] + old_c["plan_reward"]).max()),
                "maxabs_e_dyn_at_Kstar_vs_test_candidates": float(np.abs(ed[:, Kstar] - old_c["feasibility_mean_l2_future"]).max()),
            }
            d = safe_io.ensure_dir(run_dir / "partA_K20" / rname / v)
            safe_io.write_npz(d / "per_iteration_K20.npz", c_plan=cp, c_rollout=cr, e_dyn=ed,
                              history_plan_norm=np.stack(hists).astype(np.float32),
                              seed=np.asarray(seeds, dtype=np.int32),
                              init_nfe=np.array([0 if spec["init"] == "xT" else (20 if spec["init"] == "mc20" else 10)] * len(seeds)))
            summ = {"spec": spec_long, "frozen_K_star": Kstar, "checks": chk,
                    "mean_by_K": {"c_plan": cp.mean(0).tolist(), "c_rollout": cr.mean(0).tolist(), "e_dyn": ed.mean(0).tolist()},
                    "std_by_K": {"c_plan": cp.std(0, ddof=1).tolist(), "c_rollout": cr.std(0, ddof=1).tolist(),
                                 "e_dyn": ed.std(0, ddof=1).tolist()}}
            safe_io.write_json(d / "summary.json", summ)
            out[f"{rname}/{v}"] = chk
            print(f"[A] {rname} {v}: c_rollout K=0,{Kstar},10,20 = {cr.mean(0)[0]:.4f}, {cr.mean(0)[Kstar]:.4f}, "
                  f"{cr.mean(0)[10]:.4f}, {cr.mean(0)[20]:.4f} | checks {chk}", flush=True)
        ctx.env.close()
    return out


def part_b(setup, specs, run_dir, refs, seeds):
    spec = specs["pnp_pgd"]
    out = {}

    def body(protocol, ctx, s):
        rec = {}
        if protocol == "mc_only":
            o = uc.run_baseline_candidate(setup, ctx, "monte_carlo", s)
            rec["sample_s"] = o["seconds"]
            uc.evaluate_plan(setup, ctx, o["plan_norm"])
        elif protocol == "init_only":
            o = uc.unguided_init(setup, ctx, s, 20)
            rec["sample_s"] = o["seconds"]
            uc.evaluate_plan(setup, ctx, o["plan_norm"])
        elif protocol == "init_plus_iter":
            init, res, _, _ = run_pnp_candidate(setup, ctx, spec, s, with_history=False)
            rec["init_s"], rec["iter_s"] = init["seconds"], res["seconds"]
            uc.evaluate_plan(setup, ctx, res["plan_norm"])
        else:  # test_protocol(_nogc)
            init, res, hist, _ = run_pnp_candidate(setup, ctx, spec, s, with_history=True)
            rec["init_s"], rec["iter_s"] = init["seconds"], res["seconds"]
            [uc.evaluate_plan(setup, ctx, h) for h in hist]
            uc.evaluate_plan(setup, ctx, res["plan_norm"])
        return rec

    protocols = ["mc_only", "init_only", "init_plus_iter", "test_protocol", "test_protocol_nogc", "mc_only_repeat"]
    for rname in refs:
        ctx = uc.make_ref(setup, rname)
        for p in protocols:
            proto = "mc_only" if p == "mc_only_repeat" else p
            nogc = proto == "test_protocol_nogc"
            for ws in WARMUP_SEEDS:
                body(proto, ctx, ws)
            if nogc:
                gc.collect()
                gc.disable()
            recs = []
            try:
                for s in seeds:
                    recs.append(body(proto, ctx, s))
                    if nogc:
                        gc.collect()
            finally:
                if nogc:
                    gc.enable()
            arr = {k: np.array([r[k] for r in recs]) for k in recs[0]}
            safe_io.write_npz(run_dir / "partB_timing" / rname / f"{p}.npz", **arr, seed=np.asarray(seeds))
            summ = {k: {"mean_ms": 1e3 * float(v.mean()), "std_ms": 1e3 * float(v.std(ddof=1)),
                        "median_ms": 1e3 * float(np.median(v))} for k, v in arr.items()}
            out[f"{rname}/{p}"] = summ
            print(f"[B] {rname} {p}: " + ", ".join(f"{k} {v['mean_ms']:.2f}±{v['std_ms']:.2f} (med {v['median_ms']:.2f}) ms"
                                                  for k, v in summ.items()), flush=True)
        ctx.env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--selection-json", required=True, type=Path)
    ap.add_argument("--ref-test-dir", required=True, type=Path)
    ap.add_argument("--references", nargs="+", default=uc.TEST_REFERENCES)
    ap.add_argument("--n-candidates", type=int, default=100)
    ap.add_argument("--parts", nargs="+", default=["A", "B"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    selection = json.load(open(args.selection_json))["selected"]
    specs = pnp_specs(selection)
    seeds = list(range(args.n_candidates))
    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "unicycle", run_id=args.run_id)
    safe_io.write_json(run_dir / "config.json", {
        "phase": "post-selection supplementary (reporting/diagnostic only; NOT used for selection)",
        "selection_json": str(args.selection_json), "selection": selection, "pnp_specs": specs, "K_max": K_MAX,
        "ref_test_dir": str(args.ref_test_dir), "references": args.references, "seeds": seeds,
        "warmup_seeds": WARMUP_SEEDS, "parts": args.parts, "static": uc.static_config(),
        "provenance": safe_io.provenance(), "argv": sys.argv, "doc": __doc__,
    })
    setup = uc.load_setup(args.device)
    t0 = time.time()
    res = {}
    if "A" in args.parts:
        res["partA_checks"] = part_a(setup, specs, run_dir, args.ref_test_dir, args.references, seeds)
    if "B" in args.parts:
        res["partB_timing_summary"] = part_b(setup, specs, run_dir, args.references, seeds)
    res["wallclock_s"] = time.time() - t0
    safe_io.write_json(run_dir / "supp_summary.json", res)
    print(f"Wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
