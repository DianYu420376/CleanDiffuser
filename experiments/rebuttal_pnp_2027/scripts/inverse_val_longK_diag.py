#!/usr/bin/env python3
"""DIAGNOSTIC ONLY (validation set, never used for selection, never run on test).

The pre-registered PnP-PGD selection ended at the K=20 grid maximum with the
validation error still decreasing. This script records validation per-iteration
unobserved-xy MSE / f for longer runs (K up to --kmax) at the extended-grid
(noise_index, eta) values closest to the selection, so the reader can judge whether
PnP-PGD is budget-limited. Writes results/inverse/<run_id>/.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inverse_common as ic  # noqa: E402
import inverse_run as ir  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--kmax", type=int, default=200)
    ap.add_argument("--noise-indices", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--etas", type=float, nargs="+", default=[1.0, 1.5])
    args = ap.parse_args()
    ctx = ir.Ctx(args.data_dir, "val")
    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "inverse", run_id=args.run_id, tag="val_longK_diag")
    prior, xT = ctx.full()
    x0 = ic.sample_unguided(ctx.agent20, prior, xT, 20)
    out, summ = {}, {}
    for k, eta in itertools.product(args.noise_indices, args.etas):
        metric = ic.TorchMetric(ctx.normalizer, ctx.gt_obs, ctx.obs_idx, ctx.rew, ir.DEVICE)
        res, D = ic.run_pnp(ctx.agent20, ctx.clf, prior, x0, k, eta, args.kmax, metrics_fn=metric)
        c = np.stack([d["unobs_xy_mse"] for d in res.metrics])
        f = np.stack([d["f_norm"] for d in res.metrics])
        out[f"k{k}_eta{eta}__iter_unobs_xy_mse"] = c
        out[f"k{k}_eta{eta}__iter_f_norm"] = f
        m = c.mean(1)
        summ[f"k{k}_eta{eta}"] = {"sigma_bar": D.sigma_bar, "mean_at_K": {int(K): float(m[K]) for K in (0, 5, 10, 20, 50, 100, 200) if K <= args.kmax},
                                  "min_mean": float(m.min()), "argmin_K": int(m.argmin()),
                                  "f_mean_at_K": {int(K): float(f.mean(1)[K]) for K in (0, 20, 50, 100, 200) if K <= args.kmax}}
        ir.log(f"k={k} eta={eta}: {summ[f'k{k}_eta{eta}']}")
    safe_io.write_npz(run_dir / "diag_raw.npz", **out)
    safe_io.write_json(run_dir / "diag_summary.json", {
        "NOTE": "diagnostic only; validation set; not used for selection; test results unaffected",
        "args": vars(args), "summary": summ, "data_npz_sha256": ic.sha256_file(ctx.data_path),
        "checkpoint_sha256": ic.sha256_file(ctx.proto["checkpoint"]), "gpu": torch.cuda.get_device_name(0),
        "provenance": safe_io.provenance()})
    ir.log(f"DIAG_RUN_DIR {run_dir}")


if __name__ == "__main__":
    main()
