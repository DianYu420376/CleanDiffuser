# Sparse trajectory reconstruction (linear inverse problem) — Table 3

Fixed-noise PnP-PGD `x^{k+1} = D_sigma_bar(x^k - eta * grad f(x^k))` vs. unguided diffusion, PDG
(standard guidance, `w_cg`) and DCG (optimization guidance), all using the EXISTING pretrained unicycle
diffusion model (no retraining).

## Problem

* Model (read-only): `/u/rzhang26/CleanDiffuser/results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt`
  (sha256 `6e6eb4f2...dbda8`), loaded with `pipelines/unicycle_eval.py::_load_agent` (JannerUNet1d, model_dim 32,
  dim_mult 1 2 2, horizon 64, x0-prediction, cosine schedule, T=20, EMA weights).
  Trajectory `x` (64, 6) = [normalized obs (x, y, cos th, sin th), raw action (v, w)].
* `o_0` inpainting (fix_mask[0,:4] = 1, prior row 0 = normalized o_0) is the model's training convention and is
  applied identically for all methods. No other hard data consistency (measured positions are never overwritten).
* Normalizer: the TRAINING normalizer of full64_6k
  (`UnicycleDataset(load_unicycle_hdf5(.../unicycle_offline/full64_6k/unicycle_offline.hdf5), horizon=64).get_normalizer()`).
  Note: the paper's tracking runs used an older dataset's normalizer by default (`unicycle_eval` falls back to
  `results/unicycle_offline/unicycle_offline.hdf5` unless `--dataset-h5path` is given); that does not apply here.
* Ground truth: FRESH trajectories from the same data-generating process,
  `collect_unicycle_offline_data.collect_dataset(...)` with exactly the settings of `run_collect_unicycle_full64.sbatch`
  (require_full_horizon, horizon 64, dt 0.1, x/y in [-8, 8], v in [0, 2], w in [-1.5, 1.5]); only the seed differs:
  test seed 2027 -> 100 trajectories, validation seed 2028 -> 20 trajectories.
  Verified: (i) `collect_dataset` with seed 0 reproduces the first 50 training episodes bit-exactly;
  (ii) min distance of any val/test initial state to any of the 6000 training initial states is 0.115 (test) /
  0.152 (val) (no coincidences); min full-trajectory max-abs distance 0.68 / 0.89; val and test disjoint (min init dist 0.56).
* Observations (noise-free): `y = A x*_norm`, A selects (x, y) at 16 random timesteps out of {1..63} per trajectory
  (`rng = np.random.default_rng(12027 test / 12028 val)`; `sorted(rng.choice(np.arange(1, 64), 16, replace=False))`).
  `f(x) = 0.5 ||A x - y||^2` in normalized coordinates, `grad f = A^T (A x - y)`, L = 1; reward `R = -f` via
  `RuntimeRewardClassifier` (batched, per-sample masks). Checked: autograd gradient == A^T(Ax - y) exactly
  (max abs diff 0.0), R == -f exactly, f(x*) = 0.
* x_T: trajectory j uses `torch.manual_seed(base + j); torch.randn((1,64,6), device='cuda:0')`, base 20270000 (test) /
  20280000 (val); shared by all methods (passed as `initial_xt`).

## Methods (`scripts/inverse_common.py`)

| Method | Sampler call | NFE | grad f evals | net backward |
|---|---|---|---|---|
| Unguided | `agent20.sample`, ddim, 20 steps, `guidance_mode="standard"`, `w_cg=0`, classifier None | 20 | 0 | 0 |
| PDG | same, classifier = reward, `w_cg` tuned | 20 | 20 | 0 |
| DCG | `agent100.sample` (diffusion_steps 100, training_diffusion_steps 20), ddim, 20 sampling steps, `guidance_mode="optimization"`, `optimization_guidance_last_steps=10`, eta tuned | 20 | 10 | 0 |
| PnP-PGD | init = unguided DDIM-20 sample of the same x_T; `FixedNoiseDenoiser.from_agent(agent20, k, prior=prior, input_scaling="vp")`, `pnp_pgd(..., condition_fn=pin_fn(prior, fix_mask))`, `grad_f = reward_classifier_grad_f(clf, fix_mask)` | 20 + K | K | 0 |
| PnP-PGD ablation | init = unguided DDIM-10 (agent20, 10 sampling steps), K = 10, (k, eta) tuned separately on val | 10 + 10 | 10 | 0 |

## Protocol (pre-registered in `configs/inverse_protocol.json` before any val/test run)

* Selection criterion, same for every tuned method: minimize the validation mean (20 trajectories) of the
  per-trajectory unobserved-xy MSE in physical units (mean over the 47 timesteps t in {1..63} \ S and dims x, y).
  Tie-break: lower NFE, lower grads, grid order.
* Grids: PDG `w_cg` {0.3, 1, 3, 10, 30}; DCG eta {1, 3, 10, 30, 100}; PnP-PGD k {2, 5, 9} x eta {0.25, 0.5, 1.0}/L x
  K {5, 10, 20} (K nested: one K=20 run per (k, eta), iterates at K = 5, 10, 20 evaluated); ablation k x eta, K = 10.
* Boundary rule: a selected `w_cg` / eta / noise index on its grid boundary triggers ONE extension by the
  pre-registered point (PDG 0.1 / 100; DCG 0.3 / 300; PnP k 1 / 14, eta 0.125 / 1.5 — 1.5 rather than 2.0 to stay
  strictly below the 2/L step-stability limit), crossed with the full grid of the other hyper-parameters, then re-select;
  never extend again. K is a compute budget and is not extended.
* Outcome (`results/inverse/val_v1/val_grid.json`, `selected.json`):
  PDG `w_cg = 10` (interior, no extension); DCG eta = 10 (interior, no extension);
  PnP-PGD first selected (k=2, eta=1.0, K=20) — on the boundary in k (low) and eta (high) -> extended once with
  k = 1 and eta = 1.5 -> final (k=1, sigma_bar=0.095, eta=1.5/L, K=20), STILL on the boundary of k, eta and K
  (not extended further, per protocol). Ablation: (k=2, eta=1.0) -> extended (k=1, eta=1.5) -> (k=2, sigma_bar=0.179, eta=1.5/L).
* Test: all methods in ONE GPU job (NVIDIA GH200 120GB, job 3565076) with frozen hyper-parameters.
* Timing convention: batch size 1, every test trajectory timed separately (torch.cuda.synchronize before/after),
  3 warm-up calls per method, model loading and the CPU e_dyn rollouts excluded; PnP-PGD time includes the
  DDIM initialization (also reported separately). Secondary: one batch of all 100 trajectories / 100.
  Costs counted with `rebuttal_pnp.instrument.NetworkCounter` / `GradCounter` in a separate batched run; all equal
  the analytic counts above.
* Metrics (per trajectory, physical units): unobserved-xy MSE (primary), unobserved 4-dim obs MSE, action MSE
  (clipped actions, t = 0..63), observed-xy RMSE, f (normalized), e_dyn = `unicycle_eval._feasibility_gaps(planned_obs,
  _open_loop_rollout_obs(Unicycle-v0(terminate_on_oob=False), true o_0, clipped planned actions, 64))["mean_l2_future"]`.
  Mean ± std use ddof = 1 over trajectories.

## Reproduce

```bash
cd /u/rzhang26/CleanDiffuser_rebuttal_pnp
export PYTHONPATH=$PWD/pipelines:$PWD:$PWD/experiments/rebuttal_pnp_2027 D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1
PY=/u/rzhang26/venvs/CleanDiffuser/bin/python
E=experiments/rebuttal_pnp_2027
# 1. data (CPU, ~20 s) -> results/inverse/data_v1
$PY $E/scripts/inverse_make_data.py --run-id data_v1
# 2. smoke test (5 validation trajectories, untuned params) -> results/inverse/smoke_v1   [job 3564981]
RUN_ID=smoke_v1 sbatch $E/sbatch/inverse_smoke.sbatch
# 3. validation tuning + test in one job -> results/inverse/val_v1, results/inverse/test_v1   [job 3565076]
VAL_ID=val_v1 TEST_ID=test_v1 sbatch $E/sbatch/inverse_val_then_test.sbatch
#    (equivalently: RUN_ID=val_v1 sbatch $E/sbatch/inverse_val.sbatch; then
#     VAL_RUN=$PWD/$E/results/inverse/val_v1 RUN_ID=test_v1 sbatch $E/sbatch/inverse_test.sbatch)
# 4. optional validation-only long-K diagnostic -> results/inverse/val_longK_diag_v1   [job 3565101]
RUN_ID=val_longK_diag_v1 sbatch $E/sbatch/inverse_val_longK_diag.sbatch
# 5. tables + figures (CPU)
R=$E/results/inverse
$PY $E/scripts/inverse_report.py --test-run $R/test_v1 --val-run $R/val_v1 --diag-run $R/val_longK_diag_v1 --prefix test_v1_r3
```
Run ids/prefixes must be new (outputs never overwrite).

## Outputs

* `results/inverse/data_v1/`: `{test,val}_data.npz` (gt_obs (n,64,4) physical, gt_act (n,64,2), x_star_norm (n,64,6),
  obs_idx (n,16), M (n,64,6) entry mask = A^T A, Y = A^T y, next_obs); `data_config.json` (settings, checks, normalizer,
  sha256 of training hdf5, provenance).
* `results/inverse/val_v1/`: `config.json` (protocol, seeds, checkpoint/data/hdf5 sha256, normalizer, GPU, provenance),
  `checks.json`, `val_grid.json` (every grid point: params, per-trajectory and summary metrics, measured/analytic costs,
  PnP per-iteration mean curves, sigma_bar, extension record), `selected.json`, `val_raw.npz` (reconstructions,
  per-trajectory metrics, PnP per-iteration per-trajectory curves for every grid point).
* `results/inverse/test_v1/`: `config.json`, `checks.json`, `results.json` (summaries, costs, timings, batch-1 vs batched
  consistency), `raw.npz` (gt, masks, observations, x_T, seeds, per-method reconstructions batch-1 `<m>__x` and batched
  `<m>__x_batched`, per-trajectory metrics `<m>__<metric>`, per-trajectory timings, PnP init and per-iteration curves).
* `results/inverse/smoke_v1/`: same layout as test, on validation trajectories 0..4 with untuned parameters.
* `results/inverse/val_longK_diag_v1/`: DIAGNOSTIC ONLY (validation, not used for selection): PnP-PGD to K = 200.
* `results/logs/inverse_*_<jobid>.{out,err}`.
* Tables (`tables/inverse/`), FINAL = prefix `test_v1_r3`: `_table3.{md,csv,tex}`, `_table3_extra.md`,
  `_paired.md` (paired per-trajectory differences with bootstrap CIs), `_val_grids.{md,csv}`.
  Prefixes `test_v1`, `test_v1_r2` are earlier renderings of the SAME runs (identical numbers; only figure layout fixed);
  `smoke_v1_*` are from the smoke test.
* Figures (`figures/inverse/test_v1_r3/`): `recon_percentiles` (test trajectories at the 25/50/75th percentile of the
  unguided unobserved-xy MSE: #15, #88, #19), `recon_fixed_0-3`, `mse_vs_nfe`, `pnp_iterations`, `val_grids`,
  `val_longK_diagnostic`, `figure_manifest.json`.

## Known caveats

* `experiments/rebuttal_pnp_2027/scripts/` is matched by the repository's root `.gitignore` (`scripts/`): add with
  `git add -f`. sha256 of the scripts used: inverse_common 504d1df1, inverse_make_data 54e2e151, inverse_run 2770e230,
  inverse_val_longK_diag 546744f6, inverse_report cfc610e7.
* Batch-1 and batched runs differ by floating-point noise (cuDNN algorithm/TF32 defaults left unchanged, as in the
  existing pipelines): max |dx| up to 0.02 (unguided) - 0.16 (ablation) normalized units on single entries; method means
  agree to <= 1e-3 m^2. Validation was run batched (n=20); test numbers are from batch-1 runs.
* A `results/_debug_cpu/inverse/dbg_smoke1` directory comes from an aborted CPU debugging run; ignore it.
