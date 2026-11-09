# Unicycle trajectory tracking: fixed-noise PnP-PGD baseline (ICLR 2027 rebuttal)

PnP-PGD: `x^{k+1} = pin( D_{sigma_bar}( x^k - eta * grad f(x^k) ) )`, with `D` = the existing
unicycle diffusion denoiser (same checkpoint, EMA weights) evaluated at ONE fixed noise index of the
T=20 schedule (`FixedNoiseDenoiser.from_agent(agent_T20, k, prior=prior, input_scaling="vp")`), no
DDIM relaxation, never differentiated. `f = -R`, `R(x) = -5 * mean_{t=1..63} ||x~_t^{xy} - y~_t||^2`
(normalized coordinates), so `grad f = (10/63)(x~ - y~)` on xy dims, t>=1, and `L = 10/63`.

Everything lives in `/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/`.
No existing repository file was modified; the original repo `/u/rzhang26/CleanDiffuser` is read only
(checkpoint, normalizer dataset, published results).

## FINAL deliverables (status after the reviewer fixes, 2026-11-09)

| what | FINAL | superseded (kept, do not use) |
|---|---|---|
| validation run (selection) | `results/unicycle/val_3565048` (`selection.json`) | — |
| test run (all Table 2 numbers, costs, runtimes, candidates.npz) | `results/unicycle/test_3565048` | — |
| post-selection supplementary run (K up to 20 on test, timing diagnostic) | `results/unicycle/supp_3565377` (job 3565377, gh143) | — |
| tables | `tables/unicycle/job3565048_v3_*` | `job3565048_*` (v1), `job3565048_v2_*` (identical numbers to v3; v2 figures had ±std bands clipped on log axes) |
| figures | `figures/unicycle/job3565048_v3/*` plus the main PnP-PGD plan plots `figures/unicycle/job3565048/plan_plots/plan_<traj>_pnp_pgd_rankRR_candI.*` (unchanged, still final) | `figures/unicycle/job3565048/unicycle_*` (v1: wrong "(m)" unit on e_dyn, ablation (b) barely visible, "K sweep" mislabelled), `figures/unicycle/job3565048_v2/` |

Not deliverables: `results/unicycle/_smoke_*` (GPU smoke `_smoke_gpu_test_3565048`; CPU smokes `_smoke_cpu_val_0{1,2}`,
`_smoke_cpu_test_0{1,2}`, `_smoke_cpu_supp_0{1,2}`, `_smoke_analysis_cpu02`) and `results/unicycle/_smoke_dummy_selection.json`
(a dummy selection for the smoke step; it sits outside a run dir, which should not be repeated).

Reproduce the v3 outputs:
```bash
cd /u/rzhang26/CleanDiffuser_rebuttal_pnp
sbatch experiments/rebuttal_pnp_2027/sbatch/unicycle_pnp_supp.sbatch      # GPU, ~16 min -> results/unicycle/supp_<JOBID>
export PYTHONPATH=$PWD/pipelines:$PWD:$PWD/experiments/rebuttal_pnp_2027 D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable
R=experiments/rebuttal_pnp_2027/results/unicycle
/u/rzhang26/venvs/CleanDiffuser/bin/python experiments/rebuttal_pnp_2027/scripts/unicycle_pnp_analyze_v2.py \
    --val-dir $R/val_3565048 --test-dir $R/test_3565048 --supp-dir $R/supp_<JOBID> --tag <new_tag>   # CPU
```

v3 files:
* `job3565048_v3_table2.{md,csv,tex}`: Table 2 with **two** PnP-PGD rows: the spec's primary variant (init = Monte Carlo
  plan) and ablation (b) (init = Gaussian x_T), which was selected on validation with the same grid, criterion and
  boundary rule and is the best PnP variant on validation (0.2995 vs 0.6198). In the `.tex` the existing methods carry the
  **published** values (they remain the reference); the `.md/.csv` carry our re-run values with a per-row note on agreement.
  `job3565048_v3_table2_reproduced.tex` is the same table with our re-run values and a dagger footnote on DRGD.
  The `.md/.csv` also give the runtime split (init + iterations) and an alternative PnP runtime = Monte Carlo sampling time
  + PnP iteration time (see Timing below).
* `job3565048_v3_table_ablations.*`: the 4 PnP variants (main, (a), (b), (c)).
* `job3565048_v3_paired_comparison.{md,csv,json}`: per test reference, each PnP variant vs DCG/DRGD/PDG/MC paired over the
  100 candidates (same seed and x_T): medians, win rate, mean difference with a 95% paired bootstrap CI (10000 resamples).
* `job3565048_v3_validation_vs_K.*`: validation metrics vs K=0..20 at the selected (k, eta) (descriptive only).
* `job3565048_v3_test_vs_K_post_selection.*`: test metrics vs K=0..20 for each PnP variant with its frozen (k, eta),
  from supp_3565377 (post-selection reporting, NOT used for selection).
* `job3565048_v3_timing_diagnostic.*`: timing diagnostic (supp_3565377 part B).
* Figures `job3565048_v3/`: `unicycle_crollout_vs_edyn` (all variants; (b) drawn opaque), `unicycle_crollout_vs_edyn_main`
  (Table 2 rows only), `unicycle_crollout_vs_nfe` (budget plot; PnP curves = iterates K=0..20 of the frozen config, big marker
  = frozen K*, bands = interquartile range), `unicycle_pnp_per_iteration` (c_plan, c_rollout, e_dyn vs K; bands = IQR),
  `plan_plots/plan_<traj>_pnp_pgd_from_xT_rankRR_candI.*` (paper-style plan plots for ablation (b), top 4 by plan reward).
  e_dyn has no physical unit: it is the L2 over (x, y, cos θ, sin θ), which mixes metres with unitless components.

Corrections to earlier statements:
* **eta = 2/L is an exact reflection.** On the tracked coordinates f is quadratic with Hessian L·I, so
  x − (2/L)∇f(x) = 2y − x (|1 − ηL| = 1). The gradient step is not contractive; all contraction comes from the
  denoiser, and the iterates oscillate with period 2 (visible in c_plan and e_dyn over K; e.g. RT2 mean e_dyn over
  K=0..5: 0.213, 3.309, 1.385, 2.047, 1.628, 1.722).
* **Validation optimum in K.** The v1 claim that "the optimum may lie at a smaller K" was wrong. At k=2, eta=2/L the
  validation c_rollout over K=0..6 is 0.9935, 0.7298, 0.6813, 0.6430, 0.6283, 0.6198, 0.6195, then rises again
  (0.6562 at K=10, 0.8041 at K=20). The validation optimum is at K≈6, next to the selected K=5.
  For x_T init it is K=8 (0.2917), against the selected K=10 (0.2995).
* **Noise level.** The v1 claim that "the best settings use the smallest noise level" was wrong. The extension
  evaluated k=1 (sigma_bar 0.095), and it lost to k=2 (0.6625 vs 0.6198 at K=5). eta, however, is still
  improving at the boundary (0.692 at 1/L, 0.620 at 2/L).
* **Test vs validation K-dependence (post-selection, descriptive).** On test, the frozen main config keeps
  improving past K*=5 on Circle and RT2, but degrades on RT1:
  * Circle: K=20 gives 0.379 (vs 0.735 at K=5).
  * RT2: K=20 gives 1.780 (vs 3.393 at K=5).
  * RT1: best at K=3 (0.497); 1.114 at K=20.

  So the validation-chosen K is not test-optimal on 2 of the 3 references. Even the per-reference oracle K on test
  stays far behind DCG (0.222 / 0.074 / 0.185): main PnP 0.379 / 0.497 / 1.780; (b) 0.736 / 0.373 / 0.205;
  (c) 0.308 on Circle at K=20.
* **Timing.** In test_3565048 the PnP-PGD init was timed 5.2-9.4 ms slower than the identical Monte Carlo
  computation (MC 120.4 / 122.0 / 121.8 ms vs init 129.8 / 127.1 / 127.2 ms). In the diagnostic job
  (supp_3565377, a different node gh143 with noisier timings, std about 8 ms) the means were:
  * mc_only: 131.9-135.0 ms.
  * init_only: 133.9-135.3 ms.
  * init_plus_iter: 132.7-135.2 ms.
  * full test-loop protocol (with the untimed history pass and per-iteration rollouts interleaved): 135.2-136.2 ms.

  That gap is only 1-4 ms. Disabling Python GC (with gc.collect between candidates) made sampling slower
  (about 172 ms), so GC is not the explanation. The cause is unresolved; it is a loop-context effect, not a cost
  of PnP-PGD. Table 2 keeps the measured totals; the md/csv also give "MC sampling + PnP iterations" =
  155.7 / 156.9 / 156.6 ms.
* **K-curve sanity checks (supp_3565377).** For every variant and reference, iterates 0..K* of the K=20 pass are
  bit-identical to the test_3565048 history, and the metrics at K* equal the test candidates.npz (max abs diff 0.0).

## Files

| file | purpose |
|---|---|
| `scripts/unicycle_common.py` | shared code: loads the checkpoint into two `DiscreteDiffusionSDE` agents with `ue._load_agent` (T=20 for MC/PDG/DRGD/PnP-PGD; T=100 with training T=20 for DCG, as in the paper), builds per-reference context (`gym.make("Unicycle-v0", max_episode_steps=64, terminate_on_oob=False)`, reward fn, `RuntimeRewardClassifier`, prior), runs one candidate of each method with counters and timing, and evaluates plans with the same `ue` functions as `ue.run_plan_eval` |
| `scripts/unicycle_pnp_validate.py` | validation grid + pre-specified selection + one-time boundary extension, for the `mc` init (main PnP-PGD) and the `xT` init (ablation b) |
| `scripts/unicycle_pnp_test.py` | test phase: all methods in one GPU job, 100 candidates per (reference, method), costs, timing, per-iteration metrics |
| `scripts/unicycle_pnp_analyze.py` | CPU only: Table 2, ablation table, validation-grid tables, reproduction check against the published runs, figures |
| `sbatch/unicycle_pnp_full.sbatch` | one GPU job: (1) GPU smoke test, (2) validation, (3) test |

## Reproduce

```bash
cd /u/rzhang26/CleanDiffuser_rebuttal_pnp
# (one-time, already done) dummy selection used only by the smoke step of the sbatch job:
#   results/unicycle/_smoke_dummy_selection.json  (k=5, eta=0.5/L, K=5 for mc; k=9, eta=1/L, K=5 for xT)
sbatch experiments/rebuttal_pnp_2027/sbatch/unicycle_pnp_full.sbatch     # -> JOBID
# outputs: results/unicycle/_smoke_gpu_test_<JOBID>, results/unicycle/val_<JOBID>, results/unicycle/test_<JOBID>
export PYTHONPATH=$PWD/pipelines:$PWD:$PWD/experiments/rebuttal_pnp_2027
R=experiments/rebuttal_pnp_2027/results/unicycle
/u/rzhang26/venvs/CleanDiffuser/bin/python experiments/rebuttal_pnp_2027/scripts/unicycle_pnp_analyze.py \
    --val-dir $R/val_<JOBID> --test-dir $R/test_<JOBID> --tag job<JOBID>
```
Individual phases:
```bash
PY=/u/rzhang26/venvs/CleanDiffuser/bin/python; S=experiments/rebuttal_pnp_2027/scripts
$PY $S/unicycle_pnp_validate.py --run-id val_<id>                       # GPU
$PY $S/unicycle_pnp_test.py --run-id test_<id> --selection-json $R/val_<id>/selection.json   # GPU
```
Environment: `D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4`.
Note: sampling on CPU gives different candidates (CPU and CUDA random streams differ), so the published
numbers are only reproducible on CUDA.

## Protocol decisions

* **Inputs.** Checkpoint `CleanDiffuser/results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt`
  (sha256 recorded in each `config.json`). Normalizer: the OLD dataset
  `CleanDiffuser/results/unicycle_offline/unicycle_offline.hdf5`. This is what the paper runs used
  (`ue` default via `repo_dir=/u/rzhang26/CleanDiffuser`); it is used for ALL methods.
* **Per-candidate protocol (identical to the paper):** sequential, `n_samples=1`, `set_seed(seed)` immediately
  before each candidate, DDIM, uniform schedule, temperature 1.0, EMA, reward temperature 5.0, reference with
  64 steps. Monte Carlo = `standard_w_cg0` (20 NFE); PDG = `standard_w_cg30`; DCG = `optimization_scale_30`
  (diffusion_steps 100, training 20, sampling_steps 20, last 10 steps guided); DRGD = `drgd` with
  drgd_steps 20, gamma 15, noise index -1. Candidate i of every method uses the same CUDA seed, so the same x_T.
* **Metrics:** `c_plan = -plan_reward` (unclipped normalized plan), `c_rollout = -rollout_reward`
  (open-loop rollout of the clipped planned actions from o_0), `e_dyn = feasibility_mean_l2_future`.
  Mean ± std with ddof=1 over the candidates.
* **PnP-PGD (main):** x_init = the Monte Carlo DDIM-20 plan of seed i (the unclipped normalized `plan_norm`,
  bit-identical to the Monte Carlo baseline; checked per candidate as `init_vs_monte_carlo_maxabsdiff`), then K
  iterations with `pnp_pgd(...)`, `grad_f = reward_classifier_grad_f(classifier, agent.fix_mask)`,
  `condition_fn = pin_fn(prior, agent.fix_mask)`. The final plan goes through the same
  unnormalization, clipping, rollout and metrics. Cost = 20 NFE (init) + K NFE + K grad evals.
* **Validation (disjoint from test):** references `half_heart`, `random_smooth_2`, `random_smooth_4`; seeds
  1000..1019. Grid k ∈ {2, 5, 9} (sigma_bar = 0.179, 0.451, 0.934), eta ∈ {0.25, 0.5, 1.0}/L, K ∈ {5, 10, 20}.
  K values come from the iterates of one K=20 run; this is exact because the iteration is deterministic.
  Criterion (fixed in advance): minimum of the validation mean c_rollout averaged over the 3 references;
  ties (equal at 1e-9) go to the smaller e_dyn.
  Boundary rule: if the selected k or eta is on the grid boundary, extend ONCE by one point in that direction
  (k: 1 below / 13 above, sigma_bar 0.095 / 1.867; eta: 0.125/L below / 2.0/L above), evaluate the full cross
  product of the extended grid, and reselect. K is not extended (the spec rule covers k and eta only); whether
  the selected K is on its boundary is reported. Every grid point and every iteration 0..20 is saved.
* **Ablations** (test only, labelled): (a) `pnp_pgd_budget20`: DDIM-10 unguided init on T=20 + K=10 at the
  selected k, eta (20 NFE total, equal to DCG); (b) `pnp_pgd_from_xT`: PnP-PGD from the Gaussian x_T of seed i
  (the same draw DDIM starts from), with k/eta/K selected separately on validation using the same grid,
  criterion and boundary rule; (c) `pnp_pgd_noscale`: `input_scaling="none"` (DRGD's convention), with the
  selected k/eta/K of the main variant.
* **Costs:** `NetworkCounter` forward hooks on both agents' EMA networks (forward samples = NFE; backward
  passes through the net), plus `GradCounter` on the reward classifier (objective-gradient evaluations).
  Counters are attached in the timed region for every method. PnP-PGD init costs are stored separately
  (`init_*`) and included in the totals.
* **Runtime:** one GPU job for all methods. For each (reference, method): 3 warm-up candidates (seeds
  90000..90002, discarded), then per candidate `torch.cuda.synchronize()` before and after the sampling call.
  Excluded: model loading, env rollout, and metrics (rollout plus metric time is reported separately as
  `rollout_and_metric_seconds`). Batching: n_samples=1, sequential, as in the paper. For PnP-PGD, the
  per-iteration metrics come from a second, untimed pass with `record_history=True`; its final iterate is
  checked against the timed pass (`history_final_maxabsdiff`).

## Outputs

`results/unicycle/val_<JOBID>/`: `config.json`, `validation_grid_{mc,xT}.npz` (arrays `[grid point, ref, seed,
iteration 0..20]` of c_plan, c_rollout, e_dyn, plus the init metrics), `validation_{mc,xT}.json` (all rows,
selection on the initial and extended grid, extension applied, boundary flags, cost counts),
`selection.json` (frozen hyperparameters).

`results/unicycle/test_<JOBID>/<traj>/<method>/reference_plan/trajectories/`: `candidates.npz` (same keys as the
original `ue` cache), `candidates.meta.json` (ranked list), `plans_norm.npz` (unclipped normalized plans),
`summary.json`, `costs_timing.json` (per-candidate NFE, backward passes, grads, seconds, init/iteration split),
and for PnP-PGD variants `per_iteration.npz` (c_plan, c_rollout, e_dyn per candidate and iteration, plus the
iterates). Methods: `monte_carlo, pdg, dcg, drgd, pnp_pgd, pnp_pgd_budget20, pnp_pgd_from_xT, pnp_pgd_noscale`.

`tables/unicycle/<tag>_table2.{md,csv,tex}` (layout of `drgd_paper_table_gamma15.tex` plus NFE / Net bwd /
Grad evals / Runtime columns), `<tag>_table_ablations.*`, `<tag>_validation_grid_{mc,xT}.{md,csv}`,
`<tag>_reproduction_check.{md,csv}` plus `_details.json` (per-candidate max abs/rel differences against the
cached `CleanDiffuser/results/unicycle_eval/reference_plan_paper_v1/<traj>/<cfg>/reference_plan/trajectories/candidates.npz`,
and a 4-decimal comparison with the published table).

`figures/unicycle/<tag>/`: `plan_plots/plan_<traj>_pnp_pgd_rankRR_candI.{png,pdf}` (top 4 by plan reward;
a verbatim replica of `rp.plot_plan_vs_rollout(style=rp.PLOT_STYLE_EMPHASIS)` after `rp.apply_paper_plot_style()`,
saved through safe_io at 200 dpi), `plan_plots/reference_plan_legend.*`; `unicycle_crollout_vs_edyn[_main_only].*`
(mean ± std, faint per-candidate scatter, log y; the lower whisker is omitted where mean − std ≤ 0);
`unicycle_crollout_vs_nfe.*` (budget plot: baselines at their NFE, PnP-PGD curves over K including init NFE);
`unicycle_pnp_per_iteration.*` (c_plan, c_rollout, e_dyn vs iteration for the PnP variants, with final
DCG/DRGD/MC values as reference lines).
Colors: DCG #E66101, PDG #2166AC, DRGD #984EA3, PnP-PGD #E7298A (ablations in lighter or darker pink,
dashed or dotted), Monte Carlo #7F7F7F; each method also has its own marker shape.

Smoke-test directories (`results/unicycle/_smoke_*`) are not deliverables. The `_smoke_cpu_*` ones ran on the
login-node CPU, so their numbers are not comparable with the paper.

## Run record (deliverable run)

* Slurm job **3565048** (`sbatch/unicycle_pnp_full.sbatch`), node gh133 (ghx4 / GH200), COMPLETED in 16 min 11 s
  (2026-11-09 16:10-16:26). Its time limit was lowered from 04:00:00 to 01:00:00 with `scontrol update`
  while pending, to get backfilled in a congested queue. The sbatch file still says 04:00:00.
* Run dirs: `results/unicycle/_smoke_gpu_test_3565048` (smoke), `results/unicycle/val_3565048`,
  `results/unicycle/test_3565048`. Analysis tag `job3565048` (tables `tables/unicycle/job3565048_*`, figures
  `figures/unicycle/job3565048/`). Log: `results/logs/unicycle_pnp_full_3565048.{out,err}`.
* Selection (validation): main PnP-PGD: the initial-grid winner was k=2, eta=1.0/L, K=5. k and eta were both
  on the boundary, so the grid was extended once (k=1, eta=2.0/L). Final: **k=2 (sigma_bar=0.179),
  eta=2.0/L=12.6, K=5** (val c_rollout 0.6198). After the extension eta is still on the boundary, and K=5 is
  the smallest K in the grid; no further extension, per protocol. Ablation (b), x_T init: initial winner
  k=2, eta=1.0/L, K=10 (0.5991), extended. Final **k=2, eta=2.0/L, K=10** (val c_rollout 0.2995); eta is still
  on the boundary. Note that eta=2/L is the classical PGD step-size limit and, for this quadratic f, an exact reflection through the target (see "Corrections" above).

* Supplementary job **3565377** (`sbatch/unicycle_pnp_supp.sbatch`, gh143, COMPLETED in 15 min 24 s). Run dir
  `results/unicycle/supp_3565377`; log `results/logs/unicycle_pnp_supp_3565377.{out,err}`. Analysis
  `scripts/unicycle_pnp_analyze_v2.py --tag job3565048_v3` (FINAL); `job3565048_v2` was an earlier pass of the same script.
