# Synthetic constrained optimization: fixed-noise PnP-PGD baseline (ICLR 2027 rebuttal)

This experiment adds the PnP-PGD baseline `x^{k+1} = D_sigma_bar(x^k - eta * grad f(x^k))` to the synthetic subspace
Tests A and B:

* Test A: a box in a 2-D subspace of R^10, f* = 4.5.
* Test B: a 5-D ellipsoid in a 5-D subspace, f* = 7.7089.

It also re-runs the four existing methods with the paper hyperparameters from the cached paper initializations, with
every run instrumented:

* DCG ("optimization")
* PDG ("standard"; labelled PCG in the old plots)
* GGDM (Guo et al., `ggdm_g_loss_mean`)
* DRGD (Landing with the score)

The original repo `/u/rzhang26/CleanDiffuser` is read-only input. We use its checkpoints, normalizers, geometry, the
paper trajectory caches and `baseline_tune_best.json`, and `config.json` records the sha256 of each. All outputs go
through `rebuttal_pnp/safe_io.py`, which never overwrites a file.

## FINAL outputs (v2)

| what | path |
|---|---|
| compute run | `results/synthetic/20261109_165722_full_v2` (SLURM job 3565447, node gh147, GH200, 7 min 17 s) |
| figures | `figures/synthetic/20261109_165722_full_v2__20261109_170536/` |
| tables | `tables/synthetic/20261109_165722_full_v2__20261109_170536_v2_{table1.md,table1.csv,table1.tex,pnp_validation_grid.md,pnp_validation_grid.csv,reproduction.md,dcg_diagnostic.md}` |
| code | `scripts/synthetic_run_v2.py`, `scripts/synthetic_report_v2.py`, `configs/synthetic_v2.json`, `sbatch/synthetic_full_v2.sbatch` |

**Superseded (do not use):**
* `results/synthetic/20261109_162815_full` (v1, job 3565313) and its report
  `figures/synthetic/20261109_162815_full__20261109_163704/` + `tables/synthetic/20261109_162815_full__20261109_163704_*`.
  All npz data in v1 are bitwise identical to v2 (max |Δ| = 0 for every method and test). v1 is superseded for two
  reasons: its boundary-extension rule left the Test A no-VP-scaling ablation on a never-extended k boundary, and its
  report had the problems listed under "Changes v1 -> v2". The v1 code was `scripts/synthetic_run.py` (sha256
  `dca6674c…0041f`), `scripts/synthetic_report.py` (`16bc1781…ab77345`) and `configs/synthetic_v1.json` (`9685af1f…c9b85f`).
  Those file mtimes predate the v1 run.
* `figures/synthetic/20261109_165722_full_v2__20261109_170454/` and `tables/synthetic/20261109_165722_full_v2__20261109_170454_v2_*`:
  the first v2 report. It is identical to the final one except for the table-header wording ("identity" ->
  "numerically near-identity" last DDIM step, plus a timing note).
* Anything named `*quick_smoke*` or `*quick_cpusmoke*`: smoke tests (4 paths).

## Reproduce

```bash
cd /u/rzhang26/CleanDiffuser_rebuttal_pnp
# GPU compute (~7-9 min on one GH200): writes results/synthetic/<timestamp>_full_v2/
sbatch experiments/rebuttal_pnp_2027/sbatch/synthetic_full_v2.sbatch
# CPU: figures + tables from the run dir (new timestamped figure dir + *_v2 table files)
source experiments/rebuttal_pnp_2027/sbatch/env_synthetic.sh
$PY experiments/rebuttal_pnp_2027/scripts/synthetic_report_v2.py \
    experiments/rebuttal_pnp_2027/results/synthetic/20261109_165722_full_v2
# unit tests for the exact projection / metrics (2 passed)
$PY -m pytest -q experiments/rebuttal_pnp_2027/tests/test_synthetic_geometry.py
```

**Code provenance.** The repo `.gitignore` rules `scripts/` and `*.csv` match `experiments/rebuttal_pnp_2027/scripts/`
and the CSV tables, so the git commit alone does not identify the code. Use `git add -f` when committing.

* The v2 run's `config.json` contains `code_sha256` for `synthetic_run_v2.py`, `synthetic_geometry.py`, `safe_io.py`,
  `instrument.py`, `pnp_pgd.py`, `runtime_reward.py` and both synthetic pipelines.
* The report's `report_meta.json` holds the report script's sha256.

Final hashes:

* `synthetic_run_v2.py` 9c1a88ec8f1c00e7f26823095bfd52080202b6ed15ae8364c0276d9a6a4c12b1
* `synthetic_report_v2.py` 17fc918b2b366872acfc50e17093bd77a2e02deb2936884fd409038d64ca1179
* `configs/synthetic_v2.json` b8da9ca6a012446dce07ae62b434970467bd1ea7bf7cfdab2010bcda0f5c827e
* `pnp_pgd.py` c9a601e0…085ce
* `safe_io.py` b0744b73…f5756
* `instrument.py` 27e27350…ced32
* `synthetic_geometry.py` c983e896…5bad

## Files

| file | content |
|---|---|
| `configs/synthetic_v2.json` | all hyperparameters: paper settings, PnP grid, validation seeds, extension rule (v2), timing |
| `scripts/synthetic_run_v2.py` | GPU compute: re-runs, PnP grids + selection, test evaluation, counts, timing |
| `scripts/synthetic_report_v2.py` | figures (i)–(iv); Table 1; validation-grid, reproduction and DCG-diagnostic tables |
| `rebuttal_pnp/synthetic_geometry.py` | exact Euclidean P_K (box clip / ellipsoid KKT bisection), x* = P_K(c), all metrics |
| `tests/test_synthetic_geometry.py` | KKT/optimality of the projection; x* matches the existing SLSQP solver; metric equivalences |
| run dir `config.json` | full config, argv, sha256 of every input and of the code, DRGD tuned json, `provenance()`, GPU/torch |
| `test_x/{standard,optimization,ggdm,drgd,unguided,pnp,pnp_ddim_init,pnp_noscale}.npz` | `hist_phys`/`hist_norm` (n, T, 10) ambient history; `m_<metric>` (n, T) for objective, suboptimality, violation_existing, dist_K, dist_subspace, dist_xstar, G_norm; `cum_nfe`, `cum_grad`, `cum_bwd` (T,) per path. `ggdm.npz` also has `round_finals_phys` (n, 51, 10) (index 0 = unguided landing), `round_m_*`, `round_cum_{nfe,grad,bwd}`. PnP files have `selected_K`. |
| `test_x/val_grid_<variant>.npz` | every validation grid point, extension points included: `grid_k`, `grid_eta_over_L`, `grid_sigma_bar`, `m_<metric>` (G, 20, 51) |
| `test_x/val_grid_rows.json` | per (variant, k, eta, K ∈ {5,10,20,50}): validation means/stds, selected flag, extension flag |
| `test_x/summary.json` | reproduction deviations, cost cross-checks (expected vs measured), selections with `extension_rounds`, final metrics, timing (all repeats) |
| `test_x/inits.npz` | the 100 test inits (= cached paper `initial_xts`) and the 20 validation inits |
| tables `*_v2_table1.{md,csv,tex}` | Table 1 (the .tex has a GGDM gradient-count footnote) |
| tables `*_v2_pnp_validation_grid.{md,csv}` | full validation grids: all variants, extension points, K=50 info rows |
| tables `*_v2_reproduction.md`, `*_v2_dcg_diagnostic.md` | reproduction check; Reviewer-3 DCG numbers |
| figure `test_{a,b}_pnp_rebuttal_s50_convergence` | (i) replica of the paper convergence figure + PnP-PGD (selected K marked) |
| figure `test_{a,b}_pnp_rebuttal_budget_nfe` | (ii) objective, paper violation metric and exact dist_K vs cumulative NFE (log x): all methods + ablations; GGDM per-round finals |
| figure `test_{a,b}_pnp_rebuttal_objective_vs_feasibility` | (iii) final objective vs final exact dist_K (mean ± std); validation grid points faint |
| figure `test_{a,b}_dcg_reviewer3_diagnostic` | (iv) DCG: distance to subspace and exact dist_K, objective, ‖G_η‖ across reverse steps |

## Protocol decisions

### Existing methods

* We call the existing `agent.sample` with exactly the kwargs of `vis.sample_paths_with_history`.
* PDG/DCG/DRGD run per path (n_samples = 1), as in the paper run.
* GGDM runs as one batch of 100 coupled particles with `ggdm_round_seed = 0`. `set_seed(0)` is called first, as in
  the paper. GGDM is run round by round through the public `sample(..., ggdm_rounds=0, ggdm_query_xt=...)` path, so
  that each round's final iterate is recorded. This replica is bitwise identical to `_sample_ggdm_algorithm1` (checked
  in-run).
* All four methods reproduce the cached paper curves: max |Δ reward| = 0 for PDG/DCG/GGDM and ≤ 5.3e-7 for DRGD.
  Figures and tables use these re-run curves.

### Metrics (physical coordinates)

* f = ‖x − c‖², with f* = ‖P_K(c) − c‖² from the exact projection.
* Existing violation: the paper metric, which uses radial scaling for Test B.
* Exact dist_K, dist to the subspace ‖x_⊥‖, and ‖x − x*‖.
* ‖G_η(x)‖ = ‖x − P_K(x − η∇f)‖/η, with η = 1/(2 L_phys) = 0.25.
* mean ± std uses `np.std` (ddof = 0).

### PnP-PGD primary variant

* Denoiser: `FixedNoiseDenoiser.from_agent(agent, k, input_scaling="vp")` (EMA weights, no_grad, fixed k).
* Gradient: `grad_f = reward_classifier_grad_f(RuntimeRewardClassifier(reward_fn), agent.fix_mask)`. The fix_mask is all
  zeros.
* Initial iterate: the cached paper `initial_xts`.
* L = 2·max std²: 0.429796 for A, 0.147240 for B.
* Grid: k ∈ {5, 13, 24} (σ̄ 0.17/0.45/0.98), η ∈ {0.25, 0.5, 1}/L, K ∈ {5, 10, 20}.
* Validation: 20 inits from CUDA seeds 10000..10019. Criterion, fixed in advance: minimize the validation mean of
  ‖x_K − x*‖. Ties are broken by grid order with strict `<`.
* Each (k, η) is run for 50 iterations and K is read off that prefix. The method is deterministic, so this is identical
  to separate runs.

### Boundary extension (v2 rule)

* Each direction (k lower/upper, η lower/upper) may be extended at most once, on validation only. The extension points
  are k = 2 (σ̄ ≈ 0.077) below, k = 35 above, and η/2 or 2η.
* After an extension we re-select. If the new selection sits on the boundary in a direction that has not been extended
  yet, that direction is extended once too. K is never extended.
* The full trace is in `summary.json → pnp_selection.<variant>.extension_rounds`.

Extensions actually performed:

* **Test A primary and DDIM-init:** none, because (13, 0.5/L) is interior.
* **Test A no-VP-scaling:** η lower → 0.125/L, then k lower → 2. The second extension is new in v2. k = 2 is much worse
  (best value 0.32), so the selection is unchanged at (5, 0.125/L, 20). It still lies on the extended η boundary, which
  was not extended again.
* **Test B primary and DDIM-init:** k lower → 2 and η lower → 0.125/L. Selection unchanged at (5, 0.25/L, 20), now
  interior.
* **Test B no-VP-scaling:** η upper → 2/L. Selection unchanged at (13, 1/L, 5).

The frozen setting is evaluated on the 100 test inits for 50 iterations. The table reports both K_sel and K = 50.

### Ablations

Both use the same grid, criterion and extension rule.

* (a) Initialization from the unguided 50-step DDIM sample of the same noise. Its 50 NFE are counted.
* (b) `input_scaling="none"` (DRGD's convention). Here σ̄ is reported as σ_k.

### Costs

* Measured with `NetworkCounter`/`GradCounter` and cross-checked against the analytic counts. Every check matches.
* NFE counts every network evaluation actually executed, including the last DDIM step (schedule 0→0). That step is
  numerically a near-identity: for DCG max |Δx| = 2.5e-6 (A) and 8.5e-6 (B) in physical units.
* GGDM, as executed: 2550 NFE and 2500.5 gradient evaluations per path. Only 50 shared anchor gradients are
  algorithmically needed (see the table footnote).
* DRGD: 2 NFE + 1 VJP per step. Its `drgd_noise_index` indexes the sampling schedule: 45 → agent index 44 (A),
  40 → 39 (B).

### Runtime

* All timings come from one GPU job (3565447, GH200), with model loading excluded.
* 2 warm-up calls, then 5 timed repeats with `torch.cuda.synchronize`; we report the median.
* No history recording or hooks are active while timing.
* Batching convention: one call on the batch of 100 test paths, and one call on batch = 1. The net is launch-bound, so
  the two are almost equal.
* Absolute times vary by about 15–20 % between nodes: v1 on a different node gave 164 ms for DCG, v2 gives 138 ms.
  Compare methods only within one run.

### Figure (i)

* The paper style of `vis.plot_convergence` (11.5 in wide panels, mean lw 2.6, ±std band α 0.18, violation floor 1e-4,
  `_configure_violation_axis`, framealpha 0.55, `_apply_paper_style`), with PDG labelled PDG.
* v2 deviations, to avoid hiding data:
  * The legend sits below the panels (fig-level, 4 columns), not inside the left panel. In v1, inside the panel, it hid
    the PnP-PGD curve in Test B and faded the PDG/DCG/GGDM curves in Test A.
  * Each panel has its own x label at font size 13, because the long label does not fit at the paper size.
  * Figure height is 5.6 in instead of 4.8 to make room for the legend.
* The PnP curve is drawn on top (zorder).

### Figure (ii)

Three panels: objective, the paper violation metric, and exact dist_K. They differ only in Test B, where the paper
metric is radial scaling.

### Figure (iv) and the DCG diagnostic

* Neighbourhood thresholds on the distance to the subspace are 0.1 (annotated) and 0.05, in physical units; the box
  half-width is 1.
* Best iterate: the minimum of ‖G_η‖ along each path. Because the last DDIM step only changes ‖G‖ at the 1e-6 to 6e-5
  level, the "best = final" fraction uses a tolerance of 1e-4 on ‖G‖, with ties broken toward the latest step.
* The strict first-argmin statistic (v1) is also reported, but it is a float-noise tie-break between steps 49 and 50.
* The figure shows paths 0–4 plus, dotted, the path with the largest final-vs-best gap.
* This is an empirical description only. No convergence or last-iterate claim is made.

## Changes v1 -> v2 (in response to the internal review)

1. DCG diagnostic: tolerance-based best iterate. Test A "best = final" goes from 0 % (strict, an artefact) to 69 %
   (tol 1e-4). Test B: 98 % under both definitions. The step histograms and the last-step change magnitudes are now in
   the table.
2. Figure (i): the legend no longer hides data.
3. Boundary-extension rule applied per direction (Test A no-VP-scaling: k = 2 now evaluated; selection unchanged).
4. Code sha256 values recorded (run: `config.json → code_sha256`; report: `report_meta.json`).
5. Ablation colours are now lighter tints of #E7298A with non-solid lines: DDIM-init #EE7DBA dashed, no-VP-scaling
   #F5B0D5 dash-dot. v1 used a darker #C9507F for no-VP-scaling.
6. Figure (ii) gained the paper-violation panel.
7. The LaTeX table has the GGDM gradient-count footnote.
8. The figure (iv) annotation was moved off the data, and the worst path was added.
9. Text: DRGD is the most feasible *guided* method; unguided DDIM is more feasible in Test A. The sensitivity statement
   was made precise (see the results summary).

## Results summary (final run 20261109_165722_full_v2; see the tables for all columns)

Each entry gives the test mean over 100 paths of final f, ‖x − x*‖ and exact dist_K, then the NFE per path.

| method | Test A | Test B |
|---|---|---|
| DCG | 5.236, 0.176, 5.4e-2, 50 NFE | 8.087, 0.069, 7.2e-3, 50 NFE |
| PDG | 6.439, 0.538, 0.236, 50 NFE | 2.785, 1.146, 1.14, 50 NFE |
| GGDM | 12.905, 1.530, 2.7e-2, 2550 NFE | 12.915, 0.901, 2.8e-3, 2550 NFE |
| DRGD | 6.658, 0.459, 5.5e-3, 100 NFE + 50 VJP | 10.196, 0.425, 7.5e-4, 100 NFE + 50 VJP |
| PnP-PGD | 4.812, 0.0745, 1.04e-2, 20 NFE (k = 13, η = 0.5/L, K = 20) | 7.685, 0.0272, 6.7e-3, 20 NFE (k = 5, η = 0.25/L, K = 20) |
| unguided DDIM | 13.645, 1.599, 4.7e-3, 50 NFE | 13.382, 0.976, 2.7e-3, 50 NFE |

* **Infeasibility in Test B:** PnP-PGD's f < f* (f − f* = −0.024) because the iterate is slightly infeasible.
* **No sample diversity:** the PnP-PGD fixed point does not depend on the initialization (std ≤ 1e-4). The DDIM-init
  ablation reaches the same point for 50 extra NFE.
* **Feasibility of the existing methods:** DRGD is the most feasible *guided* method. Unguided DDIM is more feasible
  than DRGD in Test A (4.7e-3 vs 5.5e-3).
* **Sensitivity (validation ‖x_K − x*‖ at K = 20):** PnP-PGD is strongly sensitive to the noise level k and, at some
  k, to η. At the Test A selection (13, 0.5/L) the η neighbours are within 10–16 % (0.082 at 0.25/L, 0.086 at 1/L),
  while changing k gives 1.36 (k = 5) or 0.23 (k = 24). In Test B, doubling η at k = 5 raises 0.027 to 0.76, while
  halving it gives 0.048. Several grid points stall at 0.5–1.4.
