# ICLR 2027 rebuttal: fixed-noise PnP-PGD baseline — consolidated summary

Branch `iclr-rebuttal-pnp-baselines` (worktree `/u/rzhang26/CleanDiffuser_rebuttal_pnp`).
Per-experiment details, all deviations, and superseded renderings are in
`README_synthetic.md`, `README_unicycle.md`, `README_inverse.md`.

PnP-PGD: `x_{k+1} = D_σ̄(x_k − η ∇f(x_k))`, one fixed noise level, no backprop through the
denoiser, no DDIM relaxation, primary variant uses VP input scaling (`net(α_k x, k)`,
σ̄ = σ_k/α_k). Step sizes are in units of 1/L with L the true Lipschitz constant of ∇f in
normalized coordinates. All hyperparameters (PnP-PGD, and PDG/DCG where tuned) were selected on
a separate validation set with a criterion fixed in advance; grid-boundary selections were
extended once. Existing methods use their original code and paper hyperparameters.

## Final artifacts

| | Tables | Figures | Raw data |
|---|---|---|---|
| T1 synthetic | `tables/synthetic/20261109_165722_full_v2__20261109_170536_v2_*` | `figures/synthetic/20261109_165722_full_v2__20261109_170536/` | `results/synthetic/20261109_165722_full_v2/` |
| T2 unicycle | `tables/unicycle/job3565048_v3_*` (+ `job3565048_validation_grid_*`, `job3565048_reproduction_check*`) | `figures/unicycle/job3565048_v3/`, plan plots in `figures/unicycle/job3565048/plan_plots/` | `results/unicycle/{val,test}_3565048/`, `supp_3565377/` |
| T3 inverse | `tables/inverse/test_v1_r3_*` | `figures/inverse/test_v1_r3/` | `results/inverse/{data,val,test,val_longK_diag}_v1/` |

## Reproduction of existing results

- Synthetic: PDG, DCG and GGDM are bitwise identical to the cached paper curves on both tests. DRGD differs by at most 5e-7.
- Unicycle: Monte Carlo, PDG and DCG reproduce the published Table 2 exactly. DRGD matches to 3–4 significant digits, because its backward pass is non-deterministic.

## Headline results (test sets)

**T1 synthetic** (100 paper inits). Lower ‖x−x*‖ is better. Times are ms per batch of 100.

| Method | A: f (f*=4.5) | A: ‖x−x*‖ | A: dist_K | B: f (f*=7.709) | B: ‖x−x*‖ | B: dist_K | NFE | ms |
|---|---|---|---|---|---|---|---|---|
| DCG | 5.236 | 0.176 | 5.4e-2 | 8.087 | 0.069 | 7.2e-3 | 50 | 137 |
| PDG | 6.439 | 0.538 | 0.236 | 2.785 | 1.146 | 1.14 | 50 | 135 |
| GGDM | 12.905 | 1.530 | 2.7e-2 | 12.915 | 0.901 | 2.8e-3 | 2550 | 7740 |
| DRGD | 6.658 | 0.459 | 5.5e-3 | 10.196 | 0.425 | 7.5e-4 | 100 (+50 bwd) | 326 |
| PnP-PGD (K=20) | 4.812 | 0.075 | 1.0e-2 | 7.685 | 0.027 | 6.7e-3 | 20 | 51 |

What this table shows:
- **Accuracy:** PnP-PGD gets closest to x* on both tests, at the lowest cost.
- **No diversity:** it collapses to one point regardless of the initialization (std ≤ 1e-4), so here it behaves as an optimizer rather than a sampler.
- **Feasibility:** in Test B it ends slightly outside K, which is why f < f*.
- **Tuning sensitivity:** results depend strongly on k (and on η at some k). See the validation grids.

**Reviewer-3 DCG diagnostic** (from the original inits, descriptive only):
- **Distance to the subspace:** DCG enters the 0.1-neighbourhood only in the last reverse steps (step 46 in A, step 42 in B).
- **Final vs best iterate:** the final sample is within 1e-4 of the best ‖G‖ iterate for 69% of paths in A and 98% in B. In A, 29% of paths pass through a better iterate at steps 44–48. The mean gap is 0.014 and the maximum is 0.25.
- **Stationarity:** ‖G‖ stays at 0.70 in A and 0.27 in B, so DCG is not stationary after 50 steps.

**T2 unicycle** (100 candidates; c_rollout / e_dyn):

| Method | Circle | RT1 | RT2 | NFE | ms/sample |
|---|---|---|---|---|---|
| Monte Carlo | 1.488 / 0.222 | 0.572 / 0.213 | 6.365 / 0.213 | 20 | 121 |
| PDG | 1.169 / 1.645 | 0.358 / 0.832 | 3.883 / 2.868 | 20 | 140 |
| DCG | **0.222** / 0.503 | **0.074** / 0.317 | **0.185** / 0.587 | 20 | 134 |
| DRGD γ=15 | 0.179 / 0.706 | 0.126 / 0.421 | 1.290 / 0.899 | 40 (+20 bwd) | 366 |
| PnP-PGD (MC init, spec primary) | 0.735 / 0.705 | 0.518 / 0.778 | 3.393 / 1.722 | 25 (20 init) | 162–165 |
| PnP-PGD (x_T init, ablation b) | 0.906 / 0.826 | 0.383 / 0.643 | 0.221 / 0.423 | 10 | 70 |

What this table shows:
- **Spec primary variant (MC init):** it is weak. It improves on Monte Carlo everywhere but loses to DCG on 98–100% of paired candidates.
- **Ablation (b), x_T init:** it was best on validation.
  - On RT2 it is statistically tied with DCG: paired mean diff +0.036, 95% CI [−0.011, 0.098]. It does this at half the cost and with lower e_dyn.
  - On Circle and RT1 it is clearly worse than DCG.
- **Selected step size:** η = 2/L, which is still on the grid boundary after the one allowed extension. That step is a pure reflection on the tracked coordinates, so the iterates oscillate with period 2.

**T3 sparse inverse** (100 held-out trajectories, 25% of positions observed, noise-free):

| Method | Unobs. xy MSE [m²] | Obs. xy RMSE [m] | e_dyn | NFE | ms/traj |
|---|---|---|---|---|---|
| Unguided | 5.431 | 2.084 | 0.263 | 20 | 128.5 |
| PDG (w=10) | 3.183 | 0.729 | 0.851 | 20 | 148.9 |
| DCG (η=10) | **1.068** | 0.785 | 0.865 | 20 | 145.5 |
| PnP-PGD (k=1, η=1.5/L, K=20) | 1.550 | 1.059 | 0.432 | 40 (20 init) | 288.2 |
| PnP-PGD, budget-matched (DDIM-10 + K=10) | 1.860 | 1.180 | 0.443 | 20 | 143.0 |

What this table shows:
- **DCG vs PnP-PGD:** DCG is significantly better (paired diff +0.48, 95% CI [0.13, 0.85]).
- **PnP-PGD vs PDG:** PnP-PGD beats PDG on 86 of 100 trajectories and has about half the e_dyn of PDG and DCG.
- **Boundary and budget:** the PnP-PGD selection sits on the k, η and K boundaries. A validation-only diagnostic reaches DCG's validation error only after 50–200 iterations (70–220 NFE).

## Table 4: runtime and cost (per sample, init cost included)

All times are on a GH200, synchronized, with warm-up. Compare times only within a column, because each column comes from one job.

| Method | Synthetic A (ms per batch of 100) | Unicycle (ms per sample, n=1) | Inverse (ms per traj, n=1) | NFE syn/uni/inv | net bwd | ∇f |
|---|---|---|---|---|---|---|
| Unguided / MC | 108.9 | 120–122 | 128.5 | 50 / 20 / 20 | 0 | 0 |
| PDG | 135.8 | 140 | 148.9 | 50 / 20 / 20 | 0 | 50 / 20 / 20 |
| DCG | 137.8 | 134–135 | 145.5 | 50 / 20 / 20 | 0 | 50 / 10 / 10 |
| GGDM | 7752 | – | – | 2550 / – / – | 0 | 2500.5† / – / – |
| DRGD | 327.3 | 365–367 | – | 100 / 40 / – | 50 / 20 / – | 50 / 20 / – |
| PnP-PGD (selected) | 51.3 (K=20) | 162–165 (MC init + K=5) | 288.2 (DDIM-20 init + K=20) | 20 / 25 / 40 | 0 | 20 / 5 / 20 |
| PnP-PGD (alternative init) | 164.5 (DDIM-50 init + K=20) | 70 (x_T, K=10) | 143.0 (DDIM-10 + K=10) | 70 / 10 / 20 | 0 | 20 / 10 / 10 |

† GGDM gradient count as executed. Only 50 shared anchor gradients are algorithmically required.

Measured NFE, backward and gradient counts (forward hooks and gradient counters) equal the analytic counts everywhere.

## Known limitations and open items

- **Boundary selections.** Several PnP-PGD selections still sit on a grid boundary after the single permitted extension:
  - unicycle η = 2/L;
  - inverse k, η and K;
  - synthetic A no-VP ablation, η.

  Each is documented in the per-experiment README.
- **Unused smoke and debug runs.** These remain in `results/` and are labelled as non-deliverables. Nothing was deleted.
- **`scripts/` is gitignored.** The repo-root `.gitignore` excludes `scripts/` and `*.csv`, so these were committed with `git add -f`.
- **Inverse LaTeX table.** The column spec has one extra `c`. It still compiles, but drop the spare column before pasting into the paper.
- **DRGD at the lowest noise level.** Slurm job 3564987 runs schedule index 0 (σ = 0.0064) with γ ∈ {5, 10, 15, 20}. Its results go to `results/unicycle_drgd_lownoise/drgd_noise0_3564987/`. It is a supplementary run, not part of Tables 1–4.

## Preservation check (original repo `/u/rzhang26/CleanDiffuser`)

The following match before and after:
- all 6848 sha256 hashes of files under `results/ data/ outputs/ configs/`;
- the sha256 hashes of all modified and untracked code;
- `HEAD` (8043da1);
- `git status --porcelain`;
- `git diff`.

The checkpoint hashes recorded in every run config equal the inventory:
- full64_6k: `6e6eb4f2…`
- synthetic test_a: `38182620…`
- synthetic test_b: `ea2c1f27…`

Unit tests: 17 passed (`tests/test_pnp_pgd.py`, `experiments/rebuttal_pnp_2027/tests/`).
