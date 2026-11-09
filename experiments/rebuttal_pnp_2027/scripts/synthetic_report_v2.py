#!/usr/bin/env python3
"""Figures + tables for the synthetic PnP-PGD rebuttal experiment from a ``synthetic_run.py`` run dir.

CPU only. Writes (never overwriting) to
  figures/synthetic/<run_id>__<report_stamp>/...   (png + pdf)
  tables/synthetic/<run_id>__<report_stamp>_*.{md,csv,tex}
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REB))

import guidance_synthetic_visualize_subspace as vis  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402

COL = {"optimization": "#E66101", "standard": "#2166AC", "ggdm": "#4DAF4A", "drgd": "#984EA3",
       "pnp": "#E7298A", "pnp_ddim_init": "#EE7DBA", "pnp_noscale": "#F5B0D5", "unguided": "#7F7F7F"}
# v2: both ablations are LIGHTER tints of #E7298A with non-solid line styles (v1 used a darker #C9507F dotted)
LAB = {"optimization": "DCG", "standard": "PDG", "ggdm": "GGDM", "drgd": "DRGD", "pnp": "PnP-PGD",
       "pnp_ddim_init": "PnP-PGD (DDIM init)", "pnp_noscale": "PnP-PGD (no VP scaling)",
       "unguided": "Unguided DDIM"}
LS = {"pnp_ddim_init": "--", "pnp_noscale": "-."}
BEST_TOL = 1e-4  # tolerance on ||G|| for "final iterate is (within float noise of) the best iterate"; > max|G50-G49|
EXISTING = ("standard", "optimization", "ggdm", "drgd")  # paper order (vis.CONVERGENCE_METHODS)
PNP = ("pnp", "pnp_ddim_init", "pnp_noscale")
TEST_TITLE = {"test_a": "Test A (box in 2-D subspace)", "test_b": "Test B (ellipsoid in 5-D subspace)"}
VIOL_FLOOR = 1e-4  # as in vis.plot_convergence


def load(run_dir: Path, test: str):
    t = run_dir / test
    with open(t / "summary.json") as f:
        summ = json.load(f)
    data = {}
    for m in EXISTING + PNP + ("unguided",):
        p = t / f"{m}.npz"
        if p.is_file():
            z = np.load(p)
            data[m] = {k: z[k] for k in z.files}
    vg = {}
    for v in PNP:
        p = t / f"val_grid_{v}.npz"
        if p.is_file():
            z = np.load(p)
            vg[v] = {k: z[k] for k in z.files}
    with open(t / "val_grid_rows.json") as f:
        rows = json.load(f)
    return summ, data, vg, rows


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pm(m, s, fmt="{:.4f}"):
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def sci(x):
    return f"{x:.2e}"


# ------------------------------------------------------------------ (i) convergence replica
def fig_convergence(summ, data, out_stem):
    vis._apply_paper_style()
    opt_objective = summ["f_star_exact"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True)
    panels = [("m_objective", "Objective", axes[0]), ("m_violation_existing", "Constraint violation", axes[1])]
    Ksel = int(summ["pnp_selection"]["pnp"]["selection"][2])
    for key, subtitle, ax in panels:
        for mode in EXISTING + ("pnp",):
            curves = data[mode][key]
            steps = np.arange(curves.shape[1])
            mean, std = curves.mean(0), curves.std(0)
            if key == "m_violation_existing":
                mean = np.maximum(mean, VIOL_FLOOR)
                lo, hi = np.maximum(mean - std, VIOL_FLOOR), mean + std
            else:
                lo, hi = mean - std, mean + std
            ax.plot(steps, mean, color=COL[mode], linewidth=2.6, label=LAB[mode], zorder=4 if mode == "pnp" else 2)
            ax.fill_between(steps, lo, hi, color=COL[mode], alpha=0.18)
            if mode == "pnp":
                ax.plot([Ksel], [mean[Ksel]], marker="o", markersize=9, color=COL[mode], markeredgecolor="black",
                        markeredgewidth=1.0, linestyle="none", zorder=6, label=f"PnP-PGD selected K={Ksel}")
        if key == "m_objective":
            ax.axhline(opt_objective, color="black", linestyle="--", linewidth=2.0, label="Optimal (feasible)", zorder=3)
        else:
            vis._configure_violation_axis(ax)
        ax.set_title(subtitle)
        vis._style_axis_grid(ax, minor=True)
    # v2: per-panel x label (slightly smaller font so the long label fits), and the legend is drawn below the
    # panels: inside the left panel (paper position, loc='best') it hid the PnP-PGD curve in Test B.
    for ax in axes:
        ax.set_xlabel("Step (reverse diffusion / DRGD refine / PnP iteration)", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=True,
               framealpha=vis.LEGEND_FRAME_ALPHA, edgecolor="#CCCCCC", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    safe_io.save_figure(fig, out_stem)
    plt.close(fig)


# ------------------------------------------------------------------ (ii) budget plots
def budget_curves(data, mode, which="dist_K"):
    d = data[mode]
    if mode == "ggdm":
        return d["round_cum_nfe"], d["round_m_objective"], d[f"round_m_{which}"]
    return d["cum_nfe"], d["m_objective"], d[f"m_{which}"]


def fig_budget(summ, data, out_stem, title):
    vis._apply_paper_style()
    modes = ("unguided",) + EXISTING + PNP
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), sharex=True)
    for mode in modes:
        nfe, obj, dk = budget_curves(data, mode, "dist_K")
        _, _, ve = budget_curves(data, mode, "violation_existing")
        msk = nfe >= 1
        for ax, y, logy in ((axes[0], obj, False), (axes[1], ve, True), (axes[2], dk, True)):
            mean, std = y.mean(0)[msk], y.std(0)[msk]
            lw = 2.2 if mode in EXISTING + ("pnp",) else 2.0
            ax.plot(nfe[msk], mean, color=COL[mode], lw=lw, ls=LS.get(mode, "-"), label=LAB[mode],
                    marker="." if mode == "ggdm" else None, ms=4, zorder=4 if mode == "pnp" else 2)
            lo = np.maximum(mean - std, 1e-6) if logy else mean - std
            ax.fill_between(nfe[msk], lo, mean + std, color=COL[mode], alpha=0.12)
        if mode == "pnp":
            K = int(summ["pnp_selection"]["pnp"]["selection"][2])
            for ax, y in ((axes[0], obj), (axes[1], ve), (axes[2], dk)):
                ax.plot([nfe[K]], [y.mean(0)[K]], marker="o", ms=8, color=COL[mode], mec="black", ls="none", zorder=6,
                        label=f"PnP-PGD selected K={K}" if ax is axes[0] else None)
    axes[0].axhline(summ["f_star_exact"], color="black", ls="--", lw=1.8, label="Optimal (feasible)")
    for ax, which in ((axes[1], "violation_existing"), (axes[2], "dist_K")):
        ax.set_yscale("log")
        lo_ = min(float(np.min(budget_curves(data, m, which)[2].mean(0)[budget_curves(data, m, which)[0] >= 1])) for m in modes)
        hi_ = max(float(np.max(budget_curves(data, m, which)[2].mean(0) + budget_curves(data, m, which)[2].std(0))) for m in modes)
        ax.set_ylim(lo_ / 3.0, hi_ * 1.5)  # std bands clipped to the view on the log axis
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Cumulative NFE per path (log)")
        vis._style_axis_grid(ax, minor=True)
    axes[0].set_title("Objective $f(x)$")
    axes[1].set_title("Constraint violation (paper metric)")
    axes[2].set_title(r"Exact distance to $\mathcal{K}$")
    axes[0].set_ylabel("objective")
    axes[1].set_ylabel(r"$\|x - \mathrm{proj}(x)\|$ (physical units)")
    axes[2].set_ylabel(r"$\mathrm{dist}(x,\mathcal{K})$ (physical units)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=11, frameon=True,
               framealpha=0.55, edgecolor="#CCCCCC")
    fig.suptitle(title + " — quality vs. compute (mean ± std over 100 test paths; GGDM: final iterate of each round; "
                 "paper violation metric = radial scaling for Test B)", fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    safe_io.save_figure(fig, out_stem)
    plt.close(fig)


# ------------------------------------------------------------------ (iii) objective-feasibility plane
def final_vals(summ, data, mode, metric):
    d = data[mode][f"m_{metric}"]
    if mode in PNP:
        K = int(summ["pnp_selection"][mode]["selection"][2])
        return d[:, K]
    return d[:, -1]


def fig_plane(summ, data, vg, rows, out_stem, title):
    vis._apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for v in PNP:
        rr = [r for r in rows if r["variant"] == v and r["in_selection_grid"]]
        ax.scatter([r["dist_K_mean"] for r in rr], [r["objective_mean"] for r in rr], s=18, color=COL[v],
                   alpha=0.28, marker="o" if v == "pnp" else ("s" if v == "pnp_ddim_init" else "^"), lw=0,
                   label=f"{LAB[v]} grid (validation)")
    for mode in ("unguided",) + EXISTING + PNP:
        x = final_vals(summ, data, mode, "dist_K")
        y = final_vals(summ, data, mode, "objective")
        fmt = {"pnp": "*", "pnp_ddim_init": "D", "pnp_noscale": "^"}.get(mode, "o")
        ax.errorbar(x.mean(), y.mean(), xerr=x.std(), yerr=y.std(), fmt=fmt,
                    color=COL[mode], ms=16 if mode == "pnp" else 8, mec="black", mew=0.8, capsize=3, lw=1.4,
                    label=LAB[mode] + " (test)", zorder=7 if mode == "pnp" else 5)
    ax.axhline(summ["f_star_exact"], color="black", ls="--", lw=1.6, label="Optimal (feasible) $f^*$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Final exact $\mathrm{dist}(x,\mathcal{K})$ (log)")
    ax.set_ylabel("Final objective $f(x)$")
    ax.set_title(title, fontsize=14)
    vis._style_axis_grid(ax, minor=True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10, frameon=True, framealpha=0.55, edgecolor="#CCCCCC")
    fig.tight_layout()
    safe_io.save_figure(fig, out_stem)
    plt.close(fig)


# ------------------------------------------------------------------ (iv) DCG diagnostic
def dcg_diag(summ, data, out_stem, title, thr, n_thin=5):
    vis._apply_paper_style()
    d = data["optimization"]
    dsub, dK, f, G = d["m_dist_subspace"], d["m_dist_K"], d["m_objective"], d["m_G_norm"]
    n, T = dsub.shape
    steps = np.arange(T)
    best_strict = np.argmin(G, axis=1)  # first minimum (v1 definition; float-noise tie-breaking between steps 49/50)
    Gmin = G[np.arange(n), best_strict]
    # v2: best iterate with tolerance BEST_TOL on ||G||, ties broken toward the LATEST step (i.e. toward the final sample)
    within = G <= (Gmin[:, None] + BEST_TOL)
    best = (T - 1) - np.argmax(within[:, ::-1], axis=1)
    gap_G = G[:, -1] - Gmin
    gap_f = f[:, -1] - f[np.arange(n), best_strict]
    gap_x = d["m_dist_xstar"][:, -1] - d["m_dist_xstar"][np.arange(n), best_strict]
    hp = d["hist_phys"]
    last_step = {"max_abs_dx_phys_final_vs_prev": float(np.abs(hp[:, -1] - hp[:, -2]).max()),
                 "max_abs_dx_norm_final_vs_prev": float(np.abs(d["hist_norm"][:, -1] - d["hist_norm"][:, -2]).max()),
                 "max_abs_dG_final_vs_prev": float(np.abs(G[:, -1] - G[:, -2]).max()),
                 "max_abs_dx_phys_prev_vs_prevprev": float(np.abs(hp[:, -2] - hp[:, -3]).max())}
    frac_final_tol = {f"{tl:g}": float(np.mean(gap_G <= tl)) for tl in (0.0, 1e-5, BEST_TOL, 1e-3, 1e-2)}
    hist_strict = {int(t): int(c) for t, c in zip(*np.unique(best_strict, return_counts=True))}
    hist_tol = {int(t): int(c) for t, c in zip(*np.unique(best, return_counts=True))}
    # first step after which a path stays inside the neighbourhood dist_sub < thr (several thresholds)
    neigh = {}
    for th in thr:
        entry = []
        for i in range(n):
            idx = np.where(~(dsub[i] < th))[0]
            entry.append(int(idx[-1] + 1) if idx.size else 0)
        entry = np.array(entry)
        ev = entry[entry < T]
        neigh[f"{th:g}"] = {
            "step_mean_dist_subspace_below_thr": int(np.argmax(dsub.mean(0) < th)) if np.any(dsub.mean(0) < th) else None,
            "step_all_paths_stay_below_thr": int(entry.max()) if np.all(entry < T) else None,
            "per_path_entry_step_median": float(np.median(ev)) if ev.size else None,
            "per_path_entry_step_min": int(ev.min()) if ev.size else None,
            "per_path_entry_step_max": int(ev.max()) if ev.size else None,
            "n_paths_never_entering": int(np.sum(entry >= T)),
        }
    thr0 = thr[0]
    mean_entry = neigh[f"{thr0:g}"]["step_mean_dist_subspace_below_thr"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    spec = [(axes[0], [(dsub, r"dist. to subspace $\|x_\perp\|$", "-"), (dK, r"exact dist. to $\mathcal{K}$", "--")], True,
             "Distance to manifold / feasible set"),
            (axes[1], [(f, "objective $f$", "-")], False, "Objective"),
            (axes[2], [(G, r"$\|G_{\eta}(x)\|$, $\eta=1/(2L)$", "-")], True, "Stationarity residual")]
    for ax, series, logy, ttl in spec:
        for arr, lab, ls in series:
            c = "#8C2D04" if ls == "--" else COL["optimization"]
            mean, std = arr.mean(0), arr.std(0)
            ax.plot(steps, mean, color=c, lw=2.6, ls=ls, label=f"{lab} (mean ± std)")
            lo = np.maximum(mean - std, 1e-6) if logy else mean - std
            ax.fill_between(steps, lo, mean + std, color=c, alpha=0.15)
            for i in range(min(n_thin, n)):
                ax.plot(steps, arr[i], color=c, lw=0.7, ls=ls, alpha=0.55)
                ax.plot([best[i]], [arr[i, best[i]]], marker="*", ms=13, color="#FFD92F", mec="black", mew=0.6,
                        ls="none", zorder=7)
                ax.plot([T - 1], [arr[i, -1]], marker="s", ms=4, color="black", ls="none", zorder=8)
            # v2: additionally show the path with the LARGEST final-minus-best ||G|| gap (worst case, labelled)
            iw = int(np.argmax(gap_G))
            ax.plot(steps, arr[iw], color="#7F2704", lw=1.0, ls=":", alpha=0.9, zorder=5)
            ax.plot([best[iw]], [arr[iw, best[iw]]], marker="*", ms=13, color="#FFD92F", mec="black", mew=0.6,
                    ls="none", zorder=7)
            ax.plot([T - 1], [arr[iw, -1]], marker="s", ms=4, color="black", ls="none", zorder=8)
        if logy:
            ax.set_yscale("log")
            lo_ = min(min(float(arr.mean(0).min()), float(arr[:n_thin].min())) for arr, _, _ in series)
            hi_ = max(float((arr.mean(0) + arr.std(0)).max()) for arr, _, _ in series)
            ax.set_ylim(lo_ / 2.0, hi_ * 1.5)
        ax.set_xlabel("Reverse diffusion step")
        ax.set_title(ttl)
        vis._style_axis_grid(ax, minor=True)
    axes[0].axhline(thr0, color="#555555", lw=1.0, ls=":")
    if mean_entry is not None:
        for ax in axes:
            ax.axvline(mean_entry, color="#555555", lw=1.0, ls=":")
        axes[0].annotate(f"mean dist. to subspace\n< {thr0:g} from step {mean_entry}", xy=(mean_entry, thr0),
                         xytext=(0.06, 0.40), textcoords="axes fraction", fontsize=10,
                         bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", alpha=0.85),
                         arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    axes[2].text(0.03, 0.04, f"final − best $\\|G\\|$ over {n} paths:\nmean {gap_G.mean():.3g}, median {np.median(gap_G):.2g}, "
                 f"max {gap_G.max():.3g}\nfinal within {BEST_TOL:g} of best: {100 * frac_final_tol[f'{BEST_TOL:g}']:.0f}% of paths",
                 transform=axes[2].transAxes, fontsize=10, va="bottom",
                 bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", alpha=0.8))
    axes[1].axhline(summ["f_star_exact"], color="black", ls="--", lw=1.8, label="Optimal (feasible) $f^*$")
    axes[0].plot([], [], color="#7F2704", lw=1.0, ls=":", label=f"path with largest final−best gap (#{int(np.argmax(gap_G))})")
    axes[0].plot([], [], marker="s", color="black", ls="none", label="final sample (shown paths)")
    axes[0].plot([], [], marker="*", color="#FFD92F", mec="black", ls="none", ms=10,
                 label=r"best iterate by $\|G\|$" + f" (tol {BEST_TOL:g}, latest)")
    axes[0].legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.55, edgecolor="#CCCCCC")
    axes[1].legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.55, edgecolor="#CCCCCC")
    axes[2].legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.55, edgecolor="#CCCCCC")
    fig.suptitle(f"{title}: DCG from the original initialization (100 paths; thin lines = paths 0-4, dotted = worst final-vs-best path)", fontsize=14)
    fig.tight_layout()
    safe_io.save_figure(fig, out_stem)
    plt.close(fig)
    return {
        "thresholds_dist_subspace_physical": list(thr),
        "annotated_threshold": thr0,
        "neighbourhood_entry": neigh,
        "best_iterate_definition": (f"strict: first argmin_t ||G_eta(x_t)|| (v1); tolerant (v2, used for markers): latest t with "
                                    f"||G(x_t)|| <= min_t ||G|| + {BEST_TOL:g}. The last DDIM step (schedule index 0->0) is a "
                                    f"numerical near-identity, so strict ties between steps 49 and 50 are float noise."),
        "last_ddim_step_change": last_step,
        "best_iterate_step_strict": {"median": float(np.median(best_strict)), "min": int(best_strict.min()),
                                     "max": int(best_strict.max()), "frac_is_final": float(np.mean(best_strict == T - 1)),
                                     "histogram": hist_strict},
        "best_iterate_step_tolerant": {"tol": BEST_TOL, "median": float(np.median(best)), "min": int(best.min()),
                                       "max": int(best.max()), "frac_is_final": float(np.mean(best == T - 1)),
                                       "histogram": hist_tol},
        "frac_paths_final_within_tol_of_best_G": frac_final_tol,
        "n_paths_final_minus_best_G_gt_1e-3": int(np.sum(gap_G > 1e-3)),
        "final_minus_best_G": {"mean": float(gap_G.mean()), "max": float(gap_G.max()), "median": float(np.median(gap_G))},
        "final_minus_best_objective": {"mean": float(gap_f.mean()), "max": float(gap_f.max()), "min": float(gap_f.min())},
        "final_minus_best_dist_xstar": {"mean": float(gap_x.mean()), "max": float(gap_x.max())},
        "final_G": {"mean": float(G[:, -1].mean()), "std": float(G[:, -1].std())},
        "best_G": {"mean": float(Gmin.mean()), "std": float(Gmin.std())},
        "final_dist_subspace": {"mean": float(dsub[:, -1].mean()), "max": float(dsub[:, -1].max())},
        "final_dist_K": {"mean": float(dK[:, -1].mean()), "max": float(dK[:, -1].max())},
    }


# ------------------------------------------------------------------ tables
COLS = [("objective", "f"), ("suboptimality", "f - f*"), ("violation_existing", "viol. (existing)"),
        ("dist_K", "dist_K (exact)"), ("dist_xstar", "||x - x*||"), ("G_norm", "||G_eta||")]


def table_rows(summ, data):
    out = []
    tim = summ["timing"]
    for mode in ("optimization", "standard", "ggdm", "drgd", "pnp", "pnp_ddim_init", "pnp_noscale", "unguided"):
        Ks = [None]
        if mode in PNP:
            Ks = [int(summ["pnp_selection"][mode]["selection"][2]), data[mode]["m_objective"].shape[1] - 1]
        for K in Ks:
            r = {"method": LAB[mode], "key": mode, "K": K}
            idx = -1 if K is None else K
            for m, _ in COLS:
                v = data[mode][f"m_{m}"][:, idx]
                r[f"{m}_mean"], r[f"{m}_std"] = float(v.mean()), float(v.std())
            c = summ["costs"][mode]
            if mode in PNP:
                sel = summ["pnp_selection"][mode]
                r["method"] = f"{LAB[mode]} (k={sel['selection'][0]}, σ̄={sel['sigma_bar']:.3f}, η={sel['selection'][1]:g}/L, K={K})"
                r["K"] = K
                r["nfe"] = c["init_nfe_per_path"] + K
                r["grad"] = float(K)
                r["bwd"] = 0.0
                r["init_nfe"] = c["init_nfe_per_path"]
                tkey = f"{mode}@K{K}"
            else:
                e = c["measured_per_path"]
                r["nfe"], r["grad"], r["bwd"] = e["nfe"], e["grad"], e["bwd"]
                r["init_nfe"] = 50.0 if mode == "ggdm" else 0.0
                tkey = mode
            r["time_b100_ms"] = 1e3 * tim["batch100"][tkey]["median_s"]
            r["time_b1_ms"] = 1e3 * tim["batch1"][tkey]["median_s"]
            r["counts_match_analytic"] = c["match"]
            out.append(r)
    return out


def write_tables(run_dir, stem, tests_data):
    md = io.StringIO()
    csv_rows = []
    tex = io.StringIO()
    md.write("# Table 1 — synthetic constrained optimization (100 test paths, paper initializations)\n\n")
    md.write("Mean ± std over paths (np.std, ddof=0, as in the existing code). f* from exact Euclidean projection "
             "x* = P_K(c). 'viol. (existing)' = paper metric ||x - project_to_feasible(x)|| (radial scaling for test B); "
             "dist_K = exact Euclidean distance. ||G|| = ||x - P_K(x - η∇f(x))||/η, η = 1/(2 L_phys) = 0.25, physical "
             "coordinates. Costs per path: NFE = denoiser network forward evaluations actually executed (incl. the numerically near-identity "
             "last DDIM step and, for GGDM / PnP-PGD (DDIM init), the 50-NFE unguided initialization); grad = objective-gradient "
             "evaluations executed; bwd = backward passes through the network. Wall-clock: one call on the full batch of "
             "100 paths (and on batch = 1), GH200, median of 5 after warm-up, synchronized, no history recording; all methods "
             "timed in the same SLURM job as the compute run (absolute times vary by ~15-20% between nodes/jobs).\n\n")
    for test, (summ, data) in tests_data.items():
        rows = table_rows(summ, data)
        md.write(f"## {TEST_TITLE[test]}  (f* = {summ['f_star_exact']:.4f}, L_norm = {summ['L_norm']:.4f})\n\n")
        md.write("| Method | f | f − f* | viol. (existing) | dist_K (exact) | ‖x − x*‖ | ‖G‖ | NFE/path | grad/path | bwd/path | ms / batch 100 | ms / batch 1 |\n")
        md.write("|" + "---|" * 12 + "\n")
        for r in rows:
            vals = [pm(r["objective_mean"], r["objective_std"]), pm(r["suboptimality_mean"], r["suboptimality_std"]),
                    pm(r["violation_existing_mean"], r["violation_existing_std"], "{:.2e}"),
                    pm(r["dist_K_mean"], r["dist_K_std"], "{:.2e}"), pm(r["dist_xstar_mean"], r["dist_xstar_std"]),
                    pm(r["G_norm_mean"], r["G_norm_std"])]
            md.write(f"| {r['method']} | " + " | ".join(vals) +
                     f" | {r['nfe']:g} | {r['grad']:g} | {r['bwd']:g} | {r['time_b100_ms']:.1f} | {r['time_b1_ms']:.1f} |\n")
            csv_rows.append({"test": test, **r})
        md.write("\nGGDM grad/path counts every classifier.gradients call executed by the existing implementation: "
                 "R·S = 2500 per-step calls (used only for logging log_p, not for the update) + R = 50 anchor gradients at "
                 "the particle mean (one batch-of-1 evaluation shared by all 100 particles → 0.5/path). "
                 "Algorithmically required: 50 shared anchor gradients.\n\n")
        # LaTeX
        tex.write(f"% {TEST_TITLE[test]}\n\\begin{{tabular}}{{lrrrrrrrrrr}}\n\\toprule\n")
        tex.write("Method & $f$ & $f-f^*$ & viol. & $\\mathrm{dist}_\\mathcal{K}$ & $\\|x-x^*\\|$ & $\\|G\\|$ & NFE & grad & bwd & ms/100 \\\\\n\\midrule\n")
        for r in rows:
            name = r["method"].replace("σ̄", "$\\bar\\sigma$").replace("η", "$\\eta$")
            dag = "$^\\dagger$" if r["key"] == "ggdm" else ""
            tex.write(f"{name} & ${r['objective_mean']:.3f}\\pm{r['objective_std']:.3f}$ & {r['suboptimality_mean']:.3f} & "
                      f"{r['violation_existing_mean']:.1e} & {r['dist_K_mean']:.1e} & {r['dist_xstar_mean']:.3f} & "
                      f"{r['G_norm_mean']:.3f} & {r['nfe']:g} & {r['grad']:g}{dag} & "
                      f"{r['bwd']:g} & {r['time_b100_ms']:.0f} \\\\\n")
        tex.write("\\bottomrule\n\\end{tabular}\n")
        tex.write("\\par\\smallskip{\\footnotesize $^\\dagger$GGDM grad/path counts every objective-gradient call executed "
                  "by the existing implementation: $R\\cdot S=2500$ per-step calls (used only to log $\\log p$, not in the update) "
                  "plus $R=50$ anchor gradients at the particle mean (one batch-of-1 evaluation shared by 100 particles, "
                  "$0.5$/path). Algorithmically required: 50 shared anchor gradients. "
                  "Mean $\\pm$ std over 100 test paths (ddof $=0$); ms/100 = wall-clock for one call on the batch of 100.}\n\n")
    safe_io.write_text(Path(f"{stem}_table1.md"), md.getvalue())
    safe_io.write_text(Path(f"{stem}_table1.tex"), tex.getvalue())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)
    safe_io.write_text(Path(f"{stem}_table1.csv"), buf.getvalue())


def write_valgrid(stem, tests_rows, tests_summ):
    md = io.StringIO()
    md.write("# PnP-PGD validation grids (20 validation inits, CUDA seeds 10000..10019)\n\n"
             "Criterion (fixed in advance): minimize validation mean ||x_K − x*|| (physical). Rows with K = 50 are reported "
             "for information only (not part of the selection grid). 'ext' = boundary-extension point.\n\n")
    allrows = []
    for test, rows in tests_rows.items():
        s = tests_summ[test]
        for v in PNP:
            sel = s["pnp_selection"][v]
            md.write(f"## {TEST_TITLE[test]} — {LAB[v]}\n\nSelected (k, η/L, K) = {tuple(sel['selection'])}, "
                     f"value {sel['value']:.4f}; initial selection {tuple(sel['initial_selection'])}; "
                     f"extensions: {sel['extended'] or 'none'}; selected on extended boundary: "
                     f"{sel['selected_on_extended_boundary'] or 'no'}.\n\n")
            md.write("| k | σ̄ | η/L | η | K | ‖x_K − x*‖ | f | dist_K | viol. (existing) | ‖G‖ | sel | ext |\n|" + "---|" * 12 + "\n")
            for r in rows:
                if r["variant"] != v:
                    continue
                md.write(f"| {r['k']} | {r['sigma_bar']:.3f} | {r['eta_over_L']:g} | {r['eta']:.4f} | {r['K']}{'' if r['in_selection_grid'] else ' (info)'} | "
                         f"{pm(r['dist_xstar_mean'], r['dist_xstar_std'])} | {pm(r['objective_mean'], r['objective_std'])} | "
                         f"{r['dist_K_mean']:.2e} | {r['violation_existing_mean']:.2e} | {r['G_norm_mean']:.4f} | "
                         f"{'**✓**' if r['selected'] else ''} | {'ext' if r['is_extension'] else ''} |\n")
                allrows.append({"test": test, **r})
            md.write("\n")
    safe_io.write_text(Path(f"{stem}_pnp_validation_grid.md"), md.getvalue())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(allrows[0].keys()))
    w.writeheader()
    w.writerows(allrows)
    safe_io.write_text(Path(f"{stem}_pnp_validation_grid.csv"), buf.getvalue())


def write_repro(stem, tests_summ):
    md = io.StringIO()
    md.write("# Reproduction of the paper curves (re-run vs cached paper trajectories)\n\n"
             "max |Δ| over 100 paths × 51 steps. Reward = −f, violation = existing metric, subspace path = B^T x.\n\n")
    md.write("| test | method | max Δ reward curve | max Δ violation curve | max Δ subspace path | final f rerun | final f paper | final viol rerun | final viol paper |\n|" + "---|" * 9 + "\n")
    for test, s in tests_summ.items():
        for m, r in s["reproduction"].items():
            md.write(f"| {test} | {LAB.get(m, m)} | {sci(r['max_abs_reward_curve'])} | {sci(r['max_abs_violation_curve'])} | "
                     f"{sci(r['max_abs_subspace_path'])} | {-r['final_reward_mean_rerun']:.4f} ± {r['final_reward_std_rerun']:.4f} | "
                     f"{-r['final_reward_mean_cache']:.4f} ± {r['final_reward_std_cache']:.4f} | "
                     f"{r['final_violation_mean_rerun']:.4f} | {r['final_violation_mean_cache']:.4f} |\n")
        md.write(f"| {test} | GGDM replica vs original code path (same seed) | final max Δ (normalized) = "
                 f"{sci(s['ggdm_replica_vs_original_max_abs_final'])}, history max Δ = {sci(s['ggdm_replica_vs_original_max_abs_hist'])} | | | | | | |\n")
        md.write(f"| {test} | batched (n=100) vs per-path (paper) max Δ history (normalized) | "
                 + ", ".join(f"{LAB[k]}: {sci(v)}" for k, v in s["batched_vs_perpath_max_abs_hist_norm"].items()) + " | | | | | | |\n")
        md.write(f"| {test} | cached-vs-regenerated initial noise max Δ | {sci(s['init_regen_max_abs_diff'])} | | | | | | |\n")
        md.write(f"| {test} | x* exact vs cached SLSQP x* (‖Δ‖) | "
                 f"{sci(float(np.linalg.norm(np.array(s['x_star_exact']) - np.array(s['x_star_cache']))))}; f* exact {s['f_star_exact']:.6f} vs cached {s['f_star_cache']:.6f} | | | | | | |\n")
    safe_io.write_text(Path(f"{stem}_reproduction.md"), md.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.1, 0.05],
                    help="distance-to-subspace neighbourhood thresholds (physical); first one is annotated")
    a = ap.parse_args()
    run_dir = Path(a.run_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = safe_io.new_run_dir(safe_io.FIGURES_ROOT / "synthetic", run_id=f"{run_dir.name}__{stamp}")
    tab_dir = safe_io.ensure_dir(safe_io.TABLES_ROOT / "synthetic")
    stem = tab_dir / f"{run_dir.name}__{stamp}_v2"
    with open(run_dir / "config.json") as f:
        tests = json.load(f)["tests"]
    tests_data, tests_rows, tests_summ, diag = {}, {}, {}, {}
    for test in tests:
        summ, data, vg, rows = load(run_dir, test)
        tests_data[test], tests_rows[test], tests_summ[test] = (summ, data), rows, summ
        ttl = TEST_TITLE[test]
        fig_convergence(summ, data, fig_dir / f"{test}_pnp_rebuttal_s50_convergence")
        fig_budget(summ, data, fig_dir / f"{test}_pnp_rebuttal_budget_nfe", ttl)
        fig_plane(summ, data, vg, rows, fig_dir / f"{test}_pnp_rebuttal_objective_vs_feasibility", ttl)
        diag[test] = dcg_diag(summ, data, fig_dir / f"{test}_dcg_reviewer3_diagnostic", ttl, tuple(a.thr))
    write_tables(run_dir, stem, tests_data)
    write_valgrid(stem, tests_rows, tests_summ)
    write_repro(stem, tests_summ)
    md = io.StringIO()
    md.write("# Reviewer-3 DCG diagnostic (DCG from the original paper initialization, 100 test paths)\n\n"
             "Best iterate along each path by ||G_eta|| (eta = 1/(2 L_phys) = 0.25, physical coords). Gaps are final minus the "
             "minimum over steps (>= 0 for G by construction). The last DDIM step (schedule index 0->0) changes the DCG iterate "
             "only at float-noise level (see last_ddim_step_change), so 'best = final' is reported with a tolerance "
             f"({BEST_TOL:g} on ||G||, ties broken toward the latest step); the strict first-argmin statistic is also given but "
             "is a tie-breaking artefact between steps 49 and 50. Empirical observation only; no convergence or last-iterate "
             "claim.\n\n")
    for test, dd in diag.items():
        md.write(f"## {TEST_TITLE[test]}\n\n```json\n{json.dumps(dd, indent=2)}\n```\n\n")
    safe_io.write_text(Path(f"{stem}_dcg_diagnostic.md"), md.getvalue())
    safe_io.write_json(fig_dir / "report_meta.json", {"run_dir": str(run_dir), "tables_stem": str(stem),
                                                      "dcg_diagnostic": diag, "provenance": safe_io.provenance(),
                                                      "report_script_sha256": _sha256(Path(__file__).resolve()),
                                                      "run_code_sha256": json.load(open(run_dir / "config.json")).get("code_sha256"),
                                                      "note": "existing-method curves in all figures are the re-run curves from run_dir"})
    print(f"figures -> {fig_dir}\ntables -> {stem}_*")


if __name__ == "__main__":
    main()
