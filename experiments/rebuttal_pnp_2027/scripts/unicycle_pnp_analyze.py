#!/usr/bin/env python
"""CPU-only analysis for the unicycle PnP-PGD rebuttal experiment.

Reads a validation run dir and a test run dir (under results/unicycle/), and writes
  tables/unicycle/<tag>_*.{md,csv,tex}  : Table 2 (+costs), ablation table, validation grid,
                                          reproduction check vs. the published paper runs
  figures/unicycle/<tag>/...            : PnP-PGD paper-style plan plots (top-4 by plan reward),
                                          c_rollout vs e_dyn, c_rollout vs NFE (budget), per-iteration curves
All writes go through rebuttal_pnp.safe_io (no overwrite).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E = Path("/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027")
W = Path("/u/rzhang26/CleanDiffuser_rebuttal_pnp")
for p in (E, W / "pipelines"):
    sys.path.insert(0, str(p))
from rebuttal_pnp import safe_io  # noqa: E402
import reference_plan_replot as rp  # noqa: E402  (pure matplotlib/numpy, no torch)

PAPER_ROOT = safe_io.ORIG_RESULTS / "unicycle_eval" / "reference_plan_paper_v1"
TEST_REFS = ["circle", "random_smooth_1", "random_smooth_7"]
REF_LABEL = {"circle": "Circle", "random_smooth_1": "Random Trajectory 1", "random_smooth_7": "Random Trajectory 2"}
VAL_REFS = ["half_heart", "random_smooth_2", "random_smooth_4"]
PAPER_DIR = {"monte_carlo": "standard_w_cg0", "pdg": "standard_w_cg30", "dcg": "optimization_scale_30", "drgd": "drgd_gamma15"}
MAIN = ["monte_carlo", "pdg", "dcg", "drgd", "pnp_pgd"]
ABL = ["pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale"]
LABEL = {
    "monte_carlo": "Monte Carlo", "pdg": "PDG", "dcg": "DCG", "drgd": r"DRGD ($\gamma=15$)", "pnp_pgd": "PnP-PGD",
    "pnp_pgd_budget20": "PnP-PGD (a) budget-matched", "pnp_pgd_from_xT": r"PnP-PGD (b) init $x_T$",
    "pnp_pgd_noscale": "PnP-PGD (c) no input scaling",
}
TEX_LABEL = {
    "monte_carlo": "Monte Carlo", "pdg": "PDG", "dcg": "DCG", "drgd": r"Landing DRGD ($\gamma=15$)", "pnp_pgd": "PnP-PGD",
    "pnp_pgd_budget20": "PnP-PGD (a) budget-matched", "pnp_pgd_from_xT": r"PnP-PGD (b) init $x_T$",
    "pnp_pgd_noscale": r"PnP-PGD (c) no input scaling",
}
COLOR = {"monte_carlo": "#7F7F7F", "pdg": "#2166AC", "dcg": "#E66101", "drgd": "#984EA3", "pnp_pgd": "#E7298A",
         "pnp_pgd_budget20": "#F197C4", "pnp_pgd_from_xT": "#F4B6D4", "pnp_pgd_noscale": "#C9227A"}
MARKER = {"monte_carlo": "o", "pdg": "s", "dcg": "D", "drgd": "^", "pnp_pgd": "P",
          "pnp_pgd_budget20": "v", "pnp_pgd_from_xT": "<", "pnp_pgd_noscale": ">"}
TABLES_DIR = safe_io.TABLES_ROOT / "unicycle"
LS = {"pnp_pgd": "-", "pnp_pgd_budget20": "--", "pnp_pgd_from_xT": ":", "pnp_pgd_noscale": "-."}


def load(run_dir, ref, method):
    d = run_dir / ref / method / "reference_plan" / "trajectories"
    out = {"npz": dict(np.load(d / "candidates.npz")), "summary": json.load(open(d / "summary.json")),
           "costs": json.load(open(d / "costs_timing.json"))}
    if (d / "per_iteration.npz").is_file():
        out["iter"] = dict(np.load(d / "per_iteration.npz"))
    return out


def ms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0


def parse_paper_table():
    """Published numbers (drgd_paper_table_gamma15.tex), read-only."""
    txt = (PAPER_ROOT / "drgd_paper_table_gamma15.tex").read_text()
    out, setting = {}, None
    inv = {v: k for k, v in REF_LABEL.items()}
    alg_map = {"Monte Carlo": "monte_carlo", "PDG": "pdg", "DCG": "dcg", "Landing DRGD ($\\gamma=15$)": "drgd"}
    for line in txt.splitlines():
        m = re.search(r"\\multirow\{\d+\}\{\*\}\{(.+?)\}", line)
        if m:
            setting = inv[m.group(1)]
            continue
        m = re.match(r"\s*&\s*(.+?)&\s*\$(.+?)\$&\s*\$(.+?)\$&\s*\$(.+?)\$", line)
        if m and setting:
            vals = []
            for cell in m.groups()[1:]:
                a, b = cell.split("\\pm")
                vals.append((float(a), float(b)))
            out[(setting, alg_map[m.group(1).strip()])] = dict(zip(["c_plan", "c_rollout", "e_dyn"], vals))
    return out


def fmt(m, s, nd=4):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def write_tables(tag, rows, header, stem, tex_rows=None, tex_colspec=None, tex_header=None):
    out_dir = safe_io.ensure_dir(TABLES_DIR)
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)
    safe_io.write_text(out_dir / f"{tag}_{stem}.md", md)
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(header)
    wr.writerows(rows)
    safe_io.write_text(out_dir / f"{tag}_{stem}.csv", buf.getvalue())
    if tex_rows is not None:
        safe_io.write_text(out_dir / f"{tag}_{stem}.tex", tex_rows)
    return out_dir / f"{tag}_{stem}.md"


def tex_table(test_dir, methods, refs):
    n_cols = 9
    lines = [r"\resizebox{\linewidth}{!}{", r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Setting & Algorithm & $c_{\mathrm{plan}}$ & $c_{\mathrm{rollout}}$ & $e_{\mathrm{dyn}}$ & NFE & Net bwd & Grad evals & Runtime (ms) \\",
             r"\midrule"]
    for ri, ref in enumerate(refs):
        for mi, m in enumerate(methods):
            d = load(test_dir, ref, m)
            s, c = d["summary"]["metrics"], d["costs"]["summary"]
            first = rf"\multirow{{{len(methods)}}}{{*}}{{{REF_LABEL[ref]}}}" if mi == 0 else ""
            lines.append(
                f"{first} & {TEX_LABEL[m]} & ${s['c_plan']['mean']:.4f} \\pm {s['c_plan']['std']:.4f}$"
                f" & ${s['c_rollout']['mean']:.4f} \\pm {s['c_rollout']['std']:.4f}$"
                f" & ${s['e_dyn']['mean']:.4f} \\pm {s['e_dyn']['std']:.4f}$"
                f" & {c['nfe']['mean']:.0f} & {c['net_backward']['mean']:.0f} & {c['grad_evals']['mean']:.0f}"
                f" & ${1e3 * c['seconds']['mean']:.1f} \\pm {1e3 * c['seconds']['std']:.1f}$ \\\\")
        lines.append(r"\midrule" if ri < len(refs) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", "}"]
    return "\n".join(lines) + "\n"


def table_rows(test_dir, methods, refs, paper=None):
    rows = []
    for ref in refs:
        for m in methods:
            d = load(test_dir, ref, m)
            s, c = d["summary"]["metrics"], d["costs"]["summary"]
            row = [REF_LABEL[ref], LABEL[m].replace("$", "").replace("\\gamma", "gamma"),
                   fmt(s["c_plan"]["mean"], s["c_plan"]["std"]), fmt(s["c_rollout"]["mean"], s["c_rollout"]["std"]),
                   fmt(s["e_dyn"]["mean"], s["e_dyn"]["std"]),
                   f"{c['nfe']['mean']:.0f}", f"{c['net_backward']['mean']:.0f}", f"{c['grad_evals']['mean']:.0f}",
                   f"{1e3 * c['seconds']['mean']:.2f} ± {1e3 * c['seconds']['std']:.2f}"]
            if m.startswith("pnp"):
                row.append(f"init {c['init_nfe']['mean']:.0f} NFE / {1e3 * c['init_seconds']['mean']:.2f} ms; "
                           f"iters {c['iter_nfe']['mean']:.0f} NFE + {c['iter_grad_evals']['mean']:.0f} grads / {1e3 * c['iter_seconds']['mean']:.2f} ms")
            elif paper is not None and (ref, m) in paper:
                p = paper[(ref, m)]
                row.append("published: " + "; ".join(f"{k} {p[k][0]:.4f} ± {p[k][1]:.4f}" for k in ("c_plan", "c_rollout", "e_dyn")))
            else:
                row.append("")
            rows.append(row)
    return rows


def reproduction(test_dir, paper):
    rows, details = [], {}
    for ref in TEST_REFS:
        for m, pdir in PAPER_DIR.items():
            ours = load(test_dir, ref, m)["npz"]
            theirs = dict(np.load(PAPER_ROOT / ref / pdir / "reference_plan" / "trajectories" / "candidates.npz"))
            pos = {int(sd): i for i, sd in enumerate(theirs["seed"])}
            sel = np.array([pos[int(sd)] for sd in ours["seed"]])  # identity for the full 100-candidate run
            theirs = {k: (v[sel] if v.ndim and v.shape[0] == len(theirs["seed"]) and k not in ("reference_xy",) else v)
                      for k, v in theirs.items()}
            assert np.array_equal(ours["seed"], theirs["seed"])
            det = {"keys_equal": sorted(ours) == sorted(theirs)}
            for k in ("plan_reward", "rollout_reward", "feasibility_mean_l2_future", "planned_obs", "planned_act", "rollout_xy",
                      "reference_xy", "init_obs"):
                det[f"maxabs_{k}"] = float(np.abs(ours[k].astype(np.float64) - theirs[k].astype(np.float64)).max())
            for k in ("plan_reward", "rollout_reward", "feasibility_mean_l2_future"):
                rel = np.abs(ours[k] - theirs[k]) / np.maximum(np.abs(theirs[k]), 1e-12)
                det[f"maxrel_{k}"] = float(rel.max())
                det[f"n_rel_gt_1e-3_{k}"] = int((rel > 1e-3).sum())
            pub = paper[(ref, m)]
            ours_s = {"c_plan": ms(-ours["plan_reward"]), "c_rollout": ms(-ours["rollout_reward"]), "e_dyn": ms(ours["feasibility_mean_l2_future"])}
            cache_s = {"c_plan": ms(-theirs["plan_reward"]), "c_rollout": ms(-theirs["rollout_reward"]), "e_dyn": ms(theirs["feasibility_mean_l2_future"])}
            det["ours"], det["cached_paper_npz"], det["published_table"] = ours_s, cache_s, pub
            ok4 = all(f"{ours_s[k][0]:.4f}" == f"{pub[k][0]:.4f}" and f"{ours_s[k][1]:.4f}" == f"{pub[k][1]:.4f}" for k in pub)
            det["all_table_entries_identical_at_4dp"] = ok4
            details[f"{ref}/{m}"] = det
            rows.append([REF_LABEL[ref], LABEL[m].replace("$", "").replace("\\gamma", "gamma"),
                         " / ".join(f"{ours_s[k][0]:.4f}±{ours_s[k][1]:.4f}" for k in ("c_plan", "c_rollout", "e_dyn")),
                         " / ".join(f"{pub[k][0]:.4f}±{pub[k][1]:.4f}" for k in ("c_plan", "c_rollout", "e_dyn")),
                         "yes" if ok4 else "NO",
                         f"{det['maxabs_plan_reward']:.2e}", f"{det['maxabs_rollout_reward']:.2e}",
                         f"{det['maxabs_feasibility_mean_l2_future']:.2e}", f"{det['maxabs_planned_obs']:.2e}"])
    return rows, details


# ---------------------------------------------------------------- figures
def plan_plot(planned_xy, rollout_xy, reference_xy, init_xy, stem):
    """Verbatim replica of rp.plot_plan_vs_rollout(style=rp.PLOT_STYLE_EMPHASIS), saved via safe_io."""
    style = rp.PLOT_STYLE_EMPHASIS
    x_lim, y_lim = rp.centered_xy_limits(reference_xy, planned_xy, rollout_xy, init_xy[None, :])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], color="0.15", linestyle="--", linewidth=style["ref_line_width"], alpha=0.85, zorder=2)
    ax.plot(planned_xy[:, 0], planned_xy[:, 1], color="#2166ac", linewidth=style["line_width"], alpha=0.95, zorder=4)
    ax.plot(rollout_xy[:, 0], rollout_xy[:, 1], color="#d95f02", linewidth=style["line_width"], linestyle="--", alpha=0.95, zorder=3)
    ax.scatter(init_xy[0], init_xy[1], color="black", s=style["start_size"], marker="*", zorder=7, edgecolors="white", linewidths=0.8)
    rp.setup_plan_axes(ax, x_lim, y_lim)
    fig.tight_layout()
    safe_io.save_figure(fig, stem, formats=("png", "pdf"), dpi=rp.PLAN_PLOT_DPI)
    plt.close(fig)


def legend_figure(stem):
    """Replica of rp.save_plan_legend_figure(style=EMPHASIS), saved via safe_io."""
    from matplotlib.lines import Line2D
    style = rp.PLOT_STYLE_EMPHASIS
    fig, ax = plt.subplots(figsize=(11.0, 0.55))
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="0.15", linestyle="--", linewidth=style["ref_line_width"], label="reference"),
        Line2D([0], [0], color="#2166ac", linewidth=style["line_width"], label="diffusion plan"),
        Line2D([0], [0], color="#d95f02", linestyle="--", linewidth=style["line_width"], label="open-loop rollout"),
        Line2D([0], [0], marker="*", color="black", markerfacecolor="black", markersize=float(np.sqrt(style["start_size"]) * 1.15),
               linestyle="None", label="starting point"),
    ]
    ax.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=rp.PLAN_PLOT_LEGEND_FONTSIZE,
              handlelength=2.8, handletextpad=0.6, columnspacing=1.6)
    fig.tight_layout(pad=0.05)
    safe_io.save_figure(fig, stem, formats=("png", "pdf"), dpi=rp.PLAN_PLOT_DPI)
    plt.close(fig)


def fig_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10, "axes.labelsize": 11,
                         "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.default": "regular"})


def fig_cost_vs_dyn(test_dir, methods, stem):
    fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, ref in zip(axes, TEST_REFS):
        for m in methods:
            d = load(test_dir, ref, m)["npz"]
            ed, cr = d["feasibility_mean_l2_future"], -d["rollout_reward"]
            abl = m in ABL
            ax.scatter(ed, cr, s=6, color=COLOR[m], alpha=0.12 if not abl else 0.06, linewidths=0, zorder=1)
            (em, es), (cm, cs) = ms(ed), ms(cr)
            lo = cs if cm - cs > 0 else 0.0  # lower whisker omitted when mean - std <= 0 (log axis)
            eb = ax.errorbar(em, cm, xerr=es, yerr=[[lo], [cs]], fmt=MARKER[m], ms=8 if not abl else 6.5, color=COLOR[m],
                             mec="white", mew=0.8, capsize=3, elinewidth=1.4, alpha=1.0 if not abl else 0.75, zorder=3,
                             label=LABEL[m])
            if abl:
                for line in eb[2]:
                    line.set_linestyle("--")
        ax.set_yscale("log")
        ax.set_xlabel(r"dynamics error $e_{\mathrm{dyn}}$ (m, mean $L_2$ over $t\geq1$)")
        ax.set_title(REF_LABEL[ref])
    axes[0].set_ylabel(r"execution cost $c_{\mathrm{rollout}}$ (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    safe_io.save_figure(fig, stem)
    plt.close(fig)


def fig_budget(test_dir, stem):
    fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, ref in zip(axes, TEST_REFS):
        for m in ["monte_carlo", "pdg", "dcg", "drgd"]:
            d = load(test_dir, ref, m)
            cr = -d["npz"]["rollout_reward"]
            nfe = d["costs"]["summary"]["nfe"]["mean"]
            lab = LABEL[m] + (" (+20 net bwd)" if m == "drgd" else "")
            sd_ = cr.std(ddof=1)
            ax.errorbar(nfe, cr.mean(), yerr=[[sd_ if cr.mean() - sd_ > 0 else 0.0], [sd_]], fmt=MARKER[m], ms=8,
                        color=COLOR[m], mec="white", mew=0.8, capsize=3, zorder=4, label=lab)
        for m in ["pnp_pgd", "pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale"]:
            d = load(test_dir, ref, m)
            it = d["iter"]
            c = it["c_rollout"]  # (n_cand, K+1)
            nfe = it["init_nfe"][0] + np.arange(c.shape[1])
            mean, sd = c.mean(0), c.std(0, ddof=1)
            ax.plot(nfe, mean, LS[m], color=COLOR[m], lw=2.0 if m == "pnp_pgd" else 1.5, label=LABEL[m] + r" (K sweep)", zorder=3)
            ax.fill_between(nfe, np.where(mean - sd > 0, mean - sd, mean), mean + sd, color=COLOR[m], alpha=0.15 if m == "pnp_pgd" else 0.07, lw=0)
            ax.plot(nfe[-1], mean[-1], MARKER[m], color=COLOR[m], ms=7, mec="white", mew=0.8, zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("denoiser NFE per candidate (incl. initialization)")
        ax.set_title(REF_LABEL[ref])
    axes[0].set_ylabel(r"execution cost $c_{\mathrm{rollout}}$ (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    safe_io.save_figure(fig, stem)
    plt.close(fig)


def fig_per_iter(test_dir, stem):
    fig_style()
    metrics = [("c_plan", r"plan cost $c_{\mathrm{plan}}$"), ("c_rollout", r"execution cost $c_{\mathrm{rollout}}$"),
               ("e_dyn", r"dynamics error $e_{\mathrm{dyn}}$ (m)")]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.0), sharex="col")
    for j, ref in enumerate(TEST_REFS):
        for i, (key, lab) in enumerate(metrics):
            ax = axes[i, j]
            for m in ["pnp_pgd", "pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale"]:
                it = load(test_dir, ref, m)["iter"]
                v = it[key]
                K = np.arange(v.shape[1])
                ax.plot(K, v.mean(0), LS[m], color=COLOR[m], lw=2.0 if m == "pnp_pgd" else 1.5, label=LABEL[m])
                ax.fill_between(K, v.mean(0) - v.std(0, ddof=1), v.mean(0) + v.std(0, ddof=1), color=COLOR[m], alpha=0.1, lw=0)
            for m in ["dcg", "drgd", "monte_carlo"]:
                npz = load(test_dir, ref, m)["npz"]
                val = {"c_plan": -npz["plan_reward"], "c_rollout": -npz["rollout_reward"], "e_dyn": npz["feasibility_mean_l2_future"]}[key]
                ax.axhline(val.mean(), color=COLOR[m], lw=1.0, ls=(0, (4, 2)), label=LABEL[m] + " (final)")
            if key != "e_dyn":
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(REF_LABEL[ref])
            if i == 2:
                ax.set_xlabel("PnP-PGD iteration K")
            if j == 0:
                ax.set_ylabel(lab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    safe_io.save_figure(fig, stem)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True, type=Path)
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = args.tag
    global TABLES_DIR
    if args.smoke:  # smoke tests: keep tables/figures out of the deliverable folders
        base = safe_io.RESULTS_ROOT / "unicycle" / f"_smoke_analysis_{tag}"
        TABLES_DIR, fig_dir = base / "tables", base / "figures"
    else:
        fig_dir = safe_io.FIGURES_ROOT / "unicycle" / tag
    paper = parse_paper_table()
    fig_dir = safe_io.ensure_dir(fig_dir)
    tbl_dir = safe_io.ensure_dir(TABLES_DIR)
    selection = json.load(open(args.val_dir / "selection.json"))

    hdr = ["Setting", "Algorithm", "c_plan", "c_rollout", "e_dyn", "NFE", "Net bwd", "Grad evals", "Runtime/sample (ms)", "Note"]
    write_tables(tag, table_rows(args.test_dir, MAIN, TEST_REFS, paper), hdr, "table2", tex_rows=tex_table(args.test_dir, MAIN, TEST_REFS))
    write_tables(tag, table_rows(args.test_dir, ["pnp_pgd"] + ABL, TEST_REFS), hdr, "table_ablations",
                 tex_rows=tex_table(args.test_dir, ["pnp_pgd"] + ABL, TEST_REFS))

    rrows, rdet = reproduction(args.test_dir, paper)
    write_tables(tag, rrows, ["Setting", "Algorithm", "reproduced c_plan / c_rollout / e_dyn", "published", "identical @4dp",
                              "max|Δ plan_reward|", "max|Δ rollout_reward|", "max|Δ e_dyn|", "max|Δ planned_obs|"], "reproduction_check")
    safe_io.write_json(tbl_dir / f"{tag}_reproduction_check_details.json", rdet)

    # Validation grid tables (pre-specified + extension rows, K in {5,10,20}).
    for variant in ("mc", "xT"):
        v = json.load(open(args.val_dir / f"validation_{variant}.json"))
        sel = v["selected"]
        rows = []
        for r in v["rows_all_iterations"]:
            if r["K"] not in (5, 10, 20):
                continue
            is_sel = (r["k"] == sel["k"] and r["eta_mult"] == sel["eta_mult"] and r["K"] == sel["K"])
            rows.append([r["k"], f"{r['sigma_bar']:.3f}", f"{r['eta_mult']}/L", f"{r['eta']:.4f}", r["K"],
                         "yes" if r["in_prespecified_grid"] else "extension",
                         *[f"{r[f'{ref}/c_rollout_mean']:.4f}" for ref in VAL_REFS],
                         f"{r['avg/c_rollout']:.4f}", f"{r['avg/c_plan']:.4f}", f"{r['avg/e_dyn']:.4f}", "**SELECTED**" if is_sel else ""])
        write_tables(tag, rows, ["k", "sigma_bar", "eta", "eta value", "K", "grid", *[f"c_rollout {r}" for r in VAL_REFS],
                                 "avg c_rollout", "avg c_plan", "avg e_dyn", "selected"], f"validation_grid_{variant}")

    # (i) PnP-PGD per-candidate plan plots, top-4 by plan reward, paper style.
    rp.apply_paper_plot_style()
    plot_index = []
    for ref in TEST_REFS:
        d = load(args.test_dir, ref, "pnp_pgd")["npz"]
        order = np.argsort(-d["plan_reward"], kind="stable")
        for rank, ci in enumerate(order[:4], start=1):
            cidx = int(d["candidate_idx"][ci])
            stem = fig_dir / "plan_plots" / f"plan_{ref}_pnp_pgd_rank{rank:02d}_cand{cidx}"
            plan_plot(d["planned_xy"][ci], d["rollout_xy"][ci], d["reference_xy"], d["init_xy"], stem)
            plot_index.append({"reference": ref, "rank": rank, "candidate_idx": cidx, "seed": int(d["seed"][ci]),
                               "c_plan": float(-d["plan_reward"][ci]), "c_rollout": float(-d["rollout_reward"][ci]),
                               "e_dyn": float(d["feasibility_mean_l2_future"][ci]), "png": f"{stem}.png"})
    legend_figure(fig_dir / "plan_plots" / "reference_plan_legend")
    safe_io.write_json(fig_dir / "plan_plots" / "plot_index.json", plot_index)

    fig_cost_vs_dyn(args.test_dir, MAIN + ABL, fig_dir / "unicycle_crollout_vs_edyn")
    fig_cost_vs_dyn(args.test_dir, MAIN, fig_dir / "unicycle_crollout_vs_edyn_main_only")
    fig_budget(args.test_dir, fig_dir / "unicycle_crollout_vs_nfe")
    fig_per_iter(args.test_dir, fig_dir / "unicycle_pnp_per_iteration")
    safe_io.write_json(fig_dir / "analysis_provenance.json", {"val_dir": str(args.val_dir), "test_dir": str(args.test_dir),
                                                              "selection": selection, "provenance": safe_io.provenance()})
    print("done", fig_dir, tbl_dir)


if __name__ == "__main__":
    main()
