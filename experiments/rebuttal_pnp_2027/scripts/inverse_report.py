#!/usr/bin/env python3
"""Table 3, validation-grid tables and figures for the sparse trajectory reconstruction experiment (CPU).

Reads a test (or smoke) run dir and optionally the validation run dir; writes
  tables/inverse/<prefix>_table3.{md,csv,tex}, <prefix>_table3_extra.md, <prefix>_paired.md,
  <prefix>_val_grids.{md,csv}
  figures/inverse/<prefix>/{recon_percentiles, recon_fixed_0-3, mse_vs_nfe, val_grids, pnp_iterations}.{png,pdf}
Never overwrites (safe_io); pick a new --prefix for a new report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inverse_common as ic  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402

METHODS = ["unguided", "pdg", "dcg", "pnp_pgd", "pnp_pgd_ddim10_K10"]
STYLE = {
    "unguided": dict(color=ic.METHOD_COLORS["unguided"], ls="-", marker="o"),
    "pdg": dict(color=ic.METHOD_COLORS["pdg"], ls="-", marker="s"),
    "dcg": dict(color=ic.METHOD_COLORS["dcg"], ls="-", marker="D"),
    "pnp_pgd": dict(color=ic.METHOD_COLORS["pnp_pgd"], ls="-", marker="^"),
    "pnp_pgd_ddim10_K10": dict(color=ic.METHOD_COLORS["pnp_pgd_ddim10_K10"], ls="--", marker="v"),
}
SHORT = {"unguided": "Unguided", "pdg": "PDG", "dcg": "DCG", "pnp_pgd": "PnP-PGD",
         "pnp_pgd_ddim10_K10": "PnP-PGD (DDIM-10 init, K=10)"}


def param_str(method, p, res=None):
    if method == "pdg":
        return f"w_cg={p['w_cg']:g}"
    if method == "dcg":
        return f"eta={p['eta']:g}"
    if method.startswith("pnp"):
        sb = f", sigma_bar={res['sigma_bar']:.3f}" if res and "sigma_bar" in res else ""
        return f"k={p['noise_index']}{sb}, eta={p['eta']:g}/L, K={p['K']}"
    return ""


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def boot_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def style():
    import unicycle_eval as ue

    ue._apply_paper_plot_style()
    plt.rcParams.update({"legend.fontsize": 11, "axes.labelsize": 13, "axes.titlesize": 12})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--val-run", default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--diag-run", default=None, help="optional validation-only long-K PnP diagnostic run dir")
    args = ap.parse_args()
    test_dir = Path(args.test_run)
    R = json.load(open(test_dir / "results.json"))
    cfg = json.load(open(test_dir / "config.json"))
    raw = np.load(test_dir / "raw.npz")
    n = int(raw["gt_obs"].shape[0])
    tab_dir = safe_io.ensure_dir(safe_io.TABLES_ROOT / "inverse")
    fig_dir = safe_io.ensure_dir(safe_io.FIGURES_ROOT / "inverse" / args.prefix)
    pre = args.prefix
    normalizer_mean = np.asarray(cfg["normalizer_mean"])
    normalizer_std = np.asarray(cfg["normalizer_std"])
    present = [m for m in METHODS if m in R]

    # ------------------------------------------------------------------ Table 3
    rows = []
    hdr = ["Method", "Hyper-parameters (val-selected)", "Unobs. xy MSE [m^2]", "Obs. xy RMSE [m]", "e_dyn [m]",
           "NFE (init)", "Grad evals", "Net backward", "Runtime/traj [ms] (init)"]
    gt = R["ground_truth"]["summary"]
    rows.append(["Ground truth", "-", fmt(gt["unobs_xy_mse"]["mean"], gt["unobs_xy_mse"]["std_ddof1"], 3),
                 fmt(gt["obs_xy_rmse"]["mean"], gt["obs_xy_rmse"]["std_ddof1"], 3),
                 f"{gt['e_dyn']['mean']:.1e}", "-", "-", "-", "-"])
    for m in present:
        r = R[m]
        s = r["summary"]
        c = r["cost"]
        meas = c["measured_per_sample"]
        init = c["analytic"]["init_nfe"]
        nfe = f"{meas['nfe']:g}" + (f" ({init})" if init else "")
        t = r["time_batch1_ms"]
        rt = f"{t['mean']:.1f} ± {t['std_ddof1']:.1f}" + (f" ({t['init_mean']:.1f})" if m.startswith("pnp") else "")
        rows.append([SHORT[m], param_str(m, r["params"], r), fmt(s["unobs_xy_mse"]["mean"], s["unobs_xy_mse"]["std_ddof1"]),
                     fmt(s["obs_xy_rmse"]["mean"], s["obs_xy_rmse"]["std_ddof1"]), fmt(s["e_dyn"]["mean"], s["e_dyn"]["std_ddof1"]),
                     nfe, f"{meas['grads']:g}", f"{meas['net_backward']:g}", rt])
    note = (f"n = {n} test trajectories; mean ± std (ddof=1) over trajectories. Physical units. NFE = denoiser network "
            f"forward evaluations per trajectory (measured by forward hook, all equal to the analytic count: "
            f"{all(R[m]['cost']['match'] for m in present)}); (init) = cost of the unguided DDIM initialization included "
            f"in the total. Grad evals = evaluations of grad f; Net backward = backward passes through the denoiser. "
            f"Runtime: batch size 1, {cfg['warmup_calls']} warm-up calls, torch.cuda.synchronize around each trajectory, "
            f"model loading and CPU e_dyn rollout excluded, GPU {cfg['gpu']}.")
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    safe_io.write_text(tab_dir / f"{pre}_table3.md", f"# Table 3: sparse trajectory reconstruction\n\n{md}\n\n{note}\n")
    # CSV with raw numbers
    csv = ["method,params,unobs_xy_mse_mean,unobs_xy_mse_std,obs_xy_rmse_mean,obs_xy_rmse_std,e_dyn_mean,e_dyn_std,"
           "unobs_obs4_mse_mean,unobs_obs4_mse_std,action_mse_mean,action_mse_std,f_norm_mean,f_norm_std,nfe,init_nfe,grads,"
           "net_backward,time_ms_mean,time_ms_std,time_init_ms_mean,time_batched_per_traj_ms,n"]
    for m in ["ground_truth"] + present:
        s = R[m]["summary"]
        vals = [s[k][q] for k in ("unobs_xy_mse", "obs_xy_rmse", "e_dyn", "unobs_obs4_mse", "action_mse", "f_norm") for q in ("mean", "std_ddof1")]
        if m == "ground_truth":
            extra = ["", "", "", "", "", "", "", ""]
            pstr = ""
        else:
            r = R[m]
            mm = r["cost"]["measured_per_sample"]
            extra = [mm["nfe"], r["cost"]["analytic"]["init_nfe"], mm["grads"], mm["net_backward"], r["time_batch1_ms"]["mean"],
                     r["time_batch1_ms"]["std_ddof1"], r["time_batch1_ms"]["init_mean"], r["time_batched_per_traj_ms"]]
            pstr = param_str(m, r["params"], r)
        csv.append(",".join([m, f"\"{pstr}\""] + [f"{v:.6g}" for v in vals] + [f"{v:.6g}" if v != "" else "" for v in extra] + [str(n)]))
    safe_io.write_text(tab_dir / f"{pre}_table3.csv", "\n".join(csv) + "\n")
    # LaTeX booktabs
    tex = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r"Method & Unobs.\ $xy$ MSE [m$^2$] & Obs.\ $xy$ RMSE [m] & $e_{\mathrm{dyn}}$ [m] & NFE & $\nabla f$ evals & Net bwd & Time/traj [ms] \\",
           r"\midrule"]
    gtt = R["ground_truth"]["summary"]
    tex.append(f"Ground truth & {gtt['unobs_xy_mse']['mean']:.3f} & {gtt['obs_xy_rmse']['mean']:.3f} & "
               f"{gtt['e_dyn']['mean']:.1e} & -- & -- & -- & -- \\\\")
    for m in present:
        r = R[m]
        s = r["summary"]
        mm = r["cost"]["measured_per_sample"]
        init = r["cost"]["analytic"]["init_nfe"]
        lab = {"unguided": "Unguided", "pdg": "PDG", "dcg": "DCG", "pnp_pgd": "PnP-PGD",
               "pnp_pgd_ddim10_K10": r"PnP-PGD (DDIM-10, $K{=}10$)"}[m]
        nfe = f"{mm['nfe']:g}" + (f" ({init})" if init else "")
        tex.append(f"{lab} & ${s['unobs_xy_mse']['mean']:.3f} \\pm {s['unobs_xy_mse']['std_ddof1']:.3f}$ & "
                   f"${s['obs_xy_rmse']['mean']:.3f} \\pm {s['obs_xy_rmse']['std_ddof1']:.3f}$ & "
                   f"${s['e_dyn']['mean']:.3f} \\pm {s['e_dyn']['std_ddof1']:.3f}$ & {nfe} & {mm['grads']:g} & {mm['net_backward']:g} & "
                   f"${r['time_batch1_ms']['mean']:.1f} \\pm {r['time_batch1_ms']['std_ddof1']:.1f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", "% " + note]
    safe_io.write_text(tab_dir / f"{pre}_table3.tex", "\n".join(tex) + "\n")

    # extra metrics + medians
    ex_hdr = ["Method", "Unobs. 4-dim obs MSE", "Action MSE", "f (normalized)", "Unobs. xy MSE median", "e_dyn (normalized)",
              "o_0 pin err [phys]", "batch-1 vs batched max|dx|", "batched time/traj [ms]"]
    ex = []
    for m in ["ground_truth"] + present:
        s = R[m]["summary"]
        r = R[m]
        ex.append([SHORT.get(m, "Ground truth"), fmt(s["unobs_obs4_mse"]["mean"], s["unobs_obs4_mse"]["std_ddof1"]),
                   fmt(s["action_mse"]["mean"], s["action_mse"]["std_ddof1"]), fmt(s["f_norm"]["mean"], s["f_norm"]["std_ddof1"]),
                   f"{s['unobs_xy_mse']['median']:.3f}", fmt(s["e_dyn_norm"]["mean"], s["e_dyn_norm"]["std_ddof1"]),
                   f"{s['pin_err_o0']['mean']:.1e}",
                   f"{r.get('batch1_vs_batched_x_maxabs', float('nan')):.1e}", f"{r.get('time_batched_per_traj_ms', float('nan')):.2f}"])
    safe_io.write_text(tab_dir / f"{pre}_table3_extra.md",
                       "# Table 3 (supplementary metrics)\n\n| " + " | ".join(ex_hdr) + " |\n|" + "---|" * len(ex_hdr) + "\n"
                       + "\n".join("| " + " | ".join(r) + " |" for r in ex)
                       + f"\n\nmean ± std (ddof=1), n={n}. Action MSE over t=0..63 of clipped actions. "
                         "Batched time = one batch of all trajectories / n (secondary convention).\n")
    # paired differences (per-trajectory, same x_T and masks)
    pr = ["# Paired per-trajectory differences of unobserved-xy MSE (row minus column method), mean and 95% bootstrap CI "
          "(10000 resamples of trajectories), and #trajectories where row is better\n",
          "| A | B | mean(A-B) [m^2] | 95% CI | #A<B of n |", "|---|---|---|---|---|"]
    pairs = [("pnp_pgd", "dcg"), ("pnp_pgd", "pdg"), ("pnp_pgd", "unguided"), ("dcg", "pdg"), ("dcg", "unguided"),
             ("pdg", "unguided"), ("pnp_pgd_ddim10_K10", "pnp_pgd"), ("pnp_pgd_ddim10_K10", "dcg")]
    for a, b in pairs:
        if a in present and b in present:
            d = raw[f"{a}__unobs_xy_mse"] - raw[f"{b}__unobs_xy_mse"]
            lo, hi = boot_ci(d)
            pr.append(f"| {SHORT[a]} | {SHORT[b]} | {d.mean():.4f} | [{lo:.4f}, {hi:.4f}] | {(d < 0).sum()} of {n} |")
    safe_io.write_text(tab_dir / f"{pre}_paired.md", "\n".join(pr) + "\n")

    # ------------------------------------------------------------------ figures
    style()
    U = raw["unguided__unobs_xy_mse"]
    order = np.argsort(U, kind="stable")
    pct_idx = [int(order[int(round(q * (n - 1)))]) for q in (0.25, 0.5, 0.75)]

    def xy_of(key):
        return raw[key][..., :2] * normalizer_std[:2] + normalizer_mean[:2]

    recon = {m: xy_of(f"{m}__x") for m in present}
    gt_xy = raw["gt_obs"][..., :2]

    def recon_fig(indices, titles, stem):
        fig, axes = plt.subplots(1, len(indices), figsize=(4.3 * len(indices), 4.6))
        axes = np.atleast_1d(axes)
        for ax, j, ttl in zip(axes, indices, titles):
            pts = [gt_xy[j]] + [recon[m][j] for m in present]
            xl, yl = ic.ue._centered_xy_limits(*pts, pad_frac=0.12)
            ax.plot(gt_xy[j, :, 0], gt_xy[j, :, 1], color="black", ls="--", lw=1.8, zorder=5)
            for m in present:
                st = STYLE[m]
                ax.plot(recon[m][j, :, 0], recon[m][j, :, 1], color=st["color"], ls=st["ls"],
                        lw=1.6 if m != "pnp_pgd_ddim10_K10" else 1.3, zorder=4, alpha=0.95)
            oi = raw["obs_idx"][j]
            ax.scatter(gt_xy[j, oi, 0], gt_xy[j, oi, 1], s=42, facecolor="white", edgecolor="black", lw=1.3, zorder=7)
            ax.scatter([gt_xy[j, 0, 0]], [gt_xy[j, 0, 1]], marker="*", s=230, color="gold", edgecolor="black", lw=0.9, zorder=8)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, alpha=0.35)
            ax.set_title(ttl, fontsize=10)
        handles = [Line2D([], [], color="black", ls="--", lw=1.8, label="Ground truth"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markersize=7,
                          label="Observed (x, y) (16 of 63)"),
                   Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=13, label="Start (pinned $o_0$)")]
        handles += [Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=1.8, label=SHORT[m]) for m in present]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        safe_io.save_figure(fig, fig_dir / stem)
        plt.close(fig)

    def ttl(j, lab):
        ab = {"unguided": "Ung.", "pdg": "PDG", "dcg": "DCG", "pnp_pgd": "PnP"}
        it = [f"{ab[m]} {raw[f'{m}__unobs_xy_mse'][j]:.2f}" for m in ab if m in present]
        return f"{lab}, test #{j}\nunobs. MSE [m$^2$]: " + " | ".join(it[:2]) + "\n" + " | ".join(it[2:])

    recon_fig(pct_idx, [ttl(j, f"{q}th pct. of unguided error") for j, q in zip(pct_idx, (25, 50, 75))], "recon_percentiles")
    fixed = [j for j in range(4) if j < n]
    recon_fig(fixed, [ttl(j, "fixed index") for j in fixed], "recon_fixed_0-3")

    # MSE vs NFE
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    dx = {"unguided": -1.2, "pdg": -0.4, "dcg": 0.4, "pnp_pgd": 0.0, "pnp_pgd_ddim10_K10": 1.2}
    for m in present:
        v = raw[f"{m}__unobs_xy_mse"]
        nfe = R[m]["cost"]["measured_per_sample"]["nfe"] + dx[m]
        se = v.std(ddof=1) / np.sqrt(len(v))
        st = STYLE[m]
        ax.errorbar([nfe], [v.mean()], yerr=[[1.96 * se], [1.96 * se]], fmt=st["marker"], color=st["color"], ms=9,
                    capsize=4, lw=1.5, mfc=st["color"] if m != "pnp_pgd_ddim10_K10" else "white", mew=1.6, label=SHORT[m])
    nfes = sorted({R[m]["cost"]["measured_per_sample"]["nfe"] for m in present})
    ax.set_xticks(nfes)
    ax.set_xlim(min(nfes) - 4, max(nfes) + 4)
    ax.set_xlabel("Denoiser NFE per trajectory (incl. initialization;\npoints at equal NFE offset horizontally)")
    ax.set_ylabel("Unobserved-$xy$ MSE [m$^2$]")
    ax.grid(True, alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.set_title(f"Test set (n={n}); mean ± 1.96 SE")
    fig.tight_layout()
    safe_io.save_figure(fig, fig_dir / "mse_vs_nfe")
    plt.close(fig)

    # PnP iterations (test, batched instrumented run)
    pk = [m for m in ("pnp_pgd", "pnp_pgd_ddim10_K10") if f"{m}__iter_unobs_xy_mse_batched" in raw.files]
    if pk:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        for m in pk:
            c = raw[f"{m}__iter_unobs_xy_mse_batched"].mean(1)
            init = R[m]["cost"]["analytic"]["init_nfe"]
            ax.plot(init + np.arange(len(c)), c, color=STYLE[m]["color"], ls=STYLE[m]["ls"], marker=".", label=SHORT[m])
        for m in ("unguided", "pdg", "dcg"):
            if m in present:
                ax.axhline(raw[f"{m}__unobs_xy_mse"].mean(), color=STYLE[m]["color"], lw=1.2, ls=":", label=f"{SHORT[m]} (20 NFE)")
        ax.set_xlabel("Denoiser NFE (initialization + PnP iterations)")
        ax.set_ylabel("Unobserved-$xy$ MSE [m$^2$]")
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("PnP-PGD iterates on the test set (batched run)")
        fig.tight_layout()
        safe_io.save_figure(fig, fig_dir / "pnp_iterations")
        plt.close(fig)

    # ------------------------------------------------------------------ validation grids
    if args.val_run:
        V = json.load(open(Path(args.val_run) / "val_grid.json"))["grids"]
        lines = ["# Validation grids (n=20 validation trajectories; criterion = mean unobserved-xy MSE [m^2]; "
                 "selected entry in bold; obs RMSE and e_dyn shown for information only)\n"]
        csvl = ["method,noise_index,sigma_bar,eta_or_wcg,K,nfe,grads,val_unobs_xy_mse_mean,val_unobs_xy_mse_std,val_obs_xy_rmse_mean,val_e_dyn_mean,selected"]
        u = V["unguided"][0]
        lines.append(f"Unguided reference: {u['val_mean_unobs_xy_mse']:.4f}\n")
        for m in ("pdg", "dcg", "pnp_pgd", "pnp_pgd_ddim10_K10"):
            g = V[m]
            lines.append(f"\n## {SHORT[m]}\n\nselected: `{g['selected']}`; extension: `{g.get('extension')}`; "
                         + (f"selected on boundary after extension: {g.get('selected_on_boundary_after_extension')}"
                            if not m.startswith("pnp") else f"final boundary flags: {g.get('selected_on_boundary_final')}, "
                            f"K at grid max: {g.get('K_selected_on_grid_max')}") + "\n")
            lines.append("| params | sigma_bar | NFE | grads | val unobs-xy MSE | val obs-xy RMSE | val e_dyn |\n|---|---|---|---|---|---|---|")
            for e in g["entries"]:
                isel = e["params"] == g["selected"]
                s = e["summary"]
                ps = param_str(m, e["params"])
                sb = f"{e['sigma_bar']:.4f}" if "sigma_bar" in e else "-"
                cell = f"{s['unobs_xy_mse']['mean']:.4f} ± {s['unobs_xy_mse']['std_ddof1']:.4f}"
                lines.append(f"| {ps} | {sb} | {e['cost']['nfe']} | {e['cost']['grads']} | {'**' + cell + '**' if isel else cell} | "
                             f"{s['obs_xy_rmse']['mean']:.4f} | {s['e_dyn']['mean']:.4f} |")
                p = e["params"]
                csvl.append(",".join(str(x) for x in [m, p.get("noise_index", ""), e.get("sigma_bar", ""),
                                                      p.get("eta", p.get("w_cg", "")), p.get("K", ""), e["cost"]["nfe"], e["cost"]["grads"],
                                                      f"{s['unobs_xy_mse']['mean']:.6g}", f"{s['unobs_xy_mse']['std_ddof1']:.6g}",
                                                      f"{s['obs_xy_rmse']['mean']:.6g}", f"{s['e_dyn']['mean']:.6g}", int(isel)]))
        safe_io.write_text(tab_dir / f"{pre}_val_grids.md", "\n".join(lines) + "\n")
        safe_io.write_text(tab_dir / f"{pre}_val_grids.csv", "\n".join(csvl) + "\n")

        # figure
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
        for ax, m, key, lab in ((axes[0], "pdg", "w_cg", "$w_{cg}$"), (axes[1], "dcg", "eta", r"DCG step $\eta$")):
            es = sorted(V[m]["entries"], key=lambda e: e["params"][key])
            xv = [e["params"][key] for e in es]
            yv = [e["val_mean_unobs_xy_mse"] for e in es]
            ax.plot(xv, yv, color=STYLE[m]["color"], marker=STYLE[m]["marker"], lw=1.6)
            sel = V[m]["selected"][key]
            ax.plot([sel], [V[m]["selected_val_mean_unobs_xy_mse"]], marker="o", ms=14, mfc="none", mec="black", mew=1.5)
            ax.axhline(u["val_mean_unobs_xy_mse"], color=ic.METHOD_COLORS["unguided"], ls=":", lw=1.2, label="Unguided")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(lab)
            ax.set_ylabel("Val. unobserved-$xy$ MSE [m$^2$]")
            ax.set_title(SHORT[m])
            ax.grid(True, alpha=0.35, which="both")
            ax.legend(frameon=False, fontsize=9)
        for ax, m in ((axes[2], "pnp_pgd"), (axes[3], "pnp_pgd_ddim10_K10")):
            es = V[m]["entries"]
            ks = sorted({e["params"]["noise_index"] for e in es})
            Ks = sorted({e["params"]["K"] for e in es})
            lss = {K: ls for K, ls in zip(Ks, ["-", "--", ":"] if len(Ks) > 1 else ["-"])}
            cmap = plt.get_cmap("RdPu")
            for i, k in enumerate(ks):
                col = cmap(0.35 + 0.6 * i / max(len(ks) - 1, 1))
                for K in Ks:
                    sub = sorted([e for e in es if e["params"]["noise_index"] == k and e["params"]["K"] == K], key=lambda e: e["params"]["eta"])
                    sbar = sub[0].get("sigma_bar", float("nan"))
                    ax.plot([e["params"]["eta"] for e in sub], [e["val_mean_unobs_xy_mse"] for e in sub], color=col, ls=lss[K],
                            marker="^", ms=5, lw=1.4, label=f"k={k} ($\\bar\\sigma$={sbar:.2f})" if K == Ks[0] else None)
            for K in Ks if len(Ks) > 1 else []:
                ax.plot([], [], color="gray", ls=lss[K], label=f"K={K}")
            s = V[m]["selected"]
            ax.plot([s["eta"]], [V[m]["selected_val_mean_unobs_xy_mse"]], marker="o", ms=14, mfc="none", mec="black", mew=1.5)
            ax.axhline(u["val_mean_unobs_xy_mse"], color=ic.METHOD_COLORS["unguided"], ls=":", lw=1.2)
            ax.set_xscale("log", base=2)
            ax.set_xlabel(r"PnP step $\eta$ [$1/L$]")
            ax.set_ylabel("Val. unobserved-$xy$ MSE [m$^2$]")
            ax.set_title(SHORT[m])
            ax.grid(True, alpha=0.35, which="both")
            ax.legend(frameon=False, fontsize=8, ncol=1)
        fig.tight_layout()
        safe_io.save_figure(fig, fig_dir / "val_grids")
        plt.close(fig)

    if args.diag_run:
        Dg = json.load(open(Path(args.diag_run) / "diag_summary.json"))
        draw = np.load(Path(args.diag_run) / "diag_raw.npz")
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        keys = sorted(Dg["summary"].keys())
        cmap = plt.get_cmap("RdPu")
        ks = sorted({int(k.split("_")[0][1:]) for k in keys})
        for key in keys:
            k = int(key.split("_")[0][1:])
            eta = float(key.split("eta")[1])
            c = draw[f"{key}__iter_unobs_xy_mse"].mean(1)
            col = cmap(0.4 + 0.55 * ks.index(k) / max(len(ks) - 1, 1))
            ax.plot(np.arange(len(c)), c, color=col, ls="-" if eta == max(float(q.split("eta")[1]) for q in keys) else "--", lw=1.5,
                    label=f"k={k} ($\\bar\\sigma$={Dg['summary'][key]['sigma_bar']:.2f}), $\\eta$={eta:g}/L")
        if args.val_run:
            ax.axhline(V["dcg"]["selected_val_mean_unobs_xy_mse"], color=ic.METHOD_COLORS["dcg"], ls=":", lw=1.4, label="DCG (val-selected)")
            ax.axhline(V["pdg"]["selected_val_mean_unobs_xy_mse"], color=ic.METHOD_COLORS["pdg"], ls=":", lw=1.4, label="PDG (val-selected)")
            ax.axvline(20, color="black", ls="--", lw=1.0, label="pre-registered K grid max (20)")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, Dg["args"]["kmax"])
        ax.set_xlabel("PnP-PGD iteration K (after 20-NFE DDIM init)")
        ax.set_ylabel("Val. unobserved-$xy$ MSE [m$^2$]")
        ax.grid(True, alpha=0.35, which="both")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("DIAGNOSTIC (validation only, not used for selection)", fontsize=10)
        fig.tight_layout()
        safe_io.save_figure(fig, fig_dir / "val_longK_diagnostic")
        plt.close(fig)

    safe_io.write_json(fig_dir / "figure_manifest.json", {
        "test_run": str(test_dir), "val_run": args.val_run, "percentile_rule": "25/50/75th pct of unguided unobs-xy MSE, "
        "nearest rank order[round(q*(n-1))]", "percentile_indices": pct_idx, "fixed_indices": fixed,
        "colors": {m: STYLE[m]["color"] for m in present}, "provenance": safe_io.provenance()})
    print("tables ->", tab_dir, "figures ->", fig_dir, "percentile idx", pct_idx)


if __name__ == "__main__":
    main()
