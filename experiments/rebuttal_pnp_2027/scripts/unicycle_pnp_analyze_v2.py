#!/usr/bin/env python
"""CPU-only analysis v2 for the unicycle PnP-PGD rebuttal experiment (reviewer fixes).

Inputs (read only): validation run dir (val_3565048), test run dir (test_3565048), supplementary run dir
(supp_<J>: K<=20 post-selection reporting pass + timing diagnostic). Outputs (new files only, via safe_io):
  tables/unicycle/<tag>_table2.{md,csv,tex}          Table 2 v2: published values for existing methods (they remain
                                                     the reference; reproduced values in the md/csv), PnP-PGD (main,
                                                     spec primary variant) AND PnP-PGD init x_T (ablation b, selected
                                                     on validation with the same criterion), costs, runtime
  tables/unicycle/<tag>_table2_reproduced.tex        same, with OUR re-run numbers for the existing methods + footnote
  tables/unicycle/<tag>_table_ablations.{md,csv,tex}
  tables/unicycle/<tag>_paired_comparison.{md,csv}   per-reference paired comparison (median, win rate, paired bootstrap CI)
  tables/unicycle/<tag>_validation_vs_K.{md,csv}     validation metrics vs K (0..20) at the selected (k, eta) (descriptive)
  tables/unicycle/<tag>_test_vs_K_post_selection.{md,csv}  test metrics vs K (0..20), post-selection reporting only
  tables/unicycle/<tag>_timing_diagnostic.{md,csv}
  figures/unicycle/<tag>/...                         v2 figures (units fixed, ablation (b) visible, K<=20 curves)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unicycle_pnp_analyze as A  # noqa: E402  (v1 helpers; pure numpy/matplotlib)
from rebuttal_pnp import safe_io  # noqa: E402

rp = A.rp
TEST_REFS, REF_LABEL, VAL_REFS = A.TEST_REFS, A.REF_LABEL, A.VAL_REFS
EXISTING = ["monte_carlo", "pdg", "dcg", "drgd"]
MAIN_V2 = EXISTING + ["pnp_pgd", "pnp_pgd_from_xT"]
PNP = ["pnp_pgd", "pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale"]
LABEL = dict(A.LABEL)
LABEL.update({"pnp_pgd": "PnP-PGD (init: Monte Carlo)", "pnp_pgd_from_xT": r"PnP-PGD (init: $x_T$; abl. b)",
              "pnp_pgd_budget20": "PnP-PGD (a) budget-matched", "pnp_pgd_noscale": "PnP-PGD (c) no input scaling"})
TEX_LABEL = dict(A.TEX_LABEL)
TEX_LABEL.update({"pnp_pgd": "PnP-PGD (init: Monte Carlo)", "pnp_pgd_from_xT": r"PnP-PGD (init: $x_T$)$^{\ddagger}$"})
# Opaque colours; ablations are lighter variants of #E7298A, with dashed error bars / dashed or dotted lines.
COLOR = {"monte_carlo": "#7F7F7F", "pdg": "#2166AC", "dcg": "#E66101", "drgd": "#984EA3", "pnp_pgd": "#E7298A",
         "pnp_pgd_budget20": "#F4A6CD", "pnp_pgd_from_xT": "#EF6DB0", "pnp_pgd_noscale": "#F7C6DF"}
EDGE = {"pnp_pgd_budget20": "#9E1B5E", "pnp_pgd_from_xT": "#7A0F47", "pnp_pgd_noscale": "#9E1B5E"}
MARKER, LS = A.MARKER, A.LS
EDYN_LABEL = r"dynamics error $e_{\mathrm{dyn}}$ (mean over $t\geq1$ of $\|\hat o_t-o_t\|_2$, $o=(x,y,\cos\theta,\sin\theta)$)"
EDYN_SHORT = r"dynamics error $e_{\mathrm{dyn}}$"


def metrics(npz):
    return {"c_plan": -npz["plan_reward"], "c_rollout": -npz["rollout_reward"], "e_dyn": npz["feasibility_mean_l2_future"]}


def paired_boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bm = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def runtime_parts(test_dir, ref, m):
    c = A.load(test_dir, ref, m)["costs"]["summary"]
    out = {"measured_ms": 1e3 * c["seconds"]["mean"], "measured_std_ms": 1e3 * c["seconds"]["std"]}
    if m.startswith("pnp"):
        out["init_ms"] = 1e3 * c["init_seconds"]["mean"]
        out["iter_ms"] = 1e3 * c["iter_seconds"]["mean"]
        spec = A.load(test_dir, ref, m)["summary"]["spec"]
        if spec["init"] == "mc20":  # identical computation to the Monte Carlo baseline
            mc = A.load(test_dir, ref, "monte_carlo")["costs"]["summary"]["seconds"]["mean"]
            out["alt_ms"] = 1e3 * mc + out["iter_ms"]
    return out


def table2(test_dir, paper, methods, tag, stem):
    hdr = ["Setting", "Algorithm", "c_plan", "c_rollout", "e_dyn", "NFE (init)", "Net bwd", "Grad evals",
           "Runtime/sample measured (ms)", "Runtime alt: MC sampling + PnP iters (ms)", "Note"]
    rows = []
    for ref in TEST_REFS:
        for m in methods:
            d = A.load(test_dir, ref, m)
            s, c = d["summary"]["metrics"], d["costs"]["summary"]
            rt = runtime_parts(test_dir, ref, m)
            nfe = f"{c['nfe']['mean']:.0f}" + (f" ({c['init_nfe']['mean']:.0f})" if m.startswith("pnp") else "")
            note = ""
            if m in EXISTING:
                p = paper[(ref, m)]
                same = all(f"{s[k]['mean']:.4f}" == f"{p[k][0]:.4f}" and f"{s[k]['std']:.4f}" == f"{p[k][1]:.4f}" for k in p)
                note = ("re-run identical to published at 4 dp" if same else
                        "re-run (DRGD, non-deterministic backward) differs from published: published "
                        + "; ".join(f"{k} {p[k][0]:.4f} ± {p[k][1]:.4f}" for k in ("c_plan", "c_rollout", "e_dyn")))
            elif m == "pnp_pgd":
                sp = d["summary"]["spec"]
                note = f"spec primary variant; frozen k={sp['k']}, eta={sp['eta_mult']}/L, K={sp['K']}"
            elif m == "pnp_pgd_from_xT":
                sp = d["summary"]["spec"]
                note = (f"ablation (b), selected on validation with the same criterion (best PnP variant on validation: "
                        f"0.2995 vs 0.6198); frozen k={sp['k']}, eta={sp['eta_mult']}/L, K={sp['K']}")
            rows.append([REF_LABEL[ref], LABEL[m].replace("$", "").replace("\\gamma", "gamma"),
                         *[A.fmt(s[k]["mean"], s[k]["std"]) for k in ("c_plan", "c_rollout", "e_dyn")],
                         nfe, f"{c['net_backward']['mean']:.0f}", f"{c['grad_evals']['mean']:.0f}",
                         f"{rt['measured_ms']:.1f} ± {rt['measured_std_ms']:.1f}" + (
                             f" (init {rt['init_ms']:.1f} + iters {rt['iter_ms']:.1f})" if "init_ms" in rt else ""),
                         f"{rt['alt_ms']:.1f}" if "alt_ms" in rt else "", note])
    A.write_tables(tag, rows, hdr, stem)


def tex2(test_dir, paper, methods, use_published: bool, extra_note: str):
    lines = [r"\resizebox{\linewidth}{!}{", r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Setting & Algorithm & $c_{\mathrm{plan}}$ & $c_{\mathrm{rollout}}$ & $e_{\mathrm{dyn}}$ & NFE & Net bwd & Grad evals & Runtime (ms) \\",
             r"\midrule"]
    for ri, ref in enumerate(TEST_REFS):
        for mi, m in enumerate(methods):
            d = A.load(test_dir, ref, m)
            s, c = d["summary"]["metrics"], d["costs"]["summary"]
            if m in EXISTING and use_published:
                vals = [paper[(ref, m)][k] for k in ("c_plan", "c_rollout", "e_dyn")]
            else:
                vals = [(s[k]["mean"], s[k]["std"]) for k in ("c_plan", "c_rollout", "e_dyn")]
            lab = TEX_LABEL[m] + (r"$^{\dagger}$" if (m == "drgd" and not use_published) else "")
            first = rf"\multirow{{{len(methods)}}}{{*}}{{{REF_LABEL[ref]}}}" if mi == 0 else ""
            nfe = f"{c['nfe']['mean']:.0f}"
            if m.startswith("pnp") and c["init_nfe"]["mean"] > 0:
                nfe = f"{c['iter_nfe']['mean']:.0f}+{c['init_nfe']['mean']:.0f}"
            lines.append(f"{first} & {lab} & " + " & ".join(f"${a:.4f} \\pm {b:.4f}$" for a, b in vals)
                         + f" & {nfe} & {c['net_backward']['mean']:.0f} & {c['grad_evals']['mean']:.0f}"
                         + f" & ${1e3 * c['seconds']['mean']:.1f}$ \\\\")
        lines.append(r"\midrule" if ri < len(TEST_REFS) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", "}"]
    lines += ["% Notes (put in caption/footnote):", *[f"% {l}" for l in extra_note.strip().splitlines()]]
    return "\n".join(lines) + "\n"


def paired_table(test_dir, tag):
    hdr = ["Setting", "PnP variant", "vs", "median c_rollout PnP", "median c_rollout other", "win rate PnP (c_rollout lower)",
           "mean diff c_rollout (PnP - other) [95% paired bootstrap CI]", "median e_dyn PnP", "median e_dyn other"]
    rows, js = [], {}
    for ref in TEST_REFS:
        for p in ["pnp_pgd", "pnp_pgd_from_xT", "pnp_pgd_budget20", "pnp_pgd_noscale"]:
            mp = metrics(A.load(test_dir, ref, p)["npz"])
            for o in ["dcg", "drgd", "pdg", "monte_carlo"]:
                mo = metrics(A.load(test_dir, ref, o)["npz"])
                md, lo, hi = paired_boot(mp["c_rollout"], mo["c_rollout"])
                win = float((mp["c_rollout"] < mo["c_rollout"]).mean())
                rows.append([REF_LABEL[ref], LABEL[p].replace("$", ""), LABEL[o].replace("$", "").replace("\\gamma", "gamma"),
                             f"{np.median(mp['c_rollout']):.4f}", f"{np.median(mo['c_rollout']):.4f}", f"{100 * win:.0f}%",
                             f"{md:+.4f} [{lo:+.4f}, {hi:+.4f}]", f"{np.median(mp['e_dyn']):.4f}", f"{np.median(mo['e_dyn']):.4f}"])
                js[f"{ref}/{p}/vs/{o}"] = {"median_pnp": float(np.median(mp["c_rollout"])), "median_other": float(np.median(mo["c_rollout"])),
                                          "win_rate": win, "mean_diff": md, "ci95": [lo, hi],
                                          "median_e_dyn_pnp": float(np.median(mp["e_dyn"])), "median_e_dyn_other": float(np.median(mo["e_dyn"]))}
    A.write_tables(tag, rows, hdr, "paired_comparison")
    safe_io.write_json(A.TABLES_DIR / f"{tag}_paired_comparison.json",
                       {"note": "paired over the 100 test candidates (same seed / x_T); bootstrap 10000 resamples, rng seed 0", **js})
    return js


def val_vs_K(val_dir, tag):
    hdr = ["variant", "k", "eta", "K", "in pre-specified K grid", "avg val c_rollout", "avg val c_plan", "avg val e_dyn", "note"]
    rows = []
    for v in ("mc", "xT"):
        j = json.load(open(val_dir / f"validation_{v}.json"))
        sel = j["selected"]
        rr = sorted([r for r in j["rows_all_iterations"] if r["k"] == sel["k"] and r["eta_mult"] == sel["eta_mult"]], key=lambda r: r["K"])
        best = min(rr, key=lambda r: r["avg/c_rollout"])
        for r in rr:
            note = []
            if r["K"] == sel["K"]:
                note.append("SELECTED (grid K in {5,10,20})")
            if r["K"] == best["K"]:
                note.append("min over all K (descriptive, not used)")
            rows.append([v, r["k"], f"{r['eta_mult']}/L", r["K"], "yes" if r["K"] in (5, 10, 20) else "no",
                         f"{r['avg/c_rollout']:.4f}", f"{r['avg/c_plan']:.4f}", f"{r['avg/e_dyn']:.4f}", "; ".join(note)])
    A.write_tables(tag, rows, hdr, "validation_vs_K")


def test_vs_K(supp_dir, test_dir, tag):
    hdr = ["Setting", "variant", "K", "NFE total", "c_plan", "c_rollout", "e_dyn", "note"]
    rows, curves = [], {}
    for ref in TEST_REFS:
        for v in PNP:
            it = dict(np.load(supp_dir / "partA_K20" / ref / v / "per_iteration_K20.npz"))
            ks = json.load(open(supp_dir / "partA_K20" / ref / v / "summary.json"))
            Ks = ks["frozen_K_star"]
            curves[(ref, v)] = (it, Ks)
            for K in range(it["c_rollout"].shape[1]):
                rows.append([REF_LABEL[ref], LABEL[v].replace("$", ""), K, int(it["init_nfe"][0]) + K,
                             *[A.fmt(it[k][:, K].mean(), it[k][:, K].std(ddof=1)) for k in ("c_plan", "c_rollout", "e_dyn")],
                             "frozen K* (Table 2)" if K == Ks else ""])
    A.write_tables(tag, rows, hdr, "test_vs_K_post_selection")
    return curves


def timing_table(supp_dir, test_dir, tag):
    js = json.load(open(supp_dir / "supp_summary.json"))["partB_timing_summary"]
    hdr = ["Setting", "loop protocol", "MC / init sampling (ms, mean ± std)", "median (ms)", "PnP iterations (ms)",
           "test_3565048 MC (ms)", "test_3565048 PnP init (ms)"]
    rows = []
    for ref in TEST_REFS:
        mc_t = A.load(test_dir, ref, "monte_carlo")["costs"]["summary"]["seconds"]["mean"] * 1e3
        pi_t = A.load(test_dir, ref, "pnp_pgd")["costs"]["summary"]["init_seconds"]["mean"] * 1e3
        for p in ["mc_only", "init_only", "init_plus_iter", "test_protocol", "test_protocol_nogc", "mc_only_repeat"]:
            s = js[f"{ref}/{p}"]
            k = "sample_s" if "sample_s" in s else "init_s"
            rows.append([REF_LABEL[ref], p, f"{s[k]['mean_ms']:.2f} ± {s[k]['std_ms']:.2f}", f"{s[k]['median_ms']:.2f}",
                         f"{s['iter_s']['mean_ms']:.2f}" if "iter_s" in s else "", f"{mc_t:.2f}", f"{pi_t:.2f}"])
    A.write_tables(tag, rows, hdr, "timing_diagnostic")
    return js


# ------------------------------------------------------------------ figures
def _err(ax, x, xs, y, ys, m, abl, label):
    lo = ys if y - ys > 0 else 0.0
    eb = ax.errorbar(x, y, xerr=xs, yerr=[[lo], [ys]], fmt=MARKER[m], ms=9 if not abl else 8, color=COLOR[m],
                     mec=EDGE.get(m, "white"), mew=1.0 if abl else 0.8, capsize=3, elinewidth=1.4, zorder=4 if not abl else 3,
                     label=label)
    if abl:
        for line in eb[2]:
            line.set_linestyle("--")


def fig_cost_vs_dyn(test_dir, methods, stem):
    A.fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    for ax, ref in zip(axes, TEST_REFS):
        for m in methods:
            mm = metrics(A.load(test_dir, ref, m)["npz"])
            abl = m in ("pnp_pgd_budget20", "pnp_pgd_from_xT", "pnp_pgd_noscale")
            ax.scatter(mm["e_dyn"], mm["c_rollout"], s=6, color=COLOR[m], alpha=0.12, linewidths=0, zorder=1)
            (em, es), (cm, cs) = A.ms(mm["e_dyn"]), A.ms(mm["c_rollout"])
            _err(ax, em, es, cm, cs, m, abl, LABEL[m])
        ax.set_yscale("log")
        ax.set_xlabel(EDYN_SHORT)
        ax.set_title(REF_LABEL[ref])
    axes[0].set_ylabel(r"execution cost $c_{\mathrm{rollout}}$ (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.17, "mean ± std over 100 test candidates (lower whisker omitted where mean − std ≤ 0); faint dots: "
             r"individual candidates. $e_{\mathrm{dyn}}$ = mean over $t\geq1$ of the $L_2$ gap between planned and executed "
             r"$(x, y, \cos\theta, \sin\theta)$.", ha="center", fontsize=8.5)
    fig.tight_layout()
    safe_io.save_figure(fig, stem)
    plt.close(fig)


def fig_budget(test_dir, curves, stem):
    A.fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    for ax, ref in zip(axes, TEST_REFS):
        for m in EXISTING:
            d = A.load(test_dir, ref, m)
            cr = -d["npz"]["rollout_reward"]
            nfe = d["costs"]["summary"]["nfe"]["mean"]
            sd_ = cr.std(ddof=1)
            ax.errorbar(nfe, cr.mean(), yerr=[[sd_ if cr.mean() - sd_ > 0 else 0.0], [sd_]], fmt=MARKER[m], ms=8,
                        color=COLOR[m], mec="white", mew=0.8, capsize=3, zorder=5,
                        label=LABEL[m] + (" (+20 net bwd)" if m == "drgd" else ""))
        for m in PNP:
            it, Ks = curves[(ref, m)]
            c = it["c_rollout"]
            nfe = it["init_nfe"][0] + np.arange(c.shape[1])
            mean = c.mean(0)
            q25, q75 = np.quantile(c, 0.25, axis=0), np.quantile(c, 0.75, axis=0)
            ax.plot(nfe, mean, LS[m], color=COLOR[m], lw=2.2 if m == "pnp_pgd" else 1.7, zorder=3,
                    label=LABEL[m] + r", iterates $K=0..20$")
            if m in ("pnp_pgd", "pnp_pgd_from_xT"):
                ax.fill_between(nfe, q25, q75, color=COLOR[m], alpha=0.12, lw=0)
            ax.plot(nfe[Ks], mean[Ks], MARKER[m], color=COLOR[m], ms=9, mec=EDGE.get(m, "white"), mew=1.0, zorder=6)
        ax.set_yscale("log")
        ax.set_xlabel("denoiser NFE per candidate (incl. initialization)")
        ax.set_title(REF_LABEL[ref])
    axes[0].set_ylabel(r"execution cost $c_{\mathrm{rollout}}$ (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.text(0.5, -0.19, r"PnP-PGD curves: iterates of the frozen (k, $\eta$) up to K=20 (post-selection reporting pass on test, "
             r"not used for selection); large marker = frozen K* used in Table 2. Lines: mean; bands: interquartile range; baselines: mean ± std.",
             ha="center", fontsize=8.5)
    fig.tight_layout()
    safe_io.save_figure(fig, stem)
    plt.close(fig)


def fig_per_iter(test_dir, curves, stem):
    A.fig_style()
    mets = [("c_plan", r"plan cost $c_{\mathrm{plan}}$"), ("c_rollout", r"execution cost $c_{\mathrm{rollout}}$"),
            ("e_dyn", EDYN_SHORT)]
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 10.5), sharex="col")
    for j, ref in enumerate(TEST_REFS):
        for i, (key, lab) in enumerate(mets):
            ax = axes[i, j]
            for m in PNP:
                it, Ks = curves[(ref, m)]
                v = it[key]
                K = np.arange(v.shape[1])
                ax.plot(K, v.mean(0), LS[m], color=COLOR[m], lw=2.2 if m == "pnp_pgd" else 1.6, label=LABEL[m])
                if m in ("pnp_pgd", "pnp_pgd_from_xT"):
                    ax.fill_between(K, np.quantile(v, 0.25, axis=0), np.quantile(v, 0.75, axis=0), color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(Ks, v.mean(0)[Ks], MARKER[m], color=COLOR[m], ms=8, mec=EDGE.get(m, "white"), mew=1.0)
            for m in ["dcg", "drgd", "monte_carlo"]:
                val = metrics(A.load(test_dir, ref, m)["npz"])[key]
                ax.axhline(val.mean(), color=COLOR[m], lw=1.1, ls=(0, (4, 2)), label=LABEL[m] + " (final)")
            if key != "e_dyn":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0.0, 3.6)
                ax.annotate(r"init $x_T$ at K=0: %.1f (off scale)" % curves[(ref, "pnp_pgd_from_xT")][0]["e_dyn"][:, 0].mean(),
                            xy=(0.3, 3.5), fontsize=8, color=EDGE["pnp_pgd_from_xT"], va="top")
            if i == 0:
                ax.set_title(REF_LABEL[ref])
            if i == 2:
                ax.set_xlabel("PnP-PGD iteration K")
            if j == 0:
                ax.set_ylabel(lab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, -0.07, r"Frozen (k, $\eta$); iterates up to K=20 are a post-selection reporting pass on test (not used for "
             r"selection); markers = frozen K*; lines: mean, bands: interquartile range. With $\eta=2/L$ the gradient step is an exact reflection through the "
             r"target on the tracked coordinates, hence the period-2 oscillation.", ha="center", fontsize=8.5)
    fig.tight_layout()
    safe_io.save_figure(fig, stem)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True, type=Path)
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--supp-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag
    paper = A.parse_paper_table()
    fig_dir = safe_io.ensure_dir(safe_io.FIGURES_ROOT / "unicycle" / tag)
    safe_io.ensure_dir(A.TABLES_DIR)

    table2(args.test_dir, paper, MAIN_V2, tag, "table2")
    note = (
        "Monte Carlo, PDG, DCG: our re-run is bit-identical per candidate to the published runs. "
        "DRGD: values are the published ones; our re-run differs in the 4th digit because DRGD's backward/JVP path is "
        "non-deterministic on CUDA (two runs in the same job differ by up to 4.2e-3 in planned obs).\n"
        "PnP-PGD NFE = K iterations + init (unguided DDIM-20 plan); 0 network backward passes. Runtime: ms per sample, n=1 "
        "sequential, one GH200 job, 3 warm-up candidates, synchronized, excluding rollout/metrics. PnP-PGD (init: Monte Carlo) runtime = measured init + iterations; its init "
        "(the identical computation to Monte Carlo) was timed 5.2-9.4 ms slower than the Monte Carlo row in that job, so "
        "Monte Carlo sampling + PnP iterations gives 155.7 / 156.9 / 156.6 ms instead of 165.1 / 162.0 / 162.1 ms; a separate "
        "timing-diagnostic job (gh143) reproduced only a 1-4 ms loop-context gap, cause unresolved.\n"
        "ddagger: ablation (b), PnP-PGD started from the Gaussian x_T (no unguided init), selected on validation with the same "
        "criterion and grid; best PnP variant on validation.")
    safe_io.write_text(A.TABLES_DIR / f"{tag}_table2.tex", tex2(args.test_dir, paper, MAIN_V2, True, note))
    note_r = note.replace("DRGD: values are the published ones; our re-run differs",
                          "dagger: DRGD values are OUR re-run (published: Circle e_dyn std 0.0557; RT1 c_rollout 0.1259, "
                          "e_dyn 0.4209; RT2 0.5722±0.0790 / 1.2894±0.4768 / 0.8985±0.2637); it differs")
    safe_io.write_text(A.TABLES_DIR / f"{tag}_table2_reproduced.tex", tex2(args.test_dir, paper, MAIN_V2, False, note_r))
    table2(args.test_dir, paper, PNP, tag, "table_ablations")
    safe_io.write_text(A.TABLES_DIR / f"{tag}_table_ablations.tex",
                       tex2(args.test_dir, paper, PNP, False, "PnP-PGD variants on test; (a) DDIM-10 init + K=10 (20 NFE); "
                            "(b) init x_T (separately selected on validation); (c) input_scaling='none'."))
    paired = paired_table(args.test_dir, tag)
    val_vs_K(args.val_dir, tag)
    curves = test_vs_K(args.supp_dir, args.test_dir, tag)
    timing = timing_table(args.supp_dir, args.test_dir, tag)

    # Plan plots for ablation (b) (the main PnP-PGD plots of job3565048 remain final and unchanged).
    rp.apply_paper_plot_style()
    idx = []
    for ref in TEST_REFS:
        d = A.load(args.test_dir, ref, "pnp_pgd_from_xT")["npz"]
        order = np.argsort(-d["plan_reward"], kind="stable")
        for rank, ci in enumerate(order[:4], start=1):
            cidx = int(d["candidate_idx"][ci])
            stem = fig_dir / "plan_plots" / f"plan_{ref}_pnp_pgd_from_xT_rank{rank:02d}_cand{cidx}"
            A.plan_plot(d["planned_xy"][ci], d["rollout_xy"][ci], d["reference_xy"], d["init_xy"], stem)
            idx.append({"reference": ref, "rank": rank, "candidate_idx": cidx, "c_plan": float(-d["plan_reward"][ci]),
                        "c_rollout": float(-d["rollout_reward"][ci]), "e_dyn": float(d["feasibility_mean_l2_future"][ci]),
                        "png": f"{stem}.png"})
    safe_io.write_json(fig_dir / "plan_plots" / "plot_index_from_xT.json", idx)

    fig_cost_vs_dyn(args.test_dir, MAIN_V2 + ["pnp_pgd_budget20", "pnp_pgd_noscale"], fig_dir / "unicycle_crollout_vs_edyn")
    fig_cost_vs_dyn(args.test_dir, MAIN_V2, fig_dir / "unicycle_crollout_vs_edyn_main")
    fig_budget(args.test_dir, curves, fig_dir / "unicycle_crollout_vs_nfe")
    fig_per_iter(args.test_dir, curves, fig_dir / "unicycle_pnp_per_iteration")
    safe_io.write_json(fig_dir / "analysis_provenance.json", {
        "val_dir": str(args.val_dir), "test_dir": str(args.test_dir), "supp_dir": str(args.supp_dir),
        "selection": json.load(open(args.val_dir / "selection.json")), "provenance": safe_io.provenance(), "argv": sys.argv})
    print("done", fig_dir)
    for k in [f"{r}/pnp_pgd_from_xT/vs/dcg" for r in TEST_REFS] + [f"{r}/pnp_pgd/vs/dcg" for r in TEST_REFS]:
        print(k, paired[k])


if __name__ == "__main__":
    main()
