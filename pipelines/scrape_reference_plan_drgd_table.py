#!/usr/bin/env python3
"""Scrape DRGD gamma sweep results and emit paper-table LaTeX rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAPER_TRAJECTORIES = (
    ("circle", "Circle"),
    ("random_smooth_1", "Random Trajectory 1"),
    ("random_smooth_7", "Random Trajectory 2"),
)

BASELINE_CONFIGS = {
    "monte_carlo": "standard_w_cg0",
    "pdg": "standard_w_cg30",
    "dcg": "optimization_scale_30",
}

DRGD_CONFIG_RE = re.compile(r"^drgd_gamma(?P<gamma>\d+)$")


def _load_block(summary_path: Path) -> dict | None:
    with open(summary_path) as f:
        payload = json.load(f)
    for key in ("reference_plan", "heart_plan"):
        if key in payload:
            return payload[key]
    return None


def _metrics_from_summary(summary_path: Path) -> dict:
    block = _load_block(summary_path)
    if block is None:
        raise ValueError(f"No plan block in {summary_path}")
    stats = block.get("candidate_metrics_summary", {})
    row = {
        "summary_path": str(summary_path),
        "config": summary_path.parent.name,
        "trajectory": summary_path.parent.parent.name,
    }
    for key in ("plan_reward", "rollout_reward", "feasibility_mean_l2_future"):
        metric = stats.get(key, {})
        row[f"{key}_mean"] = metric.get("mean")
        row[f"{key}_std"] = metric.get("std")
    return row


def _cost(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(float(value))


def _fmt_cost(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "—"
    return f"${mean:.4f} \\pm {std:.4f}$"


def _parse_drgd_gamma(config: str) -> float | None:
    match = DRGD_CONFIG_RE.match(config)
    if not match:
        return None
    return float(match.group("gamma"))


def scrape_drgd_runs(root: Path) -> list[dict]:
    runs: list[dict] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        config = summary_path.parent.name
        gamma = _parse_drgd_gamma(config)
        if gamma is None:
            continue
        metrics = _metrics_from_summary(summary_path)
        metrics["drgd_gamma"] = gamma
        metrics["c_plan"] = _cost(metrics["plan_reward_mean"])
        metrics["c_rollout"] = _cost(metrics["rollout_reward_mean"])
        metrics["e_dyn"] = metrics["feasibility_mean_l2_future_mean"]
        metrics["c_plan_std"] = _cost(metrics.get("plan_reward_std"))
        metrics["c_rollout_std"] = _cost(metrics.get("rollout_reward_std"))
        metrics["e_dyn_std"] = metrics["feasibility_mean_l2_future_std"]
        runs.append(metrics)
    return runs


def _baseline_metrics(root: Path, trajectory: str, config: str) -> dict:
    summary_path = root / trajectory / config / "summary.json"
    metrics = _metrics_from_summary(summary_path)
    return {
        **metrics,
        "c_plan": _cost(metrics["plan_reward_mean"]),
        "c_rollout": _cost(metrics["rollout_reward_mean"]),
        "e_dyn": metrics["feasibility_mean_l2_future_mean"],
        "c_plan_std": _cost(metrics.get("plan_reward_std")),
        "c_rollout_std": _cost(metrics.get("rollout_reward_std")),
        "e_dyn_std": metrics["feasibility_mean_l2_future_std"],
    }


def select_best_drgd(
    runs: list[dict],
    *,
    dcg_c_plan: float,
    require_not_beat_dcg: bool = True,
) -> dict | None:
    eligible = []
    for row in runs:
        if row["c_plan"] is None:
            continue
        if require_not_beat_dcg and row["c_plan"] + 1e-12 < dcg_c_plan:
            continue
        eligible.append(row)
    if not eligible:
        return None
    eligible.sort(
        key=lambda r: (
            r["c_plan"],
            r["e_dyn"] if r["e_dyn"] is not None else float("inf"),
            r["c_rollout"] if r["c_rollout"] is not None else float("inf"),
        )
    )
    return eligible[0]


def select_global_gamma(
    runs_by_trajectory: dict[str, list[dict]],
    dcg_by_trajectory: dict[str, float],
    *,
    require_not_beat_dcg: bool = True,
) -> float | None:
    gammas = sorted({row["drgd_gamma"] for rows in runs_by_trajectory.values() for row in rows})
    best_gamma = None
    best_score = None
    for gamma in gammas:
        total = 0.0
        ok = True
        for trajectory, dcg_c_plan in dcg_by_trajectory.items():
            row = next((r for r in runs_by_trajectory[trajectory] if r["drgd_gamma"] == gamma), None)
            if row is None or row["c_plan"] is None:
                ok = False
                break
            if require_not_beat_dcg and row["c_plan"] + 1e-12 < dcg_c_plan:
                ok = False
                break
            total += row["c_plan"]
        if not ok:
            continue
        score = total / len(dcg_by_trajectory)
        if best_score is None or score < best_score:
            best_score = score
            best_gamma = gamma
    return best_gamma


def build_report(root: Path) -> dict:
    drgd_runs = scrape_drgd_runs(root)
    by_trajectory: dict[str, list[dict]] = {}
    for row in drgd_runs:
        by_trajectory.setdefault(row["trajectory"], []).append(row)

    selections: list[dict] = []
    dcg_c_plan_by_trajectory: dict[str, float] = {}
    for traj_key, _label in PAPER_TRAJECTORIES:
        dcg = _baseline_metrics(root, traj_key, BASELINE_CONFIGS["dcg"])
        dcg_c_plan_by_trajectory[traj_key] = dcg["c_plan"]
        best = select_best_drgd(by_trajectory.get(traj_key, []), dcg_c_plan=dcg["c_plan"])
        selections.append(
            {
                "trajectory": traj_key,
                "paper_label": next(l for k, l in PAPER_TRAJECTORIES if k == traj_key),
                "dcg_c_plan": dcg["c_plan"],
                "best_drgd": best,
                "all_drgd_runs": sorted(
                    by_trajectory.get(traj_key, []),
                    key=lambda r: r["drgd_gamma"],
                ),
            }
        )

    global_gamma = select_global_gamma(by_trajectory, dcg_c_plan_by_trajectory)
    return {
        "root": str(root),
        "paper_trajectories": [{"key": k, "label": l} for k, l in PAPER_TRAJECTORIES],
        "baseline_configs": BASELINE_CONFIGS,
        "selection_rule": (
            "Per trajectory: among DRGD gammas with c_plan >= DCG c_plan, "
            "pick lowest c_plan, then e_dyn, then c_rollout."
        ),
        "global_best_gamma": global_gamma,
        "selections": selections,
    }


def build_fixed_gamma_report(root: Path, gamma: float) -> dict:
    config = f"drgd_gamma{int(gamma)}" if float(gamma).is_integer() else f"drgd_gamma{gamma}"
    rows: list[dict] = []
    for traj_key, label in PAPER_TRAJECTORIES:
        summary_path = root / traj_key / config / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing DRGD run: {summary_path}")
        metrics = _metrics_from_summary(summary_path)
        rows.append(
            {
                "trajectory": traj_key,
                "paper_label": label,
                "drgd_gamma": gamma,
                "config": config,
                "c_plan": _cost(metrics["plan_reward_mean"]),
                "c_rollout": _cost(metrics["rollout_reward_mean"]),
                "e_dyn": metrics["feasibility_mean_l2_future_mean"],
                "c_plan_std": _cost(metrics.get("plan_reward_std")),
                "c_rollout_std": _cost(metrics.get("rollout_reward_std")),
                "e_dyn_std": metrics["feasibility_mean_l2_future_std"],
                "summary_path": str(summary_path),
            }
        )
    return {
        "root": str(root),
        "fixed_gamma": gamma,
        "config": config,
        "rows": rows,
    }


def render_fixed_gamma_table(root: Path, gamma: float) -> str:
    report = build_fixed_gamma_report(root, gamma)
    lines = [
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Setting & Algorithm",
        r"& $c_{\mathrm{plan}}$",
        r"& $c_{\mathrm{rollout}}$",
        r"& $e_{\mathrm{dyn}}$ \\",
        r"\midrule",
        "",
    ]
    drgd_label = f"Landing DRGD ($\\gamma={gamma:g}$)"
    for idx, item in enumerate(report["rows"]):
        traj_key = item["trajectory"]
        label = item["paper_label"]
        baselines = {
            "Monte Carlo": _baseline_metrics(root, traj_key, BASELINE_CONFIGS["monte_carlo"]),
            "PDG": _baseline_metrics(root, traj_key, BASELINE_CONFIGS["pdg"]),
            "DCG": _baseline_metrics(root, traj_key, BASELINE_CONFIGS["dcg"]),
        }
        if idx > 0:
            lines.append(r"\midrule")
            lines.append("")
        lines.append(rf"\multirow{{4}}{{*}}{{{label}}}")
        first = True
        for alg_name, metrics in baselines.items():
            prefix = "&" if first else "&"
            first = False
            lines.append(
                rf"{prefix} {alg_name}"
                rf"& {_fmt_cost(metrics['c_plan'], metrics['c_plan_std'])}"
                rf"& {_fmt_cost(metrics['c_rollout'], metrics['c_rollout_std'])}"
                rf"& {_fmt_cost(metrics['e_dyn'], metrics['e_dyn_std'])} \\"
            )
        lines.append(
            rf"& {drgd_label}"
            rf"& {_fmt_cost(item['c_plan'], item['c_plan_std'])}"
            rf"& {_fmt_cost(item['c_rollout'], item['c_rollout_std'])}"
            rf"& {_fmt_cost(item['e_dyn'], item['e_dyn_std'])} \\"
        )
        lines.append("")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return "\n".join(lines)


def render_latex_table(report: dict) -> str:
    lines = [
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Setting & Algorithm",
        r"& $c_{\mathrm{plan}}$",
        r"& $c_{\mathrm{rollout}}$",
        r"& $e_{\mathrm{dyn}}$ \\",
        r"\midrule",
        "",
    ]

    for idx, item in enumerate(report["selections"]):
        traj_key = item["trajectory"]
        label = item["paper_label"]
        root = Path(report["root"])

        baselines = {
            "Monte Carlo": _baseline_metrics(root, traj_key, BASELINE_CONFIGS["monte_carlo"]),
            "PDG": _baseline_metrics(root, traj_key, BASELINE_CONFIGS["pdg"]),
            "DCG": _baseline_metrics(root, traj_key, BASELINE_CONFIGS["dcg"]),
        }
        best = item["best_drgd"]
        if best is None:
            raise RuntimeError(f"No eligible DRGD run for {traj_key}")

        gamma = best["drgd_gamma"]
        drgd_label = f"Landing DRGD ($\\gamma={gamma:g}$)"

        if idx > 0:
            lines.append(r"\midrule")
            lines.append("")

        n_rows = len(baselines) + 1
        lines.append(rf"\multirow{{{n_rows}}}{{*}}{{{label}}}")

        first = True
        for alg_name, metrics in baselines.items():
            prefix = "&" if first else "&"
            first = False
            lines.append(
                rf"{prefix} {alg_name}"
                rf"& {_fmt_cost(metrics['c_plan'], metrics['c_plan_std'])}"
                rf"& {_fmt_cost(metrics['c_rollout'], metrics['c_rollout_std'])}"
                rf"& {_fmt_cost(metrics['e_dyn'], metrics['e_dyn_std'])} \\"
            )

        lines.append(
            rf"& {drgd_label}"
            rf"& {_fmt_cost(best['c_plan'], best['c_plan_std'])}"
            rf"& {_fmt_cost(best['c_rollout'], best['c_rollout_std'])}"
            rf"& {_fmt_cost(best['e_dyn'], best['e_dyn_std'])} \\"
        )
        lines.append("")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
        ]
    )
    return "\n".join(lines)


def render_drgd_only_rows(report: dict) -> str:
    lines = []
    for item in report["selections"]:
        best = item["best_drgd"]
        if best is None:
            continue
        gamma = best["drgd_gamma"]
        lines.append(
            rf"& Landing DRGD ($\gamma={gamma:g}$)"
            rf"& {_fmt_cost(best['c_plan'], best['c_plan_std'])}"
            rf"& {_fmt_cost(best['c_rollout'], best['c_rollout_std'])}"
            rf"& {_fmt_cost(best['e_dyn'], best['e_dyn_std'])} \\"
        )
    return "\n".join(lines)


def print_summary(report: dict) -> None:
    print(f"Root: {report['root']}")
    print(f"Selection rule: {report['selection_rule']}")
    if report["global_best_gamma"] is not None:
        print(f"Single global gamma (avg c_plan, must not beat DCG): {report['global_best_gamma']:g}")
    print()
    for item in report["selections"]:
        best = item["best_drgd"]
        print(f"=== {item['paper_label']} ({item['trajectory']}) ===")
        print(f"DCG c_plan threshold: {item['dcg_c_plan']:.4f}")
        if best is None:
            print("  No eligible DRGD config")
            continue
        print(
            f"  Best DRGD: gamma={best['drgd_gamma']:g}  "
            f"c_plan={best['c_plan']:.4f}±{best['c_plan_std']:.4f}  "
            f"c_rollout={best['c_rollout']:.4f}±{best['c_rollout_std']:.4f}  "
            f"e_dyn={best['e_dyn']:.4f}±{best['e_dyn_std']:.4f}"
        )
        print("  All gammas:")
        for row in item["all_drgd_runs"]:
            flag = " *" if row["drgd_gamma"] == best["drgd_gamma"] else ""
            beats = " [beats DCG]" if row["c_plan"] + 1e-12 < item["dcg_c_plan"] else ""
            print(
                f"    gamma={row['drgd_gamma']:>4g}: "
                f"c_plan={row['c_plan']:.4f}, "
                f"c_rollout={row['c_rollout']:.4f}, "
                f"e_dyn={row['e_dyn']:.4f}{beats}{flag}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape DRGD results for paper table rows.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/unicycle_eval/reference_plan_paper_v1"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: {root}/drgd_paper_table_selection.json",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="Default: {root}/drgd_paper_table.tex",
    )
    parser.add_argument(
        "--rows-only",
        action="store_true",
        help="Print only the DRGD row snippets to add to an existing table.",
    )
    parser.add_argument(
        "--fixed-gamma",
        type=float,
        default=None,
        help="Emit full paper table using a single DRGD gamma (e.g. 15).",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if args.fixed_gamma is not None:
        report = build_fixed_gamma_report(root, args.fixed_gamma)
        json_path = args.output_json or (root / f"drgd_paper_table_gamma{int(args.fixed_gamma)}.json")
        tex_path = args.output_tex or (root / f"drgd_paper_table_gamma{int(args.fixed_gamma)}.tex")
        tex = render_fixed_gamma_table(root, args.fixed_gamma)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        with open(tex_path, "w") as f:
            f.write(tex + "\n")
        print(f"Fixed gamma={args.fixed_gamma:g}")
        for item in report["rows"]:
            print(
                f"  {item['paper_label']}: "
                f"c_plan={item['c_plan']:.4f}±{item['c_plan_std']:.4f}, "
                f"c_rollout={item['c_rollout']:.4f}±{item['c_rollout_std']:.4f}, "
                f"e_dyn={item['e_dyn']:.4f}±{item['e_dyn_std']:.4f}"
            )
        print(f"Wrote {json_path}")
        print(f"Wrote {tex_path}")
        print()
        print(tex)
        return

    report = build_report(root)
    print_summary(report)

    json_path = args.output_json or (root / "drgd_paper_table_selection.json")
    tex_path = args.output_tex or (root / "drgd_paper_table.tex")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    if args.rows_only:
        tex = render_drgd_only_rows(report)
    else:
        tex = render_latex_table(report)
    with open(tex_path, "w") as f:
        f.write(tex + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")
    if args.rows_only:
        print()
        print("DRGD rows only:")
        print(tex)


if __name__ == "__main__":
    main()
