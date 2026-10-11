"""Plot calibrated uncertainty and intervention-utility results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "paper_assets" / "figures"
COLORS = {
    "navy": "#153B5B",
    "teal": "#168A83",
    "amber": "#E39A2D",
    "coral": "#D65F4A",
    "slate": "#718096",
    "blue": "#5596B3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FBFAF7",
            "grid.color": "#D8D5CE",
            "grid.alpha": 0.55,
        }
    )


def plot_uncertainty() -> Path:
    with (PROJECT_ROOT / "outputs" / "calibrated_selective_v4" / "summary.json").open() as handle:
        payload = json.load(handle)
    models = {item["model"]: item for item in payload["models"]}
    names = ["safe_hybrid", "temporal_only"]
    labels = ["Safe hybrid", "Temporal only"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    fig.suptitle(
        "Uncertainty Audit Under Joint Structural and Telemetry Shift",
        fontsize=15,
        fontweight="bold",
        color=COLORS["navy"],
    )

    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.35
    id_acc = [models[name]["raw"]["id"]["accuracy"] for name in names]
    ood_acc = [models[name]["raw"]["ood"]["accuracy"] for name in names]
    ax.bar(x - width / 2, id_acc, width, label="In distribution", color=COLORS["teal"])
    ax.bar(x + width / 2, ood_acc, width, label="OOD", color=COLORS["coral"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Origin hit@1")
    ax.set_title("(a) Shift causes a real performance collapse")
    ax.grid(axis="y")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    detector_names = [
        "one_minus_max_probability",
        "normalized_predictive_entropy",
        "mc_mutual_information",
        "mc_epistemic_variance",
    ]
    detector_labels = ["1-max p", "Entropy", "MC mutual\ninformation", "MC variance"]
    for offset, (name, label, color) in enumerate(
        zip(names, labels, (COLORS["teal"], COLORS["navy"]))
    ):
        values = [models[name]["ood_detection_auroc"][metric] for metric in detector_names]
        ax.bar(np.arange(4) + (offset - 0.5) * width, values, width, label=label, color=color)
    ax.axhline(0.5, color=COLORS["slate"], linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(4), detector_labels)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("OOD detection AUROC")
    ax.set_title("(b) Uncertainty detects deployment shift")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 0]
    positions = np.arange(4)
    raw_values = [
        models["temporal_only"]["raw"]["id"]["nll"],
        models["temporal_only"]["temperature_scaled"]["id"]["nll"],
        models["temporal_only"]["raw"]["ood"]["nll"],
        models["temporal_only"]["temperature_scaled"]["ood"]["nll"],
    ]
    colors = [COLORS["teal"], COLORS["amber"], COLORS["coral"], "#A23E48"]
    bars = ax.bar(positions, raw_values, color=colors)
    ax.set_xticks(positions, ["ID raw", "ID temp.", "OOD raw", "OOD temp."])
    ax.set_ylabel("Negative log likelihood")
    ax.set_title("(c) Naive temperature scaling is not shift-safe")
    ax.grid(axis="y")
    for bar, value in zip(bars, raw_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center")

    ax = axes[1, 1]
    for split, color in (("id", COLORS["teal"]), ("ood", COLORS["coral"])):
        points = models["temporal_only"]["raw"][split]["risk_coverage"]["selective"]
        coverage = [point["coverage"] for point in points]
        accuracy = [point["accuracy"] for point in points]
        ax.plot(coverage, accuracy, marker="o", linewidth=2.2, color=color, label=split.upper())
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Accepted-set accuracy")
    ax.set_title("(d) Abstention is reliable only in distribution")
    ax.grid()
    ax.legend(frameon=False)
    path = OUTPUT_DIR / "fig_uncertainty_shift_audit_v4.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_intervention() -> Path:
    with (PROJECT_ROOT / "outputs" / "intervention_utility_v4" / "summary.json").open() as handle:
        summary = json.load(handle)
    with (
        PROJECT_ROOT / "outputs" / "intervention_utility_v4" / "statistical_analysis.json"
    ).open() as handle:
        stats = json.load(handle)
    aggregate = {(row["method"], row["k"]): row for row in summary["intervention_aggregate"]}
    early = {row["method"]: row for row in summary["early_order_aggregate"]}
    methods = [
        "random",
        "temporal_only",
        "degree",
        "outbound_activity",
        "intervention_risk_temporal",
        "oracle_early",
    ]
    labels = ["Random", "Source-only", "Degree", "Outbound", "Decision-aware", "Oracle early"]
    colors = [COLORS["slate"], COLORS["coral"], COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["amber"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    fig.suptitle(
        "Origin Localization and Containment Are Different Objectives",
        fontsize=15,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax = axes[0]
    for method, label, color in zip(methods, labels, colors):
        values = [aggregate[(method, k)]["mean_prevented_fraction"] for k in (1, 3, 5)]
        ax.plot((1, 3, 5), values, marker="o", linewidth=2.2, color=color, label=label)
    ax.set_xticks((1, 3, 5))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Quarantine budget k")
    ax.set_ylabel("Mean fraction of future compromises prevented")
    ax.set_title("(a) Probability x impact converts rankings into actions")
    ax.grid()
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1]
    scatter_methods = ["activity_change", "boltzmann_prior", "degree", "outbound_activity", "safe_hybrid", "temporal_only", "intervention_risk_temporal"]
    scatter_labels = ["Activity", "Boltzmann", "Degree", "Outbound", "Hybrid", "Temporal", "Decision-aware"]
    for method, label in zip(scatter_methods, scatter_labels):
        ax.scatter(
            early[method]["origin_mrr"],
            aggregate[(method, 3)]["mean_prevented_fraction"],
            s=75,
            label=label,
        )
        ax.annotate(label, (early[method]["origin_mrr"], aggregate[(method, 3)]["mean_prevented_fraction"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    test = stats["paired_comparisons"]["k1:intervention_risk_temporal-degree"]
    ax.text(
        0.03,
        0.96,
        f"At k=1 vs degree: Δ={test['mean_delta']:+.3f}\n95% CI [{test['bootstrap_ci95'][0]:.3f}, {test['bootstrap_ci95'][1]:.3f}], p={test['sign_flip_pvalue_two_sided']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F6F1E7", "edgecolor": "#C7BFAF"},
    )
    ax.set_xlabel("Origin-localization MRR")
    ax.set_ylabel("Future compromise prevented at k=3")
    ax.set_ylim(0.03, 0.64)
    ax.set_title("(b) Better source MRR does not imply better containment")
    ax.grid()
    path = OUTPUT_DIR / "fig_intervention_utility_v4.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    for path in (plot_uncertainty(), plot_intervention()):
        print(path)


if __name__ == "__main__":
    main()
