"""Render the confirmatory cold-start experiment in a physics-journal style."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/cold_start_v1/confirmatory_evaluation/summary.json"


def _condition(data, method, horizon, retention):
    return next(
        row for row in data["condition_summary"]
        if row["method"] == method and row["horizon"] == horizon
        and row["telemetry_retention"] == retention
    )


def main() -> None:
    with SUMMARY.open(encoding="utf-8") as handle:
        data = json.load(handle)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.titleweight": "bold",
        }
    )
    colors = {"hybrid": "#2A9D8F", "temporal_only": "#264653"}
    horizons = [1, 2, 3, 6]
    retentions = [0.25, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.5), constrained_layout=True)
    fig.suptitle(
        "Fisher-Selective Boltzmann Fusion Under Cold-Start Telemetry",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0, 0]
    for method, label, marker in (
        ("hybrid", "Boltzmann hybrid", "o"),
        ("temporal_only", "Temporal only", "s"),
    ):
        means, lower, upper = [], [], []
        for horizon in horizons:
            row = _condition(data, method, horizon, 0.5)
            means.append(row["mean_mrr"])
            lower.append(row["mean_mrr"] - row["mrr_ci95_training_seeds"][0])
            upper.append(row["mrr_ci95_training_seeds"][1] - row["mean_mrr"])
        ax.errorbar(
            horizons,
            means,
            yerr=[lower, upper],
            marker=marker,
            color=colors[method],
            lw=2,
            capsize=4,
            label=label,
        )
    ax.set_title("(a) Anytime source ranking at 50% telemetry retention")
    ax.set_xlabel("observed temporal windows $T_{obs}$")
    ax.set_ylabel("MRR")
    ax.set_xticks(horizons)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    matrix = np.zeros((len(retentions), len(horizons)))
    for row_index, retention in enumerate(retentions):
        for col_index, horizon in enumerate(horizons):
            row = next(
                item for item in data["paired_condition_effects"]
                if item["horizon"] == horizon
                and item["telemetry_retention"] == retention
            )
            matrix[row_index, col_index] = row["mean_delta_mrr"]
    bound = max(abs(float(matrix.min())), abs(float(matrix.max())))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:+.3f}",
                ha="center",
                va="center",
                color="white" if abs(matrix[row_index, col_index]) > 0.55 * bound else "black",
                fontweight="bold",
            )
    ax.set_title("(b) Paired prior effect across evidence regimes")
    ax.set_xlabel("observed windows $T_{obs}$")
    ax.set_ylabel("telemetry retention")
    ax.set_xticks(range(len(horizons)), horizons)
    ax.set_yticks(range(len(retentions)), [f"{value:.0%}" for value in retentions])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(r"$\Delta$MRR (hybrid $-$ temporal)")

    ax = axes[1, 0]
    for retention, color, marker in (
        (0.25, "#E76F51", "o"),
        (0.5, "#E9C46A", "s"),
        (1.0, "#2A9D8F", "D"),
    ):
        strengths = [
            _condition(data, "hybrid", horizon, retention)["mean_prior_strength"]
            for horizon in horizons
        ]
        ax.plot(
            horizons,
            strengths,
            marker=marker,
            lw=2,
            color=color,
            label=f"retention={retention:.0%}",
        )
    ax.set_title("(c) Learned gate suppresses structural evidence")
    ax.set_xlabel("observed temporal windows $T_{obs}$")
    ax.set_ylabel(r"mean residual strength $\lambda$")
    ax.set_xticks(horizons)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    quartiles = data["reliability_quartile_effect"]
    effects = np.asarray([row["mean_delta_mrr_low_evidence"] for row in quartiles])
    lower = effects - np.asarray([row["bootstrap_ci95"][0] for row in quartiles])
    upper = np.asarray([row["bootstrap_ci95"][1] for row in quartiles]) - effects
    bars = ax.bar(
        np.arange(4),
        effects,
        color=["#E76F51" if value < 0 else "#2A9D8F" for value in effects],
        alpha=0.92,
    )
    ax.errorbar(
        np.arange(4), effects, yerr=[lower, upper], fmt="none", color="#222", capsize=5
    )
    ax.axhline(0.0, color="#222", lw=1)
    ax.set_xticks(np.arange(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Fisher reliability bin (validation-fixed cut points)")
    ax.set_ylabel(r"low-evidence $\Delta$MRR")
    ax.set_title("(d) Structural prior helps only when identifiable")
    for bar, value in zip(bars, effects):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.004 if value >= 0 else -0.004),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    policy = data["validation_fixed_selective_policy"]["low_evidence"]
    selected = data["validation_fixed_selective_policy"]["selected_graph_fraction"]
    ax.text(
        0.03,
        0.97,
        "Confirmatory selective policy\n"
        + rf"selected={selected:.1%}, $\Delta$MRR={policy['delta_mrr']:+.4f}" + "\n"
        + rf"95% CI [{policy['graph_clustered_bootstrap_ci95'][0]:+.4f}, {policy['graph_clustered_bootstrap_ci95'][1]:+.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#666", "alpha": 0.92},
    )

    output = ROOT / "paper_assets/figures/fig_cold_start_fisher_selective_v6"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
