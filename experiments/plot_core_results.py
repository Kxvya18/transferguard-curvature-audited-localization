"""Generate conference-ready figures from persisted TRANSFERGUARD results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "navy": "#153B5B",
    "teal": "#168A83",
    "amber": "#E39A2D",
    "coral": "#D65F4A",
    "slate": "#718096",
    "cream": "#F6F1E7",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFAF7",
            "grid.color": "#D8D5CE",
            "grid.alpha": 0.55,
        }
    )


def plot_stress_grid(
    stress_dir: Path,
    output_dir: Path,
    output_name: str,
    title: str,
) -> Path:
    frame = pd.read_csv(stress_dir / "sample_metrics.csv")
    with (stress_dir / "statistical_analysis.json").open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    methods = pd.DataFrame(stats["method_summary"])
    method_order = ["degree", "boltzmann_prior", "outbound_activity", "no_prior", "hybrid"]
    methods = methods.set_index("method").loc[method_order].reset_index()

    paired = frame.pivot(
        index=["n", "m", "beta_true", "sample"], columns="method", values="mrr"
    ).reset_index()
    paired["delta"] = paired["hybrid"] - paired["no_prior"]
    strength = frame[frame["method"] == "hybrid"][
        ["n", "m", "beta_true", "sample", "prior_strength"]
    ]
    paired = paired.merge(strength, on=["n", "m", "beta_true", "sample"])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        color=COLORS["navy"],
    )

    ax = axes[0, 0]
    bar_colors = [COLORS["slate"], COLORS["amber"], "#5596B3", COLORS["coral"], COLORS["teal"]]
    bars = ax.bar(np.arange(len(methods)), methods["mrr"], color=bar_colors, width=0.72)
    ax.set_xticks(
        np.arange(len(methods)),
        ["Degree", "Boltzmann\nonly", "Outbound\nactivity", "Temporal\nonly", "Safe\nhybrid"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean reciprocal rank")
    ax.set_title("(a) 480 held-out scenarios")
    ax.grid(axis="y")
    for bar, value in zip(bars, methods["mrr"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")

    ax = axes[0, 1]
    beta_summary = paired.groupby("beta_true").agg(delta=("delta", "mean")).reset_index()
    ci = stats["paired_hybrid_minus_no_prior"]["bootstrap_ci95"]
    ax.axhline(0.0, color=COLORS["slate"], linewidth=1.0)
    ax.plot(
        beta_summary["beta_true"],
        beta_summary["delta"],
        marker="o",
        linewidth=2.4,
        color=COLORS["teal"],
        label="Safe hybrid - temporal only",
    )
    ax.fill_between(beta_summary["beta_true"], 0, beta_summary["delta"], color=COLORS["teal"], alpha=0.12)
    ax.set_xlabel("Generator β")
    ax.set_ylabel("Paired ΔMRR")
    mean_delta = stats["paired_hybrid_minus_no_prior"]["mean_delta_mrr"]
    ax.set_title(
        f"(b) Overall ΔMRR = {mean_delta:+.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    )
    ax.grid()

    ax = axes[1, 0]
    heat = paired.groupby(["m", "beta_true"])["delta"].mean().unstack("beta_true")
    image = ax.imshow(heat.to_numpy(), cmap="RdYlGn", vmin=-0.10, vmax=0.10, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)), [f"{value:g}" for value in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)), [str(value) for value in heat.index])
    ax.set_xlabel("Generator β")
    ax.set_ylabel("Edges per arrival m")
    ax.set_title("(c) Mean paired ΔMRR by growth regime")
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            value = heat.iloc[row, column]
            ax.text(column, row, f"{value:+.03f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Hybrid - temporal-only MRR")

    ax = axes[1, 1]
    gate = paired.groupby("beta_true").agg(
        strength=("prior_strength", "mean"),
        gain=("delta", "mean"),
    ).reset_index()
    ax.plot(gate["beta_true"], gate["strength"], marker="s", color=COLORS["amber"], linewidth=2.4)
    ax.set_xlabel("Generator β")
    ax.set_ylabel("Learned prior strength", color=COLORS["amber"])
    ax.tick_params(axis="y", labelcolor=COLORS["amber"])
    ax.set_title("(d) Learned prior strength and realized ranking gain")
    ax.grid()
    twin = ax.twinx()
    twin.plot(gate["beta_true"], gate["gain"], marker="o", color=COLORS["teal"], linewidth=2.0)
    twin.set_ylabel("Paired ΔMRR", color=COLORS["teal"])
    twin.tick_params(axis="y", labelcolor=COLORS["teal"])

    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training(
    safe_dir: Path,
    no_prior_dir: Path,
    output_dir: Path,
    output_name: str,
    title: str,
) -> Path:
    safe = pd.read_json(safe_dir / "training_history.json")
    no_prior = pd.read_json(no_prior_dir / "training_history.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=COLORS["navy"])
    axes[0].plot(safe["epoch"], safe["val_mrr"], color=COLORS["teal"], linewidth=2.2, label="Safe hybrid")
    axes[0].plot(no_prior["epoch"], no_prior["val_mrr"], color=COLORS["coral"], linewidth=2.2, label="Temporal only")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MRR")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("(a) Ranking performance")
    axes[0].grid()
    axes[0].legend(frameon=False)

    axes[1].plot(safe["epoch"], safe["mean_confidence"], color=COLORS["teal"], linewidth=2.2, label="Safe hybrid")
    axes[1].plot(no_prior["epoch"], no_prior["mean_confidence"], color=COLORS["coral"], linewidth=2.2, label="Temporal only")
    axes[1].plot(safe["epoch"], safe["val_hit@1"], color=COLORS["navy"], linestyle="--", linewidth=1.7, label="Hybrid hit@1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean confidence / hit@1")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) Confidence versus correctness")
    axes[1].grid()
    axes[1].legend(frameon=False)
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_iot23(project_root: Path, output_dir: Path) -> Path:
    metrics = pd.read_csv(project_root / "outputs" / "iot23_baselines" / "scenario_metrics.csv")
    order = ["uniform_expected", "activity_change", "directional_asymmetry", "degree", "outbound_activity", "boltzmann_prior"]
    summary = metrics.groupby("method")["mrr"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    colors = [COLORS["slate"], "#5596B3", COLORS["amber"], COLORS["navy"], COLORS["coral"], COLORS["teal"]]
    bars = ax.bar(np.arange(len(summary)), summary.values, color=colors)
    ax.set_xticks(
        np.arange(len(summary)),
        ["Uniform", "Activity\nchange", "Directional\nasymmetry", "Degree", "Outbound\nactivity", "Boltzmann\nrank"],
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean reciprocal rank across 5 scenarios")
    ax.set_title("IoT-23 is a Real-Data Sanity Check, Not a Discriminating Provenance Benchmark", fontweight="bold")
    ax.grid(axis="y")
    for bar, value in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    ax.text(
        0.02,
        0.96,
        "Degree, earliest-seen, outbound activity, and Boltzmann rank all identify the star center.\nIoT-23 supplies malicious-flow sources, not multi-hop compromise chronology.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLORS["cream"], "edgecolor": "#C7BFAF"},
    )
    output_path = output_dir / "fig_iot23_baseline_audit.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-dir", type=Path, default=project_root / "paper_assets" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    paths = [
        plot_stress_grid(
            args.project_root / "outputs" / "synthetic_stress_grid_v3",
            args.output_dir,
            "fig_stress_grid_sparse_alert_v3.png",
            "Controlled Sparse-Alert Protocol: Safe Boltzmann Fusion",
        ),
        plot_stress_grid(
            args.project_root / "outputs" / "synthetic_stress_grid_v4",
            args.output_dir,
            "fig_stress_grid_aligned_v4.png",
            "Cross-Dataset-Aligned Activity Protocol: Prior Ablation",
        ),
        plot_training(
            args.project_root / "outputs" / "train_transferguard_mixed_safe_prior",
            args.project_root / "outputs" / "train_transferguard_mixed_no_prior",
            args.output_dir,
            "fig_training_sparse_alert_v3.png",
            "Sparse-Alert Mixed-Domain Training",
        ),
        plot_training(
            args.project_root / "outputs" / "train_transferguard_mixed_safe_prior_v4",
            args.project_root / "outputs" / "train_transferguard_mixed_no_prior_v4",
            args.output_dir,
            "fig_training_aligned_v4.png",
            "Cross-Dataset-Aligned Mixed-Domain Training",
        ),
        plot_iot23(args.project_root, args.output_dir),
    ]
    print("Generated figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
