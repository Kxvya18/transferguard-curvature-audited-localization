"""Render a confirmatory phase-diagram view of Fisher-selective fusion.

This figure intentionally uses only the independent confirmatory graph grid.
It treats hybrid-minus-temporal MRR as the response order parameter, Fisher
reliability as the structural-control axis, and temporal observations as the
external evidence field.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs/cold_start_v1/confirmatory_evaluation/summary.json"
METRICS_PATH = ROOT / "outputs/cold_start_v1/confirmatory_evaluation/model_sample_metrics.csv"
LOW_CONDITIONS = {(1, 0.25), (1, 0.5), (2, 0.25), (2, 0.5)}
CONDITIONS = [(horizon, retention) for retention in (0.25, 0.5, 1.0) for horizon in (1, 2, 3, 6)]


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int = 8000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        means[draw] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _load_graph_effects() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_PATH)
    keys = ["seed", "sample", "n_nodes", "m", "beta", "horizon", "telemetry_retention"]
    scores = metrics.pivot_table(index=keys, columns="method", values="mrr", aggfunc="mean").reset_index()
    reliability = (
        metrics.loc[metrics["method"] == "hybrid", keys + ["fisher_reliability"]]
        .groupby(keys, as_index=False)["fisher_reliability"]
        .mean()
    )
    scores = scores.merge(reliability, on=keys, validate="one_to_one")
    scores["delta_mrr"] = scores["hybrid"] - scores["temporal_only"]
    graph_keys = ["sample", "n_nodes", "m", "beta", "horizon", "telemetry_retention"]
    return scores.groupby(graph_keys, as_index=False).agg(
        delta_mrr=("delta_mrr", "mean"),
        fisher_reliability=("fisher_reliability", "mean"),
    )


def _quartile(values: pd.Series, thresholds: list[float]) -> np.ndarray:
    return np.digitize(values.to_numpy(dtype=float), thresholds, right=True) + 1


def main() -> None:
    with SUMMARY_PATH.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    thresholds = [float(value) for value in summary["reliability_thresholds"]["q25_q50_q75"]]
    effects = _load_graph_effects()
    effects["reliability_bin"] = _quartile(effects["fisher_reliability"], thresholds)
    low = (
        effects.loc[
            effects.apply(lambda row: (int(row.horizon), float(row.telemetry_retention)) in LOW_CONDITIONS, axis=1)
        ]
        .groupby(["sample", "n_nodes", "m", "beta"], as_index=False)
        .agg(delta_mrr=("delta_mrr", "mean"), fisher_reliability=("fisher_reliability", "mean"))
    )
    low["reliability_bin"] = _quartile(low["fisher_reliability"], thresholds)

    rng = np.random.default_rng(20260825)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    navy, teal, coral, gold, ink = "#19324A", "#149D8C", "#DD6E42", "#E9B949", "#1E1E1E"
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.3))
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.08, top=0.90, wspace=0.18, hspace=0.30)
    fig.suptitle(
        "Fisher-Identifiability Phase Diagram for Cold-Start Source Localization",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )

    # (a) Low-evidence response order parameter against structural reliability.
    ax = axes[0, 0]
    fisher_information = 5.0 * low["fisher_reliability"].to_numpy() / np.maximum(1.0 - low["fisher_reliability"].to_numpy(), 1e-12)
    log_information = np.log10(fisher_information)
    jitter = rng.normal(0.0, 0.0015, size=len(low))
    ax.scatter(
        log_information + jitter,
        low["delta_mrr"],
        s=18,
        color=navy,
        alpha=0.25,
        linewidths=0,
        label="confirmatory graphs",
    )
    for bin_index in range(1, 5):
        subset = low.loc[low["reliability_bin"] == bin_index]
        mean = float(subset["delta_mrr"].mean())
        ci_low, ci_high = _bootstrap_mean(subset["delta_mrr"].to_numpy(), rng)
        x_value = float(np.log10(5.0 * subset["fisher_reliability"].mean() / (1.0 - subset["fisher_reliability"].mean())))
        ax.errorbar(
            x_value,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="o",
            ms=8,
            capsize=4,
            color=coral if mean < 0 else teal,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.annotate(f"Q{bin_index}", (x_value, mean), xytext=(0, 9 if mean >= 0 else -14), textcoords="offset points", ha="center")
    threshold_information = 5.0 * thresholds[-1] / (1.0 - thresholds[-1])
    ax.axvline(np.log10(threshold_information), color=gold, lw=2.0, ls="--", label="validation Q75")
    ax.axhline(0.0, color=ink, lw=0.9)
    ax.set_title("(a) Response order parameter versus Fisher information")
    ax.set_xlabel(r"observed Fisher information $\log_{10}\mathcal{I}(\hat\beta)$")
    ax.set_ylabel(r"low-evidence $\Delta$MRR (hybrid $-$ temporal)")
    ax.legend(frameon=False, loc="lower right")

    # (b) Two-dimensional response phase map.
    ax = axes[0, 1]
    matrix = np.full((4, len(CONDITIONS)), np.nan)
    for col, (horizon, retention) in enumerate(CONDITIONS):
        subset = effects.loc[
            (effects["horizon"] == horizon)
            & (effects["telemetry_retention"] == retention)
        ]
        for bin_index in range(1, 5):
            values = subset.loc[subset["reliability_bin"] == bin_index, "delta_mrr"]
            matrix[bin_index - 1, col] = float(values.mean())
    bound = max(0.04, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > bound * 0.55 else ink,
                fontweight="bold",
            )
    ax.axhline(2.5, color=gold, lw=2.0, ls="--")
    ax.set_title("(b) Discrete response phase diagram")
    ax.set_xlabel("observed windows | telemetry retention")
    ax.set_ylabel("validation-defined Fisher-reliability bin")
    ax.set_xticks(range(len(CONDITIONS)), [f"{h}|{q:.0%}" for h, q in CONDITIONS], rotation=45, ha="right")
    ax.set_yticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(r"mean $\Delta$MRR")

    # (c) Response to attachment multiplicity across the full m sweep.
    ax = axes[1, 0]
    for bin_index, label, color, marker in ((1, "low reliability (Q1)", coral, "o"), (4, "selected reliability (Q4)", teal, "s")):
        means, lower, upper = [], [], []
        attachment_counts = [1, 2, 3, 4, 5]
        for attachment_count in attachment_counts:
            values = low.loc[(low["m"] == attachment_count) & (low["reliability_bin"] == bin_index), "delta_mrr"].to_numpy()
            mean = float(values.mean())
            ci_low, ci_high = _bootstrap_mean(values, rng)
            means.append(mean)
            lower.append(mean - ci_low)
            upper.append(ci_high - mean)
        ax.errorbar(attachment_counts, means, yerr=[lower, upper], marker=marker, ms=7, lw=2.3, capsize=4, color=color, label=label)
    ax.axhline(0.0, color=ink, lw=0.9)
    ax.set_title("(c) Response across attachment multiplicity")
    ax.set_xlabel("edges per arriving node $m$")
    ax.set_ylabel(r"mean low-evidence $\Delta$MRR")
    ax.set_xticks(attachment_counts)
    ax.legend(frameon=False, loc="best")

    # (d) Operating curve for validation-defined policy thresholds.
    ax = axes[1, 1]
    threshold_rows = [("always", -np.inf)] + [(f"Q{quantile}", threshold) for quantile, threshold in zip((25, 50, 75), thresholds)]
    gains, ci_lower, ci_upper, coverage = [], [], [], []
    for _, threshold in threshold_rows:
        contribution = np.where(low["fisher_reliability"].to_numpy() > threshold, low["delta_mrr"].to_numpy(), 0.0)
        mean = float(contribution.mean())
        lo, hi = _bootstrap_mean(contribution, rng)
        gains.append(mean)
        ci_lower.append(mean - lo)
        ci_upper.append(hi - mean)
        coverage.append(float(np.mean(low["fisher_reliability"].to_numpy() > threshold)))
    indices = np.arange(len(threshold_rows))
    bars = ax.bar(indices, gains, color=[coral if value <= 0 else teal for value in gains], alpha=0.92)
    ax.errorbar(indices, gains, yerr=[ci_lower, ci_upper], fmt="none", color=ink, capsize=4, zorder=3)
    ax.axhline(0.0, color=ink, lw=0.9)
    ax.set_xticks(indices, [label for label, _ in threshold_rows])
    ax.set_xlabel("threshold fixed on validation cache")
    ax.set_ylabel(r"policy improvement $\Delta$MRR")
    ax.set_title("(d) Selective-inference operating curve")
    for bar, value in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.0015 if value >= 0 else -0.0015), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    coverage_axis = ax.twinx()
    coverage_axis.plot(indices, coverage, color=gold, marker="D", lw=2.0, label="hybrid coverage")
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_ylabel("hybrid coverage", color=gold)
    coverage_axis.tick_params(axis="y", colors=gold)
    ax.text(0.03, 0.95, "Q75 is the pre-fixed\nconfirmatory policy", transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": gold, "alpha": 0.92})

    output = ROOT / "paper_assets/figures/fig_fisher_response_phase_diagram_v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    report = {
        "validation_thresholds": thresholds,
        "low_evidence_graphs": int(len(low)),
        "low_evidence_bins": [
            {
                "bin": f"Q{bin_index}",
                "n": int((low["reliability_bin"] == bin_index).sum()),
                "mean_delta_mrr": float(low.loc[low["reliability_bin"] == bin_index, "delta_mrr"].mean()),
            }
            for bin_index in range(1, 5)
        ],
        "threshold_operating_points": [
            {"threshold": label, "coverage": float(cov), "delta_mrr": float(gain)}
            for (label, _), cov, gain in zip(threshold_rows, coverage, gains)
        ],
    }
    with (ROOT / "outputs/cold_start_v1/confirmatory_evaluation/fisher_phase_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
