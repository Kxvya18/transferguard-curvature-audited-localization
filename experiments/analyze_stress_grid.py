"""Paired statistical analysis for the synthetic n/m/beta stress grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = np.mean(values[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(values)))
    exceedances = 0
    total = 100_000
    for start in range(0, total, 2_000):
        batch = min(2_000, total - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, len(values)))
        means = np.abs(np.mean(signs * values[None, :], axis=1))
        exceedances += int(np.sum(means >= observed))
    return float((exceedances + 1) / (total + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "synthetic_stress_grid"
        / "sample_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "synthetic_stress_grid"
        / "statistical_analysis.json",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    keys = ["n", "m", "beta_true", "sample"]
    pivot = frame.pivot(index=keys, columns="method", values="mrr").reset_index()
    metadata = (
        frame[frame["method"] == "hybrid"]
        [
            keys
            + [
                "beta_hat",
                "beta_boundary_warning",
                "prior_reliability",
                "prior_strength",
            ]
        ]
        .drop_duplicates(keys)
    )
    paired = pivot.merge(metadata, on=keys, how="left")
    paired["delta_hybrid_minus_no_prior"] = paired["hybrid"] - paired["no_prior"]
    delta = paired["delta_hybrid_minus_no_prior"].to_numpy(dtype=float)

    seed = 20260810
    bootstrap_rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed + 1)
    method_summary = (
        frame.groupby("method")[["mrr", "hit@1", "hit@5"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    stratified = {}
    for column in ("n", "m", "beta_true"):
        stratified[column] = [
            {
                column: value.item() if hasattr(value, "item") else value,
                "n_samples": int(len(group)),
                "mean_delta_mrr": float(group["delta_hybrid_minus_no_prior"].mean()),
                "hybrid_mrr": float(group["hybrid"].mean()),
                "no_prior_mrr": float(group["no_prior"].mean()),
            }
            for value, group in paired.groupby(column)
        ]

    paired["reliability_quartile"] = pd.qcut(
        paired["prior_reliability"],
        q=4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )
    reliability = [
        {
            "quartile": str(value),
            "n_samples": int(len(group)),
            "mean_reliability": float(group["prior_reliability"].mean()),
            "mean_delta_mrr": float(group["delta_hybrid_minus_no_prior"].mean()),
        }
        for value, group in paired.groupby("reliability_quartile", observed=True)
    ]

    paired["strength_quartile"] = pd.qcut(
        paired["prior_strength"],
        q=4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )
    strength_groups = [
        {
            "quartile": str(value),
            "n_samples": int(len(group)),
            "mean_prior_strength": float(group["prior_strength"].mean()),
            "mean_delta_mrr": float(group["delta_hybrid_minus_no_prior"].mean()),
        }
        for value, group in paired.groupby("strength_quartile", observed=True)
    ]
    strength_by_regime = {}
    for column in ("n", "m", "beta_true"):
        strength_by_regime[column] = [
            {
                column: value.item() if hasattr(value, "item") else value,
                "mean_prior_strength": float(group["prior_strength"].mean()),
                "mean_delta_mrr": float(group["delta_hybrid_minus_no_prior"].mean()),
            }
            for value, group in paired.groupby(column)
        ]

    cells = (
        paired.groupby(["n", "m", "beta_true"])
        .agg(
            hybrid_mrr=("hybrid", "mean"),
            no_prior_mrr=("no_prior", "mean"),
            mean_prior_reliability=("prior_reliability", "mean"),
            boundary_rate=("beta_boundary_warning", "mean"),
        )
        .reset_index()
    )
    cells["delta_mrr"] = cells["hybrid_mrr"] - cells["no_prior_mrr"]
    tolerance = 1e-12
    result = {
        "n_paired_samples": int(len(paired)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": 20_000,
        "sign_flip_seed": seed + 1,
        "sign_flip_permutations": 100_000,
        "method_summary": method_summary,
        "paired_hybrid_minus_no_prior": {
            "mean_delta_mrr": float(np.mean(delta)),
            "median_delta_mrr": float(np.median(delta)),
            "bootstrap_ci95": _bootstrap_ci(delta, bootstrap_rng),
            "sign_flip_pvalue_two_sided": _sign_flip_pvalue(delta, permutation_rng),
            "hybrid_sample_win_rate": float(np.mean(delta > tolerance)),
            "tie_rate": float(np.mean(np.abs(delta) <= tolerance)),
            "hybrid_sample_loss_rate": float(np.mean(delta < -tolerance)),
        },
        "cell_comparison": {
            "n_cells": int(len(cells)),
            "hybrid_wins": int(np.sum(cells["delta_mrr"] > tolerance)),
            "ties": int(np.sum(np.abs(cells["delta_mrr"]) <= tolerance)),
            "hybrid_losses": int(np.sum(cells["delta_mrr"] < -tolerance)),
            "best_hybrid_cell": cells.loc[cells["delta_mrr"].idxmax()].to_dict(),
            "worst_hybrid_cell": cells.loc[cells["delta_mrr"].idxmin()].to_dict(),
        },
        "stratified": stratified,
        "by_prior_reliability": reliability,
        "prior_strength_diagnostics": {
            "mean": float(paired["prior_strength"].mean()),
            "correlation_with_delta_mrr": float(
                paired[["prior_strength", "delta_hybrid_minus_no_prior"]].corr().iloc[0, 1]
            ),
            "by_strength_quartile": strength_groups,
            "by_regime": strength_by_regime,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    cells.to_csv(args.output.with_name("cell_comparison.csv"), index=False)
    print(f"Saved paired stress-grid analysis to {args.output}")


if __name__ == "__main__":
    main()
