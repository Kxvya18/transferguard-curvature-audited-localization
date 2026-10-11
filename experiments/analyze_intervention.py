"""Paired statistical analysis for time-respecting quarantine utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _paired_test(delta: np.ndarray, seed: int) -> dict:
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(20_000, dtype=float)
    for idx in range(len(bootstrap)):
        bootstrap[idx] = float(np.mean(rng.choice(values, len(values), replace=True)))
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    null = np.mean(signs * values[None, :], axis=1)
    observed = float(np.mean(values))
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
    return {
        "n_paired": len(values),
        "mean_delta": observed,
        "median_delta": float(np.median(values)),
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "sign_flip_pvalue_two_sided": p_value,
        "win_rate": float(np.mean(values > 0)),
        "tie_rate": float(np.mean(values == 0)),
        "loss_rate": float(np.mean(values < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "intervention_utility_v4"
        / "sample_interventions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "intervention_utility_v4"
        / "statistical_analysis.json",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    per_scenario = (
        frame.groupby(["scenario", "n", "m", "beta_true", "method", "k"], as_index=False)
        .agg(prevented_fraction=("prevented_fraction", "mean"))
    )
    comparisons = {}
    competitors = (
        "degree",
        "outbound_activity",
        "activity_change",
        "temporal_only",
        "safe_hybrid",
        "random",
    )
    for k_value in (1, 3, 5):
        wide = per_scenario[per_scenario["k"] == k_value].pivot(
            index="scenario", columns="method", values="prevented_fraction"
        )
        for method in competitors:
            delta = (wide["intervention_risk_temporal"] - wide[method]).dropna().to_numpy()
            comparisons[f"k{k_value}:intervention_risk_temporal-{method}"] = _paired_test(
                delta, 20260820 + k_value * 17 + competitors.index(method)
            )
    stratified = {}
    for key in ("n", "m", "beta_true"):
        subset = per_scenario[per_scenario["method"].isin(
            ["intervention_risk_temporal", "degree", "outbound_activity"]
        )]
        grouped = (
            subset.groupby([key, "method", "k"], as_index=False)
            .agg(mean_prevented_fraction=("prevented_fraction", "mean"))
        )
        stratified[key] = grouped.to_dict(orient="records")
    payload = {
        "n_scenarios": int(per_scenario["scenario"].nunique()),
        "bootstrap_resamples": 20_000,
        "sign_flip_permutations": 100_000,
        "paired_comparisons": comparisons,
        "stratified": stratified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved paired intervention analysis to {args.output}")


if __name__ == "__main__":
    main()
