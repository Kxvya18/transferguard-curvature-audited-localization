"""Compute deployable-style OOD rejection operating characteristics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibrated_selective_v4"
        / "sample_uncertainty.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibrated_selective_v4"
        / "ood_operating_points.json",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    metrics = {
        "one_minus_confidence": lambda part: 1.0 - part["confidence"].to_numpy(float),
        "normalized_entropy": lambda part: part["normalized_entropy"].to_numpy(float),
        "mc_mutual_information": lambda part: part["mc_mutual_information"].to_numpy(float),
        "mc_epistemic_variance": lambda part: part["mc_epistemic_variance"].to_numpy(float),
    }
    payload = {
        "interpretation": (
            "Thresholds are retrospective operating characteristics set at the "
            "95th percentile of the independent ID test score (nominal 5% ID FPR), "
            "not deployment-tuned thresholds."
        ),
        "models": {},
    }
    for model, model_frame in frame.groupby("model"):
        id_frame = model_frame[model_frame["split"] == "id"]
        ood_frame = model_frame[model_frame["split"] == "ood"]
        model_results = {}
        for name, extractor in metrics.items():
            id_scores = extractor(id_frame)
            ood_scores = extractor(ood_frame)
            threshold = float(np.quantile(id_scores, 0.95, method="higher"))
            id_accept = id_scores <= threshold
            ood_accept = ood_scores <= threshold
            model_results[name] = {
                "threshold": threshold,
                "id_false_rejection_rate": float(np.mean(~id_accept)),
                "ood_true_rejection_rate": float(np.mean(~ood_accept)),
                "id_accepted_accuracy": float(id_frame.loc[id_accept, "correct"].mean()),
                "ood_accepted_coverage": float(np.mean(ood_accept)),
                "ood_accepted_accuracy": (
                    float(ood_frame.loc[ood_accept, "correct"].mean())
                    if np.any(ood_accept)
                    else None
                ),
            }
        payload["models"][str(model)] = model_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved OOD operating points to {args.output}")


if __name__ == "__main__":
    main()
