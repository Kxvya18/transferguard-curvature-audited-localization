"""Apply the pre-specified synthetic-ID entropy threshold to real datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    operating = _read(ROOT / "outputs/calibrated_selective_v4/ood_operating_points.json")
    rows = []
    for dataset, stem in (("IoT-23", "iot23"), ("TON_IoT", "ton")):
        for model, suffix in (("safe_hybrid", "hybrid"), ("temporal_only", "temporal")):
            real = _read(ROOT / f"outputs/cross_dataset_v4/{stem}_{suffix}.json")
            threshold = float(
                operating["models"][model]["normalized_entropy"]["threshold"]
            )
            scenario_rows = []
            for scenario in real["scenarios"]:
                score = 1.0 - float(scenario["entropy_confidence"])
                scenario_rows.append(
                    {
                        "scenario": scenario["scenario"],
                        "normalized_entropy": score,
                        "shift_flag": bool(score > threshold),
                        "origin_rank": float(scenario["origin_rank"]),
                    }
                )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_scenarios": len(scenario_rows),
                    "synthetic_id_threshold": threshold,
                    "shift_flag_rate": float(
                        np.mean([row["shift_flag"] for row in scenario_rows])
                    ),
                    "mean_normalized_entropy": float(
                        np.mean([row["normalized_entropy"] for row in scenario_rows])
                    ),
                    "scenarios": scenario_rows,
                }
            )

    payload = {
        "detector": "normalized predictive entropy",
        "threshold_contract": (
            "95th percentile of an independent synthetic ID test split; no IoT-23 "
            "or TON_IoT examples were used to select the threshold"
        ),
        "interpretation_boundary": (
            "A shift flag indicates feature-distribution incompatibility, not a "
            "guarantee that the source prediction is wrong. The synthetic threshold "
            "is a retrospective diagnostic until calibrated on a clean deployment stream."
        ),
        "rows": rows,
    }
    output = ROOT / "outputs/cross_dataset_v4/shift_detection_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(output)


if __name__ == "__main__":
    main()
