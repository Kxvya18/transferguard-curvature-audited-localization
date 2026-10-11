"""Scenario discovery for extracted TON_IoT archives."""

from __future__ import annotations

from pathlib import Path


def discover_ton_iot_scenarios(root_dir: str | Path) -> list[dict]:
    root = Path(root_dir)
    scenarios: list[dict] = []

    processed_iot = root / "Processed_datasets" / "Processed_IoT_dataset"
    gt_iot = root / "SecuityEvents_GroundTruth_datasets" / "SecurityEvents_IoT_datasets"
    if processed_iot.exists() and gt_iot.exists():
        for csv_path in sorted(processed_iot.glob("IoT_*.csv")):
            stem = csv_path.stem.replace("IoT_", "")
            gt_path = gt_iot / f"GroundTruth_IoT_{stem}.csv"
            scenarios.append(
                {
                    "modality": "iot",
                    "scenario_name": stem,
                    "data_csv": csv_path,
                    "ground_truth_csv": gt_path if gt_path.exists() else None,
                }
            )

    processed_net = root / "Processed_datasets" / "Processed_Network_dataset"
    gt_net = root / "SecuityEvents_GroundTruth_datasets" / "SecurityEvents_Network_datasets"
    if processed_net.exists() and gt_net.exists():
        for csv_path in sorted(processed_net.glob("Network_dataset_*.csv")):
            suffix = csv_path.stem.replace("Network_dataset_", "")
            gt_path = gt_net / f"GroundTruth_Network_{suffix}.csv"
            scenarios.append(
                {
                    "modality": "network",
                    "scenario_name": f"network_{suffix}",
                    "data_csv": csv_path,
                    "ground_truth_csv": gt_path if gt_path.exists() else None,
                }
            )

    return scenarios

