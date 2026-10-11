"""Build real temporal samples from extracted TON_IoT network scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loaders.ton_iot_scenarios import discover_ton_iot_scenarios
from loaders.ton_iot_temporal import build_ton_iot_network_sample_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ton-root",
        type=Path,
        default=Path("/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/data/processed/ton_iot_extracted/Ton_Iot"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/data/processed/ton_iot_temporal_samples_summary.json"),
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ton_iot_temporal_samples_v4",
    )
    parser.add_argument("--window-size", type=float, default=300.0)
    args = parser.parse_args()

    rows = []
    args.sample_dir.mkdir(parents=True, exist_ok=True)
    for scenario in discover_ton_iot_scenarios(args.ton_root):
        if scenario["modality"] != "network":
            continue
        sample = build_ton_iot_network_sample_v4(
            data_csv=scenario["data_csv"],
            ground_truth_csv=scenario["ground_truth_csv"],
            window_size=args.window_size,
        )
        sample_path = args.sample_dir / f"{scenario['scenario_name']}.npz"
        np.savez_compressed(
            sample_path,
            node_ids=np.asarray(sample.node_ids, dtype=str),
            adjacency=sample.adjacency,
            node_features=sample.node_features,
            structural_features=sample.structural_features,
            node_mask=sample.node_mask,
            origin_index=np.asarray(sample.origin_index),
            infection_times=sample.infection_times,
            observed_mask=sample.observed_mask,
            meta_json=np.asarray(json.dumps(sample.meta)),
        )
        rows.append(
            {
                "scenario_name": scenario["scenario_name"],
                "data_csv": str(scenario["data_csv"]),
                "ground_truth_csv": None if scenario["ground_truth_csv"] is None else str(scenario["ground_truth_csv"]),
                "n_nodes": int(sample.adjacency.shape[0]),
                "time_steps": int(sample.node_features.shape[0]),
                "origin_index": int(sample.origin_index),
                "beta_hat": float(sample.meta["beta_hat"]),
                "feature_contract": sample.meta["feature_contract"],
                "sample_path": str(sample_path),
            }
        )
        print(f"[ton_iot_temporal] built {scenario['scenario_name']}", flush=True)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
