"""Deterministic baseline audit for the blind-origin synthetic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate_origin_localization
from synthetic.dataset import SyntheticOriginDataset
from synthetic.iot_growth import SyntheticScenarioConfig


def _load_config(path: Path) -> tuple[SyntheticScenarioConfig, dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenario_keys = set(SyntheticScenarioConfig.__dataclass_fields__)
    scenario = SyntheticScenarioConfig(**{key: value for key, value in payload.items() if key in scenario_keys})
    training = {key: value for key, value in payload.items() if key not in scenario_keys}
    return scenario, training


def _rank_metrics(values: np.ndarray, origin_index: int) -> dict[str, float]:
    return evaluate_origin_localization(
        {idx: float(value) for idx, value in enumerate(values)},
        origin_index,
    )


def _score_sample(sample) -> dict[str, np.ndarray]:
    features = sample.node_features
    first_alert = np.full(features.shape[1], np.inf, dtype=float)
    for step in range(features.shape[0]):
        newly_visible = (features[step, :, 6] > 0.5) & ~np.isfinite(first_alert)
        first_alert[newly_visible] = step
    temporal_weights = np.linspace(1.0, 0.25, features.shape[0], dtype=float)[:, None]
    return {
        "degree": np.sum(sample.adjacency, axis=1),
        "boltzmann_prior": sample.structural_features[:, 1],
        "outbound_activity": np.sum(features[:, :, 2], axis=0),
        "activity_change": np.sum(features[:, :, 6] * temporal_weights, axis=0),
        "earliest_activity_change": -first_alert,
    }


def _uniform_expected(n_nodes: int) -> dict[str, float]:
    result = {
        "origin_rank": float((n_nodes + 1) / 2),
        "mrr": float(np.sum(1.0 / np.arange(1, n_nodes + 1)) / n_nodes),
    }
    for k_value in (1, 3, 5, 10):
        result[f"hit@{k_value}"] = min(k_value / n_nodes, 1.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_transferguard_synthetic.json",
    )
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_blind_origin_baselines",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario, training = _load_config(args.config)
    dataset = SyntheticOriginDataset(
        scenario,
        num_samples=args.samples,
        seed_offset=50_000,
        progress_label="blind_origin_baseline",
    )

    rows = []
    source_high_change = []
    for sample_idx, sample in enumerate(dataset.samples):
        source_high_change.append(float(np.max(sample.node_features[:, sample.origin_index, 6]) > 0.5))
        for method, values in _score_sample(sample).items():
            rows.append(
                {
                    "sample": sample_idx,
                    "method": method,
                    **_rank_metrics(values, sample.origin_index),
                }
            )
        rows.append(
            {
                "sample": sample_idx,
                "method": "uniform_expected",
                **_uniform_expected(len(sample.node_ids)),
            }
        )

    metrics_path = args.output_dir / "sample_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                **{
                    f"mean_{metric}": float(np.mean([row[metric] for row in method_rows]))
                    for metric in ("origin_rank", "mrr", "hit@1", "hit@3", "hit@5", "hit@10")
                },
            }
        )
    summary = {
        "feature_contract": "observable_v4_cross_dataset_aligned",
        "n_samples": args.samples,
        "source_high_activity_change_rate": float(np.mean(source_high_change)),
        "scenario_config": scenario.to_dict(),
        "run_seed": int(training.get("run_seed", 20260810)),
        "aggregate": aggregate,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved blind-origin baseline audit to {summary_path}")


if __name__ == "__main__":
    main()
