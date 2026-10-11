"""Evaluate deterministic IoT-23 origin-localization baselines."""

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


def _load_sample(path: Path) -> dict:
    with np.load(path) as payload:
        return {
            "node_ids": payload["node_ids"].astype(str).tolist(),
            "adjacency": payload["adjacency"].copy(),
            "node_features": payload["node_features"].copy(),
            "structural_features": payload["structural_features"].copy(),
            "origin_index": int(payload["origin_index"]),
            "meta": json.loads(str(payload["meta_json"])),
        }


def _score_vectors(sample: dict) -> dict[str, np.ndarray]:
    node_features = sample["node_features"]
    structural = sample["structural_features"]
    vectors = {
        "degree": np.sum(sample["adjacency"], axis=1),
        "outbound_activity": np.sum(node_features[:, :, 2], axis=0),
        "activity_change": np.max(node_features[:, :, 6], axis=0),
        "boltzmann_prior": structural[:, 1],
        "directional_asymmetry": structural[:, 3],
    }
    if "first_seen_offsets" in sample["meta"]:
        first_seen = np.asarray(
            [np.inf if value is None else float(value) for value in sample["meta"]["first_seen_offsets"]],
            dtype=float,
        )
        vectors["earliest_seen"] = -first_seen
    return vectors


def _uniform_expected_metrics(n_nodes: int) -> dict[str, float]:
    harmonic = float(np.sum(1.0 / np.arange(1, n_nodes + 1)))
    result = {
        "origin_rank": float((n_nodes + 1) / 2.0),
        "mrr": harmonic / n_nodes,
    }
    for k_value in (1, 3, 5, 10):
        result[f"hit@{k_value}"] = min(float(k_value / n_nodes), 1.0)
    return result


def _bootstrap_mean_ci(values: list[float], seed: int = 20260810) -> tuple[float, float]:
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "iot23_temporal_samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "iot23_baselines",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.sample_dir.glob("*.npz")):
        sample = _load_sample(path)
        scenario = Path(sample["meta"]["scenario_path"]).stem
        score_vectors = _score_vectors(sample)
        for method, values in score_vectors.items():
            scores = {idx: float(value) for idx, value in enumerate(values)}
            metrics = evaluate_origin_localization(scores, sample["origin_index"])
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "n_nodes": len(sample["node_ids"]),
                    "origin_node_id": sample["meta"]["origin_node_id"],
                    **metrics,
                }
            )
        rows.append(
            {
                "scenario": scenario,
                "method": "uniform_expected",
                "n_nodes": len(sample["node_ids"]),
                "origin_node_id": sample["meta"]["origin_node_id"],
                **_uniform_expected_metrics(len(sample["node_ids"])),
            }
        )

    csv_path = args.output_dir / "scenario_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    methods = sorted({row["method"] for row in rows})
    aggregate = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        record = {"method": method, "n_scenarios": len(method_rows)}
        for metric in ("origin_rank", "mrr", "hit@1", "hit@3", "hit@5", "hit@10"):
            values = [float(row[metric]) for row in method_rows]
            low, high = _bootstrap_mean_ci(values)
            record[f"mean_{metric}"] = float(np.mean(values))
            record[f"ci95_low_{metric}"] = low
            record[f"ci95_high_{metric}"] = high
        aggregate.append(record)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_scenarios": len({row["scenario"] for row in rows}),
                "bootstrap_seed": 20260810,
                "bootstrap_resamples": 10_000,
                "warning": "IoT-23 labels identify malicious flows, not full compromise chains.",
                "aggregate": aggregate,
            },
            handle,
            indent=2,
        )
    print(f"Saved IoT-23 baseline metrics to {summary_path}")


if __name__ == "__main__":
    main()
