"""Build leakage-controlled IoT-23 temporal samples and persistent artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loaders.iot23_temporal import build_iot23_samples_from_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "iot23_extracted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "iot23_temporal_samples",
    )
    parser.add_argument("--max-log-bytes", type=int, default=500_000_000)
    parser.add_argument("--window-size", type=float, default=300.0)
    parser.add_argument("--pre-attack-steps", type=int, default=2)
    parser.add_argument("--post-attack-steps", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_iot23_samples_from_root(
        args.input_root,
        max_log_bytes=args.max_log_bytes,
        window_size=args.window_size,
        pre_attack_steps=args.pre_attack_steps,
        post_attack_steps=args.post_attack_steps,
        max_nodes=args.max_nodes,
    )
    summaries = []
    for index, sample in enumerate(samples):
        scenario = Path(sample.meta["scenario_path"]).parts[-3]
        output_path = args.output_dir / f"{index:02d}_{scenario}.npz"
        np.savez_compressed(
            output_path,
            node_ids=np.asarray(sample.node_ids, dtype=str),
            adjacency=sample.adjacency,
            node_features=sample.node_features,
            structural_features=sample.structural_features,
            node_mask=sample.node_mask,
            origin_index=np.asarray(sample.origin_index, dtype=np.int64),
            infection_times=sample.infection_times,
            observed_mask=sample.observed_mask,
            meta_json=np.asarray(json.dumps(sample.meta)),
        )
        summaries.append(
            {
                "scenario": scenario,
                "artifact": str(output_path),
                "n_nodes": len(sample.node_ids),
                "n_edges": int(np.sum(sample.adjacency) // 2),
                "origin_node_id": sample.meta["origin_node_id"],
                "origin_index": sample.origin_index,
                "beta_hat": sample.meta["beta_hat"],
                "beta_ci": sample.meta["beta_ci"],
                "beta_fisher_information": sample.meta["beta_fisher_information"],
                "beta_boundary_warning": sample.meta["beta_boundary_warning"],
                "prior_reliability": sample.meta["prior_reliability"],
                "attack_label_counts": sample.meta["attack_label_counts"],
            }
        )
        print(f"[iot23-sample] saved {scenario}: nodes={len(sample.node_ids)}", flush=True)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_contract": "observable_v4_cross_dataset_aligned",
                "labels_used_as_features": False,
                "n_samples": len(samples),
                "parameters": vars(args) | {"input_root": str(args.input_root), "output_dir": str(args.output_dir)},
                "samples": summaries,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Saved IoT-23 temporal sample summary to {summary_path}")


if __name__ == "__main__":
    main()
