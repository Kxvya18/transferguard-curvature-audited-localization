"""Evaluate independent training seeds on one fixed 60-cell test grid."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_synthetic_stress_grid import _load_model
from evaluation.metrics import evaluate_origin_localization
from synthetic.dataset import SyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig


def _neutralize(batch):
    structural = batch["structural_features"].clone(); structural[:, :, 1] = 1.0 / batch["node_mask"].sum(); structural[:, :, 2] = 1.0; structural[:, :, 5:7] = 0.0
    return {**batch, "structural_features": structural}


def _ci(values):
    array = np.asarray(values, dtype=float)
    return [float(np.mean(array) - 1.96 * np.std(array, ddof=1) / np.sqrt(len(array))), float(np.mean(array) + 1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))] if len(array) > 1 else [float(array[0]), float(array[0])]


def main() -> None:
    paths = {
        "hybrid": [PROJECT_ROOT / "outputs" / "train_transferguard_mixed_safe_prior_v4" / "transferguard_synthetic_best.pt"] + sorted((PROJECT_ROOT / "outputs" / "multiseed_v4").glob("hybrid_seed*/transferguard_synthetic_best.pt")),
        "temporal_only": [PROJECT_ROOT / "outputs" / "train_transferguard_mixed_no_prior_v4" / "transferguard_synthetic_best.pt"] + sorted((PROJECT_ROOT / "outputs" / "multiseed_v4").glob("temporal_seed*/transferguard_synthetic_best.pt")),
    }
    first = torch.load(paths["hybrid"][0], map_location="cpu", weights_only=False)
    base = SyntheticScenarioConfig(**first["scenario_config"])
    samples = []
    cell = 0
    for n, m, beta in product((100, 220, 400), (1, 2, 3, 4, 5), (0.5, 1.0, 1.5, 2.0)):
        config = replace(base, n_nodes=n, m=m, beta=beta, seed=610_000 + cell * 1009)
        samples.extend(SyntheticOriginDataset(config, 2, seed_offset=70_000).samples); cell += 1
    rows = []
    for method, checkpoints in paths.items():
        for path in checkpoints:
            model, checkpoint = _load_model(path, torch.device("cpu")); metrics = []
            for sample in samples:
                batch = collate_temporal_graph_samples([sample])
                if method == "temporal_only": batch = _neutralize(batch)
                with torch.no_grad(): out = model(batch["node_features"], batch["adjacency"], batch["structural_features"], batch["node_mask"], None)
                probs = out["probs"][0, : len(sample.node_ids)]
                metrics.append(evaluate_origin_localization({i: float(probs[i]) for i in range(len(probs))}, sample.origin_index))
            rows.append({"method": method, "seed": checkpoint["run_seed"], "mrr": float(np.mean([x["mrr"] for x in metrics])), "hit1": float(np.mean([x["hit@1"] for x in metrics]))})
            print(f"[multiseed] {method} seed={checkpoint['run_seed']}", flush=True)
    summary = []
    for method in paths:
        values = [row for row in rows if row["method"] == method]
        summary.append({"method": method, "n_training_seeds": len(values), "mean_mrr": float(np.mean([x["mrr"] for x in values])), "mrr_ci95_across_training_seeds": _ci([x["mrr"] for x in values]), "mean_hit1": float(np.mean([x["hit1"] for x in values])), "hit1_ci95_across_training_seeds": _ci([x["hit1"] for x in values])})
    output = PROJECT_ROOT / "outputs" / "multiseed_v4"; output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation_summary.json").open("w") as handle: json.dump({"n_fixed_test_samples": len(samples), "seed_rows": rows, "summary": summary}, handle, indent=2)


if __name__ == "__main__": main()
