"""Profile parameter count, construction cost, latency, and resident memory."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_synthetic_stress_grid import _load_model
from synthetic.dataset import SyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0))


def main() -> None:
    device = torch.device("cpu")
    checkpoints = {
        "safe_hybrid": PROJECT_ROOT / "outputs" / "train_transferguard_mixed_safe_prior_v4" / "transferguard_synthetic_best.pt",
        "temporal_only": PROJECT_ROOT / "outputs" / "train_transferguard_mixed_no_prior_v4" / "transferguard_synthetic_best.pt",
    }
    rows = []
    for model_name, path in checkpoints.items():
        model, checkpoint = _load_model(path, device)
        base = SyntheticScenarioConfig(**checkpoint["scenario_config"])
        for n_nodes in (100, 220, 400):
            started = time.perf_counter()
            dataset = SyntheticOriginDataset(replace(base, n_nodes=n_nodes, seed=520_000 + n_nodes), 1)
            construction_ms = (time.perf_counter() - started) * 1000.0
            batch = collate_temporal_graph_samples([dataset[0]])
            kwargs = {"node_features": batch["node_features"], "adjacency": batch["adjacency"], "structural_features": batch["structural_features"], "node_mask": batch["node_mask"], "ssl_mask": None}
            model.eval()
            with torch.no_grad():
                for _ in range(5): model(**kwargs)
                timings = []
                for _ in range(50):
                    begin = time.perf_counter(); model(**kwargs); timings.append((time.perf_counter() - begin) * 1000.0)
            rows.append({
                "model": model_name,
                "n_nodes": n_nodes,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "construction_ms": construction_ms,
                "latency_median_ms": float(np.median(timings)),
                "latency_p95_ms": float(np.quantile(timings, 0.95)),
                "peak_process_rss_mb": _rss_mb(),
            })
    output = PROJECT_ROOT / "outputs" / "profiling_v4"; output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w") as handle: json.dump({"device": "CPU", "warmup": 5, "timed_repeats": 50, "rows": rows}, handle, indent=2)
    print(f"Saved profiling to {output}")


if __name__ == "__main__": main()
