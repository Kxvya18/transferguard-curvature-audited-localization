"""Evaluate quarantine rankings when surviving attackers reroute adaptively."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_intervention_utility import (
    _intervention_risk_scores,
    _load_model,
    _model_scores,
    _ranked_nodes,
)
from interventions.adaptive_spread import evaluate_adaptive_quarantine
from synthetic.dataset import build_temporal_sample
from synthetic.iot_growth import SyntheticScenarioConfig, generate_iot_graph, simulate_attack_cascade


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--decision-time", type=int, default=1)
    parser.add_argument("--adaptive-boost", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "adaptive_intervention_v4")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid, checkpoint = _load_model(PROJECT_ROOT / "outputs" / "train_transferguard_mixed_safe_prior_v4" / "transferguard_synthetic_best.pt", device)
    temporal, _ = _load_model(PROJECT_ROOT / "outputs" / "train_transferguard_mixed_no_prior_v4" / "transferguard_synthetic_best.pt", device)
    base = SyntheticScenarioConfig(**checkpoint["scenario_config"])
    rows = []; scenario = 0
    for n_nodes, m_value, beta_value in product((100, 220, 400), (1, 2, 3, 4, 5), (0.5, 1.0, 1.5, 2.0)):
        for replicate in range(args.samples_per_cell):
            config = replace(base, n_nodes=n_nodes, m=m_value, beta=beta_value, seed=410_000 + scenario * 101 + replicate)
            graph, history = generate_iot_graph(config)
            cascade = simulate_attack_cascade(graph, config, seed=config.seed + 1001)
            sample = build_temporal_sample(graph, history, cascade, config=config)
            windows = args.decision_time + 1
            temporal_scores = _model_scores(temporal, sample, device, windows, True)
            hybrid_scores = _model_scores(hybrid, sample, device, windows, False)
            activity = np.max(sample.node_features[:windows, :, 6], axis=0)
            methods = {
                "temporal_source": temporal_scores,
                "safe_hybrid_source": hybrid_scores,
                "degree": sample.adjacency.sum(axis=1),
                "outbound_activity": sample.node_features[:windows, :, 2].sum(axis=0),
                "decision_aware": _intervention_risk_scores(graph, temporal_scores, activity, args.decision_time),
            }
            for method, scores in methods.items():
                ranking = _ranked_nodes(scores)
                for k_value in (1, 3, 5):
                    outcome = evaluate_adaptive_quarantine(
                        graph, config, cascade.origin, ranking[:k_value], args.decision_time,
                        seed=config.seed + 90_001, adaptive_boost=args.adaptive_boost,
                    )
                    rows.append({"scenario": scenario, "n": n_nodes, "m": m_value, "beta_true": beta_value, "method": method, "k": k_value, **outcome})
            scenario += 1
        print(f"[adaptive] n={n_nodes} m={m_value} beta={beta_value:g}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(args.output_dir / "sample_metrics.csv", index=False)
    aggregate = frame.groupby(["method", "k"], as_index=False).agg(
        mean_net_prevented=("net_prevented_fraction", "mean"),
        mean_gross_prevented=("gross_prevented_fraction", "mean"),
        mean_rerouted_nodes=("rerouted_new_future", "mean"),
    )
    paired = {}
    for k_value in (1, 3, 5):
        wide = frame[frame.k == k_value].pivot(index="scenario", columns="method", values="net_prevented_fraction")
        for competitor in ("temporal_source", "degree", "outbound_activity"):
            delta = (wide["decision_aware"] - wide[competitor]).to_numpy()
            rng = np.random.default_rng(20260900 + k_value)
            boot = np.asarray([np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(20_000)])
            paired[f"k{k_value}:decision_aware-{competitor}"] = {"mean_delta": float(np.mean(delta)), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump({"n_scenarios": scenario, "adaptive_boost": args.adaptive_boost, "aggregate": aggregate.to_dict(orient="records"), "paired_bootstrap": paired}, handle, indent=2)
    print(f"Saved adaptive intervention results to {args.output_dir}")


if __name__ == "__main__":
    main()
