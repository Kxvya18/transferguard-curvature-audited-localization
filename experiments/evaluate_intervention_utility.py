"""Test whether early origin rankings improve time-respecting containment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    evaluate_early_compromise_ranking,
    evaluate_origin_localization,
)
from interventions.containment import time_respecting_quarantine
from models.transferguard_model import TransferGuardModel
from synthetic.dataset import build_temporal_sample
from synthetic.iot_growth import (
    SyntheticScenarioConfig,
    generate_iot_graph,
    simulate_attack_cascade,
)


def _load_model(path: Path, device: torch.device) -> tuple[TransferGuardModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    training = checkpoint["training_config"]
    state = checkpoint["model_state_dict"]
    model = TransferGuardModel(
        node_feature_dim=int(state["encoder.node_proj.weight"].shape[1]),
        structural_feature_dim=int(state["encoder.struct_proj.weight"].shape[1]),
        hidden_dim=int(training.get("hidden_dim", 64)),
        num_message_layers=int(training.get("num_message_layers", 2)),
        dropout=float(training.get("dropout", 0.15)),
        fusion_type=str(training.get("fusion_type", "gated")),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def _model_scores(
    model: TransferGuardModel,
    sample,
    device: torch.device,
    time_windows: int,
    neutralize_prior: bool,
) -> np.ndarray:
    features = torch.from_numpy(sample.node_features[:time_windows]).unsqueeze(0).to(device)
    adjacency = torch.from_numpy(sample.adjacency).unsqueeze(0).to(device)
    structural = torch.from_numpy(sample.structural_features).unsqueeze(0).to(device).clone()
    mask = torch.from_numpy(sample.node_mask).unsqueeze(0).to(device)
    # Aggregate activity used all six windows during training; remove it here to
    # enforce a strict early-decision protocol rather than leak future traffic.
    structural[:, :, 4] = 0.0
    if neutralize_prior:
        structural[:, :, 1] = 1.0 / mask.sum()
        structural[:, :, 2] = 1.0
        structural[:, :, 5:7] = 0.0
    output = model(
        node_features=features,
        adjacency=adjacency,
        structural_features=structural,
        node_mask=mask,
        ssl_mask=None,
    )
    return output["probs"][0].cpu().numpy()


def _ranked_nodes(scores: np.ndarray) -> list[int]:
    return [int(value) for value in np.argsort(-np.asarray(scores), kind="stable")]


def _rank_probabilities(scores: np.ndarray, scale: float = 10.0) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), kind="stable")
    rank = np.empty(len(order), dtype=float)
    rank[order] = np.arange(len(order), dtype=float)
    weights = np.exp(-rank / max(scale, 1e-6))
    return weights / weights.sum()


def _expected_distance_scores(
    graph: nx.Graph,
    origin_probabilities: np.ndarray,
    top_origins: int = 20,
) -> np.ndarray:
    order = np.argsort(-origin_probabilities)[: min(top_origins, len(origin_probabilities))]
    weights = origin_probabilities[order]
    weights = weights / max(float(weights.sum()), 1e-12)
    max_distance = float(graph.number_of_nodes() + 1)
    expected = np.zeros(graph.number_of_nodes(), dtype=float)
    for weight, origin in zip(weights, order):
        lengths = nx.single_source_shortest_path_length(graph, int(origin))
        expected += float(weight) * np.asarray(
            [float(lengths.get(node, max_distance)) for node in range(graph.number_of_nodes())]
        )
    return -expected


def _intervention_risk_scores(
    graph: nx.Graph,
    origin_probabilities: np.ndarray,
    activity_change: np.ndarray,
    decision_time: int,
    top_origins: int = 20,
) -> np.ndarray:
    """Rank quarantine candidates by compromise belief times downstream impact.

    Compromise belief is the noisy-OR of observed activity change and posterior
    mass whose shortest-path frontier reaches the node at decision time. Impact
    is exponentially discounted graph reach. No attack labels or realized
    future infection times enter this score.
    """
    n_nodes = graph.number_of_nodes()
    origins = np.argsort(-origin_probabilities)[: min(top_origins, n_nodes)]
    weights = origin_probabilities[origins]
    weights = weights / max(float(weights.sum()), 1e-12)
    frontier = np.zeros(n_nodes, dtype=float)
    for weight, origin in zip(weights, origins):
        lengths = nx.single_source_shortest_path_length(
            graph, int(origin), cutoff=max(int(decision_time), 0)
        )
        for node, distance in lengths.items():
            if int(distance) == int(decision_time):
                frontier[int(node)] += float(weight)
    activity = np.asarray(activity_change, dtype=float)
    activity = activity / max(float(np.max(activity)), 1e-12)
    frontier = np.clip(frontier, 0.0, 1.0)
    compromise_belief = 1.0 - (1.0 - activity) * (1.0 - frontier)

    impact = np.zeros(n_nodes, dtype=float)
    for node in range(n_nodes):
        lengths = nx.single_source_shortest_path_length(graph, node, cutoff=4)
        impact[node] = sum(
            np.exp(-float(distance))
            for target, distance in lengths.items()
            if int(target) != node
        )
    impact /= max(float(np.max(impact)), 1e-12)
    return compromise_belief * impact


def _bootstrap_delta(values: np.ndarray, seed: int, resamples: int = 20_000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        means[idx] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return {
        "mean_delta": float(np.mean(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "train_transferguard_mixed_safe_prior_v4"
        / "transferguard_synthetic_best.pt",
    )
    parser.add_argument(
        "--no-prior-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "train_transferguard_mixed_no_prior_v4"
        / "transferguard_synthetic_best.pt",
    )
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--decision-time", type=int, default=1)
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "intervention_utility_v4",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid, checkpoint = _load_model(args.hybrid_checkpoint, device)
    temporal, _ = _load_model(args.no_prior_checkpoint, device)
    base = SyntheticScenarioConfig(**checkpoint["scenario_config"])
    intervention_rows: list[dict] = []
    early_rows: list[dict] = []
    scenario_index = 0
    rng = np.random.default_rng(20260813)

    for n_nodes, m_value, beta_value in product(
        (100, 220, 400), (1, 2, 3, 4, 5), (0.5, 1.0, 1.5, 2.0)
    ):
        for replicate in range(args.samples_per_cell):
            config = replace(
                base,
                n_nodes=n_nodes,
                m=m_value,
                beta=beta_value,
                seed=int(base.seed + 300_000 + scenario_index * 101 + replicate * 17),
            )
            graph, history = generate_iot_graph(config)
            cascade = simulate_attack_cascade(graph, config, seed=config.seed + 1001)
            sample = build_temporal_sample(graph, history, cascade, config=config)
            windows = min(args.decision_time + 1, sample.node_features.shape[0])
            hybrid_scores = _model_scores(hybrid, sample, device, windows, False)
            temporal_scores = _model_scores(temporal, sample, device, windows, True)
            prefix = sample.node_features[:windows]
            score_vectors = {
                "safe_hybrid": hybrid_scores,
                "temporal_only": temporal_scores,
                "activity_change": np.sum(prefix[:, :, 6], axis=0),
                "outbound_activity": np.sum(prefix[:, :, 2], axis=0),
                "degree": np.sum(sample.adjacency, axis=1),
                "boltzmann_prior": sample.structural_features[:, 1],
            }
            prefix_activity = np.max(prefix[:, :, 6], axis=0)
            score_vectors["intervention_risk_hybrid"] = _intervention_risk_scores(
                graph,
                hybrid_scores,
                prefix_activity,
                args.decision_time,
            )
            score_vectors["intervention_risk_temporal"] = _intervention_risk_scores(
                graph,
                temporal_scores,
                prefix_activity,
                args.decision_time,
            )
            oracle_scores = np.full(graph.number_of_nodes(), -1e6, dtype=float)
            for node, time in cascade.infection_times.items():
                oracle_scores[int(node)] = -float(time)
            score_vectors["oracle_early"] = oracle_scores

            for method, scores in score_vectors.items():
                origin_metrics = evaluate_origin_localization(
                    {idx: float(value) for idx, value in enumerate(scores)},
                    cascade.origin,
                )
                origin_probs = (
                    np.asarray(scores, dtype=float)
                    if method in {"safe_hybrid", "temporal_only"}
                    else _rank_probabilities(scores)
                )
                order_scores = _expected_distance_scores(graph, origin_probs)
                early_metrics = evaluate_early_compromise_ranking(
                    {idx: float(value) for idx, value in enumerate(order_scores)},
                    cascade.infection_times,
                )
                early_rows.append(
                    {
                        "scenario": scenario_index,
                        "n": n_nodes,
                        "m": m_value,
                        "beta_true": beta_value,
                        "method": method,
                        **origin_metrics,
                        **early_metrics,
                    }
                )
                ranking = _ranked_nodes(scores)
                for k_value in (1, 3, 5):
                    outcome = time_respecting_quarantine(
                        graph,
                        cascade.infection_times,
                        ranking[:k_value],
                        args.decision_time,
                    )
                    intervention_rows.append(
                        {
                            "scenario": scenario_index,
                            "n": n_nodes,
                            "m": m_value,
                            "beta_true": beta_value,
                            "method": method,
                            "k": k_value,
                            "random_replicate": -1,
                            **outcome,
                        }
                    )
            nodes = np.arange(graph.number_of_nodes())
            for random_replicate in range(args.random_replicates):
                random_order = rng.permutation(nodes)
                for k_value in (1, 3, 5):
                    outcome = time_respecting_quarantine(
                        graph,
                        cascade.infection_times,
                        random_order[:k_value],
                        args.decision_time,
                    )
                    intervention_rows.append(
                        {
                            "scenario": scenario_index,
                            "n": n_nodes,
                            "m": m_value,
                            "beta_true": beta_value,
                            "method": "random",
                            "k": k_value,
                            "random_replicate": random_replicate,
                            **outcome,
                        }
                    )
            scenario_index += 1
        print(f"[intervention] completed n={n_nodes}, m={m_value}, beta={beta_value:g}", flush=True)

    with (args.output_dir / "sample_interventions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(intervention_rows[0]))
        writer.writeheader()
        writer.writerows(intervention_rows)
    with (args.output_dir / "sample_early_order.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(early_rows[0]))
        writer.writeheader()
        writer.writerows(early_rows)

    import pandas as pd

    intervention = pd.DataFrame(intervention_rows)
    per_scenario = (
        intervention.groupby(["scenario", "method", "k"], as_index=False)
        .agg(prevented_fraction=("prevented_fraction", "mean"))
    )
    aggregate = (
        per_scenario.groupby(["method", "k"], as_index=False)
        .agg(
            mean_prevented_fraction=("prevented_fraction", "mean"),
            std_prevented_fraction=("prevented_fraction", "std"),
        )
        .to_dict(orient="records")
    )
    early = pd.DataFrame(early_rows)
    early_aggregate = (
        early.groupby("method", as_index=False)
        .agg(
            origin_mrr=("mrr", "mean"),
            origin_hit1=("hit@1", "mean"),
            early_precision3=("early_precision@3", "mean"),
            early_precision5=("early_precision@5", "mean"),
            pairwise_order_accuracy=("pairwise_order_acc", "mean"),
        )
        .to_dict(orient="records")
    )
    paired_tests = {}
    for k_value in (1, 3, 5):
        wide = per_scenario[per_scenario["k"] == k_value].pivot(
            index="scenario", columns="method", values="prevented_fraction"
        )
        for left, right in (
            ("temporal_only", "activity_change"),
            ("safe_hybrid", "temporal_only"),
            ("temporal_only", "random"),
        ):
            delta = (wide[left] - wide[right]).dropna().to_numpy()
            paired_tests[f"k{k_value}:{left}-{right}"] = _bootstrap_delta(
                delta, seed=20260814 + k_value
            )
    summary = {
        "feature_contract": checkpoint.get("feature_contract", "unknown"),
        "decision_time": args.decision_time,
        "time_windows_used": args.decision_time + 1,
        "future_activity_structural_feature_removed": True,
        "n_scenarios": scenario_index,
        "n_values": [100, 220, 400],
        "m_values": [1, 2, 3, 4, 5],
        "beta_values": [0.5, 1.0, 1.5, 2.0],
        "intervention_aggregate": aggregate,
        "early_order_aggregate": early_aggregate,
        "paired_bootstrap": paired_tests,
        "interpretation_boundary": (
            "Retrospective containment on one fixed temporal cascade realization; "
            "it does not model adaptive attacker rerouting after quarantine."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved intervention and early-order evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
