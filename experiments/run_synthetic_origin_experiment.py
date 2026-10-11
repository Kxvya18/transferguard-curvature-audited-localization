"""Run the first end-to-end synthetic origin-localization benchmark."""

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

from evaluation.metrics import evaluate_early_compromise_ranking, evaluate_origin_localization
from fusion.simple_fusion import hybrid_origin_scores
from priors.boltzmann_prior import (
    graph_concentration_features,
    mle_beta,
    origin_prior_distribution,
)
from synthetic.iot_growth import SyntheticScenarioConfig, generate_iot_graph, simulate_attack_cascade


def _load_config(path: Path) -> SyntheticScenarioConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SyntheticScenarioConfig(**payload)


def _degree_scores(graph) -> dict[int, float]:
    return {int(node): float(graph.degree(node)) for node in graph.nodes()}


def _mean_dict(rows: list[dict], prefix: str) -> dict[str, float]:
    keys = [key for key in rows[0].keys() if key.startswith(prefix)]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run(config: SyntheticScenarioConfig, seeds: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict] = []

    for offset in range(seeds):
        local_cfg = SyntheticScenarioConfig(**config.to_dict())
        local_cfg.seed = int(config.seed + 17 * offset)
        graph, history = generate_iot_graph(local_cfg)
        cascade = simulate_attack_cascade(graph, local_cfg, seed=local_cfg.seed + 999)

        beta_est = mle_beta(history)
        prior_probs = origin_prior_distribution(graph, beta=beta_est.beta_hat, m=local_cfg.m)
        hybrid_scores = hybrid_origin_scores(
            graph,
            prior_probs=prior_probs,
            observed_infected=cascade.observed_infected,
            alpha=0.65,
        )
        degree_scores = _degree_scores(graph)

        prior_metrics = evaluate_origin_localization(prior_probs, cascade.origin)
        prior_early = evaluate_early_compromise_ranking(prior_probs, cascade.infection_times)
        hybrid_metrics = evaluate_origin_localization(hybrid_scores, cascade.origin)
        hybrid_early = evaluate_early_compromise_ranking(hybrid_scores, cascade.infection_times)
        degree_metrics = evaluate_origin_localization(degree_scores, cascade.origin)
        degree_early = evaluate_early_compromise_ranking(degree_scores, cascade.infection_times)
        concentration = graph_concentration_features(graph, beta=beta_est.beta_hat, m=local_cfg.m)

        trial_rows.append(
            {
                "seed": float(local_cfg.seed),
                "n_nodes": float(graph.number_of_nodes()),
                "n_edges": float(graph.number_of_edges()),
                "origin": float(cascade.origin),
                "infected_count": float(len(cascade.infection_times)),
                "observed_infected_count": float(len(cascade.observed_infected)),
                "beta_true": float(local_cfg.beta),
                "beta_hat": float(beta_est.beta_hat),
                "beta_ci_lower": float(beta_est.ci_lower),
                "beta_ci_upper": float(beta_est.ci_upper),
                "beta_fisher": float(beta_est.fisher_information),
                "prior_origin_rank": float(prior_metrics["origin_rank"]),
                "prior_mrr": float(prior_metrics["mrr"]),
                "hybrid_origin_rank": float(hybrid_metrics["origin_rank"]),
                "hybrid_mrr": float(hybrid_metrics["mrr"]),
                "degree_origin_rank": float(degree_metrics["origin_rank"]),
                "degree_mrr": float(degree_metrics["mrr"]),
                "prior_hit@1": float(prior_metrics["hit@1"]),
                "hybrid_hit@1": float(hybrid_metrics["hit@1"]),
                "degree_hit@1": float(degree_metrics["hit@1"]),
                "prior_hit@5": float(prior_metrics["hit@5"]),
                "hybrid_hit@5": float(hybrid_metrics["hit@5"]),
                "degree_hit@5": float(degree_metrics["hit@5"]),
                "prior_early_precision@5": float(prior_early["early_precision@5"]),
                "hybrid_early_precision@5": float(hybrid_early["early_precision@5"]),
                "degree_early_precision@5": float(degree_early["early_precision@5"]),
                "prior_pairwise_order_acc": float(prior_early["pairwise_order_acc"]),
                "hybrid_pairwise_order_acc": float(hybrid_early["pairwise_order_acc"]),
                "degree_pairwise_order_acc": float(degree_early["pairwise_order_acc"]),
                **concentration,
            }
        )

    csv_path = out_dir / "synthetic_origin_trials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    summary = {
        "config": config.to_dict(),
        "n_trials": len(trial_rows),
        "beta_hat_mean": float(np.mean([row["beta_hat"] for row in trial_rows])),
        "prior_mrr_mean": float(np.mean([row["prior_mrr"] for row in trial_rows])),
        "hybrid_mrr_mean": float(np.mean([row["hybrid_mrr"] for row in trial_rows])),
        "degree_mrr_mean": float(np.mean([row["degree_mrr"] for row in trial_rows])),
        "prior_hit1_mean": float(np.mean([row["prior_hit@1"] for row in trial_rows])),
        "hybrid_hit1_mean": float(np.mean([row["hybrid_hit@1"] for row in trial_rows])),
        "degree_hit1_mean": float(np.mean([row["degree_hit@1"] for row in trial_rows])),
        "prior_early_precision5_mean": float(
            np.mean([row["prior_early_precision@5"] for row in trial_rows])
        ),
        "hybrid_early_precision5_mean": float(
            np.mean([row["hybrid_early_precision@5"] for row in trial_rows])
        ),
        "degree_early_precision5_mean": float(
            np.mean([row["degree_early_precision@5"] for row in trial_rows])
        ),
        "prior_pairwise_order_acc_mean": float(
            np.mean([row["prior_pairwise_order_acc"] for row in trial_rows])
        ),
        "hybrid_pairwise_order_acc_mean": float(
            np.mean([row["hybrid_pairwise_order_acc"] for row in trial_rows])
        ),
        "degree_pairwise_order_acc_mean": float(
            np.mean([row["degree_pairwise_order_acc"] for row in trial_rows])
        ),
    }

    with (out_dir / "synthetic_origin_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'synthetic_origin_summary.json'}")
    print("\nMean results")
    print(f"  beta_hat mean:        {summary['beta_hat_mean']:.3f}")
    print(f"  prior MRR:            {summary['prior_mrr_mean']:.3f}")
    print(f"  hybrid MRR:           {summary['hybrid_mrr_mean']:.3f}")
    print(f"  degree MRR:           {summary['degree_mrr_mean']:.3f}")
    print(f"  prior hit@1:          {summary['prior_hit1_mean']:.3f}")
    print(f"  hybrid hit@1:         {summary['hybrid_hit1_mean']:.3f}")
    print(f"  degree hit@1:         {summary['degree_hit1_mean']:.3f}")
    print(f"  prior early P@5:      {summary['prior_early_precision5_mean']:.3f}")
    print(f"  hybrid early P@5:     {summary['hybrid_early_precision5_mean']:.3f}")
    print(f"  degree early P@5:     {summary['degree_early_precision5_mean']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "synthetic_origin_small.json",
        help="Path to the experiment JSON config.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Number of random seeds / trials to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_origin_small",
        help="Directory for CSV and JSON outputs.",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    run(config=config, seeds=args.seeds, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

