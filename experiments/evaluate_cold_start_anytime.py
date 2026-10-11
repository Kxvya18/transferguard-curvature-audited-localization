"""Evaluate the pre-specified cold-start hypothesis on a fixed 60-cell grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import evaluate_origin_localization
from experiments.cold_start_common import load_cold_start_model, prepare_prefix_batch
from synthetic.dataset import SyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig

HORIZONS = (1, 2, 3, 6)
RETENTIONS = (0.25, 0.5, 1.0)
LOW_CONDITIONS = {(1, 0.25), (1, 0.5), (2, 0.25), (2, 0.5)}
FULL_CONDITION = (6, 1.0)


def _ci(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return [float(array[0]), float(array[0])]
    half = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return [float(np.mean(array) - half), float(np.mean(array) + half)]


def _bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for index in range(draws):
        means[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _bootstrap_difference_ci(
    first: np.ndarray, second: np.ndarray, seed: int, draws: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=float)
    for index in range(draws):
        first_draw = rng.choice(first, size=len(first), replace=True)
        second_draw = rng.choice(second, size=len(second), replace=True)
        differences[index] = float(np.mean(first_draw) - np.mean(second_draw))
    return [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ]


def _prepare_test_cache(checkpoint: dict, path: Path, seed_base: int) -> list:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    base = SyntheticScenarioConfig(**checkpoint["scenario_config"])
    samples = []
    cell = 0
    sample_index = 0
    for n_nodes, m_value, beta in product(
        (100, 220, 400), (1, 2, 3, 4, 5), (0.5, 1.0, 1.5, 2.0)
    ):
        config = replace(
            base,
            n_nodes=n_nodes,
            m=m_value,
            beta=beta,
            seed=int(seed_base) + cell * 1009,
        )
        generated = SyntheticOriginDataset(config, 2, seed_offset=80_000).samples
        for sample in generated:
            sample.meta["test_index"] = sample_index
            sample.meta["test_n"] = n_nodes
            sample.meta["test_m"] = m_value
            sample.meta["test_beta"] = beta
            samples.append(sample)
            sample_index += 1
        cell += 1
        print(f"[cold-test] generated cell {cell}/60", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)
    return samples


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _score_map(values: np.ndarray) -> dict[int, float]:
    return {index: float(value) for index, value in enumerate(values)}


@torch.no_grad()
def _evaluate_checkpoint(
    path: Path,
    samples: list,
    device: torch.device,
    batch_size: int,
    method_alias: str | None = None,
) -> list[dict]:
    model, checkpoint = load_cold_start_model(path, device)
    model.eval()
    checkpoint_method = str(checkpoint["method"])
    method = checkpoint_method if method_alias is None else method_alias
    seed = int(checkpoint["run_seed"])
    rows = []
    for horizon, retention in product(HORIZONS, RETENTIONS):
        for start, chunk in _chunks(samples, batch_size):
            raw = collate_temporal_graph_samples(chunk)
            batch = prepare_prefix_batch(
                raw,
                horizon,
                retention,
                mask_seed=9_700_000 + horizon * 100_000 + int(retention * 10_000) + start,
                ablate_prior=checkpoint_method == "temporal_only",
            )
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(
                batch["node_features"],
                batch["adjacency"],
                batch["structural_features"],
                batch["node_mask"],
                None,
                batch["observation_context"],
            )
            for local_index, sample in enumerate(chunk):
                n_nodes = len(sample.node_ids)
                probability = output["probs"][local_index, :n_nodes].cpu().numpy()
                metrics = evaluate_origin_localization(
                    _score_map(probability), sample.origin_index
                )
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "sample": int(sample.meta["test_index"]),
                        "n_nodes": int(sample.meta["test_n"]),
                        "m": int(sample.meta["test_m"]),
                        "beta": float(sample.meta["test_beta"]),
                        "horizon": horizon,
                        "telemetry_retention": retention,
                        "mrr": float(metrics["mrr"]),
                        "hit1": float(metrics["hit@1"]),
                        "origin_rank": float(metrics["origin_rank"]),
                        "prior_strength": float(output["prior_strength"][local_index].cpu()),
                        "prior_applicability": float(output["prior_applicability"][local_index].cpu()),
                        "fisher_reliability": float(output["fisher_reliability"][local_index].cpu()),
                    }
                )
        print(
            f"[cold-eval] {method} seed={seed} h={horizon} retention={retention}",
            flush=True,
        )
    return rows


def _evaluate_baselines(samples: list, batch_size: int) -> list[dict]:
    rows = []
    for horizon, retention in product(HORIZONS, RETENTIONS):
        for start, chunk in _chunks(samples, batch_size):
            batch = prepare_prefix_batch(
                collate_temporal_graph_samples(chunk),
                horizon,
                retention,
                mask_seed=9_700_000 + horizon * 100_000 + int(retention * 10_000) + start,
                ablate_prior=False,
            )
            for local_index, sample in enumerate(chunk):
                n_nodes = len(sample.node_ids)
                score_sets = {
                    "boltzmann_prior": batch["structural_features"][local_index, :n_nodes, 1].numpy(),
                    "degree": batch["structural_features"][local_index, :n_nodes, 0].numpy(),
                    "activity_change": batch["node_features"][local_index, :, :n_nodes, 6].amax(dim=0).numpy(),
                }
                for method, scores in score_sets.items():
                    metrics = evaluate_origin_localization(_score_map(scores), sample.origin_index)
                    rows.append(
                        {
                            "method": method,
                            "sample": int(sample.meta["test_index"]),
                            "horizon": horizon,
                            "telemetry_retention": retention,
                            "mrr": float(metrics["mrr"]),
                            "hit1": float(metrics["hit@1"]),
                        }
                    )
    return rows


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _summarize(
    model_rows: list[dict],
    baseline_rows: list[dict],
    reliability_thresholds: list[float],
) -> dict:
    seed_aggregate = []
    seeds = sorted({int(row["seed"]) for row in model_rows})
    for method in ("hybrid", "temporal_only"):
        for seed in seeds:
            for horizon, retention in product(HORIZONS, RETENTIONS):
                subset = [
                    row for row in model_rows
                    if row["method"] == method and row["seed"] == seed
                    and row["horizon"] == horizon
                    and row["telemetry_retention"] == retention
                ]
                seed_aggregate.append(
                    {
                        "method": method,
                        "seed": seed,
                        "horizon": horizon,
                        "telemetry_retention": retention,
                        "mrr": _mean(subset, "mrr"),
                        "hit1": _mean(subset, "hit1"),
                        "mean_prior_strength": _mean(subset, "prior_strength"),
                    }
                )

    condition_summary = []
    paired_summary = []
    for horizon, retention in product(HORIZONS, RETENTIONS):
        for method in ("hybrid", "temporal_only"):
            subset = [
                row for row in seed_aggregate
                if row["method"] == method and row["horizon"] == horizon
                and row["telemetry_retention"] == retention
            ]
            condition_summary.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "telemetry_retention": retention,
                    "mean_mrr": _mean(subset, "mrr"),
                    "mrr_ci95_training_seeds": _ci([row["mrr"] for row in subset]),
                    "mean_hit1": _mean(subset, "hit1"),
                    "mean_prior_strength": _mean(subset, "mean_prior_strength"),
                }
            )
        graph_deltas = []
        for sample in sorted({int(row["sample"]) for row in model_rows}):
            differences = []
            for seed in seeds:
                hybrid = next(
                    row for row in model_rows
                    if row["method"] == "hybrid" and row["seed"] == seed
                    and row["sample"] == sample and row["horizon"] == horizon
                    and row["telemetry_retention"] == retention
                )
                temporal = next(
                    row for row in model_rows
                    if row["method"] == "temporal_only" and row["seed"] == seed
                    and row["sample"] == sample and row["horizon"] == horizon
                    and row["telemetry_retention"] == retention
                )
                differences.append(hybrid["mrr"] - temporal["mrr"])
            graph_deltas.append(float(np.mean(differences)))
        paired_summary.append(
            {
                "horizon": horizon,
                "telemetry_retention": retention,
                "mean_delta_mrr": float(np.mean(graph_deltas)),
                "paired_graph_bootstrap_ci95": _bootstrap_mean_ci(
                    np.asarray(graph_deltas), 4_200 + horizon * 10 + int(retention * 10)
                ),
            }
        )

    baseline_summary = []
    for method in ("boltzmann_prior", "degree", "activity_change"):
        for horizon, retention in product(HORIZONS, RETENTIONS):
            subset = [
                row for row in baseline_rows
                if row["method"] == method and row["horizon"] == horizon
                and row["telemetry_retention"] == retention
            ]
            baseline_summary.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "telemetry_retention": retention,
                    "mean_mrr": _mean(subset, "mrr"),
                    "mean_hit1": _mean(subset, "hit1"),
                }
            )

    regime_seed_rows = []
    for method in ("hybrid", "temporal_only"):
        for seed in seeds:
            method_seed = [
                row for row in model_rows if row["method"] == method and row["seed"] == seed
            ]
            low = [
                row for row in method_seed
                if (row["horizon"], row["telemetry_retention"]) in LOW_CONDITIONS
            ]
            full = [
                row for row in method_seed
                if (row["horizon"], row["telemetry_retention"]) == FULL_CONDITION
            ]
            regime_seed_rows.extend(
                [
                    {"method": method, "seed": seed, "regime": "low_evidence", "mrr": _mean(low, "mrr"), "hit1": _mean(low, "hit1"), "prior_strength": _mean(low, "prior_strength")},
                    {"method": method, "seed": seed, "regime": "full_evidence", "mrr": _mean(full, "mrr"), "hit1": _mean(full, "hit1"), "prior_strength": _mean(full, "prior_strength")},
                ]
            )

    regime_summary = []
    for method in ("hybrid", "temporal_only"):
        for regime in ("low_evidence", "full_evidence"):
            subset = [row for row in regime_seed_rows if row["method"] == method and row["regime"] == regime]
            regime_summary.append(
                {
                    "method": method,
                    "regime": regime,
                    "mean_mrr": _mean(subset, "mrr"),
                    "mrr_ci95_training_seeds": _ci([row["mrr"] for row in subset]),
                    "mean_hit1": _mean(subset, "hit1"),
                    "mean_prior_strength": _mean(subset, "prior_strength"),
                }
            )

    def regime_value(method: str, regime: str, field: str) -> float:
        return next(row[field] for row in regime_summary if row["method"] == method and row["regime"] == regime)

    low_delta = regime_value("hybrid", "low_evidence", "mean_mrr") - regime_value("temporal_only", "low_evidence", "mean_mrr")
    full_delta = regime_value("hybrid", "full_evidence", "mean_mrr") - regime_value("temporal_only", "full_evidence", "mean_mrr")
    per_seed_interactions = []
    for seed in seeds:
        values = {
            (row["method"], row["regime"]): row["mrr"]
            for row in regime_seed_rows if row["seed"] == seed
        }
        per_seed_interactions.append(
            (values[("hybrid", "low_evidence")] - values[("temporal_only", "low_evidence")])
            - (values[("hybrid", "full_evidence")] - values[("temporal_only", "full_evidence")])
        )

    low_graph_deltas = []
    full_graph_deltas = []
    for sample in sorted({int(row["sample"]) for row in model_rows}):
        low_seed_deltas = []
        full_seed_deltas = []
        for seed in seeds:
            for target, conditions in (
                (low_seed_deltas, LOW_CONDITIONS),
                (full_seed_deltas, {FULL_CONDITION}),
            ):
                hybrid_values = [
                    row["mrr"] for row in model_rows
                    if row["method"] == "hybrid" and row["seed"] == seed
                    and row["sample"] == sample
                    and (row["horizon"], row["telemetry_retention"]) in conditions
                ]
                temporal_values = [
                    row["mrr"] for row in model_rows
                    if row["method"] == "temporal_only" and row["seed"] == seed
                    and row["sample"] == sample
                    and (row["horizon"], row["telemetry_retention"]) in conditions
                ]
                target.append(float(np.mean(hybrid_values) - np.mean(temporal_values)))
        low_graph_deltas.append(float(np.mean(low_seed_deltas)))
        full_graph_deltas.append(float(np.mean(full_seed_deltas)))
    low_delta_ci = _bootstrap_mean_ci(np.asarray(low_graph_deltas), 6_101)
    full_delta_ci = _bootstrap_mean_ci(np.asarray(full_graph_deltas), 6_102)

    hybrid_rows = [row for row in model_rows if row["method"] == "hybrid"]
    reliabilities = {
        sample: _mean([row for row in hybrid_rows if row["sample"] == sample], "fisher_reliability")
        for sample in sorted({int(row["sample"]) for row in hybrid_rows})
    }
    boundaries = np.asarray(reliability_thresholds, dtype=float)
    sample_quartiles = {
        sample: int(np.searchsorted(boundaries, value, side="right"))
        for sample, value in reliabilities.items()
    }
    reliability_effect = []
    quartile_graph_deltas = {}
    for quartile in range(4):
        selected = {
            sample for sample, assigned in sample_quartiles.items() if assigned == quartile
        }
        deltas = []
        for sample in selected:
            sample_seed_deltas = []
            for seed in seeds:
                h_values = [row["mrr"] for row in model_rows if row["method"] == "hybrid" and row["seed"] == seed and row["sample"] == sample and (row["horizon"], row["telemetry_retention"]) in LOW_CONDITIONS]
                t_values = [row["mrr"] for row in model_rows if row["method"] == "temporal_only" and row["seed"] == seed and row["sample"] == sample and (row["horizon"], row["telemetry_retention"]) in LOW_CONDITIONS]
                sample_seed_deltas.append(float(np.mean(h_values) - np.mean(t_values)))
            deltas.append(float(np.mean(sample_seed_deltas)))
        quartile_graph_deltas[quartile + 1] = np.asarray(deltas, dtype=float)
        reliability_effect.append(
            {
                "quartile": quartile + 1,
                "n_graphs": len(deltas),
                "mean_reliability": float(np.mean([reliabilities[sample] for sample in selected])),
                "mean_delta_mrr_low_evidence": float(np.mean(deltas)),
                "bootstrap_ci95": _bootstrap_mean_ci(np.asarray(deltas), 7_300 + quartile),
            }
        )

    reliability_interaction = {
        "q4_minus_q1_delta_mrr": float(
            np.mean(quartile_graph_deltas[4]) - np.mean(quartile_graph_deltas[1])
        ),
        "graph_clustered_bootstrap_ci95": _bootstrap_difference_ci(
            quartile_graph_deltas[4], quartile_graph_deltas[1], 7_401
        ),
    }

    selection_threshold = float(reliability_thresholds[2])
    selected_samples = {
        sample for sample, value in reliabilities.items() if value > selection_threshold
    }
    selective_policy = {}
    for regime, conditions in (
        ("low_evidence", LOW_CONDITIONS),
        ("full_evidence", {FULL_CONDITION}),
    ):
        graph_deltas = []
        graph_hit1_deltas = []
        for sample in sorted(reliabilities):
            if sample not in selected_samples:
                graph_deltas.append(0.0)
                graph_hit1_deltas.append(0.0)
                continue
            seed_deltas = []
            seed_hit1_deltas = []
            for seed in seeds:
                hybrid_values = [
                    row["mrr"] for row in model_rows
                    if row["method"] == "hybrid" and row["seed"] == seed
                    and row["sample"] == sample
                    and (row["horizon"], row["telemetry_retention"]) in conditions
                ]
                temporal_values = [
                    row["mrr"] for row in model_rows
                    if row["method"] == "temporal_only" and row["seed"] == seed
                    and row["sample"] == sample
                    and (row["horizon"], row["telemetry_retention"]) in conditions
                ]
                seed_deltas.append(float(np.mean(hybrid_values) - np.mean(temporal_values)))
                hybrid_hit1 = [
                    row["hit1"] for row in model_rows
                    if row["method"] == "hybrid" and row["seed"] == seed
                    and row["sample"] == sample
                    and (row["horizon"], row["telemetry_retention"]) in conditions
                ]
                temporal_hit1 = [
                    row["hit1"] for row in model_rows
                    if row["method"] == "temporal_only" and row["seed"] == seed
                    and row["sample"] == sample
                    and (row["horizon"], row["telemetry_retention"]) in conditions
                ]
                seed_hit1_deltas.append(
                    float(np.mean(hybrid_hit1) - np.mean(temporal_hit1))
                )
            graph_deltas.append(float(np.mean(seed_deltas)))
            graph_hit1_deltas.append(float(np.mean(seed_hit1_deltas)))
        temporal_mrr = regime_value("temporal_only", regime, "mean_mrr")
        temporal_hit1 = regime_value("temporal_only", regime, "mean_hit1")
        delta = float(np.mean(graph_deltas))
        hit1_delta = float(np.mean(graph_hit1_deltas))
        selective_policy[regime] = {
            "temporal_reference_mrr": temporal_mrr,
            "selective_policy_mrr": temporal_mrr + delta,
            "delta_mrr": delta,
            "graph_clustered_bootstrap_ci95": _bootstrap_mean_ci(
                np.asarray(graph_deltas), 8_200 + len(selective_policy)
            ),
            "temporal_reference_hit1": temporal_hit1,
            "selective_policy_hit1": temporal_hit1 + hit1_delta,
            "delta_hit1": hit1_delta,
            "hit1_graph_clustered_bootstrap_ci95": _bootstrap_mean_ci(
                np.asarray(graph_hit1_deltas), 8_300 + len(selective_policy)
            ),
        }
    selective_policy.update(
        {
            "rule": "Use hybrid only when Fisher reliability exceeds validation Q75; otherwise use temporal-only.",
            "validation_q75_threshold": selection_threshold,
            "selected_graph_fraction": float(len(selected_samples) / len(reliabilities)),
            "selected_graph_count": len(selected_samples),
        }
    )

    low_strength = regime_value("hybrid", "low_evidence", "mean_prior_strength")
    full_strength = regime_value("hybrid", "full_evidence", "mean_prior_strength")
    return {
        "n_test_graphs": len({row["sample"] for row in model_rows}),
        "n_training_seeds": len(seeds),
        "seed_aggregate": seed_aggregate,
        "condition_summary": condition_summary,
        "paired_condition_effects": paired_summary,
        "baseline_summary": baseline_summary,
        "regime_summary": regime_summary,
        "reliability_quartile_effect": reliability_effect,
        "reliability_interaction": reliability_interaction,
        "validation_fixed_selective_policy": selective_policy,
        "reliability_thresholds": {
            "source": "independent training-validation graph cache",
            "q25_q50_q75": [float(value) for value in reliability_thresholds],
        },
        "pre_specified_hypothesis_test": {
            "low_evidence_delta_mrr": low_delta,
            "low_evidence_paired_graph_bootstrap_ci95": low_delta_ci,
            "full_evidence_delta_mrr": full_delta,
            "full_evidence_paired_graph_bootstrap_ci95": full_delta_ci,
            "difference_in_differences": low_delta - full_delta,
            "difference_in_differences_ci95_training_seeds": _ci(per_seed_interactions),
            "low_evidence_prior_strength": low_strength,
            "full_evidence_prior_strength": full_strength,
            "gate_suppression_delta": low_strength - full_strength,
            "supported": bool(
                low_delta > 0.0
                and low_delta_ci[0] > 0.0
                and low_delta > full_delta
                and low_strength > full_strength
                and _ci(per_seed_interactions)[0] > 0.0
            ),
            "decision_rule": "Support requires a positive paired-graph low-evidence gain CI, larger gain than at full evidence, larger gate strength under low evidence, and a positive across-seed interaction CI.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "outputs/cold_start_v1")
    parser.add_argument("--temporal-checkpoint-root", type=Path, default=None)
    parser.add_argument("--test-cache", type=Path, default=ROOT / "data/processed/cold_start_fixed_test.pt")
    parser.add_argument("--training-cache", type=Path, default=ROOT / "data/processed/cold_start_anytime_dataset.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cold_start_v1/evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-seed-base", type=int, default=910_000)
    parser.add_argument(
        "--hybrid-method",
        choices=("hybrid", "selective_hybrid"),
        default="hybrid",
    )
    args = parser.parse_args()
    checkpoint_roots = [args.checkpoint_root]
    if args.temporal_checkpoint_root is not None:
        checkpoint_roots.append(args.temporal_checkpoint_root)
    candidate_checkpoints = sorted(
        path for root in checkpoint_roots for path in root.glob("*_seed*/best.pt")
    )
    checkpoints = []
    for path in candidate_checkpoints:
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        if metadata.get("method") in {args.hybrid_method, "temporal_only"}:
            checkpoints.append(path)
    if len(checkpoints) < 6:
        raise RuntimeError(f"Expected six checkpoints, found {len(checkpoints)}")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    samples = _prepare_test_cache(first, args.test_cache, args.test_seed_base)
    training_cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    validation_reliability = np.asarray(
        [float(np.mean(sample.structural_features[:, 6])) for sample in training_cache["val"]],
        dtype=float,
    )
    reliability_thresholds = np.quantile(validation_reliability, [0.25, 0.5, 0.75]).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rows = []
    for path in checkpoints:
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        alias = "hybrid" if metadata.get("method") == args.hybrid_method else "temporal_only"
        model_rows.extend(
            _evaluate_checkpoint(path, samples, device, args.batch_size, method_alias=alias)
        )
    baseline_rows = _evaluate_baselines(samples, args.batch_size)
    summary = _summarize(model_rows, baseline_rows, reliability_thresholds)
    summary["test_seed_base"] = int(args.test_seed_base)
    summary["hybrid_training_method"] = args.hybrid_method
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for name, rows in (("model_sample_metrics.csv", model_rows), ("baseline_sample_metrics.csv", baseline_rows)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary["pre_specified_hypothesis_test"], indent=2))


if __name__ == "__main__":
    main()
