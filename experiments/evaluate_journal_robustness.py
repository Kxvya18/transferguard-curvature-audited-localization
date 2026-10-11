"""Journal robustness grid for growth misspecification and snapshot-only priors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from synthetic.dataset import build_temporal_sample, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig, generate_iot_graph, simulate_attack_cascade


CONDITIONS = ((1, 0.25), (2, 0.50), (6, 1.00))
GROWTH_MODELS = ("heterogeneous_pa", "fitness", "aging", "copying", "stochastic_block")


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _score_map(values: np.ndarray) -> dict[int, float]:
    return {index: float(value) for index, value in enumerate(values)}


def _bootstrap_ci(values: list[float], seed: int, draws: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [np.mean(rng.choice(array, size=array.size, replace=True)) for _ in range(draws)],
        dtype=float,
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _generate_samples(
    base: SyntheticScenarioConfig,
    replicates: int,
    snapshot_ensemble_size: int,
    seed_base: int,
) -> tuple[list, list[dict]]:
    samples = []
    curvature_rows = []
    sample_id = 0
    cells = list(
        product(GROWTH_MODELS, (100, 220, 400), (1, 2, 3, 4, 5), (0.5, 1.0, 1.5, 2.0))
    )
    for cell_index, (growth_model, n_nodes, m_value, beta) in enumerate(cells):
        for replicate in range(replicates):
            seed = int(seed_base + cell_index * 1009 + replicate * 97)
            observed_config = replace(
                base,
                growth_model=growth_model,
                n_nodes=n_nodes,
                m=m_value,
                beta=beta,
                seed=seed,
                prior_history_mode="observed",
                snapshot_ensemble_size=snapshot_ensemble_size,
            )
            graph, history = generate_iot_graph(observed_config)
            cascade = simulate_attack_cascade(graph, observed_config, seed=seed + 1001)
            for history_mode in ("observed", "snapshot_ensemble"):
                config = replace(observed_config, prior_history_mode=history_mode)
                sample = build_temporal_sample(graph, history, cascade, config=config)
                sample.meta.update(
                    {
                        "journal_sample_id": sample_id,
                        "growth_model": growth_model,
                        "history_mode": history_mode,
                        "test_n": n_nodes,
                        "test_m": m_value,
                        "test_beta": beta,
                        "replicate": replicate,
                    }
                )
                samples.append(sample)
                if history_mode == "observed":
                    curvature_rows.append(
                        {
                            "sample": sample_id,
                            "growth_model": growth_model,
                            "n_nodes": n_nodes,
                            "m": m_value,
                            "beta_true": beta,
                            "beta_hat": float(sample.meta["beta_hat"]),
                            "fisher_exact": float(sample.meta["fisher_information_exact"]),
                            "fisher_approx": float(sample.meta["fisher_information_approx"]),
                            "relative_error": float(sample.meta["fisher_curvature_relative_error"]),
                        }
                    )
            sample_id += 1
        if (cell_index + 1) % 20 == 0 or cell_index + 1 == len(cells):
            print(f"[journal-grid] generated {cell_index + 1}/{len(cells)} cells", flush=True)
    return samples, curvature_rows


@torch.no_grad()
def _evaluate_checkpoint(
    checkpoint_path: Path,
    samples: list,
    device: torch.device,
    batch_size: int,
    method_alias: str | None = None,
) -> list[dict]:
    model, checkpoint = load_cold_start_model(checkpoint_path, device)
    model.eval()
    checkpoint_method = str(checkpoint["method"])
    method = method_alias or checkpoint_method
    seed = int(checkpoint["run_seed"])
    rows = []
    for horizon, retention in CONDITIONS:
        for start, chunk in _chunks(samples, batch_size):
            raw = collate_temporal_graph_samples(chunk)
            batch = prepare_prefix_batch(
                raw,
                horizon,
                retention,
                mask_seed=17_000_000 + horizon * 100_000 + int(retention * 10_000) + start,
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
                metrics = evaluate_origin_localization(_score_map(probability), sample.origin_index)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "sample": int(sample.meta["journal_sample_id"]),
                        "growth_model": sample.meta["growth_model"],
                        "history_mode": sample.meta["history_mode"],
                        "n_nodes": int(sample.meta["test_n"]),
                        "m": int(sample.meta["test_m"]),
                        "beta": float(sample.meta["test_beta"]),
                        "replicate": int(sample.meta["replicate"]),
                        "horizon": horizon,
                        "telemetry_retention": retention,
                        "mrr": float(metrics["mrr"]),
                        "hit1": float(metrics["hit@1"]),
                        "origin_rank": float(metrics["origin_rank"]),
                        "prior_strength": float(output["prior_strength"][local_index].cpu()),
                        "fisher_reliability": float(output["fisher_reliability"][local_index].cpu()),
                        "beta_hat": float(sample.meta["beta_hat"]),
                    }
                )
        print(f"[journal-grid] evaluated {method} seed={seed} h={horizon} q={retention}", flush=True)
    return rows


def _aggregate(rows: list[dict], selection_threshold: float) -> dict:
    index = {
        (
            row["method"], row["seed"], row["sample"], row["history_mode"],
            row["horizon"], row["telemetry_retention"],
        ): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    policy_rows = []
    for row in rows:
        if row["method"] != "hybrid":
            continue
        temporal = index[(
            "temporal_only", row["seed"], row["sample"], row["history_mode"],
            row["horizon"], row["telemetry_retention"],
        )]
        selected = row["fisher_reliability"] > selection_threshold
        chosen = row if selected else temporal
        policy_rows.append({**row, "method": "selective_policy", "mrr": chosen["mrr"], "hit1": chosen["hit1"], "selected": selected})
    all_rows = rows + policy_rows

    aggregate = []
    keys = sorted({
        (row["method"], row["growth_model"], row["history_mode"], row["horizon"], row["telemetry_retention"])
        for row in all_rows
    })
    for group_index, key in enumerate(keys):
        method, growth_model, history_mode, horizon, retention = key
        subset = [
            row for row in all_rows
            if (row["method"], row["growth_model"], row["history_mode"], row["horizon"], row["telemetry_retention"]) == key
        ]
        graph_means = []
        for sample in sorted({int(row["sample"]) for row in subset}):
            graph_means.append(float(np.mean([row["mrr"] for row in subset if row["sample"] == sample])))
        aggregate.append(
            {
                "method": method,
                "growth_model": growth_model,
                "history_mode": history_mode,
                "horizon": horizon,
                "telemetry_retention": retention,
                "n_graphs": len(graph_means),
                "n_training_seeds": len(seeds),
                "mean_mrr": float(np.mean(graph_means)),
                "mrr_graph_bootstrap_ci95": _bootstrap_ci(graph_means, 50_000 + group_index),
                "mean_hit1": float(np.mean([row["hit1"] for row in subset])),
                "selection_rate": float(np.mean([row.get("selected", False) for row in subset])) if method == "selective_policy" else None,
            }
        )

    paired_effects = []
    for growth_model, history_mode, horizon, retention in product(
        GROWTH_MODELS, ("observed", "snapshot_ensemble"),
        [condition[0] for condition in CONDITIONS], [condition[1] for condition in CONDITIONS],
    ):
        if (horizon, retention) not in CONDITIONS:
            continue
        deltas = []
        sample_ids = sorted({
            int(row["sample"]) for row in rows
            if row["growth_model"] == growth_model and row["history_mode"] == history_mode
        })
        for sample in sample_ids:
            seed_deltas = []
            for seed in seeds:
                hybrid = index[("hybrid", seed, sample, history_mode, horizon, retention)]
                temporal = index[("temporal_only", seed, sample, history_mode, horizon, retention)]
                seed_deltas.append(float(hybrid["mrr"] - temporal["mrr"]))
            deltas.append(float(np.mean(seed_deltas)))
        paired_effects.append(
            {
                "growth_model": growth_model,
                "history_mode": history_mode,
                "horizon": horizon,
                "telemetry_retention": retention,
                "mean_hybrid_minus_temporal_mrr": float(np.mean(deltas)),
                "graph_bootstrap_ci95": _bootstrap_ci(deltas, 70_000 + len(paired_effects)),
            }
        )
    return {"aggregate": aggregate, "paired_effects": paired_effects, "policy_rows": policy_rows}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    normalized = []
    for row in rows:
        normalized.append({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "outputs/journal_v1/cold_start_exact")
    parser.add_argument("--temporal-checkpoint-root", type=Path, default=None)
    parser.add_argument(
        "--hybrid-method",
        choices=("hybrid", "selective_hybrid"),
        default="hybrid",
    )
    parser.add_argument("--training-cache", type=Path, default=ROOT / "data/processed/cold_start_exact_dataset.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/journal_v1/robustness")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--snapshot-ensemble-size", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=1_810_000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_roots = [args.checkpoint_root]
    if args.temporal_checkpoint_root is not None:
        checkpoint_roots.append(args.temporal_checkpoint_root)
    candidates = sorted(path for root in checkpoint_roots for path in root.glob("*_seed*/best.pt"))
    checkpoints = []
    for path in candidates:
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        if metadata.get("method") in {args.hybrid_method, "temporal_only"}:
            checkpoints.append(path)
    methods_by_seed = {}
    for path in checkpoints:
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        methods_by_seed.setdefault(int(metadata["run_seed"]), set()).add(str(metadata["method"]))
    complete_seeds = {
        seed for seed, methods in methods_by_seed.items()
        if {args.hybrid_method, "temporal_only"}.issubset(methods)
    }
    checkpoints = [
        path for path in checkpoints
        if int(torch.load(path, map_location="cpu", weights_only=False)["run_seed"]) in complete_seeds
    ]
    if not complete_seeds:
        raise RuntimeError(
            f"No matched {args.hybrid_method}/temporal_only checkpoint seeds were found"
        )
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    base = SyntheticScenarioConfig(**checkpoint["scenario_config"])
    samples, curvature_rows = _generate_samples(base, args.replicates, args.snapshot_ensemble_size, args.seed_base)

    training_cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    validation_reliability = np.asarray(
        [float(np.mean(sample.structural_features[:, 6])) for sample in training_cache["val"]],
        dtype=float,
    )
    selection_threshold = float(np.quantile(validation_reliability, 0.75))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for path in checkpoints:
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        alias = "hybrid" if metadata.get("method") == args.hybrid_method else "temporal_only"
        rows.extend(_evaluate_checkpoint(path, samples, device, args.batch_size, method_alias=alias))
    summary = _aggregate(rows, selection_threshold)
    summary.update(
        {
            "n_graphs": len({int(row["sample"]) for row in rows}),
            "n_training_seeds": len({int(row["seed"]) for row in rows}),
            "growth_models": list(GROWTH_MODELS),
            "conditions": [list(condition) for condition in CONDITIONS],
            "selection_threshold": selection_threshold,
            "hybrid_training_method": args.hybrid_method,
            "matched_training_seeds": sorted(complete_seeds),
            "curvature_validation": {
                "median_relative_error": float(np.median([row["relative_error"] for row in curvature_rows])),
                "q95_relative_error": float(np.quantile([row["relative_error"] for row in curvature_rows], 0.95)),
                "max_relative_error": float(np.max([row["relative_error"] for row in curvature_rows])),
            },
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in summary.items() if key != "policy_rows"}, handle, indent=2)
    _write_csv(args.output_dir / "model_sample_metrics.csv", rows)
    _write_csv(args.output_dir / "policy_sample_metrics.csv", summary["policy_rows"])
    _write_csv(args.output_dir / "aggregate.csv", summary["aggregate"])
    _write_csv(args.output_dir / "paired_effects.csv", summary["paired_effects"])
    _write_csv(args.output_dir / "fisher_curvature_validation.csv", curvature_rows)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"aggregate", "paired_effects", "policy_rows"}}, indent=2))


if __name__ == "__main__":
    main()
