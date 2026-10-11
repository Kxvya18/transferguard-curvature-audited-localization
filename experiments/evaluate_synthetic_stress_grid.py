"""Evaluate paired hybrid/no-prior checkpoints across n, m, and beta shifts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate_origin_localization
from models.transferguard_model import TransferGuardModel
from synthetic.dataset import SyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig


def _load_model(path: Path, device: torch.device) -> tuple[TransferGuardModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    training = checkpoint["training_config"]
    state = checkpoint["model_state_dict"]
    node_feature_dim = int(state["encoder.node_proj.weight"].shape[1])
    structural_dim = int(state["encoder.struct_proj.weight"].shape[1])
    model = TransferGuardModel(
        node_feature_dim=node_feature_dim,
        structural_feature_dim=structural_dim,
        hidden_dim=int(training.get("hidden_dim", 64)),
        num_message_layers=int(training.get("num_message_layers", 2)),
        dropout=float(training.get("dropout", 0.15)),
        fusion_type=str(training.get("fusion_type", "gated")),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint


def _metrics(values: np.ndarray, origin_index: int) -> dict[str, float]:
    return evaluate_origin_localization(
        {idx: float(value) for idx, value in enumerate(values)},
        origin_index,
    )


@torch.no_grad()
def _model_rows(model, loader, method: str, device, neutralize_prior: bool) -> list[dict]:
    rows = []
    sample_offset = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if neutralize_prior:
            structural = batch["structural_features"].clone()
            uniform = 1.0 / batch["node_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
            structural[:, :, 1] = uniform
            structural[:, :, 2] = 1.0
            if structural.size(-1) >= 7:
                structural[:, :, 5] = 0.0
                structural[:, :, 6] = 0.0
            batch["structural_features"] = structural
        output = model(
            node_features=batch["node_features"],
            adjacency=batch["adjacency"],
            structural_features=batch["structural_features"],
            node_mask=batch["node_mask"],
            ssl_mask=None,
        )
        probabilities = output["probs"].cpu().numpy()
        origins = batch["origin_index"].cpu().numpy()
        masks = batch["node_mask"].cpu().numpy()
        prior_strength = output.get("prior_strength")
        prior_strength_values = (
            prior_strength.cpu().numpy()
            if prior_strength is not None
            else np.full(probabilities.shape[0], np.nan)
        )
        for batch_idx in range(probabilities.shape[0]):
            n_nodes = int(np.sum(masks[batch_idx]))
            rows.append(
                {
                    "sample": sample_offset + batch_idx,
                    "method": method,
                    "prior_strength": float(prior_strength_values[batch_idx]),
                    **_metrics(probabilities[batch_idx, :n_nodes], int(origins[batch_idx])),
                }
            )
        sample_offset += probabilities.shape[0]
    return rows


def _baseline_rows(dataset) -> list[dict]:
    rows = []
    for sample_idx, sample in enumerate(dataset.samples):
        score_vectors = {
            "degree": np.sum(sample.adjacency, axis=1),
            "boltzmann_prior": sample.structural_features[:, 1],
            "outbound_activity": np.sum(sample.node_features[:, :, 2], axis=0),
        }
        for method, values in score_vectors.items():
            rows.append(
                {
                    "sample": sample_idx,
                    "method": method,
                    "prior_strength": float("nan"),
                    **_metrics(values, sample.origin_index),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "train_transferguard_synthetic_blind_origin_v2"
        / "transferguard_synthetic_best.pt",
    )
    parser.add_argument(
        "--no-prior-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "train_transferguard_synthetic_blind_origin_no_prior"
        / "transferguard_synthetic_best.pt",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[100, 220, 400])
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--beta-values", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--samples-per-cell", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_stress_grid",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_model, hybrid_checkpoint = _load_model(args.hybrid_checkpoint, device)
    no_prior_model, _ = _load_model(args.no_prior_checkpoint, device)
    base_config = SyntheticScenarioConfig(**hybrid_checkpoint["scenario_config"])

    all_rows = []
    cell_index = 0
    for n_nodes in args.n_values:
        for m_value in args.m_values:
            for beta_value in args.beta_values:
                config = replace(
                    base_config,
                    n_nodes=n_nodes,
                    m=m_value,
                    beta=beta_value,
                    seed=int(base_config.seed + cell_index * 10_003),
                )
                label = f"n{n_nodes}_m{m_value}_b{beta_value:g}"
                dataset = SyntheticOriginDataset(
                    config,
                    num_samples=args.samples_per_cell,
                    seed_offset=80_000,
                    progress_label=label,
                    progress_every=args.samples_per_cell,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=min(8, args.samples_per_cell),
                    shuffle=False,
                    collate_fn=collate_temporal_graph_samples,
                )
                cell_rows = _baseline_rows(dataset)
                cell_rows.extend(
                    _model_rows(hybrid_model, loader, "hybrid", device, neutralize_prior=False)
                )
                cell_rows.extend(
                    _model_rows(no_prior_model, loader, "no_prior", device, neutralize_prior=True)
                )
                for row in cell_rows:
                    sample = dataset.samples[int(row["sample"])]
                    all_rows.append(
                        {
                            "n": n_nodes,
                            "m": m_value,
                            "beta_true": beta_value,
                            "sample": row["sample"],
                            "method": row["method"],
                            "prior_strength": row.get("prior_strength", float("nan")),
                            "beta_hat": sample.meta["beta_hat"],
                            "beta_boundary_warning": sample.meta["beta_boundary_warning"],
                            "prior_reliability": sample.meta["prior_reliability"],
                            **{
                                key: value
                                for key, value in row.items()
                                if key not in {"sample", "method", "prior_strength"}
                            },
                        }
                    )
                print(f"[stress-grid] completed {label}", flush=True)
                cell_index += 1

    metrics_path = args.output_dir / "sample_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate = []
    group_keys = sorted({(row["n"], row["m"], row["beta_true"], row["method"]) for row in all_rows})
    for n_nodes, m_value, beta_value, method in group_keys:
        group = [
            row
            for row in all_rows
            if (row["n"], row["m"], row["beta_true"], row["method"])
            == (n_nodes, m_value, beta_value, method)
        ]
        aggregate.append(
            {
                "n": n_nodes,
                "m": m_value,
                "beta_true": beta_value,
                "method": method,
                "mean_mrr": float(np.mean([row["mrr"] for row in group])),
                "mean_hit@1": float(np.mean([row["hit@1"] for row in group])),
                "mean_hit@5": float(np.mean([row["hit@5"] for row in group])),
                "mean_beta_hat": float(np.mean([row["beta_hat"] for row in group])),
                "boundary_rate": float(np.mean([row["beta_boundary_warning"] for row in group])),
                "mean_prior_reliability": float(np.mean([row["prior_reliability"] for row in group])),
                "mean_prior_strength": float(
                    np.nanmean([row["prior_strength"] for row in group])
                )
                if method in {"hybrid", "no_prior"}
                else None,
            }
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_contract": hybrid_checkpoint.get("feature_contract", "unknown"),
                "n_values": args.n_values,
                "m_values": args.m_values,
                "beta_values": args.beta_values,
                "samples_per_cell": args.samples_per_cell,
                "n_total_samples": cell_index * args.samples_per_cell,
                "aggregate": aggregate,
            },
            handle,
            indent=2,
        )
    print(f"Saved stress-grid results to {summary_path}")


if __name__ == "__main__":
    main()
