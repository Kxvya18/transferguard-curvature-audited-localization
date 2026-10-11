"""Evaluate the frozen mixed-domain model on real IoT-23 scenario samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate_origin_localization
from models.data_structures import TemporalGraphSample
from models.transferguard_model import TransferGuardModel
from synthetic.dataset import collate_temporal_graph_samples
from uncertainty.calibration import mc_dropout_origin_distribution


def _load_model(path: Path, device: torch.device) -> tuple[TransferGuardModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    training = checkpoint["training_config"]
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


def _load_sample(path: Path) -> TemporalGraphSample:
    with np.load(path) as payload:
        return TemporalGraphSample(
            node_ids=payload["node_ids"].astype(str).tolist(),
            adjacency=payload["adjacency"].copy(),
            node_features=payload["node_features"].copy(),
            structural_features=payload["structural_features"].copy(),
            node_mask=payload["node_mask"].copy(),
            origin_index=int(payload["origin_index"]),
            infection_times=payload["infection_times"].copy(),
            observed_mask=payload["observed_mask"].copy(),
            meta=json.loads(str(payload["meta_json"])),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "train_transferguard_mixed_safe_prior"
        / "transferguard_synthetic_best.pt",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "iot23_temporal_samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "iot23_zero_shot" / "summary.json",
    )
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--neutralize-prior", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_model(args.checkpoint, device)
    samples = [_load_sample(path) for path in sorted(args.sample_dir.glob("*.npz"))]

    rows = []
    for sample in samples:
        batch = collate_temporal_graph_samples([sample])
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if args.neutralize_prior:
            structural = batch["structural_features"].clone()
            structural[:, :, 1] = 1.0 / batch["node_mask"].sum()
            structural[:, :, 2] = 1.0
            structural[:, :, 5:7] = 0.0
            batch["structural_features"] = structural
        model.eval()
        with torch.no_grad():
            output = model(
                node_features=batch["node_features"],
                adjacency=batch["adjacency"],
                structural_features=batch["structural_features"],
                node_mask=batch["node_mask"],
                ssl_mask=None,
            )
        probability = output["probs"][0, : len(sample.node_ids)].cpu().numpy()
        metrics = evaluate_origin_localization(
            {idx: float(value) for idx, value in enumerate(probability)},
            sample.origin_index,
        )
        mc = mc_dropout_origin_distribution(model, batch, n_samples=args.mc_samples)
        mc_probability = mc["mean_probs"][0, : len(sample.node_ids)].cpu().numpy()
        mc_metrics = evaluate_origin_localization(
            {idx: float(value) for idx, value in enumerate(mc_probability)},
            sample.origin_index,
        )
        rows.append(
            {
                "dataset": sample.meta.get("dataset", "unknown"),
                "scenario": Path(sample.meta["scenario_path"]).stem,
                "n_nodes": len(sample.node_ids),
                "origin_node_id": sample.meta["origin_node_id"],
                **metrics,
                "mc_origin_rank": mc_metrics["origin_rank"],
                "mc_mrr": mc_metrics["mrr"],
                "origin_probability": float(probability[sample.origin_index]),
                "mc_origin_probability": float(mc_probability[sample.origin_index]),
                "entropy_confidence": float(mc["confidence"][0].cpu()),
                "epistemic_variance": float(mc["epistemic_var"][0].cpu()),
                "prior_strength": float(output.get("prior_strength", torch.tensor([float("nan")]))[0].cpu()),
                "beta_hat": sample.meta["beta_hat"],
                "prior_reliability": sample.meta["prior_reliability"],
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_feature_contract": checkpoint.get("feature_contract"),
        "evaluation_setting": "frozen synthetic-to-real zero-shot; no real-data fitting",
        "neutralize_prior": args.neutralize_prior,
        "n_scenarios": len(rows),
        "mean_mrr": float(np.mean([row["mrr"] for row in rows])),
        "mean_hit@1": float(np.mean([row["hit@1"] for row in rows])),
        "mean_origin_probability": float(np.mean([row["origin_probability"] for row in rows])),
        "mean_entropy_confidence": float(np.mean([row["entropy_confidence"] for row in rows])),
        "warning": "Real labels define flow sources, not multi-hop compromise chronology.",
        "scenarios": rows,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved IoT-23 zero-shot evaluation to {args.output}")


if __name__ == "__main__":
    main()
