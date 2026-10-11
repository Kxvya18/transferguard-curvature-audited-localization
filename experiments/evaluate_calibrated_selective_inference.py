"""Evaluate calibration, conformal sets, abstention, and OOD awareness."""

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

from models.transferguard_model import TransferGuardModel
from synthetic.dataset import MixedSyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig
from uncertainty.calibration import (
    binary_auroc,
    classification_calibration_metrics,
    conformal_quantile,
    evaluate_prediction_sets,
    fit_temperature,
    mc_dropout_origin_distribution,
    probabilities_from_logits,
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


def _neutralize_prior(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    structural = batch["structural_features"].clone()
    uniform = 1.0 / batch["node_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
    structural[:, :, 1] = uniform
    structural[:, :, 2] = 1.0
    structural[:, :, 5:7] = 0.0
    return {**batch, "structural_features": structural}


@torch.no_grad()
def _collect(
    model: TransferGuardModel,
    dataset,
    device: torch.device,
    neutralize_prior: bool,
    mc_samples: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=False,
        collate_fn=collate_temporal_graph_samples,
    )
    logits: list[np.ndarray] = []
    labels: list[int] = []
    mc_probabilities: list[np.ndarray] = []
    mc_mutual_information: list[float] = []
    mc_epistemic_variance: list[float] = []
    model.eval()
    for raw_batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        if neutralize_prior:
            batch = _neutralize_prior(batch)
        output = model(
            node_features=batch["node_features"],
            adjacency=batch["adjacency"],
            structural_features=batch["structural_features"],
            node_mask=batch["node_mask"],
            ssl_mask=None,
        )
        mc = mc_dropout_origin_distribution(model, batch, n_samples=mc_samples)
        counts = batch["node_mask"].sum(dim=1).long().cpu().numpy()
        for idx, count in enumerate(counts):
            n_nodes = int(count)
            logits.append(output["logits"][idx, :n_nodes].detach().cpu().numpy())
            labels.append(int(batch["origin_index"][idx].cpu()))
            probs = mc["mean_probs"][idx, :n_nodes].detach().cpu().numpy().astype(float)
            probs /= max(float(probs.sum()), 1e-12)
            mc_probabilities.append(probs)
            mc_mutual_information.append(float(mc["mutual_information"][idx].cpu()))
            mc_epistemic_variance.append(float(mc["epistemic_var"][idx].cpu()))
        model.eval()
    return {
        "logits": logits,
        "labels": labels,
        "mc_probabilities": mc_probabilities,
        "mc_mutual_information": np.asarray(mc_mutual_information),
        "mc_epistemic_variance": np.asarray(mc_epistemic_variance),
    }


def _probability_scores(probabilities: list[np.ndarray]) -> dict[str, np.ndarray]:
    max_prob = np.asarray([float(np.max(value)) for value in probabilities])
    normalized_entropy = []
    for value in probabilities:
        entropy = -float(np.sum(value * np.log(np.clip(value, 1e-12, 1.0))))
        normalized_entropy.append(entropy / max(np.log(len(value)), 1e-12))
    return {
        "one_minus_max_probability": 1.0 - max_prob,
        "normalized_predictive_entropy": np.asarray(normalized_entropy),
    }


def _evaluate_model(
    name: str,
    model: TransferGuardModel,
    calibration_dataset,
    id_dataset,
    ood_dataset,
    device: torch.device,
    neutralize_prior: bool,
    mc_samples: int,
) -> tuple[dict, list[dict]]:
    calibration = _collect(
        model, calibration_dataset, device, neutralize_prior, mc_samples
    )
    id_result = _collect(model, id_dataset, device, neutralize_prior, mc_samples)
    ood_result = _collect(model, ood_dataset, device, neutralize_prior, mc_samples)
    temperature_fit = fit_temperature(calibration["logits"], calibration["labels"])

    calibrated = {
        split: [
            probabilities_from_logits(value, temperature_fit.temperature)
            for value in result["logits"]
        ]
        for split, result in (("calibration", calibration), ("id", id_result), ("ood", ood_result))
    }
    raw = {
        split: [probabilities_from_logits(value) for value in result["logits"]]
        for split, result in (("id", id_result), ("ood", ood_result))
    }
    conformal_q = conformal_quantile(
        calibrated["calibration"], calibration["labels"], alpha=0.10
    )
    id_scores = _probability_scores(calibrated["id"])
    ood_scores = _probability_scores(calibrated["ood"])
    ood_detection = {
        key: binary_auroc(id_scores[key], ood_scores[key]) for key in id_scores
    }
    ood_detection["mc_mutual_information"] = binary_auroc(
        id_result["mc_mutual_information"], ood_result["mc_mutual_information"]
    )
    ood_detection["mc_epistemic_variance"] = binary_auroc(
        id_result["mc_epistemic_variance"], ood_result["mc_epistemic_variance"]
    )
    summary = {
        "model": name,
        "temperature_fit": temperature_fit.__dict__,
        "raw": {
            "id": classification_calibration_metrics(raw["id"], id_result["labels"]),
            "ood": classification_calibration_metrics(raw["ood"], ood_result["labels"]),
        },
        "temperature_scaled": {
            "id": classification_calibration_metrics(
                calibrated["id"], id_result["labels"]
            ),
            "ood": classification_calibration_metrics(
                calibrated["ood"], ood_result["labels"]
            ),
        },
        "conformal_alpha": 0.10,
        "conformal_quantile": conformal_q,
        "conformal": {
            "id": evaluate_prediction_sets(
                calibrated["id"], id_result["labels"], conformal_q
            ),
            "ood": evaluate_prediction_sets(
                calibrated["ood"], ood_result["labels"], conformal_q
            ),
        },
        "ood_detection_auroc": ood_detection,
    }
    rows = []
    for split, result in (("id", id_result), ("ood", ood_result)):
        for idx, (probs, label) in enumerate(
            zip(calibrated[split], result["labels"])
        ):
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "sample": idx,
                    "n_nodes": len(probs),
                    "label": label,
                    "prediction": int(np.argmax(probs)),
                    "correct": int(np.argmax(probs) == label),
                    "confidence": float(np.max(probs)),
                    "normalized_entropy": entropy / max(np.log(len(probs)), 1e-12),
                    "mc_mutual_information": float(result["mc_mutual_information"][idx]),
                    "mc_epistemic_variance": float(result["mc_epistemic_variance"][idx]),
                }
            )
    return summary, rows


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
    parser.add_argument("--mc-samples", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibrated_selective_v4",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid, checkpoint = _load_model(args.hybrid_checkpoint, device)
    no_prior, _ = _load_model(args.no_prior_checkpoint, device)
    base = SyntheticScenarioConfig(**checkpoint["scenario_config"])
    id_grid = {"n_nodes": [160, 220, 280], "m": [1, 3, 5], "beta": [0.5, 1.0, 1.5, 2.0]}
    calibration_dataset = MixedSyntheticOriginDataset(
        base, 108, id_grid, seed_offset=140_000, progress_label="calibration", progress_every=27
    )
    id_dataset = MixedSyntheticOriginDataset(
        base, 144, id_grid, seed_offset=170_000, progress_label="id-test", progress_every=36
    )
    ood_base = replace(
        base,
        observation_dropout=0.72,
        origin_observation_dropout=0.98,
        benign_burst_prob=0.12,
        benign_burst_scale=7.0,
        activity_noise_logstd=0.65,
        attack_activity_scale=0.70,
    )
    ood_grid = {"n_nodes": [100, 400], "m": [1, 2, 3, 4, 5], "beta": [0.25, 2.5]}
    ood_dataset = MixedSyntheticOriginDataset(
        ood_base, 160, ood_grid, seed_offset=210_000, progress_label="ood-test", progress_every=40
    )

    summaries = []
    rows = []
    for name, model, ablate in (
        ("safe_hybrid", hybrid, False),
        ("temporal_only", no_prior, True),
    ):
        summary, model_rows = _evaluate_model(
            name,
            model,
            calibration_dataset,
            id_dataset,
            ood_dataset,
            device,
            neutralize_prior=ablate,
            mc_samples=args.mc_samples,
        )
        summaries.append(summary)
        rows.extend(model_rows)
        print(f"[calibration] completed {name}", flush=True)

    payload = {
        "feature_contract": checkpoint.get("feature_contract", "unknown"),
        "calibration_samples": len(calibration_dataset),
        "id_samples": len(id_dataset),
        "ood_samples": len(ood_dataset),
        "ood_definition": {
            "n_nodes": [100, 400],
            "m": [1, 2, 3, 4, 5],
            "beta": [0.25, 2.5],
            "observation_dropout": ood_base.observation_dropout,
            "origin_observation_dropout": ood_base.origin_observation_dropout,
            "benign_burst_prob": ood_base.benign_burst_prob,
            "activity_noise_logstd": ood_base.activity_noise_logstd,
            "attack_activity_scale": ood_base.attack_activity_scale,
        },
        "models": summaries,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (args.output_dir / "sample_uncertainty.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved calibrated selective evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
