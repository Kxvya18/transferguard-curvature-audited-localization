"""Train the first learned TRANSFERGUARD model on synthetic origin localization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate_origin_localization
from models.transferguard_model import TransferGuardModel
from synthetic.dataset import (
    MixedSyntheticOriginDataset,
    SyntheticOriginDataset,
    collate_temporal_graph_samples,
)
from synthetic.iot_growth import SyntheticScenarioConfig
from uncertainty.calibration import mc_dropout_origin_distribution


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_config(payload: dict) -> tuple[SyntheticScenarioConfig, dict]:
    scenario_keys = set(SyntheticScenarioConfig.__dataclass_fields__.keys())
    scenario_payload = {key: value for key, value in payload.items() if key in scenario_keys}
    train_payload = {key: value for key, value in payload.items() if key not in scenario_keys}
    return SyntheticScenarioConfig(**scenario_payload), train_payload


def _mean_origin_metrics(logit_batches: list[torch.Tensor], label_batches: list[torch.Tensor]) -> dict[str, float]:
    rows = []
    for logits, labels in zip(logit_batches, label_batches):
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        label_np = labels.cpu().numpy()
        for sample_idx in range(probs.shape[0]):
            score_map = {int(node): float(prob) for node, prob in enumerate(probs[sample_idx])}
            rows.append(evaluate_origin_localization(score_map, int(label_np[sample_idx])))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0].keys()}


def _apply_ablations(batch: dict, train_cfg: dict) -> dict:
    if not bool(train_cfg.get("ablate_boltzmann_prior", False)):
        return batch
    structural = batch["structural_features"].clone()
    uniform = 1.0 / batch["node_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
    structural[:, :, 1] = uniform
    structural[:, :, 2] = 1.0
    if structural.size(-1) >= 7:
        structural[:, :, 5] = 0.0
        structural[:, :, 6] = 0.0
    return {**batch, "structural_features": structural}


def train(
    config_path: Path,
    out_dir: Path,
    run_seed_override: int | None = None,
    ssl_weight_override: float | None = None,
) -> None:
    payload = _load_json(config_path)
    scenario_cfg, train_cfg = _build_config(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_seed = int(
        train_cfg.get("run_seed", 20260810)
        if run_seed_override is None
        else run_seed_override
    )
    train_cfg["run_seed"] = run_seed
    if ssl_weight_override is not None:
        train_cfg["ssl_weight"] = float(ssl_weight_override)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    loader_generator = torch.Generator().manual_seed(run_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_grid = train_cfg.get("domain_grid")
    dataset_type = MixedSyntheticOriginDataset if domain_grid else SyntheticOriginDataset
    dataset_extra = {"domain_grid": domain_grid} if domain_grid else {}
    train_ds = dataset_type(
        base_config=scenario_cfg,
        num_samples=int(train_cfg.get("train_samples", 160)),
        seed_offset=0,
        progress_label="train",
        **dataset_extra,
    ) if domain_grid else SyntheticOriginDataset(
        config=scenario_cfg,
        num_samples=int(train_cfg.get("train_samples", 160)),
        seed_offset=0,
        progress_label="train",
    )
    val_ds = dataset_type(
        base_config=scenario_cfg,
        num_samples=int(train_cfg.get("val_samples", 48)),
        seed_offset=10_000,
        progress_label="val",
        **dataset_extra,
    ) if domain_grid else SyntheticOriginDataset(
        config=scenario_cfg,
        num_samples=int(train_cfg.get("val_samples", 48)),
        seed_offset=10_000,
        progress_label="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_temporal_graph_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        collate_fn=collate_temporal_graph_samples,
    )

    sample = train_ds[0]
    model = TransferGuardModel(
        node_feature_dim=sample.node_features.shape[-1],
        structural_feature_dim=sample.structural_features.shape[-1],
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        num_message_layers=int(train_cfg.get("num_message_layers", 2)),
        dropout=float(train_cfg.get("dropout", 0.15)),
        fusion_type=str(train_cfg.get("fusion_type", "gated")),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    ce_loss = nn.CrossEntropyLoss()
    ssl_weight = float(train_cfg.get("ssl_weight", 0.25))
    epochs = int(train_cfg.get("epochs", 20))

    best = {"val_mrr": -1.0, "epoch": -1, "state_dict": None}
    history = []
    history_path = out_dir / "training_history.json"

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_logits, train_labels = [], []
        for batch in train_loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch = _apply_ablations(batch, train_cfg)
            out = model(
                node_features=batch["node_features"],
                adjacency=batch["adjacency"],
                structural_features=batch["structural_features"],
                node_mask=batch["node_mask"],
                ssl_mask=batch["ssl_mask"],
            )
            supervised = ce_loss(out["logits"], batch["origin_index"])
            loss = supervised + ssl_weight * out["ssl_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_logits.append(out["logits"].detach())
            train_labels.append(batch["origin_index"].detach())

        train_metrics = _mean_origin_metrics(train_logits, train_labels)

        model.eval()
        val_losses = []
        val_logits, val_labels = [], []
        confidence_rows = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                batch = _apply_ablations(batch, train_cfg)
                out = model(
                    node_features=batch["node_features"],
                    adjacency=batch["adjacency"],
                    structural_features=batch["structural_features"],
                    node_mask=batch["node_mask"],
                    ssl_mask=batch["ssl_mask"],
                )
                supervised = ce_loss(out["logits"], batch["origin_index"])
                loss = supervised + ssl_weight * out["ssl_loss"]
                val_losses.append(float(loss.item()))
                val_logits.append(out["logits"])
                val_labels.append(batch["origin_index"])

                mc = mc_dropout_origin_distribution(
                    model,
                    {
                        "node_features": batch["node_features"],
                        "adjacency": batch["adjacency"],
                        "structural_features": batch["structural_features"],
                        "node_mask": batch["node_mask"],
                    },
                    n_samples=int(train_cfg.get("mc_samples", 8)),
                )
                preds = mc["mean_probs"].argmax(dim=-1)
                acc = (preds == batch["origin_index"]).float()
                confidence_rows.extend(
                    {
                        "confidence": float(conf),
                        "correct": float(ok),
                        "epistemic": float(epi),
                    }
                    for conf, ok, epi in zip(
                        mc["confidence"].cpu(),
                        acc.cpu(),
                        mc["epistemic_var"].cpu(),
                    )
                )

        val_metrics = _mean_origin_metrics(val_logits, val_labels)
        mean_conf = float(np.mean([row["confidence"] for row in confidence_rows]))
        mean_epi = float(np.mean([row["epistemic"] for row in confidence_rows]))

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "train_mrr": train_metrics["mrr"],
            "val_mrr": val_metrics["mrr"],
            "train_hit@1": train_metrics["hit@1"],
            "val_hit@1": val_metrics["hit@1"],
            "mean_confidence": mean_conf,
            "mean_epistemic_var": mean_epi,
        }
        history.append(record)
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(
            f"[synthetic] epoch={epoch} train_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_mrr={record['val_mrr']:.4f} "
            f"val_hit1={record['val_hit@1']:.4f}",
            flush=True,
        )

        if val_metrics["mrr"] > best["val_mrr"]:
            best = {
                "val_mrr": val_metrics["mrr"],
                "epoch": epoch,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }

    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "scenario_config": scenario_cfg.to_dict(),
            "training_config": train_cfg,
            "run_seed": run_seed,
            "feature_contract": "observable_v4_cross_dataset_aligned",
        },
        out_dir / "transferguard_synthetic_best.pt",
    )
    summary = {
        "best_epoch": int(best["epoch"]),
        "best_val_mrr": float(best["val_mrr"]),
        "final_epoch": int(history[-1]["epoch"]),
        "final_train_mrr": float(history[-1]["train_mrr"]),
        "final_val_mrr": float(history[-1]["val_mrr"]),
        "final_val_hit1": float(history[-1]["val_hit@1"]),
        "final_mean_confidence": float(history[-1]["mean_confidence"]),
        "final_mean_epistemic_var": float(history[-1]["mean_epistemic_var"]),
        "run_seed": run_seed,
        "feature_contract": "observable_v4_cross_dataset_aligned",
        "ablate_boltzmann_prior": bool(train_cfg.get("ablate_boltzmann_prior", False)),
    }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved model and metrics to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_transferguard_synthetic.json",
    )
    parser.add_argument(
        "--ssl-weight",
        type=float,
        default=None,
        help="Override the self-supervised loss weight for a controlled ablation.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Override the model/DataLoader seed while preserving the configured graph splits.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "train_transferguard_synthetic",
    )
    args = parser.parse_args()
    train(
        config_path=args.config,
        out_dir=args.out_dir,
        run_seed_override=args.run_seed,
        ssl_weight_override=args.ssl_weight,
    )


if __name__ == "__main__":
    main()
