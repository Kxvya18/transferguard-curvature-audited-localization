"""Cross-domain synthetic transfer training for TRANSFERGUARD."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate_origin_localization
from models.transferguard_model import TransferGuardModel
from synthetic.dataset import SyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig
from transfer.adaptation import coral_loss


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _subset_loader(dataset, fraction: float, batch_size: int, seed: int):
    n_items = max(1, int(len(dataset) * fraction))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:n_items].tolist()
    subset = Subset(dataset, indices)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_temporal_graph_samples,
    )


def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    label_np = labels.cpu().numpy()
    rows = []
    for sample_idx in range(probs.shape[0]):
        score_map = {int(node): float(prob) for node, prob in enumerate(probs[sample_idx])}
        rows.append(evaluate_origin_localization(score_map, int(label_np[sample_idx])))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0].keys()}


def _evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            out = model(
                node_features=batch["node_features"],
                adjacency=batch["adjacency"],
                structural_features=batch["structural_features"],
                node_mask=batch["node_mask"],
                ssl_mask=batch["ssl_mask"],
            )
            rows.append(_metrics_from_logits(out["logits"], batch["origin_index"]))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0].keys()}


def train(config_path: Path, out_dir: Path) -> None:
    payload = _load_json(config_path)
    source_cfg = SyntheticScenarioConfig(**payload["source"])
    target_cfg = SyntheticScenarioConfig(**payload["target"])
    train_cfg = payload["training"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_seed = int(train_cfg.get("run_seed", 20260810))
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(train_cfg.get("batch_size", 8))

    source_train = SyntheticOriginDataset(
        source_cfg,
        int(train_cfg.get("source_train_samples", 192)),
        seed_offset=0,
        progress_label="source_train",
    )
    target_unlabeled = SyntheticOriginDataset(
        target_cfg,
        int(train_cfg.get("target_unlabeled_samples", 96)),
        seed_offset=20_000,
        progress_label="target_unlabeled",
    )
    target_test = SyntheticOriginDataset(
        target_cfg,
        int(train_cfg.get("target_test_samples", 64)),
        seed_offset=40_000,
        progress_label="target_test",
    )

    source_loader = DataLoader(
        source_train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(run_seed),
        collate_fn=collate_temporal_graph_samples,
    )
    target_unlabeled_loader = DataLoader(
        target_unlabeled,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(run_seed + 1),
        collate_fn=collate_temporal_graph_samples,
    )
    target_test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_graph_samples)

    sample = source_train[0]
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
    coral_weight = float(train_cfg.get("coral_weight", 0.15))
    epochs = int(train_cfg.get("epochs", 24))

    target_iter = None
    epoch_history = []
    for _epoch in range(epochs):
        model.train()
        target_iter = iter(target_unlabeled_loader)
        for src_batch in source_loader:
            try:
                tgt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_unlabeled_loader)
                tgt_batch = next(target_iter)

            src_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in src_batch.items()}
            tgt_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in tgt_batch.items()}

            src_out = model(
                node_features=src_batch["node_features"],
                adjacency=src_batch["adjacency"],
                structural_features=src_batch["structural_features"],
                node_mask=src_batch["node_mask"],
                ssl_mask=src_batch["ssl_mask"],
            )
            tgt_out = model(
                node_features=tgt_batch["node_features"],
                adjacency=tgt_batch["adjacency"],
                structural_features=tgt_batch["structural_features"],
                node_mask=tgt_batch["node_mask"],
                ssl_mask=tgt_batch["ssl_mask"],
            )

            supervised = ce_loss(src_out["logits"], src_batch["origin_index"])
            ssl = src_out["ssl_loss"] + tgt_out["ssl_loss"]
            align = coral_loss(src_out["graph_embedding"], tgt_out["graph_embedding"])
            loss = supervised + ssl_weight * ssl + coral_weight * align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_zero_shot = _evaluate(model, target_test_loader, device)
        epoch_record = {
            "epoch": int(_epoch),
            "zero_shot_mrr": float(epoch_zero_shot["mrr"]),
            "zero_shot_hit@1": float(epoch_zero_shot["hit@1"]),
        }
        epoch_history.append(epoch_record)
        with (out_dir / "cross_domain_epoch_history.json").open("w", encoding="utf-8") as handle:
            json.dump(epoch_history, handle, indent=2)
        print(
            f"[cross-domain] epoch={_epoch} zero_shot_mrr={epoch_record['zero_shot_mrr']:.4f} "
            f"zero_shot_hit1={epoch_record['zero_shot_hit@1']:.4f}",
            flush=True,
        )

    zero_shot = _evaluate(model, target_test_loader, device)

    few_shot_results = {}
    for frac in [0.01, 0.05, 0.10]:
        adapt_model = deepcopy(model)
        adapt_optimizer = AdamW(adapt_model.parameters(), lr=float(train_cfg.get("adapt_lr", 8e-4)))
        few_loader = _subset_loader(
            target_unlabeled,
            frac,
            batch_size=batch_size,
            seed=run_seed + int(frac * 10_000),
        )
        adapt_model.to(device)
        adapt_model.train()
        for _ in range(int(train_cfg.get("adapt_epochs", 5))):
            for batch in few_loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                out = adapt_model(
                    node_features=batch["node_features"],
                    adjacency=batch["adjacency"],
                    structural_features=batch["structural_features"],
                    node_mask=batch["node_mask"],
                    ssl_mask=batch["ssl_mask"],
                )
                loss = ce_loss(out["logits"], batch["origin_index"]) + ssl_weight * out["ssl_loss"]
                adapt_optimizer.zero_grad()
                loss.backward()
                adapt_optimizer.step()
        few_shot_results[f"{int(frac * 100)}pct"] = _evaluate(adapt_model, target_test_loader, device)

    with (out_dir / "cross_domain_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "zero_shot": zero_shot,
                "few_shot": few_shot_results,
                "source_config": source_cfg.to_dict(),
                "target_config": target_cfg.to_dict(),
                "training": train_cfg,
                "run_seed": run_seed,
                "feature_contract": "observable_v2_blind_origin",
            },
            handle,
            indent=2,
        )
    print(f"Saved cross-domain summary to {out_dir / 'cross_domain_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_transferguard_cross_domain.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "train_transferguard_cross_domain",
    )
    args = parser.parse_args()
    train(config_path=args.config, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
