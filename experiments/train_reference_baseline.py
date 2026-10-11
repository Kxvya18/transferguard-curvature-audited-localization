"""Train static-GCN or windowed-TGAT reference baselines on the mixed grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from experiments.cold_start_common import prepare_prefix_batch
from models.reference_baselines import (
    StaticGCNOriginBaseline,
    TGMTGNOriginBaseline,
    WindowedTGATOriginBaseline,
)
from synthetic.dataset import MixedSyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("static_gcn", "windowed_tgat", "tgm_tgn"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--protocol",
        choices=("anytime", "full_evidence"),
        default="anytime",
        help="Match the principal randomized evidence schedule or train only on complete logs.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cold_start_exact_dataset.pt",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (PROJECT_ROOT / "configs" / "train_transferguard_mixed_no_prior.json").open() as handle:
        payload = json.load(handle)
    with (PROJECT_ROOT / "configs" / "train_cold_start_anytime.json").open() as handle:
        anytime_payload = json.load(handle)
    scenario_keys = set(SyntheticScenarioConfig.__dataclass_fields__)
    base = SyntheticScenarioConfig(**{k: v for k, v in payload.items() if k in scenario_keys})
    grid = payload["domain_grid"]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cache.exists():
        cached = torch.load(args.cache, map_location="cpu", weights_only=False)
        train = cached["train"]
        val = cached["val"]
    else:
        train = MixedSyntheticOriginDataset(base, 360, grid, progress_label=f"{args.architecture}-train")
        val = MixedSyntheticOriginDataset(base, 144, grid, seed_offset=10_000, progress_label=f"{args.architecture}-val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=12, shuffle=True, generator=generator, collate_fn=collate_temporal_graph_samples)
    val_loader = DataLoader(val, batch_size=12, shuffle=False, collate_fn=collate_temporal_graph_samples)
    sample = train[0]
    classes = {
        "static_gcn": StaticGCNOriginBaseline,
        "windowed_tgat": WindowedTGATOriginBaseline,
        "tgm_tgn": TGMTGNOriginBaseline,
    }
    cls = classes[args.architecture]
    model = cls(sample.node_features.shape[-1], sample.structural_features.shape[-1], hidden_dim=96)
    optimizer = AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best = (-1.0, None, -1)
    history = []
    training_started = time.perf_counter()
    train_horizons = [int(value) for value in anytime_payload["train_horizons"]]
    train_retentions = [float(value) for value in anytime_payload["train_retentions"]]
    validation_conditions = [
        (int(horizon), float(retention))
        for horizon, retention in anytime_payload["validation_conditions"]
    ]
    if args.protocol == "full_evidence":
        train_horizons = [6]
        train_retentions = [1.0]
        validation_conditions = [(6, 1.0)]
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        schedule_rng = np.random.default_rng(args.seed + epoch * 104_729)
        for batch_index, raw_batch in enumerate(train_loader):
            horizon = int(schedule_rng.choice(train_horizons))
            retention = float(schedule_rng.choice(train_retentions))
            batch = prepare_prefix_batch(
                raw_batch,
                horizon,
                retention,
                args.seed * 1_000_003 + epoch * 10_007 + batch_index,
                True,
            )
            out = model(batch["node_features"], batch["adjacency"], batch["structural_features"], batch["node_mask"])
            loss = loss_fn(out["logits"], batch["origin_index"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss))
        model.eval(); condition_rows = []
        with torch.no_grad():
            for condition_index, (horizon, retention) in enumerate(validation_conditions):
                rows = []
                for batch_index, raw_batch in enumerate(val_loader):
                    batch = prepare_prefix_batch(
                        raw_batch,
                        horizon,
                        retention,
                        91_000 + condition_index * 10_000 + batch_index,
                        True,
                    )
                    out = model(batch["node_features"], batch["adjacency"], batch["structural_features"], batch["node_mask"])
                    for probs, label, mask in zip(out["probs"], batch["origin_index"], batch["node_mask"]):
                        n = int(mask.sum())
                        rows.append(evaluate_origin_localization({i: float(probs[i]) for i in range(n)}, int(label)))
                condition_rows.append(
                    {
                        "horizon": horizon,
                        "retention": retention,
                        "mrr": float(np.mean([row["mrr"] for row in rows])),
                        "hit1": float(np.mean([row["hit@1"] for row in rows])),
                    }
                )
        mrr = float(np.mean([row["mrr"] for row in condition_rows])); hit1 = float(np.mean([row["hit1"] for row in condition_rows]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mrr": mrr, "val_hit1": hit1, "validation_conditions": condition_rows, "epoch_seconds": float(time.perf_counter() - epoch_started)})
        print(f"[{args.architecture}] epoch={epoch} val_mrr={mrr:.4f} hit1={hit1:.4f}", flush=True)
        if mrr > best[0]: best = (mrr, {k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch)
    torch.save(
        {
            "architecture": args.architecture,
            "model_state_dict": best[1],
            "run_seed": args.seed,
            "feature_contract": "observable_v4_cross_dataset_aligned",
            "implementation_source": getattr(model, "implementation_source", "internal"),
            "training_protocol": args.protocol,
            "validation_conditions": validation_conditions,
        },
        args.out_dir / "best.pt",
    )
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump({"architecture": args.architecture, "run_seed": args.seed, "training_protocol": args.protocol, "best_epoch": best[2], "best_val_mrr": best[0], "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "total_training_seconds": float(time.perf_counter() - training_started), "history": history}, handle, indent=2)


if __name__ == "__main__":
    main()
