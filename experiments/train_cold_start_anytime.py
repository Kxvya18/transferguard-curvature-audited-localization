"""Train matched anytime models on randomly truncated and thinned telemetry."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import evaluate_origin_localization
from experiments.cold_start_common import prepare_prefix_batch
from models.transferguard_model import TransferGuardModel
from synthetic.dataset import MixedSyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _split_config(payload: dict) -> tuple[SyntheticScenarioConfig, dict]:
    scenario_keys = set(SyntheticScenarioConfig.__dataclass_fields__)
    return (
        SyntheticScenarioConfig(**{key: value for key, value in payload.items() if key in scenario_keys}),
        {key: value for key, value in payload.items() if key not in scenario_keys},
    )


def _prepare_cache(config_path: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    scenario, training = _split_config(_read(config_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split, count, offset in (
        ("train", int(training["train_samples"]), 0),
        ("val", int(training["val_samples"]), 10_000),
    ):
        datasets[split] = MixedSyntheticOriginDataset(
            base_config=scenario,
            domain_grid=training["domain_grid"],
            num_samples=count,
            seed_offset=offset,
            progress_label=f"cold-{split}",
        ).samples
    payload = {
        "scenario_config": scenario.to_dict(),
        "training_config": training,
        **datasets,
    }
    torch.save(payload, cache_path)
    print(f"[cold-start] cached dataset at {cache_path}", flush=True)
    return payload


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, float]]:
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return [
        evaluate_origin_localization(
            {node: float(value) for node, value in enumerate(probabilities[index])},
            int(labels_np[index]),
        )
        for index in range(len(labels_np))
    ]


@torch.no_grad()
def _validate(
    model: TransferGuardModel,
    loader: DataLoader,
    conditions: list[tuple[int, float]],
    ablate_prior: bool,
    device: torch.device,
) -> tuple[float, list[dict]]:
    model.eval()
    rows = []
    for horizon, retention in conditions:
        metrics = []
        strengths = []
        for batch_index, raw_batch in enumerate(loader):
            batch = prepare_prefix_batch(
                raw_batch,
                horizon,
                retention,
                mask_seed=8_100_000 + horizon * 10_000 + int(retention * 1000) + batch_index,
                ablate_prior=ablate_prior,
            )
            batch = _to_device(batch, device)
            output = model(
                batch["node_features"],
                batch["adjacency"],
                batch["structural_features"],
                batch["node_mask"],
                None,
                batch["observation_context"],
            )
            metrics.extend(_batch_metrics(output["logits"], batch["origin_index"]))
            strengths.extend(output["prior_strength"].cpu().tolist())
        rows.append(
            {
                "horizon": horizon,
                "telemetry_retention": retention,
                "mrr": float(np.mean([row["mrr"] for row in metrics])),
                "hit1": float(np.mean([row["hit@1"] for row in metrics])),
                "mean_prior_strength": float(np.mean(strengths)),
            }
        )
    return float(np.mean([row["mrr"] for row in rows])), rows


def train(args: argparse.Namespace) -> None:
    cached = _prepare_cache(args.config, args.cache)
    if args.prepare_only:
        return
    scenario = SyntheticScenarioConfig(**cached["scenario_config"])
    training = cached["training_config"]
    ablate_prior = args.method == "temporal_only"
    selection_threshold = None
    if args.method == "selective_hybrid":
        validation_reliability = np.asarray(
            [float(np.mean(sample.structural_features[:, 6])) for sample in cached["val"]],
            dtype=float,
        )
        selection_threshold = float(np.quantile(validation_reliability, 0.75))
    run_seed = int(args.run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_generator = torch.Generator().manual_seed(run_seed)
    train_loader = DataLoader(
        cached["train"],
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_temporal_graph_samples,
    )
    val_loader = DataLoader(
        cached["val"],
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=collate_temporal_graph_samples,
    )
    sample = cached["train"][0]
    model = TransferGuardModel(
        node_feature_dim=sample.node_features.shape[-1],
        structural_feature_dim=sample.structural_features.shape[-1],
        hidden_dim=int(training["hidden_dim"]),
        num_message_layers=int(training["num_message_layers"]),
        dropout=float(training["dropout"]),
        fusion_type="cold_start_residual",
        prior_in_encoder=False,
    ).to(device)
    if selection_threshold is not None:
        model.fusion.selection_threshold = selection_threshold
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    horizons = [int(value) for value in training["train_horizons"]]
    retentions = [float(value) for value in training["train_retentions"]]
    validation_conditions = [
        (int(horizon), float(retention))
        for horizon, retention in training["validation_conditions"]
    ]
    output_dir = args.output_root / f"{args.method}_seed{run_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best = {"selection_mrr": -1.0, "epoch": -1, "state_dict": None, "conditions": []}
    history = []

    training_started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        train_metrics = []
        train_strength = []
        schedule_rng = np.random.default_rng(run_seed + epoch * 104_729)
        for batch_index, raw_batch in enumerate(train_loader):
            horizon = int(schedule_rng.choice(horizons))
            retention = float(schedule_rng.choice(retentions))
            batch = prepare_prefix_batch(
                raw_batch,
                horizon,
                retention,
                mask_seed=run_seed * 1_000_003 + epoch * 10_007 + batch_index,
                ablate_prior=ablate_prior,
            )
            batch = _to_device(batch, device)
            output = model(
                batch["node_features"],
                batch["adjacency"],
                batch["structural_features"],
                batch["node_mask"],
                None,
                batch["observation_context"],
            )
            loss = criterion(output["logits"], batch["origin_index"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            train_metrics.extend(_batch_metrics(output["logits"], batch["origin_index"]))
            train_strength.extend(output["prior_strength"].detach().cpu().tolist())

        selection_mrr, condition_rows = _validate(
            model, val_loader, validation_conditions, ablate_prior, device
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_mrr": float(np.mean([row["mrr"] for row in train_metrics])),
            "train_hit1": float(np.mean([row["hit@1"] for row in train_metrics])),
            "mean_train_prior_strength": float(np.mean(train_strength)),
            "selection_mrr": selection_mrr,
            "validation_conditions": condition_rows,
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(record)
        with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(
            f"[cold-start:{args.method}:{run_seed}] epoch={epoch:02d} "
            f"loss={record['train_loss']:.4f} selection_mrr={selection_mrr:.4f} "
            f"strength={record['mean_train_prior_strength']:.4f}",
            flush=True,
        )
        if selection_mrr > best["selection_mrr"]:
            best = {
                "selection_mrr": selection_mrr,
                "epoch": epoch,
                "conditions": condition_rows,
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            }

    checkpoint = {
        "model_state_dict": best["state_dict"],
        "scenario_config": scenario.to_dict(),
        "training_config": {
            **training,
            "fusion_type": "cold_start_residual",
            "prior_in_encoder": False,
            "ablate_boltzmann_prior": ablate_prior,
            "selection_threshold": selection_threshold,
        },
        "run_seed": run_seed,
        "method": args.method,
        "feature_contract": "cold_start_prefix_v1",
    }
    torch.save(checkpoint, output_dir / "best.pt")
    summary = {
        "method": args.method,
        "run_seed": run_seed,
        "best_epoch": int(best["epoch"]),
        "best_selection_mrr": float(best["selection_mrr"]),
        "best_validation_conditions": best["conditions"],
        "n_train": len(cached["train"]),
        "n_val": len(cached["val"]),
        "total_training_seconds": float(time.perf_counter() - training_started),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "selection_threshold": selection_threshold,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/train_cold_start_anytime.json"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/processed/cold_start_anytime_dataset.pt",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/cold_start_v1"
    )
    parser.add_argument(
        "--method",
        choices=("hybrid", "selective_hybrid", "temporal_only"),
        default="hybrid",
    )
    parser.add_argument("--run-seed", type=int, default=20260820)
    parser.add_argument("--prepare-only", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
