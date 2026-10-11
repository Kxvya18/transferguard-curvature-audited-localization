"""Shared controls for anytime cold-start training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from models.transferguard_model import TransferGuardModel

MAX_WINDOWS = 6
TELEMETRY_CHANNELS = slice(2, 8)


def prepare_prefix_batch(
    batch: dict[str, torch.Tensor],
    horizon: int,
    telemetry_retention: float,
    mask_seed: int,
    ablate_prior: bool,
) -> dict[str, torch.Tensor]:
    """Create a leakage-free temporal prefix with a reproducible telemetry mask."""
    available = int(batch["node_features"].size(1))
    horizon = max(1, min(int(horizon), available))
    retention = float(max(0.0, min(float(telemetry_retention), 1.0)))
    features = batch["node_features"][:, :horizon].clone()
    generator = torch.Generator(device="cpu").manual_seed(int(mask_seed))
    observed = torch.rand(
        (features.size(0), horizon, features.size(2), 1), generator=generator
    ) < retention
    observed = observed & (batch["node_mask"].unsqueeze(1).unsqueeze(-1) > 0)
    features[:, :, :, TELEMETRY_CHANNELS] *= observed.to(features.dtype)

    structural = batch["structural_features"].clone()
    # This feature aggregates all six windows in the base dataset and would leak
    # future evidence into a prefix experiment.
    structural[:, :, 4] = 0.0
    if ablate_prior:
        uniform = 1.0 / batch["node_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
        structural[:, :, 1] = uniform
        structural[:, :, 2] = 1.0
        structural[:, :, 5:7] = 0.0

    context = torch.tensor(
        [horizon / float(MAX_WINDOWS), retention], dtype=torch.float32
    ).repeat(features.size(0), 1)
    return {
        **batch,
        "node_features": features,
        "structural_features": structural,
        "ssl_mask": batch["ssl_mask"][:, :horizon],
        "observation_context": context,
    }


def build_cold_start_model(checkpoint: dict, device: torch.device) -> TransferGuardModel:
    state = checkpoint["model_state_dict"]
    training = checkpoint["training_config"]
    model = TransferGuardModel(
        node_feature_dim=int(state["encoder.node_proj.weight"].shape[1]),
        structural_feature_dim=int(state["encoder.struct_proj.weight"].shape[1]),
        hidden_dim=int(training["hidden_dim"]),
        num_message_layers=int(training["num_message_layers"]),
        dropout=float(training["dropout"]),
        fusion_type="cold_start_residual",
        prior_in_encoder=False,
    )
    selection_threshold = training.get("selection_threshold")
    if selection_threshold is not None:
        model.fusion.selection_threshold = float(selection_threshold)
    model.load_state_dict(state)
    return model.to(device)


def load_cold_start_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TransferGuardModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return build_cold_start_model(checkpoint, device), checkpoint
