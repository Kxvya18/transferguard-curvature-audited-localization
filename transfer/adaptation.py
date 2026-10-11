"""Domain-adaptation utilities for cross-scenario and cross-dataset transfer."""

from __future__ import annotations

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL alignment loss on graph or node embeddings."""
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    source_cov = (source.T @ source) / max(source.size(0) - 1, 1)
    target_cov = (target.T @ target) / max(target.size(0) - 1, 1)
    return ((source_cov - target_cov) ** 2).mean()


def fit_prototypes(embeddings: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    prototypes: dict[int, torch.Tensor] = {}
    for label in labels.unique():
        mask = labels == label
        prototypes[int(label.item())] = embeddings[mask].mean(dim=0)
    return prototypes


def prototypical_logits(
    embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
) -> torch.Tensor:
    ordered = [prototypes[key] for key in sorted(prototypes.keys())]
    proto = torch.stack(ordered, dim=0)
    distances = torch.cdist(embeddings, proto)
    return -distances

