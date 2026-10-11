"""Trainable TRANSFERGUARD model."""

from __future__ import annotations

import torch
from torch import nn

from fusion.learned_fusion import build_fusion_module
from self_supervised.temporal_encoder import TemporalGraphEncoder, TemporalSSLPretrainer


class TransferGuardModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        structural_feature_dim: int,
        hidden_dim: int = 64,
        num_message_layers: int = 2,
        dropout: float = 0.15,
        fusion_type: str = "gated",
        prior_in_encoder: bool = True,
    ):
        super().__init__()
        self.prior_in_encoder = bool(prior_in_encoder)
        self.encoder = TemporalGraphEncoder(
            node_feature_dim=node_feature_dim,
            structural_feature_dim=structural_feature_dim,
            hidden_dim=hidden_dim,
            num_message_layers=num_message_layers,
            dropout=dropout,
        )
        self.ssl_head = TemporalSSLPretrainer(
            hidden_dim=hidden_dim,
            node_feature_dim=node_feature_dim,
        )
        self.fusion = build_fusion_module(
            fusion_type=fusion_type,
            hidden_dim=hidden_dim,
            structural_feature_dim=structural_feature_dim,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
        ssl_mask: torch.Tensor | None = None,
        observation_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_structural = structural_features
        if not self.prior_in_encoder:
            encoder_structural = structural_features.clone()
            uniform = 1.0 / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            encoder_structural[:, :, 1] = uniform
            encoder_structural[:, :, 2] = 1.0
            if encoder_structural.size(-1) >= 7:
                encoder_structural[:, :, 5:7] = 0.0
        encoded = self.encoder(
            node_features=node_features,
            adjacency=adjacency,
            structural_features=encoder_structural,
            node_mask=node_mask,
        )
        fused = self.fusion(
            node_embeddings=encoded["node_embeddings"],
            structural_features=structural_features,
            node_mask=node_mask,
            observation_context=observation_context,
        )
        out = {**encoded, **fused}
        if ssl_mask is not None:
            out.update(
                self.ssl_head(
                    sequence_embeddings=encoded["sequence_embeddings"],
                    original_features=node_features,
                    mask=ssl_mask,
                )
            )
        return out
