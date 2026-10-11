"""Temporal graph encoder with lightweight self-supervised objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype)
    adj = adjacency + identity.unsqueeze(0)
    degree = adj.sum(dim=-1)
    inv_sqrt = torch.pow(degree.clamp_min(1.0), -0.5)
    return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)


class GraphMessageLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.nbr_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        agg = torch.bmm(norm_adj, features)
        out = self.self_proj(features) + self.nbr_proj(agg)
        return self.norm(F.relu(out))


class TemporalGraphEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        structural_feature_dim: int,
        hidden_dim: int = 64,
        num_message_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.struct_proj = nn.Linear(structural_feature_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim, hidden_dim) for _ in range(num_message_layers)]
        )
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_steps, num_nodes, _ = node_features.shape
        norm_adj = _normalize_adjacency(adjacency)
        struct_embed = self.struct_proj(structural_features)

        temporal_inputs = []
        for step in range(num_steps):
            x_step = self.node_proj(node_features[:, step, :, :]) + struct_embed
            for layer in self.message_layers:
                x_step = layer(x_step, norm_adj)
            temporal_inputs.append(self.dropout(x_step))

        stacked = torch.stack(temporal_inputs, dim=2)
        flat = stacked.reshape(batch_size * num_nodes, num_steps, self.hidden_dim)
        encoded, hidden = self.temporal_gru(flat)
        final_node = hidden[-1].reshape(batch_size, num_nodes, self.hidden_dim)
        final_node = final_node * node_mask.unsqueeze(-1)
        graph_embed = final_node.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "node_embeddings": final_node,
            "sequence_embeddings": encoded.reshape(batch_size, num_nodes, num_steps, self.hidden_dim),
            "graph_embedding": graph_embed,
        }


class TemporalSSLPretrainer(nn.Module):
    def __init__(self, hidden_dim: int, node_feature_dim: int):
        super().__init__()
        self.reconstruct = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim),
        )
        self.next_state = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim),
        )

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        original_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seq = sequence_embeddings.permute(0, 2, 1, 3)
        recon = self.reconstruct(seq)
        recon_loss = (
            F.mse_loss(recon[mask], original_features[mask])
            if mask.any()
            else torch.tensor(0.0, device=original_features.device)
        )
        if original_features.size(1) > 1:
            pred_next = self.next_state(seq[:, :-1, :, :])
            target_next = original_features[:, 1:, :, :]
            next_loss = F.mse_loss(pred_next, target_next)
        else:
            next_loss = torch.tensor(0.0, device=original_features.device)
        return {
            "reconstruction_loss": recon_loss,
            "next_step_loss": next_loss,
            "ssl_loss": recon_loss + next_loss,
        }

