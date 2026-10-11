"""Learned fusion between structural prior and temporal graph evidence."""

from __future__ import annotations

import torch
from torch import nn


class GatedPosteriorFusion(nn.Module):
    def __init__(self, hidden_dim: int, structural_feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Linear(structural_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
        observation_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prior_embed = self.prior_proj(structural_features)
        gate = self.gate(torch.cat([node_embeddings, prior_embed], dim=-1))
        fused = gate * node_embeddings + (1.0 - gate) * prior_embed
        logits = self.head(fused).squeeze(-1)
        logits = logits.masked_fill(node_mask <= 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return {"logits": logits, "probs": probs, "entropy": entropy, "fused_embeddings": fused}


class ConcatPosteriorFusion(nn.Module):
    def __init__(self, hidden_dim: int, structural_feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.prior_proj = nn.Sequential(nn.Linear(structural_feature_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
        observation_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prior_embed = self.prior_proj(structural_features)
        fused = torch.cat([node_embeddings, prior_embed], dim=-1)
        logits = self.head(fused).squeeze(-1)
        logits = logits.masked_fill(node_mask <= 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return {"logits": logits, "probs": probs, "entropy": entropy, "fused_embeddings": fused}


class EnergyPosteriorFusion(nn.Module):
    def __init__(self, hidden_dim: int, structural_feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim + structural_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
        observation_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat([node_embeddings, structural_features], dim=-1)
        energies = self.energy_head(fused).squeeze(-1)
        logits = (-energies).masked_fill(node_mask <= 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return {"logits": logits, "probs": probs, "entropy": entropy, "fused_embeddings": fused}


class SafeResidualPosteriorFusion(nn.Module):
    """Add prior evidence only through an interpretable graph-level gate."""

    def __init__(self, hidden_dim: int, structural_feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.applicability = nn.Sequential(
            nn.Linear(structural_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.max_prior_strength = 2.0

    def forward(
        self,
        node_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
        observation_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = node_mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        structural_mean = (structural_features * mask).sum(dim=1) / denominator
        masked_structural = structural_features.masked_fill(mask <= 0, float("-inf"))
        structural_max = masked_structural.max(dim=1).values
        structural_max = torch.nan_to_num(structural_max, neginf=0.0)
        prior_strength = self.max_prior_strength * self.applicability(
            torch.cat([structural_mean, structural_max], dim=-1)
        )

        evidence_logits = self.evidence_head(node_embeddings).squeeze(-1)
        prior_probability = structural_features[:, :, 1].clamp_min(1e-12)
        log_prior = prior_probability.log()
        centered_log_prior = log_prior - (
            (log_prior * node_mask).sum(dim=1, keepdim=True)
            / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        centered_log_prior = centered_log_prior.clamp(-8.0, 8.0)
        logits = evidence_logits + prior_strength * centered_log_prior
        logits = logits.masked_fill(node_mask <= 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "entropy": entropy,
            "fused_embeddings": node_embeddings,
            "prior_strength": prior_strength.squeeze(-1),
            "evidence_logits": evidence_logits,
        }


class ColdStartResidualPosteriorFusion(nn.Module):
    """Fisher-bounded prior fusion conditioned on available temporal evidence.

    ``observation_context`` contains the observed horizon fraction and telemetry
    retention. The learned applicability is reported separately from Fisher
    reliability so an experiment can test, rather than assume, when the prior
    is used.
    """

    def __init__(self, hidden_dim: int, structural_feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        context_dim = structural_feature_dim * 2 + hidden_dim * 2 + 2
        self.applicability = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.max_prior_strength = 2.0
        self.selection_threshold: float | None = None

    def forward(
        self,
        node_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        node_mask: torch.Tensor,
        observation_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = node_mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        structural_mean = (structural_features * mask).sum(dim=1) / denominator
        structural_max = torch.nan_to_num(
            structural_features.masked_fill(mask <= 0, float("-inf")).max(dim=1).values,
            neginf=0.0,
        )
        temporal_mean = (node_embeddings * mask).sum(dim=1) / denominator
        centered = (node_embeddings - temporal_mean.unsqueeze(1)) * mask
        temporal_scale = torch.sqrt(
            centered.square().sum(dim=1) / denominator + 1e-8
        )
        if observation_context is None:
            observation_context = torch.ones(
                (node_embeddings.size(0), 2),
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
        observation_context = observation_context.to(
            device=node_embeddings.device, dtype=node_embeddings.dtype
        )
        applicability = self.applicability(
            torch.cat(
                [
                    structural_mean,
                    structural_max,
                    temporal_mean,
                    temporal_scale,
                    observation_context,
                ],
                dim=-1,
            )
        )
        fisher_reliability = structural_mean[:, 6:7].clamp(0.0, 1.0)
        prior_strength = self.max_prior_strength * fisher_reliability * applicability
        if self.selection_threshold is not None:
            prior_strength = prior_strength * (
                fisher_reliability > float(self.selection_threshold)
            ).to(prior_strength.dtype)

        evidence_logits = self.evidence_head(node_embeddings).squeeze(-1)
        prior_probability = structural_features[:, :, 1].clamp_min(1e-12)
        log_prior = prior_probability.log()
        centered_log_prior = log_prior - (
            (log_prior * node_mask).sum(dim=1, keepdim=True)
            / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        logits = evidence_logits + prior_strength * centered_log_prior.clamp(-8.0, 8.0)
        logits = logits.masked_fill(node_mask <= 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "entropy": entropy,
            "fused_embeddings": node_embeddings,
            "prior_strength": prior_strength.squeeze(-1),
            "prior_applicability": applicability.squeeze(-1),
            "fisher_reliability": fisher_reliability.squeeze(-1),
            "observation_context": observation_context,
            "evidence_logits": evidence_logits,
        }


def build_fusion_module(
    fusion_type: str,
    hidden_dim: int,
    structural_feature_dim: int,
    dropout: float = 0.2,
) -> nn.Module:
    fusion_key = fusion_type.lower()
    if fusion_key == "concat":
        return ConcatPosteriorFusion(hidden_dim, structural_feature_dim, dropout)
    if fusion_key == "energy":
        return EnergyPosteriorFusion(hidden_dim, structural_feature_dim, dropout)
    if fusion_key == "safe_residual":
        return SafeResidualPosteriorFusion(hidden_dim, structural_feature_dim, dropout)
    if fusion_key == "cold_start_residual":
        return ColdStartResidualPosteriorFusion(hidden_dim, structural_feature_dim, dropout)
    return GatedPosteriorFusion(hidden_dim, structural_feature_dim, dropout)
