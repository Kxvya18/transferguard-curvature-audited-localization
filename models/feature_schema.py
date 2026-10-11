"""Canonical observable feature contract shared by synthetic and real graphs."""

from __future__ import annotations


DYNAMIC_BASE_FEATURES = (
    "degree_normalized",
    "log_degree_normalized",
    "out_event_rate",
    "in_event_rate",
    "out_byte_rate",
    "in_byte_rate",
    "activity_change_signal",
    "neighbor_activity_change_signal",
)

STRUCTURAL_FEATURES = (
    "log_aggregate_degree",
    "boltzmann_origin_prior",
    "prior_ambiguity",
    "directional_asymmetry",
    "log_aggregate_activity",
    "beta_hat_scaled",
    "prior_reliability",
)


def dynamic_feature_names(role_order: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return DYNAMIC_BASE_FEATURES + tuple(f"role_{role}" for role in role_order)
