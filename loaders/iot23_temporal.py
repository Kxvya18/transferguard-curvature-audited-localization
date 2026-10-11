"""Leakage-controlled temporal origin-localization samples for IoT-23."""

from __future__ import annotations

from collections import Counter
from ipaddress import ip_address
from pathlib import Path

import networkx as nx
import numpy as np

from graph_builders.windowed_communication import CommunicationEvent
from loaders.iot23 import load_iot23_events
from loaders.ton_iot_temporal import infer_temporal_attachment_history_from_events
from models.data_structures import TemporalGraphSample
from models.feature_schema import dynamic_feature_names
from priors.boltzmann_prior import (
    beta_estimate_reliability,
    graph_concentration_features,
    mle_beta,
    origin_prior_distribution,
)
from synthetic.iot_growth import ROLE_ORDER


def is_attack_event(event: CommunicationEvent) -> bool:
    return bool(event.label and event.label.strip().lower() not in {"benign", "normal"})


def infer_attack_origin(events: list[CommunicationEvent]) -> tuple[str, float, Counter]:
    """Infer the scenario's infected source from labeled attack flows.

    Labels are used only here to define the evaluation target and attack onset.
    They are discarded before feature construction.
    """
    attack_events = [event for event in events if is_attack_event(event)]
    if not attack_events:
        raise ValueError("IoT-23 scenario has no labeled attack events")

    source_counts = Counter(event.src for event in attack_events)
    first_attack = {}
    for event in attack_events:
        first_attack[event.src] = min(event.timestamp, first_attack.get(event.src, event.timestamp))
    origin = min(source_counts, key=lambda node: (-source_counts[node], first_attack[node], node))
    onset = min(event.timestamp for event in attack_events if event.src == origin)
    labels = Counter(event.label for event in attack_events)
    return origin, float(onset), labels


def _is_private_endpoint(node: str) -> bool:
    try:
        return bool(ip_address(node).is_private)
    except ValueError:
        return False


def _role_vector(node: str) -> np.ndarray:
    vector = np.zeros(len(ROLE_ORDER), dtype=np.float32)
    if _is_private_endpoint(node):
        vector[ROLE_ORDER.index("client")] = 1.0
    return vector


def _safe_log_normalize(values: np.ndarray) -> np.ndarray:
    transformed = np.log1p(np.maximum(values, 0.0))
    maximum = float(np.max(transformed)) if transformed.size else 0.0
    return transformed / max(maximum, 1.0)


def build_iot23_scenario_sample(
    path: str | Path,
    window_size: float = 300.0,
    pre_attack_steps: int = 2,
    post_attack_steps: int = 6,
    max_nodes: int = 256,
) -> TemporalGraphSample:
    """Build one scenario-level sample from raw flows around attack onset.

    Model features contain raw flow activity, bytes, topology, and an unsupervised
    activity-change signal. IoT-23 labels determine only ``origin_index`` and
    metadata; no attack label or compromise flag enters the feature tensors.
    """
    file_path = Path(path)
    events = load_iot23_events(file_path)
    origin, attack_onset, attack_labels = infer_attack_origin(events)
    time_steps = int(pre_attack_steps + post_attack_steps)
    start_time = attack_onset - pre_attack_steps * window_size
    end_time = attack_onset + post_attack_steps * window_size
    context_events = [event for event in events if start_time <= event.timestamp < end_time]
    if not context_events:
        raise ValueError(f"No events in attack context for {file_path}")

    endpoint_activity = Counter()
    for event in context_events:
        endpoint_activity[event.src] += 1
        endpoint_activity[event.dst] += 1
    selected = [node for node, _ in endpoint_activity.most_common(max(max_nodes - 1, 0))]
    if origin not in selected:
        selected.append(origin)
    selected = selected[:max_nodes]
    selected_set = set(selected)

    filtered_events = [
        event for event in context_events if event.src in selected_set and event.dst in selected_set
    ]
    graph = nx.Graph()
    graph.add_nodes_from(selected)
    for event in filtered_events:
        if event.src != event.dst:
            graph.add_edge(event.src, event.dst)
    if origin not in graph:
        raise ValueError(f"Inferred origin {origin} was lost during graph construction")

    node_ids = sorted(graph.nodes(), key=lambda node: (-endpoint_activity[node], node))
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    num_nodes = len(node_ids)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in graph.edges():
        source_idx = node_index[source]
        target_idx = node_index[target]
        adjacency[source_idx, target_idx] = 1.0
        adjacency[target_idx, source_idx] = 1.0

    history = infer_temporal_attachment_history_from_events(filtered_events)
    beta_est = mle_beta(history) if history else None
    beta_hat = 0.0 if beta_est is None else float(beta_est.beta_hat)
    prior_reliability = 0.0 if beta_est is None else beta_estimate_reliability(beta_est)
    prior = origin_prior_distribution(
        graph,
        beta=beta_hat,
        m=1,
        reliability=prior_reliability,
    )
    concentration = graph_concentration_features(
        graph,
        beta=beta_hat,
        m=1,
        reliability=prior_reliability,
    )

    out_events = np.zeros((time_steps, num_nodes), dtype=np.float32)
    in_events = np.zeros((time_steps, num_nodes), dtype=np.float32)
    out_bytes = np.zeros((time_steps, num_nodes), dtype=np.float32)
    in_bytes = np.zeros((time_steps, num_nodes), dtype=np.float32)
    first_seen = np.full(num_nodes, np.inf, dtype=np.float64)
    for event in filtered_events:
        step = int((event.timestamp - start_time) // window_size)
        if step < 0 or step >= time_steps:
            continue
        source_idx = node_index[event.src]
        target_idx = node_index[event.dst]
        out_events[step, source_idx] += 1.0
        in_events[step, target_idx] += 1.0
        out_bytes[step, source_idx] += float(event.bytes_out)
        in_bytes[step, target_idx] += float(event.bytes_in)
        first_seen[source_idx] = min(first_seen[source_idx], event.timestamp)
        first_seen[target_idx] = min(first_seen[target_idx], event.timestamp)

    activity = out_events + in_events
    baseline = np.mean(activity[:pre_attack_steps], axis=0) if pre_attack_steps else np.zeros(num_nodes)
    anomaly_raw = np.maximum(np.log1p(activity) - np.log1p(baseline[None, :]), 0.0)
    anomaly = np.zeros_like(anomaly_raw, dtype=np.float32)
    for step in range(time_steps):
        step_max = float(np.max(anomaly_raw[step]))
        anomaly[step] = anomaly_raw[step] / max(step_max, 1e-6)

    dynamic_names = dynamic_feature_names(ROLE_ORDER)
    features = np.zeros((time_steps, num_nodes, len(dynamic_names)), dtype=np.float32)
    for step in range(time_steps):
        graph_step = nx.Graph()
        graph_step.add_nodes_from(node_ids)
        for event in filtered_events:
            event_step = int((event.timestamp - start_time) // window_size)
            if event_step == step and event.src != event.dst:
                graph_step.add_edge(event.src, event.dst)
        degrees = np.asarray([graph_step.degree(node) for node in node_ids], dtype=np.float32)
        degree_norm = degrees / max(float(np.max(degrees)), 1.0)
        log_degree_norm = _safe_log_normalize(degrees)
        out_event_norm = out_events[step] / max(float(np.max(out_events[step])), 1.0)
        in_event_norm = in_events[step] / max(float(np.max(in_events[step])), 1.0)
        out_byte_norm = _safe_log_normalize(out_bytes[step])
        in_byte_norm = _safe_log_normalize(in_bytes[step])
        for idx, node in enumerate(node_ids):
            neighbors = list(graph.neighbors(node))
            neighbor_anomaly = (
                float(np.mean([anomaly[step, node_index[nbr]] for nbr in neighbors]))
                if neighbors
                else 0.0
            )
            features[step, idx] = np.concatenate(
                (
                    np.asarray(
                        [
                            degree_norm[idx],
                            log_degree_norm[idx],
                            out_event_norm[idx],
                            in_event_norm[idx],
                            out_byte_norm[idx],
                            in_byte_norm[idx],
                            anomaly[step, idx],
                            neighbor_anomaly,
                        ],
                        dtype=np.float32,
                    ),
                    _role_vector(node),
                )
            )

    aggregate_out = np.sum(out_events, axis=0)
    aggregate_in = np.sum(in_events, axis=0)
    structural = np.zeros((num_nodes, 7), dtype=np.float32)
    for idx, node in enumerate(node_ids):
        total_activity = float(aggregate_out[idx] + aggregate_in[idx])
        asymmetry = float((aggregate_out[idx] - aggregate_in[idx]) / max(total_activity, 1.0))
        structural[idx] = np.asarray(
            [
                np.log1p(graph.degree(node)),
                float(prior.get(node, 0.0)),
                float(concentration["prior_ambiguity"]),
                asymmetry,
                np.log1p(total_activity),
                float(beta_hat / 4.0),
                float(prior_reliability),
            ],
            dtype=np.float32,
        )

    infection_times = np.full(num_nodes, -1.0, dtype=np.float32)
    infection_times[node_index[origin]] = 0.0
    observed_mask = (np.max(anomaly, axis=0) >= 0.5).astype(np.float32)
    return TemporalGraphSample(
        node_ids=node_ids,
        adjacency=adjacency,
        node_features=features,
        structural_features=structural,
        node_mask=np.ones(num_nodes, dtype=np.float32),
        origin_index=int(node_index[origin]),
        infection_times=infection_times,
        observed_mask=observed_mask,
        meta={
            "dataset": "IoT-23",
            "scenario_path": str(file_path),
            "origin_node_id": origin,
            "origin_target_definition": "dominant source among labeled attack flows",
            "attack_onset": attack_onset,
            "attack_label_counts": dict(attack_labels),
            "labels_used_as_features": False,
            "beta_hat": beta_hat,
            "beta_identifiable": beta_est is not None,
            "beta_fisher_information": 0.0 if beta_est is None else beta_est.fisher_information,
            "beta_boundary_warning": False if beta_est is None else beta_est.boundary_warning,
            "prior_reliability": prior_reliability,
            "beta_ci": None if beta_est is None else [beta_est.ci_lower, beta_est.ci_upper],
            "dynamic_feature_names": list(dynamic_names),
            "feature_contract": "observable_v4_cross_dataset_aligned",
            "role_inference": "private IP as client; external endpoint role unknown",
            "first_seen_offsets": [
                None if not np.isfinite(value) else float(value - start_time) for value in first_seen
            ],
        },
    )


def build_iot23_samples_from_root(
    root: str | Path,
    max_log_bytes: int = 500_000_000,
    **sample_kwargs,
) -> list[TemporalGraphSample]:
    paths = sorted(Path(root).rglob("conn.log.labeled"), key=lambda path: path.stat().st_size)
    return [
        build_iot23_scenario_sample(path, **sample_kwargs)
        for path in paths
        if path.stat().st_size <= max_log_bytes
    ]
