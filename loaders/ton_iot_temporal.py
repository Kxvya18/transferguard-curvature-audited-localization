"""Build real temporal graph samples from TON_IoT network scenarios."""

from __future__ import annotations

from collections import Counter, defaultdict
from ipaddress import ip_address
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from graph_builders.windowed_communication import CommunicationEvent, build_time_windowed_graphs
from models.data_structures import TemporalGraphSample
from models.feature_schema import dynamic_feature_names
from priors.boltzmann_prior import (
    beta_estimate_reliability,
    graph_concentration_features,
    mle_beta,
    origin_prior_distribution,
)
from synthetic.iot_growth import ROLE_ORDER


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def infer_temporal_attachment_history_from_events(events: list[CommunicationEvent]) -> list[dict]:
    """Construct a pseudo-growth history from first-seen communication arrivals.

    This is not identical to BA growth, but it gives a principled temporal
    sequence for estimating a structural concentration parameter on real traffic.
    """
    seen: set[str] = set()
    seen_edges: set[frozenset[str]] = set()
    degrees: dict[str, int] = {}
    history: list[dict] = []

    for event in sorted(events, key=lambda item: item.timestamp):
        source, target = event.src, event.dst
        if source == target:
            continue
        edge_key = frozenset((source, target))
        if edge_key in seen_edges:
            continue

        source_seen = source in seen
        target_seen = target in seen
        if not source_seen and not target_seen:
            seen.update((source, target))
            degrees[source] = 1
            degrees[target] = 1
        elif not source_seen or not target_seen:
            new_node = source if not source_seen else target
            attachment_target = target if not source_seen else source
            history.append(
                {
                    "deg_before": dict(degrees),
                    "targets": [attachment_target],
                    "new_node": new_node,
                    "time": event.timestamp,
                    "m": 1,
                }
            )
            seen.add(new_node)
            degrees[new_node] = 1
            degrees[attachment_target] = degrees.get(attachment_target, 0) + 1
        else:
            degrees[source] = degrees.get(source, 0) + 1
            degrees[target] = degrees.get(target, 0) + 1
        seen_edges.add(edge_key)
    return history


def _event_frame_to_events(frame: pd.DataFrame) -> list[CommunicationEvent]:
    events: list[CommunicationEvent] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        ts = _safe_float(row_dict.get("ts", row_dict.get("timestamp", 0.0)), default=0.0)
        events.append(
            CommunicationEvent(
                timestamp=ts,
                src=str(row_dict.get("src_ip")),
                dst=str(row_dict.get("dst_ip")),
                protocol=str(row_dict.get("proto", "unknown")),
                bytes_out=_safe_float(row_dict.get("src_bytes", 0.0), default=0.0),
                bytes_in=_safe_float(row_dict.get("dst_bytes", 0.0), default=0.0),
                label=str(row_dict.get("type", row_dict.get("label", "normal"))),
            )
        )
    return events


def build_ton_iot_network_sample(
    data_csv: str | Path,
    ground_truth_csv: str | Path | None,
    window_size: float = 300.0,
    time_steps: int = 8,
) -> TemporalGraphSample:
    data_frame = pd.read_csv(data_csv, low_memory=False)
    gt_frame = pd.read_csv(ground_truth_csv, low_memory=False) if ground_truth_csv else pd.DataFrame()

    events = _event_frame_to_events(data_frame)
    history = infer_temporal_attachment_history_from_events(events)
    beta_est = mle_beta(history) if history else None
    beta_val = 1.0 if beta_est is None else float(beta_est.beta_hat)

    graphs = build_time_windowed_graphs(events, window_size=window_size)
    if not graphs:
        raise ValueError(f"No graph windows could be constructed from {data_csv}")
    graphs = graphs[:time_steps]

    node_ids = sorted({event.src for event in events} | {event.dst for event in events})
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    num_nodes = len(node_ids)

    aggregate_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    aggregate_degree = defaultdict(int)
    for _, graph in graphs:
        for u, v in graph.edges():
            i = node_index[str(u)]
            j = node_index[str(v)]
            aggregate_adj[i, j] = 1.0
            aggregate_adj[j, i] = 1.0
            aggregate_degree[str(u)] += 1
            aggregate_degree[str(v)] += 1

    import networkx as nx

    aggregate_graph = nx.Graph()
    for node in node_ids:
        aggregate_graph.add_node(node)
    for i, u in enumerate(node_ids):
        for j in range(i + 1, num_nodes):
            if aggregate_adj[i, j] > 0:
                aggregate_graph.add_edge(u, node_ids[j])

    prior_probs = origin_prior_distribution(aggregate_graph, beta=beta_val, m=2)
    concentration = graph_concentration_features(aggregate_graph, beta=beta_val, m=2)

    features = np.zeros((time_steps, num_nodes, 7), dtype=np.float32)
    structural = np.zeros((num_nodes, 5), dtype=np.float32)
    observed_mask = np.zeros(num_nodes, dtype=np.float32)
    infection_times = np.full(num_nodes, fill_value=-1.0, dtype=np.float32)

    gt_by_node: dict[str, float] = {}
    if not gt_frame.empty:
        for row in gt_frame.itertuples(index=False):
            row_dict = row._asdict()
            ts = _safe_float(row_dict.get("ts", 0.0), default=0.0)
            src = str(row_dict.get("src_ip"))
            dst = str(row_dict.get("dst_ip"))
            gt_by_node[src] = min(ts, gt_by_node.get(src, ts))
            gt_by_node[dst] = min(ts, gt_by_node.get(dst, ts))

    if gt_by_node:
        origin_node = min(gt_by_node.items(), key=lambda item: item[1])[0]
    else:
        origin_node = max(prior_probs.items(), key=lambda item: item[1])[0]

    for idx, node in enumerate(node_ids):
        prior = float(prior_probs.get(node, 0.0))
        degree = float(aggregate_graph.degree(node))
        structural[idx, :] = np.asarray(
            [
                np.log1p(degree),
                prior,
                float(concentration["prior_ambiguity"]),
                float(concentration["degree_concentration"]),
                float(beta_val),
            ],
            dtype=np.float32,
        )

        if node in gt_by_node:
            infection_times[idx] = float(gt_by_node[node])

    for step_idx, (window_start, graph) in enumerate(graphs[:time_steps]):
        degs = dict(graph.degree())
        out_bytes = defaultdict(float)
        in_bytes = defaultdict(float)
        active_in_window = set()
        seen_before_or_in_window = set()

        for event in events:
            if window_start <= event.timestamp < window_start + window_size:
                out_bytes[event.src] += event.bytes_out
                in_bytes[event.dst] += event.bytes_in
                active_in_window.add(event.src)
                active_in_window.add(event.dst)
            if event.timestamp <= window_start + window_size:
                seen_before_or_in_window.add(event.src)
                seen_before_or_in_window.add(event.dst)

        max_degree = max(degs.values()) if degs else 1
        max_out = max(out_bytes.values()) if out_bytes else 1.0
        max_in = max(in_bytes.values()) if in_bytes else 1.0

        for idx, node in enumerate(node_ids):
            degree = float(degs.get(node, 0.0))
            features[step_idx, idx, :] = np.asarray(
                [
                            degree / max(max_degree, 1.0),
                            np.log1p(degree),
                            float(out_bytes.get(node, 0.0) / max(max_out, 1.0)),
                            float(in_bytes.get(node, 0.0) / max(max_in, 1.0)),
                            float(node in active_in_window),
                            float(node in seen_before_or_in_window),
                            float(prior_probs.get(node, 0.0)),
                        ],
                        dtype=np.float32,
                    )
            observed_mask[idx] = max(observed_mask[idx], features[step_idx, idx, 5])

    return TemporalGraphSample(
        node_ids=list(node_ids),
        adjacency=aggregate_adj,
        node_features=features,
        structural_features=structural,
        node_mask=np.ones(num_nodes, dtype=np.float32),
        origin_index=int(node_index[origin_node]),
        infection_times=infection_times,
        observed_mask=observed_mask,
        meta={
            "scenario_csv": str(data_csv),
            "ground_truth_csv": None if ground_truth_csv is None else str(ground_truth_csv),
            "beta_hat": beta_val,
            "origin_node": origin_node,
        },
    )


def _ton_role_vector(node: str) -> np.ndarray:
    vector = np.zeros(len(ROLE_ORDER), dtype=np.float32)
    try:
        if ip_address(node).is_private:
            vector[ROLE_ORDER.index("client")] = 1.0
    except ValueError:
        pass
    return vector


def _log_normalize(values: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(values, 0.0))
    return logged / max(float(np.max(logged)), 1.0)


def build_ton_iot_network_sample_v4(
    data_csv: str | Path,
    ground_truth_csv: str | Path,
    window_size: float = 300.0,
    pre_attack_steps: int = 2,
    post_attack_steps: int = 4,
    max_nodes: int = 256,
) -> TemporalGraphSample:
    """Build a leakage-controlled TON_IoT sample under the v4 feature contract."""
    frame = pd.read_csv(data_csv, low_memory=False)
    ground_truth = pd.read_csv(ground_truth_csv, low_memory=False)
    if ground_truth.empty:
        raise ValueError(f"No attack ground truth in {ground_truth_csv}")
    source_counts = Counter(ground_truth["src_ip"].astype(str))
    origin = min(source_counts, key=lambda node: (-source_counts[node], node))
    onset = float(ground_truth.loc[ground_truth["src_ip"].astype(str) == origin, "ts"].min())
    target_source = "packaged_ground_truth"
    time_steps = int(pre_attack_steps + post_attack_steps)
    start = onset - pre_attack_steps * window_size
    stop = onset + post_attack_steps * window_size
    context = frame[(frame["ts"] >= start) & (frame["ts"] < stop)].copy()
    if context.empty:
        attack_rows = frame[~frame["label"].astype(str).str.lower().isin({"0", "normal", "benign"})]
        if attack_rows.empty:
            raise ValueError(f"No TON_IoT flows around attack onset {onset}")
        source_counts = Counter(attack_rows["src_ip"].astype(str).tolist())
        origin = min(source_counts, key=lambda node: (-source_counts[node], node))
        onset = float(attack_rows.loc[attack_rows["src_ip"].astype(str) == origin, "ts"].min())
        start = onset - pre_attack_steps * window_size
        stop = onset + post_attack_steps * window_size
        context = frame[(frame["ts"] >= start) & (frame["ts"] < stop)].copy()
        target_source = "processed_attack_label_due_to_nonoverlapping_packaged_ground_truth"
    context["src_ip"] = context["src_ip"].astype(str)
    context["dst_ip"] = context["dst_ip"].astype(str)
    activity = Counter(context["src_ip"].tolist())
    activity.update(context["dst_ip"].tolist())
    selected = [node for node, _ in activity.most_common(max_nodes - 1)]
    if origin not in selected:
        selected.append(origin)
    selected = selected[:max_nodes]
    selected_set = set(selected)
    context = context[
        context["src_ip"].isin(selected_set) & context["dst_ip"].isin(selected_set)
    ].copy()
    graph = nx.from_pandas_edgelist(context, "src_ip", "dst_ip", create_using=nx.Graph)
    graph.add_nodes_from(selected)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    node_ids = sorted(graph.nodes(), key=lambda node: (-activity[node], node))
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    n_nodes = len(node_ids)
    adjacency = nx.to_numpy_array(graph, nodelist=node_ids, dtype=np.float32)

    events = [
        CommunicationEvent(
            timestamp=float(row.ts),
            src=str(row.src_ip),
            dst=str(row.dst_ip),
            protocol=str(row.proto),
            bytes_out=_safe_float(row.src_bytes),
            bytes_in=_safe_float(row.dst_bytes),
            label=None,
        )
        for row in context.itertuples(index=False)
    ]
    history = infer_temporal_attachment_history_from_events(events)
    beta_est = mle_beta(history) if history else None
    beta_hat = 0.0 if beta_est is None else float(beta_est.beta_hat)
    reliability = 0.0 if beta_est is None else beta_estimate_reliability(beta_est)
    prior = origin_prior_distribution(graph, beta_hat, m=1, reliability=reliability)
    concentration = graph_concentration_features(
        graph, beta_hat, m=1, reliability=reliability
    )

    out_events = np.zeros((time_steps, n_nodes), dtype=np.float32)
    in_events = np.zeros_like(out_events)
    out_bytes = np.zeros_like(out_events)
    in_bytes = np.zeros_like(out_events)
    for row in context.itertuples(index=False):
        step = int((float(row.ts) - start) // window_size)
        if not 0 <= step < time_steps:
            continue
        source = node_index[str(row.src_ip)]
        target = node_index[str(row.dst_ip)]
        out_events[step, source] += 1.0
        in_events[step, target] += 1.0
        out_bytes[step, source] += _safe_float(row.src_bytes)
        in_bytes[step, target] += _safe_float(row.dst_bytes)
    total_events = out_events + in_events
    baseline = np.mean(total_events[:pre_attack_steps], axis=0)
    change_raw = np.maximum(np.log1p(total_events) - np.log1p(baseline[None, :]), 0.0)
    change = np.asarray(
        [row / max(float(np.max(row)), 1e-6) for row in change_raw], dtype=np.float32
    )

    names = dynamic_feature_names(ROLE_ORDER)
    features = np.zeros((time_steps, n_nodes, len(names)), dtype=np.float32)
    for step in range(time_steps):
        step_rows = context[
            ((context["ts"] - start) // window_size).astype(int) == step
        ]
        step_graph = nx.from_pandas_edgelist(
            step_rows, "src_ip", "dst_ip", create_using=nx.Graph
        )
        step_graph.add_nodes_from(node_ids)
        degrees = np.asarray([step_graph.degree(node) for node in node_ids], dtype=np.float32)
        max_degree = max(float(degrees.max()) if degrees.size else 0.0, 1.0)
        log_degrees = _log_normalize(degrees)
        for idx, node in enumerate(node_ids):
            neighbors = list(graph.neighbors(node))
            neighbor_change = (
                float(np.mean([change[step, node_index[nbr]] for nbr in neighbors]))
                if neighbors
                else 0.0
            )
            features[step, idx] = np.concatenate(
                (
                    np.asarray(
                        [
                            degrees[idx] / max_degree,
                            log_degrees[idx],
                            out_events[step, idx] / max(float(np.max(out_events[step])), 1.0),
                            in_events[step, idx] / max(float(np.max(in_events[step])), 1.0),
                            _log_normalize(out_bytes[step])[idx],
                            _log_normalize(in_bytes[step])[idx],
                            change[step, idx],
                            neighbor_change,
                        ],
                        dtype=np.float32,
                    ),
                    _ton_role_vector(node),
                )
            )
    aggregate_out = np.sum(out_events, axis=0)
    aggregate_in = np.sum(in_events, axis=0)
    structural = np.zeros((n_nodes, 7), dtype=np.float32)
    for idx, node in enumerate(node_ids):
        total = float(aggregate_out[idx] + aggregate_in[idx])
        structural[idx] = np.asarray(
            [
                np.log1p(graph.degree(node)),
                float(prior.get(node, 0.0)),
                float(concentration["prior_ambiguity"]),
                float((aggregate_out[idx] - aggregate_in[idx]) / max(total, 1.0)),
                np.log1p(total),
                beta_hat / 4.0,
                reliability,
            ],
            dtype=np.float32,
        )
    infection_times = np.full(n_nodes, -1.0, dtype=np.float32)
    infection_times[node_index[origin]] = 0.0
    return TemporalGraphSample(
        node_ids=node_ids,
        adjacency=adjacency,
        node_features=features,
        structural_features=structural,
        node_mask=np.ones(n_nodes, dtype=np.float32),
        origin_index=node_index[origin],
        infection_times=infection_times,
        observed_mask=(np.max(change, axis=0) >= 0.5).astype(np.float32),
        meta={
            "dataset": "TON_IoT",
            "scenario_path": str(data_csv),
            "origin_node_id": origin,
            "origin_target_definition": "dominant source in attack ground truth",
            "origin_target_source": target_source,
            "attack_onset": onset,
            "labels_used_as_features": False,
            "feature_contract": "observable_v4_cross_dataset_aligned",
            "beta_hat": beta_hat,
            "prior_reliability": reliability,
            "beta_boundary_warning": False if beta_est is None else beta_est.boundary_warning,
            "dynamic_feature_names": list(names),
        },
    )
