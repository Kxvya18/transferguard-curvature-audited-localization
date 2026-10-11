"""Synthetic dataset construction for trainable origin-localization models."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import Dataset

from models.data_structures import TemporalGraphSample
from models.feature_schema import dynamic_feature_names
from priors.boltzmann_prior import (
    beta_estimate_reliability,
    graph_concentration_features,
    mle_beta,
    origin_prior_distribution,
    snapshot_beta_ensemble,
)
from synthetic.iot_growth import ROLE_ORDER, SyntheticScenarioConfig, generate_iot_graph, simulate_attack_cascade


ROLE_TO_INDEX = {role: idx for idx, role in enumerate(ROLE_ORDER)}


def _normalized_adjacency_matrix(graph) -> np.ndarray:
    adj = np.asarray(
        torch.tensor(
            np.array(
                [[1.0 if graph.has_edge(u, v) else 0.0 for v in graph.nodes()] for u in graph.nodes()],
                dtype=np.float32,
            )
        ),
        dtype=np.float32,
    )
    return np.asarray(adj, dtype=np.float32)


def _role_one_hot(role: str) -> np.ndarray:
    vec = np.zeros(len(ROLE_ORDER), dtype=np.float32)
    vec[ROLE_TO_INDEX.get(role, 0)] = 1.0
    return vec


def build_temporal_sample(
    graph,
    history: list[dict],
    cascade,
    time_steps: int = 6,
    config: SyntheticScenarioConfig | None = None,
) -> TemporalGraphSample:
    scenario_config = SyntheticScenarioConfig() if config is None else config
    telemetry_rng = np.random.default_rng(int(scenario_config.seed) + 7001)
    history_mode = str(scenario_config.prior_history_mode)
    snapshot_diagnostics: dict[str, float] = {}
    if history_mode == "observed":
        beta_est = mle_beta(history)
    elif history_mode == "snapshot_ensemble":
        beta_est, snapshot_diagnostics = snapshot_beta_ensemble(
            graph,
            ensemble_size=int(scenario_config.snapshot_ensemble_size),
            seed_base=int(scenario_config.seed) + 31_000,
        )
    else:
        raise ValueError(f"Unknown prior_history_mode={history_mode!r}")
    prior_reliability = beta_estimate_reliability(beta_est)
    prior_probs = origin_prior_distribution(
        graph,
        beta_est.beta_hat,
        m=history[0]["m"] if history else 2,
        reliability=prior_reliability,
    )
    concentration = graph_concentration_features(
        graph,
        beta=beta_est.beta_hat,
        m=history[0]["m"] if history else 2,
        reliability=prior_reliability,
    )

    node_ids = [int(node) for node in graph.nodes()]
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    num_nodes = len(node_ids)
    adjacency = np.asarray(nx_to_adj(graph, node_ids), dtype=np.float32)

    infection_arr = np.full(num_nodes, fill_value=-1.0, dtype=np.float32)
    observed_mask = np.zeros(num_nodes, dtype=np.float32)
    for node, time in cascade.infection_times.items():
        infection_arr[node_index[int(node)]] = float(time)
    for node in cascade.observed_infected:
        observed_mask[node_index[int(node)]] = 1.0

    feature_names = dynamic_feature_names(ROLE_ORDER)
    features = np.zeros((time_steps, num_nodes, len(feature_names)), dtype=np.float32)
    structural = np.zeros((num_nodes, 7), dtype=np.float32)

    max_degree = max((graph.degree(node) for node in node_ids), default=1)

    activity_by_step: list[dict[int, tuple[float, float, float, float]]] = []
    activity_maxima: list[tuple[float, float, float, float]] = []
    anomaly_by_step: list[dict[int, float]] = []
    baseline_activity: dict[int, float] = {}
    for step in range(time_steps):
        raw_activity: dict[int, tuple[float, float, float, float]] = {}
        anomalies: dict[int, float] = {}
        for node in node_ids:
            degree = float(graph.degree(node))
            infection_time = cascade.infection_times.get(node, time_steps + 1)
            newly_exposed = sum(
                cascade.infection_times.get(int(nbr), time_steps + 1) == step + 1
                for nbr in graph.neighbors(node)
            )
            newly_infected = float(infection_time == step)
            active_attacker = float(infection_time <= step)
            role = str(graph.nodes[node].get("role", "sensor"))
            role_activity = {
                "gateway": 1.40,
                "broker": 1.35,
                "edge_server": 1.30,
                "camera": 1.20,
                "sensor": 0.75,
                "actuator": 0.80,
                "client": 1.00,
            }.get(role, 1.0)
            noise_out = float(
                telemetry_rng.lognormal(mean=0.0, sigma=scenario_config.activity_noise_logstd)
            )
            noise_in = float(
                telemetry_rng.lognormal(mean=0.0, sigma=scenario_config.activity_noise_logstd)
            )
            benign_burst = (
                scenario_config.benign_burst_scale
                if telemetry_rng.random() < scenario_config.benign_burst_prob
                else 0.0
            )
            baseline = 0.15 * degree * role_activity
            baseline_activity[node] = 2.0 * baseline
            attack_out = (
                active_attacker
                * scenario_config.attack_activity_scale
                * (1.0 + 1.5 * newly_exposed)
            )
            attack_in = newly_infected * scenario_config.attack_activity_scale
            out_events = baseline * noise_out + attack_out + benign_burst
            in_events = baseline * noise_in + attack_in + 0.5 * benign_burst
            byte_scale = 1.5 if role in {"camera", "edge_server"} else 1.0
            raw_activity[node] = (
                out_events,
                in_events,
                out_events * byte_scale,
                in_events * byte_scale,
            )
            activity_change = max(
                np.log1p(out_events + in_events) - np.log1p(baseline_activity[node]),
                0.0,
            )
            anomalies[node] = float(activity_change)
        max_anomaly = max(anomalies.values(), default=0.0)
        if max_anomaly > 1e-12:
            anomalies = {node: value / max_anomaly for node, value in anomalies.items()}
        activity_by_step.append(raw_activity)
        activity_maxima.append(
            tuple(
                max(values[feature_idx] for values in raw_activity.values())
                for feature_idx in range(4)
            )
        )
        anomaly_by_step.append(anomalies)

    for idx, node in enumerate(node_ids):
        role = str(graph.nodes[node].get("role", "sensor"))
        degree = float(graph.degree(node))
        prior = float(prior_probs.get(node, 0.0))
        aggregate_activity = sum(
            sum(activity_by_step[step][node][:2]) for step in range(time_steps)
        )

        structural[idx, :] = np.asarray(
            [
                np.log1p(degree),
                prior,
                float(concentration["prior_ambiguity"]),
                0.0,
                np.log1p(aggregate_activity),
                float(beta_est.beta_hat / 4.0),
                float(prior_reliability),
            ],
            dtype=np.float32,
        )

        role_vec = _role_one_hot(role)
        for step in range(time_steps):
            step_activity = activity_by_step[step]
            out_events, in_events, out_bytes, in_bytes = step_activity[node]
            max_out_events, max_in_events, max_out_bytes, max_in_bytes = activity_maxima[step]
            observed_by_step = anomaly_by_step[step][node]
            neighbors = list(graph.neighbors(node))
            neighborhood_obs = (
                float(np.mean([anomaly_by_step[step][int(nbr)] for nbr in neighbors]))
                if neighbors
                else 0.0
            )
            feature_vec = np.concatenate(
                [
                    np.asarray(
                        [
                            degree / max(max_degree, 1.0),
                            np.log1p(degree) / max(np.log1p(max_degree), 1.0),
                            out_events / max(max_out_events, 1.0),
                            in_events / max(max_in_events, 1.0),
                            out_bytes / max(max_out_bytes, 1.0),
                            in_bytes / max(max_in_bytes, 1.0),
                            observed_by_step,
                            neighborhood_obs,
                        ],
                        dtype=np.float32,
                    ),
                    role_vec,
                ]
            )
            features[step, idx, :] = feature_vec

    origin_index = node_index[int(cascade.origin)]
    node_mask = np.ones(num_nodes, dtype=np.float32)
    return TemporalGraphSample(
        node_ids=node_ids,
        adjacency=adjacency,
        node_features=features,
        structural_features=structural,
        node_mask=node_mask,
        origin_index=int(origin_index),
        infection_times=infection_arr,
        observed_mask=observed_mask,
        meta={
            "beta_hat": beta_est.beta_hat,
            "beta_ci": [beta_est.ci_lower, beta_est.ci_upper],
            "beta_boundary_warning": beta_est.boundary_warning,
            "fisher_information_exact": beta_est.fisher_information,
            "fisher_information_approx": beta_est.fisher_information_approx,
            "fisher_curvature_relative_error": beta_est.curvature_relative_error,
            "prior_reliability": prior_reliability,
            "prior_history_mode": history_mode,
            "snapshot_diagnostics": snapshot_diagnostics,
            "growth_model": scenario_config.growth_model,
            "origin_node_id": int(cascade.origin),
            "dynamic_feature_names": list(feature_names),
            "feature_contract": "observable_v4_cross_dataset_aligned",
            "telemetry_model": {
                "origin_observation_dropout": scenario_config.origin_observation_dropout,
                "benign_burst_prob": scenario_config.benign_burst_prob,
                "benign_burst_scale": scenario_config.benign_burst_scale,
                "activity_noise_logstd": scenario_config.activity_noise_logstd,
                "false_alert_rate": scenario_config.false_alert_rate,
                "attack_activity_scale": scenario_config.attack_activity_scale,
            },
        },
    )


def nx_to_adj(graph, node_ids: list[int]) -> np.ndarray:
    num_nodes = len(node_ids)
    index = {node: idx for idx, node in enumerate(node_ids)}
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in graph.edges():
        i = index[int(u)]
        j = index[int(v)]
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


class SyntheticOriginDataset(Dataset):
    def __init__(
        self,
        config: SyntheticScenarioConfig,
        num_samples: int,
        seed_offset: int = 0,
        progress_label: str | None = None,
        progress_every: int = 32,
    ):
        self.samples: list[TemporalGraphSample] = []
        for idx in range(num_samples):
            local_cfg = replace(config, seed=int(config.seed + seed_offset + idx * 23))
            graph, history = generate_iot_graph(local_cfg)
            cascade = simulate_attack_cascade(graph, local_cfg, seed=local_cfg.seed + 1001)
            self.samples.append(build_temporal_sample(graph, history, cascade, config=local_cfg))
            if progress_label and (
                idx == 0 or (idx + 1) % max(progress_every, 1) == 0 or idx + 1 == num_samples
            ):
                print(
                    f"[dataset:{progress_label}] built {idx + 1}/{num_samples} samples",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TemporalGraphSample:
        return self.samples[index]


class MixedSyntheticOriginDataset(Dataset):
    """Balanced deterministic mixture over deployment-shift configurations."""

    def __init__(
        self,
        base_config: SyntheticScenarioConfig,
        num_samples: int,
        domain_grid: dict[str, list],
        seed_offset: int = 0,
        progress_label: str | None = None,
        progress_every: int = 32,
    ):
        from itertools import product

        varying_keys = sorted(domain_grid)
        combinations = list(product(*(domain_grid[key] for key in varying_keys)))
        if not combinations:
            raise ValueError("domain_grid must define at least one combination")
        self.samples: list[TemporalGraphSample] = []
        for idx in range(num_samples):
            values = combinations[idx % len(combinations)]
            overrides = dict(zip(varying_keys, values))
            overrides["seed"] = int(base_config.seed + seed_offset + idx * 23)
            local_cfg = replace(base_config, **overrides)
            graph, history = generate_iot_graph(local_cfg)
            cascade = simulate_attack_cascade(graph, local_cfg, seed=local_cfg.seed + 1001)
            sample = build_temporal_sample(graph, history, cascade, config=local_cfg)
            sample.meta["domain_cell"] = overrides | {"seed": local_cfg.seed}
            self.samples.append(sample)
            if progress_label and (
                idx == 0 or (idx + 1) % max(progress_every, 1) == 0 or idx + 1 == num_samples
            ):
                print(f"[dataset:{progress_label}] built {idx + 1}/{num_samples} samples", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TemporalGraphSample:
        return self.samples[index]


def collate_temporal_graph_samples(samples: list[TemporalGraphSample]) -> dict[str, torch.Tensor]:
    max_nodes = max(sample.adjacency.shape[0] for sample in samples)
    time_steps = samples[0].node_features.shape[0]
    feat_dim = samples[0].node_features.shape[-1]
    struct_dim = samples[0].structural_features.shape[-1]

    batch_size = len(samples)
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float32)
    node_features = torch.zeros((batch_size, time_steps, max_nodes, feat_dim), dtype=torch.float32)
    structural_features = torch.zeros((batch_size, max_nodes, struct_dim), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    origin_index = torch.zeros((batch_size,), dtype=torch.long)
    observed_mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    infection_times = torch.full((batch_size, max_nodes), -1.0, dtype=torch.float32)

    for batch_idx, sample in enumerate(samples):
        n_nodes = sample.adjacency.shape[0]
        adjacency[batch_idx, :n_nodes, :n_nodes] = torch.from_numpy(sample.adjacency)
        node_features[batch_idx, :, :n_nodes, :] = torch.from_numpy(sample.node_features)
        structural_features[batch_idx, :n_nodes, :] = torch.from_numpy(sample.structural_features)
        node_mask[batch_idx, :n_nodes] = torch.from_numpy(sample.node_mask)
        observed_mask[batch_idx, :n_nodes] = torch.from_numpy(sample.observed_mask)
        infection_times[batch_idx, :n_nodes] = torch.from_numpy(sample.infection_times)
        origin_index[batch_idx] = int(sample.origin_index)

    ssl_mask = torch.rand((batch_size, time_steps, max_nodes), dtype=torch.float32) < 0.20
    ssl_mask = ssl_mask & (node_mask.unsqueeze(1) > 0)

    return {
        "adjacency": adjacency,
        "node_features": node_features,
        "structural_features": structural_features,
        "node_mask": node_mask,
        "origin_index": origin_index,
        "observed_mask": observed_mask,
        "infection_times": infection_times,
        "ssl_mask": ssl_mask,
    }
