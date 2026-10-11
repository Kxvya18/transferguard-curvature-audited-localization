"""Heterogeneous IoT-style graph growth and attack simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

import networkx as nx
import numpy as np


ROLE_ORDER = [
    "gateway",
    "broker",
    "edge_server",
    "camera",
    "sensor",
    "actuator",
    "client",
]


DEFAULT_ROLE_PROBS: Dict[str, float] = {
    "gateway": 0.05,
    "broker": 0.08,
    "edge_server": 0.12,
    "camera": 0.16,
    "sensor": 0.27,
    "actuator": 0.12,
    "client": 0.20,
}


DEFAULT_ORIGIN_ROLE_WEIGHTS: Dict[str, float] = {
    "gateway": 1.40,
    "broker": 1.30,
    "edge_server": 1.15,
    "camera": 1.05,
    "sensor": 0.95,
    "actuator": 0.90,
    "client": 1.15,
}


DEFAULT_ATTACK_ENTRY_ROLE_WEIGHTS: Dict[str, float] = {
    "gateway": 0.80,
    "broker": 0.90,
    "edge_server": 1.00,
    "camera": 1.25,
    "sensor": 1.10,
    "actuator": 0.95,
    "client": 1.35,
}


DEFAULT_SPREAD_FACTORS: Dict[tuple[str, str], float] = {
    ("gateway", "sensor"): 1.25,
    ("gateway", "camera"): 1.15,
    ("gateway", "actuator"): 1.20,
    ("gateway", "client"): 1.10,
    ("broker", "sensor"): 1.20,
    ("broker", "camera"): 1.20,
    ("broker", "actuator"): 1.10,
    ("edge_server", "client"): 1.15,
    ("client", "gateway"): 1.05,
    ("camera", "gateway"): 1.10,
}


@dataclass
class SyntheticScenarioConfig:
    n_nodes: int = 180
    m: int = 2
    beta: float = 1.1
    n_communities: int = 4
    aging_tau: float = 120.0
    rewiring_prob: float = 0.03
    observation_dropout: float = 0.30
    max_cascade_steps: int = 8
    transmission_base: float = 0.23
    min_infected: int = 6
    max_cascade_retries: int = 6
    origin_structural_weight: float = 0.45
    observation_delay_prob: float = 0.30
    max_observation_delay: int = 2
    origin_observation_dropout: float = 0.90
    benign_burst_prob: float = 0.04
    benign_burst_scale: float = 4.0
    activity_noise_logstd: float = 0.30
    false_alert_rate: float = 0.01
    attack_activity_scale: float = 1.20
    origin_role_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ORIGIN_ROLE_WEIGHTS)
    )
    attack_entry_role_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ATTACK_ENTRY_ROLE_WEIGHTS)
    )
    role_probs: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ROLE_PROBS))
    fitness_logmean: float = 0.0
    fitness_logstd: float = 0.35
    growth_model: str = "heterogeneous_pa"
    copying_probability: float = 0.70
    copying_boost: float = 6.0
    block_affinity: float = 4.0
    prior_history_mode: str = "observed"
    snapshot_ensemble_size: int = 8
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttackCascade:
    origin: int
    infection_times: dict[int, int]
    observed_infected: list[int]
    hidden_infected: list[int]
    observed_time: dict[int, int]
    exposure_graph: nx.Graph


def _normalized_role_probs(role_probs: Dict[str, float]) -> tuple[list[str], np.ndarray]:
    roles = list(role_probs.keys())
    probs = np.asarray([float(role_probs[role]) for role in roles], dtype=float)
    probs = probs / np.sum(probs)
    return roles, probs


def _role_affinity(source_role: str, target_role: str) -> float:
    if source_role == target_role:
        return 1.15
    hubs = {"gateway", "broker", "edge_server"}
    if target_role in hubs:
        return 1.35
    if source_role in {"sensor", "camera"} and target_role == "gateway":
        return 1.30
    if source_role == "client" and target_role in {"gateway", "edge_server"}:
        return 1.25
    return 1.0


def _seed_graph(m: int, rng: np.random.Generator, n_communities: int) -> nx.Graph:
    seed_roles = ["gateway", "broker", "edge_server", "sensor", "client"]
    G = nx.Graph()
    for node, role in enumerate(seed_roles):
        G.add_node(
            node,
            role=role,
            birth_time=node,
            community=int(node % max(n_communities, 1)),
            fitness=float(rng.lognormal(mean=0.0, sigma=0.15)),
        )

    seed_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
    for u, v in seed_edges[: max(m + 1, len(seed_edges))]:
        G.add_edge(u, v, kind="seed")
    return G


def generate_iot_graph(
    config: SyntheticScenarioConfig,
) -> tuple[nx.Graph, list[dict]]:
    """Generate a heterogeneous hub-dominated communication graph.

    The history object is intentionally compatible with the sequential
    `beta`-estimation code from the prior module.
    """
    rng = np.random.default_rng(config.seed)
    G = _seed_graph(config.m, rng, config.n_communities)
    history: list[dict] = []
    roles, role_probs = _normalized_role_probs(config.role_probs)

    for new_node in range(G.number_of_nodes(), config.n_nodes):
        new_role = str(rng.choice(roles, p=role_probs))
        new_comm = int(rng.integers(0, max(config.n_communities, 1)))
        fitness = float(
            rng.lognormal(mean=config.fitness_logmean, sigma=config.fitness_logstd)
        )

        deg_before = dict(G.degree())
        existing_nodes = list(G.nodes())
        existing_deg = np.asarray(
            [float(deg_before[node]) + 1.0 for node in existing_nodes], dtype=float
        )
        birth_times = np.asarray(
            [float(G.nodes[node]["birth_time"]) for node in existing_nodes], dtype=float
        )
        ages = float(new_node) - birth_times
        aging_term = np.exp(-ages / max(config.aging_tau, 1.0))
        community_term = np.asarray(
            [
                1.20 if int(G.nodes[node]["community"]) == new_comm else 1.0
                for node in existing_nodes
            ],
            dtype=float,
        )
        role_term = np.asarray(
            [
                _role_affinity(new_role, str(G.nodes[node]["role"]))
                for node in existing_nodes
            ],
            dtype=float,
        )
        fitness_term = np.asarray(
            [float(G.nodes[node]["fitness"]) for node in existing_nodes], dtype=float
        )

        model = str(config.growth_model)
        if model == "heterogeneous_pa":
            weights = (
                np.power(existing_deg, config.beta)
                * (0.30 + aging_term)
                * role_term
                * community_term
                * np.sqrt(np.maximum(fitness_term, 1e-6))
            )
        elif model == "fitness":
            weights = (
                np.power(existing_deg, config.beta)
                * role_term
                * community_term
                * np.power(np.maximum(fitness_term, 1e-6), 2.0)
            )
        elif model == "aging":
            weights = (
                np.power(existing_deg, config.beta)
                * np.exp(-ages / max(config.aging_tau, 1.0))
                * role_term
                * community_term
            )
        elif model == "copying":
            weights = np.power(existing_deg, max(config.beta, 0.0)) * role_term
            if rng.random() < float(np.clip(config.copying_probability, 0.0, 1.0)):
                prototype = int(rng.choice(existing_nodes))
                copied = set(G.neighbors(prototype)) | {prototype}
                copy_term = np.asarray(
                    [config.copying_boost if node in copied else 1.0 for node in existing_nodes],
                    dtype=float,
                )
                weights *= copy_term
        elif model == "stochastic_block":
            block_term = np.asarray(
                [
                    config.block_affinity
                    if int(G.nodes[node]["community"]) == new_comm
                    else 1.0
                    for node in existing_nodes
                ],
                dtype=float,
            )
            weights = block_term * role_term
        else:
            raise ValueError(f"Unknown growth_model={model!r}")

        targets: list[int] = []
        remaining_nodes = existing_nodes.copy()
        remaining_weights = weights.astype(float).copy()
        for _ in range(config.m):
            probs = remaining_weights / max(float(np.sum(remaining_weights)), 1e-12)
            idx = int(rng.choice(len(remaining_nodes), p=probs))
            targets.append(int(remaining_nodes.pop(idx)))
            remaining_weights = np.delete(remaining_weights, idx)

        G.add_node(
            new_node,
            role=new_role,
            birth_time=new_node,
            community=new_comm,
            fitness=fitness,
        )
        for target in targets:
            G.add_edge(new_node, target, kind="growth")

        if rng.random() < config.rewiring_prob and G.number_of_nodes() > 4:
            pool = [node for node in G.nodes() if node != new_node and node not in targets]
            if pool:
                extra = int(rng.choice(pool))
                G.add_edge(new_node, extra, kind="rewire")

        history.append(
            {
                "step": int(new_node - config.m - 1),
                "time": int(new_node),
                "new_node": int(new_node),
                "new_role": new_role,
                "community": int(new_comm),
                "targets": [int(target) for target in targets],
                "deg_before": deg_before,
                "m": int(config.m),
                "beta_true": float(config.beta),
                "growth_model": model,
            }
        )

    return G, history


def _pick_origin(
    G: nx.Graph,
    config: SyntheticScenarioConfig,
    rng: np.random.Generator,
) -> int:
    nodes = list(G.nodes())
    weights = []
    for node in nodes:
        role = str(G.nodes[node].get("role", "sensor"))
        degree = float(G.degree(node)) + 1.0
        fitness = float(G.nodes[node].get("fitness", 1.0))
        structural = np.power(degree, 0.35)
        stealth = np.power(degree, -0.20)
        degree_term = (
            config.origin_structural_weight * structural
            + (1.0 - config.origin_structural_weight) * stealth
        )
        role_weight = float(config.attack_entry_role_weights.get(role, 1.0))
        fitness_term = float(np.power(max(fitness, 1e-6), 0.15))
        weights.append(degree_term * role_weight * fitness_term)
    probs = np.asarray(weights, dtype=float)
    probs = probs / np.sum(probs)
    return int(rng.choice(nodes, p=probs))


def _spread_factor(source_role: str, target_role: str) -> float:
    if (source_role, target_role) in DEFAULT_SPREAD_FACTORS:
        return DEFAULT_SPREAD_FACTORS[(source_role, target_role)]
    if source_role == target_role:
        return 1.05
    return 1.0


def simulate_attack_cascade(
    G: nx.Graph,
    config: SyntheticScenarioConfig,
    origin: int | None = None,
    seed: int | None = None,
) -> AttackCascade:
    rng = np.random.default_rng(config.seed + 101 if seed is None else seed)
    best_payload: tuple[int, dict[int, int]] | None = None

    for attempt in range(max(config.max_cascade_retries, 1)):
        chosen_origin = (
            _pick_origin(G, config, rng) if origin is None else int(origin)
        )
        infection_times: dict[int, int] = {chosen_origin: 0}
        frontier = [chosen_origin]
        transmission_boost = 1.0 + 0.12 * attempt

        for step in range(1, config.max_cascade_steps + 1):
            next_frontier: list[int] = []
            for source in frontier:
                source_role = str(G.nodes[source].get("role", "sensor"))
                for target in G.neighbors(source):
                    if target in infection_times:
                        continue
                    target_role = str(G.nodes[target].get("role", "sensor"))
                    same_comm = int(G.nodes[source]["community"]) == int(G.nodes[target]["community"])
                    degree_factor = 1.0 + 0.15 * np.log(float(G.degree(source)) + 1.0)
                    p_edge = (
                        config.transmission_base
                        * transmission_boost
                        * _spread_factor(source_role, target_role)
                        * degree_factor
                        * (1.10 if same_comm else 0.90)
                    )
                    p_edge = float(np.clip(p_edge, 0.01, 0.95))
                    if rng.random() < p_edge:
                        infection_times[int(target)] = step
                        next_frontier.append(int(target))
            if not next_frontier:
                break
            frontier = next_frontier

        if best_payload is None or len(infection_times) > len(best_payload[1]):
            best_payload = (int(chosen_origin), infection_times)
        if len(infection_times) >= config.min_infected:
            break

    assert best_payload is not None
    chosen_origin, infection_times = best_payload

    observed_time: dict[int, int] = {}
    observed_infected: list[int] = []
    hidden_infected: list[int] = []
    ordered_infections = sorted(infection_times.items(), key=lambda item: item[1])

    for node, time in ordered_infections:
        role = str(G.nodes[node].get("role", "sensor"))
        miss_rate = config.observation_dropout
        if role in {"gateway", "broker", "edge_server"}:
            miss_rate *= 0.7
        if node == chosen_origin:
            miss_rate = max(miss_rate, config.origin_observation_dropout)
        visible = rng.random() > miss_rate
        if visible:
            delay = 0
            if config.max_observation_delay > 0 and rng.random() < config.observation_delay_prob:
                delay = int(rng.integers(0, config.max_observation_delay + 1))
            observed_infected.append(int(node))
            observed_time[int(node)] = int(time + delay)
        else:
            hidden_infected.append(int(node))

    if not observed_infected and ordered_infections:
        fallback_k = min(3, len(ordered_infections))
        fallback_node, fallback_time = ordered_infections[int(rng.integers(0, fallback_k))]
        observed_infected.append(int(fallback_node))
        observed_time[int(fallback_node)] = int(fallback_time)
        hidden_infected = [node for node in hidden_infected if node != int(fallback_node)]

    exposure_nodes = set(observed_infected)
    for node in list(observed_infected):
        exposure_nodes.update(int(nbr) for nbr in G.neighbors(node))
    exposure_graph = G.subgraph(sorted(exposure_nodes)).copy()

    return AttackCascade(
        origin=int(chosen_origin),
        infection_times={int(k): int(v) for k, v in infection_times.items()},
        observed_infected=sorted(observed_infected),
        hidden_infected=sorted(hidden_infected),
        observed_time=observed_time,
        exposure_graph=exposure_graph,
    )
