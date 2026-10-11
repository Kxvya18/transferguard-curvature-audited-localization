"""Adaptive post-quarantine spread with common edge-time random numbers."""

from __future__ import annotations

import math

import networkx as nx

from synthetic.iot_growth import SyntheticScenarioConfig, _spread_factor


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _edge_uniform(seed: int, step: int, source: int, target: int) -> float:
    key = (
        int(seed)
        ^ (int(step) * 0x9E3779B1)
        ^ (int(source) * 0x85EBCA77)
        ^ (int(target) * 0xC2B2AE3D)
    )
    return (_splitmix64(key) >> 11) * (1.0 / (1 << 53))


def _simulate_world(
    graph: nx.Graph,
    config: SyntheticScenarioConfig,
    origin: int,
    quarantined: set[int],
    decision_time: int,
    seed: int,
    adaptive_boost: float,
) -> dict[int, int]:
    infection_times: dict[int, int] = {int(origin): 0}
    active: set[int] = {int(origin)}
    blocked: set[int] = set()
    for step in range(1, config.max_cascade_steps + 1):
        if step > int(decision_time):
            blocked = set(quarantined)
            active.difference_update(blocked)
        proposals: dict[int, int] = {}
        for source in sorted(active):
            source_role = str(graph.nodes[source].get("role", "sensor"))
            neighbors = list(graph.neighbors(source))
            blocked_fraction = sum(int(node in blocked) for node in neighbors) / max(len(neighbors), 1)
            reroute_factor = 1.0 + float(adaptive_boost) * blocked_fraction
            for target in sorted(neighbors):
                if target in infection_times or target in blocked:
                    continue
                target_role = str(graph.nodes[target].get("role", "sensor"))
                same_comm = int(graph.nodes[source]["community"]) == int(
                    graph.nodes[target]["community"]
                )
                degree_factor = 1.0 + 0.15 * math.log(float(graph.degree(source)) + 1.0)
                probability = (
                    config.transmission_base
                    * _spread_factor(source_role, target_role)
                    * degree_factor
                    * (1.10 if same_comm else 0.90)
                    * reroute_factor
                )
                probability = min(max(probability, 0.01), 0.95)
                if _edge_uniform(seed, step, int(source), int(target)) < probability:
                    proposals[int(target)] = min(int(source), proposals.get(int(target), int(source)))
        if not proposals:
            continue
        for target in sorted(proposals):
            infection_times[target] = step
            active.add(target)
    return infection_times


def evaluate_adaptive_quarantine(
    graph: nx.Graph,
    config: SyntheticScenarioConfig,
    origin: int,
    quarantined: list[int],
    decision_time: int,
    seed: int,
    adaptive_boost: float = 1.0,
) -> dict[str, float | int]:
    """Compare no intervention and quarantine under one deterministic world."""
    baseline = _simulate_world(
        graph, config, origin, set(), decision_time, seed, adaptive_boost
    )
    intervention = _simulate_world(
        graph,
        config,
        origin,
        {int(node) for node in quarantined},
        decision_time,
        seed,
        adaptive_boost,
    )
    baseline_future = {
        node for node, time in baseline.items() if int(time) > int(decision_time)
    }
    intervention_future = {
        node for node, time in intervention.items() if int(time) > int(decision_time)
    }
    prevented = len(baseline_future - intervention_future)
    added_by_rerouting = len(intervention_future - baseline_future)
    return {
        "baseline_total": len(baseline),
        "intervention_total": len(intervention),
        "baseline_future": len(baseline_future),
        "intervention_future": len(intervention_future),
        "prevented_future": prevented,
        "rerouted_new_future": added_by_rerouting,
        "net_prevented_fraction": float(
            (len(baseline_future) - len(intervention_future)) / max(len(baseline_future), 1)
        ),
        "gross_prevented_fraction": float(prevented / max(len(baseline_future), 1)),
    }
