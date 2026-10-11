"""Counterfactual quarantine on a fixed, time-respecting cascade realization."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx


def time_respecting_quarantine(
    graph: nx.Graph,
    infection_times: dict[int, int],
    quarantined: Iterable[int],
    decision_time: int,
) -> dict[str, float | int]:
    """Remove selected nodes and descendants unsupported by another live parent.

    Nodes compromised on or before the decision remain compromised unless they
    are quarantined. A later compromise survives only when at least one neighbor
    at the preceding infection time survives. This keeps every method on the
    same realized cascade and avoids method-dependent random-number streams.
    """
    if not infection_times:
        return {
            "baseline_future": 0,
            "surviving_future": 0,
            "prevented_future": 0,
            "prevented_fraction": 0.0,
            "quarantined_compromised": 0,
        }
    blocked = {int(node) for node in quarantined}
    survivors = {
        int(node)
        for node, time in infection_times.items()
        if int(time) <= int(decision_time) and int(node) not in blocked
    }
    max_time = max(int(value) for value in infection_times.values())
    for step in range(int(decision_time) + 1, max_time + 1):
        candidates = [
            int(node)
            for node, time in infection_times.items()
            if int(time) == step and int(node) not in blocked
        ]
        for node in candidates:
            if any(
                int(neighbor) in survivors
                and int(infection_times.get(int(neighbor), -10)) == step - 1
                for neighbor in graph.neighbors(node)
            ):
                survivors.add(node)
    future = {
        int(node) for node, time in infection_times.items() if int(time) > int(decision_time)
    }
    surviving_future = future & survivors
    prevented = len(future) - len(surviving_future)
    return {
        "baseline_future": len(future),
        "surviving_future": len(surviving_future),
        "prevented_future": prevented,
        "prevented_fraction": float(prevented / max(len(future), 1)),
        "quarantined_compromised": len(blocked & set(infection_times)),
    }
