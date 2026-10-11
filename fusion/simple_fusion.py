"""Early non-neural fusion baselines.

This gives us a minimal bridge between the structural prior and observed attack
evidence before the self-supervised temporal encoder is implemented.
"""

from __future__ import annotations

from collections import deque
from typing import Mapping

import networkx as nx
import numpy as np


def _distance_evidence(G: nx.Graph, observed_infected: list[int]) -> dict[int, float]:
    if not observed_infected:
        return {int(node): 0.0 for node in G.nodes()}

    dist = {int(node): float("inf") for node in G.nodes()}
    queue = deque()
    for node in observed_infected:
        if node in G:
            dist[int(node)] = 0.0
            queue.append(int(node))

    while queue:
        node = queue.popleft()
        for nbr in G.neighbors(node):
            if dist[int(nbr)] > dist[node] + 1.0:
                dist[int(nbr)] = dist[node] + 1.0
                queue.append(int(nbr))

    scores = {}
    for node, value in dist.items():
        if np.isinf(value):
            scores[int(node)] = -6.0
        else:
            scores[int(node)] = float(-value)
    return scores


def hybrid_origin_scores(
    G: nx.Graph,
    prior_probs: Mapping[int, float],
    observed_infected: list[int],
    alpha: float = 0.65,
) -> dict[int, float]:
    """Blend log prior with simple proximity evidence."""
    evidence = _distance_evidence(G, observed_infected)
    scores: dict[int, float] = {}
    for node in G.nodes():
        prior = float(max(prior_probs.get(int(node), 1e-12), 1e-12))
        log_prior = float(np.log(prior))
        evidence_term = float(evidence.get(int(node), -6.0))
        scores[int(node)] = float(alpha * log_prior + (1.0 - alpha) * evidence_term)
    return scores

