"""Boltzmann-style structural prior adapted from the legacy BA work.

The old `ba` project treated the preferential-attachment Boltzmann view as the
entire scientific story. In `ba2`, we keep the strongest parts as a reusable
structural-prior layer for attack-origin localization and early-compromise
ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Sequence

import networkx as nx
import numpy as np


DEFAULT_ORIGIN_ROLE_WEIGHTS: Dict[str, float] = {
    "gateway": 1.30,
    "broker": 1.25,
    "edge_server": 1.20,
    "camera": 1.00,
    "sensor": 0.95,
    "actuator": 0.95,
    "client": 1.05,
}

NodeId = Hashable


@dataclass(frozen=True)
class BetaEstimate:
    """Estimated concentration parameter for the structural prior."""

    beta_hat: float
    fisher_information: float
    se: float
    ci_lower: float
    ci_upper: float
    boundary_warning: bool
    fisher_information_approx: float = float("nan")
    curvature_relative_error: float = float("nan")


def _positive_nodes(deg: Mapping[NodeId, float]) -> list[NodeId]:
    return [node for node, degree in deg.items() if degree > 0]


def _degree_array(deg: Mapping[NodeId, float], nodes: Sequence[NodeId]) -> np.ndarray:
    return np.asarray([float(deg[node]) for node in nodes], dtype=float)


def _beta_weights(degrees: np.ndarray, beta: float) -> np.ndarray:
    if degrees.size == 0:
        return np.zeros(0, dtype=float)
    if beta == 0.0:
        return np.ones_like(degrees, dtype=float)
    log_w = beta * np.log(np.maximum(degrees, 1.0))
    log_w -= float(np.max(log_w))
    return np.exp(log_w)


def attachment_probabilities(
    deg: Mapping[NodeId, float], beta: float
) -> tuple[list[NodeId], np.ndarray]:
    nodes = _positive_nodes(deg)
    if not nodes:
        return [], np.zeros(0, dtype=float)
    weights = _beta_weights(_degree_array(deg, nodes), beta)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return nodes, np.full(len(nodes), 1.0 / len(nodes), dtype=float)
    return nodes, weights / total


def hamiltonian(deg: Mapping[NodeId, float]) -> float:
    """Legacy log-degree Hamiltonian retained as a descriptive signal."""
    degrees = np.asarray([float(d) for d in deg.values() if d > 0], dtype=float)
    return float(-np.sum(np.log(degrees))) if degrees.size else 0.0


def _seq_ll_step_vectorized(
    deg: Mapping[NodeId, float], targets: Sequence[NodeId], beta: float
) -> float:
    nodes = _positive_nodes(deg)
    if not nodes:
        return 0.0

    degrees = _degree_array(deg, nodes)
    pos = {node: idx for idx, node in enumerate(nodes)}
    active = np.ones(len(nodes), dtype=bool)
    weights_base = _beta_weights(degrees, beta)

    ll = 0.0
    for target in targets:
        idx = pos.get(target)
        if idx is None or not active[idx]:
            return float("-inf")
        weights = np.where(active, weights_base, 0.0)
        z_val = float(np.sum(weights))
        if z_val <= 1e-15 or weights[idx] <= 0.0:
            return float("-inf")
        ll += float(np.log(weights[idx] / z_val))
        active[idx] = False
    return ll


def history_loglikelihood_seq(history: Sequence[dict], beta: float) -> float:
    return float(
        sum(_seq_ll_step_vectorized(step["deg_before"], step["targets"], beta) for step in history)
    )


def fisher_information_approx(history: Sequence[dict], beta: float) -> float:
    """Legacy first-draw curvature multiplied by the number of targets."""
    info = 0.0
    for step in history:
        deg = step["deg_before"]
        nodes, probs = attachment_probabilities(deg, beta)
        if not nodes:
            continue
        log_degrees = np.log(np.maximum(_degree_array(deg, nodes), 1.0))
        mean_log = float(np.dot(probs, log_degrees))
        mean_sq = float(np.dot(probs, log_degrees ** 2))
        info += len(step["targets"]) * max(mean_sq - mean_log**2, 0.0)
    return float(info)


def fisher_information_exact(history: Sequence[dict], beta: float) -> float:
    """Exact negative Hessian of the sequential WOR log likelihood.

    For every target draw, the normalizer is recomputed after removing earlier
    targets. Because the attachment family is log-linear in ``log(k)``, each
    draw contributes the active-set variance of log degree.
    """
    info = 0.0
    for step in history:
        deg = step["deg_before"]
        nodes = _positive_nodes(deg)
        if not nodes:
            continue
        degrees = _degree_array(deg, nodes)
        log_degrees = np.log(np.maximum(degrees, 1.0))
        weights_base = _beta_weights(degrees, beta)
        pos = {node: idx for idx, node in enumerate(nodes)}
        active = np.ones(len(nodes), dtype=bool)
        for target in step["targets"]:
            idx = pos.get(target)
            if idx is None or not active[idx]:
                continue
            weights = np.where(active, weights_base, 0.0)
            total = float(np.sum(weights))
            if total <= 1e-15:
                continue
            probs = weights / total
            mean_log = float(np.dot(probs, log_degrees))
            mean_sq = float(np.dot(probs, log_degrees**2))
            info += max(mean_sq - mean_log**2, 0.0)
            active[idx] = False
    return float(info)


def fisher_information(history: Sequence[dict], beta: float) -> float:
    """Exact sequential without-replacement Fisher information."""
    return fisher_information_exact(history, beta)


def snapshot_pseudohistory(
    graph: nx.Graph,
    min_seed_nodes: int = 5,
    tie_break_seed: int = 0,
) -> list[dict]:
    """Infer a growth-compatible pseudo-history from one final snapshot.

    Nodes are reverse-peeled using only current structural quantities. Random
    tie breaking is pre-seeded and is intended for an ensemble sensitivity
    analysis; node identifiers are never used as an age proxy.
    """
    work = graph.copy()
    rng = np.random.default_rng(int(tie_break_seed))
    removed: list[dict] = []
    while work.number_of_nodes() > max(int(min_seed_nodes), 2):
        candidates = list(work.nodes())
        degree = dict(work.degree())
        scores = []
        for node in candidates:
            neighbors = list(work.neighbors(node))
            neighbor_degree = float(np.mean([degree[nbr] for nbr in neighbors])) if neighbors else 0.0
            clustering = float(nx.clustering(work, node)) if degree[node] > 1 else 0.0
            scores.append((float(degree[node]), neighbor_degree, clustering))
        score_array = np.asarray(scores, dtype=float)
        order = np.lexsort((rng.random(len(candidates)), score_array[:, 2], score_array[:, 1], score_array[:, 0]))
        node = candidates[int(order[0])]
        targets = list(work.neighbors(node))
        work.remove_node(node)
        removed.append(
            {
                "new_node": node,
                "targets": targets,
                "deg_before": {
                    existing: max(int(existing_degree), 1)
                    for existing, existing_degree in work.degree()
                },
                "m": len(targets),
                "source": "snapshot_reverse_peeling",
            }
        )
    removed.reverse()
    return removed


def snapshot_beta_ensemble(
    graph: nx.Graph,
    ensemble_size: int = 8,
    seed_base: int = 31_000,
) -> tuple[BetaEstimate, dict[str, float]]:
    """Fit beta to an ensemble of snapshot-derived pseudo-histories."""
    estimates = [
        mle_beta(snapshot_pseudohistory(graph, tie_break_seed=seed_base + idx))
        for idx in range(max(int(ensemble_size), 1))
    ]
    betas = np.asarray([estimate.beta_hat for estimate in estimates], dtype=float)
    infos = np.asarray([estimate.fisher_information for estimate in estimates], dtype=float)
    beta_hat = float(np.median(betas))
    info = float(np.median(infos))
    between_variance = float(np.var(betas, ddof=1)) if betas.size > 1 else 0.0
    total_variance = 1.0 / max(info, 1e-12) + between_variance
    se = float(np.sqrt(total_variance))
    boundary = bool(np.mean([estimate.boundary_warning for estimate in estimates]) >= 0.5)
    aggregate = BetaEstimate(
        beta_hat=beta_hat,
        fisher_information=info,
        se=se,
        ci_lower=float(beta_hat - 1.96 * se),
        ci_upper=float(beta_hat + 1.96 * se),
        boundary_warning=boundary,
        fisher_information_approx=float(np.median([estimate.fisher_information_approx for estimate in estimates])),
        curvature_relative_error=float(np.median([estimate.curvature_relative_error for estimate in estimates])),
    )
    diagnostics = {
        "ensemble_size": float(len(estimates)),
        "beta_between_sd": float(np.sqrt(between_variance)),
        "beta_min": float(np.min(betas)),
        "beta_max": float(np.max(betas)),
    }
    return aggregate, diagnostics


def mle_beta(
    history: Sequence[dict],
    beta_min: float = 0.0,
    beta_max: float = 4.0,
    coarse_points: int = 41,
    refine_rounds: int = 3,
) -> BetaEstimate:
    """Pure-numpy MLE to keep the new project lightweight.

    We do a coarse grid search followed by local refinements around the current
    maximizer. This is slower than a dedicated optimizer but removes a hard
    dependency on SciPy for the first implementation stage.
    """
    lo = max(beta_min, 1e-4)
    hi = beta_max
    best_beta = 1.0
    best_ll = float("-inf")

    for _ in range(refine_rounds):
        beta_grid = np.linspace(lo, hi, coarse_points, dtype=float)
        ll_vals = np.asarray(
            [history_loglikelihood_seq(history, float(beta)) for beta in beta_grid],
            dtype=float,
        )
        idx = int(np.argmax(ll_vals))
        best_beta = float(beta_grid[idx])
        best_ll = float(ll_vals[idx])

        if idx == 0:
            left = beta_grid[0]
            right = beta_grid[1]
        elif idx == len(beta_grid) - 1:
            left = beta_grid[-2]
            right = beta_grid[-1]
        else:
            left = beta_grid[idx - 1]
            right = beta_grid[idx + 1]

        pad = max((right - left) * 0.5, 1e-4)
        lo = max(beta_min, float(left - pad))
        hi = min(beta_max, float(right + pad))

    fisher = fisher_information_exact(history, best_beta)
    fisher_approx = fisher_information_approx(history, best_beta)
    relative_error = float(abs(fisher_approx - fisher) / max(fisher, 1e-12))
    se = float(1.0 / np.sqrt(max(fisher, 1e-12)))
    ci_radius = 1.96 * se
    boundary_margin = max(0.05, 0.02 * (beta_max - beta_min))
    boundary = (
        best_beta <= beta_min + boundary_margin
        or best_beta >= beta_max - boundary_margin
        or fisher < 0.01
    )

    return BetaEstimate(
        beta_hat=best_beta,
        fisher_information=fisher,
        se=se,
        ci_lower=float(best_beta - ci_radius),
        ci_upper=float(best_beta + ci_radius),
        boundary_warning=boundary,
        fisher_information_approx=fisher_approx,
        curvature_relative_error=relative_error,
    )


def _reverse_neighbor_score(G: nx.Graph, node: NodeId, beta: float) -> float:
    terms = []
    for nbr in G.neighbors(node):
        degree = float(G.degree(nbr))
        # Jeffreys-style half-count smoothing avoids infinite evidence from
        # degree-one leaves in star-like scanning graphs.
        terms.append(np.log((degree - 1.0 + 0.5) / (degree + 0.5)))
    return float(beta * np.sum(terms)) if terms else 0.0


def _mean_field_age_score(G: nx.Graph, node: NodeId, beta: float, m: int) -> float:
    degree = max(float(G.degree(node)), 1.0)
    return float((beta + 1.0) * np.log(max(float(m), 1.0) / degree))


def _late_arrival_score(
    G: nx.Graph,
    node: NodeId,
    beta: float,
    m: int,
) -> float:
    return _reverse_neighbor_score(G, node, beta) + _mean_field_age_score(G, node, beta, m)


def origin_prior_scores(
    G: nx.Graph,
    beta: float,
    m: int = 2,
    role_weights: Mapping[str, float] | None = None,
) -> dict[NodeId, float]:
    """Prior scores for likely early / origin nodes.

    The legacy reconstruction score favors late removable nodes. For origin
    localization we flip that logic: nodes that are harder to remove under the
    reverse model are more plausible early structural anchors.
    """
    weights = DEFAULT_ORIGIN_ROLE_WEIGHTS if role_weights is None else dict(role_weights)
    scores: dict[NodeId, float] = {}
    for node in G.nodes():
        late_score = _late_arrival_score(G, node, beta, m)
        role = str(G.nodes[node].get("role", "sensor"))
        role_bonus = float(np.log(max(weights.get(role, 1.0), 1e-6)))
        scores[node] = float(-late_score + role_bonus)
    return scores


def _softmax_from_scores(scores: Mapping[NodeId, float]) -> dict[NodeId, float]:
    if not scores:
        return {}
    nodes = list(scores.keys())
    values = np.asarray([scores[node] for node in nodes], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(values))
    if scale > 1e-8:
        values = np.clip((values - median) / scale, -8.0, 8.0)
    values = values - float(np.max(values))
    weights = np.exp(values)
    probs = weights / max(float(np.sum(weights)), 1e-12)
    return {node: float(prob) for node, prob in zip(nodes, probs)}


def origin_prior_distribution(
    G: nx.Graph,
    beta: float,
    m: int = 2,
    role_weights: Mapping[str, float] | None = None,
    reliability: float = 1.0,
) -> dict[NodeId, float]:
    concentrated = _softmax_from_scores(
        origin_prior_scores(G, beta=beta, m=m, role_weights=role_weights)
    )
    if not concentrated:
        return {}
    weight = float(np.clip(reliability, 0.0, 1.0))
    uniform = 1.0 / len(concentrated)
    return {
        node: float(weight * probability + (1.0 - weight) * uniform)
        for node, probability in concentrated.items()
    }


def beta_estimate_reliability(
    estimate: BetaEstimate,
    information_scale: float = 5.0,
) -> float:
    """Reliability weight for shrinking uncertain structural priors.

    Information below ``information_scale`` is treated as weak. Boundary MLEs
    retain only one quarter of that evidence because the finite estimate is not
    identified inside the fitted interval.
    """
    information_weight = estimate.fisher_information / (
        estimate.fisher_information + max(information_scale, 1e-12)
    )
    if estimate.boundary_warning:
        information_weight *= 0.25
    return float(np.clip(information_weight, 0.0, 1.0))


def ambiguity_score(prior_probs: Mapping[NodeId, float]) -> float:
    """Normalized entropy of the origin prior.

    High values indicate the graph does not provide a sharp structural guess and
    later modules should trust the prior less aggressively.
    """
    if not prior_probs:
        return 1.0
    vals = np.asarray(list(prior_probs.values()), dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    entropy = float(-np.sum(vals * np.log(vals)))
    max_entropy = float(np.log(max(len(prior_probs), 2)))
    return float(entropy / max(max_entropy, 1e-12))


def graph_concentration_features(
    G: nx.Graph,
    beta: float,
    m: int = 2,
    role_weights: Mapping[str, float] | None = None,
    reliability: float = 1.0,
) -> dict[str, float]:
    deg = dict(G.degree())
    prior_probs = origin_prior_distribution(
        G,
        beta=beta,
        m=m,
        role_weights=role_weights,
        reliability=reliability,
    )
    degrees = np.asarray(list(deg.values()), dtype=float)
    mean_degree = float(np.mean(degrees)) if degrees.size else 0.0
    max_degree = float(np.max(degrees)) if degrees.size else 0.0

    return {
        "beta": float(beta),
        "hamiltonian": hamiltonian(deg),
        "n_nodes": float(G.number_of_nodes()),
        "n_edges": float(G.number_of_edges()),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "degree_concentration": float(max_degree / max(mean_degree, 1e-12)),
        "prior_ambiguity": ambiguity_score(prior_probs),
        "prior_top1": float(max(prior_probs.values()) if prior_probs else 0.0),
    }
