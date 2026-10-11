"""Metrics for attack-origin localization and early-compromise ranking."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _rank_of_item(scores: Mapping[int, float], item: int) -> int:
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    for idx, (node, _) in enumerate(ordered, start=1):
        if int(node) == int(item):
            return idx
    return len(ordered) + 1


def evaluate_origin_localization(
    scores: Mapping[int, float],
    true_origin: int,
    topk: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    rank = _rank_of_item(scores, true_origin)
    metrics = {
        "origin_rank": float(rank),
        "mrr": float(1.0 / rank),
    }
    for k_val in topk:
        metrics[f"hit@{k_val}"] = float(rank <= k_val)
    return metrics


def evaluate_early_compromise_ranking(
    scores: Mapping[int, float],
    infection_times: Mapping[int, int],
    topk: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    if not infection_times:
        return {f"early_precision@{k_val}": 0.0 for k_val in topk} | {
            "pairwise_order_acc": 0.0
        }

    ordered_pred = [node for node, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
    true_early = [node for node, _ in sorted(infection_times.items(), key=lambda pair: pair[1])]

    results: dict[str, float] = {}
    for k_val in topk:
        pred_k = set(ordered_pred[: min(k_val, len(ordered_pred))])
        true_k = set(true_early[: min(k_val, len(true_early))])
        results[f"early_precision@{k_val}"] = float(len(pred_k & true_k) / max(len(pred_k), 1))

    comparable = 0
    concordant = 0
    scored_nodes = [node for node in ordered_pred if node in infection_times]
    pred_pos = {node: idx for idx, node in enumerate(scored_nodes)}
    inf_items = list(infection_times.items())
    for i in range(len(inf_items)):
        node_i, time_i = inf_items[i]
        for j in range(i + 1, len(inf_items)):
            node_j, time_j = inf_items[j]
            if time_i == time_j or node_i not in pred_pos or node_j not in pred_pos:
                continue
            comparable += 1
            pred_before = pred_pos[node_i] < pred_pos[node_j]
            true_before = time_i < time_j
            concordant += int(pred_before == true_before)

    results["pairwise_order_acc"] = float(concordant / comparable) if comparable else 0.0
    return results

