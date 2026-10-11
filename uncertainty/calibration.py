"""Calibration, selective prediction, and epistemic uncertainty utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(float(temperature), 1e-6)


def entropy_confidence(probs: torch.Tensor) -> torch.Tensor:
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(probs.size(-1)), device=probs.device)).clamp_min(1e-12)
    return 1.0 - entropy / max_entropy


@torch.no_grad()
def mc_dropout_origin_distribution(
    model,
    batch: dict[str, torch.Tensor],
    n_samples: int = 16,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.train()
    try:
        probs = []
        for _ in range(n_samples):
            out = model(
                node_features=batch["node_features"],
                adjacency=batch["adjacency"],
                structural_features=batch["structural_features"],
                node_mask=batch["node_mask"],
                ssl_mask=None,
            )
            probs.append(out["probs"])
        stacked = torch.stack(probs, dim=0)
        mean_probs = stacked.mean(dim=0)
        predictive_entropy = -(
            mean_probs.clamp_min(1e-12) * mean_probs.clamp_min(1e-12).log()
        ).sum(dim=-1)
        expected_entropy = -(
            stacked.clamp_min(1e-12) * stacked.clamp_min(1e-12).log()
        ).sum(dim=-1).mean(dim=0)
        epistemic = stacked.var(dim=0).mean(dim=-1)
        return {
            "mean_probs": mean_probs,
            "epistemic_var": epistemic,
            "mutual_information": (predictive_entropy - expected_entropy).clamp_min(0.0),
            "predictive_entropy": predictive_entropy,
            "confidence": entropy_confidence(mean_probs),
        }
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    iterations: int


def _sample_nll(logits: np.ndarray, label: int, temperature: float = 1.0) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    scaled = scaled - np.max(scaled)
    log_normalizer = np.log(np.exp(scaled).sum())
    return float(log_normalizer - scaled[int(label)])


def fit_temperature(
    logits: list[np.ndarray],
    labels: list[int],
    max_iter: int = 300,
    learning_rate: float = 0.04,
) -> TemperatureFit:
    """Fit one positive temperature on an independent calibration split.

    Samples may have different node counts, so the objective is accumulated
    sample-wise instead of padding class spaces that have no shared meaning.
    """
    if not logits or len(logits) != len(labels):
        raise ValueError("logits and labels must be non-empty and aligned")
    tensors = [torch.as_tensor(value, dtype=torch.float64) for value in logits]
    targets = [int(value) for value in labels]
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)
    best_loss = float("inf")
    best_temperature = 1.0
    stale = 0
    for iteration in range(max_iter):
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        losses = []
        for sample_logits, target in zip(tensors, targets):
            losses.append(
                torch.logsumexp(sample_logits / temperature, dim=0)
                - sample_logits[target] / temperature
            )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        if current + 1e-10 < best_loss:
            best_loss = current
            best_temperature = float(temperature.detach())
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    before = float(np.mean([_sample_nll(x, y) for x, y in zip(logits, labels)]))
    return TemperatureFit(
        temperature=best_temperature,
        nll_before=before,
        nll_after=best_loss,
        iterations=iteration + 1,
    )


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    return weights / max(float(weights.sum()), 1e-12)


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = False,
) -> float:
    confidence = np.asarray(confidences, dtype=float)
    correct = np.asarray(correctness, dtype=float)
    if adaptive:
        order = np.argsort(confidence)
        groups = np.array_split(order, min(n_bins, len(order)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        groups = [
            np.flatnonzero((confidence > edges[idx]) & (confidence <= edges[idx + 1]))
            for idx in range(n_bins)
        ]
        groups[0] = np.flatnonzero((confidence >= 0.0) & (confidence <= edges[1]))
    total = max(len(confidence), 1)
    return float(
        sum(
            len(group) / total
            * abs(float(np.mean(correct[group])) - float(np.mean(confidence[group])))
            for group in groups
            if len(group)
        )
    )


def risk_coverage_summary(
    confidences: np.ndarray,
    correctness: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> dict:
    confidence = np.asarray(confidences, dtype=float)
    correct = np.asarray(correctness, dtype=float)
    order = np.argsort(-confidence)
    cumulative_risk = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    selective = []
    for requested in coverages:
        count = max(1, int(np.ceil(requested * len(order))))
        selective.append(
            {
                "coverage": float(count / len(order)),
                "risk": float(cumulative_risk[count - 1]),
                "accuracy": float(1.0 - cumulative_risk[count - 1]),
            }
        )
    coverage_at_risk = {}
    for risk_target in (0.01, 0.05, 0.10):
        valid = np.flatnonzero(cumulative_risk <= risk_target)
        coverage_at_risk[f"risk<={risk_target:.2f}"] = (
            float((valid[-1] + 1) / len(order)) if len(valid) else 0.0
        )
    return {
        "aurc": float(np.mean(cumulative_risk)),
        "selective": selective,
        "coverage_at_risk": coverage_at_risk,
    }


def conformal_quantile(
    calibration_probabilities: list[np.ndarray],
    labels: list[int],
    alpha: float = 0.10,
) -> float:
    scores = np.asarray(
        [1.0 - float(probs[label]) for probs, label in zip(calibration_probabilities, labels)]
    )
    level = min(np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def evaluate_prediction_sets(
    probabilities: list[np.ndarray],
    labels: list[int],
    quantile: float,
) -> dict[str, float]:
    threshold = 1.0 - float(quantile)
    sizes = []
    covered = []
    for probs, label in zip(probabilities, labels):
        members = np.flatnonzero(np.asarray(probs) >= threshold)
        if not len(members):
            members = np.asarray([int(np.argmax(probs))])
        sizes.append(len(members))
        covered.append(int(label) in set(int(value) for value in members))
    return {
        "empirical_coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(sizes)),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float(np.mean(np.asarray(sizes) == 1)),
    }


def classification_calibration_metrics(
    probabilities: list[np.ndarray],
    labels: list[int],
) -> dict:
    confidences = np.asarray([float(np.max(probs)) for probs in probabilities])
    predictions = np.asarray([int(np.argmax(probs)) for probs in probabilities])
    correctness = predictions == np.asarray(labels, dtype=int)
    nll = np.asarray(
        [-np.log(max(float(probs[label]), 1e-12)) for probs, label in zip(probabilities, labels)]
    )
    brier = []
    reciprocal_ranks = []
    normalized_entropy = []
    for probs, label in zip(probabilities, labels):
        target = np.zeros(len(probs), dtype=float)
        target[int(label)] = 1.0
        brier.append(float(np.square(np.asarray(probs) - target).sum()))
        rank = int(np.flatnonzero(np.argsort(-np.asarray(probs)) == int(label))[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        entropy = -float(np.sum(np.asarray(probs) * np.log(np.asarray(probs).clip(1e-12))))
        normalized_entropy.append(entropy / max(np.log(len(probs)), 1e-12))
    return {
        "n_samples": len(labels),
        "accuracy": float(np.mean(correctness)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
        "ece_15": expected_calibration_error(confidences, correctness, 15, adaptive=False),
        "adaptive_ece_15": expected_calibration_error(confidences, correctness, 15, adaptive=True),
        "risk_coverage": risk_coverage_summary(confidences, correctness),
    }


def binary_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC where larger values indicate OOD; ties receive half credit."""
    in_values = np.asarray(id_scores, dtype=float)
    out_values = np.asarray(ood_scores, dtype=float)
    comparisons = out_values[:, None] - in_values[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
