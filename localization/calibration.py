"""Source sets with a coverage guarantee from the mean-field (composite) likelihood.

The mean-field score multiplies each observer's own probability as if observers
were independent: a composite likelihood. It ranks well but its softmax is far too
confident. Two corrections, both fitted on outbreaks simulated on the same network
(no real labels needed; the transmission probability is drawn from the same prior
the localizer sums over):

- temperature: posterior proportional to exp(score / k), with k chosen to make the
  simulated outbreaks' log-probability of the true source as high as possible;
- conformal: split-conformal adaptive prediction sets. A node enters the set in
  order of posterior probability until the cumulative mass reaches a threshold
  set so that, on the simulated outbreaks, a 1 - alpha share of true sources is
  inside. If test outbreaks come from the same process, coverage is at least
  1 - alpha (Vovk et al.; Romano et al., 2020).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

TEMPERATURES = np.geomspace(0.5, 200.0, 60)


def posterior(score: np.ndarray, temperature: float) -> np.ndarray:
    feasible = score > -1e11
    log_p = np.full(len(score), -np.inf)
    log_p[feasible] = score[feasible] / temperature
    log_p -= logsumexp(log_p[feasible])
    return np.exp(log_p)


def fit_temperature(scores: list[np.ndarray], sources: list[int]) -> float:
    """Temperature maximising the mean log-probability of the true sources."""
    best, best_ll = TEMPERATURES[0], -np.inf
    for k in TEMPERATURES:
        ll = np.mean([np.log(max(posterior(s, k)[src], 1e-300)) for s, src in zip(scores, sources)])
        if ll > best_ll:
            best, best_ll = k, ll
    return float(best)


def _mass_before(prob: np.ndarray, node: int) -> float:
    """Adaptive-prediction-set score: probability mass ranked strictly above `node`,
    plus `node`'s own mass (so the set must include it)."""
    return float(prob[prob > prob[node]].sum() + prob[node])


def conformal_threshold(probs: list[np.ndarray], sources: list[int], alpha: float = 0.1) -> float:
    scores = np.sort([_mass_before(p, s) for p, s in zip(probs, sources)])
    n = len(scores)
    index = min(int(np.ceil((n + 1) * (1 - alpha))) - 1, n - 1)
    return float(scores[index])


def prediction_set(prob: np.ndarray, threshold: float) -> np.ndarray:
    order = np.argsort(-prob)
    cumulative = np.cumsum(prob[order])
    size = int(np.searchsorted(cumulative, threshold - 1e-12) + 1)
    return order[:size]
