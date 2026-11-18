"""How to handle the unknown start time (and rate): variants on identical outbreaks.

All variants come from one mean-field log-likelihood matrix over every start
window and every rate in the grid, so they differ only in how the start time
and rate are summed out:

- 8 start times (current): uniform over 8 evenly spaced start windows.
- Every start time: uniform over all start windows.
- Contact-start prior: the outbreak starts at a contact of the source.
- Tempered sum: the composite likelihood is over-confident, so each (start, rate)
  term is divided by a temperature before summing (k = 2, the value fitted for
  calibrated sets); this keeps one lucky start time from dominating.
- Told rate / told start / told both: reference rows with the truth given.

Run:  python -m experiments.start_time "Hospital" 40 [any]
("any": the source may start without a contact, to test the contact-start prior.)
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import _start_times, mean_field_loglik, mean_field_scores
from localization.spread import simulate_outbreak

PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)
FRACTION = 0.1
TEMPERATURE = 2.0


def finish(score: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(score), score, est.INFEASIBLE)


def variants(ctx: est.Context, p: float) -> dict[str, np.ndarray]:
    ob, graph = ctx.outbreak, ctx.graph
    starts = _start_times(ctx, None)
    loglik = mean_field_loglik(ctx, prob_grid=PROBS, n_start_times=None).reshape(len(starts), len(PROBS), -1)
    prior = np.full((len(starts), graph.n_nodes), -np.inf)  # contact-start prior over (start, source)
    for i, tau in enumerate(starts):
        active = np.unique(graph.windows[tau].ravel())
        if len(active):
            prior[i, active] = -np.log(len(active))
    true_start = int(np.flatnonzero(starts == ob.t_start)[0])
    true_rate = PROBS.index(p)
    flat = loglik.reshape(-1, graph.n_nodes)
    with_prior = loglik + prior[:, None, :]
    return {
        "8 start times": mean_field_scores(ctx),
        "Every start time": finish(logsumexp(flat, axis=0)),
        "Contact-start prior": finish(logsumexp(with_prior.reshape(-1, graph.n_nodes), axis=0)),
        "Tempered sum": finish(TEMPERATURE * logsumexp(flat / TEMPERATURE, axis=0)),
        "Tempered sum + contact-start": finish(TEMPERATURE * logsumexp(with_prior.reshape(-1, graph.n_nodes) / TEMPERATURE, axis=0)),
        "Told rate": finish(logsumexp(loglik[:, true_rate], axis=0)),
        "Told start": finish(logsumexp(loglik[true_start], axis=0)),
        "Told both": finish(loglik[true_start, true_rate]),
    }


def main(name: str, count: int, seed: int = 501, any_start: bool = False) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / ("start_time_any" if any_start else "start_time")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}.jsonl"
    done = sum(1 for _ in path.open()) if path.exists() else 0
    for k in range(count):
        ob = None
        while ob is None:
            ob = simulate_outbreak(graph, rng, p, horizon, FRACTION, any_start=any_start)
        if k < done:
            continue
        ctx = est.build_context(graph, ob, lookback)
        row = {"dataset": name} | {label: rank_of_source(s, ob.source) for label, s in variants(ctx, p).items()}
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), any_start="any" in sys.argv[3:])
