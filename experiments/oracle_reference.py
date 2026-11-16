"""Near-optimal reference: Monte Carlo estimate of the exact posterior, given the TRUE rate.

For each possible source s and start window tau at which s had a contact, simulate many
outbreaks with the true transmission probability and count how often the observers'
reports are reproduced exactly. With enough simulations this approaches the Bayes-optimal
ranking under the benchmark's own generative model, so it bounds what any method can do.

Run:  python -m experiments.oracle_reference "Hospital" 0.1 40 3000 [seed]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import contact_start_scores, mean_field_scores
from localization.spread import earliest_arrival, simulate_outbreak, spread


def oracle_scores(ctx: est.Context, p: float, rng: np.random.Generator, n_sim: int) -> np.ndarray:
    ob, g = ctx.outbreak, ctx.graph
    y = ob.observed_times
    infected = y < ob.t_observe
    first = int(y[infected].min())
    arrival = earliest_arrival(g, ctx.t_lo, ob.t_observe)
    possible = np.flatnonzero((g.join <= first) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1))
    log_post = np.full(g.n_nodes, -np.inf)
    for tau in range(ctx.t_lo, first + 1):
        active = np.intersect1d(np.unique(g.windows[tau].ravel()), possible)
        if not len(active):
            continue
        for s in active:
            init = np.full((n_sim, g.n_nodes), np.inf)
            init[:, s] = tau
            sim = np.minimum(spread(g, init, tau, ob.t_observe, p, rng)[:, ob.observers], ob.t_observe)
            hits = np.sum(np.all(sim == y, axis=1))
            # (hits + 0.5) / n: a small pseudo-count so that rare matches do not give log(0).
            term = np.log((hits + 0.5) / n_sim) - np.log(len(np.unique(g.windows[tau].ravel())))
            log_post[s] = np.logaddexp(log_post[s], term)
    return np.where(np.isfinite(log_post), log_post, est.INFEASIBLE)


def main(name: str, fraction: float, count: int, n_sim: int, seed: int = 61) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "oracle_reference"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}_{fraction}_{n_sim}.jsonl"
    done = 0
    while done < count:
        ob = simulate_outbreak(graph, rng, p, horizon, fraction)
        if ob is None:
            continue
        ctx = est.build_context(graph, ob, lookback)
        row = {"dataset": name, "test_rate": "true", "seed": seed,
               "Oracle": rank_of_source(oracle_scores(ctx, p, rng, n_sim), ob.source),
               "Mean-field": rank_of_source(mean_field_scores(ctx), ob.source),
               "Mean-field, contact-start prior": rank_of_source(contact_start_scores(ctx), ob.source)}
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        done += 1


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), *(int(a) for a in sys.argv[5:6]))
