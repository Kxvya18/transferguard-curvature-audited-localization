"""Is localization limited by the algorithm or by the evidence?

For a range of observer fractions, rank the source with mean field, message
passing and an informed simulation oracle. The oracle is told the true start
time and transmission probability (information no real method has) and, for
every candidate source, simulates many outbreaks and weighs each by how closely
it reproduces the observers' reports:

    weight = exp(-sum over observers |simulated time - reported time| / bandwidth).

As the number of simulations grows and the bandwidth shrinks this is the exact
posterior of the easier, known-parameter problem, so (once it has converged)
its top-1 accuracy bounds what any method can reach on our harder problem.
The oracle is run at two simulation counts to show whether it has converged.

Run:  python -m experiments.evidence_ceiling "Hospital" 30 [seed]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_scores, message_passing_scores
from localization.spread import earliest_arrival, simulate_outbreak, spread

FRACTIONS = (0.025, 0.05, 0.1, 0.2, 0.4)
BANDWIDTH = 0.5


def informed_oracle(ctx: est.Context, p: float, tau: int, n_sim: int, rng: np.random.Generator) -> np.ndarray:
    ob, g = ctx.outbreak, ctx.graph
    y = ob.observed_times
    infected = y < ob.t_observe
    arrival = earliest_arrival(g, tau, ob.t_observe)
    possible = np.flatnonzero((g.join <= tau) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1))
    score = np.full(g.n_nodes, est.INFEASIBLE)
    for s in possible:
        init = np.full((n_sim, g.n_nodes), np.inf)
        init[:, s] = tau
        sim = np.minimum(spread(g, init, tau, ob.t_observe, p, rng)[:, ob.observers], ob.t_observe)
        score[s] = logsumexp(-np.abs(sim - y).sum(axis=1) / BANDWIDTH) - np.log(n_sim)
    return score


def main(name: str, count: int, seed: int = 401) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "evidence_ceiling"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}.jsonl"
    existing = [json.loads(line)["fraction"] for line in path.open()] if path.exists() else []
    for fraction in FRACTIONS:
        done = 0
        while done < count:
            ob = simulate_outbreak(graph, rng, p, horizon, fraction)
            if ob is None:
                continue
            if done < existing.count(fraction):
                done += 1
                continue
            ctx = est.build_context(graph, ob, lookback)
            row = {"dataset": name, "fraction": fraction, "observers": len(ob.observers)}
            for label, fn in (("Mean-field", mean_field_scores), ("Message passing", message_passing_scores),
                              ("Oracle, 1000 sims", lambda c: informed_oracle(c, p, ob.t_start, 1000, rng)),
                              ("Oracle, 5000 sims", lambda c: informed_oracle(c, p, ob.t_start, 5000, rng))):
                start = time.time()
                row[label] = rank_of_source(fn(ctx), ob.source)
                row[label + " seconds"] = time.time() - start
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(a) for a in sys.argv[3:4]))
