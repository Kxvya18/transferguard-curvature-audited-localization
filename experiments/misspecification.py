"""Robustness when the data do not follow the localizer's model.

Every localizer here assumes SI spreading with one transmission probability and
exact reported infection times. Outbreaks are simulated with one departure at a
time (10% observers, the network's own rate):

- Clean: the assumed model (reference).
- Recovery (SIR): infected people stop spreading after a geometric time with
  mean horizon / 2 windows.
- Uneven rates: each person's infectiousness is multiplied by a log-normal
  factor (sigma = 1, mean 1), so a few people spread much more than others.
- Noisy times: each reported infection time is shifted by a rounded normal
  error (sd = 1 window), within the observation period.
- Delayed reports: each reported time is late by a geometric delay (mean 1 window).
- Missed infections: each infected observer reports "healthy" with probability 0.2.
- Status-only tests: observers report only infected or healthy at t_observe, not when
  (the likelihoods are told so; the time-of-arrival baseline cannot use such data and
  sees every positive as infected at t_observe - 1).

Needs `sib` for belief propagation:
    conda run -n sib-bp python -m experiments.misspecification "Hospital" 40
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.belief_propagation import bp_scores
from localization.mean_field import mean_field_scores, message_passing_scores
from localization.spread import Outbreak

FRACTION = 0.1
SCENARIOS = ["Clean", "Recovery (SIR)", "Uneven rates", "Noisy times", "Delayed reports", "Missed infections",
             "Status-only tests"]
METHODS = {"Mean-field": mean_field_scores, "Message passing": message_passing_scores,
           "Belief propagation": bp_scores, "Time-respecting TOA": est.temporal_screen}


def simulate(graph, rng, p, horizon, scenario):
    """One outbreak under `scenario`; None if fewer than 3 observers are infected."""
    n = graph.n_nodes
    t_start = int(rng.integers(0, graph.n_windows - horizon))
    active = np.unique(graph.windows[t_start].ravel())
    if not len(active):
        return None
    source = int(rng.choice(active))
    t_observe = t_start + horizon
    infected = np.full(n, np.inf)
    infected[source] = t_start
    recovered = np.full(n, np.inf)
    spread_factor = np.ones(n)
    if scenario == "Uneven rates":
        spread_factor = rng.lognormal(-0.5, 1.0, n)  # mean exp(-0.5 + 0.5) = 1
    if scenario == "Recovery (SIR)":
        recovered[source] = t_start + rng.geometric(2.0 / horizon)
    for t in range(t_start, t_observe):
        contacts = graph.windows[t]
        for u, v in ((contacts[:, 0], contacts[:, 1]), (contacts[:, 1], contacts[:, 0])):
            can = (infected[u] <= t) & (t < recovered[u]) & (infected[v] > t + 1)
            hit = v[can & (rng.random(len(u)) < np.minimum(1.0, p * spread_factor[u]))]
            new = hit[infected[hit] > t + 1]
            infected[new] = t + 1
            if scenario == "Recovery (SIR)":
                recovered[new] = t + 1 + rng.geometric(2.0 / horizon, len(new))
    observers = rng.choice(n, size=max(int(round(FRACTION * n)), 3), replace=False)
    times = np.minimum(infected[observers], t_observe)
    if np.sum(times < t_observe) < 3:
        return None
    caught = times < t_observe
    if scenario == "Noisy times":
        times[caught] = np.clip(times[caught] + np.round(rng.normal(0, 1, caught.sum())), t_start, t_observe - 1)
    if scenario == "Delayed reports":
        times[caught] = np.minimum(times[caught] + rng.geometric(0.5, caught.sum()) - 1, t_observe)
    if scenario == "Missed infections":
        times[caught & (rng.random(len(times)) < 0.2)] = t_observe
    if scenario == "Status-only tests":
        times[caught] = t_observe - 1
    if np.sum(times < t_observe) < 3:
        return None
    return Outbreak(source, t_start, t_observe, observers, times)


def main(name: str, count: int, seed: int = 601) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "misspecification"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}.jsonl"
    # Larger runs: SCENARIOS="Recovery (SIR),Uneven rates", another seed, SHARD="i/n"; own files.
    shard, shards = map(int, os.environ.get("SHARD", "0/1").split("/"))
    chosen = os.environ.get("SCENARIOS", ",".join(SCENARIOS)).split(",")
    if seed != 601 or "SHARD" in os.environ:
        path = out / f"{name.replace(' ', '_')}_seed{seed}_shard{shard}.jsonl"
    existing = [json.loads(line)["scenario"] for line in path.open()] if path.exists() else []
    for scenario in SCENARIOS:
        done = k = 0
        while k < count:
            ob = simulate(graph, rng, p, horizon, scenario)
            if ob is None:
                continue
            k += 1
            if scenario not in chosen or (k - 1) % shards != shard:
                continue
            if done < existing.count(scenario):
                done += 1
                continue
            ctx = est.build_context(graph, ob, lookback)
            row = {"dataset": name, "scenario": scenario}
            for label, fn in METHODS.items():
                start = time.time()
                told = scenario == "Status-only tests" and label != "Time-respecting TOA"
                row[label] = rank_of_source(fn(ctx, status_only=True) if told else fn(ctx), ob.source)
                row[label + " seconds"] = time.time() - start
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(a) for a in sys.argv[3:4]))
